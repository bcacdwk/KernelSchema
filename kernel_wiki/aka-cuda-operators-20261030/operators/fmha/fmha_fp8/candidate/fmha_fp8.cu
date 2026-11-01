// fmha_fp8.cu — Independent FP8-E4M3 paged FMHA for SM103 (problem 14 / L20D).
// Pure CUDA C++/PTX; written from scratch for this contract. ldmatrix/mma
// fragment conventions follow the public PTX ISA (patterns cross-checked
// against flashinfer's BSD-3 wrappers; no code copied, no library called).
//
// Math (mirrors problem/reference.py):
//   s   = (q_fp8 . k_fp8) * bmm1_scale           (fp32 accumulate)
//   p   = exp(s - rowmax), online softmax over visible kv positions
//   out = (sum p * v_fp8) * bmm2_scale / sum p   -> bf16
//   lse = (max + ln sum_p) / ln 2                -> fp32 (log2 units)
//   sinks: total2 = logaddexp2(base2, sink*log2e); out *= exp2(base2-total2)
// Kernel works in the log2 domain: s2 = s*log2e; p = exp2(s2 - m2);
// base2 = m2 + log2(d) == reference lse exactly (modulo fp32 rounding).
//
// v2: double-buffered K/V tiles (one barrier per kv iteration), causal
// heavy-tile-first ordering, ex2.approx hot path, prepare/run split for low
// per-call host overhead.

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>   // FMHA_DBG_DISPATCH dev print only
#include <algorithm>
#include <vector>
#include <set>      // r19c: host-side balanced partition (prepare-time only)

#define NEG_INF_F __int_as_float(0xff800000)
#define LOG2E 1.4426950408889634f

struct Params {
  const uint8_t* q;        // [Tq, H, Dqk] fp8 e4m3, contiguous
  const uint8_t* k;        // paged fp8; strides below in bytes (== elements)
  const uint8_t* v;        // paged fp8
  const int32_t* table;    // [B, Tstride]
  const int32_t* seqlens;  // [B]
  const int32_t* cu_q;     // [B + 1]
  uint16_t* out;           // [Tq, H, Dv] bf16
  float* lse;              // [Tq, H] or null
  float* ws;               // split partials or null (S == 1)
  const float* sinks;      // [H] or null
  int64_t q_tok_stride;    // H * Dqk
  int64_t k_sp, k_sh, k_sn;
  int64_t v_sp, v_sh, v_sn;
  int Tstride, page, page_log2;
  int H, Hkv, ratio, B;
  int T;                   // q-tile upper bound per unit (grid.x)
  int S;                   // split factor (grid.z)
  int chunk;               // kv positions per split
  float qk_scale;          // bmm1_scale * log2(e)
  float bmm2;              // bmm2_scale
  int window_left, causal, has_lse, has_sink;
  unsigned* dbg;           // optional mapped-host progress markers (tc kernel)
  int exit_mode;           // 0 normal; 1 skip tmem dealloc; 2 skip final sync+dealloc
  int tile_order;          // 1 = unit-fastest CTA->tile remap (wave balance); 0 = per-unit
  // C1: TMA (cp.async.bulk.tensor.4d) K/V tile loads for the tc kernels.
  // use_tma=1 only when the host verified: tc impl 1/2 (128B rows/plane),
  // window_left<0 (lo==0 -> every kv tile is page-aligned), contiguous rows
  // (k_sn==v_sn==dqk), 16B-aligned base/strides, power-of-two page in
  // [16,128]. Maps describe K/V as 4-D [D][page_rows][pages][heads] byte
  // tensors; the loader issues one 128B x page-rows box per page segment.
  int use_tma;
  alignas(64) CUtensorMap kmap;
  alignas(64) CUtensorMap vmap;
  // C2: per-unit work-balanced split-KV. balanced=1: grid.x linearizes
  // (batch, kv-head, x-tile, split) with exact per-batch prefixes (zero empty
  // CTAs); unit b's kv span is split into unit_S[b] near-equal TILE-count
  // chunks computed on-device from each (unit,x)'s actual ntiles. p.S carries
  // S_max (combine's split bound); chunks with c >= unit_S[b] cannot occur
  // (linear grid) except per-x causal shortfalls, which self-clean as empty.
  int balanced;
  // Debug knobs (env FMHA_FORCE_CORR / FMHA_FORCE_LAG1; default 0 = off):
  // bisect the skip-corr early-arrive (v5) and PV lag-2 (v6b) mechanisms.
  int force_corr;
  int force_lag1;
  int unit_S[32];
  int batch_off[33];      // balanced: CTA prefix per batch (grid.x decode)
  int batch_tiles[32];    // balanced: x-tiles per batch
  // Prepare-time geometry cache. cu_q/seqlens are ALREADY read back to the host
  // in prepare (the span-guard block, gated on B<=32 && ws) under the
  // cached-handle ABI, so each b's (q0, qlen, kvlen) can ride in param space
  // instead of costing the kernel a cold dependent DRAM round trip: measured
  // 1152-1536 cyc = 0.61-0.81us, 60-75% of the tc setup and hop 1 of the
  // three-hop prologue chain (results/join24-prologue.md 4b/7), with hop 2
  // (page table) and hop 3 (K/V TMA) blocked behind it. No new host traffic and
  // no new assumption: that block already needs the values on the host, and the
  // balanced path below already treats them as frozen per handle. geo_valid == 0
  // (B>32, ws==null, or a failed readback) keeps the pointer path exactly, so
  // no shape gains a failure mode. Values are RAW cu_q/seqlens; every derived
  // quantity (R, tiles, lo, hi) is still computed on-device from them.
  int geo_valid;
  int geo_cu[33];         // == cu_q[0..B]
  int geo_sl[32];         // == seqlens[0..B-1]
  // r18 persistent-MT workers (impl 6/7, S==1, env FMHA_MT_PERS; appended at
  // the struct tail so every pre-existing field offset — and therefore the
  // non-persistent kernels' SASS — is untouched). pers_total = the ORIGINAL
  // grid.x*grid.y aid space the workers stride over; pers_gx/gy = the
  // original grid dims the aid->(u,bx) map (incl. tile_order remap) uses.
  int pers_total;
  int pers_gx;
  int pers_gy;
  int pers_kmax;             // per-worker list stride (ints, -1 terminated)
  const int* pers_list;      // device buffer, per-Prepared-slot private
  int pers_static;           // r19c: 1 = fmtc static balanced partition
                             // (pers_list = [off_0..off_nw][aids...]);
                             // 0 = dynamic ticket pool ([0] = counter)
};

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                            const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                  uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_fp8(float* C, const uint32_t* A, const uint32_t* B) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}
__device__ __forceinline__ void mma_fp16(float* C, const uint32_t* A, const uint32_t* B) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}
// two packed e4m3 bytes (low 16 bits of x) -> f16x2 (half0 <- byte0)
__device__ __forceinline__ uint32_t cvt_e4m3x2_f16x2(uint32_t x) {
  uint32_t d;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(d) : "h"((uint16_t)x));
  return d;
}
// Single-MUFU base-2 exponential for the hot softmax path (rel err ~2^-22;
// exact 1.0 at 0, 0 at -inf). High-precision exp2f/log2f stay in epilogues.
__device__ __forceinline__ float ex2_fast(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  __half2 h = __float22half2_rn(make_float2(lo, hi));
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 h = __float22bfloat162_rn(make_float2(lo, hi));
  return *reinterpret_cast<uint32_t*>(&h);
}
// Split-KV workspace partials are stored as bf16x2 -- two O columns per 32-bit
// word -- so the combine reads and writes HALF the bytes. m and d stay f32: they
// feed the lse path and the self-cleaning empty-split markers, where a rounding
// change would move the log-sum-exp, and they are only two words per row anyway.
// The output is bf16 regardless, so this trades accumulation precision of the
// partials (8 mantissa bits) for halving traffic on a path measured L2-bound at
// f32 (4.1 TB/s on c2). Layout per (split, unit, x-tile, row):
//   [O: Dv bf16 = Dv/2 words][m f32][d f32],  row stride = Dv/2 + 2 words.
__device__ __forceinline__ float2 bf16x2_to_f32x2(uint32_t w) {
  const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w);
  return __bfloat1622float2(b);
}

// C2 decode: linear blockIdx.x -> (batch b, kv-head-in-batch g, x-tile bx,
// split c). batch_off/batch_tiles/unit_S are host-baked at prepare; the order
// is c-innermost so consecutive CTAs share one unit's kv stream (L2 locality).
__device__ __forceinline__ void balanced_decode(const Params& p, int& b, int& g,
                                                int& bx, int& c) {
  const int L = (int)blockIdx.x;
  int lo = 0, hi = p.B - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (p.batch_off[mid] <= L) lo = mid; else hi = mid - 1;
  }
  b = lo;
  const int r = L - p.batch_off[b];
  const int sb = p.unit_S[b];
  const int ts = p.batch_tiles[b] * sb;
  g = r / ts;
  const int rem = r - g * ts;
  bx = rem / sb;
  c = rem - bx * sb;
}

#ifndef FMHA_NO_TC
#include "fmha_tc.cuh"
#include "fmha_tc2.cuh"
#include "fmha_tcmt.cuh"
#endif

// ---------------------------------------------------------------------------
// Main kernel: one CTA = (q-tile x, unit u=(b,g), kv-split c); 4 warps, M=64.
// ---------------------------------------------------------------------------
template <int DQK, int NT>
__global__ void fmha_kernel(const __grid_constant__ Params p) {
  constexpr int DV = DQK;          // contract: Dv == Dqk in this domain
  constexpr int M = 64;            // q rows per tile
  constexpr int QS = DQK + 16;     // Q/K smem row stride (bytes)
  constexpr int VS = DV + 8;       // V smem row stride (halves)
  constexpr int NF = NT / 8;       // score n8 fragments per warp
  constexpr int OF = DV / 8;       // output n8 fragments per warp
  constexpr int KC = DQK / 32;     // fp8 k32 chunks
  constexpr int NCH = DQK / 16;    // 16B chunks per row
  constexpr int NKCH = NT * NCH / 128;  // 16B chunks per thread (K and V tiles)
  constexpr bool PERSIST_A = (DQK <= 128);
  constexpr int QSB = M * QS;           // Q bytes
  constexpr int KSB = NT * QS;          // K bytes per buffer
  constexpr int VSB = NT * VS;          // V halves per buffer

  constexpr int V8B = NT * DQK;         // raw fp8 V bytes per stage
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* Qs = smem;
  uint8_t* Ks[3] = {Qs + QSB, Qs + QSB + KSB, Qs + QSB + 2 * KSB};
  uint8_t* V8[3] = {Qs + QSB + 3 * KSB, Qs + QSB + 3 * KSB + V8B,
                    Qs + QSB + 3 * KSB + 2 * V8B};
  __half* Vs[2] = {reinterpret_cast<__half*>(Qs + QSB + 3 * KSB + 3 * V8B),
                   reinterpret_cast<__half*>(Qs + QSB + 3 * KSB + 3 * V8B + VSB * 2)};

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // tile_order: remap the linearized CTA id to (unit, tile) unit-fastest so a
  // hardware wave spans many units (weight-uniform waves) instead of sweeping
  // one unit's heavy-first tiles. Pure scheduling: (u,x,c) slot math is
  // unchanged, so outputs are bitwise identical. Gated by the host to shapes
  // whose KV fits in L2 (locality of the per-unit order is not needed).
  // C2 balanced: grid.x linearizes (b, g, x, c) exactly; no empty CTAs.
  int u, bx, c, b;
  if (p.balanced) {
    int g_dec;
    balanced_decode(p, b, g_dec, bx, c);
    u = b * p.Hkv + g_dec;
  } else {
    u = (int)blockIdx.y;
    bx = (int)blockIdx.x;
    if (p.tile_order) {
      const int gid = bx + (int)blockIdx.y * (int)gridDim.x;
      const int nu = (int)gridDim.y;
      u = gid % nu;
      bx = gid / nu;
    }
    b = u / p.Hkv;
    c = (int)blockIdx.z;
  }
  const int g = u - b * p.Hkv;

  // Param-space geometry when prepare cached it (Params::geo_valid): the same
  // integers, one fewer cold dependent DRAM round trip at the head of the CTA.
  const int q0 = p.geo_valid ? p.geo_cu[b] : p.cu_q[b];
  const int qlen = (p.geo_valid ? p.geo_cu[b + 1] : p.cu_q[b + 1]) - q0;
  const int kvlen = p.geo_valid ? p.geo_sl[b] : p.seqlens[b];
  const int R = qlen * p.ratio;
  const int tiles = (R + M - 1) / M;
  // Causal: process heavy (late) tiles first for better tail balance.
  const int x = p.causal ? (tiles - 1 - bx) : bx;
  if (x < 0 || x >= tiles) return;

  const int r0 = x * M;
  const int t0 = r0 / p.ratio;
  int tlast = (r0 + M - 1) / p.ratio;
  if (tlast > qlen - 1) tlast = qlen - 1;
  const int vis_lo = kvlen - qlen + t0;
  const int vis_hi = kvlen - qlen + tlast;

  int lo = 0, hi = kvlen;
  if (p.causal) hi = min(hi, vis_hi + 1);
  if (p.window_left >= 0) lo = max(lo, vis_lo - p.window_left);
  // ws slot math uses the LOGICAL unit count (gridDim.y == 1 when balanced).
  const int nunits_g = p.B * p.Hkv;
  const int Su = p.balanced ? p.unit_S[b] : p.S;
  if (Su > 1) {
    if (p.balanced) {
      // Per-(unit,x) equal-tile chunks: balances the causal span variation
      // exactly instead of one uniform key chunk for all x-tiles.
      const int nkv = (hi - lo + NT - 1) / NT;
      const int cpt = (nkv + Su - 1) / Su;
      lo += c * cpt * NT;
      hi = min(hi, lo + cpt * NT);
    } else {
      lo += c * p.chunk;
      hi = min(hi, lo + p.chunk);
    }
  }
  const int rows_valid = min(M, R - r0);
  if (lo >= hi) {
    // Self-cleaning empty marker: combine skips chunks with d == 0, so graph
    // replay without workspace restore can never see stale nonzero state.
    if (Su > 1) {
      const int64_t slot0 = ((int64_t)c * nunits_g + u) * p.T + x;
      const int64_t base = slot0 * (M * (DV / 2 + 2));
      for (int r = tid / 2; r < rows_valid; r += 64) {
        if ((tid & 1) == 0) {
          p.ws[base + r * (DV / 2 + 2) + DV / 2] = NEG_INF_F;
          p.ws[base + r * (DV / 2 + 2) + DV / 2 + 1] = 0.0f;
        }
      }
    }
    return;
  }

  // ---- Q tile (cp.async, raw fp8) -----------------------------------------
  for (int idx = tid; idx < M * NCH; idx += 128) {
    const int r = idx / NCH;
    const int col = (idx - r * NCH) * 16;
    int rr = r0 + r;
    if (rr > R - 1) rr = R - 1;
    const int t = rr / p.ratio;
    const int h = g * p.ratio + (rr - t * p.ratio);
    const uint8_t* src = p.q + (int64_t)(q0 + t) * p.q_tok_stride + (int64_t)h * DQK + col;
    cp_async16(Qs + r * QS + col, src);
  }
  cp_async_commit();

  // Per-thread row identities
  const int rowl = 16 * warp + (lane >> 2);
  const int rowl2 = rowl + 8;
  const int rg0 = r0 + rowl, rg1 = r0 + rowl2;
  // Does this WARP own any live row? Warp w owns rows 16w..16w+15 exclusively
  // (rowl above), attention rows are independent, and every store in both
  // epilogues is already guarded per lane by rg0<R / rg1<R -- so a warp whose
  // whole band is past R computes values that are read by nothing. Decode shapes
  // make that the common case: R = qlen*ratio is 16 and 12 on c8 and 16 on c5's
  // first batch, inside an M=64 tile, so THREE OF FOUR WARPS were issuing the
  // full KC*NF fp8 QK fragment chain and the NT/16 x DV/16 fp16 PV chain for
  // discarded rows. mma.sync is warp-scoped and the four warps share one tensor
  // pipe, so that work is not free parallelism -- it is 4x the pipe occupancy.
  // Warp-uniform by construction (r0, warp and R are), which is what keeps the
  // full-mask __shfl_xor_sync reductions below legal, and both __syncthreads()
  // plus convert_v/load_k/load_v8/cp_async_commit stay OUTSIDE the guard so the
  // CTA-wide pipeline and the live warps' K/V are untouched.
  const bool warp_live = (r0 + 16 * warp) < R;
  const int tok0 = rg0 / p.ratio, tok1 = rg1 / p.ratio;
  const int h0 = g * p.ratio + (rg0 - tok0 * p.ratio);
  const int h1 = g * p.ratio + (rg1 - tok1 * p.ratio);
  const float vis0 = (float)(kvlen - qlen + tok0);
  const float vis1 = (float)(kvlen - qlen + tok1);
  const int colq = 2 * (lane & 3);

  const int32_t* tab = p.table + (int64_t)b * p.Tstride;
  const uint8_t* kb = p.k + (int64_t)g * p.k_sh;
  const uint8_t* vb = p.v + (int64_t)g * p.v_sh;
  const int plog = p.page_log2, pmask = p.page - 1;

  auto load_k = [&](int n0, int buf) {
    uint8_t* dst = Ks[buf];
#pragma unroll
    for (int i = 0; i < NKCH; ++i) {
      const int idx = tid + i * 128;
      const int r = idx / NCH;
      const int col = (idx - r * NCH) * 16;
      int pos = n0 + r;
      if (pos > hi - 1) pos = hi - 1;
      const int pgd = pos >> plog;
      const int off = pos & pmask;
      const uint8_t* src = kb + (int64_t)__ldg(tab + pgd) * p.k_sp + (int64_t)off * p.k_sn + col;
      cp_async16(dst + r * QS + col, src);
    }
  };
  auto load_v8 = [&](int n0, int buf) {
    uint8_t* dst = V8[buf];
    constexpr int NVCH = NT * NCH / 128;
#pragma unroll
    for (int i = 0; i < NVCH; ++i) {
      const int idx = tid + i * 128;
      const int r = idx / NCH;
      const int col = (idx - r * NCH) * 16;
      int pos = n0 + r;
      if (pos > hi - 1) pos = hi - 1;
      const int pgd = pos >> plog;
      const int off = pos & pmask;
      const uint8_t* src = vb + (int64_t)__ldg(tab + pgd) * p.v_sp + (int64_t)off * p.v_sn + col;
      cp_async16(dst + r * DQK + col, src);
    }
  };
  // raw fp8 V stage -> fp16 PV operand buffer
  auto convert_v = [&](int src8, int dst16) {
    const uint8_t* srcr = V8[src8];
    __half* dst = Vs[dst16];
#pragma unroll
    for (int i = 0; i < NKCH; ++i) {
      const int idx = tid + i * 128;
      const int r = idx / NCH;
      const int colh = idx - r * NCH;   // 16-element chunk index
      const uint4 raw = *reinterpret_cast<const uint4*>(srcr + r * DQK + colh * 16);
      const uint32_t* w = reinterpret_cast<const uint32_t*>(&raw);
      uint32_t hh[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        hh[2 * j] = cvt_e4m3x2_f16x2(w[j]);            // elements 4j, 4j+1
        hh[2 * j + 1] = cvt_e4m3x2_f16x2(w[j] >> 16);  // elements 4j+2, 4j+3
      }
      uint4 s0 = make_uint4(hh[0], hh[1], hh[2], hh[3]);
      uint4 s1 = make_uint4(hh[4], hh[5], hh[6], hh[7]);
      *reinterpret_cast<uint4*>(dst + r * VS + colh * 16) = s0;
      *reinterpret_cast<uint4*>(dst + r * VS + colh * 16 + 8) = s1;
    }
  };

  float m0 = NEG_INF_F, m1 = NEG_INF_F, d0 = 0.f, d1 = 0.f;
  float O[OF][4];
#pragma unroll
  for (int f = 0; f < OF; ++f) { O[f][0] = 0.f; O[f][1] = 0.f; O[f][2] = 0.f; O[f][3] = 0.f; }

  uint32_t Afr[KC][4];
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int a_col = (lane >> 4) * 16;

  int n0 = lo;
  load_k(n0, 0);
  load_v8(n0, 0);
  cp_async_commit();                     // G_0
  if (n0 + NT < hi) {
    load_k(n0 + NT, 1);
    load_v8(n0 + NT, 1);
  }
  cp_async_commit();                     // G_1 (possibly empty)
  asm volatile("cp.async.wait_group 1;\n" ::);   // Q + G_0 complete
  __syncthreads();
  if (PERSIST_A) {
#pragma unroll
    for (int kc = 0; kc < KC; ++kc)
      ldmatrix_x4(Afr[kc][0], Afr[kc][1], Afr[kc][2], Afr[kc][3],
                  Qs + (16 * warp + a_row) * QS + kc * 32 + a_col);
  }

  const int b_row = lane & 7;
  const int b_col = ((lane >> 3) & 1) * 16;
  const int v_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int v_col = (lane >> 4) * 8;
  const float wf = (float)p.window_left;
  const float fend = (float)hi;

  int it = 0;
  for (; n0 < hi; n0 += NT, ++it) {
    const int buf = it & 1;
    const int kst = it % 3;
    // G_it complete (K/V8 stage it%3 ready); PV(it-1) and convert(it-1) done.
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
    convert_v(kst, buf);          // raw fp8 stage -> fp16 PV operand buffer

    // ---- QK^T -> C[NF][4] (fp32) -------------------------------------------
    const uint8_t* Kt = Ks[kst];
    float C[NF][4];
#pragma unroll
    for (int f = 0; f < NF; ++f) { C[f][0] = 0.f; C[f][1] = 0.f; C[f][2] = 0.f; C[f][3] = 0.f; }
    if (warp_live) {
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
      uint32_t A4[4];
      const uint32_t* Au;
      if (PERSIST_A) {
        Au = Afr[kc];
      } else {
        ldmatrix_x4(A4[0], A4[1], A4[2], A4[3],
                    Qs + (16 * warp + a_row) * QS + kc * 32 + a_col);
        Au = A4;
      }
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        uint32_t B2[2];
        ldmatrix_x2(B2[0], B2[1], Kt + (f * 8 + b_row) * QS + kc * 32 + b_col);
        mma_fp8(C[f], Au, B2);
      }
    }
    }   // warp_live: QK

    __syncthreads();              // convert_v stores visible to all warps
    if (n0 + 2 * NT < hi) {       // issue G_{it+2} DMA (stage free since it-1)
      load_k(n0 + 2 * NT, (it + 2) % 3);
      load_v8(n0 + 2 * NT, (it + 2) % 3);
    }
    cp_async_commit();

    if (warp_live) {
    // ---- scale + mask -------------------------------------------------------
    const bool need_mask = p.causal || p.window_left >= 0 || (n0 + NT > hi);
    float mr0 = NEG_INF_F, mr1 = NEG_INF_F;
#pragma unroll
    for (int f = 0; f < NF; ++f) {
      float s0 = C[f][0] * p.qk_scale;
      float s1 = C[f][1] * p.qk_scale;
      float s2 = C[f][2] * p.qk_scale;
      float s3 = C[f][3] * p.qk_scale;
      if (need_mask) {
        const float cA = (float)(n0 + 8 * f + colq);
        const float cB = cA + 1.f;
        bool okA = cA < fend, okB = cB < fend;
        if (p.causal) { okA = okA && (cA <= vis0); okB = okB && (cB <= vis0); }
        if (p.window_left >= 0) { okA = okA && (cA >= vis0 - wf); okB = okB && (cB >= vis0 - wf); }
        if (!okA) s0 = NEG_INF_F;
        if (!okB) s1 = NEG_INF_F;
        bool okC = cA < fend, okD = cB < fend;
        if (p.causal) { okC = okC && (cA <= vis1); okD = okD && (cB <= vis1); }
        if (p.window_left >= 0) { okC = okC && (cA >= vis1 - wf); okD = okD && (cB >= vis1 - wf); }
        if (!okC) s2 = NEG_INF_F;
        if (!okD) s3 = NEG_INF_F;
      }
      C[f][0] = s0; C[f][1] = s1; C[f][2] = s2; C[f][3] = s3;
      mr0 = fmaxf(mr0, fmaxf(s0, s1));
      mr1 = fmaxf(mr1, fmaxf(s2, s3));
    }
    mr0 = fmaxf(mr0, __shfl_xor_sync(0xffffffffu, mr0, 1));
    mr0 = fmaxf(mr0, __shfl_xor_sync(0xffffffffu, mr0, 2));
    mr1 = fmaxf(mr1, __shfl_xor_sync(0xffffffffu, mr1, 1));
    mr1 = fmaxf(mr1, __shfl_xor_sync(0xffffffffu, mr1, 2));

    const float mn0 = fmaxf(m0, mr0), mn1 = fmaxf(m1, mr1);
    const float mu0 = (mn0 == NEG_INF_F) ? 0.f : mn0;
    const float mu1 = (mn1 == NEG_INF_F) ? 0.f : mn1;
    const float rs0 = ex2_fast(m0 - mu0), rs1 = ex2_fast(m1 - mu1);
    m0 = mn0; m1 = mn1;
    float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
    for (int f = 0; f < NF; ++f) {
      const float p0 = ex2_fast(C[f][0] - mu0);
      const float p1 = ex2_fast(C[f][1] - mu0);
      const float p2 = ex2_fast(C[f][2] - mu1);
      const float p3 = ex2_fast(C[f][3] - mu1);
      C[f][0] = p0; C[f][1] = p1; C[f][2] = p2; C[f][3] = p3;
      ps0 += p0 + p1;
      ps1 += p2 + p3;
    }
    ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
    ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
    ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
    ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
    d0 = d0 * rs0 + ps0;
    d1 = d1 * rs1 + ps1;
    if (rs0 != 1.f || rs1 != 1.f) {
#pragma unroll
      for (int f = 0; f < OF; ++f) {
        O[f][0] *= rs0; O[f][1] *= rs0; O[f][2] *= rs1; O[f][3] *= rs1;
      }
    }

    // ---- P x V -> O ---------------------------------------------------------
    const __half* Vt = Vs[buf];
#pragma unroll
    for (int kp = 0; kp < NT / 16; ++kp) {
      uint32_t A2[4];
      // PTX m16n8k16 A-fragment order: a0=(row t/4, k 2*(t%4)), a1=(row+8, k lo),
      // a2=(row, k +8), a3=(row+8, k +8).
      A2[0] = pack_f16x2(C[2 * kp][0], C[2 * kp][1]);
      A2[1] = pack_f16x2(C[2 * kp][2], C[2 * kp][3]);
      A2[2] = pack_f16x2(C[2 * kp + 1][0], C[2 * kp + 1][1]);
      A2[3] = pack_f16x2(C[2 * kp + 1][2], C[2 * kp + 1][3]);
#pragma unroll
      for (int db = 0; db < DV / 16; ++db) {
        uint32_t B4[4];
        ldmatrix_x4_trans(B4[0], B4[1], B4[2], B4[3],
                          Vt + (kp * 16 + v_row) * VS + db * 16 + v_col);
        mma_fp16(O[2 * db], A2, B4);
        mma_fp16(O[2 * db + 1], A2, B4 + 2);
      }
    }
    }   // warp_live: softmax + PV
  }

  // ---- epilogue -----------------------------------------------------------
  if (Su > 1) {
    const int64_t slot0 = ((int64_t)c * nunits_g + u) * p.T + x;
    const int64_t base = slot0 * (M * (DV / 2 + 2));
    uint32_t* __restrict__ ws32 = reinterpret_cast<uint32_t*>(p.ws);
    // colq = 2*(lane&3) is even, so the float pair (8f+colq, 8f+colq+1) is
    // exactly the 32-bit word 4f + (lane&3).
#pragma unroll
    for (int f = 0; f < OF; ++f) {
      if (rg0 < R)
        ws32[base + rowl * (DV / 2 + 2) + 4 * f + (lane & 3)] =
            pack_bf16x2(O[f][0], O[f][1]);
      if (rg1 < R)
        ws32[base + rowl2 * (DV / 2 + 2) + 4 * f + (lane & 3)] =
            pack_bf16x2(O[f][2], O[f][3]);
    }
    if ((lane & 3) == 0) {
      if (rg0 < R) { p.ws[base + rowl * (DV / 2 + 2) + DV / 2] = m0; p.ws[base + rowl * (DV / 2 + 2) + DV / 2 + 1] = d0; }
      if (rg1 < R) { p.ws[base + rowl2 * (DV / 2 + 2) + DV / 2] = m1; p.ws[base + rowl2 * (DV / 2 + 2) + DV / 2 + 1] = d1; }
    }
    return;
  }

  float inv0 = p.bmm2 / d0;
  float inv1 = p.bmm2 / d1;
  float base20 = m0 + log2f(d0);
  float base21 = m1 + log2f(d1);
  if (p.has_sink) {
    const float sk0 = p.sinks[h0] * LOG2E;
    const float sk1 = p.sinks[h1] * LOG2E;
    const float t20 = fmaxf(base20, sk0) + log2f(1.f + exp2f(-fabsf(base20 - sk0)));
    const float t21 = fmaxf(base21, sk1) + log2f(1.f + exp2f(-fabsf(base21 - sk1)));
    inv0 *= exp2f(base20 - t20);
    inv1 *= exp2f(base21 - t21);
    base20 = t20; base21 = t21;
  }
  if (p.has_lse) {
    if (rg0 < R) p.lse[(int64_t)(q0 + tok0) * p.H + h0] = base20;
    if (rg1 < R) p.lse[(int64_t)(q0 + tok1) * p.H + h1] = base21;
  }
  uint16_t* orow0 = p.out + ((int64_t)(q0 + tok0) * p.H + h0) * DV;
  uint16_t* orow1 = p.out + ((int64_t)(q0 + tok1) * p.H + h1) * DV;
#pragma unroll
  for (int f = 0; f < OF; ++f) {
    if (rg0 < R)
      *reinterpret_cast<uint32_t*>(orow0 + 8 * f + colq) = pack_bf16x2(O[f][0] * inv0, O[f][1] * inv0);
    if (rg1 < R)
      *reinterpret_cast<uint32_t*>(orow1 + 8 * f + colq) = pack_bf16x2(O[f][2] * inv1, O[f][3] * inv1);
  }
}

// ---------------------------------------------------------------------------
// Combine kernel (S > 1): merge split partials, apply sinks, write out/lse.
// grid: (T, B*Hkv, row-blocks x col-slices of 64 dims); block: 512 threads.
// Row blocking (join 20261022 arm-b adoption): this kernel is pure latency --
// each row exposes its own split-load round trip, so one row per warp across
// more CTAs beats fewer CTAs walking rows serially (arm-b measured case9
// 0.95 TB/s effective vs multi-TB/s DRAM ceiling; case5 11.07us over 6 CTAs).
// rpb = rows per row-block CTA, shape-derived from the base combine grid
// (combine_rpb(); FMHA_COMBINE_RPB overrides). Per-row math and accumulation
// order are untouched -> bitwise-identical out at any rpb.
// v3 fast path (S <= 32): batched 4-way scale broadcast (independent shfls
// instead of a serial per-split shfl->load chain, which latency-paced rows at
// ~0.3-0.7us/split-row), float2 ws loads, next-row m/d prefetch overlapped
// with the current row's reduce chain, single 4B bf16x2 out store. Math and
// accumulation order are identical to the validated v2 path.
// ---------------------------------------------------------------------------
__global__ void fmha_combine(const __grid_constant__ Params p, int Dv, int stride,
                             int rt, float lse_adj, int rpb) {
  const int u = blockIdx.y;
  const int b = u / p.Hkv;
  const int g = u - b * p.Hkv;
  // C2 balanced: units with unit_S == 1 wrote their final outputs directly
  // from the main kernel and own no ws partials; skip them entirely.
  if (p.balanced && p.unit_S[b] <= 1) return;
  const int x = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;   // 16 (512-thread launch): halves the
  const int lane = threadIdx.x & 31;    // serial rows-per-warp latency chain
  // gridDim.z packs (row block, 64-column slice): rows are split across CTAs
  // as well as columns so rows-per-warp drops to ~1 and every SM works (pure
  // latency kernel; serial rows-per-warp paced the small-grid cases).
  const int nslice = (Dv + 63) / 64;
  const int rb = blockIdx.z / nslice;
  const int cslice = (blockIdx.z - rb * nslice) * 64;
  if (cslice >= Dv) return;
  const int ncl = min(64, Dv - cslice);      // cols in this slice
  const int ncol = ncl / 32;                 // per-lane cols (1 or 2)

  // As in the main kernels: param-space geometry when prepare cached it. Under
  // PDL these reads already overlap the main kernel's teardown, so the gain here
  // is expected to be small; it removes the traffic either way.
  const int q0 = p.geo_valid ? p.geo_cu[b] : p.cu_q[b];
  const int qlen = (p.geo_valid ? p.geo_cu[b + 1] : p.cu_q[b + 1]) - q0;
  const int R = qlen * p.ratio;
  const int tiles = (R + rt - 1) / rt;
  if (x >= tiles) return;
  const int r0 = x * rt;
  const int rbase = rb * rpb;
  const int rows_tile = min(rt, R - r0);
  if (rbase >= rows_tile) return;
  const int rows = min(rpb, rows_tile - rbase);   // rows owned by this CTA
  const int nunits = p.B * p.Hkv;
  const int S = p.S;
  const int colbase = lane * ncol;
  // bf16 partials: m/d live at word offset Dv/2 in the row (O occupies Dv/2
  // packed words), and a column slice of 64 bf16 is 32 words, so every cslice
  // used as a WORD offset becomes cslice/2. A lane's two columns (colbase =
  // lane*ncol, ncol=2) are exactly one 32-bit word at index `lane`.
  const int mdoff = Dv / 2;
  // C9 PDL: this kernel may be launched with programmatic stream
  // serialization (its CTAs spawn during the main kernel's teardown). Block
  // before any workspace/output-dependent access; index math and input
  // (cu_q/seq_lens) reads above are launch-independent and overlap the
  // primary grid's drain.
  asm volatile("griddepcontrol.wait;" ::: "memory");

  if (S <= 32) {
    const float* __restrict__ ws = p.ws;
    const int64_t split_stride = (int64_t)nunits * p.T * (int64_t)(rt * stride);
    const float* tilebase = ws + (int64_t)u * p.T * (int64_t)(rt * stride)
                          + (int64_t)x * (rt * stride);
    int r = warp;
    float mown = NEG_INF_F, down = 0.f;
    if (r < rows && lane < S) {
      const float2 md = *reinterpret_cast<const float2*>(
          tilebase + (int64_t)lane * split_stride + (int64_t)(rbase + r) * stride + mdoff);
      mown = md.x; down = md.y;
    }
    while (r < rows) {
      const int rn = r + nwarps;
      // Prefetch the next row's m/d so its DRAM/L2 latency overlaps with this
      // row's reduce chain and split accumulation.
      float mown_n = NEG_INF_F, down_n = 0.f;
      if (rn < rows && lane < S) {
        const float2 md = *reinterpret_cast<const float2*>(
            tilebase + (int64_t)lane * split_stride + (int64_t)(rbase + rn) * stride + mdoff);
        mown_n = md.x; down_n = md.y;
      }
      float M = mown;
#pragma unroll
      for (int s = 16; s; s >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, s));
      const float scown = (down != 0.f) ? ex2_fast(mown - M) : 0.f;
      float D = down * scown;
#pragma unroll
      for (int s = 16; s; s >>= 1) D += __shfl_xor_sync(0xffffffffu, D, s);
      float acc0 = 0.f, acc1 = 0.f;
      const float* rowr = tilebase + (int64_t)(rbase + r) * stride + (cslice / 2);
      for (int cc0 = 0; cc0 < S; cc0 += 4) {
        const float s0 = __shfl_sync(0xffffffffu, scown, cc0);
        const float s1 = __shfl_sync(0xffffffffu, scown, cc0 + 1);
        const float s2 = __shfl_sync(0xffffffffu, scown, cc0 + 2);
        const float s3 = __shfl_sync(0xffffffffu, scown, cc0 + 3);
        if (s0 != 0.f) {   // empty splits self-clean (down == 0)
          const float2 v = bf16x2_to_f32x2(*reinterpret_cast<const uint32_t*>(
              rowr + (int64_t)cc0 * split_stride + lane));
          acc0 += v.x * s0; acc1 += v.y * s0;
        }
        if (cc0 + 1 < S && s1 != 0.f) {
          const float2 v = bf16x2_to_f32x2(*reinterpret_cast<const uint32_t*>(
              rowr + (int64_t)(cc0 + 1) * split_stride + lane));
          acc0 += v.x * s1; acc1 += v.y * s1;
        }
        if (cc0 + 2 < S && s2 != 0.f) {
          const float2 v = bf16x2_to_f32x2(*reinterpret_cast<const uint32_t*>(
              rowr + (int64_t)(cc0 + 2) * split_stride + lane));
          acc0 += v.x * s2; acc1 += v.y * s2;
        }
        if (cc0 + 3 < S && s3 != 0.f) {
          const float2 v = bf16x2_to_f32x2(*reinterpret_cast<const uint32_t*>(
              rowr + (int64_t)(cc0 + 3) * split_stride + lane));
          acc0 += v.x * s3; acc1 += v.y * s3;
        }
      }
      float inv = p.bmm2 / D;
      float base2 = M + log2f(D) + lse_adj;
      const int gr = r0 + rbase + r;
      const int t = gr / p.ratio;
      const int h = g * p.ratio + (gr - t * p.ratio);
      if (p.has_sink) {
        const float sk = p.sinks[h] * LOG2E;
        const float tot = fmaxf(base2, sk) + log2f(1.f + exp2f(-fabsf(base2 - sk)));
        inv *= exp2f(base2 - tot);
        base2 = tot;
      }
      // lse is per (row, head): exactly one CTA per row may write it. With z
      // packing (row block, col slice) that is the first col slice of EVERY
      // row block (`blockIdx.z == 0` would drop rows past block 0).
      if (blockIdx.z % nslice == 0) {
        if (p.has_lse && colbase == 0) p.lse[(int64_t)(q0 + t) * p.H + h] = base2;
      }
      uint16_t* orow = p.out + ((int64_t)(q0 + t) * p.H + h) * Dv + cslice;
      if (ncol == 2) {
        __nv_bfloat162 pk = __float22bfloat162_rn(make_float2(acc0 * inv, acc1 * inv));
        *reinterpret_cast<uint32_t*>(orow + colbase) = *reinterpret_cast<uint32_t*>(&pk);
      } else {
        __nv_bfloat16 bv = __float2bfloat16(acc0 * inv);
        orow[colbase] = *reinterpret_cast<uint16_t*>(&bv);
      }
      r = rn; mown = mown_n; down = down_n;
    }
    return;
  }

  for (int r = warp; r < rows; r += nwarps) {
    const int gr = r0 + rbase + r;
    const int t = gr / p.ratio;
    const int h = g * p.ratio + (gr - t * p.ratio);
    const int64_t rowoff = (int64_t)(rbase + r) * stride + (cslice / 2);
    float M, D;
    float acc[2];
    acc[0] = 0.f; acc[1] = 0.f;
    {
      // S > 32: single-pass chunked online rescale (32 splits per chunk).
      M = NEG_INF_F; D = 0.f;
      for (int c0 = 0; c0 < S; c0 += 32) {
        const int n = min(32, S - c0);
        float mown = NEG_INF_F, down = 0.f;
        if (lane < n) {
          const float* src = p.ws + ((int64_t)(c0 + lane) * nunits + u) * p.T * (int64_t)(rt * stride)
                           + (int64_t)x * (rt * stride) + rowoff;
          mown = src[mdoff - cslice / 2];
          down = src[mdoff - cslice / 2 + 1];
        }
        float mc = mown;
#pragma unroll
        for (int s = 16; s; s >>= 1) mc = fmaxf(mc, __shfl_xor_sync(0xffffffffu, mc, s));
        if (mc > M) {   // strict >: -inf never > -inf, so no NaN rescale
          const float f = (M == NEG_INF_F) ? 0.f : ex2_fast(M - mc);
          acc[0] *= f; acc[1] *= f; D *= f; M = mc;
        }
        const float scown = (down != 0.f) ? ex2_fast(mown - M) : 0.f;
        float dc = down * scown;
#pragma unroll
        for (int s = 16; s; s >>= 1) dc += __shfl_xor_sync(0xffffffffu, dc, s);
        D += dc;
        for (int cc = 0; cc < n; ++cc) {
          const float sc = __shfl_sync(0xffffffffu, scown, cc);
          if (sc == 0.f) continue;   // empty splits self-clean (down == 0)
          const float* src = p.ws + ((int64_t)(c0 + cc) * nunits + u) * p.T * (int64_t)(rt * stride)
                           + (int64_t)x * (rt * stride) + rowoff;
          if (ncol == 2) {
            const float2 v = bf16x2_to_f32x2(
                *reinterpret_cast<const uint32_t*>(src + lane));
            acc[0] += v.x * sc; acc[1] += v.y * sc;
          } else if (ncol == 1) {
            // defensive: a single-column slice would split a bf16x2 word across
            // two lanes. ncl is 64 in production so this does not fire.
            const float2 v = bf16x2_to_f32x2(
                *reinterpret_cast<const uint32_t*>(src + lane / 2));
            acc[0] += ((lane & 1) ? v.y : v.x) * sc;
          }
        }
      }
    }
    float inv = p.bmm2 / D;
    float base2 = M + log2f(D) + lse_adj;
    if (p.has_sink) {
      const float sk = p.sinks[h] * LOG2E;
      const float tot = fmaxf(base2, sk) + log2f(1.f + exp2f(-fabsf(base2 - sk)));
      inv *= exp2f(base2 - tot);
      base2 = tot;
    }
    if (blockIdx.z % nslice == 0) {
      if (p.has_lse && colbase == 0) p.lse[(int64_t)(q0 + t) * p.H + h] = base2;
      // note: only lane 0 (colbase==0) writes lse; sink needs same base2 on all
      // slices, which each warp computes independently (identical inputs).
    }
    uint16_t* orow = p.out + ((int64_t)(q0 + t) * p.H + h) * Dv + cslice;
#pragma unroll
    for (int i = 0; i < 2; ++i)
      if (i < ncol) {
        __nv_bfloat16 bv = __float2bfloat16(acc[i] * inv);
        orow[colbase + i] = *reinterpret_cast<uint16_t*>(&bv);
      }
  }
}

// ---------------------------------------------------------------------------
// Host side: prepared configurations + launchers
// ---------------------------------------------------------------------------
static int g_sm_count = 0;

#if defined(FMHA_TC_TIMING) || defined(FMHA_DBG_ALLOC)
// Debug builds only: mapped-host buffer for the tc kernels' per-warp markers
// (FMHA_DBG_ALLOC: TCW wait-site markers with production codegen) and clock64
// accounting (FMHA_TC_TIMING). Production defines neither -> p.dbg stays null.
static unsigned* g_dbg_buf = nullptr;
extern "C" void* fmha_fp8_dbg_ptr() { return g_dbg_buf; }
#endif

// Experiment-only dispatch overrides (production default: all unset -> 0).
// Read once per prepare on the host; no effect on numerics or run-time cost.
static int env_int(const char* name, int dflt) {
  const char* v = getenv(name);
  return v ? atoi(v) : dflt;
}

// Rows per combine CTA (gridDim.z packs row blocks x 64-column slices). The
// combine kernel is pure latency: when the base grid (T x B*Hkv x 64-col
// slices) is small, warps walk rows serially and every row exposes its own
// split-load round trip. Row blocking trades CTAs for rows-per-warp.
// Shape-derived default from prof1 sweep16 (dv_ef3dff8147a0, bitwise parity):
// base <= 16 CTAs -> 8 (case5, 8 CTAs: 35.6->27.4us, -23%); <= 64 -> 32
// (case8: -6.4%; case2 balanced: -1.0%); else 64 (case9: -2.2%, d2: -0.1%;
// larger grids are neutral, and case3 shows over-blocking costs +1.4% at
// rpb<=32). rt=64 routes make rpb=64 a no-op (single row block).
// FMHA_COMBINE_RPB > 0 overrides for experiments. Re-read per prepare (baked
// into the launch grid at prepare/graph-capture time; never per run).
static int combine_rpb(int base_ctas, int rt) {
  int v = env_int("FMHA_COMBINE_RPB", 0);
  if (v <= 0) {
    // rpb 16 (not 32) for small grids, and for rt==64 slices. This was
    // measured NEUTRAL while the partials were f32 (dv_aee7028550de: c2/c3/c8
    // read 1.0008/1.0012/0.9934, inside a +-3.8% replicate floor) because the
    // combine was L2-bandwidth-bound at 4.1 TB/s and a finer grid only added
    // launch/scheduling cost. Halving the partials to bf16 removes that
    // bound, and the peer arm's same-job A/B then measures d1 0.7988 and
    // c2 0.9483 for the finer grid -- so the two changes are coupled and must
    // be adopted together. The rpb=64 case for base_ctas > 64 and rt==128 is
    // unchanged, which is why c9 (base_ctas 144, rt 128) reads the same in
    // both arms and stays off the finer grid where it is still near bandwidth.
    v = (base_ctas <= 16) ? 8
      : (base_ctas <= 64) ? 16
      : (rt == 64)        ? 16
                          : 64;
    // Do not serialise more than two rows per warp unless the finer grid would
    // exceed HALF a wave of 512-thread CTAs (4 resident per SM). Both directions
    // are measured (dv_aee7028550de, six arms, case0 replicate floor +-3.8% on
    // event minima): d3 at rpb=64 launches 96 CTAs -- 0.16 of a wave -- with four
    // serial rows per warp and 8/16/32 all read 0.906-0.914 against it, while c9 at
    // rpb=64 is 288 CTAs moving ~12.7MB at ~1.5 TB/s and 16/32/128 all read ~1.023
    // against it, so the finer grid costs where the combine is near bandwidth.
    // Halving stops at 32 rather than 16 on purpose: 32 -> 16 would also move
    // c2/c3/c8, which read 1.0008/1.0012/0.9934, inside the replicate floor and in
    // both directions. The band this fires in is narrow by construction -- rpb=64
    // needs base_ctas > 64, and the halving needs base_ctas*ceil(rt/32) <= 2*SMs,
    // i.e. base_ctas in (64, 74] at rt=128 or (64, 148] at rt=64.
  }
  if (v < 1) v = 1;
  if (v > 128) v = 128;
  return v;
}

namespace {
struct Prepared {
  Params p;
  dim3 grid;
  int dqk = 0;
  int nt = 0;   // mma.sync kv tile (template arg); 0 default per dqk
  int nstage = 0;  // tc K/V pipeline depth (impl 1/2/3)
  int mt_n = 0;    // megatile geometry for impl 6/7: 192 or 256 (0 = not MT)
  int pers = 0;    // r18: 1 = persistent-worker MT launch (impl 6/7, S==1)
  int* pers_list = nullptr;   // r18: private LPT aid-list buffer (cudaMalloc)
  int smem = 0;
  int impl = 0;   // 0 = mma.sync kernel, 1/3 = tcgen05 D128/D64, 2 = tc D256,
                  // 4/5 = tc2 cluster (exp), 6/7 = tc megatile D128/D64
  bool valid = false;
  // Optional launch graph (captured at prepare; replayed per run). The
  // identity fastpath in solution.py guarantees identical pointers between
  // prepares, so the baked-in arguments stay valid for every replay.
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
};
Prepared g_slots[8];
int g_slot_cursor = 0;

template <int DQK, int NT>
void ensure_smem_attr() {
  static int done = 0;
  if (!done) {
    constexpr int smem = 64 * (DQK + 16) + 3 * NT * (DQK + 16) + 3 * NT * DQK
                       + 4 * NT * (DQK + 8);
    cudaFuncSetAttribute(fmha_kernel<DQK, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    done = 1;
  }
}
}  // namespace

// Encode a 4-D tiled TMA map for one paged fp8 KV tensor viewed as bytes:
// global dims [D][page_rows][pages][heads], inner box exactly 128B (SW128
// span) x page rows. Returns 0 on success. cuTensorMapEncodeTiled is fetched
// via dlsym (libcuda is already loaded in-process by torch), so the build
// needs no driver-link flags.
namespace {
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t,
                                  void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*,
                                  CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeTiledFn encode_tiled_fn() {
  static EncodeTiledFn fn = [] {
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = RTLD_DEFAULT;
    return h ? (EncodeTiledFn)dlsym(h, "cuTensorMapEncodeTiled") : nullptr;
  }();
  return fn;
}
int encode_kv_map(CUtensorMap* m, const void* base, int D, int page,
                  long long pages, int Hkv, long long sn, long long sp,
                  long long sh) {
  EncodeTiledFn fn = encode_tiled_fn();
  if (!fn) return -1;
  // Inner box = one row-plane of bytes: 128B under SWIZZLE_128B (D>=128), or
  // the full 64B row under SWIZZLE_64B (D=64). The D64 encoding was resolved
  // empirically on this device by experiments/sw64probe.cu: TMA SWIZZLE_64B
  // + UMMA smem descriptor layout=4, LBO=0, SBO=512 reproduces the fp32
  // reference exactly (bad=0/8192); the matching loader/descriptor changes
  // live in fmha_tc.cuh (ROWB=64 compact rows).
  const bool d64 = (D < 128);
  const cuuint64_t gdim[4] = {(cuuint64_t)D, (cuuint64_t)page,
                              (cuuint64_t)pages, (cuuint64_t)Hkv};
  const cuuint64_t gstr[3] = {(cuuint64_t)sn, (cuuint64_t)sp, (cuuint64_t)sh};
  const cuuint32_t box[4] = {d64 ? 64u : 128u, (cuuint32_t)page, 1u, 1u};
  const cuuint32_t estr[4] = {1u, 1u, 1u, 1u};
  return fn(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, const_cast<void*>(base), gdim,
            gstr, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
            d64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS
             ? 0
             : -2;
}
}  // namespace

static int prepare_impl(Prepared& slot,
    const void* q, const void* k, const void* v, const int* table, const int* seqlens,
    const int* cu_q, void* out, float* lse, float* ws, const float* sinks,
    long long q_tok_stride, long long k_sp, long long k_sh, long long k_sn, long long v_sp,
    long long v_sh, long long v_sn, int Tstride, int page, int H, int Hkv, int B, int Tq,
    int dqk, int max_q_len, int max_kv_len, int window_left, int causal, int has_lse,
    int has_sink, float bmm1, float bmm2, long long ws_floats, int force_impl,
    long long kv_pages) {
  Params& p = slot.p;
  p.q = (const uint8_t*)q; p.k = (const uint8_t*)k; p.v = (const uint8_t*)v;
  p.table = table; p.seqlens = seqlens; p.cu_q = cu_q;
  p.out = (uint16_t*)out; p.lse = lse; p.ws = ws; p.sinks = sinks;
  p.q_tok_stride = q_tok_stride;
  p.k_sp = k_sp; p.k_sh = k_sh; p.k_sn = k_sn;
  p.v_sp = v_sp; p.v_sh = v_sh; p.v_sn = v_sn;
  p.Tstride = Tstride; p.page = page;
  p.page_log2 = 0;
  while ((1 << p.page_log2) < page) ++p.page_log2;
  if ((1 << p.page_log2) != page) return -13;   // page must be a power of two
  p.H = H; p.Hkv = Hkv; p.ratio = H / Hkv; p.B = B;
  p.window_left = window_left; p.causal = causal;
  p.has_lse = has_lse; p.has_sink = has_sink;
  p.dbg = nullptr;
#if defined(FMHA_TC_TIMING) || defined(FMHA_DBG_ALLOC)
  if (env_int("FMHA_DBG", 0)) {
#ifdef FMHA_DBG_ALLOC
    const int dbg_ctas = 512;
#else
    const int dbg_ctas = 64;
#endif
    if (!g_dbg_buf) {
      void* h = nullptr;
      if (cudaHostAlloc(&h, dbg_ctas * 128 * sizeof(unsigned),
                        cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess)
        g_dbg_buf = (unsigned*)h;
    }
    if (g_dbg_buf) {
      cudaMemset(g_dbg_buf, 0, dbg_ctas * 128 * sizeof(unsigned));
      cudaDeviceSynchronize();
      p.dbg = g_dbg_buf;   // UVA: host pointer is the device pointer
    }
  }
#endif
  p.exit_mode = 0;
  p.balanced = 0;   // C2 sets this below when per-unit splitting engages
  p.geo_valid = 0;  // the span-guard readback below sets it when it has the data
  p.force_corr = env_int("FMHA_FORCE_CORR", 0);
  p.force_lag1 = env_int("FMHA_FORCE_LAG1", 0);
  p.qk_scale = (float)((double)bmm1 * (double)LOG2E);
  p.bmm2 = bmm2;
  if (p.ratio < 1) return -10;

  if (!g_sm_count) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return -11;
    if (cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
      return -12;
    if (g_sm_count <= 0) g_sm_count = 64;
  }
  if (force_impl == 0) force_impl = env_int("FMHA_FORCE_IMPL", 0);
  const int M = 64;
  // NT defaults (sweep dv_329c8bacdd3b): D=256 NT=64 cut case2 main 110->94us
  // (deeper 3-stage pipeline, half the barrier/iteration count; spill-free
  // regs unchanged). D=64/128 keep 128/64 (nt64 on D64 regressed 1.13-1.16x).
  int NT = (dqk == 256) ? 64 : (dqk == 64 ? 128 : 64);
  bool nt32_pin_s1 = false;   // D=256 NT=32 inside the split-gate band: keep S=1
  // D=256 shapes that the span veto sends to mma.sync run fmha_kernel<256,64>,
  // which ptxas compiles to 96 bytes of stack with 48 spill stores and 72 spill
  // loads, where fmha_kernel<256,32> reports 80/8/8; halving the kv tile also
  // halves the per-stage smem. Measured across the 27 D=256 range probes
  // (dv_e6bd483caf67): NT=32 is a 0.789x geomean -- 0.532x on the causal
  // window-8/16 family, 0.721-0.726x on the four non-causal windowed probes the
  // guard_span veto moved onto this route, 0.583-0.946x on the v256l family.
  // It is NOT safe unscoped, because the split gate is `eff_max > NT`: a shape
  // with 32 < eff_max <= 64 flips from S=1 to S>1 and pays a whole second kernel
  // for a tiny kv range (v256_c36 22.82 -> 55.65us with S 1 -> 2; dispatch tables
  // for both settings in dv_261df8d8edc5), and that single regression outweighs
  // the three neighbouring probes NT=32 would win. So this fires only where the
  // split decision is provably identical at both NT (eff_max > 64, so the gate is
  // open either way) AND the shape is certain to take the mma.sync route, i.e.
  // guard_span <= span_lim -- the same quantity the veto tests, recomputed here
  // from max_kv_len, window_left, causal and M, all already in scope. The primary
  // veto clause is exactly this comparison, so a shape that satisfies it cannot
  // reach impl 1/2/3/6/7 whatever the row-share clause does. That excludes every
  // scored shape: c2, c6 and d2 are D=256 with guard_span 32769 / 8192 / 131072
  // against span_lim 384, so their NT stays 64 and this cannot reach either suite.
  if (dqk == 256) {
    const long long span_lim_nt = ((causal != 0) && (window_left < 0)) ? 384 : 224;
    long long gspan_nt = max_kv_len;
    long long eff_nt = max_kv_len;
    if (window_left >= 0) {
      const long long w = (long long)window_left + M + 1;
      gspan_nt = std::min<long long>(gspan_nt, w);
      if (causal) eff_nt = std::min<long long>(eff_nt, w);
    }
    // Two cases, and the second is why this is not one threshold. With
    // eff_nt > 64 the split gate `eff_max > NT` is open at BOTH widths, so NT=32
    // cannot change the split decision. In the band 32 < eff_nt <= 64 the gate is
    // closed at NT=64 and open at NT=32, so NT=32 alone would split a shape whose
    // whole visible kv range is <= 64 keys and pay a second kernel for it
    // (v256_c36 22.82 -> 55.65us). Pinning S=1 there reproduces exactly what
    // NT=64 does today -- a shape with eff_max <= 64 never splits at NT=64 -- while
    // still taking the narrower tile, which is what v256_c37/c48/c64 (same band,
    // S=1 kept by the wave-packing refinement) measured at 0.68-0.80x.
    if (gspan_nt <= span_lim_nt && eff_nt > 32) {
      NT = 32;
      nt32_pin_s1 = (eff_nt <= 64);
    }
  }
  // NT=32 tiny-chunk decode rule (sweep dv_ef81a4fb9578): when the NT64
  // split heuristic would leave <= 2 kv tiles per CTA chunk, the 3-stage
  // cp.async fill dominates CTA runtime; NT=32 halves the fill granularity
  // (d6 0.963x, d8 0.988x measured). Scoped OFF everywhere else: c3 1.041,
  // c5 1.012, d1 1.050, d5 1.488 (S 1->2 flip on the unsplit case), and
  // D64's nt64 1.110 on c8 -> require D128, a real split (preview S >= 2),
  // and preview chunk <= 128 keys. The preview mirrors the NT64 heuristic
  // below; the actual S selection then re-runs on the chosen NT.
  if (dqk == 128 && ws != nullptr) {
    long long eff64 = max_kv_len;
    if (causal && window_left >= 0)
      eff64 = std::min<long long>(eff64, (long long)window_left + M + 1);
    const long long nunits64 = (long long)B * Hkv;
    const long long tl64 = std::max<long long>(
        nunits64, (long long)Hkv * (((long long)Tq * p.ratio + M - 1) / M));
    if (eff64 > 64 && tl64 < 2LL * g_sm_count) {
      int s64 = std::min(
          std::max((int)((2LL * g_sm_count + tl64 - 1) / tl64),
                   (int)((eff64 + 767) / 768)),
          (int)std::min<long long>((eff64 + 63) / 64, 32LL));
      if (tl64 <= 4) s64 = std::min(s64, 16);
      if (s64 >= 2) {
        const long long ch64 = ((eff64 + s64 - 1) / s64 + 63) / 64 * 64;
        if (ch64 <= 128) {
          // Bound the split doubling: NT32 re-selects S on 32-key alignment
          // and must not add >50% more splits, or the extra combine partials
          // outweigh the fill saving (A/B dv_1a0d471a264c: d8 S9->16/18
          // REGRESSED 1.0225; d6 S2->3 won 0.9281 - kept).
          int s32 = std::min(
              std::max((int)((2LL * g_sm_count + tl64 - 1) / tl64),
                       (int)((eff64 + 383) / 384)),
              (int)std::min<long long>((eff64 + 31) / 32, 32LL));
          if (tl64 <= 4) s32 = std::min(s32, 16);
          if (2 * s32 <= 3 * s64) NT = 32;
        }
      }
    }
  }
  const int force_nt = env_int("FMHA_FORCE_NT", 0);
  if (force_nt > 0) {
    const bool ok = (dqk == 64 && (force_nt == 64 || force_nt == 128)) ||
                    (dqk == 128 && (force_nt == 32 || force_nt == 64 || force_nt == 128)) ||
                    (dqk == 256 && (force_nt == 32 || force_nt == 64));
    if (ok) NT = force_nt;
  }
  const int nunits = B * Hkv;
  const int rows_per_unit = max_q_len * p.ratio;
  p.T = std::max(1, (rows_per_unit + M - 1) / M);

  // Effective kv-range upper bound per q-tile (drives split chunk sizing).
  long long eff_max = max_kv_len;
  if (causal && window_left >= 0)
    eff_max = std::min<long long>(eff_max, (long long)window_left + M + 1);
  long long tiles_lb = std::max<long long>(
      nunits, (long long)Hkv * (((long long)Tq * p.ratio + M - 1) / M));
  long long target = 2LL * g_sm_count;
  int S = 1;
  if (ws != nullptr && eff_max > NT && tiles_lb < target && !nt32_pin_s1) {
    int S_t = (int)((target + tiles_lb - 1) / tiles_lb);
    // Work-balance term: keep per-split chunks to ~12 kv tiles so long
    // sequences do not dominate the critical path (mixed-length batches).
    int S_w = (int)((eff_max + (long long)NT * 12 - 1) / ((long long)NT * 12));
    // Cap 32 is measured-optimal for this kernel family: S>32 (profile
    // dv_490ede3e6ab4) regressed case5 90->206us because combine latency
    // scales with the number of ACTIVE splits (dependent per-row loads) while
    // split kernels were already latency-bound, not split-count-bound (d7
    // S 32->45 gained only ~5%, within noise). Chunked combine path above
    // remains available if combine ever becomes split-parallel.
    //
    // That measurement was taken when the combine serialised up to eight rows per
    // warp on four CTAs. It is row-blocked now, so 32 is a floor rather than a
    // ceiling: lift it only where BOTH (a) every split still owns at least 16 NT
    // tiles of work, so the extra splits are not launch overhead, and (b) the
    // resulting grid still fits one resident wave, so no split pays a second wave's
    // latency. Pre-join this exact form gave five long-kv few-unit probes
    // 0.499-0.933x with no scored shape touched (their S is ws-limited well below
    // 32, or their eff_max is under 512*NT so s_chunk never exceeds it).
    const long long s_chunk = eff_max / (16LL * NT);
    const long long ctas1 = std::max<long long>(1, (long long)nunits * p.T);
    const long long smem_kb = (long long)M * (dqk + 16) + 3LL * NT * (dqk + 16)
                            + 3LL * NT * dqk + 4LL * NT * (dqk + 8);
    const long long resident = (long long)(g_sm_count > 0 ? g_sm_count : 148) *
        std::max(1, std::min(2, (int)((227LL * 1024) / std::max(1LL, smem_kb))));
    const long long s_wave = (resident + ctas1 - 1) / ctas1;
    const long long s_ceil = std::max<long long>(32LL, std::min(s_chunk, s_wave));
    int S_kv = (int)std::min<long long>((eff_max + NT - 1) / NT, s_ceil);
    S = std::min(std::max(S_t, S_w), S_kv);
    // The `tiles_lb <= 4 -> S <= 16` cap that used to sit here is GONE, measured
    // rather than argued (dv_20ffab290252, same job, two trees, minima AND wall):
    // case5 -- one of only two scored shapes with tiles_lb <= 4 -- reads 24.29us
    // against 26.40 capped on event minima and 30.34 against 32.70 on wall, i.e.
    // -7.2% on both columns, with rel_l2 moving 0.00022 -> 0.00020 (S=32 splits
    // the kv range finer, so it is also slightly more accurate). The cap's own
    // citation (dv_329c8bacdd3b, "case5 S32->16 = 0.87x") was measured on a
    // combine that serialised up to 8 rows per warp; this combine is row-blocked,
    // which is exactly the change that flips the sign, and the arm-b chain measured
    // the same flip on its own row-blocked combine before the join. The other
    // tiles_lb<=4 scored shape is d8. Nothing else moves: tiles_lb > 4 for the
    // other eighteen, and c2/c3/c9/c0 all read inside the job's replicate noise.
    long long per = (long long)nunits * p.T * M * (dqk / 2 + 2);
    while (S > 1 && (long long)S * per > ws_floats * 3 / 4) --S;

    // Wave-packing refinement (sweep dv_a215ba7105f4): the ceil()-based S_t/S_w
    // heuristics can overshoot or undershoot the occupancy-slot budget
    // (slots = SMs x CTAs/SM from the smem footprint), spilling uniform chunks
    // into an extra wave: d1 S19 -> 304 > 296 slots (2 waves), d7 S32 -> 512
    // (2 waves), d0 S22 -> 4.76 waves vs S32's tighter 7x1024 packing. Choose
    // S = argmin of the modeled makespan
    //   cost(S) = ceil(U*S/slots) * chunk_keys(S) * kt + combine_us(S)
    // over ws-feasible S. Measured: d7 S18 0.883x, d0 S32 0.844x, d1 S16
    // 0.987x, c3 S9 1.011 (launch-floor bound, neutral); controls c5/c9
    // unchanged. Guards: (1) apply only when modeled main time >= 20us --
    // tiny decode shapes are dominated by fixed per-CTA overheads the
    // chunk-proportional model cannot see (d6/d8/c8/d3 stay on the validated
    // heuristic); (2) B>=2 batches can be length-heterogeneous (host sees only
    // max_kv_len; d0 spans kv 1..32768) and finer chunks are more
    // work-conserving than the uniform model suggests -> take the DEEPEST S
    // within 10% of the modeled min (d0 -> S32 measured, uniform argmin would
    // say S9; B=1 is homogeneous -> strict argmin, matching d2-tc S4); (3) 2%
    // hysteresis vs the heuristic pick keeps measured ties stable (c5 S16).
    // kt calibration only weights main-vs-combine; waves x chunk is scale-
    // invariant.
    {
      const int smem_b = M * (dqk + 16) + 3 * NT * (dqk + 16) + 3 * NT * dqk
                       + 4 * NT * (dqk + 8);
      const int occ = std::max(1, std::min(2, (int)((227LL * 1024) / smem_b)));
      const long long slots = (long long)g_sm_count * occ;
      const long long U = (long long)nunits * p.T;
      const long long wsb = ws_floats * 3 / 4;
      const double kt = (double)M * dqk * 4.0 / ((256.0 / occ) * 2.0 * 2.03e9) * 1e6;
      const double rows = (double)U * M;
      const double bw = 1.45e6;   // DRAM bytes/us (measured ceiling)
      auto chunk_of = [&](int s) -> long long {
        return (s > 1)
            ? (long long)NT * ((eff_max + (long long)s * NT - 1) / ((long long)s * NT))
            : eff_max;
      };
      auto cost = [&](int s) -> double {
        const long long waves = (U * s + slots - 1) / slots;
        double c = (double)waves * (double)chunk_of(s) * kt;
        if (s > 1)
          c += ((double)s * rows * (dqk / 2 + 2) * 4.0 + rows * dqk * 2.0) / bw;
        return c;
      };
      const double cur = cost(S);
      const double main_cur =
          (double)((U * S + slots - 1) / slots) * (double)chunk_of(S) * kt;
      if (main_cur >= 20.0) {
        const int s_hi = S_kv;   // the <=4-tile cap is gone (see above)
        int bmin = S;
        double bmc = cur;
        for (int s = 1; s <= s_hi; ++s) {
          if (s > 1 && (long long)s * per > wsb) continue;
          const double c = cost(s);
          if (c < bmc) { bmc = c; bmin = s; }
        }
        int best = bmin;
        // Band rule only for MULTI-WAVE argmin grids: that is where chunk
        // granularity buys work-conservation across heterogeneous batches
        // (d0: 4.76 waves -> S32 measured 0.844x). Single-wave grids are fully
        // resident -- deeper S there only adds combine traffic (d1: s16 0.987x
        // measured vs s18 1.000x).
        if (B >= 2 && (U * bmin + slots - 1) / slots >= 2) {
          for (int s = s_hi; s >= 1; --s) {
            if (s > 1 && (long long)s * per > wsb) continue;
            if (cost(s) <= 1.10 * bmc) { best = s; break; }
          }
        }
        if (best != S && cost(best) > 0.98 * cur) best = S;
        S = best;
      }
    }
  }
  const int force_s = env_int("FMHA_FORCE_S", 0);
  if (force_s >= 1 && ws != nullptr) {
    S = force_s;
    const long long s_kv_max = (eff_max + NT - 1) / NT;
    if (S > s_kv_max) S = (int)std::max<long long>(s_kv_max, 1);
    long long per = (long long)nunits * p.T * M * (dqk / 2 + 2);
    while (S > 1 && (long long)S * per > ws_floats * 3 / 4) --S;
  }
  p.S = S;
  if (S > 1) {
    long long cl = (eff_max + S - 1) / S;
    cl = ((cl + NT - 1) / NT) * NT;
    p.chunk = (int)std::max<long long>(cl, NT);
  } else {
    p.chunk = max_kv_len;
  }

  // tcgen05 path: D=64/128/256 compute-bound shapes. Eligible when 128-row
  // tiles alone give enough CTAs, or when the flop estimate says the tc rate
  // wins over the split mma.sync path. M=128 tiles there; T recomputed.
  // D=64 runs through the same 128B-row pipeline with DV=64 (mma.sync tops
  // out at ~256 MAC/cyc/SM on SM103 vs ~2048 for tcgen05 - ncu dv_329c8bacd:
  // case4 tensor-saturated at 129.9% pipe active, so D64 heavy shapes move to
  // impl 3 with exact (unpadded) QK K-steps and PV N=64).
  int impl = 0;
#ifndef FMHA_NO_TC
  // FMHA_FORCE_MMA=1 (debug): bypass every tc route -> mma.sync kernel.
  if ((dqk == 64 || dqk == 128 || dqk == 256) && force_impl != 1 &&
      !env_int("FMHA_FORCE_MMA", 0)) {
    long long tiles_lb_128 = std::max<long long>(
        nunits, (long long)Hkv * (((long long)Tq * p.ratio + 127) / 128));
    double flops_est = 8.0 * (double)Tq * (double)max_kv_len * (double)H * (double)dqk;
    bool heavy = flops_est >= 1.0e10;
    // C6 decode-class routing inputs: per-batch effective kv spans (single
    // seq_lens readback; frozen-input assumption identical to C2's).
    // job18 measurements (tc forced on mma shapes): d7 0.568x (tc+TMA vs
    // cp.async-bound mma at 1.55TB/s), d0 0.849x, case3 0.857x; case5/case8
    // LOST (1.11-1.12x) - tc's 4x32KB stage fill dominates short spans with
    // few units -> span-size gates below.
    // Short-span tc correctness veto (join 20261022 increment 2; mechanism and
    // thresholds from arm-b, reproduced and re-verified on THIS tree by
    // experiments/span_probes.py, dv_22b501067fa9 + rerun): tc quantizes P to
    // e4m3 and short spans average little of that error away, so legal
    // shape_range shapes exceed the frozen rel_l2 0.01 bound on tc while the
    // mma.sync route (P kept in fp16) passes the same probes at ~20x margin
    // (zero mismatched elements). Measured here, 5 seeds, uniform causal
    // D256: L160 0.010726, L192 0.010480, L224 0.010225 FAIL; L256 0.009884
    // passes with only 1.2% headroom (inside seed spread); q128 windowed
    // D256 kv4k/kv64k 0.012924/0.012575 FAIL; mixed-batch 87.5%-short-rows
    // D256 0.010566 FAIL; D128 mixed batch passes at 0.009647 = 3.5%
    // headroom. My v6a numerics are much tighter than arm-b's v2 on the
    // windowed D256 and dqk<256 elementwise classes (their six dqk<256
    // failures do NOT reproduce here; B_w16/32_kv512 pass at 16x), so only
    // the classes above bind -- but the thresholds are kept at arm-b's
    // five-seed band-walk values (causal-bare D256 384, other D256 224,
    // dqk<256 256; row-share 40% D256 / 62.5% else) since my margins sit
    // inside their measured gaps. Cost: unlisted-domain latency moved to the
    // ~1/8-rate mma D256 path / mma decode paths, reported as a domain
    // regression where scored shapes move. FMHA_C6=0 disables the readback
    // and with it the row-share clause, falling back to the eff_max form.
    const bool causal_bare = (causal != 0) && (window_left < 0);
    const long long span_lim =
        (dqk == 256) ? (causal_bare ? 384 : 224) : 256;
    // guard_span, NOT eff_max, is what this veto has to test. eff_max caps on
    // window_left only when CAUSAL (fmha_fp8.cu:1016) because that is what split
    // chunk sizing needs; but for a NON-CAUSAL windowed shape every row still sees
    // only window_left+1 keys, and eff_max = max_kv_len hides that completely. The
    // assumption in the comment below eb -- that such shapes "pass tc with wide
    // margins" -- is true of the normal distribution and false of `peaked`, which no
    // dispatch rule can observe. Measured on the unified join-20261023 candidate
    // (dv_142ffa0f861f, 159 shape_range probes): 4 FAIL, all of them non-causal with
    // window 8, all on the 0.01 relative-L2 bound --
    //     x64_w8n_p 0.012377 (60999 mismatched)   x64_w8n_ps 0.012377 (60959)
    //     x256_w8n_p 0.010580 (54850)             v256_psw8n 0.010580 (54812)
    // Their geometry is identical to v_win8 / v256_w8n / x256_w8n_s, which pass at
    // 0.006989-0.008627, so the only separating variable is the distribution. Capping
    // on the window regardless of causality catches all four and costs the passing
    // shapes in that family their tc route, which is the price of a veto that cannot
    // see the data. guard_span is used ONLY here: chunk sizing keeps eff_max, so
    // nothing but the route moves.
    long long guard_span = max_kv_len;
    if (window_left >= 0)
      guard_span = std::min<long long>(guard_span, (long long)window_left + M + 1);
    long long emax = 0, emin = (1LL << 62);
    long long rmax = 0, rmin = (1LL << 62);
    int nshort = 0;
    long long rows_short = 0, rows_all = 0;
    bool spans_ok = false, qspans_ok = false;
    if (B >= 1 && B <= 32 && ws != nullptr && env_int("FMHA_C6", 1)) {
      int sl_h[32];
      if (cudaMemcpy(sl_h, seqlens, (size_t)B * sizeof(int),
                     cudaMemcpyDeviceToHost) == cudaSuccess) {
        // Per-sequence query lengths so the short-span test weights by ROWS
        // (tolerance counts output elements; one extra <=132B readback next
        // to the seq_lens one, both in prepare).
        int cq_h[33];
        qspans_ok = cu_q != nullptr &&
                    cudaMemcpy(cq_h, cu_q, (size_t)(B + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost) == cudaSuccess;
        spans_ok = true;
        // Publish the values this block just read to the kernel through param
        // space (Params::geo_valid). Same integers the device would load, so
        // outputs are bit-identical; it only removes the prologue's first cold
        // dependent hop. Guarded on qspans_ok so a failed cu_q readback leaves
        // geo_valid == 0 and the pointer path in force.
        if (qspans_ok) {
          for (int b = 0; b <= B; ++b) p.geo_cu[b] = cq_h[b];
          for (int b = 0; b < B; ++b) p.geo_sl[b] = sl_h[b];
          p.geo_valid = 1;
        }
        for (int b = 0; b < B; ++b) {
          long long raw = sl_h[b] < 1 ? 1 : sl_h[b];
          long long e = raw;
          if (window_left >= 0)
            e = std::min<long long>(e, (long long)window_left + 129);
          if (e < 1) e = 1;
          if (e > emax) emax = e;
          if (e < emin) emin = e;
          if (raw > rmax) rmax = raw;
          if (raw < rmin) rmin = raw;
          // This sequence's own effective span on the SAME definition eff_max
          // uses for chunk sizing: the window caps only the LEFT extent and only
          // when causal. The precision veto above deliberately uses a different,
          // window-aware quantity (guard_span); this one stays on eff_max's
          // definition because it feeds the row-share clause, whose thresholds
          // (40% / 62.5%) were measured against it.
          long long eb = raw;
          if (causal && window_left >= 0)
            eb = std::min<long long>(eb, (long long)window_left + M + 1);
          long long rows_b = qspans_ok ? (long long)(cq_h[b + 1] - cq_h[b])
                                       : (long long)max_q_len;
          if (rows_b < 1) rows_b = 1;
          rows_all += rows_b;
          if (eb <= span_lim) {
            ++nshort;
            rows_short += rows_b;
          }
        }
      }
    }
    // Routing heterogeneity uses RAW kv variance (window-capped spans can look
    // homogeneous when every unit is capped: case3 kv 2048..32768 -> capped
    // 2048..2176, yet the tc route measured 0.857x). C2's split-balance gate
    // keeps its own capped-eff variance rule (that decides S_b usefulness).
    const bool spans_het = spans_ok && B >= 2 && rmax * 2 >= rmin * 5;
    // c6_long128 guard: tc split caps at S=8 for D128, so long-span routing
    // needs enough base CTAs that S=8 still approaches a resident wave (d7:
    // 16x8=128 of 148). Few-unit long-kv shapes keep mma's deeper S<=32.
    const long long tiles128_lb =
        std::max<long long>(1, ((long long)max_q_len * p.ratio + 127) / 128);
    const bool c6_long128 = spans_ok && dqk == 128 && emax >= 16384 &&
        (long long)nunits * tiles128_lb * 8 >= g_sm_count / 2;
    // Unit-parallelism guard: tc's 4x32KB stage fill is only amortized when
    // units x row-tiles give a reasonable CTA base (case3: 32 -> 0.857x;
    // case5/d1 with 4 CTAs lost or stayed neutral -> keep mma there).
    const bool c6_units = (long long)nunits * tiles128_lb >= 16;
    const bool c6_d64_het = spans_ok && dqk == 64 && spans_het &&
                            emax >= 4096 && c6_units;
    const bool c6_128_het = spans_ok && dqk == 128 && spans_het &&
                            emax >= 1024 && c6_units;
    // D=256 goes tc when heavy OR the kv span can amortize split chunks
    // (>= 4096 keys): the mma.sync D256 path runs at ~1/8 the tensor rate
    // (256 vs 2048 MAC/cyc/SM) and pads rows<=16 decode to M=64 (4x waste);
    // tc + deep kv splits won case2 0.54x (sweep dv_a71ad30d669d, S=32).
    // NOTE: case2's flops_est is 8.6e9 (< heavy), hence the kv-length arm.
    // tc2 (2-CTA cluster) experimental selection: force_impl 4 = D128,
    // 5 = D64. Not production-gated until sweeps prove it (fallback = 1/3).
    const bool tc2_force = (force_impl == 4 && dqk == 128) ||
                           (force_impl == 5 && dqk == 64);
    // Row-share clause: a mixed-length batch whose OUTPUT ROWS are mostly
    // short-span fails (or shaves) the frozen bounds on tc even when eff_max
    // (a batch MAXIMUM) is long; weighted by rows because the tolerance
    // counts output elements, and by each sequence's OWN span (eb above).
    // Thresholds at arm-b's measured gaps: 40% for D256 (rel_l2-bound),
    // 62.5% for dqk<256 (elementwise-bound; my E_het_d128 at 87.5% share
    // holds only 3.5% rel_l2 headroom). Force hooks stay unconditional so
    // experiments can still pin tc.
    const bool short_rows = qspans_ok
        ? (dqk == 256 ? (rows_short * 5 > rows_all * 2)
                      : (rows_short * 8 > rows_all * 5))
        : (dqk == 256 ? (nshort * 5 > B * 2) : (nshort * 8 > B * 5));
    // FMHA_NO_SPAN_GUARD=1 (dev-only): disable the veto for A/B cost
    // measurement; restores the short-span tc correctness exposure. Unlike
    // FMHA_C6=0 this leaves the c6 hetero routing arms intact.
    const bool span_short = !env_int("FMHA_NO_SPAN_GUARD", 0) &&
                            ((guard_span <= span_lim) || (spans_ok && short_rows));
    if (force_impl == 2 || force_impl == 3 || tc2_force ||
        (!span_short &&
         (c6_long128 || c6_d64_het || c6_128_het ||
          (tiles_lb_128 >= 48 && (tiles_lb_128 >= g_sm_count || heavy)) ||
          (dqk == 256 && (heavy || max_kv_len >= 4096))))) {
      impl = tc2_force ? ((force_impl == 4) ? 4 : 5)
                       : ((dqk == 128) ? 1 : (dqk == 256 ? 2 : 3));
      S = 1;
      p.S = 1;
      p.chunk = max_kv_len;
      p.T = std::max(1, (max_q_len * p.ratio + 127) / 128);
      // tc split-KV: when 128-row tiling leaves SMs idle (case9: 72 CTAs,
      // d4: 64 on 148 SMs), split kv into chunk columns merged by
      // fmha_combine (rt=128, lse_adj=-8). force_impl==3 forces split
      // eligibility (probe coverage); force_impl==2 stays S=1.
      if (ws != nullptr && force_impl != 2) {
        long long tc_ctas = (long long)p.T * nunits;
        long long eff = (long long)max_kv_len;
        if (causal && window_left >= 0)
          eff = std::min<long long>(eff, (long long)window_left + 128 + 1);
        long long kv_tiles = (eff + 127) / 128;
        long long s_by_sm = (long long)g_sm_count / std::max<long long>(tc_ctas, 1LL);
        if (force_impl == 3) s_by_sm = std::max<long long>(s_by_sm, 2);
        // D=256 decode (tiny base CTA count): fill SMs with deep kv splits.
        // Chunk floor 1024 keys (8 TN tiles) via the amortization gate below
        // keeps prefill shapes safe (d4: eff/1024 = 1 -> no split, matching
        // the measured 52.4->74.2us regression at 4-tile chunks).
        // Sweep dv_a71ad30d669d case2: S32 45.3us (0.54x), S16 0.68x,
        // S8 1.11x vs mma-nt64 84.2us.
        long long s_want = s_by_sm, s_cap = 8;
        if (dqk == 256 && s_by_sm >= 4) {
          const long long ctas = std::max<long long>(tc_ctas, 1LL);
          s_want = (2LL * g_sm_count + ctas - 1) / ctas;
          s_cap = 32;
        }
        int S_tc = (int)std::min<long long>(std::min(s_want, kv_tiles), s_cap);
        // Amortization gate: each chunk needs >= 16 kv tiles (2048 keys) on
        // D<=128, >= 8 (1024 keys) on D256, or pipeline fill/drain + combine
        // dominate (measured: d4 with 4-tile chunks regressed 52.4->74.2us,
        // dv_f550c2684e0c; case9 with 65-tile chunks gained 1.58x).
        // force_impl==3 bypasses (probe coverage). C6-routed decode classes
        // measured best at 4-5 tile chunks (case3 probe S=4 -> 24.6us vs
        // S=1 38.9us): relax their floor to 512 keys.
        if (force_impl != 3) {
          const long long amort = (dqk == 256) ? 1024
                                  : ((c6_128_het || c6_d64_het) ? 512 : 2048);
          S_tc = (int)std::min<long long>(S_tc, eff / amort);
        }
        // Wave-packing argmin when the base tc grid already fills all SMs
        // (s_by_sm == 0 -> historically S=1): fractional-wave quantization on
        // near-uniform chunks is recoverable with moderate splits (d2: 256
        // CTAs = 1.73 waves quantized to 2; S=4 packs 1024 CTAs into 7
        // near-full waves, measured 0.933x, sweep dv_a215ba7105f4). The
        // combine-traffic term vetoes splits whose main gain is thin (case7
        // S2: -32us main vs +59us combine; d9/c1 ws-infeasible or tied) and
        // the 2048-key chunk floor preserves pipeline amortization (c0/c4
        // stay S=1). D256 keeps the deep-split rule above (measured c2
        // S18/S16 = 1.23/1.33x WORSE than S32).
        if (s_by_sm == 0 && dqk != 256 && force_impl != 3) {
          const long long slots = g_sm_count;              // tc: 1 CTA/SM
          const long long U = (long long)p.T * nunits;
          const long long per_t = (long long)nunits * p.T * 128 * (dqk / 2 + 2);
          const long long wsb = ws_floats * 3 / 4;
          const double kt = 128.0 * dqk * 4.0 / (2048.0 * 2.0 * 2.03e9) * 1e6;
          const double rows = (double)U * 128;
          const double bw = 1.45e6;   // DRAM bytes/us (measured ceiling)
          auto cost = [&](int s) -> double {
            const long long ch = (s > 1)
                ? 128LL * ((eff + 128LL * s - 1) / (128LL * s)) : eff;
            const long long waves = (U * s + slots - 1) / slots;
            double c = (double)waves * (double)ch * kt;
            if (s > 1)
              c += ((double)s * rows * (dqk / 2 + 2) * 4.0 + rows * dqk * 2.0) / bw;
            return c;
          };
          int best = 1;
          double bestc = cost(1);
          for (int s = 2; s <= 8 && (long long)s <= kv_tiles; ++s) {
            if ((long long)s * per_t > wsb) continue;
            const long long ch = 128LL * ((eff + 128LL * s - 1) / (128LL * s));
            if (ch < 2048) continue;   // amortization floor (16 kv tiles)
            const double c = cost(s);
            if (c < bestc) { bestc = c; best = s; }
          }
          if (best > 1 && bestc > 0.98 * cost(1)) best = 1;
          S_tc = best;
        }
        // Experiment hook: FMHA_FORCE_S overrides the tc split count too
        // (bypasses s_by_sm and the amortization gate; chunk alignment and
        // the ws-fit shrink below still apply).
        const int fs_tc = env_int("FMHA_FORCE_S", 0);
        if (fs_tc >= 2) S_tc = (int)std::min<long long>(fs_tc, kv_tiles);
        long long per = (long long)nunits * p.T * 128 * (dqk / 2 + 2);
        while (S_tc > 1 && (long long)S_tc * per > ws_floats * 3 / 4) --S_tc;
        if (S_tc >= 2) {
          S = S_tc;
          p.S = S_tc;
          long long cl = (eff + S_tc - 1) / S_tc;
          cl = ((cl + 127) / 128) * 128;
          p.chunk = (int)std::max<long long>(cl, 128);
        }
      }
    }
  }
#endif
  // C1: TMA K/V loader eligibility. tc kernels with 128B rows per plane can
  // receive whole page segments of a kv tile via cp.async.bulk.tensor.4d
  // instead of 1024-2048 cp.async16 + address ALU per tile (job1 measurement:
  // loader issue/latency paced every heavy tc case; w9-11 work ~= CTA wall).
  // The 4-D map carries the row stride (k_sn = Hkv*D for the transposed HND
  // cache), so rows need NOT be contiguous - only 16B-multiple strides and
  // 16B-aligned bases. Gated to window_left < 0 so lo == 0 and every kv tile
  // is page-aligned (page in {16..128} divides TN=128); D64 keeps cp.async
  // (64B rows cannot fill a SW128 box). Falls back to cp.async otherwise.
  p.use_tma = 0;
#ifndef FMHA_NO_TC
  // FMHA_NO_TMA=1 (debug): force the cp.async loader path.
  if ((impl == 1 || impl == 2 || impl == 3) && window_left < 0 && kv_pages > 0 &&
      !env_int("FMHA_NO_TMA", 0) &&
      dqk >= 64 && page >= 16 && page <= 128 &&
      (reinterpret_cast<uintptr_t>(k) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(v) & 15) == 0 &&
      (k_sp % 16) == 0 && (k_sh % 16) == 0 && (k_sn % 16) == 0 &&
      (v_sp % 16) == 0 && (v_sh % 16) == 0 && (v_sn % 16) == 0) {
    p.use_tma = (encode_kv_map(&p.kmap, k, dqk, page, kv_pages, Hkv,
                               p.k_sn, p.k_sp, p.k_sh) == 0 &&
                 encode_kv_map(&p.vmap, v, dqk, page, kv_pages, Hkv,
                               p.v_sn, p.v_sp, p.v_sh) == 0)
                    ? 1
                    : 0;
  }
#endif

  // C2: per-unit work-balanced split-KV. Uniform-chunk splitting sizes every
  // chunk from max_kv_len, so mixed-length batches either leave the long unit
  // pacing the makespan (S too small) or spawn many empty/tiny splits (S sized
  // for the longest). Instead each batch b gets its own split count S_b such
  // that every chunk holds ~kappa keys, and the (unit, x-tile, split) space is
  // LINEARIZED into grid.x with exact per-batch prefixes -> zero empty CTAs.
  // Device side derives chunk bounds as ceil(ntiles(u,x)/S_b), which also
  // balances the causal per-x-tile variation exactly. seq_lens/cu_q are
  // read back ONCE at prepare (frozen inputs under the cached-handle ABI;
  // the harness restores only out/lse/workspace between calls).
  // Numerics: chunk boundaries differ from the uniform policy, so split
  // rounding changes within the public tolerance (as any S change does).
  if (ws != nullptr && impl <= 3 && B >= 1 && B <= 32 &&
      env_int("FMHA_BALANCED", 1)) {
    int sl_h[32], cq_h[33];
    bool ok = cudaMemcpy(sl_h, seqlens, (size_t)B * sizeof(int),
                         cudaMemcpyDeviceToHost) == cudaSuccess;
    ok = ok && cudaMemcpy(cq_h, cu_q, (size_t)(B + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    if (ok) {
      const int TMp = (impl == 0) ? M : 128;    // q rows per tile
      const int TNp = (impl == 0) ? NT : 128;   // kv keys per tile
      const int rt = TMp;                       // partial-row granularity
      long long eff[32];
      int btiles[32], kvt[32];
      for (int b = 0; b < B; ++b) {
        int qlen = cq_h[b + 1] - cq_h[b];
        if (qlen < 1) qlen = 1;
        long long e = sl_h[b];
        if (window_left >= 0)
          e = std::min<long long>(e, (long long)window_left + TMp + 1);
        eff[b] = std::max<long long>(e, 1);
        kvt[b] = (int)((eff[b] + TNp - 1) / TNp);
        btiles[b] = std::max(1, (qlen * p.ratio + TMp - 1) / TMp);
        // Defensive: the ws slot math requires x < p.T for every batch.
        if (btiles[b] > p.T) ok = false;
      }
      const int smem_b0 = M * (dqk + 16) + 3 * NT * (dqk + 16) + 3 * NT * dqk
                        + 4 * NT * (dqk + 8);
      const long long slots = (impl == 0)
          ? (long long)g_sm_count *
                std::max(1, std::min(2, (int)((227LL * 1024) / smem_b0)))
          : (long long)g_sm_count;
      const int kmin_env = env_int("FMHA_KMIN", 0);
      // kappa floors: mma.sync chunks amortize fill from ~1 NT tile up (job8:
      // NT32 d6 wants 1-tile chunks; NT64 d8/case8 measured best at 1-2
      // tiles); tc chunks need pipeline amortization (d4 at 4-tile chunks
      // regressed 1.30x in job6 -> >= 8 TN tiles). Both are only the FLOOR:
      // when the floor spills past the resident-slot budget, the bisect
      // below enlarges kappa to fit a single wave (d0/d7-style long-kv).
      const long long kmin = kmin_env > 0
          ? kmin_env
          : (impl == 0 ? NT : 1024);
      // ws fit: S_max * nunits * T * rt * (dqk/2+2) floats <= 3/4 ws (bf16 partials)
      const long long per_c = (long long)nunits * p.T * rt * (dqk / 2 + 2);
      const long long cap_ws =
          per_c > 0 ? (long long)((ws_floats * 3 / 4) / per_c) : 0;
      // Cap 32 keeps fmha_combine on its measured-fast S<=32 broadcast path
      // (job6: Smax=64 chunked combine cost case5 +123%, d1 +117%).
      const long long cap_s = std::min<long long>(std::max<long long>(cap_ws, 1), 32);
      auto ctas_of = [&](long long kappa) {
        long long c = 0;
        for (int b = 0; b < B; ++b) {
          long long s = std::min<long long>(
              std::max<long long>((eff[b] + kappa - 1) / kappa, 1), kvt[b]);
          s = std::min(s, cap_s);
          c += (long long)Hkv * btiles[b] * s;
        }
        return c;
      };
      long long kappa = kmin;
      if (ctas_of(kmin) > slots) {
        // SMALLEST kappa with modeled CTAs <= slots: maximizes split
        // parallelism (min makespan) while staying single-wave. (v1 searched
        // the LARGEST feasible kappa - backwards: it collapsed S_b to 1 on
        // exactly the mixed-length batches C2 targets, job6 dv_ dispatch.)
        long long loK = kmin, hiK = std::max<long long>(max_kv_len, kmin);
        if (ctas_of(hiK) > slots) {
          kappa = hiK;   // even S=1 everywhere overflows: tiles dominate
        } else {
          while (loK < hiK) {
            const long long mid = (loK + hiK) / 2;
            if (ctas_of(mid) <= slots) hiK = mid; else loK = mid + 1;
          }
          kappa = loK;
        }
      }
      int Smax = 1;
      long long total = 0;
      if (cap_ws >= 1) {
        for (int b = 0; b < B; ++b) {
          long long s = std::min<long long>(
              std::max<long long>((eff[b] + kappa - 1) / kappa, 1), kvt[b]);
          s = std::min(s, cap_s);
          s = std::max<long long>(s, 1);
          p.unit_S[b] = (int)s;
          if ((int)s > Smax) Smax = (int)s;
          total += (long long)Hkv * btiles[b] * s;
        }
      }
      // C2v3 gate: engage only for HETEROGENEOUS multi-batch spans, where
      // uniform chunks provably waste waves or pace on the longest unit.
      // Near-uniform batches keep the validated uniform policy (job8 rounds2:
      // variance~2 shapes regressed 1.6-6.4% under balanced - case5 S=32
      // combine growth, case9/d7 linear-grid ordering - while variance>=8
      // shapes won or stayed neutral: d0 0.875, case2 0.952, d1 ~1.0).
      long long emin = eff[0], emax = eff[0];
      for (int b = 1; b < B; ++b) {
        if (eff[b] < emin) emin = eff[b];
        if (eff[b] > emax) emax = eff[b];
      }
      const bool heterogeneous = (B >= 2) && (emax >= 5 * emin / 2);
      if (ok && heterogeneous && Smax >= 2 && total >= 1 && total <= 0x7FFFFFFFLL) {
        long long run = 0;
        for (int b = 0; b < B; ++b) {
          p.batch_off[b] = (int)run;
          run += (long long)Hkv * btiles[b] * p.unit_S[b];
          p.batch_tiles[b] = btiles[b];
        }
        p.batch_off[B] = (int)run;
        p.balanced = 1;
        S = Smax;
        p.S = Smax;
        p.chunk = (int)std::min<long long>(kappa, 0x7FFFFFFFLL);
        p.tile_order = 0;   // decode owns the CTA->work mapping
      }
    }
  }
#ifndef FMHA_NO_TC
  // v9 megatile routing (increment 5; re-port of the archived v9i mechanism
  // with a fresh evidence chain from the unified start): TN=256 pipeline
  // (fmha_tcmt.cuh, split-barrier protocol) for the tc D128/D64 routes.
  // Eligibility mirrors the kernel's structural assumptions: S==1 (no
  // split/combine path in the MT kernel), non-balanced grid, window_left < 0
  // (lo == 0 keeps 256-key stages page-aligned for TMA; cp.async fallback
  // shares the gate). AUTO span floor 4096 keys (16 megatiles): pre-join
  // measured mt-vs-tc wins c4 -20.6%, c1 -9.7%, c7 -3.3%, d9 -9.95%; c0
  // (span 2048) LOST +2-4% -> excluded; split routes (S>1) untouched. The
  // increment-2 span_short veto covers MT structurally: impl 1/3 only
  // reaches here through the !span_short arm (force hooks stay unconditional,
  // dev-only). FMHA_MT: 0 = off, 1 = auto (default), 2 = force at span >=
  // 2048 (dev), 3 = force ignoring span (dev numerics probes).
  const int mt_env = env_int("FMHA_MT", 1);
  // MT+split (increment 6): S>1 chunks ride the megatile pipeline when the
  // per-CTA chunk keeps the AUTO span floor (4096 keys = 16 megatiles) AND
  // the split grid packs at least two waves. The wave gate is measured, not
  // guessed (sweep24, dv_321fbc289f10): d2 (1024 CTAs, 7 waves) -12.2%,
  // while single-wave grids expose the slowest chunk CTA and decode's
  // sparse-row tiles keep the full 128-row softmax cost: c9 (144 CTAs)
  // +6.7%, d7 (128 CTAs) +9.1% -> both excluded by total_ctas >= 2*SMs.
  // FMHA_MT_SPLIT default 1; =0 leaves S>1 on the plain tc route.
  const int mt_split = env_int("FMHA_MT_SPLIT", 1);
  // A megatile must be a whole number of pages: the loader issues one TMA per
  // page segment with nseg = MTN >> log2(page), so MTN=192 at page 128 would
  // silently load 1 of the 1.5 segments it needs. MTN is therefore chosen from
  // page_size -- 192 (the deep-P pipeline, PBUF=3/PVLAG=2) whenever it divides,
  // else 256 (the shipped lag-1 map, the only one errata H2 allows there). Both
  // are compile-time geometries of the same kernel body; 0 means no MT route.
  const int mt_n = ((192 % page) == 0) ? 192 : (((256 % page) == 0) ? 256 : 0);
  if (mt_env != 0 && (impl == 1 || impl == 3) && !p.balanced &&
      window_left < 0 && mt_n != 0 &&
      (p.S == 1
           ? (mt_env >= 3 ||
              (long long)max_kv_len >= (mt_env == 2 ? 2048 : 4096))
           : (mt_split != 0 &&
              (mt_env >= 3 ||
               ((long long)p.T * nunits * p.S >= 2LL * g_sm_count &&
                (long long)p.chunk >= (mt_env == 2 ? 2048 : 4096)))))) {
    if (p.S > 1) {
      // MTN-align the chunk so the megatile partition is exact; S and the ws
      // layout are untouched (a tail chunk can go empty -> self-cleans).
      p.chunk = (int)((((long long)p.chunk + mt_n - 1) / mt_n) * mt_n);  // MTN-aligned
    }
    impl = (impl == 1) ? 6 : 7;
  }
  slot.mt_n = (impl == 6 || impl == 7) ? mt_n : 0;   // selects the MT geometry
#endif
  slot.impl = impl;
  // C2 balanced: grid.x is the exact linearized CTA count; y/z collapse to 1.
  slot.grid = p.balanced ? dim3((unsigned)p.batch_off[B], 1, 1)
                         : dim3(p.T, nunits, S);
  if (!p.balanced && (impl == 4 || impl == 5))
    slot.grid.x = 2 * ((p.T + 1) / 2);  // cluster pairs
  // tile_order: unit-fastest CTA->tile remap (wave balance). It can only pay
  // where per-tile work is UNEVEN, because balancing waves is all it does:
  // causal and windowed shapes have cost(x) = min(x+1, kvt), whereas a
  // non-causal unwindowed shape gives every q-tile the whole kv range, so a
  // hardware wave is already weight-uniform and the remap can only reorder
  // memory traffic. Measured on the megatile route (dv_4c6203a1baee,
  // dv_db6f7a2dc567, results/join23-tileorder-balanced.md): case4 (24MB, the
  // fixed suite's only non-causal shape) reads 0.9676-0.9789 for per-unit
  // order, matching the pre-join tc-route reading of 1.019 against the remap,
  // while case7 (16MB, causal) needs the remap by 1.1124/1.1289 and case9 is
  // neutral. nunits==1 (case6) is an exact no-op, so no wave-count gate. The
  // mma.sync remap gave no win (case5 1.005) and REGRESSED d3 (1.036, A/B
  // dv_3b8a44644e4b): small windowed-kv mma.sync tiles rely on per-unit L2
  // locality that unit-fastest order scatters. tc2 cluster pairs need
  // contiguous same-unit x-halves -> never remap either.
  // Byte gate: locality beats scheduling once the KV footprint is large, but
  // the two routes cross over at different sizes because a megatile stage
  // streams TN=256 keys per load instead of TN=128, which buys the locality
  // the CTA order used to have to provide. The 48MB gate was calibrated on tc
  // and justified case1 by a wave SIMULATION ("sim 1.012"; wavesim.py carries
  // the same number marked "model WRONG"); measured on MT, case1 (64MB, causal)
  // is 0.9871/0.9876 FOR the remap with a 0.02us spread over three passes, and
  // d2 (256MB, MT+split) is 1.0024/1.0029 against it. MT therefore gets 128MB,
  // bracketed by those two measurements and unresolved in between; tc keeps the
  // 48MB it was measured at. FMHA_TILE_ORDER overrides for A/B.
  const long long kv_bytes_est =
      (long long)B * max_kv_len * (long long)Hkv * 2LL * dqk;
  const bool uneven_work = (causal != 0) || (window_left >= 0);
  const long long to_gate = (impl == 6 || impl == 7) ? (128LL << 20) : (48LL << 20);
  const int tord = env_int("FMHA_TILE_ORDER", -1);
  p.tile_order = (impl == 0 || impl == 4 || impl == 5)
                     ? 0
                     : (tord >= 0 ? (tord & 1)
                                  : ((uneven_work && kv_bytes_est <= to_gate) ? 1 : 0));
  if (p.balanced) p.tile_order = 0;   // decode owns the CTA->work mapping
  slot.dqk = dqk;
  slot.nt = NT;
  // tc pipeline depth: D256 -> 3 (sweep dv_bavmmfu1q: case6 577->448us,
  // 0.78x; 225.4KB smem within the 227KB cap), D128/D64 -> 4 (case1 0.98,
  // d2 0.97, d9/7 0.98, neutral on 0/4/9; 145.5KB). FMHA_TC_NSTAGE overrides
  // back to the old depths for experiments. tc2 (impl 4/5) is fixed at 4.
  if (impl != 0) {
    const int ns_env = env_int("FMHA_TC_NSTAGE", 0);
    if (impl == 2) slot.nstage = (ns_env == 2) ? 2 : 3;
    else if (impl == 4 || impl == 5) slot.nstage = 4;
    // MT: diagnostic only -- the launch takes NSTAGE from the geometry type, so
    // this must track MT192::MNS (4) and MT256::MNS (3). Literals, not those
    // names, because this line is outside the FMHA_NO_TC guard that includes
    // fmha_tcmt.cuh.
    else if (impl == 6 || impl == 7) slot.nstage = (slot.mt_n == 192) ? 4 : 3;
    else slot.nstage = (ns_env == 3) ? 3 : 4;
  }
#ifndef FMHA_NO_TC
  if (impl == 6 || impl == 7)
    slot.smem = (int)(slot.mt_n == 192
                          ? sizeof(fmtcmt::SharedMT<fmtcmt::MT192>)
                          : sizeof(fmtcmt::SharedMT<fmtcmt::MT256>));
  else if (impl == 4 || impl == 5) slot.smem = (int)sizeof(fmtc2::Shared2<128, 4>);
  else if (impl == 1 || impl == 3)
    slot.smem = (int)(slot.nstage == 4 ? sizeof(fmtc::Shared<128, 4>)
                                       : sizeof(fmtc::Shared<128, 3>));
  else if (impl == 2)
    slot.smem = (int)(slot.nstage == 3 ? sizeof(fmtc::Shared<256, 3>)
                                       : sizeof(fmtc::Shared<256, 2>));
  else
#endif
    slot.smem = M * (dqk + 16) + 3 * NT * (dqk + 16) + 3 * NT * dqk
              + 4 * NT * (dqk + 8);
#ifndef FMHA_NO_TC
  // r18 persistent MT workers (experimental, env FMHA_MT_PERS; impl 6/7,
  // S==1 only — the direct out/lse scope). Launches g_sm_count workers that
  // stride the ORIGINAL aid space (p.T * nunits) instead of one CTA per
  // (unit, m-block): r17wb (dv_54de04a193ff) measured the wave-boundary
  // serial pool at 19.5% of c7's span (prologue 3.13us + mma_epi 1.77us per
  // CTA generation + 0.96us launch gap, 3.46 CTAs/SM). Persistence pays
  // init/alloc once per worker, double-buffers Q so loaders prefetch
  // assignment a+1 during a's tail, and deletes the launch gaps. Grid/params
  // carry the original aid space so the device-side aid->(u,bx) map (incl.
  // tile_order remap + causal flip) reproduces today's assignment ORDER and
  // every per-row computation exactly; outputs are targeted bitwise-equal.
  slot.pers = 0;
  p.pers_list = nullptr;
  p.pers_kmax = 0;
  if (slot.pers_list) {   // slot re-prepare: free the previous private pool
    cudaFree(slot.pers_list);
    slot.pers_list = nullptr;
  }
  const bool mt_route = (impl == 6 || impl == 7);
  const bool tc_route = (impl == 1 || impl == 2 || impl == 3);
  // r20 AUTO policy: default -1 = engage only inside the MEASURED win
  // classes (wave thresholds below); 0 = force off; 1 = force on
  // (probes/batteries; scheduler type still follows the route policy).
  const int pers_mode = mt_route  ? env_int("FMHA_MT_PERS", -1)
                        : tc_route ? env_int("FMHA_TC_PERS", -1)
                                   : 0;
  if ((mt_route || tc_route) && p.S == 1 && p.geo_valid && !p.balanced &&
      window_left < 0 && pers_mode != 0) {
    const long long total_aid = (long long)p.T * nunits;
    const size_t base_sz =
        mt_route
            ? ((slot.mt_n == 192) ? sizeof(fmtcmt::SharedMT<fmtcmt::MT192>)
                                  : sizeof(fmtcmt::SharedMT<fmtcmt::MT256>))
            : (impl == 2
                   ? ((slot.nstage == 3) ? sizeof(fmtc::Shared<256, 3>)
                                         : sizeof(fmtc::Shared<256, 2>))
                   : ((slot.nstage == 4) ? sizeof(fmtc::Shared<128, 4>)
                                         : sizeof(fmtc::Shared<128, 3>)));
    // q2 (second Q buffer, TM*128 = 16KB) + one 1KB page holding the
    // PERS-only barriers/flags — both OUTSIDE the shared struct so the
    // non-persistent sizeof/smem are untouched. fmtc TD=256 (impl 2) has NO
    // smem headroom for q2 (base ~231.4KB): it runs single-Q-buffer with a
    // qk_done(Ja-1) refill guard instead (kernel-side PLANES==2 branch).
    // MT192/256 and fmtc TD=128 land at <= 232448 B (the cap).
    const size_t q2_bytes = (impl == 2) ? 0 : 128 * 128;
    const size_t pers_sz =
        ((base_sz + 1023) & ~(size_t)1023) + q2_bytes + 1024;
    // ---- r18k: DYNAMIC ticket pool (replaces the static LPT lists) ----
    // r18j showed the residual blocker is the VARIANCE TAX: static binding
    // makes the wall the SLOWEST worker (c7 on-arm bimodal 98.1/100.3 =
    // makespan-vs-median; c1 cad_mx/med +32%) while the hardware CTA
    // dispatcher absorbs per-CTA cost variance for free. Pure dynamic
    // pull restores that: pool = every valid aid sorted heavy-first
    // (weight nmt+3 — each assignment also costs its ~3-megatile boundary;
    // tie: aid ascending), workers take tickets via one atomicAdd until
    // the pool is empty. Heavy-first ticket order preserves the locality
    // class of today's dispatch; per-row math is aid-keyed so outputs stay
    // bitwise-deterministic (scheduling order is not). Buffer layout in a
    // PRIVATE per-slot cudaMalloc (the harness restores workspace_buffer
    // between iterations; graph bakes the pointer): [0] = ticket counter
    // (self-cleaned by the worker drawing ticket n_items+nworkers-1, so
    // every graph replay starts at 0), [1..n_items] = pool aids.
    std::vector<std::pair<int, int>> items;   // (weight, aid)
    items.reserve((size_t)std::min<long long>(total_aid, 1 << 20));
    const int MTN_h = mt_route ? slot.mt_n : 128;   // keys per tile (fmtc TN=128)
    for (int b = 0; b < B; ++b) {
      const int qlen_b = p.geo_cu[b + 1] - p.geo_cu[b];
      const int kvlen_b = p.geo_sl[b];
      const int R_b = qlen_b * p.ratio;
      const int tiles_b = (R_b + 127) / 128;
      for (int g = 0; g < p.Hkv; ++g) {
        const int u = b * p.Hkv + g;
        for (int bx = 0; bx < tiles_b && bx < p.T; ++bx) {
          const int x = p.causal ? (tiles_b - 1 - bx) : bx;
          const int r0h = x * 128;
          int tlast = (r0h + 127) / p.ratio;
          if (tlast > qlen_b - 1) tlast = qlen_b - 1;
          int hi = kvlen_b;
          if (p.causal) hi = std::min(hi, kvlen_b - qlen_b + tlast + 1);
          const int nmt = (hi + MTN_h - 1) / MTN_h;   // lo = 0 (window<0)
          if (nmt <= 0) continue;
          const int aid = p.tile_order ? (bx * nunits + u) : (u * p.T + bx);
          items.emplace_back(nmt + 3, aid);
        }
      }
    }
    int nw = g_sm_count > 0 ? g_sm_count : 148;
    // r20 AUTO thresholds (MEASURED classes only; conditional evidence,
    // revisable): MT engages at >1 wave (r18k dynamic: c4 1.46w -2.5%,
    // c7 1.95w -1.9%, c1 6.9w -8.9%; single-wave = overhead only) and runs
    // STATIC direct-bound segments iff short-chain (<=5 items/worker:
    // r18j c4 0.951, c7 0.961, d9 0.937 med vs c1 static 1.005 = variance
    // tax), else the DYNAMIC r18k pool. fmtc engages STATIC only at >=2
    // waves (c6 3.46w: 0.9373/0.9414 static vs 0.9822/0.982 dynamic vs
    // c0 1.30w: 1.000-1.007 = prologue-bound, no fold win -> stays
    // production; dynamic measured WRONG for fmtc — greedy makespan, c0
    // +0.7%). r19d datum: static lists must ride DIRECT per-role binding
    // (r18j machinery); routing them through the pull broadcast cost ~2us
    // per assignment (c7 1.02 vs r18j 0.961) -> r19e.
    const long long n_it = (long long)items.size();
    const bool wave_ok = mt_route ? (n_it > (long long)nw)
                                  : (n_it >= 2LL * (long long)nw);
    // r19f/r19g: MT static = short-chain (<=5 items/worker), ANY impl.
    // Authoritative shape<->config mapping from the r20audit dispatch
    // lines (dv_347eab0cfa7b): frozen c4 = impl7 (DV64 SPLIT epilogue,
    // B2, MT192) — static-direct measured CLEAN 0.950 (r19e dv_3b4dd065
    // c4 cell); frozen c7 = impl6 (DV128 group-1 epilogue, B1, MT192) —
    // static-direct exposed the group-0-next-publish vs group-1-state-read
    // race (r19e c7 REJECT: gates False both arms + bw False; audit
    // confirmed 1.0586/1.0775 + REJECT under the impl6-static r19f gate).
    // r19g fixes that race STRUCTURALLY (group-0 assignment-entry waits
    // o_drained(qa-1), which fires only after every epilogue state/O
    // read). This audit re-validates BOTH classes: c7 static+fix (target
    // r18j-class 0.942-0.961) and c4 static+benign-fix (target 0.950
    // repro); c1/d9 long-chain keep the r18k dynamic pool (0.9108/0.9438
    // audit-reproduced). If c7 static fails again -> fall back to the
    // impl7-only static gate (c4 0.950 verified pre-fix; the entry gate
    // is benign for the split epilogue: group 0's own arrive is part of
    // the phase, so its wait passes on group 1's arrive).
    bool use_static = tc_route || (n_it <= 5LL * (long long)nw);
    // pers_mode 2 = probe-only: force the DYNAMIC pool (scheduler bisect).
    if (pers_mode == 2) use_static = false;
    if (!items.empty() && pers_sz <= 232448 && total_aid > 1 &&
        (pers_mode == 1 || wave_ok)) {
      if ((int)items.size() < nw) nw = (int)items.size();
      std::sort(items.begin(), items.end(),
                [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.first != b.first ? a.first > b.first
                                            : a.second < b.second;
                });
      const int n_items = (int)items.size();
      std::vector<int> flat;
      bool do_static = use_static;
      std::vector<std::vector<int>> bins((size_t)nw);
      if (do_static) {
        // ---- r19c/d/i: STATIC BALANCED PARTITION with the tiny-adjacency
        // QUARANTINE. Quality rationale (r19c, sims calibrated EXACT vs
        // dv_a7efac710e94 frontiers): cap-BFD beats dynamic greedy on the
        // makespan (c6 214->189 weight; cad_mx/med 1.21->1.034 measured) and
        // direct binding removes the ~2us/assignment broadcast cost (r19d).
        // QUARANTINE (r19i): plain BFD best-fit packs the lightest items
        // CONSECUTIVELY into the tightest bins (c7 worker 147 got SEVEN
        // consecutive nmt=1 assignments) — and consecutive-tiny chains under
        // static direct binding deterministically corrupt the group-1
        // epilogue anchor on MT-D128 (r19h dv_e00efb0ae238: bad tiles =
        // exactly the chained nmt<=2 items, rows 1-127, row 0 clean =
        // anchor/state class; segment-END tiny items clean = successor-
        // dependent; dynamic/LPT/broadcast never chain tinys and are
        // bitwise-clean on the same shape). Mechanism not yet root-caused;
        // the regime is quarantined instead: a tiny item (weight <= 7, i.e.
        // nmt/ntl <= 4 = observed-corrupt class x2 margin) may never be
        // placed behind another tiny. Descending arrival order then makes
        // every bin hold AT MOST ONE tiny, always LAST (no successor -> the
        // regime cannot arise). If the constrained placement is infeasible
        // (tiny supply vs bins), static is abandoned: MT falls back to the
        // dynamic r18k pool, fmtc to production (dynamic measured the wrong
        // fmtc scheduler: c6 0.982, c0 +0.7%).
        const int TINY_W = 7;
        long long wsum = 0;
        for (const auto& it : items) wsum += it.first;
        long long lpt_max = 0;
        bool static_ok = true;
        {
          // Constrained LPT (upper bound + fallback bins).
          std::set<std::pair<long long, int>> st;   // (load, worker)
          for (int w = 0; w < nw; ++w) st.insert({0, w});
          std::vector<char> last_tiny((size_t)nw, 0);
          for (const auto& it : items) {
            const bool tiny = (it.first <= TINY_W);
            auto b = st.begin();
            while (b != st.end() && tiny && last_tiny[(size_t)b->second]) ++b;
            if (b == st.end()) { static_ok = false; break; }
            const long long nl = b->first + it.first;
            const int w = b->second;
            st.erase(b);
            bins[(size_t)w].push_back(it.second);
            last_tiny[(size_t)w] = (char)tiny;
            st.insert({nl, w});
            if (nl > lpt_max) lpt_max = nl;
          }
        }
        auto bfd = [&](long long cap, std::vector<std::vector<int>>& out)
            -> long long {
          // Constrained best-fit-decreasing; max load, or -1 if infeasible
          // within nw bins / cap / the tiny-adjacency rule.
          out.assign((size_t)nw, {});
          std::set<std::pair<long long, int>> st;
          for (int w = 0; w < nw; ++w) st.insert({0, w});
          std::vector<char> last_tiny((size_t)nw, 0);
          for (const auto& it : items) {
            const bool tiny = (it.first <= TINY_W);
            auto ub = st.upper_bound({cap - it.first, INT32_MAX});
            if (ub == st.begin()) return -1;   // no bin with room
            --ub;                              // largest load <= cap-weight
            while (tiny && last_tiny[(size_t)ub->second]) {
              if (ub == st.begin()) return -1;
              --ub;
            }
            const long long nl = ub->first + it.first;
            const int w = ub->second;
            st.erase(ub);
            out[(size_t)w].push_back(it.second);
            last_tiny[(size_t)w] = (char)tiny;
            st.insert({nl, w});
          }
          return st.rbegin()->first;
        };
        if (static_ok) {
          long long lo = (wsum + nw - 1) / nw, hi = lpt_max;
          std::vector<std::vector<int>> tmp;
          while (lo < hi) {
            const long long mid = (lo + hi) / 2;
            if (bfd(mid, tmp) >= 0) hi = mid; else lo = mid + 1;
          }
          const long long bfd_max = bfd(lo, tmp);
          if (bfd_max >= 0 && bfd_max < lpt_max) bins.swap(tmp);
        }
        if (!static_ok) do_static = false;
      }
      if (do_static) {
        flat.assign((size_t)nw + 1 + (size_t)n_items, 0);
        int acc = 0;
        for (int w = 0; w < nw; ++w) {
          flat[(size_t)w] = acc;
          for (int a : bins[(size_t)w]) flat[(size_t)nw + 1 + acc++] = a;
        }
        flat[(size_t)nw] = acc;   // == n_items
      } else if (mt_route) {
        flat.assign((size_t)n_items + 1, 0);   // [0] = counter = 0
        for (int i = 0; i < n_items; ++i)
          flat[(size_t)i + 1] = items[(size_t)i].second;
      }
      // fmtc + static-infeasible -> flat empty -> production launch.
      int* dlist = nullptr;
      if (!flat.empty() &&
          cudaMalloc(&dlist, flat.size() * sizeof(int)) == cudaSuccess &&
          cudaMemcpy(dlist, flat.data(), flat.size() * sizeof(int),
                     cudaMemcpyHostToDevice) == cudaSuccess) {
        slot.pers_list = dlist;
        p.pers_list = dlist;
        p.pers_kmax = n_items;      // pool item count (name kept; see kernel)
        p.pers_static = do_static ? 1 : 0;   // r19c/d/i: static partition
        p.pers_total = (int)total_aid;
        p.pers_gx = p.T;
        p.pers_gy = nunits;
        slot.grid = dim3((unsigned)nw, 1, 1);
        slot.smem = (int)pers_sz;
        slot.pers = 1;
      } else if (dlist) {
        cudaFree(dlist);
      }
    }
  }
#endif
  if (env_int("FMHA_DBG_DISPATCH", 0)) {
    // Dev-only dispatch table line (command.log); production never sets this.
    const int rt_dbg = (impl != 0) ? 128 : 64;
    const int nsl = (dqk + 63) / 64;
    const int base_ctas = slot.p.T * slot.p.B * slot.p.Hkv * nsl;
    const int rpb_dbg = combine_rpb(base_ctas, rt_dbg);
    printf("DISPATCH B=%d Hkv=%d H=%d S=%d T=%d dqk=%d impl=%d nt=%d nstage=%d "
           "balanced=%d rt=%d nslice=%d old_ctas=%d rpb=%d new_ctas=%d "
           "pers=%d/%d grid=(%u,%u,%u)\n",
           slot.p.B, slot.p.Hkv, slot.p.H, slot.p.S, slot.p.T, dqk, impl, NT,
           slot.nstage, (int)slot.p.balanced, rt_dbg, nsl, base_ctas, rpb_dbg,
           base_ctas * ((rt_dbg + rpb_dbg - 1) / rpb_dbg),
           slot.pers, slot.pers ? slot.p.pers_static : 0,
           slot.grid.x, slot.grid.y, slot.grid.z);
    fflush(stdout);
  }
  slot.valid = true;
  return 0;
}

#ifndef FMHA_NO_TC
template <int TD, int NS, int DVD>
static void tc_launch(const Prepared& slot, cudaStream_t st) {
  static int attr_done = 0;
  if (!attr_done) {
    cudaFuncSetAttribute(fmtc::fmha_tc_kernel<TD, NS, DVD>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(fmtc::Shared<TD, NS>));
    // r19 persistent instantiation: base + q2 (TD<256 only) + the 1KB page.
    cudaFuncSetAttribute(fmtc::fmha_tc_kernel<TD, NS, DVD, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)((((sizeof(fmtc::Shared<TD, NS>) + 1023)
                                 & ~(size_t)1023))
                               + ((TD == 256) ? 0 : 128 * 128) + 1024));
    attr_done = 1;
  }
  // v8c: PDL is OPT-IN (FMHA_PDL=1), default OFF. The programmatic-
  // serialization attribute let consecutive graph launches of the SAME tc
  // grid self-overlap; under production timing that produced a flaky
  // completion-accounting stall (all CTAs' exit markers set, device sync
  // never returns) on case-4-class shapes (eval dv_52d9c4051767 shape 4;
  // earlier dv_bc4216111188, repro dv_acab6f7881a2, stress v8). V0 carried
  // the attr on the MAIN kernel only - combine never had it - so C9's
  // main->combine overlap intent is unchanged by this removal; the measured
  // nopdl cost was ~1-5% on a few small latency-bound cases.
  if (slot.pers) {
    fmtc::fmha_tc_kernel<TD, NS, DVD, true>
        <<<slot.grid, fmtc::NTHREADS, slot.smem, st>>>(slot.p);
    return;
  }
  if (env_int("FMHA_PDL", 0)) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = slot.grid;
    cfg.blockDim = dim3(fmtc::NTHREADS, 1, 1);
    cfg.dynamicSmemBytes = slot.smem;
    cfg.stream = st;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    void* args[] = {const_cast<Params*>(&slot.p)};
    cudaLaunchKernelExC(&cfg, (const void*)fmtc::fmha_tc_kernel<TD, NS, DVD>, args);
  } else {
    fmtc::fmha_tc_kernel<TD, NS, DVD><<<slot.grid, fmtc::NTHREADS, slot.smem, st>>>(slot.p);
  }
}
// v9 megatile launch (impl 6/7): S==1, plain launch (no PDL - v8c policy),
// graph-capture safe. Two geometries, selected by slot.mt_n (host gate above):
// MT192 = 192-key stages, MNS=4, PBUF=3, PVLAG=2; MT256 = the shipped 256-key
// lag-1 map. Both smem sizes are 212992 B, so one attribute call covers both.
template <int DV>
static void tc_mt_launch(const Prepared& slot, cudaStream_t st) {
  static int attr_done = 0;
  if (!attr_done) {
    cudaFuncSetAttribute(fmtcmt::fmha_tcmt_kernel<128, fmtcmt::MT192, DV>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(fmtcmt::SharedMT<fmtcmt::MT192>));
    cudaFuncSetAttribute(fmtcmt::fmha_tcmt_kernel<128, fmtcmt::MT256, DV>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(fmtcmt::SharedMT<fmtcmt::MT256>));
    // r18 persistent instantiations: base + the 1024-aligned second Q buffer
    // + the o_drained barrier page (must match prepare's pers_sz).
    cudaFuncSetAttribute(fmtcmt::fmha_tcmt_kernel<128, fmtcmt::MT192, DV, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)((((sizeof(fmtcmt::SharedMT<fmtcmt::MT192>) + 1023)
                                 & ~(size_t)1023)) + 128 * 128 + 1024));
    cudaFuncSetAttribute(fmtcmt::fmha_tcmt_kernel<128, fmtcmt::MT256, DV, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)((((sizeof(fmtcmt::SharedMT<fmtcmt::MT256>) + 1023)
                                 & ~(size_t)1023)) + 128 * 128 + 1024));
    attr_done = 1;
  }
  if (slot.mt_n == 192) {
    if (slot.pers)
      fmtcmt::fmha_tcmt_kernel<128, fmtcmt::MT192, DV, true>
          <<<slot.grid, fmtc::NTHREADS, slot.smem, st>>>(slot.p);
    else
      fmtcmt::fmha_tcmt_kernel<128, fmtcmt::MT192, DV>
          <<<slot.grid, fmtc::NTHREADS, slot.smem, st>>>(slot.p);
  } else {
    if (slot.pers)
      fmtcmt::fmha_tcmt_kernel<128, fmtcmt::MT256, DV, true>
          <<<slot.grid, fmtc::NTHREADS, slot.smem, st>>>(slot.p);
    else
      fmtcmt::fmha_tcmt_kernel<128, fmtcmt::MT256, DV>
          <<<slot.grid, fmtc::NTHREADS, slot.smem, st>>>(slot.p);
  }
}
// 2-CTA cluster launch (cudaLaunchKernelExC + cluster attribute; graph-
// capture validated by experiments/tc2 probe dv PASS).
template <int TD, int NS, int DVD>
static void tc2_launch(const Prepared& slot, cudaStream_t st) {
  static int attr_done = 0;
  if (!attr_done) {
    cudaFuncSetAttribute(fmtc2::fmha_tc2_kernel<TD, NS, DVD>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(fmtc2::Shared2<TD, NS>));
    attr_done = 1;
  }
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = slot.grid;
  cfg.blockDim = dim3(fmtc::NTHREADS, 1, 1);
  cfg.dynamicSmemBytes = slot.smem;
  cfg.stream = st;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  void* args[] = {const_cast<Params*>(&slot.p)};
  cudaLaunchKernelExC(&cfg, (const void*)fmtc2::fmha_tc2_kernel<TD, NS, DVD>, args);
}
#endif

static int run_impl(const Prepared& slot, cudaStream_t st) {
  const Params& p = slot.p;
#ifndef FMHA_NO_TC
  if (slot.impl == 6 || slot.impl == 7) {
    // MT route: S==1 writes out/lse directly; MT+split (S>1) writes the same
    // ws partials as the tc route, merged by fmha_combine (rt=128,
    // lse_adj=-PLOG2, row-blocked rpb rule).
    if (slot.impl == 6) tc_mt_launch<128>(slot, st);
    else tc_mt_launch<64>(slot, st);
    if (cudaGetLastError() != cudaSuccess) return -4;
    if (p.S > 1) {
      const int base_ctas = p.T * p.B * p.Hkv * ((slot.dqk + 63) / 64);
      const int rpb = combine_rpb(base_ctas, 128);
      dim3 cgrid(p.T, p.B * p.Hkv,
                 ((slot.dqk + 63) / 64) * ((128 + rpb - 1) / rpb));
      fmha_combine<<<cgrid, 512, 0, st>>>(p, slot.dqk, slot.dqk / 2 + 2, 128,
                                          -(float)fmtc::PLOG2, rpb);
      if (cudaGetLastError() != cudaSuccess) return -5;
    }
    return 0;
  }
  if (slot.impl == 4 || slot.impl == 5) {
    if (slot.impl == 4) tc2_launch<128, 4, 128>(slot, st);
    else tc2_launch<128, 4, 64>(slot, st);
    if (cudaGetLastError() != cudaSuccess) return -4;
    if (p.S > 1) {
      const int base_ctas = p.T * p.B * p.Hkv * ((slot.dqk + 63) / 64);
      const int rpb = combine_rpb(base_ctas, 128);
      dim3 cgrid(p.T, p.B * p.Hkv,
                 ((slot.dqk + 63) / 64) * ((128 + rpb - 1) / rpb));
      fmha_combine<<<cgrid, 512, 0, st>>>(p, slot.dqk, slot.dqk / 2 + 2, 128,
                                          -(float)fmtc::PLOG2, rpb);
      if (cudaGetLastError() != cudaSuccess) return -5;
    }
    return 0;
  }
  if (slot.impl == 1 || slot.impl == 2 || slot.impl == 3) {
    if (slot.impl == 1) {
      if (slot.nstage == 4) tc_launch<128, 4, 128>(slot, st);
      else tc_launch<128, 3, 128>(slot, st);
    } else if (slot.impl == 3) {
      if (slot.nstage == 4) tc_launch<128, 4, 64>(slot, st);
      else tc_launch<128, 3, 64>(slot, st);
    } else {
      if (slot.nstage == 3) tc_launch<256, 3, 256>(slot, st);
      else tc_launch<256, 2, 256>(slot, st);
    }
    if (cudaGetLastError() != cudaSuccess) return -4;
    if (p.S > 1) {
      // tc split-KV: merge partials (128-row tiles; dsum carries PSCALE=2^5,
      // hence lse_adj = -PLOG2).
      const int base_ctas = p.T * p.B * p.Hkv * ((slot.dqk + 63) / 64);
      const int rpb = combine_rpb(base_ctas, 128);
      dim3 cgrid(p.T, p.B * p.Hkv,
                 ((slot.dqk + 63) / 64) * ((128 + rpb - 1) / rpb));
      fmha_combine<<<cgrid, 512, 0, st>>>(p, slot.dqk, slot.dqk / 2 + 2, 128,
                                          -(float)fmtc::PLOG2, rpb);
      if (cudaGetLastError() != cudaSuccess) return -5;
    }
    return 0;
  }
#endif
#define LAUNCH_MMA(D, NTV)                                                     \
  do {                                                                         \
    ensure_smem_attr<D, NTV>();                                                \
    if (env_int("FMHA_PDL", 0)) {                                              \
      cudaLaunchAttribute attr{};                                              \
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;            \
      attr.val.programmaticStreamSerializationAllowed = 1;                     \
      cudaLaunchConfig_t cfg{};                                                \
      cfg.gridDim = slot.grid;                                                 \
      cfg.blockDim = dim3(128, 1, 1);                                          \
      cfg.dynamicSmemBytes = slot.smem;                                        \
      cfg.stream = st;                                                         \
      cfg.attrs = &attr;                                                       \
      cfg.numAttrs = 1;                                                        \
      void* args[] = {const_cast<Params*>(&p)};                                \
      cudaLaunchKernelExC(&cfg, (const void*)fmha_kernel<D, NTV>, args);       \
    } else {                                                                   \
      fmha_kernel<D, NTV><<<slot.grid, 128, slot.smem, st>>>(p);               \
    }                                                                          \
  } while (0)
  if (slot.dqk == 64) {
    if (slot.nt == 64) LAUNCH_MMA(64, 64);
    else LAUNCH_MMA(64, 128);
  } else if (slot.dqk == 128) {
    if (slot.nt == 32) LAUNCH_MMA(128, 32);
    else if (slot.nt == 128) LAUNCH_MMA(128, 128);
    else LAUNCH_MMA(128, 64);
  } else if (slot.dqk == 256) {
    if (slot.nt == 64) LAUNCH_MMA(256, 64);
    else LAUNCH_MMA(256, 32);
  } else {
    return -1;
  }
#undef LAUNCH_MMA
  if (cudaGetLastError() != cudaSuccess) return -2;
  if (p.S > 1) {
    const int base_ctas = p.T * p.B * p.Hkv * ((slot.dqk + 63) / 64);
    const int rpb = combine_rpb(base_ctas, 64);
    dim3 cgrid(p.T, p.B * p.Hkv,
               ((slot.dqk + 63) / 64) * ((64 + rpb - 1) / rpb));
    fmha_combine<<<cgrid, 512, 0, st>>>(p, slot.dqk, slot.dqk / 2 + 2, 64, 0.f, rpb);
    if (cudaGetLastError() != cudaSuccess) return -3;
  }
  return 0;
}

static void slot_graph_destroy(Prepared& slot) {
  if (slot.exec) { cudaGraphExecDestroy(slot.exec); slot.exec = nullptr; }
  if (slot.graph) { cudaGraphDestroy(slot.graph); slot.graph = nullptr; }
}

// Capture the slot's launch sequence (1-2 kernels) into a replayable graph.
// Saves ~2-4us of per-call launch CPU (visible on <30us shapes). Any failure
// leaves exec == nullptr and fmha_fp8_run falls back to direct launches.
static void slot_graph_capture(Prepared& slot) {
  static cudaStream_t cap_stream = nullptr;
  if (!cap_stream &&
      cudaStreamCreateWithFlags(&cap_stream, cudaStreamNonBlocking) != cudaSuccess)
    return;
  if (!cap_stream) return;
  // Pre-run once outside capture: sets lazy cudaFuncSetAttribute flags and
  // warms module loading so the captured sequence is pure kernel launches.
  // v8b: SYNC the prerun before capture. Leaving it in flight let the prerun
  // grid overlap the first graph launches (two tc grids resident at once):
  // a flaky hang appeared under production timing in exactly that window
  // (case-4-class shapes, ~30% of stress sessions; watchdog found sync
  // blocked with every dbg-marked CTA 'done' - attribution ambiguous because
  // both grids write the same marker slots). The prerun is a warmup; the
  // sync costs one kernel duration per prepare (cached handle) and removes
  // the cross-grid overlap entirely.
  if (run_impl(slot, cap_stream) != 0) return;
  if (cudaStreamSynchronize(cap_stream) != cudaSuccess) return;
  cudaGraph_t g = nullptr;
  if (cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
    return;
  int rc = run_impl(slot, cap_stream);
  cudaError_t e = cudaStreamEndCapture(cap_stream, &g);
  if (rc != 0 || e != cudaSuccess || !g) {
    if (g) cudaGraphDestroy(g);
    return;
  }
  cudaGraphExec_t ex = nullptr;
  if (cudaGraphInstantiate(&ex, g, 0) != cudaSuccess || !ex) {
    cudaGraphDestroy(g);
    return;
  }
  slot.graph = g;
  slot.exec = ex;
}

extern "C" int fmha_fp8_prepare(
    const void* q, const void* k, const void* v, const int* table, const int* seqlens,
    const int* cu_q, void* out, float* lse, float* ws, const float* sinks,
    long long q_tok_stride, long long k_sp, long long k_sh, long long k_sn, long long v_sp,
    long long v_sh, long long v_sn, int Tstride, int page, int H, int Hkv, int B, int Tq,
    int dqk, int max_q_len, int max_kv_len, int window_left, int causal, int has_lse,
    int has_sink, float bmm1, float bmm2, long long ws_floats, int force_impl,
    long long kv_pages) {
  int idx = g_slot_cursor;
  g_slot_cursor = (g_slot_cursor + 1) & 7;
  Prepared& slot = g_slots[idx];
  slot_graph_destroy(slot);   // slot recycled: stale graph args would be wrong
  int rc = prepare_impl(slot, q, k, v, table, seqlens, cu_q, out, lse, ws, sinks,
                        q_tok_stride, k_sp, k_sh, k_sn, v_sp, v_sh, v_sn, Tstride, page,
                        H, Hkv, B, Tq, dqk, max_q_len, max_kv_len, window_left, causal,
                        has_lse, has_sink, bmm1, bmm2, ws_floats, force_impl, kv_pages);
  if (rc) { slot.valid = false; return rc; }
  // Experiment hook: FMHA_NO_GRAPH=1 skips graph capture (direct launches)
  // to measure the graph-vs-direct launch floor on tiny shapes.
  if (!env_int("FMHA_NO_GRAPH", 0)) slot_graph_capture(slot);
  return idx;
}

extern "C" int fmha_fp8_run(int handle, void* stream) {
  if (handle < 0 || handle >= 8 || !g_slots[handle].valid) return -20;
  Prepared& slot = g_slots[handle];
  if (slot.exec) {
    if (cudaGraphLaunch(slot.exec, (cudaStream_t)stream) != cudaSuccess) return -21;
    return 0;
  }
  return run_impl(slot, (cudaStream_t)stream);
}

extern "C" int fmha_fp8_launch(
    const void* q, const void* k, const void* v, const int* table, const int* seqlens,
    const int* cu_q, void* out, float* lse, float* ws, const float* sinks,
    long long q_tok_stride, long long k_sp, long long k_sh, long long k_sn, long long v_sp,
    long long v_sh, long long v_sn, int Tstride, int page, int H, int Hkv, int B, int Tq,
    int dqk, int max_q_len, int max_kv_len, int window_left, int causal, int has_lse,
    int has_sink, float bmm1, float bmm2, long long ws_floats, int force_impl,
    long long kv_pages, void* stream) {
  int h = fmha_fp8_prepare(q, k, v, table, seqlens, cu_q, out, lse, ws, sinks, q_tok_stride,
                           k_sp, k_sh, k_sn, v_sp, v_sh, v_sn, Tstride, page, H, Hkv, B, Tq,
                           dqk, max_q_len, max_kv_len, window_left, causal, has_lse, has_sink,
                           bmm1, bmm2, ws_floats, force_impl, kv_pages);
  if (h < 0) return h;
  return fmha_fp8_run(h, stream);
}
