// Independent FMHA BF16 implementation for L20D (SM103) - flash-attention style,
// paged KV (HND), THD varlen, GQA/MQA, causal + left window + attention sinks,
// optional base-2 LSE, split-KV (flash-decoding) with combine kernel.
// Pure CUDA C++/PTX: warp-level mma.sync m16n8k16 bf16 + cp.async pipelines.
// No external libraries. Written for the frozen fmha_bf16 contract.
//
// v2: multi-stage cp.async pipeline (STAGES), Q fragments preloaded into
// registers for DQ<=128, uniform interior stages skip masking predicates.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math.h>

#include "fmha_params.h"

#ifdef FMHA_TMA
#include <cuda.h>
#include <dlfcn.h>
#endif


#define NEG_INF (-__int_as_float(0x7f800000))

// Typed views of the ABI params (header keeps void* for cross-library reuse).
#define P_Q    (reinterpret_cast<const __nv_bfloat16*>(P.q))
#define P_K    (reinterpret_cast<const __nv_bfloat16*>(P.k))
#define P_V    (reinterpret_cast<const __nv_bfloat16*>(P.v))
#define P_OUT  (reinterpret_cast<__nv_bfloat16*>(P.out))


__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float& d0, float& d1, float& d2, float& d3,
                                         uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ uint32_t pack_bf16(float x, float y) {
  __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&h);
}

#ifdef FMHA_EXP_FFMA2
// Packed f32x2 subtract (sm_100/sm_103 Blackwell), mechanism per the
// cudnn_fe_oss sm100 reference. sub.rn.ftz.f32x2 is lane-identical to the
// scalar `a - b` lowering under --use_fast_math (-ftz=true), so P values are
// bitwise unchanged; halves the SUB issue slots of the decode exp loops
// (the FFMA2 prefill win measured x0.945-0.988 on the WS cases,
// dv_7f1d77e885cc). exp2f stays scalar (no packed f32 ex2 exists).
__device__ __forceinline__ float2 fsub2(float2 a, float2 b) {
  uint64_t c;
  asm("sub.rn.ftz.f32x2 %0, %1, %2;\n"
      : "=l"(c)
      : "l"(reinterpret_cast<const uint64_t&>(a)),
        "l"(reinterpret_cast<const uint64_t&>(b)));
  return reinterpret_cast<float2&>(c);
}
#endif

#ifdef FMHA_DEC_PACK2
// PACK2 decode (companion to FMHA_PACK2_MUL in fmha_tc.cu): packed f32x2
// multiply for the m16 lazy O-rescale. d256 fires 64 scalar FMULs per
// thread per rescale (DVH/8 x 4); fmul2 halves that to 32 issue slots.
// Per-lane RN - divergence from the ftz scalar lowering is denormal-only
// (tolerance-level). The lazy16 formal run (dv_d9d208266818, clean-sweep
// CI<1.0) showed the decode lazy-rescale path responds measurably to
// per-fire work reduction (c2 x0.9986, c3 x0.9967, c8 x0.9932).
__device__ __forceinline__ float2 fmul2(float2 a, float2 b) {
  uint64_t c;
  asm("mul.f32x2 %0, %1, %2;\n"
      : "=l"(c)
      : "l"(reinterpret_cast<const uint64_t&>(a)),
        "l"(reinterpret_cast<const uint64_t&>(b)));
  return reinterpret_cast<float2&>(c);
}
#endif

// ---------------------------------------------------------------------------
// Bulk-tensor (TMA) KV loading for the paged cache, plus the mbarrier helpers
// that gate slot reuse. One 4D map {d, page_size, kvh, pages} per operand with
// a SWIZZLE_128B box of {64 bf16, R rows}, R = min(NT, page_size): boxes never
// straddle a page, rows past `last` are clamped like the cp.async loader, and
// out-of-range slots zero-fill (they are masked out by rlo/rhi anyway).
// ---------------------------------------------------------------------------
// NOTE: these helpers stay compiled even without FMHA_TMA. m16_body's
// `if constexpr (TMA)` branch is discarded at instantiation for TMA=false,
// but tma_kv_stage/mbar_wait/fence_proxy_async are NON-DEPENDENT names that
// must resolve at definition time - guarding them broke the FMHA_TMA=0
// degraded-mode build (fallback battery dv_f82f75ccc200 pass 3: 5 compile
// errors). Unused inline wrappers emit nothing; default builds are
// token-identical.
__device__ __forceinline__ void tma_load_4d(const void* map, char* dst_smem,
                                            int c0, int c1, int c2, int c3,
                                            uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3, %4, %5}], [%6];"
      :: "r"(smem_u32(dst_smem)), "l"(reinterpret_cast<uint64_t>(map)),
         "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar)
      : "memory");
}
__device__ __forceinline__ void mbar_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(addr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t phase) {
  asm volatile("{\n\t.reg .pred c;\n\tW%=:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 c, [%0], %1;\n\t"
               "@!c bra.uni W%=;\n\t}" :: "r"(addr), "r"(phase) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// Single-thread stage issue: arms the slot barrier for K+V and emits
// NSEG*ATOMS*2 bulk-tensor copies (8-32 instructions per 32-128KB stage,
// replacing NT*D/8 cp.async issues plus their page-table/stride chains).
template <int NT, int DK, int DV>
__device__ __forceinline__ void tma_kv_stage(const FmhaParams& P, char* dstk,
                                             char* dstv, const void* kmap,
                                             const void* vmap, int kvh,
                                             int kv_base, int last,
                                             const int32_t* tab, uint32_t mbar) {
  mbar_expect_tx(mbar, NT * (DK + DV) * 2);
  const int R = min(NT, 1 << P.page_log);
  const int nseg = NT / R;   // R divides NT: R = min(NT, page_size), both pow2
#pragma unroll
  for (int seg = 0; seg < NT / 16; ++seg) {
    if (seg >= nseg) break;
    const int p = min(kv_base + seg * R, last);
    const int page = __ldg(tab + (p >> P.page_log));
    const int slot = p & P.page_mask;
#pragma unroll
    for (int a = 0; a < DK / 64; ++a)
      tma_load_4d(kmap, dstk + a * (NT * 128) + seg * R * 128, a * 64, slot,
                  kvh, page, mbar);
#pragma unroll
    for (int a = 0; a < DV / 64; ++a)
      tma_load_4d(vmap, dstv + a * (NT * 128) + seg * R * 128, a * 64, slot,
                  kvh, page, mbar);
  }
}

// Page-ID prefetch for the decode TMA loader (FMHA_DEC_PGPRE): the same
// pathology and fix as prefill PAGEPRE (ab16: x0.9629). tma_kv_stage resolves
// each segment's physical page with a dependent __ldg on tid0 and then issues
// its 8-32 boxes; every thread hits the per-stage __syncthreads right behind
// tid0, so an L1-missing table read (~300-500 cyc) lands on the critical path
// of EVERY stage. nseg = NT/R <= NT/16 <= 4 for the decode tiles, so tid0
// keeps the next stage's page IDs in registers, loaded one iteration ahead.
// Same pages, same boxes, same order: bitwise identical.
template <int NT>
__device__ __forceinline__ void tma_pages_pre(const FmhaParams& P,
                                              const int32_t* tab, int kv_base,
                                              int last, int32_t* pg) {
  const int R = min(NT, 1 << P.page_log);
  const int nseg = NT / R;
#pragma unroll
  for (int seg = 0; seg < NT / 16; ++seg) {
    if (seg >= nseg) break;
    const int p = min(kv_base + seg * R, last);
    pg[seg] = __ldg(tab + (p >> P.page_log));
  }
}

template <int NT, int DK, int DV>
__device__ __forceinline__ void tma_kv_stage_pg(const FmhaParams& P, char* dstk,
                                                char* dstv, const void* kmap,
                                                const void* vmap, int kvh,
                                                int kv_base, int last,
                                                const int32_t* pg,
                                                uint32_t mbar) {
  mbar_expect_tx(mbar, NT * (DK + DV) * 2);
  const int R = min(NT, 1 << P.page_log);
  const int nseg = NT / R;
#pragma unroll
  for (int seg = 0; seg < NT / 16; ++seg) {
    if (seg >= nseg) break;
    const int p = min(kv_base + seg * R, last);
    const int page = pg[seg];
    const int slot = p & P.page_mask;
#pragma unroll
    for (int a = 0; a < DK / 64; ++a)
      tma_load_4d(kmap, dstk + a * (NT * 128) + seg * R * 128, a * 64, slot,
                  kvh, page, mbar);
#pragma unroll
    for (int a = 0; a < DV / 64; ++a)
      tma_load_4d(vmap, dstv + a * (NT * 128) + seg * R * 128, a * 64, slot,
                  kvh, page, mbar);
  }
}

// 128B swizzle within a row of `units` 16-byte units (units in {8,16,32}).
// Row-major: used for the Q tile, which is staged with cp.async.
__device__ __forceinline__ int swz(int row, int unit, int units_per_row) {
  return row * units_per_row + ((unit & ~7) | ((unit ^ row) & 7));
}

// Atom-major SW128 byte offset for the K/V tiles: atom(unit/8) is the OUTER
// stride, rows inside an atom are 128B apart and the 16B chunk inside the atom
// is XOR-swizzled by row&7. This is exactly what a SWIZZLE_128B TMA box of
// {64 bf16, R rows} writes at a 1024B-aligned destination, so the KV tiles can
// be filled by cp.async.bulk.tensor instead of one cp.async per 16B unit.
// ldmatrix stays conflict-free: for the 8 lanes of one 8x8 matrix the row term
// vanishes mod 8 banks and (unit&7)^row takes all 8 values.
__device__ __forceinline__ int amo(int row, int unit, int rows) {
  return (unit >> 3) * (rows * 128) + row * 128 + (((unit & 7) ^ (row & 7)) << 4);
}

struct TileInfo {
  int b, qlen, kvlen, m0, mrows, lo, hi;
  bool ok;
};

__device__ TileInfo map_tile(const FmhaParams& P, int tile) {
  TileInfo ti;
  int acc = 0;
  for (int b = 0; b < P.batch; ++b) {
    int q0 = P.cu_q[b], q1 = P.cu_q[b + 1];
    int qlen = q1 - q0;
    int mrows_req = qlen * P.group;
    int t = (mrows_req + P.m_tile - 1) / P.m_tile;
    if (tile < acc + t) {
      ti.b = b; ti.qlen = qlen; ti.kvlen = P.seq_lens[b];
      ti.m0 = (tile - acc) * P.m_tile;
      ti.mrows = min(P.m_tile, mrows_req - ti.m0);
      int qf = ti.m0 / P.group;
      int ql = (ti.m0 + ti.mrows - 1) / P.group;
      int vis_min = ti.kvlen - qlen + qf;
      int vis_max = ti.kvlen - qlen + ql;
      ti.hi = P.causal ? min(vis_max + 1, ti.kvlen) : ti.kvlen;
      ti.lo = (P.window_left >= 0) ? max(0, vis_min - P.window_left) : 0;
      ti.ok = (ti.mrows > 0) && (ti.hi > ti.lo);
      return ti;
    }
    acc += t;
  }
  ti.ok = false; ti.b = 0; ti.qlen = 0; ti.kvlen = 0; ti.m0 = 0; ti.mrows = 0; ti.lo = 0; ti.hi = 0;
  return ti;
}

__device__ bool split_range(const FmhaParams& P, const TileInfo& ti, int split,
                            int& kv0, int& kv1, int& active) {
  int L = ti.hi - ti.lo;
  if (P.splits_ub <= 1) { kv0 = ti.lo; kv1 = ti.hi; active = 1; return true; }
  // Balanced per-tile splits: with chunk_target > 0 each tile uses only as
  // many splits as keep every split near chunk_target tokens; the chunk
  // derives from the tile's OWN length, so the uniform-work floor does not
  // apply on this path. active < splits_ub is fine: every consumer (fused
  // counter win, combine_tile walk, combine_only) takes `active` from here.
  if (P.chunk_target > 0) {
    const int want = (L + P.chunk_target - 1) / P.chunk_target;
    const int sp = want < 1 ? 1 : (want < P.splits_ub ? want : P.splits_ub);
    int chunk_b = (L + sp - 1) / sp;
    chunk_b = (chunk_b + P.chunk_round - 1) / P.chunk_round * P.chunk_round;
    active = (L + chunk_b - 1) / chunk_b;
    if (split >= active) return false;
    kv0 = ti.lo + split * chunk_b;
    kv1 = min(kv0 + chunk_b, ti.hi);
    return true;
  }
  int chunk = (L + P.splits_ub - 1) / P.splits_ub;
  chunk = (chunk + P.chunk_round - 1) / P.chunk_round * P.chunk_round;
  // Uniform-work floor: never hand a block less than chunk_floor keys, so every
  // resident block runs about the same number of stages. chunk only grows, so
  // active stays <= splits_ub and the (kvh, tile, split) grid still covers L.
  if (P.chunk_floor > chunk)
    chunk = (P.chunk_floor + P.chunk_round - 1) / P.chunk_round * P.chunk_round;
  active = (L + chunk - 1) / chunk;
  if (split >= active) return false;
  kv0 = ti.lo + split * chunk;
  kv1 = min(kv0 + chunk, ti.hi);
  return true;
}

template <int THREADS>
__device__ void combine_tile(const FmhaParams& P, const TileInfo& ti, int tile, int kvh,
                             int active, float* sc, int zrank, int zsplit);
__device__ __forceinline__ int* counters_ptr(const FmhaParams& P);

// ---------------------------------------------------------------------------
// Shared load helpers (cooperative, all 128 threads)
// ---------------------------------------------------------------------------
template <int MT, int DQ, int THREADS>
__device__ __forceinline__ void load_q_tile(const FmhaParams& P, const TileInfo& ti,
                                            char* smem_q, int kvh, int tid) {
  constexpr int QU = DQ / 8;
  constexpr int UNITS = MT * QU;
  const int cu = P.cu_q[ti.b];
#pragma unroll
  for (int i = 0; i < (UNITS + THREADS - 1) / THREADS; ++i) {
    const int u = tid + i * THREADS;
    if (UNITS % THREADS != 0 && u >= UNITS) break;
    const int row = u / QU, unit = u % QU;
    const int rg = ti.m0 + row;
    const int q_idx = min(rg / P.group, ti.qlen - 1);
    const int head = kvh * P.group + (rg % P.group);
    const __nv_bfloat16* src = P_Q + (int64_t)(cu + q_idx) * P.q_tok_stride
                             + (int64_t)head * P.q_head_stride + unit * 8;
    cp_async16(smem_u32(smem_q + swz(row, unit, QU) * 16), src);
  }
}

template <int NT, int DQ, int DV, int THREADS>
__device__ __forceinline__ void load_kv_stage(const FmhaParams& P, const TileInfo& ti,
                                              char* smem_k, char* smem_v,
                                              int kvh, int kv_base, int tid) {
  constexpr int KU = DQ / 8, VU = DV / 8;
  constexpr int UK = NT * KU, UV = NT * VU;
  const int last = ti.hi - 1;
  const int32_t* tab = P.table + ti.b * P.tstride;
#ifdef FMHA_LOADER_V2
  // REJECTED by A/B dv_17485e2f3b31: 20-27% SLOWER on every case (c1 x0.74,
// c6 x0.73, c2 x0.78): per-iteration __shfl_sync serializes the cp.async
// issue chain behind the head lane's __ldg latency, destroying the free
// MLP of independent per-thread issues. ALU% was not the limiter.
// Warp-cooperative row-base broadcast (see fmha_tc.cu): 1 __ldg + int64
  // stride chain per U-lane group instead of per unit; ~30% fewer int ops in
  // loaders (NCU dv_050fe200053a: ALU top pipe 45% on tc; decode issue-bound).
  const int lane = tid & 31;
  const int64_t khb = (int64_t)kvh * P.k_head_stride;
  const int64_t vhb = (int64_t)kvh * P.v_head_stride;
  const char* kbase = reinterpret_cast<const char*>(P_K);
  const char* vbase = reinterpret_cast<const char*>(P_V);
  {
    constexpr bool DIV = (UK % THREADS == 0);
    const int gsrc = lane & ~(KU - 1);
#pragma unroll
    for (int i = 0; i < (UK + THREADS - 1) / THREADS; ++i) {
      const int u = tid + i * THREADS;
      const bool act = DIV || (u < UK);
      const int row = u / KU, unit = u & (KU - 1);
      int64_t rowb = 0;
      if (act && (lane & (KU - 1)) == 0) {
        const int p = min(kv_base + row, last);
        const int page = __ldg(tab + (p >> P.page_log));
        rowb = (int64_t)page * P.k_page_stride
             + (int64_t)(p & P.page_mask) * P.k_slot_stride + khb;
      }
      rowb = __shfl_sync(0xffffffffu, rowb, gsrc);
      if (act) {
        const __nv_bfloat16* src =
            reinterpret_cast<const __nv_bfloat16*>(kbase + rowb * 2) + unit * 8;
        cp_async16(smem_u32(smem_k + amo(row, unit, NT)), src);
      }
    }
  }
  {
    constexpr bool DIV = (UV % THREADS == 0);
    const int gsrc = lane & ~(VU - 1);
#pragma unroll
    for (int i = 0; i < (UV + THREADS - 1) / THREADS; ++i) {
      const int u = tid + i * THREADS;
      const bool act = DIV || (u < UV);
      const int row = u / VU, unit = u & (VU - 1);
      int64_t rowb = 0;
      if (act && (lane & (VU - 1)) == 0) {
        const int p = min(kv_base + row, last);
        const int page = __ldg(tab + (p >> P.page_log));
        rowb = (int64_t)page * P.v_page_stride
             + (int64_t)(p & P.page_mask) * P.v_slot_stride + vhb;
      }
      rowb = __shfl_sync(0xffffffffu, rowb, gsrc);
      if (act) {
        const __nv_bfloat16* src =
            reinterpret_cast<const __nv_bfloat16*>(vbase + rowb * 2) + unit * 8;
        cp_async16(smem_u32(smem_v + amo(row, unit, NT)), src);
      }
    }
  }
#else
#pragma unroll
  for (int i = 0; i < (UK + THREADS - 1) / THREADS; ++i) {
    const int u = tid + i * THREADS;
    if (UK % THREADS != 0 && u >= UK) break;
    const int row = u / KU, unit = u % KU;
    const int p = min(kv_base + row, last);
    const int page = __ldg(tab + (p >> P.page_log));
    const __nv_bfloat16* src = P_K + (int64_t)page * P.k_page_stride
                             + (int64_t)kvh * P.k_head_stride
                             + (int64_t)(p & P.page_mask) * P.k_slot_stride + unit * 8;
    cp_async16(smem_u32(smem_k + amo(row, unit, NT)), src);
  }
#pragma unroll
  for (int i = 0; i < (UV + THREADS - 1) / THREADS; ++i) {
    const int u = tid + i * THREADS;
    if (UV % THREADS != 0 && u >= UV) break;
    const int row = u / VU, unit = u % VU;
    const int p = min(kv_base + row, last);
    const int page = __ldg(tab + (p >> P.page_log));
    const __nv_bfloat16* src = P_V + (int64_t)page * P.v_page_stride
                             + (int64_t)kvh * P.v_head_stride
                             + (int64_t)(p & P.page_mask) * P.v_slot_stride + unit * 8;
    cp_async16(smem_u32(smem_v + amo(row, unit, NT)), src);
  }
#endif
}

// Stable base-2 logaddexp: returns total2, and *scale = exp2(base2 - total2).
__device__ __forceinline__ float sink_adjust(float base2, float sink2, float* scale) {
  float d = sink2 - base2;
  float mx = fmaxf(base2, sink2);
  float total2 = mx + log2f(1.0f + exp2f(-fabsf(d)));
  *scale = exp2f(base2 - total2);
  return total2;
}

// ---------------------------------------------------------------------------
// Main kernel. MT=64: warps split M (16 rows each). MT=16 (WNSPLIT): warps
// split N (16 kv cols each) over one shared m16 slice; merged in epilogue.
// ---------------------------------------------------------------------------
#ifdef FMHA_DEC_PHASE
// ---------------------------------------------------------------------------
// DIAGNOSTIC ONLY (never in shipped defaults: solution.py adds
// -DFMHA_DEC_PHASE only when FMHA_DEC_PHASE=1 is in the environment). m16_body
// accumulates per-phase SM-cycle totals (clock64 deltas; every warp leader
// folds its sums in once at each function exit). experiments/dec_probe.py
// reads/resets the slots through the extern "C" entries below.
// Slots: 0 issue, 1 wait+sync, 2 QK, 3 softmax, 4 PV, 5 loop tail,
//        6 stage samples, 7 warp leaders, 8 prologue, 9 epilogue.
__device__ unsigned long long g_phd[16];
extern "C" void fmha_dec_phase_read(unsigned long long* out) {
  cudaMemcpyFromSymbol(out, g_phd, 16 * sizeof(unsigned long long));
}
extern "C" void fmha_dec_phase_reset() {
  const unsigned long long z[16] = {0};
  cudaMemcpyToSymbol(g_phd, z, sizeof(z));
}
#define DPH_TS(name) const long long name = clock64();
#define DPH_SET(name) name = clock64();
#define DEC_FOLD()                                                                 \
  {                                                                                \
    DPH_TS(de1)                                                                    \
    d_epi += de1 - de0;                                                            \
    if ((tid & 31) == 0) {                                                         \
      atomicAdd(&g_phd[0], (unsigned long long)d_iss);                             \
      atomicAdd(&g_phd[1], (unsigned long long)d_wait);                            \
      atomicAdd(&g_phd[2], (unsigned long long)d_qk);                              \
      atomicAdd(&g_phd[3], (unsigned long long)d_soft);                            \
      atomicAdd(&g_phd[4], (unsigned long long)d_pv);                              \
      atomicAdd(&g_phd[5], (unsigned long long)d_tail);                            \
      atomicAdd(&g_phd[6], (unsigned long long)d_n);                               \
      atomicAdd(&g_phd[7], 1ull);                                                  \
      atomicAdd(&g_phd[8], (unsigned long long)d_pro);                             \
      atomicAdd(&g_phd[9], (unsigned long long)d_epi);                             \
    }                                                                              \
  }
#else
#define DPH_TS(name)
#define DPH_SET(name)
#define DEC_FOLD()
#endif

template <int DQ, int DV, int MT, int NT, int STAGES, bool WNSPLIT, bool FUSE,
          bool TMA = false>
__device__ void m16_body(const FmhaParams& P, const TileInfo& ti, char* smem,
                         int kvh, int split, int kv0, int kv1, int nstages,
                         int active, int kv0_exact, const void* kmap,
                         const void* vmap);

template <int DQ, int DV, int MT, int NT, int STAGES, bool WNSPLIT, bool FUSE,
          int MINB = 1, bool TMA = false>
__global__ void __launch_bounds__(256, MINB) fmha_main(
    const FmhaParams P
#ifdef FMHA_TMA
    , const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap
#endif
    ) {
  extern __shared__ __align__(16) char smem[];
  // Grid is (kv head, tile, split): blockIdx.x varies fastest, so with the
  // causal tile reversal the heaviest tiles of EVERY head are dispatched first
  // (global longest-processing-time order). With the tile as x, each head's
  // full tile range was dispatched before the next head started, which pushed
  // the heaviest tiles of the later heads into the last wave (fixed c7: 520
  // blocks over 296 slots, heaviest block 128 stages vs 112 average).
  const int kvh = blockIdx.x, tile = blockIdx.y, split = blockIdx.z;
  TileInfo ti = map_tile(P, tile);
  if (!ti.ok) return;
  int kv0, kv1, active;
  if (!split_range(P, ti, split, kv0, kv1, active)) return;
  // Stage starts must be NT-aligned for TMA boxes; kv0_exact keeps the logical
  // split boundary so the leading rows stay masked (see m16_body).
  const int kv0_exact = kv0;
#ifdef FMHA_TMA
  if constexpr (TMA) kv0 &= ~(NT - 1);
  if constexpr (WNSPLIT) {
    const int nstages_t = (kv1 - kv0 + NT - 1) / NT;
    m16_body<DQ, DV, MT, NT, STAGES, WNSPLIT, FUSE, TMA>(
        P, ti, smem, kvh, split, kv0, kv1, nstages_t, active, kv0_exact,
        TMA ? (const void*)&kmap : nullptr, TMA ? (const void*)&vmap : nullptr);
    return;
  }
#else
  if constexpr (WNSPLIT) {
    const int nstages_t = (kv1 - kv0 + NT - 1) / NT;
    m16_body<DQ, DV, MT, NT, STAGES, WNSPLIT, FUSE, false>(
        P, ti, smem, kvh, split, kv0, kv1, nstages_t, active, kv0_exact,
        nullptr, nullptr);
    return;
  }
#endif
  const int nstages = (kv1 - kv0 + NT - 1) / NT;
  const int tid = threadIdx.x;
  const int q_bytes = MT * DQ * 2, kb_bytes = NT * DQ * 2, vb_bytes = NT * DV * 2;
  char* smem_q = smem;
  char* smem_kv = smem + q_bytes;
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  constexpr int QU = DQ / 8, KU = DQ / 8, VU = DV / 8;
  constexpr bool QPRE = (DQ <= 128);
  constexpr int NQF = QPRE ? DQ / 16 : 1;

  load_q_tile<MT, DQ, 128>(P, ti, smem_q, kvh, tid);
  cp_async_commit();
#pragma unroll
  for (int j = 0; j < STAGES - 1; ++j) {
    if (j < nstages)
      load_kv_stage<NT, DQ, DV, 128>(P, ti, smem_kv + j * (kb_bytes + vb_bytes),
                                     smem_kv + j * (kb_bytes + vb_bytes) + kb_bytes, kvh, kv0 + j * NT, tid);
    cp_async_commit();
  }

  float acc_o[DV / 8][4];
#pragma unroll
  for (int j = 0; j < DV / 8; ++j)
    acc_o[j][0] = acc_o[j][1] = acc_o[j][2] = acc_o[j][3] = 0.f;
  float mrow0 = NEG_INF, mrow1 = NEG_INF, lt0 = 0.f, lt1 = 0.f;

  const int r0 = warp * 16 + g, r1 = r0 + 8;
  const int rg0 = ti.m0 + r0, rg1 = ti.m0 + r1;
  const int vis0 = ti.kvlen - ti.qlen + rg0 / P.group;
  const int vis1 = ti.kvlen - ti.qlen + rg1 / P.group;
  const int rhi0 = min(P.causal ? vis0 : ti.kvlen - 1, kv1 - 1);
  const int rhi1 = min(P.causal ? vis1 : ti.kvlen - 1, kv1 - 1);
  const int rlo0 = (P.window_left >= 0) ? max(0, vis0 - P.window_left) : 0;
  const int rlo1 = (P.window_left >= 0) ? max(0, vis1 - P.window_left) : 0;
  const uint32_t qaddr = smem_u32(smem_q);

  // Warp-uniform interior bounds: rows [warp*16, last valid row of this warp].
  int rlastv = max(0, min(15, min(MT - 1, ti.mrows - 1) - warp * 16));
  const int visf = ti.kvlen - ti.qlen + (ti.m0 + warp * 16) / P.group;
  const int visl = ti.kvlen - ti.qlen + (ti.m0 + warp * 16 + rlastv) / P.group;
  const int rhi_w = min(P.causal ? visf : ti.kvlen - 1, kv1 - 1);
  const int rlo_w = (P.window_left >= 0) ? max(0, visl - P.window_left) : 0;

  uint32_t qf[NQF][4];
  if constexpr (QPRE) {
    cp_async_wait<STAGES - 1>();
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < DQ / 16; ++ks)
      ldmatrix_x4(qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3],
                  qaddr + swz(warp * 16 + (lane & 15), ks * 2 + (lane >> 4), QU) * 16);
  }

  int cur = 0;
  for (int s = 0; s < nstages; ++s) {
    const int nx = s + STAGES - 1;
    if (nx < nstages) {
      int nb = cur + STAGES - 1;
      if (nb >= STAGES) nb -= STAGES;
      char* bk = smem_kv + nb * (kb_bytes + vb_bytes);
      load_kv_stage<NT, DQ, DV, 128>(P, ti, bk, bk + kb_bytes, kvh, kv0 + nx * NT, tid);
    }
    cp_async_commit();
    cp_async_wait<STAGES - 1>();
    __syncthreads();
    const char* kbuf = smem_kv + cur * (kb_bytes + vb_bytes);
    const uint32_t kbase = smem_u32(kbuf);
    const uint32_t vbase = smem_u32(kbuf + kb_bytes);
    float sacc[NT / 8][4];
#pragma unroll
    for (int j = 0; j < NT / 8; ++j)
      sacc[j][0] = sacc[j][1] = sacc[j][2] = sacc[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < DQ / 16; ++ks) {
      uint32_t a0, a1, a2, a3;
      if constexpr (QPRE) {
        a0 = qf[ks][0]; a1 = qf[ks][1]; a2 = qf[ks][2]; a3 = qf[ks][3];
      } else {
        ldmatrix_x4(a0, a1, a2, a3,
                    qaddr + swz(warp * 16 + (lane & 15), ks * 2 + (lane >> 4), QU) * 16);
      }
#pragma unroll
      for (int nb = 0; nb < NT / 16; ++nb) {
        uint32_t b0, b1, b2, b3;
        ldmatrix_x4(b0, b1, b2, b3,
                    kbase + amo(nb * 16 + (lane & 7) + ((lane >> 4) << 3),
                                ks * 2 + ((lane >> 3) & 1), NT));
        mma_bf16(sacc[2 * nb][0], sacc[2 * nb][1], sacc[2 * nb][2], sacc[2 * nb][3],
                 a0, a1, a2, a3, b0, b1);
        mma_bf16(sacc[2 * nb + 1][0], sacc[2 * nb + 1][1], sacc[2 * nb + 1][2], sacc[2 * nb + 1][3],
                 a0, a1, a2, a3, b2, b3);
      }
    }
    const int pbase = kv0 + s * NT;
    const bool interior = (pbase >= rlo_w) && (pbase + NT - 1 <= rhi_w);
    float rmax0 = NEG_INF, rmax1 = NEG_INF;
    // Branched mask, not if-converted predicates: on interior stages (the
    // majority) the compare/select chain is dead code that still occupies issue
    // slots. Same fix as the tcgen05 kernels; see the c4 SASS census.
#ifdef FMHA_MASK_PRED
#pragma unroll
    for (int j = 0; j < NT / 8; ++j) {
      const int c = pbase + j * 8 + 2 * t;
      sacc[j][0] *= P.qk_scale2; sacc[j][1] *= P.qk_scale2;
      sacc[j][2] *= P.qk_scale2; sacc[j][3] *= P.qk_scale2;
      if (!interior) {
        if (c < rlo0 || c > rhi0) sacc[j][0] = NEG_INF;
        if (c + 1 < rlo0 || c + 1 > rhi0) sacc[j][1] = NEG_INF;
        if (c < rlo1 || c > rhi1) sacc[j][2] = NEG_INF;
        if (c + 1 < rlo1 || c + 1 > rhi1) sacc[j][3] = NEG_INF;
      }
      rmax0 = fmaxf(rmax0, fmaxf(sacc[j][0], sacc[j][1]));
      rmax1 = fmaxf(rmax1, fmaxf(sacc[j][2], sacc[j][3]));
    }
#else
    if (interior) {
#pragma unroll
      for (int j = 0; j < NT / 8; ++j) {
        sacc[j][0] *= P.qk_scale2; sacc[j][1] *= P.qk_scale2;
        sacc[j][2] *= P.qk_scale2; sacc[j][3] *= P.qk_scale2;
        rmax0 = fmaxf(rmax0, fmaxf(sacc[j][0], sacc[j][1]));
        rmax1 = fmaxf(rmax1, fmaxf(sacc[j][2], sacc[j][3]));
      }
    } else {
#pragma unroll
      for (int j = 0; j < NT / 8; ++j) {
        const int c = pbase + j * 8 + 2 * t;
        sacc[j][0] *= P.qk_scale2; sacc[j][1] *= P.qk_scale2;
        sacc[j][2] *= P.qk_scale2; sacc[j][3] *= P.qk_scale2;
        if (c < rlo0 || c > rhi0) sacc[j][0] = NEG_INF;
        if (c + 1 < rlo0 || c + 1 > rhi0) sacc[j][1] = NEG_INF;
        if (c < rlo1 || c > rhi1) sacc[j][2] = NEG_INF;
        if (c + 1 < rlo1 || c + 1 > rhi1) sacc[j][3] = NEG_INF;
        rmax0 = fmaxf(rmax0, fmaxf(sacc[j][0], sacc[j][1]));
        rmax1 = fmaxf(rmax1, fmaxf(sacc[j][2], sacc[j][3]));
      }
    }
#endif
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));
    const float mn0 = fmaxf(mrow0, rmax0), mn1 = fmaxf(mrow1, rmax1);
    const float ms0 = (mn0 == NEG_INF) ? 0.f : mn0;
    const float ms1 = (mn1 == NEG_INF) ? 0.f : mn1;
    const float sc0 = exp2f(mrow0 - ms0), sc1 = exp2f(mrow1 - ms1);
    mrow0 = mn0; mrow1 = mn1;
    lt0 = lt0 * sc0; lt1 = lt1 * sc1;
#pragma unroll
    for (int j = 0; j < DV / 8; ++j) {
      acc_o[j][0] *= sc0; acc_o[j][1] *= sc0; acc_o[j][2] *= sc1; acc_o[j][3] *= sc1;
    }
    uint32_t pfrag[NT / 16][4];
#pragma unroll
    for (int j = 0; j < NT / 8; ++j) {
#ifdef FMHA_EXP_FFMA2
      const float2 d01 = fsub2(make_float2(sacc[j][0], sacc[j][1]),
                               make_float2(ms0, ms0));
      const float2 d23 = fsub2(make_float2(sacc[j][2], sacc[j][3]),
                               make_float2(ms1, ms1));
      const float p0 = exp2f(d01.x);
      const float p1 = exp2f(d01.y);
      const float p2 = exp2f(d23.x);
      const float p3 = exp2f(d23.y);
#else
      const float p0 = exp2f(sacc[j][0] - ms0);
      const float p1 = exp2f(sacc[j][1] - ms0);
      const float p2 = exp2f(sacc[j][2] - ms1);
      const float p3 = exp2f(sacc[j][3] - ms1);
#endif
      lt0 += p0 + p1; lt1 += p2 + p3;
      sacc[j][0] = p0; sacc[j][1] = p1; sacc[j][2] = p2; sacc[j][3] = p3;
    }
#pragma unroll
    for (int jj = 0; jj < NT / 16; ++jj) {
      pfrag[jj][0] = pack_bf16(sacc[2 * jj][0], sacc[2 * jj][1]);
      pfrag[jj][1] = pack_bf16(sacc[2 * jj][2], sacc[2 * jj][3]);
      pfrag[jj][2] = pack_bf16(sacc[2 * jj + 1][0], sacc[2 * jj + 1][1]);
      pfrag[jj][3] = pack_bf16(sacc[2 * jj + 1][2], sacc[2 * jj + 1][3]);
    }
#pragma unroll
    for (int jj = 0; jj < NT / 16; ++jj) {
#pragma unroll
      for (int vb = 0; vb < DV / 16; ++vb) {
        uint32_t v0, v1, v2, v3;
        ldmatrix_x4_trans(v0, v1, v2, v3,
                          vbase + amo(jj * 16 + (lane & 7) + (((lane >> 3) & 1) << 3),
                                      vb * 2 + (lane >> 4), NT));
        mma_bf16(acc_o[2 * vb][0], acc_o[2 * vb][1], acc_o[2 * vb][2], acc_o[2 * vb][3],
                 pfrag[jj][0], pfrag[jj][1], pfrag[jj][2], pfrag[jj][3], v0, v1);
        mma_bf16(acc_o[2 * vb + 1][0], acc_o[2 * vb + 1][1], acc_o[2 * vb + 1][2], acc_o[2 * vb + 1][3],
                 pfrag[jj][0], pfrag[jj][1], pfrag[jj][2], pfrag[jj][3], v2, v3);
      }
    }
    __syncthreads();
    cur = (cur + 1 == STAGES) ? 0 : cur + 1;
  }

  // ---- M64 epilogue ----
  lt0 += __shfl_xor_sync(0xffffffffu, lt0, 1);
  lt0 += __shfl_xor_sync(0xffffffffu, lt0, 2);
  lt1 += __shfl_xor_sync(0xffffffffu, lt1, 1);
  lt1 += __shfl_xor_sync(0xffffffffu, lt1, 2);
  const bool valid0 = r0 < ti.mrows, valid1 = r1 < ti.mrows;
  const int cu = P.cu_q[ti.b];
  const int q0i = min(rg0 / P.group, ti.qlen - 1), q1i = min(rg1 / P.group, ti.qlen - 1);
  const int head0 = kvh * P.group + (rg0 % P.group), head1 = kvh * P.group + (rg1 % P.group);
  const int64_t Pb = ((int64_t)tile * P.kvh + kvh) * P.splits_ub + split;
  if (P.splits_ub > 1) {
    float* po = P.part_o + Pb * MT * DV;
    const int64_t blk = (int64_t)tile * P.kvh + kvh;
#pragma unroll
    for (int j = 0; j < DV / 8; ++j) {
      const int col = j * 8 + 2 * t;
      *reinterpret_cast<float2*>(po + r0 * DV + col) = make_float2(acc_o[j][0], acc_o[j][1]);
      *reinterpret_cast<float2*>(po + r1 * DV + col) = make_float2(acc_o[j][2], acc_o[j][3]);
    }
    if (t == 0) {
      const int Spad = (P.splits_ub + 3) & ~3;
      float* pm0 = P.part_ml + ((blk * MT + r0) * 2) * Spad + split;
      float* pm1 = P.part_ml + ((blk * MT + r1) * 2) * Spad + split;
      pm0[0] = mrow0; pm0[Spad] = lt0;
      pm1[0] = mrow1; pm1[Spad] = lt1;
    }
    cudaTriggerProgrammaticLaunchCompletion();
    return;
  }
  float lse2_0 = mrow0 + log2f(lt0), lse2_1 = mrow1 + log2f(lt1);
  float inv0 = P.v_scale / lt0, inv1 = P.v_scale / lt1;
  if (P.has_sinks) {
    float os0, os1;
    lse2_0 = sink_adjust(lse2_0, P.sinks[head0 * P.sink_stride] * P.log2e, &os0);
    lse2_1 = sink_adjust(lse2_1, P.sinks[head1 * P.sink_stride] * P.log2e, &os1);
    inv0 *= os0; inv1 *= os1;
  }
  const int64_t tok0 = cu + q0i, tok1 = cu + q1i;
  __nv_bfloat16* orow0 = P_OUT + tok0 * P.out_tok_stride + (int64_t)head0 * P.out_head_stride;
  __nv_bfloat16* orow1 = P_OUT + tok1 * P.out_tok_stride + (int64_t)head1 * P.out_head_stride;
#pragma unroll
  for (int j = 0; j < DV / 8; ++j) {
    const int col = j * 8 + 2 * t;
    if (valid0)
      *reinterpret_cast<__nv_bfloat162*>(orow0 + col) =
          __floats2bfloat162_rn(acc_o[j][0] * inv0, acc_o[j][1] * inv0);
    if (valid1)
      *reinterpret_cast<__nv_bfloat162*>(orow1 + col) =
          __floats2bfloat162_rn(acc_o[j][2] * inv1, acc_o[j][3] * inv1);
  }
  if (P.has_lse && t == 0) {
    if (valid0) P.lse[tok0 * P.lse_tok_stride + head0] = lse2_0;
    if (valid1) P.lse[tok1 * P.lse_tok_stride + head1] = lse2_1;
  }
}

// ---------------------------------------------------------------------------
// MT=16 path: warps split each KV tile into 16-column slices; merged at end.
// DV==256 uses 8 warps: (n-slice = warp&3, dv-half = warp>>2) halving the
// per-warp O accumulator; QK is duplicated across the two dv halves.
// ---------------------------------------------------------------------------
template <int DQ, int DV, int MT, int NT, int STAGES, bool WNSPLIT, bool FUSE,
          bool TMA>
__device__ void m16_body(const FmhaParams& P, const TileInfo& ti, char* smem,
                         int kvh, int split, int kv0, int kv1, int nstages,
                         int active, int kv0_exact, const void* kmap,
                         const void* vmap) {
  // Warp grid = NSL n-slices x NDV dv-halves. d256 needs 8 warps to halve the
  // per-warp O accumulator; d128 keeps 4 warps (8 warps measured slower, see
  // FMHA_DEC_W8_ON in solution.py).
  // FMHA_DEC_W8_ON selects the 8-warp d128 block for A/B (measured slower).
#ifdef FMHA_DEC_D256W4
  // NT=32 d256 decode with 4 warps: NSL=2, NDV=2, DVH=128. Keeping NDV at 2
  // holds the duplicated-QK waste at 2x (8 warps at NT=32 would need NDV=4),
  // and 128 threads + NT=32 halves the block smem to ~106 KB so TWO CTAs fit
  // per SM. c2 is DRAM-bound at 1 CTA/SM: it moves 126 MB at ~2.5 of the
  // measured 7.10 TB/s, and the peaks probe shows a single 256-thread CTA per SM
  // caps out near 3.94 TB/s while 2 CTAs/SM reach 6.30 TB/s.
  constexpr int NWARP = (DV == 256 && NT >= 64) ? 8 : 4;
#elif defined(FMHA_DEC_W8_ON)
  constexpr int NWARP = (DV >= 128) ? 8 : 4;   // measured slower; see solution.py
#else
  constexpr int NWARP = (DV == 256) ? 8 : 4;
#endif
  constexpr int THREADS = NWARP * 32;
  // Warp grid = NSL n-slices (16 kv cols each) x NDV dv-quarters. NT64:
  // NSL=4 (d256: NDV=2, DVH=128; d128 8-warp: NDV=2, DVH=64; d64: NDV=1).
  // NT32 (d256): NSL=2, NDV=4, DVH=64 -> acc_o halves, regs fit
  // __launch_bounds__(256,2) and smem drops to ~106KB -> 2 blocks/SM.
  constexpr int NSL = NT / 16;
  constexpr int NDV = NWARP / NSL;
  constexpr int DVH = DV / NDV;
  const int tid = threadIdx.x;
  const int tile = (int)blockIdx.y;   // grid is (kv head, tile, split)
  const int q_bytes = MT * DQ * 2, kb_bytes = NT * DQ * 2, vb_bytes = NT * DV * 2;
  // The SWIZZLE_128B pattern the TMA engine writes is derived from shared
  // address bits [4:9]: the 16B chunk inside a 128B row is XORed with the
  // 128B-row index mod 8. amo() reproduces that only if the tile base is
  // 1024B-aligned, so align the KV region explicitly. The FUSE instantiations
  // declare static smem (__shared__ int win), which shifts the dynamic window
  // 16B off the boundary the cp.async-only kernels enjoy (fmha_smem reserves
  // the padding); without this they silently read/write mismatched swizzles.
  const int kv_pad = TMA
      ? (int)((1024u - ((smem_u32(smem) + (unsigned)q_bytes) & 1023u)) & 1023u) : 0;
  char* smem_q = smem;
  char* smem_kv = smem + q_bytes + kv_pad;
  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  constexpr int QU = DQ / 8, KU = DQ / 8, VU = DV / 8;
#ifdef FMHA_DEC_QPRE256
  // Preload the stage-invariant Q fragments into registers for d256 too
  // (qf[16][4] = 64 regs): removes 16 of ~40 per-warp-per-stage ldmatrix
  // ops (40% of fragment traffic) from the !QPRE in-loop path. Budget:
  // ~164 (NCU c2) + ~60 <= 255 (MINB=1) without PIPE. Peer arm-b mechanism
  // (join 20261022, their shipped default); the V0-era DQ<=128 gate predates
  // the current register picture. Spill watch: fused-d256 boundary shapes.
  constexpr bool QPRE = (DQ <= 256);
#else
  constexpr bool QPRE = (DQ <= 128);
#endif
  constexpr int NQF = QPRE ? DQ / 16 : 1;
  constexpr int NK = DQ / 16;
  // Two-chain mma + fragment prefetch (see the stage loop). MEASURED SLIGHTLY
  // SLOWER (job dv_164157: fixed geomean 119.5us with, 119.1us without; c2 main
  // 48.8 vs 48.3us) despite targeting the `wait`/`short_scoreboard` stalls, so
  // it stays opt-in: the decode stage loop is smem/tensor-throughput bound, not
  // dependency-chain bound. FMHA_DEC_PIPE=1 to re-test.
#ifdef FMHA_DEC_PIPE_ON
  // FUSE excluded (peer arm-b refinement, their dv_63aeb29d5146): the fused
  // build carries combine_tile inline and PIPE's ~24 extra registers spill
  // there (boundary b6 fused-d256 x0.331 with PIPE) while non-fused c2/d256
  // won x1.033. Inert while FMHA_DEC_PIPE defaults OFF.
  constexpr bool PIPE = (NK >= 8) && !FUSE;
#else
  constexpr bool PIPE = false;
#endif
#ifdef FMHA_DEC_QKSHARE
  // d256 decode duplicates the whole QK+softmax pass across the two dv-half
  // warp groups (warp w and w+NSL compute the identical S for the same 16-kv
  // slice; the ab22 phase probe measured QK at 60.5% of the c2 stage). Share
  // it: only the producer warps (dv half 0) run the K ldmatrix + QK mmas +
  // softmax, then publish {pfrag[4], mn0, mn1} per lane into the Q staging
  // region (dead during the stage loop because QPRE preloads the Q fragments
  // to registers). Consumers take them behind one extra mid-stage barrier and
  // reproduce the mrow/skip/sc recurrence BITWISE from the shared mn (same
  // fp32 ops on same inputs), then run PV for their dv half. Consumer lt is
  // never read downstream (the epilogue ll sums only w < NSL; consumer mrow
  // equals the producer's). Output must be bitwise identical to the unshared
  // build; any mismatch is a bug. Needs QPRE (smem_q alias) and excludes
  // PIPE. Halves per-SMSP QK tensor work: NSL mma-chains instead of NWARP.
  constexpr bool QKS = (DV == 256 && NDV == 2 && NWARP == 2 * NSL
                        && QPRE && !PIPE);
#else
  constexpr bool QKS = false;
#endif
  // One mbarrier per KV slot, just past the last stage (reserved by fmha_smem).
  // TMA completions arrive there, replacing cp.async commit-group accounting.
  uint64_t* const slot_bar =
      reinterpret_cast<uint64_t*>(smem_kv + STAGES * (kb_bytes + vb_bytes));
  // kv0 may already be rounded DOWN to NT (TMA stage alignment); rows in
  // [kv0, kv0_exact) belong to the previous split and must stay masked,
  // otherwise window_left>0 (lo not NT-aligned) double-counts them.
  const int last = ti.hi - 1;
  const int32_t* const tab = P.table + ti.b * P.tstride;
#ifdef FMHA_DEC_PGPRE
  int32_t pgpre[NT / 16];   // tid0-only: next issued stage's physical page IDs
#endif
#ifdef FMHA_DEC_PHASE
  long long d_iss = 0, d_wait = 0, d_qk = 0, d_soft = 0, d_pv = 0, d_tail = 0,
            d_n = 0, d_pro = 0, d_epi = 0;
  DPH_TS(dp0)
#endif

  if constexpr (TMA) {
    if (tid == 0) {
#pragma unroll
      for (int j = 0; j < STAGES; ++j) mbar_init(smem_u32(slot_bar + j), 1);
    }
  }
  load_q_tile<MT, DQ, THREADS>(P, ti, smem_q, kvh, tid);
  cp_async_commit();
  if constexpr (TMA) {
    __syncthreads();      // barriers constructed before they are armed/used
    const int pre = (nstages < STAGES - 1) ? nstages : (STAGES - 1);
    if (tid == 0) {
      for (int j = 0; j < pre; ++j)
        tma_kv_stage<NT, DQ, DV>(P, smem_kv + j * (kb_bytes + vb_bytes),
                                 smem_kv + j * (kb_bytes + vb_bytes) + kb_bytes,
                                 kmap, vmap, kvh, kv0 + j * NT, last, tab,
                                 smem_u32(slot_bar + j));
    }
#ifdef FMHA_DEC_PGPRE
    // Page IDs for stage STAGES-1: iteration 0 issues those boxes. (The
    // prologue stages keep the blocking form; they are off the steady-state
    // critical path.)
    if (tid == 0 && STAGES - 1 < nstages)
      tma_pages_pre<NT>(P, tab, kv0 + (STAGES - 1) * NT, last, pgpre);
#endif
    // Q is staged with cp.async; the TMA loop does not use commit groups, so
    // drain Q explicitly (DQ=256 reads Q fragments inside the stage loop).
    cp_async_wait<0>();
#ifdef FMHA_DEC_NOBAR
    // ...and join it across threads HERE: with the in-loop leading barrier
    // dropped, this one-time barrier is the cross-thread completion proof for
    // the cooperative load_q_tile copies (per-thread cp_async_wait only drains
    // the caller's own group). Previously supplied by iteration 0's leading
    // __syncthreads; DQ=256 (!QPRE) ldmatrix's smem_q inside the loop.
    // Peer arm-b mechanism (join 20261022).
    __syncthreads();
#endif
  } else {
#pragma unroll
    for (int j = 0; j < STAGES - 1; ++j) {
      if (j < nstages)
        load_kv_stage<NT, DQ, DV, THREADS>(P, ti, smem_kv + j * (kb_bytes + vb_bytes),
                                           smem_kv + j * (kb_bytes + vb_bytes) + kb_bytes,
                                           kvh, kv0 + j * NT, tid);
      cp_async_commit();
    }
  }

  float acc_o[DVH / 8][4];
#pragma unroll
  for (int j = 0; j < DVH / 8; ++j)
    acc_o[j][0] = acc_o[j][1] = acc_o[j][2] = acc_o[j][3] = 0.f;
  float mrow0 = NEG_INF, mrow1 = NEG_INF, lt0 = 0.f, lt1 = 0.f;

  const int rg0 = ti.m0 + g, rg1 = ti.m0 + g + 8;
  const int vis0 = ti.kvlen - ti.qlen + rg0 / P.group;
  const int vis1 = ti.kvlen - ti.qlen + rg1 / P.group;
  const int rhi0 = min(P.causal ? vis0 : ti.kvlen - 1, kv1 - 1);
  const int rhi1 = min(P.causal ? vis1 : ti.kvlen - 1, kv1 - 1);
  const int rlo0 = max((P.window_left >= 0) ? max(0, vis0 - P.window_left) : 0,
                       kv0_exact);
  const int rlo1 = max((P.window_left >= 0) ? max(0, vis1 - P.window_left) : 0,
                       kv0_exact);
  const uint32_t qaddr = smem_u32(smem_q);
  const int wcol = (warp % NSL) * 16;
  const int dvoff = (warp / NSL) * DVH;
  // QKSHARE roles: producers = dv-half-0 warps (w < NSL). qks aliases the
  // Q staging region, which is dead once the QPRE preload below has run.
  const bool prod = !QKS || (warp < NSL);
  float* const qks = reinterpret_cast<float*>(smem_q);

  const int rlastv = min(15, ti.mrows - 1);
  const int visf = ti.kvlen - ti.qlen + ti.m0 / P.group;
  const int visl = ti.kvlen - ti.qlen + (ti.m0 + rlastv) / P.group;
  const int rhi_w = min(P.causal ? visf : ti.kvlen - 1, kv1 - 1);
  const int rlo_w = max((P.window_left >= 0) ? max(0, visl - P.window_left) : 0,
                        kv0_exact);

  uint32_t qf[NQF][4];
  if constexpr (QPRE) {
    cp_async_wait<TMA ? 0 : (STAGES - 1)>();
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < DQ / 16; ++ks)
      ldmatrix_x4(qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3],
                  qaddr + swz(lane & 15, ks * 2 + (lane >> 4), QU) * 16);
    // QKS aliases smem_q as its exchange buffer: every warp's preload
    // ldmatrix must be done before the stage loop's first producer STS.
    if constexpr (QKS) __syncthreads();
  }

  int cur = 0;
#ifdef FMHA_DEC_PHASE
  { DPH_TS(dp1) d_pro = dp1 - dp0; }
#endif
  for (int s = 0; s < nstages; ++s) {
    DPH_TS(dt0)
#ifdef FMHA_DEC_PHASE
    long long dt1, dt2, dt3;   // set inside the TMA/cp.async + QKS branches
#endif
    const int nx = s + STAGES - 1;
    int nb = cur + STAGES - 1;
    if (nb >= STAGES) nb -= STAGES;
    if constexpr (TMA) {
      // Slot nb was drained by the trailing barrier of iteration s-1, so it can
      // be re-armed and refilled; ONE thread issues the whole stage (8-32 bulk
      // copies) instead of NT*D/8 cp.async issues spread over all threads.
      if (nx < nstages && tid == 0) {
        char* bk = smem_kv + nb * (kb_bytes + vb_bytes);
#ifdef FMHA_DEC_PGPRE
        tma_kv_stage_pg<NT, DQ, DV>(P, bk, bk + kb_bytes, kmap, vmap, kvh,
                                    kv0 + nx * NT, last, pgpre,
                                    smem_u32(slot_bar + nb));
        // Resolve stage nx+1's page IDs now: the next iteration issues its
        // boxes right at the top, so the table load has this whole stage
        // (QK+softmax+PV, ~500-2700 cyc) to land instead of stalling tid0
        // with the CTA waiting at the __syncthreads behind it.
        if (nx + 1 < nstages)
          tma_pages_pre<NT>(P, tab, kv0 + (nx + 1) * NT, last, pgpre);
#else
        tma_kv_stage<NT, DQ, DV>(P, bk, bk + kb_bytes, kmap, vmap, kvh,
                                 kv0 + nx * NT, last, tab, smem_u32(slot_bar + nb));
#endif
      }
      DPH_SET(dt1)
      mbar_wait(smem_u32(slot_bar + cur), (s / STAGES) & 1);
      fence_proxy_async();
    } else {
      if (nx < nstages) {
        char* bk = smem_kv + nb * (kb_bytes + vb_bytes);
        load_kv_stage<NT, DQ, DV, THREADS>(P, ti, bk, bk + kb_bytes, kvh, kv0 + nx * NT, tid);
      }
      cp_async_commit();
      DPH_SET(dt1)
      cp_async_wait<STAGES - 1>();
    }
#ifdef FMHA_DEC_NOBAR
    // TMA mode: every thread individually waited slot_bar[cur] + proxy
    // fence, and the TRAILING barrier is the drain/re-arm proof for the
    // next iteration's tid0 issue - so this leading barrier is redundant.
    // Dropping it also decouples the compute warps from tid0's TMA issue
    // chain (page-table __ldg ~300cy sits between issue and wait). Peer
    // arm-b mechanism; their NCU c2 (dv_cda7a70b4949): barrier = 1.23 of
    // 8.39 stall cy/inst (15%).
    if constexpr (!TMA) __syncthreads();
#else
    __syncthreads();
#endif
    DPH_SET(dt2)
    const char* kbuf = smem_kv + cur * (kb_bytes + vb_bytes);
    const uint32_t kbase = smem_u32(kbuf);
    const uint32_t vbase = smem_u32(kbuf + kb_bytes);
    uint32_t pfrag[4];
    // QKSHARE: only the producer warps run the QK+softmax region below (see
    // the QKS constexpr comment); consumers pick {pfrag, mn} up afterwards.
    if (!QKS || prod) {
    float sacc[2][4];
    sacc[0][0] = sacc[0][1] = sacc[0][2] = sacc[0][3] = 0.f;
    sacc[1][0] = sacc[1][1] = sacc[1][2] = sacc[1][3] = 0.f;
    // NCU c2 (V0): stalls `wait` 1.73 and `short_scoreboard` 1.47 of 6.16
    // cycles/instr - the NK-long mma chain into one accumulator pair and the
    // ldmatrix latency in front of it. Two accumulator chains (even/odd k-step)
    // halve the dependency depth and the fragment load for ks+1 issues before
    // the mma of ks.
    if constexpr (PIPE) {
      float acc2[2][2][4];
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          acc2[c][j][0] = acc2[c][j][1] = acc2[c][j][2] = acc2[c][j][3] = 0.f;
      uint32_t af[2][4], bf[2][4];
      {
        if constexpr (QPRE) {
          af[0][0] = qf[0][0]; af[0][1] = qf[0][1];
          af[0][2] = qf[0][2]; af[0][3] = qf[0][3];
        } else {
          ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                      qaddr + swz(lane & 15, (lane >> 4), QU) * 16);
        }
        ldmatrix_x4(bf[0][0], bf[0][1], bf[0][2], bf[0][3],
                    kbase + amo(wcol + (lane & 7) + ((lane >> 4) << 3),
                                ((lane >> 3) & 1), NT));
      }
#pragma unroll
      for (int ks = 0; ks < NK; ++ks) {
        const int c = ks & 1, n = c ^ 1;
        if (ks + 1 < NK) {
          const int k2 = ks + 1;
          if constexpr (QPRE) {
            af[n][0] = qf[k2][0]; af[n][1] = qf[k2][1];
            af[n][2] = qf[k2][2]; af[n][3] = qf[k2][3];
          } else {
            ldmatrix_x4(af[n][0], af[n][1], af[n][2], af[n][3],
                        qaddr + swz(lane & 15, k2 * 2 + (lane >> 4), QU) * 16);
          }
          ldmatrix_x4(bf[n][0], bf[n][1], bf[n][2], bf[n][3],
                      kbase + amo(wcol + (lane & 7) + ((lane >> 4) << 3),
                                  k2 * 2 + ((lane >> 3) & 1), NT));
        }
        mma_bf16(acc2[c][0][0], acc2[c][0][1], acc2[c][0][2], acc2[c][0][3],
                 af[c][0], af[c][1], af[c][2], af[c][3], bf[c][0], bf[c][1]);
        mma_bf16(acc2[c][1][0], acc2[c][1][1], acc2[c][1][2], acc2[c][1][3],
                 af[c][0], af[c][1], af[c][2], af[c][3], bf[c][2], bf[c][3]);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) sacc[j][k] = acc2[0][j][k] + acc2[1][j][k];
    } else {
#ifdef FMHA_DEC_QK4
      if constexpr (DQ == 256 && NK >= 8) {
        // 4-way k-split QK chains (FMHA_DEC_QK4): the 2-chain x NK serial
        // accumulation is latency-bound - dec_probe (dv_fef6b0596561) put
        // c2's QK phase at 561 cyc/warp-stage (39% of the stage) while
        // QKSHARE proved halving the warp-level issue count does NOT help.
        // Four accumulator pairs cut the mma dependency depth NK/2 -> NK/4;
        // the fp32 merge reassociates (tolerance-level, not bitwise).
        // +24 regs (212 -> ~236 of the 255 MINB=1 budget on d256 NT64).
        float s4[2][4][4];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int c = 0; c < 4; ++c)
            s4[h][c][0] = s4[h][c][1] = s4[h][c][2] = s4[h][c][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < NK; ++ks) {
          uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
          if constexpr (QPRE) {
            a0 = qf[ks][0]; a1 = qf[ks][1]; a2 = qf[ks][2]; a3 = qf[ks][3];
          } else {
            ldmatrix_x4(a0, a1, a2, a3,
                        qaddr + swz(lane & 15, ks * 2 + (lane >> 4), QU) * 16);
          }
          ldmatrix_x4(b0, b1, b2, b3,
                      kbase + amo(wcol + (lane & 7) + ((lane >> 4) << 3),
                                  ks * 2 + ((lane >> 3) & 1), NT));
          const int c = ks & 3;
          mma_bf16(s4[0][c][0], s4[0][c][1], s4[0][c][2], s4[0][c][3],
                   a0, a1, a2, a3, b0, b1);
          mma_bf16(s4[1][c][0], s4[1][c][1], s4[1][c][2], s4[1][c][3],
                   a0, a1, a2, a3, b2, b3);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int c = 0; c < 4; ++c)
            sacc[h][c] = (s4[h][0][c] + s4[h][1][c])
                       + (s4[h][2][c] + s4[h][3][c]);
      } else
#endif
      {
#pragma unroll
      for (int ks = 0; ks < NK; ++ks) {
        uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
        if constexpr (QPRE) {
          a0 = qf[ks][0]; a1 = qf[ks][1]; a2 = qf[ks][2]; a3 = qf[ks][3];
        } else {
          ldmatrix_x4(a0, a1, a2, a3,
                      qaddr + swz(lane & 15, ks * 2 + (lane >> 4), QU) * 16);
        }
        ldmatrix_x4(b0, b1, b2, b3,
                    kbase + amo(wcol + (lane & 7) + ((lane >> 4) << 3),
                                ks * 2 + ((lane >> 3) & 1), NT));
        mma_bf16(sacc[0][0], sacc[0][1], sacc[0][2], sacc[0][3], a0, a1, a2, a3, b0, b1);
        mma_bf16(sacc[1][0], sacc[1][1], sacc[1][2], sacc[1][3], a0, a1, a2, a3, b2, b3);
      }
      }
    }
    DPH_SET(dt3)
    const int pbase = kv0 + s * NT + wcol;
    const bool interior = (pbase >= rlo_w) && (pbase + 15 <= rhi_w);
    float rmax0 = NEG_INF, rmax1 = NEG_INF;
    // Branched mask (see the m64 body and the tcgen05 kernels): the decode
    // stage only has 8 values per thread, so the predicated-off compare/select
    // chain is a *larger* relative waste here than in prefill.
#ifdef FMHA_MASK_PRED
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int c = pbase + j * 8 + 2 * t;
      sacc[j][0] *= P.qk_scale2; sacc[j][1] *= P.qk_scale2;
      sacc[j][2] *= P.qk_scale2; sacc[j][3] *= P.qk_scale2;
      if (!interior) {
        if (c < rlo0 || c > rhi0) sacc[j][0] = NEG_INF;
        if (c + 1 < rlo0 || c + 1 > rhi0) sacc[j][1] = NEG_INF;
        if (c < rlo1 || c > rhi1) sacc[j][2] = NEG_INF;
        if (c + 1 < rlo1 || c + 1 > rhi1) sacc[j][3] = NEG_INF;
      }
      rmax0 = fmaxf(rmax0, fmaxf(sacc[j][0], sacc[j][1]));
      rmax1 = fmaxf(rmax1, fmaxf(sacc[j][2], sacc[j][3]));
    }
#else
    if (interior) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        sacc[j][0] *= P.qk_scale2; sacc[j][1] *= P.qk_scale2;
        sacc[j][2] *= P.qk_scale2; sacc[j][3] *= P.qk_scale2;
        rmax0 = fmaxf(rmax0, fmaxf(sacc[j][0], sacc[j][1]));
        rmax1 = fmaxf(rmax1, fmaxf(sacc[j][2], sacc[j][3]));
      }
    } else {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int c = pbase + j * 8 + 2 * t;
        sacc[j][0] *= P.qk_scale2; sacc[j][1] *= P.qk_scale2;
        sacc[j][2] *= P.qk_scale2; sacc[j][3] *= P.qk_scale2;
        if (c < rlo0 || c > rhi0) sacc[j][0] = NEG_INF;
        if (c + 1 < rlo0 || c + 1 > rhi0) sacc[j][1] = NEG_INF;
        if (c < rlo1 || c > rhi1) sacc[j][2] = NEG_INF;
        if (c + 1 < rlo1 || c + 1 > rhi1) sacc[j][3] = NEG_INF;
        rmax0 = fmaxf(rmax0, fmaxf(sacc[j][0], sacc[j][1]));
        rmax1 = fmaxf(rmax1, fmaxf(sacc[j][2], sacc[j][3]));
      }
    }
#endif
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));
    const float mn0 = fmaxf(mrow0, rmax0), mn1 = fmaxf(mrow1, rmax1);
    const float ms0 = (mn0 == NEG_INF) ? 0.f : mn0;
    const float ms1 = (mn1 == NEG_INF) ? 0.f : mn1;
    // Lazy O-rescale (opt-in via _pad2 bit5): while the row max grows by less
    // than lazy_thresh (base-2), keep O/l in the old base. P <= 2^thresh and
    // bf16 relative precision is scale-invariant - same algebra as the tc
    // path (validated there since dv_0ec2bc87b895). Skips the 64-fmul acc_o
    // rescale and its exp2/dependency chain on most stages.
    const bool lazy = (P._pad2 & 32) != 0;
    const bool skip0 = lazy && (ms0 - mrow0) < P.lazy_thresh;
    const bool skip1 = lazy && (ms1 - mrow1) < P.lazy_thresh;
    const float sc0 = skip0 ? 1.f : exp2f(mrow0 - ms0);
    const float sc1 = skip1 ? 1.f : exp2f(mrow1 - ms1);
    const float pb0 = skip0 ? mrow0 : ms0;
    const float pb1 = skip1 ? mrow1 : ms1;
    if (!skip0) mrow0 = mn0;
    if (!skip1) mrow1 = mn1;
    lt0 = lt0 * sc0; lt1 = lt1 * sc1;
    if (__any_sync(0xffffffffu, !skip0 || !skip1)) {
#ifdef FMHA_DEC_PACK2
      const float2 sc01 = make_float2(sc0, sc0);
      const float2 sc23 = make_float2(sc1, sc1);
#pragma unroll
      for (int j = 0; j < DVH / 8; ++j) {
        const float2 o01 = fmul2(make_float2(acc_o[j][0], acc_o[j][1]), sc01);
        const float2 o23 = fmul2(make_float2(acc_o[j][2], acc_o[j][3]), sc23);
        acc_o[j][0] = o01.x; acc_o[j][1] = o01.y;
        acc_o[j][2] = o23.x; acc_o[j][3] = o23.y;
      }
#else
#pragma unroll
      for (int j = 0; j < DVH / 8; ++j) {
        acc_o[j][0] *= sc0; acc_o[j][1] *= sc0; acc_o[j][2] *= sc1; acc_o[j][3] *= sc1;
      }
#endif
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#ifdef FMHA_EXP_FFMA2
      const float2 d01 = fsub2(make_float2(sacc[j][0], sacc[j][1]),
                               make_float2(pb0, pb0));
      const float2 d23 = fsub2(make_float2(sacc[j][2], sacc[j][3]),
                               make_float2(pb1, pb1));
      const float p0 = exp2f(d01.x);
      const float p1 = exp2f(d01.y);
      const float p2 = exp2f(d23.x);
      const float p3 = exp2f(d23.y);
#else
      const float p0 = exp2f(sacc[j][0] - pb0);
      const float p1 = exp2f(sacc[j][1] - pb0);
      const float p2 = exp2f(sacc[j][2] - pb1);
      const float p3 = exp2f(sacc[j][3] - pb1);
#endif
      lt0 += p0 + p1; lt1 += p2 + p3;
      sacc[j][0] = p0; sacc[j][1] = p1; sacc[j][2] = p2; sacc[j][3] = p3;
    }
    pfrag[0] = pack_bf16(sacc[0][0], sacc[0][1]);
    pfrag[1] = pack_bf16(sacc[0][2], sacc[0][3]);
    pfrag[2] = pack_bf16(sacc[1][0], sacc[1][1]);
    pfrag[3] = pack_bf16(sacc[1][2], sacc[1][3]);
    if (QKS) {
      // Publish {pfrag, mn0, mn1} for this warp's dv-half sibling (warp+NSL).
      float* const dst = qks + (warp * 32 + lane) * 8;
      *reinterpret_cast<float4*>(dst) =
          make_float4(__uint_as_float(pfrag[0]), __uint_as_float(pfrag[1]),
                      __uint_as_float(pfrag[2]), __uint_as_float(pfrag[3]));
      *reinterpret_cast<float2*>(dst + 4) = make_float2(mn0, mn1);
    }
    }  // end producer / unshared QK+softmax region
    if (QKS) {
      // One extra mid-stage barrier: consumers take the packed P + row maxes
      // and replay the state recurrence bitwise (same fp32 ops on the shared
      // mn; consumer lt is never read - the epilogue ll sums w < NSL only).
      __syncthreads();
      if (!prod) {
        const float* const src = qks + ((warp - NSL) * 32 + lane) * 8;
        const float4 pf4 = *reinterpret_cast<const float4*>(src);
        pfrag[0] = __float_as_uint(pf4.x);
        pfrag[1] = __float_as_uint(pf4.y);
        pfrag[2] = __float_as_uint(pf4.z);
        pfrag[3] = __float_as_uint(pf4.w);
        const float2 mn2 = *reinterpret_cast<const float2*>(src + 4);
        const float mn0 = mn2.x, mn1 = mn2.y;
        const float ms0 = (mn0 == NEG_INF) ? 0.f : mn0;
        const float ms1 = (mn1 == NEG_INF) ? 0.f : mn1;
        const bool lazy = (P._pad2 & 32) != 0;
        const bool skip0 = lazy && (ms0 - mrow0) < P.lazy_thresh;
        const bool skip1 = lazy && (ms1 - mrow1) < P.lazy_thresh;
        const float sc0 = skip0 ? 1.f : exp2f(mrow0 - ms0);
        const float sc1 = skip1 ? 1.f : exp2f(mrow1 - ms1);
        if (!skip0) mrow0 = mn0;
        if (!skip1) mrow1 = mn1;
        if (__any_sync(0xffffffffu, !skip0 || !skip1)) {
#pragma unroll
          for (int j = 0; j < DVH / 8; ++j) {
            acc_o[j][0] *= sc0; acc_o[j][1] *= sc0;
            acc_o[j][2] *= sc1; acc_o[j][3] *= sc1;
          }
        }
        DPH_SET(dt3)
      }
    }
    DPH_TS(dt4)
    if constexpr (PIPE) {
      // Same treatment for PV: the V fragment of vb+1 is in flight while the
      // two mma of vb consume the current one (acc_o entries are independent,
      // so only the ldmatrix latency needed covering).
      uint32_t vf[2][4];
      ldmatrix_x4_trans(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                        vbase + amo(wcol + (lane & 7) + (((lane >> 3) & 1) << 3),
                                    (dvoff >> 3) + (lane >> 4), NT));
#pragma unroll
      for (int vb = 0; vb < DVH / 16; ++vb) {
        const int c = vb & 1, n = c ^ 1;
        if (vb + 1 < DVH / 16) {
          const int vn = vb + 1;
          ldmatrix_x4_trans(vf[n][0], vf[n][1], vf[n][2], vf[n][3],
                            vbase + amo(wcol + (lane & 7) + (((lane >> 3) & 1) << 3),
                                        (dvoff >> 3) + vn * 2 + (lane >> 4), NT));
        }
        mma_bf16(acc_o[2 * vb][0], acc_o[2 * vb][1], acc_o[2 * vb][2], acc_o[2 * vb][3],
                 pfrag[0], pfrag[1], pfrag[2], pfrag[3], vf[c][0], vf[c][1]);
        mma_bf16(acc_o[2 * vb + 1][0], acc_o[2 * vb + 1][1], acc_o[2 * vb + 1][2], acc_o[2 * vb + 1][3],
                 pfrag[0], pfrag[1], pfrag[2], pfrag[3], vf[c][2], vf[c][3]);
      }
    } else {
#pragma unroll
      for (int vb = 0; vb < DVH / 16; ++vb) {
        uint32_t v0, v1, v2, v3;
        ldmatrix_x4_trans(v0, v1, v2, v3,
                          vbase + amo(wcol + (lane & 7) + (((lane >> 3) & 1) << 3),
                                      (dvoff >> 3) + vb * 2 + (lane >> 4), NT));
        mma_bf16(acc_o[2 * vb][0], acc_o[2 * vb][1], acc_o[2 * vb][2], acc_o[2 * vb][3],
                 pfrag[0], pfrag[1], pfrag[2], pfrag[3], v0, v1);
        mma_bf16(acc_o[2 * vb + 1][0], acc_o[2 * vb + 1][1], acc_o[2 * vb + 1][2], acc_o[2 * vb + 1][3],
                 pfrag[0], pfrag[1], pfrag[2], pfrag[3], v2, v3);
      }
    }
    DPH_TS(dt5)
    __syncthreads();
    cur = (cur + 1 == STAGES) ? 0 : cur + 1;
    DPH_TS(dt6)
#ifdef FMHA_DEC_PHASE
    d_iss += dt1 - dt0; d_wait += dt2 - dt1; d_qk += dt3 - dt2;
    d_soft += dt4 - dt3; d_pv += dt5 - dt4; d_tail += dt6 - dt5; ++d_n;
#endif
  }
  DPH_TS(de0)

  // ---- merge across warps (same 16 rows) ----
  lt0 += __shfl_xor_sync(0xffffffffu, lt0, 1);
  lt0 += __shfl_xor_sync(0xffffffffu, lt0, 2);
  lt1 += __shfl_xor_sync(0xffffffffu, lt1, 1);
  lt1 += __shfl_xor_sync(0xffffffffu, lt1, 2);
  __syncthreads();
  float* st = reinterpret_cast<float*>(smem + q_bytes);      // [NWARP][16][DVH]
  float* ml = st + NWARP * 16 * DVH;                         // [2*NWARP][16]
  float* mstar = ml + 2 * NWARP * 16;                        // [16]
  float* sc4 = mstar + 16;                                   // [NWARP][16]
  float* lst = sc4 + NWARP * 16;                             // [16]
  float* rowinv = lst + 16;                                  // [16]
  float* rowlse = rowinv + 16;                               // [16]
  if (t == 0) {
    ml[warp * 16 + g] = mrow0; ml[warp * 16 + g + 8] = mrow1;
    ml[(NWARP + warp) * 16 + g] = lt0; ml[(NWARP + warp) * 16 + g + 8] = lt1;
  }
  __syncthreads();
  if (tid < 16) {
    float mm = NEG_INF;
#pragma unroll
    for (int w = 0; w < NWARP; ++w) mm = fmaxf(mm, ml[w * 16 + tid]);
    const float msafe = (mm == NEG_INF) ? 0.f : mm;
    float ll = 0.f;
#pragma unroll
    for (int w = 0; w < NWARP; ++w) {
      float scw = exp2f(ml[w * 16 + tid] - msafe);
      sc4[w * 16 + tid] = scw;
      if (w < NSL) ll += ml[(NWARP + w) * 16 + tid] * scw;
    }
    mstar[tid] = mm; lst[tid] = ll;
    const int rg = ti.m0 + tid;
    const int head = kvh * P.group + (rg % P.group);
    float lse2 = mm + log2f(ll);
    float inv = P.v_scale / ll;
    if (P.has_sinks) {
      float os;
      lse2 = sink_adjust(lse2, P.sinks[head * P.sink_stride] * P.log2e, &os);
      inv *= os;
    }
    rowinv[tid] = inv; rowlse[tid] = lse2;
  }
  __syncthreads();
  {
    const float scg = sc4[warp * 16 + g], scg8 = sc4[warp * 16 + g + 8];
    float* stw = st + warp * 16 * DVH;
#pragma unroll
    for (int j = 0; j < DVH / 8; ++j) {
      const int col = j * 8 + 2 * t;
      *reinterpret_cast<float2*>(stw + g * DVH + col) = make_float2(acc_o[j][0] * scg, acc_o[j][1] * scg);
      *reinterpret_cast<float2*>(stw + (g + 8) * DVH + col) = make_float2(acc_o[j][2] * scg8, acc_o[j][3] * scg8);
    }
  }
  __syncthreads();
  const int64_t Pb = ((int64_t)tile * P.kvh + kvh) * P.splits_ub + split;
  const int cu = P.cu_q[ti.b];
  if (P.splits_ub > 1) {
    float* po = P.part_o + Pb * MT * DV;
#ifdef FMHA_SCALAR_PO
    for (int idx = tid; idx < 16 * DV; idx += THREADS) {
      const int r = idx / DV, c = idx % DV;
      const int h = c / DVH, cc = c % DVH;
      float accp = 0.f;
#pragma unroll
      for (int s2 = 0; s2 < NSL; ++s2)
        accp += st[((h * NSL + s2) * 16 + r) * DVH + cc];
      po[r * DV + c] = accp;
    }
#else
    // float4 on both sides: DVH and DV are multiples of 4, `st` is 16 B aligned
    // (smem is __align__(16) and q_bytes = 32*DQ), and part_o comes from a
    // 256 B-aligned allocation, so a group of four consecutive channels of one
    // row is a legal 16 B load and store. Same values, same summation order per
    // element, 1/4 of the memory instructions on every split decode epilogue.
    for (int g = tid; g < (16 * DV) / 4; g += THREADS) {
      const int c0 = (g * 4) % DV, r = (g * 4) / DV;
      const int h = c0 / DVH, cc = c0 % DVH;
      const float* base = st + ((h * NSL) * 16 + r) * DVH + cc;
      float4 acc4 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int s2 = 0; s2 < NSL; ++s2) {
        const float4 x = *reinterpret_cast<const float4*>(base + s2 * 16 * DVH);
        acc4.x += x.x; acc4.y += x.y; acc4.z += x.z; acc4.w += x.w;
      }
      *reinterpret_cast<float4*>(po + r * DV + c0) = acc4;
    }
#endif
    if (tid < 16) {
      const int64_t blk = (int64_t)tile * P.kvh + kvh;
      const int Spad = (P.splits_ub + 3) & ~3;
      float* pm = P.part_ml + ((blk * MT + tid) * 2) * Spad + split;
      pm[0] = mstar[tid];
      pm[Spad] = lst[tid];
    }
    if constexpr (FUSE) {
      // Fused combine: last block of this (tile,kvh) reduces all splits,
      // overlapping other tiles' main work and saving a kernel launch.
      // Compile-time excluded from unfused instantiations (register budget).
      __threadfence();
      __syncthreads();
      __shared__ int win;
      if (tid == 0) {
        int* cnt = counters_ptr(P) + tile * P.kvh + kvh;
        win = (atomicAdd(cnt, 1) == active - 1) ? 1 : 0;
      }
      __syncthreads();
      if (win) {
        if (tid == 0) counters_ptr(P)[tile * P.kvh + kvh] = 0;
        combine_tile<THREADS>(P, ti, tile, kvh, active,
                              reinterpret_cast<float*>(smem), 0, 1, 0);
      }
      DEC_FOLD()
      return;
    }
    cudaTriggerProgrammaticLaunchCompletion();
    DEC_FOLD()
    return;
  }
  for (int idx = tid; idx < 16 * DV; idx += THREADS) {
    const int r = idx / DV, c = idx % DV;
    if (r >= ti.mrows) continue;
    const int h = c / DVH, cc = c % DVH;
    float acc = 0.f;
#pragma unroll
    for (int s2 = 0; s2 < NSL; ++s2)
      acc += st[((h * NSL + s2) * 16 + r) * DVH + cc];
    const int rg = ti.m0 + r;
    const int q_idx = min(rg / P.group, ti.qlen - 1);
    const int head = kvh * P.group + (rg % P.group);
    const int64_t tok = cu + q_idx;
    P_OUT[tok * P.out_tok_stride + (int64_t)head * P.out_head_stride + c] =
        __float2bfloat16_rn(acc * rowinv[r]);
    if (c == 0 && P.has_lse) P.lse[tok * P.lse_tok_stride + head] = rowlse[r];
  }
  DEC_FOLD()
}

// Device-side combine for one (tile, kvh): reduces split partials to out/lse.
// `sc` points to a scratch region of at least 64*128 + 256 floats.
// NOTE: must NOT be reachable from unfused fmha_main instantiations: inlining
// it into m16_body spilled registers under __launch_bounds__(256) (c2 x0.82 in
// v3.5 eval). The FUSE template parameter excludes it at compile time.
// (__noinline__ was tried first: every ABI call site IMA'd, job dv_3039adf33448.)
template <int THREADS>
__device__ void combine_tile(const FmhaParams& P, const TileInfo& ti, int tile, int kvh,
                             int active, float* sc, int zrank, int zsplit, int row0) {
  const int tid = threadIdx.x;
  const int MT = P.m_tile, DV = P.dv;
  float* rowinv = sc + 64 * 128;
  float* rowlse = rowinv + 128;
  float* mstar = rowlse + 128;   // per-row msafe for the windowed sc rebuild
  const int64_t Pb0 = ((int64_t)tile * P.kvh + kvh) * P.splits_ub;
  const int cu = P.cu_q[ti.b];
  {
    // Parallel phase 1: RBASE row groups x LANES lanes; lane j walks splits
    // s = j*4 + k*LANES*4 (float4) + strided scalar tail. Cuts the serial
    // L2-latency split walk by LANES (NCU dv_15b642cdc94d: combine 30us at
    // 12.5% occ, DRAM 2.1% -> pure latency; the 16-thread serial phase 1 was
    // ~half the kernel). Max/sum reduce via width-LANES shfl_xor; summation
    // order changes are fp32-rounding-level vs the old walk (tolerance-safe).
    const int RBASE = (ti.mrows <= 16) ? 16 : 128;
    const int LANES = 1 << (31 - __clz(THREADS / RBASE));  // shfl width: pow2
    const int r = tid / LANES, j = tid % LANES;
    const int64_t blk = (int64_t)tile * P.kvh + kvh;
    const int Spad = (P.splits_ub + 3) & ~3;
    const float* pm = (r < ti.mrows)
        ? P.part_ml + ((blk * MT + row0 + r) * 2) * Spad : nullptr;
    float mm = NEG_INF;
    if (pm) {
      for (int s = j * 4; s + 4 <= active; s += LANES * 4) {
        const float4 m4 = *reinterpret_cast<const float4*>(pm + s);
        mm = fmaxf(mm, fmaxf(fmaxf(m4.x, m4.y), fmaxf(m4.z, m4.w)));
      }
      for (int s = (active / 4) * 4 + j; s < active; s += LANES)
        mm = fmaxf(mm, pm[s]);
    }
#pragma unroll
    for (int o = LANES / 2; o >= 1; o >>= 1)
      mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, o, LANES));
    const float msafe = (mm == NEG_INF) ? 0.f : mm;
    float ll = 0.f;
    // ll walk WITHOUT sc stores: sc now holds only a 64-split window (phase 2
    // rebuilds weights per window from mstar - exp2f of identical inputs, so
    // values and summation order stay bit-identical for any active <= 64, and
    // active > 64 (wavefill splits) becomes representable).
    if (pm) {
      for (int s = j; s < active; s += LANES) {
        const float scw = exp2f(pm[s] - msafe);
        ll += pm[Spad + s] * scw;
      }
    }
#pragma unroll
    for (int o = LANES / 2; o >= 1; o >>= 1)
      ll += __shfl_xor_sync(0xffffffffu, ll, o, LANES);
    if (pm && j == 0) {
      const int rg = ti.m0 + r;
      const int head = kvh * P.group + (rg % P.group);
      float lse2 = mm + log2f(ll);
      float inv = P.v_scale / ll;
      if (P.has_sinks) {
        float os;
        lse2 = sink_adjust(lse2, P.sinks[head * P.sink_stride] * P.log2e, &os);
        inv *= os;
      }
      rowinv[r] = inv; rowlse[r] = lse2; mstar[r] = msafe;
    }
  }
  __syncthreads();
  const int dv_log = 31 - __clz(DV), dv_mask = DV - 1;
  const int nel = ti.mrows * DV;
  // float4-vectorized channels: part_o rows are MT*DV floats (MT*DV % 4 == 0
  // and the workspace base is 256B-aligned, so 16B loads are always valid).
  // MAXI 32 -> 8 keeps acc/rowi within the 128-register budget: MAXI=32
  // spilled (ptxas: 136B st / 572B ld under __launch_bounds__(256,2)).
  const int nel4 = nel >> 2;
  const int nch = (nel4 + THREADS - 1) / THREADS;
  const int per_z = (nch + zsplit - 1) / zsplit;
  const int cz0 = zrank * per_z;
  constexpr int MAXI = 4;
  const int64_t stride4 = ((int64_t)MT * DV) >> 2;
  // row0: DH2 (m_tile 256) combines in 128-row half-tiles; the half's rows
  // start at row0 within the (tile, kvh) part_o block.
  const float4* const po4 =
      reinterpret_cast<const float4*>(P.part_o) + Pb0 * stride4
      + (((int64_t)row0 * DV) >> 2);
  // Windowed sc rebuild state: sc covers SCW=64 splits; each window is
  // rebuilt by the phase-1 (row, lane) mapping from pmw/mstar. Bit-identical
  // weights for any active (exp2f of the same inputs), single window == the
  // historical <=64-split behavior plus two extra __syncthreads.
  constexpr int SCW = 64;
  const int RBASEW = (ti.mrows <= 16) ? 16 : 128;
  const int LANESW = 1 << (31 - __clz(THREADS / RBASEW));
  const int rw = tid / LANESW, jw = tid % LANESW;
  const int SpadW = (P.splits_ub + 3) & ~3;
  const float* pmw = (rw < ti.mrows)
      ? P.part_ml + (((int64_t)tile * P.kvh + kvh) * MT + row0 + rw) * 2 * SpadW
      : nullptr;
#define CT_BUILD_SC(W0, WHI)                                          \
  {                                                                   \
    if (pmw) {                                                        \
      for (int s = (W0) + jw; s < (WHI); s += LANESW)                 \
        sc[(s - (W0)) * 128 + rw] = exp2f(pmw[s] - mstar[rw]);        \
    }                                                                 \
  }
  for (int c0 = cz0; c0 < min(nch, cz0 + per_z); c0 += MAXI) {
    const int cz = min(MAXI, min(nch, cz0 + per_z) - c0);
    float4 acc[MAXI];
    int rowi[MAXI];
#pragma unroll
    for (int i = 0; i < MAXI; ++i) {
      acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      rowi[i] = min(((c0 + i) * THREADS + tid) >> (dv_log - 2), ti.mrows - 1);
    }
    if (cz == 1) {
      const int r0i = rowi[0];
      const float4* base = po4 + c0 * THREADS + tid;
      const bool in4 = (c0 * THREADS + tid < nel4);
      for (int w0 = 0; w0 < active; w0 += SCW) {
        const int whi = min(active, w0 + SCW);
        CT_BUILD_SC(w0, whi)
        __syncthreads();
        if (in4) {
#pragma unroll 8
          for (int s = w0; s < whi; ++s) {
            const float4 v4 = base[s * stride4];
            const float scv = sc[(s - w0) * 128 + r0i];
            acc[0].x += v4.x * scv; acc[0].y += v4.y * scv;
            acc[0].z += v4.z * scv; acc[0].w += v4.w * scv;
          }
        }
        __syncthreads();
      }
    } else {
      for (int w0 = 0; w0 < active; w0 += SCW) {
        const int whi = min(active, w0 + SCW);
        CT_BUILD_SC(w0, whi)
        __syncthreads();
#pragma unroll 4
        for (int s = w0; s < whi; ++s) {
          const float4* base = po4 + s * stride4;
          const float* scs = sc + (s - w0) * 128;
#pragma unroll
          for (int i = 0; i < MAXI; ++i) {
            if (i < cz && (c0 + i) * THREADS + tid < nel4) {
              const float4 v4 = base[(c0 + i) * THREADS + tid];
              const float scv = scs[rowi[i]];
              acc[i].x += v4.x * scv; acc[i].y += v4.y * scv;
              acc[i].z += v4.z * scv; acc[i].w += v4.w * scv;
            }
          }
        }
        __syncthreads();
      }
    }
#pragma unroll
    for (int i = 0; i < MAXI; ++i) {
      if (i < cz) {
        const int e4 = (c0 + i) * THREADS + tid;
        if (e4 < nel4) {
          const int e = e4 << 2;
          const int r = e >> dv_log, c = e & dv_mask;
          const int rg = ti.m0 + r;
          const int q_idx = min(rg / P.group, ti.qlen - 1);
          const int head = kvh * P.group + (rg % P.group);
          const int64_t tok = cu + q_idx;
          __nv_bfloat16* orow =
              P_OUT + tok * P.out_tok_stride + (int64_t)head * P.out_head_stride + c;
          const float inv = rowinv[r];
          if (((reinterpret_cast<uintptr_t>(orow) & 7) == 0)) {
            uint2 pk;
            pk.x = pack_bf16(acc[i].x * inv, acc[i].y * inv);
            pk.y = pack_bf16(acc[i].z * inv, acc[i].w * inv);
            *reinterpret_cast<uint2*>(orow) = pk;
          } else {
            orow[0] = __float2bfloat16_rn(acc[i].x * inv);
            orow[1] = __float2bfloat16_rn(acc[i].y * inv);
            orow[2] = __float2bfloat16_rn(acc[i].z * inv);
            orow[3] = __float2bfloat16_rn(acc[i].w * inv);
          }
          if (c == 0 && P.has_lse) P.lse[tok * P.lse_tok_stride + head] = rowlse[r];
        }
      }
    }
  }
}

__device__ __forceinline__ int* counters_ptr(const FmhaParams& P) {
  return reinterpret_cast<int*>(P.counters);
}

// ---------------------------------------------------------------------------
// Combine kernel: reduces split partials (only used when splits_ub > 1).
// launch_bounds(256,2): <=128 regs -> 2 blocks/SM (255 regs pinned 1; NCU
// showed 12.5% occupancy on a pure-latency kernel).
// PDL variant: waits for the producer grid before touching partials; the
// launch attribute lets the block prologue + launch latency overlap the
// producer's tail (map_tile/split_range read inputs only, safe pre-wait).
// ---------------------------------------------------------------------------
template <bool PDL>
__global__ void __launch_bounds__(256, 2) fmha_combine(const FmhaParams P) {
  // DH2 (m_tile 256): combine_tile's row parallelism (RBASE/sc layout) is
  // 128 rows per block, so grid.y carries tiles x row-halves; each block
  // reduces one 128-row half of one (tile, kvh). m_tile <= 128 keeps the
  // historical mapping exactly (halves == 1, row0 == 0).
  const int halves = (P.m_tile > 128) ? (P.m_tile / 128) : 1;
  const int kvh = blockIdx.x;
  const int tile = (halves > 1) ? ((int)blockIdx.y / halves) : (int)blockIdx.y;
  const int halfrow = (halves > 1) ? ((int)blockIdx.y % halves) : 0;
  const int z = blockIdx.z;
  TileInfo ti = map_tile(P, tile);
  if (!ti.ok) return;
  if (halves > 1) {
    // Adjust BEFORE split_range: hi/lo stay the full-tile range so the
    // chunk boundaries and `active` match the DH2 main kernel exactly.
    ti.mrows -= halfrow * 128;
    if (ti.mrows <= 0) return;
    if (ti.mrows > 128) ti.mrows = 128;
    ti.m0 += halfrow * 128;
  }
  int kv0, kv1, active;
  split_range(P, ti, 0, kv0, kv1, active);
  if constexpr (PDL) cudaGridDependencySynchronize();
  // TAIL-split mode (_pad2 bit30, tc prefill): unsplit tiles (active == 1)
  // wrote their final out/lse directly in the main kernel - their part_o
  // slots are stale, so the combine must not touch them.
  if ((P._pad2 & (1 << 30)) && P.splits_ub > 1 && active == 1) return;
  __shared__ float sc[64 * 128 + 384];   // + rowinv/rowlse/mstar (3x128)
  combine_tile<256>(P, ti, tile, kvh, active, sc, z, gridDim.z, halfrow * 128);
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------
extern "C" int fmha_smem(int dq, int dv, int mt, int nt, int stages) {
  int q = mt * dq * 2;
  int kv = stages * nt * (dq * 2 + dv * 2);
  int tot = q + kv;
  if (mt == 16) {
    // per-slot mbarriers live directly past the last KV stage, and the KV
    // region may need up to 1023B of lead padding to reach the 1024B boundary
    // the SWIZZLE_128B TMA pattern requires (static smem in the FUSE
    // instantiations shifts the dynamic shared-memory window).
    tot += stages * 8 + 8 + 1024;
    // merge staging: [NWARP][16][DVH] floats == (nt/16)*16*dv*4 bytes
    int stg = q + (nt / 16) * 16 * dv * 4 + 512 * 4 + 256;
    if (stg > tot) tot = stg;
  }
  return (tot + 127) & ~127;
}

template <class K>
static int launch_kernel(K fn, dim3 grid, dim3 block, int smem, cudaStream_t st,
                         const FmhaParams& P) {
  fn<<<grid, block, smem, st>>>(P);
  return (int)cudaGetLastError();
}

#ifdef FMHA_TMA
// fmha_main always carries the two tensormap parameters (zeroed when the
// cp.async instantiation is selected), so it needs its own launch helper.
template <class K>
static int launch_kernel_tma(K fn, dim3 grid, dim3 block, int smem, cudaStream_t st,
                             const FmhaParams& P, const void* kmap, const void* vmap) {
  const CUtensorMap z{};
  const CUtensorMap* km = reinterpret_cast<const CUtensorMap*>(kmap);
  const CUtensorMap* vm = reinterpret_cast<const CUtensorMap*>(vmap);
  fn<<<grid, block, smem, st>>>(P, km ? *km : z, vm ? *vm : z);
  return (int)cudaGetLastError();
}
#else
template <class K>
static int launch_kernel_tma(K fn, dim3 grid, dim3 block, int smem, cudaStream_t st,
                             const FmhaParams& P, const void*, const void*) {
  fn<<<grid, block, smem, st>>>(P);
  return (int)cudaGetLastError();
}
#endif

template <class K>
static int launch_pdl(K fn, dim3 grid, dim3 block, int smem, cudaStream_t st,
                      const FmhaParams& P) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = (size_t)smem;
  cfg.stream = st;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  return (int)cudaLaunchKernelEx(&cfg, fn, P);
}

// d128 decode asks ptxas for 2 blocks/SM (MINB=2) when it runs 8 warps; with
// the 4-warp fallback build it keeps MINB=1 and 128 threads.
#ifdef FMHA_DEC_W8_ON
#define DEC128_MINB 2
#else
#define DEC128_MINB 1
#endif
// Decode (MT=16) configs, shared by the plain and the bulk-tensor lists: the
// dispatched instantiation MUST be the one fmha_init registered, otherwise the
// launch asks for >48KB of dynamic smem without the attribute and fails with
// cudaErrorInvalidValue (that mismatch is exactly what a MINB=1 TMA list against
// a MINB=2 dispatch list produced).
#define FMHA_DEC_LIST(X)             \
  X(64, 64, 16, 64, 4, 1)            \
  X(128, 128, 16, 64, 3, DEC128_MINB) \
  X(256, 256, 16, 64, 3, 1)          \
  X(256, 256, 16, 32, 3, 2)

#define FMHA_KERNEL_LIST(X)          \
  X(64, 64, 64, 64, 4, 1)            \
  X(128, 128, 64, 64, 3, 1)          \
  X(256, 256, 64, 32, 2, 1)          \
  FMHA_DEC_LIST(X)

template <int DQ, int DV, int MT, int NT, int STAGES, int MINB, bool TMA = false>
static int set_smem_one() {
  const int sm = fmha_smem(DQ, DV, MT, NT, STAGES);
  cudaError_t e = cudaFuncSetAttribute(
      (const void*)fmha_main<DQ, DV, MT, NT, STAGES, (MT == 16), false, MINB, TMA>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  if (e != cudaSuccess) return (int)e;
  if constexpr (MT == 16) {
    e = cudaFuncSetAttribute(
        (const void*)fmha_main<DQ, DV, MT, NT, STAGES, true, true, MINB, TMA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    if (e != cudaSuccess) return (int)e;
  }
  return 0;
}

#ifdef FMHA_TMA
// Decode (MT=16) kernels with bulk-tensor KV loading: distinct functions, so
// their dynamic-smem attribute must be set per instantiation too - with the
// same MINB the dispatcher selects (see FMHA_DEC_LIST).
#define FMHA_TMA_LIST(X) FMHA_DEC_LIST(X)
#endif

extern "C" int fmha_init() {
#define SET_SMEM(DQ, DV, MT, NT, STAGES, MINB)                                        \
  {                                                                                   \
    int r = set_smem_one<DQ, DV, MT, NT, STAGES, MINB>();                             \
    if (r) return r;                                                                  \
  }
  FMHA_KERNEL_LIST(SET_SMEM)
#undef SET_SMEM
#ifdef FMHA_TMA
#define SET_SMEM_T(DQ, DV, MT, NT, STAGES, MINB)                                      \
  {                                                                                   \
    int r = set_smem_one<DQ, DV, MT, NT, STAGES, MINB, true>();                       \
    if (r) return r;                                                                  \
  }
  FMHA_TMA_LIST(SET_SMEM_T)
#undef SET_SMEM_T
#endif
  return 0;
}

extern "C" int fmha_combine_only(const FmhaParams* p, void* stream, void*, void*);

template <int DQ, int DV, int MT, int NT, int STAGES, int MINB>
static int dispatch_one(dim3 grid, int fuse, cudaStream_t st, const FmhaParams& P,
                        const void* kmap, const void* vmap) {
#ifdef FMHA_DEC_D256W4
  const dim3 thr((MT == 16 && DV == 256 && NT >= 64) ? 256 : 128);
#elif defined(FMHA_DEC_W8_ON)
  const dim3 thr((MT == 16 && DV >= 128) ? 256 : 128);
#else
  const dim3 thr((MT == 16 && DV == 256) ? 256 : 128);
#endif
  const int sm = fmha_smem(DQ, DV, MT, NT, STAGES);
  if constexpr (MT == 16) {
#ifdef FMHA_TMA
    if (kmap && vmap) {
      if (fuse)
        return launch_kernel_tma(fmha_main<DQ, DV, MT, NT, STAGES, true, true, MINB, true>,
                                 grid, thr, sm, st, P, kmap, vmap);
      return launch_kernel_tma(fmha_main<DQ, DV, MT, NT, STAGES, true, false, MINB, true>,
                               grid, thr, sm, st, P, kmap, vmap);
    }
#endif
    if (fuse)
      return launch_kernel_tma(fmha_main<DQ, DV, MT, NT, STAGES, true, true, MINB>,
                               grid, thr, sm, st, P, nullptr, nullptr);
    return launch_kernel_tma(fmha_main<DQ, DV, MT, NT, STAGES, true, false, MINB>,
                             grid, thr, sm, st, P, nullptr, nullptr);
  } else {
    return launch_kernel_tma(fmha_main<DQ, DV, MT, NT, STAGES, false, false, MINB>,
                             grid, thr, sm, st, P, nullptr, nullptr);
  }
}

extern "C" int fmha_run(const FmhaParams* p, void* stream, void* kmap, void* vmap) {
  cudaStream_t st = (cudaStream_t)stream;
  const FmhaParams& P = *p;
  if (P.splits_ub > 128 || P.splits_ub < 1) return -100;  // 128: wavefill decode splits (windowed combine)
  dim3 grid(P.kvh, P.tiles_ub, P.splits_ub);
  int rc = -105;
#define DISPATCH(DQ, DV, MT, NT, STAGES, MINB)                                        \
  if (P.dq == DQ && P.dv == DV && P.m_tile == MT && P.n_tile == NT) {                 \
    rc = dispatch_one<DQ, DV, MT, NT, STAGES, MINB>(grid, (P._pad2 & 4), st, P,       \
                                                    kmap, vmap);                      \
  } else
  FMHA_KERNEL_LIST(DISPATCH)
  {
    return -104;
  }
#undef DISPATCH
  if (rc) return rc;
  if (P.splits_ub > 1 && !((P._pad2 & 4) && P.m_tile == 16))
    return fmha_combine_only(p, stream, nullptr, nullptr);
  return 0;
}

extern "C" int fmha_combine_only(const FmhaParams* p, void* stream, void*, void*) {
  const FmhaParams& P = *p;
  if (P.splits_ub <= 1) return 0;
  const int z = P.combine_z > 0 ? P.combine_z : 1;
  // grid.y = tiles x 128-row halves (halves == 1 for every m_tile <= 128
  // plan, so all existing dispatches keep the exact historical grid).
  const unsigned ty = (unsigned)P.tiles_ub
                    * (unsigned)((P.m_tile > 128) ? (P.m_tile / 128) : 1);
  if (P._pad2 & 8)
    return launch_pdl(fmha_combine<true>, dim3(P.kvh, ty, z), dim3(256),
                      0, (cudaStream_t)stream, P);
  return launch_kernel(fmha_combine<false>, dim3(P.kvh, ty, z), dim3(256),
                       0, (cudaStream_t)stream, P);
}

extern "C" int fmha_params_bytes() { return (int)sizeof(FmhaParams); }

#ifdef FMHA_TMA
// Host-side tensormap encoding for the decode path (driver API via dlopen, no
// link-flag change). 4D HND paged view {d, page_size, kvh, pages} with a
// SWIZZLE_128B box of {64 bf16, min(nt, page_size) rows}: per-segment boxes
// start on R-aligned rows (R divides page_size), so they never straddle a page
// and write exactly the atom-major layout the ldmatrix reads expect. Maps are
// encoded once per dispatch key into caller-provided 128B-aligned slots and
// passed by value as __grid_constant__ kernel parameters.
extern "C" int fmha_encode_kv_maps(void* kmap_out, void* vmap_out,
                                   void* kptr, void* vptr,
                                   int dq, int dv, int page_size, int kvh,
                                   int nt,
                                   long long num_pages,
                                   long long slot_sk, long long slot_sv,
                                   long long head_sk, long long head_sv,
                                   long long page_sk, long long page_sv) {
  typedef CUresult (*enc_t)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                            const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                            const cuuint32_t*, CUtensorMapInterleave,
                            CUtensorMapSwizzle, CUtensorMapL2promotion,
                            CUtensorMapFloatOOBfill);
  static enc_t enc = nullptr;
  if (!enc) {
    void* h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!h) h = dlopen("libcuda.so", RTLD_LAZY);
    if (!h) return -1;
    enc = (enc_t)(uintptr_t)dlsym(h, "cuTensorMapEncodeTiled");
    if (!enc) return -2;
  }
  const cuuint64_t dimsk[4] = {(cuuint64_t)dq, (cuuint64_t)page_size,
                               (cuuint64_t)kvh, (cuuint64_t)num_pages};
  const cuuint64_t strk[3] = {(cuuint64_t)(slot_sk * 2), (cuuint64_t)(head_sk * 2),
                              (cuuint64_t)(page_sk * 2)};
  const cuuint64_t dimsv[4] = {(cuuint64_t)dv, dimsk[1], dimsk[2], dimsk[3]};
  const cuuint64_t strv[3] = {(cuuint64_t)(slot_sv * 2), (cuuint64_t)(head_sv * 2),
                              (cuuint64_t)(page_sv * 2)};
  const int boxr = nt < page_size ? nt : page_size;
  const cuuint32_t box[4] = {64, (cuuint32_t)boxr, 1, 1};
  const cuuint32_t es[4] = {1, 1, 1, 1};
  CUresult r = enc((CUtensorMap*)kmap_out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                   kptr, dimsk, strk, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) return 1000 + (int)r;
  r = enc((CUtensorMap*)vmap_out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          vptr, dimsv, strv, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? 0 : 2000 + (int)r;
}
#endif

// ---------------------------------------------------------------------------
// Graph-replay helpers (host-side launch machinery only; no numerics).
// Capture happens on a private side stream (legacy default stream cannot be
// captured); replays run on the caller's current stream. If any step fails,
// the Python side permanently falls back to direct launches for that key.
// ---------------------------------------------------------------------------
extern "C" int fmha_is_capturing(void* stream, int* status) {
  cudaStreamCaptureStatus s = cudaStreamCaptureStatusNone;
  int rc = (int)cudaStreamIsCapturing((cudaStream_t)stream, &s);
  *status = (int)s;
  return rc;
}

extern "C" int fmha_side_stream(void** stream) {
  static cudaStream_t s = nullptr;
  if (s == nullptr) {
    int rc = (int)cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (rc != cudaSuccess) return rc;
  }
  *stream = (void*)s;
  return 0;
}

extern "C" int fmha_capture_begin(void* stream) {
  return (int)cudaStreamBeginCapture((cudaStream_t)stream,
                                     cudaStreamCaptureModeThreadLocal);
}

extern "C" int fmha_capture_end(void* stream, void** exec) {
  cudaGraph_t g = nullptr;
  int rc = (int)cudaStreamEndCapture((cudaStream_t)stream, &g);
  if (rc != cudaSuccess || g == nullptr) return rc ? rc : -1;
  cudaGraphExec_t e = nullptr;
  rc = (int)cudaGraphInstantiate(&e, g, nullptr, nullptr, 0);
  cudaGraphDestroy(g);
  if (rc != cudaSuccess || e == nullptr) return rc ? rc : -1;
  *exec = (void*)e;
  return 0;
}

extern "C" int fmha_graph_launch(void* exec, void* stream) {
  return (int)cudaGraphLaunch((cudaGraphExec_t)exec, (cudaStream_t)stream);
}

extern "C" void fmha_graph_free(void* exec) {
  if (exec) cudaGraphExecDestroy((cudaGraphExec_t)exec);
}
