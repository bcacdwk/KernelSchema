// 2-CTA cluster tcgen05 BF16 FMHA prefill kernel for L20D SM103 (M=256).
// Independent implementation for the frozen fmha_bf16 contract. cta_group::2
// UMMA: each CTA holds its Q/M half, half of each K tile (N-split) and half of
// each V tile (DV-split); rank 0 issues MMAs; completion multicasts to both
// CTAs' mbarriers. Softmax/P/O stay fully per-CTA local. cp.async producers
// with paged-KV gather. Requires -gencode arch=compute_103a,code=sm_103a.
// Cluster/commit/alloc protocol cross-checked against published SM100 sources
// (Atrex-FA pure_cuda 2cta); all code here independently written.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math.h>

#include "fmha_params.h"

#define NEG_INF (-__int_as_float(0x7f800000))

#define P_Q    (reinterpret_cast<const __nv_bfloat16*>(P.q))
#define P_K    (reinterpret_cast<const __nv_bfloat16*>(P.k))
#define P_V    (reinterpret_cast<const __nv_bfloat16*>(P.v))
#define P_OUT  (reinterpret_cast<__nv_bfloat16*>(P.out))

__device__ __forceinline__ uint32_t su32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__host__ __device__ constexpr uint32_t tc2_idesc(int M, int N, bool b_major_mn) {
  return (uint32_t)((M >> 5) << 25) | (uint32_t)((N >> 3) << 17)
       | (uint32_t(b_major_mn) << 16) | (1u << 10) | (1u << 7) | (1u << 4);
}
__device__ __forceinline__ uint64_t tc_desc(uint32_t smem_addr, uint32_t sbo_bytes,
                                            uint32_t lbo_bytes) {
  return (uint64_t(smem_addr >> 4) & 0x3FFF)
       | (uint64_t(1) << 46)
       | (uint64_t(sbo_bytes >> 4) << 32)
       | (uint64_t(lbo_bytes >> 4) << 16)
       | (uint64_t(2) << 61);
}
__device__ __forceinline__ void mma_ss2(uint32_t tmem_c, uint64_t da, uint64_t db,
                                        uint32_t idesc, bool accum) {
  asm volatile("{\n\t.reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}"
               :: "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"(int(accum)));
}
__device__ __forceinline__ void mma_ts2(uint32_t tmem_c, uint32_t tmem_a, uint64_t db,
                                        uint32_t idesc, bool accum) {
  asm volatile("{\n\t.reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n\t}"
               :: "r"(tmem_c), "r"(tmem_a), "l"(db), "r"(idesc), "r"(int(accum)));
}
__device__ __forceinline__ void umma_commit2(uint32_t mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one."
               "shared::cluster.multicast::cluster.b64 [%0], %1;"
               :: "r"(mbar_addr), "h"(uint16_t(0x3)) : "memory");
}
__device__ __forceinline__ void mbar_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t phase) {
  asm volatile("{\n\t.reg .pred c;\n\tW%=:\n\t"
               "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 c, [%0], %1;\n\t"
               "@!c bra.uni W%=;\n\t}" :: "r"(addr), "r"(phase) : "memory");
}
__device__ __forceinline__ void mbar_arrive_local(uint32_t addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void mbar_arrive_rank0(uint32_t local_addr) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(remote) : "r"(local_addr), "r"(0));
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void tmem_alloc2(uint32_t dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
               :: "r"(dst_smem), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc2(uint32_t base, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
               :: "r"(base), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish2() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}
__device__ __forceinline__ void tmem_ld32(float* d, uint32_t addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]),
                 "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]),
                 "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]),
                 "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
                 "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]),
                 "=f"(d[30]), "=f"(d[31])
               : "r"(addr));
}
__device__ __forceinline__ void tmem_st32(uint32_t addr, const float* v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
               "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
               :: "r"(addr), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]),
                  "f"(v[5]), "f"(v[6]), "f"(v[7]), "f"(v[8]), "f"(v[9]), "f"(v[10]),
                  "f"(v[11]), "f"(v[12]), "f"(v[13]), "f"(v[14]), "f"(v[15]), "f"(v[16]),
                  "f"(v[17]), "f"(v[18]), "f"(v[19]), "f"(v[20]), "f"(v[21]), "f"(v[22]),
                  "f"(v[23]), "f"(v[24]), "f"(v[25]), "f"(v[26]), "f"(v[27]), "f"(v[28]),
                  "f"(v[29]), "f"(v[30]), "f"(v[31]));
}
__device__ __forceinline__ void tmem_st16(uint32_t addr, const uint32_t* v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
               "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
               :: "r"(addr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]),
                  "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]),
                  "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;"); }
__device__ __forceinline__ void fence_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;"); }
__device__ __forceinline__ void fence_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_cluster() { asm volatile("fence.proxy.async.shared::cluster;" ::: "memory"); }
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
__device__ __forceinline__ float sink_adjust(float base2, float sink2, float* scale) {
  float d = sink2 - base2;
  float mx = fmaxf(base2, sink2);
  float total2 = mx + log2f(1.0f + exp2f(-fabsf(d)));
  *scale = exp2f(base2 - total2);
  return total2;
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

// ---------------------------------------------------------------------------
// Cooperative loaders (128 threads/CTA), SW128 atom-blocked layouts.
// ---------------------------------------------------------------------------
template <int ROWS, int D>
__device__ __forceinline__ void tc2_load_q(const FmhaParams& P, const TileInfo& ti,
                                           char* dst, int kvh, int row_base, int tid) {
  constexpr int QU = D / 8;
  const int cu = P.cu_q[ti.b];
#pragma unroll
  for (int i = 0; i < (ROWS * QU) / 128; ++i) {
    const int u = tid + i * 128;
    const int row = u / QU, unit = u % QU;
    const int rg = ti.m0 + row_base + row;
    const int q_idx = min(rg / P.group, ti.qlen - 1);
    const int head = kvh * P.group + (rg % P.group);
    const __nv_bfloat16* src = P_Q + (int64_t)(cu + q_idx) * P.q_tok_stride
                             + (int64_t)head * P.q_head_stride + unit * 8;
    const int atom = unit >> 3, uin = unit & 7;
    cp_async16(su32(dst + atom * (ROWS * 128) + row * 128 + (((uin ^ row) & 7) * 16)), src);
  }
}

template <int ROWS, int D>
__device__ __forceinline__ void tc2_load_k(const FmhaParams& P, const TileInfo& ti,
                                           char* dst, int kvh, int kv_base, int tid) {
  constexpr int U = D / 8;
  const int last = ti.hi - 1;
  const int32_t* tab = P.table + ti.b * P.tstride;
#pragma unroll
  for (int i = 0; i < (ROWS * U) / 128; ++i) {
    const int u = tid + i * 128;
    const int row = u / U, unit = u % U;
    const int p = min(kv_base + row, last);
    const int page = __ldg(tab + (p >> P.page_log));
    const __nv_bfloat16* src = P_K + (int64_t)page * P.k_page_stride
                             + (int64_t)kvh * P.k_head_stride
                             + (int64_t)(p & P.page_mask) * P.k_slot_stride + unit * 8;
    const int atom = unit >> 3, uin = unit & 7;
    cp_async16(su32(dst + atom * (ROWS * 128) + row * 128 + (((uin ^ row) & 7) * 16)), src);
  }
}

// V half-tile: all NT kv rows, dv columns [col_base, col_base + D/2).
template <int NT, int D>
__device__ __forceinline__ void tc2_load_v(const FmhaParams& P, const TileInfo& ti,
                                           char* dst, int kvh, int kv_base,
                                           int col_base, int tid) {
  constexpr int U = D / 16;  // 16B units per row within the half
  const int last = ti.hi - 1;
  const int32_t* tab = P.table + ti.b * P.tstride;
#pragma unroll
  for (int i = 0; i < (NT * U) / 128; ++i) {
    const int u = tid + i * 128;
    const int row = u / U, unit = u % U;
    const int p = min(kv_base + row, last);
    const int page = __ldg(tab + (p >> P.page_log));
    const __nv_bfloat16* src = P_V + (int64_t)page * P.v_page_stride
                             + (int64_t)kvh * P.v_head_stride
                             + (int64_t)(p & P.page_mask) * P.v_slot_stride
                             + col_base + unit * 8;
    const int atom = unit >> 3, uin = unit & 7;
    cp_async16(su32(dst + atom * (NT * 128) + row * 128 + (((uin ^ row) & 7) * 16)), src);
  }
}

// ---------------------------------------------------------------------------
// Cluster kernel. D=128, NT=128 kv per stage (64 per CTA), SB=3, M=256.
// TMEM per CTA: S0[0,128) S1[128,256) P[256,320) O[320,448). alloc 512.
// ---------------------------------------------------------------------------
template <int D, int NT, int SB>
__global__ void __launch_bounds__(128) __cluster_dims__(2, 1, 1)
fmha_tc2_main(const FmhaParams P) {
  constexpr int DV = D;
  constexpr int NT2 = NT / 2;
  constexpr int Q_BYTES = 128 * D * 2;
  constexpr int ST_BYTES = NT2 * D * 2 + NT * (D / 2) * 2;   // K half + V half
  constexpr int S_COLS = NT;
  extern __shared__ __align__(1024) char smem[];
  char* smem_q = smem;
  char* smem_k[SB];
  char* smem_v[SB];
#pragma unroll
  for (int j = 0; j < SB; ++j) {
    smem_k[j] = smem + Q_BYTES + j * ST_BYTES;
    smem_v[j] = smem_k[j] + NT2 * D * 2;
  }
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + Q_BYTES + SB * ST_BYTES);
  // mbar[0]=qk_done [1]=pv_done [2..2+SB)=kv_ready [2+SB..4+SB)=p_ready [4+SB]=dealloc
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(mbar + SB + 4);
  const uint32_t mb_qk = su32(mbar), mb_pv = su32(mbar + 1);
  const uint32_t mb_kv0 = su32(mbar + 2), mb_p0 = su32(mbar + 2 + SB);

  const int tid = threadIdx.x;
  const uint32_t rank = cluster_rank();
  // Tile reversal: descending-work order helps causal only (see fmha_tc.cu;
  // non-causal bimodal mixes regressed +30% in eval dv_e12772509b19).
  const int tile = ((P._pad2 & 2) && P.causal)
                       ? (P.tiles_ub - 1 - (int)(blockIdx.x >> 1))
                       : (int)(blockIdx.x >> 1);
  const int kvh = blockIdx.y, split = blockIdx.z;
  TileInfo ti = map_tile(P, tile);
  int kv0 = 0, kv1 = 0, active = 1;
  bool dead = !ti.ok || !split_range(P, ti, split, kv0, kv1, active);
  // Cluster kernels must not diverge on cluster_sync/alloc: dead clusters exit
  // collectively after the mandatory cluster ops are avoided entirely.
  if (dead) return;
  const int nstages = (kv1 - kv0 + NT - 1) / NT;

  if (tid == 0) {
    mbar_init(mb_qk, 1);
    mbar_init(mb_pv, 1);
#pragma unroll
    for (int i = 0; i < SB; ++i) mbar_init(su32(mbar + 2 + i), 2);
#pragma unroll
    for (int i = 0; i < 2; ++i) mbar_init(su32(mbar + 2 + SB + i), 2);
  }
  mbar_init_fence();
  cluster_sync();
  if (tid < 32) tmem_alloc2(su32(tmem_slot), 512);
  fence_before_sync();
  __syncthreads();
  fence_after_sync();
  const uint32_t tmem = *tmem_slot;
  if (tid < 32) tmem_relinquish2();
  cluster_sync();
  const uint32_t tmem_s = tmem;                 // S0 at +0, S1 at +S_COLS
  const uint32_t tmem_p = tmem + 2 * S_COLS;    // 64 words
  const uint32_t tmem_o = tmem + 2 * S_COLS + NT / 2;

  const int row_base = (int)rank * 128;
  const int rg = ti.m0 + row_base + tid;
  const bool valid = (row_base + tid) < ti.mrows;
  const int q_idx = min(rg / P.group, ti.qlen - 1);
  const int head = kvh * P.group + (rg % P.group);
  const int vis = ti.kvlen - ti.qlen + rg / P.group;
  const int rhi = min(P.causal ? vis : ti.kvlen - 1, kv1 - 1);
  const int rlo = (P.window_left >= 0) ? max(0, vis - P.window_left) : 0;
  float mbase = NEG_INF, lt = 0.f;

  tc2_load_q<128, D>(P, ti, smem_q, kvh, row_base, tid);
  tc2_load_k<NT2, D>(P, ti, smem_k[0], kvh, kv0 + rank * NT2, tid);
  tc2_load_v<NT, D>(P, ti, smem_v[0], kvh, kv0, rank * (D / 2), tid);
  cp_async_commit();
#pragma unroll
  for (int j = 1; j < SB - 1; ++j) {
    tc2_load_k<NT2, D>(P, ti, smem_k[j], kvh, kv0 + j * NT + rank * NT2, tid);
    tc2_load_v<NT, D>(P, ti, smem_v[j], kvh, kv0 + j * NT, rank * (D / 2), tid);
    cp_async_commit();
  }

  constexpr uint32_t IDESC_QK = tc2_idesc(256, NT, false);
  constexpr uint32_t IDESC_PV = tc2_idesc(256, DV, true);
  constexpr uint64_t A_ATOM = (128 * 128) >> 4;   // Q: 128 rows per k-atom
  constexpr uint64_t B_ATOM = (uint64_t)NT2 * 8;  // K half: NT2 rows per k-atom
  const uint64_t qd0 = tc_desc(su32(smem_q), 1024, 0);
  constexpr uint32_t V_LBO = (DV >= 256) ? NT * 128 : 0;  // CTA V half: DV/2 cols

  cp_async_wait<(SB >= 3 ? 1 : 0)>();
  fence_proxy_cluster();
  __syncthreads();
  if (tid == 0) mbar_arrive_rank0(mb_kv0);  // stage 0 ready (kv_ready[0])
  if (rank == 0 && tid == 0) {
    mbar_wait(mb_kv0, 0);
    fence_after_sync();
    uint64_t da = qd0, db = tc_desc(su32(smem_k[0]), 1024, 0);
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      mma_ss2(tmem_s, da, db, IDESC_QK, ks != 0);
      da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
      db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
    }
    umma_commit2(mb_qk);
  }

  for (int s = 0; s < nstages; ++s) {
    // ---- A: wait S(s) & V-buffer release; read S; stats; queue stage s+SB-1
    mbar_wait(mb_qk, s & 1);
    fence_after_sync();
    float sreg[NT];
#pragma unroll
    for (int c = 0; c < NT / 32; ++c)
      tmem_ld32(sreg + c * 32, tmem_s + (s & 1) * S_COLS + c * 32);
    tmem_wait_ld();
    const int pbase = kv0 + s * NT;
    const bool interior = (pbase >= rlo) && (pbase + NT - 1 <= rhi);
    float rmax_raw = NEG_INF;
    // Branched mask, not if-converted predicates (same fix and same reasoning as
    // fmha_tc.cu: the predicated-off compare/select chain was 39% of executed
    // instructions on c4). This 2-CTA path is off by default (FMHA_TC2=1) but
    // must not carry a known 1.6x instruction overhead if it is ever enabled.
#ifdef FMHA_MASK_PRED
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      float x = sreg[i];
      if (!interior) {
        const int c = pbase + i;
        if (c < rlo || c > rhi) x = NEG_INF;
      }
      sreg[i] = x;
      rmax_raw = fmaxf(rmax_raw, x);
    }
#else
    if (interior) {
#pragma unroll
      for (int i = 0; i < NT; ++i) rmax_raw = fmaxf(rmax_raw, sreg[i]);
    } else {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int c = pbase + i;
        float x = sreg[i];
        if (c < rlo || c > rhi) x = NEG_INF;
        sreg[i] = x;
        rmax_raw = fmaxf(rmax_raw, x);
      }
    }
#endif
    const float mn = fmaxf(mbase, rmax_raw * P.qk_scale2);
    const float msafe = (mn == NEG_INF) ? 0.f : mn;
    const bool skip = (msafe - mbase) < P.lazy_thresh;
    const float alpha = skip ? 1.f : exp2f(mbase - msafe);
    const float nqk = P.qk_scale2;
    const float npb = -(skip ? mbase : msafe);
    if (!skip) { lt *= alpha; mbase = msafe; }
    if (s > 0) mbar_wait(mb_pv, (s - 1) & 1);  // V buf reuse + O-fix ordering
    if (s + SB - 1 < nstages) {
      const int nb = (s + SB - 1) % SB;
      tc2_load_k<NT2, D>(P, ti, smem_k[nb], kvh,
                         kv0 + (s + SB - 1) * NT + rank * NT2, tid);
      tc2_load_v<NT, D>(P, ti, smem_v[nb], kvh, kv0 + (s + SB - 1) * NT,
                        rank * (D / 2), tid);
    }
    cp_async_commit();
    cp_async_wait<1>();
    fence_proxy_cluster();
    __syncthreads();
    if (tid == 0) mbar_arrive_rank0(mb_kv0 + (uint32_t)(((s + 1) % SB) * 8));
    if (rank == 0 && tid == 0 && s + 1 < nstages) {
      mbar_wait(mb_kv0 + (uint32_t)(((s + 1) % SB) * 8), ((s + 1) / SB) & 1);
      fence_after_sync();
      const int kb = (s + 1) % SB;
      uint64_t da = qd0, dbq = tc_desc(su32(smem_k[kb]), 1024, 0);
      const uint32_t ts = tmem_s + ((s + 1) & 1) * S_COLS;
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks) {
        mma_ss2(ts, da, dbq, IDESC_QK, ks != 0);
        da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
        dbq += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
      }
      umma_commit2(mb_qk);
    }
    // ---- B: exp2 in place, row sum ----
    float lsum = 0.f;
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const float p = exp2f(fmaf(sreg[i], nqk, npb));
      lsum += p;
      sreg[i] = p;
    }
    lt += lsum;
    // ---- C: O rescale; P store; signal p_ready ----
    if (s > 0) {
      if (__any_sync(0xffffffffu, alpha < 1.0f)) {
#pragma unroll
        for (int c = 0; c < DV / 32; ++c) {
          float oreg[32];
          tmem_ld32(oreg, tmem_o + c * 32);
          tmem_wait_ld();
#pragma unroll
          for (int i = 0; i < 32; ++i) oreg[i] *= alpha;
          tmem_st32(tmem_o + c * 32, oreg);
        }
        tmem_wait_st();
      }
    }
#pragma unroll
    for (int c = 0; c < NT / 32; ++c) {
      uint32_t pw[16];
#pragma unroll
      for (int j = 0; j < 16; ++j)
        pw[j] = pack_bf16(sreg[c * 32 + 2 * j], sreg[c * 32 + 2 * j + 1]);
      tmem_st16(tmem_p + c * 16, pw);
    }
    tmem_wait_st();
    fence_before_sync();
    __syncthreads();
    if (tid == 0) mbar_arrive_rank0(mb_p0 + (uint32_t)((s & 1) * 8));
    // ---- D (rank 0): PV(s) then done ----
    if (rank == 0 && tid == 0) {
      mbar_wait(mb_p0 + (uint32_t)((s & 1) * 8), (s / 2) & 1);
      fence_after_sync();
      uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
#pragma unroll
      for (int st = 0; st < NT / 16; ++st) {
        mma_ts2(tmem_o, tmem_p + st * 8, db, IDESC_PV, (st != 0) || (s > 0));
        db += 128;
      }
      umma_commit2(mb_pv);
    }
  }

  // ---- epilogue (per CTA, own 128 rows) ----
  mbar_wait(mb_pv, (nstages - 1) & 1);
  fence_after_sync();
  const int64_t Pb = ((int64_t)tile * P.kvh + kvh) * P.splits_ub + split;
  const int cu = P.cu_q[ti.b];
  const int64_t tok = cu + q_idx;
  if (P.splits_ub > 1) {
    float* po = P.part_o + (Pb * P.m_tile + row_base + tid) * DV;
    const int64_t blk = (int64_t)tile * P.kvh + kvh;
    const int Spad = (P.splits_ub + 3) & ~3;
    float* pm = P.part_ml + ((blk * P.m_tile + row_base + tid) * 2) * Spad + split;
#pragma unroll
    for (int c = 0; c < DV / 32; ++c) {
      float oreg[32];
      tmem_ld32(oreg, tmem_o + c * 32);
      tmem_wait_ld();
#pragma unroll
      for (int i = 0; i < 32; ++i) po[c * 32 + i] = oreg[i];
    }
    if (valid) { pm[0] = mbase; pm[Spad] = lt; }
  } else {
    float lse2 = mbase + log2f(lt);
    float inv = P.v_scale / lt;
    if (P.has_sinks) {
      float os;
      lse2 = sink_adjust(lse2, P.sinks[head * P.sink_stride] * P.log2e, &os);
      inv *= os;
    }
    __nv_bfloat16* orow = P_OUT + tok * P.out_tok_stride + (int64_t)head * P.out_head_stride;
#pragma unroll
    for (int c = 0; c < DV / 32; ++c) {
      float oreg[32];
      tmem_ld32(oreg, tmem_o + c * 32);
      tmem_wait_ld();
      uint32_t packed[16];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        packed[i] = pack_bf16(oreg[2 * i] * inv, oreg[2 * i + 1] * inv);
      if (valid) {
#pragma unroll
        for (int u = 0; u < 4; ++u)
          *reinterpret_cast<uint4*>(orow + c * 32 + u * 8) =
              *reinterpret_cast<uint4*>(packed + u * 4);
      }
    }
    if (valid && P.has_lse) P.lse[tok * P.lse_tok_stride + head] = lse2;
  }
  // coordinated 2-CTA TMEM release: cluster barrier orders both epilogues.
  fence_before_sync();
  __syncthreads();
  cluster_sync();
  if (tid < 32) tmem_dealloc2(tmem, 512);
}

extern "C" int fmha_tc2_smem(int d, int nt, int sb) {
  return 128 * d * 2 + sb * ((nt / 2) * d * 2 + nt * (d / 2) * 2) + 256;
}

extern "C" int fmha_tc2_init() {
  cudaError_t e = cudaFuncSetAttribute((const void*)fmha_tc2_main<128, 128, 3>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       fmha_tc2_smem(128, 128, 3));
  if (e != cudaSuccess) return (int)e;
  return 0;
}

extern "C" int fmha_tc2_run(const FmhaParams* p, void* stream, void*, void*) {
  cudaStream_t st = (cudaStream_t)stream;
  const FmhaParams& P = *p;
  if (P.dq != 128 || P.dv != 128 || P.m_tile != 256 || P.n_tile != 128)
    return -200;
  if (P.splits_ub > 64 || P.splits_ub < 1) return -100;
  dim3 grid(2 * P.tiles_ub, P.kvh, P.splits_ub);
  fmha_tc2_main<128, 128, 3><<<grid, dim3(128), fmha_tc2_smem(128, 128, 3), st>>>(P);
  return (int)cudaGetLastError();
}

extern "C" int fmha_tc2_params_bytes() { return (int)sizeof(FmhaParams); }
