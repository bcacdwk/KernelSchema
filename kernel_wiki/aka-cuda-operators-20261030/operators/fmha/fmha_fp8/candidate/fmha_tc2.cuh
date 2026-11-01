// fmha_tc2.cuh — 2-CTA cluster (cta_group::2) FP8 paged FMHA for SM103a.
// Independent implementation for problem 14; mechanism patterns validated by
// experiments/tc2/probe.cu ON THIS DEVICE (exact match vs fp32 reference,
// direct + graph-capture launch; dv probe PASS) and cross-checked against
// the public PTX ISA and the mounted atrex-fa pure-CUDA SM100 2-CTA
// reference (read for mechanism only; no code copied, no library called).
//
// Cluster = 2 CTAs = 2 adjacent 128-row q-tiles (x0, x0+1) of one unit.
//   QK: M=256 (128 rows/CTA in own smem A), N=128 kv, B halves: CTA r holds
//       kv rows [64r, 64r+64) of the tile (K smem 8KB/stage/CTA).
//   PV: M=256, A = P from each CTA's own tmem, B = V^T with N=dv split by
//       CTA; each CTA stores FULL 128B V rows (validated SW128 descriptor;
//       hardware reads its dv half) -> V traffic per cluster equals ONE
//       1-CTA tile instead of two.
//   rank0's mma warp issues all UMMAs; commits multicast (mask 3) to both
//   CTAs' local qk_done/output_ready/k_released/v_released barriers.
//   rank1's w8 lane0 is the K/V/Q pair-forwarder (cp.async.mbarrier.arrive
//   is CTA-local): waits local loaded barriers, remote-arrives rank0's
//   *_pair barriers (count includes rank0's own 96/4 arrivals as applicable).
//   Softmax warps of BOTH CTAs arrive rank0's score_pair/prob_pair/
//   rescaled_pair (count 8); state handoff stays CTA-local.
//   Teardown: cluster_sync then collective cta_group::2 dealloc.
// Softmax math is the v5 anchored-max pipeline (mirrors fmha_tc.cuh: f16
// y/exp/pack, rebase-margin correction, dsum carries PSCALE=256); per-CTA
// epilogue/ws partial layout unchanged, so fmha_combine works as-is.

#pragma once

#include "fmha_tc.cuh"

namespace fmtc2 {

using fmtc::TN;
using fmtc::TM;
using fmtc::NTHREADS;
using fmtc::O_TMEM;
using fmtc::TMEM_COLS;
using fmtc::setmaxnreg_inc;
using fmtc::setmaxnreg_dec;

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(void* m, uint32_t rank) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(ra) : "r"(fmtc::tc_smem_u32(m)), "r"(rank));
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(ra) : "memory");
}
// rank0-local arrive or remote arrive on rank0's copy, by cluster rank.
__device__ __forceinline__ void arrive_rank0(void* m, uint32_t rank) {
  if (rank == 0) fmtc::mbar_arrive(m);
  else mbar_arrive_remote(m, 0);
}
__device__ __forceinline__ void tmem_alloc2(uint32_t* smem_dst, uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
               ::"r"(fmtc::tc_smem_u32(smem_dst)), "r"(cols));
}
__device__ __forceinline__ void tmem_relinq2() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc2(uint32_t base, uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(base), "r"(cols)
               : "memory");
}
__device__ __forceinline__ void umma_f8_ss_2cta(uint32_t d_tmem, uint64_t a, uint64_t b,
                                                uint32_t idesc, bool accum) {
  const uint32_t zero = 0;
  const uint32_t sc = accum ? 1u : 0u;
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, {%5,%5,%5,%5,%5,%5,%5,%5}, p;\n}"
      ::"r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(sc), "r"(zero));
}
__device__ __forceinline__ void umma_f8_ts_2cta(uint32_t d_tmem, uint32_t a_tmem, uint64_t b,
                                                uint32_t idesc, bool accum) {
  const uint32_t zero = 0;
  const uint32_t sc = accum ? 1u : 0u;
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, {%5,%5,%5,%5,%5,%5,%5,%5}, p;\n}"
      ::"r"(d_tmem), "r"(a_tmem), "l"(b), "r"(idesc), "r"(sc), "r"(zero));
}
__device__ __forceinline__ void umma_commit2_mc(void* m, uint16_t mask) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
      "[%0], %1;" ::"r"(fmtc::tc_smem_u32(m)), "h"(mask));
}

template <int TD, int NSTAGE>
struct Shared2 {
  static constexpr int PLANES = TD / 128;
  static constexpr int KHALF = TN / 2;               // kv rows per CTA (B N-half)
  alignas(1024) uint8_t q[PLANES][TM * 128];         // this CTA's 128 q rows
  alignas(1024) uint8_t k[NSTAGE][PLANES][KHALF * 128];
  alignas(1024) uint8_t v[NSTAGE][PLANES][TN * 128]; // full kv rows (dv-half read by hw)
  alignas(16) uint64_t q_loaded;        // local, 96
  alignas(16) uint64_t q_pair;          // rank0, count 1 (rank1 forwarder)
  alignas(16) uint64_t k_loaded[NSTAGE];  // local, 96
  alignas(16) uint64_t v_loaded[NSTAGE];  // local, 96
  alignas(16) uint64_t k_pair[NSTAGE];    // rank0, count 1 (rank1 forwarder)
  alignas(16) uint64_t v_pair[NSTAGE];    // rank0, count 1
  alignas(16) uint64_t qk_done[2];        // local, multicast commit (count 1)
  alignas(16) uint64_t score_pair[2];     // rank0, count 8 (both CTAs' softmax)
  alignas(16) uint64_t probability_ready[2];  // local, count 4 (state handoff)
  alignas(16) uint64_t prob_pair[2];      // rank0, count 8 (PV gate)
  alignas(16) uint64_t output_ready[2];   // local, multicast commit (count 1)
  alignas(16) uint64_t pv_last;           // v6c: epilogue barrier (mc, count 1)
  alignas(16) uint64_t rescaled_pair;     // rank0, count 8 (corrections done)
  alignas(16) uint64_t k_released[NSTAGE];  // local, multicast commit (count 1)
  alignas(16) uint64_t v_released[NSTAGE];  // local, multicast commit (count 1)
  float state_m2[TM];
  float state_d[TM];
  uint32_t tmem_base;
  int tab_cache[TN / 16 + 2];
};

template <int TD, int NSTAGE, int DV = TD>
__global__ __launch_bounds__(NTHREADS, 1)
void fmha_tc2_kernel(const __grid_constant__ Params p) {
  constexpr int PLANES = TD / 128;
  constexpr int KHALF = TN / 2;
  constexpr int CPR = DV / (16 * PLANES);     // 16B chunks per row per plane
  constexpr int KSTEPS = DV / (32 * PLANES);  // QK K-dim 32B steps per plane
  constexpr int NPV = DV / PLANES;            // PV idesc N per plane
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Shared2<TD, NSTAGE>& sh = *reinterpret_cast<Shared2<TD, NSTAGE>*>(smem_raw);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const uint32_t rank = cluster_rank();   // == blockIdx.x & 1 (clusterDim.x=2)
  const int cta_id = blockIdx.y * gridDim.x + blockIdx.x;
  volatile unsigned* tcdbg =
      (p.dbg && cta_id < 64) ? (volatile unsigned*)(p.dbg + 128 * cta_id) : nullptr;

  const int u = blockIdx.y;
  // Cluster covers the x-tile pair (x0, x0+1); causal orders PAIRS heavy-first.
  const int npairs = (int)gridDim.x >> 1;
  const int cidx = (int)blockIdx.x >> 1;
  int b, g, q0, qlen, kvlen, R, tiles, r0, lo, hi;
  fmtc::tile_geometry(p, u, 0, b, g, q0, qlen, kvlen, R, tiles, r0, lo, hi);
  const int cpairs = (tiles + 1) / 2;
  const int c = p.causal ? (cpairs - 1 - cidx) : cidx;   // pair index (heavy first)
  const int x0 = 2 * c;                                   // cidx < npairs; host
                                                            // pads grid.x to 2*npairs
  // Whole-cluster no-op when the pair index maps outside this unit's tiles
  // (units smaller than max_q_len, or the odd-tail padding pair). Both ranks
  // take this uniformly, so skipping the collective setup is cluster-safe.
  if (x0 < 0 || x0 >= tiles) return;
  const int x = x0 + (int)rank;                           // this CTA's row tile
  const bool x_valid = (x < tiles);
  const int x_ref = min(x0 + 1, tiles - 1);               // kv geometry driver

  // Row geometry from this CTA's x (invalid x -> rows all masked downstream).
  int b2, g2, q02, qlen2, kvlen2, R2, tiles2, r02, lo2, hi2;
  fmtc::tile_geometry(p, u, min(x, tiles - 1), b2, g2, q02, qlen2, kvlen2, R2,
                      tiles2, r02, lo2, hi2);
  r0 = x * TM;                       // may exceed R for an invalid tail tile
  const bool rows_live = (r0 < R);   // false -> every row masked (row_ok)
  // Cluster kv window: lo from x0 (widest), hi from x_ref (widest).
  int lo_c = lo2, hi_c = hi2;
  {
    int b3, g3, q03, qlen3, kvlen3, R3, tiles3, r03, lo3, hi3;
    fmtc::tile_geometry(p, u, x0, b3, g3, q03, qlen3, kvlen3, R3, tiles3, r03, lo3, hi3);
    lo_c = lo3;
    int b4, g4, q04, qlen4, kvlen4, R4, tiles4, r04, lo4, hi4;
    fmtc::tile_geometry(p, u, x_ref, b4, g4, q04, qlen4, kvlen4, R4, tiles4, r04, lo4, hi4);
    hi_c = hi4;
  }
  const int ntiles = (hi_c - lo_c + TN - 1) / TN;

  // Split-KV (S>1): chunk c covers LOCAL tiles [c*cpt, (c+1)*cpt).
  const int csplit = (p.S > 1) ? (int)blockIdx.z : 0;
  int t_begin = 0, t_end = ntiles;
  if (p.S > 1) {
    const int cpt = p.chunk / TN;
    t_begin = min(ntiles, csplit * cpt);
    t_end = min(ntiles, t_begin + cpt);
    if (t_begin >= t_end) {
      // Empty chunk: self-cleaning markers, then STILL run the cluster
      // machinery with nloc=0 (a cluster peer may have work; cluster ops
      // are collective and cannot be skipped).
      const int rows = x_valid ? min(TM, R - r0) : 0;
      float* slot = p.ws + (((int64_t)csplit * gridDim.y + u) * p.T + x)
                               * (int64_t)(TM * (DV / 2 + 2));
      if (tid < rows) {
        slot[(int64_t)tid * (DV / 2 + 2) + DV / 2] = -INFINITY;
        slot[(int64_t)tid * (DV / 2 + 2) + DV / 2 + 1] = 0.f;
      }
      TCW(63, 1);
    }
  }
  const int nloc = (t_begin >= t_end) ? 0 : (t_end - t_begin);

  if (tid == 0) {
    fmtc::mbar_init(&sh.q_loaded, 96);
    fmtc::mbar_init(&sh.q_pair, 1);
    fmtc::mbar_init(&sh.rescaled_pair, 8);
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) {
      fmtc::mbar_init(&sh.k_loaded[s], 96);
      fmtc::mbar_init(&sh.v_loaded[s], 96);
      fmtc::mbar_init(&sh.k_pair[s], 1);
      fmtc::mbar_init(&sh.v_pair[s], 1);
      fmtc::mbar_init(&sh.k_released[s], 1);
      fmtc::mbar_init(&sh.v_released[s], 1);
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      fmtc::mbar_init(&sh.qk_done[s], 1);
      fmtc::mbar_init(&sh.score_pair[s], 8);
      fmtc::mbar_init(&sh.probability_ready[s], 4);
      fmtc::mbar_init(&sh.prob_pair[s], 8);
      fmtc::mbar_init(&sh.output_ready[s], 1);
    }
    fmtc::mbar_init(&sh.pv_last, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster_sync();   // barrier init visible cluster-wide before any remote arrive
  if (warp == 8) {
    tmem_alloc2(&sh.tmem_base, TMEM_COLS);
    tmem_relinq2();
  }
  __syncthreads();
  const uint32_t tb = sh.tmem_base;

  const int32_t* tab = p.table + (int64_t)b * p.Tstride;
  const uint8_t* kb = p.k + (int64_t)g * p.k_sh;
  const uint8_t* vb = p.v + (int64_t)g * p.v_sh;
  const int plog = p.page_log2, pmask = p.page - 1;

  if (warp >= 9) {
    // ------------------------------ loaders ---------------------------------
    setmaxnreg_dec<96>();
    TCW(1 + warp, warp + 1);
    TINIT();
    const int ltid = tid - 288;
    // Q: this CTA's own 128 rows (clamped when the tail tile is invalid).
    for (int idx = ltid; idx < TM * CPR * PLANES; idx += 96) {
      const int pl = idx / (TM * CPR);
      const int rem = idx - pl * (TM * CPR);
      const int row = rem / CPR;
      const int col = (rem - row * CPR) * 16;
      int rg = r0 + row;
      if (rg > R - 1) rg = R - 1;
      const int t = rg / p.ratio;
      const int h = g * p.ratio + (rg - t * p.ratio);
      fmtc::tc_cp_async16(sh.q[pl] + fmtc::swz128(row, col),
                          p.q + (int64_t)(q0 + t) * p.q_tok_stride + (int64_t)h * DV
                              + pl * 128 + col);
    }
    fmtc::cp_async_mbar_arrive(&sh.q_loaded);
    TACC(ck_);
#pragma unroll 1
    for (int tl = 0; tl < nloc; ++tl) {
      const int tile = t_begin + tl;
      const int stage = tl % NSTAGE;
      if (tl >= NSTAGE) {
        const int wN = tl - NSTAGE;
        TCW(25 + warp, tile * 16 + 1);
        fmtc::mbar_wait(&sh.k_released[wN % NSTAGE], (wN / NSTAGE) & 1);
        TCW(25 + warp, tile * 16 + 2);
        fmtc::mbar_wait(&sh.v_released[wN % NSTAGE], (wN / NSTAGE) & 1);
        TACC(cw_);
      }
      const int n0t = lo_c + tile * TN;
      int posmax = n0t + TN - 1;
      if (posmax > hi_c - 1) posmax = hi_c - 1;
      const int tab_lo = n0t >> plog;
      const int nent = (posmax >> plog) - tab_lo + 1;   // <= TN/16+1 = 9
      if (ltid < nent) sh.tab_cache[ltid] = __ldg(tab + tab_lo + ltid);
      asm volatile("bar.sync 1, 96;" ::: "memory");
      // K: this CTA's kv half [n0t + 64*rank, +64).
#pragma unroll 4
      for (int idx = ltid; idx < KHALF * CPR * PLANES; idx += 96) {
        const int pl = idx / (KHALF * CPR);
        const int rem = idx - pl * (KHALF * CPR);
        const int row = rem / CPR;
        const int col = (rem - row * CPR) * 16;
        int pos = n0t + (int)rank * KHALF + row;
        if (pos > hi_c - 1) pos = hi_c - 1;
        const int64_t poff = (int64_t)sh.tab_cache[(pos >> plog) - tab_lo];
        const int off = pos & pmask;
        fmtc::tc_cp_async16(sh.k[stage][pl] + fmtc::swz128(row, col),
                            kb + poff * p.k_sp + (int64_t)off * p.k_sn + pl * 128 + col);
      }
      // V: this CTA's dv-half (cols [rank*NPV/2, +NPV/2) per plane) stored at
      // LOGICAL cols [0, NPV/2) of the 128B-strided rows: the 2-CTA PV B
      // descriptor covers each CTA's own N-half from row byte 0 (same SW128
      // mapping; upper-half bytes stay garbage and are never addressed).
#pragma unroll 4
      for (int idx = ltid; idx < TN * (CPR / 2) * PLANES; idx += 96) {
        const int pl = idx / (TN * (CPR / 2));
        const int rem = idx - pl * (TN * (CPR / 2));
        const int row = rem / (CPR / 2);
        const int col = (rem - row * (CPR / 2)) * 16;
        int pos = n0t + row;
        if (pos > hi_c - 1) pos = hi_c - 1;
        const int64_t poff = (int64_t)sh.tab_cache[(pos >> plog) - tab_lo];
        const int off = pos & pmask;
        fmtc::tc_cp_async16(sh.v[stage][pl] + fmtc::swz128(row, col),
                            vb + poff * p.v_sp + (int64_t)off * p.v_sn + pl * 128
                                + (int)rank * (NPV / 2) + col);
      }
      fmtc::cp_async_mbar_arrive(&sh.k_loaded[stage]);
      fmtc::cp_async_mbar_arrive(&sh.v_loaded[stage]);
      TACC(ck_);
      TCW(13 + warp, tile + 1);
    }
    TDUMP(nloc);
  } else if (warp == 8) {
    // --------------------- mma issue (rank0) / forwarder (rank1) --------------
    setmaxnreg_dec<96>();
    TCW(1 + warp, warp + 1);
    if (lane == 0) {
      TINIT();
      TCW(25 + warp, 8);
      fmtc::mbar_wait(&sh.q_loaded, 0);
      if (rank == 1) {
        // Forwarder: local cp.async completions -> rank0's pair barriers.
        arrive_rank0(&sh.q_pair, rank);
#pragma unroll 1
        for (int tl = 0; tl < nloc; ++tl) {
          const int stage = tl % NSTAGE;
          fmtc::mbar_wait(&sh.k_loaded[stage], (tl / NSTAGE) & 1);
          arrive_rank0(&sh.k_pair[stage], rank);
          fmtc::mbar_wait(&sh.v_loaded[stage], (tl / NSTAGE) & 1);
          arrive_rank0(&sh.v_pair[stage], rank);
        }
        TDUMP(nloc);
      } else {
        fmtc::mbar_wait(&sh.q_pair, 0);
        TACC(cw_);
        // M=256 cluster-wide; per-CTA tmem D holds this CTA's 128 rows.
        const uint32_t id_qk = fmtc::idesc_f8(2 * TM, TN, false);
        const uint32_t id_pv = fmtc::idesc_f8(2 * TM, NPV, true);
#pragma unroll 1
        for (int tl = 0; tl <= nloc; ++tl) {
          if (tl < nloc) {
            const int tile = t_begin + tl;
            const int s2 = tl & 1;
            const int stage = tl % NSTAGE;
            TCW(25 + warp, tile * 16 + 3);
            fmtc::mbar_wait(&sh.k_loaded[stage], (tl / NSTAGE) & 1);
            fmtc::mbar_wait(&sh.k_pair[stage], (tl / NSTAGE) & 1);
            if (tl >= 2) {
              TCW(25 + warp, tile * 16 + 4);
              fmtc::mbar_wait(&sh.score_pair[s2], ((tl - 2) >> 1) & 1);
            }
            TACC(cw_);
            const uint32_t dmem = tb + s2 * 128;
#pragma unroll
            for (int pl = 0; pl < PLANES; ++pl) {
              uint64_t ad = fmtc::make_desc(2, 16, 8 * 128, sh.q[pl]);
              uint64_t bd = fmtc::make_desc(2, 16, 8 * 128, sh.k[stage][pl]);
#pragma unroll
              for (int step = 0; step < KSTEPS; ++step) {
                umma_f8_ss_2cta(dmem, ad, bd, id_qk, pl != 0 || step != 0);
                if (step != KSTEPS - 1) { fmtc::desc_add(ad, 32); fmtc::desc_add(bd, 32); }
              }
            }
            umma_commit2_mc(&sh.qk_done[s2], 0x3);
            umma_commit2_mc(&sh.k_released[stage], 0x3);
            TACC(ck_);
            TCW(13 + warp, tile + 1);
          }
          if (tl >= 1) {
            const int pv = tl - 1;
            const int ps = pv & 1;
            const int pvstage = pv % NSTAGE;
            TCW(25 + warp, (t_begin + tl) * 16 + 5);
            fmtc::mbar_wait(&sh.v_loaded[pvstage], (pv / NSTAGE) & 1);
            fmtc::mbar_wait(&sh.v_pair[pvstage], (pv / NSTAGE) & 1);
            TCW(25 + warp, (t_begin + tl) * 16 + 6);
            fmtc::mbar_wait(&sh.prob_pair[ps], (pv >> 1) & 1);
            if (pv >= 1) {
              TCW(25 + warp, (t_begin + tl) * 16 + 7);
              fmtc::mbar_wait(&sh.rescaled_pair, (pv - 1) & 1);
            }
            TACC(cw_);
            const uint32_t pa = tb + ps * 128 + 32;
#pragma unroll
            for (int pl = 0; pl < PLANES; ++pl) {
              uint64_t vd = fmtc::make_desc(2, 0, 8 * 128, sh.v[pvstage][pl]);
              const uint32_t om = tb + O_TMEM + pl * 128;
#pragma unroll
              for (int step = 0; step < 4; ++step) {
                umma_f8_ts_2cta(om, pa + step * 8, vd, id_pv, pv != 0 || step != 0);
                if (step != 3) fmtc::desc_add(vd, 32 * 128);
              }
            }
            umma_commit2_mc(&sh.output_ready[ps], 0x3);
            umma_commit2_mc(&sh.v_released[pvstage], 0x3);
            if (pv == nloc - 1) umma_commit2_mc(&sh.pv_last, 0x3);
            TACC(ck_);
          }
        }
        TDUMP(nloc);
      }
    }
  } else {
    // ------------- softmax groups A/B + merged correction (per CTA) ----------
    // Identical math to the promoted v2 1-CTA kernel; only the barrier arrive
    // targets change (pair barriers live on rank0; rank1 arrives remotely).
    setmaxnreg_inc<200>();
    const int grp = warp >> 2;
    TCW(1 + warp, warp + 1);
    TINIT();
    const int r = (warp & 3) * 32 + lane;
    const int rg = r0 + r;
    const int tok = rg / p.ratio;
    const int vis_i = kvlen - qlen + tok;
    const bool row_ok = rows_live && (rg < R);
    const uint32_t st_lane = ((uint32_t)(warp & 3) * 32) << 16;
    const uint32_t oaddr = tb + O_TMEM + st_lane;

    float m2r = -INFINITY;   // running max in RAW domain (qk_scale > 0 -> max
    float dsum = 0.f;        // commutes); tracks sum(p_hat) = PSCALE*sum(p).
#pragma unroll 1
    for (int tl = grp; tl < nloc; tl += 2) {
      const int tile = t_begin + tl;
      const int s2 = tl & 1;
      TCW(25 + warp, tile * 16 + 9);
      fmtc::mbar_wait(&sh.qk_done[s2], (tl >> 1) & 1);
      TACC(cw_);
      uint32_t sc[128];
      // C3b (mirrors fmha_tc.cuh): segmented tmem read + 4-way max chains;
      // state handoff wait placed after the release-arrive.
      fmtc::tmem_ld32(sc, tb + s2 * 128);
      fmtc::tmem_ld_wait();
#pragma unroll
      for (int seg = 1; seg < 4; ++seg)
        fmtc::tmem_ld32(sc + seg * 32, tb + s2 * 128 + seg * 32);
      TACC(ck_);

      const int n0t = lo_c + tile * TN;
      int c_lo = 0, c_hi = 128;
      if (!row_ok) {
        c_hi = 0;
      } else {
        int ub = hi_c;
        if (p.causal) ub = min(ub, vis_i + 1);
        c_hi = min(128, ub - n0t);
        if (p.window_left >= 0) c_lo = max(0, vis_i - p.window_left - n0t);
        if (c_hi < c_lo) c_hi = c_lo;
      }
      const unsigned span = (unsigned)(c_hi - c_lo);
      float mx0 = -INFINITY, mx1 = -INFINITY, mx2 = -INFINITY, mx3 = -INFINITY;
      // Span branch is lane-divergent: wait/sync/arrive stay warp-uniform.
      if (span >= 128) {
#pragma unroll
        for (int i = 0; i < 32; ++i) mx0 = fmaxf(mx0, __uint_as_float(sc[i]));
      } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const bool in = (unsigned)(i - c_lo) < span;
          float raw = __uint_as_float(sc[i]);
          raw = in ? raw : -INFINITY;
          sc[i] = __float_as_uint(raw);
          mx0 = fmaxf(mx0, raw);
        }
      }
      fmtc::tmem_ld_wait();
      __syncwarp(0xffffffffu);
      if (lane == 0) arrive_rank0(&sh.score_pair[s2], rank);
      if (span >= 128) {
#pragma unroll
        for (int i = 32; i < 64; ++i) mx1 = fmaxf(mx1, __uint_as_float(sc[i]));
#pragma unroll
        for (int i = 64; i < 96; ++i) mx2 = fmaxf(mx2, __uint_as_float(sc[i]));
#pragma unroll
        for (int i = 96; i < 128; ++i) mx3 = fmaxf(mx3, __uint_as_float(sc[i]));
      } else {
#pragma unroll
        for (int i = 32; i < 64; ++i) {
          const bool in = (unsigned)(i - c_lo) < span;
          float raw = __uint_as_float(sc[i]);
          raw = in ? raw : -INFINITY;
          sc[i] = __float_as_uint(raw);
          mx1 = fmaxf(mx1, raw);
        }
#pragma unroll
        for (int i = 64; i < 96; ++i) {
          const bool in = (unsigned)(i - c_lo) < span;
          float raw = __uint_as_float(sc[i]);
          raw = in ? raw : -INFINITY;
          sc[i] = __float_as_uint(raw);
          mx2 = fmaxf(mx2, raw);
        }
#pragma unroll
        for (int i = 96; i < 128; ++i) {
          const bool in = (unsigned)(i - c_lo) < span;
          float raw = __uint_as_float(sc[i]);
          raw = in ? raw : -INFINITY;
          sc[i] = __float_as_uint(raw);
          mx3 = fmaxf(mx3, raw);
        }
      }
      if (tl >= 1) {
        TCW(25 + warp, tile * 16 + 11);
        fmtc::mbar_wait(&sh.probability_ready[(tl - 1) & 1], ((tl - 1) >> 1) & 1);
        m2r = sh.state_m2[r];
        dsum = sh.state_d[r];
        TACC(cw_);
      }
      const float tmaxr = fmaxf(fmaxf(mx0, mx1), fmaxf(mx2, mx3));
      const float qs = p.qk_scale;
      // v5 anchored max (mirrors fmha_tc.cuh): rebase only when the tile max
      // exceeds the anchor by REBASE_MARGIN; p_hat <= 2^(8+margin) < 448.
      const bool rebase = ((tmaxr - m2r) * qs) > fmtc::REBASE_MARGIN;
      const float mnr = rebase ? tmaxr : m2r;
      const float mbase = (mnr == -INFINITY) ? 0.f : mnr;
      // +PLOG2 folds PSCALE into the exponent (dsum tracks 32*sum(p)).
      const float nmus8 = fmtc::PLOG2 - mbase * qs;
      const float accs = rebase ? fmtc::tc_ex2((m2r - mnr) * qs) : 1.0f;
      uint32_t pw[32];
      uint32_t ps2[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int widx = 0; widx < 32; ++widx) {
        const float y0 = fmaf(__uint_as_float(sc[4 * widx + 0]), qs, nmus8);
        const float y1 = fmaf(__uint_as_float(sc[4 * widx + 1]), qs, nmus8);
        const float y2 = fmaf(__uint_as_float(sc[4 * widx + 2]), qs, nmus8);
        const float y3 = fmaf(__uint_as_float(sc[4 * widx + 3]), qs, nmus8);
        const uint32_t pa = fmtc::ex2_f16x2(fmtc::cvt_f16x2_sat(y0, y1));
        const uint32_t pb = fmtc::ex2_f16x2(fmtc::cvt_f16x2_sat(y2, y3));
        ps2[widx & 3] = fmtc::hadd2_u32(ps2[widx & 3], pa);
        ps2[widx & 3] = fmtc::hadd2_u32(ps2[widx & 3], pb);
        pw[widx] = fmtc::pack_e4m3x4(pa, pb);
      }
      const uint32_t pst = fmtc::hadd2_u32(fmtc::hadd2_u32(ps2[0], ps2[1]),
                                           fmtc::hadd2_u32(ps2[2], ps2[3]));
      const float psum = fmtc::f16x2_lo(pst) + fmtc::f16x2_hi(pst);
      dsum = dsum * accs + psum;
      m2r = mnr;
      // Publish state BEFORE the probability arrives below (release).
      sh.state_m2[r] = m2r;
      sh.state_d[r] = dsum;
      fmtc::tmem_st32(tb + s2 * 128 + 32 + st_lane, pw);
      fmtc::tmem_st_wait();
      __syncwarp(0xffffffffu);
      if (lane == 0) {
        fmtc::mbar_arrive(&sh.probability_ready[s2]);   // CTA-local handoff
        arrive_rank0(&sh.prob_pair[s2], rank);          // rank0's PV gate
      }
      TACC(ck_);
      if (tl >= 1) {
        // v8 (mirrors fmha_tc.cuh): unconditional PV(tl-1)-completion wait
        // before the rescaled_pair arrival (V0 ordering; skip-arrive was a
        // flaky-deadlock suspect); only the rescale stays rebase-conditional.
        const bool do_corr = __ballot_sync(0xffffffffu, rebase);
#ifdef FMHA_TC_TIMING
        if (tcdbg && lane == 0 && do_corr) tcdbg[112 + warp] += 1;
#endif
        TCW(25 + warp, tile * 16 + 10);
        fmtc::mbar_wait(&sh.output_ready[(tl - 1) & 1], ((tl - 1) >> 1) & 1);
        TACC(cw_);
        if (do_corr) {
          // C3c (mirrors fmha_tc.cuh): batched correction loads/stores.
          constexpr int NB = DV / 32;
          constexpr int BB = NB < 4 ? NB : 4;
          uint32_t vals[BB * 32];
#pragma unroll
          for (int b0 = 0; b0 < NB; b0 += BB) {
#pragma unroll
            for (int j = 0; j < BB; ++j)
              fmtc::tmem_ld32(vals + j * 32, oaddr + (b0 + j) * 32);
            fmtc::tmem_ld_wait();
#pragma unroll
            for (int i = 0; i < BB * 32; ++i)
              vals[i] = __float_as_uint(__uint_as_float(vals[i]) * accs);
#pragma unroll
            for (int j = 0; j < BB; ++j)
              fmtc::tmem_st32(oaddr + (b0 + j) * 32, vals + j * 32);
            fmtc::tmem_st_wait();
          }
        }
        __syncwarp(0xffffffffu);
        if (lane == 0) arrive_rank0(&sh.rescaled_pair, rank);
        TACC(ck_);
      }
      TCW(13 + warp, tile + 1);
    }

    if (nloc > 0 && grp == ((nloc - 1) & 1)) {
      // Final epilogue, owned by the last tile's group; state lives in regs.
      TCW(25 + warp, (t_begin + nloc) * 16 + 13);
      fmtc::mbar_wait(&sh.pv_last, 0);
      TACC(cw_);
      if (p.S > 1) {
        float* slot = p.ws + (((int64_t)csplit * gridDim.y + u) * p.T + x)
                                 * (int64_t)(TM * (DV / 2 + 2));
        float* srow = slot + (int64_t)r * (DV / 2 + 2);
        uint32_t* wrow = reinterpret_cast<uint32_t*>(srow);
        if (row_ok) {
          srow[DV / 2] = m2r * p.qk_scale;
          srow[DV / 2 + 1] = dsum;
        }
        // C3d (mirrors fmha_tc.cuh): batched epilogue loads.
        constexpr int NBp = DV / 32;
        constexpr int BBp = NBp < 4 ? NBp : 4;
#pragma unroll
        for (int b0 = 0; b0 < NBp; b0 += BBp) {
          uint32_t vals[BBp * 32];
#pragma unroll
          for (int j = 0; j < BBp; ++j)
            fmtc::tmem_ld32(vals + j * 32, oaddr + (b0 + j) * 32);
          fmtc::tmem_ld_wait();
          if (row_ok) {
#pragma unroll
            for (int i = 0; i < BBp * 32; i += 2)
              wrow[(b0 * 32 + i) / 2] =
                  pack_bf16x2(__uint_as_float(vals[i]),
                              __uint_as_float(vals[i + 1]));
          }
        }
      } else if (rows_live) {
        const int t = rg / p.ratio;
        const int h = g * p.ratio + (rg - t * p.ratio);
        float base2 = 0.f;
        float inv = 0.f;
        if (row_ok) {
          base2 = m2r * p.qk_scale + log2f(dsum) - fmtc::PLOG2;
          inv = p.bmm2 / dsum;
          if (p.has_sink) {
            const float sk = p.sinks[h] * 1.4426950408889634f;
            const float tot = fmaxf(base2, sk) + log2f(1.f + exp2f(-fabsf(base2 - sk)));
            inv *= exp2f(base2 - tot);
            base2 = tot;
          }
          if (p.has_lse) p.lse[(int64_t)(q0 + t) * p.H + h] = base2;
        }
        uint16_t* orow = p.out + ((int64_t)(q0 + min(t, qlen - 1)) * p.H + min(h, p.H - 1)) * DV;
        // C3d: batched epilogue loads.
        constexpr int NBo = DV / 32;
        constexpr int BBo = NBo < 4 ? NBo : 4;
#pragma unroll
        for (int b0 = 0; b0 < NBo; b0 += BBo) {
          uint32_t vals[BBo * 32];
#pragma unroll
          for (int j = 0; j < BBo; ++j)
            fmtc::tmem_ld32(vals + j * 32, oaddr + (b0 + j) * 32);
          fmtc::tmem_ld_wait();
          if (row_ok) {
#pragma unroll
            for (int pr = 0; pr < BBo * 16; ++pr) {
              const float x2 = __uint_as_float(vals[2 * pr]) * inv;
              const float x1 = __uint_as_float(vals[2 * pr + 1]) * inv;
              __nv_bfloat162 pk = __float22bfloat162_rn(make_float2(x2, x1));
              *reinterpret_cast<uint32_t*>(orow + b0 * 32 + 2 * pr) =
                  *reinterpret_cast<uint32_t*>(&pk);
            }
          }
        }
      } else {
        // Invalid tail tile (rank1 past `tiles`): still must drain the last
        // PV's completion for the cluster teardown ordering.
        ;
      }
      TACC(ck_);
    }
    TDUMP((nloc - grp + 1) / 2);
  }

  TCW(37 + warp, 1);
  if (p.exit_mode != 2) {
    cluster_sync();   // both CTAs finished every tmem read
    TCW(50 + warp, 1);
  }
  if (warp == 8) {
    if (p.exit_mode == 0) {
      tmem_dealloc2(tb, TMEM_COLS);
      TCW(62, 1);
    } else {
      TCW(61, 1);
    }
  }
}

}  // namespace fmtc2
