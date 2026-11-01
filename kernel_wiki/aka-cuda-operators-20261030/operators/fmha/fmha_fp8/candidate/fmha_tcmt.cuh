// fmha_tcmt.cuh — MTN=192 STAGE-B variant of the v9 MEGATILE tcgen05 FP8 paged
// FMHA: 192 keys per megatile (halves 128 + 64), P buffers of 48 columns, PV
// k-steps = 6, host restricts MT to page_size <= 64 so a megatile is a whole
// number of pages — AND the deeper P pipeline that the 192 geometry exists to
// make reachable: PBUF=3, PVLAG=2, MNS=4, output_ready widened to NSTAGE slots
// and the loader's redundant qk_done release wait dropped. Stage A
// (experiments/mt192_src) is the same geometry with the shipped choreography and
// is the control that separates the two changes. Original:
// v9 MEGATILE tcgen05 FP8 paged FMHA (TN=256 keys per
// softmax round) for Dqk=Dv=64/128, S==1, window-free routes. Independent
// implementation for problem 14; mechanism = the promoted v8f fmha_tc.cuh
// pipeline re-tiled to halve per-key protocol cost and run the tensor pipe
// at the measured N=256 instruction rates (umma_bench sec 31-33:
// QK_D128_N256 7448 MAC/cyc -> 563 cyc/256-key tile vs 814 for 2xN128;
// PV_K256 7610 MAC/cyc -> 551 vs 764; D64 QK_D64_N256 328 vs 566).
//
// Topology (same 12 warps / 384 threads as fmha_tc.cuh):
//   w0-3 group A (EVEN megatiles), w4-7 group B (ODD), w8 mma issue,
//   w9-11 loaders (192-key stages; v8d split count-1 TMA tx barriers).
// tmem (512 cols): S [0,192) SINGLE buffer; P x3 [192,240)/[240,288)/
// [288,336) (48 cols = 192 e4m3/row); O [336, 336+DV).
// Pipeline (PVLAG=2): step j issues QK(j) [waits score_released[(j-1)&1]: S is
// single-buffered] then PV(j-2) [waits probability_ready[(j-2)%3] and
// output_rescaled[(j-2)%3], parity ((j-2)/3)&1]. qk_done/score_released stay
// 2-SLOTTED by megatile parity (single-slot forms early-pass: QK(m-2) shares
// softmax(m)'s wait parity and release(j-3) shares QK(j)'s - the original
// proto_sim SimMT flagged both); probability_ready/output_rescaled are
// PBUF-slotted and output_ready is NSTAGE-slotted, each with parity
// (index/slots)&1 = the completion count that index implies. At lag 2 the mma's
// PV waits are ALWAYS already satisfied when it reaches them, because P(j-2)
// was stored by the group of the opposite parity, which woke on qk_done(j-2)
// two steps earlier — that is the whole point: at lag 1 the mma instead blocks
// on probability_ready(j-1), stored by the SAME group that just released S, and
// the tensor pipe idles through the entire exp pass.
// P-buffer lifetime: the group writing P(m') must not start before PV(m'-PBUF)
// retired. Step j issues QK(j) then PV(j-PVLAG), so PV(m) is issued right after
// QK(m+PVLAG) and retires before qk_done(m+PVLAG+1) — exactly the barrier the
// group owning m'=m+PVLAG+1 waits on. Hence PBUF == PVLAG+1, with zero margin,
// which is the same equality (2 == 1+1) the shipped lag-1 build relies on.
// Softmax per megatile: issue BOTH halves' S segment loads with an f16x2
// compaction of half0 in between (sc[] freed for half1); score_released
// arrives once both halves are register-resident (~T+480cyc, unblocking
// QK(m+1) early). Anchor math mirrors v8f EXACTLY (raw-domain m2r,
// REBASE_MARGIN=3, mbase clamp, accs = ex2((m2r_in - mnr)*qs)) but is
// decided per HALF: half0's pab stays live in registers so a half1 rebase
// rescales it with one hmul2 pass before the SINGLE e4m3 pack (no tmem
// round-trip, no double quantization). Fully-masked halves force tmaxr to
// -INFINITY (the f16 max would read -65504 and poison the anchor; v8f gets
// this free from its fp32 max over masked -INF selects). dsum/lse/out
// identities are the v8f ones (anchor-invariant; PSCALE=2^5; MT is
// host-gated to S==1, so no combine path). Register peak ~215 -> ptxas
// spills ~15 around the half1 load window (accepted; overlaps tmem waits).
// Barrier parity windows: every wait's target+1 completion is issue-ordered
// behind the waiter's own arrival (proto_sim SimMT matrix, nmt 1..33).
//
// v9i REVERT NOTICE (permanent hardware/platform knowledge): merged
// producer gates (v9d: qk_gate = 4 releases + loader arrive_expect_tx in
// ONE count-5 phase; v_gate = count-1 arrive_expect_tx) produced RARE
// nondeterministic partial deadlocks (2-7/216 CTAs at lag-1, 85-129/216 at
// lag-2; dv_...205100Z/210450Z/211803Z) on every tmem map tried, while
// every split-barrier config (v8d-proven: plain count arrives + separate
// expect_tx) is clean (D64 ~50k+, D128 ~200k+ CTA launches). This is the
// v7b merged-expect_tx flaky-hang class again - SM103 mbarrier phases
// mixing expect_tx with plain multi-arrivals are NOT safe here, at any
// measured ~2% perf benefit. Also killed by the bisects: (H1) S-region
// (QK N=256 D) at cols >= 256 -> rare stuck CTAs; (H2) PV TS A-operand at
// cols >= 384 (P x3 layout) -> frequent stuck; hence PBUF=3/lag-2 is
// unreachable on both D and the single map below is the shipped one.
// tmem: S [0,192) | P x2 [192,288) | O [288, 288+DV). PBUF=2, lag 1.

#pragma once

#include "fmha_tc.cuh"

namespace fmtcmt {

using fmtc::TM;
using fmtc::NTHREADS;
using fmtc::PLOG2;
using fmtc::REBASE_MARGIN;
using fmtc::TMEM_COLS;
using fmtc::make_desc;
using fmtc::desc_add;
using fmtc::mbar_init;
using fmtc::mbar_wait;
using fmtc::mbar_arrive;
using fmtc::mbar_arrive_expect_tx;
using fmtc::tc_cp_async16;
using fmtc::cp_async_mbar_arrive;
using fmtc::tma_4d;
using fmtc::umma_commit;
using fmtc::umma_f8_ss;
using fmtc::umma_f8_ts;
using fmtc::idesc_f8;
using fmtc::tmem_alloc;
using fmtc::tmem_relinq;
using fmtc::tmem_dealloc;
using fmtc::tmem_ld32;
using fmtc::tmem_st32;
using fmtc::tmem_ld_wait;
using fmtc::tmem_st_wait;
using fmtc::tc_ex2;
using fmtc::cvt_f16x2_sat;
using fmtc::ex2_f16x2;
using fmtc::hadd2_u32;
using fmtc::hmax2_u32;
using fmtc::hfma2_u32;
using fmtc::hmax_halves2;
using fmtc::f16x2_dup;
using fmtc::f16x2_lo;
using fmtc::f16x2_hi;
using fmtc::pack_e4m3x4;
using fmtc::swz128;
using fmtc::setmaxnreg_inc;
using fmtc::setmaxnreg_dec;
using fmtc::tile_geometry;

// 16-column tmem store. The half1 P buffer at MTN=192 is 16 columns (64 e4m3
// bytes per row) where the shipped MTN=256 half is 32; tcgen05.st comes in
// x8/x16/x32 (results/device-facts.md).
__device__ __forceinline__ void tmem_st16(uint32_t a, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
      ::"r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]),
        "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]),
        "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]));
}
// NOTE: TCW/TINIT/TACC/TDUMP are preprocessor macros (fmha_tc.cuh) — visible
// via the include, no using-declaration possible or needed.

// STAGE A geometry: MTN=192 keys per megatile. 192 is the largest multiple of 32
// (the tcgen05.ld column granularity) that is also a multiple of every legal
// page_size up to 64, and it is what makes a deeper P pipeline reachable at all:
// S[0,192) + P x2 [192,288) + O[288,288+DV) here, and S[0,192) + P x4 [192,384) +
// O[384,512) in stage B -- every P buffer below column 384 and S below 256, i.e.
// inside both recorded SM103 errata, where MTN=256 leaves no room for a third P
// buffer at all (256 + 192 + 128 = 576 > 512). peak4 measured N=192 at 7929
// MAC/cyc/SM against 7907 at N=128 and 7998 at N=256, so the tensor rate is
// unchanged; 192 keys per stage at MNS=4 also costs exactly the same smem as 256
// keys at MNS=3 (4*192*128*2 = 3*256*128*2 = 192KB).
// A megatile geometry. Two are carried because page_size decides which is
// legal: the loader issues one TMA per whole page segment (nseg = MTN >>
// log2(page)), so MTN must be a multiple of page. 192 covers page <= 64 and is
// the faster map; page 128/256 can only use 256, where errata H2 forbids the
// deeper P pipeline (PBUF=3 at MTN=256 puts P at columns >= 384 -> "frequent
// stuck CTAs"), so those shapes keep the shipped lag-1 choreography.
//
// Every invariant the two maps satisfy is asserted here, so a third geometry
// cannot be added without meeting them. Each cites the measurement or the
// bisect that established it; see results/join23-mt192a.md and join23-mt192b.md.
template <int MTN_, int PBUF_, int PVLAG_, int MNS_>
struct MTGeo {
  static constexpr int MTN = MTN_;         // keys per megatile
  static constexpr int PBUF = PBUF_;       // P buffers in tmem
  static constexpr int PVLAG = PVLAG_;     // step j issues QK(j) then PV(j-PVLAG)
  static constexpr int MNS = MNS_;         // smem K/V stages
  static constexpr int PB = MTN_ / 4;      // P columns per buffer (MTN e4m3 bytes)
  static constexpr int PVSTEPS = MTN_ / 32;   // PV k-steps of 32 keys
  static constexpr int H1K = MTN_ - 128;   // half1 keys (half0 is always 128)
  static constexpr int H1W = H1K / 2;      // half1 f16x2 words
  static constexpr int H1S = H1K / 32;     // half1 tmem_ld32 segments
  static constexpr uint32_t S_TMEM = 0;                    // S [0, MTN)
  static constexpr uint32_t P_TMEM = MTN_;                 // P [MTN, MTN+PBUF*PB)
  static constexpr uint32_t O_TMEM_MT = MTN_ + PBUF_ * (MTN_ / 4);

  static_assert(MTN_ % 32 == 0 && H1K % 64 == 0,
                "tcgen05.ld granularity is 32 columns and half1's P store is one "
                "x16 or x32 tcgen05.st, so H1K must be 64 or 128");
  static_assert(PBUF_ == PVLAG_ + 1,
                "P(m) must survive until PV(m) retires: step j issues QK(j) then "
                "PV(j-PVLAG), so PV(m) retires before qk_done(m+PVLAG+1), exactly "
                "the barrier the group owning m+PBUF waits on. Also, at "
                "PBUF==PVLAG the mma's probability_ready wait finds done two ahead "
                "of target, the parity matches, and it deadlocks (simmt.py).");
  static_assert(MNS_ - PVLAG_ >= 2,
                "loader refill window is MNS-PVLAG steps (it frees stage tl only "
                "after PV(tl-MNS) COMPLETES); shipped MTN=256 is 3-1=2 and going "
                "below that halves the bytes/cycle budget for a K+V stage");
  static_assert(MTN_ <= 256, "errata H1: S at columns >= 256 gives rare stuck CTAs");
  static_assert(MTN_ + PBUF_ * (MTN_ / 4) <= 384,
                "errata H2: a PV A-operand (P) at columns >= 384 gives frequent "
                "stuck CTAs");
  static_assert(MTN_ + PBUF_ * (MTN_ / 4) + 128 <= 512,
                "O needs DV=128 columns and TMEM_COLS is 512, fully allocated");
  static_assert(MNS_ * MTN_ * 128 * 2 + 128 * 128 <= 232448,
                "K+V stages plus Q must fit the 227KB dynamic smem limit");
};
// page <= 64: the deep-P pipeline. 4*192*128*2 = 196608 B of stages, exactly
// what the shipped 3*256*128*2 costs, so smem is free; tmem is
// S[0,192) | P x3 [192,336) | O[336,464).
using MT192 = MTGeo<192, 3, 2, 4>;
// page 128/256: the shipped map and choreography, unchanged.
// tmem S[0,256) | P x2 [256,384) | O[384,512).
using MT256 = MTGeo<256, 2, 1, 3>;

// STAGE B: the deeper P pipeline, which is the entire point of moving to 192.
// Three invariants fix the shape of it, all three readable off the shipped tree:
//
// (1) PBUF == PVLAG + 1. The group that writes P(m') must not start before
//     PV(m' - PBUF) has consumed its A-operand. The tensor pipe executes in
//     issue order and step j issues QK(j) THEN PV(j-PVLAG), so PV(m) is issued
//     immediately after QK(m+PVLAG) and therefore completes before
//     qk_done(m+PVLAG+1) -- which is exactly the barrier the group owning
//     m' = m+PVLAG+1 waits on. The shipped MTN=256 build is PBUF=2/PVLAG=1,
//     i.e. the same equality with zero margin, and it is the margin that is
//     preserved here, not merely the depth.
//
// (2) loader slack = MNS - PVLAG >= 2 steps. The loader frees stage `tl` only
//     after PV(tl-MNS) COMPLETES (output_ready), and PV(w) is issued at step
//     w+PVLAG, so the refill window is MNS-PVLAG steps. Shipped: 3-1 = 2. The
//     MNS=4/PVLAG=3 map that peak4's header sketched would give 4-3 = 1, i.e.
//     HALF the shipped refill window for a 48KB K+V stage, in exchange for
//     slack that invariant (1) does not need. MNS=4/PVLAG=2 keeps 2.
//
// (3) every P column < 384 and every S column < 256 (SM103 errata H1/H2 in the
//     header above). P spans [192,336), S spans [0,192). O at [336,464) is a
//     PV D-operand, and D-operands are precedented at both 256 (fmha_tc.cuh's
//     O_TMEM) and 384 (this file at MTN=256) in shipped builds.
//
// What it buys. At lag 1 the single mma warp issues QK(j) and then blocks on
// probability_ready(j-1), which the group owning j-1 arrives only AFTER that
// megatile's whole exp+pack pass -- so the tensor pipe idles through exp, and
// results/join23-mt-decomposition.md measured exactly that: 3290 cyc/megatile
// against a 1062-cycle tensor floor and a 1050-cycle exp floor that are on
// different pipes and could overlap. At lag 2, PV(j-2)'s P comes from the OTHER
// group (parity (j-2)&1 == j&1), which woke on qk_done(j-2) two steps earlier,
// so both of the mma's waits are already satisfied when it reaches them and the
// only remaining gate on QK(j) is score_released(j-1), which fires right after
// the S loads, before cvt/exp. Tensor floor at MTN=192 is 791 cyc/megatile
// (6.29M MAC / 7950); exp floor is 768. Two softmax groups in parallel give the
// pair ~500 cyc/megatile of throughput, so the tensor becomes the limit.

// Fine-grained MT phase trace (FMHA_TC_TIMING builds only): softmax warp0
// accumulates 8 phase deltas into dbg slots 112..119, mma lane0 into
// 120..127 (kcyc summed over iterations; divide by the role's TDUMP tile
// count). Independent of TACC's tp_ chain; ~17 extra registers, timing
// builds only. Phase maps:
//  sm: 0 qk-wake | 1 ld0+cvt0 | 2 ld1a+cvtA | 3 ld1b+release-arrive
//      4 cvtB+handoff-wait | 5 dec+E0+E1 | 6 pack/st/publish-arrive
//      7 corr-wait(+body)+out_rs-arrive (loop tail)
//  mma: 0 k-load waits | 1 score_rel wait | 2 QK issue+commit
//       3 v-load waits | 4 prob_ready wait | 5 out_rs wait
//       6 PV issue+commit | 7 step tail
// One-shot absolute timestamps (clock64>>7, 128-cyc granularity) on the
// TPH_TGT-th role iteration only: ~80 cyc/iter perturbation elsewhere, vs
// the ~1.7k/step the accumulative-delta form cost (accounting gap in the
// dv_...180443Z phase run: wall 8.2k/iter vs phase-sum 5.0k). Slots:
// sm w0 -> 112..119 (0 qkwake 1 ld0cvt0 2 ld1acvtA 3 release 4 handoff
// 5 E0E1 6 publish 7 corr+tail), mma lane0 -> 120..126 (0 kwait 1 score_rel
// 2 qkissue 3 vwait 4 prob 5 out_rs 6 pvissue) + 127 = next-step head.
#ifdef FMHA_TC_TIMING
#define TPH_TGT 5
#define TPH(i)                                                                 \
  do {                                                                         \
    if (tcph_ && phc_ == TPH_TGT)                                              \
      tcdbg[tphb_ + (i)] = (unsigned)(clock64() >> 7);                         \
  } while (0)
#define TPHH()                                                                 \
  do {                                                                         \
    if (tcph_ && phc_ == TPH_TGT + 1)                                          \
      tcdbg[tphb_ + 7] = (unsigned)(clock64() >> 7);                           \
  } while (0)
#define TPH_INIT() unsigned phc_ = 0u
#define TPH_STEP() phc_++
#else
#define TPH(i)
#define TPHH()
#define TPH_INIT()
#define TPH_STEP()
#endif

// r17wb (compile flag FMHA_WB; requires !FMHA_TC_TIMING — the TDUMP region
// would collide — and FMHA_NO_TCW so the TCW marker stores stay off):
// wave-boundary structural timestamps, per-CTA dbg slots 64..78:
//   64/65 entry globaltimer lo/hi   66 entry clock64    67 smid
//   68 x (post-flip m-block)        69 u (unit)         70 nmt
//   71 early-return flag            72 init-done c64 (tid0, post alloc sync)
//   73 mma j-loop entry c64         74 first QK issued c64 (j==0)
//   75 mma j-loop end c64           76 loader w9 final-sync c64
//   77 tid0 final c64 (post final sync)   78 tid0 final globaltimer lo
// Pure stores to the existing dbg region; math/protocol/ordering untouched;
// production (p.dbg null) never executes a store. Question: reconcile the
// MT wall against the per-SM cadence sum (r8 cadence x tiles / 148 SMs
// underpredicts c7 by ~13-17%) — decompose the gap into launch gaps +
// prologue + epilogue per wave boundary and measure the TRUE in-run clock
// (globaltimer/clock64 ratio); decides the persistent-CTA question.
#if defined(FMHA_WB) && defined(FMHA_TC_TIMING)
#error "FMHA_WB collides with TDUMP slots; build without FMHA_TC_TIMING"
#endif
#ifdef FMHA_WB
__device__ __forceinline__ unsigned long long wb_gt() {
  unsigned long long v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
  return v;
}
__device__ __forceinline__ unsigned wb_smid() {
  unsigned v;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
  return v;
}
#define WB(i, v)                                                              \
  do {                                                                        \
    if (tcdbg) tcdbg[64 + (i)] = (unsigned)(v);                               \
  } while (0)
#else
#define WB(i, v)
#endif

__device__ __forceinline__ uint32_t hmul2_u32(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

template <class G>
struct SharedMT {
  static constexpr int NSTAGE = G::MNS;
  static constexpr int MTN = G::MTN;
  static constexpr int PBUF = G::PBUF;
  alignas(1024) uint8_t q[TM * 128];
  alignas(1024) uint8_t k[NSTAGE][MTN * 128];
  alignas(1024) uint8_t v[NSTAGE][MTN * 128];
  alignas(16) uint64_t q_loaded;
  alignas(16) uint64_t k_loaded[NSTAGE];
  alignas(16) uint64_t v_loaded[NSTAGE];
  alignas(16) uint64_t k_tx[NSTAGE];
  alignas(16) uint64_t v_tx[NSTAGE];
  alignas(16) uint64_t qk_done[2];          // count 1, megatile-parity slots
  alignas(16) uint64_t score_released[2];   // count 4, megatile-parity slots
  alignas(16) uint64_t probability_ready[PBUF];  // count 4, P-buffer slots
  alignas(16) uint64_t output_ready[NSTAGE];  // count 1, STAGE-indexed slots
  alignas(16) uint64_t output_rescaled[PBUF];    // count 4, P-buffer slots
  alignas(16) uint64_t pv_last;               // count 1
  float state_m2[TM];   // raw-domain anchor handoff (v8f convention)
  float state_d[TM];
  uint32_t tmem_base;
  int tab_cache[MTN / 16 + 2];
};

// TD fixed at 128 (single-plane 128B rows; D64 TMA uses compact SW64 rows).
// r18: PERS=true instantiates the persistent-worker path (see the fork
// inside); PERS=false (default) compiles the shipped per-CTA body verbatim.
struct MTAssign {
  int b, g, q0, qlen, kvlen, R, r0, lo, hi, x, u, nmt;
};

// aid -> (u, bx): EXACTLY the non-persistent cta_id decomposition (linear id
// over the original (gridDim.x=T, gridDim.y=nunits) grid, tile_order remap
// included), then the causal flip + second tile_geometry pass + S==1
// megatile count. Pure function of (p, aid): every role recomputes the same
// sequence independently, so workers need NO cross-role communication to
// agree on their assignments. Returns false for aids the non-persistent grid
// early-returns on (padded m-blocks) or with an empty span.
template <int MTN>
__device__ __forceinline__ bool mt_aid_geom(const Params& p, int aid,
                                            MTAssign& A) {
  int bx;
  if (p.tile_order) {
    A.u = aid % p.pers_gy;
    bx = aid / p.pers_gy;
  } else {
    A.u = aid / p.pers_gx;
    bx = aid % p.pers_gx;
  }
  int tiles;
  tile_geometry(p, A.u, bx, A.b, A.g, A.q0, A.qlen, A.kvlen, A.R, tiles,
                A.r0, A.lo, A.hi);
  A.x = p.causal ? (tiles - 1 - bx) : bx;
  if (A.x < 0 || A.x >= tiles) return false;
  tile_geometry(p, A.u, A.x, A.b, A.g, A.q0, A.qlen, A.kvlen, A.R, tiles,
                A.r0, A.lo, A.hi);
  A.nmt = (A.hi - A.lo + MTN - 1) / MTN;
  return A.nmt > 0;
}

template <int TD, class G, int DV, bool PERS = false>
__global__ __launch_bounds__(NTHREADS, 1)
void fmha_tcmt_kernel(const __grid_constant__ Params p) {
  static_assert(TD == 128 && (DV == 64 || DV == 128), "MT: D64/D128 only");
  // The geometry arrives as a type so one body serves both maps. These aliases
  // are the only names the body uses.
  constexpr int NSTAGE = G::MNS;
  constexpr int MTN = G::MTN;
  constexpr int PBUF = G::PBUF;
  constexpr int PVLAG = G::PVLAG;
  constexpr int PB = G::PB;
  constexpr int PVSTEPS = G::PVSTEPS;
  constexpr int H1K = G::H1K;
  constexpr int H1W = G::H1W;
  constexpr int H1S = G::H1S;
  constexpr uint32_t S_TMEM = G::S_TMEM;
  constexpr uint32_t P_TMEM = G::P_TMEM;
  constexpr uint32_t O_TMEM_MT = G::O_TMEM_MT;
  constexpr int ROWB = (DV >= 128) ? 128 : 64;   // TMA D64 = SW64 compact
  constexpr int CPR = DV / 16;                   // 16B chunks per row (cp.async/Q)
  constexpr int KSTEPS = DV / 32;                // QK K-dim 32B steps
  const bool tma64 = (DV < 128) && p.use_tma;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  SharedMT<G>& sh = *reinterpret_cast<SharedMT<G>*>(smem_raw);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int cta_id = blockIdx.y * gridDim.x + blockIdx.x;
#ifdef FMHA_DBG_ALLOC
  constexpr int DBG_CTAS = 512;
#else
  constexpr int DBG_CTAS = 64;
#endif
  volatile unsigned* tcdbg =
      (p.dbg && cta_id < DBG_CTAS) ? (volatile unsigned*)(p.dbg + 128 * cta_id) : nullptr;

#ifdef FMHA_WB
  if (tid == 0) {
    const unsigned long long wb_g0 = wb_gt();
    WB(0, wb_g0);
    WB(1, wb_g0 >> 32);
    WB(2, (unsigned)clock64());
    WB(3, wb_smid());
  }
#endif

  if constexpr (!PERS) {
  int u = blockIdx.y;
  int bx = (int)blockIdx.x;
  if (p.tile_order) {
    const int gid = bx + (int)blockIdx.y * (int)gridDim.x;
    const int nu = (int)gridDim.y;
    u = gid % nu;
    bx = gid / nu;
  }
  int b, g, q0, qlen, kvlen, R, tiles, r0, lo, hi;
  tile_geometry(p, u, bx, b, g, q0, qlen, kvlen, R, tiles, r0, lo, hi);
  const int x = p.causal ? (tiles - 1 - bx) : bx;
#ifdef FMHA_WB
  if (tid == 0 && (x < 0 || x >= tiles)) WB(7, 1u);
#endif
  if (x < 0 || x >= tiles) return;
  tile_geometry(p, u, x, b, g, q0, qlen, kvlen, R, tiles, r0, lo, hi);
  // MT+split (host knob FMHA_MT_SPLIT, impl 6/7 with p.S > 1): chunk c =
  // blockIdx.z covers LOCAL megatiles [c*cpt, (c+1)*cpt) of this row-tile's
  // [lo,hi) span; the host bumps p.chunk to MTN alignment so the partition
  // is exact and each megatile belongs to exactly one chunk. S==1 keeps the
  // full span (mt0 = 0).
  const int nunits_g = p.B * p.Hkv;
  const int c = (p.S > 1) ? (int)blockIdx.z : 0;
  const int nmt_all = (hi - lo + MTN - 1) / MTN;
  const int cpt = max(1, p.chunk / MTN);
  const int mt0 = (p.S > 1) ? min(nmt_all, c * cpt) : 0;
  int nmt = nmt_all - mt0;
  if (p.S > 1) nmt = min(nmt, cpt);
  if (nmt <= 0) {
    if (p.S > 1) {
      // Empty chunk: self-cleaning markers (m=-inf, d=0) so the combine
      // kernel skips this split; exit before any collective setup (mirrors
      // the tc route's empty-chunk path exactly).
      const int rows = min(TM, R - x * TM);
      float* slot = p.ws + (((int64_t)c * nunits_g + u) * p.T + x)
                               * (int64_t)(TM * (DV / 2 + 2));
      if (tid < rows) {
        slot[(int64_t)tid * (DV / 2 + 2) + DV / 2] = -INFINITY;
        slot[(int64_t)tid * (DV / 2 + 2) + DV / 2 + 1] = 0.f;
      }
    }
    return;
  }

#ifdef FMHA_WB
  if (tid == 0) {
    WB(4, x);
    WB(5, u);
    WB(6, nmt);
  }
#endif

  if (tid == 0) {
    mbar_init(&sh.q_loaded, 96);
    mbar_init(&sh.pv_last, 1);
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      mbar_init(&sh.qk_done[s], 1);
      mbar_init(&sh.score_released[s], 4);
    }
#pragma unroll
    for (int s = 0; s < PBUF; ++s) {
      mbar_init(&sh.probability_ready[s], 4);
      mbar_init(&sh.output_rescaled[s], 4);
    }
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) {
      mbar_init(&sh.output_ready[s], 1);
      mbar_init(&sh.k_loaded[s], p.use_tma ? 1 : 96);
      mbar_init(&sh.v_loaded[s], p.use_tma ? 1 : 96);
      mbar_init(&sh.k_tx[s], 1);
      mbar_init(&sh.v_tx[s], 1);
    }
  }
  if (tid == 0) {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 8) {
    tmem_alloc(&sh.tmem_base, TMEM_COLS);
    tmem_relinq();
  }
  __syncthreads();
  const uint32_t tb = sh.tmem_base;
#ifdef FMHA_WB
  if (tid == 0) WB(8, (unsigned)clock64());
#endif

  const int32_t* tab = p.table + (int64_t)b * p.Tstride;
  const uint8_t* kb = p.k + (int64_t)g * p.k_sh;
  const uint8_t* vb = p.v + (int64_t)g * p.v_sh;
  const int plog = p.page_log2, pmask = p.page - 1;

  if (warp >= 9) {
    // ------------------------------ loaders ---------------------------------
    // 192-key stages: TMA = 192/page segments (<= 12 per tensor), cp.async =
    // 256*CPR/96 chunks per thread. Release waits mirror v8d: stage wN frees
    // after QK(wN) exec (qk_done[wN&1] parity (wN>>1)&1) AND PV(wN)
    // completion (output_ready[wN&1]); the next same-slot completion is
    // issue-ordered behind this iteration's own k/v arrive -> window-safe.
    setmaxnreg_dec<72>();
    TCW(1 + warp, warp + 1);
    TINIT();
    const int ltid = tid - 288;
    for (int idx = ltid; idx < TM * CPR; idx += 96) {
      const int row = idx / CPR;
      const int col = (idx - row * CPR) * 16;
      int rg = r0 + row;
      if (rg > R - 1) rg = R - 1;
      const int t = rg / p.ratio;
      const int h = g * p.ratio + (rg - t * p.ratio);
      tc_cp_async16(sh.q + swz128(row, col),
                    p.q + (int64_t)(q0 + t) * p.q_tok_stride + (int64_t)h * DV + col);
    }
    cp_async_mbar_arrive(&sh.q_loaded);
    TACC(ck_);
    if (p.use_tma) {
#pragma unroll 1
      for (int tl = 0; tl < nmt; ++tl) {
        const int stage = tl % NSTAGE;
        const int n0t = lo + (mt0 + tl) * MTN;
        const int nseg = MTN >> plog;   // page segments per megatile (<=16)
        const int tab_lo = n0t >> plog;
        // C3a: table LDG before the release waits (latency overlap).
        int pg_reg = 0;
        if (ltid < nseg)
          pg_reg = __ldg(tab + min(tab_lo + ltid, p.Tstride - 1));
        if (tl >= NSTAGE) {
          const int wN = tl - NSTAGE;
          TCW(25 + warp, tl * 16 + 1);
          // ONE release wait, not two. tcgen05.commit fires only once every
          // umma this warp issued earlier has retired, and PV(wN) is issued
          // after QK(wN), so output_ready[wN] already implies K(wN) is free:
          // the shipped build's extra qk_done wait was redundant. At PVLAG=2 it
          // is fatal -- PV(wN) is issued at step wN+2, the SAME step as
          // QK(wN+2), so by the time the loader arrives, qk_done[wN&1] has
          // taken its next completion and a 2-slot parity window can never
          // reopen. experiments/simmt.py reproduces that deadlock at nmt=6 with
          // or_slots=2 and clears it with or_slots=NSTAGE + no qk_done wait.
          mbar_wait(&sh.output_ready[wN % NSTAGE], (wN / NSTAGE) & 1);
          TCW(25 + warp, tl * 16 + 2);
          TACC(cw_);
        }
        if (ltid < nseg) sh.tab_cache[ltid] = pg_reg;
        asm volatile("bar.sync 1, 96;" ::: "memory");
        // v8d split TMA barriers (thread0 arms tx + plain-arrives "issued";
        // mma waits loaded THEN tx). v9c: the 2*nseg bulk copies issue
        // PARALLEL over lanes 0..nseg-1 (thread0-serial issue cost ~2.3k
        // cyc/step of mma kwait at page16/32 ops, phase trace dv_...180443Z).
        // Ordering: arm(expect_tx) -> bar.sync -> lane issues -> tx barrier
        // gates DATA (complete_tx can never precede its arm), so k_loaded
        // may arrive before all lanes issue; k_tx still gates consumption.
        if (ltid == 0) {
#if FMHA_ABLATE == 9
          // zero-byte expect_tx completes the tx barriers without data moves
          mbar_arrive_expect_tx(&sh.k_tx[stage], 0);
          mbar_arrive_expect_tx(&sh.v_tx[stage], 0);
#else
          mbar_arrive_expect_tx(&sh.k_tx[stage], MTN * ROWB);
          mbar_arrive_expect_tx(&sh.v_tx[stage], MTN * ROWB);
#endif
        }
#if FMHA_ABLATE != 9
        asm volatile("bar.sync 1, 96;" ::: "memory");   // arms visible
        if (ltid < nseg) {
          const int pg = sh.tab_cache[ltid];
          const int seg_b = p.page * ROWB;
          tma_4d(sh.k[stage] + ltid * seg_b, &p.kmap, 0, 0, pg, g, &sh.k_tx[stage]);
          tma_4d(sh.v[stage] + ltid * seg_b, &p.vmap, 0, 0, pg, g, &sh.v_tx[stage]);
        }
#endif
        if (ltid == 0) {
          mbar_arrive(&sh.k_loaded[stage]);
          mbar_arrive(&sh.v_loaded[stage]);
        }
        TACC(ck_);
        TCW(13 + warp, tl + 1);
      }
      TDUMP(nmt);
    } else {
#pragma unroll 1
      for (int tl = 0; tl < nmt; ++tl) {
        const int stage = tl % NSTAGE;
        const int n0t = lo + (mt0 + tl) * MTN;
        int posmax = n0t + MTN - 1;
        if (posmax > hi - 1) posmax = hi - 1;
        const int tab_lo = n0t >> plog;
        const int nent = (posmax >> plog) - tab_lo + 1;   // <= MTN/16+1 = 17
        int pg_reg = 0;
        if (ltid < nent) pg_reg = __ldg(tab + tab_lo + ltid);
        if (tl >= NSTAGE) {
          const int wN = tl - NSTAGE;
          TCW(25 + warp, tl * 16 + 1);
          // ONE release wait, not two. tcgen05.commit fires only once every
          // umma this warp issued earlier has retired, and PV(wN) is issued
          // after QK(wN), so output_ready[wN] already implies K(wN) is free:
          // the shipped build's extra qk_done wait was redundant. At PVLAG=2 it
          // is fatal -- PV(wN) is issued at step wN+2, the SAME step as
          // QK(wN+2), so by the time the loader arrives, qk_done[wN&1] has
          // taken its next completion and a 2-slot parity window can never
          // reopen. experiments/simmt.py reproduces that deadlock at nmt=6 with
          // or_slots=2 and clears it with or_slots=NSTAGE + no qk_done wait.
          mbar_wait(&sh.output_ready[wN % NSTAGE], (wN / NSTAGE) & 1);
          TCW(25 + warp, tl * 16 + 2);
          TACC(cw_);
        }
        if (ltid < nent) sh.tab_cache[ltid] = pg_reg;
        asm volatile("bar.sync 1, 96;" ::: "memory");
#if FMHA_ABLATE != 9
#pragma unroll 4
        for (int idx = ltid; idx < MTN * CPR; idx += 96) {
          const int row = idx / CPR;
          const int col = (idx - row * CPR) * 16;
          int pos = n0t + row;
          if (pos > hi - 1) pos = hi - 1;
          const int64_t poff = (int64_t)sh.tab_cache[(pos >> plog) - tab_lo];
          const int off = pos & pmask;
          // cp.async rows are ALWAYS 128B-strided (swz128), also for D64.
          tc_cp_async16(sh.k[stage] + swz128(row, col),
                        kb + poff * p.k_sp + (int64_t)off * p.k_sn + col);
          tc_cp_async16(sh.v[stage] + swz128(row, col),
                        vb + poff * p.v_sp + (int64_t)off * p.v_sn + col);
        }
#endif
        cp_async_mbar_arrive(&sh.k_loaded[stage]);
        cp_async_mbar_arrive(&sh.v_loaded[stage]);
        TACC(ck_);
        TCW(13 + warp, tl + 1);
      }
      TDUMP(nmt);
    }
  } else if (warp == 8) {
    // ------------------------------ mma issue --------------------------------
    // dec<80> matches loaders w9-11 (setmaxnreg is warp-group collective).
    setmaxnreg_dec<72>();
    TCW(1 + warp, warp + 1);
    if (lane == 0) {
      TINIT();
      const bool tcph_ = (tcdbg != nullptr);
      const unsigned tphb_ = 120u;
      TPH_INIT();
      TCW(25 + warp, 8);
      mbar_wait(&sh.q_loaded, 0);
      TACC(cw_);
      const uint32_t id_qk = idesc_f8(TM, MTN, false);   // N=192 (7929 MAC/c, peak4)
      const uint32_t id_pv = idesc_f8(TM, DV, true);
#ifdef FMHA_WB
      WB(9, (unsigned)clock64());
#endif
      // Lag-PVLAG megatile loop: step j issues QK(j) then PV(j-PVLAG), so the
      // loop runs PVLAG steps past the last QK to drain the tail. qk_done and
      // score_released stay 2-SLOTTED BY MEGATILE PARITY (S is single-buffered,
      // so their cadence is set by QK, not by the PV depth); probability_ready
      // and output_rescaled are PBUF-slotted by P-buffer index with parity
      // (n/PBUF)&1, which is the k-th completion of that slot. Every wait's
      // window is still exactly {target, target+1}: the next completion of any
      // of them is issue-ordered behind a step the mma cannot reach without the
      // waiter's own next arrival (see the PBUF == PVLAG+1 derivation above).
#pragma unroll 1
      for (int j = 0; j < nmt + PVLAG; ++j) {
        TPHH();
        if (j < nmt) {
          const int stage = j % NSTAGE;
          TCW(25 + warp, j * 16 + 3);
#if FMHA_ABLATE != 13 && FMHA_ABLATE != 14
          mbar_wait(&sh.k_loaded[stage], (j / NSTAGE) & 1);
          if (p.use_tma) mbar_wait(&sh.k_tx[stage], (j / NSTAGE) & 1);
#endif
          TPH(0);
#if FMHA_ABLATE != 14
          if (j >= 1) {
            TCW(25 + warp, j * 16 + 4);
            mbar_wait(&sh.score_released[(j - 1) & 1], ((j - 1) >> 1) & 1);
          }
#endif
          TPH(1);
          TACC(cw_);
          {
            uint64_t ad = make_desc(2, 16, 8 * 128, sh.q);
            uint64_t bd = tma64 ? make_desc(4, 0, 8 * 64, sh.k[stage])
                                : make_desc(2, 16, 8 * 128, sh.k[stage]);
#pragma unroll
            for (int step = 0; step < KSTEPS; ++step) {
              umma_f8_ss(tb + S_TMEM, ad, bd, id_qk, step != 0);
              if (step != KSTEPS - 1) { desc_add(ad, 32); desc_add(bd, 32); }
            }
          }
          umma_commit(&sh.qk_done[j & 1]);
          TACC(ck_);
          TPH(2);
#ifdef FMHA_WB
          if (j == 0) WB(10, (unsigned)clock64());
#endif
          TCW(13 + warp, j + 1);
        }
        if (j >= PVLAG) {
          const int pv = j - PVLAG;
          const int pvstage = pv % NSTAGE;
          const int pbuf = pv % PBUF;
          TCW(25 + warp, j * 16 + 5);
#if FMHA_ABLATE != 13 && FMHA_ABLATE != 14
          mbar_wait(&sh.v_loaded[pvstage], (pv / NSTAGE) & 1);
          if (p.use_tma) mbar_wait(&sh.v_tx[pvstage], (pv / NSTAGE) & 1);
#endif
          TPH(3);
#if FMHA_ABLATE != 12 && FMHA_ABLATE != 14
          TCW(25 + warp, j * 16 + 6);
          mbar_wait(&sh.probability_ready[pbuf], (pv / PBUF) & 1);
          TPH(4);
          if (pv >= 1) {
            TCW(25 + warp, j * 16 + 7);
            mbar_wait(&sh.output_rescaled[pbuf], (pv / PBUF) & 1);
          }
#endif
          TPH(5);
          TACC(cw_);
          {
            const uint32_t pa = tb + P_TMEM + pbuf * PB;
            uint64_t vd = tma64 ? make_desc(4, 0, 8 * 64, sh.v[pvstage])
                                : make_desc(2, 0, 8 * 128, sh.v[pvstage]);
#pragma unroll
            for (int step = 0; step < PVSTEPS; ++step) {
              umma_f8_ts(tb + O_TMEM_MT, pa + step * 8, vd, id_pv,
                         pv != 0 || step != 0);
              if (step != PVSTEPS - 1) desc_add(vd, tma64 ? 32 * 64 : 32 * 128);
            }
          }
          umma_commit(&sh.output_ready[pv % NSTAGE]);
          if (pv == nmt - 1) umma_commit(&sh.pv_last);
          TACC(ck_);
          TPH(6);
        }
        TPH_STEP();
      }
#ifdef FMHA_WB
      WB(11, (unsigned)clock64());
#endif
      TDUMP(nmt);
    }
  } else {
    // ------------- softmax groups A/B (megatile two-half passes) --------------
    // Dual-group ping-pong as v8f, each iteration owning 192 keys: S loads
    // batch per half, half0 compacts to f16x2 (s16a) freeing sc[] for
    // half1's loads; the release arrive fires right after half1 lands (all S
    // data register-resident). Anchor math is v8f's (raw m2r, margin 3,
    // mbase clamp) evaluated per half; pab0 survives in registers so a
    // half1 rebase rescales it with one hmul2 pass before the SINGLE e4m3
    // pack. exp/psum use f16x2 SIMD exactly like v8f (pa <= ~266 < 448).
    setmaxnreg_inc<216>();
    const int grp = warp >> 2;
    TCW(1 + warp, warp + 1);
    TINIT();
    const bool tcph_ = (tcdbg != nullptr) && (warp == 0);
    const unsigned tphb_ = 112u;
    TPH_INIT();
    const int r = (warp & 3) * 32 + lane;
    const int rg = r0 + r;
    const int tok = rg / p.ratio;
    const int vis_i = kvlen - qlen + tok;
    const bool row_ok = (rg < R);
    const uint32_t st_lane = ((uint32_t)(warp & 3) * 32) << 16;
    const uint32_t oaddr = tb + O_TMEM_MT + st_lane;
    const float qs = p.qk_scale;
    const uint32_t qs2 = f16x2_dup(qs);

    float m2r = -INFINITY;   // running max in RAW domain (v8f convention)
    float dsum = 0.f;        // tracks sum(p_hat) = PSCALE*sum(p)
#pragma unroll 1
    for (int m = grp; m < nmt; m += 2) {
      // sbuf indexes the megatile-parity barriers: S is single-buffered, so
      // qk_done/score_released stay 2-slotted by m&1 no matter how deep the P
      // pipeline is. pbuf indexes the P buffers. The two coincide only at
      // PBUF=2, which is why the shipped tree could use one variable for both.
      const int sbuf = m & 1;
      const int pbuf = m % PBUF;
      TCW(25 + warp, m * 16 + 9);
      mbar_wait(&sh.qk_done[sbuf], (m >> 1) & 1);
      TACC(cw_);
      TPH(0);
      const int n0t = lo + (mt0 + m) * MTN;
      // Visible key window [c_lo, c_hi) within the 192-key megatile (v8f
      // int-domain selection; MT route has window_left < 0 -> c_lo == 0,
      // the general form is kept as a defensive no-op).
      int c_lo = 0, c_hi = MTN;
      if (!row_ok) {
        c_hi = 0;
      } else {
        int ub = hi;
        if (p.causal) ub = min(ub, vis_i + 1);
        c_hi = min(MTN, ub - n0t);
        if (p.window_left >= 0) c_lo = max(0, vis_i - p.window_left - n0t);
        if (c_hi < c_lo) c_hi = c_lo;
      }
      const unsigned span = (unsigned)(c_hi - c_lo);
      // Per-half spans: fully-masked halves MUST yield tmaxr = -INF (the f16
      // max reads -65504, which would rebase the anchor onto -65504 and give
      // masked keys weight ~2^PLOG2). span_h > 0 restores the v8f semantics.
      const int span0 = max(0, min(c_hi, 128) - min(c_lo, 128));
      const int span1 = max(0, (c_hi - 128) - max(c_lo - 128, 0));

      uint32_t sc[128];
      uint32_t s16a[64];
      uint32_t mx0 = 0xfc00fc00u;   // f16x2(-inf, -inf)
      // ---- half0: 4 segment loads, then mask+cvt+max into s16a ----
#if FMHA_ABLATE == 3 || FMHA_ABLATE == 11
#pragma unroll
      for (int i = 0; i < 128; ++i)
        sc[i] = __float_as_uint((float)((i + tid) & 15) - 8.f);
#else
      tmem_ld32(sc, tb + S_TMEM);
#pragma unroll
      for (int seg = 1; seg < 4; ++seg)
        tmem_ld32(sc + seg * 32, tb + S_TMEM + seg * 32);
      tmem_ld_wait();
#endif
#if FMHA_ABLATE == 11
#pragma unroll
      for (int w = 0; w < 64; ++w) s16a[w] = 0x30303030u;
      mx0 = 0x30303030u;
#else
      if (span0 >= 128) {
#pragma unroll
        for (int w = 0; w < 64; ++w) {
          s16a[w] = cvt_f16x2_sat(__uint_as_float(sc[2 * w]),
                                  __uint_as_float(sc[2 * w + 1]));
          mx0 = hmax2_u32(mx0, s16a[w]);
        }
      } else {
#pragma unroll
        for (int w = 0; w < 64; ++w) {
          const bool in0 = (unsigned)(2 * w - c_lo) < span;
          const bool in1 = (unsigned)(2 * w + 1 - c_lo) < span;
          const float f0v = in0 ? __uint_as_float(sc[2 * w]) : -INFINITY;
          const float f1v = in1 ? __uint_as_float(sc[2 * w + 1]) : -INFINITY;
          s16a[w] = cvt_f16x2_sat(f0v, f1v);
          mx0 = hmax2_u32(mx0, s16a[w]);
        }
      }
#endif   // FMHA_ABLATE == 11 (synthetic s16a)
      TPH(1);
      // ---- half1 loads in TWO 2-seg batches (v9b): the 4-seg batch peaked
      // at ~192+ regs (s16a[64] live across sc[128]) and ptxas spilled ~15
      // regs (104B st / 152B ld measured) whose LDL/STL drag showed up as
      // +1383 cyc/megatile in the abl1 decomposition (dv_f39727aef591).
      // Batched form peaks at s16a 64 + sc 64 + s16b <=64 = ~185+overhead:
      // spill-free; cost = one extra serialized tmem wait (~200 cyc) on the
      // release path, absorbed while the tensor floor (~1967) paces.
      uint32_t s16b[H1W];
      uint32_t mx1 = 0xfc00fc00u;
#if FMHA_ABLATE == 3 || FMHA_ABLATE == 11
#pragma unroll
      for (int i = 0; i < H1K; ++i)
        sc[i] = __float_as_uint((float)((i + tid + 3) & 15) - 8.f);
#else
      // ONE 4-segment batch for half1 instead of two 2-segment batches, so the
      // drain of S costs two serialized tmem waits per megatile instead of three.
      // The two-batch form existed because sc[128] live across s16a[64] peaks at
      // ~215 registers and spilled ~15 (104B st / 152B ld, +1383 cyc/megatile of
      // LDL/STL drag) at the old 200-register softmax budget; the extra wait was
      // accepted as "absorbed while the tensor floor (~1967) paces", and that
      // premise is refuted -- the kernel is latency-bound at 30.8% tensor
      // utilization (dv_116a762870d0), no-eligible 59%, and the S-load path is
      // 425 cyc of its 3310 cyc/megatile. The budget below is what makes both
      // hold: 8x216 softmax + 4x72 loader/mma = 2016 of the 2016 warp-register slots this launch reserves (168 regs x 384 threads = 64512 of 65536), which is the real pool ceiling -- 2048 would exceed it.
      // Indexing is unchanged in meaning: word w of half1 is key 128+2w, held in
      // sc[2w] by the segment that loaded it, and the mx1 chain keeps its order,
      // so this is bitwise identical to the two-batch form.
      // H1S segments in ONE batch, then one wait: 2 segments at MTN=192, 4 at
      // MTN=256 (promotion 7's form). Word w of half1 is key 128+2w, held in
      // sc[2w] by the segment that loaded it, for either geometry.
#pragma unroll
      for (int seg = 0; seg < H1S; ++seg)
        tmem_ld32(sc + seg * 32, tb + S_TMEM + 128 + seg * 32);
      tmem_ld_wait();
#endif
      // S fully register-resident (s16a + sc[0..127]) -> release for QK(m+1).
      // Warp sync first: lane0's arrive must imply EVERY lane's loads completed
      // (v3a lesson). Ahead of the cvt/max pass, as in the two-batch form.
      __syncwarp(0xffffffffu);
      if (lane == 0) mbar_arrive(&sh.score_released[sbuf]);
      TPH(2);
#if FMHA_ABLATE == 11
#pragma unroll
      for (int w = 0; w < H1W; ++w) s16b[w] = 0x30303030u;
      mx1 = 0x30303030u;
#else
      // span1 counts the VISIBLE keys of half1, whose width is MTN-128, not a
      // fixed 128: the shipped MTN=256 tree could write 128 because the two
      // coincide. At MTN=192 half1 is 64 keys wide, so `span1 >= 128` is never
      // true and EVERY megatile would take the bounds-masked branch. Both
      // branches are bit-identical when half1 is fully visible (in0/in1 are then
      // all true), so this is a pure speed fix, not a semantics change.
      if (span1 >= MTN - 128) {
#pragma unroll
        for (int w = 0; w < H1W; ++w) {
          s16b[w] = cvt_f16x2_sat(__uint_as_float(sc[2 * w]),
                                  __uint_as_float(sc[2 * w + 1]));
          mx1 = hmax2_u32(mx1, s16b[w]);
        }
      } else {
#pragma unroll
        for (int w = 0; w < H1W; ++w) {
          const bool in0 = (unsigned)(128 + 2 * w - c_lo) < span;
          const bool in1 = (unsigned)(128 + 2 * w + 1 - c_lo) < span;
          const float f0v = in0 ? __uint_as_float(sc[2 * w]) : -INFINITY;
          const float f1v = in1 ? __uint_as_float(sc[2 * w + 1]) : -INFINITY;
          s16b[w] = cvt_f16x2_sat(f0v, f1v);
          mx1 = hmax2_u32(mx1, s16b[w]);
        }
      }
#endif   // FMHA_ABLATE == 11 (synthetic s16b)
      TPH(3);
      // ---- state handoff (m-1's publisher arrived prob_ready[(m-1)&1]) ----
      // Placed after the release arrive (v8f C3b rationale: the S turnaround
      // never waits on the handoff).
      if (m >= 1) {
        TCW(25 + warp, m * 16 + 11);
        mbar_wait(&sh.probability_ready[(m - 1) % PBUF], ((m - 1) / PBUF) & 1);
        m2r = sh.state_m2[r];
        dsum = sh.state_d[r];
        TACC(cw_);
      }
      TPH(4);
      const float m2r_in = m2r;
      const float tmaxr0 = (span0 > 0) ? f16x2_lo(hmax_halves2(mx0)) : -INFINITY;
      const float tmaxr1 = (span1 > 0) ? f16x2_lo(hmax_halves2(mx1)) : -INFINITY;
      // ---- half0 anchor decision (v8f math; edges: first finite half ->
      // (+INF)*qs > margin -> rebase; masked half (tmaxr=-INF, anchor
      // finite) -> no rebase; both -INF -> NaN -> no rebase) ----
      const bool rebase0 =
          p.force_corr || (((tmaxr0 - m2r_in) * qs) > REBASE_MARGIN);
      const float mnr0 = rebase0 ? tmaxr0 : m2r_in;
      const float mbase0 = (mnr0 == -INFINITY) ? 0.f : mnr0;
      const uint32_t nm2a = f16x2_dup(PLOG2 - mbase0 * qs);
      // ---- exp half0 -> pab0 (kept live for a possible half1 rescale) ----
      uint32_t pab0[64];
      float psum0;
#if FMHA_ABLATE == 1 || FMHA_ABLATE == 11
#pragma unroll
      for (int w = 0; w < 64; ++w) pab0[w] = 0x30303030u;
      psum0 = 128.0f;
#else
      uint32_t ps2[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int w = 0; w < 64; ++w) {
        const uint32_t pa = ex2_f16x2(hfma2_u32(s16a[w], qs2, nm2a));
        pab0[w] = pa;
        ps2[w & 3] = hadd2_u32(ps2[w & 3], pa);
      }
      // f16 partial sums (<= 16*266 = 4256 per chain) finish in fp32.
      const uint32_t pst0 = hadd2_u32(hadd2_u32(ps2[0], ps2[1]),
                                      hadd2_u32(ps2[2], ps2[3]));
      psum0 = f16x2_lo(pst0) + f16x2_hi(pst0);
#endif
      // ---- half1 anchor decision; rescale pab0 in registers if it fires ----
      const bool rebase1 =
          p.force_corr || (((tmaxr1 - mnr0) * qs) > REBASE_MARGIN);
      const float mnr1 = rebase1 ? tmaxr1 : mnr0;
      if (__any_sync(0xffffffffu, rebase1)) {
        // mnr0 == -INF (lane's half0 fully masked AND fresh state): pab0 is
        // all-zero there; force f0 = 1 to keep the mixed live/dead warp
        // NaN-free (-INF - -INF would poison 0*f0 = NaN). rebase1 with a
        // finite tmaxr1 from mnr0 = -INF gives f0 = ex2(-INF) = 0, and
        // 0*0 = 0 keeps the dead half's P exact.
        const float f0 = (mnr0 == -INFINITY && mnr1 == -INFINITY)
                             ? 1.f
                             : tc_ex2((mnr0 - mnr1) * qs);
        const uint32_t f02 = f16x2_dup(f0);
#pragma unroll
        for (int w = 0; w < 64; ++w) pab0[w] = hmul2_u32(pab0[w], f02);
        psum0 *= f0;
      }
      const float mbase1 = (mnr1 == -INFINITY) ? 0.f : mnr1;
      const uint32_t nm2b = f16x2_dup(PLOG2 - mbase1 * qs);
      uint32_t pab1[H1W];
      float psum1;
#if FMHA_ABLATE == 1 || FMHA_ABLATE == 11
#pragma unroll
      for (int w = 0; w < H1W; ++w) pab1[w] = 0x30303030u;
      psum1 = 128.0f;
#else
      uint32_t ps2b[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int w = 0; w < H1W; ++w) {
        const uint32_t pa = ex2_f16x2(hfma2_u32(s16b[w], qs2, nm2b));
        pab1[w] = pa;
        ps2b[w & 3] = hadd2_u32(ps2b[w & 3], pa);
      }
      const uint32_t pst1 = hadd2_u32(hadd2_u32(ps2b[0], ps2b[1]),
                                      hadd2_u32(ps2b[2], ps2b[3]));
      psum1 = f16x2_lo(pst1) + f16x2_hi(pst1);
#endif
      TPH(5);
      // ---- state update (single accs for dsum AND the O correction) ----
      const bool rebased = rebase0 || rebase1;
      const float accs = rebased ? tc_ex2((m2r_in - mnr1) * qs) : 1.0f;
      dsum = dsum * accs + psum0 + psum1;
      m2r = mnr1;
      // ---- pack + store both P halves (columns [0,32) / [32,48) of the
      // 48-col P buffer; word w of half h = megatile keys 128h+4w..+3) ----
      uint32_t pw[32];
#if FMHA_ABLATE != 11
#pragma unroll
      for (int w = 0; w < 32; ++w)
        pw[w] = pack_e4m3x4(pab0[2 * w], pab0[2 * w + 1]);
      tmem_st32(tb + P_TMEM + pbuf * PB + st_lane, pw);
#pragma unroll
      for (int w = 0; w < H1W / 2; ++w)
        pw[w] = pack_e4m3x4(pab1[2 * w], pab1[2 * w + 1]);
      // half1 occupies PB-32 columns = H1W/2: 16 at MTN=192, 32 at MTN=256.
      // tcgen05.st comes only in x8/x16/x32, and both values are legal widths.
      if constexpr (H1W == 32)
        tmem_st16(tb + P_TMEM + pbuf * PB + 32 + st_lane, pw);
      else
        tmem_st32(tb + P_TMEM + pbuf * PB + 32 + st_lane, pw);
      tmem_st_wait();
#endif
      // Publish state BEFORE the probability_ready arrive (release/acquire,
      // cumulative via __syncwarp — v8f race-fix discipline).
      sh.state_m2[r] = m2r;
      sh.state_d[r] = dsum;
      __syncwarp(0xffffffffu);
      if (lane == 0) mbar_arrive(&sh.probability_ready[pbuf]);
      TACC(ck_);
      TPH(6);
      // ---- merged O correction (v8 arrival discipline: unconditional
      // output_ready wait for m>=1; scale body ballot-conditional) ----
      if (m >= 1) {
        const bool do_corr = __ballot_sync(0xffffffffu, rebased);
#if FMHA_ABLATE != 12 && FMHA_ABLATE != 14
        TCW(25 + warp, m * 16 + 10);
        mbar_wait(&sh.output_ready[(m - 1) % NSTAGE], ((m - 1) / NSTAGE) & 1);
        TACC(cw_);
#endif
#if FMHA_ABLATE != 11 && FMHA_ABLATE != 12 && FMHA_ABLATE != 14
        if (do_corr) {
          constexpr int NB = DV / 32;             // 2 (D64), 4 (D128)
          constexpr int BB = NB < 4 ? NB : 4;
          uint32_t vals[BB * 32];
#pragma unroll
          for (int b0 = 0; b0 < NB; b0 += BB) {
#pragma unroll
            for (int j = 0; j < BB; ++j)
              tmem_ld32(vals + j * 32, oaddr + (b0 + j) * 32);
            tmem_ld_wait();
#pragma unroll
            for (int i = 0; i < BB * 32; ++i)
              vals[i] = __float_as_uint(__uint_as_float(vals[i]) * accs);
#pragma unroll
            for (int j = 0; j < BB; ++j)
              tmem_st32(oaddr + (b0 + j) * 32, vals + j * 32);
            tmem_st_wait();
          }
        }
#endif   // FMHA_ABLATE != 11 && != 12 && != 14 (corr body)
        __syncwarp(0xffffffffu);
      }
      if (lane == 0) mbar_arrive(&sh.output_rescaled[pbuf]);
      TACC(ck_);
      TPH(7);
      TPH_STEP();
      TCW(13 + warp, m + 1);
    }

    if (grp == ((nmt - 1) & 1)) {
      // Final epilogue, owned by the last megatile's group; state in regs.
      TCW(25 + warp, nmt * 16 + 13);
      mbar_wait(&sh.pv_last, 0);
      TACC(cw_);
      if (p.S > 1) {
        // Split-KV partials (MT+split): O raw (PSCALE baked in); dsum
        // carries 2^PLOG2, combine's out = bmm2*acc/D exact with
        // lse_adj = -PLOG2. ws layout and conventions identical to the tc
        // route, so fmha_combine works unchanged.
        float* slot = p.ws + (((int64_t)c * nunits_g + u) * p.T + x)
                                 * (int64_t)(TM * (DV / 2 + 2));
        float* srow = slot + (int64_t)r * (DV / 2 + 2);
        uint32_t* wrow = reinterpret_cast<uint32_t*>(srow);
        if (row_ok) {
          srow[DV / 2] = m2r * p.qk_scale;
          srow[DV / 2 + 1] = dsum;
        }
        constexpr int NBp = DV / 32;
        constexpr int BBp = NBp < 4 ? NBp : 4;
#pragma unroll
        for (int b0 = 0; b0 < NBp; b0 += BBp) {
          uint32_t vals[BBp * 32];
#pragma unroll
          for (int j = 0; j < BBp; ++j)
            tmem_ld32(vals + j * 32, oaddr + (b0 + j) * 32);
          tmem_ld_wait();
          if (row_ok) {
#pragma unroll
            for (int i = 0; i < BBp * 32; i += 2)
              wrow[(b0 * 32 + i) / 2] =
                  pack_bf16x2(__uint_as_float(vals[i]),
                              __uint_as_float(vals[i + 1]));
          }
        }
      } else {
      const int t = rg / p.ratio;
      const int h = g * p.ratio + (rg - t * p.ratio);
      float base2 = 0.f;
      float inv = 0.f;
      if (row_ok) {
        base2 = m2r * p.qk_scale + log2f(dsum) - PLOG2;
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
      // 128-bit output stores. Each lane owns ONE row of `orow`, and the address
      // advances 4 B per pr, so four consecutive pr are 16 CONTIGUOUS bytes of that
      // lane's own row: one uint4 replaces four 32-bit stores, cutting this loop's
      // store instructions 4x (64 -> 16 per lane at DV=128, 128 -> 32 at DV=256). Same
      // values to the same addresses, so output identity is bit-preserved -- the
      // measured rel_l2 is identical to the 4-byte form on every case.
      //
      // Worth a lot because the epilogue is a TAIL, not a throughput term: it runs on
      // the four warps of the group that owned the last tile, after the other eight
      // warps are finished, and the CTA cannot retire until it completes -- and at
      // 196,608 B of smem exactly one CTA is resident per SM, so the SM idles for the
      // whole tail. Deleting the epilogue outright is worth 6.9% on c1, 18.9% on c7,
      // 15.8% on c6 and 13.2% on c0 (dv_d8ddf5ea48cf, epi_src); this recovers 64%,
      // 64%, 80% and 51% of that. The per-CTA tail is 11 kcyc on c1, 19 on c7 and 21
      // on c6, solved out of the group A/B work asymmetry in dv_c66b4bdfbbc7 -- which
      // is why no per-megatile decomposition ever saw it: over c1's 43 megatiles it is
      // 256 cyc/megatile, 6.5% of the wall, and FMHA_ABLATE=11 bundles it with the
      // whole softmax skeleton. An evict-first hint (__stcs) on top of this measured
      // as nothing (dv_6ff83aa462cf: 0.9553 vs 0.9574 on c1, 0.8739 vs 0.8747 on c6),
      // so the store stays a plain one.
      //
      // `wide` is belt-and-braces: `out` is required to be contiguous bf16, but
      // contiguity alone does not promise 16-B alignment and a misaligned uint4 store
      // is a hard device fault, so the ABI must not gain a new failure mode. The test
      // is on p.out only -- every row offset is row*DV*2 B with DV in {64,128,256},
      // b0*32 elements is 64 B, and 2*pr with pr a multiple of 4 is 16 B -- so it is
      // uniform across the whole grid. Fresh allocations are 512-B aligned, so the
      // narrow path is expected to be dead in practice.
      const bool wide = ((reinterpret_cast<uintptr_t>(p.out) & 15u) == 0u);
      constexpr int NBo = DV / 32;
      constexpr int BBo = NBo < 4 ? NBo : 4;
#pragma unroll
      for (int b0 = 0; b0 < NBo; b0 += BBo) {
        uint32_t vals[BBo * 32];
#pragma unroll
        for (int j = 0; j < BBo; ++j)
          tmem_ld32(vals + j * 32, oaddr + (b0 + j) * 32);
        tmem_ld_wait();
        if (row_ok) {
          if (wide) {
#pragma unroll
            for (int pr = 0; pr < BBo * 16; pr += 4) {
              uint4 pk4;
              uint32_t* w4 = reinterpret_cast<uint32_t*>(&pk4);
#pragma unroll
              for (int k = 0; k < 4; ++k) {
                const float x0 = __uint_as_float(vals[2 * (pr + k)]) * inv;
                const float x1 = __uint_as_float(vals[2 * (pr + k) + 1]) * inv;
                const __nv_bfloat162 pk =
                    __float22bfloat162_rn(make_float2(x0, x1));
                w4[k] = *reinterpret_cast<const uint32_t*>(&pk);
              }
              *reinterpret_cast<uint4*>(orow + b0 * 32 + 2 * pr) = pk4;
            }
          } else {
#pragma unroll
            for (int pr = 0; pr < BBo * 16; ++pr) {
              const float x0 = __uint_as_float(vals[2 * pr]) * inv;
              const float x1 = __uint_as_float(vals[2 * pr + 1]) * inv;
              __nv_bfloat162 pk = __float22bfloat162_rn(make_float2(x0, x1));
              *reinterpret_cast<uint32_t*>(orow + b0 * 32 + 2 * pr) =
                  *reinterpret_cast<uint32_t*>(&pk);
            }
          }
        }
      }
      }   // S > 1 partials vs direct out/lse
      TACC(ck_);
    }
    TDUMP((nmt - grp + 1) / 2);
  }

  TCW(37 + warp, 1);
  if (p.exit_mode != 2) {
    __syncthreads();
    TCW(50 + warp, 1);
#ifdef FMHA_WB
    if (warp == 9 && lane == 0) WB(12, (unsigned)clock64());
    if (tid == 0) {
      WB(13, (unsigned)clock64());
      WB(14, (unsigned)wb_gt());
    }
#endif
  }
  if (warp == 8) {
    if (p.exit_mode == 0) {
      tmem_dealloc(tb, TMEM_COLS);
      TCW(62, 1);
    } else {
      TCW(61, 1);
    }
  }
  }   // if constexpr (!PERS) — shipped per-CTA body + tail, verbatim

  if constexpr (PERS) {
    // ------------------ r18 persistent workers (S==1 scope) -----------------
    // LPT-balanced assignment lists written to p.ws by prepare (ws is unused
    // at S==1): worker w walks list w (heaviest first, -1 terminated). The
    // aid ENCODING is the original cta_id space, so mt_aid_geom reproduces
    // today's aid->(u,bx)->(x) mapping exactly (tile_order remap + causal
    // flip included) — assignment CONTENT, per-row math and locality class
    // are unchanged; only the worker->assignment binding differs (host LPT
    // instead of the hardware's work-conserving CTA dispatch).
    // Barriers are NEVER re-inited: every phase/parity expression rides the
    // GLOBAL step counter Ja (worker-cumulative megatiles, identical in all
    // roles because mt_aid_geom is pure) and qa (worker-local VALID-
    // assignment index): q_loaded/pv_last/o_drained phase qa-derived, ring/P
    // slots indexed (Ja+j) % NSTAGE/PBUF. Init/alloc/tmem pay ONCE. Q is
    // double-buffered (sh.q / q2 in the PERS-only smem page) so loaders
    // prefetch assignment a+1's Q while the mma still reads a's.
    // Boundary safety (no races, no new stalls beyond ~0):
    //  * PV(a+1,0) overwrites O (accumulate=false at pv==0): gated by
    //    o_drained(qa-1) — the SPLIT epilogue's 8 warps (both groups, each
    //    draining its column half) arrive after their LAST O tmem_ld_wait
    //    (global stores overlap onward) — and by the probability_ready
    //    chain. The non-owning group reads the final state from
    //    state_m2/state_d behind a probability_ready(J_last) wait, so its
    //    inv/base2 recompute is bit-identical to the owner's registers.
    //  * Q(qv) cp.async overwrites buffer qv&1 last read by QK(qv-2): the
    //    ring throttle alone does NOT bound this when assignments are tiny
    //    (nmt < NSTAGE), so the loader additionally waits output_ready(Ja -
    //    NSTAGE) before the Q fill: PV(Ja-4) retirement implies (tensor issue
    //    ordering) QK(Ja-2) retired, covering every QK of qv-2 (last tile
    //    Ja-nmt_prev-1 <= Ja-2). The slot's NEXT completion is PV(Ja), which
    //    the mma cannot issue before this loader's q_loaded(qv) arrive ->
    //    parity can never alias past the target (overshoot-free; the original
    //    qk_done-parity form could, and is the c7-hang suspect of
    //    dv_8a1f111790ec). Tiny-Ja (< NSTAGE+1) fallback: qk_done(Ja-1),
    //    whose slot next-completes at QK(Ja+1) = loader-gated as well.
    //  * mma's first QK(a+1) waits score_released(Ja'-1) = a's last tile,
    //    arrived by softmax BEFORE its epilogue -> S reuse safe.
    uint8_t* const q2 =
        smem_raw + ((sizeof(SharedMT<G>) + 1023u) & ~(size_t)1023u);
    uint64_t* const o_drained = reinterpret_cast<uint64_t*>(q2 + TM * 128);
    // q_loaded alternates between the original barrier and this PERS-page
    // copy (slot qa&1, phase (qa>>1)&1): a single slot could take TWO
    // arrives before the mma's first wait when the worker's first two
    // assignments are tiny (nmt <= NSTAGE => the loader runs unthrottled),
    // and a parity wait cannot see a double flip. With two slots, the qv>=2
    // qk_done guard (below) bounds every same-slot re-arrive behind the
    // mma's consumption of the previous phase.
    uint64_t* const qld_b = reinterpret_cast<uint64_t*>(q2 + TM * 128 + 16);
    // Monotone mma progress counter (q_loaded consumption count). The loader
    // throttles its arrive(qv) until the mma has consumed wait(qv-2), so a
    // slot can never take two arrives before its wait — parity aliasing is
    // impossible by construction even for chains of tiny (nmt~1) assignments,
    // where the ring throttle alone does not bound the loader's lead.
    volatile int* const mma_qa_p =
        reinterpret_cast<volatile int*>(q2 + TM * 128 + 32);
    // r18k dynamic-pull channel: the loader warp-9 lane0 is the single
    // puller (one atomicAdd per assignment); consumers spin on pull_seq
    // (monotone, publish count) then read pull_aid. sm_read_ctr is the
    // cumulative count of softmax warps that have READ the current aid
    // (8 per assignment; each warp's lane0 adds 1 after __syncwarp) — the
    // puller may overwrite pull_aid only once every consumer of the
    // previous aid is accounted for (mma via mma_qa_p, softmax via
    // sm_read_ctr). Ticket counter = pers_list[0] (global), pool aids at
    // pers_list[1..]; the worker drawing ticket n+nworkers-1 resets the
    // counter (single resetter, after every other pull completed) so each
    // graph replay starts at 0.
    volatile int* const pull_seq =
        reinterpret_cast<volatile int*>(q2 + TM * 128 + 40);
    volatile int* const pull_aid =
        reinterpret_cast<volatile int*>(q2 + TM * 128 + 44);
    volatile int* const sm_read_ctr =
        reinterpret_cast<volatile int*>(q2 + TM * 128 + 48);
    int* const ticket_ctr = const_cast<int*>(p.pers_list);
    const int* const pool_aids = p.pers_list + 1;
    const int nworkers = (int)gridDim.x;
    // r19d: STATIC balanced-partition segments for SHORT-CHAIN MT shapes
    // (host policy: items <= 5*nworkers; measured r18j-static beats
    // r18k-dynamic on c4 0.951/0.975, c7 0.961/0.981, d9 0.937/0.946 med —
    // the variance tax only dominates on c1-class LONG chains, 6.9 items/w,
    // which stay on the dynamic r18k ticket pool below). Read-only segment
    // walk, no atomics/counter — graph-replay safe by construction.
    const int* const seg_beg =
        p.pers_static ? (p.pers_list + (nworkers + 1) + p.pers_list[blockIdx.x])
                      : nullptr;
    const int seg_len =
        p.pers_static ? (p.pers_list[blockIdx.x + 1] - p.pers_list[blockIdx.x])
                      : 0;
    if (tid == 0) {
      *mma_qa_p = 0;
      *pull_seq = 0;
      *pull_aid = -1;
      *sm_read_ctr = 0;
      mbar_init(&sh.q_loaded, 96);
      mbar_init(qld_b, 96);
      mbar_init(&sh.pv_last, 1);
      mbar_init(o_drained, (DV == 64) ? 8 : 4);   // D64: split epilogue (8)
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        mbar_init(&sh.qk_done[s], 1);
        mbar_init(&sh.score_released[s], 4);
      }
#pragma unroll
      for (int s = 0; s < PBUF; ++s) {
        mbar_init(&sh.probability_ready[s], 4);
        mbar_init(&sh.output_rescaled[s], 4);
      }
#pragma unroll
      for (int s = 0; s < NSTAGE; ++s) {
        mbar_init(&sh.output_ready[s], 1);
        mbar_init(&sh.k_loaded[s], p.use_tma ? 1 : 96);
        mbar_init(&sh.v_loaded[s], p.use_tma ? 1 : 96);
        mbar_init(&sh.k_tx[s], 1);
        mbar_init(&sh.v_tx[s], 1);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 8) {
      tmem_alloc(&sh.tmem_base, TMEM_COLS);
      tmem_relinq();
    }
    __syncthreads();
    const uint32_t tb = sh.tmem_base;
#ifdef FMHA_WB
    if (tid == 0) WB(8, (unsigned)clock64());
#endif

    if (warp >= 9) {
      // ------------------------- loaders (persistent) -----------------------
      // Register pool stays at the shipped 8x216 + 4x72 = 2016 slots (the
      // launch reservation IS the ceiling: 224/64 = 2048 overshoots it and
      // HANGS the inc — dv_da8973cc4737 canary; census showed 224 did not
      // clear the D128 PERS spill anyway).
      setmaxnreg_dec<72>();
      const int ltid = tid - 288;
      // r18k DYNAMIC ticket pull (the loader warp-9 lane0 pulls for the
      // whole worker): the variance tax measured in r18j (wall = slowest
      // STATICALLY-bound worker) is what the hardware CTA dispatcher
      // absorbs for free in the non-persistent kernel; pulling restores it
      // (heavy-first ticket order preserves the locality class). Pacing:
      // publish aid(qa) only after mma started qa-1 (mma_qa_p >= qa) and
      // all 8 softmax warps READ aid(qa-1) (sm_read_ctr >= 8*qa).
      int Ja = 0, qa = 0;
      for (;;) {
        int aid;
        if (p.pers_static) {
          // r19e: DIRECT segment binding (the r18j machinery — every role
          // reads its own read-only segment; NO broadcast/pacing/ctr
          // atomics). r19d measured the broadcast machinery at ~2us per
          // assignment (c7 static-via-broadcast 1.02 vs r18j direct 0.961).
          aid = (qa < seg_len) ? seg_beg[qa] : -1;
        } else {
          if (ltid == 0) {
            if (qa >= 1) {
              while (*mma_qa_p < qa || *sm_read_ctr < 8 * qa) {
              }
            }
            const int t = atomicAdd(ticket_ctr, 1);
            if (t >= p.pers_kmax) {
              if (t == p.pers_kmax + nworkers - 1) {
                *ticket_ctr = 0;   // self-clean for the next graph replay
                __threadfence();
              }
              aid = -1;
            } else {
              aid = pool_aids[t];
            }
            *pull_aid = aid;
            __threadfence_block();
            *pull_seq = qa + 1;
          }
          asm volatile("bar.sync 1, 96;" ::: "memory");
          aid = *pull_aid;
        }
        if (aid < 0) break;
        MTAssign A;
        mt_aid_geom<MTN>(p, aid, A);
        const int nmt = A.nmt;
        const int32_t* tab = p.table + (int64_t)A.b * p.Tstride;
        const uint8_t* kb = p.k + (int64_t)A.g * p.k_sh;
        const uint8_t* vb = p.v + (int64_t)A.g * p.v_sh;
        const int plog = p.page_log2, pmask = p.page - 1;
        uint8_t* const qb = (qa & 1) ? q2 : sh.q;
#ifdef FMHA_WB
        if (ltid == 0) WB(19, (unsigned)aid);
#endif
        // Q(qv-2) overwrite guard (see the boundary-safety comment above).
        // Overshoot-free form: PV(Ja-NSTAGE) retirement implies — by tensor
        // issue ordering — that QK(Ja-2) and hence EVERY QK of assignment
        // qv-2 (last tile Ja-nmt_prev-1 <= Ja-2) retired, so the buffer is
        // read-free; the slot's NEXT completion is PV(Ja), which the mma
        // cannot issue before this loader's own q_loaded(qv) arrive -> the
        // parity can never alias past the target. (The qk_done-parity form
        // this replaces could wait an even overshoot onto a tile the mma
        // cannot reach without the loader — c7 hang suspect, dv_8a1f111790ec.)
        // Tiny-Ja fallback: QK(Ja-1) retirement (its slot's next completion
        // is likewise loader-gated).
        if (qa >= 2) {
          if (Ja >= NSTAGE + 1) {
            const int Xg = Ja - NSTAGE;
            mbar_wait(&sh.output_ready[Xg % NSTAGE], (Xg / NSTAGE) & 1);
          } else {
            const int Jt = Ja - 1;
            mbar_wait(&sh.qk_done[Jt & 1], (Jt >> 1) & 1);
          }
        }
#ifdef FMHA_WB
        if (ltid == 0) WB(20, (unsigned)qa);
#endif
        for (int idx = ltid; idx < TM * CPR; idx += 96) {
          const int row = idx / CPR;
          const int col = (idx - row * CPR) * 16;
          int rg = A.r0 + row;
          if (rg > A.R - 1) rg = A.R - 1;
          const int t = rg / p.ratio;
          const int h = A.g * p.ratio + (rg - t * p.ratio);
          tc_cp_async16(qb + swz128(row, col),
                        p.q + (int64_t)(A.q0 + t) * p.q_tok_stride
                            + (int64_t)h * DV + col);
        }
        // Arrive throttle: bound pending per-slot arrivals at one (see the
        // mma_qa_p comment). Deadlock-free: wait(qv-2) needed only arrive(qv-2),
        // which this loader issued two iterations ago.
        if (qa >= 2) {
          while (*mma_qa_p < qa - 1) {
          }
        }
        cp_async_mbar_arrive((qa & 1) ? qld_b : &sh.q_loaded);
        if (p.use_tma) {
#pragma unroll 1
          for (int tl = 0; tl < nmt; ++tl) {
            const int Jtl = Ja + tl;
            const int stage = Jtl % NSTAGE;
            const int n0t = A.lo + tl * MTN;
            const int nseg = MTN >> plog;   // page segments per megatile
            const int tab_lo = n0t >> plog;
#ifdef FMHA_WB
            if (ltid == 0) WB(18, (unsigned)Jtl);
#endif
            int pg_reg = 0;
            if (ltid < nseg)
              pg_reg = __ldg(tab + min(tab_lo + ltid, p.Tstride - 1));
            if (Jtl >= NSTAGE) {
              const int wN = Jtl - NSTAGE;
              mbar_wait(&sh.output_ready[wN % NSTAGE], (wN / NSTAGE) & 1);
            }
            if (ltid < nseg) sh.tab_cache[ltid] = pg_reg;
            asm volatile("bar.sync 1, 96;" ::: "memory");
            if (ltid == 0) {
              mbar_arrive_expect_tx(&sh.k_tx[stage], MTN * ROWB);
              mbar_arrive_expect_tx(&sh.v_tx[stage], MTN * ROWB);
            }
            asm volatile("bar.sync 1, 96;" ::: "memory");   // arms visible
            if (ltid < nseg) {
              const int pg = sh.tab_cache[ltid];
              const int seg_b = p.page * ROWB;
              tma_4d(sh.k[stage] + ltid * seg_b, &p.kmap, 0, 0, pg, A.g,
                     &sh.k_tx[stage]);
              tma_4d(sh.v[stage] + ltid * seg_b, &p.vmap, 0, 0, pg, A.g,
                     &sh.v_tx[stage]);
            }
            if (ltid == 0) {
              mbar_arrive(&sh.k_loaded[stage]);
              mbar_arrive(&sh.v_loaded[stage]);
            }
          }
        } else {
#pragma unroll 1
          for (int tl = 0; tl < nmt; ++tl) {
            const int Jtl = Ja + tl;
            const int stage = Jtl % NSTAGE;
            const int n0t = A.lo + tl * MTN;
            int posmax = n0t + MTN - 1;
            if (posmax > A.hi - 1) posmax = A.hi - 1;
            const int tab_lo = n0t >> plog;
            const int nent = (posmax >> plog) - tab_lo + 1;
            int pg_reg = 0;
            if (ltid < nent) pg_reg = __ldg(tab + tab_lo + ltid);
            if (Jtl >= NSTAGE) {
              const int wN = Jtl - NSTAGE;
              mbar_wait(&sh.output_ready[wN % NSTAGE], (wN / NSTAGE) & 1);
            }
            if (ltid < nent) sh.tab_cache[ltid] = pg_reg;
            asm volatile("bar.sync 1, 96;" ::: "memory");
#pragma unroll 4
            for (int idx = ltid; idx < MTN * CPR; idx += 96) {
              const int row = idx / CPR;
              const int col = (idx - row * CPR) * 16;
              int pos = n0t + row;
              if (pos > A.hi - 1) pos = A.hi - 1;
              const int64_t poff = (int64_t)sh.tab_cache[(pos >> plog) - tab_lo];
              const int off = pos & pmask;
              tc_cp_async16(sh.k[stage] + swz128(row, col),
                            kb + poff * p.k_sp + (int64_t)off * p.k_sn + col);
              tc_cp_async16(sh.v[stage] + swz128(row, col),
                            vb + poff * p.v_sp + (int64_t)off * p.v_sn + col);
            }
            cp_async_mbar_arrive(&sh.k_loaded[stage]);
            cp_async_mbar_arrive(&sh.v_loaded[stage]);
          }
        }
        Ja += nmt;
        ++qa;
      }
    } else if (warp == 8) {
      // ------------------------ mma issue (persistent) ----------------------
      setmaxnreg_dec<72>();
      if (lane == 0) {
        const uint32_t id_qk = idesc_f8(TM, MTN, false);
        const uint32_t id_pv = idesc_f8(TM, DV, true);
        int Ja = 0, qa = 0;
        for (;;) {
          int aid;
          if (p.pers_static) {
            aid = (qa < seg_len) ? seg_beg[qa] : -1;   // r19e direct binding
          } else {
            while (*pull_seq < qa + 1) {
            }
            aid = *pull_aid;
          }
          if (aid < 0) break;
          MTAssign A;
          mt_aid_geom<MTN>(p, aid, A);
          const int nmt = A.nmt;
#ifdef FMHA_WB
          WB(5, (unsigned)aid);
#endif
          mbar_wait((qa & 1) ? qld_b : &sh.q_loaded, (qa >> 1) & 1);
          __threadfence_block();   // release: aid read precedes the flag
          *mma_qa_p = qa + 1;   // loader arrive-throttle + pull pacing
#ifdef FMHA_WB
          WB(28, (unsigned)qa);
#endif
          const uint8_t* qb = (qa & 1) ? q2 : (const uint8_t*)sh.q;
#ifdef FMHA_WB
          if (qa == 0) WB(9, (unsigned)clock64());
#endif
#pragma unroll 1
          for (int j = 0; j < nmt + PVLAG; ++j) {
            const int J = Ja + j;
#ifdef FMHA_WB
            WB(4, (unsigned)J);
#endif
            if (j < nmt) {
              const int stage = J % NSTAGE;
              mbar_wait(&sh.k_loaded[stage], (J / NSTAGE) & 1);
              if (p.use_tma) mbar_wait(&sh.k_tx[stage], (J / NSTAGE) & 1);
              if (J >= 1) {
                mbar_wait(&sh.score_released[(J - 1) & 1], ((J - 1) >> 1) & 1);
              }
              {
                uint64_t ad = make_desc(2, 16, 8 * 128, qb);
                uint64_t bd = tma64 ? make_desc(4, 0, 8 * 64, sh.k[stage])
                                    : make_desc(2, 16, 8 * 128, sh.k[stage]);
#pragma unroll
                for (int step = 0; step < KSTEPS; ++step) {
                  umma_f8_ss(tb + S_TMEM, ad, bd, id_qk, step != 0);
                  if (step != KSTEPS - 1) { desc_add(ad, 32); desc_add(bd, 32); }
                }
              }
              umma_commit(&sh.qk_done[J & 1]);
#ifdef FMHA_WB
              if (J == 0) WB(10, (unsigned)clock64());
#endif
            }
            if (j >= PVLAG) {
              const int pv = j - PVLAG;
              const int Jpv = Ja + pv;
              const int pvstage = Jpv % NSTAGE;
              const int pbuf = Jpv % PBUF;
              // O-handoff gate: assignment >0 may not touch O before the
              // previous assignment's epilogue finished READING it.
              if (pv == 0 && qa >= 1) mbar_wait(o_drained, (qa - 1) & 1);
              mbar_wait(&sh.v_loaded[pvstage], (Jpv / NSTAGE) & 1);
              if (p.use_tma) mbar_wait(&sh.v_tx[pvstage], (Jpv / NSTAGE) & 1);
              mbar_wait(&sh.probability_ready[pbuf], (Jpv / PBUF) & 1);
              if (pv >= 1) {
                mbar_wait(&sh.output_rescaled[pbuf], (Jpv / PBUF) & 1);
              }
              {
                const uint32_t pa = tb + P_TMEM + pbuf * PB;
                uint64_t vd = tma64 ? make_desc(4, 0, 8 * 64, sh.v[pvstage])
                                    : make_desc(2, 0, 8 * 128, sh.v[pvstage]);
#pragma unroll
                for (int step = 0; step < PVSTEPS; ++step) {
                  umma_f8_ts(tb + O_TMEM_MT, pa + step * 8, vd, id_pv,
                             pv != 0 || step != 0);
                  if (step != PVSTEPS - 1) desc_add(vd, tma64 ? 32 * 64 : 32 * 128);
                }
              }
              umma_commit(&sh.output_ready[Jpv % NSTAGE]);
              if (pv == nmt - 1) umma_commit(&sh.pv_last);
            }
          }
          Ja += nmt;
          ++qa;
        }
#ifdef FMHA_WB
        WB(6, (unsigned)Ja);
        WB(11, (unsigned)clock64());
#endif
      }
    } else {
      // ---------------------- softmax groups (persistent) -------------------
      setmaxnreg_inc<216>();
      const int grp = warp >> 2;
      const int r = (warp & 3) * 32 + lane;
      const uint32_t st_lane = ((uint32_t)(warp & 3) * 32) << 16;
      const uint32_t oaddr = tb + O_TMEM_MT + st_lane;
      const float qs = p.qk_scale;
      const uint32_t qs2 = f16x2_dup(qs);
      int Ja = 0, qa = 0;
      for (;;) {
        int aid;
        if (p.pers_static) {
          aid = (qa < seg_len) ? seg_beg[qa] : -1;   // r19e direct binding
        } else {
          while (*pull_seq < qa + 1) {
          }
          aid = *pull_aid;
          // Account this warp's aid READ to the puller's pacing counter
          // (8 warps per assignment; the puller overwrites pull_aid only
          // once sm_read_ctr >= 8*next and mma_qa_p >= next).
          __syncwarp(0xffffffffu);
          __threadfence_block();   // release: aid read precedes the counter
          if (lane == 0) atomicAdd((int*)sm_read_ctr, 1);
        }
        if (aid < 0) break;
        MTAssign A;
        mt_aid_geom<MTN>(p, aid, A);
        const int nmt = A.nmt;
        const int rg = A.r0 + r;
        const int tok = rg / p.ratio;
        const int vis_i = A.kvlen - A.qlen + tok;
        const bool row_ok = (rg < A.R);
        float m2r = -INFINITY;   // per-assignment online-softmax state
        float dsum = 0.f;
        // r19j: the r19g group-0 o_drained entry gate is REMOVED. It ordered
        // the (real, narrow) group-0-next-publish vs group-1-state-read
        // window structurally, but cost ~2us per boundary — measured as the
        // whole c7-static win (audit#2 106.5→112.6; eval c7 0.9811 = dyn
        // class instead of the 0.942-0.961 static class). The c7 corruption
        // it was added for turned out to be the BFD tiny-chain regime
        // (r19h fingerprint), quarantined at the PARTITION level (r19i) —
        // independent of this gate. Without the gate the state-read window
        // keeps its r18j/r19c/r19e-c4 empirical ~1-2us cadence margin
        // (publish needs qk_done(Ja_next) = drain + QK issue/retire + S
        // read; the read follows probability(Jl) by ~instructions; clean
        // across r18j LPT-static, r19c/e static and 28/28 r19i runs).
        // Documented fallback if it ever re-arms: a dedicated state_free
        // barrier (group 1 arrives right after its epilogue state read,
        // BEFORE pv_last/O-read; group 0 waits it at entry) = structural
        // ordering at ~zero cost. Removed-by: r19j, validated by r19j probe
        // (c7/c4/c6 static rowdiff x20) + the promotion eval re-run.
#ifdef FMHA_WB
        if (warp == 0 && lane == 0) WB(22, (unsigned)aid);
#endif
#pragma unroll 1
        for (int m = grp; m < nmt; m += 2) {
          const int Jm = Ja + m;
          const int sbuf = Jm & 1;
          const int pbuf = Jm % PBUF;
#ifdef FMHA_WB
          if (lane == 0) {
            if (warp == 0) WB(21, (unsigned)Jm);
            if (warp == 4) WB(25, (unsigned)Jm);
          }
#endif
          mbar_wait(&sh.qk_done[sbuf], (Jm >> 1) & 1);
          const int n0t = A.lo + m * MTN;
          int c_lo = 0, c_hi = MTN;
          if (!row_ok) {
            c_hi = 0;
          } else {
            int ub = A.hi;
            if (p.causal) ub = min(ub, vis_i + 1);
            c_hi = min(MTN, ub - n0t);
            if (p.window_left >= 0) c_lo = max(0, vis_i - p.window_left - n0t);
            if (c_hi < c_lo) c_hi = c_lo;
          }
          const unsigned span = (unsigned)(c_hi - c_lo);
          const int span0 = max(0, min(c_hi, 128) - min(c_lo, 128));
          const int span1 = max(0, (c_hi - 128) - max(c_lo - 128, 0));
          uint32_t sc[128];
          uint32_t s16a[64];
          uint32_t mx0 = 0xfc00fc00u;   // f16x2(-inf, -inf)
          tmem_ld32(sc, tb + S_TMEM);
#pragma unroll
          for (int seg = 1; seg < 4; ++seg)
            tmem_ld32(sc + seg * 32, tb + S_TMEM + seg * 32);
          tmem_ld_wait();
          if (span0 >= 128) {
#pragma unroll
            for (int w = 0; w < 64; ++w) {
              s16a[w] = cvt_f16x2_sat(__uint_as_float(sc[2 * w]),
                                      __uint_as_float(sc[2 * w + 1]));
              mx0 = hmax2_u32(mx0, s16a[w]);
            }
          } else {
#pragma unroll
            for (int w = 0; w < 64; ++w) {
              const bool in0 = (unsigned)(2 * w - c_lo) < span;
              const bool in1 = (unsigned)(2 * w + 1 - c_lo) < span;
              const float f0v = in0 ? __uint_as_float(sc[2 * w]) : -INFINITY;
              const float f1v = in1 ? __uint_as_float(sc[2 * w + 1]) : -INFINITY;
              s16a[w] = cvt_f16x2_sat(f0v, f1v);
              mx0 = hmax2_u32(mx0, s16a[w]);
            }
          }
          uint32_t s16b[H1W];
          uint32_t mx1 = 0xfc00fc00u;
#pragma unroll
          for (int seg = 0; seg < H1S; ++seg)
            tmem_ld32(sc + seg * 32, tb + S_TMEM + 128 + seg * 32);
          tmem_ld_wait();
          // S fully register-resident -> release for the next QK. Warp sync
          // first: lane0's arrive must imply EVERY lane's loads completed.
          __syncwarp(0xffffffffu);
          if (lane == 0) mbar_arrive(&sh.score_released[sbuf]);
          if (span1 >= MTN - 128) {
#pragma unroll
            for (int w = 0; w < H1W; ++w) {
              s16b[w] = cvt_f16x2_sat(__uint_as_float(sc[2 * w]),
                                      __uint_as_float(sc[2 * w + 1]));
              mx1 = hmax2_u32(mx1, s16b[w]);
            }
          } else {
#pragma unroll
            for (int w = 0; w < H1W; ++w) {
              const bool in0 = (unsigned)(128 + 2 * w - c_lo) < span;
              const bool in1 = (unsigned)(128 + 2 * w + 1 - c_lo) < span;
              const float f0v = in0 ? __uint_as_float(sc[2 * w]) : -INFINITY;
              const float f1v = in1 ? __uint_as_float(sc[2 * w + 1]) : -INFINITY;
              s16b[w] = cvt_f16x2_sat(f0v, f1v);
              mx1 = hmax2_u32(mx1, s16b[w]);
            }
          }
          // ---- state handoff (LOCAL m>=1 gate: fresh state per assignment;
          // GLOBAL Jm-1 slot/parity: the publisher's arrive count is global) ----
          if (m >= 1) {
            mbar_wait(&sh.probability_ready[(Jm - 1) % PBUF],
                      ((Jm - 1) / PBUF) & 1);
            m2r = sh.state_m2[r];
            dsum = sh.state_d[r];
          }
          const float m2r_in = m2r;
          const float tmaxr0 = (span0 > 0) ? f16x2_lo(hmax_halves2(mx0)) : -INFINITY;
          const float tmaxr1 = (span1 > 0) ? f16x2_lo(hmax_halves2(mx1)) : -INFINITY;
          const bool rebase0 =
              p.force_corr || (((tmaxr0 - m2r_in) * qs) > REBASE_MARGIN);
          const float mnr0 = rebase0 ? tmaxr0 : m2r_in;
          const float mbase0 = (mnr0 == -INFINITY) ? 0.f : mnr0;
          const uint32_t nm2a = f16x2_dup(PLOG2 - mbase0 * qs);
          uint32_t pab0[64];
          float psum0;
          {
            uint32_t ps2[4] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (int w = 0; w < 64; ++w) {
              const uint32_t pa = ex2_f16x2(hfma2_u32(s16a[w], qs2, nm2a));
              pab0[w] = pa;
              ps2[w & 3] = hadd2_u32(ps2[w & 3], pa);
            }
            const uint32_t pst0 = hadd2_u32(hadd2_u32(ps2[0], ps2[1]),
                                            hadd2_u32(ps2[2], ps2[3]));
            psum0 = f16x2_lo(pst0) + f16x2_hi(pst0);
          }
          const bool rebase1 =
              p.force_corr || (((tmaxr1 - mnr0) * qs) > REBASE_MARGIN);
          const float mnr1 = rebase1 ? tmaxr1 : mnr0;
          if (__any_sync(0xffffffffu, rebase1)) {
            const float f0 = (mnr0 == -INFINITY && mnr1 == -INFINITY)
                                 ? 1.f
                                 : tc_ex2((mnr0 - mnr1) * qs);
            const uint32_t f02 = f16x2_dup(f0);
#pragma unroll
            for (int w = 0; w < 64; ++w) pab0[w] = hmul2_u32(pab0[w], f02);
            psum0 *= f0;
          }
          const float mbase1 = (mnr1 == -INFINITY) ? 0.f : mnr1;
          const uint32_t nm2b = f16x2_dup(PLOG2 - mbase1 * qs);
          uint32_t pab1[H1W];
          float psum1;
          {
            uint32_t ps2b[4] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (int w = 0; w < H1W; ++w) {
              const uint32_t pa = ex2_f16x2(hfma2_u32(s16b[w], qs2, nm2b));
              pab1[w] = pa;
              ps2b[w & 3] = hadd2_u32(ps2b[w & 3], pa);
            }
            const uint32_t pst1 = hadd2_u32(hadd2_u32(ps2b[0], ps2b[1]),
                                            hadd2_u32(ps2b[2], ps2b[3]));
            psum1 = f16x2_lo(pst1) + f16x2_hi(pst1);
          }
          const bool rebased = rebase0 || rebase1;
          const float accs = rebased ? tc_ex2((m2r_in - mnr1) * qs) : 1.0f;
          dsum = dsum * accs + psum0 + psum1;
          m2r = mnr1;
          uint32_t pw[32];
#pragma unroll
          for (int w = 0; w < 32; ++w)
            pw[w] = pack_e4m3x4(pab0[2 * w], pab0[2 * w + 1]);
          tmem_st32(tb + P_TMEM + pbuf * PB + st_lane, pw);
#pragma unroll
          for (int w = 0; w < H1W / 2; ++w)
            pw[w] = pack_e4m3x4(pab1[2 * w], pab1[2 * w + 1]);
          if constexpr (H1W == 32)
            tmem_st16(tb + P_TMEM + pbuf * PB + 32 + st_lane, pw);
          else
            tmem_st32(tb + P_TMEM + pbuf * PB + 32 + st_lane, pw);
          tmem_st_wait();
          // Publish state BEFORE the probability_ready arrive (release/acquire,
          // cumulative via __syncwarp — v8f race-fix discipline).
          sh.state_m2[r] = m2r;
          sh.state_d[r] = dsum;
          __syncwarp(0xffffffffu);
          if (lane == 0) mbar_arrive(&sh.probability_ready[pbuf]);
#ifdef FMHA_WB
          if (warp == 0 && lane == 0) WB(26, (unsigned)Jm);
#endif
          // ---- merged O correction (LOCAL m>=1 gate, GLOBAL Jm-1 slot) ----
          if (m >= 1) {
            const bool do_corr = __ballot_sync(0xffffffffu, rebased);
            mbar_wait(&sh.output_ready[(Jm - 1) % NSTAGE],
                      ((Jm - 1) / NSTAGE) & 1);
            if (do_corr) {
              constexpr int NB = DV / 32;             // 2 (D64), 4 (D128)
              constexpr int BB = NB < 4 ? NB : 4;
              uint32_t vals[BB * 32];
#pragma unroll
              for (int b0 = 0; b0 < NB; b0 += BB) {
#pragma unroll
                for (int jj = 0; jj < BB; ++jj)
                  tmem_ld32(vals + jj * 32, oaddr + (b0 + jj) * 32);
                tmem_ld_wait();
#pragma unroll
                for (int i = 0; i < BB * 32; ++i)
                  vals[i] = __float_as_uint(__uint_as_float(vals[i]) * accs);
#pragma unroll
                for (int jj = 0; jj < BB; ++jj)
                  tmem_st32(oaddr + (b0 + jj) * 32, vals + jj * 32);
                tmem_st_wait();
              }
            }
            __syncwarp(0xffffffffu);
          }
          if (lane == 0) mbar_arrive(&sh.output_rescaled[pbuf]);
        }
        if constexpr (DV == 64) {
          // ---- split epilogue (D64 arm; r18d measured c4 min 0.973->0.950;
          // r18e measured D128 regressions +1.2..+1.9% -> D128 keeps the
          // single-owner form below) ----
          // O rows belong to the reading warp's (warp&3) lane slice either
          // way, so the groups split the COLUMNS: owner takes the lower
          // half of the 32-col batches, non-owner the upper half. The final
          // state (m2r/dsum) is in sh.state_m2/state_d — the owner's last-
          // tile publish precedes its probability_ready arrive, which the
          // non-owner waits before reading — so its inv/base2 recompute is
          // BIT-IDENTICAL to the owner's register values. lse stays owner-
          // written (one writer per row). Halves the per-boundary epilogue
          // serialization (dominant PERS boundary-stall term) and halves
          // the vals[] live array. o_drained count 8 (all softmax warps).
          const bool owner = (grp == ((nmt - 1) & 1));
          float m2r_e = m2r;
          float dsum_e = dsum;
          if (!owner) {
            const int Jl = Ja + nmt - 1;   // global index of the last tile
            mbar_wait(&sh.probability_ready[Jl % PBUF], (Jl / PBUF) & 1);
            m2r_e = sh.state_m2[r];
            dsum_e = sh.state_d[r];
          }
#ifdef FMHA_WB
          if (lane == 0) WB(23, (unsigned)qa);
#endif
          mbar_wait(&sh.pv_last, qa & 1);
#ifdef FMHA_WB
          if (lane == 0) WB(24, (unsigned)qa);
#endif
          const int t = rg / p.ratio;
          const int h = A.g * p.ratio + (rg - t * p.ratio);
          float base2 = 0.f;
          float inv = 0.f;
          if (row_ok) {
            base2 = m2r_e * p.qk_scale + log2f(dsum_e) - PLOG2;
            inv = p.bmm2 / dsum_e;
            if (p.has_sink) {
              const float sk = p.sinks[h] * 1.4426950408889634f;
              const float tot = fmaxf(base2, sk)
                                + log2f(1.f + exp2f(-fabsf(base2 - sk)));
              inv *= exp2f(base2 - tot);
              base2 = tot;
            }
            if (owner && p.has_lse)
              p.lse[(int64_t)(A.q0 + t) * p.H + h] = base2;
          }
          uint16_t* orow =
              p.out + ((int64_t)(A.q0 + min(t, A.qlen - 1)) * p.H
                       + min(h, p.H - 1)) * DV;
          const bool wide = ((reinterpret_cast<uintptr_t>(p.out) & 15u) == 0u);
          constexpr int NBo = DV / 32;   // 2 (D64) or 4 (D128)
          constexpr int NBh = NBo / 2;   // batches per group (1 or 2)
          const int bstart = owner ? 0 : NBh;
          uint32_t vals[NBh * 32];
#pragma unroll
          for (int jj = 0; jj < NBh; ++jj)
            tmem_ld32(vals + jj * 32, oaddr + (bstart + jj) * 32);
          tmem_ld_wait();
          // O reads complete -> release the tmem region to the next
          // assignment's PV(pv==0) overwrite; stores overlap onward.
          if (lane == 0) mbar_arrive(o_drained);
          if (row_ok) {
#pragma unroll
            for (int jj = 0; jj < NBh; ++jj) {
              if (wide) {
#pragma unroll
                for (int pr = 0; pr < 16; pr += 4) {
                  uint4 pk4;
                  uint32_t* w4 = reinterpret_cast<uint32_t*>(&pk4);
#pragma unroll
                  for (int k = 0; k < 4; ++k) {
                    const float x0 =
                        __uint_as_float(vals[jj * 32 + 2 * (pr + k)]) * inv;
                    const float x1 =
                        __uint_as_float(vals[jj * 32 + 2 * (pr + k) + 1]) * inv;
                    const __nv_bfloat162 pk =
                        __float22bfloat162_rn(make_float2(x0, x1));
                    w4[k] = *reinterpret_cast<const uint32_t*>(&pk);
                  }
                  *reinterpret_cast<uint4*>(orow + (bstart + jj) * 32
                                            + 2 * pr) = pk4;
                }
              } else {
#pragma unroll
                for (int pr = 0; pr < 16; ++pr) {
                  const float x0 = __uint_as_float(vals[jj * 32 + 2 * pr]) * inv;
                  const float x1 =
                      __uint_as_float(vals[jj * 32 + 2 * pr + 1]) * inv;
                  __nv_bfloat162 pk =
                      __float22bfloat162_rn(make_float2(x0, x1));
                  *reinterpret_cast<uint32_t*>(orow + (bstart + jj) * 32
                                               + 2 * pr) =
                      *reinterpret_cast<uint32_t*>(&pk);
                }
              }
            }
          }
        } else {
          // ---- always-group-1 epilogue (D128 arm, r18j) ----
          // The nested drain decouples pv_last from the QK gate (no r18g
          // cycle), but the owner-group epilogue still delays that group's
          // next-assignment m=0 — and for ODD nmt the owner IS group 0, the
          // m=0 group — stalling the score_released -> QK chain by the whole
          // O drain (~1.7us of the ~2.6us boundary residual). Pinning the
          // epilogue to group 1 frees group 0; when group 1 did not own the
          // last tile (odd nmt) it takes the final state from
          // state_m2/state_d behind probability_ready(J_last) — published
          // before the owner's arrive — so inv/base2 are BIT-IDENTICAL to
          // the owner-register path (mechanism proven in r18d/r18e).
          if (grp == 1) {
            const bool owner = ((nmt & 1) == 0);   // even nmt -> group1 owned last
            float m2r_e = m2r;
            float dsum_e = dsum;
            if (!owner) {
              const int Jl = Ja + nmt - 1;   // global index of the last tile
              mbar_wait(&sh.probability_ready[Jl % PBUF], (Jl / PBUF) & 1);
              m2r_e = sh.state_m2[r];
              dsum_e = sh.state_d[r];
            }
#ifdef FMHA_WB
            if (lane == 0) WB(23, (unsigned)qa);
#endif
            mbar_wait(&sh.pv_last, qa & 1);
#ifdef FMHA_WB
            if (lane == 0) WB(24, (unsigned)qa);
#endif
            const int t = rg / p.ratio;
            const int h = A.g * p.ratio + (rg - t * p.ratio);
            float base2 = 0.f;
            float inv = 0.f;
            if (row_ok) {
              base2 = m2r_e * p.qk_scale + log2f(dsum_e) - PLOG2;
              inv = p.bmm2 / dsum_e;
              if (p.has_sink) {
                const float sk = p.sinks[h] * 1.4426950408889634f;
                const float tot = fmaxf(base2, sk)
                                  + log2f(1.f + exp2f(-fabsf(base2 - sk)));
                inv *= exp2f(base2 - tot);
                base2 = tot;
              }
              if (p.has_lse) p.lse[(int64_t)(A.q0 + t) * p.H + h] = base2;
            }
            uint16_t* orow =
                p.out + ((int64_t)(A.q0 + min(t, A.qlen - 1)) * p.H
                         + min(h, p.H - 1)) * DV;
            const bool wide = ((reinterpret_cast<uintptr_t>(p.out) & 15u) == 0u);
            constexpr int NBo = DV / 32;
            constexpr int BBo = NBo < 4 ? NBo : 4;
#pragma unroll
            for (int b0 = 0; b0 < NBo; b0 += BBo) {
              uint32_t vals[BBo * 32];
#pragma unroll
              for (int jj = 0; jj < BBo; ++jj)
                tmem_ld32(vals + jj * 32, oaddr + (b0 + jj) * 32);
              tmem_ld_wait();
              // O reads done -> release tmem to the next assignment's
              // PV(pv==0) overwrite; the global stores overlap onward.
              if (b0 + BBo >= NBo && lane == 0) mbar_arrive(o_drained);
              if (row_ok) {
                if (wide) {
#pragma unroll
                  for (int pr = 0; pr < BBo * 16; pr += 4) {
                    uint4 pk4;
                    uint32_t* w4 = reinterpret_cast<uint32_t*>(&pk4);
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                      const float x0 =
                          __uint_as_float(vals[2 * (pr + k)]) * inv;
                      const float x1 =
                          __uint_as_float(vals[2 * (pr + k) + 1]) * inv;
                      const __nv_bfloat162 pk =
                          __float22bfloat162_rn(make_float2(x0, x1));
                      w4[k] = *reinterpret_cast<const uint32_t*>(&pk);
                    }
                    *reinterpret_cast<uint4*>(orow + b0 * 32 + 2 * pr) = pk4;
                  }
                } else {
#pragma unroll
                  for (int pr = 0; pr < BBo * 16; ++pr) {
                    const float x0 = __uint_as_float(vals[2 * pr]) * inv;
                    const float x1 = __uint_as_float(vals[2 * pr + 1]) * inv;
                    __nv_bfloat162 pk =
                        __float22bfloat162_rn(make_float2(x0, x1));
                    *reinterpret_cast<uint32_t*>(orow + b0 * 32 + 2 * pr) =
                        *reinterpret_cast<uint32_t*>(&pk);
                  }
                }
              }
            }
          }
        }
        Ja += nmt;
        ++qa;
      }
    }

    TCW(37 + warp, 1);
  if (p.exit_mode != 2) {
    __syncthreads();
    TCW(50 + warp, 1);
#ifdef FMHA_WB
    if (warp == 9 && lane == 0) WB(12, (unsigned)clock64());
    if (tid == 0) {
      WB(13, (unsigned)clock64());
      WB(14, (unsigned)wb_gt());
    }
#endif
  }
  if (warp == 8) {
    if (p.exit_mode == 0) {
      tmem_dealloc(tb, TMEM_COLS);
      TCW(62, 1);
    } else {
      TCW(61, 1);
    }
  }
  }   // if constexpr (PERS)
}

}  // namespace fmtcmt
