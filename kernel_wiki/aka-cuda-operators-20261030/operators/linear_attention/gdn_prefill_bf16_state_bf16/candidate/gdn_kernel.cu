// GDN (gated delta rule) prefill - independent CUDA C++/PTX implementation for L20D (sm_103).
// See gdn_common.cuh header block for the math summary; precision notes are there too.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#ifdef GDN_HOSTPROF
#include <chrono>
#include <vector>
#endif
#include <optional>
#include <unordered_map>
#include <utility>

#include "gdn_common.cuh"

namespace gdn {
#ifdef GDN_HOSTPROF
// V-J37 host-preamble census (HOST-ONLY build flag; experiments/
// host_preamble_census.py): 5 steady_clock stamps -> 4 intervals:
// [0->1] tensor prep/checks/ws/maps/dispatch, [1->2] P1 launch + error
// check, [2->3] chain launch(es) + check, [3->4] return construction.
// Python-side total minus (hp4-hp0) approximates pybind arg/return
// conversion. Production builds compile WITHOUT the flag: zero impact.
thread_local long long g_hp[5];
static inline long long hp_now() {
  return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
#define HP(i) (g_hp[i] = hp_now())
#else
#define HP(i) ((void)0)
#endif

// ============================ P1: prep kernel (V2d) =====================
// GQA=true: grid (slots_ub, H); block loads k/q ONCE per qk head, computes
// gram/qk once, then loops over the ratio value heads of the group (wins when
// slots*H still fills the machine). GQA=false: grid (slots_ub, HV), one value
// head per block (lower latency for small workloads).
// Per head: b cumsum, system build, unit-lower-triangular inverse (8-level
// fp32 SIMT register forward substitution; 16/32/64-level off-diagonals via
// single-bf16 mma on the working planes — E1 converted lvl16/32 from SIMT to
// the lvl64 idiom), store Tinv hi/lo planes + b + QKd hi/lo.
// V3f: RATIO>0 => cnt==RATIO compile-time; ALL g/beta head loads are
// prefetched in the preamble (one cp.async group per head) so the per-hv
// cp_wait immediates are exact and the wait exposure disappears (measured
// 0.3-0.4us/hv in the cumsum phase). RATIO==0 keeps the V2m double-buffer
// prefetch for unexpected ratios.
template <bool GQA, int RATIO, bool QHFAST>
__global__ void __launch_bounds__(NTHR, 2)
gdn_prep(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
         const float* __restrict__ g, const float* __restrict__ beta,
         const int* __restrict__ cu, const int* __restrict__ chunk_indices,
         const int* __restrict__ chunk_offsets, int N, int H, int HV,
         char* __restrict__ scratch, char* __restrict__ scratchq,
         int* __restrict__ perm, const int skip_q, const int ci_rows,
         const __grid_constant__ CUtensorMap kmap,   // V-J2 TMA prologue
         const __grid_constant__ CUtensorMap qmap
#ifdef GDN_P1PHASE
         , long long* __restrict__ phases
#endif
         ) {
#ifdef GDN_P1PHASE
  // phases[0]=loads/gram/qk/sKQ-store shared preamble; per-hv accumulations:
  // [1]=cumsum [2]=system [3]=qkd [4]=lvl8 [5]=lvl16 [6]=lvl32 [7]=lvl64
  // [8]=publish [9]=hv iters; written by block(0,0) thread 0.
  long long ph[16] = {};
  long long _pt = clock64();
#define TICK(i) do { long long _n = clock64(); ph[i] += _n - _pt; _pt = _n; } while (0)
  // V-J24 preamble sub-phase census (GDN_P1PHASE ONLY; experiments/
  // p1_dual_solo_phases.py): ph[10]=setup+box/gb issue, ph[11]=mbar box
  // wait, ph[12]=cp g/beta wait + join sync, ph[13]=gram/qk compute block
  // (on tid128 = the rho branch + convergence-sync wait), ph[14]=sKQ bulk
  // store issue region. ph[0] keeps its role (Region-2 residue ~0 after
  // the sub-phase ticks; old preamble = ph[0]+ph[10..14]).
  // Buffer layout v2: [0..9] block0 ph vec | [10..19] solo-probe ph vec |
  // [20..23] meta | [24..29] block0 tid0 sub-phases | [30..35] solo tid0
  // sub-phases | [36..41] block0 tid128 (rho warp) sub-phases | [42..47]
  // solo tid128 sub-phases | TAIL_BASE=64: {t_start,t_end,smid,lin+1} per
  // block. Harnesses MUST size >= 64 + 4*32*288 elements.
  // V-J16 tail census (GDN_P1PHASE builds ONLY; experiments/
  // p1_tail_census.py): per-block wall timeline — %globaltimer (ns,
  // GPU-global, cross-SM comparable) at entry/exit + %smid + linear id.
  // Layout: phases[0..9] = block(0,0) phase accumulations (unchanged);
  // phases[32 + lin*4 + {0,1,2,3}] = {t_start, t_end, smid, lin+1} for
  // EVERY block incl. slot>=total early exits, lin = x + gridDim.x*y.
  // V-J24-fix: tail base moved 32 -> 64 (layout v2: [24..47] are the
  // subphase vectors; the first census run aliased lin 0-4 tails onto the
  // solo-probe slots). Harness MUST size the buffer >= 64 + 4*GX*slots_ub
  // entries (shape_range bound: 32*288 blocks => 36,928 int64). Overhead ~2 SFR reads + one 32B
  // store per block — census builds only; production compiles WITHOUT the
  // flag so the production binary is untouched.
  long long _ts = 0;
  unsigned _smid = 0;
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_ts));
    asm volatile("mov.u32 %0, %%smid;" : "=r"(_smid));
  }
  // V-J17a dual/solo phase probe (GDN_P1PHASE ONLY; experiments/
  // p1_dual_solo_phases.py): block(0,0) is DUAL-class whenever n_blocks >
  // 148 SMs (SM0 also hosts lin=148 — verified exactly by the V-J16 smid
  // census: dual pairs are (i, i+148), ratio 1.29-1.35). phases[10..19] =
  // the same tick vector recorded by a SOLO probe block lin==P. P is the
  // middle of the ACTIVE range: na = total*(QHFAST ? gx : gy) active lins
  // under the round-robin lin->SM mapping; P = na/2 if na <= 148 (all-solo)
  // else (na-1)/2 — which satisfies P >= n-148 (solo: partner P+148 >= n)
  // AND P < na (active) for every fixed-suite geometry (trivial tail is
  // small: na >= 2n-295 holds with margin). phases[20]=P, [21]=n_blocks,
  // [22]=na, [23]=148 (hardcoded SM count — L20D census build only).
  // For n > 296 (multi-wave: c0/c6) "solo" is unreliable — reference only.
  const long long _nblk = (long long)gridDim.x * gridDim.y;
  const long long _lin0 = (long long)blockIdx.x +
      (long long)gridDim.x * blockIdx.y;
  const long long _na = (long long)chunk_offsets[N] *
      (QHFAST ? (long long)gridDim.x : (long long)gridDim.y);
  const long long _P = (_na <= 148) ? (_na / 2) : ((_na - 1) / 2);
#define TAIL_REC() do { if (phases != nullptr && threadIdx.x == 0) {           \
    long long _te;                                                             \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_te));                    \
    const long long _lin = (long long)blockIdx.x +                             \
        (long long)gridDim.x * blockIdx.y;                                     \
    long long* _tb = phases + 64 + _lin * 4;                                   \
    _tb[0] = _ts; _tb[1] = _te; _tb[2] = (long long)_smid; _tb[3] = _lin + 1;  \
  } } while (0)
#else
#define TICK(i) do {} while (0)
#endif
  extern __shared__ char smem_raw[];
  // V-J2 (arm-a V5.35a-fix port): 1024B-aligned base — MANDATORY, not
  // cosmetic: TMA SWIZZLE bakes the WRITER's dst-address phase
  // ((base>>7)+row)&7 into the bytes; the s2g image is read back by the
  // chain at ITS stage base. Bases equal mod 1024 (phase 0) on both sides =>
  // image path phase-transparent for every instantiation pair (arm-a bench
  // T4 proved mismatched bases corrupt; T1-3 same-base correctness at any
  // phase). Offset-form keeps addrspace(3) provenance (V5.15a).
  char* smem = smem_raw +
      (int)((1024u - (reinterpret_cast<uintptr_t>(smem_raw) & 1023u)) & 1023u);
  __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);   // kT0 kT1 (16,384B)
  __nv_bfloat16* sQ = sK + 8192;                                // qT0 qT1 (16,384B)
  // V25: INTERNAL working planes are bf16 (were fp32). v26prof: P1 is
  // L1/TEX-pipe-bound (74-76%) with the fp32 gram/sys/tinv planes + the
  // lvl64 hi/lo overlay conversions dominating its L1 bytes; occupancy is
  // register-capped at 2 CTAs/SM (85-reg cap at 3x256 vs 114 actual), so
  // traffic reduction is the ONLY P1 lever. The PUBLISHED interface is
  // unchanged: slots already carry bf16 hi-only TinvH/QkdH (V3d-1/V3e) -
  // chain untouched. Numerics: ~7 extra 2^-9 roundings along the Tinv path
  // (~0.5-1% rel) vs frozen rtol 0.05 (5-10x headroom); mma accum stays
  // fp32; sYf region is used as the bf16 sYb Y plane at every level (E1:
  // lvl16/32 joined the lvl64 single-bf16-Y mma class; the fp32 sYf form
  // is gone).
  __nv_bfloat16* sGram = reinterpret_cast<__nv_bfloat16*>(sQ + 8192);  // [64][PL_STR] bf16
  // V-J44: group-of-2 head-lockstep merge (fused RT2/RT3) runs two heads'
  // sys/Tinv/scratch sets concurrently through the merge spine (each phase
  // was 25-50% thread-idle serial-per-head; heads are independent). P1NS=1
  // for every other ratio => offsets byte-identical to the serial layout.
  constexpr int P1NS = (RATIO == 2 || RATIO == 3 || RATIO == 8) ? 2 : 1;   // V-J44x
  __nv_bfloat16* sSys = sGram + CDIM * PL_STR;                    // P1NS x [64][PL_STR]
  __nv_bfloat16* sTinv = sSys + P1NS * CDIM * PL_STR;             // P1NS x [64][PL_STR]
  // V26: per-head rho slots (was single [64]). cumsum(b)+rho for ALL heads
  // are precomputed on the 2 warps idle during gram/qk (tile_active covers
  // exactly 6/8 warps: idle = (mw<2, nw=1) = tid 128-191) for RATIO>0,
  // removing cnt*(0.58us serial scan + 4 full-CTA barriers) from the head
  // loop. Named barrier bar.sync 1 (64 thr) replaces the internal syncs.
  // Bitwise: identical fp32 op order (32-lane shfl scan + one cross-warp
  // correction + same expf). Buffer region 17->24 x CDIM x 4 (host matched).
  constexpr int SBC = (RATIO > 0) ? RATIO : 2;   // g/beta/rho buffer count
  float* sb0 = reinterpret_cast<float*>(sTinv + P1NS * CDIM * PL_STR);   // SBC x [64] buffers
  float* sbeta0 = sb0 + SBC * CDIM;
  float* sRho = sbeta0 + SBC * CDIM;            // SBC x [64] V2q decay ratios
  float* sYf = sRho + SBC * CDIM;               // P1NS x [32][40] fp32 scratch
  // V25: lvl64 Y plane reuses the sYf scratch as bf16 [32][40] (lvl16/32
  // sYf reads all complete before lvl64, barrier-separated). The 8 hi/lo
  // overlay planes on sK/sQ are GONE (direct ldm from the bf16 working
  // planes), and with them the pre-lvl64 bulk wait_group.read (nothing
  // writes over sK/sQ anymore; grid drain covers s2g visibility).
  __nv_bfloat16* sYb = reinterpret_cast<__nv_bfloat16*>(sYf);
  // V-J2: TMA-prologue mbar (after sYf; all offsets above are 128B
  // multiples from the aligned base => 8B-safe).
  uint64_t* p1mbar = reinterpret_cast<uint64_t*>(sYf + P1NS * 32 * 40);

  // V19 (= V13 port), V19.1 SELECTIVE: QHFAST grid (GX, slots_ub) vs
  // slot-fastest (slots_ub, GX), chosen by the host. Under
  // TIMING_PROTOCOL_V2 every forward streams k/q cold from DRAM;
  // slot-fastest made the ~296 concurrent CTAs read 256B column slices of
  // DIFFERENT 2-4KB rows (row-buffer thrash), while qh-fastest lets the
  // H/HV CTAs of one slot cooperatively sweep contiguous row blocks.
  // Measured on V18.1: qh-fastest wins replicated c6 -2.05..-2.74us
  // (4 samples across 2 harnesses + old-tree 3x), c0 -0.93..-2.02, c1
  // sentinel flat — BUT official same-job eval (dv_7ddca0e52748) showed
  // c8 (ONLY segmented fixed case, tp8 N=1 T=8192, G>1) +1.86us (+2.04%):
  // its 30-sample window is bimodal (fast ~91.2 / slow ~93.2 bands; the
  // slow mode pre-exists at ~23% on V18.1) and qh-fastest SHIFTS THE MODE
  // MIXTURE to ~60% — the 3-kernel segmented window (P1 -> pass A ->
  // pass C) interacts with P1 completion ordering. V19.1 therefore keeps
  // qh-fastest ONLY for Gseg==1 shapes (proven wins) and reverts
  // segmented shapes to the certified V18.1 slot-fastest behavior.
  // Per-CTA work identical under both orders => outputs bitwise identical
  // either way. Audits: all 3 P1 blockIdx sites keyed on QHFAST (the two
  // (0,0) guards are symmetric and stay); slots_ub <= 288 fits gridDim.y,
  // GX <= 32 fits gridDim.x under both orders; LPT/c0 hot-set unchanged
  // (linear-id tail = highest slots under BOTH orders); chain/l2norm
  // grids untouched.
  const int slot = QHFAST ? (int)blockIdx.y : (int)blockIdx.x;
  // V-J45: one-round parallel mapping prefetch. The old path was a
  // 3-deep dependent cold LDG chain (chunk_offsets[N] -> chunk_indices[slot]
  // -> cu[n],cu[n+1]) = region-15 1.22us cold (CLASSD/FCOMP censuses),
  // serial before the KQ box issue on EVERY P1 CTA. co[N] and the full
  // cu[0..N] (N<=32 shape_range-guaranteed => <=33 ints, ~3 coalesced
  // sectors) are mutually independent: ONE parallel round staged to static
  // smem, published by the mbar-init __syncthreads HOISTED here (zero added
  // barriers). ci[slot] stays the authoritative mapping, read after the
  // early-exit guard below (no OOB). Same values => bitwise (V-J18b class).
  __shared__ int s_cu[33];
  __shared__ int s_total;
  // E2: ci[min(slot, ci_rows-1)] joins the SAME parallel cold round as
  // co[N]/cu[] (the loads are mutually independent; the old path re-read
  // ci[slot] in a SECOND serial round after the early-exit guard = ~0.4-0.6us
  // extra cold LDG latency per CTA, census region-15 0.89-1.07us). The
  // clamp keeps the speculative read in-bounds for trivial (slot >= total)
  // blocks; ci_rows = ci.size(0) = chunk_offsets[N] by the frozen ABI, so
  // for every ACTIVE slot (slot < total) min(slot, ci_rows-1) == slot and
  // the value is bitwise the old one.
  __shared__ int2 s_ci;
  {
    const int te = (int)threadIdx.x;
    if (te == 0) s_total = chunk_offsets[N];
    if (te <= N) s_cu[te] = cu[te];
    if (te == 33) s_ci = reinterpret_cast<const int2*>(chunk_indices)[min(slot, ci_rows - 1)];
  }
  if (threadIdx.x == 0) {
    mbar_init(smem_u32(p1mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  const int total = s_total;
  if (slot >= total) {
#ifdef GDN_P1PHASE
    TAIL_REC();                         // V-J16: trivial blocks consume launch
#endif                                  // slots too — record their timeline
#ifndef GDN_E29_OFF
    if (threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // E29
#endif
    return;
  }
  const int2 nc = s_ci;   // E2: staged in the parallel mapping round above
  const int n = nc.x, c = nc.y;
  const int start = s_cu[n] + c * CDIM;
  const int end = min(start + CDIM, s_cu[n + 1]);   // V-J45: smem-resolved
  const int L = end - start;
  TICK(15);   // V-J24b: entry -> mapping resolved. Composition probe for
              // the setup+issue mass (V-J24: solo floor 0.86us = 44% of the
              // preamble): ph[15] = smem/addr setup + the 3-deep dependent
              // LDG chain (chunk_offsets[N] -> chunk_indices[slot] -> cu[n],
              // cu[n+1]); ph[10] (TICK(10) below) = the remainder (perm
              // fold on block0-tid0, mbar_init+syncthreads, 4x TMA box
              // issue, issue_gb cp.async groups).
  const int ratio = HV / H;
  int qh, hv0, cnt;
  if (GQA) { qh = QHFAST ? (int)blockIdx.x : (int)blockIdx.y; hv0 = qh * ratio; cnt = (RATIO > 0) ? RATIO : ratio; }
  else { hv0 = QHFAST ? (int)blockIdx.x : (int)blockIdx.y; qh = hv0 / ratio; cnt = 1; }
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int mw = warp & 3, nw = warp >> 2;   // 4M x 2N warp tiling of [64,64]

  // V2g: fold the P0 LPT request ordering into P1 block (0,0) — the chain
  // consumes perm only after P1 completes (stream order), so the separate
  // gdn_order launch is redundant (~2-3us saved on small shapes).
  if (perm != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
    int key[32], ch[32];
    for (int i = 0; i < N; i++) {
      key[i] = i;
      ch[i] = (s_cu[i + 1] - s_cu[i] + CDIM - 1) / CDIM;   // V-J45
    }
    for (int i = 1; i < N; i++) {  // insertion sort, descending chunks (stable)
      int kk = key[i], cc = ch[i], j = i - 1;
      while (j >= 0 && ch[j] < cc) { ch[j + 1] = ch[j]; key[j + 1] = key[j]; j--; }
      ch[j + 1] = cc; key[j + 1] = kk;
    }
    for (int i = 0; i < N; i++) perm[i] = key[i];
  }

  // ---- V-J2 TMA-swizzle k/q prologue (arm-a V5.35a port): FOUR {64,64,1}
  // SWIZZLE_128B boxes on p1mbar replace the 2,048-op cp.async16 swarm.
  // Peer V25 measured P1 L1/TEX-pipe-bound (74-76%) — TMA bypasses the
  // L1/LSU path entirely (arm-a: prep −0.4..−2.0µs on an fp32-plane P1;
  // expected larger here). Rows >= L of partial chunks hold FOREIGN FINITE
  // rows (next request) or hardware OOB zeros (>= T) — containment wire
  // (arm-a V5.33a §2, re-proven by this tree's smoke X/Y/Z): beta/g
  // predicated zero-fills below UNCHANGED => sys rows >= L = I => Tinv/Qkd
  // rows >= L contained; the s2g image carries them exactly as the
  // KQTMA-direct path does. NEVER weaken those fills.
  // V-J45: mbar_init + the publishing barrier were HOISTED into the
  // mapping block above (the barrier now also publishes s_cu/s_total).
  const float* gh = g + (size_t)start * HV;
  const float* bh = beta + (size_t)start * HV;
  auto issue_gb = [&](int r) {
    int hv = hv0 + r;
    float* sbs = sb0 + ((RATIO > 0) ? r : (r & 1)) * CDIM;
    float* sbe = sbeta0 + ((RATIO > 0) ? r : (r & 1)) * CDIM;
    if (tid < CDIM) {
      cp_async4(&sbs[tid], gh + (size_t)tid * HV + hv, tid < L);
      cp_async4(&sbe[tid], bh + (size_t)tid * HV + hv, tid < L);
    }
    cp_commit();
  };
  if (tid == 0) {
    bulk_expect_tx(smem_u32(p1mbar), KQ_TMA_BYTES);
    tma3d_box(&kmap, sK, 0, start, qh, smem_u32(p1mbar));
    tma3d_box(&kmap, sK + 4096, 64, start, qh, smem_u32(p1mbar));
    tma3d_box(&qmap, sQ, 0, start, qh, smem_u32(p1mbar));
    tma3d_box(&qmap, sQ + 4096, 64, start, qh, smem_u32(p1mbar));
  }
  if constexpr (RATIO > 0) {
    #pragma unroll
    for (int r = 0; r < RATIO; r++) issue_gb(r);   // groups G0..G(RATIO-1)
  } else {
    issue_gb(0);                        // legacy: single g/beta group
  }
  // V26: wait ALL g/beta groups pre-gram (the idle-warp cumsum precompute
  // needs every head's g) + the KQ mbar (boxes land ~1.5-2.5us later).
  TICK(10);                             // V-J24: setup + box/gb issue
#ifndef GDN_E6_SERIAL_WAIT
  if constexpr (RATIO > 0) {
    // E6: split the join. The rho warps (tid 128-191 = the ONLY
    // !tile_active warps) start the cumsum/rho precompute as soon as the
    // g/beta cp.async groups land — released by the ISSUING warps 0-1's own
    // cp_wait via bar.sync 2 (128 thr = warps {0,1,4,5}) — instead of after
    // the 32KB TMA boxes that gate the gram warps. Preamble wall:
    // old = boxwait + max(gram, rho); new = max(cpwait+rho, boxwait+gram):
    // saves ~min(rho, boxwait-cpwait) per CTA on cold large-T shapes
    // (census sub11 box waits: c1 5783, d5 4332, d3 3717, d1 3160 cyc vs
    // rho 0.9-2.5us; expected c1 -3us, d5 -11, d1 -4, d3 -1.3, c0 ~0).
    // BITWISE: the per-head rho op sequences are the verbatim V26/V-J18
    // texts; cp.async visibility = issuer cp_wait + bar.sync 2 (subset
    // barrier orders smem across participants); the TMA mbar acquire stays
    // on every consumer of the box data (gram warps' ldmatrix, tid0's s2g
    // bulk — the rho warps never read sK/sQ). RT0 keeps the legacy order.
    if (tid >= 128 && tid < 192) {
      asm volatile("bar.sync 2, 128;" ::: "memory");
      const int tid2 = tid - 128;
      if constexpr (RATIO >= 4) {
      #pragma unroll
      for (int rr = 0; rr < RATIO; rr++) {          // pass 1: local scans
        float* sbm = sb0 + rr * CDIM;
        float a2 = sbm[tid2];
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
          float o = __shfl_up_sync(0xffffffffu, a2, off, 32);
          if ((tid2 & 31) >= off) a2 += o;
        }
        sbm[tid2] = a2;
      }
      asm volatile("bar.sync 1, 64;" ::: "memory");
      #pragma unroll
      for (int rr = 0; rr < RATIO; rr++) {          // pass 2: cross fixup
        float* sbm = sb0 + rr * CDIM;
        if (tid2 >= 32) sbm[tid2] += sbm[31];
      }
      asm volatile("bar.sync 1, 64;" ::: "memory");
      #pragma unroll
      for (int rr = 0; rr < RATIO; rr++) {          // pass 3: rho stores
        float* sbm = sb0 + rr * CDIM;
        float* sRr = sRho + rr * CDIM;
        sRr[tid2] = (tid2 == 0) ? 1.f : __expf(sbm[tid2 - 1] - sbm[tid2]);
      }
      } else {
        #pragma unroll
        for (int rr = 0; rr < RATIO; rr++) {        // vj11-identical text
          float* sbm = sb0 + rr * CDIM;
          float a2 = sbm[tid2];
          #pragma unroll
          for (int off = 1; off < 32; off <<= 1) {
            float o = __shfl_up_sync(0xffffffffu, a2, off, 32);
            if ((tid2 & 31) >= off) a2 += o;
          }
          sbm[tid2] = a2;
          asm volatile("bar.sync 1, 64;" ::: "memory");
          if (tid2 >= 32) sbm[tid2] += sbm[31];
          asm volatile("bar.sync 1, 64;" ::: "memory");
          float* sRr = sRho + rr * CDIM;
          sRr[tid2] = (tid2 == 0) ? 1.f : __expf(sbm[tid2 - 1] - sbm[tid2]);
        }
      }
    } else {
      cp_wait<0>();
      if (tid < 64) asm volatile("bar.sync 2, 128;" ::: "memory");
      mbar_wait(smem_u32(p1mbar), 0);     // k/q image landed (acquire)
    }
    __syncthreads();
  } else {
    mbar_wait(smem_u32(p1mbar), 0);
    cp_wait<0>();
    __syncthreads();
  }
#else
  mbar_wait(smem_u32(p1mbar), 0);       // k/q image landed (acquire)
  cp_wait<0>();
  __syncthreads();
#endif
  TICK(11);                             // V-J24: box wait (E6: split join)
  TICK(12);                             // V-J24: g/beta wait + join
#ifndef GDN_E29_OFF
  // E29: EXPLICIT PDL launch-dependents trigger. Without it the implicit
  // trigger fires only at PRIMARY-GRID-COMPLETION, so the chain kernel —
  // despite its programmaticStreamSerializationAllowed launch, its
  // griddepcontrol.wait and its P1-independent prologue design (V3b/V5.8a/
  // V-J3) — only begins CTA dispatch AFTER P1 fully drains: mode_probe
  // measured a +5-6us chain-entry gap on fast iterations (and ALL wall
  // variance sat in this gap; P1/chain internals stable within +-0.3us).
  // Trigger placement: every CTA has its k/q boxes + g/beta cp groups
  // landed (remaining P1 = compute + publish, ~60% of block duration for
  // single-wave P1). Dependent grids then become dispatch-eligible while
  // P1 computes; DATA ORDER IS UNCHANGED: the chain's griddepcontrol.wait
  // still binds to primary-grid COMPLETION, and all pre-wait chain reads
  // (PRE0 boxes, sreg_load, cu) are input-only by design. Multi-wave P1
  // (c0/c1: 13 waves) reaches here near grid end => no change there;
  // single/dual-wave P1 (all remaining fixed cases) gains the dispatch
  // ramp + prologue overlap. Under-triggering would merely fall back to
  // the implicit completion trigger (current behavior) — failure mode is
  // neutral.
  if (threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif

  TICK(0);
  // ---- gram = K @ K^T -> sGram; qk = Q @ K^T (raw, kept in registers
  // across the head loop) ----
  // V24 (= V3m fusion, RT-GATED to {2,3}): identical B-frags shared
  // (-16 x4-ldm/active warp in the preamble); per-accumulator mma order
  // + values unchanged => BITWISE. RT8/RT1/RT0 keep the original separate
  // loops: V14 measured fusion overflowing the 128-reg launch_bounds(256,2)
  // budget at RT8 (acc+acc2 ~32 live regs) with the regression canceling
  // the c0/c1 wins. Current-tree RT2/RT3 = 114 regs -> fused needs ptxas
  // to fit ~130 into 128 (census gate: 0-spill PASS, any spill REJECT).
  float acc[4][4] = {};
  float acc2[4][4] = {};
  // V-J2 fast swizzle addressing (arm-a V5.35b, lesson 20): kbase is
  // 1024B-aligned => phase == lane&7 and k&7 == 0 at every frag site.
  // A-frag rows: mw*16 + (m&1)*8 + r; B-frag rows: nw*32{+16} + ((m>>1)&1)*8
  // + r. kg groups: A-frags (2i + (lane>>4)&1) -> xa, B-frags (2i +
  // (lane>>3)&1) -> xb, i = ks&3; sub-tile s = ks>=4 folds to +8192
  // immediates; the q tile rides at +16384 (qT0/qT1). ONE IADD3 per site,
  // bit-identical addresses to the generic swz_addr form.
  const uint32_t kbase = smem_u32(sK);
  const int l7p = lane & 7;
  const uint32_t rba = kbase + (uint32_t)((mw * 16 + ((lane >> 3) & 1) * 8 + l7p) * 128);
  const uint32_t rbb = kbase + (uint32_t)((nw * 32 + ((lane >> 4) & 1) * 8 + l7p) * 128);
  uint32_t xa[4], xb[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    xa[i] = (uint32_t)((((2 * i + ((lane >> 4) & 1)) ^ l7p) << 4));
    xb[i] = (uint32_t)((((2 * i + ((lane >> 3) & 1)) ^ l7p) << 4));
  }
  const bool tile_active = (nw * 32) <= (mw * 16 + 15);
#ifdef GDN_E22_OFF
  const bool hi_half = false;   // E22 attribution: legacy full-tile compute
#else
  // E22: partial pure-upper skip WITHIN active warps. For (mw0,nw0) and
  // (mw2,nw1) the b1 column-half (nw*32+16..31) lies entirely above the
  // diagonal (16>15, 48>47): its nt2/3 gram+qk products are consumed by
  // nobody (sys build selects by i>j / i==j ternaries - stale sGram upper
  // is converted but NEVER selected; QKd publish skips ks>mw == exactly
  // these nt, E4). Skip the b1 ldm, the four nt2/3 mma and the nt2/3 sGram
  // stores: -17% of gram/qk mma, -8% of their x4-ldm wavefronts. BITWISE.
  const bool hi_half = (nw * 32 + 16) > (mw * 16 + 15);
#endif
  if constexpr (RATIO == 2 || RATIO == 3) {
    if (tile_active) {
      #pragma unroll
      for (int ks = 0; ks < KDIM / 16; ks++) {
        uint32_t ak[4], aq[4], b0[4], b1[4];
        // V-J2: fast swz sites (see reg block above); generic-equivalent
        // addresses, bit-identical frags (arm-a bench T1-T3).
        ldm_swzf(ak, rba + xa[ks & 3] + ((ks >= 4) ? 8192u : 0u));
        ldm_swzf(aq, rba + xa[ks & 3] + ((ks >= 4) ? 24576u : 16384u));
        ldm_swzf(b0, rbb + xb[ks & 3] + ((ks >= 4) ? 8192u : 0u));
        if (!hi_half) ldm_swzf(b1, rbb + xb[ks & 3] + ((ks >= 4) ? 10240u : 2048u));   // E22
        mma_bf16(acc[0], ak, b0);
        mma_bf16(acc[1], ak, b0 + 2);
        mma_bf16(acc2[0], aq, b0);
        mma_bf16(acc2[1], aq, b0 + 2);
        if (!hi_half) {   // E22: nt2/3 pure-upper, never consumed
          mma_bf16(acc[2], ak, b1);
          mma_bf16(acc[3], ak, b1 + 2);
          mma_bf16(acc2[2], aq, b1);
          mma_bf16(acc2[3], aq, b1 + 2);
        }
      }
      int gid = lane >> 2, tig = lane & 3;
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        if (hi_half && nt >= 2) continue;   // E22: pure-upper tile never read
        int row0 = mw * 16 + gid;
        int col0 = nw * 32 + nt * 8 + tig * 2;
        // V25: bf16 plane store (was 4x scalar fp32); pairs -> 2x STS.32
        __nv_bfloat162 g0 = {f_hi(acc[nt][0]), f_hi(acc[nt][1])};
        __nv_bfloat162 g1 = {f_hi(acc[nt][2]), f_hi(acc[nt][3])};
        *reinterpret_cast<__nv_bfloat162*>(&sGram[row0 * PL_STR + col0]) = g0;
        *reinterpret_cast<__nv_bfloat162*>(&sGram[(row0 + 8) * PL_STR + col0]) = g1;
      }
#ifdef GDN_E6_SERIAL_WAIT
    } else {
      // V26: warps 4-5 (the ONLY !tile_active warps: (mw<2,nw=1), tid
      // 128-191) precompute cumsum(b)+rho for ALL heads while the 6 active
      // warps run the fused gram+qk (~2.5us shadow >= 1.16/1.74us for
      // RT2/RT3). bar.sync 1 (64 thr) is warp-pair-local; the main
      // __syncthreads below publishes sb/sRho slots CTA-wide. Bitwise:
      // same fp32 op order as the old in-loop scan (32-lane shfl_up scan,
      // single cross-warp +sbm[31] correction, same __expf rho).
      // V-J18-RT8-only: fused-path ratios are {2,3} (V24 gate) — the
      // phase split is DISABLED here: the official stack eval (dv_8aeee52d9c11)
      // showed d9 ([31..49]x10 hr2 = RT2, 320-block multi-wave P1) +3.3%
      // (cand 54.05 vs vj11-official 52.22, incumbent draws equal) while my
      // WARM harnesses measured D9 flat — cold-protocol multi-wave walls
      // amplify small per-block deltas (2 waves x dL). Reverting this copy
      // to the vj11 text makes the RT2/RT3 instantiations source-identical
      // to the officially clean vj11 tree; the replicated J18 win lives in
      // the RT8 non-fused copy below.
      // E6: production hoists this body into the split-wait region above;
      // this copy is the GDN_E6_SERIAL_WAIT attribution fallback.
      const int tid2 = tid - 128;
      #pragma unroll
      for (int rr = 0; rr < RATIO; rr++) {
        float* sbm = sb0 + rr * CDIM;
        float a2 = sbm[tid2];
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
          float o = __shfl_up_sync(0xffffffffu, a2, off, 32);
          if ((tid2 & 31) >= off) a2 += o;
        }
        sbm[tid2] = a2;
        asm volatile("bar.sync 1, 64;" ::: "memory");
        if (tid2 >= 32) sbm[tid2] += sbm[31];
        asm volatile("bar.sync 1, 64;" ::: "memory");
        float* sRr = sRho + rr * CDIM;
        sRr[tid2] = (tid2 == 0) ? 1.f : __expf(sbm[tid2 - 1] - sbm[tid2]);
      }
#endif
    }
    __syncthreads();
  } else {
    if (tile_active) {
      #pragma unroll
      for (int ks = 0; ks < KDIM / 16; ks++) {
        uint32_t a[4], b0[4], b1[4];
        // V-J2: fast swz (gram: A + B both from the k image).
        ldm_swzf(a, rba + xa[ks & 3] + ((ks >= 4) ? 8192u : 0u));
        ldm_swzf(b0, rbb + xb[ks & 3] + ((ks >= 4) ? 8192u : 0u));
        if (!hi_half) ldm_swzf(b1, rbb + xb[ks & 3] + ((ks >= 4) ? 10240u : 2048u));   // E22
        mma_bf16(acc[0], a, b0);
        mma_bf16(acc[1], a, b0 + 2);
        if (!hi_half) {   // E22: nt2/3 pure-upper, never consumed
          mma_bf16(acc[2], a, b1);
          mma_bf16(acc[3], a, b1 + 2);
        }
      }
      int gid = lane >> 2, tig = lane & 3;
      #pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        if (hi_half && nt >= 2) continue;   // E22: pure-upper tile never read
        int row0 = mw * 16 + gid;
        int col0 = nw * 32 + nt * 8 + tig * 2;
        // V25: bf16 plane store (was 4x scalar fp32); pairs -> 2x STS.32
        __nv_bfloat162 g0 = {f_hi(acc[nt][0]), f_hi(acc[nt][1])};
        __nv_bfloat162 g1 = {f_hi(acc[nt][2]), f_hi(acc[nt][3])};
        *reinterpret_cast<__nv_bfloat162*>(&sGram[row0 * PL_STR + col0]) = g0;
        *reinterpret_cast<__nv_bfloat162*>(&sGram[(row0 + 8) * PL_STR + col0]) = g1;
      }
#ifdef GDN_E6_SERIAL_WAIT
    } else if constexpr (RATIO > 0) {
      // V26: same idle-warp precompute for the non-fused path (RT8/RT1;
      // RT0 keeps the legacy in-loop rotation). RT8: 8 heads x ~0.6us may
      // overrun gram (~1.2us) into the post-gram barrier - still net-
      // positive (removes 8 x (0.58us + 4 barriers) from the serial loop).
      // V-J18 (RT>=4 ONLY): 3-pass phase split — 16 serialized bar.sync
      // rendezvous (~0.49us/head, V-J17a probe) collapse to 2, and the
      // shfl/fixup/expf chains of the 8 heads interleave for ILP.
      // BITWISE: per-head op sequences verbatim; only barrier placement.
      // Gated to RATIO>=4 after the official stack eval (dv_8aeee52d9c11)
      // showed the RT2 domain case d9 +3.3% under the cold protocol (multi-
      // wave walls amplify small per-block deltas); RT1/RT2/RT3 keep the
      // vj11-identical per-rr loop below (replicated win: c9 prep -0.7/-0.5us
      // across dv_f690fc2fc4f8 + dv_e5c8ae0bf94b).
      // E6: production hoists this body into the split-wait region above;
      // this copy is the GDN_E6_SERIAL_WAIT attribution fallback.
      const int tid2 = tid - 128;
      if constexpr (RATIO >= 4) {
      #pragma unroll
      for (int rr = 0; rr < RATIO; rr++) {          // pass 1: local scans
        float* sbm = sb0 + rr * CDIM;
        float a2 = sbm[tid2];
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
          float o = __shfl_up_sync(0xffffffffu, a2, off, 32);
          if ((tid2 & 31) >= off) a2 += o;
        }
        sbm[tid2] = a2;
      }
      asm volatile("bar.sync 1, 64;" ::: "memory");
      #pragma unroll
      for (int rr = 0; rr < RATIO; rr++) {          // pass 2: cross fixup
        float* sbm = sb0 + rr * CDIM;
        if (tid2 >= 32) sbm[tid2] += sbm[31];
      }
      asm volatile("bar.sync 1, 64;" ::: "memory");
      #pragma unroll
      for (int rr = 0; rr < RATIO; rr++) {          // pass 3: rho stores
        float* sbm = sb0 + rr * CDIM;
        float* sRr = sRho + rr * CDIM;
        sRr[tid2] = (tid2 == 0) ? 1.f : __expf(sbm[tid2 - 1] - sbm[tid2]);
      }
      } else {
        #pragma unroll
        for (int rr = 0; rr < RATIO; rr++) {        // vj11-identical (RT1)
          float* sbm = sb0 + rr * CDIM;
          float a2 = sbm[tid2];
          #pragma unroll
          for (int off = 1; off < 32; off <<= 1) {
            float o = __shfl_up_sync(0xffffffffu, a2, off, 32);
            if ((tid2 & 31) >= off) a2 += o;
          }
          sbm[tid2] = a2;
          asm volatile("bar.sync 1, 64;" ::: "memory");
          if (tid2 >= 32) sbm[tid2] += sbm[31];
          asm volatile("bar.sync 1, 64;" ::: "memory");
          float* sRr = sRho + rr * CDIM;
          sRr[tid2] = (tid2 == 0) ? 1.f : __expf(sbm[tid2 - 1] - sbm[tid2]);
        }
      }
#endif
    }
    __syncthreads();
    if (tile_active) {
      #pragma unroll
      for (int ks = 0; ks < KDIM / 16; ks++) {
        uint32_t a[4], b0[4], b1[4];
        // V-J2: fast swz (qk: A from the q image at +16384, B from k).
        ldm_swzf(a, rba + xa[ks & 3] + ((ks >= 4) ? 24576u : 16384u));
        ldm_swzf(b0, rbb + xb[ks & 3] + ((ks >= 4) ? 8192u : 0u));
        if (!hi_half) ldm_swzf(b1, rbb + xb[ks & 3] + ((ks >= 4) ? 10240u : 2048u));   // E22
        mma_bf16(acc2[0], a, b0);
        mma_bf16(acc2[1], a, b0 + 2);
        if (!hi_half) {   // E22: nt2/3 pure-upper, publish skips ks>mw
          mma_bf16(acc2[2], a, b1);
          mma_bf16(acc2[3], a, b1 + 2);
        }
      }
    }
  }

  TICK(13);   // V-J24: gram/qk block done (tid128: rho + conv-sync)
  // ---- sKQ store (V-J2): the CONTIGUOUS 32,768B swizzle image [kT0 kT1
  // qT0 qT1] emitted verbatim from smem — the chain's bulk path stages this
  // byte image and consumes it with the SAME swz ldmatrix code as KQTMA
  // (once per qk head; non-GQA: only the group's first hv block writes).
  // Rows >= L hold foreign/OOB data (containment wire, see prologue).
  // V-J2 (arm-a V5.33a S2): skipped when the chain runs KQTMA (direct TMA
  // reads of the input q/k) — eliminates the whole s2g pass for that regime.
  if (!skip_q && (GQA || hv0 == qh * ratio)) {
    // V3g: sK..sQ are contiguous (32,768B image) -> one bulk s2g (TMA).
    // Nothing overlays sK/sQ anymore (V25) — grid drain covers visibility.
    char* gkq = scratchq + ((size_t)slot * H + qh) * KSQB_BYTES;
    if (tid == 0) {
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                   :: "l"(gkq), "r"(smem_u32(sK)), "n"(KSQB_BYTES) : "memory");
      asm volatile("cp.async.bulk.commit_group;");
    }
  }

  TICK(14);   // V-J24: sKQ store issue region
  TICK(0);
  // ---- per value-head: system, inverse, stores ----
  if constexpr (RATIO == 2 || RATIO == 3 || RATIO == 8) {   // V-J44x
    // V-J44: group-of-2 head-lockstep merge (fused RT2/RT3 ONLY; RT8/RT1/RT0
    // keep the serial loop verbatim below). Each phase runs the group's nh
    // INDEPENDENT head item-spaces in ONE pass over the previously idle
    // thread capacity (lvl8 was 64/256 threads, lvl16/32 128/256, lvl64 4/8
    // warps — serial per head), with ONE barrier set per group instead of
    // per head. Every per-element op chain is VERBATIM (same FFMA order,
    // same __expf, same mma/ldmatrix sequences, same stored values and
    // addresses per head) — only the item->thread mapping and barrier
    // placement move => BITWISE (V-J18b class). Census premise (V-J17a
    // dual): per-head serial chain = 2.40us x RATIO = 68% of the c0 P1
    // block (10.54us); heads are independent (V26 pre-lands all sb/sRho).
#ifndef GDN_E19_OFF
    // E19: ONE-TIME sTinv identity + pad-col zero init for BOTH group planes.
    // The per-group re-init was redundant: each head's lvl8/16/32/64 rewrite
    // EXACTLY the strict-lower tiles (they tile the full strict-lower 64x64),
    // while diag(=1), strict-upper(=0) and pad cols 64-71(=0, the E9 blk3
    // A-tile NaN guard) are never written afterwards — the skeleton is loop-
    // invariant. Same values at every consumption point => BITWISE. Savings
    // scale with group count: RT8 4 groups -> 1 init (c9 P1 is 21% of wall),
    // RT3 2 -> 1, RT2 single-group neutral.
    for (int idx = tid; idx < 2 * CDIM * 8; idx += NTHR) {
      const int h = idx >> 9, e = idx & 511;
      const int i = e >> 3, j8 = (e & 7) * 8;
      __nv_bfloat16* tp = sTinv + h * (CDIM * PL_STR);
      uint4 tu;
      __nv_bfloat162* t2 = reinterpret_cast<__nv_bfloat162*>(&tu);
      #pragma unroll
      for (int q = 0; q < 4; q++)
        t2[q] = {f_hi((i == j8 + 2 * q) ? 1.f : 0.f),
                 f_hi((i == j8 + 2 * q + 1) ? 1.f : 0.f)};
      *reinterpret_cast<uint4*>(&tp[i * PL_STR + j8]) = tu;
    }
    for (int idx = tid; idx < 2 * 128; idx += NTHR) {
      const int hz = idx >> 7, ez = idx & 127;
      const int plane = ez >> 6, row = ez & 63;
      __nv_bfloat16* zb = (plane ? sTinv : sSys) + hz * (CDIM * PL_STR);
      const uint4 z4 = {0u, 0u, 0u, 0u};
      *reinterpret_cast<uint4*>(&zb[row * PL_STR + 64]) = z4;
    }
    __syncthreads();
#endif
    for (int g0 = 0; g0 < cnt; g0 += 2) {
      const int nh = (cnt - g0 >= 2) ? 2 : 1;
      TICK(1);
      // ---- system build, all group heads: item = (h, jg, i) ----
      // V-J47: 8-col retile of the V2h 4-col grouping (V27-series
      // successor). 128-bit gram LDS / rho LDS x2 / sSys+sTinv STS per
      // item replace the 4-col pair's ops => LSU/thread -36% (56->36)
      // on P1's largest head item (system build = 1.05us of the 3.59us
      // fused head sum, CLASSD; LSU-throughput-bound to ~2%).
      // i-MAJOR mapping (consecutive lanes = consecutive rows, shared
      // jg): rho/sb/sbeta reads are warp-uniform BROADCASTS; the
      // 128-bit gram/sSys/sTinv ops are conflict-free (144B pitch =
      // 36 words => per-quarter-warp bank offsets 4r distinct).
      // BITWISE: dual expf anchors - E0=expf(bi-sb[j8]) for cols 0-3
      // and F0=expf(bi-sb[j8+4]) for cols 4-7, each with its own
      // 4-step rho chain (SAME arguments, order, masks, f_hi packs and
      // addresses as the 4-col form; a continued 8-step chain would
      // shift values ~1e-7 and is deliberately NOT used). V-J18b class.
#ifdef GDN_E23_OFF
      for (int idx = tid; idx < nh * (CDIM * 8); idx += NTHR) {
        const int h = idx >> 9, e = idx & 511;
        const int jg = e >> 6, i = e & 63;
        const int j8 = jg * 8;
        const int r = g0 + h;
        const float* sb = sb0 + r * CDIM;
        const float* sbeta = sbeta0 + r * CDIM;
        const float* sRh = sRho + r * CDIM;
        __nv_bfloat16* sSysh = sSys + h * (CDIM * PL_STR);
        __nv_bfloat16* sTinvh = sTinv + h * (CDIM * PL_STR);
        const uint4 gu = *reinterpret_cast<const uint4*>(&sGram[i * PL_STR + j8]);
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gu);
        const float4 rvA = *reinterpret_cast<const float4*>(&sRh[j8]);
        const float4 rvB = *reinterpret_cast<const float4*>(&sRh[j8 + 4]);
        const float be = sbeta[i], bi = sb[i];
        const float E0 = __expf(bi - sb[j8]);          // one expf per 4 cols
        const float E1 = E0 * rvA.y, E2 = E1 * rvA.z, E3 = E2 * rvA.w;
        const float F0 = __expf(bi - sb[j8 + 4]);
        const float F1 = F0 * rvB.y, F2 = F1 * rvB.z, F3 = F2 * rvB.w;
        float4 va, vb;
        va.x = (i > j8 + 0) ? be * E0 * __bfloat162float(g2[0].x) : ((i == j8 + 0) ? 1.f : 0.f);
        va.y = (i > j8 + 1) ? be * E1 * __bfloat162float(g2[0].y) : ((i == j8 + 1) ? 1.f : 0.f);
        va.z = (i > j8 + 2) ? be * E2 * __bfloat162float(g2[1].x) : ((i == j8 + 2) ? 1.f : 0.f);
        va.w = (i > j8 + 3) ? be * E3 * __bfloat162float(g2[1].y) : ((i == j8 + 3) ? 1.f : 0.f);
        vb.x = (i > j8 + 4) ? be * F0 * __bfloat162float(g2[2].x) : ((i == j8 + 4) ? 1.f : 0.f);
        vb.y = (i > j8 + 5) ? be * F1 * __bfloat162float(g2[2].y) : ((i == j8 + 5) ? 1.f : 0.f);
        vb.z = (i > j8 + 6) ? be * F2 * __bfloat162float(g2[3].x) : ((i == j8 + 6) ? 1.f : 0.f);
        vb.w = (i > j8 + 7) ? be * F3 * __bfloat162float(g2[3].y) : ((i == j8 + 7) ? 1.f : 0.f);
        uint4 su;
        *reinterpret_cast<__nv_bfloat162*>(&su.x) = {f_hi(va.x), f_hi(va.y)};
        *reinterpret_cast<__nv_bfloat162*>(&su.y) = {f_hi(va.z), f_hi(va.w)};
        *reinterpret_cast<__nv_bfloat162*>(&su.z) = {f_hi(vb.x), f_hi(vb.y)};
        *reinterpret_cast<__nv_bfloat162*>(&su.w) = {f_hi(vb.z), f_hi(vb.w)};
        *reinterpret_cast<uint4*>(&sSysh[i * PL_STR + j8]) = su;
#ifdef GDN_E19_OFF
        float4 ta, tb;
        ta.x = (i == j8 + 0) ? 1.f : 0.f;
        ta.y = (i == j8 + 1) ? 1.f : 0.f;
        ta.z = (i == j8 + 2) ? 1.f : 0.f;
        ta.w = (i == j8 + 3) ? 1.f : 0.f;
        tb.x = (i == j8 + 4) ? 1.f : 0.f;
        tb.y = (i == j8 + 5) ? 1.f : 0.f;
        tb.z = (i == j8 + 6) ? 1.f : 0.f;
        tb.w = (i == j8 + 7) ? 1.f : 0.f;
        uint4 tu;
        *reinterpret_cast<__nv_bfloat162*>(&tu.x) = {f_hi(ta.x), f_hi(ta.y)};
        *reinterpret_cast<__nv_bfloat162*>(&tu.y) = {f_hi(ta.z), f_hi(ta.w)};
        *reinterpret_cast<__nv_bfloat162*>(&tu.z) = {f_hi(tb.x), f_hi(tb.y)};
        *reinterpret_cast<__nv_bfloat162*>(&tu.w) = {f_hi(tb.z), f_hi(tb.w)};
        *reinterpret_cast<uint4*>(&sTinvh[i * PL_STR + j8]) = tu;
#endif
      }
#else
      // E23: head-shared gram load. sGram is GROUP-INVARIANT (gram = K@K^T
      // is shared by the group's q-heads — the fused preamble computes it
      // once), but the legacy (h,jg,i) item space re-loaded the same uint4
      // per head. item = (jg,i) with an inner head loop halves the gram
      // LDS.128 stream (~-12% sysbuild LSU; ph2 = the largest remaining P1
      // phase after E16/E19/E22). Per-head op chains VERBATIM (same dual
      // expf anchors, rho chains, masks, packs, addresses) => BITWISE.
      for (int idx = tid; idx < CDIM * 8; idx += NTHR) {
        const int jg = idx >> 6, i = idx & 63;
        const int j8 = jg * 8;
        const uint4 gu = *reinterpret_cast<const uint4*>(&sGram[i * PL_STR + j8]);
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gu);
        for (int h = 0; h < nh; h++) {
          const int r = g0 + h;
          const float* sb = sb0 + r * CDIM;
          const float* sbeta = sbeta0 + r * CDIM;
          const float* sRh = sRho + r * CDIM;
          __nv_bfloat16* sSysh = sSys + h * (CDIM * PL_STR);
          const float4 rvA = *reinterpret_cast<const float4*>(&sRh[j8]);
          const float4 rvB = *reinterpret_cast<const float4*>(&sRh[j8 + 4]);
          const float be = sbeta[i], bi = sb[i];
          const float E0 = __expf(bi - sb[j8]);          // one expf per 4 cols
          const float E1 = E0 * rvA.y, E2 = E1 * rvA.z, E3 = E2 * rvA.w;
          const float F0 = __expf(bi - sb[j8 + 4]);
          const float F1 = F0 * rvB.y, F2 = F1 * rvB.z, F3 = F2 * rvB.w;
          float4 va, vb;
          va.x = (i > j8 + 0) ? be * E0 * __bfloat162float(g2[0].x) : ((i == j8 + 0) ? 1.f : 0.f);
          va.y = (i > j8 + 1) ? be * E1 * __bfloat162float(g2[0].y) : ((i == j8 + 1) ? 1.f : 0.f);
          va.z = (i > j8 + 2) ? be * E2 * __bfloat162float(g2[1].x) : ((i == j8 + 2) ? 1.f : 0.f);
          va.w = (i > j8 + 3) ? be * E3 * __bfloat162float(g2[1].y) : ((i == j8 + 3) ? 1.f : 0.f);
          vb.x = (i > j8 + 4) ? be * F0 * __bfloat162float(g2[2].x) : ((i == j8 + 4) ? 1.f : 0.f);
          vb.y = (i > j8 + 5) ? be * F1 * __bfloat162float(g2[2].y) : ((i == j8 + 5) ? 1.f : 0.f);
          vb.z = (i > j8 + 6) ? be * F2 * __bfloat162float(g2[3].x) : ((i == j8 + 6) ? 1.f : 0.f);
          vb.w = (i > j8 + 7) ? be * F3 * __bfloat162float(g2[3].y) : ((i == j8 + 7) ? 1.f : 0.f);
          uint4 su;
          *reinterpret_cast<__nv_bfloat162*>(&su.x) = {f_hi(va.x), f_hi(va.y)};
          *reinterpret_cast<__nv_bfloat162*>(&su.y) = {f_hi(va.z), f_hi(va.w)};
          *reinterpret_cast<__nv_bfloat162*>(&su.z) = {f_hi(vb.x), f_hi(vb.y)};
          *reinterpret_cast<__nv_bfloat162*>(&su.w) = {f_hi(vb.z), f_hi(vb.w)};
          *reinterpret_cast<uint4*>(&sSysh[i * PL_STR + j8]) = su;
#ifdef GDN_E19_OFF
          // flag orthogonality: E19_OFF restores the in-loop identity even
          // on the E23 item mapping (E23_OFF+E19_OFF == legacy verbatim).
          __nv_bfloat16* sTinvh = sTinv + h * (CDIM * PL_STR);
          float4 ta, tb;
          ta.x = (i == j8 + 0) ? 1.f : 0.f;
          ta.y = (i == j8 + 1) ? 1.f : 0.f;
          ta.z = (i == j8 + 2) ? 1.f : 0.f;
          ta.w = (i == j8 + 3) ? 1.f : 0.f;
          tb.x = (i == j8 + 4) ? 1.f : 0.f;
          tb.y = (i == j8 + 5) ? 1.f : 0.f;
          tb.z = (i == j8 + 6) ? 1.f : 0.f;
          tb.w = (i == j8 + 7) ? 1.f : 0.f;
          uint4 tu;
          *reinterpret_cast<__nv_bfloat162*>(&tu.x) = {f_hi(ta.x), f_hi(ta.y)};
          *reinterpret_cast<__nv_bfloat162*>(&tu.y) = {f_hi(ta.z), f_hi(ta.w)};
          *reinterpret_cast<__nv_bfloat162*>(&tu.z) = {f_hi(tb.x), f_hi(tb.y)};
          *reinterpret_cast<__nv_bfloat162*>(&tu.w) = {f_hi(tb.z), f_hi(tb.w)};
          *reinterpret_cast<uint4*>(&sTinvh[i * PL_STR + j8]) = tu;
#endif
        }
      }
#endif
      __syncthreads();

      TICK(2);
      // ---- QKd publish (V2i position preserved: ahead of the inverse
      // levels), head-serial within the group: GMEM-only stores, no barrier,
      // acc2 register liveness unchanged. E4: TRIANGLE-PACKED slot image —
      // only 16x16 frag tiles (m=mw, ks=nw*2+(nt>>1)) with ks <= mw exist in
      // gmem (the chain never loads ks > mw); tiles are 512B, row pitch 32B.
      // Values/masks BITWISE unchanged; skipped tiles were exact zeros. ----
      for (int hh = 0; hh < nh; hh++) {
        const int r = g0 + hh;
        const int hv = hv0 + r;
        const float* sb = sb0 + r * CDIM;
        const float* sRh = sRho + r * CDIM;
        __nv_bfloat16* gQ = reinterpret_cast<__nv_bfloat16*>(
            scratch + ((size_t)slot * HV + hv) * SLOT_BYTES) + PLANE_PACKED_BYTES / 2;
        if (tile_active) {
          int gid = lane >> 2, tig = lane & 3;
          #pragma unroll
          for (int nt = 0; nt < 4; nt++) {
            if (nw * 2 + (nt >> 1) > mw) continue;   // E4: no such tile
            const int ks = nw * 2 + (nt >> 1);
            const int tofs = tri_tile(mw, ks) + (nt & 1) * 128 + tig * 2;
            #pragma unroll
            for (int half = 0; half < 2; half++) {
              int i = mw * 16 + gid + half * 8;
              int j = nw * 32 + nt * 8 + tig * 2;
              float e0 = acc2[nt][half * 2 + 0] * QSCALE;
              float e1 = acc2[nt][half * 2 + 1] * QSCALE;
              const float E = __expf(sb[i] - sb[j]);   // V2q: e1 via rho chain
              e0 *= (i >= j) ? E : 0.f;
              e1 *= (i >= j + 1) ? E * sRh[j + 1] : 0.f;   // V26: per-head rho slot
              __nv_bfloat162 pr = {f_hi(e0), f_hi(e1)};   // hi-only (V2f)
              *reinterpret_cast<__nv_bfloat162*>(&gQ[tofs + (gid + half * 8) * 8]) = pr;
            }
          }
        }
      }

      TICK(3);
      // ---- level 8 lockstep (V2n register forward substitution, item =
      // (h, w8, j)): one lane per column, chain VERBATIM ----
#ifdef GDN_E19_OFF
      {
        // E9: pad-col zero-fill relocated here (cols 64-71 of sSys/sTinv,
        // all group heads) — runs concurrently with lvl8 (disjoint addrs)
        // and rides lvl8's trailing barrier, so the fused lvl16 below needs
        // NO extra barrier for its blk3 A-tile pad reads. (Uninitialized
        // smem there could carry NaN patterns; 0*NaN=NaN would poison valid
        // C rows even though the k8-15 A slice is "killed".)
        for (int idx = tid; idx < nh * 128; idx += NTHR) {
          const int hz = idx >> 7, ez = idx & 127;
          const int plane = ez >> 6, row = ez & 63;
          __nv_bfloat16* zb = plane ? sTinv : sSys;
          const uint4 z4 = {0u, 0u, 0u, 0u};
          *reinterpret_cast<uint4*>(&zb[hz * (CDIM * PL_STR) + row * PL_STR + 64]) = z4;
        }
      }
#endif
      if (tid < nh * 64) {
        const int h = tid >> 6;
        const int base = ((tid >> 3) & 7) * 8;
        const int j = tid & 7;
        const __nv_bfloat16* sSysh = sSys + h * (CDIM * PL_STR);
        __nv_bfloat16* sTinvh = sTinv + h * (CDIM * PL_STR);
        float tt[8];
        tt[0] = 1.f;                       // diagonal
        #pragma unroll
        for (int d = 1; d < 8; d++) {
          const __nv_bfloat16* rowp = &sSysh[(base + ((j + d < 8) ? (j + d) : 7)) * PL_STR + base + j];  // V25: bf16
          float a2 = 0.f;
          #pragma unroll
          for (int q = 0; q < 7; q++) a2 += (q < d) ? __bfloat162float(rowp[q]) * tt[q] : 0.f;
          tt[d] = -a2;
        }
        #pragma unroll
        for (int d = 1; d < 8; d++)
          if (j + d < 8) sTinvh[(base + j + d) * PL_STR + base + j] = f_hi(tt[d]);   // V25: bf16
      }
      __syncthreads();
      TICK(4);
      // ---- level 16 (E9): fused single-pass mma form of
      // X = -((invD2 @ L21)_hi+lo) @ invD1, all FOUR 8x8 off-diagonals
      // (blk 0..3, one per 16x16 diagonal block — the E1 first cut wrongly
      // mirrored lvl32's TWO-block structure and left rows 40-63 unwritten;
      // unit test e1_unit16.py caught it). One warp per (head, blk):
      // warps 0..4*nh-1, h = warp>>2, blk = warp&3.
      // History: E1b materialized Y = L21@invD1 in sYb (hi/lo) and ran a
      // second X pass behind an extra barrier — measured 1757-2184 cyc/CTA
      // (census dv_5b5ba2738931), NO faster than the SIMT it replaced
      // (1637): the smem round-trip + barrier, not the mma, was the cost.
      // E9 keeps Z = invD2@L21 in C registers and reuses the C-frag lane
      // map as the next mma's A-frag (a0 = (c0,c1), a1 = (c2,c3), a2/a3=0)
      // => one pass, one barrier, no sYb, ~3 ldm + 3 mma + packs per warp.
      // Padded-m16n8k16 zero-kills + the blk3 pad-col finiteness guard are
      // documented at the fused block below and at the relocated pad loop
      // (beside lvl8). Numerics: Z hi/lo bf16 pair (~2^-17) — same declared
      // class as E1b's Y hi/lo; X16 feeds lvl32/64 + published Tinv (the
      // BOTTOM of the hierarchical compose; single-bf16 intermediates are
      // bounded to the terminal level, lvl64, by the E1 smoke history).
      // GDN_E1_SIMT16: attribution fallback (debug builds) = the verbatim
      // certified V-J44 SIMT text; production compiles WITHOUT the flag.
#ifndef GDN_E1_SIMT16
      {
        // E9 FUSED lvl16: X = -(Z_hi + Z_lo) @ invD1 with Z = invD2 @ L21,
        // all in ONE warp pass — no sYb plane, no Y/X barrier (the E1b form
        // measured 1757-2184 cyc: two dependent passes + smem round-trip +
        // extra barrier ~= the old SIMT 1637; the round-trip, not the mma,
        // was the cost). C-frag == A-frag lane map (a0 = (c0,c1), a1 =
        // (c2,c3)) makes the Z -> A conversion pure register packing.
        // Padded m16n8k16 kills: Z-pass A k8-15 = sTinv strict upper /
        // blk3 pad (zero-filled pre-lvl8 barrier) kills B k8-15 = next
        // blk's finite L21 rows; A rows 8-15 garbage -> C rows 8-15
        // discarded (finite: all factors bf16-finite). X-pass a2/a3 = 0
        // kills invD1's zero-padded k8-15 rows. Z kept hi+lo (~2^-17):
        // X16 feeds lvl32/64 + published Tinv (bottom of the compose).
        const int gid = lane >> 2, tig = lane & 3;
        if (warp < 4 * nh) {
          const int h = warp >> 2, blk = warp & 3;
          const int r0 = blk * 16 + 8, c0 = blk * 16;
          const __nv_bfloat16* sSysh = sSys + h * (CDIM * PL_STR);
          __nv_bfloat16* sTinvh = sTinv + h * (CDIM * PL_STR);
          uint32_t a[4], b[2];
          float zacc[4] = {};
          ldm_a(a, sTinvh, PL_STR, r0, r0);                    // invD2 (padded)
          ldm_b1_kmajor(b, &sSysh[r0 * PL_STR + c0], PL_STR, 0, 0);   // L21 k16n8
          mma_bf16(zacc, a, b);                                 // Z rows 0-7
          uint32_t b2[2], a2[4];
          ldm_b1_kmajor(b2, &sTinvh[c0 * PL_STR + c0], PL_STR, 0, 0); // invD1 k16n8
          float xacc[4] = {};
          const __nv_bfloat162 zh01 = {f_hi(zacc[0]), f_hi(zacc[1])};
          const __nv_bfloat162 zh23 = {f_hi(zacc[2]), f_hi(zacc[3])};
          a2[0] = *reinterpret_cast<const uint32_t*>(&zh01);
          a2[1] = *reinterpret_cast<const uint32_t*>(&zh23);
          a2[2] = 0u; a2[3] = 0u;
          mma_bf16(xacc, a2, b2);
#ifdef GDN_E16_OFF
          const __nv_bfloat162 zl01 = {f_lo(zacc[0], zh01.x), f_lo(zacc[1], zh01.y)};
          const __nv_bfloat162 zl23 = {f_lo(zacc[2], zh23.x), f_lo(zacc[3], zh23.y)};
          a2[0] = *reinterpret_cast<const uint32_t*>(&zl01);
          a2[1] = *reinterpret_cast<const uint32_t*>(&zl23);
          mma_bf16(xacc, a2, b2);              // X = -(Z_hi+Z_lo) @ invD1
#else
          // E16: X = -Z_hi @ invD1 — single-bf16 intermediate, the V25
          // lvl64 declared rounding class extended to lvl16 (X16 is stored
          // bf16-quantized anyway; the lo pass only shifts which bf16 value
          // it lands on, ~2^-9 relative class either way).
#endif
          *reinterpret_cast<__nv_bfloat162*>(&sTinvh[(r0 + gid) * PL_STR + c0 + tig * 2]) =
              {f_hi(-xacc[0]), f_hi(-xacc[1])};
        }
      }
#else
      // ---- level 16 lockstep: X = -(invD2 @ (L21 @ invD1)), item = (h,blk,e)
      for (int idx = tid; idx < nh * 128; idx += NTHR) {
        const int h = idx >> 7, e7 = idx & 127;
        const int blk = e7 >> 5, e = e7 & 31;
        const int i = e >> 2, j2 = (e & 3) * 2;
        const int r0 = blk * 16 + 8, c0 = blk * 16;
        const __nv_bfloat16* sSysh = sSys + h * (CDIM * PL_STR);
        const __nv_bfloat16* sTinvh = sTinv + h * (CDIM * PL_STR);
        float y0 = 0.f, y1 = 0.f;
        #pragma unroll
        for (int m = 0; m < 8; m++) {
          float sv = __bfloat162float(sSysh[(r0 + i) * PL_STR + c0 + m]);   // V25: bf16 plane
          __nv_bfloat162 t2 = *reinterpret_cast<const __nv_bfloat162*>(&sTinvh[(c0 + m) * PL_STR + c0 + j2]);
          y0 += sv * __bfloat162float(t2.x);
          y1 += sv * __bfloat162float(t2.y);
        }
        *reinterpret_cast<float2*>(&sYf[h * (32 * 40) + blk * 64 + i * 8 + j2]) = make_float2(y0, y1);
      }
      __syncthreads();
      for (int idx = tid; idx < nh * 128; idx += NTHR) {
        const int h = idx >> 7, e7 = idx & 127;
        const int blk = e7 >> 5, e = e7 & 31;
        const int i = e >> 2, j2 = (e & 3) * 2;
        const int r0 = blk * 16 + 8, c0 = blk * 16;
        __nv_bfloat16* sTinvh = sTinv + h * (CDIM * PL_STR);
        float x0 = 0.f, x1 = 0.f;
        #pragma unroll
        for (int m = 0; m < 8; m++) {
          float iv = __bfloat162float(sTinvh[(r0 + i) * PL_STR + r0 + m]);   // V25
          float2 yv = *reinterpret_cast<const float2*>(&sYf[h * (32 * 40) + blk * 64 + m * 8 + j2]);
          x0 += iv * yv.x;
          x1 += iv * yv.y;
        }
        *reinterpret_cast<__nv_bfloat162*>(&sTinvh[(r0 + i) * PL_STR + c0 + j2]) = {f_hi(-x0), f_hi(-x1)};   // V25: bf16 store
      }
#endif
      __syncthreads();
      TICK(5);
      // ---- level 32 (E10): FUSED single-pass mma form, same C-frag->A-frag
      // register reuse as E9's lvl16: Z = invD2 @ L21 (full 16x16 tiles, no
      // padding — every factor lies inside written plane regions, no NaN
      // guard needed), then X = -(Z_hi + Z_lo) @ invD1 with the A-frag
      // packed from Z's C registers (a0=(z0.c0,z0.c1), a1=(z0.c2,z0.c3),
      // a2=(z1.c0,z1.c1), a3=(z1.c2,z1.c3)). Removes the sYb round-trip +
      // its barrier (E1b form: 660-721 cyc/CTA). Z hi/lo = ~2^-17, strictly
      // tighter than the certified single-bf16-Y lvl32 class it replaces.
      // GDN_E1_SIMT32: attribution fallback = verbatim certified V2o/V27 text.
#ifndef GDN_E1_SIMT32
      {
        const int gid = lane >> 2, tig = lane & 3;
        if (warp < 2 * nh) {
          const int h = warp >> 1, blk = warp & 1;
          const int r0 = blk * 32 + 16, c0 = blk * 32;
          const __nv_bfloat16* sSysh = sSys + h * (CDIM * PL_STR);
          __nv_bfloat16* sTinvh = sTinv + h * (CDIM * PL_STR);
          uint32_t a[4], b[4];
          float zacc[2][4] = {};
          ldm_a(a, sTinvh, PL_STR, r0, r0);                          // invD2
          ldm_b2_kmajor(b, &sSysh[r0 * PL_STR + c0], PL_STR, 0, 0);  // L21 k16n16
          mma_bf16(zacc[0], a, b);
          mma_bf16(zacc[1], a, b + 2);                     // Z = invD2 @ L21
          uint32_t b2[4], a2[4];
          ldm_b2_kmajor(b2, &sTinvh[c0 * PL_STR + c0], PL_STR, 0, 0); // invD1
          float xacc[2][4] = {};
          const __nv_bfloat162 zh0 = {f_hi(zacc[0][0]), f_hi(zacc[0][1])};
          const __nv_bfloat162 zh1 = {f_hi(zacc[0][2]), f_hi(zacc[0][3])};
          const __nv_bfloat162 zh2 = {f_hi(zacc[1][0]), f_hi(zacc[1][1])};
          const __nv_bfloat162 zh3 = {f_hi(zacc[1][2]), f_hi(zacc[1][3])};
          a2[0] = *reinterpret_cast<const uint32_t*>(&zh0);
          a2[1] = *reinterpret_cast<const uint32_t*>(&zh1);
          a2[2] = *reinterpret_cast<const uint32_t*>(&zh2);
          a2[3] = *reinterpret_cast<const uint32_t*>(&zh3);
          mma_bf16(xacc[0], a2, b2);
          mma_bf16(xacc[1], a2, b2 + 2);
#ifdef GDN_E16_OFF
          const __nv_bfloat162 zl0 = {f_lo(zacc[0][0], zh0.x), f_lo(zacc[0][1], zh0.y)};
          const __nv_bfloat162 zl1 = {f_lo(zacc[0][2], zh1.x), f_lo(zacc[0][3], zh1.y)};
          const __nv_bfloat162 zl2 = {f_lo(zacc[1][0], zh2.x), f_lo(zacc[1][1], zh2.y)};
          const __nv_bfloat162 zl3 = {f_lo(zacc[1][2], zh3.x), f_lo(zacc[1][3], zh3.y)};
          a2[0] = *reinterpret_cast<const uint32_t*>(&zl0);
          a2[1] = *reinterpret_cast<const uint32_t*>(&zl1);
          a2[2] = *reinterpret_cast<const uint32_t*>(&zl2);
          a2[3] = *reinterpret_cast<const uint32_t*>(&zl3);
          mma_bf16(xacc[0], a2, b2);
          mma_bf16(xacc[1], a2, b2 + 2);          // X = -(Z_hi+Z_lo) @ invD1
#else
          // E16: single-bf16 Z (V25 lvl64 declared class extended to lvl32).
#endif
          #pragma unroll
          for (int nt = 0; nt < 2; nt++)
            #pragma unroll
            for (int half = 0; half < 2; half++) {
              const int row = r0 + gid + 8 * half;
              const int col = c0 + nt * 8 + tig * 2;
              *reinterpret_cast<__nv_bfloat162*>(&sTinvh[row * PL_STR + col]) =
                  {f_hi(-xacc[nt][half * 2 + 0]), f_hi(-xacc[nt][half * 2 + 1])};
            }
        }
      }
#else
      // ---- level 32 lockstep (V2o/V27 retile preserved per item) ----
      for (int idx = tid; idx < nh * 128; idx += NTHR) {
        const int h = idx >> 7, e7 = idx & 127;
        const int blk = e7 >> 6, e = e7 & 63;
        const int i = e >> 2, j4 = (e & 3) * 4;
        const int r0 = blk * 32 + 16, c0 = blk * 32;
        const __nv_bfloat16* sSysh = sSys + h * (CDIM * PL_STR);
        const __nv_bfloat16* sTinvh = sTinv + h * (CDIM * PL_STR);
        const __nv_bfloat16* lrow = &sSysh[(r0 + i) * PL_STR + c0];   // V25: bf16
        float y[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int m4 = 0; m4 < 16; m4 += 4) {
          const uint2 l4u = *reinterpret_cast<const uint2*>(lrow + m4);   // V25: 4xbf16 = 8B
          const __nv_bfloat162* l2p = reinterpret_cast<const __nv_bfloat162*>(&l4u);
          const float lp[4] = {__bfloat162float(l2p[0].x), __bfloat162float(l2p[0].y),
                               __bfloat162float(l2p[1].x), __bfloat162float(l2p[1].y)};
          #pragma unroll
          for (int q = 0; q < 4; q++) {
            const uint2 t4u = *reinterpret_cast<const uint2*>(&sTinvh[(c0 + m4 + q) * PL_STR + c0 + j4]);
            const __nv_bfloat162* t2p = reinterpret_cast<const __nv_bfloat162*>(&t4u);
            y[0] += lp[q] * __bfloat162float(t2p[0].x);
            y[1] += lp[q] * __bfloat162float(t2p[0].y);
            y[2] += lp[q] * __bfloat162float(t2p[1].x);
            y[3] += lp[q] * __bfloat162float(t2p[1].y);
          }
        }
        *reinterpret_cast<float4*>(&sYf[h * (32 * 40) + blk * 256 + i * 16 + j4]) = make_float4(y[0], y[1], y[2], y[3]);
      }
      __syncthreads();
      for (int idx = tid; idx < nh * 128; idx += NTHR) {
        const int h = idx >> 7, e7 = idx & 127;
        const int blk = e7 >> 6, e = e7 & 63;
        const int i = e >> 2, j4 = (e & 3) * 4;
        const int r0 = blk * 32 + 16, c0 = blk * 32;
        __nv_bfloat16* sTinvh = sTinv + h * (CDIM * PL_STR);
        float x[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int m4 = 0; m4 < 16; m4 += 4) {
          const uint2 e4u = *reinterpret_cast<const uint2*>(&sTinvh[(r0 + i) * PL_STR + r0 + m4]);   // V25
          const __nv_bfloat162* e2p = reinterpret_cast<const __nv_bfloat162*>(&e4u);
          const float ep[4] = {__bfloat162float(e2p[0].x), __bfloat162float(e2p[0].y),
                               __bfloat162float(e2p[1].x), __bfloat162float(e2p[1].y)};
          #pragma unroll
          for (int q = 0; q < 4; q++) {
            const float4 yv = *reinterpret_cast<const float4*>(&sYf[h * (32 * 40) + blk * 256 + (m4 + q) * 16 + j4]);
            x[0] += ep[q] * yv.x;
            x[1] += ep[q] * yv.y;
            x[2] += ep[q] * yv.z;
            x[3] += ep[q] * yv.w;
          }
        }
        uint2 ou;   // V27: 4 bf16 = one 8B STS (was bf162)
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ou);
        o2[0] = {f_hi(-x[0]), f_hi(-x[1])};
        o2[1] = {f_hi(-x[2]), f_hi(-x[3])};
        *reinterpret_cast<uint2*>(&sTinvh[(r0 + i) * PL_STR + c0 + j4]) = ou;   // V25: bf16 store
      }
#endif
      __syncthreads();
      TICK(6);
      // ---- level 64 lockstep: warps 0-3 = head g0, warps 4-7 = head g0+1;
      // intra-warp (wm,wn) tiling and the V25 direct single-bf16 mma chains
      // VERBATIM. Y = L21 @ invD1, then X = -invD2 @ Y. ----
      {
        const int lh = warp >> 2;
        if (lh < nh) {
          const int sub = warp & 3, wm = sub & 1, wn = sub >> 1;
          const __nv_bfloat16* sSysh = sSys + lh * (CDIM * PL_STR);
          const __nv_bfloat16* sTinvh = sTinv + lh * (CDIM * PL_STR);
          __nv_bfloat16* sYbh = reinterpret_cast<__nv_bfloat16*>(sYf + lh * (32 * 40));
          float yacc[2][4] = {};
          #pragma unroll
          for (int ks = 0; ks < 2; ks++) {
            uint32_t ah[4], bh[4];
            ldm_a(ah, sSysh + 32 * PL_STR, PL_STR, wm * 16, ks * 16);
            ldm_b2_kmajor(bh, sTinvh, PL_STR, ks * 16, wn * 16);
            mma_bf16(yacc[0], ah, bh);
            mma_bf16(yacc[1], ah, bh + 2);
          }
          int gid = lane >> 2, tig = lane & 3;
          #pragma unroll
          for (int nt = 0; nt < 2; nt++)
            #pragma unroll
            for (int half = 0; half < 2; half++) {
              int row = wm * 16 + gid + 8 * half;
              int col = wn * 16 + nt * 8 + tig * 2;
              *reinterpret_cast<__nv_bfloat162*>(&sYbh[row * 40 + col]) =
                  {f_hi(yacc[nt][half * 2 + 0]), f_hi(yacc[nt][half * 2 + 1])};
            }
        }
      }
      __syncthreads();
      {
        const int lh = warp >> 2;
        if (lh < nh) {
          const int sub = warp & 3, wm = sub & 1, wn = sub >> 1;
          __nv_bfloat16* sTinvh = sTinv + lh * (CDIM * PL_STR);
          __nv_bfloat16* sYbh = reinterpret_cast<__nv_bfloat16*>(sYf + lh * (32 * 40));
          float xacc[2][4] = {};
          #pragma unroll
          for (int ks = 0; ks < 2; ks++) {
            uint32_t ah[4], bh[4];
            ldm_a(ah, sTinvh + 32 * PL_STR + 32, PL_STR, wm * 16, ks * 16);
            ldm_b2_kmajor(bh, sYbh, 40, ks * 16, wn * 16);
            mma_bf16(xacc[0], ah, bh);
            mma_bf16(xacc[1], ah, bh + 2);
          }
          int gid = lane >> 2, tig = lane & 3;
          #pragma unroll
          for (int nt = 0; nt < 2; nt++)
            #pragma unroll
            for (int half = 0; half < 2; half++) {
              int row = 32 + wm * 16 + gid + 8 * half;
              int col = wn * 16 + nt * 8 + tig * 2;
              *reinterpret_cast<__nv_bfloat162*>(&sTinvh[row * PL_STR + col]) =
                  {f_hi(-xacc[nt][half * 2 + 0]), f_hi(-xacc[nt][half * 2 + 1])};
            }
        }
      }
      __syncthreads();

      TICK(7);
      // ---- publish lockstep: Tinv/b(+beta+Ai/Dl) for all group heads ----
      // E4: triangle-packed Tinv image — 10 frag tiles t (512B each,
      // tile-major t(m,ks) = m(m+1)/2 + ks), item = (h, t, row16, chunk16B).
      // Same bytes copied per element; upper tiles never published (chain
      // never reads them). Bitwise.
      for (int idx = tid; idx < nh * (TRI_TILES * 32); idx += NTHR) {
        const int h = (idx >= TRI_TILES * 32) ? 1 : 0;   // nh <= 2
        const int e = idx - h * (TRI_TILES * 32);
        const int t = e >> 5, rem = e & 31;
        const int rr = rem >> 1, cc = rem & 1;
        // tile -> (m, ks) without a runtime-indexed table (local-mem risk)
        const int m = (t >= 6) ? 3 : ((t >= 3) ? 2 : ((t >= 1) ? 1 : 0));
        const int ks = t - (m * (m + 1)) / 2;
        char* slot_base = scratch + ((size_t)slot * HV + hv0 + g0 + h) * SLOT_BYTES;
        __nv_bfloat16* gTinvH = reinterpret_cast<__nv_bfloat16*>(slot_base);
        const __nv_bfloat16* sTinvh = sTinv + h * (CDIM * PL_STR);
        // V25: sTinv IS the published element format -> pure 16B copies.
        *reinterpret_cast<uint4*>(&gTinvH[t * 256 + cc * 128 + rr * 8]) =
            *reinterpret_cast<const uint4*>(
                &sTinvh[(m * 16 + rr) * PL_STR + ks * 16 + cc * 8]);
      }
      if (tid < nh * CDIM) {
        const int h = tid >> 6, c = tid & 63;
        const int r = g0 + h;
        const float* sb = sb0 + r * CDIM;
        const float* sbeta = sbeta0 + r * CDIM;
        char* slot_base = scratch + ((size_t)slot * HV + hv0 + r) * SLOT_BYTES;
        float* gb = reinterpret_cast<float*>(
            reinterpret_cast<__nv_bfloat16*>(slot_base) + PLANE_PACKED_BYTES);
        gb[c] = sb[c];                       // V5.12a: b in slot tail
        (gb + CDIM)[c] = sbeta[c];           // V5.12a: beta (bit-identical)
        const float bi = sb[c];              // V5.18a: Ai/Dl planes, same
        (gb + 2 * CDIM)[c] = __expf(bi);     // __expf on the same fp32 values
        (gb + 3 * CDIM)[c] = __expf(sb[CDIM - 1] - bi);
      }
      // sets are reused by the next group: separate publish reads from the
      // next sys-build writes (last group: no successor, barrier dropped as
      // in the serial loop's r+1<cnt guard).
      if (g0 + 2 < cnt) { __syncthreads(); }
      TICK(8);
#ifdef GDN_P1PHASE
      ph[9] += nh;
#endif
    }
  } else {
#ifndef GDN_E19_OFF
  // E19: one-time sTinv identity + pad-col zero init (serial path; see the
  // lockstep copy for the invariance proof). RT1/RT0: single init replaces
  // the per-head identity portion of the system-build loop and the per-head
  // E9 pad fill.
  for (int idx = tid; idx < CDIM * 8; idx += NTHR) {
    const int i = idx >> 3, j8 = (idx & 7) * 8;
    uint4 tu;
    __nv_bfloat162* t2 = reinterpret_cast<__nv_bfloat162*>(&tu);
    #pragma unroll
    for (int q = 0; q < 4; q++)
      t2[q] = {f_hi((i == j8 + 2 * q) ? 1.f : 0.f),
               f_hi((i == j8 + 2 * q + 1) ? 1.f : 0.f)};
    *reinterpret_cast<uint4*>(&sTinv[i * PL_STR + j8]) = tu;
  }
  for (int idx = tid; idx < 128; idx += NTHR) {
    const int plane = idx >> 6, row = idx & 63;
    __nv_bfloat16* zb = plane ? sTinv : sSys;
    const uint4 z4 = {0u, 0u, 0u, 0u};
    *reinterpret_cast<uint4*>(&zb[row * PL_STR + 64]) = z4;
  }
  __syncthreads();
#endif
  for (int r = 0; r < cnt; r++) {
    const int hv = hv0 + r;
    const float* sb = sb0 + ((RATIO > 0) ? r : (r & 1)) * CDIM;
    const float* sbeta = sbeta0 + ((RATIO > 0) ? r : (r & 1)) * CDIM;
    const float* sRh = sRho + ((RATIO > 0) ? r : (r & 1)) * CDIM;   // V26 per-head rho
    if constexpr (RATIO > 0) {
      // V26: all g/beta landed pre-gram (cp_wait<0>) and this head's
      // cumsum+rho were precomputed on the gram idle warps (bitwise same
      // values). The staged wait, the 0.58us serial scan and 4 full-CTA
      // barriers per head are gone; the post-gram syncthreads published
      // the sb/sRho slots CTA-wide.
    } else {
      if (r > 0) { cp_wait<0>(); __syncthreads(); }
      // b = inclusive cumsum(g)   [legacy RT0 rotating-buffer path]
      {
        // cumsum64 uses its own syncs; operates on the current set in place
        float* sbm = const_cast<float*>(sb);
        if (tid < 64) {
          float a2 = sbm[tid];
          #pragma unroll
          for (int off = 1; off < 32; off <<= 1) {
            float o = __shfl_up_sync(0xffffffffu, a2, off, 32);
            if ((tid & 31) >= off) a2 += o;
          }
          sbm[tid] = a2;
        }
        __syncthreads();
        if (tid >= 32 && tid < 64) sbm[tid] += sbm[31];
        __syncthreads();
      }
      // V2q: rho[j] = exp(b_{j-1} - b_j). Frozen input g = -exp(A_log)*softplus(.)
      // with A_log in [-3,1] and dt in [-3,3] gives |g| <~ 35 << 88, so rho is
      // finite (<= e^35) and the product form below cannot overflow/NaN;
      // slow_decay/zero domains give rho ~= 1. Relative error ~1e-7 per factor,
      // far below the bf16 plane quantization (2^-9) used downstream.
      if (tid < CDIM) const_cast<float*>(sRh)[tid] = (tid == 0) ? 1.f : __expf(sb[tid - 1] - sb[tid]);   // V26: rotating slot (matches sRh reads)
      __syncthreads();
    }

    TICK(1);
    // system = I + tril(beta_i * exp(b_i-b_j) * gram_ij, -1); V2h: 4-col
    // groups with float4 LDS/STS (4x fewer smem ops; expf math unchanged).
    for (int idx = tid; idx < CDIM * 16; idx += NTHR) {
      const int i = idx >> 4, j4 = (idx & 15) * 4;
      // V25: bf16 plane LDS (uint2 = 4 elems); math stays fp32; expf unchanged
      const uint2 gu = *reinterpret_cast<const uint2*>(&sGram[i * PL_STR + j4]);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gu);
      const float4 gv = {__bfloat162float(g2[0].x), __bfloat162float(g2[0].y),
                         __bfloat162float(g2[1].x), __bfloat162float(g2[1].y)};
      const float4 rv = *reinterpret_cast<const float4*>(&sRh[j4]);   // V26: per-head rho slot
      const float be = sbeta[i], bi = sb[i];
      const float E0 = __expf(bi - sb[j4]);          // one expf per 4 cols
      const float E1 = E0 * rv.y, E2 = E1 * rv.z, E3 = E2 * rv.w;
      float4 vv;
      vv.x = (i > j4 + 0) ? be * E0 * gv.x : ((i == j4 + 0) ? 1.f : 0.f);
      vv.y = (i > j4 + 1) ? be * E1 * gv.y : ((i == j4 + 1) ? 1.f : 0.f);
      vv.z = (i > j4 + 2) ? be * E2 * gv.z : ((i == j4 + 2) ? 1.f : 0.f);
      vv.w = (i > j4 + 3) ? be * E3 * gv.w : ((i == j4 + 3) ? 1.f : 0.f);
      uint2 su;
      *reinterpret_cast<__nv_bfloat162*>(&su.x) = {f_hi(vv.x), f_hi(vv.y)};
      *reinterpret_cast<__nv_bfloat162*>(&su.y) = {f_hi(vv.z), f_hi(vv.w)};
      *reinterpret_cast<uint2*>(&sSys[i * PL_STR + j4]) = su;
#ifdef GDN_E19_OFF
      float4 tv;
      tv.x = (i == j4 + 0) ? 1.f : 0.f;
      tv.y = (i == j4 + 1) ? 1.f : 0.f;
      tv.z = (i == j4 + 2) ? 1.f : 0.f;
      tv.w = (i == j4 + 3) ? 1.f : 0.f;
      uint2 tu;
      *reinterpret_cast<__nv_bfloat162*>(&tu.x) = {f_hi(tv.x), f_hi(tv.y)};
      *reinterpret_cast<__nv_bfloat162*>(&tu.y) = {f_hi(tv.z), f_hi(tv.w)};
      *reinterpret_cast<uint2*>(&sTinv[i * PL_STR + j4]) = tu;
#endif
    }
    __syncthreads();

    TICK(2);
    // V2i: QKd publish depends only on acc2 regs + sb — hoisted ahead of the
    // inverse levels so the warps idle during the 64-thread level-8 forward
    // substitution overlap it (P1 is per-hv serial in GQA mode).
    // E4: triangle-packed slot image (see the lockstep copy).
    {
      __nv_bfloat16* gQ = reinterpret_cast<__nv_bfloat16*>(
          scratch + ((size_t)slot * HV + hv) * SLOT_BYTES) + PLANE_PACKED_BYTES / 2;
      if (tile_active) {
        int gid = lane >> 2, tig = lane & 3;
        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          if (nw * 2 + (nt >> 1) > mw) continue;   // E4: no such tile
          const int ks = nw * 2 + (nt >> 1);
          const int tofs = tri_tile(mw, ks) + (nt & 1) * 128 + tig * 2;
          #pragma unroll
          for (int half = 0; half < 2; half++) {
            int i = mw * 16 + gid + half * 8;
            int j = nw * 32 + nt * 8 + tig * 2;
            float e0 = acc2[nt][half * 2 + 0] * QSCALE;
            float e1 = acc2[nt][half * 2 + 1] * QSCALE;
            const float E = __expf(sb[i] - sb[j]);   // V2q: e1 via rho chain
            e0 *= (i >= j) ? E : 0.f;
            e1 *= (i >= j + 1) ? E * sRh[j + 1] : 0.f;   // V26: per-head rho slot
            __nv_bfloat162 pr = {f_hi(e0), f_hi(e1)};   // hi-only (V2f)
            *reinterpret_cast<__nv_bfloat162*>(&gQ[tofs + (gid + half * 8) * 8]) = pr;
          }
        }
      }
    }

    TICK(3);
    // V2m: kick next head's g/beta prefetch HERE (was in publish) — buffer
    // (r+1)&1 is free once iteration r-1 published, and the wait at the top
    // of r+1 then hides behind ~3us of inverse levels (measured exposure was
    // 0.19-0.37us/hv at publish-position).
    if (RATIO == 0 && r + 1 < cnt) { issue_gb(r + 1); }
#ifdef GDN_E19_OFF
    {
      // E9: pad-col zero-fill relocated (cols 64-71 of sSys/sTinv) —
      // concurrent with lvl8 (disjoint addrs), rides its trailing barrier.
      for (int idx = tid; idx < 128; idx += NTHR) {
        const int plane = idx >> 6, row = idx & 63;
        __nv_bfloat16* zb = plane ? sTinv : sSys;
        const uint4 z4 = {0u, 0u, 0u, 0u};
        *reinterpret_cast<uint4*>(&zb[row * PL_STR + 64]) = z4;
      }
    }
#endif
    // level 8 (V2n): REGISTER forward substitution — column j's recurrence
    // only consumes column j's own prior outputs (unit-lower-triangular), so
    // one lane per column keeps t[] in registers: no STS->LDS forwarding
    // chain (wavefront measured 1.47-1.70us/hv; both prior forms pay 7
    // smem round-trips). sSys reads are dependency-free (batched/pipelined).
    if (lane < 8) {
      const int base = warp * 8;
      const int j = lane;
      float tt[8];
      tt[0] = 1.f;                       // diagonal
      #pragma unroll
      for (int d = 1; d < 8; d++) {
        const __nv_bfloat16* rowp = &sSys[(base + ((j + d < 8) ? (j + d) : 7)) * PL_STR + base + j];  // V25: bf16
        float a2 = 0.f;
        #pragma unroll
        for (int q = 0; q < 7; q++) a2 += (q < d) ? __bfloat162float(rowp[q]) * tt[q] : 0.f;
        tt[d] = -a2;
      }
      #pragma unroll
      for (int d = 1; d < 8; d++)
        if (j + d < 8) sTinv[(base + j + d) * PL_STR + base + j] = f_hi(tt[d]);   // V25: bf16
    }
    __syncthreads();
    TICK(4);
    // level 16 (E9): FUSED single-pass mma form (see the lockstep copy for
    // the full kill/padding proof and the round-trip-cost rationale).
    // Serial single-head path: warps 0-3 = blk 0-3.
    // GDN_E1_SIMT16: attribution fallback = verbatim certified V2q text.
#ifndef GDN_E1_SIMT16
    {
      const int gid = lane >> 2, tig = lane & 3;
      if (warp < 4) {
        const int blk = warp;
        const int r0 = blk * 16 + 8, c0 = blk * 16;
        uint32_t a[4], b[2];
        float zacc[4] = {};
        ldm_a(a, sTinv, PL_STR, r0, r0);                    // invD2 (padded)
        ldm_b1_kmajor(b, &sSys[r0 * PL_STR + c0], PL_STR, 0, 0);   // L21 k16n8
        mma_bf16(zacc, a, b);                                // Z rows 0-7
        uint32_t b2[2], a2[4];
        ldm_b1_kmajor(b2, &sTinv[c0 * PL_STR + c0], PL_STR, 0, 0); // invD1 k16n8
        float xacc[4] = {};
        const __nv_bfloat162 zh01 = {f_hi(zacc[0]), f_hi(zacc[1])};
        const __nv_bfloat162 zh23 = {f_hi(zacc[2]), f_hi(zacc[3])};
        a2[0] = *reinterpret_cast<const uint32_t*>(&zh01);
        a2[1] = *reinterpret_cast<const uint32_t*>(&zh23);
        a2[2] = 0u; a2[3] = 0u;
        mma_bf16(xacc, a2, b2);
#ifdef GDN_E16_OFF
        const __nv_bfloat162 zl01 = {f_lo(zacc[0], zh01.x), f_lo(zacc[1], zh01.y)};
        const __nv_bfloat162 zl23 = {f_lo(zacc[2], zh23.x), f_lo(zacc[3], zh23.y)};
        a2[0] = *reinterpret_cast<const uint32_t*>(&zl01);
        a2[1] = *reinterpret_cast<const uint32_t*>(&zl23);
        mma_bf16(xacc, a2, b2);              // X = -(Z_hi+Z_lo) @ invD1
#else
        // E16: X = -Z_hi @ invD1 — single-bf16 intermediate (V25 lvl64
        // declared rounding class extended to lvl16).
#endif
        *reinterpret_cast<__nv_bfloat162*>(&sTinv[(r0 + gid) * PL_STR + c0 + tig * 2]) =
            {f_hi(-xacc[0]), f_hi(-xacc[1])};
      }
    }
#else
    // level 16: off-diag 8x8: X = -(invD2 @ (L21 @ invD1)) ; sYf as [4][64]
    // V2q: (blk, i, j-pair) mapping with float2 column reads.
    for (int idx = tid; idx < 128; idx += NTHR) {
      int blk = idx >> 5, e = idx & 31;
      int i = e >> 2, j2 = (e & 3) * 2;
      int r0 = blk * 16 + 8, c0 = blk * 16;
      float y0 = 0.f, y1 = 0.f;
      #pragma unroll
      for (int m = 0; m < 8; m++) {
        float sv = __bfloat162float(sSys[(r0 + i) * PL_STR + c0 + m]);   // V25: bf16 plane
        __nv_bfloat162 t2 = *reinterpret_cast<const __nv_bfloat162*>(&sTinv[(c0 + m) * PL_STR + c0 + j2]);
        y0 += sv * __bfloat162float(t2.x);
        y1 += sv * __bfloat162float(t2.y);
      }
      *reinterpret_cast<float2*>(&sYf[blk * 64 + i * 8 + j2]) = make_float2(y0, y1);
    }
    __syncthreads();
    for (int idx = tid; idx < 128; idx += NTHR) {
      int blk = idx >> 5, e = idx & 31;
      int i = e >> 2, j2 = (e & 3) * 2;
      int r0 = blk * 16 + 8, c0 = blk * 16;
      float x0 = 0.f, x1 = 0.f;
      #pragma unroll
      for (int m = 0; m < 8; m++) {
        float iv = __bfloat162float(sTinv[(r0 + i) * PL_STR + r0 + m]);   // V25
        float2 yv = *reinterpret_cast<const float2*>(&sYf[blk * 64 + m * 8 + j2]);
        x0 += iv * yv.x;
        x1 += iv * yv.y;
      }
      *reinterpret_cast<__nv_bfloat162*>(&sTinv[(r0 + i) * PL_STR + c0 + j2]) = {f_hi(-x0), f_hi(-x1)};   // V25: bf16 store
    }
#endif
    __syncthreads();
    TICK(5);
    // level 32 (E10): FUSED single-pass mma form (see the lockstep copy for
    // the design note). Serial single-head path: warps 0-1 = blk 0-1.
    // GDN_E1_SIMT32: attribution fallback = verbatim certified V2o/V27 text.
#ifndef GDN_E1_SIMT32
    {
      const int gid = lane >> 2, tig = lane & 3;
      if (warp < 2) {
        const int blk = warp;
        const int r0 = blk * 32 + 16, c0 = blk * 32;
        uint32_t a[4], b[4];
        float zacc[2][4] = {};
        ldm_a(a, sTinv, PL_STR, r0, r0);                          // invD2
        ldm_b2_kmajor(b, &sSys[r0 * PL_STR + c0], PL_STR, 0, 0);  // L21 k16n16
        mma_bf16(zacc[0], a, b);
        mma_bf16(zacc[1], a, b + 2);                     // Z = invD2 @ L21
        uint32_t b2[4], a2[4];
        ldm_b2_kmajor(b2, &sTinv[c0 * PL_STR + c0], PL_STR, 0, 0);  // invD1
        float xacc[2][4] = {};
        const __nv_bfloat162 zh0 = {f_hi(zacc[0][0]), f_hi(zacc[0][1])};
        const __nv_bfloat162 zh1 = {f_hi(zacc[0][2]), f_hi(zacc[0][3])};
        const __nv_bfloat162 zh2 = {f_hi(zacc[1][0]), f_hi(zacc[1][1])};
        const __nv_bfloat162 zh3 = {f_hi(zacc[1][2]), f_hi(zacc[1][3])};
        a2[0] = *reinterpret_cast<const uint32_t*>(&zh0);
        a2[1] = *reinterpret_cast<const uint32_t*>(&zh1);
        a2[2] = *reinterpret_cast<const uint32_t*>(&zh2);
        a2[3] = *reinterpret_cast<const uint32_t*>(&zh3);
        mma_bf16(xacc[0], a2, b2);
        mma_bf16(xacc[1], a2, b2 + 2);
#ifdef GDN_E16_OFF
        const __nv_bfloat162 zl0 = {f_lo(zacc[0][0], zh0.x), f_lo(zacc[0][1], zh0.y)};
        const __nv_bfloat162 zl1 = {f_lo(zacc[0][2], zh1.x), f_lo(zacc[0][3], zh1.y)};
        const __nv_bfloat162 zl2 = {f_lo(zacc[1][0], zh2.x), f_lo(zacc[1][1], zh2.y)};
        const __nv_bfloat162 zl3 = {f_lo(zacc[1][2], zh3.x), f_lo(zacc[1][3], zh3.y)};
        a2[0] = *reinterpret_cast<const uint32_t*>(&zl0);
        a2[1] = *reinterpret_cast<const uint32_t*>(&zl1);
        a2[2] = *reinterpret_cast<const uint32_t*>(&zl2);
        a2[3] = *reinterpret_cast<const uint32_t*>(&zl3);
        mma_bf16(xacc[0], a2, b2);
        mma_bf16(xacc[1], a2, b2 + 2);          // X = -(Z_hi+Z_lo) @ invD1
#else
        // E16: single-bf16 Z (V25 lvl64 declared class extended to lvl32).
#endif
        #pragma unroll
        for (int nt = 0; nt < 2; nt++)
          #pragma unroll
          for (int half = 0; half < 2; half++) {
            const int row = r0 + gid + 8 * half;
            const int col = c0 + nt * 8 + tig * 2;
            *reinterpret_cast<__nv_bfloat162*>(&sTinv[row * PL_STR + col]) =
                {f_hi(-xacc[nt][half * 2 + 0]), f_hi(-xacc[nt][half * 2 + 1])};
          }
      }
    }
#else
    // level 32 (V2o): off-diag 16x16 — V27 (blk, i, j4) float4 retile.
    for (int idx = tid; idx < 128; idx += NTHR) {
      const int blk = idx >> 6, e = idx & 63;
      const int i = e >> 2, j4 = (e & 3) * 4;
      const int r0 = blk * 32 + 16, c0 = blk * 32;
      const __nv_bfloat16* lrow = &sSys[(r0 + i) * PL_STR + c0];   // V25: bf16
      float y[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int m4 = 0; m4 < 16; m4 += 4) {
        const uint2 l4u = *reinterpret_cast<const uint2*>(lrow + m4);   // V25: 4xbf16 = 8B
        const __nv_bfloat162* l2p = reinterpret_cast<const __nv_bfloat162*>(&l4u);
        const float lp[4] = {__bfloat162float(l2p[0].x), __bfloat162float(l2p[0].y),
                             __bfloat162float(l2p[1].x), __bfloat162float(l2p[1].y)};
        #pragma unroll
        for (int q = 0; q < 4; q++) {
          const uint2 t4u = *reinterpret_cast<const uint2*>(&sTinv[(c0 + m4 + q) * PL_STR + c0 + j4]);
          const __nv_bfloat162* t2p = reinterpret_cast<const __nv_bfloat162*>(&t4u);
          y[0] += lp[q] * __bfloat162float(t2p[0].x);
          y[1] += lp[q] * __bfloat162float(t2p[0].y);
          y[2] += lp[q] * __bfloat162float(t2p[1].x);
          y[3] += lp[q] * __bfloat162float(t2p[1].y);
        }
      }
      *reinterpret_cast<float4*>(&sYf[blk * 256 + i * 16 + j4]) = make_float4(y[0], y[1], y[2], y[3]);
    }
    __syncthreads();
    for (int idx = tid; idx < 128; idx += NTHR) {
      const int blk = idx >> 6, e = idx & 63;
      const int i = e >> 2, j4 = (e & 3) * 4;
      const int r0 = blk * 32 + 16, c0 = blk * 32;
      float x[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int m4 = 0; m4 < 16; m4 += 4) {
        const uint2 e4u = *reinterpret_cast<const uint2*>(&sTinv[(r0 + i) * PL_STR + r0 + m4]);   // V25
        const __nv_bfloat162* e2p = reinterpret_cast<const __nv_bfloat162*>(&e4u);
        const float ep[4] = {__bfloat162float(e2p[0].x), __bfloat162float(e2p[0].y),
                             __bfloat162float(e2p[1].x), __bfloat162float(e2p[1].y)};
        #pragma unroll
        for (int q = 0; q < 4; q++) {
          const float4 yv = *reinterpret_cast<const float4*>(&sYf[blk * 256 + (m4 + q) * 16 + j4]);
          x[0] += ep[q] * yv.x;
          x[1] += ep[q] * yv.y;
          x[2] += ep[q] * yv.z;
          x[3] += ep[q] * yv.w;
        }
      }
      uint2 ou;   // V27: 4 bf16 = one 8B STS (was bf162)
      __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ou);
      o2[0] = {f_hi(-x[0]), f_hi(-x[1])};
      o2[1] = {f_hi(-x[2]), f_hi(-x[3])};
      *reinterpret_cast<uint2*>(&sTinv[(r0 + i) * PL_STR + c0 + j4]) = ou;   // V25: bf16 store
    }
#endif
    __syncthreads();
    TICK(6);
    // V25: level 64 via DIRECT single-bf16 mma from the working planes —
    // hi/lo overlay conversion (emit8 x3), the 8 overlay planes, the
    // pre-lvl64 bulk wait_group.read+sync (nothing overlays sK/sQ anymore;
    // grid drain covers s2g visibility) and the 3-product hi/lo mma sets are
    // ALL removed: 12 mma -> 4 per warp, ~20KB/head overlay STS gone, 2 fewer
    // barriers/head. X = -(invD2 @ (L21 @ invD1)), all [32,32]:
    //   A-frags ldm direct from sSys[32:64][0:32] (L21) / sTinv[32:64][32:64]
    //   (invD2); B-frags from sTinv[0:32][0:32] (invD1) / the single bf16 Y
    //   plane on the sYf scratch (lvl16/32 sYf reads complete, barrier-
    //   separated). Row pitches 144B (PL_STR) / 80B (Y) are 16B multiples ->
    //   ldmatrix-aligned. Numerics: single-rounding (2^-9) merge inputs vs
    //   hi/lo (~2^-17): the declared per-level rounding class; fp32 mma
    //   accumulation kept; gate = mm==0 + margin class bounds.
    // Y = L21 @ invD1; warps 0-3 (2M x 2N)
    if (warp < 4) {
      const int wm = warp & 1, wn = warp >> 1;
      float yacc[2][4] = {};
      #pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        uint32_t ah[4], bh[4];
        ldm_a(ah, sSys + 32 * PL_STR, PL_STR, wm * 16, ks * 16);
        ldm_b2_kmajor(bh, sTinv, PL_STR, ks * 16, wn * 16);
        mma_bf16(yacc[0], ah, bh);
        mma_bf16(yacc[1], ah, bh + 2);
      }
      int gid = lane >> 2, tig = lane & 3;
      #pragma unroll
      for (int nt = 0; nt < 2; nt++)
        #pragma unroll
        for (int half = 0; half < 2; half++) {
          int row = wm * 16 + gid + 8 * half;
          int col = wn * 16 + nt * 8 + tig * 2;
          *reinterpret_cast<__nv_bfloat162*>(&sYb[row * 40 + col]) =
              {f_hi(yacc[nt][half * 2 + 0]), f_hi(yacc[nt][half * 2 + 1])};
        }
    }
    __syncthreads();
    // X = -invD2 @ Y; write sTinv off-diagonal (bf16)
    if (warp < 4) {
      const int wm = warp & 1, wn = warp >> 1;
      float xacc[2][4] = {};
      #pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        uint32_t ah[4], bh[4];
        ldm_a(ah, sTinv + 32 * PL_STR + 32, PL_STR, wm * 16, ks * 16);
        ldm_b2_kmajor(bh, sYb, 40, ks * 16, wn * 16);
        mma_bf16(xacc[0], ah, bh);
        mma_bf16(xacc[1], ah, bh + 2);
      }
      int gid = lane >> 2, tig = lane & 3;
      #pragma unroll
      for (int nt = 0; nt < 2; nt++)
        #pragma unroll
        for (int half = 0; half < 2; half++) {
          int row = 32 + wm * 16 + gid + 8 * half;
          int col = wn * 16 + nt * 8 + tig * 2;
          *reinterpret_cast<__nv_bfloat162*>(&sTinv[row * PL_STR + col]) =
              {f_hi(-xacc[nt][half * 2 + 0]), f_hi(-xacc[nt][half * 2 + 1])};
        }
    }
    __syncthreads();

    TICK(7);
    // ---- publish: store Tinv/b (QKd hoisted, g/beta prefetch hoisted) ----
    // E4: triangle-packed Tinv image (see the lockstep copy).
    char* slot_base = scratch + ((size_t)slot * HV + hv) * SLOT_BYTES;
    __nv_bfloat16* gTinvH = reinterpret_cast<__nv_bfloat16*>(slot_base);
    float* gb = reinterpret_cast<float*>(gTinvH + PLANE_PACKED_BYTES);

    // V2h: 16B gmem stores; V3e: hi plane only (TinvL dead).
    // V25: sTinv IS the published element format -> pure 16B smem->gmem
    // copies, zero cvt/pack ops.
    for (int idx = tid; idx < TRI_TILES * 32; idx += NTHR) {
      const int t = idx >> 5, rem = idx & 31;
      const int rr = rem >> 1, cc = rem & 1;
      const int m = (t >= 6) ? 3 : ((t >= 3) ? 2 : ((t >= 1) ? 1 : 0));
      const int ks = t - (m * (m + 1)) / 2;
      *reinterpret_cast<uint4*>(&gTinvH[t * 256 + cc * 128 + rr * 8]) =
          *reinterpret_cast<const uint4*>(
              &sTinv[(m * 16 + rr) * PL_STR + ks * 16 + cc * 8]);
    }
    if (tid < CDIM) gb[tid] = sb[tid];
    // V5.12a: publish beta into the slot tail (gb+CDIM) so the chain's
    // single bulk covers planes+b+beta — removes the chain's 64 strided
    // cp_async4 beta requests per chunk (64 excessive-heavy LSU wavefronts
    // on its top pipe) and the 64 b requests. Values bit-identical: P1's
    // sbeta beyond L is cp.async-zero-filled (b = flat cumsum tail), which
    // is exactly what the chain's old predicated (tid<Lc, zero-fill) load
    // produced for the same slot (same L per slot).
    if (tid < CDIM) (gb + CDIM)[tid] = sbeta[tid];
    // V5.18a: Ai/Dl decay planes in the slot tail. The chain's per-chunk
    // 5x __expf (+arg FADDs, ~4.5% of its stall samples: MUFU.EX2 482 +
    // FADD 278, s1 census dv_7f13f039ad89) are pure functions of this
    // head's b cumsum: Ai[i] = exp(b_i), Dl[i] = exp(b63 - b_i), and
    // Ai[63] == exp(b63) == the chain's alphaL. P1 computes them ONCE per
    // (slot, head) with the SAME __expf intrinsic on the SAME fp32 values
    // -> BIT-IDENTICAL results; they ride the chain's existing plane bulk
    // (+512B/slot). sb beyond L is the flat cumsum tail (finite, |b| <~ 35)
    // so the planes are finite everywhere; the chain's rows >= L are
    // zero-masked by beta as before.
    if (tid < CDIM) {
      const float bi = sb[tid];
      (gb + 2 * CDIM)[tid] = __expf(bi);
      (gb + 3 * CDIM)[tid] = __expf(sb[CDIM - 1] - bi);
    }
    // (QKd publish hoisted before the inverse levels — V2i)
    // ensure stores of this iteration complete before next reuses sSys/sTinv:
    if (r + 1 < cnt) { __syncthreads(); }
    TICK(8);
#ifdef GDN_P1PHASE
    ph[9]++;
#endif
  }
  }
  // V3g: s2g bulk writes must be complete (globally visible) before P1
  // retires - P2 reads them under stream/PDL ordering. No-op when the
  // group already completed or was never issued.
  if (tid == 0) asm volatile("cp.async.bulk.wait_group 0;");
#ifdef GDN_P1PHASE
#undef TICK
  TAIL_REC();                           // V-J16: after the s2g drain wait =
  if (phases != nullptr) {
    if (blockIdx.x == 0 && blockIdx.y == 0) {
      if (tid == 0) {
        for (int i = 0; i < 10; i++) phases[i] = ph[i];
        for (int i = 0; i < 6; i++) phases[24 + i] = ph[10 + i];  // V-J24
        phases[20] = _P;                // V-J17a probe metadata
        phases[21] = _nblk;
        phases[22] = _na;
        phases[23] = 148;
      } else if (tid == 128) {
        for (int i = 0; i < 6; i++) phases[36 + i] = ph[10 + i];  // rho warp
      }
    } else if (_lin0 == _P) {
      if (tid == 0) {
        for (int i = 0; i < 10; i++) phases[10 + i] = ph[i];      // SOLO probe
        for (int i = 0; i < 6; i++) phases[30 + i] = ph[10 + i];
      } else if (tid == 128) {
        for (int i = 0; i < 6; i++) phases[42 + i] = ph[10 + i];
      }
    }
  }
#undef TAIL_REC
#endif
}

// ============================ P0: chain order ============================
// Longest-chain-first (LPT) request ordering for wave-tail balance. N<=32.
// ============================ P2: chain kernel ============================
// grid (VDIM/VT, HV, N), block 256. Each block walks the chunk chain of
// (request n, head hv, v-tile vt), carrying S [128, VT] fp32 in smem.
// Warp tiling for [64,VTT] outputs: mw = warp&3 (m16 rows); nw = warp>>2
// covers VTT/2 cols (VTT=16: n8; VTT=32: n16; VTT=64: 2x n16 NPER groups).
// State update [128,VTT]: warp = one m16 key-tile, VTT/8 n8 col-tiles.
// NSTAGE=2 cp.async double-buffered pipeline over chunks.
// VTT=16 halves per-chunk tensor/epilogue work per CTA and doubles grid.x:
// wins when the extra CTAs still fit one wave (host heuristic).
// V18.1: MODE template folds the segment machinery per launch class:
// MODE 0 = monolithic (EXACTLY the common-tree computation - all segment
// logic constexpr-elided, zero added instructions on the dependency-bound
// chain); MODE 1 = pass A (output steps 2q/5u/6/7 dead-code eliminated,
// publish compiled in); MODE 2 = pass C (compose compiled in, full rate).
// Segmentation is vt16-only by the host rule => MODE 1/2 instantiate for
// VTT=16 only. G/pass remain args for launch uniformity; MODE0 ignores them.
// V-J1.5 (arm-a V5.20a port): weak-decay fallback threshold for segmented
// chains (pass C). The segment composition S_true(g) = B-composition drops
// the cross-segment delta-feedback term; its relative error is O(1) when a
// segment's decay a_g = prod exp(b63) is ~1 (slow-decay regime: |b63| tiny).
// Pass A publishes a_g (segA); pass C computes amax = max_g a_g per (n,hv,vt):
//   amax <= GDN_SEG_THETA -> composed path (error bounded by the theta class;
//     arm-b official precedent v5.8/v5.9 PASS x2 + domain);
//   amax >  GDN_SEG_THETA -> EXACT serial fallback: g>0 blocks exit, g==0
//     reruns the FULL chain from state0 (seg_lo==0, nseg==chunks => bitwise
//     the monolithic MODE0 sequence incl the final-state store).
// The predicate is uniform per (n,hv): aprod is vt-invariant (built from the
// (slot,hv) Ai planes), so every block of a chain group takes the same branch.
// Content-dependent device-side branch (no D2H, no case identity): slow-decay
// content on a seg-triggering shape is legal frozen-domain input, and WITHOUT
// this fallback it silently corrupts (join-tree smoke R_seg_weak_fb: out
// max_abs 1.04e-1, 88,376 mismatches — pre-existing gap, arm-a V5.20a had
// added exactly this device-side fallback the peer arm lacked).
constexpr float GDN_SEG_THETA = 1e-3f;
#ifdef GDN_CPHASE
// Chain per-phase census (GDN_CPHASE builds ONLY; experiments/chain_phases.py).
// arm-a instrumentation ported onto the peer tree (V-J4-census): tid0 of
// block(0,0,0) stamps clock64 at chunk-step boundaries; per-phase deltas
// accumulate in registers and flush via atomicAdd at kernel end (3
// back-to-back runs sum). MODE-separated index ranges: MODE0 -> [0,16),
// MODE1 (pass A) -> [24,40), MODE2 (pass C) -> [48,64). Slot map within a
// range: 0=W(wait+sync) 1=ISS+setup 2=S2(kS/qS) 3=S3(rhs) 4=B1+ah4 5=S4(u)
// 6=S5(planes) 7=B3+ah6/at8 8=S67(attn2+7a STS) 9=S8(dS+Sreg+sShi+7b sweep)
// 10=FULL(loop only) 11=chunks(sum nseg) 12=first-chunk FULL 13=last-chunk
// FULL 14=nseg(store) 15=weak(store) 16=postloop(publish+epilogue, per-run)
// 17=prologue(entry->loop, per-run). Drain estimate = profiler wall -
// (17+10+16)/runs. Distortion: ~24 clock reads/chunk on ONE thread
// (~0.1-0.2us/chunk) — relative shares are the signal. Non-instrumented
// builds: all macros empty -> token-identical preprocessing.
#define CPH_ACT_DECL()                                                        \
  const bool cph_act = (cph != nullptr) && tid == 0 && blockIdx.x == 0 &&     \
                       blockIdx.y == 0 && blockIdx.z == 0;                    \
  const int cph_base = (MODE == 1) ? 24 : (MODE == 2) ? 48 : 0;               \
  long long cph_acc[12] = {};                                                 \
  long long cph_fi = 0, cph_la = 0;                                           \
  long long cph_gt0 = 0, cph_gtM = 0, cph_gtE = 0, cph_cycM = 0;
// V-J27 throttle probe (GDN_CPHASE builds ONLY; chain_phases.py): %globaltimer
// (ns, GPU-global) at t_entry / loop midpoint / t_kend + clock64 at midpoint.
// effective GHz = cycles/ns over each half answers the 11-18% "drain/untracked"
// question: DVFS sag under sustained dense-mma load vs boundary/drain overhead.
// Slots (LAST-RUN values, not accumulated): base+18=gt0, +19=gtE, +20=gtM,
// +21=cycM, +22=t_entry, +23=t_kend.
#define CPH_GT(v) do { if (cph_act)                                           \
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v)); } while (0)
#define CPH_MID(cl, ns) do { if (cph_act && (cl) == (ns) / 2) {               \
    cph_cycM = clock64(); CPH_GT(cph_gtM); } } while (0)
#define CPH_DECL(n) const long long n = cph_act ? clock64() : 0LL
#define CPH_TICK(i, t0) do { if (cph_act) cph_acc[i] += clock64() - (t0); } while (0)
#define CPH_ADD(p, val) \
  atomicAdd(reinterpret_cast<unsigned long long*>(p), (unsigned long long)(val))
#define CPH_FL(cl, lh, t0, tn) do { if (cph_act) {                            \
      if ((cl) == 0) cph_fi = (tn) - (t0);                                    \
      if ((cl) + 1 == (lh)) cph_la = (tn) - (t0); } } while (0)
#define CPH_FLUSH() do { if (cph_act) {                                       \
      for (int i = 0; i <= 10; i++) CPH_ADD(cph + cph_base + i, cph_acc[i]);  \
      CPH_ADD(cph + cph_base + 11, (long long)nseg);                          \
      CPH_ADD(cph + cph_base + 12, cph_fi);                                   \
      CPH_ADD(cph + cph_base + 13, cph_la);                                   \
      CPH_ADD(cph + cph_base + 16, t_kend - t_postloop);                      \
      CPH_ADD(cph + cph_base + 17, t_preloop - t_entry);                      \
      cph[cph_base + 14] = nseg;                                              \
      cph[cph_base + 15] = weak ? 1 : 0;                                      \
      cph[cph_base + 18] = cph_gt0;  cph[cph_base + 19] = cph_gtE;            \
      cph[cph_base + 20] = cph_gtM;  cph[cph_base + 21] = cph_cycM;           \
      cph[cph_base + 22] = t_entry;  cph[cph_base + 23] = t_kend; } } while (0)
#else
#define CPH_ACT_DECL()
#define CPH_DECL(n)
#define CPH_TICK(i, t0)
#define CPH_FL(cl, lh, t0, tn)
#define CPH_FLUSH()
#define CPH_GT(v)
#define CPH_MID(cl, ns)
#endif

template <int NSTAGE, int VTT, int MODE, bool KQTMA = false, bool PRE0 = false>
__global__ void __launch_bounds__(NTHR, 1)
gdn_chain(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
          const __nv_bfloat16* __restrict__ v,
          const __grid_constant__ CUtensorMap vmap,   // V5.13b (zero/unused at VTT=64)
          const __grid_constant__ CUtensorMap kmap,   // V-J2 (KQTMA only)
          const __grid_constant__ CUtensorMap qmap,   // V-J2 (KQTMA only)
          const float* __restrict__ beta,
          const void* __restrict__ state0, const int state_is_fp32,
          const int* __restrict__ cu, const int* __restrict__ chunk_offsets,
          const int* __restrict__ perm,
          const int H, const int HV,
          const char* __restrict__ scratch, const char* __restrict__ scratchq,
          __nv_bfloat16* __restrict__ out,
          __nv_bfloat16* __restrict__ core, float* __restrict__ final_state,
          const int G, const int pass,
          float* __restrict__ segB, float* __restrict__ segA
#ifdef GDN_CPHASE
          , long long* __restrict__ cph
#endif
          ) {
  // V3r: VTT>32 widens plane/v strides to 72 (64 cols + 8 pad; multiple
  // of 8 bf16 -> ldmatrix rows stay 16B-aligned). VTT<=32 byte-identical.
  constexpr int SPS = (VTT > 32) ? 72 : SP_STR;
  constexpr int USS = (VTT > 32) ? 72 : US_STR;
  // V5.13b: TMA box writes PACKED [64][VTT] rows for VTT<=32 (E1 read
  // conflicts: vt16 2-way / vt32 4-way — net LSU win per bench wavefront
  // math). V5.14a: vt64 joins the TMA path with SWIZZLE_128B (bench
  // dv_6a4643de6c8c: byte-exact chunk^(row&7) model, 158ns per 8KB box =
  // op-bound; replaces 256 cp_async16 wavefronts/chunk) — packed pitch 64
  // elems = 128B rows, swizzle keeps E1 conflict-free (vs 8-way unswizzled).
  constexpr int VSS = VTT;
  // V-J2: the KQ stage region is ALWAYS the swizzle image (4 sub-tiles,
  // 32,768B) — the bulk path stages P1's byte-identical s2g image, the
  // KQTMA path TMA-writes the same layout from the inputs; consumption is
  // unified on swz ldmatrix. KQTMA selects only the ISSUE mechanism.
  constexpr int KQBYTES = KQ_TMA_BYTES;
  constexpr int ST_BYTES =
      KQBYTES                      // sKQ: [kT0 kT1 qT0 qT1] swizzle image
      + CDIM * VSS * 2             // sV
      + 2 * PLANE_PACKED_BYTES    // sTinvH sQkdH (E4: triangle-packed; V3e)
      + 4 * CDIM * 4;              // sb, sbeta, sAi, sDl (V5.18a)
  extern __shared__ char smem_raw[];
  char* smem = SMEM_ALIGN16(smem_raw);   // V5.15a: shared-qualified (LDS/STS)
  // V3c: S lives in registers (Sreg, step8 ownership pattern) - the fp32
  // sS smem tile is gone; planes start at smem base.
  __nv_bfloat16* sShi = reinterpret_cast<__nv_bfloat16*>(smem);               // [128][SPS]
  __nv_bfloat16* sUh = sShi + 2 * CDIM * SPS;                                // [64][USS]
  __nv_bfloat16* sRHh = sUh + CDIM * USS;                                    // rhs hi plane
  __nv_bfloat16* sDUh = sRHh + CDIM * USS;                                   // decay*u hi plane
  uint64_t* loadmbar = reinterpret_cast<uint64_t*>(sDUh + CDIM * USS);  // NSTAGE load barriers
  // V5.13b-fix: TMA tensor ops require 128B-aligned smem dst (1D bulk needs
  // only 16B — why V5.12a was fine; sanitizer: "Misaligned shared address"
  // in gdn_chain at the box op, dv_3f2e5c4bb108). stV(s) = stages +
  // s*ST_BYTES + 34,816 with ST_BYTES % 128 == 0 and 34,816 % 128 == 0, so
  // 128B-aligning the stage base aligns every stage's v-box dst. Host
  // chain_smem slack (+128B) covers the padding.
  // V5.15a: offset-based 128B round-up (no uintptr_t/inttoptr round-trip) —
  // keeps NVVM addrspace(3) provenance so plain derefs of stage-derived
  // pointers (E1 sVs reads, sShi/plane access, final-state staging) emit as
  // LDS/STS (short scoreboard) instead of generic LD.E/ST.E (long scoreboard
  // + L1TEX global path). Identical address as the old (x+127)&~127 form.
  char* mbar_end = reinterpret_cast<char*>(loadmbar + NSTAGE);
  // V-J2 (lesson 19): stages at 1024B alignment (was 128B) — swizzle-phase
  // transparency for the bulk image path requires stage KQ bases == P1 sK
  // base (mod 1024); both are then phase row&7. s*ST_BYTES keeps it:
  // ST_BYTES/1024 = 53/55/59 (vt16/32/64), sub-tile offsets 8,192/16,384/
  // 24,576 all == 0 mod 1024. Arm-a bench T4: mismatched bases corrupt.
  char* stages = mbar_end +
      (int)((1024u - (reinterpret_cast<uintptr_t>(mbar_end) & 1023u)) & 1023u);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2, tig = lane & 3;
  constexpr int NPER = VTT / 16;   // n16-groups per warp (2 for 32, 1 for 16)
  constexpr int NS8 = VTT / 8;     // n8 col-tiles for state update
  const int vt = blockIdx.x;
  const int hv = blockIdx.y;
  const int vbase = vt * VTT;
  const int mw = warp & 3, nw = warp >> 2;
  // V-J2 (arm-a V5.35b, lesson 20): fast alignment-invariant swizzle
  // addressing lane constants. V-J5: extended to vt64 with a register-lean
  // variant — the 4 xg4 regs + row128a persist (5 regs; vt64 measured REG
  // ~247/255 at KSPLIT-off), while the trans helpers (xg_at, row128t) are
  // RECOMPUTED per chunk (~6 ops amortized over 4 at8 sites) instead of
  // persisting, keeping the persistent delta within the vt64 budget.
  // vt16/32 keep the full 7-reg persistent form. Spill check: any vt64
  // regression at the gate = REJECT (c0 chain +us = ptxas spill signature).
  // xg4: A-frag kg groups (2i + (lane>>4)&1); xg_at: trans-frag kg
  // ((warp*2 + (lane>>3)&1) & 7); row128a/t: the lane's row*128 terms.
  constexpr bool SWZFAST = true;
  constexpr bool SWZPERSIST_T = (VTT < 64);
  uint32_t xg4[4];
  uint32_t xg_at = 0;
  int row128a, row128t = 0;
  {
    const int l7 = lane & 7;
    #pragma unroll
    for (int i = 0; i < 4; i++)
      xg4[i] = (uint32_t)((((2 * i + ((lane >> 4) & 1)) ^ l7) << 4));
    row128a = (mw * 16 + ((lane >> 3) & 1) * 8 + l7) * 128;
    if constexpr (SWZPERSIST_T) {
      xg_at = (uint32_t)((((warp * 2 + ((lane >> 3) & 1)) & 7) ^ l7) << 4);
      row128t = (((lane >> 4) & 1) * 8 + l7) * 128;
    }
  }
  CPH_ACT_DECL();
  CPH_DECL(t_entry);   // census: prologue span start
  CPH_GT(cph_gt0);     // V-J27: globaltimer at entry

  // V3b PDL: this kernel launches with programmatic stream serialization vs
  // P1; blocks pre-launch and run P1-independent prologue while P1 drains.
  // griddepcontrol.wait = completion semantics (same data ordering as a
  // serialized launch). perm lookup needs P1 output -> waits first.
  // V5.16a: mbar init is P1-independent — hoisted above the wait so the
  // init+barrier overlaps P1's tail instead of serializing after perm.
  if (tid == 0) {
    for (int j = 0; j < NSTAGE; j++) mbar_init(smem_u32(&loadmbar[j]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
#ifndef GDN_E29_OFF
  // E29: as PDL PRIMARY of the MODE1->MODE2 seg pair, trigger dependents at
  // entry (all pass-A CTAs are resident/launched by the time the last one
  // reaches here). MODE2's griddepcontrol.wait still binds to this grid's
  // COMPLETION (segB/segA reads stay post-wait); only its dispatch ramp +
  // pre-wait prologue (mbar_init, sreg_load, weak-decay predicate inputs
  // are all input/post-wait-safe) move earlier. For MODE0/MODE2 primaries
  // this is a harmless hint (no PDL secondary follows within the forward).
  if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  // V-J3 (PRE0): one-wave vt16 chunk-0 PRE-WAIT prefetch. Host guarantees
  // MODE0, VTT=16, Gseg==1, !KQTMA, !l2norm, perm==nullptr (=> n =
  // blockIdx.z identity, no P1 output needed for the geometry). cu is an
  // INPUT tensor — PDL-legal pre-wait read (V5.8a sreg_load / V5.16a warm
  // precedents); the k/q/v boxes read the INPUTS directly = byte-identical
  // to what P1's s2g image would stage (V-J2 unification: same maps, same
  // coords) => stage-0 contents BITWISE identical, containment wire
  // unchanged. ONE arrive.expect_tx covers the FULL chunk-0 byte set
  // (lesson 10); the planes/vec bulk (P1 outputs) issues post-wait via
  // issue_loads(0, 0, planes_only). Effect: chunk-0's cold KQ+v DRAM round
  // (~0.6-0.8us under the official evict protocol; arm-a cold census:
  // prologue +0.4 / first-FULL +0.2-0.4us vs warm) overlaps P1's tail
  // instead of serializing after griddepcontrol.wait.
  const bool pre0_on = PRE0 && (perm == nullptr);
  if (pre0_on) {
    const int n0 = (int)blockIdx.z;
    const int start0 = cu[n0];
    const int qh0 = hv / (HV / H);
    if (tid == 0) {
      const uint32_t mb0 = smem_u32(&loadmbar[0]);
      bulk_expect_tx(mb0, KSQB_BYTES + 2 * PLANE_PACKED_BYTES + 4 * CDIM * 4
                        + CDIM * VTT * 2);
      tma3d_box(&kmap, stages, 0, start0, qh0, mb0);
      tma3d_box(&kmap, stages + 8192, 64, start0, qh0, mb0);
      tma3d_box(&qmap, stages + 16384, 0, start0, qh0, mb0);
      tma3d_box(&qmap, stages + 24576, 64, start0, qh0, mb0);
      tma3d_box(&vmap, stages + KQBYTES, vbase, hv, start0, mb0);
    }
  }
  // V18 (port of arm-b V4a/V4b onto the join-20261024 common tree):
  // segmented chains for the low-occupancy vt16 class. grid.z = n_idx*G + g
  // (G==1 -> identical to old blockIdx.z). pass 0 (A): runs its chunk
  // segment from state0 (g==0) or zero-state (g>0) at the cheap propagation
  // rate (no qS/attn2/out work in pass A at all), and publishes the affine
  // transfer B_g (end-state, fp32) + a_g = prod exp(b63). pass 1 (C,
  // launched only when G>1): g==0 reruns segment 0 from state0 (the exact
  // monolithic computation -> bitwise-identical outputs); g>0 composes
  // S_true(g) = sum_i (prod_{i<j<g} a_j)*B_i (B_0 subsumes state0; G=2
  // compose is the exact B_0 roundtrip -> bitwise) and reruns with full
  // outputs; g==G-1 stores final_state.
  const int n_idx = (MODE == 0) ? (int)blockIdx.z : ((int)blockIdx.z / G);
  const int g = (MODE == 0) ? 0 : ((int)blockIdx.z - n_idx * G);
  int n;
  if (perm != nullptr) {
    // V5.16a: speculatively warm L2 with the tiny index arrays BEFORE the
    // wait — cu/chunk_offsets/perm are 1-3 L2 lines each (N<=32 ints), so
    // even a wrong-index read warms the exact lines the post-wait dependent
    // loads need (census s0: first LDSM long_sb=360 = wait->perm->cu/
    // chunk_offsets->issue_loads(0), three serial DRAM steps ~600ns each;
    // warm turns steps 2-3 into ~250ns L2 hits). Values DISCARDED: perm is
    // P1 output — pre-wait content undefined by PDL, safe to read, illegal
    // to use. The wait's memory clobber blocks CSE/sinking of the real loads.
    const int warm = cu[n_idx] + cu[n_idx + 1]
                   + chunk_offsets[n_idx] + perm[n_idx];
    asm volatile("" :: "r"(warm));
    asm volatile("griddepcontrol.wait;" ::: "memory");
    n = perm[n_idx];
  } else {
    n = n_idx;
  }
  const int start = cu[n];
  const int seqlen = cu[n + 1] - start;
  const int chunks = (seqlen + CDIM - 1) / CDIM;
  const int seg_lo = (MODE == 0) ? 0 : (chunks * g) / G;
  int nseg = (MODE == 0) ? chunks : ((chunks * (g + 1)) / G - seg_lo);
  bool weak = false;   // V-J1.5: weak-decay serial fallback (MODE2-only)
  const int slot0 = chunk_offsets[n] + seg_lo;   // MODE0: seg_lo==0 folds away
  const int qh = hv / (HV / H);
  // V4b (ported): ALL outputs (incl segment 0) are emitted by the full-rate
  // pass - pass C when G>1, the single pass when G==1. Pass A runs the cheap
  // zero/state0-propagation rate on EVERY segment including g==0, cutting
  // the A->C critical path. C-g0 reruns segment 0 from state0 = the exact
  // monolithic computation -> bitwise-identical outputs => the port is
  // bitwise-neutral on every case, segmented or not.
  constexpr bool full_out = (MODE != 1);   // V18.1: compile-time

  __nv_bfloat16* stKQ = reinterpret_cast<__nv_bfloat16*>(stages);
  auto stV = [&](int s) { return stKQ + s * (ST_BYTES / 2) + KQBYTES / 2; };   // V-J2: image 32,768B
  // byte-addressed stage helpers
  auto stPlane = [&](int s, int p) {  // p: 0=TinvH 1=QkdH (V3e)
    return reinterpret_cast<__nv_bfloat16*>(stages + s * ST_BYTES +
                                            (KQBYTES / 2 + CDIM * VSS) * 2 +
                                            p * PLANE_PACKED_BYTES);
  };
  auto stB = [&](int s) {
    return reinterpret_cast<float*>(stages + s * ST_BYTES +
                                    (KQBYTES / 2 + CDIM * VSS) * 2 + 2 * PLANE_PACKED_BYTES);
  };
  auto stBeta = [&](int s) { return stB(s) + CDIM; };
  auto stAi = [&](int s) { return stB(s) + 2 * CDIM; };   // V5.18a exp(b_i)
  auto stDl = [&](int s) { return stB(s) + 3 * CDIM; };   // V5.18a exp(b63-b_i)

  auto issue_loads = [&](int c, int s, bool planes_only = false) {   // c = segment-local chunk idx (V18)
    const int tok0 = start + (seg_lo + c) * CDIM;
    const int Lc = min(CDIM, seqlen - (seg_lo + c) * CDIM);
    __nv_bfloat16* dKQ = stKQ + s * (ST_BYTES / 2);
    const char* sl = scratch + ((size_t)(slot0 + c) * HV + hv) * SLOT_BYTES;
    // V2e: two cp.async.bulk 1D copies replace the ~4350-op cp.async16 stream
    // for the contiguous regions (V-J2: q/k = the 32,768B swizzle image
    // 4 planes 4x[64][PL_STR] from the slot). bulk drain measured ~320ns/80KB
    // (TMA deep queue) vs ~3.5us/88KB queue-limited cp.async16. v/b stay cp.async.
    const uint32_t mb = smem_u32(&loadmbar[s]);   // V5.13b: shared by bulk + v box
    if (planes_only) {
      // V-J3 (PRE0): chunk-0 KQ image + v box were issued PRE-wait on this
      // same mbar under ONE expect_tx covering the full chunk-0 byte set;
      // only the planes/vec bulk (P1 outputs — illegal pre-wait) remains.
      if (tid == 0)
        bulk_g2s(reinterpret_cast<char*>(stPlane(s, 0)), sl,
                 2 * PLANE_PACKED_BYTES + 4 * CDIM * 4, mb);
      return;
    }
    if constexpr (KQTMA) {
      // V-J2 (arm-a V5.33a S1): four {64,64,1} SWIZZLE_128B boxes (kT0 kT1
      // qT0 qT1), coords (key0, tok0, qh) — k/q read DIRECT from the input
      // tensors; scratchq is never written in this regime (P1 skip_q). Each
      // sub-tile dst 8,192B at 128B-aligned offsets (8,192 % 128 == 0).
      // tok0 may run past the request end for last partial chunks —
      // containment wire: foreign finite rows are zeroed by beta/DU and
      // masked by i<Lc stores; past-T rows are hardware OOB zero-fill.
      // ONE arrive.expect_tx for boxes + planes + v (lesson 10).
      if (tid == 0) {
        bulk_expect_tx(mb, KQ_TMA_BYTES + 2 * PLANE_PACKED_BYTES + 4 * CDIM * 4
                         + CDIM * VTT * 2);
        char* d = reinterpret_cast<char*>(dKQ);
        tma3d_box(&kmap, d, 0, tok0, qh, mb);
        tma3d_box(&kmap, d + 8192, 64, tok0, qh, mb);
        tma3d_box(&qmap, d + 16384, 0, tok0, qh, mb);
        tma3d_box(&qmap, d + 24576, 64, tok0, qh, mb);
        bulk_g2s(reinterpret_cast<char*>(stPlane(s, 0)), sl,
                 2 * PLANE_PACKED_BYTES + 4 * CDIM * 4, mb);
      }
    } else if (tid == 0) {
      if constexpr (MODE == 1) {
        // V20: pass A provably never reads the q half of the KQ tile (its a1
        // ldm is full_out-DCE'd) nor the QkdH plane (sole reader ah6 is
        // full_out-gated) -> skip both dead loads: -26624B/chunk = -47.3% of
        // vt16 per-chunk bulk bytes (accounting dv_f34ef5797f64: pass A ran
        // at FULL chain rate 1.087us/chunk despite q-side DCE = load-bound).
        // Bitwise: dead loads only; K = first half of the flat [k rows 0-63;
        // q rows 64-127] KQ block; slot = [TinvH][QkdH][b/beta/Ai/Dl] so
        // TinvH bulk comes from slot+0 and the floats from slot+2*PLANE_B.
        // All dsts/srcs stay 16B-aligned; sizes are 16B multiples.
        bulk_expect_tx(mb, KSQB_BYTES / 2 + PLANE_PACKED_BYTES + 4 * CDIM * 4
                         + CDIM * VTT * 2);
        bulk_g2s(dKQ, scratchq + ((size_t)(slot0 + c) * H + qh) * KSQB_BYTES,
                 KSQB_BYTES / 2, mb);
        bulk_g2s(reinterpret_cast<char*>(stPlane(s, 0)), sl,
                 PLANE_PACKED_BYTES, mb);
        bulk_g2s(reinterpret_cast<char*>(stB(s)), sl + 2 * PLANE_PACKED_BYTES,
                 4 * CDIM * 4, mb);
      } else {
        // V5.12a: one more bulk segment — planes+b+beta are contiguous in the
        // slot (SLOT_BYTES grew by CDIM*4) and land contiguously at stPlane.
        bulk_expect_tx(mb, KSQB_BYTES + 2 * PLANE_PACKED_BYTES + 4 * CDIM * 4
                         + CDIM * VTT * 2);   // V5.13b/V5.14a: v box, all VTT
        bulk_g2s(dKQ, scratchq + ((size_t)(slot0 + c) * H + qh) * KSQB_BYTES, KSQB_BYTES, mb);
        bulk_g2s(reinterpret_cast<char*>(stPlane(s, 0)), sl,
                 2 * PLANE_PACKED_BYTES + 4 * CDIM * 4, mb);
      }
    }
    {
      // V5.13b/V5.14a: ONE TMA box op replaces CDIM*VTT/8 cp_async16
      // requests (~64 LSU wavefronts/chunk at vt16, ~256 at vt64). Bench:
      // 158ns/op pipelined regardless of 2-8KB size (op-bound), hidden
      // under the chunk; expect_tx covers the FULL box (TMA signals all 64
      // rows incl. zero-filled OOB). Rows >= Lc within T load real (finite)
      // v — rhs rows >= Lc stay exact zero via slot beta zero-fill (V5.12a).
      // Swizzle: NONE for VTT<=32, 128B for VTT=64 (host map matches).
      if (tid == 0) {
        tma3d_box(&vmap, &stV(s)[0], vbase, hv, tok0, mb);
      }
    }
    // V5.12a: b+beta now arrive inside the bulk above (stB/stBeta pointers
    // unchanged — they sit at the end of the bulk-written stage region).
  };

  // (V5.16a: mbar_init + syncthreads hoisted above the PDL wait)

  // S carried in registers across chunks, owned in the step-8 accumulator
  // pattern: Sreg[nt][half*2+c] = S[key=warp*16+gid+8*half][v=nt*8+tig*2+c].
  float Sreg[NS8][4];
  float aprod = 1.0f;   // V4a port: running prod exp(b63) over this segment
  // V5.8a: state0 -> Sreg factored so the perm==nullptr path issues the gmem
  // loads BEFORE griddepcontrol.wait — latency overlaps P1's tail (arm-a
  // V5d3 mechanism: chain −1% there). state0 is a caller input P1 never
  // writes, so the pre-wait read is stream-safe. perm path keeps the legacy
  // post-loads position (n comes from P1 output).
  auto sreg_load = [&]() {
    const size_t sofs = ((size_t)n * HV + hv) * 16384 + (size_t)vbase * KDIM;
    const __nv_bfloat16* st_bf = reinterpret_cast<const __nv_bfloat16*>(state0);
    const float* st_f = reinterpret_cast<const float*>(state0);
    const int k0 = warp * 16 + gid, k1 = k0 + 8;
    #pragma unroll
    for (int nt = 0; nt < NS8; nt++) {
      #pragma unroll
      for (int half = 0; half < 2; half++) {
        const int kk = half ? k1 : k0;
        #pragma unroll
        for (int c2 = 0; c2 < 2; c2++) {
          const int vl = nt * 8 + tig * 2 + c2;
          Sreg[nt][half * 2 + c2] = state_is_fp32
              ? st_f[sofs + (size_t)vl * KDIM + kk]
              : __bfloat162float(st_bf[sofs + (size_t)vl * KDIM + kk]);
        }
      }
    }
  };
  auto zero_sreg = [&]() {   // V18 port: pass-A g>0 zero-state init
    #pragma unroll
    for (int nt = 0; nt < NS8; nt++)
      #pragma unroll
      for (int j = 0; j < 4; j++) Sreg[nt][j] = 0.f;
  };
  if (perm == nullptr) {
    // V5.8a pre-wait state0 fetch (pure input read). V18.1: MODE0 always
    // loads (common-tree behavior); pass-A g>0 zeroes Sreg pre-wait
    // (register op, safe); pass-C g>0 composes post-wait from pass A's
    // segB/segA (PDL-ordered outputs).
    if constexpr (MODE == 0) {
      sreg_load();
    } else if constexpr (MODE == 1) {
      if (g == 0) sreg_load();
      else zero_sreg();
    } else {
      if (g == 0) sreg_load();
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");   // P1 outputs ready
  }

  // V-J1.5 (arm-a V5.20a port): pass-C weak-decay predicate + serial
  // fallback control. Placement: AFTER griddepcontrol.wait on BOTH perm
  // paths (segA is a pass-A output) and BEFORE the prologue issues (nseg
  // widens to the full chain under the fallback; g==0 => seg_lo==0 so the
  // prologue/loop index sequence becomes exactly the monolithic one).
  if constexpr (MODE == 2) {
    const float* sa = segA + ((size_t)n * HV + hv) * gridDim.x * G + vt * G;
    float amax = sa[0];
    for (int i = 1; i < G; i++) amax = fmaxf(amax, sa[i]);
    weak = (amax > GDN_SEG_THETA);
    if (weak) {
      if (g > 0) return;      // exact fallback: g==0 reruns the full chain
      nseg = chunks;
    }
  }

  // ---- prologue: kick off loads for the first NSTAGE-1 segment chunks, then
  // init state (V3o: loads(cl+NSTAGE-1) issue at the top of iter cl; state
  //  init overlaps DMA; V4a port: empty segments skip loads entirely;
  //  V21: depth generalized - NSTAGE=2 folds to the exact old single issue) ----
  #pragma unroll
  for (int p = 0; p < NSTAGE - 1; p++) {
    if (p < nseg) { issue_loads(p, p, pre0_on && p == 0); }   // V-J1a: cp_commit purged; V-J3: chunk-0 planes-only under PRE0
  }

  {
    const int k0 = warp * 16 + gid, k1 = k0 + 8;
    if constexpr (MODE == 2) {
      if (g > 0) {
      // V4a pass C: compose true segment input from pass-A transfers
      // (deterministic descending-i fp32 accumulation; ~2^-24 relative vs
      // the unsplit register evolution - negligible vs bf16 plane quant;
      // G=2 g=1 compose is the exact B_0 roundtrip -> bitwise).
      // V4a-fix: slice keyed by (n, hv, vt) - segments of DIFFERENT
      // requests collided on (hv,vt,g) for N>=2.
      const size_t sliceb =
          (((size_t)n * HV + hv) * gridDim.x + vt) * G * (2 * CDIM * VTT);
      const size_t slicea = ((size_t)n * HV + hv) * gridDim.x * G + vt * G;
      zero_sreg();
      float coef = 1.0f;
      for (int i = g - 1; i >= 1; --i) {
        const float* Bi = segB + sliceb + (size_t)i * (2 * CDIM * VTT);
        #pragma unroll
        for (int nt = 0; nt < NS8; nt++) {
          #pragma unroll
          for (int half = 0; half < 2; half++) {
            const size_t key = (size_t)(half ? k1 : k0) * VTT + nt * 8 + tig * 2;
            Sreg[nt][half * 2 + 0] += coef * Bi[key + 0];
            Sreg[nt][half * 2 + 1] += coef * Bi[key + 1];
          }
        }
        coef *= segA[slicea + i];
      }
      {
        const float* B0 = segB + sliceb;
        #pragma unroll
        for (int nt = 0; nt < NS8; nt++) {
          #pragma unroll
          for (int half = 0; half < 2; half++) {
            const size_t key = (size_t)(half ? k1 : k0) * VTT + nt * 8 + tig * 2;
            Sreg[nt][half * 2 + 0] += coef * B0[key + 0];
            Sreg[nt][half * 2 + 1] += coef * B0[key + 1];
          }
        }
      }
      } else if (perm != nullptr) {
        sreg_load();
      }
    } else if constexpr (MODE == 1) {
      if (perm != nullptr) {
        if (g == 0) sreg_load();
        else zero_sreg();
      }
    } else {
      if (perm != nullptr) sreg_load();   // common-tree behavior (MODE0)
    }
    #pragma unroll
    for (int nt = 0; nt < NS8; nt++) {
      #pragma unroll
      for (int half = 0; half < 2; half++) {
        const int kk = (half ? k1 : k0);
        const int vg = nt * 8 + tig * 2;
        __nv_bfloat162 hp = {f_hi(Sreg[nt][half * 2 + 0]),
                             f_hi(Sreg[nt][half * 2 + 1])};
        *reinterpret_cast<__nv_bfloat162*>(&sShi[kk * SPS + vg]) = hp;
      }
    }
    __syncthreads();
  }

  CPH_DECL(t_preloop);   // census: prologue span end
  // ---- chunk chain (V18: segment-local cl; global chunk = seg_lo + cl) ----
  for (int cl = 0; cl < nseg; cl++) {
    const int c = seg_lo + cl;
    const int s = cl % NSTAGE;
    const int Lc = min(CDIM, seqlen - c * CDIM);
    const int tok0 = start + c * CDIM;
    CPH_MID(cl, nseg);   // V-J27: midpoint throttle probe
    CPH_DECL(tw);
    // V-J1a (arm-a V5.30a port): cp groups PURGED — every chain load is
    // bulk/TMA tracked on loadmbar since V2e/V5.13b, so cp_commit created
    // empty groups and this cp_wait<0> waited on nothing (measured vt16 win
    // in the arm-a V5.29a bundle bisect; mbar_wait below is the ONLY
    // completion mechanism, __syncthreads the only visibility fence).
    mbar_wait(smem_u32(&loadmbar[s]), (uint32_t)((cl / NSTAGE) & 1));  // bulk stage data
    __syncthreads();
    CPH_DECL(ti); CPH_TICK(0, tw);
    // V3o: issue loads(cl+NSTAGE-1) AFTER the top sync - stage
    // (cl+NSTAGE-1)%NSTAGE == (cl-1)%NSTAGE was last read by iteration cl-1,
    // whose reads all completed before this sync. Replaces the end-of-loop B4
    // barrier + issue (net -1 barrier per chunk). The peer's "hoist measured
    // NET SLOWER" note was about hoisting WITHIN the iteration (needs an
    // extra barrier); this variant removes one instead.
    // V21: depth NSTAGE-1 (was hardcoded 1 at NSTAGE=2 - folds identically);
    // at NSTAGE=3 the load gets TWO iterations of latency slack (issued top
    // of cl, waited top of cl+2).
    if (cl + NSTAGE - 1 < nseg) { issue_loads(cl + NSTAGE - 1, (cl + NSTAGE - 1) % NSTAGE); }   // V-J1a: cp_commit purged

    const __nv_bfloat16* sKQ = stKQ + s * (ST_BYTES / 2);
    // V-J2: unified swizzle consumption — k sub-tile0 at the stage KQ
    // region base, q sub-tile0 at +16,384B (bulk path = P1's byte-identical
    // s2g image; KQTMA = TMA-written same layout). Per-chunk rowbase hoists
    // (lesson 20): rba for a0/a1, rbt folds the trans sub-tile immediate.
    const uint32_t kq_base = smem_u32(sKQ);
    uint32_t rba = 0u, rbt = 0u;
    if constexpr (SWZFAST) {
      rba = kq_base + (uint32_t)row128a;
      if constexpr (SWZPERSIST_T) {
        rbt = kq_base + (uint32_t)row128t + (uint32_t)((warp >> 2) * 8192);
      } else {
        // V-J5: vt64 register-lean trans helpers — recomputed per chunk
        // (~6 ops, amortized over the 4 at8 sites) instead of 2 persistent
        // regs (REG cap). Same values as the persistent form.
        const int l7c = lane & 7;
        xg_at = (uint32_t)((((warp * 2 + ((lane >> 3) & 1)) & 7) ^ l7c) << 4);
        rbt = kq_base + (uint32_t)((((lane >> 4) & 1) * 8 + l7c) * 128)
            + (uint32_t)((warp >> 2) * 8192);
      }
    }
    const __nv_bfloat16* sVs = stV(s);
    // V5.14a-fix: SWIZZLE_128B phase of this stage's v tile (0 unless VTT=64)
    const int vph = (VTT == 64) ? (int)((smem_u32(&sVs[0]) >> 7) & 7) : 0;
    const __nv_bfloat16* sTinvH = stPlane(s, 0);
    const __nv_bfloat16* sQkdH = stPlane(s, 1);
    const float* sbeta = stBeta(s);
    const float* sAi = stAi(s);   // V5.18a: P1-precomputed exp(b_i)
    const float* sDl = stDl(s);   // V5.18a: P1-precomputed exp(b63 - b_i)
    // per-lane decay values (rows this lane's accumulators cover) —
    // plane LDS replaces 4x __expf + arg FADDs (bit-identical values);
    // sAi[63] == exp(b63) == alphaL (used in the S-update below).
    const int r0 = mw * 16 + gid;
    const float Ai0 = sAi[r0], Ai1 = sAi[r0 + 8];
    const float AiQ0 = Ai0 * QSCALE, AiQ1 = Ai1 * QSCALE;
    const float Dl0 = sDl[r0], Dl1 = sDl[r0 + 8];
    CPH_DECL(t2); CPH_TICK(1, ti);

    // step 2: combined kS/qS = [k;q] @ (S_hi + S_lo)
    // V2j: qS keeps ONLY the S_hi term. qS feeds just o = AiQ*qS + attn2 with
    // AiQ = exp(b_i)*QSCALE <= 0.088, and large-AiQ (early tokens) coincides
    // with small carried |S| — the dropped q@S_lo contributes <~1e-4 abs to o
    // (measured margins >=4x inside tolerance across all suites). kS keeps
    // both planes (feeds rhs -> u -> state). Saves 16 ldm + 32 mma per warp.
    // V5.8a: ks-parity accumulator split for VTT<=32 (arm-a V10b mechanism:
    // depth-8 chains -> 2x depth-4; bought chain −2% same-job there).
    // VTT=64 (REG 247/255, headroom 8) keeps the unsplit path via constexpr;
    // accKx/accQx fold away there. Merges at the step-3/step-7 use sites
    // (fp32 regroup ~1e-7 rel, arm-a precedent passed official ×2 seeds).
    constexpr bool KSPLIT = (VTT <= 32);
    float accK[NPER][4] = {};
    float accQ[NPER][4] = {};
    float accKx[NPER][4] = {};
    float accQx[NPER][4] = {};
    {
      // V3d-1: state planes hi-only (single-plane bf16 S operand; the
      // dropped S_lo term is ~2^-9 relative on kS - v4-prejoin numerical
      // class, official-PASS precedent with >=13x fin margin).
      #pragma unroll
      for (int ks = 0; ks < KDIM / 16; ks++) {
        uint32_t a0[4], a1[4], b[(NPER > 2 ? 2 * NPER : 4)];
        // V-J2: fast swz = ONE IADD3/site (rba + xg4[ks&3] + sub-tile/q
        // immediates); generic swz_addr kept for vt64 (SWZFAST off, REG cap).
        if constexpr (SWZFAST) ldm_swzf(a0, rba + xg4[ks & 3] + ((ks >= 4) ? 8192u : 0u));
        else ldm_a_swz(a0, kq_base, mw * 16, ks * 16);
        // V4a port: pass A skips the qS half (feeds only step-7 outputs)
        if constexpr (full_out) {
          if constexpr (SWZFAST) ldm_swzf(a1, rba + xg4[ks & 3] + ((ks >= 4) ? 24576u : 16384u));
          else ldm_a_swz(a1, kq_base + 16384, mw * 16, ks * 16);
        }
        float (*dK)[4] = (KSPLIT && (ks & 1)) ? accKx : accK;
        float (*dQ)[4] = (KSPLIT && (ks & 1)) ? accQx : accQ;
        if constexpr (NPER == 1) {
          // V3j: x2 B-frag (n8) - half the ldmatrix bytes, identical regs
          ldm_b1_kmajor(b, sShi, SPS, ks * 16, nw * (VTT / 2));
          mma_bf16(dK[0], a0, b);
          if constexpr (full_out) mma_bf16(dQ[0], a1, b);
        } else {
          // V3r-fix: NPER=4 spans 32 cols -> NPER/2 x4-frags at n0+q*16
          // (NPER=2: single iteration, b+0/n0+0 = bitwise identical).
          #pragma unroll
          for (int q = 0; q < NPER / 2; q++)
            ldm_b2_kmajor(b + 4 * q, sShi, SPS, ks * 16, nw * (VTT / 2) + q * 16);
          #pragma unroll
          for (int nt = 0; nt < NPER; nt++) {
            mma_bf16(dK[nt], a0, b + 2 * nt);
            if constexpr (full_out) mma_bf16(dQ[nt], a1, b + 2 * nt);
          }
        }
      }
    }

    CPH_DECL(t3); CPH_TICK(2, t2);
    // V-J10b: mini-chunk RESTRICTED to vt64. Measured (dv_c2702d056562 +
    // dv_00ad0b7ee509, both bitwise): vt64 c0 chain 194.6 -> 182.0 (−12.7µs,
    // deterministic ×2 runs, total-backed — exceeds the 2 mini-chunks' own
    // ~−3µs, so part of the win is a vt64 codegen/register-allocation shift;
    // RE-VERIFY at every gate), while vt16 pays a code-size tax in the hot
    // loop (+27ns/chunk on 256-chunk c1/X = +7µs; +0.3..0.8µs on floors)
    // that exceeds the single-mini-chunk saving, and vt32 is ~neutral.
    // constexpr restriction => vt16/32 compile the EXACT V-J9 loop (tax
    // gone); domain d0/d2 (vt32, mini-dominated) forgo the benefit —
    // domain is reported, fixed-suite decides.
    if constexpr (VTT == 64) {
    if (Lc == 1) {
      // ---- V-J10: Lc==1 MINI-CHUNK (BITWISE vs the full path) ----
      // Every row >= 1 quantity of the full path is EXACTLY zero/identity
      // under the containment fills: sbeta_i>=1 = 0 => rhs_i>=1 = 0; Tinv
      // rows >= 1 = e_i => u_i>=1 = 0 => DU_i>=1 = 0 => dS = k0^T (x) DU_0
      // (rank-1); Tinv row 0 = e_0 EXACT => u_0 = rhs_0 through the same
      // single bf16 rounding as the full path's sUh plane (u0f = fp32 of
      // f_hi(rhs_0)); QKd row 0 zero except col 0 => attn2_00 = QKd_00*u_0
      // is ONE mma term (bf16xbf16 products are exact in fp32; sums of one
      // nonzero term + exact zeros/ones are bitwise); o rows >= 1 store-
      // masked. Skips: plane STS/LDS traffic (sRHh row 0 = DU_0 scratch),
      // B1+B3 barriers (one sync publishes DU_0), ah4/ah6/at8 frags, steps
      // 4/6/8 mmas, 7a staging + 7b sweep (o_0 stored direct). Saves
      // ~0.4-0.6us per Lc=1 tail chunk: c2/c5/c7 (2049=64k+1) and c6
      // (4097=64k+1) carry exactly one each; c0 two; domain d0 [1] / d2
      // [1x32] / d3 (65,129 tails) are mini-dominated.
      if (mw == 0 && gid == 0) {
        // row-0 owners: lanes 0-3 of each nw warp hold row 0 (c0/c1 accumulator
        // entries) spanning cols nw*(VTT/2) + nt*8 + tig*2 + {0,1}.
        const float be0 = sbeta[0];
        const float Aiv = sAi[0], Dl0v = sDl[0];
        const float qkd00 = __bfloat162float(sQkdH[0]);
        #pragma unroll
        for (int nt = 0; nt < NPER; nt++) {
          const int vg = nw * (VTT / 2) + nt * 8 + tig * 2;
          const int voff0 = (VTT == 64)
              ? ((((vg >> 3) ^ (vph & 7)) << 3) | (vg & 7)) : vg;   // i=0 row
          const __nv_bfloat162 vv =
              *reinterpret_cast<const __nv_bfloat162*>(&sVs[voff0]);
          const float kS0 = KSPLIT ? accK[nt][0] + accKx[nt][0] : accK[nt][0];
          const float kS1 = KSPLIT ? accK[nt][1] + accKx[nt][1] : accK[nt][1];
          const float r0 = be0 * (__bfloat162float(vv.x) - Aiv * kS0);
          const float r1 = be0 * (__bfloat162float(vv.y) - Aiv * kS1);
          const __nv_bfloat16 ub0 = f_hi(r0), ub1 = f_hi(r1);   // sUh-equal rounding
          const float u0f = __bfloat162float(ub0);
          const float u1f = __bfloat162float(ub1);
          if constexpr (full_out) {
            const float AiQv = Aiv * QSCALE;
            const float qS0 = KSPLIT ? accQ[nt][0] + accQx[nt][0] : accQ[nt][0];
            const float qS1 = KSPLIT ? accQ[nt][1] + accQx[nt][1] : accQ[nt][1];
            const __nv_bfloat162 pr = {f_hi(AiQv * qS0 + qkd00 * u0f),
                                       f_hi(AiQv * qS1 + qkd00 * u1f)};
            const size_t off = ((size_t)tok0 * HV + hv) * VDIM + vbase + vg;
            *reinterpret_cast<__nv_bfloat162*>(out + off) = pr;
            if (core != nullptr) *reinterpret_cast<__nv_bfloat162*>(core + off) = pr;
          }
          const __nv_bfloat162 dh = {f_hi(Dl0v * u0f), f_hi(Dl0v * u1f)};  // S5-equal
          *reinterpret_cast<__nv_bfloat162*>(&sRHh[vg]) = dh;   // DU_0 scratch (row 0)
        }
      }
      __syncthreads();   // publish DU_0 (replaces B1+B3 for this chunk)
      float accS[NS8][4] = {};
      {
        // rank-1 dS: accS[nt][half*2+c] = k0[key_half] * DU0[v]; k0 row of the
        // swizzle image is phase-linear (row&7 == 0 => XOR term = kg only).
        const char* img = reinterpret_cast<const char*>(sKQ);
        float k0f[2];
        #pragma unroll
        for (int half = 0; half < 2; half++) {
          const int key = warp * 16 + gid + 8 * half;
          const char* kb = img + (key >> 6) * 8192 + (((key & 63) >> 3) << 4)
                             + ((key & 7) << 1);
          k0f[half] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(kb));
        }
        #pragma unroll
        for (int nt = 0; nt < NS8; nt++) {
          const __nv_bfloat162 du =
              *reinterpret_cast<const __nv_bfloat162*>(&sRHh[nt * 8 + tig * 2]);
          const float d0 = __bfloat162float(du.x), d1 = __bfloat162float(du.y);
          accS[nt][0] = k0f[0] * d0;
          accS[nt][1] = k0f[0] * d1;
          accS[nt][2] = k0f[1] * d0;
          accS[nt][3] = k0f[1] * d1;
        }
      }
      {
        // shared tail (identical code/values to the full path's step-8 update)
        const float alphaL = sAi[CDIM - 1];
        if constexpr (MODE == 1) aprod *= alphaL;   // V18.1 pass-A scalar
        #pragma unroll
        for (int nt = 0; nt < NS8; nt++) {
          const int vg = nt * 8 + tig * 2;
          #pragma unroll
          for (int half = 0; half < 2; half++) {
            const int key = warp * 16 + gid + 8 * half;
            float x0 = alphaL * Sreg[nt][half * 2 + 0] + accS[nt][half * 2 + 0];
            float x1 = alphaL * Sreg[nt][half * 2 + 1] + accS[nt][half * 2 + 1];
            Sreg[nt][half * 2 + 0] = x0;
            Sreg[nt][half * 2 + 1] = x1;
            __nv_bfloat162 hp = {f_hi(x0), f_hi(x1)};   // V3d-1: hi-only plane
            *reinterpret_cast<__nv_bfloat162*>(&sShi[key * SPS + vg]) = hp;
          }
        }
      }
      CPH_DECL(t4); CPH_TICK(3, t3);
      CPH_DECL(t5); CPH_TICK(4, t4);
      CPH_DECL(t6); CPH_TICK(5, t5);
      CPH_DECL(t7); CPH_TICK(6, t6);
      CPH_DECL(t8); CPH_TICK(7, t7);
      CPH_DECL(t9); CPH_TICK(8, t8);
      CPH_DECL(ta); CPH_TICK(9, t9); CPH_TICK(10, tw); CPH_FL(cl, nseg, tw, ta);
      continue;
    }
    }  // V-J10b: mini-chunk vt64-only (constexpr)
    // step 3: rhs = beta * (v - A * kS) -> hi/lo planes
    {
      const float be0 = sbeta[mw * 16 + gid], be1 = sbeta[mw * 16 + gid + 8];
      #pragma unroll
      for (int nt = 0; nt < NPER; nt++) {
        int vg = nw * (VTT / 2) + nt * 8 + tig * 2;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
          int i = mw * 16 + gid + 8 * half;
          float Ai = half ? Ai1 : Ai0, be = half ? be1 : be0;
          // V5.14a-fix: vt64 rows are SWIZZLE_128B — the HW phase derives
          // from the DEST ADDRESS (chunk ^= ((dst/128 + row) & 7), i.e.
          // addr[6:4]^addr[9:7]); the bench passed at phase 0 only because
          // its tile sat 1024B-aligned. Compute the phase at runtime from
          // the actual stV address (vph, hoisted per chunk) instead of
          // assuming alignment. vt16/32 stay linear-packed (SWIZZLE_NONE).
          const int voff = (VTT == 64)
              ? i * 64 + ((((vg >> 3) ^ ((i + vph) & 7)) << 3) | (vg & 7))
              : i * VSS + vg;
          const __nv_bfloat162 vv = *reinterpret_cast<const __nv_bfloat162*>(&sVs[voff]);
          float v0 = __bfloat162float(vv.x);
          float v1 = __bfloat162float(vv.y);
          const float kS0 = KSPLIT ? accK[nt][half * 2 + 0] + accKx[nt][half * 2 + 0]
                                   : accK[nt][half * 2 + 0];
          const float kS1 = KSPLIT ? accK[nt][half * 2 + 1] + accKx[nt][half * 2 + 1]
                                   : accK[nt][half * 2 + 1];
          float r0 = be * (v0 - Ai * kS0);
          float r1 = be * (v1 - Ai * kS1);
          __nv_bfloat162 rh = {f_hi(r0), f_hi(r1)};   // V3d-2: rhs hi-only
          *reinterpret_cast<__nv_bfloat162*>(&sRHh[i * USS + vg]) = rh;
        }
      }
    }
    CPH_DECL(t4); CPH_TICK(3, t3);
    // V3l: hoist step-4 A-frags (TinvH is stage data, stable since the
    // loop-top sync) above B1 - barrier convergence overlaps the LSU issues.
    uint32_t ah4[4][4];
    #pragma unroll
    for (int ks = 0; ks <= 3; ks++) {
      if (ks > mw) break;
      ldm_a_pk(ah4[ks], sTinvH + tri_tile(mw, ks));   // E4b half-split packed tile
    }
    __syncthreads();  // B1
    CPH_DECL(t5); CPH_TICK(4, t4);

    // step 4: u = Tinv @ rhs; Tinv unit-lower-triangular => k-steps beyond
    // this warp's m-tile contribute exactly zero and are skipped.
    // V3d-2: u = TinvH @ rhsH only. V3e: TinvL no longer published.
    float accU[NPER][4] = {};
    #pragma unroll
    for (int ks = 0; ks <= 3; ks++) {
      if (ks > mw) break;
      uint32_t bh[(NPER > 2 ? 2 * NPER : 4)];
      if constexpr (NPER == 1) {   // V3j: x2 frag
        ldm_b1_kmajor(bh, sRHh, USS, ks * 16, nw * (VTT / 2));
        mma_bf16(accU[0], ah4[ks], bh);
      } else {
        #pragma unroll
        for (int q = 0; q < NPER / 2; q++)   // V3r-fix: 32 cols at NPER=4
          ldm_b2_kmajor(bh + 4 * q, sRHh, USS, ks * 16, nw * (VTT / 2) + q * 16);
        #pragma unroll
        for (int nt = 0; nt < NPER; nt++) mma_bf16(accU[nt], ah4[ks], bh + 2 * nt);
      }
    }

    CPH_DECL(t6); CPH_TICK(5, t5);
    // step 5: write u planes and du = D_last*u planes (dedicated buffers; rhs stays)
    {
      #pragma unroll
      for (int nt = 0; nt < NPER; nt++) {
        int vg = nw * (VTT / 2) + nt * 8 + tig * 2;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
          int i = mw * 16 + gid + 8 * half;
          float u0 = accU[nt][half * 2 + 0];
          float u1 = accU[nt][half * 2 + 1];
          // V3a: u-lo plane dead (step6 keeps Qh@Uh only) - hi stores suffice.
          // V4a port: sUh feeds only step 6 -> skipped on pass A.
          if constexpr (full_out) {
            __nv_bfloat162 uh = {f_hi(u0), f_hi(u1)};
            *reinterpret_cast<__nv_bfloat162*>(&sUh[i * USS + vg]) = uh;
          }
          float d = half ? Dl1 : Dl0;
          __nv_bfloat162 dh = {f_hi(d * u0), f_hi(d * u1)};   // V3d-1
          *reinterpret_cast<__nv_bfloat162*>(&sDUh[i * USS + vg]) = dh;
        }
      }
    }
    CPH_DECL(t7); CPH_TICK(6, t6);
    // V3l: hoist step-6 (QkdH) and step-8 (K^T from sKQ) A-frags above B3 -
    // both read stage data stable since the loop-top sync.
    // V4a port: ah6 serves only step 6 -> hoisted under full_out only.
    uint32_t ah6[4][4], at8[4][4];
    if constexpr (full_out) {
      #pragma unroll
      for (int ks = 0; ks <= 3; ks++) {
        if (ks > mw) break;
        ldm_a_pk(ah6[ks], sQkdH + tri_tile(mw, ks));   // E4b half-split packed tile
      }
    }
    #pragma unroll
    for (int ks = 0; ks < CDIM / 16; ks++) {
      // V-J2 fast form: rbt folds row128t + sub-tile imm; srow*128 = ks*2048
      // immediate; xg_at = lane-constant trans kg group.
      if constexpr (SWZFAST) ldm_trans_swzf(at8[ks], rbt + xg_at + (uint32_t)(ks * 2048));
      else ldm_a_trans_swz(at8[ks], kq_base, ks * 16, warp * 16);
    }
    __syncthreads();  // B3
    CPH_DECL(t8); CPH_TICK(7, t7);

    if constexpr (full_out) {
    // step 6: attn2 = QKd @ u (V3a: Qh@Uh only; V2f already dropped Ql@Uh,
    // and the removed Qh@Ul term is the same ~2^-9 relative class on attn2,
    // OUT-path only - the state path (du planes) keeps both hi+lo).
    // QKd lower-triangular => k-steps beyond this warp's m-tile are
    // exactly zero and skipped.
    float accA[NPER][4] = {};
    #pragma unroll
    for (int ks = 0; ks <= 3; ks++) {
      if (ks > mw) break;
      uint32_t bh[(NPER > 2 ? 2 * NPER : 4)];
      if constexpr (NPER == 1) {   // V3j: x2 frag
        ldm_b1_kmajor(bh, sUh, USS, ks * 16, nw * (VTT / 2));
        mma_bf16(accA[0], ah6[ks], bh);
      } else {
        #pragma unroll
        for (int q = 0; q < NPER / 2; q++)   // V3r-fix: 32 cols at NPER=4
          ldm_b2_kmajor(bh + 4 * q, sUh, USS, ks * 16, nw * (VTT / 2) + q * 16);
        #pragma unroll
        for (int nt = 0; nt < NPER; nt++) mma_bf16(accA[nt], ah6[ks], bh + 2 * nt);
      }
    }

    // step 7a (V-J1b: arm-a V5.31a-fix/b port): o = A*scale*qS + attn2,
    // STAGED into sRHh at stride USS instead of scattered STG.32 — sRHh is
    // dead after step-4's B-frag reads and B3 already separates those reads
    // from this write (ZERO extra barrier). Legacy path issued 2*NPER
    // scattered STG.32/thread (8-row sectors); stride USS keeps STS.32
    // conflict-free in every class (word 20*gid+tig vt16/32; 36*gid+16nw+
    // 4nt+tig vt64 — 32 distinct banks; US_STR=40 identical in both trees).
    // The coalesced STG.128 sweep runs after step 8 (step 7b). VALUES
    // BITWISE IDENTICAL (same o0/o1 regs -> same bf16 bytes, store path
    // only). Arm-a evidence: official c0 −14.3us; gates 0.9871/0.9850 with
    // all chain kernel deltas ≤ 0 (dv_7cace6e949ce lineage).
    {
      #pragma unroll
      for (int nt = 0; nt < NPER; nt++) {
        #pragma unroll
        for (int half = 0; half < 2; half++) {
          int i = mw * 16 + gid + 8 * half;
          if (i < Lc) {
            float AiQ = half ? AiQ1 : AiQ0;
            const float qS0 = KSPLIT ? accQ[nt][half * 2 + 0] + accQx[nt][half * 2 + 0]
                                     : accQ[nt][half * 2 + 0];
            const float qS1 = KSPLIT ? accQ[nt][half * 2 + 1] + accQx[nt][half * 2 + 1]
                                     : accQ[nt][half * 2 + 1];
            float o0 = AiQ * qS0 + accA[nt][half * 2 + 0];
            float o1 = AiQ * qS1 + accA[nt][half * 2 + 1];
            __nv_bfloat162 pr = {f_hi(o0), f_hi(o1)};
            *reinterpret_cast<__nv_bfloat162*>(
                &sRHh[i * USS + nw * (VTT / 2) + nt * 8 + tig * 2]) = pr;
          }
        }
      }
    }
    }  // full_out (V4a port: steps 6+7 skipped on pass A)
    CPH_DECL(t9); CPH_TICK(8, t8);

    // step 8: dS = K^T @ du ; S = exp(b63)*S + dS ; rebuild S planes
    float accS[NS8][4] = {};
    #pragma unroll
    for (int ks = 0; ks < CDIM / 16; ks++) {
      // V3l: A-frags hoisted above B3 (at8); du hi plane only (V3d-1).
      #pragma unroll
      for (int nb = 0; nb < VTT / 16; nb++) {
        uint32_t b0[4];
        ldm_b2_kmajor(b0, sDUh, USS, ks * 16, nb * 16);
        mma_bf16(accS[2 * nb], at8[ks], b0);
        mma_bf16(accS[2 * nb + 1], at8[ks], b0 + 2);
      }
    }
    {
      const float alphaL = sAi[CDIM - 1];   // V5.18a: == exp(b63), from plane
      if constexpr (MODE == 1) aprod *= alphaL;   // V18.1: pass-A transfer scalar
      #pragma unroll
      for (int nt = 0; nt < NS8; nt++) {
        int vg = nt * 8 + tig * 2;
        #pragma unroll
        for (int half = 0; half < 2; half++) {
          int key = warp * 16 + gid + 8 * half;
          float x0 = alphaL * Sreg[nt][half * 2 + 0] + accS[nt][half * 2 + 0];
          float x1 = alphaL * Sreg[nt][half * 2 + 1] + accS[nt][half * 2 + 1];
          Sreg[nt][half * 2 + 0] = x0;
          Sreg[nt][half * 2 + 1] = x1;
          __nv_bfloat162 hp = {f_hi(x0), f_hi(x1)};   // V3d-1: hi-only plane
          *reinterpret_cast<__nv_bfloat162*>(&sShi[key * SPS + vg]) = hp;
        }
      }
    }

    // step 7b (V-J1b: arm-a V5.34a port): WARP-LOCAL coalesced o sweep.
    // Ownership: warp (mw,nw) staged EXACTLY rows mw*16..+15 x cols
    // nw*(VTT/2)..+VTT/2-1 in step 7a — the sweep reads only its OWN warp's
    // STS bytes, so __syncwarp() suffices (NO block barrier) and all 8 warps
    // sweep in parallel (legacy scattered STG.32 ran on every thread with
    // 8-row sectors). Per warp: VTT 16B chunks; vt16 lanes 0-15 one each,
    // vt32 32x1, vt64 x2. Bitwise-safe: identical staged bytes, identical
    // gmem addresses/values; out/core row stride 256B and every col offset
    // (vbase, nw*VTT/2, cc*8) is a 16B multiple => STG.128 aligned; LDS.128
    // side: USS*2 = 80/144B row pitch, both 16B multiples. Retirement: next
    // sRHh writer (step 3 of the next chunk) sits behind the loop-top sync.
    if constexpr (full_out) {
      __syncwarp();
      constexpr int CSW = (VTT == 16) ? 0 : ((VTT == 32) ? 1 : 2);
      #pragma unroll
      for (int j = lane; j < VTT; j += 32) {
        const int i = mw * 16 + (j >> CSW);
        const int cc = j & ((1 << CSW) - 1);
        if (i < Lc) {
          const uint4 ov = *reinterpret_cast<const uint4*>(
              &sRHh[i * USS + nw * (VTT / 2) + cc * 8]);
          const size_t off = ((size_t)(tok0 + i) * HV + hv) * VDIM + vbase +
                             nw * (VTT / 2) + cc * 8;
          *reinterpret_cast<uint4*>(out + off) = ov;
          if (core != nullptr) *reinterpret_cast<uint4*>(core + off) = ov;
        }
      }
    }

    CPH_DECL(ta); CPH_TICK(9, t9); CPH_TICK(10, tw); CPH_FL(cl, nseg, tw, ta);
    // V3o: B4 removed - both duties (stage reads retired before re-issue;
    // S-plane rebuild visible cross-warp) are discharged by the NEXT
    // iteration's top syncthreads, after which loads(cl+1) issues.
  }

  CPH_DECL(t_postloop);   // census: publish+epilogue span start
  // ---- V4a port: pass A publishes the affine segment transfer ----
  // B_g = segment end-state given its pass-A input (state0 for g==0, zero
  // for g>0); a_g = prod exp(b63) over the segment (1 if empty). Pass C
  // composes S_true(g) = coef*B_0 + sum_i coef_i*B_i from these. Store
  // layout per (n, hv, vt, g): [2*CDIM][VTT] fp32 rows key-major, matching
  // the Sreg ownership pattern (float2 pairs at even v-offsets).
  if constexpr (MODE == 1) {   // V18.1: pass A only (segB guaranteed non-null)
    const size_t sliceb =
        (((size_t)n * HV + hv) * gridDim.x + vt) * G * (2 * CDIM * VTT)
        + (size_t)g * (2 * CDIM * VTT);
    float* Bp = segB + sliceb;
    #pragma unroll
    for (int nt = 0; nt < NS8; nt++) {
      #pragma unroll
      for (int half = 0; half < 2; half++) {
        const size_t key =
            (size_t)(warp * 16 + gid + 8 * half) * VTT + nt * 8 + tig * 2;
        *reinterpret_cast<float2*>(&Bp[key]) = {Sreg[nt][half * 2 + 0],
                                                Sreg[nt][half * 2 + 1]};
      }
    }
    if (tid == 0)
      segA[((size_t)n * HV + hv) * gridDim.x * G + vt * G + g] = aprod;
  }

  // ---- final state store: final[n,hv,v,k] = S[k,v] ----
  // V18.1: MODE0 stores when requested (common-tree behavior); MODE2 only
  // on the LAST segment (true end state); MODE1 never.
  // V-J1.5: under the weak-decay fallback the g==0 FULL-chain rerun holds
  // the true end state (all g>0 blocks exited) and stores it.
  if (final_state != nullptr &&
      (MODE == 0 || (MODE == 2 && (weak ? (g == 0) : (g == G - 1))))) {
    float* fdst = final_state + ((size_t)n * HV + hv) * 16384 + (size_t)vbase * KDIM;
    // stage through the free stage-0 smem (chain done) for coalesced float4.
    // V5.16a: ODD pitch VTT+1 — at pitch VTT every read quad was 32-way
    // bank-conflicted ((4*lane+i)*VTT mod 32 = const for all lanes, since
    // VTT in {16,32,64} makes 4*VTT a multiple of 32); VTT+1 leaves 4-way.
    // Max staging = 127*(VTT+1)+VTT words = 33,276B (vt64) <= ST_BYTES.
    constexpr int EPITCH = VTT + 1;
    float* stg = reinterpret_cast<float*>(stages);
    #pragma unroll
    for (int nt = 0; nt < NS8; nt++) {
      #pragma unroll
      for (int half = 0; half < 2; half++) {
        const int kk = warp * 16 + gid + 8 * half;
        const int vg = nt * 8 + tig * 2;
        stg[kk * EPITCH + vg] = Sreg[nt][half * 2 + 0];
        stg[kk * EPITCH + vg + 1] = Sreg[nt][half * 2 + 1];
      }
    }
    __syncthreads();
    // V5.16a: two-stage software pipeline — LDS quad(j+1) issues before
    // STG(j) so load latency overlaps the store (census s0: epilogue
    // STG.E.128 short_sb=476 + VIADD 251 ~ 5% of chain stall samples;
    // epilogue dominates short-chain tail blocks: case0 has 4-chunk reqs).
    constexpr int EITER = VTT * 32 / NTHR;   // 2 / 4 / 8 for vt16/32/64
    auto ldq = [&](int j) -> float4 {
      const int e = tid + j * NTHR;
      const int vl = e >> 5, kq = (e & 31) << 2;
      float4 f4;
      f4.x = stg[(kq + 0) * EPITCH + vl];
      f4.y = stg[(kq + 1) * EPITCH + vl];
      f4.z = stg[(kq + 2) * EPITCH + vl];
      f4.w = stg[(kq + 3) * EPITCH + vl];
      return f4;
    };
    float4 cur = ldq(0);
    #pragma unroll
    for (int j = 0; j < EITER; j++) {
      const int e = tid + j * NTHR;
      const int vl = e >> 5, kq = (e & 31) << 2;
      float4 nxt;
      if (j + 1 < EITER) nxt = ldq(j + 1);
      *reinterpret_cast<float4*>(fdst + (size_t)vl * KDIM + kq) = cur;
      cur = nxt;
    }
  }
  CPH_DECL(t_kend);   // census: postloop span end (retirement drain NOT
                      // included — profiler wall minus ticks estimates it)
  CPH_GT(cph_gtE);    // V-J27: globaltimer at t_kend
  CPH_FLUSH();
}

// ==================== optional l2norm pre-kernel ====================
// One warp per row of 128; y = x * rsqrt(sumsq + 1e-6) (matches reference within
// fp32 rounding; result cast to bf16 like the pre-normalized frozen inputs).
__global__ void l2norm_qk(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                          __nv_bfloat16* __restrict__ qo, __nv_bfloat16* __restrict__ ko,
                          const long rows) {
  const long r0 = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  for (long r = r0; r < 2 * rows; r += (long)gridDim.x * (blockDim.x >> 5)) {
    const __nv_bfloat16* src = (r < rows ? q : k) + (r < rows ? r : r - rows) * KDIM;
    __nv_bfloat16* dst = (r < rows ? qo : ko) + (r < rows ? r : r - rows) * KDIM;
    float x[4];
    uint2 u0 = *reinterpret_cast<const uint2*>(src + lane * 4);
    // each lane handles 4 elems; warp covers the 128-wide row
    __nv_bfloat162 p0 = *reinterpret_cast<__nv_bfloat162*>(&u0.x);
    __nv_bfloat162 p1 = *reinterpret_cast<__nv_bfloat162*>(&u0.y);
    x[0] = __bfloat162float(p0.x); x[1] = __bfloat162float(p0.y);
    x[2] = __bfloat162float(p1.x); x[3] = __bfloat162float(p1.y);
    float ss = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    float s = rsqrtf(ss + 1e-6f);
    __nv_bfloat16 o[4];
    #pragma unroll
    for (int t = 0; t < 4; t++) o[t] = __float2bfloat16_rn(x[t] * s);
    *reinterpret_cast<uint2*>(dst + lane * 4) = *reinterpret_cast<uint2*>(o);
  }
}

// ============================ host side ============================
constexpr int P1_SMEM = 1024 + KQ_TMA_BYTES + 5 * CDIM * PL_STR * 2 + 24 * CDIM * 4
                        + 2 * 32 * 40 * 4 + 256;
  // V-J44: 96,512B = 1024 base-align + swizzle KQ image 32,768 + bf16 planes
  // 46,080 (sGram + TWO sSys/sTinv sets for group-of-2 head-lockstep) +
  // sb/sbeta/sRho x8 6,144 + sYf/sYb TWO sets 10,240 + p1mbar/slack 256.
  // (was 72,960 single-set.) 2 blocks/SM: 193,024 <= 233,472B cap — the
  // measured REG-114 cap binds occupancy either way (V25 note). P1NS=1
  // ratios over-reserve 23,552B harmlessly (same 2/SM).
constexpr int NSTAGE_V = 2;   // vt32/vt64 (V21.1: vt32 REGRESSED at 3 stages
                              // c9 +1.63% sd 0.09; vt64 3-stage exceeds optin)
constexpr int NSTAGE_32 = 3;  // VARIANT PROBE: vt32 3rd stage (peer V21.1 closed
                              // vt32@3 at +1.63% c9 PRE-staged-o; S8/W balance
                              // changed since — re-measure. smem ~206K <= optin.)
constexpr int NSTAGE_S = 3;   // V21.1: vt16-ONLY 3rd stage (deeper TMA
                              // prefetch: 2 iterations of latency slack).
                              // Measured (dv_a12c278a791f): c1 -5.84us
                              // (-1.54%, chain rate -2.0%), c8 -4.06
                              // (-4.66%), c4/c6 flat = no vt16 regression;
                              // smem @3 <= optin 232448B (launch-validated),
                              // occupancy unchanged (already 1 CTA/SM).
// V3r: chain smem depends on VTT (strides widen at VTT>32); must match the
// in-kernel SPS/USS/VSS layout exactly.
constexpr int chain_smem(int vtt, bool kqtma = false, bool ns3 = true) {
  (void)kqtma;   // V-J2: KQ region unified (swizzle image both paths)
  const int sps = vtt > 32 ? 72 : SP_STR;
  const int uss = vtt > 32 ? 72 : US_STR;
  const int vss = vtt;   // V5.13b/V5.14a: packed TMA rows (vt64 pitch 128B)
  // V21.1: vt16-only 3rd stage. V-J9: vt32 joined at 3 stages (c9 −1.4µs,
  // Y −2.1%). V-J9b: vt32 depth-3 GATED on chain length (avg chunks >= 8) —
  // one-chunk-per-request vt32 shapes REGRESSED at depth 3 (domain d2
  // 32x[1] +3.3%, d9 10x[31..49] +3.8%, dv_80de94c37a41): zero pipeline
  // benefit, pure smem-footprint overhead. Fixed-suite dispatch unchanged
  // (c9 avg 25 chunks keeps depth 3; all other fixed cases vt16/vt64/seg).
  const int ns = (vtt == 16) ? NSTAGE_S
               : ((vtt == 32 && ns3) ? NSTAGE_32 : NSTAGE_V);
  return (2 * CDIM * sps * 2) + 3 * (CDIM * uss * 2)
      + ns * (KQ_TMA_BYTES + CDIM * vss * 2 + 2 * PLANE_PACKED_BYTES + 4 * CDIM * 4)
      + ns * 8 + 64   // load mbars + slack
      + 1088;         // V-J2: 1024 stage-align slack + 64
}

static void set_smem_attrs() {
  static bool done = false;
  if (!done) {
    // V19.1: both grid orders can launch (qhfast iff Gseg==1) -> both need
    // the >48KB dynamic-smem attribute. q = qh-fastest, s = slot-fastest.
#define P1_ATTR(GQ, RT, QF, TAG)                                                                \
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_prep<GQ, RT, QF>,                         \
                cudaFuncAttributeMaxDynamicSharedMemorySize, P1_SMEM) == cudaSuccess, "P1 attr " TAG);
    P1_ATTR(true, 0, true, "t0q")
    P1_ATTR(true, 0, false, "t0s")
    P1_ATTR(true, 2, true, "t2q")
    P1_ATTR(true, 2, false, "t2s")
    P1_ATTR(true, 3, true, "t3q")
    P1_ATTR(true, 3, false, "t3s")
    P1_ATTR(true, 8, true, "t8q")
    P1_ATTR(true, 8, false, "t8s")
    P1_ATTR(false, 1, true, "f1q")
    P1_ATTR(false, 1, false, "f1s")
#undef P1_ATTR
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_V, 32, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(32, false, false)) == cudaSuccess,
                "P2 smem attr failed for ", chain_smem(32, false, false), "B: ", cudaGetErrorString(cudaGetLastError()));
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_S, 16, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(16)) == cudaSuccess,
                "P2 smem attr failed(16) for ", chain_smem(16), "B: ", cudaGetErrorString(cudaGetLastError()));
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_V, 64, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(64)) == cudaSuccess,
                "P2 smem attr failed(64) for ", chain_smem(64), "B: ", cudaGetErrorString(cudaGetLastError()));
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_S, 16, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(16)) == cudaSuccess,
                "P2 smem attr failed(16,A) for ", chain_smem(16), "B: ", cudaGetErrorString(cudaGetLastError()));
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_S, 16, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(16)) == cudaSuccess,
                "P2 smem attr failed(16,C) for ", chain_smem(16), "B: ", cudaGetErrorString(cudaGetLastError()));
    // V-J2 (arm-a V5.33a S1): KQTMA instantiations (non-seg vt32/vt64 only).
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_V, 32, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(32, true, false)) == cudaSuccess,
                "P2 smem attr failed(32,kqtma,ns2)");
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_V, 64, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(64, true)) == cudaSuccess,
                "P2 smem attr failed(64,kqtma)");
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_32, 32, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(32)) == cudaSuccess,
                "P2 smem attr failed(32,ns3)");
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_32, 32, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(32, true)) == cudaSuccess,
                "P2 smem attr failed(32,ns3,kqtma)");
    // V-J3: PRE0 instantiation (one-wave vt16 chunk-0 pre-wait prefetch).
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_S, 16, 0, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(16)) == cudaSuccess,
                "P2 smem attr failed(16,pre0)");
    // V-J51: MODE1/2 x vt64 wide-tile seg instantiations.
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_V, 64, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(64)) == cudaSuccess,
                "P2 smem attr failed(64,A)");
    TORCH_CHECK(cudaFuncSetAttribute((const void*)gdn_chain<NSTAGE_V, 64, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, chain_smem(64)) == cudaSuccess,
                "P2 smem attr failed(64,C)");
    cudaFuncSetAttribute(l2norm_qk, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
    done = true;
  }
}

// V5.13b: host-side v tensor map, cached by EXACT key. Encode measured
// 57.8us on this platform (results/tma-vbox-unit.txt) — per-forward is a
// non-starter; the map holds address+layout metadata only (NO data), so
// reuse under an exact (ptr, HV, T, box) key is semantics-preserving: every
// forward still TMA-reads v's current bytes; a new tensor/geometry misses
// and re-encodes (first call absorbed by warmup). Leaked heap allocation
// (matches the V3h ws_cache pattern; avoids static-destructor vs
// torch-shutdown ordering hazards).
static CUtensorMap get_vmap(const void* vptr, long HV, long T, int boxc) {
  struct Key {
    const void* p; long hv; long t; int b;
    bool operator==(const Key& o) const { return p == o.p && hv == o.hv && t == o.t && b == o.b; }
  };
  struct KeyH {
    size_t operator()(const Key& k) const {
      return std::hash<const void*>()(k.p) ^ (std::hash<long>()(k.hv) * 31)
           ^ (std::hash<long>()(k.t) * 131) ^ ((size_t)k.b * 7);
    }
  };
  static auto* cache = new std::unordered_map<Key, CUtensorMap, KeyH>();
  Key key{vptr, HV, T, boxc};
  auto it = cache->find(key);
  if (it != cache->end()) return it->second;
  CUtensorMap map;
  uint64_t gdim[3] = {128, (uint64_t)HV, (uint64_t)T};
  uint64_t gstride[2] = {256, (uint64_t)HV * 256};
  uint32_t box[3] = {(uint32_t)boxc, 1, 64};
  uint32_t estr[3] = {1, 1, 1};
  // V5.14a: vt64 box = 128B rows -> SWIZZLE_128B (byte-exact bench model
  // chunk^(row&7), results/tma-swz-unit.txt); vt16/32 stay SWIZZLE_NONE
  // (sub-128B swizzle layout unverified — bench FAILED the xor model).
  const CUtensorMapSwizzle swz = (boxc == 64) ? CU_TENSOR_MAP_SWIZZLE_128B
                                              : CU_TENSOR_MAP_SWIZZLE_NONE;
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(vptr),
      gdim, gstride, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "v tensormap encode failed: ", (int)r);
  cache->emplace(key, map);
  return map;
}

// V-J2 (arm-a V5.33a/V5.35a): k/q tensor maps for the SWIZZLE_128B two-box
// sub-tile form — P1's TMA prologue consumes them in EVERY case; the chain's
// KQTMA regime shares. Layout k/q = [1,T,H,128]: gdim {128, T, H}, strides
// {H*256B token, 256B head}, box {64, 64, 1} (128B inner extent = the
// SWIZZLE_128B constraint => two 64-key boxes per tile). Same caching
// discipline as get_vmap (encode 57.8us first call, warmup-absorbed; map
// holds address+layout metadata only — every forward TMA-reads current
// bytes; exact (ptr, H, T) key).
static CUtensorMap get_kqmap(const void* ptr, long H, long T) {
  struct Key {
    const void* p; long h; long t;
    bool operator==(const Key& o) const { return p == o.p && h == o.h && t == o.t; }
  };
  struct KeyH {
    size_t operator()(const Key& k) const {
      return std::hash<const void*>()(k.p) ^ (std::hash<long>()(k.h) * 31)
           ^ (std::hash<long>()(k.t) * 131);
    }
  };
  static auto* cache = new std::unordered_map<Key, CUtensorMap, KeyH>();
  Key key{ptr, H, T};
  auto it = cache->find(key);
  if (it != cache->end()) return it->second;
  CUtensorMap map;
  uint64_t gdim[3] = {128, (uint64_t)T, (uint64_t)H};
  uint64_t gstride[2] = {(uint64_t)H * 256, 256};   // token stride, head stride
  uint32_t box[3] = {64, 64, 1};
  uint32_t estr[3] = {1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr),
      gdim, gstride, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "kq tensormap encode failed: ", (int)r);
  cache->emplace(key, map);
  return map;
}

std::pair<torch::Tensor, std::optional<torch::Tensor>> gdn_forward(
    const torch::Tensor& q_, const torch::Tensor& k_, const torch::Tensor& v_,
    const torch::Tensor& g_, const torch::Tensor& beta_,
    const torch::Tensor& state_, const torch::Tensor& cu_,
    const torch::Tensor& ci_, const torch::Tensor& co_,
    const bool l2norm, const bool out_final,
    const std::optional<torch::Tensor>& core_
#ifdef GDN_P1PHASE
    , const std::optional<torch::Tensor>& phases_
#endif
#ifdef GDN_CPHASE
    , const std::optional<torch::Tensor>& cph_
#endif
    ) {
  HP(0);   // V-J37: entry
  TORCH_CHECK(q_.is_cuda() && k_.is_cuda() && v_.is_cuda(), "inputs must be CUDA");
  // V3e: skip no-op contiguous() dispatch on the common contiguous path
  auto q = q_.is_contiguous() ? q_ : q_.contiguous();
  auto k = k_.is_contiguous() ? k_ : k_.contiguous();
  auto v = v_.is_contiguous() ? v_ : v_.contiguous();
  auto g = g_.is_contiguous() ? g_ : g_.contiguous();
  auto beta = beta_.is_contiguous() ? beta_ : beta_.contiguous();
  auto state = state_.is_contiguous() ? state_ : state_.contiguous();
  auto cu = cu_.is_contiguous() ? cu_ : cu_.contiguous();
  auto ci = ci_.is_contiguous() ? ci_ : ci_.contiguous();
  auto co = co_.is_contiguous() ? co_ : co_.contiguous();
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
              v.scalar_type() == at::kBFloat16, "q/k/v must be bf16");
  TORCH_CHECK(g.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat, "g/beta fp32");
  TORCH_CHECK(cu.scalar_type() == at::kInt && ci.scalar_type() == at::kInt &&
              co.scalar_type() == at::kInt, "cu/chunk ints");
  TORCH_CHECK(state.scalar_type() == at::kBFloat16 || state.scalar_type() == at::kFloat,
              "state bf16/fp32");
  const int T = q.size(1), H = q.size(2), HV = v.size(2), N = cu.size(0) - 1;
  TORCH_CHECK(q.size(3) == KDIM && v.size(3) == VDIM && HV % H == 0, "D=128 geometry");
  TORCH_CHECK(state.size(0) == N && state.size(1) == HV && state.size(2) == VDIM &&
              state.size(3) == KDIM, "state shape");
  // E2: ci row count for the parallel mapping prefetch clamp (frozen ABI:
  // chunk_indices has exactly chunk_offsets[N] rows).
  TORCH_CHECK(ci.dim() == 2 && ci.size(1) == 2 && ci.size(0) >= 1, "chunk_indices shape");
  const int ci_rows = (int)ci.size(0);
  torch::Tensor core;
  __nv_bfloat16* core_ptr = nullptr;
  if (core_.has_value()) {
    core = core_->contiguous();
    TORCH_CHECK(core.scalar_type() == at::kBFloat16, "core bf16");
    TORCH_CHECK(core.numel() >= (long)T * HV * VDIM, "core capacity");
    core_ptr = reinterpret_cast<__nv_bfloat16*>(core.data_ptr());
  }
  auto opts = q.options();
  // V3h: out (+final) carved from ONE fresh byte allocation - fresh storage
  // per call preserved, one fewer allocator round-trip. 256B-aligned split.
  // V5.8a: the allocation itself MOVED below the P1 launch (arm-a V5d4:
  // ~2-4us of allocator+view time overlaps P1's GPU execution; fixed −0.68%
  // there). P1 touches neither tensor; chain launch consumes the pointers.
  const long out_bytes = (long)T * HV * VDIM * 2;
  const long fin_bytes = out_final ? (long)N * HV * VDIM * KDIM * 4 : 0;
  const long fin_ofs = fin_bytes ? ((out_bytes + 255) & ~255L) : 0;
  torch::Tensor outbuf;
  torch::Tensor out;
  std::optional<torch::Tensor> final_state;
  float* final_ptr = nullptr;
  const long slots_ub = (T + CDIM - 1) / CDIM + N;
  // V2g: ONE workspace allocation (planes | 256B pad | scratchq | 256B pad/
  // perm) — each at::empty costs ~2-4us of caching-allocator+dispatch time,
  // measurable on ~150us shapes. Stream-order lifetime is unchanged (buffer
  // tensor lives to return; same-stream reuse is safe).
  const long planes_bytes = (long)slots_ub * HV * SLOT_BYTES;
  const long q_bytes = (long)slots_ub * H * KSQB_BYTES;
  // V5.11a: hoist the V3r2 vtile dispatch above the perm decision — the
  // LPT condition now needs the chain grid size.
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // V3r2: three-tier vtile dispatch (metadata-only; no D2H):
  //  vt16: one-wave rule (unchanged) - chain-serial latency class.
  //  vt64: wave-bound AND long multi-request batches (T >= 4096, N >= 2).
  //    Stage traffic is ~90% vtile-duplicated (sKQ+planes identical across
  //    vt of one (n,hv)); vt64 halves L2 reads (c0: 445MB -> 238MB) and
  //    waves (c0: 256 -> 128 CTAs = one wave). Measured (dv_4037bb7dedff):
  //    c0 (T=7424,N=4) 494.2 -> 448.6us (-9.2%), margins bitwise == vt32.
  //    Excluded: short chains (c9 T=3072: vt64 = +1.1%, serial-rate cost
  //    dominates) and one-wave-at-vt32 geometries (4*HV*N <= sm_count:
  //    serial-bound - domain d3 [4096,4096] Hv16 would ~+20%; d1 [8192]
  //    Hv24 measured serial-bound at vt32). Multi-wave-at-vt32 N==1 IS
  //    included: domain d5 ([16384] Hv128, 3.5 waves, measured 3.15TB/s
  //    stage traffic = L2-bound not serial-bound) - traffic win applies.
  //  vt32: everything else (status quo).
  //  (dv 20261022T045634Z: always-16 regressed wave-bound c0 759->887,
  //   c9 217->272 - amplification; V3r blanket-64: geo 133.19 > 132.8 bar.)
  const bool vt16 = ((long)(VDIM / 16) * HV * N <= sm_count);
  // V-J42: one-chunk class (avg chunks/req <= 1, from T and N metadata only —
  // no D2H, no case identity). Its chains have ZERO serial length, so the
  // vt64 serial-rate penalty behind the T>=4096 gate (V3r2: c9 +1.1%, d3
  // +20% on LONG chains) cannot apply; the wall is waves x per-CTA latency,
  // and vt64 halves the CTAs (V-J41 census: d2 = 512 vt32 CTAs x 4.31us at
  // 1 CTA/SM = 3.46 waves = 14.9us chain, 77% of it fixed per-CTA overhead).
  // Wave-bound-at-vt32 clause retained: single-wave geometries keep vt32's
  // lower per-CTA latency. Expected bitwise (V3r2 evidence: vtt classes
  // bitwise-equal — per-element K accumulation order is vtt-invariant).
  // V-J43: extended to avg <= 2 chunks/req (ceil(T/64) <= 2N) — d3-class
  // (T576 N6, 1.30 waves at vt32, 20.76us chain): wave-halving vs the
  // vt64 per-chunk penalty (~+0.3us x <=2 serial chunks) — A/B decides.
  // (V-J42 note: the "expected bitwise" line held only for Lc<=32 chunks;
  // KSPLIT=(VTT<=32) makes multi-ks flips margins-class — smoke EXACT.)
  const bool onechunk = (long)((T + CDIM - 1) / CDIM) <= 2L * (long)N;
  // E5a: wave-arithmetic vt64 clause — MULTI-WAVE vt32 geometries also go
  // vt64 when the wave-count drop outweighs the vt64/vt32 serial-rate ratio
  // (census rates: vt32 1.44us/chunk vs vt64 2.16 => vt64 wins iff
  // ceil64*3 < ceil32*2). Scope: c9 (2 waves -> 1, chain 76.6 -> ~71us) and
  // the d9 domain class (4 -> 2 waves, plus ~halved stage traffic). vt32
  // SINGLE-wave stays vt32 (V3r2 d3/d6 lesson) and T>=4096/onechunk clauses
  // are unchanged. Margins-class (KSPLIT flips at VTT>32), not bitwise.
  const long ctas32 = (long)(VDIM / 32) * HV * N;
  const long w32 = (ctas32 + sm_count - 1) / sm_count;
  const long w64 = ((long)(VDIM / 64) * HV * N + sm_count - 1) / sm_count;
  const bool vt64 = !vt16 && ctas32 > sm_count &&
                    ((T >= 4096 || onechunk) || w64 * 3 < w32 * 2);
  int vtt = vt16 ? 16 : (vt64 ? 64 : 32);   // V-J51: wide-seg may retile
  long chain_blocks = (long)(VDIM / vtt) * HV * N;   // V-J51: retile-aware
  // V4a port (V18): segment-parallel chains for the LOW-OCCUPANCY vt16
  // class (arm-b evidence: c8-class P2 32 CTAs -> 78% of SMs idle, chain
  // rate per SM unchanged => occupancy-starved). Split each chain into G
  // segments (pass A: zero/state0-init partial chains at the cheap
  // propagation rate publishing affine transfers B_g/a_g; pass C: composed
  // true input + exact full-rate rerun emitting ALL outputs). G>1 only when
  // it at least doubles occupancy (vtCTAs*G <= sm) and keeps >=16
  // chunks/segment on average (avg>=32). G==1 -> single launch,
  // segB=nullptr, kernel path bitwise-identical to the common tree.
  // Numerics: G==1 bitwise; G=2 pass-C compose is the exact B_0 fp32
  // roundtrip (bitwise); G>=3 ~2^-24 composition class (arm-b official
  // precedent: v5.8/v5.9 PASS x2 + domain, c8 margins moved ~1e-6).
  // V-J51: wide-tile segmentation second chance for OCCUPANCY-STARVED
  // LONG-chain geometries (c1 class: vt16-selected, 128 cols x 256 chunks,
  // 20 idle SMs). vt64 columns are 4x fewer, so G>=2 stays SINGLE-WAVE per
  // pass (32x4=128<=148). E5b: the wide arm is now evaluated SIDE-BY-SIDE
  // with the V4a vt16 arm (previously V-J51 required Gseg==1, so V4a's
  // occupancy-based G preempted it — mis-tiering the d7 domain class:
  // [16384] tp4 hr2 ran vt16 G=2 = 128-chunk A+C passes (233us chain) while
  // vt64 G=8 gives 32-chunk passes (~124us predicted from census rates)).
  // Arm choice = predicted serial wall: chunks/G x (passA+passC rate-sum),
  // measured vt16 (0.75+1.07) = 1.82us/chunk vs vt64 (1.57+2.16) = 3.73
  // (dv_ee72c184d3e0 + dv_d312a930b726) => wide wins iff Gw*100 > Gc*205.
  // Single-feasible-arm cases keep the exact V4a/V-J51 behavior (c1 wide,
  // c8 vt16 — c8's avgc=128 fails the wide gate; ratios reproduced:
  // 128/4 x 1.82 = 58us vt16 vs 128/4... wide infeasible => vt16 kept).
  // Numerics: G>=3 compose = the ~2^-24 class CERTIFIED on c8/c1.
  int Gseg = 1;
  long seg_bytes = 0;
  if (vt16) {
    const long vtCTAs = (long)(VDIM / 16) * HV * N;
    const long avg_chunks = ((long)T + (long)N * 63) / ((long)N * 64);
    int Gc = 0;   // V4a arm (vt16)
    if (vtCTAs * 2 <= sm_count && avg_chunks >= 32) {
      Gc = (int)(sm_count / vtCTAs);
      if (Gc > 8) Gc = 8;
      while (Gc > 1 && avg_chunks / Gc < 16) --Gc;
      if (Gc < 2) Gc = 0;
    }
    int Gw = 0;   // V-J51 arm (vt64 wide)
    const long cols64 = (long)(VDIM / 64) * HV * N;
    {
      int g = (int)(sm_count / cols64);
      if (g > 8) g = 8;
      while (g > 1 && avg_chunks / g < 32) --g;
      if (g >= 2 && avg_chunks >= 192 && cols64 * g <= sm_count) Gw = g;
    }
    const bool wide = (Gw >= 2) && ((Gc < 2) || ((long)Gw * 100 > (long)Gc * 205));
    if (wide) {
      vtt = 64;
      Gseg = Gw;
      chain_blocks = cols64;
      seg_bytes = (long)N * HV * (VDIM / 64) * Gw * ((long)2 * CDIM * 64 * 4 + 4) + 256;
    } else if (Gc >= 2) {
      Gseg = Gc;
      seg_bytes = (long)N * HV * (VDIM / 16) * Gc * ((long)2 * CDIM * 16 * 4 + 4) + 256;
    }
  }
  // V5.13b/V5.14a: v box map for the TMA path (all VTT; vt64 = SWIZZLE_128B)
  const CUtensorMap vmap = get_vmap(v.data_ptr(), HV, T, vtt);

  // V3h: grow-only persistent workspace (internal scratch, never returned;
  // P1 rewrites every consumed slot before P2 reads it each call).
  // Heap-allocated + intentionally leaked: avoids static-destructor vs
  // torch-shutdown ordering hazards in extension .so teardown.
  static torch::Tensor* ws_cache = new torch::Tensor();
  const long need = planes_bytes + q_bytes + 512 + seg_bytes;
  if (!ws_cache->defined() || ws_cache->numel() < need ||
      ws_cache->device() != q.device()) {
    *ws_cache = at::empty({need}, opts.dtype(at::kByte));
  }
  torch::Tensor scratchbuf = *ws_cache;   // keep alive through the call
  char* scratch_ptr = reinterpret_cast<char*>(scratchbuf.data_ptr());
  char* scratchq_ptr = scratch_ptr + planes_bytes + 256;
  // V4a port: seg transfers live past the perm tail pad (16B-aligned; perm
  // uses <=128B of the 512B pad, segB starts at pad+256).
  float* segB_ptr = seg_bytes
      ? reinterpret_cast<float*>(scratchq_ptr + q_bytes + 256) : nullptr;
  float* segA_ptr = seg_bytes
      ? segB_ptr + (long)N * HV * (VDIM / vtt) * Gseg * (2 * CDIM * vtt) : nullptr;   // V-J51: vtt-general (vt16 values byte-identical)

  auto stream = at::cuda::getCurrentCUDAStream();
  set_smem_attrs();

  const __nv_bfloat16* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  const __nv_bfloat16* kp = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  torch::Tensor qn, kn;
  if (l2norm) {
    // V-J2 (arm-a V5.35a): PERSISTENT grow-only normalized-q/k buffers (V3h
    // ws pattern). Stable pointers across same-shape calls => get_kqmap
    // cache hits (a per-call at::empty_like would miss => 2x57.8us encode
    // per call now that P1's TMA prologue consumes maps EVERY call).
    // Internal scratch, never returned; same-stream ordering makes reuse
    // safe (l2norm writes, P1/chain read, next call's l2norm rewrites).
    static torch::Tensor* norm_cache = new torch::Tensor[2]();
    const long qbytes = q.numel() * 2;
    if (!norm_cache[0].defined() || norm_cache[0].numel() < qbytes ||
        norm_cache[0].device() != q.device()) {
      norm_cache[0] = at::empty({qbytes}, opts.dtype(at::kByte));
      norm_cache[1] = at::empty({qbytes}, opts.dtype(at::kByte));
    }
    qn = norm_cache[0].narrow(0, 0, qbytes).view(at::kBFloat16).view(q.sizes());
    kn = norm_cache[1].narrow(0, 0, qbytes).view(at::kBFloat16).view(k.sizes());
    const long rows = (long)T * H;
    long blocks = (2 * rows + 7) / 8;  // 8 warps (256 thr) per block
    if (blocks > 1048576) blocks = 1048576;
    l2norm_qk<<<(int)blocks, 256, 0, stream>>>(
        qp, kp, reinterpret_cast<__nv_bfloat16*>(qn.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(kn.data_ptr()), rows);
    qp = reinterpret_cast<const __nv_bfloat16*>(qn.data_ptr());
    kp = reinterpret_cast<const __nv_bfloat16*>(kn.data_ptr());
  }

  // V-J2 (arm-a V5.33a S1): KQTMA regime dispatch — direct q/k TMA boxes for
  // NON-seg chains whose workspace + q-input footprint exceeds L2 (the
  // DRAM-pass reduction pays only where the scratchq round-trip actually
  // hits DRAM). VTT >= 32 REQUIRED (arm-a gate dv_48cab87b90e6 regime map,
  // all bitwise): vt64 c0 0.976 WIN / vt32 Y[9000,7383] 0.981 WIN / vt16 c1
  // 1.027 + X[16383] 1.027 LOSS — the +3 TMA ops/chunk (~70ns admission)
  // land serial on vt16's long 1.0us chains while vt32/64 absorb them and
  // cash the s2g-write elimination (prep −4.8us on c0). l2norm=True falls
  // back to bulk for the CHAIN (kqtma=false) — but the P1 maps below are
  // built on the persistent norm buffers, so they stay cache-stable.
  static int l2_bytes_cache = 0;
  if (l2_bytes_cache == 0)
    cudaDeviceGetAttribute(&l2_bytes_cache, cudaDevAttrL2CacheSize, q.get_device());
  const long qin_bytes = (long)T * H * KDIM * 2;
  const bool kqtma = !l2norm && Gseg == 1 && vtt >= 32 &&
                     (need + qin_bytes > (long)l2_bytes_cache);
  // V-J2: k/q maps are ALWAYS built — P1's TMA prologue consumes them in
  // every case (chain KQTMA shares). Built AFTER the l2norm block: qp/kp
  // are the final pointers.
  const CUtensorMap kq_kmap = get_kqmap(kp, H, T);
  const CUtensorMap kq_qmap = get_kqmap(qp, H, T);

  // (V5.11a tier dispatch + V4a-port Gseg + V5.13b vmap: HOISTED above the
  // workspace block - seg_bytes sizes the ws allocation.)

  // Longest-chain-first ordering when many requests compete for waves;
  // perm is filled by P1 block (0,0) (folded P0) before the chain reads it.
  // V5.11a: ALSO engage for 2..7-request grids exceeding one wave — case0
  // (N=4, lengths 255/1023/2049/4097, vt64: 192 blocks = 1.30 waves) today
  // linearizes its 64-chunk request LAST (IDs 144-191): most of its blocks
  // start only as wave-1 SMs retire (~25us late; chain 319us vs ~294us
  // pure-serial). LPT puts the longest chains at IDs 0-47 (start ~0).
  // The old N>=8 rule is kept verbatim as a superset branch: zero behavior
  // change for every shape that had perm before, and one-wave shapes stay
  // perm-free (fixed cases 1-9 all <= 0.86 waves under the V3r2 dispatch).
  const int* perm_ptr = nullptr;
  int* perm_buf = nullptr;
  if ((N >= 8 && N <= 32) ||
      (N >= 2 && N <= 32 && chain_blocks > sm_count)) {
    perm_buf = reinterpret_cast<int*>(scratchq_ptr + q_bytes);  // N<=32 ints in the tail pad
    perm_ptr = perm_buf;
  }

  // V-J3 (PRE0): chunk-0 pre-wait KQ+v prefetch for the one-wave vt16 bulk
  // regime. Requires perm-free dispatch (n = blockIdx.z pre-wait) and the
  // input-backed maps (l2norm's norm buffers are stream-safe pre-wait too —
  // l2norm fully precedes P1 — but stay conservative until measured).
  const bool pre0 = !kqtma && !l2norm && Gseg == 1 && vtt == 16 &&
                    perm_ptr == nullptr;

  // GQA-shared prep only when the shared grid still fills the machine; small
  // workloads prefer one block per value head (shorter serial ratio loop).
  const bool gqa = (long)slots_ub * H >= sm_count;
#ifdef GDN_P1PHASE
  long long* phases_ptr = phases_.has_value()
      ? reinterpret_cast<long long*>(phases_->data_ptr<int64_t>()) : nullptr;
#endif
#ifdef GDN_CPHASE
  long long* cph_ptr = cph_.has_value()
      ? reinterpret_cast<long long*>(cph_->data_ptr<int64_t>()) : nullptr;
  #define cph_arg , cph_ptr
#else
  #define cph_arg
#endif
    // V3f: dispatch the compile-time RATIO matching HV/H (2/3/8 cover the
  // shape_range head_ratio values); anything else uses the generic path.
  const int ratio = HV / H;
#ifdef GDN_P1PHASE
  #define phases_arg , phases_ptr
#else
  #define phases_arg
#endif
#define GDN_LAUNCH_PREP(GQ, RT, GX, QF)                                          \
  gdn_prep<GQ, RT, QF><<<dim3((QF) ? (unsigned)(GX) : (unsigned)slots_ub,        \
                              (QF) ? (unsigned)slots_ub : (unsigned)(GX)),        \
      NTHR, P1_SMEM, stream>>>(                                                  \
      qp, kp, g.data_ptr<float>(), beta.data_ptr<float>(), cu.data_ptr<int>(),   \
      ci.data_ptr<int>(), co.data_ptr<int>(), N, H, HV,                          \
      scratch_ptr, scratchq_ptr, perm_buf, (int)kqtma, ci_rows, kq_kmap, kq_qmap phases_arg)
  // V19.1: qh-fastest P1 order ONLY for non-segmented shapes (Gseg==1).
  // Segmented (G>1) shapes keep the certified V18.1 slot-fastest order:
  // official c8 evidence (dv_7ddca0e52748) = +1.86us slow-mode-mixture
  // shift on the 3-kernel segmented window under qh-fastest.
  HP(1);   // V-J37: prep/checks/ws/maps/dispatch done
  const bool qhfast = (Gseg == 1);
  if (gqa) {
    switch (ratio) {
      case 2: if (qhfast) GDN_LAUNCH_PREP(true, 2, H, true); else GDN_LAUNCH_PREP(true, 2, H, false); break;
      case 3: if (qhfast) GDN_LAUNCH_PREP(true, 3, H, true); else GDN_LAUNCH_PREP(true, 3, H, false); break;
      case 8: if (qhfast) GDN_LAUNCH_PREP(true, 8, H, true); else GDN_LAUNCH_PREP(true, 8, H, false); break;
      default: if (qhfast) GDN_LAUNCH_PREP(true, 0, H, true); else GDN_LAUNCH_PREP(true, 0, H, false); break;
    }
  } else {
    if (qhfast) GDN_LAUNCH_PREP(false, 1, HV, true); else GDN_LAUNCH_PREP(false, 1, HV, false);
  }
#undef GDN_LAUNCH_PREP
#undef phases_arg
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "P1 launch error: ",
              cudaGetErrorString(cudaGetLastError()));
  HP(2);   // V-J37: P1 launch + error check

  // V5.8a: out(+final) carve here — overlaps P1 GPU time (see note above).
  outbuf = at::empty({out_bytes + fin_ofs + fin_bytes}, opts.dtype(at::kByte));
  out = outbuf.narrow(0, 0, out_bytes).view(at::kBFloat16).view({1, T, HV, VDIM});
  if (out_final) {
    auto f = outbuf.narrow(0, fin_ofs, fin_bytes).view(at::kFloat)
                     .view({N, HV, VDIM, KDIM});
    final_ptr = reinterpret_cast<float*>(outbuf.data_ptr<uint8_t>() + fin_ofs);
    final_state = f;
  }

  // (V3r2 vtile dispatch hoisted above the perm decision — V5.11a.)
  // V3b PDL: programmatic stream serialization lets P2 blocks pre-launch
  // and run their P1-independent prologue while P1 drains (kernel-side
  // griddepcontrol.wait keeps exact data ordering).
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(VDIM / vtt), (unsigned)HV, (unsigned)(N * Gseg));
  cfg.blockDim = dim3(NTHR);
  // V-J9b: vt32 deep pipeline ONLY for long chains (avg chunks >= 8; the
  // fixed-suite dispatch is unchanged — c9 avg 25 keeps depth 3 — while
  // one-chunk-per-request vt32 domain shapes revert to depth 2 and recover
  // the +3.3/+3.8% d2/d9 regressions measured in dv_80de94c37a41).
  const long avg_chunks_meta = ((long)T + (long)N * 63) / ((long)N * 64);
  const bool ns3_32 = (avg_chunks_meta >= 8);
  cfg.dynamicSmemBytes = chain_smem(vtt, kqtma, ns3_32);
  cfg.stream = stream;
  cudaLaunchAttribute pdl_attr[1];
  pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl_attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = pdl_attr;
  cfg.numAttrs = 1;
  auto launch_chain = [&](auto* fn, int pass_) {
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, fn,
        qp, kp, reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), vmap,
        kq_kmap, kq_qmap,   // V-J2 (KQTMA only)
        beta.data_ptr<float>(),
        state.data_ptr(), state.scalar_type() == at::kFloat ? 1 : 0,
        cu.data_ptr<int>(), co.data_ptr<int>(), perm_ptr, H, HV,
        scratch_ptr,
        scratchq_ptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), core_ptr, final_ptr,
        Gseg, pass_, segB_ptr, segA_ptr cph_arg) == cudaSuccess,
        "P2 PDL launch error: ", cudaGetErrorString(cudaGetLastError()));
  };
  if (Gseg > 1) {
    // V18.1: segmented => vt16 by the Gseg rule; pass A then pass C (C's
    // PDL wait binds to A via stream order; segB/segA reads sit post-wait).
    // V-J51: wide-tile seg (c1 class) => the SAME MODE1/2 device text at
    // VTT=64 (VTT-generic templates, zero new kernel code).
    if (vtt == 64) {
      launch_chain(gdn_chain<NSTAGE_V, 64, 1>, 0);
      launch_chain(gdn_chain<NSTAGE_V, 64, 2>, 1);
    } else {
      launch_chain(gdn_chain<NSTAGE_S, 16, 1>, 0);
      launch_chain(gdn_chain<NSTAGE_S, 16, 2>, 1);
    }
  } else if (vt16) {
    if (pre0) launch_chain(gdn_chain<NSTAGE_S, 16, 0, false, true>, 0);
    else launch_chain(gdn_chain<NSTAGE_S, 16, 0>, 0);
  } else if (vt64) {
    // V-J2 (arm-a V5.33a): KQTMA regime = issue-mechanism-only delta (the
    // swizzle image is the universal stage form now).
    if (kqtma) launch_chain(gdn_chain<NSTAGE_V, 64, 0, true>, 0);
    else launch_chain(gdn_chain<NSTAGE_V, 64, 0>, 0);
  } else {
    // V-J9b: depth-3 only for long vt32 chains (see ns3_32 above).
    if (ns3_32) {
      if (kqtma) launch_chain(gdn_chain<NSTAGE_32, 32, 0, true>, 0);
      else launch_chain(gdn_chain<NSTAGE_32, 32, 0>, 0);
    } else {
      if (kqtma) launch_chain(gdn_chain<NSTAGE_V, 32, 0, true>, 0);
      else launch_chain(gdn_chain<NSTAGE_V, 32, 0>, 0);
    }
  }
  // Debug-phase launch checks: attribute failures surface immediately.
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "P2 launch error: ",
              cudaGetErrorString(cudaGetLastError()));
  HP(3);   // V-J37: chain launch(es) + check
#undef cph_arg
  HP(4);   // V-J37: return construction
  return {out, final_state};
}

}  // namespace gdn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef GDN_HOSTPROF
  m.def("hostprof", [] {
    return std::vector<long long>(gdn::g_hp, gdn::g_hp + 5);
  });
#endif
#ifdef GDN_P1PHASE
  m.def("gdn_forward", &gdn::gdn_forward, "GDN prefill forward (P1-phase instrumented)");
#else
  m.def("gdn_forward", &gdn::gdn_forward, "GDN prefill forward");
#endif
}
