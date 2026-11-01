// mla_expanded_bf16 - tcgen05 TWO-CTA CLUSTER kernel (tc9): PERSISTENT clusters.
//
// tc9 = tc8 (pair M256N128K16 + packed-f16x2 exp + true masked-tile max, all
// officially evaluated at 117.079 us fixed geomean as best/v17) with the CTA
// lifetime turned into a WORK LOOP over items, which is the half of arm a's
// tc2p win that the cluster engine never had. Everything else - TMEM map, pair
// MMA shapes, barrier protocol, softmax, epilogue math, host ABI - is tc8's.
//
// Why: the refit#7 cost model prices tc8's per-item overhead at ~17k cycles
// (8.94 us per item per cluster), against ~12k for a serial tc2 CTA, because a
// cluster launch adds the pair TMEM alloc by both CTAs, two cluster rendezvous
// and a 49152 B Q box on top of what tc2 pays. f0 (5.19 items/cluster) and f1
// (18.16) are exactly the two fixed cases tc8 loses - 1.211 and 1.410 - and both
// are prologue-dominated rather than rate-limited.
//
// What is different from tc2p's persistence, and why it is cheaper here: there is
// NO device work counter, NO smem item publication and NO CONS/NEXT handshake.
// The item sequence is a pure function of (blockIdx.x, gridDim.x), and
// decode_item() is a pure function of the item index plus immutable cum_q/cum_kv,
// so all ten warp roles in both CTAs of a pair derive identical sequences, skip
// identical empty items and advance an identical global tile counter. Balance
// comes from the strided assignment plus the LPT q-tile order already in the
// decode; the price against a counter is that a ragged remainder cannot be
// stolen, which matters most on f1 (1344 items over 74 clusters).
//
// Three things had to change structurally, all forced by the pipeline no longer
// restarting at an item boundary:
//   1. The tile index is GLOBAL. Every `i % STAGES` / `(i / STAGES) & 1` in the
//      file is therefore already the right slot and parity across items, and only
//      the three places where the ITEM matters use G0 explicitly (this item's
//      first tile, the kv column offset, and the O-reset predicate).
//   2. The epilogue can no longer stage O over "the dead K stages" - the loader is
//      refilling them for the next item while this epilogue packs - so the staging
//      image moved to its own 32768 B at EPI_OFF (206080 B of the 232448 opt-in).
//   3. The pair has ONE O accumulator (3 S/P slots x 128 + O 128 = all 512 TMEM
//      columns), and the next item's first PV overwrites it with accumulate off,
//      so a new ODR barrier (count 2: own softmax + the peer's warp-10 relay,
//      mirroring PF/PL) gates that PV on both CTAs' epilogue O reads.
// The Q refill waits SF of the previous item's last QK, so the 49152 B transfer
// overlaps that item's softmax tail and epilogue instead of serialising.
//
// Internal namespace/symbol names stay mla_tc7* on purpose: it keeps the diff
// against kernel_tc8.cu reviewable as the change it is.
//
// ---- the tc8 header follows, unchanged ----
//
// mla_expanded_bf16 - tcgen05 TWO-CTA CLUSTER kernel (tc8), post-join step 1.
//
// tc8 = the cluster engine that the pre-join arm b promoted (kernel_tc7.cu at
// best/correct-124, officially evaluated at 124.359 us fixed geomean, taken
// byte-for-byte as the starting point) PLUS the softmax exp of the join winner
// (kernel_tc2p.cu's production -DTC2_EX2F16 path, copied textually). Nothing
// else in this file differs from the promoted tc7: same TMEM map, same barrier
// protocol, same pair MMA shapes, same epilogue, same host wrapper.
//
// Why combine exactly these two: the join left the two arms' wins orthogonal.
// The winner (arm a) is faster on the many-item shapes because kernel_tc2p.cu
// makes the CTA persistent and because its exp is packed f16x2 (f1 195.6 us vs
// arm b's 234.5). Arm b was faster on the steady-state shapes because a pair
// MMA is M256N128K16 = 8189 flop/cyc/SM against tc2's M128N64K16 A-from-smem
// 5459, i.e. 1280.3 cyc per 128x128 tile against tc2's 1643.2 for the same
// 16384 score elements (f5 750.3 vs 792.8, f4 246.8 vs 270.2, f2 118.2 vs
// 127.7, f9 50.4 vs 53.7). Per-case best-of-both is 119.98 us against the
// adopted 123.58, and that estimate still charges tc8 with arm b's UNPACKED
// exp - the two levers are independent and this file is their sum.
//
// The exp phase is the reason to expect more than the sum: tc7's per-tile period
// was softmax-paced (~1470 cyc against its 1280 cyc MMA stream), so cutting the
// exp instruction count in half is what lets the pair MMA rate actually set the
// period instead of being masked by the softmax. kernel_tc2p.cu's own accounting
// for the same change is sm_exp 1236 -> 736 cyc.
//
// Status: FORCE-ONLY (MLA_FORCE=tc8). _pick_engine never returns it. It has to
// pass the boundary set, the repeat-launch race hunt and both official suites on
// identical bytes before any dispatch or tag moves.
//
// ---- original tc7 header follows, unchanged ----
//
// Same frozen contract as kernel_tc2.cu. Structurally different: BM=256 is split
// across a cluster PAIR (128 q rows per CTA) and BN=128, so every MMA is
// M256N128K16 at cta_group::2. Design and evidence:
// experiments/notes/cluster-design.md and experiments/notes/mma-rates.md.
//
// Why: tc2's f5 steady state is 2255 cyc per dual-tile against 1643 cyc of MMA
// and ~1470 cyc of softmax, and both halves of that gap are TMEM-column bound at
// BM=256 in one CTA (3 S/P slots would need 640 columns; BN=128 would need 768).
// In a pair each CTA owns only its own 128 rows, so 3 S/P slots x 128 columns +
// O 128 = 512 exactly, and the MMA shape becomes the one that runs at peak.
//
// Measured on this device (jobs dv_9ef28d29d53c, dv_1f1369184437):
//   * pair M256N128K16 A-from-smem: 64.02 cyc/MMA = 8189 flop/cyc/SM = 100% of
//     the 8192 peak, identical at grid=2 and grid=148 (74 clusters, min=max over
//     all 74 leaders)
//   * the full per-tile stream (12 QK A-from-smem + 8 PV A-from-TMEM):
//     1280.3 cyc, against tc2's 1643.2 for the same 16384 score elements and the
//     same 128 output dims -> 1.284x on the MMA floor
//   * a pair MMA takes A from TMEM only with the bracketed "[%0], [%1], ..."
//     spelling tc2 has always used; the unbracketed one is rejected by ptxas
//   * tcgen05.commit.cta_group::2 alone reaches ONLY the leader's CTA (the peer
//     spun a 2M-cycle budget), but the .multicast::cluster form lands on both
//     CTAs' copies of the barrier at the same time (561/561 cyc) -> used for
//     every MMA-completion signal here
//   * remote arrives work in both directions with count-2 aggregation (139 cyc)
//     but cost 329-824 cyc to issue, so they are used only where a full pipeline
//     slot of slack exists, never on an MMA-completion path
//   * a leader-only tcgen05.alloc.cta_group::2 gives the peer TMEM, the pair MMA
//     writes M=256 across the pair (both CTAs read back 192.0 from an all-1.0
//     QK), and tcgen05.ld works on pair-allocated TMEM from either CTA
//   * a peer CANNOT poll the leader's barrier (try_wait rejects .shared::cluster)
//
// Per-CTA budgets: TMEM 3 x 128 (S, later packed bf16 P in the first 64 columns
// of each slot) + 128 (O) = 512. smem Q 49152 + 3 stages x (K 24576 + V 16384)
// + ctrl 256 = 172288 B, i.e. 49152 B LIGHTER than tc2's 221440, because B is
// split along N: each CTA holds 64 of the tile's 128 kv tokens of K and 64 of the
// 128 output dims of V, so a stage costs the same 40960 B as tc2's while covering
// twice the kv tokens. Global traffic per unit work is unchanged.
//
// Softmax: 8 warps over the CTA's 128 rows x 128 kv. Warp w owns rows
// 32*(w&3)..+31 and column half (w>>2), so each thread handles exactly the 64
// values tc2's threads handle - but the row max now spans two warps, so the two
// halves exchange their hardware row max through smem (maxbuf) under bar_softmax,
// and the O correction and the epilogue split by output dims instead. The two
// partial row sums are combined once, in the epilogue.
//
// Barrier protocol, per tile i with slot s = i % 3 (one ring index for both the
// KV stage and the S/P slot, because PV(i) is the last consumer of BOTH):
//   BF[s]  count 1, local   - this CTA's TMA for stage s landed
//   BR[s]  count 2, leader  - both CTAs' stage s landed (each CTA's warp 8 waits
//                             its own BF[s] then arrives, remotely from the peer)
//   SF[s]  count 1, both    - QK(i) retired (multicast commit)
//   PF[s]  count 2, leader  - both CTAs' softmax finished tile i: P stored in
//                             TMEM and the O correction applied
//   PV[s]  count 1, both    - PV(i) retired (multicast commit); frees stage s and
//                             S/P slot s together
//   QF     count 1, local   - the Q TMA landed
//
// Warp roles (320 threads, 1 CTA/SM, smem 172288 B):
//   warps 0-7: softmax (rows 32*(w&3), column half w>>2)
//   warp  8:   leader - elected-lane pair MMA issuer; peer - the BR relay
//   warp  9:   elected-lane TMA loader (each CTA loads its own halves)
//
// Step 1 is a correctness skeleton: the pipeline above is complete, but every
// wait is BOUNDED and a timeout is counted into dbg[slot] so a protocol mistake
// reports itself instead of hanging the job. Passes with no timeouts are
// indistinguishable from unbounded waits; -DTC7_UNBOUNDED removes them.
//
// Build: nvcc -gencode=arch=compute_103a,code=sm_103a -lcuda

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define DIN __device__ __forceinline__

namespace mla_tc7 {

constexpr int DQK = 192;
constexpr int DV = 128;
constexpr int SUB = 128;            // q rows per CTA
constexpr int BMP = 2 * SUB;        // q rows per cluster pair
constexpr int BN = 128;             // kv tokens per tile, pair-wide
constexpr int BNH = BN / 2;         // kv tokens of K held by one CTA
constexpr int DVH = DV / 2;         // output dims of V held by one CTA
constexpr int KV_STAGES = 3;
constexpr int S_STAGES = 3;
#ifdef TC7_PF_RELAY
constexpr int THREADS = 352;   // + warp10: the peer's PF relay
#else
constexpr int THREADS = 320;
#endif
constexpr int SOFT_WARPS = 8;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -1.0e30f;
constexpr float LAG_MAX_LOG2 = 8.0f;

constexpr int Q_BYTES = SUB * DQK * 2;                        // 49152
constexpr int K_BYTES = BNH * DQK * 2;                        // 24576
constexpr int V_BYTES = BN * DVH * 2;                         // 16384
constexpr int KV_TOTAL = KV_STAGES * (K_BYTES + V_BYTES);     // 122880
constexpr int CTRL_OFF = Q_BYTES + KV_TOTAL;                  // 172032
// ctrl: NBARS mbarriers (8 B each) + the pair TMEM holder (8 B) + the 8x32
// cross-warp row-max exchange buffer (1024 B, reused for the epilogue's partial
// row sums)
#ifdef TC7_XMAX1
constexpr int CTRL_BYTES = 2560;   // two rotating 8x32 max buffers
#else
constexpr int CTRL_BYTES = 1280;
#endif
// tc9 is PERSISTENT: a cluster keeps its smem, its TMEM and its pipeline across
// items, so the epilogue can no longer stage O over "the dead K stages" - with a
// global ring the loader is refilling those stages for the NEXT item while this
// item's O is being packed. The staging image gets its own 32768 B instead, which
// fits: 180224 + 32768 = 212992 of the 232448 B opt-in, 19456 B spare, and still
// one CTA per SM either way so nothing else about the launch changes.
// The staged image is the SOURCE of a SW128-swizzled cp.async.bulk.tensor store,
// so its base has to carry the alignment the swizzle pattern is defined against:
// tc8 staged at Q_BYTES = 49152, which is 16384-aligned. A merely 128-byte-aligned
// base (CTRL_OFF + CTRL_BYTES = 173312) leaves the hardware and ob_addr()
// disagreeing about the swizzle phase and permutes the image. Measured that way in
// dv_74bfaa0a55f0: every shape whose bulk_ok is true wrong by up to 8.25 while
// bad_lse stayed 0 and the two shapes that take the scattered path (x1, x127)
// were bit-clean - lse never passes through the staging buffer, which is what made
// the split diagnosable from the correctness pattern alone.
constexpr int EPI_OFF = (CTRL_OFF + CTRL_BYTES + 16383) / 16384 * 16384;  // 180224
constexpr int STAGE_BYTES = SUB * DV * 2;                     // 32768 epilogue stage
constexpr int SMEM_BYTES = EPI_OFF + STAGE_BYTES;             // 212992
// Clusters a persistent launch may occupy: 148 SMs / 2 CTAs per pair. Launching
// more than this only re-introduces the waves the work loop exists to remove.
constexpr int MAX_CLUSTERS = 74;

// TMEM column map: S/P slot s at 128s (f32 S in all 128 columns, packed bf16 P
// in the first 64), O at 384.
DIN int TCOL_S(int s) { return s * 128; }
constexpr int TCOL_O = 384;

enum { BF0 = 0, BR0 = 3, SF0 = 6, PF0 = 9, PV0 = 12, QF = 15,
#ifdef TC7_PF_RELAY
       PL0 = 16,
       // OL: this CTA's epilogue has finished READING O out of TMEM. ODR: the
       // leader's aggregated copy (count 2 = its own softmax + the peer's warp10
       // relay), which is what the issuer waits before the first PV of the next
       // item overwrites the pair's single O accumulator. Both exist only because
       // the kernel is persistent: within one item, O is written by PV and read by
       // the epilogue in program order.
#ifdef TC9_KVSPLIT
       OL = 19, ODR = 20, BFK0 = 21, BRK0 = 24, NBARS = 27 };
#else
       OL = 19, ODR = 20, NBARS = 21 };
#endif
#else
#ifdef TC9_KVSPLIT
       OL = 16, ODR = 17, BFK0 = 18, BRK0 = 21, NBARS = 24 };
#else
       OL = 16, ODR = 17, NBARS = 18 };
#endif
#endif
// dbg slots for bounded-wait timeouts (step 1 only)
enum { DBG_TO_BF = 16, DBG_TO_BR, DBG_TO_SF, DBG_TO_PF, DBG_TO_PV, DBG_TO_QF,
#ifdef TC9_KVSPLIT
       DBG_TO_BFV = 22,   // KVSPLIT: own-V wait at the PV point
#endif
       DBG_TO_ODR = 23, DBG_TILES = 24, DBG_CTAS = 25, DBG_ITEMS = 26 };

#ifdef TC9_TIMING
// Stage accounting in kernel_tc2p.cu's slot layout, so experiments/timing_tc2.py
// decodes it unchanged and the cluster engine's cycles per tile can be set next to
// the single-CTA engine's without a second decoder: softmax 16-22 accumulated by
// tid==0 (16 tiles, 17 tmem_ld, 18 masked max + cross-warp exchange, 19 exp/pack/
// row-sum/P-store, 20 the SF wait for this tile's QK result, 21 the O correction,
// 22 whole-tile period), issuer 24-30 from the leader's warp 8 (24 tiles, 25 the
// BF/QF/BR waits, 26 the PV slot-free wait, 27 QK issue, 28 the ODR+PF waits, 29 the
// issuer's whole-tile period - tc2p uses 29 for its OC wait, which tc9 does not have
// because PF merges PF and OC - and 30 PV issue), lifetime 48-52 from
// tid==0 (48 span_sum, 49 ctas, 50 span_max, 51 prologue_sum, 52 epilogue_sum,
// 54 tile count for the decoder's T_mean).
// Collisions this respects: DBG_CTAS(25) and DBG_ITEMS(26) would land on the
// issuer's slots, so both are compiled out under TC9_TIMING; DBG_TILES is 24, which
// is exactly is_tiles, so it stays; the DBG_TO_* slots are written only by the
// bounded-wait build and a timing build is always -DTC7_UNBOUNDED.
// TC9_TIMING2 (E34, requires TC9_TIMING; never in production builds) adds the
// contiguous-chain decomposition of the softmax warp's unaccounted cadence:
// 31 item-boundary interval (decode_item + loop admin; chain: previous
// epilogue's c0 restart -> first tile's SF-wait start), 32 first-tile SF
// wait = refill bubble (SUBSET of 20), 33 epilogue last-PV retire wait
// (SUBSET of 52), 34 inter-tile loop residue for non-first tiles (~0 =
// conservation check), 35 items completed per CTA. Conservation identity:
// span(48) ~= prologue(51) + boundary(31) + tiles*(17..22) + gap(34) +
// epilogue(52), all per CTA; residual reports measurement leakage.
// The macros expand to nothing without the flag, so the production binary is
// byte-for-byte what it was - the contract kernel_tc2p.cu's TC2_TIMING blocks have.
#define T9MARK(cond, v) do { if (cond) { (v) = clock64(); } } while (0)
#define T9ACC(cond, acc, from) do { if (cond) { \
    const long long _t = clock64(); (acc) += _t - (from); (from) = _t; } } while (0)
#define T9DECL_SM long long sm_tiles = 0, sm_sld = 0, sm_max = 0, sm_exp = 0, \
    sm_bew = 0, sm_corr = 0, sm_per = 0, c0 = 0, e0 = 0; \
    const bool timed_s = (dbg != nullptr) && (tid == 0);
#ifdef TC9_TIMING2
// E34 fine-grained decomposition of the softmax warp's UNACCOUNTED cadence
// (E32: 28-35% of span/tile on shallow scored cases sits outside every
// named stage). Contiguous-chain slots: 31 item-boundary interval
// [previous chain end -> first tile's SF-wait start] = decode_item +
// loop admin; 32 first-tile SF wait (refill bubble; SUBSET of 20);
// 33 epilogue PV-retire wait (SUBSET of 52); 34 inter-tile loop residue
// for non-first tiles (should be ~0 - conservation check); 35 items done
// per CTA (the timing build compiles out DBG_ITEMS=26). Inert unless BOTH
// TC9_TIMING and TC9_TIMING2 are defined; production builds define
// neither.
#define T9DECL_SM2 long long sm_boundary = 0, sm_bew_first = 0, epi_pv = 0, \
    sm_gap = 0, items_done = 0;
#else
#define T9DECL_SM2
#endif
#ifdef TC9_TIMING4
#ifndef TC9_TIMING
#error "TC9_TIMING4 requires TC9_TIMING (shares the timed_s/c0 chain)"
#endif
// E43 (probe198): substage decomposition of slot 19 (exp/pack/rowsum/
// P-store) to localise the tc9-vs-tc2p per-element residual (755-768 vs
// 581-712 cyc/tile at identical 16384 S-elements/CTA/tile and identical
// masked fractions; probe197 census). Slots: 36 = the exp/pack loops incl.
// the EARLY_PST store ISSUES, 37 = tmem_wait_st, 38 = the d update.
// Conservation: 36+37+38 == 19 exactly (t4s is captured at the sm_max
// ACC; nothing moves c0 in between). tc2p's TC2_TIMING4 mirrors this at
// slots 55-57 with one span difference: tc2p's tail (57) also contains
// fence_before_sync + the pfB mbar_arrive, tc9's PF arrive sits after
// sm_corr - compare 36-vs-55 (loop) and 37-vs-56 (wait_st) directly and
// read 38-vs-57 with that asymmetry in mind. Never in production builds.
#define T9DECL_SM4 long long smx_loop = 0, smx_wst = 0, smx_tail = 0;
#else
#define T9DECL_SM4
#endif
#ifdef TC9_TIMING5
#ifndef TC9_TIMING
#error "TC9_TIMING5 requires TC9_TIMING"
#endif
// E45 (probe202): split sm_per (slot 22 = the PF path: fence_before_sync +
// bar_softmax convergence + the warp0-lane0 local arrive) into slot 41 =
// fence+bar convergence and slot 42 = the arrive itself. Conservation:
// 41+42 == 22. Question: how much of the 128 cyc/tile is the 8-warp
// bar.sync convergence (attackable via a tc2p-style local-count PF
// barrier) vs the mbar_arrive (arm-b's measured relay optimum)?
#define T9DECL_SM5 long long sm_barconv = 0, sm_pfarr = 0;
#else
#define T9DECL_SM5
#endif
#define T9DECL_IS long long is_bf = 0, is_pvd = 0, is_qk = 0, is_pfw = 0, \
    is_pv = 0, is_per = 0, i0 = 0; \
    const bool timed_i = (dbg != nullptr) && (tid == 256);
#else
#define T9MARK(cond, v) do {} while (0)
#define T9ACC(cond, acc, from) do {} while (0)
#define T9DECL_SM
#define T9DECL_SM2
#define T9DECL_SM4
#define T9DECL_SM5
#define T9DECL_IS
#endif

DIN uint32_t sm32_(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
DIN uint32_t bar(char* ctrl, int idx) { return sm32_(ctrl) + idx * 8; }

// ---------------------------------------------------------------------------
// PTX primitives. The 1-CTA forms are copied from the promoted kernel_tc2.cu;
// the cluster forms are the spellings measured by experiments/micro.cu (ladder
// rungs r5-r8 and cls0-cls7).
// ---------------------------------------------------------------------------
DIN void tma_load_4d(const CUtensorMap* map, uint32_t dst, uint32_t mbar,
                     int c0, int c1, int c2, int c3) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];"
      ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar)
      : "memory");
}
DIN void tma_prefetch_4d(const CUtensorMap* map, int c0, int c1, int c2,
                         int c3) {
  asm volatile(
      "cp.async.bulk.prefetch.tensor.4d.L2.global.tile"
      " [%0, {%1, %2, %3, %4}];"
      ::"l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
DIN void tma_store_4d(const CUtensorMap* map, uint32_t src, int c0, int c1,
                      int c2, int c3) {
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
      " [%0, {%1, %2, %3, %4}], [%5];"
      ::"l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(src) : "memory");
}
DIN void bulk_commit() { asm volatile("cp.async.bulk.commit_group;"); }
DIN void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;"); }

DIN void mbar_init(uint32_t a, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
}
DIN void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               ::"r"(a), "r"(bytes) : "memory");
}
DIN void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
               ::"r"(a) : "memory");
}
DIN void mbar_wait_h(uint32_t a, int phase, uint32_t hint) {
  asm volatile(
      "{\n\t.reg .pred P;\n\tLABW%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, %2;\n\t"
      "@P bra.uni DONEW%=;\n\tbra.uni LABW%=;\n\tDONEW%=:\n\t}\n"
      ::"r"(a), "r"(phase), "r"(hint)
      : "memory");
}
DIN void mbar_wait(uint32_t a, int phase) { mbar_wait_h(a, phase, 0x989680u); }

// cluster scope
DIN unsigned cluster_rank() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
// address of the same smem location in CTA `rank` of this cluster
DIN uint32_t mapa_(uint32_t addr, unsigned rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
// Cross-CTA arrive. Measured per call (micro.bin clbar, 200 iterations):
//   release, back to back ............ 276.6 cyc
//   release, after an MMA wait ....... 272 cyc  (the state tc7's softmax is in)
//   relaxed, back to back ............  79.6 cyc
//   cluster barrier arrive+wait ...... 364.7 cyc
// The relaxed form is 3.5x cheaper. It is defensible here because the payload the
// signal publishes is in TMEM, not generic memory: the producer has already done
// tcgen05.wait::st and tcgen05.fence::before_thread_sync, and the consumer does
// tcgen05.fence::after_thread_sync before issuing, which is the documented
// ordering for TMEM across a thread sync. Nothing in generic memory is being
// published, so the arrive's own release semantics are not what carries the
// ordering. It is still a correctness risk and is validated by the boundary set
// and the race hunt before it goes anywhere near the policy.
#ifdef TC7_RELAX_ARRIVE
DIN void mbar_arrive_remote(uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
               ::"r"(cluster_addr) : "memory");
}
#else
DIN void mbar_arrive_remote(uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
               ::"r"(cluster_addr) : "memory");
}
#endif
// TC7_NO_CLSYNC replaces the cluster rendezvous with a CTA-local one. Only a
// diagnostic: it separates "the pair TMEM alloc blocks" from "barrier.cluster
// blocks", which the staged ladder cannot do on its own because stage 2 contains
// both. Without a real rendezvous the peer could read tholder before the leader's
// alloc published it, so results from this build are meaningless - only whether
// it RETURNS is.
#ifdef TC7_NO_CLSYNC
DIN void cluster_sync() { __syncthreads(); }
#else
DIN void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
#endif
// Bounded wait: step 1 counts a timeout into dbg[slot] and proceeds rather than
// hanging the job, so a protocol mistake is diagnosable from the output.
#ifdef TC7_UNBOUNDED
DIN void mbwait(uint32_t a, int phase, unsigned long long* dbg, int slot,
                uint32_t hint) {
  (void)dbg; (void)slot;
  mbar_wait_h(a, phase, hint);
}
#else
DIN void mbwait(uint32_t a, int phase, unsigned long long* dbg, int slot,
                uint32_t hint) {
  (void)hint;
  long long t0 = clock64();
  int done = 0;
  while (!done && clock64() - t0 < 200000LL) {
    asm volatile("{\n\t.reg .pred P;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P;\n\t}\n" : "=r"(done) : "r"(a), "r"(phase));
  }
  if (!done && dbg) atomicAdd(dbg + slot, 1ull);
}
#endif

DIN int elect_one() {
  int pred = 0;
  asm volatile("{\n\t.reg .pred p;\n\telect.sync _|p, 0xFFFFFFFF;\n\t"
               "selp.s32 %0, 1, 0, p;\n\t}\n" : "+r"(pred));
  return pred;
}
DIN uint64_t make_desc(uint32_t smem_addr, uint32_t lbo_bytes,
                       uint32_t sbo_bytes) {
  uint32_t lo = ((smem_addr >> 4) & 0x3FFF) | ((lbo_bytes >> 4) << 16);
  uint32_t hi = (2u << 29) | (1u << 14) | ((sbo_bytes >> 4) & 0x3FFF);
  return (static_cast<uint64_t>(hi) << 32) | lo;
}
// pair idesc: the M field is bits 24-28 = M>>4, so 256>>4 = 16 (tc2 uses 8 for
// M=128). N=128, bf16 A and B, f32 accumulate; the PV form also sets b_major.
constexpr uint32_t IDESC_QK = (uint32_t(256 >> 4) << 24) |
                              (uint32_t(128 >> 3) << 17) |
                              (1u << 10) | (1u << 7) | (1u << 4);
constexpr uint32_t IDESC_PV = IDESC_QK | (1u << 16);
// A operand from a smem descriptor, pair granularity
DIN void mma_ss2(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc,
                 int accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::
          "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum));
}
// A operand from TMEM, pair granularity. The BRACKETS are load-bearing: without
// them ptxas reports "Arguments mismatch for instruction tcgen05.mma" (ladder
// rung r7), and at runtime the TMEM address must be 8-column aligned or the
// launch dies with "misaligned address".
DIN void mma_ts2(uint32_t tmem_d, uint32_t tmem_a, uint64_t db, uint32_t idesc,
                 int accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" ::
          "r"(tmem_d), "r"(tmem_a), "l"(db), "r"(idesc), "r"(accum));
}
// Completion of all previously issued pair MMAs, signalled to BOTH CTAs of the
// pair at the same local offset. Without .multicast::cluster only the leader's
// barrier fires (measured: the peer spun a 2M-cycle budget).
DIN void mma_commit_mc(uint32_t mbar) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
      ".multicast::cluster.b64 [%0], %1;" ::"r"(mbar), "h"(uint16_t(3)));
}

DIN void tmem_ld32(float* d, uint32_t ta) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32"
      " {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]),
        "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]),
        "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]),
        "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]),
        "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]),
        "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
      : "r"(ta));
}
// 64-column load that also returns the row maximum over those 64 columns
// (measured: same rate as a plain x64 load, bit-exact against a FMNMX tree).
DIN float tmem_ld64_redmax(float* d, uint32_t ta) {
  float m;
  asm volatile(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32 "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,"
      "%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,"
      "%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64}, "
      "%0, [%65];"
      : "=f"(m), "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
        "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]),
        "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]),
        "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
        "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]),
        "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]),
        "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]),
        "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]),
        "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]),
        "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]),
        "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]),
        "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]),
        "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
      : "r"(ta));
  return m;
}
DIN void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
DIN void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
DIN void tmem_st32(uint32_t ta, const float* d) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::
      "r"(ta), "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "f"(d[4]), "f"(d[5]),
      "f"(d[6]), "f"(d[7]), "f"(d[8]), "f"(d[9]), "f"(d[10]), "f"(d[11]),
      "f"(d[12]), "f"(d[13]), "f"(d[14]), "f"(d[15]), "f"(d[16]), "f"(d[17]),
      "f"(d[18]), "f"(d[19]), "f"(d[20]), "f"(d[21]), "f"(d[22]), "f"(d[23]),
      "f"(d[24]), "f"(d[25]), "f"(d[26]), "f"(d[27]), "f"(d[28]), "f"(d[29]),
      "f"(d[30]), "f"(d[31]) : "memory");
}
DIN void tmem_st16bf(uint32_t ta, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::
      "r"(ta), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]),
      "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]),
      "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]) : "memory");
}
DIN void fence_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;"); }
DIN void fence_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }

DIN uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
DIN float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// Packed exp2 for the softmax (TC8_EX2F16). Textually the variant that
// kernel_tc2p.cu ships as its production softmax (-DTC2_EX2F16), so the two
// engines quantise the probabilities identically: the exp ARGUMENT is rounded
// to f16, one ex2.approx.f16x2 evaluates the pair, and the results are unpacked
// to f32 for the row sum and repacked to bf16 for P (P storage, the IDESC and
// every MMA operand are unchanged).
//
// Why it pays: the exp chain is issue/throughput bound at ~1.7x its XU floor
// and is flat in warps per scheduler partition (experiments/notes/mma-rates.md:
// 866.5 cyc per 16384 values at 2 warps/partition, 828.8 at 4, 846.7 at 8), so
// the lever is instruction count, not more parallelism. Packing halves the MUFU
// instructions per tile; measured on this device ex2.approx.f16x2 delivers the
// SAME 32 values/cyc/SM as ex2.approx.f32, i.e. the win is the halved issue
// slots, and kernel_tc2p.cu's own accounting reports sm_exp 1236 -> 736 cyc.
//
// Precision: dominant weights (argument near 0) keep full accuracy; tail weights
// take ~0.3-1% relative input-quantization error against a 5% checker rtol. P is
// stored bf16 (8 mantissa bits) either way, so the f16 intermediate (11 bits)
// is never the narrowest format in the chain. Masked columns are fed a -200.0f
// sentinel rather than a post-hoc select so no inf/NaN can reach the f16 exp.
#ifdef TC8_EX2F16
DIN uint32_t cvt_f16x2(float lo, float hi) {
  uint32_t y;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(y) : "f"(hi), "f"(lo));
  return y;
}
DIN uint32_t ex2_f16x2(uint32_t x) {
  uint32_t y;
  // No .ftz on the f16x2 form (ptxas: "Illegal modifier '.ftz'"); underflow to
  // zero is implicit, which is what makes the -200.0f sentinel exact.
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}
DIN float f16lo_f32(uint32_t p) {
  float y;
  asm("{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, %1;\n\t"
      "cvt.f32.f16 %0, lo;\n\t}\n" : "=f"(y) : "r"(p));
  return y;
}
DIN float f16hi_f32(uint32_t p) {
  float y;
  asm("{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, %1;\n\t"
      "cvt.f32.f16 %0, hi;\n\t}\n" : "=f"(y) : "r"(p));
  return y;
}
#endif
// SW128 image addresses. Q and K are K-major with 128-row (Q) / 64-row (K) atom
// planes; the epilogue staging image matches the rank-4 out map's 128-row box.
DIN uint32_t ob_addr(int row, int chunk16) {
  int atom = chunk16 >> 3;
  int c = chunk16 & 7;
  return atom * 16384 + (row >> 3) * 1024 + (row & 7) * 128 + ((c ^ (row & 7)) << 4);
}
// 8 softmax warps (256 threads); the issuer/loader warps never arrive.
DIN void bar_softmax() { asm volatile("bar.sync 1, 256;" ::: "memory"); }

// ---------------------------------------------------------------------------
// Per-item geometry (tc9 is persistent: this runs once per item, per role)
// ---------------------------------------------------------------------------
// There is NO device work counter and NO cross-CTA item publication. The item
// sequence is a pure function of (cluster id, cluster count) and this decode is a
// pure function of the item index plus the immutable cum_q/cum_kv, so all ten warp
// roles in both CTAs of a pair iterate the same items, skip the same empty ones and
// advance their global tile counter identically - which is what keeps the
// cluster-scope barriers (BR, PF, and the new ODR) in step. A counter would need
// the leader to fetch, publish through DSMEM and gate on a CONS aggregation; a
// pure function needs nothing, at the cost of static rather than dynamic balance
// (the strided assignment below is what spreads heavy and light items).
struct Item {
  int q_base, kv_base, qlen, kvlen, pm0;
  int m0, kv_off, T, Tfull, h, h_kv, tok0;
  int64_t o_off;
  float sink2;
  bool windowed, bulk_ok;
};

// ---------------------------------------------------------------------------
// Static item-assignment balance: a globally descending item order (TC9_GSORT)
// and a serpentine stride (TC9_SERP). Both are pure functions of (blockIdx.x,
// gridDim.x) plus the immutable cum_q/cum_kv, so neither adds a counter, a
// handshake or a barrier to the item sequence, and neither can change an output
// element: they change WHICH cluster computes an item, not what the item is.
//
// Why. decode_item's own comment prices the plain stride at 1.270x the mean load
// over the ten fixed cases and 1.713x on f9, against 1.084x for a dynamic
// counter, and rejects the counter because it needs a leader-fetch/peer-publish
// handshake per item. experiments/item_balance.py rebuilds that simulation from
// this file's constants and reproduces both recorded numbers exactly (f9
// rank/stride 1.713, head/stride 2.398), which is what makes the rest of it
// usable. Two things fall out:
//   * the list is ONE descending ramp only when batch == 1. With batch > 1 it is
//     `batch` CONCATENATED ramps, so one request's lightest tiles precede the
//     next request's heaviest - which is why a serpentine stride alone helps
//     f2/f4/f9 and badly hurts f6 (1.310 -> 1.438) and f8 (1.358 -> 1.697).
//   * sorting the pair ids by descending T across all requests, then zigzagging
//     the stride, improves or ties every one of the ten cases (mean 1.281 ->
//     1.113) and lands within 2% of the dynamic counter's 1.089, statically.
// A latency model fitted to the measured tc9r oracle (rmse 1.13%, and its
// 0.958 us/tile independently matches the 2033-cycle span/tile measured in
// dv_8bac3774fc32) then puts f9 at 38.30 -> 29.96, f7 91.95 -> 82.41, f3
// 52.37 -> 48.56, f4 231.91 -> 200.67 and f2 100.77 -> 85.51.
//
// What it deliberately does NOT change is the locality property rank-major was
// chosen for: the `heads` items of one pair-tile stay consecutive (only the
// ORDER OF PAIRS is permuted, item % heads is untouched), so the ~74 concurrent
// clusters still span one or two pair-tiles and their KV working set is the size
// decode_item's comment prices, not the 776 MB head-major-avoiding case. Sorting
// individual items rather than pairs would have balanced marginally better and
// reopened exactly the trade-off arm b measured.
#ifdef TC9_PREMETA
// cum_q/cum_kv smem mirrors - port of the adopted TC2P_PREMETA mechanism (E5
// measured d3 -2.1%, d5 -1.6%, f1 -1.1% on the tc2p family). decode_item's
// per-item cum scan re-reads the two small global arrays from five call sites
// (both CTAs' loader/issuer/softmax/epilogue roles), and their L1 lines are
// continuously evicted by the KV stream, so every item re-pays L2/DRAM
// latency for the same bytes. The 64-entry mirrors live in the dead alignment
// gap between the ctrl/GSORT region and the 16384-aligned staging image
// ([174336, 174848) of the [173312, 180224) gap), so SMEM_BYTES and every
// other layout constant stay untouched. Warp 9 issues the cold reads at entry
// (latency overlaps the init/fence/alloc chain, tc2p's trick) and lands them
// before the init __syncthreads(), which publishes them to every role;
// batch > 63 (outside the frozen domain) falls back to the global arrays.
#define T9_CUMQ_OFF (CTRL_OFF + CTRL_BYTES + 1024)   // past any GSORT table
#define T9_CUMKV_OFF (T9_CUMQ_OFF + 64 * 4)
static_assert(T9_CUMKV_OFF + 64 * 4 <= EPI_OFF,
              "cum mirror must stay inside the ctrl/epilogue padding");
#endif
#ifdef TC9_GSORT
// The table lives in the alignment padding between the ctrl block and the
// 16384-aligned epilogue staging image, so SMEM_BYTES does not move, EPI_OFF's
// swizzle alignment is untouched, and nothing else in the layout shifts.
#define TC9_PERM_OFF (CTRL_OFF + CTRL_BYTES)
// sum(query_lengths) <= 16384 and batch <= 32 bound pairs_ub at 96; the cap is
// headroom, and exceeding it degrades to today's order instead of overflowing.
#define TC9_PERM_CAP 128
#define TC9_PERM_BYTES (TC9_PERM_CAP * 2 * sizeof(int))   // perm[] and T[]
static_assert(TC9_PERM_BYTES <= (size_t)(EPI_OFF - TC9_PERM_OFF),
              "the permutation must fit the ctrl/epilogue alignment padding");

// Tiles for one pair id: decode_item's T arithmetic with the per-CTA fields
// dropped, so a table of them can be built before the roles split.
DIN int pair_tiles(const int32_t* cum_q, const int32_t* cum_kv, int batch,
                   int causal, int pair_id) {
  int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
  for (int i = 0; i < batch; ++i) {
    const int nq = cum_q[i + 1], nkv = cum_kv[i + 1];
    const int len_q = nq - prev_q, len_kv = nkv - prev_kv;
    const int tiles = (len_q + BMP - 1) / BMP;
    if (pair_id < acc + tiles) {
      const int pm0 = (tiles - 1 - (pair_id - acc)) * BMP;
      int nmax = causal ? min(len_kv, (len_kv - len_q) + pm0 + BMP) : len_kv;
      nmax = min(nmax, len_kv);
      const int T = (nmax + BN - 1) / BN;
      return T > 0 ? T : 0;
    }
    acc += tiles; prev_q = nq; prev_kv = nkv;
  }
  return 0;                                   // surplus pair id: zero work
}

// perm[r] = the pair id holding the r-th largest T, ties by ASCENDING pair id.
// Ranked by a scan rather than a value histogram on purpose: T is bounded by
// kvlen/BN and the domain allows sum(kv_lengths) up to 1048576, so T reaches
// 8192 and a histogram would not fit the padding, while the scan is O(pairs_ub)
// per thread with pairs_ub <= 96. The tie rule is what makes the table
// bit-identical in both CTAs of a pair and in all ten warp roles without any of
// them communicating - an atomic placement would order equal-T pairs by arrival
// and the two CTAs could then walk different item sequences, which is exactly
// the deadlock the pure-function design exists to avoid.
// Caller must __syncthreads() afterwards to publish it.
// __noinline__ deliberately, and against this file's DIN convention: this runs
// ONCE per CTA, and force-inlining it puts its whole body - the per-pair tile
// arithmetic and the O(pairs_ub) rank scan - into the kernel's entry block ahead
// of five tight warp-specialised loops. dv_897e0ef8a5db measured the cost of
// carrying that code at all: on the six gate-firing shapes where the mechanism
// provably cannot act (single-pass, so k is always 0, or non-causal, so both flags
// are off) and the item order is therefore IDENTICAL to the pre-adoption build,
// the adopted engine was still 1.7-3.3% slower with 9/9 and 8/9 round consistency
// against a same-binary noise floor of 0.7%. A call once per CTA is ~20 cycles.
__device__ __noinline__ void build_perm(const int32_t* cum_q,
                                        const int32_t* cum_kv, int batch,
                                        int causal, int pairs_ub, char* smem) {
  int* const perm = reinterpret_cast<int*>(smem + TC9_PERM_OFF);
  int* const tbuf = perm + TC9_PERM_CAP;
  const int tid = threadIdx.x;
  if (pairs_ub > TC9_PERM_CAP) {
    for (int p = tid; p < TC9_PERM_CAP; p += blockDim.x) perm[p] = p;
    return;                        // uniform across the CTA: no barrier skew
  }
  for (int p = tid; p < pairs_ub; p += blockDim.x)
    tbuf[p] = pair_tiles(cum_q, cum_kv, batch, causal, p);
  __syncthreads();
  for (int p = tid; p < pairs_ub; p += blockDim.x) {
    const int tp = tbuf[p];
    int r = 0;
    for (int q = 0; q < pairs_ub; ++q) {
      const int tq = tbuf[q];
      r += (tq > tp) | ((tq == tp) & (q < p));
    }
    perm[r] = p;
  }
}
#endif

// Slot bookkeeping for the work loops, hoisted out of them. Pass k deals slot
// slotA when k is even and slotB when k is odd, so pass k+1 fills the clusters
// that drew the LIGHTEST item in pass k instead of piling onto the ones that drew
// the heaviest. With t9_serp false slotB == slotA == cid and the loops reduce to
// today's `item = cid; item += ncl` exactly, one add per iteration. The increment
// MUST stay in the for clause and not at the end of the body: all five loops
// `continue` on an empty item, and a body-tail increment would be skipped and hang
// the kernel.
struct T9Slots {
  int a, b;
};

DIN bool decode_item(const int32_t* cum_q, const int32_t* cum_kv, int batch,
                     int heads, int kv_heads, int causal, int window_left,
                     int has_sink, const float* sinks, int pairs_ub,
                     long long item, unsigned rank, Item& it,
                     const int* perm = nullptr) {
  // Flattening of the (pair-tile, head) item space, and it is a real trade-off:
  //   head-major (item % pairs_ub) keeps the ~74 concurrently running items
  //     inside one or two heads, so their K/V shares an L2 working set of ~21 MB
  //     on f5 instead of ~126 MB against 132 MB of L2 - but the item list is then
  //     `heads` separate descending ramps, and a strided cluster samples one ramp
  //     repeatedly: simulated static makespan 1.586x the mean over the ten fixed
  //     cases (2.398x on f9).
  //   rank-major (item / heads) makes the list ONE globally descending ramp, so
  //     the same stride samples every weight band: 1.270x (1.713x on f9). It
  //     gives up the head locality above.
  // A device work counter would have both - dynamic assignment reaches 1.084x AND
  // hands out adjacent items, so concurrency stays head-local - at the cost of a
  // leader-fetch/peer-publish handshake per item. Measured both ways before
  // paying for that.
#ifdef TC9_RANKMAJOR
#ifdef TC9_GSORT
  // item / heads is the pair's RANK in the item list. TC9_GSORT replaces the list
  // order with globally descending T, so the rank goes through the table the
  // prologue built once per CTA. The bound check costs nothing in-domain
  // (pairs_ub <= 96 < TC9_PERM_CAP) and keeps an out-of-domain shape reading its
  // own identity tail instead of running off the array.
  const int prank = static_cast<int>(item / heads);
  const int pair_id = (perm != nullptr && prank < TC9_PERM_CAP) ? perm[prank]
                                                                : prank;
#else
  (void)perm;
  const int pair_id = static_cast<int>(item / heads);
#endif
  it.h = static_cast<int>(item % heads);
#else
  (void)perm;      // head-major has no single descending rank to sort
  const int pair_id = static_cast<int>(item % pairs_ub);
  it.h = static_cast<int>(item / pairs_ub);
#endif
  // ---- pair mapping: BM=256 q-tiles, LPT order within each request ----
  int b = -1, pm0 = 0, q_base = 0, kv_base = 0, qlen = 0, kvlen = 0;
  {
    int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
    for (int i = 0; i < batch; ++i) {
      int nq = cum_q[i + 1], nkv = cum_kv[i + 1];
      int len_q = nq - prev_q;
      int tiles = (len_q + BMP - 1) / BMP;
      if (b < 0 && pair_id < acc + tiles) {
        b = i;
        // LPT: heaviest q-tile of the request first (longest causal KV sweep)
        pm0 = (tiles - 1 - (pair_id - acc)) * BMP;
        q_base = prev_q; kv_base = prev_kv;
        qlen = len_q; kvlen = nkv - prev_kv;
        break;
      }
      acc += tiles; prev_q = nq; prev_kv = nkv;
    }
  }
  if (b < 0) return false;             // surplus pair id: both CTAs agree
  it.q_base = q_base; it.kv_base = kv_base; it.qlen = qlen; it.kvlen = kvlen;
  it.pm0 = pm0;
  it.m0 = pm0 + static_cast<int>(rank) * SUB;   // this CTA's first q row
  it.kv_off = kvlen - qlen;
  it.windowed = window_left >= 0;
  // T is a PAIR quantity: both CTAs must run the same number of tiles or the
  // cluster-scope barriers would deadlock. Rows past qlen produce finite garbage
  // that is row-local through softmax/PV and never stored.
  int nmax_p = causal ? min(kvlen, it.kv_off + pm0 + BMP) : kvlen;
  nmax_p = min(nmax_p, kvlen);
  const int Traw = (nmax_p + BN - 1) / BN;
#if defined(TC7_STAGE) && TC7_STAGE == 4
  it.T = Traw < 1 ? Traw : 1;
#elif defined(TC7_STAGE) && TC7_STAGE == 5
  it.T = Traw < 2 ? Traw : 2;
#else
  it.T = Traw;
#endif
  // per-CTA masking: the first tile whose columns are all unmasked for THIS CTA
  int nsafe = causal ? min(kvlen, it.kv_off + it.m0 + 1) : kvlen;
  if (it.windowed) nsafe = 0;
  it.Tfull = nsafe / BN;
  if (it.T <= 0) return false;         // both CTAs skip together
  it.h_kv = it.h / (heads / kv_heads);
  it.tok0 = kv_base;
  it.o_off = static_cast<int64_t>(q_base) * heads + it.h;
  it.sink2 = has_sink ? sinks[it.h] * LOG2E : 0.0f;
  it.bulk_ok = (it.m0 + SUB <= qlen);
  return true;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
#ifdef TC7_LAUNCH_EX
__global__ void __launch_bounds__(THREADS, 1)
#else
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
#endif
mla_tc7_kernel(const __grid_constant__ CUtensorMap krmap,
               const __grid_constant__ CUtensorMap vrmap,
               const __grid_constant__ CUtensorMap qrmap,
               const __grid_constant__ CUtensorMap ormap,
               __nv_bfloat16* __restrict__ out,
               float* __restrict__ lse,
               const float* __restrict__ sinks,
               const int32_t* __restrict__ cum_q,
               const int32_t* __restrict__ cum_kv,
               int32_t total_q, int32_t batch, int heads, int kv_heads,
               int window_left, int causal, int has_lse, int has_sink,
               float bmm1_scale, float bmm2_scale,
               unsigned long long* __restrict__ dbg) {
  (void)total_q;
  extern __shared__ char smem[];
  char* sq = smem;                              // 49152 B Q
  char* sk = smem + Q_BYTES;                    // 3 x 24576 B K halves
  char* sv = sk + KV_STAGES * K_BYTES;          // 3 x 16384 B V halves
  char* ctrl = smem + CTRL_OFF;
  uint32_t* tholder = reinterpret_cast<uint32_t*>(ctrl + NBARS * 8);
  float* maxbuf = reinterpret_cast<float*>(ctrl + NBARS * 8 + 16);  // 8 x 32 f32

  const int tid = threadIdx.x;
#ifdef TC9_TIMING
  const unsigned long long t_ent = (tid == 0) ? clock64() : 0ull;
#endif
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const unsigned rank = cluster_rank();
  const int cid = blockIdx.x >> 1;            // this cluster's id
  const int ncl = gridDim.x >> 1;             // clusters in this launch
  const int pairs_ub = total_q / BMP + batch;
  const long long total_items = static_cast<long long>(pairs_ub) * heads;
#ifdef TC9_PREMETA
  // Issue the cold cum_q/cum_kv reads BEFORE the init/fence/alloc chain so
  // their DRAM latency overlaps ~1-2k cycles of setup (tc2p PREMETA form).
  // Each warp-9 lane covers mirror entries lane and lane+32; entries 0..batch.
  int pm_q0 = 0, pm_kv0 = 0, pm_q1 = 0, pm_kv1 = 0;
  const int pm_cap = batch < 63 ? batch : 63;
  if (warp == 9) {
    if (lane <= pm_cap) { pm_q0 = cum_q[lane]; pm_kv0 = cum_kv[lane]; }
    if (lane + 32 <= pm_cap) {
      pm_q1 = cum_q[lane + 32]; pm_kv1 = cum_kv[lane + 32];
    }
  }
#endif
#ifdef TC9_SERP
  // Serpentine runs only where the work per item VARIES, and that is a mechanism
  // condition rather than a fitted threshold: a non-causal item sweeps the whole
  // KV, so T is constant across a request's tiles and no assignment can beat
  // another, while the zigzag still changes each cluster's item SEQUENCE and so
  // its Q/KV access order. dv_6af9f5aa63a7 measured exactly that split on the 15
  // shapes the quantisation gate fires on: the 8 whose simulated makespan improves
  // won by 4.2-15.9%, and of the 7 whose makespan cannot improve, 5 sit inside the
  // same-binary noise floor - `auto` and tc9r are the SAME build on those shapes
  // and still differ by up to 2.3%, on xr8b256 - while the two multi-pass
  // non-causal ones cost +1.7% (xn4096h12) and +1.0% (xnb2h12). Gating on causal
  // removes those, and forgoes one tile of makespan on one shape in 486 (xgqanc,
  // the only non-causal one a global sort can help). Shapes with total_items <= ncl
  // need no guard: every cluster gets one item, k is always 0, and the zigzag
  // degenerates to today's stride exactly.
  const bool t9_serp = (causal != 0);
  const T9Slots t9sl = {cid, t9_serp ? (ncl - 1 - cid) : cid};
#else
  const bool t9_serp = false;
  // Written through t9_serp so the symbol is used in every flag combination (the
  // build carries no -Werror, but an unused-variable warning in a kernel this
  // size is noise worth not adding); it folds to {cid, cid} at compile time.
  const T9Slots t9sl = {cid, t9_serp ? (ncl - 1 - cid) : cid};
#endif
#ifdef TC9_GSORT
  // A single request's ramp is ALREADY globally descending - T falls with d and the
  // surplus tail is zero - and ties break by ascending pair id, which is the
  // existing order, so with batch == 1 the sort is provably the identity. Building
  // the table anyway costs ~1.4-2.0 us of exposed latency at the top of the
  // prologue. Measured that way in dv_3ac96f20d630: on the batch==1 shapes the
  // gsort builds were slower than their non-gsort twins by exactly that much while
  // producing the identical item order (f9 tc9b 31.90 against tc9s 29.92, f3
  // 48.08/46.68, f0 38.10/36.55, xq156 52.70/51.42, f2 85.28/84.33), and the same
  // constant disappears into the noise on the long batch==1 kernels (f4
  // 203.44/203.71, f5 832.67/836.03), which is what identifies it as a fixed
  // prologue cost rather than a per-item one.
  //
  // causal is in the condition for the same reason batch is: a non-causal item
  // sweeps the whole KV, so T is constant across a request's tiles and the only
  // variation left is BETWEEN requests of different length. Of the 364 batch>1
  // shapes in the 486-shape replay set the sort changes the makespan on 81 and is
  // the identity on 283, and of those 81 exactly one is non-causal.
  //
  // With TC9_SERP compiled in the sort additionally REQUIRES the zigzag: measured
  // alone it is worse than today's order (f6 tc9g 113.74 against tc9r 109.98, f7
  // 93.39/87.16, f8 48.81/46.03) because it reorders the list without changing how
  // the list is dealt. Without TC9_SERP - the tc9g attribution build - it keeps its
  // batch-only condition, so that build still measures what it was built to
  // measure. Both conditions are uniform across the CTA (kernel arguments only), so
  // the barrier inside build_perm is never taken by some threads and not others.
#ifdef TC9_SERP
  const bool t9_sort = t9_serp && (batch > 1);
#else
  const bool t9_sort = (batch > 1);
#endif
  int* const t9_perm = t9_sort ? reinterpret_cast<int*>(smem + TC9_PERM_OFF)
                               : nullptr;
#else
  int* const t9_perm = nullptr;    // decode_item's default: the branch folds away
#endif
  // tc2 measured the suspend-hint form better than a pure spin (its opts bit3) and
  // ships 0x989680. tc7 waits more often per tile than tc2 does - BR and PF are
  // new - so wake latency may cost more here; TC7_SPIN tests hint 0.
#ifdef TC7_SPIN
  const uint32_t whint = 0u;
#else
  const uint32_t whint = 0x989680u;
#endif

  // Everything above the work loop is per-LAUNCH; everything an item needs is
  // decoded inside it, by every role, from the same pure function.
  const int64_t o_stride = static_cast<int64_t>(heads) * DV;
  const float sc2 = bmm1_scale * LOG2E;
  // Global tile counter. The KV-stage ring, the S/P slot ring and EVERY barrier
  // parity run on it across item boundaries, so the pipeline never drains between
  // items: that is the whole point of persisting. Each role advances it by the
  // same it.T over the same items, so the roles cannot drift apart.
  long long Gbase = 0;
  long long fitem = 0;             // this cluster's item ordinal: QF parity

#ifndef TC9_TIMING
  if (dbg && tid == 0) atomicAdd(dbg + DBG_CTAS, 1ull);   // 25 = is_bf under timing
#endif
#if defined(TC7_STAGE) && TC7_STAGE == 0
  // stage 0 tests only that the cluster launch happens at all: 2 CTAs per
  // cluster, 173312 B of dynamic smem each, 320 threads, and a dbg write that
  // proves the kernel body ran. If this hangs, nothing below matters.
  if (dbg && tid == 0) atomicMax(dbg + 40, 0ull);
  return;
#endif

#ifdef TC9_PFL
#ifndef TC7_PF_RELAY
#error "TC9_PFL requires TC7_PF_RELAY (per-warp arrives target the local PF/PL barriers; the peer relay forwards)"
#endif
#ifdef TC9_BISECT_NOEXP
#error "TC9_PFL is incompatible with TC9_BISECT_NOEXP (the bisect's single arrive cannot complete the count-9 PF)"
#endif
#endif
  // ---- control init (parallel: one thread per barrier) ----
  if (tid < NBARS) {
    int cnt = 1;
    const int i = tid;
    if (i >= BR0 && i < BR0 + KV_STAGES) cnt = 2;       // both CTAs' warp 8
#ifdef TC9_KVSPLIT
    else if (i >= BRK0 && i < BRK0 + KV_STAGES) cnt = 2;  // KVSPLIT K relay
#endif
#ifdef TC9_PFL
    else if (i >= PF0 && i < PF0 + S_STAGES) cnt = 9;   // E46: 8 warp lane0 + peer relay
#else
    else if (i >= PF0 && i < PF0 + S_STAGES) cnt = 2;   // both CTAs' softmax
#endif
    else if (i == ODR) cnt = 2;      // own softmax + the peer's relay
    else if (i == OL) cnt = 1;       // this CTA's softmax, once per item
#ifdef TC7_PF_RELAY
#ifdef TC9_PFL
    else if (i >= PL0 && i < PL0 + S_STAGES) cnt = 8;   // E46: 8 warp lane0
#else
    else if (i >= PL0 && i < PL0 + S_STAGES) cnt = 1;   // this CTA's softmax only
#endif
#endif
    mbar_init(bar(ctrl, i), cnt);
  }
  asm volatile("fence.mbarrier_init.release.cluster;");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#ifdef TC9_PREMETA
  // Land the entry-issued reads in the mirrors BEFORE the init barrier below:
  // unlike tc2p (loader-only consumer, __syncwarp) every role of both CTAs'
  // warps decodes from these pointers, so publication rides the CTA-wide sync.
  if (warp == 9) {
    if (lane <= pm_cap) {
      reinterpret_cast<int*>(smem + T9_CUMQ_OFF)[lane] = pm_q0;
      reinterpret_cast<int*>(smem + T9_CUMKV_OFF)[lane] = pm_kv0;
    }
    if (lane + 32 <= pm_cap) {
      reinterpret_cast<int*>(smem + T9_CUMQ_OFF)[lane + 32] = pm_q1;
      reinterpret_cast<int*>(smem + T9_CUMKV_OFF)[lane + 32] = pm_kv1;
    }
  }
#endif
  __syncthreads();
#ifdef TC9_PREMETA
  // Flag-off reduction: cq/ck are const aliases of the global pointers -
  // identical code by construction. batch > 63 (outside the frozen domain):
  // the 64-entry mirror cannot hold entries 0..batch, every role falls back.
  const int32_t* const cq = (batch <= 63)
      ? reinterpret_cast<const int32_t*>(smem + T9_CUMQ_OFF) : cum_q;
  const int32_t* const ck = (batch <= 63)
      ? reinterpret_cast<const int32_t*>(smem + T9_CUMKV_OFF) : cum_kv;
#else
  const int32_t* const cq = cum_q;
  const int32_t* const ck = cum_kv;
#endif
#ifdef TC9_GSORT
  // Once per CTA, before the roles split: every warp role reads this table, and
  // both CTAs of a pair build the same one from the same immutable inputs, so
  // the item sequence stays a pure function of (blockIdx.x, gridDim.x) and the
  // cluster-scope barriers stay in step with no handshake added.
  if (t9_sort) {
    build_perm(cq, ck, batch, causal, pairs_ub, smem);
    __syncthreads();          // publish perm[] to every warp role
  }
#endif
#if defined(TC7_STAGE) && TC7_STAGE <= 1
  if (dbg && tid == 0) atomicMax(dbg + 40, 1ull);   // reached: ctrl init done
  return;
#endif

  // ---- pair TMEM: the leader allocates for BOTH CTAs (measured: the peer reads
  // back its own 128 lanes of the pair's D) ----
  uint32_t tmem = 0;
  // BOTH CTAs issue the collective alloc, and both issue the collective dealloc at
  // the end. This is not a style choice: with the leader issuing them alone, the
  // first launch completes and the second one never does (run_repeat.sh
  // dv_c08b4dd5599f: launch 0 tiles=12 ctas=24 on x1 and tiles=240 on x1024, then
  // launch 1 hangs on both shapes, on a different shape too, and with bounded and
  // unbounded waits alike). Leader-only leaves the peer's half of the pair TMEM
  // unreleased, so the next launch's tcgen05.alloc blocks forever waiting for
  // columns that are still charged to the SM. Issuing from both CTAs relaunches
  // indefinitely - 5/5 on x1 and 3/3 on x1024 - and produces bit-identical output
  // (out_absmax 3.812 and 4.75, the same values launch 0 gave before). micro.cu's
  // cl2 had already measured the two forms as equal in rate (64.02 cyc/MMA either
  // way), so this costs nothing. A cluster rendezvous AFTER the dealloc does not
  // help on its own: that variant still hung on launch 1.
  if (warp == 8) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 ::"r"(sm32_(tholder)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
  }
  __syncthreads();
  fence_after_sync();
  tmem = *(volatile uint32_t*)tholder;
  // cluster rendezvous: both CTAs must have TMEM before any pair MMA
  cluster_sync();
#if defined(TC7_STAGE) && TC7_STAGE <= 2
  if (dbg && tid == 0) atomicMax(dbg + 40, 2ull);   // reached: pair TMEM + rendezvous
  // release the pair TMEM before leaving, or the next synchronize reports
  // cudaErrorTensorMemoryLeak and masks the fact that this stage passed
  cluster_sync();
  if (warp == 8)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 ::"r"(tmem), "r"(512) : "memory");
  return;
#endif

  // tc8 filled Q here once per launch and pre-loaded the first KV_STAGES tiles
  // before the roles split. tc9 cannot: with a persistent cluster both are per
  // ITEM, and the KV ring is one continuous global stream, so both moved into the
  // loader's work loop below. The issuer's `i == G0` QF wait and the loader's
  // `G < KV_STAGES` no-PV-wait case cover the first item exactly as before.

#if defined(TC7_STAGE) && TC7_STAGE <= 3
  if (dbg && tid == 0) atomicMax(dbg + 40, 3ull);   // reached: fills issued
  cluster_sync();
  if (warp == 8)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 ::"r"(tmem), "r"(512) : "memory");
  return;
#endif
  const uint32_t qsbo = 1024u;         // 8-row core matrix in a 128-row plane
  const uint32_t qa16 = 1024u;         // atom stride, 16384 B in 16 B units
  const uint32_t ksbo = 1024u;         // 8-row core matrix in a 64-row plane
  const uint32_t vsbo = 1024u;         // 8-token core matrix (8 x 128 B)
  const int vstep16 = 2048 / 16;       // 16 kv tokens of V = 2048 B

  // Every role below walks the SAME item sequence and keeps its own copy of the
  // global tile counter Gbase, advancing it by it.T on exactly the items that
  // decode_item accepts. That invariant is what replaces tc2p's device counter,
  // its smem publication and its CONS/NEXT handshake: nothing is communicated
  // between roles or between CTAs at item granularity except the barriers that
  // already existed (plus ODR), and a disagreement would show up as a bounded-wait
  // timeout counter rather than as a wrong answer.
  if (warp == 9) {
    // ---------------- TMA LOADER (persistent) ----------------
    // One continuous global stream of KV stages across every item this cluster
    // owns, plus the per-item Q refill. The ring never restarts at an item
    // boundary - stage and parity come from the global tile index - so the loader
    // is fetching item f+1's first tiles while the issuer is still in item f's.
    if (elect_one()) {
      for (long long k9 = 0, base9 = 0, item = t9sl.a;
           item < total_items;
           (base9 += ncl, ++k9,
            item = base9 + ((k9 & 1) ? t9sl.b : t9sl.a))) {
        Item it;
        if (!decode_item(cq, ck, batch, heads, kv_heads, causal,
                         window_left, has_sink, sinks, pairs_ub, item, rank, it,
                         t9_perm))
          continue;
        const long long G0 = Gbase;
        // Q refill. The single Q buffer is being read by the previous item's QK
        // stream until that stream's last commit lands, so wait SF of tile G0-1;
        // the transfer then overlaps the previous item's softmax tail and
        // epilogue, which is tc2p's trick and the reason persisting pays.
        if (fitem > 0) {
          const long long gp = G0 - 1;
          mbwait(bar(ctrl, SF0 + gp % S_STAGES), (int)((gp / S_STAGES) & 1), dbg,
                 DBG_TO_SF, whint);
          fence_after_sync();
        }
        mbar_expect_tx(bar(ctrl, QF), Q_BYTES);
        tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF), 0,
                    it.q_base + it.m0, 0, it.h);
#ifdef TC9_QPF
        // E42 (probe197): L2 prefetch of the NEXT item's Q. The refill above
        // is serialised behind SF(G0-1) (the single sq buffer is being read
        // by the previous item's QK stream until its last commit), so the
        // cold DRAM Q fetch overlaps only the softmax tail + epilogue - and
        // on short-tile items that window is smaller than the fetch: slot-32
        // refill bubble 4155 cyc/item on f9 (34% of the item chain), 2800
        // f8, 2303 f3 (probe194 dv_7d44fb9afc7e). A full second Q buffer
        // does not fit (KV ring 3x40K + Q 98K is already ~221K of 227K
        // smem), but decode_item is a pure function of the item ordinal, so
        // the loader can decode item f+1 here and fire a fire-and-forget
        // TMA L2 prefetch of its Q box: no smem, no barrier, no ordering
        // hazard (it writes nothing), and it has the whole item period to
        // land, turning the boundary refill into an L2 hit. Placed AFTER
        // this item's own Q issue so it can never delay the critical path;
        // a decode miss (tail/unowned item) simply skips the prefetch.
        {
          const long long item_n =
              (base9 + ncl) + (((k9 + 1) & 1) ? t9sl.b : t9sl.a);
          if (item_n < total_items) {
            Item it2;
            if (decode_item(cq, ck, batch, heads, kv_heads, causal,
                            window_left, has_sink, sinks, pairs_ub, item_n,
                            rank, it2, t9_perm))
              tma_prefetch_4d(&qrmap, 0, it2.q_base + it2.m0, 0, it2.h);
          }
        }
#endif
#ifdef TC9_KVSPLIT
        for (long long j = G0; j < G0 + it.T; ++j) {
          const int s = (int)(j % KV_STAGES);
          const int jt = (int)(j - G0);
          // KVSPLIT: K(s) is consumed when QK(j-3) COMMITS (SF0+s, multicast
          // to both CTAs - tcgen05 completion implies its SMEM reads are
          // done, the same guarantee the V/PV0 release already rides), ~one
          // tile earlier than PV(j-3) retirement. V(s) keeps the PV-retire
          // trigger. Same 3 stages, same SMEM: only the triggers move.
          if (j >= KV_STAGES) {
            mbwait(bar(ctrl, SF0 + s), (int)((j / KV_STAGES - 1) & 1), dbg,
                   DBG_TO_SF, whint);
          }
          mbar_expect_tx(bar(ctrl, BFK0 + s), K_BYTES);
          tma_load_4d(&krmap, sm32_(sk) + s * K_BYTES, bar(ctrl, BFK0 + s), 0,
                      it.tok0 + jt * BN + (int)rank * BNH, 0, it.h_kv);
          if (j >= KV_STAGES) {
            mbwait(bar(ctrl, PV0 + s), (int)((j / KV_STAGES - 1) & 1), dbg,
                   DBG_TO_PV, whint);
          }
          mbar_expect_tx(bar(ctrl, BF0 + s), V_BYTES);
          tma_load_4d(&vrmap, sm32_(sv) + s * V_BYTES, bar(ctrl, BF0 + s), 0,
                      it.tok0 + jt * BN, (int)rank, it.h_kv);
        }
#else
        for (long long j = G0; j < G0 + it.T; ++j) {
          const int s = (int)(j % KV_STAGES);
          // PV(j-3) retired frees stage s (K read by QK, V by PV). The first
          // KV_STAGES tiles of the LAUNCH have nothing to wait for; note the test
          // is on the global index, so a later item's first tile still waits.
          if (j >= KV_STAGES) {
            mbwait(bar(ctrl, PV0 + s), (int)((j / KV_STAGES - 1) & 1), dbg,
                   DBG_TO_PV, whint);
          }
          const int jt = (int)(j - G0);
          mbar_expect_tx(bar(ctrl, BF0 + s), K_BYTES + V_BYTES);
          tma_load_4d(&krmap, sm32_(sk) + s * K_BYTES, bar(ctrl, BF0 + s), 0,
                      it.tok0 + jt * BN + (int)rank * BNH, 0, it.h_kv);
          tma_load_4d(&vrmap, sm32_(sv) + s * V_BYTES, bar(ctrl, BF0 + s), 0,
                      it.tok0 + jt * BN, (int)rank, it.h_kv);
        }
#endif
        Gbase += it.T; ++fitem;
      }
    }
  } else if (warp == 8) {
    // ---------------- warp 8: peer relays BR, leader issues the MMAs ----------
    if (elect_one()) {
      if (rank != 0) {
        // The peer's only job here: tell the leader that this CTA's stage has
        // landed. Off the loader's critical path, which matters because a remote
        // arrive costs 329-824 cyc to issue (measured).
        const uint32_t lbr = mapa_(bar(ctrl, BR0), 0);
#ifdef TC9_KVSPLIT
        const uint32_t lbrk = mapa_(bar(ctrl, BRK0), 0);
#endif
        for (long long k9 = 0, base9 = 0, item = t9sl.a;
             item < total_items;
             (base9 += ncl, ++k9,
              item = base9 + ((k9 & 1) ? t9sl.b : t9sl.a))) {
          Item it;
          if (!decode_item(cq, ck, batch, heads, kv_heads, causal,
                           window_left, has_sink, sinks, pairs_ub, item, rank, it,
                         t9_perm))
            continue;
          const long long G0 = Gbase;
          for (long long i = G0; i < G0 + it.T; ++i) {
#ifdef TC9_KVSPLIT
            // KVSPLIT: two relay legs. The K leg (BRK) certifies this
            // CTA's K(i) - and, at an item's head, its Q - for the
            // leader's QK(i); the V leg keeps the old BR0 barrier and now
            // gates the leader's PV(i) instead of its QK(i).
            mbwait(bar(ctrl, BFK0 + i % KV_STAGES),
                   (int)((i / KV_STAGES) & 1), dbg, DBG_TO_BF, whint);
            if (i == G0) {
              mbwait(bar(ctrl, QF), (int)(fitem & 1), dbg, DBG_TO_QF, whint);
              fence_after_sync();
            }
            mbar_arrive_remote(lbrk + (int)(i % KV_STAGES) * 8);
            mbwait(bar(ctrl, BF0 + i % KV_STAGES), (int)((i / KV_STAGES) & 1),
                   dbg, DBG_TO_BFV, whint);
            mbar_arrive_remote(lbr + (int)(i % KV_STAGES) * 8);
#else
            mbwait(bar(ctrl, BF0 + i % KV_STAGES), (int)((i / KV_STAGES) & 1),
                   dbg, DBG_TO_BF, whint);
            // this item's first BR arrive also certifies this CTA's Q, which the
            // leader's pair QK reads out of THIS smem
            if (i == G0) {
              mbwait(bar(ctrl, QF), (int)(fitem & 1), dbg, DBG_TO_QF, whint);
              fence_after_sync();
            }
            mbar_arrive_remote(lbr + (int)(i % KV_STAGES) * 8);
#endif
          }
          Gbase += it.T; ++fitem;
        }
      } else {
        const uint32_t qb0 = sm32_(sq);
        T9DECL_IS
        for (long long k9 = 0, base9 = 0, item = t9sl.a;
             item < total_items;
             (base9 += ncl, ++k9,
              item = base9 + ((k9 & 1) ? t9sl.b : t9sl.a))) {
          Item it;
          if (!decode_item(cq, ck, batch, heads, kv_heads, causal,
                           window_left, has_sink, sinks, pairs_ub, item, rank, it,
                         t9_perm))
            continue;
          const long long G0 = Gbase;
          for (long long i = G0; i < G0 + it.T; ++i) {
            const int s = (int)(i % KV_STAGES);
            const int par = (int)((i / KV_STAGES) & 1);
            T9MARK(timed_i, i0);
#ifdef TC9_KVSPLIT
            mbwait(bar(ctrl, BFK0 + s), par, dbg, DBG_TO_BF, whint);
            if (i == G0) {
              mbwait(bar(ctrl, QF), (int)(fitem & 1), dbg, DBG_TO_QF, whint);
              fence_after_sync();
            }
            mbar_arrive(bar(ctrl, BRK0 + s));
#ifndef TC7_NO_BR
            mbwait(bar(ctrl, BRK0 + s), par, dbg, DBG_TO_BR, whint);
#endif
#else
            mbwait(bar(ctrl, BF0 + s), par, dbg, DBG_TO_BF, whint);
            if (i == G0) {
              mbwait(bar(ctrl, QF), (int)(fitem & 1), dbg, DBG_TO_QF, whint);
              fence_after_sync();
            }
            mbar_arrive(bar(ctrl, BR0 + s));
#ifndef TC7_NO_BR
            mbwait(bar(ctrl, BR0 + s), par, dbg, DBG_TO_BR, whint);
#endif
#endif
            fence_after_sync();
            T9ACC(timed_i, is_bf, i0);   // KV bulk + this item's Q + the BR relay
            // S/P slot s is free once PV(i-3) retired (same barrier, same parity
            // as the loader's stage wait)
            if (i >= KV_STAGES) {
              mbwait(bar(ctrl, PV0 + s), (int)((i / KV_STAGES - 1) & 1), dbg,
                     DBG_TO_PV, whint);
              fence_after_sync();
            }
            T9ACC(timed_i, is_pvd, i0);  // S/P slot free: PV(i-3) retired
            // ---- QK(i): 12 k-steps of M256N128K16, A=Q from smem ----
            {
              const uint32_t ts = tmem + TCOL_S(s);
              const uint64_t da = make_desc(qb0, 0, qsbo);
              const uint64_t db = make_desc(sm32_(sk) + s * K_BYTES, 0, ksbo);
              #pragma unroll
              for (int kk = 0; kk < 12; ++kk)
                mma_ss2(ts, da + ((kk >> 2) * qa16 + (kk & 3) * 2),
                        db + ((kk >> 2) * (8192 / 16) + (kk & 3) * 2),
                        IDESC_QK, kk > 0);
              mma_commit_mc(bar(ctrl, SF0 + s));
            }
            T9ACC(timed_i, is_qk, i0);   // the pair QK issue, 12 M256N128K16
            // ---- PV(i-1): 8 k-steps, A=P from TMEM, B=V half from smem ----
            const long long ip = i - 1;
            if (ip >= G0) {
              const int sp = (int)(ip % S_STAGES);
              // PV(G0) is the first PV to touch THIS item's O, and the pair has
              // exactly one O accumulator (3 S/P slots x 128 + O 128 = all 512
              // TMEM columns, so there is no second O to ping-pong into). The
              // previous item's epilogue reads O out of TMEM, so its drain must
              // have completed in BOTH CTAs first - hence ODR, count 2, arrived by
              // this CTA's softmax and relayed from the peer by its warp 10.
              if (ip == G0 && fitem > 0) {
                mbwait(bar(ctrl, ODR), (int)((fitem - 1) & 1), dbg, DBG_TO_ODR,
                       whint);
                fence_after_sync();
              }
#ifndef TC7_NO_PF
              mbwait(bar(ctrl, PF0 + sp), (int)((ip / S_STAGES) & 1), dbg,
                     DBG_TO_PF, whint);
#endif
#ifdef TC9_KVSPLIT
              // KVSPLIT: own V(ip) (BF0, armed by the loader's V leg) and
              // the peer's V(ip) (BR0, cnt 2: this self-arrive + the relay's
              // V leg). Both are ~0-wait in steady state - V(ip) landed a
              // tile ago - but they are the issuer-side anchors that make
              // the pair PV's B halves safe (bit0 design rule).
              mbwait(bar(ctrl, BF0 + sp), (int)((ip / KV_STAGES) & 1), dbg,
                     DBG_TO_BFV, whint);
              mbar_arrive(bar(ctrl, BR0 + sp));
#ifndef TC7_NO_BR
              mbwait(bar(ctrl, BR0 + sp), (int)((ip / KV_STAGES) & 1), dbg,
                     DBG_TO_BR, whint);
#endif
#endif
              fence_after_sync();
              T9ACC(timed_i, is_pfw, i0);  // ODR + PF: BOTH CTAs' softmax stored P
              const uint32_t tp = tmem + TCOL_S(sp);
              const uint64_t db = make_desc(sm32_(sv) + sp * V_BYTES, 0, vsbo);
              #pragma unroll
              for (int kk = 0; kk < 8; ++kk)
                mma_ts2(tmem + TCOL_O, tp + kk * 8, db + kk * vstep16, IDESC_PV,
                        (ip > G0) | (kk > 0));
              mma_commit_mc(bar(ctrl, PV0 + sp));
            }
            T9ACC(timed_i, is_pv, i0);   // the pair PV issue, 8 k-steps
            T9ACC(timed_i, is_per, i0);
            if (dbg) atomicAdd(dbg + DBG_TILES, 1ull);
          }
          // ---- tail: PV(G0+T-1), this item's last ----
          {
            const long long il = G0 + it.T - 1;
            const int sl = (int)(il % S_STAGES);
            if (it.T == 1 && fitem > 0) {
              mbwait(bar(ctrl, ODR), (int)((fitem - 1) & 1), dbg, DBG_TO_ODR,
                     whint);
              fence_after_sync();
            }
#ifndef TC7_NO_PF
            mbwait(bar(ctrl, PF0 + sl), (int)((il / S_STAGES) & 1), dbg,
                   DBG_TO_PF, whint);
#endif
#ifdef TC9_KVSPLIT
            mbwait(bar(ctrl, BF0 + sl), (int)((il / KV_STAGES) & 1), dbg,
                   DBG_TO_BFV, whint);
            mbar_arrive(bar(ctrl, BR0 + sl));
#ifndef TC7_NO_BR
            mbwait(bar(ctrl, BR0 + sl), (int)((il / KV_STAGES) & 1), dbg,
                   DBG_TO_BR, whint);
#endif
#endif
            fence_after_sync();
            const uint32_t tp = tmem + TCOL_S(sl);
            const uint64_t db = make_desc(sm32_(sv) + sl * V_BYTES, 0, vsbo);
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk)
              mma_ts2(tmem + TCOL_O, tp + kk * 8, db + kk * vstep16, IDESC_PV,
                      (it.T > 1) | (kk > 0));
            mma_commit_mc(bar(ctrl, PV0 + sl));
          }
          Gbase += it.T; ++fitem;
        }
#ifdef TC9_TIMING
        if (timed_i) {                   // slot 24 (is_tiles) is DBG_TILES above
          atomicAdd(dbg + 25, (unsigned long long)is_bf);
          atomicAdd(dbg + 26, (unsigned long long)is_pvd);
          atomicAdd(dbg + 27, (unsigned long long)is_qk);
          atomicAdd(dbg + 28, (unsigned long long)is_pfw);
          atomicAdd(dbg + 29, (unsigned long long)is_per);
          atomicAdd(dbg + 30, (unsigned long long)is_pv);
        }
#endif
      }
    }
#ifdef TC7_PF_RELAY
  } else if (warp == 10) {
    // -------- PEER RELAY (warp 10, peer CTA only): PF per tile, OL per item ----
    // Nothing else in the peer CTA issues MMAs, so this warp would otherwise be
    // idle. It waits the peer's own local barriers and forwards them to the
    // leader, which is the only place a remote arrive is affordable.
    if (rank != 0 && elect_one()) {
      const uint32_t lpf = mapa_(bar(ctrl, PF0), 0);
      const uint32_t lodr = mapa_(bar(ctrl, ODR), 0);
      for (long long k9 = 0, base9 = 0, item = t9sl.a;
           item < total_items;
           (base9 += ncl, ++k9,
            item = base9 + ((k9 & 1) ? t9sl.b : t9sl.a))) {
        Item it;
        if (!decode_item(cq, ck, batch, heads, kv_heads, causal,
                         window_left, has_sink, sinks, pairs_ub, item, rank, it,
                         t9_perm))
          continue;
        const long long G0 = Gbase;
        for (long long i = G0; i < G0 + it.T; ++i) {
          const int s = (int)(i % S_STAGES);
          mbwait(bar(ctrl, PL0 + s), (int)((i / S_STAGES) & 1), dbg, DBG_TO_PF,
                 whint);
          mbar_arrive_remote(lpf + s * 8);
        }
        // the epilogue's O reads are done in this CTA: release the leader's next
        // PV to overwrite the pair's single O accumulator
        mbwait(bar(ctrl, OL), (int)(fitem & 1), dbg, DBG_TO_ODR, whint);
        mbar_arrive_remote(lodr);
        Gbase += it.T; ++fitem;
      }
    }
#endif
  } else if (warp < SOFT_WARPS) {
    // ---------------- SOFTMAX (column-split over 8 warps) ----------------
    const int w3 = warp & 3;                 // row group: rows 32*w3 .. +31
    const int half = warp >> 2;              // column half: kv 64*half .. +63
    const int row = w3 * 32 + lane;          // row within this CTA's 128
    const uint32_t lb = static_cast<uint32_t>(w3 * 32) << 16;
    float sv2[2][32];
    // this thread's slice of the cross-warp max exchange: one float per (warp,
    // lane), i.e. per row of the CTA's 128
    // Cross-warp row-max exchange buffer: one float per (warp, lane), i.e. per
    // row of this CTA's 128. With TC7_XMAX1 tile i uses half (i & 1) of a
    // double-sized buffer, which removes the second barrier - its only job was to
    // stop the next tile overwriting a value its partner had not read yet. Tile
    // i+2 reuses the half, and its write sits behind tile i+1's barrier, which no
    // warp reaches until after its tile-i read.
    const int mrot = 256;            // floats per rotating half
#ifdef TC7_XMAX1
#define TC7_MXOFF(i) (((i) & 1) * 256)
#else
#define TC7_MXOFF(i) 0
#endif

    T9DECL_SM
    T9DECL_SM2
    T9DECL_SM4
    T9DECL_SM5
#ifdef TC9_TIMING
    if (timed_s) {                       // prologue: init, pair TMEM, rendezvous
      const long long t = clock64();
      atomicAdd(dbg + 51, (unsigned long long)(t - (long long)t_ent));
      c0 = t;
    }
#endif
    for (long long k9 = 0, base9 = 0, item = t9sl.a;
         item < total_items;
         (base9 += ncl, ++k9,
          item = base9 + ((k9 & 1) ? t9sl.b : t9sl.a))) {
    Item it;
    if (!decode_item(cq, ck, batch, heads, kv_heads, causal,
                     window_left, has_sink, sinks, pairs_ub, item, rank, it,
                         t9_perm))
      continue;
    const long long G0 = Gbase;
    const int m0 = it.m0;
    const int kv_off = it.kv_off;
    const bool windowed = it.windowed;
    const int Tfull = it.Tfull;
    const int qlen = it.qlen;
    const int kvlen = it.kvlen;              // the masked-tile column bounds need it
    const int grow = m0 + row;               // global q row
    const int vis = kv_off + grow;
    const bool active = (grow < qlen);
    const float sink2 = it.sink2;
    __nv_bfloat16* outp = out + it.o_off * DV;
    float* lsep = has_lse ? lse + it.o_off : nullptr;
    float mr = NEG_BIG, d = 0.0f;            // per-item online-softmax state
    // The tile index i is GLOBAL: `i % S_STAGES` and `(i / S_STAGES) & 1` below
    // are therefore already the right slot and parity across item boundaries, and
    // the ring never restarts. Only the three places where the ITEM matters use
    // G0 explicitly: `i == G0` for this item's first tile, `i - G0` for the kv
    // column offset and the masked/unmasked split, and the O-reset predicate.
    for (long long i = G0; i < G0 + it.T; ++i) {
      const int s = (int)(i % S_STAGES);
      const uint32_t ts = tmem + TCOL_S(s) + half * 64 + lb;
#if defined(TC9_TIMING) && defined(TC9_TIMING2)
      const long long tprev = c0;        // previous chain end (epilogue restart / sm_per)
#endif
      T9MARK(timed_s, c0);
#if defined(TC9_TIMING) && defined(TC9_TIMING2)
      const long long tbew0 = c0;
#endif
      mbwait(bar(ctrl, SF0 + s), (int)((i / S_STAGES) & 1), dbg, DBG_TO_SF,
             whint);
      fence_after_sync();
      T9ACC(timed_s, sm_bew, c0);        // waiting for this tile's QK result
#if defined(TC9_TIMING) && defined(TC9_TIMING2)
      if (timed_s) {
        if (i == G0) {                   // item boundary: decode_item + loop admin
          sm_boundary += tbew0 - tprev;
          sm_bew_first += c0 - tbew0;    // refill bubble (subset of sm_bew)
        } else {
          sm_gap += tbew0 - tprev;       // loop admin only; expect ~0
        }
      }
#endif
#ifdef TC9_BISECT_NOEXP
      // BISECTION ONLY - WRONG RESULTS, NEVER SHIPPED (same class as the
      // tc2p opts-bit2 "never scale O" bisection). Consumes the SF phase and
      // arrives PF with ZERO softmax work: no S load, no max/exchange, no
      // exp/P store, no correction. Protocol stays complete (SF consumed,
      // PF/relay arrived per tile, epilogue ODR/OL arrivals untouched), so
      // the pipeline cannot hang; d accumulates a constant and O is garbage.
      // Purpose: expose the issuer+feed+MMA chain's cadence floor - the
      // difference to the production arm is the softmax side's total
      // contribution (work + wake latency), which E4a's reorder-only flags
      // could not attribute. Pairs with the pre-existing TC7_NO_PF issuer
      // bisection. Assumes the production TC7_PF_RELAY arrive pattern.
      d += 64.0f;
      fence_before_sync();
      bar_softmax();
      if (warp == 0 && lane == 0) {
        mbar_arrive(bar(ctrl, (rank == 0 ? PF0 : PL0) + s));
      }
      continue;
#endif
      float* mx = maxbuf + TC7_MXOFF(i) + warp * 32 + lane;
      float* mxp = maxbuf + TC7_MXOFF(i) + (warp ^ 4) * 32 + lane;
      // n0 and `masked` are ITEM-relative: the kv column offset restarts at each
      // item's own request, and Tfull counts this item's fully-visible tiles.
      const int n0 = (int)(i - G0) * BN + half * 64;   // this warp's first kv col
      const bool masked = (i - G0) >= Tfull;

      // hardware row max over this warp's 64 columns, then the cross-warp
      // exchange that makes it a row max over all 128
      float tm = tmem_ld64_redmax(&sv2[0][0], ts);
      tmem_wait_ld();
      T9ACC(timed_s, sm_sld, c0);        // TMEM load of S with the fused row max
#ifndef TC8_HW_MASKMAX
      if (masked) {
        // The fused reduction is an UPPER bound, not the row max: on a masked
        // tile it also sees the columns causality hides, and any of those can
        // outscore every visible one. With an f32 exp that is harmless - softmax
        // is shift-invariant and ex2.approx.f32 stays nonzero down to 2^-149, so
        // the ratio O/d survives - which is why the promoted tc7 could use it.
        // ex2.approx.f16x2 underflows to EXACT ZERO below 2^-24, so an inflated
        // max zeroes every probability of the tile, including the tile that
        // carries the row's own mass: dtot becomes 0, inv becomes 0, out becomes
        // 0 and lse becomes log2f(0) = -inf. Measured (job dv_b2a40f651e23):
        // 10 of 19 boundary shapes and 8 of 10 fixed cases wrong, xgqa8 (peaked)
        // worst at 20507 elements and an lse error of 43. So masked tiles take
        // the true max over the visible columns, exactly as kernel_tc2p.cu does;
        // the load itself is unchanged (the x64 reduction load measures the same
        // as a plain x64 load) and unmasked tiles keep the free fused max, where
        // every column is visible so the bound is tight.
        const int jhi_m = min(causal ? vis + 1 : INT32_MAX, kvlen);
        const int jlo_m = windowed ? max(0, vis - window_left) : 0;
        float pm = NEG_BIG;
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
#ifdef TC9_MSKIP
          // A fully hidden 32-column block contributes the fmaxf IDENTITY
          // (NEG_BIG) on all 32 lanes - skipping its scan is bit-exact.
          const int jb0 = n0 + cc * 32;
          if (jb0 < jhi_m && !(windowed && jb0 + 31 < jlo_m))
#endif
          #pragma unroll
          for (int x = 0; x < 32; ++x) {
            const int j = n0 + cc * 32 + x;
            pm = fmaxf(pm, (j >= jlo_m && j < jhi_m) ? sv2[cc][x] : NEG_BIG);
          }
        }
        tm = pm;
      }
#endif
#ifndef TC7_NO_XMAX
      *mx = tm;
      bar_softmax();
      tm = fmaxf(tm, *mxp);
#ifndef TC7_XMAX1
      bar_softmax();
#endif
#endif

      // lagged running max (threshold 2^8 in the scaled log2 domain): alpha is
      // exactly 1.0f on almost every tile, so the O correction is skipped
      float nr = fmaxf(mr, tm);
      if ((nr - mr) * sc2 <= LAG_MAX_LOG2) nr = mr;
      const float nm2 = nr * sc2;
      const float alpha = (nr == mr) ? 1.0f : ex2_approx(fmaf(mr, sc2, -nm2));
      mr = nr;

      T9ACC(timed_s, sm_max, c0);        // masked max + the cross-warp exchange
#ifdef TC9_TIMING4
      const long long t4s = c0;
      long long t4a = c0, t4b = c0;
#endif
      const bool need_corr = (i > G0) && __any_sync(0xffffffffu, alpha != 1.0f);

      // exp / pack / row sum over this warp's 64 columns. This MUST come before
      // the O correction: the correction reuses sv2 as its O scratch, and sv2
      // still holds this tile's scores.
      float pc = 0.0f;
#ifndef TC9_PR16
      uint32_t pr[32];
#endif
#ifdef TC9_EARLY_PST
      // P image address, hoisted above the exp loops so each half can be stored as
      // soon as it is packed. kernel_tc2p.cu has always done it that way - its
      // tmem_st16bf sits INSIDE the cc loop - and its s_exp measures 581-712 cycles
      // per tile where this kernel's measures 824-883 for the same 16384 score
      // elements per CTA and a textually identical exp (dv_8bac3774fc32 against
      // dv_e8daa299ce32). Deferring both stores to the end keeps all 32 packed
      // registers live across the second half and serialises the two TMEM stores
      // behind the whole exp, so the wait_st at the end pays for both.
      const uint32_t tp = tmem + TCOL_S(s) + half * 32 + lb;
#ifdef TC9_PR16
      // E44 (probe199): tc2p-exact per-cc packed window. probe198
      // (dv_5f7fe077a161) localised the whole tc9-vs-tc2p exp residual
      // (+105..+235 cyc/tile, R1 4/4) to the LOOP BODY at identical
      // 2048-element warp shares; the last structural difference is that
      // tc9 keeps pr[32] live across BOTH cc halves (the store takes
      // pr+cc*16) while tc2p scopes pr[16] inside each cc iteration and
      // its registers die at the store. Same values, same pc order, same
      // two tmem_st16bf addresses/data/order -> bit-identical by
      // construction; only the register lifetime changes.
#define T9_PRDECL uint32_t pr16[16];
#define T9_PRW(cc, x2, v) (pr16[x2] = (v))
#define T9_STORE_HALF(cc) tmem_st16bf(tp + (cc) * 16, pr16)
#else
#define T9_PRDECL
#define T9_PRW(cc, x2, v) (pr[(cc) * 16 + (x2)] = (v))
#define T9_STORE_HALF(cc) tmem_st16bf(tp + (cc) * 16, pr + (cc) * 16)
#endif
#else
#define T9_PRDECL
#define T9_PRW(cc, x2, v) (pr[(cc) * 16 + (x2)] = (v))
#define T9_STORE_HALF(cc) do {} while (0)
#endif
#ifdef TC9_PR16
#ifndef TC9_EARLY_PST
#error "TC9_PR16 requires TC9_EARLY_PST (tp must be hoisted; tail store uses pr[32])"
#endif
#endif
      if (!masked) {
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          T9_PRDECL
          #pragma unroll
          for (int x2 = 0; x2 < 16; ++x2) {
#ifdef TC8_EX2F16
            const uint32_t pf = ex2_f16x2(cvt_f16x2(
                fmaf(sv2[cc][x2 * 2], sc2, -nm2),
                fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2)));
            const float p0 = f16lo_f32(pf), p1 = f16hi_f32(pf);
            pc += p0 + p1;
            T9_PRW(cc, x2, pack_bf16(p0, p1));
#else
            float p0 = ex2_approx(fmaf(sv2[cc][x2 * 2], sc2, -nm2));
            float p1 = ex2_approx(fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2));
            pc += p0 + p1;
            T9_PRW(cc, x2, pack_bf16(p0, p1));
#endif
          }
          T9_STORE_HALF(cc);
        }
      } else {
        const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
        const int jlo = windowed ? max(0, vis - window_left) : 0;
        if (!windowed) {
          // prefix mask: unconditional ex2, then zero the columns at or past
          // the first invalid one (tc2's promoted bit14 path)
          const int hi = jhi - n0;
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            T9_PRDECL
#ifdef TC9_MSKIP
            // Fully hidden block: every lane would take the -200 sentinel,
            // whose ex2.approx.f16x2 flushes to EXACT +0 (dv_b2a40f651e23);
            // pc += 0.0f + 0.0f is a bit identity (pc starts at +0.0f and
            // only grows) and pack_bf16(0,0) == 0, so zeroing pr and
            // skipping the block is bit-exact. On tiny-sequence items
            // (f7-class ql=1) and diagonal tiles this removes most of the
            // exp/pack work - the cluster engine's pacing stage (E17:
            // sm_exp 2.5-2.7k cyc/tile).
            if (cc * 32 >= hi) {
              #pragma unroll
              for (int x2 = 0; x2 < 16; ++x2) T9_PRW(cc, x2, 0u);
              T9_STORE_HALF(cc);
              continue;
            }
#endif
            #pragma unroll
            for (int x2 = 0; x2 < 16; ++x2) {
              const int x = cc * 32 + x2 * 2;
#ifdef TC8_EX2F16
              const float a0 = (x < hi) ? fmaf(sv2[cc][x2 * 2], sc2, -nm2)
                                        : -200.0f;
              const float a1 = (x + 1 < hi)
                                   ? fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2)
                                   : -200.0f;
              const uint32_t pf = ex2_f16x2(cvt_f16x2(a0, a1));
              const float p0 = f16lo_f32(pf), p1 = f16hi_f32(pf);
#else
              const float e0 = ex2_approx(fmaf(sv2[cc][x2 * 2], sc2, -nm2));
              const float e1 = ex2_approx(fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2));
              const float p0 = (x < hi) ? e0 : 0.0f;
              const float p1 = (x + 1 < hi) ? e1 : 0.0f;
#endif
              pc += p0 + p1;
              T9_PRW(cc, x2, pack_bf16(p0, p1));
            }
            T9_STORE_HALF(cc);
          }
        } else {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            T9_PRDECL
#ifdef TC9_MSKIP
            const int jb0 = n0 + cc * 32;
            if (jb0 >= jhi || jb0 + 31 < jlo) {   // fully hidden block
              #pragma unroll
              for (int x2 = 0; x2 < 16; ++x2) T9_PRW(cc, x2, 0u);
              T9_STORE_HALF(cc);
              continue;
            }
#endif
            #pragma unroll
            for (int x2 = 0; x2 < 16; ++x2) {
              float p0, p1;
#ifdef TC8_EX2F16
              {
                const int j = n0 + cc * 32 + x2 * 2;
                const float a0 = (j >= jlo && j < jhi)
                                     ? fmaf(sv2[cc][x2 * 2], sc2, -nm2)
                                     : -200.0f;
                const int j1 = j + 1;
                const float a1 = (j1 >= jlo && j1 < jhi)
                                     ? fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2)
                                     : -200.0f;
                const uint32_t pf = ex2_f16x2(cvt_f16x2(a0, a1));
                p0 = f16lo_f32(pf);
                p1 = f16hi_f32(pf);
              }
#else
              {
                int j = n0 + cc * 32 + x2 * 2;
                p0 = (j >= jlo && j < jhi)
                         ? ex2_approx(fmaf(sv2[cc][x2 * 2], sc2, -nm2)) : 0.0f;
              }
              {
                int j = n0 + cc * 32 + x2 * 2 + 1;
                p1 = (j >= jlo && j < jhi)
                         ? ex2_approx(fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2)) : 0.0f;
              }
#endif
              pc += p0 + p1;
              T9_PRW(cc, x2, pack_bf16(p0, p1));
            }
            T9_STORE_HALF(cc);
          }
        }
      }
      // P image: 64 bf16 pairs into columns half*32 .. half*32+31 of the slot.
      // Under TC9_EARLY_PST both halves are already stored and tp is declared
      // above the exp loops.
#ifndef TC9_EARLY_PST
      const uint32_t tp = tmem + TCOL_S(s) + half * 32 + lb;
      tmem_st16bf(tp, pr);
      tmem_st16bf(tp + 16, pr + 16);
#endif
#ifdef TC9_TIMING4
      if (timed_s) t4a = clock64();
#endif
      tmem_wait_st();
#ifdef TC9_TIMING4
      if (timed_s) t4b = clock64();
#endif
      d = d * alpha + pc;
      T9ACC(timed_s, sm_exp, c0);        // exp / pack / row sum / P store
#ifdef TC9_TIMING4
      if (timed_s) {
        smx_loop += t4a - t4s;
        smx_wst += t4b - t4a;
        smx_tail += c0 - t4b;
      }
#endif
#undef T9_STORE_HALF
#undef T9_PRDECL
#undef T9_PRW

      // O correction (skipped on tile 0 and, thanks to the lagged max, on almost
      // every later tile because alpha is then exactly 1.0f). PV(i-1) writes this
      // CTA's half of O, so it must have retired first. This warp rescales only
      // its own 64 output dims, so the two halves never touch the same columns.
      if (need_corr) {
        mbwait(bar(ctrl, PV0 + (int)((i - 1) % S_STAGES)),
               (int)(((i - 1) / S_STAGES) & 1), dbg, DBG_TO_PV, whint);
        fence_after_sync();
        const uint32_t to = tmem + TCOL_O + half * 64 + lb;
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) tmem_ld32(sv2[cc], to + cc * 32);
        tmem_wait_ld();
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          #pragma unroll
          for (int x = 0; x < 32; ++x) sv2[cc][x] *= alpha;
          tmem_st32(to + cc * 32, sv2[cc]);
        }
        tmem_wait_st();
      }
      T9ACC(timed_s, sm_corr, c0);       // the O correction, when alpha != 1
      // PF merges tc2's PF and OC barriers: it is arrived only after P is in
      // TMEM AND the correction is done, which is exactly the pair of facts the
      // leader's issuer needs before PV(i) may accumulate into either CTA's O.
#ifdef TC9_TIMING5
      long long t5a = 0, t5b = 0;
      T9MARK(timed_s, t5a);
#endif
      fence_before_sync();
#ifdef TC9_PFL
      // E46 (probe203): per-warp local PF arrive - tc2p's production
      // pattern (its pfB arrive is per-thread, count 128, no convergence
      // bar). probe202 (dv_5923287e45e9) priced fence+bar_softmax
      // convergence in this PF path at slot 41: 94.8 (f2) / 168.5 (f9) /
      // 81.7 (f4) cyc/tile, and the bar-induced desched additionally
      // loads gap34 (249-271/tile). Each warp fences its OWN tcgen05
      // stores and arrives from lane0; the count-9 PF (8 warps + the peer
      // relay) / count-8 PL completes exactly when all 16 softmax warps of
      // the pair have fenced - the same fact the leader's PV wait and the
      // peer relay consume, without an 8-warp bar.sync convergence on the
      // paced path. Parity math is count-independent: the relay (PL-phase
      // mbwait) and the issuer (PF-phase mbwait) are byte-identical. No
      // numeric change by construction (protocol-only).
#ifdef TC9_TIMING5
      T9MARK(timed_s, t5b);
#endif
      if (lane == 0)
        mbar_arrive(bar(ctrl, (rank == 0 ? PF0 : PL0) + s));
#else
      bar_softmax();
#ifdef TC9_TIMING5
      T9MARK(timed_s, t5b);
#endif
      if (warp == 0 && lane == 0) {
#ifdef TC7_PF_RELAY
        // Both CTAs arrive LOCALLY. A remote arrive costs 329 cyc to issue bare
        // and 824 cyc issued from the state the softmax is in right after an MMA
        // or TMEM wait (measured, cls6/cls7), and it sits directly on the
        // leader's PV issue path. The peer's warp 10 picks its local barrier up
        // and does the remote arrive instead, so the softmax warps go straight
        // on to tile i+1 and the arrive latency overlaps the next tile.
        mbar_arrive(bar(ctrl, (rank == 0 ? PF0 : PL0) + s));
#else
        const uint32_t a = bar(ctrl, PF0 + s);
        if (rank == 0) mbar_arrive(a);
        else mbar_arrive_remote(mapa_(a, 0));
#endif
      }
#endif
      T9ACC(timed_s, sm_per, c0);
#ifdef TC9_TIMING5
      if (timed_s) {
        sm_barconv += t5b - t5a;
        sm_pfarr += c0 - t5b;
      }
#endif
#ifdef TC9_TIMING
      if (timed_s) sm_tiles += 1;
#endif
    }

    T9MARK(timed_s, e0);
    // ---- epilogue ----
    // (1) the last PV must have retired; (2) combine the two halves' row sums;
    // (3) scale/pack into the staging image and issue one bulk tensor store per
    // CTA, or scattered per-row stores at a boundary.
    const long long il = G0 + it.T - 1;
    mbwait(bar(ctrl, PV0 + (int)(il % S_STAGES)), (int)((il / S_STAGES) & 1), dbg,
           DBG_TO_PV, whint);
    fence_after_sync();
#if defined(TC9_TIMING) && defined(TC9_TIMING2)
    if (timed_s) epi_pv += clock64() - e0;   // last-PV retire wait (subset of 52)
#endif
    // partial row sums reuse the max buffer, one float per row of this CTA
    float* rs = maxbuf;              // fixed half: the loop's reads are all done
    (void)mrot;
    bar_softmax();
    rs[warp * 32 + lane] = d;
    bar_softmax();
    const float dtot = rs[w3 * 32 + lane] + rs[(w3 + 4) * 32 + lane];
    bar_softmax();

    float g = 0.0f, lse2 = 0.0f;
    const float inv = dtot > 0.0f ? 1.0f / dtot : 0.0f;
    lse2 = mr * sc2 + log2f(dtot);
    float extra = 1.0f;
    if (has_sink) {
      float t = fmaxf(lse2, sink2) + log2f(1.0f + exp2f(-fabsf(lse2 - sink2)));
      extra = exp2f(lse2 - t);
      lse2 = t;
    }
    g = bmm2_scale * inv * extra;

#ifdef TC9_DIRECT_EPI
    // Direct-store epilogue (tc2p64/tc10's promoted pattern): every item
    // takes the scattered path, so the staging image, the fence.proxy, the
    // bulk tensor store AND the DRAINLATE cross-item read drain all
    // disappear. Byte-identical coverage: bulk_ok == (m0+SUB <= qlen) means
    // every row of a bulk box is active, so the active-guarded per-lane
    // stores write exactly the staged path's bytes - and E10 proved the
    // substitution bit-equal in the other direction (tc10's direct epilogue
    // reproduced this staging epilogue's outputs on every tc9r-routed
    // shape). The scattered-path discipline stays: WARP-UNIFORM predicate
    // around the .sync.aligned TMEM loads (the dv_8ad2fc9f7d9c hang law),
    // per-lane active guard on the stores. The row-sum maxbuf reuse is
    // already ordered by the third bar_softmax above; the staging-reuse
    // rendezvous below is what this variant drops.
    if (__any_sync(0xffffffffu, active)) {
      const uint32_t to = tmem + TCOL_O + half * 64 + lb;
      __nv_bfloat16* orow = outp + static_cast<int64_t>(m0 + row) * o_stride;
      #pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        tmem_ld32(sv2[cc], to + cc * 32);
        tmem_wait_ld();
        if (active) {
          __align__(16) uint32_t pk[16];
          #pragma unroll
          for (int x2 = 0; x2 < 16; ++x2)
            pk[x2] = pack_bf16(sv2[cc][x2 * 2] * g, sv2[cc][x2 * 2 + 1] * g);
          uint4* dst = reinterpret_cast<uint4*>(orow + (half * 2 + cc) * 32);
          const uint4* src4 = reinterpret_cast<const uint4*>(pk);
          dst[0] = src4[0]; dst[1] = src4[1];
          dst[2] = src4[2]; dst[3] = src4[3];
        }
      }
    }
#else
    // The rendezvous must be the 256-thread named barrier, NOT __syncthreads() -
    // warps 8, 9 and 10 never enter this branch and would leave it waiting forever.
#ifdef TC9_DRAINLATE
    if (tid == 0) bulk_wait_read0();   // previous item's store: stg now reusable
#endif
    bar_softmax();
    const bool bulk_ok = it.bulk_ok;
    // warp-uniform form of the per-row validity test, for the aligned TMEM loads
    const bool any_active = __any_sync(0xffffffffu, active);
    // tc8 staged this image over the K stages on the grounds that they were dead
    // by now. Under persistence they are NOT: the loader is already refilling them
    // for the next item while this epilogue packs, so the image gets the dedicated
    // EPI_OFF buffer added to the layout (206080 B of the 232448 B opt-in).
    char* stg = smem + EPI_OFF;
    if (bulk_ok) {
      const uint32_t to = tmem + TCOL_O + half * 64 + lb;
      #pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        tmem_ld32(sv2[cc], to + cc * 32);
        tmem_wait_ld();
        __align__(16) uint32_t pk[16];
        #pragma unroll
        for (int x2 = 0; x2 < 16; ++x2)
          pk[x2] = pack_bf16(sv2[cc][x2 * 2] * g, sv2[cc][x2 * 2 + 1] * g);
        const uint4* src4 = reinterpret_cast<const uint4*>(pk);
        #pragma unroll
        for (int q4 = 0; q4 < 4; ++q4)
          *reinterpret_cast<uint4*>(stg + ob_addr(row, (half * 2 + cc) * 4 + q4)) =
              src4[q4];
      }
    } else if (any_active) {
      // tcgen05.ld is .sync.aligned: every lane of the warp must execute it, so
      // the load is guarded by a WARP-UNIFORM predicate and only the global
      // stores are per-row conditional. The first version of this kernel had
      // `else if (active)` around the load itself, which is per-lane - on x1
      // (qlen=1) warp 0 has exactly one active lane and the aligned TMEM load
      // then waits forever for the other 31. That is the hang dv_8ad2fc9f7d9c
      // and dv_c5d1bb496415 measured, and it is why x1024 (all rows active, and
      // bulk_ok so this path is not even taken) ran clean while x1 and x127 did
      // not. tc2's scattered path has the same discipline.
      const uint32_t to = tmem + TCOL_O + half * 64 + lb;
      __nv_bfloat16* orow = outp + static_cast<int64_t>(m0 + row) * o_stride;
      #pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        tmem_ld32(sv2[cc], to + cc * 32);
        tmem_wait_ld();
        if (active) {
          __align__(16) uint32_t pk[16];
          #pragma unroll
          for (int x2 = 0; x2 < 16; ++x2)
            pk[x2] = pack_bf16(sv2[cc][x2 * 2] * g, sv2[cc][x2 * 2 + 1] * g);
          uint4* dst = reinterpret_cast<uint4*>(orow + (half * 2 + cc) * 32);
          const uint4* src4 = reinterpret_cast<const uint4*>(pk);
          dst[0] = src4[0]; dst[1] = src4[1];
          dst[2] = src4[2]; dst[3] = src4[3];
        }
      }
    }
    if (bulk_ok) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      bar_softmax();
      if (tid == 0) {
        tma_store_4d(&ormap, sm32_(stg), 0, it.q_base + m0, 0, it.h);
        bulk_commit();
#ifdef TC9_DRAINLATE
        // DRAINLATE: the read drain moves to the NEXT epilogue's rendezvous
        // (tid0, before the 256-thread bar) plus a final drain after the item
        // loop - identical lifetime guarantees, zero boundary-path exposure.
#else
        bulk_wait_read0();
#endif
      }
    }
#endif   // TC9_DIRECT_EPI
    if (active && has_lse) {
      // one write per row: the half-0 warps own the row sums
      if (half == 0) lsep[static_cast<int64_t>(m0 + row) * heads] = lse2;
    }

    // ---- O drain: this CTA's epilogue has finished READING O out of TMEM ----
    // The pair has exactly one O accumulator (3 S/P slots x 128 + O 128 = all 512
    // TMEM columns, so there is no second O to ping-pong into), and the leader's
    // first PV of the NEXT item overwrites it with accumulate off. ODR - count 2,
    // this CTA's own arrive plus the peer's warp-10 relay - is what that PV waits
    // on. Same shape as the PF/PL pair, and for the same measured reason: a remote
    // arrive costs 329 cyc bare and 824 cyc from the state a warp is in after an
    // MMA or TMEM wait, so the softmax arrives locally and an otherwise-idle warp
    // forwards it. bar_softmax first so one arrive covers all eight warps' reads,
    // fence_before_sync so those TMEM loads are ordered before the arrive, and it
    // also closes the staging-buffer lifetime for the next item's epilogue.
    fence_before_sync();
    bar_softmax();
    if (warp == 0 && lane == 0)
      mbar_arrive(bar(ctrl, rank == 0 ? ODR : OL));
#ifdef TC9_TIMING
    if (timed_s) {
      atomicAdd(dbg + 52, (unsigned long long)(clock64() - e0));
      T9MARK(timed_s, c0);               // restart the period clock for the next
#ifdef TC9_TIMING2
      items_done += 1;
#endif
    }                                    // item's first tile
#else
    if (dbg && tid == 0) atomicAdd(dbg + DBG_ITEMS, 1ull);   // 26 = an issuer slot
#endif
    Gbase += it.T; ++fitem;
    }                                   // end of this cluster's item loop
#if defined(TC9_DRAINLATE) && !defined(TC9_DIRECT_EPI)
    if (tid == 0) bulk_wait_read0();   // last item's store: exit semantics as shipped
#endif
#ifdef TC9_TIMING
    if (timed_s) {
      atomicAdd(dbg + 16, (unsigned long long)sm_tiles);
      atomicAdd(dbg + 17, (unsigned long long)sm_sld);
      atomicAdd(dbg + 18, (unsigned long long)sm_max);
      atomicAdd(dbg + 19, (unsigned long long)sm_exp);
      atomicAdd(dbg + 20, (unsigned long long)sm_bew);
      atomicAdd(dbg + 21, (unsigned long long)sm_corr);
      atomicAdd(dbg + 22, (unsigned long long)sm_per);
#ifdef TC9_TIMING2
      atomicAdd(dbg + 31, (unsigned long long)sm_boundary);
      atomicAdd(dbg + 32, (unsigned long long)sm_bew_first);
      atomicAdd(dbg + 33, (unsigned long long)epi_pv);
      atomicAdd(dbg + 34, (unsigned long long)sm_gap);
      atomicAdd(dbg + 35, (unsigned long long)items_done);
#endif
#ifdef TC9_TIMING4
      atomicAdd(dbg + 36, (unsigned long long)smx_loop);
      atomicAdd(dbg + 37, (unsigned long long)smx_wst);
      atomicAdd(dbg + 38, (unsigned long long)smx_tail);
#endif
#ifdef TC9_TIMING5
      atomicAdd(dbg + 41, (unsigned long long)sm_barconv);
      atomicAdd(dbg + 42, (unsigned long long)sm_pfarr);
#endif
      atomicAdd(dbg + 54, (unsigned long long)sm_tiles);
      const unsigned long long sp = clock64() - t_ent;   // excludes the teardown
      atomicAdd(dbg + 48, sp);
      atomicMax(dbg + 50, sp);
      atomicAdd(dbg + 49, 1ull);
    }
#endif
  }

  // ---- teardown: both CTAs release the pair TMEM (see the alloc comment) ----
  // Skipping this entirely is not an option either: the driver reports "tensor
  // memory not completely freed" at the next synchronize, which the staged ladder
  // measured when its early returns bypassed this point.
  __syncthreads();
  cluster_sync();
  if (warp == 8)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 ::"r"(tmem), "r"(512) : "memory");
}

}  // namespace mla_tc7

// ---------------------------------------------------------------------------
// Host: rank-4 direct maps + cluster launch (no prep, no workspace).
// ---------------------------------------------------------------------------
namespace {
struct MapKey4 {
  const void* ptr; uint64_t tokens, atoms, heads; int box_tok; int box_atoms;
  bool operator==(const MapKey4& o) const {
    return ptr == o.ptr && tokens == o.tokens && atoms == o.atoms &&
           heads == o.heads && box_tok == o.box_tok && box_atoms == o.box_atoms;
  }
};
struct MapEntry4 { MapKey4 key; CUtensorMap map; bool valid = false; };
MapEntry4 g_cache4[32];
int g_next4 = 0;
int encode4(CUtensorMap* m, const void* ptr, uint64_t tokens, uint64_t heads,
            uint64_t atoms, int box_tok, int box_atoms) {
  cuuint64_t gdim[4] = {64, tokens, atoms, heads};
  cuuint64_t gstr[3] = {heads * atoms * 128, 128, atoms * 128};
  cuuint32_t box[4] = {64, static_cast<cuuint32_t>(box_tok),
                       static_cast<cuuint32_t>(box_atoms), 1};
  cuuint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(ptr),
      gdim, gstr, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return static_cast<int>(r);
}
int cached4(CUtensorMap* m, const void* ptr, uint64_t tokens, uint64_t heads,
            uint64_t atoms, int box_tok, int box_atoms) {
  MapKey4 key{ptr, tokens, atoms, heads, box_tok, box_atoms};
  for (auto& e : g_cache4) {
    if (e.valid && e.key == key) { *m = e.map; return 0; }
  }
  int rc = encode4(m, ptr, tokens, heads, atoms, box_tok, box_atoms);
  if (rc == 0) {
    MapEntry4& e = g_cache4[g_next4];
    g_next4 = (g_next4 + 1) & 31;
    e.key = key; e.map = *m; e.valid = true;
  }
  return rc;
}
}  // namespace

extern "C" void mla_tc_ws(int total_kv, int batch, int kv_heads,
                          long long* kvblks, long long* kbytes,
                          long long* vbytes) {
  (void)total_kv; (void)batch; (void)kv_heads;
  *kvblks = 0; *kbytes = 0; *vbytes = 0;
}

extern "C" int mla_tc_forward(const void* q, const void* k, const void* v,
                              void* out, void* lse, const void* sinks,
                              const void* cum_q, const void* cum_kv,
                              void* kws, void* vws,
                              int total_q, int total_kv, int batch, int heads,
                              int kv_heads, int kvblks,
                              int window_left, int causal, int has_lse,
                              int has_sink, float bmm1_scale, float bmm2_scale,
                              int max_kv_len, void* stream,
                              void* dbg, int dbg_mode,
                              int pf_mode, int lag_max, int direct,
                              int box_tok, int opts) {
  (void)total_q; (void)kws; (void)vws; (void)kvblks; (void)max_kv_len;
  (void)dbg_mode;
  (void)pf_mode; (void)lag_max; (void)direct; (void)box_tok; (void)opts;
  using namespace mla_tc7;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  static bool attr_set = false;
  static int g_sms = 0;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(mla_tc7_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return static_cast<int>(e);
    // Ask the device rather than trusting MAX_CLUSTERS: a cluster needs two CTAs
    // on two SMs and 212992 B of dynamic smem per CTA leaves exactly one CTA per
    // SM, so the concurrency limit is half the SM count. tc2p queries the same
    // attribute for the same reason.
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess ||
        g_sms < 2) {
      g_sms = 2 * MAX_CLUSTERS;
    }
    attr_set = true;
  }
  int pairs_ub = total_q / BMP + batch;
  if (pairs_ub <= 0) return static_cast<int>(cudaGetLastError());
  // Persistent launch: one cluster per SM pair, each looping over a strided slice
  // of the (pair-tile, head) item space. The kernel derives the item sequence from
  // (blockIdx.x, gridDim.x) alone, so both CTAs of a pair - and every warp role in
  // them - iterate identically without a device counter or any cross-CTA item
  // handshake. heads moves out of gridDim.y and into the flattened item index.
  const long long total_items = (long long)pairs_ub * heads;
  int ncl = g_sms / 2;
  if (ncl < 1) ncl = 1;
  if (total_items < ncl) ncl = (int)total_items;   // never launch idle clusters

  CUtensorMap krmap, vrmap, qrmap, ormap;
  int rc;
  // K: 64-token x 192-dim boxes, one per CTA half of a BN=128 tile
  if ((rc = cached4(&krmap, k, (uint64_t)total_kv, (uint64_t)kv_heads, 3, 64, 3)))
    return 5000 + rc;
  // V: 128-token x 64-dim boxes; the atom coordinate picks the dim half
  if ((rc = cached4(&vrmap, v, (uint64_t)total_kv, (uint64_t)kv_heads, 2, 128, 1)))
    return 6000 + rc;
  // Q: one 128-row x 192-dim box per CTA
  if ((rc = cached4(&qrmap, q, (uint64_t)total_q, (uint64_t)heads, 3, 128, 3)))
    return 7000 + rc;
  // out: one 128-row x 128-dim box per CTA for the bulk store
  if ((rc = cached4(&ormap, out, (uint64_t)total_q, (uint64_t)heads, 2, 128, 2)))
    return 8000 + rc;

  dim3 grid(2 * ncl, 1, 1);
#ifdef TC7_LAUNCH_EX
  // Explicit cluster launch attribute instead of the compile-time
  // __cluster_dims__ on the kernel. Kept behind a flag so the two mechanisms can
  // be A/B'd: if a 2-CTA cluster launch with 173312 B of dynamic smem per CTA is
  // not being scheduled by the <<<>>> path, the launch returns success and the
  // kernel simply never runs, which looks exactly like a hang.
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.gridDim = grid;
  cfg.blockDim = dim3(THREADS, 1, 1);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = s;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaError_t le = cudaLaunchKernelEx(
      &cfg, mla_tc7_kernel, krmap, vrmap, qrmap, ormap,
      static_cast<__nv_bfloat16*>(out), static_cast<float*>(lse),
      static_cast<const float*>(sinks),
      static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
      total_q, batch, heads, kv_heads, window_left, causal, has_lse, has_sink,
      bmm1_scale, bmm2_scale, reinterpret_cast<unsigned long long*>(dbg));
  if (le != cudaSuccess) return 9000 + static_cast<int>(le);
  return static_cast<int>(cudaGetLastError());
#else
  mla_tc7_kernel<<<grid, THREADS, SMEM_BYTES, s>>>(
      krmap, vrmap, qrmap, ormap, static_cast<__nv_bfloat16*>(out),
      static_cast<float*>(lse), static_cast<const float*>(sinks),
      static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
      total_q, batch, heads, kv_heads, window_left, causal, has_lse, has_sink,
      bmm1_scale, bmm2_scale,
      reinterpret_cast<unsigned long long*>(dbg));
  return static_cast<int>(cudaGetLastError());
#endif
}
