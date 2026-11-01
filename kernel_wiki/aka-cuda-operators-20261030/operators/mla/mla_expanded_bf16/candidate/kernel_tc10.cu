// mla_expanded_bf16 - tcgen05 SINGLE-CTA persistent kernel (tc10): BM=128,
// BN=128, M128N128K16 at the full 8192 flop/cyc/SM rate.
//
// ============================ AXIS VERDICT: REJECTED =======================
// E10 (dv_e4233ffb187d, results/e10_tc10_axis_decision.md): after the
// perm-table root-cause fix, tc10 is fully CORRECT (census==model on 11/11
// shapes, zero timeouts, bit-equal to tc9r on every tc9r-routed shape incl.
// batch>1 sorted, zero tolerance/reference mismatch) but LOSES on latency
// everywhere: vs tc2p64 on its class f0 1.060, f2 1.163, d5 1.126, f1 1.291,
// f6 1.174, d3 1.108; vs tc9r on its class f3 1.061, f9 1.063, f8 1.024,
// x2048b2sl 1.040, f7 1.087 (0/8 rounds faster on 10 of 11 shapes). The
// 8192 flop/cyc/SM instruction rate does not convert to achieved throughput
// at 2 full-width KV stages: the KV-hostile shapes (f1, ws 126MB) lose worst
// - the design note's risk #2 - and the cluster's pair-MMA/relay overlap
// still beats the shorter single-CTA barrier chain. Kept in-tree with full
// evidence per the failed-arms rule; NOT routed in production. The perm-table
// placement bug (ctrl-relative offsets added to the raw smem base -> table
// inside the Q TMA buffer) and its E9e proof are recorded at the perm pointer
// and in results/e9e_perm_stomp_root_cause.md.
// ===========================================================================
//
// Why this kernel exists (join-20261026 phase, measured anchors only):
//   * kernel_tc9.cu's header records the arm-b micro measurement: a pair
//     M256N128K16 costs 64.02 cyc, "the same as a single M128N128K16" -
//     i.e. ONE CTA issuing N=128 instructions runs the tensor core at
//     8192 flop/cyc/SM, while kernel_tc2p.cu's M128N64K16 A-from-smem
//     stream sustains 5459 flop/cyc/SM. The single-CTA production engine
//     therefore leaves a 1.5x MMA-rate gap on the table.
//   * the E2 census + span math put the tc2p64 steady cadence at
//     ~1323-1575 cyc per 128x64 subtile tile (~3963 flop/cyc/SM achieved
//     on f1) and the cluster cadence at ~2033-2050 cyc per 256x128 pair
//     tile (~5116 flop/cyc/SM per CTA, 0.958 us/tile recorded in the
//     kernel_tc9.cu GSORT comment). tc10 targets the cluster's per-CTA
//     throughput WITHOUT the cluster: no pair MMA, no BR/PF/ODR relays,
//     no remote arrives (329-824 cyc issue cost each, measured cls6/cls7),
//     no cluster rendezvous, no 2-CTA launch scheduling.
//   * E4a proved the cluster steady state is NOT softmax-instruction-bound
//     (EARLY_PST/XMAX1 = ties), so the per-tile softmax text below is tc9's
//     unchanged (fused ld64+redmax, column-split 8 warps, packed f16x2 exp,
//     lagged max); the win has to come from the MMA rate and the shorter
//     barrier chain, and the A/B decides.
//
// Structure (320 threads, grid = min(#SM, items), 1 CTA/SM, persistent):
//   warps 0-7 softmax (rows 32*(w&3), column half w>>2 - tc9's layout)
//   warp  8: elected-lane MMA issuer (QK + PV, cta_group::1)
//   warp  9: elected-lane TMA loader (Q refill + flat global KV ring)
// Item space: (128-row q-tile, head) pairs enumerated by a pure function of
// (blockIdx.x, gridDim.x) - no device counter, no publication handshake -
// with tc9's GSORT (globally descending-T order, causal batch>1) and SERP
// (serpentine stride, causal) static balance, both self-gating exactly as
// the adopted tc9r build does.
//
// TMEM (512 cols, cta_group::1): S/P slots 3 x 128 at cols 0..383 (f32 S in
// all 128, packed bf16 P in the first 64 of the warp-half's 32-col image),
// O accumulator 128 cols at 384..511.
//
// SMEM (215680 B of the 232448 opt-in):
//   Q      49152  (128 rows x 192 dims, one TMA box, refilled per item)
//   K      2 x 49152 (BN=128 x 192, FULL tile per CTA - no cluster split)
//   V      2 x 32768 (BN=128 x 128 dims)
//   ctrl   2688   (15 mbarriers + tmem holder + 8x32 maxbuf + perm table)
// KV_STAGES is 2 (not tc9's 3): a full-width stage is 81920 B and 3 of them
// do not fit next to Q. The KVSPLIT release law makes 2 enough: K(s) frees at
// QK(j-2) COMMITMENT (SF), V(s) at PV(j-2) retirement - at the 1280-cyc MMA
// floor K lands ~1190 cyc into a 2560-cyc window and V ~1560, both ahead of
// need. Epilogue is tc2p-style direct st.global.v8 stores (no 32 KB staging
// image fits, and the staged TMA store was a cluster-era artifact).
//
// Barrier protocol per tile i (s2 = i%2 stage, s3 = i%3 S/P slot):
//   BFK[s2] cnt1 - K(i) TMA landed (loader expect_tx)
//   BF[s2]  cnt1 - V(i) TMA landed
//   SF[s3]  cnt1 - QK(i) committed (tcgen05.commit; also frees K(i+2))
//   PF[s3]  cnt1 - softmax stored P(i) and applied the O correction
//                  (warp0 lane0 after the 8-warp bar; LOCAL - no relay)
//   PV[s3]  cnt1 - PV(i) committed; frees S/P slot s3 (QK(i+3)) AND V(i+2)
//   QF      cnt1 - the item's Q box landed (parity fitem&1)
//   ODR     cnt1 - epilogue finished reading O (gates the NEXT item's first
//                  PV, which overwrites the single O accumulator)
//
// v1 probe discipline: ALL waits are BOUNDED (200k-cycle budget, timeout
// counted into dbg[slot]) so a protocol mistake reports itself instead of
// hanging a job; dbg also carries CTAS/ITEMS/TILES censuses. Production
// adoption (if the A/B earns it) switches to the unbounded spelling like
// tc7/tc9 did.
//
// Numerics: per-row math is tc9's textually (same BN=128 tile boundaries,
// same 12-step QK / 8-step PV k-order, same lagged max, same packed f16x2
// exp with the -200 sentinel, same two-half row-sum combine order), so
// tc10 outputs are expected BIT-EQUAL to tc9r's on any shape both run -
// that bit equality is the probe's primary correctness gate, with the fp32
// reference as the second.
//
// Build: nvcc -gencode=arch=compute_103a,code=sm_103a -lcuda.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define DIN __device__ __forceinline__

namespace mla_tc10 {

constexpr int DQK = 192;
constexpr int DV = 128;
constexpr int BM = 128;             // q rows per item (one CTA)
constexpr int BN = 128;             // kv tokens per tile
constexpr int KV_STAGES = 2;
constexpr int S_STAGES = 3;
constexpr int THREADS = 320;
constexpr int SOFT_WARPS = 8;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -1.0e30f;
constexpr float LAG_MAX_LOG2 = 8.0f;

constexpr int Q_BYTES = BM * DQK * 2;                        // 49152
constexpr int K_BYTES = BN * DQK * 2;                        // 49152
constexpr int V_BYTES = BN * DV * 2;                         // 32768
constexpr int KV_TOTAL = KV_STAGES * (K_BYTES + V_BYTES);    // 163840
constexpr int CTRL_OFF = Q_BYTES + KV_TOTAL;                 // 212992

// ctrl: NBARS mbarriers (8B) + tmem holder (8B) + maxbuf (8x32 f32) +
// GSORT counting-sort tables.
enum { BFK0 = 0, BFK1, BF0, BF1, SF0, SF1, SF2, PF0, PF1, PF2,
       PV0, PV1, PV2, QF, ODR, NBARS };
constexpr int THOLD_OFF = NBARS * 8;                 // 120
constexpr int MAXBUF_OFF = THOLD_OFF + 8;            // 128
constexpr int PERM_OFF = MAXBUF_OFF + SOFT_WARPS * 32 * 4;   // 1152
// GSORT table capacities, bounded by the FROZEN domain (batch <= 32,
// sum(query_lengths) <= 16384 -> tiles_ub <= 16384/128 + 32 = 160). ANY
// bound violation (out-of-domain metadata) degrades to the IDENTITY order
// via do_sort's tiles_ub<=TC10_TILES_CAP gate - never a partial table. The
// build is tc9r's deterministic O(tiles_ub^2) rank scan (ties by ascending
// tile id): every CTA derives a bit-identical rank->tile_id table with zero
// communication.
//
// BUG RECORD - the E8/E9/E9c/E9d "table stomp" (CLOSED, root-caused by E9e
// dv_44f8e0254d30): v1-v3 computed the table pointers as smem+PERMTBL_OFF /
// smem+TVAL_OFF, but these offsets are CTRL-RELATIVE - the table landed at
// absolute [2312,2952) INSIDE the 49152B Q TMA buffer, and every item's Q
// refill overwrote the ranks with Q bf16 data. Signature: build-time
// readback correct in 148/148 CTAs, exit values float-like and
// data-dependent per CTA, loader (running ahead) decoded more good items
// than issuer/softmax, NOSORT and batch==1 immune (no table), GPERM immune
// (global table) and BIT-EQUAL to tc9r on f3/f8/x2048b2sl. E9d's
// "destroyed canaries" were the same missing-CTRL_OFF bug in the canary
// expressions; E9e's ctrl-based canaries/patterns all survived while only
// the table words mutated. Fix: perm = ctrl + PERMTBL_OFF, build_perm takes
// ctrl. The E8-era "cap fallback read stale smem" story was this same bug.
constexpr int T_MAX = 128;                       // ceil(16384 / BN)
#define TC10_TILES_CAP 160                       // 16384/128 + 32 domain bound
constexpr int TVAL_OFF = PERM_OFF;                          // 160 int
constexpr int HIST_OFF = TVAL_OFF + TC10_TILES_CAP * 4;     // 130 int
// TC10_GUARD / TC10_TAILPAD: E9c/E9e triage flags, theories SUPERSEDED by
// the root cause above (the stomp never came from a neighboring region or
// the allocation top). Kept inert: TAILPAD's 4KB additionally absorbs the
// one real top-relative write E9e found - a repdigit u64 at
// [SMEM_BYTES-16, SMEM_BYTES-8) in every launch (benign hw/driver
// bookkeeping; lands in the stock +128B pad otherwise).
#ifdef TC10_GUARD
constexpr int GUARD_BYTES = 2048;
#else
constexpr int GUARD_BYTES = 0;
#endif
constexpr int PERMTBL_OFF = HIST_OFF + (T_MAX + 2) * 4 + GUARD_BYTES;
// TC10_TAILPAD: dead 4 KB behind the table (E9e top-tracking test - theory
// falsified: tc10x reproduced the corruption identically because the table
// was never in the ctrl tail; see the bug record above). Flag kept inert.
#ifdef TC10_TAILPAD
constexpr int TAIL_PAD = 4096;
#else
constexpr int TAIL_PAD = 0;
#endif
constexpr int CTRL_BYTES = PERMTBL_OFF + TC10_TILES_CAP * 4 + 128 + TAIL_PAD;
constexpr int SMEM_BYTES = CTRL_OFF + CTRL_BYTES;
static_assert(SMEM_BYTES <= 232448, "tc10 smem over the opt-in max");

// TMEM column map (per-CTA half of tc9's): S/P slot s at 128s, O at 384.
DIN int TCOL_S(int s) { return s * 128; }
constexpr int TCOL_O = 384;

// dbg slots (bounded-wait timeouts + census; tc9's non-timing layout)
enum { DBG_TO_BFK = 16, DBG_TO_BF, DBG_TO_SF, DBG_TO_PF, DBG_TO_PV,
       DBG_TO_QF, DBG_TO_ODR = 23, DBG_TILES = 24, DBG_CTAS = 25,
       DBG_ITEMS = 26 };

DIN uint32_t sm32_(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
DIN uint32_t bar(char* ctrl, int idx) { return sm32_(ctrl) + idx * 8; }

// ---------------------------------------------------------------------------
// PTX primitives. 1-CTA forms copied from the promoted kernel_tc2.cu /
// kernel_tc9.cu (both measured); the bounded wait is tc9's step-1 form.
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
// Bounded wait (v1 probe): a protocol mistake reports into dbg[slot] and
// proceeds instead of hanging the job. Production adoption builds with
// -DTC7_UNBOUNDED (tc9's flag name): the bounded form must never ship -
// a timed-out wait proceeds on unconsumed data.
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
// idesc: M128 N128 bf16 f32acc (M field = 128>>4 = 8 at bit 24, N = 128>>3
// = 16 at bit 17); the PV form adds b_major. Field layout as documented in
// kernel_tc9.cu (pair M256 form) and kernel_tc2.cu (single M128N64 form).
constexpr uint32_t IDESC_QK = (uint32_t(128 >> 4) << 24) |
                              (uint32_t(128 >> 3) << 17) |
                              (1u << 10) | (1u << 7) | (1u << 4);
constexpr uint32_t IDESC_PV = IDESC_QK | (1u << 16);
DIN void mma_ss(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc,
                int accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::
          "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum));
}
DIN void mma_ts(uint32_t tmem_d, uint32_t tmem_a, uint64_t db, uint32_t idesc,
                int accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" ::
          "r"(tmem_d), "r"(tmem_a), "l"(db), "r"(idesc), "r"(accum));
}
DIN void mma_commit(uint32_t mbar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::
      "r"(mbar));
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
// 64-column load that also returns the row maximum (kernel_tc9.cu: same rate
// as a plain x64 load, bit-exact against an FMNMX tree).
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
      "f"(d[12]), "f"(d[13]), "f"(d[14]), "f"(d[15]), "f"(d[16]),
      "f"(d[17]), "f"(d[18]), "f"(d[19]), "f"(d[20]), "f"(d[21]),
      "f"(d[22]), "f"(d[23]), "f"(d[24]), "f"(d[25]), "f"(d[26]),
      "f"(d[27]), "f"(d[28]), "f"(d[29]), "f"(d[30]), "f"(d[31]), "f"(d[32])
      : "memory");
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
// Packed exp2 (TC8_EX2F16 semantics, baked in: this is the promoted softmax
// quantization of both production engines - f16-rounded argument, one
// ex2.approx.f16x2 per pair, -200 sentinel for masked columns so underflow
// is exact zero and no inf/NaN can reach the f16 exp).
DIN uint32_t cvt_f16x2(float lo, float hi) {
  uint32_t y;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(y) : "f"(hi), "f"(lo));
  return y;
}
DIN uint32_t ex2_f16x2(uint32_t x) {
  uint32_t y;
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
// 8 softmax warps (256 threads); the issuer/loader warps never arrive.
DIN void bar_softmax() { asm volatile("bar.sync 1, 256;" ::: "memory"); }

// ---------------------------------------------------------------------------
// Per-item geometry (persistent: pure function of the item index + immutable
// cum_q/cum_kv, decoded identically by all ten warps - tc9's law).
// ---------------------------------------------------------------------------
struct Item {
  int q_base, kv_base, qlen, kvlen;
  int m0, kv_off, T, Tfull, h, h_kv, tok0;
  int64_t o_off;
  float sink2;
  bool windowed;
};

// Tiles for one tile_id (GSORT ranking): decode's T arithmetic minus the
// per-thread fields.
DIN int tile_T(const int32_t* cum_q, const int32_t* cum_kv, int batch,
               int causal, int tile_id) {
  int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
  for (int i = 0; i < batch; ++i) {
    const int nq = cum_q[i + 1], nkv = cum_kv[i + 1];
    const int len_q = nq - prev_q, len_kv = nkv - prev_kv;
    const int tiles = (len_q + BM - 1) / BM;
    if (tile_id < acc + tiles) {
      const int m0 = (tiles - 1 - (tile_id - acc)) * BM;
      int nmax = causal ? min(len_kv, (len_kv - len_q) + m0 + BM) : len_kv;
      nmax = min(nmax, len_kv);
      const int T = (nmax + BN - 1) / BN;
      return T > 0 ? T : 0;
    }
    acc += tiles; prev_q = nq; prev_kv = nkv;
  }
  return 0;
}

// Table read: GPERM keeps the table in a per-CTA global scratch slice
// (immune to whatever stomps the smem ctrl tail - E9c/E9d); the default
// build reads the smem table.
#ifdef TC10_GPERM
DIN int perm_read(const int* perm, int idx) { return __ldcg(perm + idx); }
#else
DIN int perm_read(const int* perm, int idx) { return perm[idx]; }
#endif

DIN int smid_() {
  int s;
  asm("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

// perm[r] = the tile id holding the r-th largest T, ties by ASCENDING tile
// id - EXACTLY tc9's rule, and the determinism is load-bearing in a way the
// E9 triage (dv_a682d95b42c1) proved the hard way: a counting-sort variant
// with atomic scatter produced a VALID, correctly SORTED table in every
// CTA - and still desynced the roles, because the tie order differed
// between CTAs (arrival-order atomics) while the serpentine stride makes
// each CTA consume ranks its PEERS built differently... each CTA reads its
// OWN table consistently, but the balance assumption (and the E8 census
// arithmetic) presumes one global order; more fatally, the v1 E8 scan form
// corrupted within a single CTA (roles saw mid-launch table changes - the
// O(n^2) read-back phase remains unexplained at the mechanism level). The
// deterministic scan below is the form that ships in tc9r and measured
// clean on f3/f9-class batch==1 shapes in E8; ties-by-ascending-id makes
// every CTA's table bit-identical with zero communication.
// Caller must __syncthreads() afterwards to publish.
// __noinline__ deliberately (entry-block size law, see kernel_tc9.cu).
__device__ __noinline__ void build_perm(const int32_t* cum_q,
                                        const int32_t* cum_kv, int batch,
                                        int causal, int tiles_ub, char* ctrl,
                                        int* perm_out) {
  int* const perm = perm_out;
  int* const tval = reinterpret_cast<int*>(ctrl + TVAL_OFF);
  const int tid = threadIdx.x;
  for (int p = tid; p < tiles_ub; p += blockDim.x)
    tval[p] = tile_T(cum_q, cum_kv, batch, causal, p);
  __syncthreads();
  for (int p = tid; p < tiles_ub; p += blockDim.x) {
    const int tp = tval[p];
    int r = 0;
    for (int q = 0; q < tiles_ub; ++q) {
      const int tq = tval[q];
      r += (tq > tp) | ((tq == tp) & (q < p));
    }
    perm[r] = p;
  }
#ifdef TC10_GPERM
  __threadfence();   // publish the global-scratch table to this CTA's roles
#endif
}

DIN bool decode_item(const int32_t* cum_q, const int32_t* cum_kv, int batch,
                     int heads, int kv_heads, int causal, int window_left,
                     int has_sink, const float* sinks, int tiles_ub,
                     long long item, Item& it, const int* perm) {
  (void)tiles_ub;   // kept for signature parity with the tc9 decode
  // rank-major flattening (item / heads = tile rank, item % heads = head):
  // ONE globally descending ramp under GSORT, the order arm-b measured at
  // 1.113x mean load vs 1.270 plain / 2.398 head-major (f9).
  const int trank = static_cast<int>(item / heads);
  const int tile_id = (perm != nullptr && trank < TC10_TILES_CAP)
                          ? perm_read(perm, trank) : trank;
  it.h = static_cast<int>(item % heads);
  int b = -1, m0 = 0, q_base = 0, kv_base = 0, qlen = 0, kvlen = 0;
  {
    int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
    for (int i = 0; i < batch; ++i) {
      int nq = cum_q[i + 1], nkv = cum_kv[i + 1];
      int len_q = nq - prev_q;
      int tiles = (len_q + BM - 1) / BM;
      if (b < 0 && tile_id < acc + tiles) {
        b = i;
        // LPT within the request: heaviest q-tile first.
        m0 = (tiles - 1 - (tile_id - acc)) * BM;
        q_base = prev_q; kv_base = prev_kv;
        qlen = len_q; kvlen = nkv - prev_kv;
        break;
      }
      acc += tiles; prev_q = nq; prev_kv = nkv;
    }
  }
  if (b < 0) return false;             // surplus tile id: every role agrees
  it.q_base = q_base; it.kv_base = kv_base; it.qlen = qlen; it.kvlen = kvlen;
  it.m0 = m0;
  it.kv_off = kvlen - qlen;
  it.windowed = window_left >= 0;
  int nmax = causal ? min(kvlen, it.kv_off + m0 + BM) : kvlen;
  nmax = min(nmax, kvlen);
  it.T = (nmax + BN - 1) / BN;
  // first tile whose columns are ALL unmasked for these 128 rows
  int nsafe = causal ? min(kvlen, it.kv_off + m0 + 1) : kvlen;
  if (it.windowed) nsafe = 0;
  it.Tfull = nsafe / BN;
  if (it.T <= 0) return false;
  it.h_kv = it.h / (heads / kv_heads);
  it.tok0 = kv_base;
  it.o_off = static_cast<int64_t>(q_base) * heads + it.h;
  it.sink2 = has_sink ? sinks[it.h] * LOG2E : 0.0f;
  return true;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS, 1)
mla_tc10_kernel(const __grid_constant__ CUtensorMap krmap,
                const __grid_constant__ CUtensorMap vrmap,
                const __grid_constant__ CUtensorMap qrmap,
                __nv_bfloat16* __restrict__ out,
                float* __restrict__ lse,
                const float* __restrict__ sinks,
                const int32_t* __restrict__ cum_q,
                const int32_t* __restrict__ cum_kv,
                int32_t total_q, int32_t batch, int heads, int kv_heads,
                int window_left, int causal, int has_lse, int has_sink,
                float bmm1_scale, float bmm2_scale,
                unsigned long long* __restrict__ dbg,
                int* __restrict__ gperm) {
  extern __shared__ char smem[];
  char* sq = smem;                              // 49152 B Q
  char* sk = smem + Q_BYTES;                    // 2 x 49152 B K stages
  char* sv = sk + KV_STAGES * K_BYTES;          // 2 x 32768 B V stages
  char* ctrl = smem + CTRL_OFF;
  uint32_t* tholder = reinterpret_cast<uint32_t*>(ctrl + THOLD_OFF);
  float* maxbuf = reinterpret_cast<float*>(ctrl + MAXBUF_OFF);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int cid = blockIdx.x;
  const int ncl = gridDim.x;
  const int tiles_ub = total_q / BM + batch;
  const long long total_items = static_cast<long long>(tiles_ub) * heads;
  // SERP (causal only) + GSORT (causal batch>1): tc9r's adopted static
  // balance, self-gating on exactly its measured conditions.
  const bool serp = (causal != 0);
  const int slotA = cid;
  const int slotB = serp ? (ncl - 1 - cid) : cid;
#ifdef TC10_NOSORT
  const bool do_sort = false;   // triage: identity order (serp stride stays)
#else
  // Domain guard: the counting-sort tables cover tiles_ub <= TC10_TILES_CAP;
  // out-of-domain shapes keep the identity order (decode's nullptr guard),
  // never a partial table.
  const bool do_sort = serp && (batch > 1) && (tiles_ub <= TC10_TILES_CAP);
#endif
#ifdef TC10_GPERM
  int* const perm = do_sort ? (gperm + static_cast<size_t>(cid) * TC10_TILES_CAP)
                            : nullptr;
#else
  (void)gperm;
  // ROOT CAUSE of the E8/E9/E9c/E9d "table stomp" (fixed post-E9e,
  // dv_44f8e0254d30): PERMTBL_OFF/TVAL_OFF are CTRL-RELATIVE constants, but
  // v1-v3 added them to the raw smem base - the table (and build_perm's
  // tval) landed at absolute [2312,2952), INSIDE the 49152B Q TMA buffer,
  // and every item's Q refill overwrote the ranks with Q bf16 data (the
  // "float-like" exit payload; loader-ahead-of-issuer progressive desync;
  // NOSORT/batch==1 immune because they never build a table; GPERM immune
  // because its table is global). E9e proved the confinement: ctrl-based
  // canaries/pattern words all survived while only the buggy-pointer table
  // words mutated. The table belongs at ctrl + PERMTBL_OFF.
  int* const perm = do_sort ? reinterpret_cast<int*>(ctrl + PERMTBL_OFF)
                            : nullptr;
#endif
  const uint32_t whint = 0x989680u;
  const int64_t o_stride = static_cast<int64_t>(heads) * DV;
  const float sc2 = bmm1_scale * LOG2E;
  long long Gbase = 0;
  long long fitem = 0;

  if (dbg && tid == 0) atomicAdd(dbg + DBG_CTAS, 1ull);

  // ---- control init (parallel: one thread per barrier; all count 1) ------
  if (tid < NBARS) mbar_init(bar(ctrl, tid), 1);
  asm volatile("fence.mbarrier_init.release.cluster;");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (do_sort) {
    // NOTE: build_perm's base MUST be ctrl (TVAL_OFF is ctrl-relative -
    // see the perm-pointer root-cause comment above).
    build_perm(cum_q, cum_kv, batch, causal, tiles_ub, ctrl, perm);
    __syncthreads();          // publish perm[] to every warp role
#ifdef TC10_TRIAGE
    if (dbg) {
      if (tid < 120 && tid < tiles_ub) dbg[40 + tid] = perm_read(perm, tid);
      // E9e: pattern-fill every dead ctrl word around the table (hist gap +
      // tail pad, each word tagged with its own ctrl offset) so the exit
      // scan maps the stomp footprint exactly; canaries bracket the table,
      // one sits at the allocation top; watch slots armed for the per-role
      // first-corruption item index.
      volatile int* cw = reinterpret_cast<volatile int*>(ctrl);
      for (int w = HIST_OFF + tid * 4; w < PERMTBL_OFF; w += blockDim.x * 4)
        cw[w >> 2] = static_cast<int>(0x5A5A0000u | static_cast<unsigned>(w >> 2));
      for (int w = PERMTBL_OFF + TC10_TILES_CAP * 4 + tid * 4;
           w < CTRL_BYTES - 16; w += blockDim.x * 4)
        cw[w >> 2] = static_cast<int>(0x5A5A0000u | static_cast<unsigned>(w >> 2));
      __syncthreads();
      if (tid == 0) {
        dbg[38] = 1; dbg[39] = tiles_ub;
        // canary_left (top-896 without pad), canary_right (top-64 without
        // pad), top canary at the allocation's last word
        cw[(PERMTBL_OFF - 128) >> 2] = static_cast<int>(0xC0DE1234);
        cw[(PERMTBL_OFF + TC10_TILES_CAP * 4 + 64) >> 2] =
            static_cast<int>(0xBEEF5678);
        cw[(CTRL_BYTES - 8) >> 2] = static_cast<int>(0x0BADF00D);
        // watch slots dbg[1550..1552] are armed HOST-SIDE (all-ones): a
        // per-CTA init here would race another CTA's earlier atomicMin.
        dbg[400 + cid] = perm_read(perm, 0);
        dbg[600 + cid] = (tiles_ub > 1) ? perm_read(perm, 1) : -1;
      }
    }
#endif
  }
  if (warp == 8) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 ::"r"(sm32_(tholder)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  fence_after_sync();
  const uint32_t tmem = *(volatile uint32_t*)tholder;

  // Descriptor constants. Q/K: SW128 K-major, 8-row core matrices (sbo
  // 1024 B); atom plane = 128 tokens x 128 B = 16384 B -> 1024 in 16 B
  // units. V: two 64-dim atoms of 128 tokens -> lbo 16384, k-step 16
  // tokens x 128 B = 2048 B -> 128 in 16 B units. (tc2p box64's measured
  // spelling, scaled from its 64-token planes.)
  const uint32_t qsbo = 1024u, qa16 = 1024u;
  const uint32_t ksbo = 1024u, ka16 = 1024u;
  const uint32_t vlbo = 16384u, vsbo = 1024u;
  const int vstep16 = 2048 / 16;

  if (warp == 9) {
    // ---------------- TMA LOADER (persistent) ----------------
    if (elect_one()) {
      for (long long k10 = 0, base10 = 0, item = slotA;
           item < total_items;
           (base10 += ncl, ++k10,
            item = base10 + ((k10 & 1) ? slotB : slotA))) {
        Item it;
        if (!decode_item(cum_q, cum_kv, batch, heads, kv_heads, causal,
                         window_left, has_sink, sinks, tiles_ub, item, it,
                         perm))
          continue;
        const long long G0 = Gbase;
#ifdef TC10_TRIAGE
        if (dbg) {
          atomicAdd(dbg + 170, 1ull);
          if (do_sort &&
              reinterpret_cast<volatile int*>(ctrl + PERMTBL_OFF - 128)[0] !=
                  static_cast<int>(0xC0DE1234))
            atomicMin(dbg + 1550, static_cast<unsigned long long>(G0 + 1));
        }
#endif
        // Q refill: the single Q buffer is being read by the previous item's
        // QK stream until its last commit lands (SF of tile G0-1); the
        // transfer then overlaps the previous item's softmax tail/epilogue.
        if (fitem > 0) {
          const long long gp = G0 - 1;
          mbwait(bar(ctrl, SF0 + (int)(gp % S_STAGES)),
                 (int)((gp / S_STAGES) & 1), dbg, DBG_TO_SF, whint);
          fence_after_sync();
        }
        mbar_expect_tx(bar(ctrl, QF), Q_BYTES);
        tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF), 0,
                    it.q_base + it.m0, 0, it.h);
        for (long long j = G0; j < G0 + it.T; ++j) {
          const int s2 = (int)(j % KV_STAGES);
          const int jt = (int)(j - G0);
          // KVSPLIT release law on 2 stages: K(s2) frees at QK(j-2)'s
          // COMMITMENT (SF), V(s2) at PV(j-2) retirement - both barriers
          // are S-slot-indexed (mod 3) while the stage is mod 2; the parity
          // arithmetic stays exact because every commit flips its barrier
          // once per 3 tiles.
          if (j >= KV_STAGES) {
            mbwait(bar(ctrl, SF0 + (int)((j - KV_STAGES) % S_STAGES)),
                   (int)(((j - KV_STAGES) / S_STAGES) & 1), dbg, DBG_TO_SF,
                   whint);
          }
          mbar_expect_tx(bar(ctrl, BFK0 + s2), K_BYTES);
          tma_load_4d(&krmap, sm32_(sk) + s2 * K_BYTES, bar(ctrl, BFK0 + s2),
                      0, it.tok0 + jt * BN, 0, it.h_kv);
          if (j >= KV_STAGES) {
            mbwait(bar(ctrl, PV0 + (int)((j - KV_STAGES) % S_STAGES)),
                   (int)(((j - KV_STAGES) / S_STAGES) & 1), dbg, DBG_TO_PV,
                   whint);
          }
          mbar_expect_tx(bar(ctrl, BF0 + s2), V_BYTES);
          tma_load_4d(&vrmap, sm32_(sv) + s2 * V_BYTES, bar(ctrl, BF0 + s2),
                      0, it.tok0 + jt * BN, 0, it.h_kv);
        }
        Gbase += it.T; ++fitem;
      }
    }
  } else if (warp == 8) {
    // ---------------- ISSUER (elected lane) ----------------
    if (elect_one()) {
      const uint32_t qb0 = sm32_(sq);
      for (long long k10 = 0, base10 = 0, item = slotA;
           item < total_items;
           (base10 += ncl, ++k10,
            item = base10 + ((k10 & 1) ? slotB : slotA))) {
        Item it;
        if (!decode_item(cum_q, cum_kv, batch, heads, kv_heads, causal,
                         window_left, has_sink, sinks, tiles_ub, item, it,
                         perm))
          continue;
        const long long G0 = Gbase;
#ifdef TC10_TRIAGE
        if (dbg && do_sort &&
            reinterpret_cast<volatile int*>(ctrl + PERMTBL_OFF - 128)[0] !=
                static_cast<int>(0xC0DE1234))
          atomicMin(dbg + 1551, static_cast<unsigned long long>(G0 + 1));
#endif
        for (long long i = G0; i < G0 + it.T; ++i) {
          const int s2 = (int)(i % KV_STAGES);
          const int s3 = (int)(i % S_STAGES);
          mbwait(bar(ctrl, BFK0 + s2), (int)((i / KV_STAGES) & 1), dbg,
                 DBG_TO_BFK, whint);
          if (i == G0) {
            mbwait(bar(ctrl, QF), (int)(fitem & 1), dbg, DBG_TO_QF, whint);
            fence_after_sync();
          }
          // S/P slot s3 frees when PV(i-3) retired.
          if (i >= S_STAGES) {
            mbwait(bar(ctrl, PV0 + s3), (int)((i / S_STAGES - 1) & 1), dbg,
                   DBG_TO_PV, whint);
            fence_after_sync();
          }
          // ---- QK(i): 12 k-steps of M128N128K16, A=Q from smem ----
          {
            const uint32_t ts = tmem + TCOL_S(s3);
            const uint64_t da = make_desc(qb0, 0, qsbo);
            const uint64_t db =
                make_desc(sm32_(sk) + s2 * K_BYTES, 0, ksbo);
            #pragma unroll
            for (int kk = 0; kk < 12; ++kk)
              mma_ss(ts, da + ((kk >> 2) * qa16 + (kk & 3) * 2),
                     db + ((kk >> 2) * ka16 + (kk & 3) * 2),
                     IDESC_QK, kk > 0);
            mma_commit(bar(ctrl, SF0 + s3));
          }
          // ---- PV(i-1): 8 k-steps, A=P from TMEM, B=V from smem ----
          const long long ip = i - 1;
          if (ip >= G0) {
            const int sp3 = (int)(ip % S_STAGES);
            const int sp2 = (int)(ip % KV_STAGES);
            // First PV of the item overwrites the single O accumulator:
            // the previous item's epilogue O reads must have drained.
            if (ip == G0 && fitem > 0) {
              mbwait(bar(ctrl, ODR), (int)((fitem - 1) & 1), dbg,
                     DBG_TO_ODR, whint);
              fence_after_sync();
            }
            mbwait(bar(ctrl, PF0 + sp3), (int)((ip / S_STAGES) & 1), dbg,
                   DBG_TO_PF, whint);
            mbwait(bar(ctrl, BF0 + sp2), (int)((ip / KV_STAGES) & 1), dbg,
                   DBG_TO_BF, whint);
            fence_after_sync();
            const uint32_t tp = tmem + TCOL_S(sp3);
            const uint64_t db =
                make_desc(sm32_(sv) + sp2 * V_BYTES, vlbo, vsbo);
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk)
              mma_ts(tmem + TCOL_O, tp + kk * 8, db + kk * vstep16, IDESC_PV,
                     (ip > G0) | (kk > 0));
            mma_commit(bar(ctrl, PV0 + sp3));
          }
          if (dbg) atomicAdd(dbg + DBG_TILES, 1ull);
        }
        // ---- tail: PV(G0+T-1), this item's last ----
        {
          const long long il = G0 + it.T - 1;
          const int sl3 = (int)(il % S_STAGES);
          const int sl2 = (int)(il % KV_STAGES);
          if (it.T == 1 && fitem > 0) {
            mbwait(bar(ctrl, ODR), (int)((fitem - 1) & 1), dbg, DBG_TO_ODR,
                   whint);
            fence_after_sync();
          }
          mbwait(bar(ctrl, PF0 + sl3), (int)((il / S_STAGES) & 1), dbg,
                 DBG_TO_PF, whint);
          mbwait(bar(ctrl, BF0 + sl2), (int)((il / KV_STAGES) & 1), dbg,
                 DBG_TO_BF, whint);
          fence_after_sync();
          const uint32_t tp = tmem + TCOL_S(sl3);
          const uint64_t db =
              make_desc(sm32_(sv) + sl2 * V_BYTES, vlbo, vsbo);
          #pragma unroll
          for (int kk = 0; kk < 8; ++kk)
            mma_ts(tmem + TCOL_O, tp + kk * 8, db + kk * vstep16, IDESC_PV,
                   (it.T > 1) | (kk > 0));
          mma_commit(bar(ctrl, PV0 + sl3));
        }
        Gbase += it.T; ++fitem;
        if (dbg) atomicAdd(dbg + DBG_ITEMS, 1ull);   // elected lane only
      }
    }
  } else if (warp < SOFT_WARPS) {
    // ---------------- SOFTMAX (column-split over 8 warps) ----------------
    const int w3 = warp & 3;                 // row group: rows 32*w3 .. +31
    const int half = warp >> 2;              // column half: kv 64*half .. +63
    const int row = w3 * 32 + lane;          // row within this CTA's 128
    const uint32_t lb = static_cast<uint32_t>(w3 * 32) << 16;
    float sv2[2][32];
    for (long long k10 = 0, base10 = 0, item = slotA;
         item < total_items;
         (base10 += ncl, ++k10,
          item = base10 + ((k10 & 1) ? slotB : slotA))) {
      Item it;
      if (!decode_item(cum_q, cum_kv, batch, heads, kv_heads, causal,
                       window_left, has_sink, sinks, tiles_ub, item, it, perm))
        continue;
      const long long G0 = Gbase;
#ifdef TC10_TRIAGE
      if (dbg && warp == 0 && lane == 0) {
        atomicAdd(dbg + 171, 1ull);
        if (do_sort &&
            reinterpret_cast<volatile int*>(ctrl + PERMTBL_OFF - 128)[0] !=
                static_cast<int>(0xC0DE1234))
          atomicMin(dbg + 1552, static_cast<unsigned long long>(G0 + 1));
      }
#endif
      const int m0 = it.m0;
      const int kv_off = it.kv_off;
      const bool windowed = it.windowed;
      const int Tfull = it.Tfull;
      const int qlen = it.qlen;
      const int kvlen = it.kvlen;
      const int grow = m0 + row;
      const int vis = kv_off + grow;
      const bool active = (grow < qlen);
      const float sink2 = it.sink2;
      __nv_bfloat16* outp = out + it.o_off * DV;
      float* lsep = has_lse ? lse + it.o_off : nullptr;
      float mr = NEG_BIG, d = 0.0f;
      for (long long i = G0; i < G0 + it.T; ++i) {
        const int s3 = (int)(i % S_STAGES);
        const uint32_t ts = tmem + TCOL_S(s3) + half * 64 + lb;
        mbwait(bar(ctrl, SF0 + s3), (int)((i / S_STAGES) & 1), dbg,
               DBG_TO_SF, whint);
        fence_after_sync();
#ifdef TC10_TRIAGE
        if (dbg && warp == 0 && lane == 0) atomicAdd(dbg + 172, 1ull);
#endif
        float* mx = maxbuf + warp * 32 + lane;
        float* mxp = maxbuf + (warp ^ 4) * 32 + lane;
        const int n0 = (int)(i - G0) * BN + half * 64;
        const bool masked = (i - G0) >= Tfull;
        // fused 64-column load + hardware row max over this warp's columns
        float tm = tmem_ld64_redmax(&sv2[0][0], ts);
        tmem_wait_ld();
        if (masked) {
          // The fused reduction is an UPPER bound: on a masked tile it also
          // sees causality-hidden columns, and ex2.approx.f16x2 underflows
          // to EXACT ZERO below 2^-24 - an inflated max would zero every
          // probability of the tile (tc9 law, dv_b2a40f651e23). Masked
          // tiles take the true max over the visible columns.
          const int jhi_m = min(causal ? vis + 1 : INT32_MAX, kvlen);
          const int jlo_m = windowed ? max(0, vis - window_left) : 0;
          float pm = NEG_BIG;
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            #pragma unroll
            for (int x = 0; x < 32; ++x) {
              const int j = n0 + cc * 32 + x;
              pm = fmaxf(pm, (j >= jlo_m && j < jhi_m) ? sv2[cc][x] : NEG_BIG);
            }
          }
          tm = pm;
        }
        // cross-warp exchange: two bar_softmax rounds on the shared maxbuf
        *mx = tm;
        bar_softmax();
        tm = fmaxf(tm, *mxp);
        bar_softmax();
        // lagged running max (2^8 threshold): alpha == 1.0f exactly on
        // almost every tile, so the O correction is skipped
        float nr = fmaxf(mr, tm);
        if ((nr - mr) * sc2 <= LAG_MAX_LOG2) nr = mr;
        const float nm2 = nr * sc2;
        const float alpha = (nr == mr) ? 1.0f : ex2_approx(fmaf(mr, sc2, -nm2));
        mr = nr;
        const bool need_corr = (i > G0) && __any_sync(0xffffffffu, alpha != 1.0f);
        // exp / pack / row sum over this warp's 64 columns (MUST precede the
        // O correction: sv2 is its scratch and still holds this tile's P).
        float pc = 0.0f;
        uint32_t pr[32];
        if (!masked) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            #pragma unroll
            for (int x2 = 0; x2 < 16; ++x2) {
              const uint32_t pf = ex2_f16x2(cvt_f16x2(
                  fmaf(sv2[cc][x2 * 2], sc2, -nm2),
                  fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2)));
              const float p0 = f16lo_f32(pf), p1 = f16hi_f32(pf);
              pc += p0 + p1;
              pr[cc * 16 + x2] = pack_bf16(p0, p1);
            }
          }
        } else {
          const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
          const int jlo = windowed ? max(0, vis - window_left) : 0;
          if (!windowed) {
            // prefix mask: unconditional ex2 on a -200 sentinel argument,
            // whose f16x2 exp underflows to exact zero (tc2's promoted form)
            const int hi = jhi - n0;
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              #pragma unroll
              for (int x2 = 0; x2 < 16; ++x2) {
                const int x = cc * 32 + x2 * 2;
                const float a0 = (x < hi)
                    ? fmaf(sv2[cc][x2 * 2], sc2, -nm2) : -200.0f;
                const float a1 = (x + 1 < hi)
                    ? fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2) : -200.0f;
                const uint32_t pf = ex2_f16x2(cvt_f16x2(a0, a1));
                const float p0 = f16lo_f32(pf), p1 = f16hi_f32(pf);
                pc += p0 + p1;
                pr[cc * 16 + x2] = pack_bf16(p0, p1);
              }
            }
          } else {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              #pragma unroll
              for (int x2 = 0; x2 < 16; ++x2) {
                const int j0 = n0 + cc * 32 + x2 * 2;
                const float a0 = (j0 >= jlo && j0 < jhi)
                    ? fmaf(sv2[cc][x2 * 2], sc2, -nm2) : -200.0f;
                const float a1 = (j0 + 1 >= jlo && j0 + 1 < jhi)
                    ? fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2) : -200.0f;
                const uint32_t pf = ex2_f16x2(cvt_f16x2(a0, a1));
                const float p0 = f16lo_f32(pf), p1 = f16hi_f32(pf);
                pc += p0 + p1;
                pr[cc * 16 + x2] = pack_bf16(p0, p1);
              }
            }
          }
        }
        // P image: 64 bf16 pairs into columns half*32 .. half*32+31 of the
        // slot (both halves stored inside the loops keeps the packed
        // registers from spanning the whole exp - tc2p's promoted form).
        {
          const uint32_t tp = tmem + TCOL_S(s3) + half * 32 + lb;
          tmem_st16bf(tp, pr);
          tmem_st16bf(tp + 16, pr + 16);
        }
        tmem_wait_st();
        d = d * alpha + pc;
        // O correction (skipped on the item's first tile and, via the lagged
        // max, on almost every later tile). PV(i-1) wrote this CTA's O half;
        // this warp rescales only its own 64 output dims.
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
        fence_before_sync();
        bar_softmax();
        if (warp == 0 && lane == 0) mbar_arrive(bar(ctrl, PF0 + s3));
      }

      // ---- epilogue (per item, direct global stores) ----
      const long long il = G0 + it.T - 1;
      mbwait(bar(ctrl, PV0 + (int)(il % S_STAGES)),
             (int)((il / S_STAGES) & 1), dbg, DBG_TO_PV, whint);
      fence_after_sync();
      // combine the two column halves' row sums through the max buffer
      float* rs = maxbuf;
      bar_softmax();
      rs[warp * 32 + lane] = d;
      bar_softmax();
      const float dtot = rs[w3 * 32 + lane] + rs[(w3 + 4) * 32 + lane];
      bar_softmax();
      float lse2 = mr * sc2 + log2f(dtot);
      float extra = 1.0f;
      if (has_sink) {
        float t = fmaxf(lse2, sink2) + log2f(1.0f + exp2f(-fabsf(lse2 - sink2)));
        extra = exp2f(lse2 - t);
        lse2 = t;
      }
      const float inv = dtot > 0.0f ? 1.0f / dtot : 0.0f;
      const float g = bmm2_scale * inv * extra;
      const uint32_t to = tmem + TCOL_O + half * 64 + lb;
      __nv_bfloat16* orow =
          outp + static_cast<int64_t>(m0 + row) * o_stride;
      #pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        tmem_ld32(sv2[cc], to + cc * 32);
        tmem_wait_ld();
        if (active) {
          __align__(16) uint32_t pk[16];
          #pragma unroll
          for (int x2 = 0; x2 < 16; ++x2)
            pk[x2] = pack_bf16(sv2[cc][x2 * 2] * g, sv2[cc][x2 * 2 + 1] * g);
          // 256-bit stores: 32 dims bf16 = 64 B per cc = 2 x v8
          uint32_t* dstv = reinterpret_cast<uint32_t*>(orow + half * 64 + cc * 32);
          asm volatile(
              "st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::
              "l"(dstv), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]),
              "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
          asm volatile(
              "st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::
              "l"(dstv + 8), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]),
              "r"(pk[11]), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]),
              "r"(pk[15]) : "memory");
        }
      }
      // O drain: all eight warps' O reads are done -> release the single O
      // accumulator to the NEXT item's first PV (ODR, count 1, local).
      fence_before_sync();
      bar_softmax();
      if (warp == 0 && lane == 0) mbar_arrive(bar(ctrl, ODR));
      if (active && has_lse && half == 0) {
        lsep[static_cast<int64_t>(m0 + row) * heads] = lse2;
      }
      Gbase += it.T; ++fitem;
    }
  }

#ifdef TC10_TRIAGE
  // Exit-time perm re-dump + E9e footprint scan: per CTA the lowest/highest
  // ctrl offset whose pattern word was mutated (canary slots excluded), a
  // 32-word raw dump of the stomp zone from CTA 0, the allocation-top canary
  // and 16 words below it, and the CTA->SM id map.
  if (dbg && do_sort) {
    if (tid < 120 && tid < tiles_ub) dbg[180 + tid] = perm_read(perm, tid);
    if (tid == 0) {
      volatile int* cw = reinterpret_cast<volatile int*>(ctrl);
      dbg[800 + cid] = perm_read(perm, 0);
      dbg[1000 + cid] = (tiles_ub > 1) ? perm_read(perm, 1) : -1;
      dbg[1200 + cid] = cw[(PERMTBL_OFF - 128) >> 2];
      dbg[1400 + cid] = cw[(PERMTBL_OFF + TC10_TILES_CAP * 4 + 64) >> 2];
      dbg[1860 + cid] = smid_();
      int lo = INT32_MAX, hi = -1;
      for (int w = HIST_OFF; w < PERMTBL_OFF; w += 4) {
        if (w == PERMTBL_OFF - 128) continue;      // canary_left slot
        if (cw[w >> 2] != static_cast<int>(0x5A5A0000u |
                                           static_cast<unsigned>(w >> 2))) {
          if (w < lo) lo = w;
          if (w > hi) hi = w;
        }
      }
      for (int w = PERMTBL_OFF + TC10_TILES_CAP * 4; w < CTRL_BYTES - 16;
           w += 4) {
        if (w == PERMTBL_OFF + TC10_TILES_CAP * 4 + 64) continue;  // canary_r
        if (cw[w >> 2] != static_cast<int>(0x5A5A0000u |
                                           static_cast<unsigned>(w >> 2))) {
          if (w < lo) lo = w;
          if (w > hi) hi = w;
        }
      }
      dbg[1560 + cid] = (lo == INT32_MAX) ? -1 : lo;
      dbg[1712 + cid] = hi;
      if (cid == 0) {
        // E9e instrumentation-bug fixes: (1) the zone base missed CTRL_OFF
        // (read Q data at smem+2184 instead of the table zone); (2) the
        // 1900-range slots collided with dbg[1860+cid] smids for cid>=40 -
        // zone/top/near-top moved to 2100+ (script dbg tensor is 4096).
        const volatile int* zl = cw + ((PERMTBL_OFF - 128) >> 2);
        for (int k = 0; k < 48; ++k) dbg[2100 + k] = zl[k];
        dbg[2160] = cw[(CTRL_BYTES - 8) >> 2];
        for (int k = 0; k < 16; ++k)
          dbg[2170 + k] = cw[(CTRL_BYTES - 72 + 4 * k) >> 2];
      }
      dbg[300] = 1;
    }
  }
#endif
  // ---- teardown ----
  __syncthreads();
  if (warp == 8)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 ::"r"(tmem), "r"(512) : "memory");
}

}  // namespace mla_tc10

// ---------------------------------------------------------------------------
// Host: rank-4 direct maps + plain persistent launch (no cluster attrs).
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

namespace mla_tc10 { int* g_perm_scratch = nullptr; }
extern "C" void* mla_tc10_gperm() { return mla_tc10::g_perm_scratch; }

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
  (void)kws; (void)vws; (void)kvblks; (void)max_kv_len;
  (void)dbg_mode; (void)pf_mode; (void)lag_max; (void)direct;
  (void)box_tok; (void)opts;
  using namespace mla_tc10;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  static bool attr_set = false;
  static int g_sms = 0;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(mla_tc10_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return static_cast<int>(e);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess ||
        g_sms < 1) {
      g_sms = 148;
    }
    // Per-CTA global perm scratch (TC10_GPERM fallback / triage): allocated
    // on the FIRST plain launch, which always precedes any graph capture.
    if (mla_tc10::g_perm_scratch == nullptr) {
      if (cudaMalloc(&mla_tc10::g_perm_scratch,
                     (size_t)g_sms * TC10_TILES_CAP * sizeof(int))
          != cudaSuccess) {
        mla_tc10::g_perm_scratch = nullptr;
        return static_cast<int>(cudaGetLastError());
      }
    }
    attr_set = true;
  }
  int tiles_ub = total_q / BM + batch;
  if (tiles_ub <= 0) return static_cast<int>(cudaGetLastError());
  const long long total_items = (long long)tiles_ub * heads;
  int ncta = g_sms;
  if (total_items < ncta) ncta = (int)total_items;   // never launch idle CTAs

  CUtensorMap krmap, vrmap, qrmap;
  int rc;
  // K: full BN=128-token x 192-dim boxes (3 atoms)
  if ((rc = cached4(&krmap, k, (uint64_t)total_kv, (uint64_t)kv_heads, 3, 128, 3)))
    return 5000 + rc;
  // V: full 128-token x 128-dim boxes (2 atoms)
  if ((rc = cached4(&vrmap, v, (uint64_t)total_kv, (uint64_t)kv_heads, 2, 128, 2)))
    return 6000 + rc;
  // Q: one 128-row x 192-dim box per item
  if ((rc = cached4(&qrmap, q, (uint64_t)total_q, (uint64_t)heads, 3, 128, 3)))
    return 7000 + rc;

  mla_tc10_kernel<<<ncta, THREADS, SMEM_BYTES, s>>>(
      krmap, vrmap, qrmap, static_cast<__nv_bfloat16*>(out),
      static_cast<float*>(lse), static_cast<const float*>(sinks),
      static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
      total_q, batch, heads, kv_heads, window_left, causal, has_lse, has_sink,
      bmm1_scale, bmm2_scale,
      reinterpret_cast<unsigned long long*>(dbg),
      mla_tc10::g_perm_scratch);
  return static_cast<int>(cudaGetLastError());
}
