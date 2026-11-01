// kernel_tc7.cu - TWO-CTA CLUSTER tcgen05 kernel (arm-b join import).
// M256N128K16 cta_group::2, BN=128, BM=256 split as 128 q-rows per CTA;
// per-CTA TMEM = 3 S/P slots x 128 cols + O 128 = 512 exactly; measured
// (arm-b jobs dv_9ef28d29d53c/dv_1f1369184437): pair MMA at 100% of the
// 8192 flop/cyc/SM peak, full per-tile stream 1280.3 cyc vs tc2's 1643.2.
// PROVENANCE: manual join 20261022, arm-b best/correct-124
// (.join-exchange/20261022/b/candidate/kernel_tc7.cu, source_commit
// 6c02725). Copied UNMODIFIED below this banner; promoted build config =
// arm-b's (-DTC7_UNBOUNDED -DTC7_PF_RELAY -DTC7_RELAX_ARRIVE). Per
// campaign law, arm-b timings are hypothesis-grade for this tree: all
// adoption decisions come from same-job probe44+ re-measurement.
// mla_expanded_bf16 - tcgen05 TWO-CTA CLUSTER kernel (tc7), step 1.
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
// ctrl: 16 mbarriers (128 B) + the pair TMEM holder (8 B) + the 8x32 cross-warp
// row-max exchange buffer (1024 B, reused for the epilogue's partial row sums)
#ifdef TC7_XMAX1
constexpr int CTRL_BASE = 2560;   // two rotating 8x32 max buffers
#else
constexpr int CTRL_BASE = 1280;
#endif
#ifdef TC7_LPTG
// Static global-LPT tables (probe57): ord[100] descending-T pair-tile order,
// tids[100]/ths[100] scratch, hist[132] counting sort, nreal at [432].
// 433 ints = 1732B -> 1792B region. Built by tid==0, read after syncthreads.
constexpr int LPTG_OFF = CTRL_BASE;
constexpr int LPTG_NREAL = 432;
constexpr int CTRL_BYTES = CTRL_BASE + 1792;
#else
constexpr int CTRL_BYTES = CTRL_BASE;
#endif
constexpr int SMEM_BYTES = CTRL_OFF + CTRL_BYTES;             // 173312/175104
constexpr int STAGE_BYTES = SUB * DV * 2;                     // 32768 epilogue stage

// TMEM column map: S/P slot s at 128s (f32 S in all 128 columns, packed bf16 P
// in the first 64), O at 384.
DIN int TCOL_S(int s) { return s * 128; }
constexpr int TCOL_O = 384;

enum { BF0 = 0, BR0 = 3, SF0 = 6, PF0 = 9, PV0 = 12, QF = 15,
#ifdef TC7_PF_RELAY
       PL0 = 16, NBARS = 19 };   // PL: peer-local "P stored", relayed by warp10
#else
       NBARS = 16 };
#endif
// dbg slots for bounded-wait timeouts (step 1 only)
enum { DBG_TO_BF = 16, DBG_TO_BR, DBG_TO_SF, DBG_TO_PF, DBG_TO_PV, DBG_TO_QF,
       DBG_TILES = 24, DBG_CTAS = 25 };

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
#if defined(TC7_EX2F16) || defined(TC7_EX2F16_UNMASK) || defined(TC7_EX2F16_PREFIX)
// E2 (ported from kernel_tc2.cu v14, probe27/28-validated, production
// numerics class since v14): one ex2.approx.f16x2 per score PAIR (halves
// MUFU count; f16 mantissa 4x finer than the bf16 variant that failed the
// checker - E1 lesson). NOTE: no .ftz on the f16x2 form (ptxas rejects);
// underflow to zero is implicit (the -200 sentinel and deep tails are
// unrepresentable in f16).
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
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const unsigned rank = cluster_rank();
#ifdef TC7_LPTG
  // ---- static global LPT (probe57): grid = (2*heads, pairs_ub); rank r =
  // blockIdx.y indexes a descending-T pair-tile table every CTA rebuilds
  // identically (deterministic counting sort, stable tile_id-asc ties), so
  // both CTAs of a pair map to the same tile and the joint-return symmetry
  // of the b<0 / T<=0 paths is preserved. Kills the late-heavy-start
  // residual of the tiles-fastest-within-head dispatch order (the static
  // analogue of tc2p's measured 0.5*max_unit residual, probe54/56).
  int* const lptg_ord = reinterpret_cast<int*>(ctrl + LPTG_OFF);
  if (tid == 0) {
    int* const tids = lptg_ord + 100;
    int* const ths = tids + 100;
    int* const hist = ths + 100;
    for (int t = 0; t <= 128; ++t) hist[t] = 0;
    int n = 0;
    int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
    for (int i = 0; i < batch; ++i) {
      const int nq = cum_q[i + 1], nkv = cum_kv[i + 1];
      const int len_q = nq - prev_q, len_kv = nkv - prev_kv;
      const int tiles = (len_q + BMP - 1) / BMP;
      const int kv_off = len_kv - len_q;
      for (int local = 0; local < tiles && n < 100; ++local) {
        const int pm0 = (tiles - 1 - local) * BMP;  // in-seq LPT position
        const int nmax = causal ? min(len_kv, kv_off + pm0 + BMP) : len_kv;
        const int T = (nmax + BN - 1) / BN;         // == the mapping's Traw
        if (T > 0 && T <= 128) {
          tids[n] = acc + local; ths[n] = T; hist[T]++; n++;
        }
      }
      acc += tiles; prev_q = nq; prev_kv = nkv;
    }
    int p = 0;
    for (int t = 128; t >= 0; --t) { const int c = hist[t]; hist[t] = p; p += c; }
    for (int j = 0; j < n; ++j) lptg_ord[hist[ths[j]]++] = tids[j];
    lptg_ord[LPTG_NREAL] = n;
  }
  __syncthreads();
  const int pair_id = (blockIdx.y < lptg_ord[LPTG_NREAL])
                        ? lptg_ord[blockIdx.y] : 0x7FFFFFFF;
  const int h = blockIdx.x >> 1;
#else
  const int pair_id = blockIdx.x >> 1;
  const int h = blockIdx.y;
#endif
  // tc2 measured the suspend-hint form better than a pure spin (its opts bit3) and
  // ships 0x989680. tc7 waits more often per tile than tc2 does - BR and PF are
  // new - so wake latency may cost more here; TC7_SPIN tests hint 0.
#ifdef TC7_SPIN
  const uint32_t whint = 0u;
#else
  const uint32_t whint = 0x989680u;
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
  // Both CTAs of a pair compute the same pm0/qlen/kvlen, so they return together:
  // a cluster kernel must never let one CTA leave the others at a cluster scope.
  if (b < 0) return;

  const int m0 = pm0 + (int)rank * SUB;         // this CTA's first q row
  const int kv_off = kvlen - qlen;
  const bool windowed = window_left >= 0;
  // T is a PAIR quantity: both CTAs must run the same number of tiles or the
  // cluster-scope barriers would deadlock. Rows past qlen produce finite garbage
  // that is row-local through softmax/PV and never stored (the same contract
  // tc2's qrows-skip path relies on).
  int nmax_p = causal ? min(kvlen, kv_off + pm0 + BMP) : kvlen;
  nmax_p = min(nmax_p, kvlen);
  const int Traw = (nmax_p + BN - 1) / BN;
  // Staged diagnosis: 4 runs exactly one tile end to end (QK, softmax, PV,
  // epilogue) and 5 runs two, which is the smallest T that exercises the ring
  // wraparound and the deferred PV(i-1). Both CTAs clamp identically, so the
  // cluster-scope barriers still see both of them.
#if defined(TC7_STAGE) && TC7_STAGE == 4
  const int T = Traw < 1 ? Traw : 1;
#elif defined(TC7_STAGE) && TC7_STAGE == 5
  const int T = Traw < 2 ? Traw : 2;
#else
  const int T = Traw;
#endif
  // per-CTA masking: the first tile whose columns are all unmasked for THIS CTA
  int nsafe = causal ? min(kvlen, kv_off + m0 + 1) : kvlen;
  if (windowed) nsafe = 0;
  const int Tfull = nsafe / BN;
  if (T <= 0) return;

  const int group_div = heads / kv_heads;
  const int h_kv = h / group_div;
  __nv_bfloat16* outp = out + (static_cast<int64_t>(q_base) * heads + h) * DV;
  float* lsep = has_lse ? lse + static_cast<int64_t>(q_base) * heads + h : nullptr;
  const int64_t o_stride = static_cast<int64_t>(heads) * DV;
  const float sc2 = bmm1_scale * LOG2E;
  const float sink2 = has_sink ? sinks[h] * LOG2E : 0.0f;
  const int tok0 = kv_base;

  if (dbg && tid == 0) atomicAdd(dbg + DBG_CTAS, 1ull);
#if defined(TC7_STAGE) && TC7_STAGE == 0
  // stage 0 tests only that the cluster launch happens at all: 2 CTAs per
  // cluster, 173312 B of dynamic smem each, 320 threads, and a dbg write that
  // proves the kernel body ran. If this hangs, nothing below matters.
  if (dbg && tid == 0) atomicMax(dbg + 40, 0ull);
  return;
#endif

  // ---- control init (parallel: one thread per barrier) ----
  if (tid < NBARS) {
    int cnt = 1;
    const int i = tid;
    if (i >= BR0 && i < BR0 + KV_STAGES) cnt = 2;       // both CTAs' warp 8
    else if (i >= PF0 && i < PF0 + S_STAGES) cnt = 2;   // both CTAs' softmax
#ifdef TC7_PF_RELAY
    else if (i >= PL0 && i < PL0 + S_STAGES) cnt = 1;   // this CTA's softmax only
#endif
    mbar_init(bar(ctrl, i), cnt);
  }
  asm volatile("fence.mbarrier_init.release.cluster;");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
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

  // ---- Q: one 128-row x 192-dim box per CTA (49152 B) ----
  if (tid == 0) {
    mbar_expect_tx(bar(ctrl, QF), Q_BYTES);
    tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF), 0, q_base + m0, 0, h);
  }
  // ---- KV prologue: each CTA loads its own halves of tiles 0..min(2,T-1) ----
  //   K half: 64 kv tokens x 192 dims at token offset i*BN + rank*BNH
  //   V half: 128 kv tokens x 64 dims, atom coordinate = rank
  if (warp == 9 && elect_one()) {
    const int pre = (T < KV_STAGES) ? T : KV_STAGES;
    for (int j = 0; j < pre; ++j) {
      mbar_expect_tx(bar(ctrl, BF0 + j), K_BYTES + V_BYTES);
      tma_load_4d(&krmap, sm32_(sk) + j * K_BYTES, bar(ctrl, BF0 + j), 0,
                  tok0 + j * BN + (int)rank * BNH, 0, h_kv);
      tma_load_4d(&vrmap, sm32_(sv) + j * V_BYTES, bar(ctrl, BF0 + j), 0,
                  tok0 + j * BN, (int)rank, h_kv);
    }
  }

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

  if (warp == 9) {
    // ---------------- TMA LOADER ----------------
    if (elect_one()) {
      for (int j = KV_STAGES; j < T; ++j) {
        const int s = j % KV_STAGES;
        // PV(j-3) retired: that frees stage s (K read by QK, V by PV)
        mbwait(bar(ctrl, PV0 + s), (j / KV_STAGES - 1) & 1, dbg, DBG_TO_PV, whint);
        mbar_expect_tx(bar(ctrl, BF0 + s), K_BYTES + V_BYTES);
        tma_load_4d(&krmap, sm32_(sk) + s * K_BYTES, bar(ctrl, BF0 + s), 0,
                    tok0 + j * BN + (int)rank * BNH, 0, h_kv);
        tma_load_4d(&vrmap, sm32_(sv) + s * V_BYTES, bar(ctrl, BF0 + s), 0,
                    tok0 + j * BN, (int)rank, h_kv);
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
        for (int i = 0; i < T; ++i) {
          mbwait(bar(ctrl, BF0 + i % KV_STAGES), (i / KV_STAGES) & 1, dbg,
                 DBG_TO_BF, whint);
          // tile 0's BR arrive also certifies this CTA's Q, which the leader's
          // pair QK reads out of THIS smem
          if (i == 0) {
            mbwait(bar(ctrl, QF), 0, dbg, DBG_TO_QF, whint);
            fence_after_sync();
          }
          mbar_arrive_remote(lbr + (i % KV_STAGES) * 8);
        }
      } else {
        const uint32_t qb0 = sm32_(sq);
        for (int i = 0; i < T; ++i) {
          const int s = i % KV_STAGES;
          const int par = (i / KV_STAGES) & 1;
          mbwait(bar(ctrl, BF0 + s), par, dbg, DBG_TO_BF, whint);
          if (i == 0) {
            mbwait(bar(ctrl, QF), 0, dbg, DBG_TO_QF, whint);
            fence_after_sync();
          }
          mbar_arrive(bar(ctrl, BR0 + s));
#ifndef TC7_NO_BR
          mbwait(bar(ctrl, BR0 + s), par, dbg, DBG_TO_BR, whint);
#endif
          fence_after_sync();
          // S/P slot s is free once PV(i-3) retired (same barrier, same parity as
          // the loader's stage wait)
          if (i >= KV_STAGES) {
            mbwait(bar(ctrl, PV0 + s), (i / KV_STAGES - 1) & 1, dbg, DBG_TO_PV,
                   whint);
            fence_after_sync();
          }
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
          // ---- PV(i-1): 8 k-steps, A=P from TMEM, B=V half from smem ----
          const int ip = i - 1;
          if (ip >= 0) {
            const int sp = ip % S_STAGES;
#ifndef TC7_NO_PF
            mbwait(bar(ctrl, PF0 + sp), (ip / S_STAGES) & 1, dbg, DBG_TO_PF,
                   whint);
#endif
            fence_after_sync();
            const uint32_t tp = tmem + TCOL_S(sp);
            const uint64_t db = make_desc(sm32_(sv) + sp * V_BYTES, 0, vsbo);
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk)
              mma_ts2(tmem + TCOL_O, tp + kk * 8, db + kk * vstep16, IDESC_PV,
                      (ip > 0) | (kk > 0));
            mma_commit_mc(bar(ctrl, PV0 + sp));
          }
          if (dbg) atomicAdd(dbg + DBG_TILES, 1ull);
        }
        // ---- tail: PV(T-1) ----
        {
          const int il = T - 1;
          const int sl = il % S_STAGES;
#ifndef TC7_NO_PF
          mbwait(bar(ctrl, PF0 + sl), (il / S_STAGES) & 1, dbg, DBG_TO_PF, whint);
#endif
          fence_after_sync();
          const uint32_t tp = tmem + TCOL_S(sl);
          const uint64_t db = make_desc(sm32_(sv) + sl * V_BYTES, 0, vsbo);
          #pragma unroll
          for (int kk = 0; kk < 8; ++kk)
            mma_ts2(tmem + TCOL_O, tp + kk * 8, db + kk * vstep16, IDESC_PV,
                    (il > 0) | (kk > 0));
          mma_commit_mc(bar(ctrl, PV0 + sl));
        }
      }
    }
#ifdef TC7_PF_RELAY
  } else if (warp == 10) {
    // ---------------- PEER PF RELAY (warp 10, peer CTA only) ----------------
    // Nothing else in the peer CTA issues MMAs, so this warp would otherwise be
    // idle. It waits the peer's own "P stored" barrier and forwards it to the
    // leader, which is the only place a remote arrive is affordable.
    if (rank != 0 && elect_one()) {
      const uint32_t lpf = mapa_(bar(ctrl, PF0), 0);
      for (int i = 0; i < T; ++i) {
        const int s = i % S_STAGES;
        mbwait(bar(ctrl, PL0 + s), (i / S_STAGES) & 1, dbg, DBG_TO_PF, whint);
        mbar_arrive_remote(lpf + s * 8);
      }
    }
#endif
  } else if (warp < SOFT_WARPS) {
    // ---------------- SOFTMAX (column-split over 8 warps) ----------------
    const int w3 = warp & 3;                 // row group: rows 32*w3 .. +31
    const int half = warp >> 2;              // column half: kv 64*half .. +63
    const int row = w3 * 32 + lane;          // row within this CTA's 128
    const int grow = m0 + row;               // global q row
    const uint32_t lb = static_cast<uint32_t>(w3 * 32) << 16;
    const int vis = kv_off + grow;
    const bool active = (grow < qlen);
    float mr = NEG_BIG, d = 0.0f;
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

    for (int i = 0; i < T; ++i) {
      const int s = i % S_STAGES;
      const uint32_t ts = tmem + TCOL_S(s) + half * 64 + lb;
      mbwait(bar(ctrl, SF0 + s), (i / S_STAGES) & 1, dbg, DBG_TO_SF, whint);
      fence_after_sync();
      float* mx = maxbuf + TC7_MXOFF(i) + warp * 32 + lane;
      float* mxp = maxbuf + TC7_MXOFF(i) + (warp ^ 4) * 32 + lane;
      const int n0 = i * BN + half * 64;      // this warp's first kv column
      const bool masked = i >= Tfull;

      // hardware row max over this warp's 64 columns, then the cross-warp
      // exchange that makes it a row max over all 128
#ifdef TC7_MASKMAX
      float tm;
      if (masked) {
        // Register-masked max (tc2-style j-boundary filter) for MASKED
        // tiles: E2-safety law (probe48 / results/tc7_e2_underflow_law.md)
        // - the hardware redmax covers masked columns, so valid-column
        // exp args could sit << -24 where ex2.approx.f16x2 underflows to
        // exact 0 (row-sum -> 0 -> row zeroed). With a masked max, the
        // best valid column always has arg == 0 -> p == 1 -> dtot >= 1.
        tmem_ld32(sv2[0], ts);
        tmem_ld32(sv2[1], ts + 32);
        tmem_wait_ld();
        const int jhi_m = min(causal ? vis + 1 : INT32_MAX, kvlen);
        const int jlo_m = windowed ? max(0, vis - window_left) : 0;
        tm = NEG_BIG;
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          #pragma unroll
          for (int x = 0; x < 32; ++x) {
            const int j = n0 + cc * 32 + x;
            tm = fmaxf(tm, (j >= jlo_m && j < jhi_m) ? sv2[cc][x] : NEG_BIG);
          }
        }
      } else {
        tm = tmem_ld64_redmax(&sv2[0][0], ts);
        tmem_wait_ld();
      }
#else
      float tm = tmem_ld64_redmax(&sv2[0][0], ts);
      tmem_wait_ld();
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

      const bool need_corr = (i > 0) && __any_sync(0xffffffffu, alpha != 1.0f);

      // exp / pack / row sum over this warp's 64 columns. This MUST come before
      // the O correction: the correction reuses sv2 as its O scratch, and sv2
      // still holds this tile's scores.
      float pc = 0.0f;
      uint32_t pr[32];
#ifdef TC7_EARLY_PST
      // Store each packed P half inside the exp loop, as kernel_tc2p.cu has
      // always done; port of kernel_tc9.cu's TC9_EARLY_PST (see the comment
      // there: deferring both stores to the end keeps all 32 packed registers
      // live across the second half and serialises the two TMEM stores behind
      // the whole exp - tc9's sm_exp measured 824-883 cyc/tile against
      // tc2p's 581-712 for the same 16384 scores/CTA and a textually
      // identical exp, dv_8bac3774fc32 vs dv_e8daa299ce32). Reorder-only:
      // identical values, identical P image.
      const uint32_t tp = tmem + TCOL_S(s) + half * 32 + lb;
#define T7_STORE_HALF(cc) tmem_st16bf(tp + (cc) * 16, pr + (cc) * 16)
#else
#define T7_STORE_HALF(cc) do {} while (0)
#endif
      if (!masked) {
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          #pragma unroll
          for (int x2 = 0; x2 < 16; ++x2) {
#if defined(TC7_EX2F16) || defined(TC7_EX2F16_UNMASK)
            const uint32_t pf = ex2_f16x2(cvt_f16x2(
                fmaf(sv2[cc][x2 * 2], sc2, -nm2),
                fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2)));
            const float p0 = f16lo_f32(pf), p1 = f16hi_f32(pf);
#else
            float p0 = ex2_approx(fmaf(sv2[cc][x2 * 2], sc2, -nm2));
            float p1 = ex2_approx(fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2));
#endif
            pc += p0 + p1;
            pr[cc * 16 + x2] = pack_bf16(p0, p1);
          }
          T7_STORE_HALF(cc);
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
            #pragma unroll
            for (int x2 = 0; x2 < 16; ++x2) {
              const int x = cc * 32 + x2 * 2;
#if defined(TC7_EX2F16) || defined(TC7_EX2F16_PREFIX)
              const uint32_t pf = ex2_f16x2(cvt_f16x2(
                  fmaf(sv2[cc][x2 * 2], sc2, -nm2),
                  fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2)));
              const float e0 = f16lo_f32(pf), e1 = f16hi_f32(pf);
#else
              const float e0 = ex2_approx(fmaf(sv2[cc][x2 * 2], sc2, -nm2));
              const float e1 = ex2_approx(fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2));
#endif
              const float p0 = (x < hi) ? e0 : 0.0f;
              const float p1 = (x + 1 < hi) ? e1 : 0.0f;
              pc += p0 + p1;
              pr[cc * 16 + x2] = pack_bf16(p0, p1);
            }
            T7_STORE_HALF(cc);
          }
        } else {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            #pragma unroll
            for (int x2 = 0; x2 < 16; ++x2) {
#ifdef TC7_EX2F16
              // out-of-range -> -200 sentinel: ex2.f16x2 underflows to
              // exact zero (tc2-validated masked-E2 form)
              const int j0 = n0 + cc * 32 + x2 * 2;
              const float a0 = (j0 >= jlo && j0 < jhi)
                                   ? fmaf(sv2[cc][x2 * 2], sc2, -nm2) : -200.0f;
              const float a1 = (j0 + 1 >= jlo && j0 + 1 < jhi)
                                   ? fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2) : -200.0f;
              const uint32_t pf = ex2_f16x2(cvt_f16x2(a0, a1));
              const float p0 = f16lo_f32(pf), p1 = f16hi_f32(pf);
#else
              float p0, p1;
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
              pr[cc * 16 + x2] = pack_bf16(p0, p1);
            }
            T7_STORE_HALF(cc);
          }
        }
      }
      // P image: 64 bf16 pairs into columns half*32 .. half*32+31 of the slot.
      // Under TC7_EARLY_PST both halves are already stored and tp is declared
      // above the exp loops.
#ifndef TC7_EARLY_PST
      const uint32_t tp = tmem + TCOL_S(s) + half * 32 + lb;
      tmem_st16bf(tp, pr);
      tmem_st16bf(tp + 16, pr + 16);
#endif
      tmem_wait_st();
      d = d * alpha + pc;
#undef T7_STORE_HALF

      // O correction (skipped on tile 0 and, thanks to the lagged max, on almost
      // every later tile because alpha is then exactly 1.0f). PV(i-1) writes this
      // CTA's half of O, so it must have retired first. This warp rescales only
      // its own 64 output dims, so the two halves never touch the same columns.
      if (need_corr) {
        mbwait(bar(ctrl, PV0 + (i - 1) % S_STAGES), ((i - 1) / S_STAGES) & 1,
               dbg, DBG_TO_PV, whint);
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
      // PF merges tc2's PF and OC barriers: it is arrived only after P is in
      // TMEM AND the correction is done, which is exactly the pair of facts the
      // leader's issuer needs before PV(i) may accumulate into either CTA's O.
      fence_before_sync();
      bar_softmax();
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
    }

    // ---- epilogue ----
    // (1) the last PV must have retired; (2) combine the two halves' row sums;
    // (3) scale/pack into the staging image over the dead K stages and issue one
    // bulk tensor store per CTA, or scattered per-row stores at a boundary.
    const int il = T - 1;
    mbwait(bar(ctrl, PV0 + il % S_STAGES), (il / S_STAGES) & 1, dbg, DBG_TO_PV,
           whint);
    fence_after_sync();
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

    // Staging over the K stages, which are dead: every PV has retired and the
    // loader issued nothing past tile T-1. The rendezvous must be the 256-thread
    // named barrier, NOT __syncthreads() - warps 8 and 9 never enter this branch
    // and would leave it waiting forever.
    bar_softmax();
    const bool bulk_ok = (m0 + SUB <= qlen);
    // warp-uniform form of the per-row validity test, for the aligned TMEM loads
    const bool any_active = __any_sync(0xffffffffu, active);
    char* stg = smem + Q_BYTES;
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
        tma_store_4d(&ormap, sm32_(stg), 0, q_base + m0, 0, h);
        bulk_commit();
        bulk_wait_read0();
      }
    }
    if (active && has_lse) {
      // one write per row: the half-0 warps own the row sums
      if (half == 0) lsep[static_cast<int64_t>(m0 + row) * heads] = lse2;
    }
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
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(mla_tc7_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return static_cast<int>(e);
    attr_set = true;
  }
  int pairs_ub = total_q / BMP + batch;
  if (pairs_ub <= 0) return static_cast<int>(cudaGetLastError());

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

#ifdef TC7_LPTG
  // x-fastest dispatch = head-fastest within sorted rank r (grid.y): all
  // heads' heaviest pair tile start at t~0. Cluster pairs stay x-adjacent
  // (2*heads is even; __cluster_dims__(2,1,1) unchanged).
  dim3 grid(2 * heads, pairs_ub, 1);
#else
  dim3 grid(2 * pairs_ub, heads, 1);
#endif
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
