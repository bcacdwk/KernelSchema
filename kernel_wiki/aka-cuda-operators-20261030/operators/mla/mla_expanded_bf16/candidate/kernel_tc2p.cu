// mla_expanded_bf16 — tcgen05 persistent dual-subtile kernel (tc2p).
// Derived from kernel_tc2.cu (v1, official-proven): identical per-tile math
// and per-tile pipeline protocol; the CTA lifetime becomes a WORK LOOP over
// items (q-tile, head) fetched from a device counter.
//
// Why: probe4/5 measured ~12k cycles of fixed cost per CTA (prologue ~7.4k:
// init/fences/alloc/Q+KV cold-DRAM arrival; epilogue ~3.5-5k) which serial
// CTAs pay once per ITEM: f1 8.4 items/SM, d4 7.8, f0/c6/c7 2.6. Persistent
// CTAs pay it once per SM and overlap item k+1's Q refill with item k's
// softmax/epilogue. Estimated -5..-15% on multi-wave mid cases.
//
// Structure (352 threads, grid = min(#SM, items), dual-issuer + TMA-Q only,
// box8 only):
//   warps 0-3 softmax subtile 0, warps 4-7 subtile 1 (as tc2)
//   warp 8: s0 MMA issuer + Q refill issuer
//   warp 9: item fetcher/publisher + KV stage loader (flat global ring)
//   warp 10: s1 MMA issuer
// Ring/barrier phases run on GLOBAL counters across items:
//   g     = global KV stage index  (BF/BE: instance g%3, parity (g/3)&1)
//   C0/C1 = global per-subtile tile counters
//            (SF/PF/PV: instance C&1, parity (C/2)&1; OC: parity C&1)
//   f     = item index             (NEXT/ODR: parity f&1; QF: ping-pong
//            QF(f&1) parity (f/2)&1; CONS: completion #f)
// Item handshake: loader lane0 atomicAdd(counter) -> decode -> smem publish
// -> NEXT arrive; every consumer warp waits NEXT(f&1), copies geometry to
// registers, arrives CONS. publish(k) is GATED on CONS(k-1): this bounds the
// publisher to <=1 phase ahead of every consumer's NEXT wait (parity
// aliasing would deadlock otherwise) and keeps the 2-slot items[] ring from
// overwriting unconsumed geometry. Surplus/empty items (b<0 or T==0) are
// re-fetched silently; the final sentinel (valid=0) unwinds everyone.
// O reuse across items: softmax arrives ODR(s) after its epilogue O reads
// (once per item, also when Ts==0); issuers wait ODR(s) of item f-1 before
// the FIRST PV of item f (QK writes only the S region, needs no ODrain).
// Q refill: at item f start warp8 waits the previous item's last SF(0)/
// SF(1) commits (Q smem reads done - the SF(1) wait also fences warp10's
// QK stream: warp10 cannot commit item f's SF(1) before QF(f) fires), then
// issues the two 48KB Q boxes -> the transfer overlaps item f-1's softmax
// tail/epilogue. QF ping-pong (QF0/QF1) makes the fill wait protocol-safe:
// warp8 can never be two fills ahead of warp10's wait.
// BE stage release keys on the GLOBAL g (item f's i==0 releases item f-1's
// last stage), both issuers commit (cnt=2) exactly as v1 di-mode.
//
// Numerics: per-tile math is textually identical to kernel_tc2.cu v1.
// Host: static device counter with an in-kernel SELF-RESET (v24, probe81
// dv_705e95947c95): total increments per launch are exactly n_eff +
// gridDim.x (each CTA exits on its first failing fetch; invalid-decode
// fetches stay within the n_eff budget), so the lane whose atomicAdd
// returns the final w stores 0 for the next launch - graph-replay safe
// (no per-launch host state) and it removes the per-call 4-byte
// cudaMemsetAsync, whose serialized graph node cost every tc2p-family
// call 0.5-3.2us (cold A/B: median -1.55us, bit-equal 10/10, node count
// 2 -> 1). One-time cudaMalloc + synchronous zero run on the first plain
// launch, which always precedes any capture in solution.py.
// Build: nvcc -gencode=arch=compute_103a,code=sm_103a -lcuda.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define DIN __device__ __forceinline__

namespace mla_tc2p {

constexpr int DQK = 192;
constexpr int DV = 128;
constexpr int BM = 256;             // 2 subtiles x 128 rows
constexpr int SUB = 128;
constexpr int BN = 64;
constexpr int KV_STAGES = 3;
constexpr int THREADS = 352;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -1.0e30f;
constexpr float LAG_MAX_LOG2 = 8.0f;

constexpr int Q_BYTES = SUB * DQK * 2;          // 49152 per subtile region
constexpr int Q_TOTAL = BM * DQK * 2;           // 98304
constexpr int K_BYTES = BN * DQK * 2;           // 24576 per stage
constexpr int V_BYTES = BN * DV * 2;            // 16384 per stage
constexpr int KV_TOTAL = KV_STAGES * (K_BYTES + V_BYTES);   // 122880
constexpr int CTRL_OFF = Q_TOTAL + KV_TOTAL;    // 221184
// ctrl: NBARS mbarriers (26*8=208B) + tmem holder (8B at 208) + items ring
// (2 x 48B at 216). Region size 512B. Under TC2P_QSPLIT (experiment flag)
// NBARS = 28 and the items ring moves to 232 (28*8+8); the static_assert
// after the barrier enum pins the pairing. Next consumer ORD_OFF=512 is
// untouched either way (232+96 = 328 < 512).
#ifdef TC2P_QSPLIT
constexpr int ITEMS_OFF = 232;
#else
constexpr int ITEMS_OFF = 216;
#endif
// bit20 LPTG sort tables (loader-lane-private, at 512): ord[100] tile ids in
// descending-T order, tids[100]/ths[100] scratch, hist[257] counting sort.
// tiles_ub <= total_q/BM + batch <= 16384/256 + 32 = 96; T <= 16384/BN = 256.
constexpr int ORD_OFF = 512;
constexpr int TIDS_OFF = ORD_OFF + 400;
constexpr int THS_OFF = TIDS_OFF + 400;
constexpr int HIST_OFF = THS_OFF + 400;
constexpr int CTRL_BYTES = HIST_OFF + 260 * 4;   // 512+3*400+1040 = 2752
#ifdef TC2P_PREMETA
// cum_q/cum_kv smem mirrors (batch+1 int32 each; 64 entries cover the frozen
// batch<=32 domain with margin, and the loader guards batch>63 back to the
// global arrays). E5 bug record: the first PREMETA cut clamped the mirror
// fill at 32 lanes (entries 0..32), but batch==32 needs 33+1=34 entries -
// decode then read uninitialized smem, items decoded invalid and were
// silently skipped (d3 ran 3.5x "faster" with 6.3M wrong elements; the E5
// bit-equality gate caught it). Never clamp below batch+1.
constexpr int CUMQ_OFF = CTRL_BYTES;                // 2752
constexpr int CUMKV_OFF = CUMQ_OFF + 64 * 4;        // 3016
constexpr int CTRL_BYTES_E = CUMKV_OFF + 64 * 4;    // 3280
#else
constexpr int CTRL_BYTES_E = CTRL_BYTES;
#endif
constexpr int SMEM_BYTES = CTRL_OFF + ((CTRL_BYTES_E + 511) / 512) * 512;  // 3072

// TMEM column map (identical to tc2)
DIN int TCOL_S(int s, int st) { return s * 128 + st * 64; }
DIN int TCOL_O(int s) { return 256 + s * 128; }

// barrier indices
enum { BF0 = 0, BF1, BF2, BE0, BE1, BE2,
       SF00, SF01, SF10, SF11,
       PF00, PF01, PF10, PF11,
       PV00, PV01, PV10, PV11,   // BEPV[s][st]
       OC0, OC1, QF0, QF1,
#ifdef TC2P_QSPLIT
       // E38b (experiment flag, off by default; production enum unchanged):
       // per-subtile Q arrival barriers. Each issuer warp's MMA descriptor
       // reads ONLY its own 49KB half (qb = sq + (s1warp ? Q_BYTES : 0)),
       // so half-box arrivals let subtile-0's first QK start ~half the
       // cold-DRAM Q transfer earlier at every item boundary (E38a: is_qf
       // = 81% of f0's item chain, 61% f1, 74% f2, 45% f6). Parity mirrors
       // QF0/QF1 exactly: slot (f&1), phase (f>>1)&1.
       QF2, QF3,
#endif
       NEXT, CONS, ODR0, ODR1, NBARS };
#ifdef TC2P_QSPLIT
static_assert(NBARS == 28, "QSPLIT ctrl layout: ITEMS_OFF=232 assumes 28 barriers");
#else
static_assert(NBARS == 26, "ctrl layout: ITEMS_OFF=216 assumes 26 barriers");
#endif
#if defined(TC2P_QEARLY) && !defined(TC2P_QSPLIT)
#error "TC2P_QEARLY requires TC2P_QSPLIT (per-half arrival barriers)"
#endif
DIN uint32_t sm32_(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
DIN uint32_t bar(char* ctrl, int idx) { return sm32_(ctrl) + idx * 8; }

// ---------------------------------------------------------------------------
// PTX primitives (probe-verified forms, identical to kernel_tc2.cu / V0)
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
// box8 canonical: 8 requests/operand, 8-row groups (K stride 3072B, V 2048B)
DIN void load_kv_tile(const CUtensorMap& krmap, const CUtensorMap& vrmap,
                      uint32_t kd, uint32_t vd, uint32_t mbar, int tok,
                      int h_kv) {
#ifdef TC2P_BOX64
  // one full-tile request per operand (rows contiguous, atom planes at
  // 8192B; ~116B/cyc ingest vs ~15B/cyc for 3KB boxes - job8 probe)
  tma_load_4d(&krmap, kd, mbar, 0, tok, 0, h_kv);
  tma_load_4d(&vrmap, vd, mbar, 0, tok, 0, h_kv);
#else
  #pragma unroll
  for (int g = 0; g < 8; ++g) {
    tma_load_4d(&krmap, kd + g * 3072, mbar, 0, tok + g * 8, 0, h_kv);
    tma_load_4d(&vrmap, vd + g * 2048, mbar, 0, tok + g * 8, 0, h_kv);
  }
#endif
}
DIN void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               ::"r"(a), "r"(bytes) : "memory");
}
DIN void mbar_init(uint32_t a, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
}
DIN void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
DIN void mbar_wait_h(uint32_t a, int phase, uint32_t hint) {
  asm volatile(
      "{\n\t.reg .pred P;\n\tLABW%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, %2;\n\t"
      "@P bra.uni DONEW%=;\n\tbra.uni LABW%=;\n\tDONEW%=:\n\t}\n"
      ::"r"(a), "r"(phase), "r"(hint)
      : "memory");
}
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
// idesc: M128 N64 bf16 f32acc
constexpr uint32_t IDESC_QK64 = (uint32_t(128 >> 4) << 24) |
                                (uint32_t(64 >> 3) << 17) |
                                (1u << 10) | (1u << 7) | (1u << 4);
constexpr uint32_t IDESC_PV = (uint32_t(128 >> 4) << 24) |
                              (uint32_t(128 >> 3) << 17) |
                              (1u << 16) | (1u << 10) | (1u << 7) | (1u << 4);
DIN void mma_ss(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc, int accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::
          "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum));
}
DIN void mma_ts(uint32_t tmem_d, uint32_t tmem_a, uint64_t db, uint32_t idesc, int accum) {
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
DIN void tmem_ld32_nw(float* d, uint32_t ta) {
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
DIN void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
DIN void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
#ifdef TC2P_REDMAX
// 64-column load that also returns the row maximum over those 64 columns.
// Mechanism imported verbatim from kernel_tc9.cu (measured there: same rate
// as a plain x64 load, bit-exact against an FMNMX tree). The tc2p S slot is
// exactly 64 TMEM columns wide with the same 32x32b lane mapping, and
// sv2[2][32] is contiguous, so the fused form drops in without any layout
// change.
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
#endif
DIN void tmem_st32(uint32_t ta, const float* d) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::
          "r"(ta), "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "f"(d[4]),
          "f"(d[5]), "f"(d[6]), "f"(d[7]), "f"(d[8]), "f"(d[9]), "f"(d[10]),
          "f"(d[11]), "f"(d[12]), "f"(d[13]), "f"(d[14]), "f"(d[15]),
          "f"(d[16]), "f"(d[17]), "f"(d[18]), "f"(d[19]), "f"(d[20]),
          "f"(d[21]), "f"(d[22]), "f"(d[23]), "f"(d[24]), "f"(d[25]),
          "f"(d[26]), "f"(d[27]), "f"(d[28]), "f"(d[29]), "f"(d[30]),
          "f"(d[31]) : "memory");
}
DIN void tmem_st16bf(uint32_t ta, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::
          "r"(ta), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]),
          "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]),
          "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]) : "memory");
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
// Softmax exp variants (compile-time; probe26): the exp phase runs 64 fp32
// ex2 per thread per tile (~512 cyc MUFU-bound of sm_exp 880-1040). The
// packed variants halve the MUFU count. P storage stays bf16 in TMEM in
// BOTH variants (bf16x2 directly; f16x2 via unpack+repack) -> no IDESC or
// MMA-operand changes. Precision: dominant weights (arg near 0) keep full
// accuracy; tail weights take ~0.3-1% relative input-quantization error,
// well inside the 5% checker tolerance (numerics-gated in probe26).
#ifdef TC2_EX2BF16
DIN uint32_t cvt_bf16x2(float lo, float hi) {
  uint32_t y;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(y) : "f"(hi), "f"(lo));
  return y;
}
DIN uint32_t ex2_bf16x2(uint32_t x) {
  uint32_t y;
  asm("ex2.approx.ftz.bf16x2 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}
DIN float bf16lo_f32(uint32_t p) { return __uint_as_float(p << 16); }
DIN float bf16hi_f32(uint32_t p) { return __uint_as_float(p & 0xFFFF0000u); }
#endif
#ifdef TC2_EX2F16
DIN uint32_t cvt_f16x2(float lo, float hi) {
  uint32_t y;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(y) : "f"(hi), "f"(lo));
  return y;
}
DIN uint32_t ex2_f16x2(uint32_t x) {
  uint32_t y;
  // NOTE: no .ftz on the f16x2 form (ptxas: "Illegal modifier '.ftz'" -
  // probe27 dv_4d4953ff1098 build gate); underflow to zero is implicit
  // (2^-200 sentinel and deep tails are unrepresentable in f16).
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

// ---------------------------------------------------------------------------
// Item geometry (published by the loader warp through the items[] ring)
// ---------------------------------------------------------------------------
struct ItemGeom {
  int valid;
  int q_base;     // absolute q row base of the batch
  int tok0;       // absolute kv token base of the batch
  int m0;         // q-tile start within the batch (LPT-ordered)
  int qlen;
  int kvlen;
  int Ts0, Tf0;   // subtile-0 kv tile count / full (unmasked) count
  int Ts1, Tf1;
  int T;          // max(Ts0, Ts1)
  int h;          // head index
};

// Geometry for an explicit (tile_id, h) pair. Enumeration-independent: the
// bit20 LPTG global-sort order feeds different w->(tile,h) mappings through
// here while the per-item math (and therefore the output bytes) stays
// identical to the tiles-fastest-within-head default.
DIN void decode_item_th(const int32_t* __restrict__ cum_q,
                        const int32_t* __restrict__ cum_kv, int batch,
                        int tile_id, int h, int causal, int window_left,
                        int opts, ItemGeom& it) {
  it.h = h;
  it.valid = 0;
  it.Ts0 = 0; it.Ts1 = 0; it.T = 0; it.Tf0 = 0; it.Tf1 = 0;
  it.q_base = 0; it.tok0 = 0; it.m0 = 0; it.qlen = 0; it.kvlen = 0;
  int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
  for (int i = 0; i < batch; ++i) {
    int nq = cum_q[i + 1], nkv = cum_kv[i + 1];
    int len_q = nq - prev_q;
    int tiles = (len_q + BM - 1) / BM;
    if (tile_id < acc + tiles) {
      const int local = tile_id - acc;
      it.valid = 1;
      it.m0 = ((opts & 2) ? (tiles - 1 - local) : local) * BM;
      it.q_base = prev_q;
      it.tok0 = prev_kv;
      it.qlen = len_q;
      it.kvlen = nkv - prev_kv;
      break;
    }
    acc += tiles; prev_q = nq; prev_kv = nkv;
  }
  if (!it.valid) return;
  const int kv_off = it.kvlen - it.qlen;
  const bool windowed = window_left >= 0;
  const bool has_sub0 = (it.m0 < it.qlen);
  const bool has_sub1 = (it.m0 + SUB < it.qlen);
  {
    int nmax = causal ? min(it.kvlen, kv_off + it.m0 + SUB) : it.kvlen;
    int nsafe = causal ? min(it.kvlen, kv_off + it.m0 + 1) : it.kvlen;
    if (windowed) nsafe = 0;
    if (!has_sub0) { nmax = 0; nsafe = 0; }
    nmax = min(nmax, it.kvlen);
    it.Ts0 = (nmax + BN - 1) / BN;
    it.Tf0 = nsafe / BN;
  }
  {
    int nmax = causal ? min(it.kvlen, kv_off + it.m0 + 2 * SUB) : it.kvlen;
    int nsafe = causal ? min(it.kvlen, kv_off + it.m0 + SUB + 1) : it.kvlen;
    if (windowed) nsafe = 0;
    if (!has_sub1) { nmax = 0; nsafe = 0; }
    nmax = min(nmax, it.kvlen);
    it.Ts1 = (nmax + BN - 1) / BN;
    it.Tf1 = nsafe / BN;
  }
  it.T = max(it.Ts0, it.Ts1);
  if (it.T <= 0) it.valid = 0;
}

DIN void decode_item(const int32_t* __restrict__ cum_q,
                     const int32_t* __restrict__ cum_kv, int batch,
                     int tiles_ub, int w, int causal, int window_left,
                     int opts, ItemGeom& it) {
  decode_item_th(cum_q, cum_kv, batch, w % tiles_ub, w / tiles_ub, causal,
                 window_left, opts, it);
}

// T of tile (b, m0) - textually the decode_item Ts0/Ts1 nmax formulas (T is
// independent of nsafe/windowed: it.T = max(Ts0, Ts1), Ts keys on nmax only).
DIN int tile_T(int len_q, int len_kv, int kv_off, int m0, int causal) {
  const bool has_sub0 = (m0 < len_q);
  const bool has_sub1 = (m0 + SUB < len_q);
  int nmax = causal ? min(len_kv, kv_off + m0 + SUB) : len_kv;
  if (!has_sub0) nmax = 0;
  nmax = min(nmax, len_kv);
  int nmax1 = causal ? min(len_kv, kv_off + m0 + 2 * SUB) : len_kv;
  if (!has_sub1) nmax1 = 0;
  nmax1 = min(nmax1, len_kv);
  return max((nmax + BN - 1) / BN, (nmax1 + BN - 1) / BN);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS, 1)
mla_tc2p_kernel(const __grid_constant__ CUtensorMap krmap,
                const __grid_constant__ CUtensorMap vrmap,
                const __grid_constant__ CUtensorMap qrmap,
                const __nv_bfloat16* __restrict__ qraw,
                __nv_bfloat16* __restrict__ out,
                float* __restrict__ lse,
                const float* __restrict__ sinks,
                const int32_t* __restrict__ cum_q,
                const int32_t* __restrict__ cum_kv,
                int32_t total_q, int32_t batch, int32_t heads, int32_t kv_heads,
                int32_t window_left, int32_t causal, int32_t has_lse,
                int32_t has_sink, float bmm1_scale, float bmm2_scale,
                int32_t tiles_ub, unsigned int* __restrict__ counter,
                int32_t opts, unsigned long long* __restrict__ dbg) {
  (void)qraw;
  extern __shared__ char smem[];
  char* sq = smem;                       // 2 x 49152B Q regions
  char* sk = smem + Q_TOTAL;             // 3 x 24576B K stages
  char* sv = sk + KV_STAGES * K_BYTES;   // 3 x 16384B V stages
  char* ctrl = smem + CTRL_OFF;
  uint32_t* tholder = reinterpret_cast<uint32_t*>(ctrl + NBARS * 8);
  ItemGeom* items = reinterpret_cast<ItemGeom*>(ctrl + ITEMS_OFF);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const uint32_t whint = (opts & 8) ? 0u : 0x989680u;
  const int group_div = heads / kv_heads;
  const float sc2 = bmm1_scale * LOG2E;

#ifdef TC2P_PREMETA
  // Cold-chain trim (E2 census dv_...: tc2p prologues run 14.5-25k cycles
  // against the cluster engines' 5.7k; the dbg51 chain is counter atomic ->
  // cum_q/cum_kv DRAM reads -> table walk/decode -> first Q TMA, all serial
  // and cold after the protocol's L2 eviction). Warp 9's mirror lanes issue
  // the cum_q/cum_kv reads BEFORE the init/fences/TMEM alloc so that latency
  // overlaps ~1-2k cycles of setup; tid0 prefetches the work-counter line
  // whose first atomicAdd would otherwise miss cold too. Each warp-9 lane
  // covers entries l and l+32 of the 64-entry mirror (one warp suffices for
  // any THREADS build; the loads are independent and pipeline).
  int pm_q0 = 0, pm_kv0 = 0, pm_q1 = 0, pm_kv1 = 0;
  const int pm_cap = batch < 63 ? batch : 63;   // entries 0..batch, cap 64
  if (warp == 9) {
    const int l = tid & 31;
    if (l <= pm_cap) {
      pm_q0 = cum_q[l];
      pm_kv0 = cum_kv[l];
    }
    if (l + 32 <= pm_cap) {
      pm_q1 = cum_q[l + 32];
      pm_kv1 = cum_kv[l + 32];
    }
  }
  if (tid == 0) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(counter));
  }
#endif

#ifdef TC2_TIMING
  const unsigned long long t_ent = (tid == 0) ? clock64() : 0ull;
#endif
#ifdef TC2_TIMING3
#if !defined(TC2_TIMING)
#error "TC2_TIMING3 requires TC2_TIMING (shares the timed roles and slot 51)"
#endif
  // E40 (probe195): worst-CTA prologue timeline, entry-relative. Slots
  // 38-46 via atomicMax (each mark fires once per CTA from the warp that
  // owns the event): 38 init/alloc done (tid0), 39 loader ready (warp9),
  // 40 first counter fetch returned, 41 item0 published (NEXT arrive),
  // 42 first Q TMAs issued, 43 issuer's first QF wait done (warp8),
  // 44 first QK0 committed, 45 first SF wait done (softmax tid0; the
  // slot-51 endpoint), 46 first BF (K0) wait done. Requires TC2_TIMING
  // (shares the timed roles); never in production builds.
  const unsigned long long t_ent3 = clock64();
#define T3MARK(slot) \
  do { if (dbg) atomicMax(dbg + (slot), \
        (unsigned long long)(clock64() - t_ent3)); } while (0)
#endif
  if (dbg && tid == 0) atomicAdd(dbg + 8, 1ull);

  // ---- control init (distributed) ----
  if (tid < NBARS) {
    const int i = tid;
    int cnt = 1;
    if (i >= PF00 && i <= PF11) cnt = 128;      // 4 softmax warps/subtile
    else if (i == OC0 || i == OC1) cnt = 128;
    else if (i == ODR0 || i == ODR1) cnt = 128;
    else if (i == CONS) cnt = 10;               // 8 softmax + 2 issuer warps
    else if (i >= BE0 && i <= BE2) cnt = 2;     // dual issuer commits
    mbar_init(bar(ctrl, i), cnt);
  }
  asm volatile("fence.mbarrier_init.release.cluster;");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (warp == 8) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 ::"r"(sm32_(tholder)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  fence_after_sync();
  const uint32_t tmem = *(volatile uint32_t*)tholder;
#ifdef TC2_TIMING3
  if (tid == 0) T3MARK(38);
#endif
  const int total_items = tiles_ub * heads;

  if (warp == 9) {
    // ============ LOADER: fetch, publish, Q(0) fill, flat KV ring =========
#ifdef TC2P_PREMETA
    // Land the entry-issued metadata reads in the smem mirror (entries l and
    // l+32 per lane; same warp consumes them, so __syncwarp is sufficient).
    {
      const int l = tid & 31;
      if (l <= pm_cap) {
        reinterpret_cast<int*>(ctrl + CUMQ_OFF)[l] = pm_q0;
        reinterpret_cast<int*>(ctrl + CUMKV_OFF)[l] = pm_kv0;
      }
      if (l + 32 <= pm_cap) {
        reinterpret_cast<int*>(ctrl + CUMQ_OFF)[l + 32] = pm_q1;
        reinterpret_cast<int*>(ctrl + CUMKV_OFF)[l + 32] = pm_kv1;
      }
      __syncwarp();
    }
#endif
#ifdef TC2P_LPTGP
    // E41 (probe196): WARP-PARALLEL LPTG table build, bit-identical to the
    // elected-lane serial build inside elect_one() below (same stable
    // counting sort; all scans are integer sums, associative). All 32
    // loader lanes run this BEFORE elect_one(); the elected lane then
    // skips its own build (flag in hist[257]; CTRL_BYTES reserves 260
    // ints for the 257-bin histogram) and only runs the serial stable
    // scatter. Motivation (probe195 dv_b4aab122264d timelines): the
    // serial build is the dominant loader-prep segment on every b>1
    // LPTG shape - 6.9k cyc (f1, 13 tiles), 11.8k (f6, 30), 19.5k (d3,
    // 47) per CTA, one lane doing a 257-bin zero, the tile_T walk and a
    // 257-step SERIALLY DEPENDENT descending prefix while 31 lanes idle.
    // Structure: lane i <-> sequence i (batch <= 32; >32 falls back to
    // serial via the flag), inclusive tile scan -> flat tile ranges;
    // each lane walks ITS OWN sequence's tiles (zero shuffles in the
    // hot loop; raw T -> tids[j], valid flag -> ths[j] at the flat
    // index j = acc_i + local, which is exactly the serial enumeration
    // order); 257-bin zero strided; stable compaction of valid entries
    // by ascending j (per-round exclusive scan + carry; all reads
    // complete before any write via __syncwarp; compaction writes land
    // strictly below the next round's read region since pos < 32(r+1));
    // hist counts by smem atomicAdd (commutative); three-phase
    // descending 257-bin prefix (lane l owns bins [8l, 8l+8+ (l==31)),
    // per-lane serial descending, offsets = suffix sums of partials via
    // one ascending inclusive scan, rewrite descending with the
    // offset). Tiles bound: t_tot = sum(ceil(len_q/BM)) <= tiles_ub =
    // total_q/BM + batch <= 96 < 100 under the frozen contract, so the
    // 100-entry scratch and the serial build's n_real<100 cap never
    // bind; the t_tot<=100 gate below is the belt (falls back to
    // serial if ever violated).
    {
      const int lane = tid & 31;
#ifdef TC2P_PREMETA
      const int32_t* const cqp = (batch <= 63)
          ? reinterpret_cast<const int32_t*>(ctrl + CUMQ_OFF) : cum_q;
      const int32_t* const ckp = (batch <= 63)
          ? reinterpret_cast<const int32_t*>(ctrl + CUMKV_OFF) : cum_kv;
#else
      const int32_t* const cqp = cum_q;
      const int32_t* const ckp = cum_kv;
#endif
      const bool lptg_p = (opts & (3 << 20)) != 0;
      bool tid_skip = false;
#ifdef TC2P_TID
      if (batch == 1 && (opts & 2)) {
        const int lq1 = cqp[1] - cqp[0];
        const int lk1 = ckp[1] - ckp[0];
        const int tiles1 = (lq1 + BM - 1) / BM;
        if (lk1 >= lq1 && tiles1 <= 100 && lk1 <= 256 * BN) tid_skip = true;
      }
#endif
      if (lptg_p && !tid_skip) {
        int* const hist_p = reinterpret_cast<int*>(ctrl + HIST_OFF);
        bool built = false;
        if (batch <= 32) {
          int* const tids_p = reinterpret_cast<int*>(ctrl + TIDS_OFF);
          int* const ths_p = reinterpret_cast<int*>(ctrl + THS_OFF);
          // Phase 0: lane <-> seq metadata + inclusive tile scan.
          int my_tiles = 0, my_lenq = 0, my_lenkv = 0;
          if (lane < batch) {
            my_lenq = cqp[lane + 1] - cqp[lane];
            my_lenkv = ckp[lane + 1] - ckp[lane];
            my_tiles = (my_lenq + BM - 1) / BM;
          }
          int incl = my_tiles;
          #pragma unroll
          for (int d = 1; d < 32; d <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, incl, d);
            if (lane >= d) incl += v;
          }
          const int my_acc = incl - my_tiles;
          const int t_tot = __shfl_sync(0xffffffffu, incl, 31);
          const int my_kvoff = my_lenkv - my_lenq;
          if (t_tot <= 100) {
            // Phase 1: own-sequence tile walk (raw index = serial order).
            for (int local = 0; local < my_tiles; ++local) {
              const int j = my_acc + local;
              if (j < 100) {
                const int li = (opts & 2) ? (my_tiles - 1 - local) : local;
                const int T = tile_T(my_lenq, my_lenkv, my_kvoff, li * BM,
                                     causal);
                tids_p[j] = T;
                ths_p[j] = (T > 0 && T <= 256) ? 1 : 0;
              }
            }
            for (int t = lane; t <= 256; t += 32) hist_p[t] = 0;
            __syncwarp();
            // Phase 2: stable compaction of valid entries by ascending j.
            int carry = 0;
            for (int base = 0; base < t_tot; base += 32) {
              const int idx = base + lane;
              const int v = (idx < t_tot) ? ths_p[idx] : 0;
              const int rawT = (idx < t_tot) ? tids_p[idx] : 0;
              int incl2 = v;
              #pragma unroll
              for (int d = 1; d < 32; d <<= 1) {
                const int w = __shfl_up_sync(0xffffffffu, incl2, d);
                if (lane >= d) incl2 += w;
              }
              const int round_total = __shfl_sync(0xffffffffu, incl2, 31);
              const int pos = carry + incl2 - v;
              __syncwarp();
              if (v && pos < 100) {
                tids_p[pos] = idx;
                ths_p[pos] = rawT;
                atomicAdd(&hist_p[rawT], 1);
              }
              carry += round_total;
              __syncwarp();
            }
            // Phase 3: descending exclusive prefix over the 257 bins.
            const int lo = lane * 8;
            const int cnt = (lane == 31) ? 9 : 8;
            int partial = 0;
            for (int k = cnt - 1; k >= 0; --k) partial += hist_p[lo + k];
            int incl3 = partial;
            #pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
              const int v = __shfl_up_sync(0xffffffffu, incl3, d);
              if (lane >= d) incl3 += v;
            }
            const int total_all = __shfl_sync(0xffffffffu, incl3, 31);
            int run = total_all - incl3;   // = sum of partials of higher lanes
            for (int k = cnt - 1; k >= 0; --k) {
              const int c = hist_p[lo + k];
              hist_p[lo + k] = run;
              run += c;
            }
            __syncwarp();
            built = true;
          }
        }
        if (lane == 0) hist_p[257] = built ? 1 : 0;
        __syncwarp();
      }
    }
#endif
    if (elect_one()) {
#ifdef TC2P_PREMETA
      // batch > 63 (outside the frozen domain and the 64-entry mirror):
      // walk/TID/decode fall back to the global arrays. Flag-off: cq/ck are
      // const aliases of the global pointers - identical code by construction.
      const int32_t* const cq = (batch <= 63)
          ? reinterpret_cast<const int32_t*>(ctrl + CUMQ_OFF) : cum_q;
      const int32_t* const ck = (batch <= 63)
          ? reinterpret_cast<const int32_t*>(ctrl + CUMKV_OFF) : cum_kv;
#else
      const int32_t* const cq = cum_q;
      const int32_t* const ck = cum_kv;
#endif
      // ---- bit20 LPTG: global descending-T enumeration table ----
      // Per-head-block LPT (opts bit1) leaves each head's heaviest tile at
      // ~94-96% of the steal order (measured residual = span_max - span_avg
      // ~= 0.5 x max_unit on f3/f7/f8, probe41/42). Global sort starts every
      // heavy item at t~0. Single elected lane builds + reads the table ->
      // no sync needed; counting sort is stable (ties: ascending tile_id)
      // so every CTA derives the identical order from cum_q/cum_kv.
      int* ord = reinterpret_cast<int*>(ctrl + ORD_OFF);
      const bool lptg = (opts & (3 << 20)) != 0;   // bit20 plain | bit21 grouped
      // bit21 grouped mode: group-major enumeration over the same sorted
      // table. Concurrent heads are bounded to gsz = bits22-29 + 1, so the
      // tile-major KV working set stays gsz*mk*640 <= L2 (the bit20 locality
      // law) while the descending-T order still kills the 0.5*max_unit
      // residual WITHIN each group. gsz == heads degenerates exactly to
      // bit20 order. Host picks gsz = largest divisor of heads under the ws
      // budget (f1: 48 of 96; d4: 64 of 128).
      const bool grouped = (opts & (1 << 21)) != 0;
      const int gsz = grouped ? (((opts >> 22) & 0xFF) + 1) : heads;
      int n_real = 0;
      int n_eff = total_items;
      int grp_items = total_items;
#ifdef TC2P_TID
      bool tid_ident = false;
#endif
      if (lptg) {
#ifdef TC2P_TID
      // TC2P_TID: at batch == 1 with opts bit1 the counting sort below is
      // provably the IDENTITY, so the table never has to be built and ord[r] is
      // just r. What that removes from the loader's elected lane - the one serial
      // thing every CTA does before its first KV fetch - is a 257-entry histogram
      // zero, the tile_T walk, a 257-step SERIALLY DEPENDENT descending prefix and
      // the scatter. Measured in the arm that first shipped it (dv_9cbb9ad2d2c5,
      // same-job official): the walk+sort term went 1795.4 -> 85.6 cycles on f0,
      // worth f0 +1.968 us, f2 +2.048, f4 +2.960.
      //
      // The identity is proved, not asserted: experiments/verify_lptg_identity.py
      // transcribes this build and enumerates 491 domain shapes - 126 with
      // batch == 1, 0 divergences - plus a 220-point synthetic grid that leaves the
      // frozen domain on purpose, and shows a witness for the necessity of every
      // clause below. bit1 is load-bearing: 44 of the 126 batch-1 shapes FAIL the
      // identity with it off.
      //
      // The three extra guards cover what the domain enumeration cannot, because
      // they only bite outside the frozen ranges:
      //   lk1 >= lq1     kv_off >= 0, so nmax >= 1 for every block and no tile is
      //                  dropped for T == 0 (which would make n_real < tiles and
      //                  break ord[r] == r). Witness without it: ql=257 kv=1.
      //   tiles1 <= 100  the `n_real < 100` cap in the walk never binds.
      //                  Witness without it: ql=25856 kv=100000, n_real=0 of 101.
      //   lk1 <= 256*BN  T = ceil(nmax/BN) <= ceil(lk1/BN) <= 256, so no tile is
      //                  dropped by the `T <= 256` histogram-range filter either.
      //                  Witness without it: ql=1 kv=16385, T=256+ -> n_real=0.
      // Any guard failing falls through to the unmodified build, so the flag can
      // only ever skip work that has been proved unnecessary. Every line of it
      // lives inside #ifdef TC2P_TID, including the braces that scope the skipped
      // block, so with the flag undefined the reduced source is byte-identical to
      // the join start - which is what check_production_unchanged verifies, and
      // why the enclosed code is deliberately NOT re-indented.
        if (batch == 1 && (opts & 2)) {
          const int lq1 = cq[1] - cq[0];
          const int lk1 = ck[1] - ck[0];
          const int tiles1 = (lq1 + BM - 1) / BM;
          if (lk1 >= lq1 && tiles1 <= 100 && lk1 <= 256 * BN) {
            tid_ident = true;
            n_real = tiles1;
          }
        }
        if (!tid_ident) {
#endif
        int* tids = reinterpret_cast<int*>(ctrl + TIDS_OFF);
        int* ths = reinterpret_cast<int*>(ctrl + THS_OFF);
        int* hist = reinterpret_cast<int*>(ctrl + HIST_OFF);
#ifdef TC2P_LPTGP
        // E41: reuse the warp-parallel build's table when it ran (flag in
        // hist[257]); n_real = hist[0] because the descending exclusive
        // prefix leaves bin 0 holding #{T > 0} = the valid-entry count,
        // and the scatter below only increments bins >= 1 (valid T). The
        // serial stable scatter stays on the elected lane (<= 96
        // iterations; parallelising it per-bin would need the same
        // ascending-j stability proof the compaction already carries).
        if (hist[257] != 0) {
          n_real = hist[0];
          for (int j = 0; j < n_real; ++j) ord[hist[ths[j]]++] = tids[j];
        } else {
#endif
        for (int t = 0; t <= 256; ++t) hist[t] = 0;
        int acc = 0, prev_q = cq[0], prev_kv = ck[0];
        for (int i = 0; i < batch; ++i) {
          const int nq = cq[i + 1], nkv = ck[i + 1];
          const int len_q = nq - prev_q, len_kv = nkv - prev_kv;
          const int tiles = (len_q + BM - 1) / BM;
          const int kv_off = len_kv - len_q;
          for (int local = 0; local < tiles && n_real < 100; ++local) {
            const int li = (opts & 2) ? (tiles - 1 - local) : local;
            const int T = tile_T(len_q, len_kv, kv_off, li * BM, causal);
            if (T > 0 && T <= 256) {
              tids[n_real] = acc + local;
              ths[n_real] = T;
              hist[T]++;
              n_real++;
            }
          }
          acc += tiles; prev_q = nq; prev_kv = nkv;
        }
        int p = 0;
        for (int t = 256; t >= 0; --t) { const int c = hist[t]; hist[t] = p; p += c; }
        for (int j = 0; j < n_real; ++j) ord[hist[ths[j]]++] = tids[j];
#ifdef TC2P_LPTGP
        }
#endif
#ifdef TC2P_TID
        }
#endif
        n_eff = n_real * heads;
        grp_items = n_real * gsz;
#ifdef TC2_TIMING
        // E41 order proof: position-weighted checksum of the final ord[]
        // (plus n_real). Serial and LPTGP timing builds must produce the
        // SAME slot-47 sum on every LPTG shape - item-order changes are
        // invisible to output bit-equality (per-item independence), so
        // this is the only host-visible proof of enumeration identity.
#ifdef TC2P_TID
        if (dbg && !tid_ident) {
#else
        if (dbg) {
#endif
          unsigned long long cs = (unsigned long long)n_real * 1000003ull;
          for (int j = 0; j < n_real; ++j)
            cs += (unsigned long long)(unsigned)ord[j]
                  * (unsigned long long)(j + 1);
          atomicAdd(dbg + 47, cs);
        }
#endif
      }
      int f = 0;                 // publication index
      long long g = 0;           // global KV stage index
#ifdef TC2_TIMING3
      bool t3_40 = false;
      if ((tid & 31) == 0) T3MARK(39);
#endif
      while (true) {
        // fetch next REAL item (skip surplus/empty silently)
        ItemGeom it;
        while (true) {
          const int w = (int)atomicAdd(counter, 1u);
#ifdef TC2_TIMING3
          if (!t3_40 && (tid & 31) == 0) { t3_40 = true; T3MARK(40); }
#endif
          if (w == n_eff + (int)gridDim.x - 1)
            *counter = 0u;   // SR: final increment of this launch zeroes
                             // the counter for the next launch (no further
                             // counter accesses follow; kernel exit flushes)
          if (w >= n_eff) { it.valid = 0; it.T = 0; break; }
          if (lptg) {
            int r, hh;
            if (grouped) {
              const int gidx = w / grp_items;
              const int rem = w - gidx * grp_items;
              r = rem / gsz;
              hh = gidx * gsz + (rem - r * gsz);
            } else {
              r = w / heads;
              hh = w - r * heads;
            }
#ifdef TC2P_TID
            decode_item_th(cq, ck, batch, tid_ident ? r : ord[r], hh, causal,
                           window_left, opts, it);
#else
            decode_item_th(cq, ck, batch, ord[r], hh, causal,
                           window_left, opts, it);
#endif
          } else {
            decode_item(cq, ck, batch, tiles_ub, w, causal,
                        window_left, opts, it);
          }
          if (it.valid) break;
        }
        // publish(k) writes slot k&1. Gate on CONS(k-1): every consumer has
        // copied item k-1's geometry, and the NEXT phase can never run 2
        // ahead of a waiter (parity aliasing would deadlock).
        if (f >= 1) mbar_wait_h(bar(ctrl, CONS), (f - 1) & 1, whint);
        items[f & 1] = it;                       // valid=0 marks the sentinel
        mbar_arrive(bar(ctrl, NEXT));            // release: geometry visible
#ifdef TC2_TIMING3
        if (f == 0 && (tid & 31) == 0) T3MARK(41);
#endif
        if (!it.valid) break;
        if (f == 0) {
          // first Q fill (later refills are issued by issuer warp 8)
#ifdef TC2P_QSPLIT
          mbar_expect_tx(bar(ctrl, QF0), Q_BYTES);
          tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF0), 0,
                      it.q_base + it.m0, 0, it.h);
          mbar_expect_tx(bar(ctrl, QF2), Q_BYTES);
          tma_load_4d(&qrmap, sm32_(sq) + Q_BYTES, bar(ctrl, QF2), 0,
                      it.q_base + it.m0 + SUB, 0, it.h);
#else
          mbar_expect_tx(bar(ctrl, QF0), 2 * Q_BYTES);
          tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF0), 0,
                      it.q_base + it.m0, 0, it.h);
          tma_load_4d(&qrmap, sm32_(sq) + Q_BYTES, bar(ctrl, QF0), 0,
                      it.q_base + it.m0 + SUB, 0, it.h);
#endif
#ifdef TC2_TIMING3
          if ((tid & 31) == 0) T3MARK(42);
#endif
        }
        const int h_kv = it.h / group_div;
        for (int i = 0; i < it.T; ++i, ++g) {
          const int st3 = (int)(g % KV_STAGES);
          if (g >= KV_STAGES)
            mbar_wait_h(bar(ctrl, BE0 + st3), (int)((g / KV_STAGES - 1) & 1),
                        whint);
          mbar_expect_tx(bar(ctrl, BF0 + st3), K_BYTES + V_BYTES);
          load_kv_tile(krmap, vrmap, sm32_(sk) + st3 * K_BYTES,
                       sm32_(sv) + st3 * V_BYTES, bar(ctrl, BF0 + st3),
                       it.tok0 + i * BN, h_kv);
        }
        ++f;
      }
    }
  } else if (warp == 8 || warp == 10) {
    // ============ ISSUERS (warp8: s0 + Q refill; warp10: s1) ==============
    if (elect_one()) {
      const bool s1warp = (warp == 10);
      const uint32_t qb = sm32_(sq) + (s1warp ? Q_BYTES : 0);
      // K-major SW128 canonical descriptor constants (as v1 TMA-Q mode)
#ifdef TC2P_BOX64
      const int kplane = 8192;
      const uint32_t ksbo = 1024u;
      const uint32_t vlbo = 8192u;
      const uint32_t vsbo = 1024u;
      const int vstep = 2048;
#else
      const int kplane = 1024;
      const uint32_t ksbo = 3072u;
      const uint32_t vlbo = 1024u;
      const uint32_t vsbo = 2048u;
      const int vstep = 4096;
#endif
      const uint32_t qsbo = 1024u;
      const int qa16 = 1024;      // TMA-Q atom stride in 16B units
      const uint32_t sfB = s1warp ? SF10 : SF00;
      const uint32_t pfB = s1warp ? PF10 : PF00;
      const uint32_t pvB = s1warp ? PV10 : PV00;
      const uint32_t ocB = s1warp ? OC1 : OC0;
      const uint32_t odrB = s1warp ? ODR1 : ODR0;
      const int tcol_s = s1warp ? 128 : 0;
      const int tcol_o = s1warp ? 384 : 256;
      int f = 0;                  // item index
      long long Gbase = 0;        // global stage counter at item start
      int C = 0;                  // own-subtile global tile counter
      int otherC = 0;             // other subtile's global counter (warp8)
      int prevTs_own = 0, prevTs_other = 0;
#ifdef TC2P_QEARLY
      bool early0_done = false;   // half-0 of item f+1 already in flight
#endif
#ifdef TC2_TIMING
      const bool timed = (dbg != nullptr);
      long long is_tiles = 0, is_bf = 0, is_bew = 0, is_qk = 0,
                is_pfw = 0, is_ocw = 0, is_pv = 0;
      long long u0 = 0, u1 = 0, u2 = 0, p0 = 0, p1 = 0, bw = 0;
#ifdef TC2_TIMING2
      // E37 (requires TC2_TIMING; never in production builds): 31 is_item =
      // issuer's per-item boundary chain [previous item's tail-PV end ->
      // this item's first BF wait start] = NEXT supply wait + CONS arrive +
      // Q REFILL (prev-last-QK SF waits + 2x49KB Q TMA issues + QF arrival
      // wait) + loop admin; 32 is_items = real items (sentinel excluded -
      // the issuer breaks before any tile).
      // E38a composition split: 35 is_next_i = the NEXT supply wait;
      // 36 is_qsf = prev-last-QK SF waits + fence; 37 is_qf = the QF
      // arrival wait; residual (31-35-36-37) = TMA issues + admin.
      long long is_item = 0, is_items = 0, i_chain = 0;
      long long is_next_i = 0, is_qsf = 0, is_qf = 0;
#endif
#endif
      while (true) {
#if defined(TC2_TIMING) && defined(TC2_TIMING2)
        if (timed && !s1warp) i_chain = clock64();
#endif
        mbar_wait_h(bar(ctrl, NEXT), f & 1, whint);
#if defined(TC2_TIMING) && defined(TC2_TIMING2)
        if (timed && !s1warp) { is_next_i += clock64() - i_chain; }
#endif
        fence_after_sync();
        const ItemGeom it = items[f & 1];
        mbar_arrive(bar(ctrl, CONS));
        if (!it.valid) break;
        const int Ts = s1warp ? it.Ts1 : it.Ts0;
        const int T = it.T;
        if (!s1warp) {
          // ---- Q refill for this item (item 0 was filled by the loader) ----
          if (f > 0) {
            // Wait until the previous item's last QK MMAs completed (they
            // were the last readers of the Q smem regions).
#if defined(TC2_TIMING) && defined(TC2_TIMING2)
            const long long tq0 = (timed && !s1warp) ? clock64() : 0;
#endif
            if (prevTs_own > 0)
              mbar_wait_h(bar(ctrl, SF00 + ((C - 1) & 1)),
                          ((C - 1) / 2) & 1, whint);
            if (prevTs_other > 0)
              mbar_wait_h(bar(ctrl, SF10 + ((otherC - 1) & 1)),
                          ((otherC - 1) / 2) & 1, whint);
            fence_after_sync();
#if defined(TC2_TIMING) && defined(TC2_TIMING2)
            if (timed && !s1warp) is_qsf += clock64() - tq0;
#endif
#ifdef TC2P_QSPLIT
#ifdef TC2P_QEARLY
            if (!early0_done) {
              mbar_expect_tx(bar(ctrl, QF0 + (f & 1)), Q_BYTES);
              tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF0 + (f & 1)), 0,
                          it.q_base + it.m0, 0, it.h);
            }
            early0_done = false;
#else
            mbar_expect_tx(bar(ctrl, QF0 + (f & 1)), Q_BYTES);
            tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF0 + (f & 1)), 0,
                        it.q_base + it.m0, 0, it.h);
#endif
            mbar_expect_tx(bar(ctrl, QF2 + (f & 1)), Q_BYTES);
            tma_load_4d(&qrmap, sm32_(sq) + Q_BYTES,
                        bar(ctrl, QF2 + (f & 1)), 0,
                        it.q_base + it.m0 + SUB, 0, it.h);
#else
            mbar_expect_tx(bar(ctrl, QF0 + (f & 1)), 2 * Q_BYTES);
            tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF0 + (f & 1)), 0,
                        it.q_base + it.m0, 0, it.h);
            tma_load_4d(&qrmap, sm32_(sq) + Q_BYTES,
                        bar(ctrl, QF0 + (f & 1)), 0,
                        it.q_base + it.m0 + SUB, 0, it.h);
#endif
          }
          prevTs_own = it.Ts0;
          prevTs_other = it.Ts1;
        }
#if defined(TC2_TIMING) && defined(TC2_TIMING2)
        const long long tq1 = (timed && !s1warp) ? clock64() : 0;
#endif
#ifdef TC2P_QSPLIT
        mbar_wait_h(bar(ctrl, (s1warp ? QF2 : QF0) + (f & 1)),
                    (f >> 1) & 1, whint);
#else
        mbar_wait_h(bar(ctrl, QF0 + (f & 1)), (f >> 1) & 1, whint);
#endif
#if defined(TC2_TIMING) && defined(TC2_TIMING2)
        if (timed && !s1warp) is_qf += clock64() - tq1;
#endif
#ifdef TC2_TIMING3
        if (f == 0 && !s1warp && (tid & 31) == 0) T3MARK(43);
#endif
        fence_after_sync();
        for (int i = 0; i < T; ++i) {
          const long long g = Gbase + i;
          const int st3 = (int)(g % KV_STAGES);
#ifdef TC2_TIMING
          if (timed && !s1warp) { bw = 0; u0 = clock64(); }
#ifdef TC2_TIMING2
          if (timed && !s1warp && i == 0) {
            is_item += u0 - i_chain; is_items += 1;
          }
#endif
#endif
          mbar_wait_h(bar(ctrl, BF0 + st3), (int)((g / KV_STAGES) & 1), whint);
          fence_after_sync();
#ifdef TC2_TIMING
          if (timed && !s1warp) { u1 = clock64(); is_bf += u1 - u0; }
#endif
#ifdef TC2_TIMING3
          if (f == 0 && i == 0 && !s1warp && (tid & 31) == 0) T3MARK(46);
#endif
          // ---- QK(s, i) ----
          const int Cg = C + i;                 // global own-subtile tile idx
          if (i < Ts) {
            if (Cg >= 2) {
#ifdef TC2_TIMING
              if (timed && !s1warp) u2 = clock64();
#endif
              mbar_wait_h(bar(ctrl, pvB + (Cg & 1)),
                          (int)(((Cg - 2) / 2) & 1), whint);
              fence_after_sync();
#ifdef TC2_TIMING
              if (timed && !s1warp) {
                long long t = clock64(); is_bew += t - u2; bw += t - u2;
              }
#endif
            }
            const uint32_t tmem_s = tmem + tcol_s + (Cg & 1) * 64;
            const uint64_t da0 = make_desc(qb, 0, qsbo);
            const uint64_t db0 =
                make_desc(sm32_(sk) + st3 * K_BYTES, 0, ksbo);
            #pragma unroll
            for (int kk = 0; kk < 12; ++kk) {
              mma_ss(tmem_s, da0 + ((kk >> 2) * qa16 + (kk & 3) * 2),
                     db0 + ((kk >> 2) * (kplane >> 4) + (kk & 3) * 2),
                     IDESC_QK64, kk > 0);
            }
            mma_commit(bar(ctrl, sfB + (Cg & 1)));
          }
#ifdef TC2_TIMING
          if (timed && !s1warp) {
            u2 = clock64(); is_qk += u2 - u1 - bw; is_tiles += 1;
          }
#endif
#ifdef TC2_TIMING3
          if (f == 0 && Cg == 0 && !s1warp && (tid & 31) == 0) T3MARK(44);
#endif
          // ---- PV(s, i-1) ----
          if (i > 0 && (i - 1) < Ts) {
            const int ip = Cg - 1;              // global idx of the PV tile
            const int stp = ip & 1;
            const uint32_t vbase_p =
                sm32_(sv) + (int)((g - 1) % KV_STAGES) * V_BYTES;
#ifdef TC2_TIMING
            if (timed && !s1warp) p0 = clock64();
#endif
            mbar_wait_h(bar(ctrl, pfB + stp), (int)((ip / 2) & 1), whint);
#ifdef TC2_TIMING
            if (timed && !s1warp) { p1 = clock64(); is_pfw += p1 - p0; p0 = p1; }
#endif
            mbar_wait_h(bar(ctrl, ocB), ip & 1, whint);
#ifdef TC2_TIMING
            if (timed && !s1warp) { p1 = clock64(); is_ocw += p1 - p0; }
#endif
            // First PV of the item reuses the O accumulator last written by
            // the PREVIOUS item: wait for its epilogue reads (ODR).
            if (i == 1 && f > 0) {
              mbar_wait_h(bar(ctrl, odrB), (f - 1) & 1, whint);
            }
            fence_after_sync();
            const uint32_t tmem_p = tmem + tcol_s + stp * 64;
            const uint64_t db0 = make_desc(vbase_p, vlbo, vsbo);
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              mma_ts(tmem + tcol_o, tmem_p + kk * 8,
                     db0 + kk * (vstep >> 4), IDESC_PV, (i > 1) | (kk > 0));
            mma_commit(bar(ctrl, pvB + stp));
#ifdef TC2_TIMING
            if (timed && !s1warp) is_pv += clock64() - p1;
#endif
          }
          // ---- release KV stage of global tile g-1 (pairs across warps) ----
          if (g >= 1) {
            mma_commit(bar(ctrl, BE0 + (int)((g - 1) % KV_STAGES)));
          }
#ifdef TC2P_QEARLY
          // E39 early half-0 refill (requires QSPLIT; warp 8 only). Fires on
          // the FIRST loop pass with no subtile-0 QK work (i == Ts; causal
          // Ts0 < Ts1 leaves T-Ts0 such passes): subtile-0's Q half died at
          // its last QK commit (i == Ts-1), and the SF wait below is the
          // same "tcgen05 completion implies SMEM reads done" guarantee the
          // boundary refill rides. The NEXT(f+1) wait is an idempotent phase
          // OBSERVATION (mbarrier waits don't consume phases; the loop-top
          // wait for f+1 then returns immediately), and items[(f+1)&1] is
          // visible behind it. QF0-slot ownership stays single-warp (warp 8
          // is the only QF0 expect/issue agent; warp 10 waits QF2 only).
          // The loader publishes item f+1 gated only on CONS(f) (arrived at
          // this item's top) + its steal/decode, so this wait cannot
          // deadlock against the KV ring; worst case it absorbs the loader's
          // decode latency once per item, vs the full 49KB transfer saved.
          if (!s1warp && i == Ts && !early0_done) {
            const long long cl = C + Ts - 1;      // own last global QK tile
            mbar_wait_h(bar(ctrl, SF00 + (int)(cl & 1)),
                        (int)((cl / 2) & 1), whint);
            fence_after_sync();
            mbar_wait_h(bar(ctrl, NEXT), (f + 1) & 1, whint);
            fence_after_sync();
            const ItemGeom itn = items[(f + 1) & 1];
            if (itn.valid) {
              mbar_expect_tx(bar(ctrl, QF0 + ((f + 1) & 1)), Q_BYTES);
              tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF0 + ((f + 1) & 1)),
                          0, itn.q_base + itn.m0, 0, itn.h);
              early0_done = true;
            }
          }
#endif
        }
        // ---- tail: PV(s, Ts-1) when this subtile ran to T ----
        {
          const int il = Ts - 1;
          if (il >= 0 && il == T - 1) {
            const int ip = C + il;              // global idx
            const int stl = ip & 1;
            const long long gl = Gbase + il;
            mbar_wait_h(bar(ctrl, pfB + stl), (int)((ip / 2) & 1), whint);
            mbar_wait_h(bar(ctrl, ocB), ip & 1, whint);
            if (f > 0 && T == 1) {
              // single-tile item: the tail PV IS the first PV -> ODR guard
              mbar_wait_h(bar(ctrl, odrB), (f - 1) & 1, whint);
            }
            fence_after_sync();
            const uint32_t tmem_p = tmem + tcol_s + stl * 64;
            const uint64_t db0 =
                make_desc(sm32_(sv) + (int)(gl % KV_STAGES) * V_BYTES, vlbo,
                          vsbo);
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              mma_ts(tmem + tcol_o, tmem_p + kk * 8,
                     db0 + kk * (vstep >> 4), IDESC_PV, (il > 0) | (kk > 0));
            mma_commit(bar(ctrl, pvB + stl));
          }
        }
        C += Ts;
        if (!s1warp) otherC += it.Ts1;
        Gbase += T;
        ++f;
      }
#ifdef TC2_TIMING
      if (timed && !s1warp) {
        atomicAdd(dbg + 24, (unsigned long long)is_tiles);
        atomicAdd(dbg + 25, (unsigned long long)is_bf);
        atomicAdd(dbg + 26, (unsigned long long)is_bew);
        atomicAdd(dbg + 27, (unsigned long long)is_qk);
        atomicAdd(dbg + 28, (unsigned long long)is_pfw);
        atomicAdd(dbg + 29, (unsigned long long)is_ocw);
        atomicAdd(dbg + 30, (unsigned long long)is_pv);
#ifdef TC2_TIMING2
        atomicAdd(dbg + 31, (unsigned long long)is_item);
        atomicAdd(dbg + 32, (unsigned long long)is_items);
        atomicAdd(dbg + 35, (unsigned long long)is_next_i);
        atomicAdd(dbg + 36, (unsigned long long)is_qsf);
        atomicAdd(dbg + 37, (unsigned long long)is_qf);
#endif
      }
#endif
    }
  } else if (warp < 8) {
    // ============ SOFTMAX + CORRECTION + EPILOGUE (per item) ===============
    const int s = warp >> 2;               // subtile
    const int w4 = warp & 3;               // warp within subtile
    const int lane = tid & 31;
    const int row = w4 * 32 + lane;        // row within the subtile [0,128)
    const uint32_t lb = static_cast<uint32_t>(w4 * 32) << 16;
    const uint32_t sfB = s ? SF10 : SF00;
    const uint32_t pfB = s ? PF10 : PF00;
    const uint32_t pvB = s ? PV10 : PV00;
    const uint32_t ocB = s ? OC1 : OC0;
    const uint32_t odrB = s ? ODR1 : ODR0;
    const int tcol_s = s * 128;
    const int tcol_o = 256 + s * 128;
    float sv2[2][32];
    int f = 0;                  // item index
    int C = 0;                  // own-subtile global tile counter
#ifdef TC2_TIMING
    const bool timed_s = (dbg != nullptr) && (tid == 0);
#ifdef TC2_TIMING4
#ifndef TC2_TIMING
#error "TC2_TIMING4 requires TC2_TIMING (shares the timed_s/c0 chain)"
#endif
    // E43 (probe198): slot-19 substage split mirroring tc9's TC9_TIMING4
    // (slots 36-38): 55 = exp/pack loops incl. the in-loop tmem_st16bf
    // issues, 56 = tmem_wait_st, 57 = fence_before_sync + pfB arrive + the
    // d update (tc9's tail is JUST the d update - its PF arrive sits after
    // sm_corr). Conservation: 55+56+57 == 19 exactly.
    long long smx_loop = 0, smx_wst = 0, smx_tail = 0;
#endif
#ifdef TC2_TIMING5
#ifndef TC2_TIMING
#error "TC2_TIMING5 requires TC2_TIMING"
#endif
    // E45 (probe202): first-class accounting of the loop-top SF wait
    // (mbar_wait_h(sfB+st)), which today falls UNACCOUNTED inside the
    // sm_per interval (cprev -> c0): probe194's tc2p "unaccounted"
    // 906/936/694 cyc/tile on f1/f0/f6 is suspected to be mostly this
    // wait = softmax idling on the QK MMA result (latency-bound
    // ping-pong). Slot 58 = SF wait + fence_after_sync per tile.
    long long sm_sfw2 = 0;
#endif
    long long sm_tiles = 0, sm_sld = 0, sm_max = 0, sm_exp = 0,
              sm_bew = 0, sm_corr = 0, sm_per = 0;
    long long c0 = 0, cprev = 0;
#ifdef TC2_TIMING2
    // E37 (requires TC2_TIMING; never in production builds): 33 sm_next =
    // softmax warp's per-item NEXT supply wait (SUBSET of the item-boundary
    // sm_per interval); 34 sm_items = item count incl. the sentinel pull.
    long long sm_next = 0, sm_items = 0;
#endif
#endif
    while (true) {
#if defined(TC2_TIMING) && defined(TC2_TIMING2)
      const long long tn0 = timed_s ? clock64() : 0;
#endif
      mbar_wait_h(bar(ctrl, NEXT), f & 1, whint);
#if defined(TC2_TIMING) && defined(TC2_TIMING2)
      if (timed_s) { sm_next += clock64() - tn0; sm_items += 1; }
#endif
      fence_after_sync();
      const ItemGeom it = items[f & 1];
      if (lane == 0) mbar_arrive(bar(ctrl, CONS));
      if (!it.valid) break;
      const int grow = it.m0 + s * SUB + row;   // batch-local q row
      const int vis = (it.kvlen - it.qlen) + grow;
      const int Ts = s ? it.Ts1 : it.Ts0;
      const int Tfull = s ? it.Tf1 : it.Tf0;
      const int kvlen = it.kvlen;
      const int qlen = it.qlen;
      __nv_bfloat16* outp =
          out + (static_cast<int64_t>(it.q_base) * heads + it.h) * DV;
      float* lsep = has_lse
          ? lse + static_cast<int64_t>(it.q_base) * heads + it.h : nullptr;
      const int64_t o_stride = static_cast<int64_t>(heads) * DV;
      const float sink2 = has_sink ? sinks[it.h] * LOG2E : 0.0f;
      const bool windowed = window_left >= 0;
      float mr = NEG_BIG, d = 0.0f;

      for (int i = 0; i < Ts; ++i) {
        const int Cg = C + i;
        const int st = Cg & 1;
        const uint32_t ts = tmem + tcol_s + st * 64 + lb;
#ifdef TC2_TIMING5
        const long long t5s = timed_s ? clock64() : 0;
#endif
        mbar_wait_h(bar(ctrl, sfB + st), (int)((Cg / 2) & 1), whint);
        fence_after_sync();
#ifdef TC2_TIMING5
        if (timed_s) sm_sfw2 += clock64() - t5s;
#endif
#ifdef TC2_TIMING
        if (timed_s) {
          c0 = clock64();
          if (Cg > 0) sm_per += c0 - cprev;
          else {
            atomicAdd(dbg + 51, (unsigned long long)(c0 - t_ent));
#ifdef TC2_TIMING3
            T3MARK(45);
#endif
          }
          cprev = c0;
        }
#endif
        const int n0 = i * BN;
        const bool masked = i >= Tfull;

#ifdef TC2P_REDMAX
        // Fused 64-column load + hardware row max (kernel_tc9.cu mechanism;
        // measured there: same rate as a plain x64 load, bit-exact against
        // an FMNMX tree). Removes the whole software max phase from unmasked
        // tiles - probe32 priced it at 378-422 cyc/tile on f1 (with MAXTREE)
        // and up to 752-793 on d7 (serial), latency-dominated.
        // E4b (dv_00f066dcbfe2): unconditional REDMAX measured f1 0.9947
        // (9/10) but d3 1.0072 (mask-heavy: 372 of 564 items are 1-tile
        // masked, so the fused load's discarded bound is paid ~everywhere).
        // TC2P_REDMAXU therefore keeps masked tiles on the ORIGINAL load+
        // software-max path: per-tile work is then a strict subset of both
        // parents (unmasked: fused load, no software max; masked: byte-for-
        // byte the production path).
        float tm;
#ifdef TC2P_REDMAXU
        if (!masked) {
          tm = tmem_ld64_redmax(&sv2[0][0], ts);
          tmem_wait_ld();
#ifdef TC2_TIMING
          if (timed_s) { long long t = clock64(); sm_sld += t - c0; c0 = t; }
#endif
        } else {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc)
            tmem_ld32_nw(sv2[cc], ts + cc * 32);
          tmem_wait_ld();
#ifdef TC2_TIMING
          if (timed_s) { long long t = clock64(); sm_sld += t - c0; c0 = t; }
#endif
          {
            const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
            const int jlo = windowed ? max(0, vis - window_left) : 0;
            float pm = NEG_BIG;
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              #pragma unroll
              for (int x = 0; x < 32; ++x) {
                int j = n0 + cc * 32 + x;
                pm = fmaxf(pm, (j >= jlo && j < jhi) ? sv2[cc][x] : NEG_BIG);
              }
            }
            tm = pm;
          }
        }
#else
        tm = tmem_ld64_redmax(&sv2[0][0], ts);
        tmem_wait_ld();
#ifdef TC2_TIMING
        if (timed_s) { long long t = clock64(); sm_sld += t - c0; c0 = t; }
#endif
        if (masked) {
          // The fused reduction is an UPPER bound, not the row max: it also
          // sees the columns causality hides. ex2.approx.f16x2 underflows to
          // EXACT ZERO below 2^-24, so an inflated max zeroes every
          // probability of the tile (tc9 law, dv_b2a40f651e23 measured 8/10
          // fixed cases wrong without the recompute). Masked tiles take the
          // true max over the visible columns from the already-loaded
          // values; the load itself is unchanged.
          tm = NEG_BIG;
          const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
          const int jlo = windowed ? max(0, vis - window_left) : 0;
#ifdef TC2_MAXTREE
          {
            float m32[32];
            #pragma unroll
            for (int x = 0; x < 32; ++x) {
              const int ja = n0 + x, jb = n0 + 32 + x;
              const float va = (ja >= jlo && ja < jhi) ? sv2[0][x] : NEG_BIG;
              const float vb = (jb >= jlo && jb < jhi) ? sv2[1][x] : NEG_BIG;
              m32[x] = fmaxf(va, vb);
            }
            #pragma unroll
            for (int st = 16; st >= 1; st >>= 1) {
              #pragma unroll
              for (int x = 0; x < st; ++x)
                m32[x] = fmaxf(m32[x], m32[x + st]);
            }
            tm = m32[0];
          }
#else
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            float pm = NEG_BIG;
            #pragma unroll
            for (int x = 0; x < 32; ++x) {
              int j = n0 + cc * 32 + x;
              pm = fmaxf(pm, (j >= jlo && j < jhi) ? sv2[cc][x] : NEG_BIG);
            }
            tm = fmaxf(tm, pm);
          }
#endif
        }
#endif  // TC2P_REDMAXU
#else
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc)
          tmem_ld32_nw(sv2[cc], ts + cc * 32);
        tmem_wait_ld();
#ifdef TC2_TIMING
        if (timed_s) { long long t = clock64(); sm_sld += t - c0; c0 = t; }
#endif

        float tm = NEG_BIG;
        if (!masked) {
#ifdef TC2_MAXTREE
          // Balanced max tree (probe32): fold the two cc halves, then 5
          // in-place rounds with full ILP (~30-cycle latency chain vs
          // ~250 for the serial 63-fmax accumulator; sm_max measured
          // 378-422 cyc/tile on f1, 752-793 on d7 - latency-dominated).
          // BIT-EXACT: fmaxf is exact and max is associative/commutative.
          {
            float m32[32];
            #pragma unroll
            for (int x = 0; x < 32; ++x)
              m32[x] = fmaxf(sv2[0][x], sv2[1][x]);
            #pragma unroll
            for (int st = 16; st >= 1; st >>= 1) {
              #pragma unroll
              for (int x = 0; x < st; ++x)
                m32[x] = fmaxf(m32[x], m32[x + st]);
            }
            tm = m32[0];
          }
#else
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            float pm = sv2[cc][0];
            #pragma unroll
            for (int x = 1; x < 32; ++x) pm = fmaxf(pm, sv2[cc][x]);
            tm = fmaxf(tm, pm);
          }
#endif
        } else {
#ifdef TC2_MAXTREE
          {
            const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
            const int jlo = windowed ? max(0, vis - window_left) : 0;
            float m32[32];
            #pragma unroll
            for (int x = 0; x < 32; ++x) {
              const int ja = n0 + x, jb = n0 + 32 + x;
              const float va = (ja >= jlo && ja < jhi) ? sv2[0][x] : NEG_BIG;
              const float vb = (jb >= jlo && jb < jhi) ? sv2[1][x] : NEG_BIG;
              m32[x] = fmaxf(va, vb);
            }
            #pragma unroll
            for (int st = 16; st >= 1; st >>= 1) {
              #pragma unroll
              for (int x = 0; x < st; ++x)
                m32[x] = fmaxf(m32[x], m32[x + st]);
            }
            tm = m32[0];
          }
#else
          const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
          const int jlo = windowed ? max(0, vis - window_left) : 0;
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            float pm = NEG_BIG;
#ifdef TC2P_MSKIPM
            // E25/E26: a fully hidden 32-column block contributes the fmaxf
            // IDENTITY (NEG_BIG) on all 32 lanes, so skipping the scan is
            // bit-exact. Split out from E25's TC2P_MSKIP (whose exp-side
            // `continue` form regressed this kernel 1.05-1.43 - codegen, not
            // mechanism) so the max-side guard can be attributed alone.
            const int jb0 = n0 + cc * 32;
            if (jb0 < jhi && !(windowed && jb0 + 31 < jlo))
#endif
            #pragma unroll
            for (int x = 0; x < 32; ++x) {
              int j = n0 + cc * 32 + x;
              pm = fmaxf(pm, (j >= jlo && j < jhi) ? sv2[cc][x] : NEG_BIG);
            }
            tm = fmaxf(tm, pm);
          }
#endif
        }
#endif  // TC2P_REDMAX
        // lagged running max (threshold 2^8 in scaled log2 domain)
        float nr = fmaxf(mr, tm);
        if ((nr - mr) * sc2 <= LAG_MAX_LOG2) nr = mr;
        const float nm2 = nr * sc2;
        const float alpha = (nr == mr) ? 1.0f : ex2_approx(fmaf(mr, sc2, -nm2));
        mr = nr;
#ifdef TC2_TIMING
        if (timed_s) { long long t = clock64(); sm_max += t - c0; c0 = t; }
#ifdef TC2_TIMING4
        const long long t4s = c0;
        long long t4a = c0, t4b = c0;
#endif
#endif

        float pc = 0.0f;
        if (!masked) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            uint32_t pr[16];
            #pragma unroll
            for (int x2 = 0; x2 < 16; ++x2) {
#if defined(TC2_EX2BF16)
              const uint32_t pb = ex2_bf16x2(cvt_bf16x2(
                  fmaf(sv2[cc][x2 * 2], sc2, -nm2),
                  fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2)));
              pr[x2] = pb;
              pc += bf16lo_f32(pb) + bf16hi_f32(pb);
#elif defined(TC2_EX2F16)
              const uint32_t pf = ex2_f16x2(cvt_f16x2(
                  fmaf(sv2[cc][x2 * 2], sc2, -nm2),
                  fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2)));
              const float p0 = f16lo_f32(pf), p1 = f16hi_f32(pf);
              pc += p0 + p1;
              pr[x2] = pack_bf16(p0, p1);
#else
              float p0 = ex2_approx(fmaf(sv2[cc][x2 * 2], sc2, -nm2));
              float p1 = ex2_approx(fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2));
              pc += p0 + p1;
              pr[x2] = pack_bf16(p0, p1);
#endif
            }
            tmem_st16bf(ts + cc * 16, pr);
          }
        } else {
          const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
          const int jlo = windowed ? max(0, vis - window_left) : 0;
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            uint32_t pr[16];
#ifdef TC2P_MSKIPE
            // Fully hidden 32-column block: every lane would take the -200
            // sentinel, whose ex2.approx.f16x2 flushes to EXACT +0 (the
            // dv_b2a40f651e23 underflow law), pc += 0.0f + 0.0f is a bit
            // identity (pc starts at +0.0f and only ever grows), and
            // pack_bf16(0,0) == 0 - zeroing pr is bit-exact. E26 form:
            // single tmem_st16bf site kept below, if/else instead of E25's
            // `continue` (whose duplicated store site + early exit regressed
            // this kernel 1.05-1.43 on codegen, not mechanism).
            const int jb0 = n0 + cc * 32;
            if (jb0 >= jhi || (windowed && jb0 + 31 < jlo)) {
              #pragma unroll
              for (int x2 = 0; x2 < 16; ++x2) pr[x2] = 0u;
            } else {
#endif
            #pragma unroll
            for (int x2 = 0; x2 < 16; ++x2) {
#if defined(TC2_EX2BF16) || defined(TC2_EX2F16)
              // out-of-range -> sentinel far below the f16/bf16 exp range;
              // ex2.approx.ftz flushes it to exact zero
              const int j0 = n0 + cc * 32 + x2 * 2;
              const float a0 = (j0 >= jlo && j0 < jhi)
                                   ? fmaf(sv2[cc][x2 * 2], sc2, -nm2) : -200.0f;
              const float a1 = (j0 + 1 >= jlo && j0 + 1 < jhi)
                                   ? fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2) : -200.0f;
#ifdef TC2_EX2BF16
              const uint32_t pb = ex2_bf16x2(cvt_bf16x2(a0, a1));
              pr[x2] = pb;
              pc += bf16lo_f32(pb) + bf16hi_f32(pb);
#else
              const uint32_t pf = ex2_f16x2(cvt_f16x2(a0, a1));
              const float p0 = f16lo_f32(pf), p1 = f16hi_f32(pf);
              pc += p0 + p1;
              pr[x2] = pack_bf16(p0, p1);
#endif
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
              pc += p0 + p1;
              pr[x2] = pack_bf16(p0, p1);
#endif
            }
#ifdef TC2P_MSKIPE
            }
#endif
            tmem_st16bf(ts + cc * 16, pr);
          }
        }
#ifdef TC2_TIMING4
        if (timed_s) t4a = clock64();
#endif
        tmem_wait_st();
#ifdef TC2_TIMING4
        if (timed_s) t4b = clock64();
#endif
        fence_before_sync();
        mbar_arrive(bar(ctrl, pfB + st));
        d = d * alpha + pc;
#ifdef TC2_TIMING
        if (timed_s) { long long t = clock64(); sm_exp += t - c0; c0 = t; }
#endif
#ifdef TC2_TIMING4
        if (timed_s) {
          smx_loop += t4a - t4s;
          smx_wst += t4b - t4a;
          smx_tail += c0 - t4b;
        }
#endif

        // O correction (skip tile 0 of the ITEM; skip when alpha==1 warp-
        // wide). Fast-arrive (opts bit0, always safe in dual-issuer mode):
        // with no correction the warp touches nothing in O, so OC arrives
        // without waiting for PV(s,i-1) completion.
        const bool need_corr =
            (i > 0) && __any_sync(0xffffffffu, alpha != 1.0f);
#ifdef TC2_TIMING
        long long cw = c0;
#endif
        if (need_corr || (i > 0 && !(opts & 1))) {
          const int ip = Cg - 1;
          mbar_wait_h(bar(ctrl, pvB + (ip & 1)), (int)((ip / 2) & 1), whint);
          fence_after_sync();
#ifdef TC2_TIMING
          if (timed_s) { long long t = clock64(); sm_bew += t - cw; cw = t; }
#endif
        }
        if (need_corr) {
          const uint32_t to = tmem + tcol_o + lb;
          #pragma unroll
          for (int half = 0; half < 2; ++half) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc)
              tmem_ld32_nw(sv2[cc], to + (half * 2 + cc) * 32);
            tmem_wait_ld();
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              #pragma unroll
              for (int x = 0; x < 32; ++x) sv2[cc][x] *= alpha;
              tmem_st32(to + (half * 2 + cc) * 32, sv2[cc]);
            }
            tmem_wait_st();
          }
        }
        fence_before_sync();
        mbar_arrive(bar(ctrl, ocB));
#ifdef TC2_TIMING
        if (timed_s) {
          long long t = clock64(); sm_corr += t - cw; sm_tiles += 1;
        }
#endif
      }
      C += Ts;

      // ---- epilogue (per item) ----
      if (Ts > 0) {
#ifdef TC2_TIMING
        const unsigned long long t_ep = timed_s ? clock64() : 0ull;
#endif
        const int cl = C - 1;                 // global last tile idx
        mbar_wait_h(bar(ctrl, pvB + (cl & 1)), (int)((cl / 2) & 1), whint);
        fence_after_sync();
        float inv = d > 0.0f ? 1.0f / d : 0.0f;
        float lse2 = mr * sc2 + log2f(d);
        float extra = 1.0f;
        if (has_sink) {
          float t = fmaxf(lse2, sink2) + log2f(1.0f + exp2f(-fabsf(lse2 - sink2)));
          extra = exp2f(lse2 - t);
          lse2 = t;
        }
        const float gsc = bmm2_scale * inv * extra;
        const bool active = (grow < qlen);
        const uint32_t to = tmem + tcol_o + lb;
        __nv_bfloat16* orow =
            outp + static_cast<int64_t>(it.m0 + s * SUB + row) * o_stride;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc)
            tmem_ld32_nw(sv2[cc], to + (half * 2 + cc) * 32);
          tmem_wait_ld();
          if (half == 1) {
            // All O reads of this item are done; release the accumulator to
            // the next item's first PV (stores below are global-only).
            fence_before_sync();
            mbar_arrive(bar(ctrl, odrB));
          }
          if (active) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              __align__(16) uint32_t pk[16];
              #pragma unroll
              for (int x2 = 0; x2 < 16; ++x2)
                pk[x2] = pack_bf16(sv2[cc][x2 * 2] * gsc,
                                   sv2[cc][x2 * 2 + 1] * gsc);
              // 256-bit vector stores (sm_100+, PTX ISA >= 8.8): each
              // instruction covers a full 32B sector pair; the previous
              // 4x uint4 (16B) pattern wrote half-sectors (ncu: 49%
              // excessive store sectors on the epilogue).
              uint32_t* dstv =
                  reinterpret_cast<uint32_t*>(orow + (half * 2 + cc) * 32);
              asm volatile(
                  "st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::
                      "l"(dstv), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]),
                      "r"(pk[3]), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]),
                      "r"(pk[7]) : "memory");
              asm volatile(
                  "st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::
                      "l"(dstv + 8), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]),
                      "r"(pk[11]), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]),
                      "r"(pk[15]) : "memory");
            }
          }
        }
        if (active && has_lse) {
          lsep[static_cast<int64_t>(it.m0 + s * SUB + row) * heads] = lse2;
        }
#ifdef TC2_TIMING
        if (timed_s)
          atomicAdd(dbg + 52, (unsigned long long)(clock64() - t_ep));
#endif
      } else {
        // Ts == 0: O region untouched this item; still advance the phase.
        fence_before_sync();
        mbar_arrive(bar(ctrl, odrB));
      }
      ++f;
    }
#ifdef TC2_TIMING
    if (timed_s) {
      atomicAdd(dbg + 16, (unsigned long long)sm_tiles);
      atomicAdd(dbg + 17, (unsigned long long)sm_sld);
      atomicAdd(dbg + 18, (unsigned long long)sm_max);
      atomicAdd(dbg + 19, (unsigned long long)sm_exp);
      atomicAdd(dbg + 20, (unsigned long long)sm_bew);
      atomicAdd(dbg + 21, (unsigned long long)sm_corr);
      atomicAdd(dbg + 22, (unsigned long long)sm_per);
#ifdef TC2_TIMING4
      atomicAdd(dbg + 55, (unsigned long long)smx_loop);
      atomicAdd(dbg + 56, (unsigned long long)smx_wst);
      atomicAdd(dbg + 57, (unsigned long long)smx_tail);
#endif
#ifdef TC2_TIMING5
      atomicAdd(dbg + 58, (unsigned long long)sm_sfw2);
#endif
#ifdef TC2_TIMING2
      atomicAdd(dbg + 33, (unsigned long long)sm_next);
      atomicAdd(dbg + 34, (unsigned long long)sm_items);
#endif
    }
#endif
  }

  if (dbg && tid == 0) atomicAdd(dbg + 5, 1ull);
  __syncthreads();
  if (warp == 8) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 ::"r"(tmem), "r"(512) : "memory");
  }
  if (dbg && tid == 0) atomicAdd(dbg + 6, 1ull);
#ifdef TC2_TIMING
  if (dbg && tid == 0) {
    const unsigned long long sp = clock64() - t_ent;
    atomicAdd(dbg + 48, sp);            // span sum (CTAs)
    atomicAdd(dbg + 49, 1ull);          // CTA count
    atomicMax(dbg + 50, sp);            // worst CTA span
  }
#endif
}

}  // namespace mla_tc2p

// ---------------------------------------------------------------------------
// Host: rank-4 direct maps + persistent launch.
// ---------------------------------------------------------------------------
namespace {
struct MapKey4 {
  const void* ptr; uint64_t tokens, atoms, heads; int box_tok;
  bool operator==(const MapKey4& o) const {
    return ptr == o.ptr && tokens == o.tokens && atoms == o.atoms &&
           heads == o.heads && box_tok == o.box_tok;
  }
};
struct MapEntry4 { MapKey4 key; CUtensorMap map; bool valid = false; };
MapEntry4 g_cache4[16];
int g_next4 = 0;

// Identical rank-4 encoding to kernel_tc2.cu (box8 KV, box128 Q).
int encode4(CUtensorMap* m, const void* ptr, uint64_t tokens, uint64_t heads,
            uint64_t atoms, int box_tok) {
  cuuint64_t gdim[4] = {64, tokens, atoms, heads};
  cuuint64_t gstr[3] = {heads * atoms * 128, 128, atoms * 128};
  cuuint32_t box[4] = {64, static_cast<cuuint32_t>(box_tok),
                       static_cast<cuuint32_t>(atoms), 1};
  cuuint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(ptr),
      gdim, gstr, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
#if defined(TC2_PROMO256)
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
#elif defined(TC2_PROMO_NONE)
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
#else
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
#endif
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return static_cast<int>(r);
}
int cached4(CUtensorMap* m, const void* ptr, uint64_t tokens, uint64_t heads,
            uint64_t atoms, int box_tok) {
  MapKey4 key{ptr, tokens, atoms, heads, box_tok};
  for (auto& e : g_cache4) {
    if (e.valid && e.key == key) { *m = e.map; return 0; }
  }
  int rc = encode4(m, ptr, tokens, heads, atoms, box_tok);
  if (rc == 0) {
    MapEntry4& e = g_cache4[g_next4];
    g_next4 = (g_next4 + 1) & 15;
    e.key = key; e.map = *m; e.valid = true;
  }
  return rc;
}
unsigned* g_counter = nullptr;
int g_sms = 0;
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
  (void)kws; (void)vws; (void)kvblks; (void)max_kv_len; (void)dbg_mode;
  (void)pf_mode; (void)lag_max; (void)direct; (void)box_tok;
  using namespace mla_tc2p;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  static bool attr_set = false;
  if (!attr_set) {
    // One-time init; always runs on the first plain launch (solution.py
    // executes fn(*args) directly before any graph capture can occur).
    cudaError_t e = cudaFuncSetAttribute(mla_tc2p_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return static_cast<int>(e);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 9002;
    if (cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess)
      return 9003;
    if (g_sms <= 0) g_sms = 148;
    if (cudaMalloc(&g_counter, 16) != cudaSuccess) return 9001;
    cudaMemset(g_counter, 0, 16);   // one-time zero; kernel self-resets after
    attr_set = true;
  }
  int tiles_ub = total_q / BM + batch;
  if (tiles_ub <= 0) return static_cast<int>(cudaGetLastError());
  const int total_items = tiles_ub * heads;
  int grid = total_items < g_sms ? total_items : g_sms;
  if (grid <= 0) return 5001;

  CUtensorMap krmap, vrmap, qrmap;
  int rc;
#ifdef TC2P_BOX64
  const int kvbox = 64;
#else
  const int kvbox = 8;
#endif
  if ((rc = cached4(&krmap, k, (uint64_t)total_kv, (uint64_t)kv_heads, 3,
                    kvbox)))
    return 5000 + rc;
  if ((rc = cached4(&vrmap, v, (uint64_t)total_kv, (uint64_t)kv_heads, 2,
                    kvbox)))
    return 6000 + rc;
  if ((rc = cached4(&qrmap, q, (uint64_t)total_q, (uint64_t)heads, 3, 128)))
    return 7000 + rc;

  mla_tc2p_kernel<<<dim3(grid), THREADS, SMEM_BYTES, s>>>(
      krmap, vrmap, qrmap, static_cast<const __nv_bfloat16*>(q),
      static_cast<__nv_bfloat16*>(out), static_cast<float*>(lse),
      static_cast<const float*>(sinks),
      static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
      total_q, batch, heads, kv_heads, window_left, causal, has_lse, has_sink,
      bmm1_scale, bmm2_scale, tiles_ub, g_counter, opts,
      reinterpret_cast<unsigned long long*>(dbg));
  return static_cast<int>(cudaGetLastError());
}
