// mla_expanded_bf16 - tcgen05 CLUSTER split-KV dual-subtile kernel (tc2c).
// Derived from kernel_tc2s.cu (gmem-merge split-KV, retired by probe22:
// base machinery +10.3% vs split gain -7.3% on f9 - the gmem merge cost
// ep0 11.3k + ep1 28.2k cyc). probe34 validated the two facts this design
// needs on SM103: DSMEM exchange is contention-free at 74 concurrent
// clusters (128KB remote read 7.1k cyc, +0.4% vs 1 cluster; cluster_sync
// 396 cyc; remote mbarrier ping-pong 356 cyc; cluster launch +1.2us).
//
// Architecture (differences from tc2s in CAPS):
//   * NO PREP KERNEL, NO TICKETS, NO GMEM WS. Each CTA redundantly runs
//     PHASE0 (one elect-lane of the loader warp, ~1k cyc, before
//     tcgen05.alloc so idle CTAs exit cheap): replicate the tc2s_prep cut
//     scan (Tl[<=16], hist, smallest cut>=4 with units<=gridDim.x), then
//     CLOSED-FORM decode of this CTA's unit from u = blockIdx.x.
//     ENUMERATION: split pairs first, head-blocked (u < 2*sp*heads:
//     h = u/(2sp), pair p = (u%(2sp))/2, part = u&1 - pairs are always
//     cluster-aligned because the block size 2*sp is even), then unsplit
//     units (h = u'/(R-sp), j = u'%(R-sp)). f9: 16 pairs + 112 unsplit,
//     U=144 of 148 CTAs. Host launches EXACTLY gridDim = 148 as 74
//     clusters of 2: unit u -> cluster u/2, rank u&1. Idle units (u>=U)
//     exit before init/alloc; a dangling rank1 (U odd) exits without ever
//     touching a cluster op (unsplit units use none), so asymmetric exit
//     is safe. Fallback: no feasible cut -> U = real*heads unsplit units
//     (= tc2 semantics + phase0 cost).
//   * Main loop inherited from tc2s unchanged (windowed Ts/Tf, absolute
//     n0 masking, NEXT/CONS/QF/ODR ring algebra); with one unit per CTA
//     the f>=1 fetch degenerates to an immediate invalid publish.
//   * DSMEM MERGE EPILOGUE, v2 BIDIRECTIONAL (v1's serial one-way round
//     trip measured +10.2% on f9: sigwait 8.3k + read/merge ~18k ate the
//     whole split saving, span_max 80.2k == unsplit tc2; probe35
//     dv_330207a67702). v2: each part EXPORTS its own subtile (s==part)
//     and MERGES the other (s==1-part): two parallel 64KB exchanges.
//     EXPORT: after BOTH subtiles' final PV retire (payload half s spans
//     sq_s + freed K stages; md in PAY_MD_OFF; cross-wait skipped when
//     Ts_other==0), tmem_ld own O(s) -> STS chunk-blocked fp32 payload
//     (s*64KB + ((half*2+cc)*SUB + row)*128B), __threadfence +
//     __syncwarp, 4 lane0 remote-arrives into the PEER's SIG0 (count 4);
//     then wait own SIG1 (ACK) before ODR/exit - smem lifetime guard.
//     MERGE: own pvB wait, full-warp try_wait.acquire.cluster on own
//     SIG0, ld.shared::cluster.v4 remote reads of the peer's payload(s)
//     + md feed the UNCHANGED tc2s merge math + sink/lse/store tail;
//     __syncwarp + 4 lane0 remote-arrives into the peer's SIG1 (ACK).
//     Exchange predicate is LOCAL and symmetric: bothx = Ts>0 && sib_s>0
//     (sib_s = peer's window count for subtile s). No cluster_sync in the
//     steady-state path (only the post-init one, before any exit).
//   * POLICY: shipped with op198 (conservative OC): split part1 windows
//     are causal + asymmetric (Ts0<Ts1) at T~16 - d7's exact bit0-race
//     trigger geometry (results/bit0_race_notes.md) - so bit0 stays gated
//     until stress-proven on tc2c itself.
//
// Numerics: per-tile math identical to tc2s/tc2 v1 (f32 ex2 softmax);
// split merge exact up to fp32 recombination rounding; transport change
// (gmem -> DSMEM) is numerics-inert. Not bit-identical to tc2 (split
// recombination order) - gated by tolerance numerics like tc2s was.
// Build: nvcc -gencode=arch=compute_103a,code=sm_103a -lcuda.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define DIN __device__ __forceinline__

namespace mla_tc2c {

constexpr int DQK = 192;
constexpr int DV = 128;
constexpr int BM = 256;
constexpr int SUB = 128;
constexpr int BN = 64;
constexpr int KV_STAGES = 3;
constexpr int THREADS = 352;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -1.0e30f;
constexpr float LAG_MAX_LOG2 = 8.0f;

constexpr int Q_BYTES = SUB * DQK * 2;          // 49152
constexpr int Q_TOTAL = BM * DQK * 2;           // 98304
constexpr int K_BYTES = BN * DQK * 2;           // 24576
constexpr int V_BYTES = BN * DV * 2;            // 16384
constexpr int KV_TOTAL = KV_STAGES * (K_BYTES + V_BYTES);   // 122880
constexpr int CTRL_OFF = Q_TOTAL + KV_TOTAL;    // 221184
// ctrl: 28 mbarriers (224B) + tmem holder (8B at 224) + items ring
// (2 x 72B at 232..376) + phase0 scratch (pl_Tl 64B at 512, pl_hist 264B
// at 576). Region 1024B. NOTE: NO static __shared__ anywhere in this
// kernel - static shared shifts the dynamic smem base off the 128B
// alignment the swizzled TMA boxes require (probe36/37: UTMALDG.4D
// misaligned-address fault at +0x4690 on every stage).
constexpr int ITEMS_OFF = 232;     // 2 x 72B ItemGeom = 144B (232..376)
constexpr int PLTL_OFF = 512;      // phase0 Tl[16]
constexpr int PLHIST_OFF = 576;    // phase0 hist[66]
constexpr int SMEM_BYTES = CTRL_OFF + 1024;
// DSMEM merge payload (part0 only; epilogue-time reuse of the sq+sk span):
// O partials at [0, PAY_O_BYTES), chunk-blocked ((half*2+cc)*SUB + row)*32
// floats per subtile half (s*SUB*DV*4 base); md at PAY_MD_OFF, 2 floats per
// (subtile,row). Region fully free only after BOTH subtiles' final PV.
constexpr int PAY_O_BYTES = 2 * SUB * DV * 4;   // 131072
constexpr int PAY_MD_OFF = PAY_O_BYTES;         // 2KB

constexpr int MAX_SLOTS = 148;      // dispatch gate: tiles_ub*heads <= 148
constexpr int MAX_UNITS = 2 * MAX_SLOTS;

DIN int TCOL_S(int s, int st) { return s * 128 + st * 64; }
DIN int TCOL_O(int s) { return 256 + s * 128; }

enum { BF0 = 0, BF1, BF2, BE0, BE1, BE2,
       SF00, SF01, SF10, SF11,
       PF00, PF01, PF10, PF11,
       PV00, PV01, PV10, PV11,
       OC0, OC1, QF0, QF1, NEXT, CONS, ODR0, ODR1, SIG0, SIG1, NBARS };
DIN uint32_t sm32_(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
DIN uint32_t bar(char* ctrl, int idx) { return sm32_(ctrl) + idx * 8; }

// ---------------------------------------------------------------------------
// PTX primitives (identical to kernel_tc2p.cu / kernel_tc2.cu)
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
DIN void load_kv_tile(const CUtensorMap& krmap, const CUtensorMap& vrmap,
                      uint32_t kd, uint32_t vd, uint32_t mbar, int tok,
                      int h_kv) {
  #pragma unroll
  for (int g = 0; g < 8; ++g) {
    tma_load_4d(&krmap, kd + g * 3072, mbar, 0, tok + g * 8, 0, h_kv);
    tma_load_4d(&vrmap, vd + g * 2048, mbar, 0, tok + g * 8, 0, h_kv);
  }
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
// ---- cluster/DSMEM primitives (tc3-proven forms; probe34-validated) ----
DIN uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DIN uint32_t mapa(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(remote) : "r"(local_addr), "r"(rank));
  return remote;
}
// remote arrive (release at cluster scope; default semantics of the op)
DIN void mbar_arrive_rank(uint32_t local_a, uint32_t rank) {
  const uint32_t remote = mapa(local_a, rank);
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(remote)
               : "memory");
}
// acquire.cluster wait: orders the peer's released writes for this waiter
DIN void mbar_wait_cl(uint32_t a, int phase, uint32_t hint) {
  asm volatile(
      "{\n\t.reg .pred P;\n\tLABWC%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], %1, %2;\n\t"
      "@P bra.uni DONEWC%=;\n\tbra.uni LABWC%=;\n\tDONEWC%=:\n\t}\n"
      ::"r"(a), "r"(phase), "r"(hint)
      : "memory");
}
DIN void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
DIN void ld_remote_v4(uint32_t a, uint4& u) {
  asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w)
               : "r"(a) : "memory");
}
DIN float ld_remote_f32(uint32_t a) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(a)
               : "memory");
  return v;
}
DIN uint64_t make_desc(uint32_t smem_addr, uint32_t lbo_bytes,
                       uint32_t sbo_bytes) {
  uint32_t lo = ((smem_addr >> 4) & 0x3FFF) | ((lbo_bytes >> 4) << 16);
  uint32_t hi = (2u << 29) | (1u << 14) | ((sbo_bytes >> 4) & 0x3FFF);
  return (static_cast<uint64_t>(hi) << 32) | lo;
}
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

// ---------------------------------------------------------------------------
// Unit table (written by the prep kernel) and item geometry
// ---------------------------------------------------------------------------
struct UnitEntry {
  int tile_id;
  int h;
  int k0k1;      // k0 | (k1 << 16), absolute KV tile window [k0, k1)
  int flags;     // bit0 is_split, bit1 part
};

struct ItemGeom {
  int valid;
  int q_base;
  int tok0;
  int m0;
  int qlen;
  int kvlen;
  int Ts0, Tf0;   // split-windowed subtile-0 counts
  int Ts1, Tf1;
  int T;          // max(Ts0, Ts1) for this unit
  int h;
  int k0;         // absolute KV tile offset (masking n0)
  int is_split;
  int part;
  int slot;       // tile_id*heads + h (< 148 by the dispatch gate)
  int sib0, sib1; // sibling part's tile count per subtile (merge verdict)
};

DIN void decode_unit(const int32_t* __restrict__ cum_q,
                     const int32_t* __restrict__ cum_kv, int batch,
                     int causal, int window_left, int opts, int heads,
                     const UnitEntry& e, ItemGeom& it) {
  const int k0 = e.k0k1 & 0xFFFF;
  const int k1 = e.k0k1 >> 16;
  it.h = e.h;
  it.k0 = k0;
  it.is_split = e.flags & 1;
  it.part = (e.flags >> 1) & 1;
  it.slot = e.tile_id * heads + e.h;
  it.valid = 0;
  it.Ts0 = 0; it.Ts1 = 0; it.T = 0; it.Tf0 = 0; it.Tf1 = 0;
  it.q_base = 0; it.tok0 = 0; it.m0 = 0; it.qlen = 0; it.kvlen = 0;
  // batch mapping for tile_id (same scan as tc2/tc2p; the prep kernel put
  // only real tiles in the table, so b >= 0 is guaranteed, but the scan
  // stays defensive)
  int b = -1, m0 = 0, q_base = 0, tok0 = 0, qlen = 0, kvlen = 0;
  {
    int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
    for (int i = 0; i < batch; ++i) {
      int nq = cum_q[i + 1], nkv = cum_kv[i + 1];
      int len_q = nq - prev_q;
      int tiles = (len_q + BM - 1) / BM;
      if (b < 0 && e.tile_id < acc + tiles) {
        const int local = e.tile_id - acc;
        b = i;
        m0 = ((opts & 2) ? (tiles - 1 - local) : local) * BM;
        q_base = prev_q; tok0 = prev_kv;
        qlen = len_q; kvlen = nkv - prev_kv;
        break;
      }
      acc += tiles; prev_q = nq; prev_kv = nkv;
    }
  }
  if (b < 0) return;
  it.valid = 1;
  it.m0 = m0; it.q_base = q_base; it.tok0 = tok0;
  it.qlen = qlen; it.kvlen = kvlen;
  const int kv_off = kvlen - qlen;
  const bool windowed = window_left >= 0;
  const bool has_sub0 = (m0 < qlen);
  const bool has_sub1 = (m0 + SUB < qlen);
  int Ts0f, Ts1f, Tf0f, Tf1f;
  {
    int nmax = causal ? min(kvlen, kv_off + m0 + SUB) : kvlen;
    int nsafe = causal ? min(kvlen, kv_off + m0 + 1) : kvlen;
    if (windowed) nsafe = 0;
    if (!has_sub0) { nmax = 0; nsafe = 0; }
    nmax = min(nmax, kvlen);
    Ts0f = (nmax + BN - 1) / BN;
    Tf0f = nsafe / BN;
  }
  {
    int nmax = causal ? min(kvlen, kv_off + m0 + 2 * SUB) : kvlen;
    int nsafe = causal ? min(kvlen, kv_off + m0 + SUB + 1) : kvlen;
    if (windowed) nsafe = 0;
    if (!has_sub1) { nmax = 0; nsafe = 0; }
    nmax = min(nmax, kvlen);
    Ts1f = (nmax + BN - 1) / BN;
    Tf1f = nsafe / BN;
  }
  // apply the split window [k0, k1)
  const int ks0 = min(k0, Ts0f), ke0 = min(k1, Ts0f);
  const int ks1 = min(k0, Ts1f), ke1 = min(k1, Ts1f);
  (void)ks0; (void)ks1;
  it.Ts0 = ke0 - ks0;
  it.Ts1 = ke1 - ks1;
  it.Tf0 = min(max(Tf0f - k0, 0), it.Ts0);
  it.Tf1 = min(max(Tf1f - k0, 0), it.Ts1);
  it.T = max(it.Ts0, it.Ts1);
  // sibling part's per-subtile window size -> the merge verdict is LOCAL:
  // part0 exports iff its sibling holds data; part1 merges iff it does.
  if (!it.is_split) {
    it.sib0 = 0;
    it.sib1 = 0;
  } else if (it.part == 0) {
    it.sib0 = max(Ts0f - k1, 0);
    it.sib1 = max(Ts1f - k1, 0);
  } else {
    it.sib0 = min(k0, Ts0f);
    it.sib1 = min(k0, Ts1f);
  }
  if (it.T <= 0) it.valid = 0;
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS, 1)
mla_tc2c_kernel(const __grid_constant__ CUtensorMap krmap,
                const __grid_constant__ CUtensorMap vrmap,
                const __grid_constant__ CUtensorMap qrmap,
                __nv_bfloat16* __restrict__ out,
                float* __restrict__ lse,
                const float* __restrict__ sinks,
                const int32_t* __restrict__ cum_q,
                const int32_t* __restrict__ cum_kv,
                int32_t batch, int32_t heads, int32_t kv_heads,
                int32_t window_left, int32_t causal, int32_t has_lse,
                int32_t has_sink, float bmm1_scale, float bmm2_scale,
                int32_t opts, unsigned long long* __restrict__ dbg) {
  extern __shared__ char smem[];
  char* sq = smem;
  char* sk = smem + Q_TOTAL;
  char* sv = sk + KV_STAGES * K_BYTES;
  char* ctrl = smem + CTRL_OFF;
  uint32_t* tholder = reinterpret_cast<uint32_t*>(ctrl + NBARS * 8);
  ItemGeom* items = reinterpret_cast<ItemGeom*>(ctrl + ITEMS_OFF);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const uint32_t crank = cluster_rank();
  const uint32_t whint = (opts & 8) ? 0u : 0x989680u;
  const int group_div = heads / kv_heads;
  const float sc2 = bmm1_scale * LOG2E;

#ifdef TC2_TIMING
  const unsigned long long t_ent = (tid == 0) ? clock64() : 0ull;
#endif
  // ---- PHASE0: in-kernel split plan + static unit decode (no prep
  // kernel, no tickets). One elect lane of the loader warp replicates the
  // tc2s_prep cut scan (uniform across CTAs), then closed-form decodes
  // this CTA's unit from u = blockIdx.x with splits enumerated FIRST so
  // every (part0,part1) pair is cluster-aligned (see file header). Idle
  // CTAs (u >= U; whole clusters, except a possible odd-U rank1 tail that
  // touches no cluster op) exit HERE, before barrier init and
  // tcgen05.alloc (probe18: ~1us sentinel vs full init/alloc churn).
  if (warp == 9 && elect_one()) {
    int* pl_Tl = reinterpret_cast<int*>(ctrl + PLTL_OFF);
    int* pl_hist = reinterpret_cast<int*>(ctrl + PLHIST_OFF);
    const int grid_cap = static_cast<int>(gridDim.x);
    int overflow = 0;
    for (int i = 0; i < 16; ++i) pl_Tl[i] = -1;
    int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
    int tmax = 0;
    for (int b = 0; b < batch; ++b) {
      const int nq = cum_q[b + 1], nkv = cum_kv[b + 1];
      const int len_q = nq - prev_q, len_kv = nkv - prev_kv;
      const int tiles = (len_q + BM - 1) / BM;
      const int kv_off = len_kv - len_q;
      for (int local = 0; local < tiles; ++local) {
        if (acc + local >= 16) { overflow = 1; break; }
        const int m0 = ((opts & 2) ? (tiles - 1 - local) : local) * BM;
        const bool has1 = (m0 + SUB < len_q);
        int nmax = causal ? min(len_kv, kv_off + m0 + (has1 ? 2 * SUB : SUB))
                          : len_kv;
        nmax = min(nmax, len_kv);
        if (m0 >= len_q) nmax = 0;
        const int T = (nmax + BN - 1) / BN;
        pl_Tl[acc + local] = T;
        if (T > tmax) tmax = T;
      }
      acc += tiles; prev_q = nq; prev_kv = nkv;
    }
    int real = 0;
    for (int i = 0; i < 16 && !overflow; ++i)
      if (pl_Tl[i] > 0) real++;
    if (!overflow && real != acc) overflow = 1;   // T==0 tile: gate breach
    int cut = tmax + 1;                           // no splits by default
    int sp = 0;
    if (!overflow && !(opts & 16) && tmax >= 4 && tmax < 66) {
      for (int i = 0; i <= tmax; ++i) pl_hist[i] = 0;
      for (int i = 0; i < acc; ++i) pl_hist[pl_Tl[i]]++;
      int suff = 0;
      for (int c = tmax; c >= 4; --c) {
        suff += pl_hist[c];
        if ((real + suff) * heads > grid_cap) break;
        cut = c;
      }
      if (cut <= tmax)
        for (int i = 0; i < acc; ++i)
          if (pl_Tl[i] >= cut) sp++;
    }
    const int U = (real + sp) * heads;
    if (dbg) {
      dbg[72] = static_cast<unsigned long long>(U);
      dbg[73] = static_cast<unsigned long long>(cut);
      dbg[74] = static_cast<unsigned long long>(sp);
      dbg[75] = static_cast<unsigned long long>(real);
    }
    const int u = static_cast<int>(blockIdx.x);
    if (overflow || u >= U) {
      items[0].valid = 0;
    } else {
      UnitEntry e;
      int t = -1;
      if (sp > 0 && u < 2 * sp * heads) {
        const int h = u / (2 * sp);
        const int rem = u - h * (2 * sp);
        const int pi = rem >> 1, part = rem & 1;
        int cnt = 0;
        for (int i = 0; i < acc; ++i)
          if (pl_Tl[i] >= cut && cnt++ == pi) { t = i; break; }
        if (t >= 0) {
          const int T = pl_Tl[t];
          const int mid = (T + 1) >> 1;
          e.tile_id = t; e.h = h;
          e.k0k1 = part ? (mid | (T << 16)) : (mid << 16);
          e.flags = part ? 3 : 1;
        }
      } else {
        const int up = u - 2 * sp * heads;
        const int nus = real - sp;
        const int h = up / nus, j = up - h * nus;
        int cnt = 0;
        for (int i = 0; i < acc; ++i)
          if (pl_Tl[i] > 0 && pl_Tl[i] < cut && cnt++ == j) { t = i; break; }
        if (t >= 0) {
          e.tile_id = t; e.h = h;
          e.k0k1 = pl_Tl[t] << 16;
          e.flags = 0;
        }
      }
      if (t < 0) {
        items[0].valid = 0;              // defensive: plan/enum mismatch
      } else {
        ItemGeom it;
        decode_unit(cum_q, cum_kv, batch, causal, window_left, opts, heads,
                    e, it);
        items[0] = it;
      }
    }
  }
  __syncthreads();
  const bool have_work = (items[0].valid != 0);
  // Bars are initialized on EVERY CTA (idle ones too) and published with
  // fence.mbarrier_init + cluster_sync BEFORE any exit: this is what makes
  // part0's later remote SIG arrive formally safe against part1's init
  // (tc3 pattern), while idle CTAs still skip the tcgen05.alloc churn.
  if (tid < NBARS) {
    const int i = tid;
    int cnt = 1;
    if (i >= PF00 && i <= PF11) cnt = 128;
    else if (i == OC0 || i == OC1) cnt = 128;
    else if (i == ODR0 || i == ODR1) cnt = 128;
    else if (i == CONS) cnt = 10;
    else if (i == SIG0 || i == SIG1) cnt = 4;
    else if (i >= BE0 && i <= BE2) cnt = 2;
    mbar_init(bar(ctrl, i), cnt);
  }
  asm volatile("fence.mbarrier_init.release.cluster;");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  cluster_sync();
  if (!have_work) {
#ifdef TC2_TIMING
    if (dbg && tid == 0) atomicAdd(dbg + 66, 1ull);
#endif
    return;
  }
  if (dbg && tid == 0) atomicAdd(dbg + 8, 1ull);
  if (warp == 8) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 ::"r"(sm32_(tholder)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  fence_after_sync();
  const uint32_t tmem = *(volatile uint32_t*)tholder;

  if (warp == 9) {
    // ============ LOADER: publish units, Q(0) fill, flat KV ring ==========
    if (elect_one()) {
      int f = 0;
      long long g = 0;
#ifdef TC2_TIMING
      const unsigned long long t_ld = clock64();
#endif
      while (true) {
        ItemGeom it;
        if (f == 0) {
          it = items[0];                     // staged by the pre-init fetch
        } else {
          // tc2c: one unit per CTA. f>=1 is always the invalid publish
          // that retires every warp's item loop; the CONS wait collects
          // the f==0 arrivals first (ring algebra unchanged).
          it.valid = 0; it.T = 0; it.is_split = 0;
          mbar_wait_h(bar(ctrl, CONS), (f - 1) & 1, whint);
          items[f & 1] = it;
        }
        mbar_arrive(bar(ctrl, NEXT));
        if (!it.valid) break;
#ifdef TC2_TIMING
        if (dbg) {
          atomicAdd(dbg + 63, 1ull);
          if (it.is_split) atomicAdd(dbg + 65, 1ull);
        }
#endif
        if (f == 0) {
          mbar_expect_tx(bar(ctrl, QF0), 2 * Q_BYTES);
          tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF0), 0,
                      it.q_base + it.m0, 0, it.h);
          tma_load_4d(&qrmap, sm32_(sq) + Q_BYTES, bar(ctrl, QF0), 0,
                      it.q_base + it.m0 + SUB, 0, it.h);
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
                       it.tok0 + (it.k0 + i) * BN, h_kv);
        }
        ++f;
      }
#ifdef TC2_TIMING
      if (dbg) atomicAdd(dbg + 57, (unsigned long long)(clock64() - t_ld));
#endif
    }
  } else if (warp == 8 || warp == 10) {
    // ============ ISSUERS (identical protocol to tc2p) ====================
    if (elect_one()) {
      const bool s1warp = (warp == 10);
      const uint32_t qb = sm32_(sq) + (s1warp ? Q_BYTES : 0);
      const int kplane = 1024;
      const uint32_t ksbo = 3072u;
      const uint32_t vlbo = 1024u;
      const uint32_t vsbo = 2048u;
      const int vstep = 4096;
      const uint32_t qsbo = 1024u;
      const int qa16 = 1024;
      const uint32_t sfB = s1warp ? SF10 : SF00;
      const uint32_t pfB = s1warp ? PF10 : PF00;
      const uint32_t pvB = s1warp ? PV10 : PV00;
      const uint32_t ocB = s1warp ? OC1 : OC0;
      const uint32_t odrB = s1warp ? ODR1 : ODR0;
      const int tcol_s = s1warp ? 128 : 0;
      const int tcol_o = s1warp ? 384 : 256;
      int f = 0;
      long long Gbase = 0;
      int C = 0;
      int otherC = 0;
      int prevTs_own = 0, prevTs_other = 0;
#ifdef TC2_TIMING
      const bool timed = (dbg != nullptr);
      long long is_tiles = 0, is_bf = 0, is_bew = 0, is_qk = 0,
                is_pfw = 0, is_ocw = 0, is_pv = 0;
      long long u0 = 0, u1 = 0, u2 = 0, p0 = 0, p1 = 0, bw = 0;
#endif
      while (true) {
        mbar_wait_h(bar(ctrl, NEXT), f & 1, whint);
        fence_after_sync();
        const ItemGeom it = items[f & 1];
        mbar_arrive(bar(ctrl, CONS));
        if (!it.valid) break;
        const int Ts = s1warp ? it.Ts1 : it.Ts0;
        const int T = it.T;
        if (!s1warp) {
          if (f > 0) {
#ifdef TC2_TIMING
            const unsigned long long tr0 = timed ? clock64() : 0ull;
#endif
            if (prevTs_own > 0)
              mbar_wait_h(bar(ctrl, SF00 + ((C - 1) & 1)),
                          ((C - 1) / 2) & 1, whint);
            if (prevTs_other > 0)
              mbar_wait_h(bar(ctrl, SF10 + ((otherC - 1) & 1)),
                          ((otherC - 1) / 2) & 1, whint);
            fence_after_sync();
            mbar_expect_tx(bar(ctrl, QF0 + (f & 1)), 2 * Q_BYTES);
            tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF0 + (f & 1)), 0,
                        it.q_base + it.m0, 0, it.h);
            tma_load_4d(&qrmap, sm32_(sq) + Q_BYTES,
                        bar(ctrl, QF0 + (f & 1)), 0,
                        it.q_base + it.m0 + SUB, 0, it.h);
#ifdef TC2_TIMING
            if (timed)
              atomicAdd(dbg + 58, (unsigned long long)(clock64() - tr0));
#endif
          }
          prevTs_own = it.Ts0;
          prevTs_other = it.Ts1;
        }
        mbar_wait_h(bar(ctrl, QF0 + (f & 1)), (f >> 1) & 1, whint);
        fence_after_sync();
        for (int i = 0; i < T; ++i) {
          const long long g = Gbase + i;
          const int st3 = (int)(g % KV_STAGES);
#ifdef TC2_TIMING
          if (timed && !s1warp) { bw = 0; u0 = clock64(); }
#endif
          mbar_wait_h(bar(ctrl, BF0 + st3), (int)((g / KV_STAGES) & 1), whint);
          fence_after_sync();
#ifdef TC2_TIMING
          if (timed && !s1warp) { u1 = clock64(); is_bf += u1 - u0; }
#endif
          const int Cg = C + i;
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
          if (i > 0 && (i - 1) < Ts) {
            const int ip = Cg - 1;
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
          if (g >= 1) {
            mma_commit(bar(ctrl, BE0 + (int)((g - 1) % KV_STAGES)));
          }
        }
        {
          const int il = Ts - 1;
          if (il >= 0 && il == T - 1) {
            const int ip = C + il;
            const int stl = ip & 1;
            const long long gl = Gbase + il;
            mbar_wait_h(bar(ctrl, pfB + stl), (int)((ip / 2) & 1), whint);
            mbar_wait_h(bar(ctrl, ocB), ip & 1, whint);
            if (f > 0 && T == 1) {
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
      }
#endif
    }
  } else if (warp < 8) {
    // ============ SOFTMAX + CORRECTION + EPILOGUE (+ split merge) =========
    const int s = warp >> 2;
    const int w4 = warp & 3;
    const int lane = tid & 31;
    const int row = w4 * 32 + lane;
    const uint32_t lb = static_cast<uint32_t>(w4 * 32) << 16;
    const uint32_t sfB = s ? SF10 : SF00;
    const uint32_t pfB = s ? PF10 : PF00;
    const uint32_t pvB = s ? PV10 : PV00;
    const uint32_t ocB = s ? OC1 : OC0;
    const uint32_t odrB = s ? ODR1 : ODR0;
    const int tcol_s = s * 128;
    const int tcol_o = 256 + s * 128;
    float sv2[2][32];
    int f = 0;
    int C = 0;
#ifdef TC2_TIMING
    const bool timed_s = (dbg != nullptr) && (tid == 0);
    long long sm_tiles = 0, sm_sld = 0, sm_max = 0, sm_exp = 0,
              sm_bew = 0, sm_corr = 0, sm_per = 0;
    long long c0 = 0, cprev = 0;
#endif
    while (true) {
      mbar_wait_h(bar(ctrl, NEXT), f & 1, whint);
      fence_after_sync();
      const ItemGeom it = items[f & 1];
      if (lane == 0) mbar_arrive(bar(ctrl, CONS));
      if (!it.valid) break;
      const int grow = it.m0 + s * SUB + row;
      const int vis = (it.kvlen - it.qlen) + grow;
      const int Ts = s ? it.Ts1 : it.Ts0;
      const int Tfull = s ? it.Tf1 : it.Tf0;
      const int kvlen = it.kvlen;
      const int qlen = it.qlen;
      const int k0 = it.k0;
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
        mbar_wait_h(bar(ctrl, sfB + st), (int)((Cg / 2) & 1), whint);
        fence_after_sync();
#ifdef TC2_TIMING
        if (timed_s) {
          c0 = clock64();
          if (Cg > 0) sm_per += c0 - cprev;
          else atomicAdd(dbg + 51, (unsigned long long)(c0 - t_ent));
          cprev = c0;
        }
#endif
        const int n0 = (k0 + i) * BN;
        const bool masked = i >= Tfull;

        #pragma unroll
        for (int cc = 0; cc < 2; ++cc)
          tmem_ld32_nw(sv2[cc], ts + cc * 32);
        tmem_wait_ld();
#ifdef TC2_TIMING
        if (timed_s) { long long t = clock64(); sm_sld += t - c0; c0 = t; }
#endif
        float tm = NEG_BIG;
        if (!masked) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            float pm = sv2[cc][0];
            #pragma unroll
            for (int x = 1; x < 32; ++x) pm = fmaxf(pm, sv2[cc][x]);
            tm = fmaxf(tm, pm);
          }
        } else {
          const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
          const int jlo = windowed ? max(0, vis - window_left) : 0;
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
        }
        float nr = fmaxf(mr, tm);
        if ((nr - mr) * sc2 <= LAG_MAX_LOG2) nr = mr;
        const float nm2 = nr * sc2;
        const float alpha = (nr == mr) ? 1.0f : ex2_approx(fmaf(mr, sc2, -nm2));
        mr = nr;
#ifdef TC2_TIMING
        if (timed_s) { long long t = clock64(); sm_max += t - c0; c0 = t; }
#endif
        float pc = 0.0f;
        if (!masked) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            uint32_t pr[16];
            #pragma unroll
            for (int x2 = 0; x2 < 16; ++x2) {
              float p0 = ex2_approx(fmaf(sv2[cc][x2 * 2], sc2, -nm2));
              float p1 = ex2_approx(fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2));
              pc += p0 + p1;
              pr[x2] = pack_bf16(p0, p1);
            }
            tmem_st16bf(ts + cc * 16, pr);
          }
        } else {
          const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
          const int jlo = windowed ? max(0, vis - window_left) : 0;
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            uint32_t pr[16];
            #pragma unroll
            for (int x2 = 0; x2 < 16; ++x2) {
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
            }
            tmem_st16bf(ts + cc * 16, pr);
          }
        }
        tmem_wait_st();
        fence_before_sync();
        mbar_arrive(bar(ctrl, pfB + st));
        d = d * alpha + pc;
#ifdef TC2_TIMING
        if (timed_s) { long long t = clock64(); sm_exp += t - c0; c0 = t; }
#endif
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

      // ---- epilogue (per unit) ----
      const bool active = (grow < qlen);
      const uint32_t to = tmem + tcol_o + lb;
      __nv_bfloat16* orow =
          outp + static_cast<int64_t>(it.m0 + s * SUB + row) * o_stride;
      // v2 bidirectional merge: each part EXPORTS its own subtile
      // (s == part) and MERGES the other (s == 1-part) from its own TMEM
      // O(s) plus the peer's remote payload - two parallel 64KB exchanges
      // instead of v1's serial 128KB round trip (probe35: v1 chain
      // sigwait 8.3k + read/merge ~18k ate the whole split saving;
      // span_max 80.2k == unsplit tc2). Verdicts stay local: exchange
      // happens iff Ts>0 && sib_s>0 (sib_s = peer's window count for
      // subtile s, symmetric on both CTAs).
      const int sib_s = s ? it.sib1 : it.sib0;
      const bool bothx = it.is_split && Ts > 0 && sib_s > 0;
      const bool am_merge = bothx && (s != it.part);
      const bool am_export = bothx && (s == it.part);
      if (am_export) {
        // ---- EXPORT: STS own-subtile payload, signal peer SIG0, then
        // hold the CTA (smem lifetime) until the peer's read-done ACK ----
#ifdef TC2_TIMING
        const unsigned long long tx0 = timed_s ? clock64() : 0ull;
#endif
        const int cl = C - 1;
        mbar_wait_h(bar(ctrl, pvB + (cl & 1)), (int)((cl / 2) & 1), whint);
        fence_after_sync();
        // Payload [0,128KB) spans sq (+sk head): free only once BOTH
        // subtiles' final PVs retired (Q and KV stages live until then).
        // Cross-wait skipped when the sibling subtile is empty - the own
        // 64KB half then lives inside sq alone (own pvB wait suffices).
        {
          const int cx = s ? (it.Ts0 - 1) : (it.Ts1 - 1);
          if (cx >= 0) {
            const uint32_t xb = s ? PV00 : PV10;
            mbar_wait_h(bar(ctrl, xb + (cx & 1)), (int)((cx / 2) & 1), whint);
            fence_after_sync();
          }
        }
        // chunk-blocked fp32 layout (tc2s-probed: consecutive lanes write
        // consecutive 128B; row-major measured 11.8k/31.5k on f9):
        float* payO = reinterpret_cast<float*>(smem) + s * (SUB * DV);
        float* payMD = reinterpret_cast<float*>(smem + PAY_MD_OFF) +
                       (s * SUB + row) * 2;
        payMD[0] = mr;
        payMD[1] = d;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc)
            tmem_ld32_nw(sv2[cc], to + (half * 2 + cc) * 32);
          tmem_wait_ld();
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            float4* dst4 = reinterpret_cast<float4*>(
                payO + ((half * 2 + cc) * SUB + row) * 32);
            #pragma unroll
            for (int q4 = 0; q4 < 8; ++q4)
              dst4[q4] = make_float4(sv2[cc][q4 * 4], sv2[cc][q4 * 4 + 1],
                                     sv2[cc][q4 * 4 + 2], sv2[cc][q4 * 4 + 3]);
          }
        }
        // release the payload at gpu scope (subsumes cluster), then signal:
        // SIG_s count=4 = one lane0 arrive per exporting warp; part1's
        // acquire.cluster wait synchronizes-with each arrive.
        __threadfence();
        __syncwarp();
        if (lane == 0) mbar_arrive_rank(bar(ctrl, SIG0), crank ^ 1u);
        // ACK (own SIG1, count 4): the peer's merge warps arrive here
        // after their remote reads retired; waiting keeps this CTA's smem
        // payload alive until then (exit would tear the DSMEM window).
        mbar_wait_cl(bar(ctrl, SIG1), 0, whint);
#ifdef TC2_TIMING
        if (timed_s)
          atomicAdd(dbg + 67, (unsigned long long)(clock64() - tx0));
#endif
        fence_before_sync();
        mbar_arrive(bar(ctrl, odrB));
#ifdef TC2_TIMING
        if (timed_s) atomicAdd(dbg + 70, 1ull);
#endif
      } else if (am_merge) {
        // ---- MERGE subtile s (== 1 - part): own TMEM O + peer payload.
        // The exporter provably exists and signals: sib_s > 0 here is the
        // same predicate as the peer's am_export for this subtile. ----
#ifdef TC2_TIMING
        const unsigned long long tm0 = timed_s ? clock64() : 0ull;
        const unsigned long long te1 = tm0;
#endif
        const int cl = C - 1;
        mbar_wait_h(bar(ctrl, pvB + (cl & 1)), (int)((cl / 2) & 1), whint);
        fence_after_sync();
        // bounded one-way wait: the exporter never waits on the merger.
        // SIG0 (count 4) completes on the peer export warps' four remote
        // arrives; the full-warp acquire.cluster wait gives every lane the
        // release ordering on the peer's payload STS (probe34-validated
        // pattern; no extra fence).
        mbar_wait_cl(bar(ctrl, SIG0), 0, whint);
#ifdef TC2_TIMING
        if (timed_s) atomicAdd(dbg + 59, (unsigned long long)(clock64() - tm0));
#endif
        const uint32_t payb =
            mapa(sm32_(smem) + static_cast<uint32_t>(s * (SUB * DV * 4)),
                 crank ^ 1u);
        const uint32_t mdb =
            mapa(sm32_(smem + PAY_MD_OFF) +
                     static_cast<uint32_t>((s * SUB + row) * 8),
                 crank ^ 1u);
        const float mr_o = ld_remote_f32(mdb);
        const float d_o = ld_remote_f32(mdb + 4);
        const float a = mr * sc2, b = mr_o * sc2;
        const float M = fmaxf(a, b);
        const float w1 = ex2_approx(a - M);
        const float w2 = ex2_approx(b - M);
        const float d_m = fmaf(d, w1, d_o * w2);
        float lse2 = M + log2f(d_m);
        float extra = 1.0f;
        if (has_sink) {
          float t = fmaxf(lse2, sink2) + log2f(1.0f + exp2f(-fabsf(lse2 - sink2)));
          extra = exp2f(lse2 - t);
          lse2 = t;
        }
        const float gsc = bmm2_scale * (d_m > 0.0f ? 1.0f / d_m : 0.0f) * extra;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc)
            tmem_ld32_nw(sv2[cc], to + (half * 2 + cc) * 32);
          tmem_wait_ld();
          if (active) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              const uint32_t ob4 = payb + static_cast<uint32_t>(
                  (((half * 2 + cc) * SUB + row) * 32) * 4);
              __align__(16) uint32_t pk[16];
              #pragma unroll
              for (int q4 = 0; q4 < 8; ++q4) {
                uint4 vu;
                ld_remote_v4(ob4 + q4 * 16u, vu);
                const float m0f = fmaf(sv2[cc][q4 * 4], w1,
                                       __uint_as_float(vu.x) * w2);
                const float m1f = fmaf(sv2[cc][q4 * 4 + 1], w1,
                                       __uint_as_float(vu.y) * w2);
                const float m2f = fmaf(sv2[cc][q4 * 4 + 2], w1,
                                       __uint_as_float(vu.z) * w2);
                const float m3f = fmaf(sv2[cc][q4 * 4 + 3], w1,
                                       __uint_as_float(vu.w) * w2);
                pk[q4 * 2] = pack_bf16(m0f * gsc, m1f * gsc);
                pk[q4 * 2 + 1] = pack_bf16(m2f * gsc, m3f * gsc);
              }
              uint4* dst =
                  reinterpret_cast<uint4*>(orow + (half * 2 + cc) * 32);
              const uint4* src4 = reinterpret_cast<const uint4*>(pk);
              dst[0] = src4[0]; dst[1] = src4[1];
              dst[2] = src4[2]; dst[3] = src4[3];
            }
          }
        }
        // read-done ACK into the peer's SIG1: every remote read of this
        // warp has retired (__syncwarp converges the lanes after their
        // loads), so the peer may exit and tear down its smem afterwards.
        __syncwarp();
        if (lane == 0) mbar_arrive_rank(bar(ctrl, SIG1), crank ^ 1u);
        if (active && has_lse) {
          lsep[static_cast<int64_t>(it.m0 + s * SUB + row) * heads] = lse2;
        }
#ifdef TC2_TIMING
        if (timed_s) atomicAdd(dbg + 62, (unsigned long long)(clock64() - tm0));
#endif
        fence_before_sync();
        mbar_arrive(bar(ctrl, odrB));
#ifdef TC2_TIMING
        if (timed_s) {
          atomicAdd(dbg + 69, (unsigned long long)(clock64() - te1));
          atomicAdd(dbg + 71, 1ull);
        }
#endif
      } else if (Ts > 0) {
#ifdef TC2_TIMING
        const unsigned long long t_ep = timed_s ? clock64() : 0ull;
#endif
        const int cl = C - 1;
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
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc)
            tmem_ld32_nw(sv2[cc], to + (half * 2 + cc) * 32);
          tmem_wait_ld();
          if (half == 1) {
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
              uint4* dst =
                  reinterpret_cast<uint4*>(orow + (half * 2 + cc) * 32);
              const uint4* src4 = reinterpret_cast<const uint4*>(pk);
              dst[0] = src4[0]; dst[1] = src4[1];
              dst[2] = src4[2]; dst[3] = src4[3];
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
    atomicAdd(dbg + 48, sp);
    atomicAdd(dbg + 49, 1ull);
    atomicMax(dbg + 50, sp);
  }
#endif
}

}  // namespace mla_tc2c

// ---------------------------------------------------------------------------
// Host: rank-4 direct maps + single cluster launch (no prep, no ws).
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
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
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

// tc2c has NO device workspace: the split plan is in-kernel phase0 and
// the merge travels through intra-cluster DSMEM.
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
  using namespace mla_tc2c;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(mla_tc2c_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return static_cast<int>(e);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 9002;
    if (cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess)
      return 9003;
    if (g_sms <= 0) g_sms = 148;
    attr_set = true;
  }
  const int tiles_ub = total_q / BM + batch;
  if (tiles_ub <= 0) return static_cast<int>(cudaGetLastError());
  const int ub_items = tiles_ub * heads;
  // dispatch gate (solution.py) guarantees ub_items <= MAX_SLOTS and
  // tiles_ub <= 14 for tc2c shapes; be defensive anyway: phase0's Tl[16]
  // and hist[66] bounds rely on these (breach -> loud checker failure via
  // invalid units, never silent miscounts).
  if (ub_items > MAX_SLOTS || tiles_ub > 14) return 9100;

  CUtensorMap krmap, vrmap, qrmap;
  int rc;
  if ((rc = cached4(&krmap, k, (uint64_t)total_kv, (uint64_t)kv_heads, 3, 8)))
    return 5000 + rc;
  if ((rc = cached4(&vrmap, v, (uint64_t)total_kv, (uint64_t)kv_heads, 2, 8)))
    return 6000 + rc;
  if ((rc = cached4(&qrmap, q, (uint64_t)total_q, (uint64_t)heads, 3, 128)))
    return 7000 + rc;

  // one cluster launch: grid = SM count rounded down to even (74 clusters
  // of 2 on the 148-SM L20D); unit u -> cluster u/2, rank u&1. Idle CTAs
  // (u >= U) exit in phase0 before init/alloc.
  cudaLaunchConfig_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  cfg.gridDim = dim3(static_cast<unsigned>(g_sms & ~1), 1, 1);
  cfg.blockDim = dim3(THREADS, 1, 1);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = s;
  cudaError_t le = cudaLaunchKernelEx(
      &cfg, mla_tc2c_kernel, krmap, vrmap, qrmap,
      static_cast<__nv_bfloat16*>(out), static_cast<float*>(lse),
      static_cast<const float*>(sinks),
      static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
      batch, heads, kv_heads, window_left, causal, has_lse, has_sink,
      bmm1_scale, bmm2_scale, opts,
      reinterpret_cast<unsigned long long*>(dbg));
  if (le != cudaSuccess) return static_cast<int>(le);
  return static_cast<int>(cudaGetLastError());
}
