// mla_expanded_bf16 — tcgen05 persistent dual-subtile kernel (tc2p-v14 ARCHIVE).
// Frozen A/B baseline: kernel_tc2p.cu exactly as promoted in best/v14
// (E2 ex2.f16x2 + v8 stores, refill-at-item-start). Registered as forced
// engine "tc2p14"; NOT built by _build_all.
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
// Host: static device counter, cudaMemsetAsync before each launch (both
// capture cleanly into CUDA graphs; the one-time cudaMalloc runs on the
// first plain launch, which always precedes any capture in solution.py).
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
// (2 x 48B at 216). Region size 512B.
constexpr int ITEMS_OFF = 216;
constexpr int SMEM_BYTES = CTRL_OFF + 512;

// TMEM column map (identical to tc2)
DIN int TCOL_S(int s, int st) { return s * 128 + st * 64; }
DIN int TCOL_O(int s) { return 256 + s * 128; }

// barrier indices
enum { BF0 = 0, BF1, BF2, BE0, BE1, BE2,
       SF00, SF01, SF10, SF11,
       PF00, PF01, PF10, PF11,
       PV00, PV01, PV10, PV11,   // BEPV[s][st]
       OC0, OC1, QF0, QF1, NEXT, CONS, ODR0, ODR1, NBARS };
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

DIN void decode_item(const int32_t* __restrict__ cum_q,
                     const int32_t* __restrict__ cum_kv, int batch,
                     int tiles_ub, int w, int causal, int window_left,
                     int opts, ItemGeom& it) {
  const int tile_id = w % tiles_ub;
  const int h = w / tiles_ub;
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

#ifdef TC2_TIMING
  const unsigned long long t_ent = (tid == 0) ? clock64() : 0ull;
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
  const int total_items = tiles_ub * heads;

  if (warp == 9) {
    // ============ LOADER: fetch, publish, Q(0) fill, flat KV ring =========
    if (elect_one()) {
      int f = 0;                 // publication index
      long long g = 0;           // global KV stage index
      while (true) {
        // fetch next REAL item (skip surplus/empty silently)
        ItemGeom it;
        while (true) {
          const int w = (int)atomicAdd(counter, 1u);
          if (w >= total_items) { it.valid = 0; it.T = 0; break; }
          decode_item(cum_q, cum_kv, batch, tiles_ub, w, causal, window_left,
                      opts, it);
          if (it.valid) break;
        }
        // publish(k) writes slot k&1. Gate on CONS(k-1): every consumer has
        // copied item k-1's geometry, and the NEXT phase can never run 2
        // ahead of a waiter (parity aliasing would deadlock).
        if (f >= 1) mbar_wait_h(bar(ctrl, CONS), (f - 1) & 1, whint);
        items[f & 1] = it;                       // valid=0 marks the sentinel
        mbar_arrive(bar(ctrl, NEXT));            // release: geometry visible
        if (!it.valid) break;
        if (f == 0) {
          // first Q fill (later refills are issued by issuer warp 8)
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
      // box8 K-major SW128 canonical descriptor constants (as v1 TMA-Q mode)
      const int kplane = 1024;
      const uint32_t ksbo = 3072u;
      const uint32_t vlbo = 1024u;
      const uint32_t vsbo = 2048u;
      const int vstep = 4096;
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
          // ---- Q refill for this item (item 0 was filled by the loader) ----
          if (f > 0) {
            // Wait until the previous item's last QK MMAs completed (they
            // were the last readers of the Q smem regions).
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
        const int n0 = i * BN;
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
        // lagged running max (threshold 2^8 in scaled log2 domain)
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
    attr_set = true;
  }
  int tiles_ub = total_q / BM + batch;
  if (tiles_ub <= 0) return static_cast<int>(cudaGetLastError());
  const int total_items = tiles_ub * heads;
  int grid = total_items < g_sms ? total_items : g_sms;
  if (grid <= 0) return 5001;

  CUtensorMap krmap, vrmap, qrmap;
  int rc;
  if ((rc = cached4(&krmap, k, (uint64_t)total_kv, (uint64_t)kv_heads, 3, 8)))
    return 5000 + rc;
  if ((rc = cached4(&vrmap, v, (uint64_t)total_kv, (uint64_t)kv_heads, 2, 8)))
    return 6000 + rc;
  if ((rc = cached4(&qrmap, q, (uint64_t)total_q, (uint64_t)heads, 3, 128)))
    return 7000 + rc;

  if (cudaMemsetAsync(g_counter, 0, 4, s) != cudaSuccess)
    return static_cast<int>(cudaGetLastError());
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
