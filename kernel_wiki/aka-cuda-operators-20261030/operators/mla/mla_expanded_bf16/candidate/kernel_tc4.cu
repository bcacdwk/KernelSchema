// mla_expanded_bf16 — tcgen05 dual-subtile kernel, BN=32/4-stage (tc4).
// *** ABANDONED (probe14 dv_96428f6e5fda): m128n32k16 tensor floor = 1207
// cyc/dual = 2172 MAC/cyc (68% of the 3191 n64 rate) -> tc4's per-flop
// tensor floor is +47% vs tc2; best-case steady state cannot beat tc2's
// measured 2280. Also had a vstep bug on the box8 path (2048 vs 4096) -
// box32 numerics were clean (0 mismatch on 10 cases), protocol sound (no
// hangs). Kept as evidence; not built or dispatched by solution.py. ***
// Derived from kernel_tc2.cu (v1-proven) by parametric transformation:
// BN 64->32, KV_STAGES 3->4, S_STAGES 2->4. The 4-deep S ring lets QK(i)
// run three tiles ahead of PV(i-...)/softmax(i-...), hiding the softmax
// chain (probe4/5: is_pfw 182-922 cyc/tile was the top steady stall) and
// the 4-deep KV ring relaxes is_bf/is_bew. TMEM: S 4x32x2=256 + O 256.
// SMEM: Q 96K + 4x(K 12K + V 8K) = 176K + ctrl. Tensor floor target:
// 822 cyc per BN=32 dual tile (2.62M MAC @ 3191 MAC/cyc) IF m128n32k16
// runs at full rate (probe13 gates this design).
//
// Same frozen contract as kernel_tc.cu; structurally different pipeline
// (design + evidence: experiments/notes/dualsub.md):
//   * BM=256 as TWO 128-row subtiles in one CTA, BN=64, 3 KV stages,
//     2 S stages per subtile, 1 O accumulator per subtile (TMEM 512 cols:
//     S0a 0-63, S0b 64-127, S1a 128-192, S1b 192-255, O0 256-383,
//     O1 384-511).
//   * KV bytes per MMA flop halved vs the BM=128 kernel: 40960B per
//     1280-cycle dual tile (32B/cyc/SM required vs ~22B/cyc measured
//     L2->smem ingest wall => expected ~1.7x on feed-bound cases).
//   * DIRECT-TMA only: rank-4 maps over raw K [total_kv,h,192] /
//     V [total_kv,h,128]; box {64,8,atoms,1} = one canonical 8-token
//     SW128 group (same smem image as the probe-verified workspace map,
//     job5-validated end-to-end). 8 K + 8 V boxes per BN=64 tile.
//   * Softmax warps own a FULL row (64 cols): no pair exchange, no named
//     barriers in the softmax path (vs the column-split pairs of tc1).
//   * Lagged running max (threshold 2^8): alpha == 1.0f exactly on ~all
//     tiles -> O correction skipped (mechanism as cudnn_fe_oss reference).
//
// Warp roles (320 threads, 1 CTA/SM, smem 221440B):
//   warps 0-3: softmax subtile 0 (rows m0..m0+127)
//   warps 4-7: softmax subtile 1 (rows m0+128..m0+255)
//   warp  8: elected-lane tcgen05 MMA issuer (both subtiles)
//   warp  9: elected-lane TMA loader
//
// PTX primitives are the probe-verified forms from kernel_tc.cu (V0).
// Build: nvcc -gencode=arch=compute_103a,code=sm_103a -lcuda.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define DIN __device__ __forceinline__

namespace mla_tc4 {

constexpr int DQK = 192;
constexpr int DV = 128;
constexpr int BM = 256;             // 2 subtiles x 128 rows
constexpr int SUB = 128;
constexpr int BN = 32;
constexpr int S_STAGES = 4;   // TMEM S ring depth (tc2: 2)
constexpr int KV_STAGES = 4;
#ifdef TC2_DUAL3
constexpr int THREADS = 352;   // + warp10: subtile-1 MMA issuer
#else
constexpr int THREADS = 320;
#endif
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -1.0e30f;
constexpr float LAG_MAX_LOG2 = 8.0f;

constexpr int Q_BYTES = SUB * DQK * 2;          // 49152 per subtile region
constexpr int Q_TOTAL = BM * DQK * 2;           // 98304
constexpr int K_BYTES = BN * DQK * 2;           // 24576 per stage
constexpr int V_BYTES = BN * DV * 2;            // 16384 per stage
constexpr int KV_TOTAL = KV_STAGES * (K_BYTES + V_BYTES);   // 122880
constexpr int CTRL_OFF = Q_TOTAL + KV_TOTAL;    // 221184
// ctrl: 35 mbarriers (280B) + tmem holder (8B) + slack
constexpr int SMEM_BYTES = CTRL_OFF + 512;

// TMEM column map
DIN int TCOL_S(int s, int st) { return s * 128 + st * 32; }
DIN int TCOL_O(int s) { return 256 + s * 128; }

// barrier indices
enum { BF0 = 0, BF1, BF2, BF3, BE0, BE1, BE2, BE3,
       SF00, SF01, SF02, SF03, SF10, SF11, SF12, SF13,
       PF00, PF01, PF02, PF03, PF10, PF11, PF12, PF13,
       PV00, PV01, PV02, PV03, PV10, PV11, PV12, PV13,   // BEPV[s][st]
       OC0, OC1, QF, NBARS };
// per-(subtile,stage) barrier index helpers
DIN int SF_(int s, int st) { return SF00 + s * 4 + st; }
DIN int PF_(int s, int st) { return PF00 + s * 4 + st; }
DIN int PV_(int s, int st) { return PV00 + s * 4 + st; }
DIN uint32_t sm32_(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
DIN uint32_t bar(char* ctrl, int idx) { return sm32_(ctrl) + idx * 8; }

// ---------------------------------------------------------------------------
// PTX primitives (probe-verified forms, identical to kernel_tc.cu / V0)
// ---------------------------------------------------------------------------
DIN void cp_async16(uint32_t s, const void* g, int src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               ::"r"(s), "l"(g), "r"(src_size));
}
DIN void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
DIN void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

DIN void tma_load_4d(const CUtensorMap* map, uint32_t dst, uint32_t mbar,
                     int c0, int c1, int c2, int c3) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];"
      ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar)
      : "memory");
}
// KV tile loader for one BN=64 stage buffer. box_tok = TMA box rows:
//   8:  canonical km/mn groups (8 requests per operand; atom plane 1024B,
//       8-row groups at 3072/2048B) - matches tc1's probe-verified descs.
//   64: ONE full-tile box per operand (24KB K + 16KB V): rows contiguous
//       (row*128B) across the whole stage, atom plane at 8192B. job8 probe:
//       3KB boxes ingest ~15B/cyc/SM vs ~116B/cyc for 48KB boxes; the tc2
//       dual-tile period was feed-bound at ~2650cyc. Descriptors in the
//       issuer switch SBO/LBO/stepping to match; host encodes the map with
//       box {64, box_tok, atoms, 1}.
DIN void load_kv_tile(const CUtensorMap& krmap, const CUtensorMap& vrmap,
                      uint32_t kd, uint32_t vd, uint32_t mbar, int tok,
                      int h_kv, int box_tok) {
  if (box_tok == 32) {
    tma_load_4d(&krmap, kd, mbar, 0, tok, 0, h_kv);
    tma_load_4d(&vrmap, vd, mbar, 0, tok, 0, h_kv);
  } else {
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
      tma_load_4d(&krmap, kd + g * 3072, mbar, 0, tok + g * 8, 0, h_kv);
      tma_load_4d(&vrmap, vd + g * 2048, mbar, 0, tok + g * 8, 0, h_kv);
    }
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
// hint = suspend time hint for try_wait; opts bit 3 passes 0 (pure spin) to
// test whether deep-sleep wake latency is part of the per-tile chain.
DIN void mbar_wait_h(uint32_t a, int phase, uint32_t hint) {
  asm volatile(
      "{\n\t.reg .pred P;\n\tLABW%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, %2;\n\t"
      "@P bra.uni DONEW%=;\n\tbra.uni LABW%=;\n\tDONEW%=:\n\t}\n"
      ::"r"(a), "r"(phase), "r"(hint)
      : "memory");
}
DIN void mbar_wait(uint32_t a, int phase) { mbar_wait_h(a, phase, 0x989680u); }
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
// idesc: M128 N64 bf16 f32acc (N field = 64>>3 = 8 at bit 17)
constexpr uint32_t IDESC_QK32 = (uint32_t(128 >> 4) << 24) |
                                (uint32_t(32 >> 3) << 17) |
                                (1u << 10) | (1u << 7) | (1u << 4);  // m128 n32
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
// canonical UMMA smem addresses within a tile region
DIN uint32_t km_addr(int row, int kchunk16) {   // K-major SW128
  int atom = kchunk16 >> 3;
  int c = kchunk16 & 7;
  return (row >> 3) * 3072 + atom * 1024 + (row & 7) * 128 + ((c ^ (row & 7)) << 4);
}
DIN uint32_t mn_addr(int krow, int nchunk16) {  // MN-major SW128 (V)
  int atom = nchunk16 >> 3;
  int c = nchunk16 & 7;
  return (krow >> 3) * 2048 + atom * 1024 + (krow & 7) * 128 + ((c ^ (krow & 7)) << 4);
}
DIN void mark(unsigned long long* dbg, int slot, unsigned long long val) {
  if (dbg) atomicMax(dbg + slot, val);
}
// TC2_TIMING builds (engine "tc2t"): stage-cycle accounting via clock64
// sums into tc1-compatible dbg slots (16-22 softmax, 24-30 issuer,
// 32-34 loader). The progress marks are compiled OUT in timing builds -
// contended global atomicMax on one address from every tile/thread
// perturbed the very pipeline being measured (job21: 13x slowdown,
// ld_issue inflated to ~34k cyc/tile). Production builds have ZERO
// timing code (job21 also showed the runtime-gated version cost 21%
// on the default path: 144.7 -> 175.4us fixed geomean).
#ifdef TC2_TIMING
DIN void markp(unsigned long long* dbg, int slot, unsigned long long val) {
  (void)dbg; (void)slot; (void)val;
}
#else
#define markp mark
#endif

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS, 1)
mla_tc4_kernel(const __grid_constant__ CUtensorMap krmap,
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
               int32_t box_tok, int32_t opts,
               unsigned long long* __restrict__ dbg) {
  extern __shared__ char smem[];
  char* sq = smem;                       // 2 x 49152B Q regions
  char* sk = smem + Q_TOTAL;             // 3 x 24576B K stages
  char* sv = sk + KV_STAGES * K_BYTES;   // 3 x 16384B V stages
  char* ctrl = smem + CTRL_OFF;
  uint32_t* tholder = reinterpret_cast<uint32_t*>(ctrl + NBARS * 8);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int tile_id = blockIdx.x;
  const int h = blockIdx.y;
#ifdef TC2_TIMING
  // per-CTA lifetime accounting (dbg slots 48-55, tid0): span/prologue/
  // epilogue sums + T distribution. Answers "where does the un-accounted
  // per-CTA time go on short-T cases" (job2 c0: issuer+softmax spans ~= half
  // the CTA slot).
  const unsigned long long t_ent = (tid == 0) ? clock64() : 0ull;
#endif

  // ---- CTA mapping (BM=256 q-tiles) ----
  int b = -1, m0 = 0, q_base = 0, kv_base = 0, qlen = 0, kvlen = 0;
  {
    int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
    for (int i = 0; i < batch; ++i) {
      int nq = cum_q[i + 1], nkv = cum_kv[i + 1];
      int len_q = nq - prev_q;
      int tiles = (len_q + BM - 1) / BM;
      if (b < 0 && tile_id < acc + tiles) {
        // LPT order (opts bit 1): heaviest q-tile of the batch first
        // (longest causal KV sweep -> first wave; short tiles fill the tail).
        b = i;
        m0 = ((opts & 2) ? (tiles - 1 - (tile_id - acc)) : (tile_id - acc)) * BM;
        q_base = prev_q; kv_base = prev_kv;
        qlen = len_q; kvlen = nkv - prev_kv;
        break;
      }
      acc += tiles; prev_q = nq; prev_kv = nkv;
    }
  }
  if (b < 0) return;

  const int kv_off = kvlen - qlen;
  const bool windowed = window_left >= 0;
  // Per-subtile KV iteration bounds. A fully-OOB subtile (all rows >= qlen)
  // gets T=0: no QK/PV issued for it, its softmax warps skip to the end.
  const bool has_sub0 = (m0 < qlen);
  const bool has_sub1 = (m0 + SUB < qlen);
  int Ts0, Ts1, Tf0, Tf1;
  {
    int nmax = causal ? min(kvlen, kv_off + m0 + SUB) : kvlen;
    int nsafe = causal ? min(kvlen, kv_off + m0 + 1) : kvlen;
    if (windowed) nsafe = 0;
    if (!has_sub0) { nmax = 0; nsafe = 0; }
    nmax = min(nmax, kvlen);
    Ts0 = (nmax + BN - 1) / BN;
    Tf0 = nsafe / BN;
  }
  {
    int nmax = causal ? min(kvlen, kv_off + m0 + 2 * SUB) : kvlen;
    int nsafe = causal ? min(kvlen, kv_off + m0 + SUB + 1) : kvlen;
    if (windowed) nsafe = 0;
    if (!has_sub1) { nmax = 0; nsafe = 0; }
    nmax = min(nmax, kvlen);
    Ts1 = (nmax + BN - 1) / BN;
    Tf1 = nsafe / BN;
  }
  const int T = max(Ts0, Ts1);
  if (T <= 0) return;

  const int group_div = heads / kv_heads;
  const int h_kv = h / group_div;
  __nv_bfloat16* outp = out + (static_cast<int64_t>(q_base) * heads + h) * DV;
  float* lsep = has_lse ? lse + static_cast<int64_t>(q_base) * heads + h : nullptr;
  const int64_t o_stride = static_cast<int64_t>(heads) * DV;
  const float sc2 = bmm1_scale * LOG2E;
  const float sink2 = has_sink ? sinks[h] * LOG2E : 0.0f;
  const int tok0 = kv_base;
  const uint32_t whint = (opts & 8) ? 0u : 0x989680u;   // bit3: pure spin
  // opts bit 7 (128): Q loaded by TMA as two {64,128,3,1} boxes -> the box
  // writes ATOM-major SW128 (atom plane = 128 rows x 128B = 16384B, 8-row
  // core matrices at 1024B; tc3-proven) instead of the cp.async km_addr
  // group-major layout (8-row groups at 3072B, atom planes at 1024B).
  // QK A-descs follow: SBO 1024 + atom stride 16384B vs SBO 3072 + 1024B.
  const uint32_t qsbo = (opts & 128) ? 1024u : 3072u;
  const int qa16 = (opts & 128) ? 1024 : 64;   // atom stride in 16B units
#ifdef TC2_DUAL3
  const bool di = (opts & 64);   // dual-issuer: warp8 drives s0, warp10 s1
#else
  // bit64 retired in 320-thread builds: job23 (dv_5f9057c94859) merged
  // loader+s1-issuer into warp9 and the TMA feed fell behind the s1 chain
  // (warp8 is_bf_wait 124 -> 1062 cyc/dual-tile; fixed 0.978x, domain
  // 0.962x). The 352-thread tc2d build keeps warp9 a pure loader.
  const bool di = false;
#endif

  if (dbg && tid == 0) atomicAdd(dbg + 8, 1ull);

  // ---- control init ----
  if (tid == 0) {
    for (int i = 0; i < NBARS; ++i) {
      int cnt = 1;
      if (i >= PF00 && i < PV00) cnt = 128;       // 4 softmax warps/subtile
      else if (i == OC0 || i == OC1) cnt = 128;
      else if (i == QF) cnt = (opts & 128) ? 1 : THREADS;
      // dual-issuer (opts bit 6): one tcgen05.commit per issuer warp
      else if (di && i >= BE0 && i <= BE3) cnt = 2;
      mbar_init(bar(ctrl, i), cnt);
    }
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

  // ---- K/V TMA prologue (warp 9 elected; tiles 0..min(2,T-1)) ----
  if (warp == 9 && elect_one()) {
    #pragma unroll
    for (int j = 0; j < KV_STAGES; ++j) {
      if (j < T) {
        mbar_expect_tx(bar(ctrl, BF0 + j), K_BYTES + V_BYTES);
        load_kv_tile(krmap, vrmap, sm32_(sk) + j * K_BYTES,
                     sm32_(sv) + j * V_BYTES, bar(ctrl, BF0 + j),
                     tok0 + j * BN, h_kv, box_tok);
        markp(dbg, 0, (unsigned long long)(j + 1));
      }
    }
  }

  // ---- Q direct load ----
  if (opts & 128) {
    // TMA-Q: two 48KB boxes (one per subtile region). ~900 cyc fill vs
    // ~2550 for the cp.async path; only tid0 issues. Rows past total_q
    // zero-fill (OOB); in-range rows of a following batch are finite
    // garbage - row-local through softmax/PV and never stored (same
    // contract as the opts bit2 qrows-skip path).
    if (tid == 0) {
      mbar_expect_tx(bar(ctrl, QF), 2 * Q_BYTES);
      tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF), 0, q_base + m0, 0, h);
      tma_load_4d(&qrmap, sm32_(sq) + Q_BYTES, bar(ctrl, QF), 0,
                  q_base + m0 + SUB, 0, h);
    }
  } else {
  // ---- cp.async path (ALL threads): 256 rows x 24 16B chunks ----
  {
    const __nv_bfloat16* qp =
        qraw + (static_cast<int64_t>(q_base) * heads + h) * DQK;
    const int64_t q_stride = static_cast<int64_t>(heads) * DQK;
    // opts bit 2: load only rows < qlen (skip zero-fill traffic for OOB
    // rows; garbage rows keep row-local NaN/garbage through softmax and are
    // never stored). opts=0 keeps the proven zero-fill path.
    const int qrows = (opts & 4) ? min(qlen - m0, BM) : BM;
    #pragma unroll 4
    for (int idx = tid; idx < qrows * 24; idx += THREADS) {
      int r = idx / 24, c = idx % 24;
      const __nv_bfloat16* src =
          qp + static_cast<int64_t>(m0 + r) * q_stride + c * 8;
      cp_async16(sm32_(sq) + (r >> 7) * Q_BYTES + km_addr(r & 127, c), src,
                 (m0 + r < qlen) ? 16 : 0);
    }
    cp_commit();
    cp_wait0();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    mbar_arrive(bar(ctrl, QF));
  }
  }

#ifdef TC2_DUAL3
  if (warp == 10) {
    // ---------------- SUBTILE-1 ISSUER (dual-issuer mode) ----------------
    // Subtile accumulator chains are independent: warp8 drives s0, this
    // warp drives s1; warp9 stays a pure TMA loader (job23 lesson: folding
    // the feed into an issuer warp starves BF and costs more than the
    // pipe-packing wins).
    if (di && elect_one()) {
      mbar_wait_h(bar(ctrl, QF), 0, whint);
      fence_after_sync();
      const uint32_t qb1 = sm32_(sq) + Q_BYTES;
      const int kplane = (box_tok == 32) ? 4096 : 1024;
      const uint32_t ksbo = (box_tok == 32) ? 1024u : 3072u;
      const uint32_t vlbo = (box_tok == 32) ? 4096u : 1024u;
      const uint32_t vsbo = (box_tok == 32) ? 1024u : 2048u;
      const int vstep = 2048;   // 16 tokens x 128B per k16 step (box-indep)
#ifdef TC2_TIMING
      const bool timed = (dbg != nullptr);
      long long i1_tiles = 0, i1_bf = 0, i1_bew = 0, i1_qk = 0,
                i1_pfw = 0, i1_ocw = 0, i1_pv = 0;
      long long u0 = 0, u1 = 0, u2 = 0, p0 = 0, p1 = 0, bw = 0;
#endif
      for (int i = 0; i < T; ++i) {
        const int st3 = i % KV_STAGES;
        const int st = i & (S_STAGES - 1);
#ifdef TC2_TIMING
        if (timed) { bw = 0; u0 = clock64(); }
#endif
        mbar_wait_h(bar(ctrl, BF0 + st3), (i / KV_STAGES) & 1, whint);
        fence_after_sync();
#ifdef TC2_TIMING
        if (timed) { u1 = clock64(); i1_bf += u1 - u0; }
#endif
        // ---- QK(s1,i) ----
        if (i < Ts1) {
          if (i >= S_STAGES) {
#ifdef TC2_TIMING
            if (timed) u2 = clock64();
#endif
            mbar_wait_h(bar(ctrl, PV_(1, st)),
                        ((i - S_STAGES) / S_STAGES) & 1, whint);
            fence_after_sync();
#ifdef TC2_TIMING
            if (timed) { long long t = clock64(); i1_bew += t - u2; bw += t - u2; }
#endif
          }
          const uint32_t tmem_s = tmem + TCOL_S(1, st);
          const uint64_t da0 = make_desc(qb1, 0, qsbo);
          const uint64_t db0 =
              make_desc(sm32_(sk) + st3 * K_BYTES, 0, ksbo);
          #pragma unroll
          for (int kk = 0; kk < 12; ++kk) {
            mma_ss(tmem_s, da0 + ((kk >> 2) * qa16 + (kk & 3) * 2),
                   db0 + ((kk >> 2) * (kplane >> 4) + (kk & 3) * 2),
                   IDESC_QK32, kk > 0);
          }
          mma_commit(bar(ctrl, SF_(1, st)));
        }
        markp(dbg, 4, (unsigned long long)(i + 1));
#ifdef TC2_TIMING
        if (timed) { u2 = clock64(); i1_qk += u2 - u1 - bw; i1_tiles += 1; }
#endif
        // ---- PV(s1,i-1) ----
        if (i > 0 && (i - 1) < Ts1) {
          const int ip = i - 1;
          const int stp = ip & (S_STAGES - 1);
          const uint32_t vbase_p = sm32_(sv) + (ip % KV_STAGES) * V_BYTES;
#ifdef TC2_TIMING
          if (timed) p0 = clock64();
#endif
          mbar_wait_h(bar(ctrl, PF_(1, stp)), (ip / S_STAGES) & 1, whint);
#ifdef TC2_TIMING
          if (timed) { p1 = clock64(); i1_pfw += p1 - p0; p0 = p1; }
#endif
          mbar_wait_h(bar(ctrl, OC1), ip & 1, whint);
#ifdef TC2_TIMING
          if (timed) { p1 = clock64(); i1_ocw += p1 - p0; }
#endif
          fence_after_sync();
          const uint32_t tmem_p = tmem + TCOL_S(1, stp);
          const uint64_t db0 = make_desc(vbase_p, vlbo, vsbo);
          #pragma unroll
          for (int kk = 0; kk < BN / 16; ++kk)
            mma_ts(tmem + TCOL_O(1), tmem_p + kk * 8,
                   db0 + kk * (vstep >> 4), IDESC_PV, (ip > 0) | (kk > 0));
          mma_commit(bar(ctrl, PV_(1, stp)));
#ifdef TC2_TIMING
          if (timed) i1_pv += clock64() - p1;
#endif
        }
        // ---- release stage i-1 (s1 share; pairs with warp8's commit) ----
        if (i >= 1) {
          mma_commit(bar(ctrl, BE0 + (i - 1) % KV_STAGES));
        }
      }
      // ---- tail: PV(s1, Ts1-1) when s1 ran to T ----
      {
        const int il = Ts1 - 1;
        if (il == T - 1) {
          const int stl = il & (S_STAGES - 1);
          const int st3l = il % KV_STAGES;
          mbar_wait_h(bar(ctrl, PF_(1, stl)), (il / S_STAGES) & 1, whint);
          mbar_wait_h(bar(ctrl, OC1), il & 1, whint);
          fence_after_sync();
          const uint32_t tmem_p = tmem + TCOL_S(1, stl);
          const uint64_t db0 =
              make_desc(sm32_(sv) + st3l * V_BYTES, vlbo, vsbo);
          #pragma unroll
          for (int kk = 0; kk < BN / 16; ++kk)
            mma_ts(tmem + TCOL_O(1), tmem_p + kk * 8,
                   db0 + kk * (vstep >> 4), IDESC_PV, (il > 0) | (kk > 0));
          mma_commit(bar(ctrl, PV_(1, stl)));
        }
      }
#ifdef TC2_TIMING
      if (timed) {
        atomicAdd(dbg + 40, (unsigned long long)i1_tiles);
        atomicAdd(dbg + 41, (unsigned long long)i1_bf);
        atomicAdd(dbg + 42, (unsigned long long)i1_bew);
        atomicAdd(dbg + 43, (unsigned long long)i1_qk);
        atomicAdd(dbg + 44, (unsigned long long)i1_pfw);
        atomicAdd(dbg + 45, (unsigned long long)i1_ocw);
        atomicAdd(dbg + 46, (unsigned long long)i1_pv);
      }
#endif
    }
  } else if (warp == 9) {
#else
  if (warp == 9) {
#endif
    // ---------------- TMA LOADER (steady state) ----------------
    if (elect_one()) {
#ifdef TC2_TIMING
      const bool timed_l = (dbg != nullptr);
      long long ld_tiles = 0, ld_bew = 0, ld_iss = 0;
      long long x0 = 0, x1 = 0;
#endif
      for (int j = KV_STAGES; j < T; ++j) {
        const int st3 = j % KV_STAGES;
#ifdef TC2_TIMING
        if (timed_l) x0 = clock64();
#endif
        mbar_wait_h(bar(ctrl, BE0 + st3), (j / KV_STAGES - 1) & 1, whint);
#ifdef TC2_TIMING
        if (timed_l) x1 = clock64();
#endif
        mbar_expect_tx(bar(ctrl, BF0 + st3), K_BYTES + V_BYTES);
        load_kv_tile(krmap, vrmap, sm32_(sk) + st3 * K_BYTES,
                     sm32_(sv) + st3 * V_BYTES, bar(ctrl, BF0 + st3),
                     tok0 + j * BN, h_kv, box_tok);
        markp(dbg, 0, (unsigned long long)(j + 1));
#ifdef TC2_TIMING
        if (timed_l) {
          long long x2 = clock64();
          ld_tiles += 1; ld_bew += x1 - x0; ld_iss += x2 - x1;
        }
#endif
      }
#ifdef TC2_TIMING
      if (timed_l) {
        atomicAdd(dbg + 32, (unsigned long long)ld_tiles);
        atomicAdd(dbg + 33, (unsigned long long)ld_bew);
        atomicAdd(dbg + 34, (unsigned long long)ld_iss);
      }
#endif
    }
  } else if (warp == 8) {
    // ---------------- MMA ISSUER ----------------
    if (elect_one()) {
      mbar_wait_h(bar(ctrl, QF), 0, whint);
      fence_after_sync();
      const uint32_t qb0 = sm32_(sq);
      const uint32_t qb1 = sm32_(sq) + Q_BYTES;
      // KV smem layout depends on the TMA box rows (see load_kv_tile):
      //   box8: km/mn canonical (atom plane 1024, 8-row groups at 3072/2048)
      //   box64: rows contiguous; atom plane at 8192B (K) / 8192B (V);
      //          8-row core matrices contiguous -> SBO 1024.
      const int kplane = (box_tok == 32) ? 4096 : 1024;
      const uint32_t ksbo = (box_tok == 32) ? 1024u : 3072u;
      const uint32_t vlbo = (box_tok == 32) ? 4096u : 1024u;
      const uint32_t vsbo = (box_tok == 32) ? 1024u : 2048u;
      const int vstep = 2048;   // 16 tokens x 128B per k16 step (box-indep)
#ifdef TC2_TIMING
      const bool timed = (dbg != nullptr);
      long long is_tiles = 0, is_bf = 0, is_bew = 0, is_qk = 0,
                is_pfw = 0, is_ocw = 0, is_pv = 0;
      long long u0 = 0, u1 = 0, u2 = 0, p0 = 0, p1 = 0, bw = 0;
#endif
      if (di) {
      // ===== dual-issuer: this warp drives subtile 0 only; warp10 drives
      // subtile 1 (tc2d build). Slots 24-30 then measure the s0 stream. =====
      for (int i = 0; i < T; ++i) {
        const int st3 = i % KV_STAGES;
        const int st = i & (S_STAGES - 1);
#ifdef TC2_TIMING
        if (timed) { bw = 0; u0 = clock64(); }
#endif
        mbar_wait_h(bar(ctrl, BF0 + st3), (i / KV_STAGES) & 1, whint);
        fence_after_sync();
#ifdef TC2_TIMING
        if (timed) { u1 = clock64(); is_bf += u1 - u0; }
#endif
        // ---- QK(s0,i) ----
        if (i < Ts0) {
          if (i >= S_STAGES) {
#ifdef TC2_TIMING
            if (timed) u2 = clock64();
#endif
            mbar_wait_h(bar(ctrl, PV_(0, st)),
                        ((i - S_STAGES) / S_STAGES) & 1, whint);
            fence_after_sync();
#ifdef TC2_TIMING
            if (timed) { long long t = clock64(); is_bew += t - u2; bw += t - u2; }
#endif
          }
          const uint32_t tmem_s = tmem + TCOL_S(0, st);
          const uint64_t da0 = make_desc(qb0, 0, qsbo);
          const uint64_t db0 =
              make_desc(sm32_(sk) + st3 * K_BYTES, 0, ksbo);
          #pragma unroll
          for (int kk = 0; kk < 12; ++kk) {
            mma_ss(tmem_s, da0 + ((kk >> 2) * qa16 + (kk & 3) * 2),
                   db0 + ((kk >> 2) * (kplane >> 4) + (kk & 3) * 2),
                   IDESC_QK32, kk > 0);
          }
          mma_commit(bar(ctrl, SF_(0, st)));
        }
        markp(dbg, 1, (unsigned long long)(i + 1));
#ifdef TC2_TIMING
        if (timed) { u2 = clock64(); is_qk += u2 - u1 - bw; is_tiles += 1; }
#endif
        // ---- PV(s0,i-1) ----
        if (i > 0 && (i - 1) < Ts0) {
          const int ip = i - 1;
          const int stp = ip & (S_STAGES - 1);
          const uint32_t vbase_p = sm32_(sv) + (ip % KV_STAGES) * V_BYTES;
#ifdef TC2_TIMING
          if (timed) p0 = clock64();
#endif
          mbar_wait_h(bar(ctrl, PF_(0, stp)), (ip / S_STAGES) & 1, whint);
#ifdef TC2_TIMING
          if (timed) { p1 = clock64(); is_pfw += p1 - p0; p0 = p1; }
#endif
          mbar_wait_h(bar(ctrl, OC0), ip & 1, whint);
#ifdef TC2_TIMING
          if (timed) { p1 = clock64(); is_ocw += p1 - p0; }
#endif
          fence_after_sync();
          const uint32_t tmem_p = tmem + TCOL_S(0, stp);
          const uint64_t db0 = make_desc(vbase_p, vlbo, vsbo);
          #pragma unroll
          for (int kk = 0; kk < BN / 16; ++kk)
            mma_ts(tmem + TCOL_O(0), tmem_p + kk * 8,
                   db0 + kk * (vstep >> 4), IDESC_PV, (ip > 0) | (kk > 0));
          mma_commit(bar(ctrl, PV_(0, stp)));
#ifdef TC2_TIMING
          if (timed) is_pv += clock64() - p1;
#endif
        }
        // ---- release stage i-1 (s0 share; pairs with warp9's commit) ----
        if (i >= 1) {
          mma_commit(bar(ctrl, BE0 + (i - 1) % KV_STAGES));
        }
      }
      // ---- tail: PV(s0, Ts0-1) when s0 ran to T ----
      {
        const int il = Ts0 - 1;
        if (il == T - 1) {
          const int stl = il & (S_STAGES - 1);
          const int st3l = il % KV_STAGES;
          mbar_wait_h(bar(ctrl, PF_(0, stl)), (il / S_STAGES) & 1, whint);
          mbar_wait_h(bar(ctrl, OC0), il & 1, whint);
          fence_after_sync();
          const uint32_t tmem_p = tmem + TCOL_S(0, stl);
          const uint64_t db0 =
              make_desc(sm32_(sv) + st3l * V_BYTES, vlbo, vsbo);
          #pragma unroll
          for (int kk = 0; kk < BN / 16; ++kk)
            mma_ts(tmem + TCOL_O(0), tmem_p + kk * 8,
                   db0 + kk * (vstep >> 4), IDESC_PV, (il > 0) | (kk > 0));
          mma_commit(bar(ctrl, PV_(0, stl)));
        }
      }
      } else {
      for (int i = 0; i < T; ++i) {
        const int st3 = i % KV_STAGES;
        const int st = i & (S_STAGES - 1);
#ifdef TC2_TIMING
        if (timed) { bw = 0; u0 = clock64(); }
#endif
        mbar_wait_h(bar(ctrl, BF0 + st3), (i / KV_STAGES) & 1, whint);
        fence_after_sync();
#ifdef TC2_TIMING
        if (timed) { u1 = clock64(); is_bf += u1 - u0; }
#endif
        const uint32_t kbase = sm32_(sk) + st3 * K_BYTES;
        const uint32_t vbase = sm32_(sv) + st3 * V_BYTES;
        // ---- QK(s,i) for both subtiles ----
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
          const int Ts_sel = s ? Ts1 : Ts0;
          if (i < Ts_sel) {
            if (i >= S_STAGES) {
#ifdef TC2_TIMING
              if (timed) u2 = clock64();
#endif
              mbar_wait_h(bar(ctrl, PV_(s, st)),
                          ((i - S_STAGES) / S_STAGES) & 1, whint);
              fence_after_sync();
#ifdef TC2_TIMING
              if (timed) { long long t = clock64(); is_bew += t - u2; bw += t - u2; }
#endif
            }
            const uint32_t qbase = s ? qb1 : qb0;
            const uint32_t tmem_s = tmem + TCOL_S(s, st);
            // Descriptor fast-add: smem desc addr field is (addr>>4) in
            // bits 0-13; all offsets below are 16B units and the sum never
            // leaves the 14-bit field (max ~13.6k < 16384), so a single
            // uint64 add replaces the full make_desc per phase (job21 A/B:
            // +7.8% fixed-suite geomean).
            const uint64_t da0 = make_desc(qbase, 0, qsbo);
            const uint64_t db0 = make_desc(kbase, 0, ksbo);
            #pragma unroll
            for (int kk = 0; kk < 12; ++kk) {
              mma_ss(tmem_s, da0 + ((kk >> 2) * qa16 + (kk & 3) * 2),
                     db0 + ((kk >> 2) * (kplane >> 4) + (kk & 3) * 2),
                     IDESC_QK32, kk > 0);
            }
            mma_commit(bar(ctrl, SF_(s, st)));
          }
        }
        markp(dbg, 1, (unsigned long long)(i + 1));
#ifdef TC2_TIMING
        if (timed) { u2 = clock64(); is_qk += u2 - u1 - bw; is_tiles += 1; }
#endif
        // ---- PV(s,i-1): reads KV stage (i-1)%3 ----
        if (i > 0) {
          const int ip = i - 1;
          const int stp = ip & (S_STAGES - 1);
          const uint32_t vbase_p =
              sm32_(sv) + (ip % KV_STAGES) * V_BYTES;
          #pragma unroll
          for (int s = 0; s < 2; ++s) {
            const int Ts_sel = s ? Ts1 : Ts0;
            if (ip < Ts_sel) {
#ifdef TC2_TIMING
              if (timed) p0 = clock64();
#endif
              mbar_wait_h(bar(ctrl, PF_(s, stp)),
                          (ip / S_STAGES) & 1, whint);
#ifdef TC2_TIMING
              if (timed) { p1 = clock64(); is_pfw += p1 - p0; p0 = p1; }
#endif
              mbar_wait_h(bar(ctrl, OC0 + s), ip & 1, whint);
#ifdef TC2_TIMING
              if (timed) { p1 = clock64(); is_ocw += p1 - p0; }
#endif
              fence_after_sync();
              const uint32_t tmem_p = tmem + TCOL_S(s, stp);
              const uint64_t db0 = make_desc(vbase_p, vlbo, vsbo);
              #pragma unroll
              for (int kk = 0; kk < BN / 16; ++kk)
                mma_ts(tmem + TCOL_O(s), tmem_p + kk * 8,
                       db0 + kk * (vstep >> 4), IDESC_PV,
                       (ip > 0) | (kk > 0));
              mma_commit(bar(ctrl, PV_(s, stp)));
#ifdef TC2_TIMING
              if (timed) is_pv += clock64() - p1;
#endif
            }
          }
        }
        // ---- release KV stage i-1 (all its MMAs issued above) ----
        if (i >= 1) {
          mma_commit(bar(ctrl, BE0 + (i - 1) % KV_STAGES));
        }
      }
      // ---- tail: PV(s, T_s-1) for subtiles that ran to T ----
      #pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int il = (s ? Ts1 : Ts0) - 1;
        if (il == T - 1) {
          const int stl = il & (S_STAGES - 1);
          const int st3 = il % KV_STAGES;
#ifdef TC2_TIMING
          if (timed) p0 = clock64();
#endif
          mbar_wait_h(bar(ctrl, PF_(s, stl)), (il / S_STAGES) & 1, whint);
#ifdef TC2_TIMING
          if (timed) { p1 = clock64(); is_pfw += p1 - p0; p0 = p1; }
#endif
          mbar_wait_h(bar(ctrl, OC0 + s), il & 1, whint);
#ifdef TC2_TIMING
          if (timed) { p1 = clock64(); is_ocw += p1 - p0; }
#endif
          fence_after_sync();
          const uint32_t tmem_p = tmem + TCOL_S(s, stl);
          const uint32_t vbase = sm32_(sv) + st3 * V_BYTES;
          const uint64_t db0 = make_desc(vbase, vlbo, vsbo);
          #pragma unroll
          for (int kk = 0; kk < BN / 16; ++kk)
            mma_ts(tmem + TCOL_O(s), tmem_p + kk * 8,
                   db0 + kk * (vstep >> 4), IDESC_PV,
                   (il > 0) | (kk > 0));
          mma_commit(bar(ctrl, PV_(s, stl)));
#ifdef TC2_TIMING
          if (timed) is_pv += clock64() - p1;
#endif
        }
      }
      }
      markp(dbg, 2, 1ull);
#ifdef TC2_TIMING
      if (timed) {
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
    // ---------------- SOFTMAX + CORRECTION + EPILOGUE ----------------
    const int s = warp >> 2;               // subtile
    const int w4 = warp & 3;               // warp within subtile
    const int lane = tid & 31;
    const int row = w4 * 32 + lane;        // row within the subtile [0,128)
    const int grow = m0 + s * SUB + row;   // global q row
    const uint32_t lb = static_cast<uint32_t>(w4 * 32) << 16;
    const int vis = kv_off + grow;
    const int Ts = s ? Ts1 : Ts0;
    const int Tfull = s ? Tf1 : Tf0;
    float mr = NEG_BIG, d = 0.0f;
    float sv2[2][32];
#ifdef TC2_TIMING
    const bool timed_s = (dbg != nullptr) && (tid == 0);
    long long sm_tiles = 0, sm_sld = 0, sm_max = 0, sm_exp = 0,
              sm_bew = 0, sm_corr = 0, sm_per = 0;
    long long c0 = 0, cprev = 0;
#endif

    for (int i = 0; i < Ts; ++i) {
      const int st = i & (S_STAGES - 1);
      const uint32_t ts = tmem + TCOL_S(s, st) + lb;
      mbar_wait_h(bar(ctrl, SF_(s, st)), (i / S_STAGES) & 1, whint);
      fence_after_sync();
#ifdef TC2_TIMING
      if (timed_s) {
        c0 = clock64();
        if (i > 0) sm_per += c0 - cprev;
        else atomicAdd(dbg + 51, (unsigned long long)(c0 - t_ent));
        cprev = c0;
      }
#endif
      const int n0 = i * BN;
      const bool masked = i >= Tfull;

      #pragma unroll
      for (int cc = 0; cc < BN / 32; ++cc)
        tmem_ld32_nw(sv2[cc], ts + cc * 32);
      tmem_wait_ld();
#ifdef TC2_TIMING
      if (timed_s) { long long t = clock64(); sm_sld += t - c0; c0 = t; }
#endif

      float tm = NEG_BIG;
      if (!masked) {
        #pragma unroll
        for (int cc = 0; cc < BN / 32; ++cc) {
          float pm = sv2[cc][0];
          #pragma unroll
          for (int x = 1; x < 32; ++x) pm = fmaxf(pm, sv2[cc][x]);
          tm = fmaxf(tm, pm);
        }
      } else {
        const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
        const int jlo = windowed ? max(0, vis - window_left) : 0;
        #pragma unroll
        for (int cc = 0; cc < BN / 32; ++cc) {
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
        for (int cc = 0; cc < BN / 32; ++cc) {
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
        for (int cc = 0; cc < BN / 32; ++cc) {
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
      mbar_arrive(bar(ctrl, PF_(s, st)));
      d = d * alpha + pc;
#ifdef TC2_TIMING
      if (timed_s) { long long t = clock64(); sm_exp += t - c0; c0 = t; }
#endif

      // O correction (skip tile 0; skip when alpha==1 warp-wide).
      // Fast-arrive: with no correction the warp touches nothing in O, so
      // OC can arrive WITHOUT waiting for PV(s,i-1) completion - the issuer
      // may then put PV(s,i) on the in-order tensor pipe immediately and the
      // serial PV->corr->OC->PV chain collapses (lagged-max makes alpha==1
      // the ~always case after tile 0).
      const bool need_corr =
          (i > 0) && __any_sync(0xffffffffu, alpha != 1.0f);
#ifdef TC2_TIMING
      long long cw = c0;
#endif
      if (need_corr || (i > 0 && !(opts & 1))) {
        mbar_wait_h(bar(ctrl, PV_(s, (i - 1) & (S_STAGES - 1))),
                    ((i - 1) / S_STAGES) & 1, whint);
        fence_after_sync();
#ifdef TC2_TIMING
        if (timed_s) { long long t = clock64(); sm_bew += t - cw; cw = t; }
#endif
      }
      if (need_corr) {
        const uint32_t to = tmem + TCOL_O(s) + lb;
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
      mbar_arrive(bar(ctrl, OC0 + s));
      markp(dbg, 3, (unsigned long long)(i + 1));
#ifdef TC2_TIMING
      if (timed_s) { long long t = clock64(); sm_corr += t - cw; sm_tiles += 1; }
#endif
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

    // ---- epilogue ----
    if (Ts > 0) {
#ifdef TC2_TIMING
      const unsigned long long t_ep = timed_s ? clock64() : 0ull;
#endif
      const int il = Ts - 1;
      mbar_wait_h(bar(ctrl, PV_(s, il & (S_STAGES - 1))),
                  (il / S_STAGES) & 1, whint);
      fence_after_sync();
      float inv = d > 0.0f ? 1.0f / d : 0.0f;
      float lse2 = mr * sc2 + log2f(d);
      float extra = 1.0f;
      if (has_sink) {
        float t = fmaxf(lse2, sink2) + log2f(1.0f + exp2f(-fabsf(lse2 - sink2)));
        extra = exp2f(lse2 - t);
        lse2 = t;
      }
      const float g = bmm2_scale * inv * extra;
      const bool active = (grow < qlen);
      const uint32_t to = tmem + TCOL_O(s) + lb;
      __nv_bfloat16* orow = outp + static_cast<int64_t>(m0 + s * SUB + row) * o_stride;
      #pragma unroll
      for (int half = 0; half < 2; ++half) {
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc)
          tmem_ld32_nw(sv2[cc], to + (half * 2 + cc) * 32);
        tmem_wait_ld();
        if (active) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
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
      if (active && has_lse) {
        lsep[static_cast<int64_t>(m0 + s * SUB + row) * heads] = lse2;
      }
#ifdef TC2_TIMING
      if (timed_s) atomicAdd(dbg + 52, (unsigned long long)(clock64() - t_ep));
#endif
    }
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
    atomicAdd(dbg + 48, sp);            // span sum (working CTAs)
    atomicAdd(dbg + 49, 1ull);          // working CTA count
    atomicMax(dbg + 50, sp);            // worst CTA span
    atomicAdd(dbg + 54, (unsigned long long)T);
    atomicAdd(dbg + 55, (unsigned long long)T * (unsigned long long)T);
  }
#endif
}

}  // namespace mla_tc4

// ---------------------------------------------------------------------------
// Host: rank-4 direct maps + launch (no prep, no workspace).
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

// box {64, box_tok, atoms, 1}: box_tok=8 reproduces the canonical km/mn
// group layout; box_tok=64 is one full-tile request writing rows
// contiguously with atom planes at 8192B (job8 probe: 3KB boxes ingest
// ~15B/cyc/SM vs ~116B/cyc for large boxes -> lift the tc2 feed wall).
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
}  // namespace

// Same exported ABI as kernel_tc.cu so solution.py can load this .so as an
// alternative "tc2" engine; ws sizing is zero (direct path), and the
// kws/vws/kvblks/dbg_mode/direct args are ignored.
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
  (void)pf_mode; (void)lag_max; (void)direct;
  if (box_tok != 8 && box_tok != 32) box_tok = 8;
  using namespace mla_tc4;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(mla_tc4_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return static_cast<int>(e);
    attr_set = true;
  }
  int tiles_ub = total_q / BM + batch;
  if (tiles_ub <= 0) return static_cast<int>(cudaGetLastError());

  CUtensorMap krmap, vrmap, qrmap;
  int rc;
  if ((rc = cached4(&krmap, k, (uint64_t)total_kv, (uint64_t)kv_heads, 3,
                    box_tok)))
    return 5000 + rc;
  if ((rc = cached4(&vrmap, v, (uint64_t)total_kv, (uint64_t)kv_heads, 2,
                    box_tok)))
    return 6000 + rc;
  if ((opts & 128) &&
      (rc = cached4(&qrmap, q, (uint64_t)total_q, (uint64_t)heads, 3, 128)))
    return 7000 + rc;
  if (!(opts & 128)) __builtin_memset(&qrmap, 0, sizeof(qrmap));

  dim3 grid(tiles_ub, heads, 1);
  mla_tc4_kernel<<<grid, THREADS, SMEM_BYTES, s>>>(
      krmap, vrmap, qrmap, static_cast<const __nv_bfloat16*>(q),
      static_cast<__nv_bfloat16*>(out), static_cast<float*>(lse),
      static_cast<const float*>(sinks),
      static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
      total_q, batch, heads, kv_heads, window_left, causal, has_lse, has_sink,
      bmm1_scale, bmm2_scale, box_tok, opts,
      reinterpret_cast<unsigned long long*>(dbg));
  return static_cast<int>(cudaGetLastError());
}
