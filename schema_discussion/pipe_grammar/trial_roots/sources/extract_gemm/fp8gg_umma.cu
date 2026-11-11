// V1.6 (wd9): fifth wide-gate clause — kb_count>=16 &&
// items256>=8*nSM && mtiles==ne (high-wave single-mtile-group class;
// wk16's mtiles>=2*ne relaxed to ==ne). Flips exactly d9 (domain;
// c7 excluded at items256 896 < 1184 — its own probe measured null).
// Same-job A/B: d9 -3.21% (wl16b r1, identical arming) / -3.43% (r2)
// = 2/2 meet <=-2%; controls 0.9999; numchk bitwise every block.
// Fixed dispatch identical to V1.5; host dispatch only.
//
// V1.5 (wk12): fourth wide-gate clause — kb_count 8..12 &&
// items256>=10*nSM && mtiles>=ne (high-wave tiny-k class; flips exactly
// c8: mtiles/ne=1.5 misses wk16 structurally). Same-job A/B (3 runs):
// c8 -3.71/-1.98/-3.70% (2/3 >=2%; r2 fully separated min-vs-max; r3
// tiebreak per wk16/wd1 precedent); controls c3/c4 flat, c0 -0.8%;
// numchk bitwise=True maxabs=0 every block of all three jobs (new
// verified class: kb=12, 64-row half-tile A boxes). Kernels identical
// to V1.4; host dispatch only.
//
// V1.4 (wk16): third wide-gate clause — kb_count>=16 && items256>=8*nSM
// && mtiles>=2*ne (high-wave small-k class). Same-job A/B (3 runs +
// domain): c4 -2.8%, c9 -3.1% (2/3 runs >=2%, r3 tiebreak), d2 -4.2%;
// bitwise-identical outputs (numchk clean every block); controls flat.
// Companion pp out-cap probe 66->80MB REJECTED (c8 +1.7%): cap stands.
// Kernels identical to V1.3; host dispatch only.
//
// Cross-call benchmark-specific traversal removed by timing-v2 maintenance.
// V1.2 (wg2): wide-dispatch class addition — kb_count<=48 (k<=6144) &&
// items256*2>=5*nSM && mtiles>=ne (low-wave k=4096 class; c2 -4.02/-4.81%
// in two same-job A/Bs, wg1/wg2). k=7168 low-wave wide EXCLUDED (c0 +7.1%
// measured: 1.84MB per-item B tile, HBM latency unamortized over ~3 waves).
// Kernels identical to V1.1; host dispatch only. results/wg1-findings.md.
//
// V1.1 (wd1): wide-engine rotated tmem drain — next x16-pair load issues
// BEFORE the current FFMA batch (tcgen05.wait::ld is all-outstanding, so
// order is ld,fma,wait); two buffer pairs alternate, fully unrolled; pe
// arrive after the last load retires. Bitwise-identical outputs; 168 regs,
// 0 spills (candidate V1 had 40B spills). Measured (2-variant same-job ABBA,
// dv_14ff0612d664 + 3-block tie-break 20261021T112155Z): c5 -3.7%,
// x1 -5..-7%, c9 0.0/0.0/-0.8; V0 engine byte-identical to V1.
//
// Merged candidate engine pair (V1): [wide] w1f BN=256 engine + [V0] d22g3
// pair-batched 128x128 engine, host-dispatched by shape-derived geometry
// (sched_policy: one indptr D2H per identity). Wide class (measured ab12):
// n%256==0 && kb>=32 && items256>=4*nSM && mtiles>=4*active_groups
// -> c5-class -7.5%, x1-class -7.1%; all other classes stay on V0 bitwise-
// identical paths. Both engines: per-128K fp32 scale promotion, exact e4m3
// MMA accumulation -> cross-engine outputs within <=1 bf16 ulp on eval
// inputs (0 checker mismatches; results/v1-eval-notes.md).
//
// ---- original V0 header ----
// fp8gg_d22g3 — d22g2 + hardened gate identity key (indptr,G,m,n,k):
// (indptr,G) collided across freed-tensor address reuse (job37b: d6 got
// d5's stale sareg). Perf-only bug; paths remain bitwise identical.
// d22g2 note: template-split gated sa path: umma_gemm<1> = d22r
// register prefetch (exact allocation), umma_gemm<0> = V0/d22 cp.async
// ring; host gate G1 (single mega-group) || G2 (uniform <=512 groups &&
// kb_count>=48) selects the instantiation. Bitwise identical paths.
// d22g note: d22r + host-gated sa path (register prefetch OR V0
// cp.async ring; identical bitwise numerics, perf-only policy; gate G1
// = single non-empty group && mtiles>=4 && pairs>=16 -> cp.async).
// d22r note: d22 + register pair-ahead a_scale prefetch (drops the
// cp.async sa ring: no wait_g1 stall, no refill issue in the epilogue).
// Bitwise-identical values/order. Base: c1_d22.
//
// Pair-batched warp-specialized persistent grouped GEMM (PB pipeline):
//   warp0 lane0: TMA producer — 2D tensor-map loads of A/B fp8 tiles
//                (SWIZZLE_128B) into a 3-deep ring of 64KB pair-stages,
//                each holding TWO consecutive 128-K blocks [A0|B0|A1|B1].
//   warp1 lane0: MMA issuer — per pair: tcgen05.mma.cta_group::1.kind::f8f6f4
//                (e4m3, K-major A/B, M=128 N=128, 4x k32 per K block) into
//                tmem accumulator pair ts {(j&1)*2, +1}; ONE tcgen05.commit
//                per pair releases the stage AND signals accumulators ready.
//                v_dualg: when p.dual (host gate: est steady pairs/CTA >= 48)
//                warp2 lane0 co-issues the kb1 quad and both commit
//                (empty_bar count 2) — overlaps issuer management with the
//                dispatch stream; gate-off reproduces the V0 path exactly.
//   warps4-7:    epilogue/promotion (128 threads, thread == tile row):
//                per pair: drain both tmem buffers (tcgen05.ld 128 cols each),
//                FFMA with (a_scale[row,kb] * b_scale[e,kb,nt]) into a
//                register-resident fp32 row accumulator; ONE pair-empty
//                arrive per pair. Tile store: FULL tiles (hi-row0 >= 128)
//                stage 128x128 bf16 swizzled in smem and issue 2 TMA bulk
//                stores (coalesced, offloads the epilogue — interleaved A/B
//                dv_41cc4b9be85d: -4..-20% per shape, replicated within
//                0.5%); ragged tail tiles keep predicated direct uint4
//                stores (no clobber by construction).
// tmem: 512 cols allocated = 4 x 128-col accumulators (TMS=4 via pair ping-
// pong), so MMA of pair j+1 overlaps the epilogue drain of pair j.
// Device-side group scheduler (m_indptr scanned in smem; no host syncs).
// Host: single-node CUDA-graph launch cache (params baked; permanent direct
// fallback on API failure) + 4-slot exact-key tensormap caches (A/B UINT8
// box {128,128}, out UINT16 box {64,128} SWIZZLE_128B).
//
// Why pair-batched: per-k-block sync/wake/commit overhead (~500 cyc) paced
// the single-kb pipeline at ~900 cyc/kb while the MMA issue floor is 256
// (no-TMA diagnostic: 752 cyc/kb, 1646 TF). Batching 2 K blocks per sync
// round halves the events per kb and measured -8..-20% across shapes
// (c5 342.8->274.9us = 1750 TF); evidence: phase2_pb probe dv_5a108fe433b4
// run 20261019T162736Z, all numerics 0-mismatch incl odd-KB pair remainder.
//
// Numerics: e4m3 products accumulate exactly in fp32 tensor-core/tmem; the
// per-128-K-block promotion reproduces the reference dequantize-then-matmul
// mathematics to within fp32 accumulation order (bitwise-equal in practice,
// far inside the frozen tolerance).
//
// Mechanism notes: tcgen05/TMA instruction forms and descriptor encodings
// were studied from public DeepGEMM sm100 headers and the atrex-fa pure-CUDA
// sm100 primitives (smem desc = swz<<61 | 1<<46 | SBO>>4<<32 | LBO>>4<<16 |
// addr>>4 with swz=2 for 128B swizzle, SBO=1024, LBO=16; idesc =
// (M>>5)<<25 | (N>>3)<<17 | 1<<4 for f32 accumulator). All code here is an
// independent implementation; no external library code is called.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

#define DI __device__ __forceinline__

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 128;
constexpr int KB_BYTES = BM * BK;                // 16384 per operand per k-block
constexpr int PSTAGE_BYTES = 4 * KB_BYTES;       // 65536: A0|B0|A1|B1
constexpr int PSTAGES = 3;                       // pair-stages (6 kb lookahead)
constexpr int STAGING_BYTES = 32768;             // TMA-store staging (2 x 64-col halves)
constexpr int TAIL_BYTES = 1024;                 // barriers + schedule (G <= 116)
// X8: V0 epilogue width. NEPI=4 reproduces the certified 128-thread epilogue
// exactly; NEPI=8 (FP8GG_V0_8EPI) splits the 128 tile cols across 8 warps
// (fin[64]/thread, 384-thread block) to cut the per-pair drain span
// (x7 diag dv_8f5760fc..: drain 506-608 cyc -> issuer pe_wait 394-602).
#ifdef FP8GG_V0_8EPI
constexpr int NEPI = 8;
constexpr int THREADS_V = 384;
#else
constexpr int NEPI = 4;
constexpr int THREADS_V = 256;
#endif
constexpr int ECOLS = 512 / NEPI;   // cols per epilogue thread (BN=128: 4->128, 8->64)
// X10: x16-rotated V0 pair drain (wide wd1 pattern) for the certified
// ECOLS=128 layout. Per-element FFMA order unchanged -> bitwise-identical
// (verified x10 dv_64c00c7330cd + interleaved x10b dv_03408bbbcbfa).
// ADOPTED: d6 -0.8% replicated cold+warm in both jobs; fixed-suite V0
// shapes all neutral (1.000); no consistent regression. x10's larger d5/d7
// "wins" were within-job drift artifacts (interleaved re-bench: 1.000).
constexpr bool V0_X16 = (ECOLS == 128);
template <int W>
DI void epi_bar_n() { asm volatile("bar.sync 1, %0;" :: "n"(W) : "memory"); }
constexpr unsigned IDESC_128x128 = ((128u >> 5) << 25) | ((128u >> 3) << 17) | (1u << 4);
constexpr unsigned SWZ_128B = 2;
constexpr unsigned LBO_BYTES = 16;

struct Params {
  const uint8_t* a;   // (m, k) e4m3
  const uint8_t* b;   // (e, n, k) e4m3
  const float* sa;    // (k/128, m)
  const float* sb;    // (e, k/128, n/128)
  const int* indptr;  // (G+1)
  uint16_t* out;      // (m, n) bf16
  int m, n, k, G;
  int n_tiles, kb_count;
  int abox;           // v_abox: adaptive A-box selection enabled
  int dual;           // v_dualg: gated kb-split dual-thread MMA issue
  int sareg;          // d22g: 1 = register sa prefetch, 0 = cp.async ring
  int bstream;        // X3-B: 1 = B read exactly once (mtiles==ne) -> TMA
                      // loads of B use an L2 evict_first policy hint
};

DI uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

DI bool bar_wait(uint32_t bar, unsigned parity) {
  unsigned done;
  asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               "selp.u32 %0, 1, 0, p;\n}"
               : "=r"(done) : "r"(bar), "r"(parity));
  return done != 0;
}

DI void bar_spin(uint32_t bar, unsigned parity) {
  while (!bar_wait(bar, parity)) __nanosleep(32);
}

DI void bar_init(uint32_t bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}

DI void bar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar));
}

DI void bar_expect_tx(uint32_t bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes));
}

DI void tma_load_2d(const CUtensorMap* map, uint32_t dst, uint32_t bar, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(bar)
               : "memory");
}

// X3-B: L2 evict_first policy hint for stream-class B tiles (read exactly
// once per kernel). Same-job A/B (x3_harness dv_d045d9cca98f): c1 -9.7%,
// c7 -8.5%, d9 -2.4%, bitwise-identical; gated at runtime by Params::bstream
// because any B reuse regresses (c4 +4.4%, c8 +1.0%).
DI uint64_t make_evict_first_policy() {
  uint64_t pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
DI void tma_load_2d_ef(const CUtensorMap* map, uint32_t dst, uint32_t bar, int x, int y,
                       uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(bar),
                  "l"(pol)
               : "memory");
}
DI void tma_store_2d(const CUtensorMap* map, uint32_t src, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
               :: "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(src) : "memory");
}
DI void bulk_commit_group() { asm volatile("cp.async.bulk.commit_group;"); }
DI void bulk_wait_group_read0() { asm volatile("cp.async.bulk.wait_group.read 0;"); }
DI void bulk_wait_group0() { asm volatile("cp.async.bulk.wait_group 0;"); }
DI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;"); }
DI void epi_bar() { asm volatile("bar.sync 1, 128;" ::: "memory"); }
DI void cp_async4_pred(uint32_t dst, const void* src, bool pred) {
  const int sz = pred ? 4 : 0;   // 0 -> zero-fill, src not dereferenced
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
               :: "r"(dst), "l"(src), "r"(sz));
}
DI void cp_async_commit_g() { asm volatile("cp.async.commit_group;"); }
DI void cp_async_wait_g1() { asm volatile("cp.async.wait_group 1;"); }

DI void umma_f8(uint32_t d_tmem, uint64_t ad, uint64_t bd, uint32_t idesc, unsigned acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}"
               :: "r"(d_tmem), "l"(ad), "l"(bd), "r"(idesc), "r"(acc));
}

DI void umma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(bar));
}

DI void tmem_ld_x32(uint32_t taddr, uint32_t v[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=r"(v[0]),"=r"(v[1]),"=r"(v[2]),"=r"(v[3]),"=r"(v[4]),"=r"(v[5]),"=r"(v[6]),"=r"(v[7]),
                 "=r"(v[8]),"=r"(v[9]),"=r"(v[10]),"=r"(v[11]),"=r"(v[12]),"=r"(v[13]),"=r"(v[14]),"=r"(v[15]),
                 "=r"(v[16]),"=r"(v[17]),"=r"(v[18]),"=r"(v[19]),"=r"(v[20]),"=r"(v[21]),"=r"(v[22]),"=r"(v[23]),
                 "=r"(v[24]),"=r"(v[25]),"=r"(v[26]),"=r"(v[27]),"=r"(v[28]),"=r"(v[29]),"=r"(v[30]),"=r"(v[31])
               : "r"(taddr));
}

DI void tmem_ld_x16(uint32_t taddr, uint32_t v[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(v[0]),"=r"(v[1]),"=r"(v[2]),"=r"(v[3]),"=r"(v[4]),"=r"(v[5]),"=r"(v[6]),"=r"(v[7]),
                 "=r"(v[8]),"=r"(v[9]),"=r"(v[10]),"=r"(v[11]),"=r"(v[12]),"=r"(v[13]),"=r"(v[14]),"=r"(v[15])
               : "r"(taddr));
}

DI void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
DI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;"); }

DI uint64_t make_desc(uint32_t smem_addr) {
  return ((uint64_t)SWZ_128B << 61) | (1ULL << 46) | ((1024ULL >> 4) << 32) |
         (((uint64_t)(LBO_BYTES >> 4)) << 16) | ((smem_addr & 0xFFFFFFu) >> 4);
}

DI uint32_t pack_bf16(float x, float y) {
  __nv_bfloat162 h = __float22bfloat162_rn(make_float2(x, y));
  return *reinterpret_cast<uint32_t*>(&h);
}

}  // namespace

// ---- V0 diagnostic instrumentation (compile-time only; inert in shipped
// builds: with V0_DIAG==0 every guarded statement constant-folds away) ----
// The array itself is unconditional (512B inert device global) so that the
// runtime-guarded instrumentation statements compile in shipped builds, where
// V0_DIAG==0 constant-folds them away (zero code, zero runtime cost).
__device__ unsigned long long g_v0_diag[64];
#ifdef FP8GG_DIAG_V0
#define V0_DIAG 1
extern "C" int fp8umma_v0diag_pull(unsigned long long* h, int n) {
  if (n > 64) n = 64;
  return (int)cudaMemcpyFromSymbol(h, g_v0_diag, (size_t)n * 8, 0,
                                   cudaMemcpyDeviceToHost);
}
extern "C" int fp8umma_v0diag_reset() {
  const unsigned long long z[64] = {0};
  return (int)cudaMemcpyToSymbol(g_v0_diag, z, sizeof(z));
}
#else
#define V0_DIAG 0
#endif

template <int SAREG>
__global__ void __launch_bounds__(THREADS_V, 1)
umma_gemm(const Params p, const __grid_constant__ CUtensorMap tA,
          const __grid_constant__ CUtensorMap tA64, const __grid_constant__ CUtensorMap tA32,
          const __grid_constant__ CUtensorMap tA16,
          const __grid_constant__ CUtensorMap tB, const __grid_constant__ CUtensorMap tOut) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // smem layout: pair-stage ring | TMA-store staging (32KB) | barrier tail:
  //   full_bar[PSTAGES] | empty_bar[PSTAGES] (commit targets) | pe[2]
  uint8_t* staging = smem + PSTAGES * PSTAGE_BYTES;
  // v_sload: 2KB double-buffered row-private a_scale prefetch region
  float* sa_s = reinterpret_cast<float*>(staging + STAGING_BYTES);
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(staging + STAGING_BYTES + 2048);
  uint64_t* empty_bar = full_bar + PSTAGES;
  uint64_t* pe_bar = empty_bar + PSTAGES;   // pair epilogue-done, count 128
  uint32_t* s_tmem = reinterpret_cast<uint32_t*>(pe_bar + 2);
  int* s_ind = reinterpret_cast<int*>(s_tmem + 1);
  int* s_pref = s_ind + p.G + 1;

  for (int i = tid; i <= p.G; i += THREADS_V) s_ind[i] = p.indptr[i];
  __syncthreads();
  if (tid == 0) {
    int acc = 0;
    for (int e = 0; e < p.G; ++e) {
      s_pref[e] = acc;
      acc += (s_ind[e + 1] - s_ind[e] + BM - 1) / BM;
    }
    s_pref[p.G] = acc;
    for (uint32_t i = 0; i < PSTAGES * 2 + 2; ++i)
      bar_init(su32(full_bar + i),
                 (i >= PSTAGES * 2) ? (unsigned)(NEPI * 32)
                                    : (i >= PSTAGES && p.dual ? 2u : 1u));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 1) {
    // 512 tmem cols = 4 x 128-col accumulators (pair ping-pong x 2 pairs).
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(su32(s_tmem)), "r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  const uint32_t tbase = *s_tmem;
  const int W = s_pref[p.G] * p.n_tiles;
  const int KB = p.kb_count;
  const int NB = p.n_tiles;
  const int PAIRS = (KB + 1) >> 1;

  if (warp == 0 && lane == 0) {
    // ---------------- TMA producer (one round per K-block PAIR) ----------------
    uint32_t stage = 0, phase = 0;
    int e_hint = 0;
    const uint64_t bpol = p.bstream ? make_evict_first_policy() : 0ull;
    const bool diag = (V0_DIAG != 0) && (blockIdx.x == 0);
    unsigned long long acc_ew = 0, acc_is = 0, acc_fw = 0, n_fw = 0, n_pr = 0,
                       p_t0 = 0, p_t1 = 0;
    for (int w = blockIdx.x; w < W; w += gridDim.x) {
      // Monotonic group scan: items arrive in increasing w per CTA, so the
      // group index never decreases — start from the previous item's group
      // (O(G) -> amortized O(G/items); measured -9.6% c8, -9.0% d2-class,
      // dv_8af8766c22ff).
      const int wi = (int)w;
      int e = e_hint;
      while (s_pref[e + 1] * NB <= wi) ++e;
      e_hint = e;
      const int r = wi - s_pref[e] * NB;
      const int tiles_e = s_pref[e + 1] - s_pref[e];
      const int nt = r / tiles_e, mt = r % tiles_e;
      const int row0 = s_ind[e] + mt * BM;
      const int brow = e * p.n + nt * BN;
      // v_abox: tightest A box for this tile's valid rows (stale smem above
      // the box feeds masked-away MMA rows, exactly like OOB zero-fill).
      // Box rounds UP: vrows in (64,128) -> 128, (32,64] -> 64, (16,32] -> 32, else 16.
      const int vrows = s_ind[e + 1] - row0;
      int boxh = 128;
      if (p.abox && vrows < 128)
        boxh = vrows > 64 ? 128 : (vrows > 32 ? 64 : (vrows > 16 ? 32 : 16));
      const CUtensorMap* mapA =
          boxh == 128 ? &tA : (boxh == 64 ? &tA64 : (boxh == 32 ? &tA32 : &tA16));
      const unsigned abytes = (unsigned)(boxh * BK);
      #pragma unroll 1
      for (int j = 0; j < PAIRS; ++j) {
        const int kb0 = 2 * j, kb1 = 2 * j + 1;
        const bool has1 = kb1 < KB;
        if (diag) p_t0 = clock64();
        bar_spin(su32(empty_bar + stage), phase ^ 1);
        if (diag) p_t1 = clock64();
        const uint32_t fb = su32(full_bar + stage);
        const uint8_t* st = smem + stage * PSTAGE_BYTES;
        bar_expect_tx(fb, has1 ? (2u * abytes + 2u * KB_BYTES) : (abytes + KB_BYTES));
        tma_load_2d(mapA, su32(st), fb, kb0 * BK, row0);
        if (p.bstream) tma_load_2d_ef(&tB, su32(st + KB_BYTES), fb, kb0 * BK, brow, bpol);
        else           tma_load_2d(&tB, su32(st + KB_BYTES), fb, kb0 * BK, brow);
        if (has1) {
          tma_load_2d(mapA, su32(st + 2 * KB_BYTES), fb, kb1 * BK, row0);
          if (p.bstream) tma_load_2d_ef(&tB, su32(st + 3 * KB_BYTES), fb, kb1 * BK, brow, bpol);
          else           tma_load_2d(&tB, su32(st + 3 * KB_BYTES), fb, kb1 * BK, brow);
        }
        if (diag) {
          const unsigned long long t2 = clock64();
          acc_ew += p_t1 - p_t0; acc_is += t2 - p_t1; ++n_pr;
          if (j < 2) { acc_fw += p_t1 - p_t0; ++n_fw; }
          g_v0_diag[8] = acc_ew; g_v0_diag[9] = acc_is; g_v0_diag[10] = n_pr;
          g_v0_diag[11] = acc_fw; g_v0_diag[12] = n_fw;
        }
        if (++stage == PSTAGES) { stage = 0; phase ^= 1; }
      }
    }
  } else if ((warp == 1 || (warp == 2 && p.dual)) && lane == 0) {
    // ------- UMMA issuer (v_dualg2: gate-hoisted kb-split dual issue) ------
    // Mode branch hoisted OUT of the loops: the single-mode copy below is
    // textually V0-exact (job-16 showed runtime isA/dual predicates inside
    // the unroll-2 body cost ~1.5% on ~26-pair/CTA gate-off shapes, d7).
    // dual: warp1 = kb0 quad -> dt0, warp2 lane0 = kb1 quad -> dt1, both
    // commit (empty_bar count 2); identical stage/phase/pj bookkeeping.
    uint32_t stage = 0, phase = 0;
    unsigned long pj = 0;   // global pair index (continues across items)
    const bool diag = (V0_DIAG != 0) && (blockIdx.x == 0);
    const int dbase = (warp == 1) ? 0 : 4;
    unsigned long long acc_f = 0, acc_p = 0, acc_i = 0, i_t0 = 0, i_t1 = 0, i_t2 = 0;
    if (diag && warp == 1) { i_t0 = clock64(); g_v0_diag[30] = i_t0; }
    if (p.dual) {
      const bool isA = (warp == 1);
      for (int w = blockIdx.x; w < W; w += gridDim.x) {
        #pragma unroll 2
        for (int j = 0; j < PAIRS; ++j, ++pj) {
          const bool has1 = (2 * j + 1) < KB;
          if (diag) i_t0 = clock64();
          bar_spin(su32(full_bar + stage), phase);
          if (diag) i_t1 = clock64();
          const unsigned pb = pj & 1;             // tmem pair slot
          const unsigned long puse = pj >> 1;     // use # of pe_bar[pb]
          bar_spin(su32(pe_bar + pb), (unsigned)((puse - 1) & 1));
          tc_fence_after();
          if (diag) i_t2 = clock64();
          const uint32_t sb0 = su32(smem + stage * PSTAGE_BYTES);
          if (isA) {
            uint64_t ad = make_desc(sb0);
            uint64_t bd = make_desc(sb0 + KB_BYTES);
            const uint32_t dt0 = tbase + (pb * 2) * BN;
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
              umma_f8(dt0, ad, bd, IDESC_128x128, step > 0 ? 1u : 0u);
              ad += 2; bd += 2;
            }
          } else if (has1) {
            uint64_t ad1 = make_desc(sb0 + 2 * KB_BYTES);
            uint64_t bd1 = make_desc(sb0 + 3 * KB_BYTES);
            const uint32_t dt1 = tbase + (pb * 2 + 1) * BN;
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
              umma_f8(dt1, ad1, bd1, IDESC_128x128, step > 0 ? 1u : 0u);
              ad1 += 2; bd1 += 2;
            }
          }
          umma_commit(su32(empty_bar + stage));
          if (diag) {
            const unsigned long long t3 = clock64();
            acc_f += i_t1 - i_t0; acc_p += i_t2 - i_t1; acc_i += t3 - i_t2;
            g_v0_diag[dbase + 0] = acc_f; g_v0_diag[dbase + 1] = acc_p;
            g_v0_diag[dbase + 2] = acc_i; g_v0_diag[dbase + 3] = pj + 1;
            if (warp == 1) g_v0_diag[31] = t3;
          }
          if (++stage == PSTAGES) { stage = 0; phase ^= 1; }
        }
      }
    } else {
      for (int w = blockIdx.x; w < W; w += gridDim.x) {
        // unroll 2: halves loop bookkeeping on the issue-critical warp;
        // interleaved A/B (dv_f8749c372a4c): 8/8 shapes improved (d3 -2.1%,
        // c0 -1.4%, f9 -1.1%, c5 -0.8%), numerics 15/15 clean.
        #pragma unroll 2
        for (int j = 0; j < PAIRS; ++j, ++pj) {
          const bool has1 = (2 * j + 1) < KB;
          if (diag) i_t0 = clock64();
          bar_spin(su32(full_bar + stage), phase);
          if (diag) i_t1 = clock64();
          const unsigned pb = pj & 1;             // tmem pair slot
          const unsigned long puse = pj >> 1;     // use # of pe_bar[pb]
          bar_spin(su32(pe_bar + pb), (unsigned)((puse - 1) & 1));
          tc_fence_after();
          if (diag) i_t2 = clock64();
          const uint32_t sb0 = su32(smem + stage * PSTAGE_BYTES);
          uint64_t ad = make_desc(sb0);
          uint64_t bd = make_desc(sb0 + KB_BYTES);
          const uint32_t dt0 = tbase + (pb * 2) * BN;
          #pragma unroll
          for (int step = 0; step < 4; ++step) {
            umma_f8(dt0, ad, bd, IDESC_128x128, step > 0 ? 1u : 0u);
            ad += 2; bd += 2;
          }
          if (has1) {
            uint64_t ad1 = make_desc(sb0 + 2 * KB_BYTES);
            uint64_t bd1 = make_desc(sb0 + 3 * KB_BYTES);
            const uint32_t dt1 = tbase + (pb * 2 + 1) * BN;
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
              umma_f8(dt1, ad1, bd1, IDESC_128x128, step > 0 ? 1u : 0u);
              ad1 += 2; bd1 += 2;
            }
          }
          // ONE commit per pair: releases the stage to the producer AND tells
          // the epilogue both accumulators are ready (single-commit pattern,
          // fence lab F4; halved per-kb events in the PB pipeline).
          umma_commit(su32(empty_bar + stage));
          if (diag) {
            const unsigned long long t3 = clock64();
            acc_f += i_t1 - i_t0; acc_p += i_t2 - i_t1; acc_i += t3 - i_t2;
            g_v0_diag[dbase + 0] = acc_f; g_v0_diag[dbase + 1] = acc_p;
            g_v0_diag[dbase + 2] = acc_i; g_v0_diag[dbase + 3] = pj + 1;
            g_v0_diag[31] = t3;
          }
          if (++stage == PSTAGES) { stage = 0; phase ^= 1; }
        }
      }
    }
  } else if (warp >= 4) {
  if constexpr (SAREG == 1) {
    // ---------------- epilogue / promotion (thread == tile row) ----------------
    const int row = (warp & 3) * 32 + lane;
    const int grp = (warp - 4) >> 2;                 // X8: col group (NEPI=4 -> 0)
    const int coloff = grp * ECOLS;
    const bool issuer = (warp == 4 && lane == 0);   // TMA-store issuer
    unsigned long pj = 0;
    int e_hint = 0;
    bool prev_store = false;
    const bool diag = (V0_DIAG != 0) && (blockIdx.x == 0) && (warp == 4) && (lane == 0);
    unsigned long long acc_dw = 0, acc_dr = 0, acc_sc = 0, acc_st = 0, n_ep = 0,
                       n_it = 0, acc_few = 0, acc_fdr = 0, n_fpair = 0,
                       e_t0 = 0, e_t1 = 0, t_sc = 0;
    for (int w = blockIdx.x; w < W; w += gridDim.x) {
      const int wi = (int)w;
      int e = e_hint;   // monotonic scan (see producer comment)
      while (s_pref[e + 1] * NB <= wi) ++e;
      e_hint = e;
      const int r = wi - s_pref[e] * NB;
      const int tiles_e = s_pref[e + 1] - s_pref[e];
      const int nt = r / tiles_e, mt = r % tiles_e;
      const int lo = s_ind[e], hi = s_ind[e + 1];
      const int row0 = lo + mt * BM;
      const bool valid = row < hi - row0;
      const float* sap = p.sa + row0 + row;
      const float* sbp = p.sb + (long)e * KB * NB + nt;
      uint16_t* op = p.out + (long)(row0 + row) * p.n + nt * BN + coloff;

      float fin[ECOLS];
      #pragma unroll
      for (int j = 0; j < ECOLS; ++j) fin[j] = 0.f;

      // d22r: a_scale via register pair-ahead __ldg (c = pair j, n = pair
      // j+1; in-loop refill targets pair j+2 = kb 2j+4/2j+5). Row-private
      // registers; both kb words per pair, predicated past the tensor end.
      const bool inb = (row0 + row) < p.m;
      float sa_c0 = inb ? __ldg(sap) : 0.f;
      float sa_c1 = (inb && 1 < KB) ? __ldg(sap + (long)1 * p.m) : 0.f;
      float sa_n0 = (inb && 2 < KB) ? __ldg(sap + (long)2 * p.m) : 0.f;
      float sa_n1 = (inb && 3 < KB) ? __ldg(sap + (long)3 * p.m) : 0.f;
      float sb_c = __ldg(sbp);
      // v_sbpre: kb1's b_scale rides the same one-pair-ahead register
      // prefetch as sb_0 -> both scale loads issue with ~1250 cyc slack
      // instead of ~300 (borderline vs L2 latency). Values/order unchanged.
      float sb_n = (1 < KB) ? __ldg(sbp + (long)1 * NB) : 0.f;
      #pragma unroll 1
      for (int j = 0; j < PAIRS; ++j, ++pj) {
        const int kb1 = 2 * j + 1;
        const bool has1 = kb1 < KB;
        const float sa_0 = sa_c0;
        const float sb_0 = sb_c;
        const float sa_1 = has1 ? sa_c1 : 0.f;
        const float sb_1 = has1 ? sb_n : 0.f;
        sa_c0 = sa_n0; sa_c1 = sa_n1;
        const int kbn0 = 2 * j + 4;    // pair j+2's kb0
        sa_n0 = (inb && kbn0 < KB) ? __ldg(sap + (long)kbn0 * p.m) : 0.f;
        sa_n1 = (inb && kbn0 + 1 < KB) ? __ldg(sap + (long)(kbn0 + 1) * p.m) : 0.f;
        const bool hasnext = (j + 1 < PAIRS);
        const int kbN = 2 * j + 2;
        sb_c = hasnext ? __ldg(sbp + (long)kbN * NB) : 0.f;
        sb_n = (hasnext && (kbN + 1) < KB) ? __ldg(sbp + (long)(kbN + 1) * NB) : 0.f;

        const unsigned pb = pj & 1;
        // accumulator-ready == the pair's commit completion on empty_bar
        if (diag) e_t0 = clock64();
        bar_spin(su32(empty_bar + (pj % PSTAGES)), (unsigned)((pj / PSTAGES) & 1));
        if (diag) e_t1 = clock64();
        tc_fence_after();
        if constexpr (V0_X16) {
          // X10: x16-rotated pair drain (wide wd1 pattern generalized to the
          // 2-kb pair): the next 32-col batch's loads issue before the
          // current batch's FFMA so tmem-port reads overlap ALU (wait::ld is
          // all-outstanding -> issue,fma,wait order). Two buffer pairs
          // (v0*,v1*) alternate; fully unrolled so ptxas renames without
          // copies. pe arrive after the LAST load retires, before the final
          // FFMA batch (same relative point as wide wd1). Per-element FFMA
          // order (kb0 then kb1, increasing col) matches the x32-batched
          // code exactly -> bitwise-identical outputs.
          const float sc0 = sa_0 * sb_0;
          const float sc1 = has1 ? (sa_1 * sb_1) : 0.f;
          if (diag) t_sc = clock64();
          const uint32_t ta0 = tbase + (pb * 2) * BN + coloff;
          uint32_t v0a[16], v0b[16], v1a[16], v1b[16];
          tmem_ld_x16(ta0 + 0 * 32 + 0, v0a);
          tmem_ld_x16(ta0 + 0 * 32 + 16, v0b);
          tmem_ld_wait();
          tmem_ld_x16(ta0 + 1 * 32 + 0, v1a);
          tmem_ld_x16(ta0 + 1 * 32 + 16, v1b);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[0 * 32 + i] = fmaf(sc0, __uint_as_float(v0a[i]), fin[0 * 32 + i]);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[0 * 32 + 16 + i] = fmaf(sc0, __uint_as_float(v0b[i]), fin[0 * 32 + 16 + i]);
          tmem_ld_wait();
          tmem_ld_x16(ta0 + 2 * 32 + 0, v0a);
          tmem_ld_x16(ta0 + 2 * 32 + 16, v0b);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[1 * 32 + i] = fmaf(sc0, __uint_as_float(v1a[i]), fin[1 * 32 + i]);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[1 * 32 + 16 + i] = fmaf(sc0, __uint_as_float(v1b[i]), fin[1 * 32 + 16 + i]);
          tmem_ld_wait();
          tmem_ld_x16(ta0 + 3 * 32 + 0, v1a);
          tmem_ld_x16(ta0 + 3 * 32 + 16, v1b);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[2 * 32 + i] = fmaf(sc0, __uint_as_float(v0a[i]), fin[2 * 32 + i]);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[2 * 32 + 16 + i] = fmaf(sc0, __uint_as_float(v0b[i]), fin[2 * 32 + 16 + i]);
          tmem_ld_wait();
          if (has1) {
            const uint32_t ta1 = ta0 + BN;
            tmem_ld_x16(ta1 + 0 * 32 + 0, v0a);
            tmem_ld_x16(ta1 + 0 * 32 + 16, v0b);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + i] = fmaf(sc0, __uint_as_float(v1a[i]), fin[3 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + 16 + i] = fmaf(sc0, __uint_as_float(v1b[i]), fin[3 * 32 + 16 + i]);
            tmem_ld_wait();
            tmem_ld_x16(ta1 + 1 * 32 + 0, v1a);
            tmem_ld_x16(ta1 + 1 * 32 + 16, v1b);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[0 * 32 + i] = fmaf(sc1, __uint_as_float(v0a[i]), fin[0 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[0 * 32 + 16 + i] = fmaf(sc1, __uint_as_float(v0b[i]), fin[0 * 32 + 16 + i]);
            tmem_ld_wait();
            tmem_ld_x16(ta1 + 2 * 32 + 0, v0a);
            tmem_ld_x16(ta1 + 2 * 32 + 16, v0b);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[1 * 32 + i] = fmaf(sc1, __uint_as_float(v1a[i]), fin[1 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[1 * 32 + 16 + i] = fmaf(sc1, __uint_as_float(v1b[i]), fin[1 * 32 + 16 + i]);
            tmem_ld_wait();
            tmem_ld_x16(ta1 + 3 * 32 + 0, v1a);
            tmem_ld_x16(ta1 + 3 * 32 + 16, v1b);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[2 * 32 + i] = fmaf(sc1, __uint_as_float(v0a[i]), fin[2 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[2 * 32 + 16 + i] = fmaf(sc1, __uint_as_float(v0b[i]), fin[2 * 32 + 16 + i]);
            tmem_ld_wait();               // all tmem reads of this slot done
            bar_arrive(su32(pe_bar + pb));
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + i] = fmaf(sc1, __uint_as_float(v1a[i]), fin[3 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + 16 + i] = fmaf(sc1, __uint_as_float(v1b[i]), fin[3 * 32 + 16 + i]);
          } else {
            // odd-KB tail pair: only kb0 was drained; its batch 3 sits in v1*.
            bar_arrive(su32(pe_bar + pb));
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + i] = fmaf(sc0, __uint_as_float(v1a[i]), fin[3 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + 16 + i] = fmaf(sc0, __uint_as_float(v1b[i]), fin[3 * 32 + 16 + i]);
          }
        } else {
        {
          const float sc = sa_0 * sb_0;
          if (diag) t_sc = clock64();
          const uint32_t ta = tbase + (pb * 2) * BN + coloff;
          // d22: batch tcgen05.ld 2-deep per wait; X8: ECOLS=64 (NEPI=8)
          // covers the thread's cols in one batch, ECOLS=128 (NEPI=4,
          // certified layout) in two. Per-element FFMA order unchanged.
          uint32_t va[32], vb[32];
          tmem_ld_x32(ta + 0 * 32, va);
          if constexpr (ECOLS > 32) tmem_ld_x32(ta + 1 * 32, vb);
          tmem_ld_wait();
          #pragma unroll
          for (int i = 0; i < 32; ++i) fin[0 * 32 + i] = fmaf(sc, __uint_as_float(va[i]), fin[0 * 32 + i]);
          if constexpr (ECOLS > 32) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[1 * 32 + i] = fmaf(sc, __uint_as_float(vb[i]), fin[1 * 32 + i]);
          }
          if constexpr (ECOLS > 64) {
            tmem_ld_x32(ta + 2 * 32, va);
            tmem_ld_x32(ta + 3 * 32, vb);
            tmem_ld_wait();
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[2 * 32 + i] = fmaf(sc, __uint_as_float(va[i]), fin[2 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[3 * 32 + i] = fmaf(sc, __uint_as_float(vb[i]), fin[3 * 32 + i]);
          }
        }
        if (has1) {
          const float sc = sa_1 * sb_1;
          const uint32_t ta = tbase + (pb * 2 + 1) * BN + coloff;
          uint32_t va[32], vb[32];
          tmem_ld_x32(ta + 0 * 32, va);
          if constexpr (ECOLS > 32) tmem_ld_x32(ta + 1 * 32, vb);
          tmem_ld_wait();
          #pragma unroll
          for (int i = 0; i < 32; ++i) fin[0 * 32 + i] = fmaf(sc, __uint_as_float(va[i]), fin[0 * 32 + i]);
          if constexpr (ECOLS > 32) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[1 * 32 + i] = fmaf(sc, __uint_as_float(vb[i]), fin[1 * 32 + i]);
          }
          if constexpr (ECOLS > 64) {
            tmem_ld_x32(ta + 2 * 32, va);
            tmem_ld_x32(ta + 3 * 32, vb);
            tmem_ld_wait();
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[2 * 32 + i] = fmaf(sc, __uint_as_float(va[i]), fin[2 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[3 * 32 + i] = fmaf(sc, __uint_as_float(vb[i]), fin[3 * 32 + i]);
          }
        }
        bar_arrive(su32(pe_bar + pb));
        }
        if (diag) {
          const unsigned long long t2 = clock64();
          acc_dw += e_t1 - e_t0; acc_dr += t2 - e_t1; ++n_ep;
          acc_sc += t_sc - e_t1;
          if (j < 2) { acc_few += e_t1 - e_t0; acc_fdr += t2 - e_t1; ++n_fpair; }
          g_v0_diag[16] = acc_dw; g_v0_diag[17] = acc_dr; g_v0_diag[18] = n_ep;
          g_v0_diag[21] = acc_few; g_v0_diag[22] = acc_fdr; g_v0_diag[23] = n_fpair;
          g_v0_diag[24] = acc_sc;
        }
      }

      // ---- tile store: TMA-store for FULL tiles (all 128 box rows owned by
      // this tile -> no clobber of neighboring groups), direct stores for
      // ragged tail tiles. Interleaved A/B (dv_41cc4b9be85d, 2 replicates
      // within 0.5%): -4.2% c0, -12.8% c6, -4.5% mt, -9.6% d0, -19.7%
      // fixed9-class, -17.3% d2-class, neutral at m=4 (path never taken).
      if (diag) e_t0 = clock64();
      const bool full_tile = (hi - row0) >= BM;
      if (full_tile) {
        if (issuer && prev_store) bulk_wait_group_read0();
        epi_bar_n<NEPI * 32>();
        #pragma unroll
        for (int cc = 0; cc < ECOLS / 8; ++cc) {
          uint4 pk;
          const int col = cc * 8;
          pk.x = pack_bf16(fin[col + 0], fin[col + 1]);
          pk.y = pack_bf16(fin[col + 2], fin[col + 3]);
          pk.z = pack_bf16(fin[col + 4], fin[col + 5]);
          pk.w = pack_bf16(fin[col + 6], fin[col + 7]);
          // canonical SWIZZLE_128B staging: chunk ci of row r at ((ci^(r&7))*16)
          // inside the 16KB box holding tile cols [64*box, 64*box+64).
          const int box = (coloff + col) >> 6, ci = cc & 7;
          *reinterpret_cast<uint4*>(staging + box * 16384 + row * 128
                                    + ((ci ^ (row & 7)) * 16)) = pk;
        }
        fence_proxy_async();
        epi_bar_n<NEPI * 32>();
        if (issuer) {
          tma_store_2d(&tOut, su32(staging), nt * 128, row0);
          tma_store_2d(&tOut, su32(staging + 16384), nt * 128 + 64, row0);
          bulk_commit_group();
          prev_store = true;
        }
      } else if (valid) {
        #pragma unroll
        for (int c = 0; c < ECOLS / 8; ++c) {
          uint4 pk;
          pk.x = pack_bf16(fin[c * 8 + 0], fin[c * 8 + 1]);
          pk.y = pack_bf16(fin[c * 8 + 2], fin[c * 8 + 3]);
          pk.z = pack_bf16(fin[c * 8 + 4], fin[c * 8 + 5]);
          pk.w = pack_bf16(fin[c * 8 + 6], fin[c * 8 + 7]);
          *reinterpret_cast<uint4*>(op + c * 8) = pk;
        }
      }
      if (diag) { e_t1 = clock64(); acc_st += e_t1 - e_t0; ++n_it;
                  g_v0_diag[19] = acc_st; g_v0_diag[20] = n_it; }
    }
    if (issuer && prev_store) bulk_wait_group0();   // stores complete before exit
  } else {
    // ---------------- epilogue / promotion (thread == tile row) ----------------
    const int row = (warp & 3) * 32 + lane;
    const int grp = (warp - 4) >> 2;                 // X8: col group (NEPI=4 -> 0)
    const int coloff = grp * ECOLS;
    const bool issuer = (warp == 4 && lane == 0);   // TMA-store issuer
    unsigned long pj = 0;
    int e_hint = 0;
    bool prev_store = false;
    const bool diag = (V0_DIAG != 0) && (blockIdx.x == 0) && (warp == 4) && (lane == 0);
    unsigned long long acc_dw = 0, acc_dr = 0, acc_sc = 0, acc_st = 0, n_ep = 0,
                       n_it = 0, acc_few = 0, acc_fdr = 0, n_fpair = 0,
                       e_t0 = 0, e_t1 = 0, t_sc = 0;
    for (int w = blockIdx.x; w < W; w += gridDim.x) {
      const int wi = (int)w;
      int e = e_hint;   // monotonic scan (see producer comment)
      while (s_pref[e + 1] * NB <= wi) ++e;
      e_hint = e;
      const int r = wi - s_pref[e] * NB;
      const int tiles_e = s_pref[e + 1] - s_pref[e];
      const int nt = r / tiles_e, mt = r % tiles_e;
      const int lo = s_ind[e], hi = s_ind[e + 1];
      const int row0 = lo + mt * BM;
      const bool valid = row < hi - row0;
      const float* sap = p.sa + row0 + row;
      const float* sbp = p.sb + (long)e * KB * NB + nt;
      uint16_t* op = p.out + (long)(row0 + row) * p.n + nt * BN + coloff;

      float fin[ECOLS];
      #pragma unroll
      for (int j = 0; j < ECOLS; ++j) fin[j] = 0.f;

      // d22g sareg=0: V0/d22 cp.async sa ring (2KB double-buffered,
      // row-private). Chosen for single-mega-group shapes (gate G1) where
      // the register path measured -9% (domain1, jobs 29/30/32).
      const bool inb = (row0 + row) < p.m;
      float* mybuf = sa_s + row;
      #pragma unroll
      for (int q = 0; q < 2; ++q) {
        if (q < PAIRS) {
          const int kbq0 = 2 * q;
          cp_async4_pred(su32(mybuf + q * 256), sap + (long)kbq0 * p.m, inb);
          cp_async4_pred(su32(mybuf + q * 256 + 128),
                         sap + (long)(kbq0 + 1) * p.m, inb && (kbq0 + 1 < KB));
        }
        cp_async_commit_g();   // always: keeps the wait_group-1 invariant
      }
      float sb_c = __ldg(sbp);
      // v_sbpre: kb1's b_scale rides the same one-pair-ahead register
      // prefetch as sb_0 -> both scale loads issue with ~1250 cyc slack
      // instead of ~300 (borderline vs L2 latency). Values/order unchanged.
      float sb_n = (1 < KB) ? __ldg(sbp + (long)1 * NB) : 0.f;
      #pragma unroll 1
      for (int j = 0; j < PAIRS; ++j, ++pj) {
        const int kb1 = 2 * j + 1;
        const bool has1 = kb1 < KB;
        cp_async_wait_g1();    // pair j's slot (group j) complete
        const float sa_0 = mybuf[(j & 1) * 256];
        const float sb_0 = sb_c;
        const float sa_1 = has1 ? mybuf[(j & 1) * 256 + 128] : 0.f;
        const float sb_1 = has1 ? sb_n : 0.f;
        const bool hasnext = (j + 1 < PAIRS);
        const int kbN = 2 * j + 2;
        sb_c = hasnext ? __ldg(sbp + (long)kbN * NB) : 0.f;
        sb_n = (hasnext && (kbN + 1) < KB) ? __ldg(sbp + (long)(kbN + 1) * NB) : 0.f;

        const unsigned pb = pj & 1;
        // accumulator-ready == the pair's commit completion on empty_bar
        if (diag) e_t0 = clock64();
        bar_spin(su32(empty_bar + (pj % PSTAGES)), (unsigned)((pj / PSTAGES) & 1));
        if (diag) e_t1 = clock64();
        tc_fence_after();
        if constexpr (V0_X16) {
          // X10: x16-rotated pair drain — see the SAREG=1 copy for the full
          // rationale. Bitwise-identical per-element FFMA order; pe arrive
          // after the last load wait, before the final FFMA batch.
          const float sc0 = sa_0 * sb_0;
          const float sc1 = has1 ? (sa_1 * sb_1) : 0.f;
          if (diag) t_sc = clock64();
          const uint32_t ta0 = tbase + (pb * 2) * BN + coloff;
          uint32_t v0a[16], v0b[16], v1a[16], v1b[16];
          tmem_ld_x16(ta0 + 0 * 32 + 0, v0a);
          tmem_ld_x16(ta0 + 0 * 32 + 16, v0b);
          tmem_ld_wait();
          tmem_ld_x16(ta0 + 1 * 32 + 0, v1a);
          tmem_ld_x16(ta0 + 1 * 32 + 16, v1b);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[0 * 32 + i] = fmaf(sc0, __uint_as_float(v0a[i]), fin[0 * 32 + i]);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[0 * 32 + 16 + i] = fmaf(sc0, __uint_as_float(v0b[i]), fin[0 * 32 + 16 + i]);
          tmem_ld_wait();
          tmem_ld_x16(ta0 + 2 * 32 + 0, v0a);
          tmem_ld_x16(ta0 + 2 * 32 + 16, v0b);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[1 * 32 + i] = fmaf(sc0, __uint_as_float(v1a[i]), fin[1 * 32 + i]);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[1 * 32 + 16 + i] = fmaf(sc0, __uint_as_float(v1b[i]), fin[1 * 32 + 16 + i]);
          tmem_ld_wait();
          tmem_ld_x16(ta0 + 3 * 32 + 0, v1a);
          tmem_ld_x16(ta0 + 3 * 32 + 16, v1b);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[2 * 32 + i] = fmaf(sc0, __uint_as_float(v0a[i]), fin[2 * 32 + i]);
          #pragma unroll
          for (int i = 0; i < 16; ++i) fin[2 * 32 + 16 + i] = fmaf(sc0, __uint_as_float(v0b[i]), fin[2 * 32 + 16 + i]);
          tmem_ld_wait();
          if (has1) {
            const uint32_t ta1 = ta0 + BN;
            tmem_ld_x16(ta1 + 0 * 32 + 0, v0a);
            tmem_ld_x16(ta1 + 0 * 32 + 16, v0b);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + i] = fmaf(sc0, __uint_as_float(v1a[i]), fin[3 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + 16 + i] = fmaf(sc0, __uint_as_float(v1b[i]), fin[3 * 32 + 16 + i]);
            tmem_ld_wait();
            tmem_ld_x16(ta1 + 1 * 32 + 0, v1a);
            tmem_ld_x16(ta1 + 1 * 32 + 16, v1b);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[0 * 32 + i] = fmaf(sc1, __uint_as_float(v0a[i]), fin[0 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[0 * 32 + 16 + i] = fmaf(sc1, __uint_as_float(v0b[i]), fin[0 * 32 + 16 + i]);
            tmem_ld_wait();
            tmem_ld_x16(ta1 + 2 * 32 + 0, v0a);
            tmem_ld_x16(ta1 + 2 * 32 + 16, v0b);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[1 * 32 + i] = fmaf(sc1, __uint_as_float(v1a[i]), fin[1 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[1 * 32 + 16 + i] = fmaf(sc1, __uint_as_float(v1b[i]), fin[1 * 32 + 16 + i]);
            tmem_ld_wait();
            tmem_ld_x16(ta1 + 3 * 32 + 0, v1a);
            tmem_ld_x16(ta1 + 3 * 32 + 16, v1b);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[2 * 32 + i] = fmaf(sc1, __uint_as_float(v0a[i]), fin[2 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[2 * 32 + 16 + i] = fmaf(sc1, __uint_as_float(v0b[i]), fin[2 * 32 + 16 + i]);
            tmem_ld_wait();               // all tmem reads of this slot done
            bar_arrive(su32(pe_bar + pb));
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + i] = fmaf(sc1, __uint_as_float(v1a[i]), fin[3 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + 16 + i] = fmaf(sc1, __uint_as_float(v1b[i]), fin[3 * 32 + 16 + i]);
          } else {
            // odd-KB tail pair: only kb0 was drained; its batch 3 sits in v1*.
            bar_arrive(su32(pe_bar + pb));
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + i] = fmaf(sc0, __uint_as_float(v1a[i]), fin[3 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 16; ++i) fin[3 * 32 + 16 + i] = fmaf(sc0, __uint_as_float(v1b[i]), fin[3 * 32 + 16 + i]);
          }
        } else {
        {
          const float sc = sa_0 * sb_0;
          if (diag) t_sc = clock64();
          const uint32_t ta = tbase + (pb * 2) * BN + coloff;
          // d22: batch tcgen05.ld 2-deep per wait; X8: ECOLS=64 (NEPI=8)
          // covers the thread's cols in one batch, ECOLS=128 (NEPI=4,
          // certified layout) in two. Per-element FFMA order unchanged.
          uint32_t va[32], vb[32];
          tmem_ld_x32(ta + 0 * 32, va);
          if constexpr (ECOLS > 32) tmem_ld_x32(ta + 1 * 32, vb);
          tmem_ld_wait();
          #pragma unroll
          for (int i = 0; i < 32; ++i) fin[0 * 32 + i] = fmaf(sc, __uint_as_float(va[i]), fin[0 * 32 + i]);
          if constexpr (ECOLS > 32) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[1 * 32 + i] = fmaf(sc, __uint_as_float(vb[i]), fin[1 * 32 + i]);
          }
          if constexpr (ECOLS > 64) {
            tmem_ld_x32(ta + 2 * 32, va);
            tmem_ld_x32(ta + 3 * 32, vb);
            tmem_ld_wait();
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[2 * 32 + i] = fmaf(sc, __uint_as_float(va[i]), fin[2 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[3 * 32 + i] = fmaf(sc, __uint_as_float(vb[i]), fin[3 * 32 + i]);
          }
        }
        if (has1) {
          const float sc = sa_1 * sb_1;
          const uint32_t ta = tbase + (pb * 2 + 1) * BN + coloff;
          uint32_t va[32], vb[32];
          tmem_ld_x32(ta + 0 * 32, va);
          if constexpr (ECOLS > 32) tmem_ld_x32(ta + 1 * 32, vb);
          tmem_ld_wait();
          #pragma unroll
          for (int i = 0; i < 32; ++i) fin[0 * 32 + i] = fmaf(sc, __uint_as_float(va[i]), fin[0 * 32 + i]);
          if constexpr (ECOLS > 32) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[1 * 32 + i] = fmaf(sc, __uint_as_float(vb[i]), fin[1 * 32 + i]);
          }
          if constexpr (ECOLS > 64) {
            tmem_ld_x32(ta + 2 * 32, va);
            tmem_ld_x32(ta + 3 * 32, vb);
            tmem_ld_wait();
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[2 * 32 + i] = fmaf(sc, __uint_as_float(va[i]), fin[2 * 32 + i]);
            #pragma unroll
            for (int i = 0; i < 32; ++i) fin[3 * 32 + i] = fmaf(sc, __uint_as_float(vb[i]), fin[3 * 32 + i]);
          }
        }
        }
        // v_sload: refill this slot for pair j+2 (reads above are complete)
        {
          const int jn = j + 2;
          if (jn < PAIRS) {
            const int kbn0 = 2 * jn;
            cp_async4_pred(su32(mybuf + (jn & 1) * 256),
                           sap + (long)kbn0 * p.m, inb);
            cp_async4_pred(su32(mybuf + (jn & 1) * 256 + 128),
                           sap + (long)(kbn0 + 1) * p.m, inb && (kbn0 + 1 < KB));
          }
          cp_async_commit_g();
        }
        if constexpr (!V0_X16) bar_arrive(su32(pe_bar + pb));
        if (diag) {
          const unsigned long long t2 = clock64();
          acc_dw += e_t1 - e_t0; acc_dr += t2 - e_t1; ++n_ep;
          acc_sc += t_sc - e_t1;
          if (j < 2) { acc_few += e_t1 - e_t0; acc_fdr += t2 - e_t1; ++n_fpair; }
          g_v0_diag[16] = acc_dw; g_v0_diag[17] = acc_dr; g_v0_diag[18] = n_ep;
          g_v0_diag[21] = acc_few; g_v0_diag[22] = acc_fdr; g_v0_diag[23] = n_fpair;
          g_v0_diag[24] = acc_sc;
        }
      }

      // ---- tile store: TMA-store for FULL tiles (all 128 box rows owned by
      // this tile -> no clobber of neighboring groups), direct stores for
      // ragged tail tiles. Interleaved A/B (dv_41cc4b9be85d, 2 replicates
      // within 0.5%): -4.2% c0, -12.8% c6, -4.5% mt, -9.6% d0, -19.7%
      // fixed9-class, -17.3% d2-class, neutral at m=4 (path never taken).
      if (diag) e_t0 = clock64();
      const bool full_tile = (hi - row0) >= BM;
      if (full_tile) {
        if (issuer && prev_store) bulk_wait_group_read0();
        epi_bar_n<NEPI * 32>();
        #pragma unroll
        for (int cc = 0; cc < ECOLS / 8; ++cc) {
          uint4 pk;
          const int col = cc * 8;
          pk.x = pack_bf16(fin[col + 0], fin[col + 1]);
          pk.y = pack_bf16(fin[col + 2], fin[col + 3]);
          pk.z = pack_bf16(fin[col + 4], fin[col + 5]);
          pk.w = pack_bf16(fin[col + 6], fin[col + 7]);
          // canonical SWIZZLE_128B staging: chunk ci of row r at ((ci^(r&7))*16)
          // inside the 16KB box holding tile cols [64*box, 64*box+64).
          const int box = (coloff + col) >> 6, ci = cc & 7;
          *reinterpret_cast<uint4*>(staging + box * 16384 + row * 128
                                    + ((ci ^ (row & 7)) * 16)) = pk;
        }
        fence_proxy_async();
        epi_bar_n<NEPI * 32>();
        if (issuer) {
          tma_store_2d(&tOut, su32(staging), nt * 128, row0);
          tma_store_2d(&tOut, su32(staging + 16384), nt * 128 + 64, row0);
          bulk_commit_group();
          prev_store = true;
        }
      } else if (valid) {
        #pragma unroll
        for (int c = 0; c < ECOLS / 8; ++c) {
          uint4 pk;
          pk.x = pack_bf16(fin[c * 8 + 0], fin[c * 8 + 1]);
          pk.y = pack_bf16(fin[c * 8 + 2], fin[c * 8 + 3]);
          pk.z = pack_bf16(fin[c * 8 + 4], fin[c * 8 + 5]);
          pk.w = pack_bf16(fin[c * 8 + 6], fin[c * 8 + 7]);
          *reinterpret_cast<uint4*>(op + c * 8) = pk;
        }
      }
      if (diag) { e_t1 = clock64(); acc_st += e_t1 - e_t0; ++n_it;
                  g_v0_diag[19] = acc_st; g_v0_diag[20] = n_it; }
    }
    if (issuer && prev_store) bulk_wait_group0();   // stores complete before exit
  }
  }

  __syncthreads();
  if (warp == 1) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tbase), "r"(512));
  }
}


// ============================ wide-tile engine (w1f) =========================
// BN=256 per-kb rounds; 4 x 48KB stages (192KB in flight); single 32KB
// half-tile staging (halves stored sequentially); 320 threads (8-warp
// 2-group epilogue, fin[128], x16 drains); all-local barriers. Dispatched
// for n%256==0 && kb>=32 && items256>=4*nSM && mtiles>=4*active_groups
// (measured: c5-class -7.5%, x1-class -7.1%; elsewhere V0 engine wins).
namespace {
constexpr int BNW = 256;
constexpr int AW_BYTES = BM * BK;                   // 16384
constexpr int BW_BYTES = BNW * BK;                  // 32768
constexpr int STAGEW_BYTES = AW_BYTES + BW_BYTES;   // 49152
constexpr int PSTAGESW = 4;
constexpr int STAGW_BYTES = BM * 128 * 2;           // 32768 half-tile buffer
constexpr int THREADS_W = 320;
constexpr int EPIW_THREADS = 256;
constexpr unsigned IDESC_128x256 = ((128u >> 5) << 25) | ((256u >> 3) << 17) | (1u << 4);

struct ParamsW {
  const uint8_t* a;
  const uint8_t* b;
  const float* sa;
  const float* sb;
  const int* indptr;
  uint16_t* out;
  int m, n, k, G;
  int n2_tiles;    // n / 256
  int n_tiles;     // n / 128 (b-scale block stride)
  int kb_count;
  int abox;
  int bstream;     // X3-B: as Params::bstream
};

DI void epi_bar_w() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
}  // namespace

// Template parameters (V1.8):
//   GSTAGES  pipeline ring depth (48KB stages)
//   GSTAGB   TMA-store staging bytes
//   GSTORE   0 = half-sequential store (32KB staging, 2 passes, certified
//                w1f behavior); 1 = full-tile single-pass store (64KB staging,
//                ONE bulk drain-wait + 2 bar.syncs per item instead of 2+4).
// Instantiations: <4,32768,0> = engine 1 (bitwise/behavior-identical to the
// certified w1f); <3,65536,1> = engine 2 (w1g) for low-kb wide shapes where
// per-item store cost amortizes worst (fe1 diag: store ~2974 cyc/item).
template <int GSTAGES, int GSTAGB, int GSTORE>
__global__ void __launch_bounds__(THREADS_W, 1)
umma_wide(const ParamsW p, const __grid_constant__ CUtensorMap tA,
        const __grid_constant__ CUtensorMap tA64, const __grid_constant__ CUtensorMap tA32,
        const __grid_constant__ CUtensorMap tA16,
        const __grid_constant__ CUtensorMap tB, const __grid_constant__ CUtensorMap tOut) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // layout: GSTAGES x 48KB stages | GSTAGB staging | tail:
  //   full[GSTAGES] | empty[GSTAGES] | pe[2] | s_tmem | s_ind[G+1] | s_pref[G+1]
  uint8_t* staging = smem + GSTAGES * STAGEW_BYTES;
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(staging + GSTAGB);
  uint64_t* empty_bar = full_bar + GSTAGES;
  uint64_t* pe_bar = empty_bar + GSTAGES;
  uint32_t* s_tmem = reinterpret_cast<uint32_t*>(pe_bar + 2);
  int* s_ind = reinterpret_cast<int*>(s_tmem + 1);
  int* s_pref = s_ind + p.G + 1;

  for (int i = tid; i <= p.G; i += THREADS_W) s_ind[i] = p.indptr[i];
  __syncthreads();
  if (tid == 0) {
    int acc = 0;
    for (int e = 0; e < p.G; ++e) {
      s_pref[e] = acc;
      acc += (s_ind[e + 1] - s_ind[e] + BM - 1) / BM;    // m-tiles (128 rows)
    }
    s_pref[p.G] = acc;
    for (uint32_t i = 0; i < GSTAGES * 2 + 2; ++i)
      bar_init(su32(full_bar + i),
               (i >= GSTAGES * 2) ? (unsigned)EPIW_THREADS : 1u);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 9) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(su32(s_tmem)), "r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  const uint32_t tbase = *s_tmem;
  const int W = s_pref[p.G] * p.n2_tiles;    // items = m-tiles x 256-col tiles
  const int KB = p.kb_count;
  const int NB2 = p.n2_tiles;                // 256-col tiles per group
  const int NBS = p.n_tiles;                 // b-scale blocks per group (128-col)

  if (warp == 8 && lane == 0) {
    // ---------------- TMA producer (one round per K block) ----------------
    uint32_t stage = 0, phase = 0;
    int e_hint = 0;
    const uint64_t bpol = p.bstream ? make_evict_first_policy() : 0ull;
    for (int w = blockIdx.x; w < W; w += gridDim.x) {
      const int wi = (int)w;
      int e = e_hint;
      while (s_pref[e + 1] * NB2 <= wi) ++e;
      e_hint = e;
      const int r = wi - s_pref[e] * NB2;
      const int tiles_e = s_pref[e + 1] - s_pref[e];
      const int nt2 = r / tiles_e, mt = r % tiles_e;
      const int row0 = s_ind[e] + mt * BM;
      const int brow = e * p.n + nt2 * BNW;
      const int vrows = s_ind[e + 1] - row0;
      int boxh = 128;
      if (p.abox && vrows < 128)
        boxh = vrows > 64 ? 128 : (vrows > 32 ? 64 : (vrows > 16 ? 32 : 16));
      const CUtensorMap* mapA =
          boxh == 128 ? &tA : (boxh == 64 ? &tA64 : (boxh == 32 ? &tA32 : &tA16));
      const unsigned abytes = (unsigned)(boxh * BK);
      #pragma unroll 1
      for (int kb = 0; kb < KB; ++kb) {
        bar_spin(su32(empty_bar + stage), phase ^ 1);
        const uint32_t fb = su32(full_bar + stage);
        const uint8_t* st = smem + stage * STAGEW_BYTES;
        bar_expect_tx(fb, abytes + (unsigned)BW_BYTES);
        tma_load_2d(mapA, su32(st), fb, kb * BK, row0);
        if (p.bstream) tma_load_2d_ef(&tB, su32(st + AW_BYTES), fb, kb * BK, brow, bpol);
        else           tma_load_2d(&tB, su32(st + AW_BYTES), fb, kb * BK, brow);
        if (++stage == GSTAGES) { stage = 0; phase ^= 1; }
      }
    }
  } else if (warp == 9 && lane == 0) {
    // ---------------- UMMA issuer (M128 N256 K32 x 4 per round) ----------
    uint32_t stage = 0, phase = 0;
    unsigned long rr = 0;    // global round index (continues across items)
    for (int w = blockIdx.x; w < W; w += gridDim.x) {
      #pragma unroll 2
      for (int kb = 0; kb < KB; ++kb, ++rr) {
        bar_spin(su32(full_bar + stage), phase);
        const unsigned pb = rr & 1;              // tmem slot (2 x 256 cols)
        const unsigned long puse = rr >> 1;      // use # of pe_bar[pb]
        bar_spin(su32(pe_bar + pb), (unsigned)((puse - 1) & 1));
        tc_fence_after();
        const uint32_t sb0 = su32(smem + stage * STAGEW_BYTES);
        uint64_t ad = make_desc(sb0);
        uint64_t bd = make_desc(sb0 + AW_BYTES);
        const uint32_t dt = tbase + pb * BNW;
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
          umma_f8(dt, ad, bd, IDESC_128x256, step > 0 ? 1u : 0u);
          ad += 2; bd += 2;
        }
        umma_commit(su32(empty_bar + stage));
        if (++stage == GSTAGES) { stage = 0; phase ^= 1; }
      }
    }
  } else if (warp < 8) {
    // -------- epilogue: thread == (row, col-group); group owns 128 cols ----
    const int row = (warp & 3) * 32 + lane;
    const int grp = warp >> 2;                   // 0: cols 0-127, 1: 128-255
    const int coloff = grp * 128;                // within the 256-col tile
    const bool issuer = (warp == 0 && lane == 0);
    unsigned long rr = 0;
    int e_hint = 0;
    bool prev_store = false;
    for (int w = blockIdx.x; w < W; w += gridDim.x) {
      const int wi = (int)w;
      int e = e_hint;
      while (s_pref[e + 1] * NB2 <= wi) ++e;
      e_hint = e;
      const int r = wi - s_pref[e] * NB2;
      const int tiles_e = s_pref[e + 1] - s_pref[e];
      const int nt2 = r / tiles_e, mt = r % tiles_e;
      const int lo = s_ind[e], hi = s_ind[e + 1];
      const int row0 = lo + mt * BM;
      const bool valid = row < hi - row0;
      const float* sap = p.sa + row0 + row;
      // b-scale block for this thread's 128 cols: 2*nt2 + grp (stride NBS per kb)
      const float* sbp = p.sb + (long)e * KB * NBS + (2 * nt2 + grp);
      uint16_t* op = p.out + (long)(row0 + row) * p.n + nt2 * BNW + coloff;

      float fin[128];
      #pragma unroll
      for (int j = 0; j < 128; ++j) fin[j] = 0.f;

      // kb-ahead register prefetch (2-deep): sa[row, kb], sb[block, kb]
      const bool inb = (row0 + row) < p.m;
      float sa_c = inb ? __ldg(sap) : 0.f;
      float sa_n = (inb && 1 < KB) ? __ldg(sap + (long)1 * p.m) : 0.f;
      float sb_c = __ldg(sbp);
      float sb_n = (1 < KB) ? __ldg(sbp + (long)1 * NBS) : 0.f;
      #pragma unroll 1
      for (int kb = 0; kb < KB; ++kb, ++rr) {
        const float sc = sa_c * sb_c;
        sa_c = sa_n; sb_c = sb_n;
        const int kbn = kb + 2;
        sa_n = (inb && kbn < KB) ? __ldg(sap + (long)kbn * p.m) : 0.f;
        sb_n = (kbn < KB) ? __ldg(sbp + (long)kbn * NBS) : 0.f;

        const unsigned pb = rr & 1;
        bar_spin(su32(empty_bar + (rr % GSTAGES)), (unsigned)((rr / GSTAGES) & 1));
        tc_fence_after();
        const uint32_t ta = tbase + pb * BNW + coloff;
        // wd1: rotated drain. Next batch's x16 pair issues before the current
        // batch's FFMA so tmem-port reads overlap ALU (wait::ld is
        // all-outstanding -> issue,fma,wait order). Two buffer pairs (v0*,v1*)
        // alternate; fully unrolled so ptxas renames without copies. pe arrive
        // after the last load retires (wait at end of batch 2), before the
        // final FFMA batch -- same relative point as the serial code.
        uint32_t v0a[16], v0b[16], v1a[16], v1b[16];
        tmem_ld_x16(ta + 0 * 32 + 0, v0a);
        tmem_ld_x16(ta + 0 * 32 + 16, v0b);
        tmem_ld_wait();
        // batch 0 (v0 = cols 0..31); load batch 1 into v1
        tmem_ld_x16(ta + 1 * 32 + 0, v1a);
        tmem_ld_x16(ta + 1 * 32 + 16, v1b);
        #pragma unroll
        for (int i = 0; i < 16; ++i)
          fin[0 * 32 + i] = fmaf(sc, __uint_as_float(v0a[i]), fin[0 * 32 + i]);
        #pragma unroll
        for (int i = 0; i < 16; ++i)
          fin[0 * 32 + 16 + i] = fmaf(sc, __uint_as_float(v0b[i]), fin[0 * 32 + 16 + i]);
        tmem_ld_wait();
        // batch 1 (v1 = cols 32..63); load batch 2 into v0
        tmem_ld_x16(ta + 2 * 32 + 0, v0a);
        tmem_ld_x16(ta + 2 * 32 + 16, v0b);
        #pragma unroll
        for (int i = 0; i < 16; ++i)
          fin[1 * 32 + i] = fmaf(sc, __uint_as_float(v1a[i]), fin[1 * 32 + i]);
        #pragma unroll
        for (int i = 0; i < 16; ++i)
          fin[1 * 32 + 16 + i] = fmaf(sc, __uint_as_float(v1b[i]), fin[1 * 32 + 16 + i]);
        tmem_ld_wait();
        // batch 2 (v0 = cols 64..95); load batch 3 into v1
        tmem_ld_x16(ta + 3 * 32 + 0, v1a);
        tmem_ld_x16(ta + 3 * 32 + 16, v1b);
        #pragma unroll
        for (int i = 0; i < 16; ++i)
          fin[2 * 32 + i] = fmaf(sc, __uint_as_float(v0a[i]), fin[2 * 32 + i]);
        #pragma unroll
        for (int i = 0; i < 16; ++i)
          fin[2 * 32 + 16 + i] = fmaf(sc, __uint_as_float(v0b[i]), fin[2 * 32 + 16 + i]);
        tmem_ld_wait();
        bar_arrive(su32(pe_bar + pb));   // all tmem reads of this slot retired
        // batch 3 (v1 = cols 96..127)
        #pragma unroll
        for (int i = 0; i < 16; ++i)
          fin[3 * 32 + i] = fmaf(sc, __uint_as_float(v1a[i]), fin[3 * 32 + i]);
        #pragma unroll
        for (int i = 0; i < 16; ++i)
          fin[3 * 32 + 16 + i] = fmaf(sc, __uint_as_float(v1b[i]), fin[3 * 32 + 16 + i]);
      }

      // ---- tile store: 128x256 bf16 = 4 TMA boxes {64 cols, 128 rows} ----
      const bool full_tile = (hi - row0) >= BM;
      if (full_tile) {
        if constexpr (GSTORE == 0) {
          // half h = column half h (== group h): group-h threads pack the single
          // 32KB buffer, fence, bar, issuer stores 2 boxes {64u16,128rows}.
          #pragma unroll 1
          for (int h = 0; h < 2; ++h) {
            if (issuer && (prev_store || h == 1)) bulk_wait_group_read0();
            epi_bar_w();
            if (grp == h) {
              #pragma unroll
              for (int c = 0; c < 16; ++c) {
                uint4 pk;
                const int col = c * 8;             // within-group column
                pk.x = pack_bf16(fin[col + 0], fin[col + 1]);
                pk.y = pack_bf16(fin[col + 2], fin[col + 3]);
                pk.z = pack_bf16(fin[col + 4], fin[col + 5]);
                pk.w = pack_bf16(fin[col + 6], fin[col + 7]);
                // two 128B boxes per half: chunks 0-7 -> staging[0,16K),
                // chunks 8-15 -> staging[16K,32K); SWIZZLE_128B within each box.
                const int box = c >> 3, cc = c & 7;
                *reinterpret_cast<uint4*>(staging + box * 16384 + row * 128
                                          + ((cc ^ (row & 7)) * 16)) = pk;
              }
              fence_proxy_async();
            }
            epi_bar_w();
            if (issuer) {
              tma_store_2d(&tOut, su32(staging), nt2 * 256 + h * 128 + 0, row0);
              tma_store_2d(&tOut, su32(staging + 16384), nt2 * 256 + h * 128 + 64, row0);
              bulk_commit_group();
              prev_store = true;
            }
          }
        } else {
          // w1g: single-pass full-tile store. One drain-wait for the PREVIOUS
          // item's store, one pack of all 4 quadrants (each thread packs its
          // group's 128 cols = 2 quadrants), one fence/bar pair, 4 TMA boxes.
          if (issuer && prev_store) bulk_wait_group_read0();
          epi_bar_w();
          #pragma unroll
          for (int c = 0; c < 16; ++c) {
            uint4 pk;
            const int col = c * 8;               // within-group column
            pk.x = pack_bf16(fin[col + 0], fin[col + 1]);
            pk.y = pack_bf16(fin[col + 2], fin[col + 3]);
            pk.z = pack_bf16(fin[col + 4], fin[col + 5]);
            pk.w = pack_bf16(fin[col + 6], fin[col + 7]);
            // quadrant q = 2*grp + (c>>3): cols [64q, 64q+64); SWIZZLE_128B
            // within each 16KB quadrant box.
            const int q = (grp << 1) + (c >> 3), cc = c & 7;
            *reinterpret_cast<uint4*>(staging + q * 16384 + row * 128
                                      + ((cc ^ (row & 7)) * 16)) = pk;
          }
          fence_proxy_async();
          epi_bar_w();
          if (issuer) {
            tma_store_2d(&tOut, su32(staging), nt2 * 256 + 0, row0);
            tma_store_2d(&tOut, su32(staging + 16384), nt2 * 256 + 64, row0);
            tma_store_2d(&tOut, su32(staging + 32768), nt2 * 256 + 128, row0);
            tma_store_2d(&tOut, su32(staging + 49152), nt2 * 256 + 192, row0);
            bulk_commit_group();
            prev_store = true;
          }
        }
      } else if (valid) {
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
          uint4 pk;
          pk.x = pack_bf16(fin[c * 8 + 0], fin[c * 8 + 1]);
          pk.y = pack_bf16(fin[c * 8 + 2], fin[c * 8 + 3]);
          pk.z = pack_bf16(fin[c * 8 + 4], fin[c * 8 + 5]);
          pk.w = pack_bf16(fin[c * 8 + 6], fin[c * 8 + 7]);
          *reinterpret_cast<uint4*>(op + c * 8) = pk;
        }
      }
    }
    if (issuer && prev_store) bulk_wait_group0();
  }

  __syncthreads();
  if (warp == 9) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tbase), "r"(512));
  }
}


// ---------------------------------------------------------------- host side ---
// Two engines in one module:
//   engine 0 (V0/d22g3): pair-batched 128x128 UMMA, template<SAREG> paths,
//            single-node CUDA-graph launch cache, identity-cached sareg gate.
//   engine 1 (w1f wide): BN=256 per-kb rounds, 4x48KB ring, half-tile staging,
//            graph-cached the same way (cluster-free).
// Dispatch (sched_policy, one indptr D2H per tensor identity — V0 mechanism):
//   wide iff n%256==0 && kb_count>=32 && items256 >= 4*nSM && mtiles >= 4*active_groups
// Both engines compute per-element identical math (per-128K fp32 promotion);
// outputs bitwise-equal across engines (A/B-verified pre-merge).
namespace {
struct MapSlot {
  const void* ptr = nullptr;
  long rows = -1, cols = -1;
  CUtensorMap map;
};
MapSlot g_cacheA[4], g_cacheB[4], g_cacheOut[4];
MapSlot g_cacheA64[2], g_cacheA32[2], g_cacheA16[2];
MapSlot g_cacheB256[2];
int g_nextA = 0, g_nextB = 0, g_nextOut = 0;
int g_nextA64 = 0, g_nextA32 = 0, g_nextA16 = 0;
int g_nextB256 = 0;

int encode_2d_h(CUtensorMap* out, const void* gptr, long rows, long cols, long boxh) {
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols};
  cuuint32_t box[2] = {128, (cuuint32_t)boxh};
  cuuint32_t estr[2] = {1, 1};
  // 256B promotion confirmed best: X9 p128 (L2_128B) regressed c0 +4.3%,
  // c4 +4.4%, d9 +3.6%, d2 +2.0% (x9_harness dv_1ef38321a03f, bitwise-clean).
  const CUtensorMapL2promotion promo = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  CUresult r = cuTensorMapEncodeTiled(out, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
      const_cast<void*>(gptr), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, promo,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int)r;
}

int encode_2d(CUtensorMap* out, const void* gptr, long rows, long cols) {
  return encode_2d_h(out, gptr, rows, cols, 128);
}

// out map for TMA stores: UINT16 (m, n) row-major, box {64 elems = 128B,
// 128 rows} with SWIZZLE_128B matching the staging write pattern.
int encode_2d_u16(CUtensorMap* out, const void* gptr, long rows, long cols) {
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols * 2};
  cuuint32_t box[2] = {64, 128};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(out, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2,
      const_cast<void*>(gptr), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int)r;
}

const CUtensorMap* cached_map(MapSlot* cache, int& next, const void* ptr, long rows, long cols) {
  for (int i = 0; i < 4; ++i)
    if (cache[i].ptr == ptr && cache[i].rows == rows && cache[i].cols == cols)
      return &cache[i].map;
  MapSlot& s = cache[next];
  next = (next + 1) & 3;
  if (encode_2d(&s.map, ptr, rows, cols) != 0) return nullptr;
  s.ptr = ptr; s.rows = rows; s.cols = cols;
  return &s.map;
}

const CUtensorMap* cached_map_h(MapSlot* cache, int& next, const void* ptr, long rows,
                                long cols, long boxh) {
  for (int i = 0; i < 2; ++i)
    if (cache[i].ptr == ptr && cache[i].rows == rows && cache[i].cols == cols)
      return &cache[i].map;
  MapSlot& s = cache[next];
  next = (next + 1) & 1;
  if (encode_2d_h(&s.map, ptr, rows, cols, boxh) != 0) return nullptr;
  s.ptr = ptr; s.rows = rows; s.cols = cols;
  return &s.map;
}

const CUtensorMap* cached_map_out(MapSlot* cache, int& next, const void* ptr, long rows, long cols) {
  for (int i = 0; i < 4; ++i)
    if (cache[i].ptr == ptr && cache[i].rows == rows && cache[i].cols == cols)
      return &cache[i].map;
  MapSlot& s = cache[next];
  next = (next + 1) & 3;
  if (encode_2d_u16(&s.map, ptr, rows, cols) != 0) return nullptr;
  s.ptr = ptr; s.rows = rows; s.cols = cols;
  return &s.map;
}

// Single-node CUDA-graph launch caches (params baked; permanent direct
// fallback on API failure). One slot per engine.
struct GraphSlot {
  bool valid = false;
  const void* a = nullptr; const void* b = nullptr;
  const void* sa = nullptr; const void* sb = nullptr;
  const void* indptr = nullptr; const void* out = nullptr;
  int m = -1, n = -1, k = -1, G = -1, grid = -1, abox = -1, dual = -1;
  int sareg = -1;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  Params params;        // stable storage referenced by node params
  CUtensorMap tA, tA64, tA32, tA16, tB, tOut;   // by-value kernel args
};
GraphSlot g_slot;
bool g_graph_off = false;

struct GraphSlotW {
  bool valid = false;
  const void* a = nullptr; const void* b = nullptr;
  const void* sa = nullptr; const void* sb = nullptr;
  const void* indptr = nullptr; const void* out = nullptr;
  int m = -1, n = -1, k = -1, G = -1, grid = -1, abox = -1;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ParamsW params;
  CUtensorMap tA, tA64, tA32, tA16, tB, tOut;
};
GraphSlotW g_slotw;
bool g_graphw_off = false;

// w1g (engine 2) launch cache; identical slot mechanics, cluster-free.
GraphSlotW g_slot2;
bool g_graphw2_off = false;
bool g_w1g_off = false;      // permanent engine-2 -> engine-1 fallback
}  // namespace

// Identity-cached policy: (sareg, engine) from ONE indptr D2H per
// (indptr,G,m,n,k) identity — hit path is a pointer compare (V0 mechanism;
// gate key hardened in d22g3). Engine thresholds are shape-derived geometry
// (ab12 measurements): wide engine wins only for L2-resident big-K shapes
// with many waves and high m-tiles per active group.
namespace {
struct SchedCache {
  const int* indptr = nullptr; int G = -1;
  int m = -1, n = -1, k = -1; int sareg = 0; int engine = 0; int pp = 0;
};
SchedCache g_sched;

void sched_policy(const int* indptr, int G, int m, int n, int k, int kb_count,
                  int nSM, cudaStream_t stream, int* sareg_out, int* engine_out,
                  int* pp_out) {
  if (g_sched.indptr == indptr && g_sched.G == G && g_sched.m == m
      && g_sched.n == n && g_sched.k == k) {
    *sareg_out = g_sched.sareg;
    *engine_out = g_sched.engine;
    *pp_out = g_sched.pp;
    return;
  }
  int sareg = 0, engine = 0, pp = 0;
  int h_ind[120];
  if (G <= 116 &&
      cudaMemcpyAsync(h_ind, indptr, (size_t)(G + 1) * sizeof(int),
                      cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
      cudaStreamSynchronize(stream) == cudaSuccess) {
    int ne = 0; long mt = 0; bool ok = h_ind[0] >= 0;
    int uni = -1; bool uniform = true;
    long mtiles = 0;
    for (int e = 0; e < G && ok; ++e) {
      const int lo = h_ind[e], hi = h_ind[e + 1];
      ok = (hi >= lo) && (hi <= m);
      const int sz = hi - lo;
      if (sz > 0) {
        ++ne;
        if (uni < 0) uni = sz;
        else if (sz != uni) uniform = false;
      }
      const long mte = ((long)sz + BM - 1) / BM;
      mtiles += mte;
      mt += mte;
    }
    if (ok) {
      // V2.0 (X33 adoption, dv_06591945b5a3): GLOBAL sareg=0 (cp.async
      // 2-slot scale ring). Post-join sub-gate audit: sareg0 beats or ties
      // sareg1 on every V0 shape tested — c6 -7.7% (3x96-sample passes,
      // plainB-controlled), d7 -7.2% (x3), c0/d6 1.000, c1 ambiguous
      // -1.8% mode-mixed, d5 min-equal; zero measured losses. The archived
      // pre-join g1/g2 classes (ne==1&&mt>=4&&pairs>=16; uniform<=512&&
      // kb>=48 -> 0 else 1) predate the X10 epilogue; register prefetch
      // (sareg1) no longer pays anywhere. Bitwise-identical (9 cases + 3
      // canaries across x32/x33). dual gate UNCHANGED (combo added nothing
      // over sareg0 alone; c0 dual0 +3.4% revalidates it).
      sareg = 0;
      (void)mt; (void)uni; (void)uniform;
      // wide engine (w1f + wg2): measured-win classes only
      // (results/wg1-findings.md): original >=4-wave class, PLUS the
      // low-wave class [2.5,4) waves when kb<=48 (k<=6144: per-item wide
      // B-tile <=1.5MB; at k=7168 low-wave wide loses +7% - c0 measured).
      if ((n % 256) == 0 && ne > 0 && nSM > 0) {
        const long items256 = mtiles * (long)(n / 256);
        engine = ((kb_count >= 32
                   && ((items256 >= 4L * nSM && mtiles >= 4L * ne)
                       || (kb_count <= 48 && items256 * 2L >= 5L * nSM
                           && mtiles >= ne)))
                  // V1.4 (wk16): high-wave small-k class (c4/c9/d2-like:
                  // kb>=16, >=8 waves256, >=2 mtiles per active group).
                  || (kb_count >= 16 && items256 >= 8L * nSM
                      && mtiles >= 2L * ne)
                  // V1.5 (wk12): high-wave tiny-k class (c8-like: kb 8..12,
                  // >=10 waves256, >=1 mtile/active group). Flips exactly
                  // c8 in fixed suite; no domain case (kb<=12 cap keeps
                  // d9 kb=16 out; c3/d4 items256 768 < 1480).
                  || (kb_count >= 8 && kb_count <= 12
                      && items256 >= 10L * nSM
                      && mtiles >= ne)
                  // V1.6 (wd9): high-wave single-mtile-group class (wk16's
                  // mtiles>=2*ne relaxed to ==ne). Flips exactly d9 domain;
                  // c7 excluded at items256 896 < 8*nSM (its own probe
                  // measured null: -2.01/-0.87/-0.02%). Fixed dispatch
                  // identical to V1.5.
                  || (kb_count >= 16 && items256 >= 8L * nSM
                      && mtiles == ne)
                  // V1.8 (C_new): many-wave low-kb high-reuse class. Extends
                  // wk16-style gating down to items256>=4*nSM when per-group
                  // m-tile reuse is >=2 (wide B tile amortized over >=2 m
                  // tiles; per-item B <= 640KB at kb<=20). x1_harness forced
                  // A/B: d4 51.2->49.2 (-3.9%), c3 61.4->60.5 (-1.5%); c7/d9
                  // (mtiles==ne streaming) and c1/d0/d7 (items256<4*nSM) keep
                  // their certified engines.
                  || (kb_count >= 8 && kb_count <= 20
                      && items256 >= 4L * nSM && mtiles >= 2L * ne)) ? 1 : 0;
        // V1.8 (w1g): single-pass full-tile store variant of the wide engine
        // for LOW-KB, B-REUSING shapes, where the certified half-sequential
        // store's second bulk drain-wait amortizes worst (fe1 diag
        // dv_5fb08c79faa6: store ~2974 cyc/item = ~250 cyc/round at kb=12).
        // Trades one TMA ring stage (4->3; full_wait measured only 86 cyc at
        // depth 4) for 64KB staging. Bitwise-identical numerics (store-only).
        // x1_harness (dv_ job 20261026T144659Z): c9 -3.3%, d2 -3.9%, d4
        // -3.9%, c4 -2.8%, c8 -1.8%, c3 -1.6%; streaming class (mtiles~ne:
        // c7 +1.9%, d9 +2.3%) REGRESSES on the shallower ring -> excluded
        // via 2*mtiles >= 3*ne; kb>20 excluded (c2 1.000, c5 1.010).
        if (engine == 1 && kb_count <= 20 && 2L * mtiles >= 3L * ne)
          engine = 2;
      }
#ifdef FP8GG_FORCE_ENGINE
      engine = FP8GG_FORCE_ENGINE;   // diagnostic builds only (harness A/B)
#endif
#ifdef FP8GG_FORCE_SAREG
      sareg = FP8GG_FORCE_SAREG;     // diagnostic builds only (V0 gate A/B)
#endif
      // X3-B (stream flag via the retained scheduler slot): B tiles are read
      // EXACTLY ONCE iff every non-empty group spans one m-tile (mtiles==ne).
      // x3_harness (dv_d045d9cca98f): evict_first on B then wins big (c1
      // -9.7%, c7 -8.5%, d9 -2.4%) while any reuse regresses (c4 +4.4%,
      // c8 +1.0%) -> gate strictly on mtiles == ne. Exact-grid (X3-C) was
      // measured neutral-to-negative (d5 1.002, warm -2.5%) -> not adopted.
      pp = (mtiles == (long)ne) ? 1 : 0;
    }
  }
  (void)cudaGetLastError();
  g_sched.indptr = indptr; g_sched.G = G;
  g_sched.m = m; g_sched.n = n; g_sched.k = k;
  g_sched.sareg = sareg; g_sched.engine = engine; g_sched.pp = pp;
  *sareg_out = sareg;
  *engine_out = engine;
  *pp_out = pp;
}
}  // namespace

extern "C" int fp8umma_smem_max() {
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  return v;
}

// `stages` is retained for ABI stability; both engines use fixed ring depths.
extern "C" int fp8umma_run(const void* a, const void* b, const void* sa, const void* sb,
                           const int* indptr, void* out, int m, int n, int k, int G, int nSM,
                           int stages, cudaStream_t stream) {
  (void)stages;
  if (m <= 0 || n <= 0 || k <= 0 || G <= 0) return 0;
  if ((n % BN) != 0 || (k % BK) != 0) return 1001;
  if (G > 116) return 1004;  // schedule smem tail bound (TAIL 1024B); contract has G <= 64

  const int kb_count = k / BK;
  int sareg = 0, engine = 0;
  int pp = 0;
  sched_policy(indptr, G, m, n, k, kb_count, nSM, stream, &sareg, &engine, &pp);

  const CUtensorMap* tA = cached_map(g_cacheA, g_nextA, a, m, k);
  const CUtensorMap* tOut = cached_map_out(g_cacheOut, g_nextOut, out, m, n);
  if (!tA || !tOut) return 1002;
  const CUtensorMap* tA64 = cached_map_h(g_cacheA64, g_nextA64, a, m, k, 64);
  const CUtensorMap* tA32 = cached_map_h(g_cacheA32, g_nextA32, a, m, k, 32);
  const CUtensorMap* tA16 = cached_map_h(g_cacheA16, g_nextA16, a, m, k, 16);
  int abox = 1;
  if (!tA64 || !tA32 || !tA16) { abox = 0; tA64 = tA32 = tA16 = tA; }

  static int cfg = 0;
  if (!cfg) {
    const int mx = fp8umma_smem_max();
    if (PSTAGES * PSTAGE_BYTES + STAGING_BYTES + 2048 + TAIL_BYTES > mx) return 1003;
    if (PSTAGESW * STAGEW_BYTES + STAGW_BYTES + TAIL_BYTES > mx) return 1003;
    if (3 * STAGEW_BYTES + BM * 256 * 2 + TAIL_BYTES > mx) return 1003;
    cudaFuncSetAttribute(umma_gemm<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         PSTAGES * PSTAGE_BYTES + STAGING_BYTES + 2048 + TAIL_BYTES);
    cudaFuncSetAttribute(umma_gemm<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         PSTAGES * PSTAGE_BYTES + STAGING_BYTES + 2048 + TAIL_BYTES);
    cudaFuncSetAttribute((const void*)umma_wide<PSTAGESW, STAGW_BYTES, 0>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         PSTAGESW * STAGEW_BYTES + STAGW_BYTES + TAIL_BYTES);
    cudaFuncSetAttribute((const void*)umma_wide<3, BM * 256 * 2, 1>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         3 * STAGEW_BYTES + BM * 256 * 2 + TAIL_BYTES);
    cfg = 1;
  }

  if (engine == 2 && g_w1g_off) engine = 1;   // permanent w1g fallback
  if (engine == 2) {
    // ---------------- w1g engine (wide, 3 stages, single-pass store) -------
    const CUtensorMap* tB = cached_map_h(g_cacheB256, g_nextB256, b, (long)G * n, k, BNW);
    if (!tB) return 1002;
    ParamsW p{(const uint8_t*)a, (const uint8_t*)b, (const float*)sa, (const float*)sb,
              indptr, (uint16_t*)out, m, n, k, G, n / BNW, n / BN, kb_count, abox, pp};
    const long wub = ((long)(m + BM - 1) / BM + G) * (long)p.n2_tiles;
    int grid = (int)(wub < (long)nSM ? wub : (long)nSM);
    if (grid < 1) grid = 1;
    const size_t smem = 3 * STAGEW_BYTES + BM * 256 * 2 + TAIL_BYTES;
    if (!g_graphw2_off) {
      GraphSlotW& s = g_slot2;
      if (s.valid && s.a == a && s.b == b && s.sa == sa && s.sb == sb
          && s.indptr == indptr && s.out == out && s.m == m && s.n == n
          && s.k == k && s.G == G && s.grid == grid && s.abox == abox) {
        if (cudaGraphLaunch(s.exec, stream) == cudaSuccess) return 0;
        g_graphw2_off = true;
      } else {
        if (s.exec) { cudaGraphExecDestroy(s.exec); s.exec = nullptr; }
        if (s.graph) { cudaGraphDestroy(s.graph); s.graph = nullptr; }
        s.valid = false;
        s.params = p; s.tA = *tA; s.tA64 = *tA64; s.tA32 = *tA32; s.tA16 = *tA16;
        s.tB = *tB; s.tOut = *tOut;
        s.a = a; s.b = b; s.sa = sa; s.sb = sb;
        s.indptr = indptr; s.out = out;
        s.m = m; s.n = n; s.k = k; s.G = G; s.grid = grid; s.abox = abox;
        cudaGraph_t gr = nullptr;
        cudaKernelNodeParams np{};
        void* kargs[7] = { &s.params, &s.tA, &s.tA64, &s.tA32, &s.tA16, &s.tB, &s.tOut };
        np.func = (void*)umma_wide<3, BM * 256 * 2, 1>;
        np.gridDim = dim3((unsigned)grid, 1u, 1u);
        np.blockDim = dim3((unsigned)THREADS_W, 1u, 1u);
        np.sharedMemBytes = (unsigned)smem;
        np.kernelParams = kargs;
        np.extra = nullptr;
        cudaGraphNode_t node = nullptr;
        cudaGraphExec_t ex = nullptr;
        if (cudaGraphCreate(&gr, 0) == cudaSuccess
            && cudaGraphAddKernelNode(&node, gr, nullptr, 0, &np) == cudaSuccess
            && cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0) == cudaSuccess) {
          s.graph = gr; s.exec = ex; s.valid = true;
          if (cudaGraphLaunch(ex, stream) == cudaSuccess) return 0;
        } else if (gr) {
          cudaGraphDestroy(gr);
        }
        g_graphw2_off = true;
        (void)cudaGetLastError();
      }
    }
    umma_wide<3, BM * 256 * 2, 1><<<grid, THREADS_W, smem, stream>>>(
        p, *tA, *tA64, *tA32, *tA16, *tB, *tOut);
    if (cudaGetLastError() == cudaSuccess) return 0;
    g_w1g_off = true;
    (void)cudaGetLastError();
    engine = 1;
  }

  if (engine == 1) {
    // ---------------- wide engine (w1f) ----------------
    const CUtensorMap* tB = cached_map_h(g_cacheB256, g_nextB256, b, (long)G * n, k, BNW);
    if (!tB) return 1002;
    ParamsW p{(const uint8_t*)a, (const uint8_t*)b, (const float*)sa, (const float*)sb,
              indptr, (uint16_t*)out, m, n, k, G, n / BNW, n / BN, kb_count, abox, pp};
    const long wub = ((long)(m + BM - 1) / BM + G) * (long)p.n2_tiles;
    int grid = (int)(wub < (long)nSM ? wub : (long)nSM);
    if (grid < 1) grid = 1;
    const size_t smem = PSTAGESW * STAGEW_BYTES + STAGW_BYTES + TAIL_BYTES;
    if (!g_graphw_off) {
      GraphSlotW& s = g_slotw;
      if (s.valid && s.a == a && s.b == b && s.sa == sa && s.sb == sb
          && s.indptr == indptr && s.out == out && s.m == m && s.n == n
          && s.k == k && s.G == G && s.grid == grid && s.abox == abox) {
        if (cudaGraphLaunch(s.exec, stream) == cudaSuccess) return 0;
        g_graphw_off = true;
      } else {
        if (s.exec) { cudaGraphExecDestroy(s.exec); s.exec = nullptr; }
        if (s.graph) { cudaGraphDestroy(s.graph); s.graph = nullptr; }
        s.valid = false;
        s.params = p; s.tA = *tA; s.tA64 = *tA64; s.tA32 = *tA32; s.tA16 = *tA16;
        s.tB = *tB; s.tOut = *tOut;
        s.a = a; s.b = b; s.sa = sa; s.sb = sb;
        s.indptr = indptr; s.out = out;
        s.m = m; s.n = n; s.k = k; s.G = G; s.grid = grid; s.abox = abox;
        cudaGraph_t gr = nullptr;
        cudaKernelNodeParams np{};
        void* kargs[7] = { &s.params, &s.tA, &s.tA64, &s.tA32, &s.tA16, &s.tB, &s.tOut };
        np.func = (void*)umma_wide<PSTAGESW, STAGW_BYTES, 0>;
        np.gridDim = dim3((unsigned)grid, 1u, 1u);
        np.blockDim = dim3((unsigned)THREADS_W, 1u, 1u);
        np.sharedMemBytes = (unsigned)smem;
        np.kernelParams = kargs;
        np.extra = nullptr;
        cudaGraphNode_t node = nullptr;
        cudaGraphExec_t ex = nullptr;
        if (cudaGraphCreate(&gr, 0) == cudaSuccess
            && cudaGraphAddKernelNode(&node, gr, nullptr, 0, &np) == cudaSuccess
            && cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0) == cudaSuccess) {
          s.graph = gr; s.exec = ex; s.valid = true;
          if (cudaGraphLaunch(ex, stream) == cudaSuccess) return 0;
        } else if (gr) {
          cudaGraphDestroy(gr);
        }
        g_graphw_off = true;
        (void)cudaGetLastError();
      }
    }
    umma_wide<PSTAGESW, STAGW_BYTES, 0><<<grid, THREADS_W, smem, stream>>>(
        p, *tA, *tA64, *tA32, *tA16, *tB, *tOut);
    return (int)cudaGetLastError();
  }

  // ---------------- V0 engine ----------------
  const CUtensorMap* tB = cached_map(g_cacheB, g_nextB, b, (long)G * n, k);
  if (!tB) return 1002;
  Params p{(const uint8_t*)a, (const uint8_t*)b, (const float*)sa, (const float*)sb,
           indptr, (uint16_t*)out, m, n, k, G, n / BN, kb_count, abox, 0, sareg, pp};

  const long wub = ((long)(m + BM - 1) / BM + G) * (long)p.n_tiles;
  int grid = (int)(wub < (long)nSM ? wub : (long)nSM);
  if (grid < 1) grid = 1;
  {
    const long pairs_ub = wub * (((long)kb_count + 1) >> 1);
    p.dual = (pairs_ub >= 48L * grid) ? 1 : 0;
#ifdef FP8GG_FORCE_DUAL
    p.dual = FP8GG_FORCE_DUAL;       // diagnostic builds only (V0 gate A/B)
#endif
  }

  const size_t smem = PSTAGES * PSTAGE_BYTES + STAGING_BYTES + 2048 + TAIL_BYTES;

  if (!g_graph_off) {
    GraphSlot& s = g_slot;
    if (s.valid && s.a == a && s.b == b && s.sa == sa && s.sb == sb
        && s.indptr == indptr && s.out == out && s.m == m && s.n == n
        && s.k == k && s.G == G && s.grid == grid && s.abox == abox
        && s.dual == p.dual && s.sareg == sareg) {
      if (cudaGraphLaunch(s.exec, stream) == cudaSuccess) return 0;
      g_graph_off = true;  // fall through to direct launch
    } else {
      if (s.exec) { cudaGraphExecDestroy(s.exec); s.exec = nullptr; }
      if (s.graph) { cudaGraphDestroy(s.graph); s.graph = nullptr; }
      s.valid = false;
      s.params = p; s.tA = *tA; s.tA64 = *tA64; s.tA32 = *tA32; s.tA16 = *tA16;
      s.tB = *tB; s.tOut = *tOut;
      s.a = a; s.b = b; s.sa = sa; s.sb = sb;
      s.indptr = indptr; s.out = out;
      s.m = m; s.n = n; s.k = k; s.G = G; s.grid = grid; s.abox = abox;
      s.dual = p.dual;
      s.sareg = sareg;
      cudaGraph_t gr = nullptr;
      cudaKernelNodeParams np{};
      void* kargs[7] = { &s.params, &s.tA, &s.tA64, &s.tA32, &s.tA16, &s.tB, &s.tOut };
      np.func = (void*)(sareg ? umma_gemm<1> : umma_gemm<0>);
      np.gridDim = dim3((unsigned)grid, 1u, 1u);
      np.blockDim = dim3((unsigned)THREADS_V, 1u, 1u);
      np.sharedMemBytes = (unsigned)smem;
      np.kernelParams = kargs;
      np.extra = nullptr;
      cudaGraphNode_t node = nullptr;
      cudaGraphExec_t ex = nullptr;
      if (cudaGraphCreate(&gr, 0) == cudaSuccess
          && cudaGraphAddKernelNode(&node, gr, nullptr, 0, &np) == cudaSuccess
          && cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0) == cudaSuccess) {
        s.graph = gr; s.exec = ex; s.valid = true;
        if (cudaGraphLaunch(ex, stream) == cudaSuccess) return 0;
      } else if (gr) {
        cudaGraphDestroy(gr);
      }
      g_graph_off = true;  // any graph-API failure: permanent direct fallback
      (void)cudaGetLastError();  // clear latched error state
    }
  }

  if (sareg)
    umma_gemm<1><<<grid, THREADS_V, smem, stream>>>(p, *tA, *tA64, *tA32, *tA16, *tB, *tOut);
  else
    umma_gemm<0><<<grid, THREADS_V, smem, stream>>>(p, *tA, *tA64, *tA32, *tA16, *tB, *tOut);
  return (int)cudaGetLastError();
}
