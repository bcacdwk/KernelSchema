// fmha_tc.cuh — tcgen05 (SM103a) FP8 paged FMHA mainloop for Dqk=Dv=128.
// Independent implementation for problem 14. Mechanism patterns (UMMA smem
// descriptors, idesc encoding, tmem P-operand PV with TransposeB, mbarrier
// choreography, warp roles) follow the public PTX ISA and were validated on
// this L20D by experiments/tc_probe.cu; the warp-specialized topology mirrors
// conventions visible in flashinfer/atrex-fa SM100 sources (read for
// mechanism; no code copied, no library called).
//
// Topology: 384 threads / 12 warps, 1 CTA/SM (tmem-bound).
//   w0-3   softmax group A (EVEN local tiles; row per lane; S/P in tmem)
//   w4-7   softmax group B (ODD local tiles). A tile's owner also runs its
//          O-rescale (merged correction; ballot identity-skip), and the LAST
//          tile's group runs the final epilogue (norm+sinks+bf16/lse or
//          split-KV partials) from register state; setmaxnreg 208 both.
//   w8     mma issue  (lane 0 issues tcgen05.mma + commits; also tmem alloc)
//   w9-11  loaders    (paged cp.async -> swizzle-128B smem; mbarrier arrive)
// Tiles: M=128 q-rows (row = token*ratio + head-in-group, as in v2), N=128 kv.
// tmem columns: S0 [0,128) (P0 packed e4m3 at [32,64)), S1 [128,256)
// (P1 at [160,192)), O [256,384). Allocation: 512 columns. NOTE: an fp16
// QK accumulator (idesc c_format=F16) was attempted and REJECTED: it yields
// NaN/garbage S on SM103 kind::f8f6f4 (parity dv_b6obouplm job; NVIDIA's own
// SM100 fmha references also fix qk_acc_dtype=Float32).
// P quantization (v6a anchored-max f16 pipeline): p_hat = ex2.approx.f16x2(
// f16(s2 - anchor + PLOG2)) packs straight to e4m3 via cvt.rn.satfinite.
// e4m3x2.f16x2. The anchor (m2r) is the running max quantized to REBASE
// events: a tile rebases the anchor to its own max only when the max exceeds
// the anchor by REBASE_MARGIN (log2 units after qk_scale), which keeps every
// p_hat <= 2^(PLOG2+margin+round) ~= 266 < 448 (e4m3 max, satfinite can
// never clamp) while making the O/dsum rescale RARE: natural per-tile max
// increments fall below the margin quickly, so PV UMMAs accumulate back-to-
// back in tensor issue order without a tmem ld->scale->st correction
// roundtrip between them (v4 corrected ~100% of tiles: tctiming
// dv_afb87f84da30 showed the corr chain + softmax latency pacing every heavy
// case at 2.4-3.4x the umma_bench tile floor; v5's 0.6 margin still fired
// the 32-row ballot most tiles -> only 1.02-1.11x, dv_15e3bca227d7).
// LSE/out identities are anchor-invariant: base2 = anchor + log2(dsum) -
// PLOG2 == log2 sum 2^s2 exactly as in v4; dsum tracks sum(p_hat) ==
// PSCALE*sum(p). Split partials keep the stored-d xPSCALE convention that
// fmha_combine's lse_adj=-PLOG2 expects.

#pragma once

#include <cuda_bf16.h>
#include <cstdint>

// EXPERIMENT-ONLY component ablations (experiments/ablate.py): each ID strips
// ONE mechanism from the real kernel to measure its marginal per-tile cost.
// Numerics are garbage under ablation; production never defines FMHA_ABLATE.
// 1: no exp/pack pipeline  2: no mask+max pass  3: no S tmem reads
// 4: no P tmem store       5: no correction body 6: no PV umma
// 7: no QK umma            9: no K/V data loads (barrier protocol intact)
// 10: no max pass AND no exp/pack (S reads + protocol intact)
// 11: full softmax skeleton: no S read/max/exp/P store/corr body - pure
//     barrier choreography + tensor + loads (protocol-cost floor)
// 12: mma skips softmax-handoff waits (prob_ready/out_rescaled)
// 13: mma skips loader waits (k/v_loaded, k/v_tx)
// 14: mma free-run: skips ALL waits (12+13+score_released) -> in-kernel
//     tensor+issue+commit floor. 12-14 produce garbage; timing only.
#ifndef FMHA_ABLATE
#define FMHA_ABLATE 0
#endif

namespace fmtc {

constexpr int TM = 128;
constexpr int TN = 128;
constexpr int NTHREADS = 384;
// v6a: PSCALE 2^5 (was 2^8). p_hat = 2^(s2 - anchor + 5) keeps max p_hat at
// 2^5 nominal; the larger rebase margin below can then allow p_hat up to
// 2^(5+3+0.07) = 266 < 448 without e4m3 clamping. e4m3 is floating, so the
// smaller nominal scale costs no relative precision; the underflow edge moves
// from 17 to 14 log2 units below the anchor (contribution <= 2^-14 there).
constexpr float PSCALE = 32.0f;
constexpr float PLOG2 = 5.0f;   // log2(PSCALE); epilogue/lse_adj convention
// v5 anchored-max softmax rebase threshold (log2 units after qk_scale).
// v6a: 0.6 -> 3.0. The correction ballot spans 32 independent rows, so the
// per-WARP rebase probability is ~1-(1-p_row)^32: margin 0.6 kept p_row high
// (natural max increments are ~0.1-1 log2 units) and v5 still corrected most
// tiles (measured only 1.02-1.11x). Margin 3.0 makes per-row rebases rare
// after the first few tiles of a row.
constexpr float REBASE_MARGIN = 3.0f;
constexpr uint32_t O_TMEM = 256;
constexpr uint32_t TMEM_COLS = 512;
// S-load wait shape, COMPILE-TIME (was the runtime field p.onewait_s). 0 = C3b's
// two waits: seg0 lands and is waited, segs1-3 fly while seg0's max runs, and
// score_released fires between the halves. 1 = one wait for all four segments with
// the release fired before the max pass. Both are kept in the source; only one is
// compiled, because ptxas allocates for the worse of two live paths: with the
// runtime branch all six fmtc::fmha_tc_kernel instantiations report 8 bytes stack /
// 8 spill stores / 112 spill loads, and with either single path compiled the four
// TD=128 ones report 0/0/0 (dv_2cb9c748a7ac). The branch was default-OFF, so every
// production tc shape was paying two spilled registers for a path it never ran.
#ifndef FMHA_ONEWAIT_S
#define FMHA_ONEWAIT_S 0
#endif

__device__ __forceinline__ uint32_t tc_smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint64_t make_desc(uint32_t swizzle, uint32_t lbo_bytes,
                                              uint32_t sbo_bytes, const void* base) {
  uint64_t d = ((uint64_t)swizzle << 61) | (1ULL << 46) | (((uint64_t)sbo_bytes >> 4) << 32)
             | (((uint64_t)lbo_bytes >> 4) << 16);
  d |= (uint64_t)(tc_smem_u32(base) & 0x3FFFFu) >> 4;
  return d;
}
__device__ __forceinline__ void desc_add(uint64_t& d, int bytes) { d += (uint32_t)(bytes >> 4); }
__device__ __forceinline__ void mbar_init(void* m, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(tc_smem_u32(m)), "r"(count));
}
__device__ __forceinline__ void mbar_wait(void* m, uint32_t phase) {
  uint32_t a = tc_smem_u32(m);
  asm volatile(
      "{\n.reg .pred p;\nW%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra.uni W%=;\n}" ::"r"(a), "r"(phase));
}
__device__ __forceinline__ void mbar_arrive(void* m) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(tc_smem_u32(m)));
}
__device__ __forceinline__ void tc_cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(tc_smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_mbar_arrive(void* m) {
  // .noinc: the async completion performs the arrival WITHOUT bumping the
  // pending count first (count is fixed at mbarrier.init) - the plain form
  // would raise pending by 1 per instruction and the barrier never completes.
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(tc_smem_u32(m)));
}
// C1 TMA loader: 4-D tiled bulk copy (byte tensor, SW128 box 128B x page rows)
// completing into an mbarrier transaction-count barrier, plus expect_tx.
__device__ __forceinline__ void tma_4d(void* dst, const void* map, int x, int y, int z,
                                       int w, void* mbar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];"
      ::"r"(tc_smem_u32(dst)), "l"(map), "r"(x), "r"(y), "r"(z), "r"(w),
        "r"(tc_smem_u32(mbar))
      : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(void* m, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               ::"r"(tc_smem_u32(m)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void umma_commit(void* m) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               ::"r"(tc_smem_u32(m)));
}
__device__ __forceinline__ void tmem_alloc(uint32_t* smem_dst, uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               ::"r"(tc_smem_u32(smem_dst)), "r"(cols));
}
__device__ __forceinline__ void tmem_relinq() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t base, uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(base), "r"(cols));
}
__device__ __forceinline__ void tmem_ld32(uint32_t* r, uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
      "%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]),
        "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
}
__device__ __forceinline__ void tmem_st32(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,"
      "%25,%26,%27,%28,%29,%30,%31,%32};"
      ::"r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]),
        "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]),
        "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]),
        "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
        "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]));
}
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;"); }
__device__ __forceinline__ void umma_f8_ss(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc,
                                           bool accum) {
  const uint32_t zero = 0;
  const uint32_t sc = accum ? 1u : 0u;
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5,%5,%5,%5}, p;\n}"
      ::"r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(sc), "r"(zero));
}
__device__ __forceinline__ void umma_f8_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b,
                                           uint32_t idesc, bool accum) {
  const uint32_t zero = 0;
  const uint32_t sc = accum ? 1u : 0u;
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, {%5,%5,%5,%5}, p;\n}"
      ::"r"(d_tmem), "r"(a_tmem), "l"(b), "r"(idesc), "r"(sc), "r"(zero));
}
__device__ __forceinline__ uint32_t idesc_f8(int M, int N, bool transB, bool cF32 = true) {
  return (uint32_t)(((M >> 5) << 25) | ((N >> 3) << 17) | ((transB ? 1 : 0) << 16)
                    | ((cF32 ? 1u : 0u) << 4));
}
template <int Registers>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(Registers));
}
template <int Registers>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(Registers));
}
// pack two floats into e4m3 bytes: low byte = lo, high byte = hi
__device__ __forceinline__ uint32_t pack_e4m3x2(float lo, float hi) {
  uint16_t d;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return (uint32_t)d;
}
__device__ __forceinline__ float tc_ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// ---- f16x2 SIMD softmax pipeline (v4) -------------------------------------
// y (fp32 log2-domain exponent, PSCALE-folded: y = s2 - anchor + PLOG2) is
// rounded to f16 (satfinite: -inf -> -65504 -> ex2 = 0, no NaN path) and
// exp2 runs on packed halves: half the MUFU issues and half the e4m3 pack
// cost of the fp32 path. f16 rounding of y contributes <= |y|*2^-11 log2
// error, i.e. vanishing exactly where p is large; e4m3 P quantization (2^-4)
// still dominates the probability error budget.
__device__ __forceinline__ uint32_t cvt_f16x2_sat(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ uint32_t ex2_f16x2(uint32_t a) {
  uint32_t d;
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(d) : "r"(a));
  return d;
}
__device__ __forceinline__ uint32_t cvt_e4m3x2_f16x2(uint32_t a) {
  uint16_t d;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(a));
  return (uint32_t)d;
}
// two f16x2 (4 probabilities) -> one packed e4m3x4 word (mov.b32 pack saves
// the shl+or of the two-step form)
__device__ __forceinline__ uint32_t pack_e4m3x4(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("{\n.reg .b16 lo, hi;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 lo, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 hi, %2;\n"
      "mov.b32 %0, {lo, hi};\n}"
      : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ uint32_t hadd2_u32(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("add.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ uint32_t hmax2_u32(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ uint32_t hsub2_u32(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("sub.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ uint32_t hfma2_u32(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
// max(f16x2.lo, f16x2.hi) broadcast to both halves (f16-exact).
__device__ __forceinline__ uint32_t hmax_halves2(uint32_t x) {
  uint32_t d;
  asm("{\n.reg .f16 lo, hi, m;\n"
      "mov.b32 {lo, hi}, %1;\n"
      "max.f16 m, lo, hi;\n"
      "mov.b32 %0, {m, m};\n}"
      : "=r"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ uint32_t f16x2_dup(float x) {
  uint32_t d;
  asm("{\n.reg .f16 h;\ncvt.rn.f16.f32 h, %1;\nmov.b32 %0, {h, h};\n}"
      : "=r"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ float f16x2_lo(uint32_t x) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((uint16_t)(x & 0xffffu)));
  return f;
}
__device__ __forceinline__ float f16x2_hi(uint32_t x) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((uint16_t)(x >> 16)));
  return f;
}
__device__ __forceinline__ int swz128(int row, int col_byte) {
  return row * 128 + ((((col_byte >> 4) ^ (row & 7)) << 4) | (col_byte & 15));
}

template <int TD, int NSTAGE>
struct Shared {
  static constexpr int PLANES = TD / 128;
  // v6b: S/P tmem buffers. DV<=128 fits THREE 128-col S buffers + O in the
  // 512-col tmem (384 + DV), letting the mma warp run PV(t) two steps behind
  // QK(t+2): two extra QKs fill the tensor pipe during the ~1.4k-cyc softmax
  // latency (period = tensor + max(0, L - 2*QK) instead of L - QK). DV=256's
  // O needs 256 cols -> stays at 2 buffers / lag 1 (identical to v5 flow).
  static constexpr int NBUF = (TD >= 256) ? 2 : 3;
  alignas(1024) uint8_t q[PLANES][TM * 128];
  alignas(1024) uint8_t k[NSTAGE][PLANES][TN * 128];
  alignas(1024) uint8_t v[NSTAGE][PLANES][TN * 128];
  alignas(16) uint64_t q_loaded;
  alignas(16) uint64_t k_loaded[NSTAGE];
  alignas(16) uint64_t v_loaded[NSTAGE];
  alignas(16) uint64_t k_tx[NSTAGE];
  alignas(16) uint64_t v_tx[NSTAGE];
  alignas(16) uint64_t qk_done[NBUF];
  alignas(16) uint64_t score_released[NBUF];
  alignas(16) uint64_t probability_ready[NBUF];
  alignas(16) uint64_t output_ready[2];
  // v6c: dedicated single-completion barrier for the LAST PV: the epilogue's
  // old output_ready[(nloc-1)&1] parity wait could pass on an earlier
  // completion (c = target-2 window) when the final iterations skip corr and
  // the wait executes before PV(nloc-3) retires. One commit, one phase ->
  // structurally early-pass-free.
  alignas(16) uint64_t pv_last;
  // v6c: NBUF slots (was single). PV(pv) waits the arrival of softmax
  // iteration pv; with PV lag 2 on a SINGLE barrier, arrival #(pv+1) (a
  // skip-corr iteration arriving right after its P-store) can land before
  // the mma warp reaches PV(pv)'s wait and supersede the parity window ->
  // deadlock (proto_sim.py proved the cycle; job dv_9ccda477f26a hung).
  // Slot pv%NBUF's next arrival comes from iteration pv+NBUF, whose QK the
  // SAME mma warp issues strictly after PV(pv) -> program-order safe.
  alignas(16) uint64_t output_rescaled[NBUF];
  // v7: loader buffer-reuse waits target qk_done[(tl-NSTAGE)%NBUF] and
  // output_ready[(tl-NSTAGE)&1] directly (the released-barrier pairs were
  // redundant commits of the SAME tensor completions). Parity-window safety:
  // the next same-slot completion (QK(tl-1) / PV(tl-2)) is issue-ordered
  // after k_loaded[tl%NSTAGE], which this very loader iteration arrives ->
  // the wait always sees c in {target, target+1}.
  // v8d: TMA completion SPLIT again (V0 structure): k/v_loaded count=1
  // (thread0 plain arrive = "loads issued") + separate k/v_tx expect_tx
  // transaction barriers = data landed; the mma warp waits both. The v7b
  // MERGED form (single count=1 arrive.expect_tx phase) is the only TMA-path
  // delta vs V0 and hangs flakily (~50% of multi-shape engine sessions at
  // case-4-class shapes; V0's split form never hung). Keep count=1 arrives
  // (not V0's 96-thread arrive storm) - only the tx accounting is split.
  float state_m2[TM];   // dual-group online-softmax state handoff (published
  float state_d[TM];    // BEFORE the owner's probability_ready arrive)
  uint32_t tmem_base;
  int tab_cache[TN / 16 + 2];   // per-tile page-table slice (page >= 16)
};

// Tile geometry (identical math to the mma.sync kernel, M=128).
__device__ __forceinline__ void tile_geometry(const Params& p, int u, int x, int& b, int& g,
                                              int& q0, int& qlen, int& kvlen, int& R, int& tiles,
                                              int& r0, int& lo, int& hi) {
  b = u / p.Hkv;
  g = u - b * p.Hkv;
  // Hop 1 of the prologue's three dependent cold hops, removed when prepare
  // cached the geometry: 1152-1536 cyc (0.61-0.81us, 60-75% of setup) that the
  // page-table __ldg and the first K/V TMA both waited on. Identical integers,
  // so every derived quantity and every output stays bit-identical;
  // geo_valid == 0 restores the pointer path exactly.
  if (p.geo_valid) {
    q0 = p.geo_cu[b];
    qlen = p.geo_cu[b + 1] - q0;
    kvlen = p.geo_sl[b];
  } else {
    q0 = p.cu_q[b];
    qlen = p.cu_q[b + 1] - q0;
    kvlen = p.seqlens[b];
  }
  R = qlen * p.ratio;
  tiles = (R + TM - 1) / TM;
  r0 = x * TM;
  int t0 = r0 / p.ratio;
  int tlast = (r0 + TM - 1) / p.ratio;
  if (tlast > qlen - 1) tlast = qlen - 1;
  const int vis_lo = kvlen - qlen + t0;
  const int vis_hi = kvlen - qlen + tlast;
  lo = 0;
  hi = kvlen;
  if (p.causal) hi = min(hi, vis_hi + 1);
  if (p.window_left >= 0) lo = max(lo, vis_lo - p.window_left);
}

// r19 persistent-fmtc helpers. aid -> (u,bx): EXACTLY the non-persistent
// cta_id decomposition (linear id over the original (gridDim.x=T,
// gridDim.y=nunits) grid, tile_order remap included), then the causal flip
// + second tile_geometry pass + S==1 tile count. Pure function of (p, aid):
// every role recomputes the same ticket sequence independently.
struct TCAssign {
  int b, g, q0, qlen, kvlen, R, r0, lo, hi, x, u, ntl;
};
__device__ __forceinline__ bool tc_aid_geom(const Params& p, int aid,
                                            TCAssign& A) {
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
  A.ntl = (A.hi - A.lo + TN - 1) / TN;
  return A.ntl > 0;
}

#ifdef FMHA_WB
__device__ __forceinline__ unsigned long long tc_wb_gt() {
  unsigned long long v;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
  return v;
}
__device__ __forceinline__ unsigned tc_wb_smid() {
  unsigned v;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
  return v;
}
// Same slot map as the MT PERS body (r17/r18): 64..78 core + 82..90
// frontiers. fmtc builds define FMHA_WB with FMHA_DBG_ALLOC + NO_TCW only.
#define WB(i, v)                                                              \
  do {                                                                        \
    if (tcdbg) tcdbg[64 + (i)] = (unsigned)(v);                               \
  } while (0)
#else
#define WB(i, v)
#endif

// Optional progress markers (mapped host memory; null in production).
// Per-CTA region of 64 slots: [1+w]=role entered, [13+w]=last milestone
// (tile+1), [25+w]=current wait site (tile*16+site), [37+w]=reached final sync.
// FMHA_NO_TCW (timing builds only) strips the marker stores: each is an
// uncached sysmem write that would distort the clock64 accounting below.
#if defined(FMHA_NO_TCW)
#define TCW(slot, val)
#else
#define TCW(slot, val)                                                        \
  do {                                                                        \
    if (tcdbg) tcdbg[(slot)] = (unsigned)(val);                               \
  } while (0)
#endif

// Optional per-warp cycle accounting, compiled in ONLY with FMHA_TC_TIMING
// (tc_debug builds). Production builds: macros expand to nothing -> identical
// codegen/registers. Slots: 64+w*4 wait kcyc, +1 work kcyc, +2 wall kcyc,
// +3 role tile counter.
#ifdef FMHA_TC_TIMING
#define TINIT()                                                               \
  unsigned tp_ = tcdbg ? (unsigned)clock64() : 0u, t0_ = tp_;                 \
  long long cw_ = 0, ck_ = 0
#define TACC(acc)                                                             \
  do {                                                                        \
    if (tcdbg) {                                                              \
      unsigned n_ = (unsigned)clock64();                                      \
      acc += (long long)(n_ - tp_);                                           \
      tp_ = n_;                                                               \
    }                                                                         \
  } while (0)
#define TDUMP(tiles_)                                                         \
  do {                                                                        \
    if (tcdbg && lane == 0) {                                                 \
      unsigned e_ = (unsigned)clock64();                                      \
      tcdbg[64 + warp * 4 + 0] = (unsigned)(cw_ >> 10);                       \
      tcdbg[64 + warp * 4 + 1] = (unsigned)(ck_ >> 10);                       \
      tcdbg[64 + warp * 4 + 2] = (unsigned)((e_ - t0_) >> 10);                \
      tcdbg[64 + warp * 4 + 3] = (unsigned)(tiles_);                          \
    }                                                                         \
  } while (0)
#else
#define TINIT()
#define TACC(acc)
#define TDUMP(tiles_)
#endif

// TD = smem/tmem tile width (128B rows, N=128 kv, O columns per plane).
// DV = REAL head dim (Dqk == Dv per contract). DV < TD (D=64) keeps the
// 128B-row swizzle layout but: loaders fill only DV bytes/row; QK runs
// DV/32 K-steps (upper smem never read); PV idesc N = DV; correction and
// epilogues iterate DV/32 tmem blocks; ws partial rows are DV+2 floats.
// Garbage in the unwritten upper halves is never consumed by any descriptor.
template <int TD, int NSTAGE, int DV = TD, bool PERS = false>
__global__ __launch_bounds__(NTHREADS, 1)
void fmha_tc_kernel(const __grid_constant__ Params p) {
  constexpr int PLANES = TD / 128;
  constexpr int CPR = DV / (16 * PLANES);     // 16B chunks per row per plane
  constexpr int KSTEPS = DV / (32 * PLANES);  // QK K-dim 32B steps per plane
  constexpr int NPV = DV / PLANES;            // PV idesc N per plane
  // C4: TMA D64 rows are compact 64B under SWIZZLE_64B (descriptor
  // layout=4, LBO=0, SBO=512 - resolved by experiments/sw64probe.cu on this
  // device). DV>=128 keeps the 128B SW128 rows (layout=2, LBO 16/0, SBO
  // 1024). cp.async fallback rows are ALWAYS 128B-strided (swz128).
  constexpr int ROWB = (DV >= 128) ? 128 : 64;
  // v6b pipeline geometry: NBUF S/P tmem buffers, PV runs PVLAG steps behind
  // QK (see Shared). O_TMEM_V = first O column (after the S buffers).
  constexpr int NBUF = Shared<TD, NSTAGE>::NBUF;
  constexpr int PVLAG = NBUF - 1;
  const int pvlag = p.force_lag1 ? 1 : PVLAG;   // debug bisect knob
  constexpr uint32_t O_TMEM_V = 128 * NBUF;
  const bool tma64 = (DV < 128) && p.use_tma;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Shared<TD, NSTAGE>& sh = *reinterpret_cast<Shared<TD, NSTAGE>*>(smem_raw);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int cta_id = blockIdx.y * gridDim.x + blockIdx.x;
#ifdef FMHA_DBG_ALLOC
  constexpr int DBG_CTAS = 512;   // wide capture for hang hunts
#else
  constexpr int DBG_CTAS = 64;
#endif
  volatile unsigned* tcdbg =
      (p.dbg && cta_id < DBG_CTAS) ? (volatile unsigned*)(p.dbg + 128 * cta_id) : nullptr;

#ifdef FMHA_WB
  if (tid == 0) {
    const unsigned long long wb_g0 = tc_wb_gt();
    WB(0, wb_g0);
    WB(1, wb_g0 >> 32);
    WB(2, (unsigned)clock64());
    WB(3, tc_wb_smid());
  }
#endif

  if constexpr (!PERS) {
  // tile_order: unit-fastest CTA->tile remap (wave balance). Pure scheduling:
  // slot indexing stays logical (u,x,c), so outputs are bitwise identical.
  // C2 balanced: grid.x linearizes (b, g, x, c) exactly; no empty CTAs.
  int u, bx, c0;
  if (p.balanced) {
    int b_dec, g_dec;
    balanced_decode(p, b_dec, g_dec, bx, c0);
    u = b_dec * p.Hkv + g_dec;
  } else {
    u = blockIdx.y;
    bx = (int)blockIdx.x;
    c0 = (int)blockIdx.z;
    if (p.tile_order) {
      const int gid = bx + (int)blockIdx.y * (int)gridDim.x;
      const int nu = (int)gridDim.y;
      u = gid % nu;
      bx = gid / nu;
    }
  }
  int b, g, q0, qlen, kvlen, R, tiles, r0, lo, hi;
  tile_geometry(p, u, bx, b, g, q0, qlen, kvlen, R, tiles, r0, lo, hi);
  const int x = p.causal ? (tiles - 1 - bx) : bx;
  if (x < 0 || x >= tiles) return;
  tile_geometry(p, u, x, b, g, q0, qlen, kvlen, R, tiles, r0, lo, hi);
  const int ntiles = (hi - lo + TN - 1) / TN;

  // Split-KV (S>1): chunk c covers LOCAL tiles [c*cpt, (c+1)*cpt) of this
  // row-tile's [lo,hi) span; chunk is TN-aligned by the host, so the tile
  // partition is exact and each tile belongs to exactly one chunk.
  // C2 balanced: cpt is derived per (unit,x) from the actual ntiles and the
  // per-batch split count unit_S[b]; the ws slot math uses the LOGICAL unit
  // count (gridDim.y == 1 when balanced).
  const int nunits_g = p.B * p.Hkv;
  const int Su = p.balanced ? p.unit_S[b] : p.S;
  const int c = (Su > 1) ? c0 : 0;
  int t_begin = 0, t_end = ntiles;
  if (Su > 1) {
    const int cpt = p.balanced ? (ntiles + Su - 1) / Su : p.chunk / TN;
    t_begin = min(ntiles, c * cpt);
    t_end = min(ntiles, t_begin + cpt);
    if (t_begin >= t_end) {
      // Empty chunk: self-cleaning markers (m=-inf, d=0) so the combine
      // kernel skips this split; exit before any collective setup.
      const int rows = min(TM, R - x * TM);
      float* slot = p.ws + (((int64_t)c * nunits_g + u) * p.T + x)
                               * (int64_t)(TM * (DV / 2 + 2));
      if (tid < rows) {
        slot[(int64_t)tid * (DV / 2 + 2) + DV / 2] = -INFINITY;
        slot[(int64_t)tid * (DV / 2 + 2) + DV / 2 + 1] = 0.f;
      }
      TCW(63, 1);
      return;
    }
  }
  const int nloc = t_end - t_begin;

  if (tid == 0) {
    mbar_init(&sh.q_loaded, 96);
#pragma unroll
    for (int s = 0; s < NBUF; ++s) mbar_init(&sh.output_rescaled[s], 4);
#pragma unroll
    for (int s = 0; s < NSTAGE; ++s) {
      // TMA: count 1 (thread0 plain arrive) + separate tx barriers.
      // cp.async: 96 loader threads, tx barriers unused.
      mbar_init(&sh.k_loaded[s], p.use_tma ? 1 : 96);
      mbar_init(&sh.v_loaded[s], p.use_tma ? 1 : 96);
      mbar_init(&sh.k_tx[s], 1);
      mbar_init(&sh.v_tx[s], 1);
    }
#pragma unroll
    for (int s = 0; s < NBUF; ++s) {
      mbar_init(&sh.qk_done[s], 1);
      mbar_init(&sh.score_released[s], 4);
      mbar_init(&sh.probability_ready[s], 4);
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      mbar_init(&sh.output_ready[s], 1);
    }
    mbar_init(&sh.pv_last, 1);
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

  const int32_t* tab = p.table + (int64_t)b * p.Tstride;
  const uint8_t* kb = p.k + (int64_t)g * p.k_sh;
  const uint8_t* vb = p.v + (int64_t)g * p.v_sh;
  const int plog = p.page_log2, pmask = p.page - 1;

  if (warp >= 9) {
    // ------------------------------ loaders ---------------------------------
    // 96 regs: lets the compiler overlap the LDS(tab_cache)->64-bit
    // address->cp.async chains across loop iterations. Group-uniform with
    // the mma warp (w8-11 = one setmaxnreg warp group). Budget:
    // 8*200 (softmax groups) + 4*96 = 1984 warp-slots <= 2048 pool.
    setmaxnreg_dec<96>();
    TCW(1 + warp, warp + 1);
    TINIT();
    const int ltid = tid - 288;
    for (int idx = ltid; idx < TM * CPR * PLANES; idx += 96) {
      const int pl = idx / (TM * CPR);
      const int rem = idx - pl * (TM * CPR);
      const int row = rem / CPR;
      const int col = (rem - row * CPR) * 16;
      int rg = r0 + row;
      if (rg > R - 1) rg = R - 1;
      const int t = rg / p.ratio;
      const int h = g * p.ratio + (rg - t * p.ratio);
      tc_cp_async16(sh.q[pl] + swz128(row, col),
                    p.q + (int64_t)(q0 + t) * p.q_tok_stride + (int64_t)h * DV
                        + pl * 128 + col);
    }
    cp_async_mbar_arrive(&sh.q_loaded);
    TACC(ck_);
    // C1.1: the TMA and cp.async tile loops are separate copies so the
    // fallback path compiles exactly like the pre-TMA kernel (a per-tile
    // runtime branch here cost ~7-11% on loader-bound cp.async cases).
    if (p.use_tma) {
#pragma unroll 1
      for (int tl = 0; tl < nloc; ++tl) {
        const int tile = t_begin + tl;
        const int stage = tl % NSTAGE;
        const int n0t = lo + tile * TN;
        // TMA loads the tile FULL: rows >= hi hold finite fp8 from real
        // pages (table entries clamped inside the row) and are softmax-
        // masked, so no per-row clamping or posmax shrink is needed.
        const int nseg = TN >> plog;               // page segments per tile
        const int tab_lo = n0t >> plog;
        // C3a: issue the table LDG BEFORE the release waits so its ~600cy
        // global latency overlaps the wait instead of serializing after it
        // (TMA-mode loader work was still ~2.1k cyc/tile, job3 tctiming).
        int pg_reg = 0;
        if (ltid < nseg)
          pg_reg = __ldg(tab + min(tab_lo + ltid, p.Tstride - 1));
        if (tl >= NSTAGE) {
          const int wN = tl - NSTAGE;   // releases stage wN%NSTAGE == stage
          TCW(25 + warp, tile * 16 + 1);
          mbar_wait(&sh.qk_done[wN % NBUF], (wN / NBUF) & 1);
          TCW(25 + warp, tile * 16 + 2);
          mbar_wait(&sh.output_ready[wN & 1], (wN >> 1) & 1);
          TACC(cw_);
        }
        if (ltid < nseg) sh.tab_cache[ltid] = pg_reg;
        asm volatile("bar.sync 1, 96;" ::: "memory");
        // One bulk copy per (plane, page segment) instead of 96 threads x ~21
        // cp.async16 + address ALU chains (job1: loader issue paced every heavy
        // tc case). The SW128 box writes byte-identical smem to swz128(row,col),
        // so all UMMA descriptors and numerics are unchanged vs the cp.async
        // path. v8d split TMA barriers (V0 structure, count-1 arrives): thread0
        // arms tx counts, then plain-arrives the "issued" barriers; the mma warp
        // waits loaded THEN tx (V0 order).
        //
        // ISSUE WIDTH, measured in both directions (dv_854efcfff824,
        // dv_ec019fc0d590, ES_REPS 2 and 4). Letting ltid==0 walk the whole
        // 2*PLANES*nseg chain alone costs roughly 100 cyc per issue and paces the
        // loader -- this file's own note prices TMA-mode loader work at ~2.1k
        // cyc/tile -- so spreading the issues over nseg lanes, the way fmtcmt's
        // v9c already does, is worth **7.5% of c0 on both instruments** (event
        // 0.9248, wall 0.9245, page 16, 16 issues) and 4.8% event / 2.5% wall of
        // d0. But the extra bar.sync that per-lane issue requires costs more than
        // it saves once the chain is short, and the two DV=256 shapes are exactly
        // that: PLANES=2 at nseg 2 and 1 regressed **c2 +3.2% wall (both jobs)
        // and c6 +5.6% wall = +14.2us on a 227us case, reproducible to 0.08us**.
        // PLANES is TD/128, a template constant, so the gate is compile-time and
        // the TD=256 instantiations keep the shipped serial chain verbatim.
        // Nothing regressed inside PLANES==1: nseg=8 (c0) -7.5%, nseg=2 (c9)
        // -0.7% wall, nseg=1 (c3 -0.4% wall, d7 -0.5% wall).
        if (ltid == 0) {
#if FMHA_ABLATE == 9
          // zero-byte expect_tx completes the tx barriers without data moves
          mbar_arrive_expect_tx(&sh.k_tx[stage], 0);
          mbar_arrive_expect_tx(&sh.v_tx[stage], 0);
#else
          mbar_arrive_expect_tx(&sh.k_tx[stage], PLANES * TN * ROWB);
          mbar_arrive_expect_tx(&sh.v_tx[stage], PLANES * TN * ROWB);
#endif
        }
        if constexpr (PLANES == 1) {
          // v9c issue order: nseg lanes issue in parallel, each its own
          // segment's K before its V, so ALL of a tile's K requests enter the TMA
          // queue before any V -- the order the consumer wants, since the mma
          // waits k_tx for QK(j) but v_tx only for PV(j-pvlag), i.e. V already
          // has pvlag steps of slack and K is the one on the critical path.
          // Ordering: arm(expect_tx) -> bar.sync -> lanes issue -> the tx barrier
          // gates DATA (complete_tx can never precede its arm), so k_loaded may
          // arrive before every lane has issued; k_tx still gates consumption.
#if FMHA_ABLATE != 9
          asm volatile("bar.sync 1, 96;" ::: "memory");   // arms visible
          if (ltid < nseg) {
            const int seg_b = p.page * ROWB;         // segment bytes in smem
            const int pg = sh.tab_cache[ltid];
            tma_4d(sh.k[stage][0] + ltid * seg_b, &p.kmap, 0, 0, pg, g,
                   &sh.k_tx[stage]);
            tma_4d(sh.v[stage][0] + ltid * seg_b, &p.vmap, 0, 0, pg, g,
                   &sh.v_tx[stage]);
          }
#endif
        } else {
#if FMHA_ABLATE != 9
          if (ltid == 0) {
            const int seg_b = p.page * ROWB;         // segment bytes in smem
            for (int pl = 0; pl < PLANES; ++pl)
              for (int s = 0; s < nseg; ++s) {
                const int pg = sh.tab_cache[s];
                tma_4d(sh.k[stage][pl] + s * seg_b, &p.kmap, pl * 128, 0, pg, g,
                       &sh.k_tx[stage]);
                tma_4d(sh.v[stage][pl] + s * seg_b, &p.vmap, pl * 128, 0, pg, g,
                       &sh.v_tx[stage]);
              }
          }
#endif
        }
        if (ltid == 0) {
          mbar_arrive(&sh.k_loaded[stage]);
          mbar_arrive(&sh.v_loaded[stage]);
        }
        TACC(ck_);
        TCW(13 + warp, tile + 1);
      }
      TDUMP(nloc);
    } else {
#pragma unroll 1
      for (int tl = 0; tl < nloc; ++tl) {
        const int tile = t_begin + tl;
        const int stage = tl % NSTAGE;
        const int n0t = lo + tile * TN;
        // Cooperative page-table prefetch: the tile's rows [n0t, posmax] touch
        // at most TN/16+1 table entries (page >= 16). The 96 loader threads
        // fetch them with independent LDGs, publish through a loader-only
        // named barrier, and each cp.async then reads its entry from smem
        // (~30cy LDS) instead of a dependent global load (~100-400cy) that
        // paced the loader role (1.4us/tile, longest work after softmax fix).
        // C3a: LDG issues before the release waits (latency overlap).
        int posmax = n0t + TN - 1;
        if (posmax > hi - 1) posmax = hi - 1;
        const int tab_lo = n0t >> plog;
        const int nent = (posmax >> plog) - tab_lo + 1;   // <= TN/16+1 = 9
        int pg_reg = 0;
        if (ltid < nent) pg_reg = __ldg(tab + tab_lo + ltid);
        if (tl >= NSTAGE) {
          const int wN = tl - NSTAGE;   // releases stage wN%NSTAGE == stage
          TCW(25 + warp, tile * 16 + 1);
          mbar_wait(&sh.qk_done[wN % NBUF], (wN / NBUF) & 1);
          TCW(25 + warp, tile * 16 + 2);
          mbar_wait(&sh.output_ready[wN & 1], (wN >> 1) & 1);
          TACC(cw_);
        }
        if (ltid < nent) sh.tab_cache[ltid] = pg_reg;
        asm volatile("bar.sync 1, 96;" ::: "memory");
#if FMHA_ABLATE != 9
#pragma unroll 4
        for (int idx = ltid; idx < TN * CPR * PLANES; idx += 96) {
          const int pl = idx / (TN * CPR);
          const int rem = idx - pl * (TN * CPR);
          const int row = rem / CPR;
          const int col = (rem - row * CPR) * 16;
          int pos = n0t + row;
          if (pos > hi - 1) pos = hi - 1;
          const int64_t poff = (int64_t)sh.tab_cache[(pos >> plog) - tab_lo];
          const int off = pos & pmask;
          tc_cp_async16(sh.k[stage][pl] + swz128(row, col),
                        kb + poff * p.k_sp + (int64_t)off * p.k_sn + pl * 128 + col);
          tc_cp_async16(sh.v[stage][pl] + swz128(row, col),
                        vb + poff * p.v_sp + (int64_t)off * p.v_sn + pl * 128 + col);
        }
#endif
        cp_async_mbar_arrive(&sh.k_loaded[stage]);
        cp_async_mbar_arrive(&sh.v_loaded[stage]);
        TACC(ck_);
        TCW(13 + warp, tile + 1);
      }
      TDUMP(nloc);
    }
  } else if (warp == 8) {
    // ------------------------------ mma issue --------------------------------
    // dec<96> to match loaders w9-11: setmaxnreg is a warp-group collective
    // (w8-11 = one group) and requires the SAME immediate in all 4 warps.
    // (v3.4b-v3.5 ran dec<32> here vs dec<96> in w9-11 = UB that happened to
    // work; the dual-group inc<208> budget then hung w1/w6/w7 at inc.)
    setmaxnreg_dec<96>();
    TCW(1 + warp, warp + 1);
    if (lane == 0) {
      TINIT();
      TCW(25 + warp, 8);
      mbar_wait(&sh.q_loaded, 0);
      TACC(cw_);
      // QK accumulates in F32 tmem (an F16 c_format attempt produced NaN on
      // SM103 kind::f8f6f4 - rejected; see file header).
      const uint32_t id_qk = idesc_f8(TM, TN, false);
      const uint32_t id_pv = idesc_f8(TM, NPV, true);
      // v6b/c: step j issues QK(j) first, then PV(j-PVLAG). With NBUF=3 the
      // tensor pipe keeps executing queued QKs while softmax(t) is still in
      // flight (PV(t) gates on probability_ready(t) ~= qk_done(t)+L): the
      // per-tile period drops from tensor+max(0, L-QK) toward pure tensor
      // time. Supersede-safety (parity window = {target, target+1} per
      // try_wait.parity): every NBUF-slotted wait's next completion requires
      // a LATER step of this same single mma warp (score_rel: softmax(j)
      // read <- QK(j) exec; probability: softmax(pv+NBUF) <- QK(pv+NBUF)
      // issued at step pv+NBUF > pv+PVLAG since NBUF > PVLAG; out_rescaled
      // slot pv%NBUF: iteration pv+NBUF <- QK(pv+NBUF) likewise) or a later
      // PV exec (out_ready: PV(tl+1) issued at step tl+3 after corr(tl)'s
      // wait; loader v_rel: PV(tl) exec). Verified by experiments/proto_sim
      // across nloc 1..33, NBUF 2/3, corr on/off.
#pragma unroll 1
      for (int j = 0; j < nloc + pvlag; ++j) {
        if (j < nloc) {
          const int tile = t_begin + j;
          const int sb = j % NBUF;
          const int stage = j % NSTAGE;
          TCW(25 + warp, tile * 16 + 3);
#if FMHA_ABLATE != 13 && FMHA_ABLATE != 14
          mbar_wait(&sh.k_loaded[stage], (j / NSTAGE) & 1);
          if (p.use_tma) mbar_wait(&sh.k_tx[stage], (j / NSTAGE) & 1);
#endif
#if FMHA_ABLATE != 14
          if (j >= NBUF) {
            TCW(25 + warp, tile * 16 + 4);
            mbar_wait(&sh.score_released[sb], (j / NBUF - 1) & 1);
          }
#endif
          TACC(cw_);
          const uint32_t dmem = tb + sb * 128;
#if FMHA_ABLATE != 7
#pragma unroll
          for (int pl = 0; pl < PLANES; ++pl) {
            uint64_t ad = make_desc(2, 16, 8 * 128, sh.q[pl]);
            uint64_t bd = tma64 ? make_desc(4, 0, 8 * 64, sh.k[stage][pl])
                                : make_desc(2, 16, 8 * 128, sh.k[stage][pl]);
#pragma unroll
            for (int step = 0; step < KSTEPS; ++step) {
              umma_f8_ss(dmem, ad, bd, id_qk, pl != 0 || step != 0);
              if (step != KSTEPS - 1) { desc_add(ad, 32); desc_add(bd, 32); }
            }
          }
#endif
          umma_commit(&sh.qk_done[sb]);
          TACC(ck_);
          TCW(13 + warp, tile + 1);
        }
        if (j >= pvlag) {
          const int pv = j - pvlag;
          const int pb = pv % NBUF;
          const int pvstage = pv % NSTAGE;
          TCW(25 + warp, (t_begin + j) * 16 + 5);
#if FMHA_ABLATE != 13 && FMHA_ABLATE != 14
          mbar_wait(&sh.v_loaded[pvstage], (pv / NSTAGE) & 1);
          if (p.use_tma) mbar_wait(&sh.v_tx[pvstage], (pv / NSTAGE) & 1);
#endif
#if FMHA_ABLATE != 12 && FMHA_ABLATE != 14
          TCW(25 + warp, (t_begin + j) * 16 + 6);
          mbar_wait(&sh.probability_ready[pb], (pv / NBUF) & 1);
          if (pv >= 1) {
            TCW(25 + warp, (t_begin + j) * 16 + 7);
            mbar_wait(&sh.output_rescaled[pv % NBUF], (pv / NBUF) & 1);
          }
#endif
          TACC(cw_);
          const uint32_t pa = tb + pb * 128 + 32;
#if FMHA_ABLATE != 6
#pragma unroll
          for (int pl = 0; pl < PLANES; ++pl) {
            uint64_t vd = tma64 ? make_desc(4, 0, 8 * 64, sh.v[pvstage][pl])
                                : make_desc(2, 0, 8 * 128, sh.v[pvstage][pl]);
            const uint32_t om = tb + O_TMEM_V + pl * 128;
#pragma unroll
            for (int step = 0; step < 4; ++step) {
              umma_f8_ts(om, pa + step * 8, vd, id_pv, pv != 0 || step != 0);
              if (step != 3) desc_add(vd, tma64 ? 32 * 64 : 32 * 128);
            }
          }
#endif
          umma_commit(&sh.output_ready[pv & 1]);
          if (pv == nloc - 1) umma_commit(&sh.pv_last);
          TACC(ck_);
        }
      }
      TDUMP(nloc);
    }
  } else {
    // ------------- softmax groups A/B + merged correction --------------------
    // Dual-group ping-pong: group A (warps 0-3) owns EVEN local tiles, group B
    // (warps 4-7) owns ODD tiles, so softmax(t+1) overlaps QK/PV UMMA exec and
    // the other group's work (ncu dv_a25ddcd25d99: 0.81 eligible warps/sched,
    // 44% no-eligible; single-group softmax serialized ~1.23us/tile while
    // correction warps idle-waited 97%). The group owning tile tl also runs
    // tl's O-rescale (accs already in registers; ballot identity-skip kept)
    // and the last tile's group runs the final epilogue from registers.
    // Online-softmax state (m2r,dsum) hands off through state_m2/state_d smem:
    // published BEFORE the owner's probability_ready arrive (release), read
    // after the consumer's wait on that barrier (acquire). Strict alternation
    // + the state dependency make every barrier wait supersede-safe.
    // Regs: setmaxnreg is a WARP-GROUP collective (4 aligned warps must use
    // the SAME immediate) - groups: w0-3 & w4-7 inc 200, w8-11 dec 96.
    // Pool: 8*200 + 4*96 = 1984 warp-slots <= 2048; transient-safe: dec
    // releases (R0-96)*4 >= inc needs (200-R0)*8 for any R0 in [96,200].
    setmaxnreg_inc<200>();
    const int grp = warp >> 2;
    TCW(1 + warp, warp + 1);
    TINIT();
    const int r = (warp & 3) * 32 + lane;
    const int rg = r0 + r;
    const int tok = rg / p.ratio;
    const int vis_i = kvlen - qlen + tok;
    const bool row_ok = (rg < R);
    const uint32_t st_lane = ((uint32_t)(warp & 3) * 32) << 16;
    const uint32_t oaddr = tb + O_TMEM_V + st_lane;

    float m2r = -INFINITY;   // running max in RAW domain (qk_scale > 0 -> max
    float dsum = 0.f;        // commutes); tracks sum(p_hat) = PSCALE*sum(p).
#pragma unroll 1
    for (int tl = grp; tl < nloc; tl += 2) {
      const int tile = t_begin + tl;
      const int sb = tl % NBUF;
      TCW(25 + warp, tile * 16 + 9);
      mbar_wait(&sh.qk_done[sb], (tl / NBUF) & 1);
      TACC(cw_);
      uint32_t sc[128];
      // C3b: segmented tmem read. seg0 lands in registers first; segs1-3 are
      // issued and fly while seg0's mask/max runs (tcgen05.wait::ld waits for
      // ALL outstanding loads). The release arrive fires once every segment is
      // in registers - same visibility guarantee as the batched form, results
      // bitwise identical, ~2 load latencies overlapped with compute.
#if FMHA_ABLATE == 3 || FMHA_ABLATE == 11
#pragma unroll
      for (int i = 0; i < 128; ++i)
        sc[i] = __float_as_uint((float)((i + tid) & 15) - 8.f);
#else
      tmem_ld32(sc, tb + sb * 128);
#if !FMHA_ONEWAIT_S
      tmem_ld_wait();
#endif
#pragma unroll
      for (int seg = 1; seg < 4; ++seg)
        tmem_ld32(sc + seg * 32, tb + sb * 128 + seg * 32);
#endif
#if FMHA_ONEWAIT_S
      // W1 (arm-b mechanism): ONE .wait::ld for all four S segments instead of two
      // -- wait points measured ~300 cyc each on their tree (dv_07cdd563747f) vs
      // 3-7% for whole arithmetic blocks, so minimizing waits can beat C3b's
      // seg0-compute/load overlap. Release fires right here: every segment is
      // register-resident, so the S buffer is free (and QK(tl+NBUF) unblocked)
      // EARLIER than the two-wait path manages. Bitwise-neutral. NOT the default:
      // measured on this tree at 0.9985 event / 0.9929 wall over the six tc-route
      // shapes against the two-wait form's 0.9985 / 0.9936, i.e. indistinguishable,
      // so the ~300cyc/wait citation does not survive contact with this kernel.
      tmem_ld_wait();
      __syncwarp(0xffffffffu);
      if (lane == 0) mbar_arrive(&sh.score_released[sb]);
#endif
      TACC(ck_);

      const int n0t = lo + tile * TN;
      // Visible key range [c_lo, c_hi) for this tile in the int domain. The
      // selection is identical to the old per-column float predicates
      // (key < hi; causal key <= vis; window key >= vis - w; all operands
      // integral < 2^24 so the fp32 compares were exact), but sc[] is no
      // longer dynamically indexed -> it stays in registers, not local mem.
      int c_lo = 0, c_hi = 128;
      if (!row_ok) {
        c_hi = 0;
      } else {
        int ub = hi;
        if (p.causal) ub = min(ub, vis_i + 1);
        c_hi = min(128, ub - n0t);
        if (p.window_left >= 0) c_lo = max(0, vis_i - p.window_left - n0t);
        if (c_hi < c_lo) c_hi = c_lo;
      }
      const unsigned span = (unsigned)(c_hi - c_lo);
      // C3b: four independent 32-deep max chains (one per segment) instead of
      // one 127-deep serial chain: ~4x shorter dependency depth. fmaxf is
      // exact and associative -> bitwise-identical maximum.
#if FMHA_ABLATE == 2 || FMHA_ABLATE == 10 || FMHA_ABLATE == 11
      const float mx0 = 0.f, mx1 = 0.f, mx2 = 0.f, mx3 = 0.f;
#else
      float mx0 = -INFINITY, mx1 = -INFINITY, mx2 = -INFINITY, mx3 = -INFINITY;
      // NOTE: the span branch is per-ROW (lane-divergent); only plain
      // register ops may live inside it. The tcgen05 wait + warp sync +
      // release arrive below stay in warp-uniform code (v3a lesson).
      if (span >= 128) {
#pragma unroll
        for (int i = 0; i < 32; ++i) mx0 = fmaxf(mx0, __uint_as_float(sc[i]));
      } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const bool in = (unsigned)(i - c_lo) < span;
          float raw = __uint_as_float(sc[i]);
          raw = in ? raw : -INFINITY;
          sc[i] = __float_as_uint(raw);
          mx0 = fmaxf(mx0, raw);
        }
      }
#endif
#if !FMHA_ONEWAIT_S
      tmem_ld_wait();
      // Publish ALL lanes' completion before lane0's release-arrive: without
      // the warp sync, lane0's mbarrier.arrive can execute while mates'
      // tcgen05.ld are still in flight and QK(tl+2) may overwrite S early.
      __syncwarp(0xffffffffu);
      if (lane == 0) mbar_arrive(&sh.score_released[sb]);
#endif
#if FMHA_ABLATE != 2 && FMHA_ABLATE != 10 && FMHA_ABLATE != 11
      if (span >= 128) {
#pragma unroll
        for (int i = 32; i < 64; ++i) mx1 = fmaxf(mx1, __uint_as_float(sc[i]));
#pragma unroll
        for (int i = 64; i < 96; ++i) mx2 = fmaxf(mx2, __uint_as_float(sc[i]));
#pragma unroll
        for (int i = 96; i < 128; ++i) mx3 = fmaxf(mx3, __uint_as_float(sc[i]));
      } else {
#pragma unroll
        for (int i = 32; i < 64; ++i) {
          const bool in = (unsigned)(i - c_lo) < span;
          float raw = __uint_as_float(sc[i]);
          raw = in ? raw : -INFINITY;
          sc[i] = __float_as_uint(raw);
          mx1 = fmaxf(mx1, raw);
        }
#pragma unroll
        for (int i = 64; i < 96; ++i) {
          const bool in = (unsigned)(i - c_lo) < span;
          float raw = __uint_as_float(sc[i]);
          raw = in ? raw : -INFINITY;
          sc[i] = __float_as_uint(raw);
          mx2 = fmaxf(mx2, raw);
        }
#pragma unroll
        for (int i = 96; i < 128; ++i) {
          const bool in = (unsigned)(i - c_lo) < span;
          float raw = __uint_as_float(sc[i]);
          raw = in ? raw : -INFINITY;
          sc[i] = __float_as_uint(raw);
          mx3 = fmaxf(mx3, raw);
        }
      }
#endif
      if (tl >= 1) {
        // State handoff: tile tl-1's group published state_m2/state_d before
        // arriving probability_ready[(tl-1)&1] (release/acquire). Placed
        // AFTER the release-arrive so the S-buffer turnaround never waits on
        // the handoff (values are first consumed by mnr/accs below).
        TCW(25 + warp, tile * 16 + 11);
        mbar_wait(&sh.probability_ready[(tl - 1) % NBUF], ((tl - 1) / NBUF) & 1);
        m2r = sh.state_m2[r];
        dsum = sh.state_d[r];
        TACC(cw_);
      }
      const float tmaxr = fmaxf(fmaxf(mx0, mx1), fmaxf(mx2, mx3));
      const float qs = p.qk_scale;
      // v5 anchored max: rebase the anchor onto this tile's max only when it
      // exceeds the anchor by REBASE_MARGIN (post-scale log2 units), so every
      // p_hat <= 2^(8+margin) ~= 388 < 448 (e4m3 satfinite can never clamp).
      // Edge cases: first finite tile: (+inf)*qs > margin -> rebase; fully
      // masked tile (tmaxr=-inf, anchor finite): (-inf)*qs -> no rebase;
      // both -inf: NaN -> no rebase (all p are 0; mbase clamps to 0).
      const bool rebase = p.force_corr ||
                          (((tmaxr - m2r) * qs) > REBASE_MARGIN);
      const float mnr = rebase ? tmaxr : m2r;
      const float mbase = (mnr == -INFINITY) ? 0.f : mnr;
      // +PLOG2 folds PSCALE into the exponent: p_hat = 2^(s2 - anchor + 5) is
      // the e4m3 P operand directly; dsum below therefore tracks 32*sum(p).
      const float nmus8 = PLOG2 - mbase * qs;
      const float accs = rebase ? tc_ex2((m2r - mnr) * qs) : 1.0f;
      uint32_t pw[32];
#if FMHA_ABLATE == 1 || FMHA_ABLATE == 10 || FMHA_ABLATE == 11
#pragma unroll
      for (int widx = 0; widx < 32; ++widx) pw[widx] = 0x30303030u;
      const float psum = 128.0f;
#else
      uint32_t ps2[4] = {0u, 0u, 0u, 0u};   // f16x2 partial sums of p_hat
      // v7a: y = s2*qs + nmus8 evaluated in f16x2 (2 cvt + 2 hfma2 replace
      // 4 fma + 2 cvt: 11 ops/4 scores vs 14). Extra error vs the f32-fma
      // form: s2 and qs f16 rounding => |dy| <= ~3*|s2*qs|*2^-11 <= 0.06 log2
      // in-domain (|s2*qs| <= ~40; masked -inf -> satfinite -65504 -> ex2=0),
      // i.e. <= ~4% on tail p, below the e4m3 P quantum (2^-4) that follows.
      const uint32_t qs2 = f16x2_dup(qs);
      const uint32_t nm2 = f16x2_dup(nmus8);
#pragma unroll
      for (int widx = 0; widx < 32; ++widx) {
        const uint32_t sa = cvt_f16x2_sat(__uint_as_float(sc[4 * widx + 0]),
                                          __uint_as_float(sc[4 * widx + 1]));
        const uint32_t sb = cvt_f16x2_sat(__uint_as_float(sc[4 * widx + 2]),
                                          __uint_as_float(sc[4 * widx + 3]));
        const uint32_t pa = ex2_f16x2(hfma2_u32(sa, qs2, nm2));
        const uint32_t pb = ex2_f16x2(hfma2_u32(sb, qs2, nm2));
        ps2[widx & 3] = hadd2_u32(ps2[widx & 3], pa);
        ps2[widx & 3] = hadd2_u32(ps2[widx & 3], pb);
        pw[widx] = pack_e4m3x4(pa, pb);
      }
      // f16 partial sums (<= 16*266 = 4256 per chain, < 65504: no overflow)
      // finish in fp32; dsum carries the 256 scale across tiles/state.
      const uint32_t pst = hadd2_u32(hadd2_u32(ps2[0], ps2[1]),
                                     hadd2_u32(ps2[2], ps2[3]));
      const float psum = f16x2_lo(pst) + f16x2_hi(pst);
#endif
      dsum = dsum * accs + psum;
      m2r = mnr;
      // Publish state BEFORE the probability_ready arrive below (release).
      sh.state_m2[r] = m2r;
      sh.state_d[r] = dsum;
#if FMHA_ABLATE != 4 && FMHA_ABLATE != 11
      tmem_st32(tb + sb * 128 + 32 + st_lane, pw);
      tmem_st_wait();
#endif
      // Release-arrive by lane0 must publish EVERY lane's state_m2/state_d
      // stores and tcgen05.st completion; __syncwarp makes them visible to
      // lane0 first (cumulative release). Fixes an inherited V0 race that
      // made tc-path outputs nondeterministic run-to-run (observed out diff
      // up to ~0.096 on case7, lse diff ~0.004 on case9 via A/B parity).
      __syncwarp(0xffffffffu);
      if (lane == 0) mbar_arrive(&sh.probability_ready[sb]);
      TACC(ck_);
      // v8 arrival discipline (reverts the v5 skip-arrive): iteration tl
      // ALWAYS waits PV(tl-1)'s completion (output_ready) before arriving
      // output_rescaled[tl % NBUF] - V0's structurally-proven ordering, where
      // arrival #tl implies PV(tl-1) DONE. The skip-arrive variant produced a
      // FLAKY production-timing deadlock (eval dv_bc4216111188 shape 4; repro
      // dv_acab6f7881a2; not re-triggerable in 45 bisect launches,
      // dv_93de6484038a, nor under marker perturbation). With PV lag-2 the
      // unconditional wait costs ~0: the same mma step issues QK(tl+2) ahead
      // of PV(tl), covering the wake latency on the tensor pipe. Only the
      // tmem rescale itself stays conditional on rebase (rare: margin 3).
      // Uniform slot numbering: PV(pv>=1) waits parity (pv/NBUF)&1; iteration
      // 0's arrival is a no-op assertion (PV(0) never waits).
      {
        if (tl >= 1) {
          const bool do_corr = __ballot_sync(0xffffffffu, rebase);
#ifdef FMHA_TC_TIMING
          if (tcdbg && lane == 0 && do_corr) tcdbg[112 + warp] += 1;
#endif
#if FMHA_ABLATE != 12 && FMHA_ABLATE != 14
          TCW(25 + warp, tile * 16 + 10);
          mbar_wait(&sh.output_ready[(tl - 1) & 1], ((tl - 1) >> 1) & 1);
          TACC(cw_);
#endif
#if FMHA_ABLATE != 5 && FMHA_ABLATE != 11 && FMHA_ABLATE != 12 && FMHA_ABLATE != 14
          if (do_corr) {
            // Rescale O accumulated through PV(tl-1) by accs (anchor rebase).
            // C3c: batched correction - issue all block loads, ONE wait, scale,
            // issue all stores, ONE wait (V0 serialized 4x ld/wait/st/wait ~
            // 2.2K cyc; batched ~0.7K). DV=256 caps at 4 blocks (128 regs) per
            // batch. Per-element scaling order unchanged -> bitwise identical.
            constexpr int NB = DV / 32;             // 2 (D64), 4 (D128), 8 (D256)
            constexpr int BB = NB < 4 ? NB : 4;     // blocks per batch
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
#endif
          __syncwarp(0xffffffffu);
        }
        if (lane == 0) mbar_arrive(&sh.output_rescaled[tl % NBUF]);
        TACC(ck_);
      }
      TCW(13 + warp, tile + 1);
    }

    if (grp == ((nloc - 1) & 1)) {
      // Final epilogue, owned by the last tile's group; state lives in regs.
      TCW(25 + warp, (t_begin + nloc) * 16 + 13);
      mbar_wait(&sh.pv_last, 0);
      TACC(cw_);
      // NOTE: tcgen05.ld is sync.aligned - ALL 32 lanes of the warp must
      // execute it. Loads stay uniform; only gmem/ws stores are row-guarded.
      if (Su > 1) {
        // Split-KV partials: O raw (PSCALE baked in); dsum already carries the
        // PSCALE=2^PLOG2 factor (f16 pipeline folds it into the exponent), so
        // the combine's out = bmm2*acc/D stays exact with lse_adj = -PLOG2.
        float* slot = p.ws + (((int64_t)c * nunits_g + u) * p.T + x)
                                 * (int64_t)(TM * (DV / 2 + 2));
        float* srow = slot + (int64_t)r * (DV / 2 + 2);
        uint32_t* wrow = reinterpret_cast<uint32_t*>(srow);
        if (row_ok) {
          srow[DV / 2] = m2r * p.qk_scale;
          srow[DV / 2 + 1] = dsum;
        }
        // C3d: batched epilogue loads (one wait per <=4-block batch).
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
          // dsum carries PSCALE=2^PLOG2: base2 subtracts PLOG2, inv divides
          // by dsum directly (== bmm2/(PSCALE*dsum_unscaled)).
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
        // C3d: batched epilogue loads (one wait per <=4-block batch).
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
      }
      TACC(ck_);
    }
    TDUMP((nloc - grp + 1) / 2);
  }

  TCW(37 + warp, 1);
  if (p.exit_mode != 2) {
    __syncthreads();
    TCW(50 + warp, 1);
#ifdef FMHA_WB
    if (warp == 9 && lane == 0) WB(12, (unsigned)clock64());
    if (tid == 0) {
      WB(13, (unsigned)clock64());
      WB(14, (unsigned)tc_wb_gt());
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

  // ================= r19 PERSISTENT-FMTC (PERS=true) =====================
  // Port of the r18k MT persistent design (all mechanisms bitwise-proven
  // there): dynamic ticket pull from the global heavy-first pool, Q double
  // buffer at PLANES==1 (D256 keeps ONE Q buffer — smem cap — guarded by a
  // qk_done wait before refill), global-J barrier parities, alternating
  // q_loaded slots + arrive throttle, overshoot-free loader guard,
  // o_drained O-handoff, ALWAYS-group-1 epilogue with state-smem pickup.
  // S==1 direct-out scope only (host-gated); balanced/S>1 never launch it.
  if constexpr (PERS) {
#include "fmha_tc_pers.cuh"
  }
}

}  // namespace fmtc
