// tcgen05 (SM100/SM103 5th-gen tensor core) BF16 FMHA prefill kernel for L20D.
// Independent implementation for the frozen fmha_bf16 contract; PTX-level
// tcgen05.mma kind::f16 with S/P/O accumulators in TMEM, bf16 paged-KV smem
// tiles in SW128 atom layouts, cp.async producers, mbarrier choreography.
// Descriptor formats cross-checked against公开发布 SM100 kernels (cudnn_fe_oss
// sm100 fprop derivation in Atrex-FA and flashinfer cake_mamba sm100 sources);
// all numerical work here is independently written.
// Requires -arch=sm_103a.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math.h>

#include "fmha_params.h"

#ifdef FMHA_TMA
#include <cuda.h>
#include <dlfcn.h>
#endif

#define NEG_INF (-__int_as_float(0x7f800000))

// Row-max/row-sum reduction shape for the 4-warp fmha_tc_main NT-wide loops.
// The historical CHAIN1 verdict (dv_889aec8778a9: serial beats 4-way overall)
// was decided by the SHORT-chain kernels (WS/main8 HC<=64); the chain0 arm on
// c4 (dv_fef6b0596561, x0.9905, engine correctness passed) showed the
// 128-element chains of main<64,128,3> prefer the 4-way tree (~512 -> ~140
// cycles of serial dependency per stage). Tree for NT >= 128, serial below;
// with FMHA_CHAIN1 off the legacy all-tree behavior is kept.
#ifdef FMHA_CHAIN1
#define FMHA_TC_CHAIN_TREE_MIN 128
#else
#define FMHA_TC_CHAIN_TREE_MIN 0
#endif

// PACK2 phase-2 (post-FFMA2 issue-slot reduction, screened separately):
// RMAX = 4-way tree for the main8ws CHAIN1 serial local row-max chain
// (bitwise: fmaxf is exact/associative; the historical serial-wins verdict
// dv_889aec8778a9 bundled rmax+lsum, and FFMA2 has since changed the exp
// loop's slot balance). LSUM = fadd2-packed paired lsum accumulation
// (tolerance: pairing order changes). MUL = fmul2-packed O-rescale
// (tolerance: denormal-flush divergence only).
#if defined(FMHA_PACK2_RMAX) || defined(FMHA_PACK2_LSUM) || defined(FMHA_PACK2_MUL)
#define FMHA_PACK2 1
#endif

#define P_Q    (reinterpret_cast<const __nv_bfloat16*>(P.q))
#define P_K    (reinterpret_cast<const __nv_bfloat16*>(P.k))
#define P_V    (reinterpret_cast<const __nv_bfloat16*>(P.v))
#define P_OUT  (reinterpret_cast<__nv_bfloat16*>(P.out))

__device__ __forceinline__ uint32_t su32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- instruction descriptor (kind::f16, bf16 a/b, f32 acc, dense) ----
__host__ __device__ constexpr uint32_t tc_idesc(int M, int N, bool b_major_mn) {
  return (uint32_t)((M >> 5) << 25) | (uint32_t)((N >> 3) << 17)
       | (uint32_t(b_major_mn) << 16) | (1u << 10) | (1u << 7) | (1u << 4);
}

// ---- smem descriptor: SW128 (2), version 1, SBO/LBO in 16B units ----
__device__ __forceinline__ uint64_t tc_desc(uint32_t smem_addr, uint32_t sbo_bytes,
                                            uint32_t lbo_bytes) {
  return (uint64_t(smem_addr >> 4) & 0x3FFF)
       | (uint64_t(1) << 46)
       | (uint64_t(sbo_bytes >> 4) << 32)
       | (uint64_t(lbo_bytes >> 4) << 16)
       | (uint64_t(2) << 61);
}

__device__ __forceinline__ void mma_ss(uint32_t tmem_c, uint64_t da, uint64_t db,
                                       uint32_t idesc, bool accum) {
  asm volatile("{\n\t.reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
               :: "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"(int(accum)));
}
__device__ __forceinline__ void mma_ts(uint32_t tmem_c, uint32_t tmem_a, uint64_t db,
                                       uint32_t idesc, bool accum) {
  asm volatile("{\n\t.reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}"
               :: "r"(tmem_c), "r"(tmem_a), "l"(db), "r"(idesc), "r"(int(accum)));
}
__device__ __forceinline__ void umma_commit(uint32_t mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(mbar_addr) : "memory");
}
__device__ __forceinline__ void mbar_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t phase) {
  asm volatile("{\n\t.reg .pred c;\n\tW%=:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 c, [%0], %1;\n\t"
               "@!c bra.uni W%=;\n\t}" :: "r"(addr), "r"(phase) : "memory");
}
// Producer arrival (release). Paired with the acquire in mbar_wait, so a
// consumer that observes the phase also observes the producer's prior tcgen05
// stores (the producer issues tcgen05.fence::before_thread_sync first).
__device__ __forceinline__ void mbar_arrive(uint32_t addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
               :: "r"(addr) : "memory");
}
// Named barrier 1 over exactly the 256 softmax threads (warps 0-7) of the
// warp-specialized kernel, so the control warp is never part of it and cannot
// be stalled by - or stall - the cross-half exchange.
__device__ __forceinline__ void bar_softmax() {
  asm volatile("bar.sync 1, 256;" ::: "memory");
}
// Same named barrier over a compile-time thread count (the WS kernel uses it
// for its NH*128 softmax threads; NH=2 reproduces bar_softmax exactly).
template <int THREADS>
__device__ __forceinline__ void bar_softmax_t() {
  asm volatile("bar.sync 1, %0;" :: "n"(THREADS) : "memory");
}
__device__ __forceinline__ void tmem_alloc(uint32_t dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :: "r"(dst_smem), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t base, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
               :: "r"(base), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_ld32(float* d, uint32_t addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]),
                 "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]),
                 "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]),
                 "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
                 "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]),
                 "=f"(d[30]), "=f"(d[31])
               : "r"(addr));
}
__device__ __forceinline__ void tmem_st32(uint32_t addr, const float* v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
               "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
               :: "r"(addr), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]),
                  "f"(v[5]), "f"(v[6]), "f"(v[7]), "f"(v[8]), "f"(v[9]), "f"(v[10]),
                  "f"(v[11]), "f"(v[12]), "f"(v[13]), "f"(v[14]), "f"(v[15]), "f"(v[16]),
                  "f"(v[17]), "f"(v[18]), "f"(v[19]), "f"(v[20]), "f"(v[21]), "f"(v[22]),
                  "f"(v[23]), "f"(v[24]), "f"(v[25]), "f"(v[26]), "f"(v[27]), "f"(v[28]),
                  "f"(v[29]), "f"(v[30]), "f"(v[31]));
}
__device__ __forceinline__ void tmem_st16(uint32_t addr, const uint32_t* v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
               "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
               :: "r"(addr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]),
                  "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]),
                  "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }

#ifdef FMHA_LDRED
// Fused load + row-max reduction from TMEM (mechanism M6 of the production
// reference, PRODUCTION_REFERENCE/context_*.cu unmasked loop:
// "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32" - sm_103/sm_101+ family
// per the reference's own #error guard; L20D is CC 10.3). Returns the 64
// column values AND their per-lane max in one pass, replacing the 2x
// tmem_ld32 + 63-fmax 4-way tree of the NH1 row-split interior path.
// max.f32 semantics match fmaxf (NaN -> the other operand); FTZ handling
// may differ from the fast-math fmaxf on denormals only -> tolerance class.
__device__ __forceinline__ void tmem_ld64_red_max(float* d, float& rmax,
                                                  uint32_t addr) {
  asm volatile(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
      " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63]), "=f"(rmax)
      : "r"(addr));
}
#endif

__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;"); }
__device__ __forceinline__ void fence_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;"); }
__device__ __forceinline__ void fence_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
// 4B cp.async for the deep page-ID ring (FMHA_PGDEEP).
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ uint32_t pack_bf16(float x, float y) {
  __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&h);
}

// Single-instruction ex2.approx.ftz for the hot softmax path. Rel error ~2^-22
// on normal range: >14 bits tighter than the bf16 P rounding (2^-8) that
// dominates softmax precision, so fixed tolerances are unaffected. exp2f may
// lower to a multi-instruction sequence; tc kernels are issue-bound (NCU c1:
// IPC 2.07, SM 44%, DRAM 0.7%).
__device__ __forceinline__ float exp2_fast(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// Packed f32x2 arithmetic (sm_100/sm_103 Blackwell FFMA2), mechanism per the
// cudnn_fe_oss sm100 fprop derivation in Atrex-FA (mounted reference).
// fma.rn.ftz.f32x2 computes two independent round-to-nearest-even FTZ FMAs -
// lane-identical to the scalar fmaf lowering under --use_fast_math (which sets
// -ftz=true), so results are BITWISE identical; it only halves the ISSUE
// slots for the score->exponent affine map in the hot softmax loop. The
// kernel is issue-bound (NCU c1: IPC 2.07, SM 44%), so cutting 16 of the
// ~112 instructions/thread/stage in the exp loop is a direct issue-slot win
// (measured x0.945-0.988 on the main8ws cases, dv_7f1d77e885cc). exp2 stays
// scalar (no packed f32 ex2 exists), so P values are unchanged.
__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
  uint64_t d;
  asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;\n"
      : "=l"(d)
      : "l"(reinterpret_cast<const uint64_t&>(a)),
        "l"(reinterpret_cast<const uint64_t&>(b)),
        "l"(reinterpret_cast<const uint64_t&>(c)));
  return reinterpret_cast<float2&>(d);
}

// PACK2 phase-2 helpers (same reference pattern): packed add/mul. add.f32x2
// changes lsum pairing order -> rounding-level only (tolerance arms).
// mul.f32x2 is per-lane RN like the scalar multiply; divergence from the
// fast-math scalar lowering is limited to denormal flushing -> tolerance.
#ifdef FMHA_PACK2
__device__ __forceinline__ float2 fadd2(float2 a, float2 b) {
  uint64_t c;
  asm("add.f32x2 %0, %1, %2;\n"
      : "=l"(c)
      : "l"(reinterpret_cast<const uint64_t&>(a)),
        "l"(reinterpret_cast<const uint64_t&>(b)));
  return reinterpret_cast<float2&>(c);
}

__device__ __forceinline__ float2 fmul2(float2 a, float2 b) {
  uint64_t c;
  asm("mul.f32x2 %0, %1, %2;\n"
      : "=l"(c)
      : "l"(reinterpret_cast<const uint64_t&>(a)),
        "l"(reinterpret_cast<const uint64_t&>(b)));
  return reinterpret_cast<float2&>(c);
}
#endif

// E2E: emulated packed exp2 for MUFU/FMA-pipe balancing. Mechanism per the
// mounted cudnn_fe_oss sm100 d128 fprop reference (exp2_emulated_2, shipped
// there interleaved with native ex2 at OK_FA_E2E_FREQ=16/RES=8: "To better
// overlap EX2 MUFU and emulation"): floor via the 1.5*2^23 magic-number
// trick (add.rm), cubic minimax polynomial for 2^f via three fma.rn.ftz.
// f32x2, exponent field splice via shl+add.s32. Independent re-encoding
// here, plus a NaN fixup (setp.neu/selp) that the reference omits - native
// ex2.approx.ftz(NaN)=NaN and the frozen nonfinite semantics are kept.
// Inputs are clamped to >= -127 like the reference (exp2 args here are
// <= ~lazy_thresh by construction; -inf/masked lanes -> exactly +0.0).
// Relative error ~2^-22 vs ex2.approx's ~2^-22: TOLERANCE-level.
#ifdef FMHA_E2E
#if FMHA_E2E >= 2
#define E2E_EMUL(i) (((i) & 15) >= 12)   // 4 of 16 pairs emulated (25%)
#else
#define E2E_EMUL(i) (((i) & 15) >= 8)    // 8 of 16 pairs emulated (50%)
#endif
__device__ __forceinline__ float2 exp2_emul2(float2 a) {
  float ox, oy;
  asm("{\n\t"
      ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
      ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
      ".reg .b32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
      ".reg .pred q1, q2;\n\t"
      "setp.neu.f32 q1, %2, %2;\n\t"
      "setp.neu.f32 q2, %3, %3;\n\t"
      "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
      "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
      "mov.b64 l1, {f1, f2};\n\t"
      "mov.f32 f3, 0f4B400000;\n\t"
      "mov.b64 l2, {f3, f3};\n\t"
      "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
      "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
      "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
      "mov.f32 f7, 0f3D9DF09D;\n\t"
      "mov.b64 l6, {f7, f7};\n\t"
      "mov.f32 f6, 0f3E6906A4;\n\t"
      "mov.b64 l5, {f6, f6};\n\t"
      "mov.f32 f5, 0f3F31F519;\n\t"
      "mov.b64 l4, {f5, f5};\n\t"
      "mov.f32 f4, 0f3F800000;\n\t"
      "mov.b64 l3, {f4, f4};\n\t"
      "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
      "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
      "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
      "mov.b64 {r1, r2}, l7;\n\t"
      "mov.b64 {r3, r4}, l10;\n\t"
      "shl.b32 r5, r1, 23;\n\t"
      "add.s32 r7, r5, r3;\n\t"
      "shl.b32 r6, r2, 23;\n\t"
      "add.s32 r8, r6, r4;\n\t"
      "selp.b32 %0, 0x7FC00000, r7, q1;\n\t"
      "selp.b32 %1, 0x7FC00000, r8, q2;\n\t"
      "}\n"
      : "=f"(ox), "=f"(oy)
      : "f"(a.x), "f"(a.y));
  return make_float2(ox, oy);
}
#endif

__device__ __forceinline__ float sink_adjust(float base2, float sink2, float* scale) {
  float d = sink2 - base2;
  float mx = fmaxf(base2, sink2);
  float total2 = mx + log2f(1.0f + exp2f(-fabsf(d)));
  *scale = exp2f(base2 - total2);
  return total2;
}

// Wave dephase (runtime knob FMHA_WS_DEPHASE, _pad2 bits 16-23 = cycles>>4).
// With 2 CTAs/SM the co-resident CTAs start in phase (same launch, same
// prologue length) and stay in phase: both run softmax while the SHARED
// tcgen05 pipe idles, then both queue PV+QK behind each other. Measured on
// c1 (WS d128 NT64): stage period 2972 cyc = softmax work ~1440 + own pipe
// work ~768 + ~764 queuing behind the sibling CTA (tensor duty 36.2%).
// Delaying one class of CTAs by ~half a pipe round overlaps one CTA's tensor
// phase with the other's softmax phase. Class = lin ^ (lin / %nsmid)
// alternates within a co-resident pair under both plausible block->SM
// mappings ((j, j+nsmid) round-robin and (2k, 2k+1) contiguous). Pure issue
// delay: same instructions, same values, same per-CTA order -> bitwise
// identical; any mismatch under this knob is a pre-existing race exposed.
__device__ __forceinline__ int dephase_cycles(const FmhaParams& P) {
  const int dph = ((P._pad2 >> 16) & 0xFF) << 4;
  if (!dph) return 0;
  unsigned smc;
  asm("mov.u32 %0, %%nsmid;" : "=r"(smc));
  const unsigned lin =
      blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  return (((lin ^ (smc ? lin / smc : 0u)) & 1u) != 0u) ? dph : 0;
}
__device__ __forceinline__ void dephase_spin(const FmhaParams& P) {
  const int d = dephase_cycles(P);
  if (d) {
    const long long t0 = clock64();
    while (clock64() - t0 < (long long)d) {
    }
  }
}

// ---- tile mapping (shared semantics with the mma.sync kernel) ----
struct TileInfo {
  int b, qlen, kvlen, m0, mrows, lo, hi;
  bool ok;
};

// Grid-order knob (_pad2 bit14, FMHA_GRIDORDER=2): the launcher swaps grid
// x/y so the TILE index walks blockIdx.x (fastest) and kvh blockIdx.y,
// i.e. kvh-MAJOR execution order. Under the evaluator's per-iteration L2
// eviction (512 MB RMW, restore-l2-evict-event-v2) the big prefill cases
// re-stream each tile's causal KV prefix from DRAM: c1's measured 755.8 us
// equals its full tile x kvh re-read traffic (~5.0 GB) at ~94% of the
// 7.09 TB/s roof, c7 ~87%, while their warm-L2 device time is 350/80 us.
// Tile-major waves mix 8 kvh x ~37 tiles -> a ~62 MB shared-prefix window
// that apparently exceeds what L2 captures; kvh-major keeps the concurrent
// window at one kv head's unique KV (c1: ~8-13 MB) so the nested prefixes
// hit. Pure permutation of the block->(tile,kvh,split) mapping: every block
// computes the same tile with the same data, so outputs are bitwise
// identical. Swap mode puts tiles on gridDim.x (2^31 cap), so the TILE_FOLD
// compliance path is only needed (and only applied) in the legacy layout.
__device__ __forceinline__ void grid_pos(const FmhaParams& P, int& kvh,
                                         int& tile_y, int& split) {
  split = (int)blockIdx.z;
  if (P._pad2 & 16384) {
    tile_y = (int)blockIdx.x;
    kvh = (int)blockIdx.y;
    return;
  }
  tile_y = (int)blockIdx.y;
  kvh = (int)blockIdx.x;
#ifdef FMHA_TILE_FOLD
  // Folded launch (host folds tiles into grid.z when tiles_ub > 65535):
  // z = split + splits_ub * tile_chunk. Identity when unfolded
  // (gridDim.z == splits_ub) - the case for every suite/domain shape.
  if (gridDim.z > (unsigned)P.splits_ub) {
    split = (int)(blockIdx.z % (unsigned)P.splits_ub);
    tile_y += (int)gridDim.y * (int)(blockIdx.z / (unsigned)P.splits_ub);
  }
#endif
}

__device__ TileInfo map_tile(const FmhaParams& P, int tile) {
  TileInfo ti;
  int acc = 0;
  for (int b = 0; b < P.batch; ++b) {
    int q0 = P.cu_q[b], q1 = P.cu_q[b + 1];
    int qlen = q1 - q0;
    int mrows_req = qlen * P.group;
    int t = (mrows_req + P.m_tile - 1) / P.m_tile;
    if (tile < acc + t) {
      ti.b = b; ti.qlen = qlen; ti.kvlen = P.seq_lens[b];
      ti.m0 = (tile - acc) * P.m_tile;
      ti.mrows = min(P.m_tile, mrows_req - ti.m0);
      int qf = ti.m0 / P.group;
      int ql = (ti.m0 + ti.mrows - 1) / P.group;
      int vis_min = ti.kvlen - qlen + qf;
      int vis_max = ti.kvlen - qlen + ql;
      ti.hi = P.causal ? min(vis_max + 1, ti.kvlen) : ti.kvlen;
      ti.lo = (P.window_left >= 0) ? max(0, vis_min - P.window_left) : 0;
      ti.ok = (ti.mrows > 0) && (ti.hi > ti.lo);
      return ti;
    }
    acc += t;
  }
  ti.ok = false; ti.b = 0; ti.qlen = 0; ti.kvlen = 0; ti.m0 = 0; ti.mrows = 0; ti.lo = 0; ti.hi = 0;
  return ti;
}

__device__ bool split_range(const FmhaParams& P, const TileInfo& ti, int split,
                            int& kv0, int& kv1, int& active) {
  int L = ti.hi - ti.lo;
  if (P.splits_ub <= 1) { kv0 = ti.lo; kv1 = ti.hi; active = 1; return true; }
  int chunk = (L + P.splits_ub - 1) / P.splits_ub;
  chunk = (chunk + P.chunk_round - 1) / P.chunk_round * P.chunk_round;
  // Uniform-work floor: never hand a block less than chunk_floor keys, so every
  // resident block runs about the same number of stages. chunk only grows, so
  // active stays <= splits_ub and the (kvh, tile, split) grid still covers L.
  if (P.chunk_floor > chunk)
    chunk = (P.chunk_floor + P.chunk_round - 1) / P.chunk_round * P.chunk_round;
  active = (L + chunk - 1) / chunk;
  if (split >= active) return false;
  kv0 = ti.lo + split * chunk;
  kv1 = min(kv0 + chunk, ti.hi);
  return true;
}

// ---------------------------------------------------------------------------
// Cooperative tile loaders -> SW128 atom-blocked smem layouts.
// K-major (Q,K): addr(row,col) = atom(col/64)*(rows*128B) + row*128B +
//                                swizzle16(col%64); atom = 64 bf16 = 128B.
// MN-major (V):  addr(kv,dv)   = natom(dv/64)*(rows*128B) + kv*128B +
//                                swizzle16(dv%64).
// ---------------------------------------------------------------------------
template <int ROWS, int D, int THREADS = 128>
__device__ __forceinline__ void tc_load_q(const FmhaParams& P, const TileInfo& ti,
                                          char* dst, int kvh, int tid) {
  constexpr int QU = D / 8;
  const int cu = P.cu_q[ti.b];
#pragma unroll
  for (int i = 0; i < (ROWS * QU) / THREADS; ++i) {
    const int u = tid + i * THREADS;
    const int row = u / QU, unit = u % QU;
    const int rg = ti.m0 + row;
    const int q_idx = min(rg / P.group, ti.qlen - 1);
    const int head = kvh * P.group + (rg % P.group);
    const __nv_bfloat16* src = P_Q + (int64_t)(cu + q_idx) * P.q_tok_stride
                             + (int64_t)head * P.q_head_stride + unit * 8;
    const int atom = unit >> 3, uin = unit & 7;
    cp_async16(su32(dst + atom * (ROWS * 128) + row * 128 + (((uin ^ row) & 7) * 16)), src);
  }
}

template <int NT, int D, bool IS_V, int THREADS = 128>
__device__ __forceinline__ void tc_load_kv(const FmhaParams& P, const TileInfo& ti,
                                           char* dst, int kvh, int kv_base, int tid) {
  constexpr int U = D / 8;
  const int last = ti.hi - 1;
  const int32_t* tab = P.table + ti.b * P.tstride;
#ifdef FMHA_LOADER_V2
  // REJECTED by A/B dv_17485e2f3b31: 20-27% SLOWER on every case (c1 x0.74,
// c6 x0.73, c2 x0.78): per-iteration __shfl_sync serializes the cp.async
// issue chain behind the head lane's __ldg latency, destroying the free
// MLP of independent per-thread issues. ALU% was not the limiter.
// Warp-cooperative addressing: each U-lane row group's head lane resolves the
  // paged row base (1 __ldg + int64 stride chain per group instead of per
  // unit) and broadcasts it via a 64-bit shfl. NCU c1 (dv_050fe200053a):
  // ALU(int) top pipe 45.2%; per-unit page/stride chains dominated loaders.
  const int64_t page_stride = IS_V ? P.v_page_stride : P.k_page_stride;
  const int64_t slot_stride = IS_V ? P.v_slot_stride : P.k_slot_stride;
  const int64_t hbase = (int64_t)kvh * (IS_V ? P.v_head_stride : P.k_head_stride);
  const char* base = reinterpret_cast<const char*>(IS_V ? P_V : P_K);
  const int lane = tid & 31;
  const int gsrc = lane & ~(U - 1);
#pragma unroll
  for (int i = 0; i < (NT * U) / THREADS; ++i) {
    const int u = tid + i * THREADS;
    const int row = u / U, unit = u & (U - 1);
    int64_t rowb = 0;
    if ((lane & (U - 1)) == 0) {
      const int p = min(kv_base + row, last);
      const int page = __ldg(tab + (p >> P.page_log));
      rowb = (int64_t)page * page_stride
           + (int64_t)(p & P.page_mask) * slot_stride + hbase;
    }
    rowb = __shfl_sync(0xffffffffu, rowb, gsrc);
    const __nv_bfloat16* src =
        reinterpret_cast<const __nv_bfloat16*>(base + rowb * 2) + unit * 8;
    const int atom = unit >> 3, uin = unit & 7;
    cp_async16(su32(dst + atom * (NT * 128) + row * 128 + (((uin ^ row) & 7) * 16)), src);
  }
#else
#pragma unroll
  for (int i = 0; i < (NT * U) / THREADS; ++i) {
    const int u = tid + i * THREADS;
    const int row = u / U, unit = u % U;
    const int p = min(kv_base + row, last);
    const int page = __ldg(tab + (p >> P.page_log));
    const __nv_bfloat16* src =
        (IS_V ? P_V : P_K)
        + (int64_t)page * (IS_V ? P.v_page_stride : P.k_page_stride)
        + (int64_t)kvh * (IS_V ? P.v_head_stride : P.k_head_stride)
        + (int64_t)(p & P.page_mask) * (IS_V ? P.v_slot_stride : P.k_slot_stride)
        + unit * 8;
    const int atom = unit >> 3, uin = unit & 7;
    cp_async16(su32(dst + atom * (NT * 128) + row * 128 + (((uin ^ row) & 7) * 16)), src);
  }
#endif
}

// REJECTED by A/B dv_d5782a1d88d4: 8-12% SLOWER on 4-warp main (c1 x0.880,
// c4 x0.883, c6 x0.923), break-even on main8 (c7 x1.000). The phase-D fill
// puts one __ldg latency serially in the barrier path (V1's per-unit __ldgs
// pipeline/overlap); +nt*4 smem likely cost d64 its 2-blocks/SM; LDS
// competes with ldmatrix. Kept behind FMHA_LOADER_V3 as a negative result.
// Loader V3: page-resident addressing. A tiny smem array pages[NT] holds
// (page<<7)|slot per kv row of the stage being loaded, filled one stage ahead
// inside phase D's existing sync window (no new barriers; race-free: C's reads
// precede D's first sync, D's second sync publishes the fill). Per unit the
// loader then does 1 LDS + unpack instead of __ldg(tab) + full int64 chain.
// Targets NCU c1 (dv_2b67f4e06ee0 raw): ALU int pipe 51.3%, stalled_wait
// 0.984, long_scoreboard 0.890. (V2's shfl variant LOST 20-27%: it serialized
// issues; V3 keeps every thread's cp.async chain independent.)
template <int NT, int D, bool IS_V, int THREADS = 128>
__device__ __forceinline__ void tc_load_kv3(const FmhaParams& P, char* dst,
                                            const int32_t* pages, int kvh,
                                            int tid) {
  constexpr int U = D / 8;
  const int64_t pstride2 = (IS_V ? P.v_page_stride : P.k_page_stride) * 2;
  const int64_t sstride2 = (IS_V ? P.v_slot_stride : P.k_slot_stride) * 2;
  const char* base = reinterpret_cast<const char*>(IS_V ? P_V : P_K)
                   + (int64_t)kvh * (IS_V ? P.v_head_stride : P.k_head_stride) * 2;
#pragma unroll
  for (int i = 0; i < (NT * U) / THREADS; ++i) {
    const int u = tid + i * THREADS;
    const int row = u / U, unit = u & (U - 1);
    const int ps = pages[row];
    const char* src = base + (int64_t)(ps >> 7) * pstride2
                           + (int64_t)(ps & 127) * sstride2 + unit * 16;
    const int atom = unit >> 3, uin = unit & 7;
    cp_async16(su32(dst + atom * (NT * 128) + row * 128 + (((uin ^ row) & 7) * 16)),
               reinterpret_cast<const __nv_bfloat16*>(src));
  }
}

// Row-major cp.async loader (FMHA_LOADER_ROW): pay the page-table lookup and the
// 64-bit stride chain ONCE per 128B row instead of once per 16B unit. After the
// branched-mask fix this is the largest remaining instruction block on the d64
// prefill path (c4 SASS census: IMAD 11.2% + SHF 4.1% + LEA 3.9% + LOP3 1.2% +
// LDG 0.9% + LDGSTS 1.0% = 22.2% of all executed instructions, i.e. ~36% of the
// post-mask total, and that path is issue-bound at 560 of ~3300 MACs/cyc/SM).
// For NT=128/D=64 with 128 threads each thread stages exactly one row: 1 lookup
// + 8 cp.async instead of 8 lookups + 8 full stride chains.
// TPR = THREADS/NT threads cooperate on a row when THREADS > NT, so the d128
// NT64 and main8 (256-thread) configurations stay fully occupied.
template <int NT, int D, bool IS_V, int THREADS = 128>
__device__ __forceinline__ void tc_load_kv_rm(const FmhaParams& P, const TileInfo& ti,
                                              char* dst, int kvh, int kv_base, int tid) {
  constexpr int U = D / 8;                                    // 16B units per row
  constexpr int TPR = (THREADS > NT) ? (THREADS / NT) : 1;     // threads per row
  constexpr int ROWS_PER_PASS = THREADS / TPR;                 // == min(NT, THREADS)
  constexpr int PASSES = NT / ROWS_PER_PASS;
  const int row0 = (TPR == 1) ? tid : (tid / TPR);
  const int u0 = (TPR == 1) ? 0 : (tid % TPR);
  const int last = ti.hi - 1;
  const int32_t* const tab = P.table + ti.b * P.tstride;
  const int64_t page_stride = IS_V ? P.v_page_stride : P.k_page_stride;
  const int64_t slot_stride = IS_V ? P.v_slot_stride : P.k_slot_stride;
  const int64_t hbase = (int64_t)kvh * (IS_V ? P.v_head_stride : P.k_head_stride);
  const char* const gbase = reinterpret_cast<const char*>(IS_V ? P_V : P_K);
#pragma unroll
  for (int i = 0; i < PASSES; ++i) {
    const int r = row0 + i * ROWS_PER_PASS;
    const int p = min(kv_base + r, last);
    const int page = __ldg(tab + (p >> P.page_log));
    const char* src = gbase + ((int64_t)page * page_stride
                               + (int64_t)(p & P.page_mask) * slot_stride
                               + hbase) * 2;
    const unsigned rowbase = (unsigned)r * 128u;
    const int rr = r & 7;
#pragma unroll
    for (int u = u0; u < U; u += TPR) {
      const unsigned off = (unsigned)(u >> 3) * (NT * 128) + rowbase
                         + (unsigned)((((u & 7) ^ rr) & 7) * 16);
      cp_async16(su32(dst + off),
                 reinterpret_cast<const __nv_bfloat16*>(src + (unsigned)u * 16u));
    }
  }
}

#ifdef FMHA_LOADER_ROW
#define TC_LOAD_KV tc_load_kv_rm
#else
#define TC_LOAD_KV tc_load_kv
#endif

// ---------------------------------------------------------------------------
// TMA (cp.async.bulk.tensor) KV loading - FMHA_TMA builds, D>=128 NT64 configs.
// The tc K/V smem layout is already canonical atom-major SW128 (atom*NT*128B +
// row*128B + chunk^(row&7)), which is exactly what a SWIZZLE_128B TMA box of
// {64 bf16, 1 row} writes at a 1024B-aligned dst with NT%8==0. Per-row boxes
// make page-boundary straddles impossible; OOB never occurs (coords clamped
// like the cp.async loader). Completion gated by per-slot kv mbarriers with
// expect_tx (armed one stage ahead in phase D / prologue), replacing the
// cp.async commit-group waits at the SAME choreography points, so all smem
// slot-reuse guards (qk_done before K reissue, pv_done before V reissue) are
// preserved. Replaces ~240 int ops/thread/stage with ~2 TMA issues.
// ---------------------------------------------------------------------------
#ifdef FMHA_TMA
__device__ __forceinline__ void tma_load_4d(const void* map, char* dst_smem,
                                            int c0, int c1, int c2, int c3,
                                            uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3, %4, %5}], [%6];"
      :: "r"(su32(dst_smem)), "l"(reinterpret_cast<uint64_t>(map)),
         "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar)
      : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(addr), "r"(bytes) : "memory");
}

// Multi-row boxes: R = min(NT, page_size) rows per op (R divides both NT and
// page_size, and stage starts are NT-aligned, so boxes never straddle pages).
// OPS = (NT/R)*(D/64) ~ 4-8 per stage instead of NT*D/64 per-row ops.
// Row start > last clamps like the cp.async loader (duplicate rows, masked by
// rhi; boxes past a page end zero-fill via TMA OOB - never fault).
template <int NT, int D, int THREADS>
__device__ __forceinline__ void tc_tma_kv(const FmhaParams& P, char* dst,
                                          const void* map, int kvh, int kv_base,
                                          int last, const int32_t* tab, int tid,
                                          uint32_t mbar) {
  constexpr int ATOMS = D / 64;
  const int R = min(NT, 1 << P.page_log);
  const int OPS = (NT / R) * ATOMS;
  for (int o = tid; o < OPS; o += THREADS) {
    const int seg = o / ATOMS, a = o % ATOMS;
    const int p = min(kv_base + seg * R, last);
    const int page = __ldg(tab + (p >> P.page_log));
    tma_load_4d(map, dst + a * (NT * 128) + seg * R * 128,
                a * 64, p & P.page_mask, kvh, page, mbar);
  }
}

// Page-ID prefetch for the warp-specialized control loop (FMHA_TC_PAGEPRE).
// tc_tma_kv resolves each box's physical page with a dependent __ldg right on
// the issue path. The FMHA_PHASE probe (dv_de0e856f253d) measured that TMA
// phase at 1345 cyc/iter on c1 - 44% of the control loop, which paces the
// whole CTA (ctrl SUM 3058 == softmax period 2972): a single warp has nothing
// else in flight, so every table miss serializes. The page IDs depend only on
// the stage index, so tc_page_pre resolves them one FULL control iteration
// before tc_tma_kv_pre issues the boxes; the load latency lands under the
// mma-issue/drain window instead of on the critical path. Bitwise identical:
// same boxes, same coords, same order - only the table read is hoisted.
// Single pass: OPS = (NT/R)*(D/64) <= 16 < 32 lanes for NT <= 128, so lane o
// holds the page for op o. DV == D in every WS config, so one register covers
// the K and V boxes of a stage. Lanes past OPS clamp seg and load a duplicate
// (in-bounds) entry.
template <int NT, int D>
__device__ __forceinline__ int32_t tc_page_pre(const FmhaParams& P,
                                               const int32_t* tab, int kv_base,
                                               int last, int lane) {
  const int R = min(NT, 1 << P.page_log);
  const int seg = min(lane / (D / 64), NT / R - 1);
  const int p = min(kv_base + seg * R, last);
  return __ldg(tab + (p >> P.page_log));
}

template <int NT, int D>
__device__ __forceinline__ void tc_tma_kv_pre(const FmhaParams& P, char* dst,
                                              const void* map, int kvh,
                                              int kv_base, int last, int lane,
                                              int32_t page, uint32_t mbar) {
  constexpr int ATOMS = D / 64;
  const int R = min(NT, 1 << P.page_log);
  const int OPS = (NT / R) * ATOMS;
  if (lane < OPS) {
    const int seg = lane / ATOMS, a = lane % ATOMS;
    const int p = min(kv_base + seg * R, last);
    tma_load_4d(map, dst + a * (NT * 128) + seg * R * 128, a * 64,
                p & P.page_mask, kvh, page, mbar);
  }
}
#endif

#ifdef FMHA_IMMMAC
// Immediate-operand tcgen05.mma issue groups (mechanism M1 from the
// production reference PRODUCTION_REFERENCE/context_*.cu; SASS census
// dv_7f06138c3a3a: the helper-loop form costs ~10 SASS slots per
// UTCHMMA (ELECT + BRA.U.ANY retry + 4x R2UR per instruction because
// the asm sits in a lane-0 divergent region); the reference's
// full-warp form with one in-asm elect and immediate descriptors is
// ~3-4 slots. Full-warp callers ONLY (the elect is inside the asm).
// mma_qk_imm_d128: QK group, 8 UTCHMMA (kind::f16, M128 N64); a(k)=1024*(k/4)+2*(k%4), b(k)=512*(k/4)+2*(k%4).
__device__ __forceinline__ void mma_qk_imm_d128(
    uint32_t tmem_d, uint32_t a_lo, uint32_t b_lo, int accum_first) {
  asm volatile(
      "{\n\t.reg .pred ldr, p0, p1;\n\t"
      ".reg .b32 hi, alo, blo;\n\t"
      ".reg .b64 da, db;\n\t"
      "elect.sync _|ldr, 0xffffffff;\n\t"
      "setp.ne.b32 p0, %3, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 hi, 0x40004040;\n\t"
      "mov.b32 alo, %0;\n\t"
      "mov.b32 blo, %1;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p0;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 1018;\n\t"
      "add.u32 blo, blo, 506;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "}\n"
      :: "r"(a_lo), "r"(b_lo), "r"(tmem_d), "r"(accum_first),
         "n"(tc_idesc(128, 64, false)));
}
// mma_qk_imm_d64nt128: QK group, 4 UTCHMMA (kind::f16, M128 N128); a(k)=1024*(k/4)+2*(k%4), b(k)=1024*(k/4)+2*(k%4).
__device__ __forceinline__ void mma_qk_imm_d64nt128(
    uint32_t tmem_d, uint32_t a_lo, uint32_t b_lo, int accum_first) {
  asm volatile(
      "{\n\t.reg .pred ldr, p0, p1;\n\t"
      ".reg .b32 hi, alo, blo;\n\t"
      ".reg .b64 da, db;\n\t"
      "elect.sync _|ldr, 0xffffffff;\n\t"
      "setp.ne.b32 p0, %3, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 hi, 0x40004040;\n\t"
      "mov.b32 alo, %0;\n\t"
      "mov.b32 blo, %1;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p0;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "}\n"
      :: "r"(a_lo), "r"(b_lo), "r"(tmem_d), "r"(accum_first),
         "n"(tc_idesc(128, 128, false)));
}
// mma_pv_imm_nt64_dv128: PV group, 4 UTCHMMA (A from TMEM P cols +8k; B smem V +128 units/k).
__device__ __forceinline__ void mma_pv_imm_nt64_dv128(
    uint32_t tmem_d, uint32_t tmem_a, uint32_t b_lo, int accum_first) {
  asm volatile(
      "{\n\t.reg .pred ldr, p0, p1;\n\t"
      ".reg .b32 hi, ta, blo;\n\t"
      ".reg .b64 db;\n\t"
      "elect.sync _|ldr, 0xffffffff;\n\t"
      "setp.ne.b32 p0, %3, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 hi, 0x40004040;\n\t"
      "mov.b32 ta, %1;\n\t"
      "mov.b32 blo, %0;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p0;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "}\n"
      :: "r"(b_lo), "r"(tmem_a), "r"(tmem_d), "r"(accum_first),
         "n"(tc_idesc(128, 128, true)));
}
#if defined(FMHA_DH2_PEARLY) && !defined(FMHA_IMMMAC)
#error "FMHA_DH2_PEARLY requires FMHA_IMMMAC (split PV issue is imm-only)"
#endif
#if defined(FMHA_DH2_PEARLY) && defined(FMHA_DH2_ORDERB)
#error "FMHA_DH2_PEARLY and FMHA_DH2_ORDERB both restructure the DH2 loop"
#endif
#ifdef FMHA_DH2_MMA2
// DH2 v2: a SECOND MMA-lead warp (one issuer per half). launch_bounds grows
// by one warp; only the DH2 instantiation launches the extra warp, and the
// (DH == 2) term keeps every DH1 bound byte-identical (protects the MINB=2
// register caps).
#define DH2_XTRA_W 32
#else
#define DH2_XTRA_W 0
#endif
#ifdef FMHA_DH2_PEARLY
// PEARLY (M2-lite, reference split-P p_full mechanism): PV sub-step PAIR
// (2x UTCHMMA = 32 keys). Called twice per PV_h(s): early on the pre_early
// barrier (P chunk c=0 stored + this stage's lazy O rescale done) with
// accum_first = (s>0), then late on the full pre barrier (chunk c=1 stored)
// with tmem_a +16 words / b_lo +256 units and constant-true accumulate.
// Same instructions/accumulation order as the 4-step group -> bitwise.
__device__ __forceinline__ void mma_pv_imm_nt32x2_dv128(
    uint32_t tmem_d, uint32_t tmem_a, uint32_t b_lo, int accum_first) {
  asm volatile(
      "{\n\t.reg .pred ldr, p0, p1;\n\t"
      ".reg .b32 hi, ta, blo;\n\t"
      ".reg .b64 db;\n\t"
      "elect.sync _|ldr, 0xffffffff;\n\t"
      "setp.ne.b32 p0, %3, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 hi, 0x40004040;\n\t"
      "mov.b32 ta, %1;\n\t"
      "mov.b32 blo, %0;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p0;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "}\n"
      :: "r"(b_lo), "r"(tmem_a), "r"(tmem_d), "r"(accum_first),
         "n"(tc_idesc(128, 128, true)));
}
#endif
// mma_pv_imm_nt128_dv64: PV group, 8 UTCHMMA (A from TMEM P cols +8k; B smem V +128 units/k).
__device__ __forceinline__ void mma_pv_imm_nt128_dv64(
    uint32_t tmem_d, uint32_t tmem_a, uint32_t b_lo, int accum_first) {
  asm volatile(
      "{\n\t.reg .pred ldr, p0, p1;\n\t"
      ".reg .b32 hi, ta, blo;\n\t"
      ".reg .b64 db;\n\t"
      "elect.sync _|ldr, 0xffffffff;\n\t"
      "setp.ne.b32 p0, %3, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 hi, 0x40004040;\n\t"
      "mov.b32 ta, %1;\n\t"
      "mov.b32 blo, %0;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p0;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "}\n"
      :: "r"(b_lo), "r"(tmem_a), "r"(tmem_d), "r"(accum_first),
         "n"(tc_idesc(128, 64, true)));
}
#ifdef FMHA_IMM_D64
// mma_qk_imm_d64nt64: QK group for the CS-NH1-d64 NT64 tile, 4 UTCHMMA
// (K=64/16; N=64 idesc). Same instructions/descriptors/step as the helper
// loop -> bitwise.
__device__ __forceinline__ void mma_qk_imm_d64nt64(
    uint32_t tmem_d, uint32_t a_lo, uint32_t b_lo, int accum_first) {
  asm volatile(
      "{\n\t.reg .pred ldr, p0, p1;\n\t"
      ".reg .b32 hi, alo, blo;\n\t"
      ".reg .b64 da, db;\n\t"
      "elect.sync _|ldr, 0xffffffff;\n\t"
      "setp.ne.b32 p0, %3, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 hi, 0x40004040;\n\t"
      "mov.b32 alo, %0;\n\t"
      "mov.b32 blo, %1;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p0;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "}\n"
      :: "r"(a_lo), "r"(b_lo), "r"(tmem_d), "r"(accum_first),
         "n"(tc_idesc(128, 64, false)));
}
// mma_pv_imm_nt64_dv64: PV group for the CS-NH1-d64 NT64 tile, 4 UTCHMMA
// (A from TMEM P words +8/st; B smem V +128 units/st; N=64 idesc).
__device__ __forceinline__ void mma_pv_imm_nt64_dv64(
    uint32_t tmem_d, uint32_t tmem_a, uint32_t b_lo, int accum_first) {
  asm volatile(
      "{\n\t.reg .pred ldr, p0, p1;\n\t"
      ".reg .b32 hi, ta, blo;\n\t"
      ".reg .b64 db;\n\t"
      "elect.sync _|ldr, 0xffffffff;\n\t"
      "setp.ne.b32 p0, %3, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 hi, 0x40004040;\n\t"
      "mov.b32 ta, %1;\n\t"
      "mov.b32 blo, %0;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p0;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "}\n"
      :: "r"(b_lo), "r"(tmem_a), "r"(tmem_d), "r"(accum_first),
         "n"(tc_idesc(128, 64, true)));
}
#endif  // FMHA_IMM_D64
// mma_qk_imm_d256: QK group for d256 NT64, 16 UTCHMMA (K=256/16).
__device__ __forceinline__ void mma_qk_imm_d256(
    uint32_t tmem_d, uint32_t a_lo, uint32_t b_lo, int accum_first) {
  asm volatile(
      "{\n\t.reg .pred ldr, p0, p1;\n\t"
      ".reg .b32 hi, alo, blo;\n\t"
      ".reg .b64 da, db;\n\t"
      "elect.sync _|ldr, 0xffffffff;\n\t"
      "setp.ne.b32 p0, %3, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 hi, 0x40004040;\n\t"
      "mov.b32 alo, %0;\n\t"
      "mov.b32 blo, %1;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p0;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 1018;\n\t"
      "add.u32 blo, blo, 506;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 1018;\n\t"
      "add.u32 blo, blo, 506;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 1018;\n\t"
      "add.u32 blo, blo, 506;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "add.u32 alo, alo, 2;\n\t"
      "add.u32 blo, blo, 2;\n\t"
      "mov.b64 da, {alo, hi};\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %4, p1;\n\t"
      "}\n"
      :: "r"(a_lo), "r"(b_lo), "r"(tmem_d), "r"(accum_first),
         "n"(tc_idesc(128, 64, false)));
}

// mma_pv_imm_nt64_dv256: PV group for d256 NT64, 4 UTCHMMA (A from TMEM).
__device__ __forceinline__ void mma_pv_imm_nt64_dv256(
    uint32_t tmem_d, uint32_t tmem_a, uint32_t b_lo, int accum_first) {
  asm volatile(
      "{\n\t.reg .pred ldr, p0, p1;\n\t"
      ".reg .b32 hi, ta, blo;\n\t"
      ".reg .b64 db;\n\t"
      "elect.sync _|ldr, 0xffffffff;\n\t"
      "setp.ne.b32 p0, %3, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "mov.b32 hi, 0x40004040;\n\t"
      "mov.b32 ta, %1;\n\t"
      "mov.b32 blo, %0;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p0;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "add.u32 ta, ta, 8;\n\t"
      "add.u32 blo, blo, 128;\n\t"
      "mov.b64 db, {blo, hi};\n\t"
      "@ldr tcgen05.mma.cta_group::1.kind::f16 [%2], [ta], db, %4, p1;\n\t"
      "}\n"
      :: "r"(b_lo), "r"(tmem_a), "r"(tmem_d), "r"(accum_first),
         "n"(tc_idesc(128, 256, true)));
}

#endif  // FMHA_IMMMAC


// ---------------------------------------------------------------------------
// tcgen05 FA main kernel: M=128 rows, N=NT kv per stage, one CTA, 128 threads.
// TMEM: S fp32 [128 x NT] @0, P bf16x2 [128 x NT/2] @NT, O fp32 [128 x DV] @NT+NT/2.
// Pipeline: stage s+SB-1 K loaded in A, V loaded in C (after PV(s-1) retires);
// PV(s) then QK(s+1) issued in D behind cp_async_wait<2*SB-3>.
// ---------------------------------------------------------------------------
template <int D, int NT, int SB, bool TMA = false>
__global__ void __launch_bounds__(128) fmha_tc_main(
    const FmhaParams P
#ifdef FMHA_TMA
    , const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap
#endif
    ) {
  extern __shared__ __align__(1024) char smem[];
  constexpr int DV = D;
  constexpr int Q_BYTES = 128 * D * 2;
  constexpr int KV_BYTES = NT * D * 2;
  constexpr int S_COLS = NT, P_COLS = NT / 2;
  constexpr int WAITN = 2 * SB - 3;
  char* smem_q = smem;
  char* smem_k[SB];
  char* smem_v[SB];
#pragma unroll
  for (int j = 0; j < SB; ++j) {
    smem_k[j] = smem + Q_BYTES + j * 2 * KV_BYTES;
    smem_v[j] = smem_k[j] + KV_BYTES;
  }
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + Q_BYTES + SB * 2 * KV_BYTES);
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(mbar + 2);
  const uint32_t mbar_qk = su32(mbar), mbar_pv = su32(mbar + 1);

  // Tile reversal = descending-work order, only valid as a win for causal
  // (work gradient along tiles). Non-causal bimodal mixes regress badly
  // (c4 eval .2315->.3021, dv_e12772509b19): heavy-first strands a full
  // heavy block in wave 2 while light blocks idle behind it.
  // Grid is (kv head, tile, split) so blockIdx.x (fastest) walks the heads and
  // the reversed tile index walks work-descending across ALL heads at once
  // (global LPT order); see fmha_main in fmha_kernel.cu for why.
  int tile_y, split, kvh;
  grid_pos(P, kvh, tile_y, split);
  const int tile = ((((P._pad2 >> 1) & 1) && P.causal)
                        ? (P.tiles_ub - 1 - tile_y)
                        : tile_y);
  TileInfo ti = map_tile(P, tile);
  if (!ti.ok) return;
  int kv0, kv1, active;
  if (!split_range(P, ti, split, kv0, kv1, active)) return;
  // This split's logical kv range is [kv0, kv1). Stage starts may be rounded
  // DOWN below (TMA page alignment); kv0_exact keeps the logical lower bound so
  // the mask can exclude the rows that belong to the previous split.
  const int kv0_exact = kv0;
  // Bulk-tensor KV loads are available for the NT64 d128/d256 tiles and for
  // the d64 NT128 tile (one atom per row, R = min(NT, page_size) segments).
  constexpr bool KV_TMA = TMA && ((D >= 128 && NT == 64) || (D == 64 && NT == 128));
#ifdef FMHA_TMA
  // Page-align stage starts: round this split's kv0 DOWN to NT. chunk is a
  // multiple of NT, so the shift amount (kv0 & (NT-1)) == (lo & (NT-1)) is the
  // same for every split; the extra leading rows are masked out with kv0_exact
  // below, which keeps split coverage disjoint when lo is not NT-aligned
  // (window_left > 0 makes lo arbitrary, so the shift is generally non-zero).
  if (KV_TMA) kv0 &= ~(NT - 1);
#endif
  const int nstages = (kv1 - kv0 + NT - 1) / NT;
  const int tid = threadIdx.x;
#ifdef FMHA_LOADER_V3
  int32_t* const pages = reinterpret_cast<int32_t*>(smem + Q_BYTES
                                                    + SB * 2 * KV_BYTES + 128
                                                    + (D >= 128 ? 2304 : 0));
  const int32_t* const tabx = P.table + ti.b * P.tstride;
#endif
#ifdef FMHA_TMA
  constexpr bool USE_TMA = KV_TMA;
  // main (4 warps) has no rm/ls exchange buffers, so its KV barriers sit right
  // after mbar+tmem_slot inside the reserved 128B; main8's live past rm/ls.
  const uint32_t mbar_kv0 = su32(mbar) + 64;          // K barriers
  const uint32_t mbar_vv0 = mbar_kv0 + SB * 8;        // V barriers
  const int32_t* const tabt = P.table + ti.b * P.tstride;
#else
  constexpr bool USE_TMA = false;
#endif

  if (tid == 0) {
    mbar_init(mbar_qk, 1); mbar_init(mbar_pv, 1);
#ifdef FMHA_TMA
    if constexpr (USE_TMA) {
      for (int j = 0; j < SB; ++j) {
        mbar_init(mbar_kv0 + j * 8, 1);
        mbar_init(mbar_vv0 + j * 8, 1);
      }
      for (int j = 0; j < SB && j < nstages; ++j) {
        mbar_expect_tx(mbar_kv0 + j * 8, NT * D * 2);
        mbar_expect_tx(mbar_vv0 + j * 8, NT * DV * 2);
      }
    }
#endif
  }
  constexpr uint32_t TC_COLS = (S_COLS + P_COLS + DV <= 256) ? 256 : 512;
  if (tid < 32) tmem_alloc(su32(tmem_slot), TC_COLS);
  __syncthreads();
  const uint32_t tmem = *tmem_slot;
  if (tid < 32) tmem_relinquish();
  const uint32_t tmem_s = tmem;
  const uint32_t tmem_p = tmem + S_COLS;
  const uint32_t tmem_o = tmem + S_COLS + P_COLS;

  const int rg = ti.m0 + tid;
  const bool valid = tid < ti.mrows;
  const int q_idx = min(rg / P.group, ti.qlen - 1);
  const int head = kvh * P.group + (rg % P.group);
  const int vis = ti.kvlen - ti.qlen + rg / P.group;
  const int rhi = min(P.causal ? vis : ti.kvlen - 1, kv1 - 1);
  // max() with kv0_exact: rows in [aligned kv0, kv0_exact) are owned by the
  // previous split; without this they would be counted twice (window_left>0).
  const int rlo = max((P.window_left >= 0) ? max(0, vis - P.window_left) : 0,
                      kv0_exact);
  float mbase = NEG_INF, lt = 0.f;  // O/l are held in exp2(.-mbase) space

  tc_load_q<128, D>(P, ti, smem_q, kvh, tid);
#ifdef FMHA_TMA
  if constexpr (USE_TMA) {
    // Prefetch stages 0..SB-2 (matching the cp.async prologue): with SB=3 the
    // in-loop issue only covers stage s+SB-1, so stage 1 must be filled here or
    // its armed expect_tx never completes and phase D waits forever.
    const int pre = nstages < SB - 1 ? nstages : SB - 1;
    for (int j = 0; j < pre; ++j) {
      tc_tma_kv<NT, D, 128>(P, smem_k[j], &kmap, kvh, kv0 + j * NT, ti.hi - 1,
                            tabt, tid, mbar_kv0 + j * 8);
      tc_tma_kv<NT, DV, 128>(P, smem_v[j], &vmap, kvh, kv0 + j * NT, ti.hi - 1,
                             tabt, tid, mbar_vv0 + j * 8);
    }
    cp_async_commit();
    cp_async_wait<0>();
    if (nstages > 0) mbar_wait(mbar_kv0, 0);
  } else
#endif
  {
  TC_LOAD_KV<NT, D, false>(P, ti, smem_k[0], kvh, kv0, tid);
  TC_LOAD_KV<NT, D, true>(P, ti, smem_v[0], kvh, kv0, tid);
  cp_async_commit();
#pragma unroll
  for (int j = 1; j < SB - 1; ++j) {
    TC_LOAD_KV<NT, D, false>(P, ti, smem_k[j], kvh, kv0 + j * NT, tid);
    cp_async_commit();
    TC_LOAD_KV<NT, D, true>(P, ti, smem_v[j], kvh, kv0 + j * NT, tid);
    cp_async_commit();
  }
  }

  constexpr uint32_t IDESC_QK = tc_idesc(128, NT, false);
  constexpr uint32_t IDESC_PV = tc_idesc(128, DV, true);
  constexpr uint64_t A_ATOM = (128 * 128) >> 4;
  constexpr uint64_t B_ATOM = (uint64_t)NT * 8;
  const uint64_t qd0 = tc_desc(su32(smem_q), 1024, 0);
  constexpr uint32_t V_LBO = (DV == 64) ? 0 : NT * 128;

#ifdef FMHA_LOADER_V3
  if (tid < NT && SB - 1 < nstages) {
    const int pf = min(kv0 + (SB - 1) * NT + tid, ti.hi - 1);
    pages[tid] = (__ldg(tabx + (pf >> P.page_log)) << 7) | (pf & P.page_mask);
  }
#endif
  cp_async_wait<(SB >= 3 ? 2 : 0)>();
  fence_proxy_async();
  __syncthreads();
  // Whole-CTA wave dephase; no-op unless the host packed the runtime knob
  // (only packed for 2-CTA/SM tiles - see solution.py _ws_dephase).
  dephase_spin(P);
  if (tid == 0) {
    uint64_t da = qd0, db = tc_desc(su32(smem_k[0]), 1024, 0);
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);
      da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
      db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
    }
    umma_commit(mbar_qk);
  }

  for (int s = 0; s < nstages; ++s) {
    // ---- A: S read + row stats; queue K(s+SB-1) ----
    mbar_wait(mbar_qk, s & 1);
    fence_after_sync();
    float sreg[NT];
#pragma unroll
    for (int c = 0; c < NT / 32; ++c) tmem_ld32(sreg + c * 32, tmem_s + c * 32);
    tmem_wait_ld();
    const int pbase = kv0 + s * NT;
    const bool interior = (pbase >= rlo) && (pbase + NT - 1 <= rhi);
    float rmax_raw = NEG_INF;
    // The mask has to be a real BRANCH around the element loop, not an
    // if-converted predicate inside it. SASS census of this exact kernel on c4
    // (ncu job dv_256cb185c164, per-instruction counters): ISETP 15.9% +
    // FSEL 15.4% + VIADD 7.9% = 39% of all executed instructions, and only 60%
    // of thread-instruction slots were predicated ON - i.e. the interior stages
    // (the overwhelming majority) were issuing ~650 dead mask instructions per
    // thread per stage against ~128 useful MUFU. Splitting the loop keeps the
    // algebra identical: when `interior` holds the mask cannot fire, so sreg is
    // already the masked value and only the max reduction is needed.
#ifdef FMHA_MASK_PRED
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      float x = sreg[i];
      if (!interior) {
        const int c = pbase + i;
        if (c < rlo || c > rhi) x = NEG_INF;
      }
      sreg[i] = x;
      rmax_raw = fmaxf(rmax_raw, x);
    }
#else
    if (interior) {
      if constexpr (NT >= FMHA_TC_CHAIN_TREE_MIN) {
        float mx0 = NEG_INF, mx1 = NEG_INF, mx2 = NEG_INF, mx3 = NEG_INF;
#pragma unroll
        for (int i = 0; i < NT; i += 4) {
          mx0 = fmaxf(mx0, sreg[i + 0]);
          mx1 = fmaxf(mx1, sreg[i + 1]);
          mx2 = fmaxf(mx2, sreg[i + 2]);
          mx3 = fmaxf(mx3, sreg[i + 3]);
        }
        rmax_raw = fmaxf(fmaxf(mx0, mx1), fmaxf(mx2, mx3));
      } else {
#pragma unroll
        for (int i = 0; i < NT; ++i) rmax_raw = fmaxf(rmax_raw, sreg[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int c = pbase + i;
        float x = sreg[i];
        if (c < rlo || c > rhi) x = NEG_INF;
        sreg[i] = x;
        rmax_raw = fmaxf(rmax_raw, x);
      }
    }
#endif
    // Row max in scaled (base-2 logit) space; qk_scale2 > 0 so the argmax is
    // scale-invariant and one multiply suffices.
    const float mn = fmaxf(mbase, rmax_raw * P.qk_scale2);
    const float msafe = (mn == NEG_INF) ? 0.f : mn;
    // Lazy rescale: keep O/l in the old base when the row max grew by less
    // than lazy_thresh (base-2) units. Algebraically exact (P and l share the
    // base); P stays <= 2^thresh so bf16 precision is unaffected.
    const bool skip = (msafe - mbase) < P.lazy_thresh;
    const float alpha = skip ? 1.f : exp2_fast(mbase - msafe);
    const float nqk = P.qk_scale2;
    const float npb = -(skip ? mbase : msafe);
    if (!skip) { lt *= alpha; mbase = msafe; }
    if (s + SB - 1 < nstages) {
#ifdef FMHA_TMA
      if constexpr (USE_TMA) {
        tc_tma_kv<NT, D, 128>(P, smem_k[(s + SB - 1) % SB], &kmap, kvh,
                              kv0 + (s + SB - 1) * NT, ti.hi - 1, tabt, tid,
                              mbar_kv0 + ((s + SB - 1) % SB) * 8);
      } else
#endif
      {
#ifdef FMHA_LOADER_V3
        tc_load_kv3<NT, D, false>(P, smem_k[(s + SB - 1) % SB], pages, kvh, tid);
#else
        TC_LOAD_KV<NT, D, false>(P, ti, smem_k[(s + SB - 1) % SB], kvh,
                                 kv0 + (s + SB - 1) * NT, tid);
#endif
      }
    }
    cp_async_commit();
    // SB>=3: K(s+1) is complete here (wait<2> covers it): issue QK(s+1) early
    // so the tensor pipe runs during the softmax/pack phase below.
    if constexpr (SB >= 3) if (s + 1 < nstages) {
      cp_async_wait<(SB >= 3 ? 2 : 1)>();
      fence_proxy_async();
      __syncthreads();
      fence_after_sync();
      if (tid == 0) {
        const int kb = (s + 1) % SB;
        uint64_t da = qd0, dbq = tc_desc(su32(smem_k[kb]), 1024, 0);
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
          mma_ss(tmem_s, da, dbq, IDESC_QK, ks != 0);
          da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
          dbq += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
        }
        umma_commit(mbar_qk);
      }
    }
    // ---- B: P = exp2(S - m) in place; row sum ----
    // Four partial accumulators instead of one serial chain: `lsum += p` is NT
    // dependent FADDs at ~4 cycles each (NT=128 on the d64 tile -> ~512 cycles
    // per thread per stage of pure dependency latency). Reassociating the sum is
    // a rounding-level change only (fp32 against a 5e-2 relative tolerance).
    // Tree shape gated by FMHA_TC_CHAIN_TREE_MIN (see file header): c4 engine
    // A/B x0.9905 with correctness passed (dv_fef6b0596561).
    if constexpr (NT >= FMHA_TC_CHAIN_TREE_MIN) {
#ifdef FMHA_EXP_FFMA2
      const float2 nqk2 = make_float2(nqk, nqk);
      const float2 npb2 = make_float2(npb, npb);
#ifdef FMHA_PACK2_LSUM
      // PACK2 lsum (tree shape): two float2 accumulators replace the four
      // scalar ones - fadd2 halves the accumulate slots; per-lane chain
      // depth is unchanged. Pairing order -> tolerance-level rounding.
      float2 acc0 = make_float2(0.f, 0.f), acc1 = make_float2(0.f, 0.f);
#pragma unroll
      for (int i = 0; i < NT; i += 4) {
        const float2 a0 = ffma2(make_float2(sreg[i + 0], sreg[i + 1]), nqk2, npb2);
        const float2 a1 = ffma2(make_float2(sreg[i + 2], sreg[i + 3]), nqk2, npb2);
        const float p0 = exp2_fast(a0.x);
        const float p1 = exp2_fast(a0.y);
        const float p2 = exp2_fast(a1.x);
        const float p3 = exp2_fast(a1.y);
        acc0 = fadd2(acc0, make_float2(p0, p1));
        acc1 = fadd2(acc1, make_float2(p2, p3));
        sreg[i + 0] = p0; sreg[i + 1] = p1; sreg[i + 2] = p2; sreg[i + 3] = p3;
      }
      const float lsum = (acc0.x + acc0.y) + (acc1.x + acc1.y);
#else
      float ls0 = 0.f, ls1 = 0.f, ls2 = 0.f, ls3 = 0.f;
#pragma unroll
      for (int i = 0; i < NT; i += 4) {
        const float2 a0 = ffma2(make_float2(sreg[i + 0], sreg[i + 1]), nqk2, npb2);
        const float2 a1 = ffma2(make_float2(sreg[i + 2], sreg[i + 3]), nqk2, npb2);
        const float p0 = exp2_fast(a0.x);
        const float p1 = exp2_fast(a0.y);
        const float p2 = exp2_fast(a1.x);
        const float p3 = exp2_fast(a1.y);
        ls0 += p0; ls1 += p1; ls2 += p2; ls3 += p3;
        sreg[i + 0] = p0; sreg[i + 1] = p1; sreg[i + 2] = p2; sreg[i + 3] = p3;
      }
      const float lsum = (ls0 + ls1) + (ls2 + ls3);
#endif
#else
      float ls0 = 0.f, ls1 = 0.f, ls2 = 0.f, ls3 = 0.f;
#pragma unroll
      for (int i = 0; i < NT; i += 4) {
        const float p0 = exp2_fast(fmaf(sreg[i + 0], nqk, npb));
        const float p1 = exp2_fast(fmaf(sreg[i + 1], nqk, npb));
        const float p2 = exp2_fast(fmaf(sreg[i + 2], nqk, npb));
        const float p3 = exp2_fast(fmaf(sreg[i + 3], nqk, npb));
        ls0 += p0; ls1 += p1; ls2 += p2; ls3 += p3;
        sreg[i + 0] = p0; sreg[i + 1] = p1; sreg[i + 2] = p2; sreg[i + 3] = p3;
      }
      const float lsum = (ls0 + ls1) + (ls2 + ls3);
#endif
      lt += lsum;
    } else {
      float lsum = 0.f;
#ifdef FMHA_EXP_FFMA2
      const float2 nqk2 = make_float2(nqk, nqk);
      const float2 npb2 = make_float2(npb, npb);
#ifdef FMHA_PACK2_LSUM
      float2 acc0 = make_float2(0.f, 0.f), acc1 = make_float2(0.f, 0.f);
#pragma unroll
      for (int i = 0; i < NT; i += 4) {
        const float2 a0 = ffma2(make_float2(sreg[i], sreg[i + 1]), nqk2, npb2);
        const float2 a1 = ffma2(make_float2(sreg[i + 2], sreg[i + 3]), nqk2, npb2);
        const float p0 = exp2_fast(a0.x);
        const float p1 = exp2_fast(a0.y);
        const float p2 = exp2_fast(a1.x);
        const float p3 = exp2_fast(a1.y);
        acc0 = fadd2(acc0, make_float2(p0, p1));
        acc1 = fadd2(acc1, make_float2(p2, p3));
        sreg[i + 0] = p0; sreg[i + 1] = p1; sreg[i + 2] = p2; sreg[i + 3] = p3;
      }
      lsum = (acc0.x + acc0.y) + (acc1.x + acc1.y);
#else
#pragma unroll
      for (int i = 0; i < NT; i += 2) {
        const float2 a2 = ffma2(make_float2(sreg[i], sreg[i + 1]), nqk2, npb2);
        const float p0 = exp2_fast(a2.x);
        const float p1 = exp2_fast(a2.y);
        lsum += p0;
        sreg[i] = p0;
        lsum += p1;
        sreg[i + 1] = p1;
      }
#endif
#else
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const float p = exp2_fast(fmaf(sreg[i], nqk, npb));
        lsum += p;
        sreg[i] = p;
      }
#endif
      lt += lsum;
    }
    // ---- C: wait PV(s-1); O rescale; pack+store P; queue V(s+SB-1) ----
    if (s > 0) {
      mbar_wait(mbar_pv, (s - 1) & 1);
      fence_after_sync();
      if (__any_sync(0xffffffffu, alpha < 1.0f)) {
#ifdef FMHA_PACK2_MUL
        const float2 alpha2 = make_float2(alpha, alpha);
#endif
#pragma unroll
        for (int c = 0; c < DV / 32; ++c) {
          float oreg[32];
          tmem_ld32(oreg, tmem_o + c * 32);
          tmem_wait_ld();
#ifdef FMHA_PACK2_MUL
          // PACK2 rescale: 16 fmul2 replace 32 scalar FMULs (tolerance-level:
          // denormal-only divergence from the ftz scalar lowering).
#pragma unroll
          for (int i = 0; i < 32; i += 2) {
            const float2 o2 = fmul2(make_float2(oreg[i], oreg[i + 1]), alpha2);
            oreg[i] = o2.x;
            oreg[i + 1] = o2.y;
          }
#else
#pragma unroll
          for (int i = 0; i < 32; ++i) oreg[i] *= alpha;
#endif
          tmem_st32(tmem_o + c * 32, oreg);
        }
        tmem_wait_st();
      }
    }
#pragma unroll
    for (int c = 0; c < NT / 32; ++c) {
      uint32_t pw[16];
#pragma unroll
      for (int j = 0; j < 16; ++j)
        pw[j] = pack_bf16(sreg[c * 32 + 2 * j], sreg[c * 32 + 2 * j + 1]);
      tmem_st16(tmem_p + c * 16, pw);
    }
    tmem_wait_st();
    if (s + SB - 1 < nstages) {
#ifdef FMHA_TMA
      if constexpr (USE_TMA) {
        tc_tma_kv<NT, DV, 128>(P, smem_v[(s + SB - 1) % SB], &vmap, kvh,
                               kv0 + (s + SB - 1) * NT, ti.hi - 1, tabt, tid,
                               mbar_vv0 + ((s + SB - 1) % SB) * 8);
      } else
#endif
      {
#ifdef FMHA_LOADER_V3
        tc_load_kv3<NT, D, true>(P, smem_v[(s + SB - 1) % SB], pages, kvh, tid);
#else
        TC_LOAD_KV<NT, D, true>(P, ti, smem_v[(s + SB - 1) % SB], kvh,
                                kv0 + (s + SB - 1) * NT, tid);
#endif
      }
    }
    cp_async_commit();
    // ---- D: issue PV(s) then QK(s+1) ----
    fence_before_sync();
    __syncthreads();
#ifdef FMHA_TMA
    if constexpr (USE_TMA) {
      // Waits BEFORE re-arm: slot (s+SB)%SB == s%SB, whose current phase is
      // stage s (V[s] waited here; K[s] was waited in D(s-1)). expect_tx on
      // an uncompleted phase would poison it (racy hang/launch-failure).
      if (s + 1 < nstages)
        mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8, ((s + 1) / SB) & 1);
      mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);
      if (tid == 0 && s + SB < nstages) {
        mbar_expect_tx(mbar_kv0 + ((s + SB) % SB) * 8, NT * D * 2);
        mbar_expect_tx(mbar_vv0 + ((s + SB) % SB) * 8, NT * DV * 2);
      }
    } else
#endif
    {
#ifdef FMHA_LOADER_V3
    if (tid < NT && s + SB < nstages) {
      const int pf = min(kv0 + (s + SB) * NT + tid, ti.hi - 1);
      pages[tid] = (__ldg(tabx + (pf >> P.page_log)) << 7) | (pf & P.page_mask);
    }
#endif
    cp_async_wait<(SB >= 3 ? 2 : 1)>();
    }
    fence_proxy_async();
    __syncthreads();
    fence_after_sync();
    if (tid == 0) {
      uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
#pragma unroll
      for (int st = 0; st < NT / 16; ++st) {
        mma_ts(tmem_o, tmem_p + st * 8, db, IDESC_PV, (st != 0) || (s > 0));
        db += 128;
      }
      umma_commit(mbar_pv);
      if constexpr (SB < 3) {
        if (s + 1 < nstages) {
          const int kb = (s + 1) % SB;
          uint64_t da = qd0, dbq = tc_desc(su32(smem_k[kb]), 1024, 0);
#pragma unroll
          for (int ks = 0; ks < D / 16; ++ks) {
            mma_ss(tmem_s, da, dbq, IDESC_QK, ks != 0);
            da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
            dbq += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
          }
          umma_commit(mbar_qk);
        }
      }
    }
  }

  // ---- epilogue ----
  mbar_wait(mbar_pv, (nstages - 1) & 1);
  fence_after_sync();
  const int64_t Pb = ((int64_t)tile * P.kvh + kvh) * P.splits_ub + split;
  const int cu = P.cu_q[ti.b];
  const int64_t tok = cu + q_idx;
  if (P.splits_ub > 1) {
    float* po = P.part_o + (Pb * P.m_tile + tid) * DV;
    const int64_t blk = (int64_t)tile * P.kvh + kvh;
    const int Spad = (P.splits_ub + 3) & ~3;
    float* pm = P.part_ml + ((blk * P.m_tile + tid) * 2) * Spad + split;
#pragma unroll
    for (int c = 0; c < DV / 32; ++c) {
      float oreg[32];
      tmem_ld32(oreg, tmem_o + c * 32);
      tmem_wait_ld();
#ifdef FMHA_SCALAR_PO
#pragma unroll
      for (int i = 0; i < 32; ++i) po[c * 32 + i] = oreg[i];
#else
      // float4 stores: DV, c*32 and i are multiples of 4 and part_o
      // comes from a 256 B-aligned allocation, so every row offset is
      // 16 B aligned. The scalar form cost STG 5.0% of all stall
      // samples on c4 (splits=2, 64 stores per thread per tile).
#pragma unroll
      for (int i = 0; i < 32; i += 4)
        *reinterpret_cast<float4*>(po + c * 32 + i) =
            make_float4(oreg[i], oreg[i + 1], oreg[i + 2], oreg[i + 3]);
#endif
    }
    if (valid) { pm[0] = mbase; pm[Spad] = lt; }
    // PDL: let the dependent combine grid launch (visibility still gated by
    // its cudaGridDependencySynchronize on this grid's completion).
    cudaTriggerProgrammaticLaunchCompletion();
  } else {
    float lse2 = mbase + log2f(lt);
    float inv = P.v_scale / lt;
    if (P.has_sinks) {
      float os;
      lse2 = sink_adjust(lse2, P.sinks[head * P.sink_stride] * P.log2e, &os);
      inv *= os;
    }
    __nv_bfloat16* orow = P_OUT + tok * P.out_tok_stride + (int64_t)head * P.out_head_stride;
#pragma unroll
    for (int c = 0; c < DV / 32; ++c) {
      float oreg[32];
      tmem_ld32(oreg, tmem_o + c * 32);
      tmem_wait_ld();
      uint32_t packed[16];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        packed[i] = pack_bf16(oreg[2 * i] * inv, oreg[2 * i + 1] * inv);
      if (valid) {
#pragma unroll
        for (int u = 0; u < 4; ++u)
          *reinterpret_cast<uint4*>(orow + c * 32 + u * 8) =
              *reinterpret_cast<uint4*>(packed + u * 4);
      }
    }
    if (valid && P.has_lse) P.lse[tok * P.lse_tok_stride + head] = lse2;
  }
  fence_before_sync();
  __syncthreads();
  if (tid < 32) tmem_dealloc(tmem, TC_COLS);
}

template <int D, int NT, int SB, int MINB = 1, bool TMA = false, bool EARLY = false>
__global__ void __launch_bounds__(256, MINB) fmha_tc_main8(
    const FmhaParams P
#ifdef FMHA_TMA
    , const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap
#endif
    );

// Warp-specialized 8 softmax + 1 control warp variant (288 threads), defined at
// the end of this file; see the comment there for the pipeline argument.
// NH = number of softmax column quarters (NH=2: the classic 8-warp split;
// NH=4: 16 softmax warps + 1 control warp = 544 threads, HC = NT/4 columns per
// quarter - the only way to run NT=128 warp-specialized at d64, where HC=64
// would bust the register budget of a 2-CTA/SM 288-thread layout).
#ifdef FMHA_TMA
// CS (CSPLIT, join-20261025 peer arm-a mechanism, ported with attribution):
// dual control warps - a TMA-lead (boxes/arms/page prefetch) and an MMA-lead
// (waits + all tcgen05 issues/commits) - turn the serial single-ctrl chain
// (peer ex2: tma 985 + mma_issue 1346 + waits ~600 = the critical 2923 vs
// softmax 2776) into a max, making the period softmax-bound. Peer evidence:
// bitwise clean (cs1 dv_145c0b1fa0cc 14/14, cs3 dv_c61f78ba783b 17/17 incl.
// all 10 fixed cases); gated wins c1 x0.969-0.973, c9 x0.924-0.973, d2
// x0.938, t2 x0.939-0.954, sd1 x0.966-0.979; c7 x1.084 in the partial-tail
// band is excluded by the host wave gate (cs2 dv_39cc7a645a52). Dispatch
// bit26 here (peer used bit14; bit14 is FMHA_GRIDORDER in this tree).
template <int D, int NT, int SB, int MINB, bool TMA, bool QKFIRST = false,
          bool QKEARLY = false, int NH = 2, bool TL = false, bool CS = false,
          int DH = 1, bool MC = false>
__global__ void __launch_bounds__(NH * 128 * DH + (CS ? 64 : 32)
                       + ((DH == 2) ? DH2_XTRA_W : 0),
                  MINB)
fmha_tc_main8ws(const FmhaParams P, const __grid_constant__ CUtensorMap kmap,
                const __grid_constant__ CUtensorMap vmap);

// Multicast-pair prefill (defined at the end of this file): 2-CTA cluster,
// adjacent m-tiles share one TMA fetch per KV stage; per-CTA mma/softmax
// choreography identical to main8ws. Selected via _pad2 bit15 (FMHA_MC2).
template <int D, int NT, int SB, int MINB>
__global__ void __launch_bounds__(288, MINB) __cluster_dims__(1, 2, 1)
fmha_tc_main8mc(const FmhaParams P, const __grid_constant__ CUtensorMap kmap,
                const __grid_constant__ CUtensorMap vmap);

// DTA dual-group alternating prefill (defined at the end of this file):
// two 256-thread softmax groups alternate S buffers, P aliased into the
// drained S stage, QK primed two stages ahead. Selected via _pad2 bit28
// (FMHA_TC_DTA); host gate: d128 NT64 m128 WS splits==1.
extern "C" int fmha_dta_smem();
__global__ void __launch_bounds__(576, 1) fmha_tc_dta(
    const FmhaParams P, const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap);
#endif

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
// w8 selects the 8-warp (main8/main8ws) tail: their rm/ls cross-half exchange
// buffers live past the mbar/tmem_slot region, so the reservation grows by
// 2304 B. For d>=128 that was always paid; for d64 the 4-warp main<64,128,3>
// must NOT pay it (114816 -> 117120 B would drop it from 2 blocks/SM to 1:
// c4 eval .2315 -> .3013, dv_81b18d6c4d87). The d64 main8<64,128,2> pair fits
// 2 blocks/SM with the tail: 16384 + 65536 + 2304 + 512 = 84736 B, x2 = 169472
// + 2x1024 driver = 171520 <= 233472 B shared-memory configuration.
// +nt*4: FMHA_LOADER_V3 pages[NT] array ((page<<7)|slot per kv row) at the tail.
// +dh: DH2 dual-M-half (M4a) doubles the Q region (two 128-row halves, each
// with the DH1-identical inner atom layout); the DH2 mbar/rm tail (2208 B)
// still fits the existing 2688 B reservation.
extern "C" int fmha_tc_smem(int dq, int nt, int sb, int w8 = 0, int nh = 2,
                            int dh = 1) {
  return 128 * dh * dq * 2 + sb * 2 * nt * dq * 2 + 128
       + ((dq >= 128 || w8) ? (nh == 4 ? 4352 : 2304) : 0) + nt * 4;
}

template <class K>
static int tc_launch(K fn, dim3 grid, int threads, int smem, cudaStream_t st,
                     const FmhaParams& P) {
#ifdef FMHA_TMA
  const CUtensorMap z0{}, z1{};   // unused by TMA=false instantiations
  fn<<<grid, dim3(threads), smem, st>>>(P, z0, z1);
#else
  fn<<<grid, dim3(threads), smem, st>>>(P);
#endif
  return (int)cudaGetLastError();
}

#ifdef FMHA_TMA
template <class K>
static int tc_launch_tma(K fn, dim3 grid, int threads, int smem, cudaStream_t st,
                         const FmhaParams& P, const void* kmap, const void* vmap) {
  fn<<<grid, dim3(threads), smem, st>>>(P, *(const CUtensorMap*)kmap,
                                        *(const CUtensorMap*)vmap);
  return (int)cudaGetLastError();
}
// Cluster launch via runtime attribute (the DH2-MC instantiation of the
// main8ws template has no compile-time __cluster_dims__ - the same template
// also serves every non-cluster dispatch).
template <class K>
static int tc_launch_tma_mc(K fn, dim3 grid, int threads, int smem,
                            cudaStream_t st, const FmhaParams& P,
                            const void* kmap, const void* vmap) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(threads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 1;
  attrs[0].val.clusterDim.y = 2;
  attrs[0].val.clusterDim.z = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  return (int)cudaLaunchKernelEx(&cfg, fn, P, *(const CUtensorMap*)kmap,
                                 *(const CUtensorMap*)vmap);
}
#endif

extern "C" int fmha_tc_init() {
#define TC_SET(D, NT, SB)                                                                \
  {                                                                                      \
    cudaError_t e = cudaFuncSetAttribute((const void*)fmha_tc_main<D, NT, SB>,           \
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,    \
                                         fmha_tc_smem(D, NT, SB));                       \
    if (e != cudaSuccess) return (int)e;                                                 \
  }
  TC_SET(64, 128, 3)
  TC_SET(128, 64, 2)
  TC_SET(128, 128, 2)
  TC_SET(256, 64, 2)
#undef TC_SET
#define TC_SET8(D, NT, SB, MINB)                                                       \
  {                                                                                      \
    cudaError_t e = cudaFuncSetAttribute((const void*)fmha_tc_main8<D, NT, SB, MINB>,    \
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,    \
                                         fmha_tc_smem(D, NT, SB, 1));                    \
    if (e != cudaSuccess) return (int)e;                                                 \
  }
// d64 main8 needs the rm/ls tail too, and SB=2 is the only depth that keeps
// 2 blocks/SM (see fmha_tc_smem).
#define TC_SET8W(D, NT, SB, MINB)                                                      \
  {                                                                                      \
    cudaError_t e = cudaFuncSetAttribute((const void*)fmha_tc_main8<D, NT, SB, MINB>,    \
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,    \
                                         fmha_tc_smem(D, NT, SB, 1));                    \
    if (e != cudaSuccess) return (int)e;                                                 \
  }
  TC_SET8(128, 64, 2, 2)
  TC_SET8(256, 64, 2, 1)
  TC_SET8W(64, 128, 2, 2)
  TC_SET8W(64, 64, 2, 2)
#undef TC_SET8
#undef TC_SET8W
#ifdef FMHA_TMA
  // The TMA=true instantiations are distinct kernels: set the dynamic-smem
  // attribute on them too (per-instantiation), else their launches fail.
#define TC_SETT(D, NT, SB)                                                             \
  {                                                                                      \
    cudaError_t e = cudaFuncSetAttribute((const void*)fmha_tc_main<D, NT, SB, true>,     \
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,    \
                                         fmha_tc_smem(D, NT, SB));                       \
    if (e != cudaSuccess) return (int)e;                                                 \
  }
  TC_SETT(64, 128, 3)
  TC_SETT(128, 64, 2)
  TC_SETT(256, 64, 2)
#undef TC_SETT
#define TC_SET8T(D, NT, SB, MINB)                                                        \
  {                                                                                      \
    cudaError_t e = cudaFuncSetAttribute(                                                \
        (const void*)fmha_tc_main8<D, NT, SB, MINB, true>,                               \
        cudaFuncAttributeMaxDynamicSharedMemorySize, fmha_tc_smem(D, NT, SB, 1));        \
    if (e != cudaSuccess) return (int)e;                                                 \
  }
  TC_SET8T(128, 64, 2, 2)
  TC_SET8T(256, 64, 2, 1)
  TC_SET8T(64, 128, 2, 2)
  TC_SET8T(64, 64, 2, 2)
#undef TC_SET8T
#define TC_SET8TE(D, NT, SB, MINB)                                                     \
  {                                                                                      \
    cudaError_t e = cudaFuncSetAttribute(                                                \
        (const void*)fmha_tc_main8<D, NT, SB, MINB, true, true>,                         \
        cudaFuncAttributeMaxDynamicSharedMemorySize, fmha_tc_smem(D, NT, SB, 1));        \
    if (e != cudaSuccess) return (int)e;                                                 \
  }
  TC_SET8TE(128, 64, 2, 2)
  TC_SET8TE(256, 64, 2, 1)
#undef TC_SET8TE
#define TC_SET8WS(D, NT, SB, MINB)                                                       \
  {                                                                                      \
    cudaError_t e = cudaFuncSetAttribute(                                                \
        (const void*)fmha_tc_main8ws<D, NT, SB, MINB, true>,                             \
        cudaFuncAttributeMaxDynamicSharedMemorySize, fmha_tc_smem(D, NT, SB, 1));        \
    if (e != cudaSuccess) return (int)e;                                                 \
    e = cudaFuncSetAttribute(                                                            \
        (const void*)fmha_tc_main8ws<D, NT, SB, MINB, true, true>,                       \
        cudaFuncAttributeMaxDynamicSharedMemorySize, fmha_tc_smem(D, NT, SB, 1));        \
    if (e != cudaSuccess) return (int)e;                                                 \
    e = cudaFuncSetAttribute(                                                            \
        (const void*)fmha_tc_main8ws<D, NT, SB, MINB, true, false, true>,                \
        cudaFuncAttributeMaxDynamicSharedMemorySize, fmha_tc_smem(D, NT, SB, 1));        \
    if (e != cudaSuccess) return (int)e;                                                 \
  }
  TC_SET8WS(128, 64, 2, 2)
  TC_SET8WS(256, 64, 2, 1)
  // NT=128 for d128: TMEM becomes S 128 + P 2x64 + O 128 = 384 -> 512 columns,
  // i.e. 1 CTA/SM, but each stage carries twice the work per barrier/mbar round
  // trip and per TMEM latency exposure, and QK runs at the N=128 rate (3042
  // MACs/cyc/SM at 1 CTA vs 2732 at N=64 with 2 CTAs).
  TC_SET8WS(128, 128, 2, 1)
  // d64: TMEM S 64 + P 2x32 + O 64 = 192 -> 256 columns, so still 2 CTAs/SM,
  // and SB=4 keeps 84608 B (2 CTAs = 169 KB). This is the d64 configuration the
  // 4-warp main<64,128,3> cannot reach: 16 softmax warps + 2 control warps per
  // SM with a decoupled mma issue, instead of 8 warps that serialise softmax,
  // cp.async addressing and mma issue on the same 4 threads-per-scheduler.
  TC_SET8WS(64, 64, 4, 2)
#undef TC_SET8WS
  // Selective TMA_LATE instantiations (dispatch bit24, FMHA_TC_TLSEL): the
  // late-issue loop won d256 (c6) and split-d128 (c9) twice but lost
  // long-tile non-split d128 (c1) - the host picks per shape.
  {
    cudaError_t e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<128, 64, 2, 2, true, false, false, 2, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(128, 64, 2, 1));
    if (e != cudaSuccess) return (int)e;
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<256, 64, 2, 1, true, false, false, 2, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(256, 64, 2, 1));
    if (e != cudaSuccess) return (int)e;
  }
  // CSPLIT instantiation (dispatch bit26, FMHA_WS_CSPLIT; peer arm-a
  // join-20261025): dual control warps, 320 threads, SAME smem/TMEM footprint
  // as the 288-thread base - only the launch width changes.
  {
    cudaError_t e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<128, 64, 2, 2, true, false, false, 2,
                                     false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(128, 64, 2, 1));
    if (e != cudaSuccess) return (int)e;
  }
  // NH1 row-split instantiations (dispatch bit29, host knob FMHA_WS_NH1):
  // 4 softmax warps x whole rows, no row-max exchange, 160 threads (192 with
  // CSPLIT's two control warps), smaller rm tail (fmha_tc_smem nh=1; same
  // 2304B bucket, occupancy unchanged). Plain / TL / CS for d128; plain / TL
  // for d256.
  {
    cudaError_t e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<128, 64, 2, 2, true, false, false, 1>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(128, 64, 2, 1, 1));
    if (e != cudaSuccess) return (int)e;
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<128, 64, 2, 2, true, false, false, 1, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(128, 64, 2, 1, 1));
    if (e != cudaSuccess) return (int)e;
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<128, 64, 2, 2, true, false, false, 1,
                                     false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(128, 64, 2, 1, 1));
    if (e != cudaSuccess) return (int)e;
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<256, 64, 2, 1, true, false, false, 1>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(256, 64, 2, 1, 1));
    if (e != cudaSuccess) return (int)e;
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<256, 64, 2, 1, true, false, false, 1, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(256, 64, 2, 1, 1));
    if (e != cudaSuccess) return (int)e;
  }
  // DH2 dual-M-half instantiation (dispatch bit7 + bit26 + bit29 with
  // m_tile == 256; host knob FMHA_WS_DH2, mechanism M4a of the production
  // reference dual-half pipeline): CSPLIT-NH1 with TWO independent 128-row
  // attention halves per CTA sharing one K/V stream, 320 threads, TMEM 512
  // (S0|S1|P0|P1|O0|O1), ~133.7KB smem, 1 CTA/SM.
  {
    cudaError_t e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<128, 64, 2, 1, true, false, false, 1,
                                     false, true, 2>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(128, 64, 2, 1, 1, 2));
    if (e != cudaSuccess) return (int)e;
    // Deep-KV-ring variants (FMHA_DH2_SB=3/4 -> dispatch bit8/bit25):
    // SB3 = 166528 B, SB4 = 199296 B dynamic smem - both still 1 CTA/SM.
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<128, 64, 3, 1, true, false, false, 1,
                                     false, true, 2>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(128, 64, 3, 1, 1, 2));
    if (e != cudaSuccess) return (int)e;
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<128, 64, 4, 1, true, false, false, 1,
                                     false, true, 2>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(128, 64, 4, 1, 1, 2));
    if (e != cudaSuccess) return (int)e;
    // Cluster-multicast pair instantiation (FMHA_DH2_MC2, bit15 under DH2;
    // launched via tc_launch_tma_mc with a runtime (1,2,1) cluster).
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<128, 64, 2, 1, true, false, false, 1,
                                     false, true, 2, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(128, 64, 2, 1, 1, 2));
    if (e != cudaSuccess) return (int)e;
    // CSPLIT-d256-NH1 (FMHA_CS_D256, bit26 under the bit10 d256 WS gate):
    // dual control warps for the c6-class tile - the TMALATE single ctrl
    // interleaves the box issue with the service-blocked mma issue in ONE
    // warp (pipe duty ~52% at 1280-cyc service vs ~2460-cyc period); the
    // split turns the sum into a max exactly as it did for d128.
    // 192 threads, 199296 B, 1 CTA/SM.
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<256, 64, 2, 1, true, false, false, 1,
                                     false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(256, 64, 2, 1, 1));
    if (e != cudaSuccess) return (int)e;
    // CSPLIT-NH1-d64-NT64 (FMHA_CS_NH1_D64): probe run 20261027T231538Z
    // shows c0's NH4 tile at ~24% pipe duty - softmax-bound (exp_pack 682
    // from 16-warp slot contention + PAIRMAX rmx 446) against a 512-cyc
    // service. The row-split NH1 (4+4 warps, no exchange, LDRED) at NT64
    // trades pipe rate (irrelevant at 24% duty) for a ~2x shorter softmax
    // chain and 2 CTA/SM. 192 threads, 51840 B, MINB=2.
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<64, 64, 2, 2, true, false, false, 1,
                                     false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(64, 64, 2, 1, 1));
    if (e != cudaSuccess) return (int)e;
  }
  // DTA instantiation (dispatch bit28, FMHA_TC_DTA): 544 threads, 1 CTA/SM,
  // ~133KB dynamic smem (Q + 3 KV slot pairs + tail).
  {
    cudaError_t e = cudaFuncSetAttribute(
        (const void*)fmha_tc_dta, cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_dta_smem());
    if (e != cudaSuccess) return (int)e;
  }
#define TC_SET8MC(D, NT, SB, MINB)                                                   \
  {                                                                                    \
    cudaError_t e = cudaFuncSetAttribute(                                              \
        (const void*)fmha_tc_main8mc<D, NT, SB, MINB>,                                 \
        cudaFuncAttributeMaxDynamicSharedMemorySize, fmha_tc_smem(D, NT, SB, 1));      \
    if (e != cudaSuccess) return (int)e;                                               \
  }
  TC_SET8MC(128, 64, 2, 2)
  TC_SET8MC(256, 64, 2, 1)
#undef TC_SET8MC
  // NH=4 d64 NT128 (544 threads, 1 CTA/SM, TMEM 512 cols): registered in the
  // base form only - QKFIRST/QKEARLY are rejected issue orders and the NH4-TL
  // instantiation measured neutral (c0 x1.0008, S13 dv_2d4a8e207441) and was
  // dropped.
#ifndef FMHA_TC_MAXROW
  {
    cudaError_t e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<64, 128, 2, 1, true, false, false, 4>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(64, 128, 2, 1, 4));
    if (e != cudaSuccess) return (int)e;
    // CSPLIT-NH4 (peer arm-a, join-20261027 exchange): 576 threads,
    // 1 CTA/SM, same smem as the NH4 base instantiation.
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<64, 128, 2, 1, true, false, false, 4,
                                     false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(64, 128, 2, 1, 4));
    if (e != cudaSuccess) return (int)e;
    // NH1-NT128-d64 CS (FMHA_CS_NH1_D64NT128): 192 threads, 1 CTA/SM.
    e = cudaFuncSetAttribute(
        (const void*)fmha_tc_main8ws<64, 128, 2, 1, true, false, false, 1,
                                     false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        fmha_tc_smem(64, 128, 2, 1, 1));
    if (e != cudaSuccess) return (int)e;
  }
#endif
#endif
  return 0;
}

// The split-KV combine kernel lives in the mma.sync library. Registering its
// address once lets fmha_tc_run launch main+combine from ONE host call, which
// removes a ctypes round-trip (~1.5-3us) from every split tc launch - the
// evaluator times the host path inside the measured region.
typedef int (*fmha_combine_fn)(const FmhaParams*, void*, void*, void*);
static fmha_combine_fn g_combine = nullptr;
extern "C" int fmha_tc_set_combine(void* fn) {
  g_combine = reinterpret_cast<fmha_combine_fn>(fn);
  return 0;
}

extern "C" int fmha_tc_run(const FmhaParams* p, void* stream,
                           const void* kmap, const void* vmap) {
  cudaStream_t st = (cudaStream_t)stream;
  const FmhaParams& P = *p;
  if (P.splits_ub > 64 || P.splits_ub < 1) return -100;
  // Grid-order bit14: tiles on x (fastest), kvh on y - kvh-major execution
  // for L2 prefix capture under the evaluator's per-iteration eviction.
  // Tiles ride gridDim.x (2^31 cap), so no fold is needed in swap mode.
  const bool swapg = (P._pad2 & 16384) != 0;
#ifdef FMHA_TILE_FOLD
  // Compliance: tiles_ub can exceed the 65535 gridDim.y cap at the range-legal
  // max corner (batch32 x q8192 x h32/kvh1 -> 65,568 tiles). Fold tiles into
  // grid.z; kernels decode via gridDim.z > splits_ub. fold > 1 implies
  // splits_ub == 1 (a split needs base_blocks < ~2*SM, i.e. tiles_ub*kvh <=
  // 295), so gridDim.z = fold stays tiny and the combine (splits>1 only) is
  // immune. At fold == 1 the grid is bit-identical to the unfolded form.
  const unsigned fold =
      swapg ? 1u : (((unsigned)P.tiles_ub + 65534u) / 65535u);
  dim3 grid(swapg ? dim3((unsigned)P.tiles_ub, (unsigned)P.kvh,
                         (unsigned)P.splits_ub)
                  : dim3(P.kvh, ((unsigned)P.tiles_ub + fold - 1u) / fold,
                         (unsigned)P.splits_ub * fold));
#else
  dim3 grid(swapg ? dim3(P.tiles_ub, P.kvh, P.splits_ub)
                  : dim3(P.kvh, P.tiles_ub, P.splits_ub));
#endif
  // bit0 = W8 (main8, packed by python for d256 only). NOTE: must test the
  // bit, not equality: with rev(bit1) default-on, _pad2==3 for d256 and the
  // old '== 1' silently disabled main8 in every default build (v3.5/v3.6b).
  const bool w8 = (P._pad2 & 1);
  const bool early = (P._pad2 & 128);   // phase-A QK(s+1) issue (main8 + TMA)
  int rc;
#ifdef FMHA_TMA
  // bit15 = multicast-pair prefill (FMHA_MC2). Host gate guarantees
  // splits_ub == 1, window_left < 0, m_tile 128 / n_tile 64 and
  // tiles_padded <= 65535 (gridDim.y cap). Cluster (1,2,1) pairs adjacent
  // tiles on y; x stays kvh-fastest so the wave order matches the base
  // launch. Falls back per-pair inside the kernel when the tiles cross a
  // batch border or the pair is the odd slack tile.
  if (kmap && vmap && (P._pad2 & 32768) && P.dq == P.dv
      && (P.dq == 128 || P.dq == 256) && P.m_tile == 128
      && P.n_tile == 64 && P.splits_ub == 1 && P.window_left < 0) {
    dim3 mgrid(P.kvh, ((unsigned)P.tiles_ub + 1u) & ~1u, 1u);
    if (P.dq == 128)
      rc = tc_launch_tma(fmha_tc_main8mc<128, 64, 2, 2>, mgrid, 288,
                         fmha_tc_smem(128, 64, 2, 1), st, P, kmap, vmap);
    else
      rc = tc_launch_tma(fmha_tc_main8mc<256, 64, 2, 1>, mgrid, 288,
                         fmha_tc_smem(256, 64, 2, 1), st, P, kmap, vmap);
    return (rc || P.splits_ub <= 1 || !g_combine)
               ? rc
               : g_combine(p, stream, nullptr, nullptr);
  }
  // bit9 = warp-specialized 8 softmax + 1 control warp prefill (288 threads).
  // Only the d128 NT64 TMA configuration is instantiated: it is the one whose
  // TMEM budget (S 64 + P 32 + O 128 = 224 of 256 columns) still allows 2
  // CTAs/SM, and the shapes that dominate the fixed suite (c1/c7/d2/d7).
  if (kmap && vmap && (P._pad2 & 512) && P.dq == 128 && P.dv == 128
      && P.m_tile == 128 && P.n_tile == 128) {
    rc = tc_launch_tma(fmha_tc_main8ws<128, 128, 2, 1, true>, grid, 288,
                       fmha_tc_smem(128, 128, 2, 1), st, P, kmap, vmap);
    return (rc || P.splits_ub <= 1 || !g_combine)
               ? rc
               : g_combine(p, stream, nullptr, nullptr);
  }
  if (kmap && vmap && (P._pad2 & 512) && (P._pad2 & 128) && P.dq == 128
      && P.dv == 128 && P.m_tile == 256 && P.n_tile == 64) {
    // DH2 (bit7 marker + bit26 + bit29 + m_tile 256; the host packs bit7 for
    // DH2 only with tc_early off, and no other fmha_tc branch accepts
    // m_tile == 256): dual-M-half CSPLIT-NH1 prefill, mechanism M4a of the
    // production reference (two independent 128-row attention halves per CTA
    // sharing one K/V stream; per-half S/P/O TMEM regions and barrier sets,
    // per-half order = the certified base order). 320 threads, TMEM 512,
    // ~133.7KB smem, 1 CTA/SM. splits_ub == 1 is host-gated.
    // bit8 (256) / bit25: deeper KV ring (SB3 166.5KB / SB4 199.3KB, still
    // 1 CTA/SM) - FMHA_DH2_SB host knob; S/P parity stays 2-deep.
#ifdef FMHA_DH2_MMA2
    constexpr int DH2_THR = 352;   // + one per-half MMA-lead warp (v2)
#else
    constexpr int DH2_THR = 320;
#endif
    if ((P._pad2 & 32768) && P.splits_ub == 1 && P.window_left < 0) {
      // bit15 under DH2 (FMHA_DH2_MC2): cluster-multicast pairs of adjacent
      // 256-row tiles - ONE K/V fetch serves 512 rows (halves SM<-L2 KV
      // traffic on top of DH2's per-tile halving). Padded-even y grid; the
      // kernel falls back to solo per pair (batch border / odd slack tile).
      // Host gate: splits==1, no window, tiles_padded <= 65535.
      dim3 mgrid(P.kvh, ((unsigned)P.tiles_ub + 1u) & ~1u, 1u);
      rc = tc_launch_tma_mc(fmha_tc_main8ws<128, 64, 2, 1, true, false, false,
                                            1, false, true, 2, true>,
                            mgrid, DH2_THR, fmha_tc_smem(128, 64, 2, 1, 1, 2),
                            st, P, kmap, vmap);
      return rc;   // splits == 1: no combine
    }
    if (P._pad2 & (1 << 25))
      rc = tc_launch_tma(fmha_tc_main8ws<128, 64, 4, 1, true, false, false, 1,
                                         false, true, 2>,
                         grid, DH2_THR, fmha_tc_smem(128, 64, 4, 1, 1, 2), st, P,
                         kmap, vmap);
    else if (P._pad2 & 256)
      rc = tc_launch_tma(fmha_tc_main8ws<128, 64, 3, 1, true, false, false, 1,
                                         false, true, 2>,
                         grid, DH2_THR, fmha_tc_smem(128, 64, 3, 1, 1, 2), st, P,
                         kmap, vmap);
    else
      rc = tc_launch_tma(fmha_tc_main8ws<128, 64, 2, 1, true, false, false, 1,
                                         false, true, 2>,
                         grid, DH2_THR, fmha_tc_smem(128, 64, 2, 1, 1, 2), st, P,
                         kmap, vmap);
    return (rc || P.splits_ub <= 1 || !g_combine)
               ? rc
               : g_combine(p, stream, nullptr, nullptr);
  }
  if (kmap && vmap && (P._pad2 & 512) && P.dq == 128 && P.dv == 128
      && P.m_tile == 128 && P.n_tile == 64) {
    // bit24: selective TMA_LATE instantiation (host: FMHA_TC_TLSEL, only for
    // split d128 shapes where the late-issue order won; see fmha_tc_init).
    // bit11: issue QK(s+1) before PV(s) so the tensor pipe runs QK(s+1) during
    // stage s's exp2/pack instead of idling (see the QKFIRST block in the loop).
    // bit26: CSPLIT dual control warps, 320 threads (join-20261025 peer arm-a
    // mechanism; host wave gate cs_wave_ok in solution.py). Checked BEFORE
    // bit24: on the peer join fixed job CSPLIT beat this tree's TL+PRO2 on
    // split-d128 (c9 102.4 vs 106.5us).
    // bit28: DTA dual-group alternating prefill (FMHA_TC_DTA). Checked
    // FIRST: the host gate suppresses CS/TL bits when dta_ok.
    if (P._pad2 & (1 << 28)) {
      rc = tc_launch_tma(fmha_tc_dta, grid, 576, fmha_dta_smem(), st, P, kmap,
                         vmap);
      return (rc || P.splits_ub <= 1 || !g_combine)
                 ? rc
                 : g_combine(p, stream, nullptr, nullptr);
    }
    if (P._pad2 & (1 << 29)) {
      // bit29: NH1 row-split softmax (host knob FMHA_WS_NH1): 4 softmax
      // warps own whole rows (32 rows x all 64 columns, sreg[64]); the
      // cross-warp row-max exchange disappears entirely. Combines with
      // bit26 (CSPLIT: +2 ctrl warps = 192 threads; TL bit ignored exactly
      // as in the NH2 CS path) or bit24 (TMA_LATE). v1 screen: c7 x0.9881
      // stable 3/3 rounds (dv_82baea629c9d).
      if (P._pad2 & (1 << 26))
        rc = tc_launch_tma(fmha_tc_main8ws<128, 64, 2, 2, true, false, false, 1,
                                           false, true>,
                           grid, 192, fmha_tc_smem(128, 64, 2, 1, 1), st, P, kmap,
                           vmap);
      else if (P._pad2 & (1 << 24))
        rc = tc_launch_tma(fmha_tc_main8ws<128, 64, 2, 2, true, false, false, 1,
                                           true>,
                           grid, 160, fmha_tc_smem(128, 64, 2, 1, 1), st, P, kmap,
                           vmap);
      else
        rc = tc_launch_tma(fmha_tc_main8ws<128, 64, 2, 2, true, false, false, 1>,
                           grid, 160, fmha_tc_smem(128, 64, 2, 1, 1), st, P, kmap,
                           vmap);
    } else if (P._pad2 & (1 << 26))
      rc = tc_launch_tma(fmha_tc_main8ws<128, 64, 2, 2, true, false, false, 2,
                                         false, true>,
                         grid, 320, fmha_tc_smem(128, 64, 2, 1), st, P, kmap,
                         vmap);
    else if (P._pad2 & (1 << 24))
      rc = tc_launch_tma(fmha_tc_main8ws<128, 64, 2, 2, true, false, false, 2,
                                         true>,
                         grid, 288, fmha_tc_smem(128, 64, 2, 1), st, P, kmap,
                         vmap);
    else if (P._pad2 & 4096)
      rc = tc_launch_tma(fmha_tc_main8ws<128, 64, 2, 2, true, false, true>, grid,
                         288, fmha_tc_smem(128, 64, 2, 1), st, P, kmap, vmap);
    else
      rc = (P._pad2 & 2048)
               ? tc_launch_tma(fmha_tc_main8ws<128, 64, 2, 2, true, true>, grid, 288,
                               fmha_tc_smem(128, 64, 2, 1), st, P, kmap, vmap)
               : tc_launch_tma(fmha_tc_main8ws<128, 64, 2, 2, true>, grid, 288,
                               fmha_tc_smem(128, 64, 2, 1), st, P, kmap, vmap);
    return (rc || P.splits_ub <= 1 || !g_combine)
               ? rc
               : g_combine(p, stream, nullptr, nullptr);
  }
  // bit10 = the same warp specialization for d256 (S 64 + P 2x32 + O 256 = 384
  // -> TC_COLS 512, so 1 CTA/SM = 8 softmax warps/SM instead of 16: the case
  // where an idle tensor pipe is most expensive). 288 threads x 1 CTA allows
  // 227 regs against main8<256,64,2,1>'s 128.
  if (kmap && vmap && (P._pad2 & 1024) && P.dq == 256 && P.dv == 256
      && P.m_tile == 128 && P.n_tile == 64) {
    // bit24: selective TMA_LATE instantiation (won c6 x0.976/x0.985 twice).
    // bit29: NH1 row-split softmax for d256 (160 threads, 1 CTA/SM; each of
    // the 4 softmax warps owns 32 rows x all NT columns; rescale/epilogue
    // walk 8 O-column chunks per warp). TL combines as usual.
    if ((P._pad2 & (1 << 29)) && (P._pad2 & (1 << 26)))
      rc = tc_launch_tma(fmha_tc_main8ws<256, 64, 2, 1, true, false, false,
                                         1, false, true>,
                         grid, 192, fmha_tc_smem(256, 64, 2, 1, 1), st, P,
                         kmap, vmap);
    else if (P._pad2 & (1 << 29))
      rc = (P._pad2 & (1 << 24))
               ? tc_launch_tma(fmha_tc_main8ws<256, 64, 2, 1, true, false, false,
                                               1, true>,
                               grid, 160, fmha_tc_smem(256, 64, 2, 1, 1), st, P,
                               kmap, vmap)
               : tc_launch_tma(fmha_tc_main8ws<256, 64, 2, 1, true, false, false,
                                               1>,
                               grid, 160, fmha_tc_smem(256, 64, 2, 1, 1), st, P,
                               kmap, vmap);
    else if (P._pad2 & (1 << 24))
      rc = tc_launch_tma(fmha_tc_main8ws<256, 64, 2, 1, true, false, false, 2,
                                         true>,
                         grid, 288, fmha_tc_smem(256, 64, 2, 1), st, P, kmap,
                         vmap);
    else if (P._pad2 & 4096)
      rc = tc_launch_tma(fmha_tc_main8ws<256, 64, 2, 1, true, false, true>, grid,
                         288, fmha_tc_smem(256, 64, 2, 1), st, P, kmap, vmap);
    else
      rc = (P._pad2 & 2048)
               ? tc_launch_tma(fmha_tc_main8ws<256, 64, 2, 1, true, true>, grid, 288,
                               fmha_tc_smem(256, 64, 2, 1), st, P, kmap, vmap)
               : tc_launch_tma(fmha_tc_main8ws<256, 64, 2, 1, true>, grid, 288,
                               fmha_tc_smem(256, 64, 2, 1), st, P, kmap, vmap);
    return (rc || P.splits_ub <= 1 || !g_combine)
               ? rc
               : g_combine(p, stream, nullptr, nullptr);
  }
#endif
#ifdef FMHA_TMA
  if (kmap && vmap && P.dq == P.dv && (P.dq == 128 || P.dq == 256)
      && P.m_tile == 128 && P.n_tile == 64) {
    const int sm128 = fmha_tc_smem(128, 64, 2), sm256 = fmha_tc_smem(256, 64, 2);
    if (P.dq == 128) {
      if (w8)
        rc = early ? tc_launch_tma(fmha_tc_main8<128, 64, 2, 2, true, true>, grid,
                                   256, sm128, st, P, kmap, vmap)
                   : tc_launch_tma(fmha_tc_main8<128, 64, 2, 2, true>, grid, 256,
                                   sm128, st, P, kmap, vmap);
      else
        rc = tc_launch_tma(fmha_tc_main<128, 64, 2, true>, grid, 128, sm128, st, P,
                           kmap, vmap);
    } else {
      if (w8)
        rc = early ? tc_launch_tma(fmha_tc_main8<256, 64, 2, 1, true, true>, grid,
                                   256, sm256, st, P, kmap, vmap)
                   : tc_launch_tma(fmha_tc_main8<256, 64, 2, 1, true>, grid, 256,
                                   sm256, st, P, kmap, vmap);
      else
        rc = tc_launch_tma(fmha_tc_main<256, 64, 2, true>, grid, 128, sm256, st, P,
                           kmap, vmap);
    }
    return (rc || P.splits_ub <= 1 || !g_combine) ? rc
                                                  : g_combine(p, stream, nullptr, nullptr);
  }
#endif
#ifdef FMHA_TMA
  if (kmap && vmap && (P._pad2 & 512) && P.dq == 64 && P.dv == 64
      && P.m_tile == 128 && P.n_tile == 64) {
    // bit26+bit29: CSPLIT-NH1 row-split (FMHA_CS_NH1_D64, SB2, MINB2,
    // 192 threads, 2 CTA/SM); else the historical NH2 SB4 base loop.
    if ((P._pad2 & (1 << 26)) && (P._pad2 & (1 << 29)))
      rc = tc_launch_tma(fmha_tc_main8ws<64, 64, 2, 2, true, false, false, 1,
                                         false, true>,
                         grid, 192, fmha_tc_smem(64, 64, 2, 1, 1), st, P,
                         kmap, vmap);
    else
      rc = tc_launch_tma(fmha_tc_main8ws<64, 64, 4, 2, true>, grid, 288,
                         fmha_tc_smem(64, 64, 4, 1), st, P, kmap, vmap);
    return (rc || P.splits_ub <= 1 || !g_combine)
               ? rc
               : g_combine(p, stream, nullptr, nullptr);
  }
#ifndef FMHA_TC_MAXROW
  if (kmap && vmap && (P._pad2 & 8192) && P.dq == 64 && P.dv == 64
      && P.m_tile == 128 && P.n_tile == 128) {
    // bit13: NH=4 warp-specialized d64 NT128 (544 threads, 1 CTA/SM). Tensor
    // floor per stage: QK N=128 344 cyc + PV N=64 688 cyc at the 1-CTA rates
    // = 8.1 cyc/key vs 12 cyc/key for the NT=64 2-CTA layout; softmax groups
    // keep the proven HC=32 per-thread shape of the d128 NT64 WS tile.
    // NH4-TL screened (S13 dv_2d4a8e207441): c0 x1.0008 neutral despite the
    // same tma(1160)-before-mma(842) serialization TL fixed on the d256 tile
    // (c0's 16-stage tiles are too short for steady-state reorder to pay;
    // the PRO2 prologue depth already covers the cold start) - NH4 stays on
    // the base loop and the host gate never sets bit24 for it.
    // bit26: CSPLIT-NH4 dual control warps (port of peer arm-a's
    // join-20261027 exchange mechanism; their probe dv_eda5a9ac9bbe
    // measured c0-NH4 tma_issue at 1058 cyc/iter - the worst serial
    // pre-mma phase in the suite, page16 -> 16 boxes/iter - against mma
    // 846). 576 threads, TMA-lead + MMA-lead, the same base-order
    // invariants as the d128 CS. Bitwise class (issue parallelism only).
    // This tree's own S13 measurement (NH4-TL neutral, c0 x1.0008) says
    // REORDER doesn't pay on c0's short 16-stage tiles; the peer's bet is
    // that PARALLELIZING the issue warps does. FMHA_CS_NH4 gates it.
    if ((P._pad2 & (1 << 26)) && (P._pad2 & (1 << 29)))
      // FMHA_CS_NH1_D64NT128: row-split NH1 on the NT128 d64 tile (192
      // threads, TMEM S128+P128+O64 = 320 -> 512, 1 CTA/SM, sreg[128]
      // fits the 341-reg MINB1 cap): no exchange, no PAIRMAX barrier,
      // half the stages of the NT64 flip. Helper mma path (the imm d64
      // blocks are NT64-shaped; d64nt128 QK imm EXISTS - wired in the
      // CS dispatch already; PV nt128_dv64 imm also exists).
      rc = tc_launch_tma(fmha_tc_main8ws<64, 128, 2, 1, true, false, false, 1,
                                         false, true>,
                         grid, 192, fmha_tc_smem(64, 128, 2, 1, 1), st, P,
                         kmap, vmap);
    else if (P._pad2 & (1 << 26))
      rc = tc_launch_tma(fmha_tc_main8ws<64, 128, 2, 1, true, false, false, 4,
                                         false, true>,
                         grid, 576, fmha_tc_smem(64, 128, 2, 1, 4), st, P,
                         kmap, vmap);
    else
      rc = tc_launch_tma(fmha_tc_main8ws<64, 128, 2, 1, true, false, false, 4>,
                         grid, 544, fmha_tc_smem(64, 128, 2, 1, 4), st, P, kmap,
                         vmap);
    return (rc || P.splits_ub <= 1 || !g_combine)
               ? rc
               : g_combine(p, stream, nullptr, nullptr);
  }
#endif
  if (kmap && vmap && P.dq == 64 && P.dv == 64 && P.m_tile == 128 && P.n_tile == 128) {
    // d64 NT128 spends 2048 cp.async issues plus their page-table lookups per
    // stage on 128 threads; 8-16 bulk-tensor boxes replace all of it.
    if (w8) {
      // 8-warp d64 tile: the 4-warp main<64,128,3> runs only 8 warps/SM (2 CTAs)
      // and NCU shows it latency bound (c0: 1.13 active warps/scheduler, 62.7%
      // no-eligible, SM throughput 22.7%; c4: 1.66 / 53.6% / 38.4%) at 16% of
      // the measured tcgen05 ceiling for its N mix. main8 splits each S row
      // across 2 warps, so 2 CTAs give 16 warps/SM; SB=2 is what still fits
      // 2 CTAs (84736 B), the same load/compute ratio as the d128 NT64 SB2
      // tile that reaches 48% of its ceiling.
      rc = tc_launch_tma(fmha_tc_main8<64, 128, 2, 2, true>, grid, 256,
                         fmha_tc_smem(64, 128, 2, 1), st, P, kmap, vmap);
      return (rc || P.splits_ub <= 1 || !g_combine) ? rc
                                                    : g_combine(p, stream, nullptr, nullptr);
    }
    rc = tc_launch_tma(fmha_tc_main<64, 128, 3, true>, grid, 128,
                       fmha_tc_smem(64, 128, 3), st, P, kmap, vmap);
    return (rc || P.splits_ub <= 1 || !g_combine) ? rc
                                                  : g_combine(p, stream, nullptr, nullptr);
  }
#endif
  if (P.dq == 64 && P.dv == 64 && P.m_tile == 128 && P.n_tile == 64)
    rc = tc_launch(fmha_tc_main8<64, 64, 2, 2>, grid, 256,
                   fmha_tc_smem(64, 64, 2, 1), st, P);
  else if (P.dq == 64 && P.dv == 64 && P.m_tile == 128 && P.n_tile == 128)
    rc = w8 ? tc_launch(fmha_tc_main8<64, 128, 2, 2>, grid, 256,
                        fmha_tc_smem(64, 128, 2, 1), st, P)
            : tc_launch(fmha_tc_main<64, 128, 3>, grid, 128, fmha_tc_smem(64, 128, 3), st, P);
  else if (P.dq == 128 && P.dv == 128 && P.m_tile == 128 && P.n_tile == 64)
    rc = w8 ? tc_launch(fmha_tc_main8<128, 64, 2, 2>, grid, 256, fmha_tc_smem(128, 64, 2), st, P)
            : tc_launch(fmha_tc_main<128, 64, 2>, grid, 128, fmha_tc_smem(128, 64, 2), st, P);
  else if (P.dq == 128 && P.dv == 128 && P.m_tile == 128 && P.n_tile == 128)
    rc = tc_launch(fmha_tc_main<128, 128, 2>, grid, 128, fmha_tc_smem(128, 128, 2), st, P);
  else if (P.dq == 256 && P.dv == 256 && P.m_tile == 128 && P.n_tile == 64)
    rc = w8 ? tc_launch(fmha_tc_main8<256, 64, 2>, grid, 256, fmha_tc_smem(256, 64, 2), st, P)
            : tc_launch(fmha_tc_main<256, 64, 2>, grid, 128, fmha_tc_smem(256, 64, 2), st, P);
  else
    return -200;  // unsupported -> caller falls back to mma.sync path
  return (rc || P.splits_ub <= 1 || !g_combine) ? rc
                                                : g_combine(p, stream, nullptr, nullptr);
}

extern "C" int fmha_tc_params_bytes() { return (int)sizeof(FmhaParams); }

#ifdef FMHA_TMA
// Host-side tensormap encoding (driver API via dlopen; no link-flag change).
// 4D HND paged view {d, page_size, kvh, pages}; box {64,1,1,1}: per-row ops
// make page straddles impossible; SWIZZLE_128B writes exactly the canonical
// atom-major layout the tc kernels already use. Maps are encoded ONCE per
// dispatch key (kv ptrs/strides pinned by the cache) into caller-provided
// 64B-aligned 128B slots and passed by value as __grid_constant__ params.
extern "C" int fmha_encode_kv_maps(void* kmap_out, void* vmap_out,
                                   void* kptr, void* vptr,
                                   int dq, int dv, int page_size, int kvh,
                                   int nt,
                                   long long num_pages,
                                   long long slot_sk, long long slot_sv,
                                   long long head_sk, long long head_sv,
                                   long long page_sk, long long page_sv) {
  typedef CUresult (*enc_t)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                            const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                            const cuuint32_t*, CUtensorMapInterleave,
                            CUtensorMapSwizzle, CUtensorMapL2promotion,
                            CUtensorMapFloatOOBfill);
  static enc_t enc = nullptr;
  if (!enc) {
    void* h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!h) h = dlopen("libcuda.so", RTLD_LAZY);
    if (!h) return -1;
    enc = (enc_t)(uintptr_t)dlsym(h, "cuTensorMapEncodeTiled");
    if (!enc) return -2;
  }
  const cuuint64_t dimsk[4] = {(cuuint64_t)dq, (cuuint64_t)page_size,
                               (cuuint64_t)kvh, (cuuint64_t)num_pages};
  const cuuint64_t strk[3] = {(cuuint64_t)(slot_sk * 2), (cuuint64_t)(head_sk * 2),
                              (cuuint64_t)(page_sk * 2)};
  const cuuint64_t dimsv[4] = {(cuuint64_t)dv, dimsk[1], dimsk[2], dimsk[3]};
  const cuuint64_t strv[3] = {(cuuint64_t)(slot_sv * 2), (cuuint64_t)(head_sv * 2),
                              (cuuint64_t)(page_sv * 2)};
  const int boxr = nt < page_size ? nt : page_size;
  const cuuint32_t box[4] = {64, (cuuint32_t)boxr, 1, 1};
  const cuuint32_t es[4] = {1, 1, 1, 1};
  CUresult r = enc((CUtensorMap*)kmap_out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                   kptr, dimsk, strk, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) return 1000 + (int)r;
  r = enc((CUtensorMap*)vmap_out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
          vptr, dimsv, strv, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? 0 : 2000 + (int)r;
}
#endif

// ---------------------------------------------------------------------------
// 8-warp (256-thread) tcgen05 variant: warp pairs share a TMEM lane quadrant
// and split the S columns in half, halving per-lane softmax ALU/TMEM work.
// Row state (mbase, lt) is replicated in both half-warps and kept consistent
// via two small smem exchanges per stage.
// ---------------------------------------------------------------------------
// EARLY: issue QK(s+1) from phase A (right after the row-max barrier proves
// every thread has drained its S(s) LDTM) instead of from phase D behind PV(s).
// S is single-buffered, so the tile is free at that point; P lives in its own
// TMEM columns. The next stage's `mbar_wait(qk)` then finds the MMA already
// retired, taking its latency off the critical path (NCU c1: tensor pipe only
// 26.8% busy, long_scoreboard 2.37 of 7.98 cycles/instr).
template <int D, int NT, int SB, int MINB, bool TMA, bool EARLY>
__global__ void __launch_bounds__(256, MINB) fmha_tc_main8(
    const FmhaParams P
#ifdef FMHA_TMA
    , const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap
#endif
    ) {
  extern __shared__ __align__(1024) char smem[];
  constexpr int DV = D;
  constexpr int Q_BYTES = 128 * D * 2;
  constexpr int KV_BYTES = NT * D * 2;
  // main8 issues QK(s+1) either in phase D (SB < 3) or in the EARLY TMA block
  // (EARLY && USE_TMA && SB == 2). Any other combination queues QK(0) and then
  // never again, so the softmax warps wait on mbar_qk(1) forever - that is a
  // hang, not a slow path (main8<64,64,4,2> deadlocked job dv_ba52853eb8e1).
  static_assert(SB < 3 || (EARLY && TMA),
                "main8 has no QK(s+1) issue site for this SB");
  constexpr int S_COLS = NT, P_COLS = NT / 2;
  constexpr int HC = NT / 2;              // S columns per half
  constexpr uint32_t TC_COLS = (S_COLS + P_COLS + DV <= 256) ? 256 : 512;
  char* smem_q = smem;
  char* smem_k[SB];
  char* smem_v[SB];
#pragma unroll
  for (int j = 0; j < SB; ++j) {
    smem_k[j] = smem + Q_BYTES + j * 2 * KV_BYTES;
    smem_v[j] = smem_k[j] + KV_BYTES;
  }
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + Q_BYTES + SB * 2 * KV_BYTES);
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(mbar + 2);
  float* rm = reinterpret_cast<float*>(tmem_slot + 8);   // [2][128]
  float* ls = rm + 2 * 128;                              // [2][128]
  const uint32_t mbar_qk = su32(mbar), mbar_pv = su32(mbar + 1);

  // Tile reversal = descending-work order, only valid as a win for causal
  // (work gradient along tiles). Non-causal bimodal mixes regress badly
  // (c4 eval .2315->.3021, dv_e12772509b19): heavy-first strands a full
  // heavy block in wave 2 while light blocks idle behind it.
  // Grid is (kv head, tile, split) so blockIdx.x (fastest) walks the heads and
  // the reversed tile index walks work-descending across ALL heads at once
  // (global LPT order); see fmha_main in fmha_kernel.cu for why.
  int tile_y, split, kvh;
  grid_pos(P, kvh, tile_y, split);
  const int tile = ((((P._pad2 >> 1) & 1) && P.causal)
                        ? (P.tiles_ub - 1 - tile_y)
                        : tile_y);
  TileInfo ti = map_tile(P, tile);
  if (!ti.ok) return;
  int kv0, kv1, active;
  if (!split_range(P, ti, split, kv0, kv1, active)) return;
  // This split's logical kv range is [kv0, kv1). Stage starts may be rounded
  // DOWN below (TMA page alignment); kv0_exact keeps the logical lower bound so
  // the mask can exclude the rows that belong to the previous split.
  const int kv0_exact = kv0;
#ifdef FMHA_TMA
  // Page-align stage starts: round this split's kv0 DOWN to NT. chunk is a
  // multiple of NT, so the shift amount (kv0 & (NT-1)) == (lo & (NT-1)) is the
  // same for every split; the extra leading rows are masked out with kv0_exact
  // below, which keeps split coverage disjoint when lo is not NT-aligned
  // (window_left > 0 makes lo arbitrary, so the shift is generally non-zero).
  if (TMA && ((D >= 128 && NT == 64) || (D == 64 && NT == 128)))
    kv0 &= ~(NT - 1);
#endif
  const int nstages = (kv1 - kv0 + NT - 1) / NT;
  const int tid = threadIdx.x;
  const int half = tid >> 7;          // 0/1: S column half
  const int row = tid & 127;          // TMEM lane == m row
#ifdef FMHA_LOADER_V3
  // main8 always reserves the 2304 B rm/ls tail (every head dim, incl. d64).
  int32_t* const pages = reinterpret_cast<int32_t*>(smem + Q_BYTES
                                                    + SB * 2 * KV_BYTES + 128
                                                    + 2304);
  const int32_t* const tabx = P.table + ti.b * P.tstride;
#endif
#ifdef FMHA_TMA
  constexpr bool USE_TMA =
      TMA && ((D >= 128 && NT == 64) || (D == 64 && NT == 128));
  const uint32_t mbar_kv0 = su32(mbar) + 2096;        // K barriers
  const uint32_t mbar_vv0 = mbar_kv0 + SB * 8;        // V barriers
  const int32_t* const tabt = P.table + ti.b * P.tstride;
#else
  constexpr bool USE_TMA = false;
#endif

  if (tid == 0) {
    mbar_init(mbar_qk, 1); mbar_init(mbar_pv, 1);
#ifdef FMHA_TMA
    if constexpr (USE_TMA) {
      for (int j = 0; j < SB; ++j) {
        mbar_init(mbar_kv0 + j * 8, 1);
        mbar_init(mbar_vv0 + j * 8, 1);
      }
      for (int j = 0; j < SB && j < nstages; ++j) {
        mbar_expect_tx(mbar_kv0 + j * 8, NT * D * 2);
        mbar_expect_tx(mbar_vv0 + j * 8, NT * DV * 2);
      }
    }
#endif
  }
  if (tid < 32) tmem_alloc(su32(tmem_slot), TC_COLS);
  __syncthreads();
  const uint32_t tmem = *tmem_slot;
  if (tid < 32) tmem_relinquish();
  const uint32_t tmem_s = tmem;
  const uint32_t tmem_p = tmem + S_COLS;
  const uint32_t tmem_o = tmem + S_COLS + P_COLS;

  const int rg = ti.m0 + row;
  const bool valid = row < ti.mrows;
  const int q_idx = min(rg / P.group, ti.qlen - 1);
  const int head = kvh * P.group + (rg % P.group);
  const int vis = ti.kvlen - ti.qlen + rg / P.group;
  const int rhi = min(P.causal ? vis : ti.kvlen - 1, kv1 - 1);
  // max() with kv0_exact: rows in [aligned kv0, kv0_exact) are owned by the
  // previous split; without this they would be counted twice (window_left>0).
  const int rlo = max((P.window_left >= 0) ? max(0, vis - P.window_left) : 0,
                      kv0_exact);
  float mbase = NEG_INF, lt = 0.f;

  tc_load_q<128, D, 256>(P, ti, smem_q, kvh, tid);
#ifdef FMHA_TMA
  if constexpr (USE_TMA) {
    const int pre = nstages < SB - 1 ? nstages : SB - 1;   // see fmha_tc_main
    for (int j = 0; j < pre; ++j) {
      tc_tma_kv<NT, D, 256>(P, smem_k[j], &kmap, kvh, kv0 + j * NT, ti.hi - 1,
                            tabt, tid, mbar_kv0 + j * 8);
      tc_tma_kv<NT, DV, 256>(P, smem_v[j], &vmap, kvh, kv0 + j * NT, ti.hi - 1,
                             tabt, tid, mbar_vv0 + j * 8);
    }
    cp_async_commit();
    cp_async_wait<0>();
    if (nstages > 0) mbar_wait(mbar_kv0, 0);
  } else
#endif
  {
  TC_LOAD_KV<NT, D, false, 256>(P, ti, smem_k[0], kvh, kv0, tid);
  TC_LOAD_KV<NT, D, true, 256>(P, ti, smem_v[0], kvh, kv0, tid);
  cp_async_commit();
#pragma unroll
  for (int j = 1; j < SB - 1; ++j) {
    TC_LOAD_KV<NT, D, false, 256>(P, ti, smem_k[j], kvh, kv0 + j * NT, tid);
    cp_async_commit();
    TC_LOAD_KV<NT, D, true, 256>(P, ti, smem_v[j], kvh, kv0 + j * NT, tid);
    cp_async_commit();
  }
  }

  constexpr uint32_t IDESC_QK = tc_idesc(128, NT, false);
  constexpr uint32_t IDESC_PV = tc_idesc(128, DV, true);
  constexpr uint64_t A_ATOM = (128 * 128) >> 4;
  constexpr uint64_t B_ATOM = (uint64_t)NT * 8;
  const uint64_t qd0 = tc_desc(su32(smem_q), 1024, 0);
  constexpr uint32_t V_LBO = (DV == 64) ? 0 : NT * 128;

#ifdef FMHA_LOADER_V3
  if (tid < NT && SB - 1 < nstages) {
    const int pf = min(kv0 + (SB - 1) * NT + tid, ti.hi - 1);
    pages[tid] = (__ldg(tabx + (pf >> P.page_log)) << 7) | (pf & P.page_mask);
  }
#endif
  cp_async_wait<(SB >= 3 ? 2 : 0)>();
  fence_proxy_async();
  __syncthreads();
  // Whole-CTA wave dephase; no-op unless the host packed the runtime knob
  // (only packed for 2-CTA/SM tiles - see solution.py _ws_dephase).
  dephase_spin(P);
  if (tid == 0) {
    uint64_t da = qd0, db = tc_desc(su32(smem_k[0]), 1024, 0);
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);
      da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
      db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
    }
    umma_commit(mbar_qk);
  }

  for (int s = 0; s < nstages; ++s) {
    // ---- A: read own S half, per-row partial max ----
    mbar_wait(mbar_qk, s & 1);
    fence_after_sync();
    float sreg[HC];
#pragma unroll
    for (int c = 0; c < HC / 32; ++c)
      tmem_ld32(sreg + c * 32, tmem_s + half * HC + c * 32);
    tmem_wait_ld();
    const int pbase = kv0 + s * NT + half * HC;
    const bool interior = (pbase >= rlo) && (pbase + HC - 1 <= rhi);
    float rmax_raw = NEG_INF;
    // Branched, not predicated: see the same fix in fmha_tc_main (39% of the
    // executed instructions on c4 were predicated-off mask ops).
#ifdef FMHA_MASK_PRED
#pragma unroll
    for (int i = 0; i < HC; ++i) {
      float x = sreg[i];
      if (!interior) {
        const int c = pbase + i;
        if (c < rlo || c > rhi) x = NEG_INF;
      }
      sreg[i] = x;
      rmax_raw = fmaxf(rmax_raw, x);
    }
#else
    if (interior) {
#ifdef FMHA_CHAIN1
#pragma unroll
      for (int i = 0; i < HC; ++i) rmax_raw = fmaxf(rmax_raw, sreg[i]);
#else
      float mx0 = NEG_INF, mx1 = NEG_INF, mx2 = NEG_INF, mx3 = NEG_INF;
#pragma unroll
      for (int i = 0; i < HC; i += 4) {
        mx0 = fmaxf(mx0, sreg[i + 0]);
        mx1 = fmaxf(mx1, sreg[i + 1]);
        mx2 = fmaxf(mx2, sreg[i + 2]);
        mx3 = fmaxf(mx3, sreg[i + 3]);
      }
      rmax_raw = fmaxf(fmaxf(mx0, mx1), fmaxf(mx2, mx3));
#endif
    } else {
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        const int c = pbase + i;
        float x = sreg[i];
        if (c < rlo || c > rhi) x = NEG_INF;
        sreg[i] = x;
        rmax_raw = fmaxf(rmax_raw, x);
      }
    }
#endif
    if (s + SB - 1 < nstages) {
#ifdef FMHA_TMA
      if constexpr (USE_TMA) {
        tc_tma_kv<NT, D, 256>(P, smem_k[(s + SB - 1) % SB], &kmap, kvh,
                              kv0 + (s + SB - 1) * NT, ti.hi - 1, tabt, tid,
                              mbar_kv0 + ((s + SB - 1) % SB) * 8);
      } else
#endif
      {
#ifdef FMHA_LOADER_V3
        tc_load_kv3<NT, D, false, 256>(P, smem_k[(s + SB - 1) % SB], pages, kvh, tid);
#else
        TC_LOAD_KV<NT, D, false, 256>(P, ti, smem_k[(s + SB - 1) % SB], kvh,
                                      kv0 + (s + SB - 1) * NT, tid);
#endif
      }
    }
    cp_async_commit();
    rm[half * 128 + row] = rmax_raw;
    __syncthreads();
#ifdef FMHA_TMA
    if constexpr (EARLY && USE_TMA && SB == 2) {
      // Every thread has finished its S(s) LDTM (tmem_wait_ld + the barrier
      // above), so the single S tile is free for QK(s+1). K(s+1) was just
      // handed to the TMA engine in this same phase; only the issuing thread
      // waits on its barrier, the other 255 threads continue into exp/pack.
      if (s + 1 < nstages && tid == 0) {
        mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8, ((s + 1) / SB) & 1);
        fence_after_sync();
        const int kb = (s + 1) % SB;
        uint64_t da = qd0, dbq = tc_desc(su32(smem_k[kb]), 1024, 0);
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
          mma_ss(tmem_s, da, dbq, IDESC_QK, ks != 0);
          da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
          dbq += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
        }
        umma_commit(mbar_qk);
      }
    }
#endif
    const float rmax = fmaxf(rm[row], rm[128 + row]);
    const float mn = fmaxf(mbase, rmax * P.qk_scale2);
    const float msafe = (mn == NEG_INF) ? 0.f : mn;
    const bool skip = (msafe - mbase) < P.lazy_thresh;
    const float alpha = skip ? 1.f : exp2_fast(mbase - msafe);
    const float nqk = P.qk_scale2;
    const float npb = -(skip ? mbase : msafe);
    if (!skip) { lt *= alpha; mbase = msafe; }
    // ---- B: exp2 on own half; P pack+store; partial row sum ----
#ifdef FMHA_CHAIN1
    float lsum = 0.f;
#ifdef FMHA_EXP_FFMA2
    const float2 nqk2 = make_float2(nqk, nqk);
    const float2 npb2 = make_float2(npb, npb);
#pragma unroll
    for (int i = 0; i < HC; i += 2) {
      const float2 a2 = ffma2(make_float2(sreg[i], sreg[i + 1]), nqk2, npb2);
      const float p0 = exp2_fast(a2.x);
      const float p1 = exp2_fast(a2.y);
      lsum += p0;
      sreg[i] = p0;
      lsum += p1;
      sreg[i + 1] = p1;
    }
#else
#pragma unroll
    for (int i = 0; i < HC; ++i) {
      const float p = exp2_fast(fmaf(sreg[i], nqk, npb));
      lsum += p;
      sreg[i] = p;
    }
#endif
#else
    float ls0 = 0.f, ls1 = 0.f, ls2 = 0.f, ls3 = 0.f;
#pragma unroll
    for (int i = 0; i < HC; i += 4) {
      const float p0 = exp2_fast(fmaf(sreg[i + 0], nqk, npb));
      const float p1 = exp2_fast(fmaf(sreg[i + 1], nqk, npb));
      const float p2 = exp2_fast(fmaf(sreg[i + 2], nqk, npb));
      const float p3 = exp2_fast(fmaf(sreg[i + 3], nqk, npb));
      ls0 += p0; ls1 += p1; ls2 += p2; ls3 += p3;
      sreg[i + 0] = p0; sreg[i + 1] = p1; sreg[i + 2] = p2; sreg[i + 3] = p3;
    }
    const float lsum = (ls0 + ls1) + (ls2 + ls3);
#endif
    ls[half * 128 + row] = lsum;
#pragma unroll
    for (int c = 0; c < HC / 32; ++c) {
      uint32_t pw[16];
#pragma unroll
      for (int j = 0; j < 16; ++j)
        pw[j] = pack_bf16(sreg[c * 32 + 2 * j], sreg[c * 32 + 2 * j + 1]);
      tmem_st16(tmem_p + half * (HC / 2) + c * 16, pw);
    }
    tmem_wait_st();
    // ---- C: wait PV(s-1); O rescale on own (rows x dv-half); V load ----
    if (s > 0) {
      mbar_wait(mbar_pv, (s - 1) & 1);
      fence_after_sync();
      if (__any_sync(0xffffffffu, alpha < 1.0f)) {
#pragma unroll
        for (int c = 0; c < DV / 64; ++c) {
          float oreg[32];
          const uint32_t oa = tmem_o + half * (DV / 2) + c * 32;
          tmem_ld32(oreg, oa);
          tmem_wait_ld();
#pragma unroll
          for (int i = 0; i < 32; ++i) oreg[i] *= alpha;
          tmem_st32(oa, oreg);
        }
        tmem_wait_st();
      }
    }
    if (s + SB - 1 < nstages) {
#ifdef FMHA_TMA
      if constexpr (USE_TMA) {
        tc_tma_kv<NT, DV, 256>(P, smem_v[(s + SB - 1) % SB], &vmap, kvh,
                               kv0 + (s + SB - 1) * NT, ti.hi - 1, tabt, tid,
                               mbar_vv0 + ((s + SB - 1) % SB) * 8);
      } else
#endif
      {
#ifdef FMHA_LOADER_V3
        tc_load_kv3<NT, D, true, 256>(P, smem_v[(s + SB - 1) % SB], pages, kvh, tid);
#else
        TC_LOAD_KV<NT, D, true, 256>(P, ti, smem_v[(s + SB - 1) % SB], kvh,
                                     kv0 + (s + SB - 1) * NT, tid);
#endif
      }
    }
    cp_async_commit();
    // ---- D: issue PV(s) (and QK(s+1) for SB<3) ----
    fence_before_sync();
    __syncthreads();
    lt += ls[row] + ls[128 + row];   // after barrier: cross-half lsum visible
#ifdef FMHA_TMA
    if constexpr (USE_TMA) {
      // Waits BEFORE re-arm: slot (s+SB)%SB == s%SB, whose current phase is
      // stage s (V[s] waited here; K[s] was waited in D(s-1), or in phase A of
      // iteration s-1 when EARLY issues QK there). expect_tx on an uncompleted
      // phase would poison it (racy hang/launch-failure).
      if (!(EARLY && USE_TMA && SB == 2) && s + 1 < nstages)
        mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8, ((s + 1) / SB) & 1);
      mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);
      if (tid == 0 && s + SB < nstages) {
        mbar_expect_tx(mbar_kv0 + ((s + SB) % SB) * 8, NT * D * 2);
        mbar_expect_tx(mbar_vv0 + ((s + SB) % SB) * 8, NT * DV * 2);
      }
    } else
#endif
    {
#ifdef FMHA_LOADER_V3
    if (tid < NT && s + SB < nstages) {
      const int pf = min(kv0 + (s + SB) * NT + tid, ti.hi - 1);
      pages[tid] = (__ldg(tabx + (pf >> P.page_log)) << 7) | (pf & P.page_mask);
    }
#endif
    cp_async_wait<(SB >= 3 ? 2 : 1)>();
    }
    fence_proxy_async();
    __syncthreads();
    fence_after_sync();
    if (tid == 0) {
      uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
#pragma unroll
      for (int st = 0; st < NT / 16; ++st) {
        mma_ts(tmem_o, tmem_p + st * 8, db, IDESC_PV, (st != 0) || (s > 0));
        db += 128;
      }
      umma_commit(mbar_pv);
      // SB<3 without EARLY: QK(s+1) can only be issued here (S is still being
      // read by the softmax above until the phase-A barrier of the next round).
      if constexpr (SB < 3) {
        if (!(EARLY && USE_TMA) && s + 1 < nstages) {
          const int kb = (s + 1) % SB;
          uint64_t da = qd0, dbq = tc_desc(su32(smem_k[kb]), 1024, 0);
#pragma unroll
          for (int ks = 0; ks < D / 16; ++ks) {
            mma_ss(tmem_s, da, dbq, IDESC_QK, ks != 0);
            da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
            dbq += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
          }
          umma_commit(mbar_qk);
        }
      }
    }
  }

  // ---- epilogue (warps 0-3 only; they cover all 128 rows) ----
  mbar_wait(mbar_pv, (nstages - 1) & 1);
  fence_after_sync();
  if (half == 0) {
    const int64_t Pb = ((int64_t)tile * P.kvh + kvh) * P.splits_ub + split;
    const int cu = P.cu_q[ti.b];
    const int64_t tok = cu + q_idx;
    if (P.splits_ub > 1) {
      float* po = P.part_o + (Pb * P.m_tile + row) * DV;
      const int64_t blk = (int64_t)tile * P.kvh + kvh;
      const int Spad = (P.splits_ub + 3) & ~3;
      float* pm = P.part_ml + ((blk * P.m_tile + row) * 2) * Spad + split;
#pragma unroll
      for (int c = 0; c < DV / 32; ++c) {
        float oreg[32];
        tmem_ld32(oreg, tmem_o + c * 32);
        tmem_wait_ld();
#ifdef FMHA_SCALAR_PO
#pragma unroll
        for (int i = 0; i < 32; ++i) po[c * 32 + i] = oreg[i];
#else
        // float4 stores: DV, c*32 and i are multiples of 4 and part_o
        // comes from a 256 B-aligned allocation, so every row offset is
        // 16 B aligned. The scalar form cost STG 5.0% of all stall
        // samples on c4 (splits=2, 64 stores per thread per tile).
#pragma unroll
        for (int i = 0; i < 32; i += 4)
          *reinterpret_cast<float4*>(po + c * 32 + i) =
              make_float4(oreg[i], oreg[i + 1], oreg[i + 2], oreg[i + 3]);
#endif
      }
      if (valid) { pm[0] = mbase; pm[Spad] = lt; }
      cudaTriggerProgrammaticLaunchCompletion();
    } else {
      float lse2 = mbase + log2f(lt);
      float inv = P.v_scale / lt;
      if (P.has_sinks) {
        float os;
        lse2 = sink_adjust(lse2, P.sinks[head * P.sink_stride] * P.log2e, &os);
        inv *= os;
      }
      __nv_bfloat16* orow = P_OUT + tok * P.out_tok_stride + (int64_t)head * P.out_head_stride;
#pragma unroll
      for (int c = 0; c < DV / 32; ++c) {
        float oreg[32];
        tmem_ld32(oreg, tmem_o + c * 32);
        tmem_wait_ld();
        uint32_t packed[16];
#pragma unroll
        for (int i = 0; i < 16; ++i)
          packed[i] = pack_bf16(oreg[2 * i] * inv, oreg[2 * i + 1] * inv);
        if (valid) {
#pragma unroll
          for (int u = 0; u < 4; ++u)
            *reinterpret_cast<uint4*>(orow + c * 32 + u * 8) =
                *reinterpret_cast<uint4*>(packed + u * 4);
        }
      }
      if (valid && P.has_lse) P.lse[tok * P.lse_tok_stride + head] = lse2;
    }
  }
  fence_before_sync();
  __syncthreads();
  if (tid < 32) tmem_dealloc(tmem, TC_COLS);
}

#ifdef FMHA_PHASE
// ---------------------------------------------------------------------------
// DIAGNOSTIC ONLY (never in shipped defaults: solution.py adds -DFMHA_PHASE
// only when FMHA_PHASE=1 is in the environment). fmha_tc_main8ws accumulates
// per-phase SM-cycle totals (clock64 deltas, lane 0 of each warp folds them in
// once at kernel exit). experiments/phase_probe.py reads/resets the slots via
// the two extern "C" entry points below.
// Slots: 0 qkwait, 1 S read, 2 rowmax+exchange, 3 exp2+pack+P store,
// 4 rescale+pready, 5 softmax stage samples, 6 softmax warp count,
// 7 ctrl PV wait, 8 ctrl TMA issue/arm, 9 ctrl pready wait, 10 ctrl V wait,
// 11 ctrl sfree wait, 12 ctrl K wait, 13 ctrl mma issue, 14 ctrl iters,
// 15 ctrl block count.
// CSPLIT (slots 16-28, peer arm-a layout): 16 MMA-lead pre wait, 17 MMA-lead
// vv wait, 18 MMA-lead PV issue, 19 MMA-lead sfree wait, 20 MMA-lead kv wait,
// 21 MMA-lead QK issue, 22 TMA-lead pv wait, 23 TMA-lead box issue,
// 24 TMA-lead landing/arm, 26 MMA-lead iters, 27 TMA-lead iters,
// 28 CSPLIT block count.
__device__ unsigned long long g_ph[32];
extern "C" void fmha_phase_read(unsigned long long* out) {
  cudaMemcpyFromSymbol(out, g_ph, 32 * sizeof(unsigned long long));
}
extern "C" void fmha_phase_reset() {
  const unsigned long long z[32] = {0};
  cudaMemcpyToSymbol(g_ph, z, sizeof(z));
}
#define PH_TS(name) const long long name = clock64();
#else
#define PH_TS(name)
#endif
// ---------------------------------------------------------------------------
// Warp-specialized 8+1 prefill (WS). Warps 0-7 run the softmax pipeline; warp 8
// is a control warp that owns every TMA issue, every KV mbarrier wait and every
// tcgen05.mma issue.
// Why: NCU on c1 shows the tensor pipe only 26.8% busy with issue 0.41-0.47 per
// scheduler and stalls long_scoreboard 2.37 + wait 1.23 + barrier 1.16 of 7.98
// warp-cycles/instruction. In main8 the mma-issuing thread IS a softmax thread,
// so the pipe sits idle for the whole exp2/pack phase, all 256 threads spin on
// mbar_qk/mbar_pv/mbar_kv/mbar_vv, and the CTA meets at three barriers per
// stage. Here the control warp waits alone, issues QK(s+1) the moment the
// softmax warps have drained S(s) (mbar_sfree) and PV(s) the moment P(s) is
// stored and O(s) rescaled (mbar_pready), so QK(s+1) executes *during* stage s's
// softmax. Per-stage arithmetic (c1 shape, 2 CTAs/SM): tensor floor ~1460
// cycles/CTA-stage, exp2 ~1024 (8192 elements at 16 SFU/cycle/SM), TMEM ld+st
// ~400 -> overlapping them is worth up to ~2x over the measured ~4300.
// Two further differences fall out of the split:
//  - The cross-half row-SUM exchange disappears. Both halves read the same
//    combined row max, so they share one mbase/alpha sequence and each can
//    accumulate its own lt; the two are added once in the epilogue. That drops
//    one CTA barrier per stage. The remaining row-MAX exchange uses named
//    barrier 1 over the 256 softmax threads, so the control warp is never part
//    of it (algebra is unchanged: lt_total follows the same recurrence
//    lt = lt*alpha + sum_h lsum_h, only the addition order differs).
//  - The O-rescale wait on mbar_pv moves inside the `alpha < 1` branch, so with
//    the lazy threshold the softmax warps skip it entirely instead of spinning
//    on the previous PV every stage. Still safe: tcgen05 mmas of one CTA run in
//    issue order and pready(s) arrives after the rescale, so PV(s) accumulates
//    into an already-rescaled O.
// TMEM: S 64 + P 2x32 + O 128 = 256 columns, exactly the TC_COLS this config
// already allocates, so still 2 CTAs/SM. The SECOND P buffer is what makes the
// issue order QK(s+1) -> PV(s) legal: tcgen05.commit is cumulative, so the
// mbar_qk commit that releases the softmax warps into stage s+1 covers every
// mma issued before QK(s+1) - which is PV(s-1), not PV(s). With one P buffer
// the warps would overwrite P(s) while PV(s) is still reading it (measured:
// job19 ws arm, c7 48295 mismatching elements, b16 1366, b18 448, all with
// rel_l2 ~0 - a few rows of a few stages). With parity buffers, the write of
// P[(s+2)&1] == P[s&1] happens after mbar_qk(s+2), whose commit does cover
// PV(s), and PV(s+1) reads P[(s+1)&1] written under pready(s+1).
// sfree/pready are PAIRS of mbarriers indexed by stage parity, not one barrier
// with alternating phases. That is load-bearing: mbarrier arrivals are not
// tagged by phase, so with a single barrier a warp that is one stage ahead
// contributes its arrival to the phase the laggards have not finished, and the
// control warp would release PV(s) while a slow warp was still storing P(s).
// Warp skew is bounded to one stage by sfree (a warp cannot reach stage s+1
// before mbar_qk(s+1), which the control warp issues only after sfree[s&1]
// completed), but pready has no such self-limiting property because QK(s+1) is
// issued *before* PV(s). Measured exactly that failure (job20/21, ws arm):
// nondeterministic, out-only (lse clean, so the softmax statistics were right),
// c7 mm=127..1807 max_abs 0.23-0.80, b18 mm=111..3074, and FMHA_LAZY_THRESH=0
// masked it because rescaling every stage re-introduces an mbar_pv wait per
// stage, which bounds the skew by hand.
// The four barriers sit past the KV barriers in the unused part of the 2688 B
// smem tail, so no kernel's smem size (or occupancy) changes.
// 288 threads x 2 CTAs/SM needs <= 113 regs; main8<128,64,2,2> compiles at 106.
// ---------------------------------------------------------------------------
// Cluster-multicast helpers (defined with fmha_tc_main8mc at the end of the
// file; forward-declared for main8ws's MC instantiation).
__device__ __forceinline__ uint32_t mapa_u32(uint32_t local, uint32_t rank);
__device__ __forceinline__ uint32_t cluster_rank();
__device__ __forceinline__ void mbar_arrive_remote(uint32_t addr);
__device__ __forceinline__ void fence_proxy_cluster();
__device__ __forceinline__ void mbar_init_fence();
__device__ __forceinline__ void cluster_sync_mc();
template <int NT, int D>
__device__ __forceinline__ void tc_tma_kv_mc(const FmhaParams& P, char* dst,
                                             const void* map, int kvh,
                                             int kv_base, int last, int lane,
                                             int32_t page, uint32_t mbar,
                                             uint16_t mask);
#ifdef FMHA_TMA
template <int D, int NT, int SB, int MINB, bool TMA, bool QKFIRST, bool QKEARLY,
          int NH, bool TL, bool CS, int DH, bool MC>
__global__ void __launch_bounds__(NH * 128 * DH + (CS ? 64 : 32)
                       + ((DH == 2) ? DH2_XTRA_W : 0),
                  MINB)
fmha_tc_main8ws(const FmhaParams P, const __grid_constant__ CUtensorMap kmap,
                const __grid_constant__ CUtensorMap vmap) {
  static_assert(!(QKFIRST && QKEARLY), "pick one QK issue order");
  // DH2 (mechanism M4a of the production reference context_*.cu dual-M-half
  // pipeline): two independent M128 attention halves in ONE CTA sharing
  // the K/V stream - two softmax chains overlap the ctrl issue chain (the
  // duty census's chain-bound fix) and KV box work per MAC halves. v1:
  // CS + NH1 + d128 NT64 SB2, MINB=1, TMEM 512 (S0|S1|P0|P1|O0|O1).
  static_assert(DH == 1 || (DH == 2 && TMA && CS && NH == 1 && D == 128 &&
                            NT == 64 && (SB == 2 || SB == 3 || SB == 4) &&
                            !QKFIRST && !QKEARLY && !TL),
                "DH2 v1: CSPLIT NH1 d128 NT64 SB2-4 only");
#ifdef FMHA_PGDEEP
  static_assert(DH == 1, "PGDEEP ring words live past the DH1 tail layout");
#endif
  // MC (cluster multicast pair) rides the DH2 config only: adjacent 256-row
  // tiles form the cluster pair; main8mc keeps serving the DH1 shapes.
  static_assert(!MC || (DH == 2 && CS && NH == 1 && D == 128 && NT == 64 &&
                        SB == 2 && TMA && !QKFIRST && !QKEARLY && !TL),
                "MC v1: DH2 CSPLIT-NH1 d128 NT64 SB2 only");
  static_assert(!CS || (TMA && !QKFIRST && !QKEARLY && !TL &&
                        (SB == 2 || (DH == 2 && (SB == 3 || SB == 4))) &&
                        ((D == 128 && NT == 64 && (NH == 2 || NH == 1)) ||
                         (D == 64 && NT == 128 && (NH == 4 || NH == 1)) ||
                         (D == 64 && NT == 64 && NH == 1) ||
                         (D == 256 && NT == 64 && NH == 1))),
                "CSPLIT v1 (peer arm-a): base-order WS d128 NT64 SB2, NH2 or "
                "the NH1 row-split softmax; + CSPLIT-NH4 (peer arm-a, "
                "join-20261027 exchange): d64 NT128 NH4 dual-ctrl, 576 thr; "
                "+ DH2 deep KV ring SB3/SB4 (FMHA_DH2_SB); + CSPLIT-d256-NH1 "
                "(own, probe-driven: c6 issuer/box split)");
  static_assert(TMA && ((D >= 128 && NT == 64 && NH == 1) ||
                        (D >= 128 && (NT == 64 || NT == 128) && NH == 2) ||
                        (D == 64 && ((NT == 64 && (NH == 2 || NH == 1)) ||
                                     (NT == 128 && (NH == 4 || NH == 1))))),
                "ws variant is TMA-only; NH=4 exists for the d64 NT128 tile; "
                "NH=1 is the row-split softmax variant (d>=128 NT64)");
#ifdef FMHA_TC_MAXROW
  static_assert(NH == 2, "FMHA_TC_MAXROW reads the sibling HALF (NH=2 only)");
#endif
  // NH softmax column groups ("quarters" when NH=4): NH*128 softmax threads +
  // one control warp. NH=2 is the classic 288-thread layout, token for token.
  constexpr int SW = NH * 128;              // softmax threads PER HALF
  extern __shared__ __align__(1024) char smem[];
  constexpr int DV = D;
  constexpr int Q_BYTES = 128 * DH * D * 2;
  constexpr int KV_BYTES = NT * D * 2;
  constexpr int S_COLS = NT, P_COLS = NT;   // two P stage buffers (parity)
  constexpr int PH = NT / 2;                // columns per P buffer
  constexpr int HC = NT / NH;               // S columns per softmax group
  constexpr uint32_t TC_COLS =
      (DH == 2) ? 512u : ((S_COLS + P_COLS + DV <= 256) ? 256u : 512u);
  char* smem_q = smem;
  char* smem_k[SB];
  char* smem_v[SB];
#pragma unroll
  for (int j = 0; j < SB; ++j) {
    smem_k[j] = smem + Q_BYTES + j * 2 * KV_BYTES;
    smem_v[j] = smem_k[j] + KV_BYTES;
  }
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + Q_BYTES + SB * 2 * KV_BYTES);
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(mbar + 2 * DH);
  // Row-max exchange, DOUBLE BUFFERED by stage parity: with the phase-D CTA
  // barrier gone, the only thing between stage s's read of rm and stage s+2's
  // write is stage s+1's named barrier, so a single buffer would let a fast warp
  // overwrite it (sporadic wrong rows). Parity makes s+1 write the other half,
  // and bar_softmax(s+1) proves every warp finished reading stage s.
  // rm[0..NH*128) = even stages, [NH*128..2*NH*128) = odd stages; the first
  // (NH-1)*128 floats of the odd block are reused for the one-shot epilogue lt
  // exchange (post-loop, so no overlap in time).
  float* rm = reinterpret_cast<float*>(tmem_slot + 8);   // [2][DH][NH][128]
  // DH-general barrier map (DH==1 reproduces the legacy offsets exactly):
  // qk_h at +8h, pv_h at +8*DH + 8h, tmem_slot 32B, rm NH*DH*1024B.
  const uint32_t mbar_qk = su32(mbar);
  const uint32_t mbar_pv = su32(mbar) + DH * 8;
  const uint32_t mbar_kv0 = su32(mbar) + 16 * DH + 32 + NH * DH * 1024;
  const uint32_t mbar_vv0 = mbar_kv0 + SB * 8;
  // Past the KV barriers: fmha_tc_smem(128,64,2) gives a 2688 B tail and the
  // layout above ends at 2128, so these two are free without touching smem size
  // (and therefore without touching occupancy of any other kernel).
  // sfree/pre: DH*2 barriers each, indexed (dhalf*2 + parity)*8.
  const uint32_t mbar_sfree0 = mbar_vv0 + SB * 8;  // even stages: S drained
  const uint32_t mbar_sfree1 = mbar_sfree0 + 8;    // odd stages
  const uint32_t mbar_pre0 = mbar_sfree0 + DH * 2 * 8;  // P + rescale
  const uint32_t mbar_pre1 = mbar_pre0 + 8;
#ifdef FMHA_DH2_PEARLY
  // PEARLY (M2-lite): early-P-chunk barriers, DH2-only (host + constexpr
  // gated; DH1 never inits/arrives/waits these). Past the pre region:
  // DH2 tail use 2208 -> 2240 B of the 2688 B fmha_tc_smem reservation.
  const uint32_t mbar_prearly0 = mbar_pre0 + DH * 2 * 8;
#endif
  // MC (cluster multicast pair): rank1 -> rank0 refill-ready handshake.
  // Past the pre region AND PEARLY's four-barrier block: DH2 tail use
  // 2248 of 2688 B. Declared unconditionally (an unused constant in
  // non-MC builds) so if-constexpr(MC) references resolve at definition.
  const uint32_t mbar_r1rd = mbar_pre0 + DH * 2 * 8 + 32;
#ifndef FMHA_TC_PAGEPRE
  static_assert(!MC, "MC v1 requires FMHA_TC_PAGEPRE (tc_tma_kv_mc page word)");
#endif
  // Each of the four is used every other stage, so its own phase index is s>>1.

  const int tid = threadIdx.x;
  const bool ctrl = (tid >= SW * DH);
  // CSPLIT (peer arm-a): two control warps - warp A = TMA-lead (boxes/arms/
  // page prefetch), warp B = MMA-lead (kv/vv/pre/sfree waits + all tcgen05.mma
  // issues + commits). Base: one warp does both serially (peer ex2: tma 985 +
  // mma_issue 1346 + waits ~600 = the critical 2923 vs softmax 2776; the split
  // turns the sum into a max). Dependencies ride the EXISTING mbarriers (no
  // new sync); the MMA-lead keeps the load-bearing PV(s)-before-QK(s+1)
  // issue+commit order, so commit coverage - and outputs - are bitwise
  // identical. The TMA-lead additionally proves slot-s landing (kv/vv phase
  // wait) before arming it for stage s+SB, since it no longer trails the
  // MMA-lead's step-(c) vv wait in one program order.
  const bool csplit_tma = CS && ctrl && (tid < SW * DH + 32);
#ifdef FMHA_DH2_MMA2
  // DH2 v2: warp A = TMA-lead, warps B/C = per-half MMA-leads (B = half 0,
  // C = half 1). tcgen05.commit tracks the ISSUING THREAD's prior async
  // ops, so with one issuer per half each commit covers exactly its own
  // half's chain - cross-half commit poisoning (the ORDER B failure) is
  // structurally impossible and no wait couples the half chains.
  const bool csplit_mma = CS && ctrl && (tid >= SW * DH + 32) && (DH != 2);
  const bool mma2_h0 = (DH == 2) && ctrl && (tid >= SW * DH + 32)
                       && (tid < SW * DH + 64);
  const bool mma2_h1 = (DH == 2) && ctrl && (tid >= SW * DH + 64);
  const int mma2_h = mma2_h1 ? 1 : 0;
#else
  const bool csplit_mma = CS && ctrl && (tid >= SW * DH + 32);
#endif
  const int lane = (tid - SW * DH) & 31;        // per-control-warp lane
  const int half = (tid >> 7) & (NH - 1);       // softmax: S column group
  // DH2: warps 0-3 = half 0 (tile rows 0-127), warps 4-7 = half 1 (rows
  // 128-255). `row` becomes the GLOBAL row in the (256-row) tile; DH1 is
  // value-identical (dhalf == 0 for every softmax thread).
  const int dhalf = (DH == 2) ? (int)((unsigned)tid / SW) : 0;
  const int row = (tid & 127) + dhalf * 128;    // global row in the tile

  int tile_y, split, kvh;
  if constexpr (MC) {
    // Padded-y cluster grid (host: mgrid(kvh, (tiles_ub+1)&~1, 1)); rank =
    // tile_y & 1; the pair covers adjacent tile_y (2t, 2t+1). Both ranks
    // compute mc_on identically from local inputs - no communication in
    // the decision path, no divergent cluster paths.
    kvh = (int)blockIdx.x;
    tile_y = (int)blockIdx.y;
    split = 0;
  } else {
    grid_pos(P, kvh, tile_y, split);
  }
  const int tile = ((((P._pad2 >> 1) & 1) && P.causal)
                        ? (P.tiles_ub - 1 - tile_y)
                        : tile_y);
  // dead_pad (MC only): the padded y slot past tiles_ub maps through rev to
  // tile == -1, and map_tile(-1) RETURNS A LIVE-LOOKING GARBAGE TILE when
  // group >= 2 (m0 = -256; ql = (-256+255)/group truncates to 0 -> hi > lo
  // -> ok == true -> negative-row OOB reads/writes). main8mc guards this
  // with dead_pad; mirror it exactly. (c7's group==1 hides it: ql == -1 ->
  // hi == 0 -> ok == false.)
  const bool dead_pad = MC && (tile_y >= P.tiles_ub);
  TileInfo ti = (dead_pad || (MC && tile < 0))
                    ? TileInfo{0, 0, 0, 0, 0, 0, 0, false}
                    : map_tile(P, tile);
  // MC rank/pairing state (declared unconditionally for name resolution in
  // non-MC builds; mc_on stays false there). Pairing gates mirror main8mc
  // and the host: both tiles valid, same batch, no window, no splits - so
  // both start at kv0 and share one stage grid.
  uint32_t rank = 0;
  bool mc_on = false;
  int peer_hi = 0, last_mc = 0;
  uint32_t peer_r1rd = 0;
  if constexpr (MC) {
    rank = cluster_rank();
    const int peer_y = tile_y ^ 1;
    const bool peer_pad = (peer_y >= P.tiles_ub);
    const int peer_tile = (((P._pad2 >> 1) & 1) && P.causal)
                              ? (P.tiles_ub - 1 - peer_y)
                              : peer_y;
    const TileInfo tp = (peer_pad || peer_tile < 0)
                            ? TileInfo{0, 0, 0, 0, 0, 0, 0, false}
                            : map_tile(P, peer_tile);
    mc_on = ti.ok && tp.ok && (ti.b == tp.b) && P.window_left < 0 &&
            P.splits_ub == 1;
#ifdef FMHA_DH2_MC_SOLO
    // Hang bisection flag: keep the cluster LAUNCH (grid/attrs/dead-pad
    // ranks) but force every pair to the solo unicast path - separates
    // cluster-launch mechanics from the pairing protocol.
    mc_on = false;
#endif
    peer_hi = tp.ok ? tp.hi : 0;
  }
  if (!ti.ok) return;   // dead slack rank: the peer computes mc_on == false
  int kv0, kv1, active;
  if (!split_range(P, ti, split, kv0, kv1, active)) return;
  const int kv0_exact = kv0;
  kv0 &= ~(NT - 1);            // page-align stage starts; rlo masks the shift
  int nstages_hi = kv1;
  if constexpr (MC) {
    // Shared stage grid: the pair walks max(own hi, peer hi); the shorter
    // tile's extra stages are fully masked by its own per-row rhi (exact).
    if (mc_on) {
      last_mc = max(kv1, peer_hi) - 1;
      nstages_hi = max(kv1, peer_hi);
    }
  }
  const int nstages = (nstages_hi - kv0 + NT - 1) / NT;
  if (nstages <= 0) return;
  const int32_t* const tabt = P.table + ti.b * P.tstride;
  // Box/page row clamp: the shared pair range under mc, else the own tile.
  int box_hi = ti.hi - 1;
  if constexpr (MC) {
    if (mc_on) {
      box_hi = last_mc;
      peer_r1rd = mapa_u32(mbar_r1rd, 0);
    }
  }

  if (tid == 0) {
#pragma unroll
    for (int h = 0; h < DH; ++h) {
      mbar_init(mbar_qk + h * 8, 1);
      mbar_init(mbar_pv + h * 8, 1);
      mbar_init(mbar_sfree0 + h * 16, NH * 4);
      mbar_init(mbar_sfree1 + h * 16, NH * 4);
      mbar_init(mbar_pre0 + h * 16, NH * 4);
      mbar_init(mbar_pre1 + h * 16, NH * 4);
#ifdef FMHA_DH2_PEARLY
      // BOTH parities (the sfree/pre pairs init via the *0/*1 constant
      // pair; prearly has one constant - the odd-parity barrier lives at
      // +8). dv_e62da3cbf8ea: missing this = arrive/wait on an
      // uninitialized mbarrier at stage 1 = deterministic launch failure.
      if constexpr (DH == 2) {
        mbar_init(mbar_prearly0 + h * 16, NH * 4);
        mbar_init(mbar_prearly0 + h * 16 + 8, NH * 4);
      }
#endif
    }
    if constexpr (MC) mbar_init(mbar_r1rd, 1);
    for (int j = 0; j < SB; ++j) {
      mbar_init(mbar_kv0 + j * 8, 1);
      mbar_init(mbar_vv0 + j * 8, 1);
    }
    for (int j = 0; j < SB && j < nstages; ++j) {
      mbar_expect_tx(mbar_kv0 + j * 8, NT * D * 2);
      // QKEARLY arms V slots inline (just before each box issue) and its
      // prologue fills ONLY V stage 0; every later V slot gets its first
      // inline arm+box at iteration s = slot-1. Arming more slots here would
      // double-arm slot 1 against iteration 0's inline arm: if the prologue
      // box completes before that second expect_tx lands, the slot's phase
      // flips early and every later parity wait on it is off by one
      // generation (the code-provable V-prologue hazard; fixed pre-join once,
      // re-derived independently here).
      if (!QKEARLY || j == 0)
        mbar_expect_tx(mbar_vv0 + j * 8, NT * DV * 2);
    }
  }
  if (tid < 32) tmem_alloc(su32(tmem_slot), TC_COLS);
  __syncthreads();
  const uint32_t tmem = *tmem_slot;
  if (tid < 32) tmem_relinquish();
  // DH2 tmem map (reference context kernel): S0|S1 | P0|P1 | O0|O1 =
  // 64+64 | 64+64 | 128+128 = 512 cols. Per-half: S_h = tmem_s + h*S_COLS,
  // P_h = tmem_p + h*P_COLS, O_h = tmem_o + h*DV. DH1 offsets unchanged.
  const uint32_t tmem_s = tmem;
  const uint32_t tmem_p = tmem + DH * S_COLS;
  const uint32_t tmem_o = tmem + DH * (S_COLS + P_COLS);

  constexpr uint32_t IDESC_QK = tc_idesc(128, NT, false);
  constexpr uint32_t IDESC_PV = tc_idesc(128, DV, true);
  constexpr uint64_t A_ATOM = (128 * 128) >> 4;
  constexpr uint64_t B_ATOM = (uint64_t)NT * 8;
  constexpr uint32_t V_LBO = (DV == 64) ? 0 : NT * 128;
  const uint64_t qd0 = tc_desc(su32(smem_q), 1024, 0);

#define WS_PROXY_FENCE fence_proxy_async();
// WS_TMA: hand one stage's boxes to the TMA engine, spread over the 32 control
// lanes (OPS is 2-8 boxes, so most lanes idle).
#define WS_TMA(KV, DST, MAP, STAGE, MB)                                       \
      tc_tma_kv<NT, KV, 32>(P, DST, MAP, kvh, kv0 + (STAGE) * NT, ti.hi - 1,  \
                            tabt, lane, MB);

  // TMA-after-mma control-loop order (FMHA_TC_TMA_LATE). Phase probe
  // dv_fc8d4bb9a57c (c1/c6/c7, FMHA_PHASE counters): the ctrl warp's box-issue
  // phase costs ~750-990 cyc/iter and sits IN PROGRAM ORDER BEFORE the PV(s)/
  // QK(s+1) issues, so the tensor pipe drains and then idles behind it every
  // stage (softmax qkwait ~1470 == ctrl mma_issue ~1350 = issue-side
  // backpressure; c1 period 2923 cyc against ~2x768 of pipe work per SM).
  // TMA_LATE moves stage s+SB's arm+issue AFTER both mma issues. Guards are
  // unchanged: the K slot is freed by sfree(s) (=> qk_done(s) => QK(s) done),
  // the V slot by an explicit mbar_pv(s) wait (which gates no mma issue and
  // is free in steady state). Per-iteration issue order stays PV(s) before
  // QK(s+1) - the load-bearing cumulative-commit invariant - and boxes,
  // coords, order and flight windows (now one iteration longer) are
  // unchanged, so results are bitwise identical.
#ifdef FMHA_TC_TMA_LATE
  constexpr bool TMALATE = !QKFIRST && !QKEARLY;
#else
  // Selective TMA_LATE via template parameter (dispatch bit24, host knob
  // FMHA_TC_TLSEL): the late-issue order won on d256 (c6 x0.976/x0.985) and
  // split-d128 (c9 x0.969/x0.985) in two independent jobs but lost on
  // long-tile non-split d128 (c1 x1.040) - shorter box lead there.
  constexpr bool TMALATE = TL && !QKFIRST && !QKEARLY;
#endif

  // Q tile is cp.async'd by the 256 softmax threads; the control warp only
  // joins the barrier and issues the prologue KV bulk loads. Under CSPLIT the
  // TMA-lead warp owns every box issue, prologue included.
  if (!ctrl) {
#pragma unroll
    for (int h = 0; h < DH; ++h) {
      // DH2: each 128-row half owns a separate 32KB Q region with the
      // DH1-identical inner layout (16KB atom stride = A_ATOM): the A
      // descriptor base advances by 128*D*2/16 = 2048 units per half and
      // the per-atom stepping is unchanged. (A single 256-row tc_load_q
      // would double the atom stride and break A_ATOM.)
      TileInfo tih = ti;
      tih.m0 += h * 128;
      tc_load_q<128, D, SW * DH>(P, tih, smem_q + h * (128 * D * 2), kvh, tid);
    }
  }
  cp_async_commit();
  cp_async_wait<0>();
  if constexpr (MC) {
    if (mc_on) {
      // Publish rank-local mbarrier inits + prologue slot arms cluster-wide
      // before rank0 issues the shared stage-0 boxes (main8mc's proven
      // pattern; cluster_sync is .aligned - every thread of both ranks).
      mbar_init_fence();
      cluster_sync_mc();
    }
  }
  if (ctrl && (!CS || csplit_tma)) {
#ifdef FMHA_TC_PRO2
    // PRO2 (FMHA_TC_PRO2): fill BOTH slot generations in the prologue (the
    // init block already armed kv1/vv1) and skip iteration 0's step-(a)
    // issue. Isolates TMA_LATE's deeper cold-start prefetch from its late
    // steady-state issue reorder (S3/S10: TMA_LATE won c6 x0.976-0.985 and
    // c9 x0.969-0.985 but lost c1 x1.040 - hypothesis: the wins are the
    // prologue depth, the loss is the shorter steady-state box lead).
    // Same boxes/coords/order -> bitwise identical.
    // CSPLIT keeps the peer-verified base prologue depth (SB-1): its
    // TMA-lead iteration 0 step (a) issues stage SB-1 (cs1/cs3 clean).
    // FMHA_CS_PRO2 (S39): give CSPLIT the same deep prologue as the base
    // kernel - both slots filled here, TMA-lead iteration-0 step (a)
    // skipped below. Same boxes/coords/order -> bitwise claim.
#ifdef FMHA_CS_PRO2
    const int pre = nstages < SB ? nstages : SB;
#else
    const int pre = CS ? (nstages < SB - 1 ? nstages : SB - 1)
                       : (nstages < SB ? nstages : SB);
#endif
#else
    const int pre = nstages < SB - 1 ? nstages : SB - 1;
#endif
    // QKEARLY arms V inline; TMA_LATE's iteration 0 issues stage SB, so the
    // prologue must fill BOTH slot generations (init block armed them).
    // QKEARLY's V prologue is exactly ONE stage (slot 0, armed in init):
    // every later V stage is armed+boxed inline at iteration s-1, one arm
    // per box per generation (see the init-block comment).
    const int prek = (QKEARLY || TMALATE) ? (nstages < SB ? nstages : SB) : pre;
    const int prev = QKEARLY ? (nstages < 1 ? nstages : 1)
                             : (TMALATE ? prek : pre);
    if constexpr (MC) {
      if (mc_on) {
        // One multicast pair for stage 0 feeds BOTH ranks' slot 0 (armed at
        // init). rank1 issues nothing; solo pairs take the stock prologue.
        if (rank == 0 && nstages >= 1) {
          const int32_t pg0 = tc_page_pre<NT, D>(P, tabt, kv0, box_hi, lane);
          fence_proxy_cluster();
          __syncwarp();
          tc_tma_kv_mc<NT, D>(P, smem_k[0], &kmap, kvh, kv0, box_hi, lane,
                              pg0, mbar_kv0, 3);
          fence_proxy_cluster();
          __syncwarp();
          tc_tma_kv_mc<NT, DV>(P, smem_v[0], &vmap, kvh, kv0, box_hi, lane,
                               pg0, mbar_vv0, 3);
        }
      } else {
        for (int j = 0; j < prek; ++j)
          WS_TMA(D, smem_k[j], &kmap, j, mbar_kv0 + j * 8)
        for (int j = 0; j < prev; ++j)
          WS_TMA(DV, smem_v[j], &vmap, j, mbar_vv0 + j * 8)
      }
    } else {
      for (int j = 0; j < prek; ++j)
        WS_TMA(D, smem_k[j], &kmap, j, mbar_kv0 + j * 8)
      for (int j = 0; j < prev; ++j)
        WS_TMA(DV, smem_v[j], &vmap, j, mbar_vv0 + j * 8)
    }
  }
  fence_proxy_async();
  __syncthreads();

  if (ctrl) {
    // ---- control warp: TMA prefetch + all mma issue, nothing else ----
    // Wave dephase (see dephase_cycles): only the control warp spins, so the
    // softmax warps race ahead to the mbar_qk wait while this CTA's tensor
    // issue shifts by ~half a pipe round relative to its sibling CTA.
    // MINB>=2 configs only (the 1-CTA/SM tiles have no sibling to overlap).
    if constexpr (MINB >= 2) dephase_spin(P);
#ifdef FMHA_PHASE
    long long c_pv = 0, c_tma = 0, c_pre = 0, c_vv = 0, c_sfr = 0, c_kv = 0,
              c_mma = 0, c_n = 0;
#define WS_ACCB c_sfr += tb1 - tb0; c_kv += tb2 - tb1; c_mma += tb3 - tb2;
#define WS_ACCC c_pre += tc1 - tc0; c_vv += tc2 - tc1; c_mma += tc3 - tc2;
#else
#define WS_ACCB
#define WS_ACCC
#endif
    fence_after_sync();
#ifdef FMHA_DH2_MMA2
    if (!CS || csplit_mma || mma2_h0) {
#else
    if (!CS || csplit_mma) {
#endif
      mbar_wait(mbar_kv0, 0);
      fence_proxy_async();
      fence_after_sync();
      // DH2: prime QK(0) for BOTH halves (own S region + own qk barrier).
      // Half h reads Q smem rows [128h, 128h+128): descriptor units are
      // 16B, so the A base advances by 128*D*2/16 = 16*D per half.
#pragma unroll
      for (int h = 0; h < DH; ++h) {
        if (lane == 0) {
          uint64_t da = qd0 + h * (16 * D), db = tc_desc(su32(smem_k[0]), 1024, 0);
#pragma unroll
          for (int ks = 0; ks < D / 16; ++ks) {
            mma_ss(tmem_s + h * S_COLS, da, db, IDESC_QK, ks != 0);
            da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
            db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
          }
          umma_commit(mbar_qk + h * 8);
        }
        __syncwarp();
      }
    }
#ifdef FMHA_TC_PAGEPRE
    // Page IDs for the first in-loop box issue: stage SB-1 for the base order
    // (iteration 0's step (a)) and CSPLIT's TMA-lead, stage SB for TMA_LATE
    // (iteration 0's step (3)).
    // (Prologue stages keep the blocking form; they overlap the Q load.)
    int pg = 0;
    if (!CS || csplit_tma)
      pg = tc_page_pre<NT, D>(P, tabt, kv0 + (TMALATE ? SB : SB - 1) * NT,
                              box_hi, lane);
#ifdef FMHA_PGDEEP
    // Deep page-ID pipeline (mechanism M5 of the production reference,
    // PRODUCTION_REFERENCE/context_*.cu warp 13: page-table loads streamed
    // into a 6-deep smem ring by a dedicated producer warp). The NCU census
    // put 54.9% of warp stalls on long_scoreboard (the dependent page
    // __ldg); PAGEPRE hoists it only ONE iteration. Here the TMA-lead
    // streams page IDs PGD stages ahead via cp.async commit groups and
    // consumes its own lane's ring word - single writer/reader warp, no
    // barriers. Ring sits in the free smem tail past the pre barriers
    // (NH1: bytes 1120..1376 of the 2688B tail; NH4: 4192..4448 of 4992;
    // NH2: 2144..2400 of 2944 - fits every main8ws tail).
    constexpr int PGD = 4;
    int* const pg_ring = reinterpret_cast<int*>(
        reinterpret_cast<char*>(mbar) + 48 + NH * 1024 + SB * 16 + 16);
    const int pgR = min(NT, 1 << P.page_log);
    const int pgseg = min(lane / (D / 64), NT / pgR - 1);
#define PG_PREFETCH(STG, SLOT)                                                 \
    do {                                                                       \
      const int _pstg = min((STG), nstages - 1);                               \
      const int _pp = min(kv0 + _pstg * NT + pgseg * pgR, ti.hi - 1);          \
      cp_async4(su32(pg_ring + ((SLOT) % PGD) * 16 + lane),                    \
                tabt + (_pp >> P.page_log));                                   \
      cp_async_commit();                                                       \
    } while (0)
    if (CS && csplit_tma) {
      // Fill the ring for the first PGD-1 in-loop box stages (the stock CS
      // TMA-lead boxes stage SB-1 at iteration 0).
#pragma unroll
      for (int d = 0; d < PGD - 1; ++d)
        PG_PREFETCH(SB - 1 + d, SB - 1 + d);
    }
#endif
#endif
    if constexpr (CS) {
      // ---- CSPLIT (peer arm-a join-20261025, ported with attribution) ----
      // MMA-lead warp: steps (c)+(b) - PV(s) then QK(s+1), the base issue
      // order and commit coverage. TMA-lead warp: steps (a)+(d)+pg. Peer
      // evidence: bitwise clean (cs1 dv_145c0b1fa0cc 14/14, cs3
      // dv_c61f78ba783b 17/17); gated wins c1 x0.969-0.973, c9 x0.924-0.973
      // (their base, without PRO2/TLSEL - this tree's A/B re-measures).
#ifdef FMHA_DH2_MMA2
      if (mma2_h0 || mma2_h1) {
        // DH2 v2 per-half MMA-lead: warp B runs half 0's chain, warp C runs
        // half 1's - each with its OWN waits (both warps wait the shared
        // vv/kv phases; mbarrier waits are non-consuming) and its OWN
        // commits. Per-half order = the certified base order; per-thread
        // commit tracking means pv_h/qk_h commits cover exactly warp h's
        // chain, so NO cross-half coupling remains (ORDER B's poisoning
        // required a shared issuer; here it is structurally impossible).
        const int h = mma2_h;
#ifdef FMHA_PHASE
        unsigned long long m_pre = 0, m_vv = 0, m_pvi = 0, m_sfr = 0,
                           m_kv = 0, m_qki = 0, m_n = 0;
        long long ma0, ma1;
#endif
        for (int s = 0; s < nstages; ++s) {
#ifdef FMHA_PHASE
          ma0 = clock64();
#endif
          mbar_wait(mbar_pre0 + (h * 2 + (s & 1)) * 8, (s >> 1) & 1);
          fence_after_sync();
#ifdef FMHA_PHASE
          ma1 = clock64(); m_pre += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
          mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);
          WS_PROXY_FENCE
          fence_after_sync();
#ifdef FMHA_PHASE
          ma1 = clock64(); m_vv += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
          {
            const uint32_t pb = tmem_p + h * P_COLS + (s & 1) * PH;
            const uint32_t vb =
                (uint32_t)((su32(smem_v[s % SB]) >> 4) & 0x3FFF) |
                (uint32_t)((V_LBO >> 4) << 16);
#ifdef FMHA_IMMMAC
            mma_pv_imm_nt64_dv128(tmem_o + h * DV, pb, vb, s > 0);
#else
            if (lane == 0) {
              uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
#pragma unroll
              for (int st = 0; st < NT / 16; ++st) {
                mma_ts(tmem_o + h * DV, pb + st * 8, db, IDESC_PV,
                       (st != 0) || (s > 0));
                db += 128;
              }
            }
#endif
            if (lane == 0) umma_commit(mbar_pv + h * 8);
            __syncwarp();
          }
#ifdef FMHA_PHASE
          ma1 = clock64(); m_pvi += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
          if (s + 1 < nstages) {
            mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8, ((s + 1) / SB) & 1);
            WS_PROXY_FENCE
            fence_after_sync();
#ifdef FMHA_PHASE
            ma1 = clock64(); m_kv += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
            mbar_wait(mbar_sfree0 + (h * 2 + (s & 1)) * 8, (s >> 1) & 1);
            fence_after_sync();
#ifdef FMHA_PHASE
            ma1 = clock64(); m_sfr += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
            const uint32_t alo = (uint32_t)(qd0 & 0xFFFFFFFFULL);
            const uint32_t blo =
                (uint32_t)((su32(smem_k[(s + 1) % SB]) >> 4) & 0x3FFF);
#ifdef FMHA_IMMMAC
            mma_qk_imm_d128(tmem_s + h * S_COLS, alo + h * (16 * D), blo, 0);
#else
            if (lane == 0) {
              uint64_t da = qd0 + h * (16 * D),
                       db = tc_desc(su32(smem_k[(s + 1) % SB]), 1024, 0);
#pragma unroll
              for (int ks = 0; ks < D / 16; ++ks) {
                mma_ss(tmem_s + h * S_COLS, da, db, IDESC_QK, ks != 0);
                da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
                db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
              }
            }
#endif
            if (lane == 0) umma_commit(mbar_qk + h * 8);
            __syncwarp();
#ifdef FMHA_PHASE
            ma1 = clock64(); m_qki += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
          }
#ifdef FMHA_PHASE
          ++m_n;
#endif
        }
#ifdef FMHA_PHASE
        if (lane == 0) {
          atomicAdd(&g_ph[16], m_pre); atomicAdd(&g_ph[17], m_vv);
          atomicAdd(&g_ph[18], m_pvi); atomicAdd(&g_ph[19], m_sfr);
          atomicAdd(&g_ph[20], m_kv);  atomicAdd(&g_ph[21], m_qki);
          atomicAdd(&g_ph[26], m_n);
          if (h == 0) atomicAdd(&g_ph[28], 1ull);
        }
#endif
      } else
#endif  // FMHA_DH2_MMA2
      if (csplit_mma) {
#ifdef FMHA_PHASE
        unsigned long long m_pre = 0, m_vv = 0, m_pvi = 0, m_sfr = 0,
                           m_kv = 0, m_qki = 0, m_n = 0;
        long long ma0, ma1;
#endif
        if constexpr (DH == 2) {
          // DH2 MMA-lead: two independent per-half chains in ONE ctrl warp.
          // Per half the certified base order holds (PV_h(s) issued+committed
          // before QK_h(s+1); pre_h gates PV_h, sfree_h gates QK_h). The
          // K/V stream is shared: one box pair per stage feeds both halves,
          // so TMA-lead work per MAC halves. Overlap comes from the two
          // chains, not from reordering within a chain.
#ifdef FMHA_DH2_ORDERB
          // ORDER B (probe-driven, results/20261027-dh2-phase-probe.md):
          // per-half interleaved issue PV0(s) -> QK0(s+1) -> PV1(s) ->
          // QK1(s+1). Per-half order and commit coverage identical to
          // order A (PV_h(s) issued+committed before QK_h(s+1)); the
          // cross-half interleave removes pre1(s)+PV1(s) from half-0's
          // S-refill path (order-A mma_pre_wait 500 cyc/iter). Per-half
          // tmem regions are disjoint; the V-slot gate absorbs QK0's
          // pipe time into tma_pv_wait's ~1190-cyc slack. Bitwise class.
          for (int s = 0; s < nstages; ++s) {
#ifdef FMHA_PHASE
            ma0 = clock64();
#endif
            const uint32_t alo = (uint32_t)(qd0 & 0xFFFFFFFFULL);
            const uint32_t blo =
                (uint32_t)((su32(smem_k[(s + 1) % SB]) >> 4) & 0x3FFF);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#ifdef FMHA_DH2_PEARLY
              mbar_wait(mbar_prearly0 + (h * 2 + (s & 1)) * 8, (s >> 1) & 1);
#else
              mbar_wait(mbar_pre0 + (h * 2 + (s & 1)) * 8, (s >> 1) & 1);
#endif
              fence_after_sync();
#ifdef FMHA_PHASE
              ma1 = clock64(); m_pre += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              if (h == 0) {
                mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);
                WS_PROXY_FENCE
                fence_after_sync();
#ifdef FMHA_PHASE
                ma1 = clock64(); m_vv += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              }
              const uint32_t pb = tmem_p + h * P_COLS + (s & 1) * PH;
              const uint32_t vb =
                  (uint32_t)((su32(smem_v[s % SB]) >> 4) & 0x3FFF) |
                  (uint32_t)((V_LBO >> 4) << 16);
#ifdef FMHA_DH2_PEARLY
              // M2-lite: keys 0-31 on pre_early, keys 32-63 on the full
              // pre; the pv commit still lands after both pairs (V-gate
              // coverage unchanged - the ORDER B lesson).
              mma_pv_imm_nt32x2_dv128(tmem_o + h * DV, pb, vb, s > 0);
#ifdef FMHA_PHASE
              ma1 = clock64(); m_pvi += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              mbar_wait(mbar_pre0 + (h * 2 + (s & 1)) * 8, (s >> 1) & 1);
              fence_after_sync();
#ifdef FMHA_PHASE
              ma1 = clock64(); m_pre += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              mma_pv_imm_nt32x2_dv128(tmem_o + h * DV, pb + 16, vb + 256, 1);
#elif defined(FMHA_IMMMAC)
              mma_pv_imm_nt64_dv128(tmem_o + h * DV, pb, vb, s > 0);
#else
              if (lane == 0) {
                uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
#pragma unroll
                for (int st = 0; st < NT / 16; ++st) {
                  mma_ts(tmem_o + h * DV, pb + st * 8, db, IDESC_PV,
                         (st != 0) || (s > 0));
                  db += 128;
                }
              }
#endif
              if (lane == 0) umma_commit(mbar_pv + h * 8);
              __syncwarp();
#ifdef FMHA_PHASE
              ma1 = clock64(); m_pvi += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              if (s + 1 < nstages) {
                if (h == 0) {
                  mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8,
                            ((s + 1) / SB) & 1);
                  WS_PROXY_FENCE
                  fence_after_sync();
#ifdef FMHA_PHASE
                  ma1 = clock64(); m_kv += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
                }
                mbar_wait(mbar_sfree0 + (h * 2 + (s & 1)) * 8, (s >> 1) & 1);
                fence_after_sync();
#ifdef FMHA_PHASE
                ma1 = clock64(); m_sfr += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
#ifdef FMHA_IMMMAC
                mma_qk_imm_d128(tmem_s + h * S_COLS, alo + h * (16 * D), blo,
                                0);
#else
                if (lane == 0) {
                  uint64_t da = qd0 + h * (16 * D),
                           db = tc_desc(su32(smem_k[(s + 1) % SB]), 1024, 0);
#pragma unroll
                  for (int ks = 0; ks < D / 16; ++ks) {
                    mma_ss(tmem_s + h * S_COLS, da, db, IDESC_QK, ks != 0);
                    da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
                    db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
                  }
                }
#endif
                if (lane == 0) umma_commit(mbar_qk + h * 8);
                __syncwarp();
#ifdef FMHA_PHASE
                ma1 = clock64(); m_qki += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              }
            }
#ifdef FMHA_PHASE
            ++m_n;
#endif
          }
#else
          for (int s = 0; s < nstages; ++s) {
#ifdef FMHA_PHASE
            ma0 = clock64();
#endif
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#ifdef FMHA_DH2_PEARLY
              mbar_wait(mbar_prearly0 + (h * 2 + (s & 1)) * 8, (s >> 1) & 1);
#else
              mbar_wait(mbar_pre0 + (h * 2 + (s & 1)) * 8, (s >> 1) & 1);
#endif
              fence_after_sync();
#ifdef FMHA_PHASE
              ma1 = clock64(); m_pre += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              if (h == 0) {
                mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);
                WS_PROXY_FENCE
                fence_after_sync();
#ifdef FMHA_PHASE
                ma1 = clock64(); m_vv += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              }
              const uint32_t pb = tmem_p + h * P_COLS + (s & 1) * PH;
              const uint32_t vb =
                  (uint32_t)((su32(smem_v[s % SB]) >> 4) & 0x3FFF) |
                  (uint32_t)((V_LBO >> 4) << 16);
#ifdef FMHA_DH2_PEARLY
              // M2-lite: keys 0-31 on pre_early, keys 32-63 on the full
              // pre; the pv commit still lands after both pairs (V-gate
              // coverage unchanged - the ORDER B lesson).
              mma_pv_imm_nt32x2_dv128(tmem_o + h * DV, pb, vb, s > 0);
#ifdef FMHA_PHASE
              ma1 = clock64(); m_pvi += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              mbar_wait(mbar_pre0 + (h * 2 + (s & 1)) * 8, (s >> 1) & 1);
              fence_after_sync();
#ifdef FMHA_PHASE
              ma1 = clock64(); m_pre += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              mma_pv_imm_nt32x2_dv128(tmem_o + h * DV, pb + 16, vb + 256, 1);
#elif defined(FMHA_IMMMAC)
              mma_pv_imm_nt64_dv128(tmem_o + h * DV, pb, vb, s > 0);
#else
              if (lane == 0) {
                uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
#pragma unroll
                for (int st = 0; st < NT / 16; ++st) {
                  mma_ts(tmem_o + h * DV, pb + st * 8, db, IDESC_PV,
                         (st != 0) || (s > 0));
                  db += 128;
                }
              }
#endif
              if (lane == 0) umma_commit(mbar_pv + h * 8);
              __syncwarp();
#ifdef FMHA_PHASE
              ma1 = clock64(); m_pvi += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
            }
            if (s + 1 < nstages) {
              // Shared K(s+1) first, then per-half sfree->QK: half h's
              // QK(s+1) overwrites S_h(s), so sfree_h(s) gates ONLY QK_h -
              // a slow half 1 must not delay half 0's S refill (skew
              // isolation the 2-CTA base gets for free).
              mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8, ((s + 1) / SB) & 1);
              WS_PROXY_FENCE
              fence_after_sync();
#ifdef FMHA_PHASE
              ma1 = clock64(); m_kv += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              const uint32_t alo = (uint32_t)(qd0 & 0xFFFFFFFFULL);
              const uint32_t blo =
                  (uint32_t)((su32(smem_k[(s + 1) % SB]) >> 4) & 0x3FFF);
#pragma unroll
              for (int h = 0; h < 2; ++h) {
                mbar_wait(mbar_sfree0 + (h * 2 + (s & 1)) * 8, (s >> 1) & 1);
                fence_after_sync();
#ifdef FMHA_PHASE
                ma1 = clock64(); m_sfr += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
#ifdef FMHA_IMMMAC
                mma_qk_imm_d128(tmem_s + h * S_COLS, alo + h * (16 * D), blo,
                                0);
#else
                if (lane == 0) {
                  uint64_t da = qd0 + h * (16 * D),
                           db = tc_desc(su32(smem_k[(s + 1) % SB]), 1024, 0);
#pragma unroll
                  for (int ks = 0; ks < D / 16; ++ks) {
                    mma_ss(tmem_s + h * S_COLS, da, db, IDESC_QK, ks != 0);
                    da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
                    db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
                  }
                }
#endif
                if (lane == 0) umma_commit(mbar_qk + h * 8);
                __syncwarp();
#ifdef FMHA_PHASE
                ma1 = clock64(); m_qki += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
              }
            }
#ifdef FMHA_PHASE
            ++m_n;
#endif
          }
#endif  // FMHA_DH2_ORDERB (else: order A above)
#ifdef FMHA_PHASE
          if (lane == 0) {
            atomicAdd(&g_ph[16], m_pre); atomicAdd(&g_ph[17], m_vv);
            atomicAdd(&g_ph[18], m_pvi); atomicAdd(&g_ph[19], m_sfr);
            atomicAdd(&g_ph[20], m_kv);  atomicAdd(&g_ph[21], m_qki);
            atomicAdd(&g_ph[26], m_n);   atomicAdd(&g_ph[28], 1ull);
          }
#endif
        } else {
        for (int s = 0; s < nstages; ++s) {
          // (c) PV(s): P(s) packed (pre), V(s) resident (vv). Issued and
          // committed BEFORE QK(s+1) - the load-bearing commit-coverage
          // order, identical to the base WS_STEP_C -> WS_STEP_B sequence.
#ifdef FMHA_PHASE
          ma0 = clock64();
#endif
          mbar_wait((s & 1) ? mbar_pre1 : mbar_pre0, (s >> 1) & 1);
          fence_after_sync();
#ifdef FMHA_PHASE
          ma1 = clock64(); m_pre += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
          mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);
          WS_PROXY_FENCE
          fence_after_sync();
#ifdef FMHA_PHASE
          ma1 = clock64(); m_vv += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
#ifdef FMHA_IMMMAC
          {
            const uint32_t pb = tmem_p + (s & 1) * PH;
            const uint32_t vb =
                (uint32_t)((su32(smem_v[s % SB]) >> 4) & 0x3FFF) |
                (uint32_t)((V_LBO >> 4) << 16);
            if constexpr (NT == 64 && DV == 128) {
              mma_pv_imm_nt64_dv128(tmem_o, pb, vb, s > 0);
            } else if constexpr (NT == 64 && DV == 256) {
              mma_pv_imm_nt64_dv256(tmem_o, pb, vb, s > 0);
            } else if constexpr (NT == 128 && DV == 64) {
              mma_pv_imm_nt128_dv64(tmem_o, pb, vb, s > 0);
#ifdef FMHA_IMM_D64
            } else if constexpr (NT == 64 && DV == 64) {
              mma_pv_imm_nt64_dv64(tmem_o, pb, vb, s > 0);
#endif
            } else {
              if (lane == 0) {
                uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
#pragma unroll
                for (int st = 0; st < NT / 16; ++st) {
                  mma_ts(tmem_o, pb + st * 8, db, IDESC_PV,
                         (st != 0) || (s > 0));
                  db += 128;
                }
              }
            }
            if (lane == 0) umma_commit(mbar_pv);
          }
          __syncwarp();
#else
#ifdef FMHA_IMMMAC
          {
            const uint32_t pb = tmem_p + (s & 1) * PH;
            const uint32_t vb =
                (uint32_t)((su32(smem_v[s % SB]) >> 4) & 0x3FFF) |
                (uint32_t)((V_LBO >> 4) << 16);
            if constexpr (NT == 64 && DV == 256) {
              mma_pv_imm_nt64_dv256(tmem_o, pb, vb, s > 0);
            } else if constexpr (NT == 64 && DV == 128) {
              mma_pv_imm_nt64_dv128(tmem_o, pb, vb, s > 0);
            } else {
              if (lane == 0) {
                uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
#pragma unroll
                for (int st = 0; st < NT / 16; ++st) {
                  mma_ts(tmem_o, pb + st * 8, db, IDESC_PV,
                         (st != 0) || (s > 0));
                  db += 128;
                }
              }
            }
            if (lane == 0) umma_commit(mbar_pv);
          }
          __syncwarp();
#else
          if (lane == 0) {
            uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
            const uint32_t pb = tmem_p + (s & 1) * PH;
#pragma unroll
            for (int st = 0; st < NT / 16; ++st) {
              mma_ts(tmem_o, pb + st * 8, db, IDESC_PV, (st != 0) || (s > 0));
              db += 128;
            }
            umma_commit(mbar_pv);
          }
          __syncwarp();
#endif
#endif
#ifdef FMHA_PHASE
          ma1 = clock64(); m_pvi += (unsigned long long)(ma1 - ma0);
#endif
          // (b) QK(s+1): S(s) drained (sfree), K(s+1) resident (kv).
          if (s + 1 < nstages) {
#ifdef FMHA_PHASE
            ma0 = clock64();
#endif
            mbar_wait((s & 1) ? mbar_sfree1 : mbar_sfree0, (s >> 1) & 1);
            fence_after_sync();
#ifdef FMHA_PHASE
            ma1 = clock64(); m_sfr += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
            mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8, ((s + 1) / SB) & 1);
            WS_PROXY_FENCE
            fence_after_sync();
#ifdef FMHA_PHASE
            ma1 = clock64(); m_kv += (unsigned long long)(ma1 - ma0); ma0 = ma1;
#endif
#ifdef FMHA_IMMMAC
            {
              const int kb = (s + 1) % SB;
              const uint32_t alo = (uint32_t)(qd0 & 0xFFFFFFFFULL);
              const uint32_t blo =
                  (uint32_t)((su32(smem_k[kb]) >> 4) & 0x3FFF);
              if constexpr (D == 256 && NT == 64) {
                mma_qk_imm_d256(tmem_s, alo, blo, 0);
              } else if constexpr (D == 128 && NT == 64) {
                mma_qk_imm_d128(tmem_s, alo, blo, 0);
              } else if constexpr (D == 64 && NT == 128) {
                mma_qk_imm_d64nt128(tmem_s, alo, blo, 0);
#ifdef FMHA_IMM_D64
              } else if constexpr (D == 64 && NT == 64) {
                mma_qk_imm_d64nt64(tmem_s, alo, blo, 0);
#endif
              } else {
                if (lane == 0) {
                  uint64_t da = qd0, db = tc_desc(su32(smem_k[kb]), 1024, 0);
#pragma unroll
                  for (int ks = 0; ks < D / 16; ++ks) {
                    mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);
                    da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
                    db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
                  }
                }
              }
              if (lane == 0) umma_commit(mbar_qk);
            }
            __syncwarp();
#else
#ifdef FMHA_IMMMAC
            {
              const int kb = (s + 1) % SB;
              const uint32_t alo = (uint32_t)(qd0 & 0xFFFFFFFFULL);
              const uint32_t blo =
                  (uint32_t)((su32(smem_k[kb]) >> 4) & 0x3FFF);
              if constexpr (D == 256 && NT == 64) {
                mma_qk_imm_d256(tmem_s, alo, blo, 0);
              } else if constexpr (D == 128 && NT == 64) {
                mma_qk_imm_d128(tmem_s, alo, blo, 0);
              } else {
                if (lane == 0) {
                  uint64_t da = qd0,
                           db = tc_desc(su32(smem_k[kb]), 1024, 0);
#pragma unroll
                  for (int ks = 0; ks < D / 16; ++ks) {
                    mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);
                    da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
                    db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
                  }
                }
              }
              if (lane == 0) umma_commit(mbar_qk);
            }
            __syncwarp();
#else
            if (lane == 0) {
              const int kb = (s + 1) % SB;
              uint64_t da = qd0, db = tc_desc(su32(smem_k[kb]), 1024, 0);
#pragma unroll
              for (int ks = 0; ks < D / 16; ++ks) {
                mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);
                da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
                db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
              }
              umma_commit(mbar_qk);
            }
            __syncwarp();
#endif
#endif
#ifdef FMHA_PHASE
            ma1 = clock64(); m_qki += (unsigned long long)(ma1 - ma0);
#endif
          }
#ifdef FMHA_PHASE
          ++m_n;
#endif
        }
#ifdef FMHA_PHASE
        if (lane == 0) {
          atomicAdd(&g_ph[16], m_pre); atomicAdd(&g_ph[17], m_vv);
          atomicAdd(&g_ph[18], m_pvi); atomicAdd(&g_ph[19], m_sfr);
          atomicAdd(&g_ph[20], m_kv);  atomicAdd(&g_ph[21], m_qki);
          atomicAdd(&g_ph[26], m_n);   atomicAdd(&g_ph[28], 1ull);
        }
#endif
        }  // DH != 2
      } else {
#ifdef FMHA_PHASE
        unsigned long long t_pv = 0, t_box = 0, t_land = 0, t_n = 0;
#endif
        for (int s = 0; s < nstages; ++s) {
#ifdef FMHA_PGDEEP
          // Bound outstanding cp.async groups at PGD-2: stage s+SB-1's page
          // IDs are resident after this. Runs EVERY iteration (box-less tail
          // ones included) so the group accounting stays constant.
          cp_async_wait<PGD - 2>();
#endif
          // (a) load stage s+SB-1 into slot (s+SB-1)%SB (V slot freedom:
          // pv#(s-1); K freedom follows in-order from the same commit).
#ifdef FMHA_PHASE
          long long ta0 = clock64();
#endif
#ifdef FMHA_CS_PRO2
          // CS_PRO2: stage SB-1 is already in flight from the prologue.
          if (s + SB - 1 < nstages && s > 0) {
#else
          if (s + SB - 1 < nstages) {
#endif
            if (s > 0) {
              // DH2: the shared V slot is freed only after BOTH halves'
              // PV(s-1) have drained.
              mbar_wait(mbar_pv, (s - 1) & 1);
              if constexpr (DH == 2) mbar_wait(mbar_pv + 8, (s - 1) & 1);
              fence_after_sync();
              if constexpr (MC) {
                if (mc_on && rank == 0) {
                  // Peer's slot (s+1)%SB is re-armed AND its stage-(s-1)
                  // consumption is proven (rank1 signals after its own
                  // pv0/pv1(s-1) drain + local arm).
                  mbar_wait(mbar_r1rd, (s - 1) & 1);
                  fence_after_sync();
                }
              }
            }
#ifdef FMHA_PHASE
            {
              long long tap = clock64();
              t_pv += (unsigned long long)(tap - ta0);
              ta0 = tap;
            }
#endif
            const int j = (s + SB - 1) % SB;
#ifdef FMHA_PGDEEP
            {
              const int pgd_page = pg_ring[((s + SB - 1) % PGD) * 16 + lane];
              tc_tma_kv_pre<NT, D>(P, smem_k[j], &kmap, kvh,
                                   kv0 + (s + SB - 1) * NT, box_hi, lane,
                                   pgd_page, mbar_kv0 + j * 8);
              tc_tma_kv_pre<NT, DV>(P, smem_v[j], &vmap, kvh,
                                    kv0 + (s + SB - 1) * NT, box_hi, lane,
                                    pgd_page, mbar_vv0 + j * 8);
            }
#elif defined(FMHA_TC_PAGEPRE)
            if constexpr (MC) {
              if (mc_on) {
                // rank0 issues ONE multicast pair per stage for both CTAs
                // (rank1's slot was armed + signaled via r1rd); cluster-scope
                // proxy fence before each box (main8mc choreography).
                if (rank == 0) {
                  fence_proxy_cluster();
                  __syncwarp();
                  tc_tma_kv_mc<NT, D>(P, smem_k[j], &kmap, kvh,
                                      kv0 + (s + SB - 1) * NT, box_hi, lane,
                                      pg, mbar_kv0 + j * 8, 3);
                  fence_proxy_cluster();
                  __syncwarp();
                  tc_tma_kv_mc<NT, DV>(P, smem_v[j], &vmap, kvh,
                                       kv0 + (s + SB - 1) * NT, box_hi, lane,
                                       pg, mbar_vv0 + j * 8, 3);
                }
              } else {
                tc_tma_kv_pre<NT, D>(P, smem_k[j], &kmap, kvh,
                                     kv0 + (s + SB - 1) * NT, box_hi, lane, pg,
                                     mbar_kv0 + j * 8);
                tc_tma_kv_pre<NT, DV>(P, smem_v[j], &vmap, kvh,
                                      kv0 + (s + SB - 1) * NT, box_hi, lane,
                                      pg, mbar_vv0 + j * 8);
              }
            } else {
              tc_tma_kv_pre<NT, D>(P, smem_k[j], &kmap, kvh,
                                   kv0 + (s + SB - 1) * NT, box_hi, lane, pg,
                                   mbar_kv0 + j * 8);
              tc_tma_kv_pre<NT, DV>(P, smem_v[j], &vmap, kvh,
                                    kv0 + (s + SB - 1) * NT, box_hi, lane, pg,
                                    mbar_vv0 + j * 8);
            }
#else
            WS_TMA(D, smem_k[j], &kmap, s + SB - 1, mbar_kv0 + j * 8)
            WS_TMA(DV, smem_v[j], &vmap, s + SB - 1, mbar_vv0 + j * 8)
#endif
            __syncwarp();
#ifdef FMHA_PHASE
            {
              long long tab = clock64();
              t_box += (unsigned long long)(tab - ta0);
              ta0 = tab;
            }
#endif
          }
#ifdef FMHA_PGDEEP
          // Stream page IDs PGD-1 stages ahead (one cp.async group). Past
          // the tail the macro clamps to duplicate loads, keeping cadence.
          PG_PREFETCH(s + SB + PGD - 2, s + SB + PGD - 2);
#endif
          // (d) arm slot s%SB for stage s+SB after slot s's TMA phases
          // completed (landing; the MMA lead's consumption waits are its own).
          if (s + SB < nstages) {
            mbar_wait(mbar_kv0 + (s % SB) * 8, (s / SB) & 1);
            mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);
            fence_after_sync();
            if constexpr (MC) {
              if (mc_on && rank == 1) {
                // Refill-ready: prove OWN consumption of slot s%SB's stage-s
                // data BEFORE the remote signal - the pv0/pv1(s) commits
                // cover QK0/QK1(s) + PV0/PV1(s) (single-issuer order), the
                // same proof rank0's step-(a) pv gate uses for itself.
                mbar_wait(mbar_pv, s & 1);
                mbar_wait(mbar_pv + 8, s & 1);
                fence_after_sync();
              }
            }
            if (lane == 0) {
              mbar_expect_tx(mbar_kv0 + (s % SB) * 8, NT * D * 2);
              mbar_expect_tx(mbar_vv0 + (s % SB) * 8, NT * DV * 2);
            }
            fence_proxy_async();
            if constexpr (MC) {
              if (mc_on && rank == 1) {
                __syncwarp();
                if (lane == 0) mbar_arrive_remote(peer_r1rd);
                __syncwarp();
              }
            }
          }
#if defined(FMHA_TC_PAGEPRE) && !defined(FMHA_PG_EARLY) && !defined(FMHA_PGDEEP)
          pg = tc_page_pre<NT, D>(P, tabt, kv0 + (s + SB) * NT, box_hi, lane);
#endif
          __syncwarp();
#ifdef FMHA_PHASE
          {
            long long ta1 = clock64();
            t_land += (unsigned long long)(ta1 - ta0);
            ++t_n;
          }
#endif
        }
#ifdef FMHA_PHASE
        if (lane == 0) {
          atomicAdd(&g_ph[22], t_pv);
          atomicAdd(&g_ph[23], t_box);
          atomicAdd(&g_ph[24], t_land);
          atomicAdd(&g_ph[27], t_n);
          atomicAdd(&g_ph[28], 1ull);
        }
#endif
      }
    } else if constexpr (QKEARLY) {
      // Deep-K QK-first pipeline. The tensor pipe currently idles through the
      // whole exp2/pack phase of every stage: with PV(s) issued before QK(s+1)
      // the control warp cannot queue QK(s+1) until pre(s) fires at the END of
      // softmax(s), so c1 runs at 36.2% tensor-pipe duty (ncu3, job
      // dv_256cb185c164) with 57% of its stall cycles in long_scoreboard.
      // Issuing QK(s+1) right after sfree(s) fills that window.
      // QKFIRST tried this and was x1.071 SLOWER because with SB=2 it issued
      // the K(s+1) box in the same iteration that waits on it. Here K(s+SB) is
      // handed to the TMA engine a full iteration earlier, so the box has the
      // whole pre(s)/PV(s) window to land.
      // Slot freedom: sfree(s) proves all 8 softmax warps finished reading
      // S(s), which implies QK(s) completed, so K slot s%SB is dead; and
      // mbar_pv(s-1) proves PV(s-1) stopped reading V(s-1), so V slot
      // (s+1)%SB is dead. Barriers are armed inline immediately before each
      // issue (the previous phase of that slot was waited one iteration back).
      for (int s = 0; s < nstages; ++s) {
        mbar_wait((s & 1) ? mbar_sfree1 : mbar_sfree0, (s >> 1) & 1);
        fence_after_sync();
#ifndef FMHA_QK_LATE_ISSUE
        // NOTE the order: QK(s+1) is issued BEFORE the K(s+SB) box is handed to
        // the TMA engine, because tc_tma_kv resolves the page table with a
        // dependent __ldg first. Putting the box issue between sfree(s) and the
        // QK(s+1) issue puts that load latency (~200-500 cycles on an L1 miss)
        // squarely on the path that feeds the tensor pipe - which is why the
        // first QKEARLY cut measured x0.99 instead of overlapping anything.
        if (s + 1 < nstages) {
          mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8, ((s + 1) / SB) & 1);
          WS_PROXY_FENCE
          fence_after_sync();
          if (lane == 0) {
            const int kb = (s + 1) % SB;
            uint64_t da = qd0, db = tc_desc(su32(smem_k[kb]), 1024, 0);
            _Pragma("unroll")
            for (int ks = 0; ks < D / 16; ++ks) {
              mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);
              da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
              db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
            }
            umma_commit(mbar_qk);
          }
          __syncwarp();
        }
#endif
        if (s + SB < nstages) {
          const int jk = s % SB;
          if (lane == 0) mbar_expect_tx(mbar_kv0 + jk * 8, NT * D * 2);
          WS_PROXY_FENCE
          // The syncwarp between the arm and the box issues is load-bearing:
          // lane0's expect_tx must be warp-ordered before EVERY lane's
          // cp.async.bulk.tensor (fence.proxy.async orders only the fencing
          // lane's own writes). Without it the TMA engine can complete_tx
          // against an unarmed mbarrier -> phase corrupts -> the kv wait
          // releases early and the mma reads half-landed smem. This is the
          // base choreography (arm/fence at step (d), syncwarp, issue next
          // iteration); its absence here is the prime suspect for the
          // measured QKEARLY corruption (dv_682ff24d5c1e/dv_5b3582d28a46:
          // massive nondeterministic out, near-clean lse).
          __syncwarp();
          WS_TMA(D, smem_k[jk], &kmap, s + SB, mbar_kv0 + jk * 8)
          __syncwarp();
        }
        if (s + 1 < nstages) {
          if (s > 0) {
            mbar_wait(mbar_pv, (s - 1) & 1);
            fence_after_sync();
          }
          const int jv = (s + 1) % SB;
          if (lane == 0) mbar_expect_tx(mbar_vv0 + jv * 8, NT * DV * 2);
          WS_PROXY_FENCE
          __syncwarp();   // same arm-before-issue ordering as the K block
          WS_TMA(DV, smem_v[jv], &vmap, s + 1, mbar_vv0 + jv * 8)
          __syncwarp();
        }
        mbar_wait((s & 1) ? mbar_pre1 : mbar_pre0, (s >> 1) & 1);
        fence_after_sync();
        mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);
        WS_PROXY_FENCE
        fence_after_sync();
        if (lane == 0) {
          uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
          const uint32_t pb = tmem_p + (s & 1) * PH;
          _Pragma("unroll")
          for (int st = 0; st < NT / 16; ++st) {
            mma_ts(tmem_o, pb + st * 8, db, IDESC_PV, (st != 0) || (s > 0));
            db += 128;
          }
          umma_commit(mbar_pv);
        }
        __syncwarp();
#ifdef FMHA_QK_LATE_ISSUE
        // BISECTION variant (diagnostic, default off): identical QKEARLY
        // structure - inline arm+issue of both boxes from this warp, V lead
        // of one iteration, wsync arrivals, single-arm V prologue - but the
        // QK(s+1) issue moves AFTER the PV(s) issue, restoring the base
        // pipe order (PV(s) before QK(s+1)) and the base cumulative-commit
        // invariant. Discriminator: if this build is CLEAN while the
        // early-issue build corrupts, the disease is the QK-before-PV pipe
        // overlap itself (QK(s+1)'s TMEM S write in flight during softmax's
        // concurrent tcgen05 P-store/O-rescale window); if it STILL corrupts,
        // the disease is the inline box mechanism. All barrier guards are
        // unchanged: sfree(s) (loop top) frees the K slot, mbar_pv(s-1)
        // (V block) frees the V slot, kv(s+1) proves K residency here.
        if (s + 1 < nstages) {
          mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8, ((s + 1) / SB) & 1);
          WS_PROXY_FENCE
          fence_after_sync();
          if (lane == 0) {
            const int kb = (s + 1) % SB;
            uint64_t da = qd0, db = tc_desc(su32(smem_k[kb]), 1024, 0);
            _Pragma("unroll")
            for (int ks = 0; ks < D / 16; ++ks) {
              mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);
              da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
              db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
            }
            umma_commit(mbar_qk);
          }
          __syncwarp();
        }
#endif
      }
    } else if constexpr (TMALATE) {
      // TMA-after-mma order (see the TMALATE comment at the prologue).
      for (int s = 0; s < nstages; ++s) {
#ifdef FMHA_PHASE
        ++c_n;
#endif
        // (1) PV(s): pre(s) proves P(s) packed + O(s) rescaled; vv(s) proves
        //     V(s) resident (prologue or step (3) of iteration s-2).
        PH_TS(tc0)
        mbar_wait((s & 1) ? mbar_pre1 : mbar_pre0, (s >> 1) & 1);
        fence_after_sync();
        PH_TS(tc1)
        mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);
        fence_proxy_async();
        fence_after_sync();
        PH_TS(tc2)
        if (lane == 0) {
          uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
          const uint32_t pb = tmem_p + (s & 1) * PH;
#pragma unroll
          for (int st = 0; st < NT / 16; ++st) {
            mma_ts(tmem_o, pb + st * 8, db, IDESC_PV, (st != 0) || (s > 0));
            db += 128;
          }
          umma_commit(mbar_pv);
        }
        PH_TS(tc3)
#ifdef FMHA_PHASE
        c_pre += tc1 - tc0; c_vv += tc2 - tc1; c_mma += tc3 - tc2;
#endif
        // (2) QK(s+1): sfree(s) proves S(s) drained (=> QK(s) complete =>
        //     K slot s%SB dead); kv(s+1) was issued one iteration back.
        if (s + 1 < nstages) {
          PH_TS(tb0)
          mbar_wait((s & 1) ? mbar_sfree1 : mbar_sfree0, (s >> 1) & 1);
          fence_after_sync();
          PH_TS(tb1)
          mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8, ((s + 1) / SB) & 1);
          fence_proxy_async();
          fence_after_sync();
          PH_TS(tb2)
          if (lane == 0) {
            const int kb = (s + 1) % SB;
            uint64_t da = qd0, db = tc_desc(su32(smem_k[kb]), 1024, 0);
#pragma unroll
            for (int ks = 0; ks < D / 16; ++ks) {
              mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);
              da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
              db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
            }
            umma_commit(mbar_qk);
          }
          __syncwarp();
          PH_TS(tb3)
#ifdef FMHA_PHASE
          c_sfr += tb1 - tb0; c_kv += tb2 - tb1; c_mma += tb3 - tb2;
#endif
        }
        // (3) Arm + issue stage s+SB into slot s%SB. K half first (its guard,
        //     sfree(s), already passed in (2)); the V half additionally waits
        //     PV(s) complete - the previous occupant V(s) is being read by
        //     the mma issued in (1). That wait gates no mma issue and is
        //     satisfied well before the next iteration's pre wait in steady
        //     state. Arm order is the proven base choreography: lane0
        //     expect_tx -> all-lane proxy fence -> syncwarp -> box issue.
        if (s + SB < nstages) {
          PH_TS(td0)
          if (lane == 0)
            mbar_expect_tx(mbar_kv0 + (s % SB) * 8, NT * D * 2);
          fence_proxy_async();
          __syncwarp();
#ifdef FMHA_TC_PAGEPRE
          tc_tma_kv_pre<NT, D>(P, smem_k[s % SB], &kmap, kvh,
                               kv0 + (s + SB) * NT, ti.hi - 1, lane, pg,
                               mbar_kv0 + (s % SB) * 8);
#else
          WS_TMA(D, smem_k[s % SB], &kmap, s + SB, mbar_kv0 + (s % SB) * 8)
#endif
          mbar_wait(mbar_pv, s & 1);
          fence_after_sync();
          if (lane == 0)
            mbar_expect_tx(mbar_vv0 + (s % SB) * 8, NT * DV * 2);
          fence_proxy_async();
          __syncwarp();
#ifdef FMHA_TC_PAGEPRE
          tc_tma_kv_pre<NT, DV>(P, smem_v[s % SB], &vmap, kvh,
                                kv0 + (s + SB) * NT, ti.hi - 1, lane, pg,
                                mbar_vv0 + (s % SB) * 8);
#else
          WS_TMA(DV, smem_v[s % SB], &vmap, s + SB, mbar_vv0 + (s % SB) * 8)
#endif
          __syncwarp();
#ifdef FMHA_PHASE
          c_tma += clock64() - td0;
#endif
        }
        // (4) Page IDs for stage s+SB+1 (issued by iteration s+1's step (3)):
        //     the table load has the whole next pre-wait window to land.
#ifdef FMHA_TC_PAGEPRE
        pg = tc_page_pre<NT, D>(P, tabt, kv0 + (s + SB + 1) * NT, ti.hi - 1,
                                lane);
#endif
        __syncwarp();
      }
    } else
    for (int s = 0; s < nstages; ++s) {
#ifdef FMHA_PHASE
      ++c_n;
#endif
      // (a) load stage s+SB-1 into slot (s+SB-1)%SB. That slot last held stage
      // s-1, whose V is released by PV(s-1); expect_tx for it was armed by
      // step (d) of iteration s-1 (or by the init block when s == 0).
#ifdef FMHA_TC_PRO2
      // PRO2: stage SB-1 is already in flight from the prologue.
      if (s + SB - 1 < nstages && !QKFIRST && s > 0) {
#else
      if (s + SB - 1 < nstages && !QKFIRST) {
#endif
        if (s > 0) {
          PH_TS(ta0)
          mbar_wait(mbar_pv, (s - 1) & 1);
          fence_after_sync();
#ifdef FMHA_PHASE
          c_pv += clock64() - ta0;
#endif
        }
        const int j = (s + SB - 1) % SB;
        PH_TS(ta1)
#ifdef FMHA_TC_PAGEPRE
        tc_tma_kv_pre<NT, D>(P, smem_k[j], &kmap, kvh, kv0 + (s + SB - 1) * NT,
                             ti.hi - 1, lane, pg, mbar_kv0 + j * 8);
        tc_tma_kv_pre<NT, DV>(P, smem_v[j], &vmap, kvh, kv0 + (s + SB - 1) * NT,
                              ti.hi - 1, lane, pg, mbar_vv0 + j * 8);
#else
        WS_TMA(D, smem_k[j], &kmap, s + SB - 1, mbar_kv0 + j * 8)
        WS_TMA(DV, smem_v[j], &vmap, s + SB - 1, mbar_vv0 + j * 8)
#endif
        __syncwarp();
#ifdef FMHA_PHASE
        c_tma += clock64() - ta1;
#endif
      }
#ifdef FMHA_PG_EARLY
      // EARLY prefetch position: issue stage s+SB's table load right after this
      // iteration's boxes, giving it the whole pready/mma window (~2000 cyc) to
      // land. The LATE position (iteration end, below) leaves only the ~80-200
      // cycle pv-wait between load and use, so an L1 miss still stalls the
      // issue path (phase probe round 2: tma 978 cyc/iter even with PAGEPRE).
      pg = tc_page_pre<NT, D>(P, tabt, kv0 + (s + SB) * NT, ti.hi - 1, lane);
#endif
      // QKFIRST: K(s+SB-1) does NOT need PV(s-1) - its slot's K half was consumed
      // by QK(s-1), which is complete because iteration s-1 waited pre(s-1) and
      // the softmax warps only arrive there after their mbar_qk(s-1) wait. So the
      // K box can be handed to the TMA engine immediately, giving it the whole
      // QK(s)/S-read window to land before step (b) needs it. V(s+SB-1) still
      // has to wait for PV(s-1) to stop reading V(s-1), so it is issued after
      // step (b) below.
      if (s + SB - 1 < nstages && QKFIRST) {
        const int j = (s + SB - 1) % SB;
        WS_TMA(D, smem_k[j], &kmap, s + SB - 1, mbar_kv0 + j * 8)
        __syncwarp();
      }
      // (c) PV(s): P(s) in TMEM, O(s) rescaled, V(s) resident. Then
      // (b) QK(s+1): S(s) drained by all 8 softmax warps, K(s+1) resident.
      // Steps (c) then (b): the order that keeps main8's invariant, see below.
#define WS_STEP_B                                                                  \
      if (s + 1 < nstages) {                                                       \
        PH_TS(tb0)                                                                 \
        mbar_wait((s & 1) ? mbar_sfree1 : mbar_sfree0, (s >> 1) & 1);              \
        fence_after_sync();                                                        \
        PH_TS(tb1)                                                                 \
        mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8, ((s + 1) / SB) & 1);              \
        WS_PROXY_FENCE                                                             \
        fence_after_sync();                                                        \
        PH_TS(tb2)                                                                 \
        if (lane == 0) {                                                           \
          const int kb = (s + 1) % SB;                                             \
          uint64_t da = qd0, db = tc_desc(su32(smem_k[kb]), 1024, 0);              \
          _Pragma("unroll")                                                        \
          for (int ks = 0; ks < D / 16; ++ks) {                                    \
            mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);                             \
            da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;                              \
            db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;                              \
          }                                                                        \
          umma_commit(mbar_qk);                                                    \
        }                                                                          \
        __syncwarp();                                                              \
        PH_TS(tb3)                                                                 \
        WS_ACCB                                                                    \
      }
#define WS_STEP_C                                                                  \
      PH_TS(tc0)                                                                   \
      mbar_wait((s & 1) ? mbar_pre1 : mbar_pre0, (s >> 1) & 1);                    \
      fence_after_sync();                                                          \
      PH_TS(tc1)                                                                   \
      mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);                            \
      WS_PROXY_FENCE                                                               \
      fence_after_sync();                                                          \
      PH_TS(tc2)                                                                   \
      if (lane == 0) {                                                             \
        uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);                  \
        const uint32_t pb = tmem_p + (s & 1) * PH;                                 \
        _Pragma("unroll")                                                          \
        for (int st = 0; st < NT / 16; ++st) {                                     \
          mma_ts(tmem_o, pb + st * 8, db, IDESC_PV, (st != 0) || (s > 0));         \
          db += 128;                                                               \
        }                                                                          \
        umma_commit(mbar_pv);                                                      \
      }                                                                            \
      PH_TS(tc3)                                                                   \
      WS_ACCC
      // PV(s) BEFORE QK(s+1), and that order is load-bearing rather than a
      // preference: tcgen05.commit is cumulative, so the mbar_qk commit that
      // releases the softmax warps into stage s+1 covers every mma issued before
      // QK(s+1). Issuing QK(s+1) first leaves PV(s) outside that commit and the
      // warps then run stage s+1 concurrently with PV(s) - measured as a
      // nondeterministic out-only corruption on fixed c7 (mm=17..1821 per run,
      // lse always clean, one block's rows at a time) that survived parity
      // double-buffered P, parity double-buffered rm, parity-indexed sfree/
      // pready pairs, a softmax lock-step barrier and the proxy fence
      // (job20/21/22/24), and disappeared 3/3 the moment PV(s) went first
      // (job24 arm d1: mm=0,0,0 at 234.0us vs 229.0us for the racy order and
      // 249.3us for main8). It is the invariant main8 has always relied on.
      if constexpr (QKFIRST) {
        // QK(s+1) first: it only needs S(s) drained (sfree) and K(s+1) resident,
        // both true long before P(s) is packed, so the tensor pipe has QK(s+1)
        // queued and executes it DURING stage s's exp2/pack instead of idling.
        // Safe because P is parity double buffered (softmax(s+1) writes the other
        // half, and its release commit - cumulative over QK(s+1) - already covers
        // PV(s-1), the previous reader of that half) and because the O rescale in
        // softmax(s+1) explicitly waits mbar_pv parity s&1, i.e. PV(s).
        WS_STEP_B
        if (s + SB - 1 < nstages) {
          if (s > 0) {
            mbar_wait(mbar_pv, (s - 1) & 1);
            fence_after_sync();
          }
          const int jv = (s + SB - 1) % SB;
          WS_TMA(DV, smem_v[jv], &vmap, s + SB - 1, mbar_vv0 + jv * 8)
          __syncwarp();
        }
        WS_STEP_C
      } else {
        WS_STEP_C
        WS_STEP_B
      }
#undef WS_STEP_B
#undef WS_STEP_C
#undef WS_ACCB
#undef WS_ACCC
#undef WS_PROXY_FENCE
#undef WS_TMA
      // (d) arm slot s%SB for stage s+SB. That slot's stage-s TMAs have both
      // completed - V(s) was waited in (c) just above and K(s) in (b) of
      // iteration s-1 (or in the prologue when s == 0) - so arming the next
      // phase is safe, and the issue itself (step (a) of iteration s+1) still
      // waits PV(s) before overwriting the slot.
      // The proxy fence publishes the expect_tx to the TMA engine before step
      // (a) of the next iteration issues into it (same order main8 uses:
      // arm -> fence.proxy.async -> sync -> issue).
      if (s + SB < nstages) {
        PH_TS(td0)
        if (lane == 0) {
          mbar_expect_tx(mbar_kv0 + (s % SB) * 8, NT * D * 2);
          mbar_expect_tx(mbar_vv0 + (s % SB) * 8, NT * DV * 2);
        }
        fence_proxy_async();
#ifdef FMHA_PHASE
        c_tma += clock64() - td0;
#endif
      }
#if defined(FMHA_TC_PAGEPRE) && !defined(FMHA_PG_EARLY)
      // Resolve stage s+SB's page IDs here: iteration s+1's step (a) issues
      // those boxes, so the table load has a full control iteration (the
      // pready/kv waits + the whole mma issue/drain window) to land.
      pg = tc_page_pre<NT, D>(P, tabt, kv0 + (s + SB) * NT, ti.hi - 1, lane);
#endif
      __syncwarp();
    }
#ifdef FMHA_PHASE
    if (lane == 0) {
      atomicAdd(&g_ph[7], (unsigned long long)c_pv);
      atomicAdd(&g_ph[8], (unsigned long long)c_tma);
      atomicAdd(&g_ph[9], (unsigned long long)c_pre);
      atomicAdd(&g_ph[10], (unsigned long long)c_vv);
      atomicAdd(&g_ph[11], (unsigned long long)c_sfr);
      atomicAdd(&g_ph[12], (unsigned long long)c_kv);
      atomicAdd(&g_ph[13], (unsigned long long)c_mma);
      atomicAdd(&g_ph[14], (unsigned long long)c_n);
      atomicAdd(&g_ph[15], 1ull);
    }
#endif
  } else {
    // ---- softmax warps 0-7 ----
    const int rg = ti.m0 + row;
    const bool valid = row < ti.mrows;
    const int q_idx = min(rg / P.group, ti.qlen - 1);
    const int head = kvh * P.group + (rg % P.group);
    const int vis = ti.kvlen - ti.qlen + rg / P.group;
    const int rhi = min(P.causal ? vis : ti.kvlen - 1, kv1 - 1);
    const int rlo = max((P.window_left >= 0) ? max(0, vis - P.window_left) : 0,
                        kv0_exact);
    float mbase = NEG_INF, lt = 0.f;
    const bool wlead = ((tid & 31) == 0);
#ifdef FMHA_DH2_PEARLY
    constexpr bool PEARLY = (DH == 2);   // M2-lite early P chunk (DH2 only)
#else
    constexpr bool PEARLY = false;
#endif
#ifdef FMHA_PHASE
    long long p_qk = 0, p_srd = 0, p_rmx = 0, p_exp = 0, p_res = 0, p_n = 0;
#endif
    for (int s = 0; s < nstages; ++s) {
      PH_TS(pa0)
      mbar_wait(mbar_qk + dhalf * 8, s & 1);
      fence_after_sync();
      PH_TS(pa1)
      float sreg[HC];
#ifdef FMHA_TC_MAXROW
      // Read the sibling half's columns too, purely to finish the row max
      // locally: that removes the STS + 256-thread named barrier + LDS round
      // trip from the critical path of every stage (c1 spends 57% of its stall
      // cycles in barrier/mbarrier waits, and its softmax phase - not the tensor
      // pipe - sets the period). Both loads issue back to back, so one
      // tcgen05.wait::ld covers them. Cost: +HC registers (dead right after the
      // max) and 2x the S read traffic, against a TMEM pipe that NCU puts at
      // 42% duty.
      float oreg[HC];
#pragma unroll
      for (int c = 0; c < HC / 32; ++c) {
        tmem_ld32(sreg + c * 32, tmem_s + half * HC + c * 32);
        tmem_ld32(oreg + c * 32, tmem_s + (half ^ 1) * HC + c * 32);
      }
      tmem_wait_ld();
#else
#ifdef FMHA_LDRED
      float red_max = NEG_INF;
      if constexpr (NH == 1 && NT == 64) {
        // M6: one fused x64 load-reduce for the NH1 row-split (half==0;
        // the row's 64 columns are contiguous from the S base). NT==64
        // guard: the x64 form covers exactly one 64-column row; wider
        // NT (NH1-NT128 d64 experiment) takes the tree path below.
        tmem_ld64_red_max(sreg, red_max, tmem_s + dhalf * S_COLS);
      } else {
#pragma unroll
        for (int c = 0; c < HC / 32; ++c)
          tmem_ld32(sreg + c * 32, tmem_s + half * HC + c * 32);
      }
      tmem_wait_ld();
#else
#pragma unroll
      for (int c = 0; c < HC / 32; ++c)
        tmem_ld32(sreg + c * 32, tmem_s + dhalf * S_COLS + half * HC + c * 32);
      tmem_wait_ld();
#endif
#endif
      PH_TS(pa2)
      // Every lane fences its own prior tcgen05 ops (tmem_wait_ld above already
      // completed them); lane 0 then publishes the arrival for the whole warp.
      fence_before_sync();
#ifdef FMHA_WS_WSYNC
      // tmem_wait_ld is PER-LANE: without a warp execution rendezvous, lane 0
      // can publish the arrival while another lane's tcgen05.ld is still in
      // flight, and the consumer (QK(s+1) under QKEARLY, or the K-slot TMA
      // refill) may overwrite TMEM/smem mid-read. The base choreography hides
      // this behind hundreds of cycles of ctrl-side waits; QKEARLY shrinks the
      // window to ~0 and corrupts scores nondeterministically (dv_682ff24d5c1e
      // / dv_5b3582d28a46 / dv_fcf90870887d signature: S-side garbage, worse
      // on long loops and last-wave CTAs). One warp barrier per arrival closes
      // it; bitwise identical on every path.
      __syncwarp();
#endif
      if (wlead) mbar_arrive(mbar_sfree0 + (dhalf * 2 + (s & 1)) * 8);
      const int pbase = kv0 + s * NT + half * HC;
      const bool interior = (pbase >= rlo) && (pbase + HC - 1 <= rhi);
      float rmax_raw = NEG_INF;
      // Branched, not predicated (see fmha_tc_main): the d128/d256 warp-
      // specialized tiles run the same interior majority, and this loop is the
      // hottest code in the whole candidate (c1/c6/c7/c9).
#ifdef FMHA_MASK_PRED
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        float x = sreg[i];
        if (!interior) {
          const int c = pbase + i;
          if (c < rlo || c > rhi) x = NEG_INF;
        }
        sreg[i] = x;
        rmax_raw = fmaxf(rmax_raw, x);
      }
#else
      if (interior) {
#ifdef FMHA_LDRED
        if constexpr (NH == 1 && NT == 64) {
          // M6: the row max arrived fused with the S load - the whole
          // 63-fmax tree disappears from the critical path (rmx slot
          // measured 244/195 cyc/stage on c1/c6, probe dv_27d51ac1b73f;
          // 467 incl. exchange on c0-NH4, dv_fc8d4bb9a57c).
          rmax_raw = red_max;
        } else {
#endif
#if defined(FMHA_CHAIN1) && !defined(FMHA_PACK2_RMAX)
        if constexpr (NH == 1) {
          // Row-split: the local chain is now 64 deep and, with the exchange
          // gone, fully exposed on the critical path - the 4-way tree cuts it
          // to depth ~19 at identical slot count (bitwise: fmaxf is exact and
          // associative, NaN semantics included).
          float mx0 = NEG_INF, mx1 = NEG_INF, mx2 = NEG_INF, mx3 = NEG_INF;
#pragma unroll
          for (int i = 0; i < HC; i += 4) {
            mx0 = fmaxf(mx0, sreg[i + 0]);
            mx1 = fmaxf(mx1, sreg[i + 1]);
            mx2 = fmaxf(mx2, sreg[i + 2]);
            mx3 = fmaxf(mx3, sreg[i + 3]);
          }
          rmax_raw = fmaxf(fmaxf(mx0, mx1), fmaxf(mx2, mx3));
        } else {
#pragma unroll
          for (int i = 0; i < HC; ++i) rmax_raw = fmaxf(rmax_raw, sreg[i]);
        }
#else
        float mx0 = NEG_INF, mx1 = NEG_INF, mx2 = NEG_INF, mx3 = NEG_INF;
#pragma unroll
        for (int i = 0; i < HC; i += 4) {
          mx0 = fmaxf(mx0, sreg[i + 0]);
          mx1 = fmaxf(mx1, sreg[i + 1]);
          mx2 = fmaxf(mx2, sreg[i + 2]);
          mx3 = fmaxf(mx3, sreg[i + 3]);
        }
        rmax_raw = fmaxf(fmaxf(mx0, mx1), fmaxf(mx2, mx3));
#endif
#ifdef FMHA_LDRED
        }
#endif
      } else {
#pragma unroll
        for (int i = 0; i < HC; ++i) {
          const int c = pbase + i;
          float x = sreg[i];
          if (c < rlo || c > rhi) x = NEG_INF;
          sreg[i] = x;
          rmax_raw = fmaxf(rmax_raw, x);
        }
      }
#endif
#ifdef FMHA_TC_MAXROW
      {
        const int opbase = kv0 + s * NT + (half ^ 1) * HC;
        const bool oint = (opbase >= rlo) && (opbase + HC - 1 <= rhi);
        float omax = NEG_INF;
        if (oint) {
#pragma unroll
          for (int i = 0; i < HC; ++i) omax = fmaxf(omax, oreg[i]);
        } else {
#pragma unroll
          for (int i = 0; i < HC; ++i) {
            const int c = opbase + i;
            omax = fmaxf(omax, (c < rlo || c > rhi) ? NEG_INF : oreg[i]);
          }
        }
        rmax_raw = fmaxf(rmax_raw, omax);
      }
      const float rmax = rmax_raw;
#else
      float rmax;
      if constexpr (NH == 1) {
        // Row-split softmax (NH=1): each warp owns whole rows (32 rows x all
        // NT columns), so the row max is lane-local - no smem exchange, no
        // 256-thread barrier, no skew absorption. This is the vendor d128
        // softmax ownership; the column-split NH2 pays 441-559 cyc/stage for
        // the exchange (phase probe dv_a7e8cbe32362).
        rmax = rmax_raw;
      } else {
        float* const rmb = rm + (s & 1) * (NH * 128);
        rmb[half * 128 + row] = rmax_raw;
#ifdef FMHA_TC_PAIRMAX
        // Pair-scoped exchange (see the same knob in fmha_tc_main8mc): sync
        // only the NH warps sharing this row group (bar id 1+warp%4, NH*32
        // threads) instead of all SW softmax threads - each group absorbs its
        // own skew, not the slowest warp's. Bitwise identical values.
        asm volatile("bar.sync %0, %1;" :: "r"(1 + ((tid >> 5) & 3)),
                     "n"(NH * 32) : "memory");
#else
        bar_softmax_t<SW>();
#endif
        rmax = rmb[row];
#pragma unroll
        for (int q = 1; q < NH; ++q) rmax = fmaxf(rmax, rmb[q * 128 + row]);
      }
#endif
      PH_TS(pa3)
      const float mn = fmaxf(mbase, rmax * P.qk_scale2);
      const float msafe = (mn == NEG_INF) ? 0.f : mn;
      const bool skip = (msafe - mbase) < P.lazy_thresh;
      const float alpha = skip ? 1.f : exp2_fast(mbase - msafe);
      const float nqk = P.qk_scale2;
      const float npb = -(skip ? mbase : msafe);
      if (!skip) { lt *= alpha; mbase = msafe; }
#ifdef FMHA_CHAIN1
      float lsum = 0.f;
#ifdef FMHA_EXP_FFMA2
      // Packed arg-FMA (bitwise identical: fma.rn.f32x2 == two __fmaf_rn).
      // Halves the FFMA issue slots of the hot exp loop; lsum accumulation
      // ORDER is untouched (still p0,p1,p2,... serial) so lsum is bitwise
      // identical too. Mechanism per the cudnn_fe_oss sm100 reference.
      const float2 nqk2 = make_float2(nqk, nqk);
      const float2 npb2 = make_float2(npb, npb);
#ifdef FMHA_PACK2_LSUM
      // PACK2 lsum: two float2 accumulators (16 fadd2 + 3 scalar adds)
      // replace the 32 serial scalar FADDs - slot count ~halved again and
      // the accumulator chain depth drops 64 -> 16. The pairing order
      // changes lsum rounding at the ~1ulp level (fp32 against a 5e-2
      // relative tolerance): TOLERANCE-level, not bitwise.
      // E2E (vendor MUFU/FMA-pipe balancing, see exp2_emul2): a fixed
      // interleaved subset of pairs takes the packed-polynomial path so
      // ex2 MUFU throughput and FMA/ALU issue run concurrently.
      float2 acc0 = make_float2(0.f, 0.f), acc1 = make_float2(0.f, 0.f);
#pragma unroll
      for (int i = 0; i < HC; i += 4) {
        const float2 a0 = ffma2(make_float2(sreg[i], sreg[i + 1]), nqk2, npb2);
        const float2 a1 = ffma2(make_float2(sreg[i + 2], sreg[i + 3]), nqk2, npb2);
#ifdef FMHA_E2E
        float2 e0, e1;
        if (E2E_EMUL(i)) {
          e0 = exp2_emul2(a0);
          e1 = exp2_emul2(a1);
        } else {
          e0 = make_float2(exp2_fast(a0.x), exp2_fast(a0.y));
          e1 = make_float2(exp2_fast(a1.x), exp2_fast(a1.y));
        }
        const float p0 = e0.x, p1 = e0.y, p2 = e1.x, p3 = e1.y;
#else
        const float p0 = exp2_fast(a0.x);
        const float p1 = exp2_fast(a0.y);
        const float p2 = exp2_fast(a1.x);
        const float p3 = exp2_fast(a1.y);
#endif
        acc0 = fadd2(acc0, make_float2(p0, p1));
        acc1 = fadd2(acc1, make_float2(p2, p3));
        sreg[i + 0] = p0; sreg[i + 1] = p1; sreg[i + 2] = p2; sreg[i + 3] = p3;
      }
      lsum = (acc0.x + acc0.y) + (acc1.x + acc1.y);
#else
#pragma unroll
      for (int i = 0; i < HC; i += 2) {
        const float2 a2 = ffma2(make_float2(sreg[i], sreg[i + 1]), nqk2, npb2);
        const float p0 = exp2_fast(a2.x);
        const float p1 = exp2_fast(a2.y);
        lsum += p0;
        sreg[i] = p0;
        lsum += p1;
        sreg[i + 1] = p1;
      }
#endif
#else
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        const float p = exp2_fast(fmaf(sreg[i], nqk, npb));
        lsum += p;
        sreg[i] = p;
      }
#endif
#else
      float ls0 = 0.f, ls1 = 0.f, ls2 = 0.f, ls3 = 0.f;
#pragma unroll
      for (int i = 0; i < HC; i += 4) {
        const float p0 = exp2_fast(fmaf(sreg[i + 0], nqk, npb));
        const float p1 = exp2_fast(fmaf(sreg[i + 1], nqk, npb));
        const float p2 = exp2_fast(fmaf(sreg[i + 2], nqk, npb));
        const float p3 = exp2_fast(fmaf(sreg[i + 3], nqk, npb));
        ls0 += p0; ls1 += p1; ls2 += p2; ls3 += p3;
        sreg[i + 0] = p0; sreg[i + 1] = p1; sreg[i + 2] = p2; sreg[i + 3] = p3;
      }
      const float lsum = (ls0 + ls1) + (ls2 + ls3);
#endif
#pragma unroll
      for (int c = 0; c < (PEARLY ? 1 : HC / 32); ++c) {
        uint32_t pw[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
          pw[j] = pack_bf16(sreg[c * 32 + 2 * j], sreg[c * 32 + 2 * j + 1]);
        tmem_st16(tmem_p + dhalf * P_COLS + (s & 1) * PH + half * (HC / 2) + c * 16, pw);
      }
      tmem_wait_st();
      PH_TS(pa4)
      // Lazy O rescale: only touch O (and only wait for PV(s-1)) when some row
      // in this warp actually exceeds the threshold.
      // With PV(s) issued before QK(s+1), the mbar_qk wait at the top of this
      // stage already implies PV(s-1) completed, so this wait is normally
      // satisfied on arrival; it stays as the explicit guard for the O read.
      if (s > 0 && __any_sync(0xffffffffu, alpha < 1.0f)) {
        mbar_wait(mbar_pv + dhalf * 8, (s - 1) & 1);
        fence_after_sync();
        // O is rescaled in 32-column TMEM units split across the NH groups;
        // when DV < NH*32 (d64, NH=4) the trailing groups skip - pready still
        // covers every unit because the low groups arrive only after theirs.
        constexpr int OU = DV / 32;               // 32-col units in O
        constexpr int UPC = (OU + NH - 1) / NH;   // units per group
        const int u0 = half * UPC;
#ifdef FMHA_PACK2_MUL
        const float2 alpha2 = make_float2(alpha, alpha);
#endif
#pragma unroll
        for (int c = 0; c < UPC; ++c) {
          if (u0 + c < OU) {
            float oreg[32];
            const uint32_t oa = tmem_o + dhalf * DV + (u0 + c) * 32;
            tmem_ld32(oreg, oa);
            tmem_wait_ld();
#ifdef FMHA_PACK2_MUL
            // PACK2 rescale: 16 fmul2 replace 32 scalar FMULs (slot halving;
            // per-lane RN so divergence from the ftz scalar lowering is
            // denormal-only -> tolerance-level).
#pragma unroll
            for (int i = 0; i < 32; i += 2) {
              const float2 o2 = fmul2(make_float2(oreg[i], oreg[i + 1]), alpha2);
              oreg[i] = o2.x;
              oreg[i + 1] = o2.y;
            }
#else
#pragma unroll
            for (int i = 0; i < 32; ++i) oreg[i] *= alpha;
#endif
            tmem_st32(oa, oreg);
          }
        }
        tmem_wait_st();
      }
      lt += lsum;                       // per-half accumulator, no exchange
#ifdef FMHA_DH2_PEARLY
      if constexpr (PEARLY) {
        // M2-lite early-P publish: chunk c=0 (keys 0-31) is stored and this
        // stage's lazy O rescale has completed -> the MMA-lead may start
        // PV_h's first sub-step pair while chunk c=1 packs. The rescale's
        // position BEFORE this arrival is what keeps the early PV
        // accumulation from overlapping the O writes (the reference fuses
        // the same two conditions into its p_full barrier).
        fence_before_sync();
        __syncwarp();
        if (wlead) mbar_arrive(mbar_prearly0 + (dhalf * 2 + (s & 1)) * 8);
#pragma unroll
        for (int c = 1; c < HC / 32; ++c) {
          uint32_t pw[16];
#pragma unroll
          for (int j = 0; j < 16; ++j)
            pw[j] = pack_bf16(sreg[c * 32 + 2 * j], sreg[c * 32 + 2 * j + 1]);
          tmem_st16(tmem_p + dhalf * P_COLS + (s & 1) * PH + half * (HC / 2) + c * 16, pw);
        }
        tmem_wait_st();
      }
#endif  // FMHA_DH2_PEARLY
      fence_before_sync();
#ifdef FMHA_WS_WSYNC
      // See the sfree-arrival comment: per-lane tmem_wait_st does not order
      // OTHER lanes' P stores before lane 0's arrival; PV(s) can read P
      // mid-store (historical QKFIRST signature: massive nondeterministic
      // out with near-clean lse - exactly a corrupt-P, clean-softmax state).
      __syncwarp();
#endif
      if (wlead) mbar_arrive(mbar_pre0 + (dhalf * 2 + (s & 1)) * 8);
#ifdef FMHA_PHASE
      {
        PH_TS(pa5)
        p_qk += pa1 - pa0; p_srd += pa2 - pa1; p_rmx += pa3 - pa2;
        p_exp += pa4 - pa3; p_res += pa5 - pa4; ++p_n;
      }
#endif
    }
#ifdef FMHA_PHASE
    if ((tid & 31) == 0) {
      atomicAdd(&g_ph[0], (unsigned long long)p_qk);
      atomicAdd(&g_ph[1], (unsigned long long)p_srd);
      atomicAdd(&g_ph[2], (unsigned long long)p_rmx);
      atomicAdd(&g_ph[3], (unsigned long long)p_exp);
      atomicAdd(&g_ph[4], (unsigned long long)p_res);
      atomicAdd(&g_ph[5], (unsigned long long)p_n);
      atomicAdd(&g_ph[6], 1ull);
    }
#endif

    // ---- epilogue (warps 0-3 write out; they cover all 128 rows) ----
    mbar_wait(mbar_pv + dhalf * 8, (nstages - 1) & 1);
    fence_after_sync();
    float* const ltx = rm + NH * 128;    // post-loop reuse of the odd rm block
    if (half >= 1) ltx[(half - 1) * 128 + row] = lt;
    bar_softmax_t<SW * DH>();
    if (half == 0) {
#pragma unroll
      for (int q = 1; q < NH; ++q) lt += ltx[(q - 1) * 128 + row];
      const int64_t Pb = ((int64_t)tile * P.kvh + kvh) * P.splits_ub + split;
      const int cu = P.cu_q[ti.b];
      const int64_t tok = cu + q_idx;
      // TAIL-split mode (_pad2 bit30): tiles the device chunker left
      // UNSPLIT (active == 1) write their final out/lse directly instead of
      // bouncing fp32 partials through part_o + a combine copy. Bitwise
      // identical: combine on a single split computes mm=m, ll=l*1,
      // inv=v_scale/l (+sinks), out=bf16(part_o*inv), lse=m+log2(l) - the
      // exact expressions and fp32 values of the direct branch.
      if (P.splits_ub > 1
          && !((P._pad2 & (1 << 30)) && active == 1)) {
        float* po = P.part_o + (Pb * P.m_tile + row) * DV;
        const int64_t blk = (int64_t)tile * P.kvh + kvh;
        const int Spad = (P.splits_ub + 3) & ~3;
        float* pm = P.part_ml + ((blk * P.m_tile + row) * 2) * Spad + split;
#pragma unroll
        for (int c = 0; c < DV / 32; ++c) {
          float oreg[32];
          tmem_ld32(oreg, tmem_o + dhalf * DV + c * 32);
          tmem_wait_ld();
#ifdef FMHA_SCALAR_PO
#pragma unroll
          for (int i = 0; i < 32; ++i) po[c * 32 + i] = oreg[i];
#else
          // float4 stores: DV, c*32 and i are multiples of 4 and part_o
          // comes from a 256 B-aligned allocation, so every row offset is
          // 16 B aligned. The scalar form cost STG 5.0% of all stall
          // samples on c4 (splits=2, 64 stores per thread per tile).
#pragma unroll
          for (int i = 0; i < 32; i += 4)
            *reinterpret_cast<float4*>(po + c * 32 + i) =
                make_float4(oreg[i], oreg[i + 1], oreg[i + 2], oreg[i + 3]);
#endif
        }
        if (valid) { pm[0] = mbase; pm[Spad] = lt; }
        cudaTriggerProgrammaticLaunchCompletion();
      } else {
        float lse2 = mbase + log2f(lt);
        float inv = P.v_scale / lt;
        if (P.has_sinks) {
          float os;
          lse2 = sink_adjust(lse2, P.sinks[head * P.sink_stride] * P.log2e, &os);
          inv *= os;
        }
        __nv_bfloat16* orow =
            P_OUT + tok * P.out_tok_stride + (int64_t)head * P.out_head_stride;
#pragma unroll
        for (int c = 0; c < DV / 32; ++c) {
          float oreg[32];
          tmem_ld32(oreg, tmem_o + dhalf * DV + c * 32);
          tmem_wait_ld();
          uint32_t packed[16];
#pragma unroll
          for (int i = 0; i < 16; ++i)
            packed[i] = pack_bf16(oreg[2 * i] * inv, oreg[2 * i + 1] * inv);
          if (valid) {
#pragma unroll
            for (int u = 0; u < 4; ++u)
              *reinterpret_cast<uint4*>(orow + c * 32 + u * 8) =
                  *reinterpret_cast<uint4*>(packed + u * 4);
          }
        }
        if (valid && P.has_lse) P.lse[tok * P.lse_tok_stride + head] = lse2;
      }
    }
  }
  fence_before_sync();
  __syncthreads();
  if (tid < 32) tmem_dealloc(tmem, TC_COLS);
}

// ---------------------------------------------------------------------------
// Multicast-pair helpers. Design notes:
// Why: under restore-l2-evict-event-v2 every timed forward starts with a cold
// L2 (512 MB RMW eviction, engine cache_policy.py), and the big prefills are
// DRAM re-stream bound - c1's 755.8 us equals ~3.6 GB of effective tile-prefix
// re-reads (bw_probe dv_2406f2e05fae: cold read roof ~4.6-5.2 TB/s, L2
// 126.5 MB) against ~350 us of warm-L2 device time, and the S5 gridorder arm
// (dv_2406f2e05fae) proved scheduling alone cannot capture the reuse (x1.14
// geomean WORSE: the kvh-fastest wave order is load-bearing for LPT and DRAM
// page locality). Pairing adjacent m-tiles in a 2-CTA cluster and fetching
// each KV stage ONCE via cp.async.bulk.tensor multicast halves the re-reads
// in hardware, at the unchanged per-CTA cta_group::1 mma rate and occupancy.
// Arm choreography extends the proven base pattern (lane0 expect_tx ->
// all-lane proxy fence -> syncwarp -> box issue) with rank0 arming rank1's
// slot barriers remotely via mapa + .shared::cluster expect_tx, and a
// cluster-scope proxy fence before issuing.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tma_load_4d_mc(const void* map, char* dst_smem,
                                               int c0, int c1, int c2, int c3,
                                               uint32_t mbar, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes.multicast::cluster [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
      :: "r"(su32(dst_smem)), "l"(reinterpret_cast<uint64_t>(map)),
         "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar), "h"(mask)
      : "memory");
}
__device__ __forceinline__ uint32_t mapa_u32(uint32_t local, uint32_t rank) {
  uint32_t r;
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local), "r"(rank));
  return r;
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void mbar_expect_tx_cluster(uint32_t addr,
                                                       uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
               :: "r"(addr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
               :: "r"(addr) : "memory");
}
__device__ __forceinline__ void fence_proxy_cluster() {
  asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;");
}
__device__ __forceinline__ void cluster_sync_mc() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Page-prefetched multicast box issue (lane-parallel like tc_tma_kv_pre).
template <int NT, int D>
__device__ __forceinline__ void tc_tma_kv_mc(const FmhaParams& P, char* dst,
                                             const void* map, int kvh,
                                             int kv_base, int last, int lane,
                                             int32_t page, uint32_t mbar,
                                             uint16_t mask) {
  constexpr int ATOMS = D / 64;
  const int R = min(NT, 1 << P.page_log);
  const int OPS = (NT / R) * ATOMS;
  if (lane < OPS) {
    const int seg = lane / ATOMS, a = lane % ATOMS;
    const int p = min(kv_base + seg * R, last);
    tma_load_4d_mc(map, dst + a * (NT * 128) + seg * R * 128, a * 64,
                   p & P.page_mask, kvh, page, mbar, mask);
  }
}

// ---------------------------------------------------------------------------
// Multicast-pair prefill: see the block comment above the forward declaration.
// Per-CTA structure is main8ws (base PV-before-QK order, parity P/rm, sfree/
// pre pairs); the differences are the padded-y cluster grid, the shared stage
// count, rank0's multicast refill with remote arms, and rank1's two mirror
// arrivals. The solo fallback (mc_on false) runs the same loop with local
// arms and unicast boxes on its own tile length.
// ---------------------------------------------------------------------------
template <int D, int NT, int SB, int MINB>
__global__ void __launch_bounds__(288, MINB) __cluster_dims__(1, 2, 1)
fmha_tc_main8mc(
    const FmhaParams P, const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap) {
  static_assert((D == 128 || D == 256) && NT == 64 && SB == 2,
                "mc kernel is instantiated for d128/d256 NT64 SB2 only");
  constexpr int NH = 2;
  constexpr int SW = NH * 128;
  extern __shared__ __align__(1024) char smem[];
  constexpr int DV = D;
  constexpr int Q_BYTES = 128 * D * 2;
  constexpr int KV_BYTES = NT * D * 2;
  constexpr int S_COLS = NT, P_COLS = NT;   // two P stage buffers (parity)
  constexpr int PH = NT / 2;                // columns per P buffer
  constexpr int HC = NT / NH;               // S columns per softmax group
  constexpr uint32_t TC_COLS = (S_COLS + P_COLS + DV <= 256) ? 256 : 512;
  char* smem_q = smem;
  char* smem_k[SB];
  char* smem_v[SB];
#pragma unroll
  for (int j = 0; j < SB; ++j) {
    smem_k[j] = smem + Q_BYTES + j * 2 * KV_BYTES;
    smem_v[j] = smem_k[j] + KV_BYTES;
  }
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + Q_BYTES + SB * 2 * KV_BYTES);
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(mbar + 2);
  // rm: [2 parity][NH][128]; the first (NH-1)*128 floats of the odd block are
  // reused for the one-shot epilogue lt exchange (post-loop, no time overlap).
  float* rm = reinterpret_cast<float*>(tmem_slot + 8);
  const uint32_t mbar_qk = su32(mbar), mbar_pv = su32(mbar + 1);
  const uint32_t mbar_kv0 = su32(mbar) + 48 + NH * 1024;
  const uint32_t mbar_vv0 = mbar_kv0 + SB * 8;
  const uint32_t mbar_sfree0 = mbar_vv0 + SB * 8;  // even stages: S drained
  const uint32_t mbar_sfree1 = mbar_sfree0 + 8;    // odd stages
  const uint32_t mbar_pre0 = mbar_sfree1 + 8;      // even: P packed + rescaled
  const uint32_t mbar_pre1 = mbar_pre0 + 8;        // odd
  const uint32_t mbar_r1rd = mbar_pre1 + 8;        // rank0: peer refill-ready

  const int tid = threadIdx.x;
  const bool ctrl = (tid >= SW);
  const int lane = tid - SW;                    // control-warp lane
  const int half = (tid >> 7) & (NH - 1);       // softmax: S column group
  const int row = tid & 127;                    // softmax: TMEM lane == m row

  const uint32_t rank = cluster_rank();         // == blockIdx.y & 1
  const int kvh = (int)blockIdx.x;
  const int tile_y = (int)blockIdx.y;
  const bool rev = ((P._pad2 >> 1) & 1) && P.causal;
  const bool dead_pad = (tile_y >= P.tiles_ub);
  const int tile = rev ? (P.tiles_ub - 1 - tile_y) : tile_y;
  const int peer_y = tile_y ^ 1;
  const bool peer_pad = (peer_y >= P.tiles_ub);
  const int peer_tile = rev ? (P.tiles_ub - 1 - peer_y) : peer_y;
  const TileInfo ti = dead_pad ? TileInfo{0, 0, 0, 0, 0, 0, 0, false}
                               : map_tile(P, tile);
  const TileInfo tp = peer_pad ? TileInfo{0, 0, 0, 0, 0, 0, 0, false}
                               : map_tile(P, peer_tile);
  // Symmetric pairing decision (both ranks compute the same value from the
  // same inputs - no communication, no divergent cluster paths). Defensive
  // gates mirror the host: pairing is only valid with lo == 0 (no window,
  // splits == 1) so both tiles start at kv0 and share one stage grid.
  const bool mc_on = ti.ok && tp.ok && (ti.b == tp.b)
                     && P.window_left < 0 && P.splits_ub == 1;
  if (!ti.ok) return;   // dead slack tile: peer computes mc_on == false too
  int kv0, kv1, active;
  if (!split_range(P, ti, 0, kv0, kv1, active)) return;
  const int kv0_exact = kv0;
  kv0 &= ~(NT - 1);            // host gate: window<0, splits==1 -> kv0 == 0
  // Shared stage geometry: the pair walks ceil(max(hi)/NT) stages; the
  // shorter tile's extra stages are fully masked by its rhi (exact).
  const int shared_hi = mc_on ? max(kv1, tp.hi) : kv1;
  const int nstages = (shared_hi - kv0 + NT - 1) / NT;
  if (nstages <= 0) return;
  const int last_mc = shared_hi - 1;
  const int32_t* const tabt = P.table + ti.b * P.tstride;

  if (tid == 0) {
    mbar_init(mbar_qk, 1); mbar_init(mbar_pv, 1);
    mbar_init(mbar_sfree0, NH * 4); mbar_init(mbar_sfree1, NH * 4);
    mbar_init(mbar_pre0, NH * 4); mbar_init(mbar_pre1, NH * 4);
    mbar_init(mbar_r1rd, 1);
    for (int j = 0; j < SB; ++j) {
      mbar_init(mbar_kv0 + j * 8, 1);
      mbar_init(mbar_vv0 + j * 8, 1);
    }
    // Prologue slot arms: every rank arms its OWN barriers locally (also in
    // mc mode - rank0 never writes rank1's mbarriers; the in-loop refills are
    // ordered by the r1rd handshake instead).
    for (int j = 0; j < SB && j < nstages; ++j) {
      mbar_expect_tx(mbar_kv0 + j * 8, NT * D * 2);
      mbar_expect_tx(mbar_vv0 + j * 8, NT * DV * 2);
    }
  }
  if (tid < 32) tmem_alloc(su32(tmem_slot), TC_COLS);
  __syncthreads();
  const uint32_t tmem = *tmem_slot;
  if (tid < 32) tmem_relinquish();
  const uint32_t tmem_s = tmem;
  const uint32_t tmem_p = tmem + S_COLS;
  const uint32_t tmem_o = tmem + S_COLS + P_COLS;
  const uint32_t peer_r1rd = mapa_u32(mbar_r1rd, 0);   // rank1 -> rank0 signal

  constexpr uint32_t IDESC_QK = tc_idesc(128, NT, false);
  constexpr uint32_t IDESC_PV = tc_idesc(128, DV, true);
  constexpr uint64_t A_ATOM = (128 * 128) >> 4;
  constexpr uint64_t B_ATOM = (uint64_t)NT * 8;
  constexpr uint32_t V_LBO = (DV == 64) ? 0 : NT * 128;
  const uint64_t qd0 = tc_desc(su32(smem_q), 1024, 0);

  // Q tile is per-CTA (own 128 rows); softmax threads cp.async it.
  if (!ctrl) tc_load_q<128, D, SW>(P, ti, smem_q, kvh, tid);
  cp_async_commit();
  cp_async_wait<0>();
  if (mc_on) {
    // Publish the rank-local mbarrier inits AND prologue arms cluster-wide
    // before rank0 issues the shared prologue boxes (init-fence +
    // cluster_sync = tc2's proven pattern; the CUTLASS multicast convention:
    // consumers arm their own barriers, one cluster-wide release/acquire
    // orders the arms before the single issuer's boxes).
    mbar_init_fence();
    cluster_sync_mc();
    if (ctrl && rank == 0) {
      const int pre = nstages < SB ? nstages : SB;
      for (int j = 0; j < pre; ++j) {
        const int32_t pgj =
            tc_page_pre<NT, D>(P, tabt, kv0 + j * NT, last_mc, lane);
        fence_proxy_cluster();
        __syncwarp();
        tc_tma_kv_mc<NT, D>(P, smem_k[j], &kmap, kvh, kv0 + j * NT, last_mc,
                            lane, pgj, mbar_kv0 + j * 8, 3);
      }
      for (int j = 0; j < pre; ++j) {
        const int32_t pgj =
            tc_page_pre<NT, D>(P, tabt, kv0 + j * NT, last_mc, lane);
        fence_proxy_cluster();
        __syncwarp();
        tc_tma_kv_mc<NT, DV>(P, smem_v[j], &vmap, kvh, kv0 + j * NT, last_mc,
                             lane, pgj, mbar_vv0 + j * 8, 3);
      }
    }
  } else if (ctrl) {
    // Solo fallback: base-equivalent prologue over BOTH slots (the unified
    // loop below refills stage s+SB at iteration s, so stage 1 must be
    // prologue-filled - the TMA_LATE prologue shape, measured perf-neutral).
    const int pre = nstages < SB ? nstages : SB;
    for (int j = 0; j < pre; ++j)
      tc_tma_kv<NT, D, 32>(P, smem_k[j], &kmap, kvh, kv0 + j * NT, ti.hi - 1,
                           tabt, lane, mbar_kv0 + j * 8);
    for (int j = 0; j < pre; ++j)
      tc_tma_kv<NT, DV, 32>(P, smem_v[j], &vmap, kvh, kv0 + j * NT, ti.hi - 1,
                            tabt, lane, mbar_vv0 + j * 8);
  }
  fence_proxy_async();
  __syncthreads();

  if (ctrl) {
    // ---- control warp ----
    fence_after_sync();
    mbar_wait(mbar_kv0, 0);
    fence_proxy_async();
    fence_after_sync();
    if (lane == 0) {
      uint64_t da = qd0, db = tc_desc(su32(smem_k[0]), 1024, 0);
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks) {
        mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);
        da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
        db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
      }
      umma_commit(mbar_qk);
    }
    __syncwarp();
    // Page IDs for stage SB (iteration 0's refill issues those boxes).
    int pg = tc_page_pre<NT, D>(P, tabt, kv0 + SB * NT, last_mc, lane);
    for (int s = 0; s < nstages; ++s) {
      // (1) PV(s): pre(s) proves P(s) packed and O(s) rescaled; vv(s) proves
      //     V(s) resident (prologue or iteration s-2's refill).
      mbar_wait((s & 1) ? mbar_pre1 : mbar_pre0, (s >> 1) & 1);
      fence_after_sync();
      mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);
      fence_proxy_async();
      fence_after_sync();
      if (lane == 0) {
        uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
        const uint32_t pb = tmem_p + (s & 1) * PH;
#pragma unroll
        for (int st = 0; st < NT / 16; ++st) {
          mma_ts(tmem_o, pb + st * 8, db, IDESC_PV, (st != 0) || (s > 0));
          db += 128;
        }
        umma_commit(mbar_pv);
      }
      __syncwarp();
      // (2) QK(s+1): sfree(s) proves S(s) drained (=> own QK(s) complete =>
      //     own K-slot s%SB dead); kv(s+1) was issued one iteration back.
      if (s + 1 < nstages) {
        mbar_wait((s & 1) ? mbar_sfree1 : mbar_sfree0, (s >> 1) & 1);
        fence_after_sync();
        mbar_wait(mbar_kv0 + ((s + 1) % SB) * 8, ((s + 1) / SB) & 1);
        fence_proxy_async();
        fence_after_sync();
        if (lane == 0) {
          const int kb = (s + 1) % SB;
          uint64_t da = qd0, db = tc_desc(su32(smem_k[kb]), 1024, 0);
#pragma unroll
          for (int ks = 0; ks < D / 16; ++ks) {
            mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);
            da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
            db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
          }
          umma_commit(mbar_qk);
        }
        __syncwarp();
      }
      // (3) rank1 refill preparation: prove its own QK(s) (step (2)'s
      //     sfree wait) and PV(s) (explicit wait) are done, re-arm its OWN
      //     slot barriers for stage s+SB locally, then release-signal rank0.
      //     Local arms + one cluster release/acquire pair is the CUTLASS
      //     multicast convention; rank0 never writes rank1's mbarriers.
      if (mc_on && rank == 1 && s + SB < nstages) {
        mbar_wait(mbar_pv, s & 1);
        fence_after_sync();
        if (lane == 0) {
          mbar_expect_tx(mbar_kv0 + (s % SB) * 8, NT * D * 2);
          mbar_expect_tx(mbar_vv0 + (s % SB) * 8, NT * DV * 2);
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive_remote(peer_r1rd);
        __syncwarp();
      }
      // (4) refill slot s%SB with stage s+SB. mc mode: rank0 waits the peer
      //     signal, arms its own slot, and issues ONE multicast box set for
      //     both CTAs. Solo mode: self arms and unicast issues (base
      //     choreography: expect_tx -> proxy fence -> syncwarp -> issue).
      if (s + SB < nstages && (!mc_on || rank == 0)) {
        if (mc_on) {
          mbar_wait(mbar_r1rd, s & 1);   // peer QK(s)/PV(s) done + peer armed
          fence_after_sync();
        }
        // Own K-slot free: own QK(s) complete (own step (2) sfree wait).
        if (lane == 0)
          mbar_expect_tx(mbar_kv0 + (s % SB) * 8, NT * D * 2);
        fence_proxy_async();
        __syncwarp();
        if (mc_on)
          tc_tma_kv_mc<NT, D>(P, smem_k[s % SB], &kmap, kvh,
                              kv0 + (s + SB) * NT, last_mc, lane, pg,
                              mbar_kv0 + (s % SB) * 8, 3);
        else
          tc_tma_kv_pre<NT, D>(P, smem_k[s % SB], &kmap, kvh,
                               kv0 + (s + SB) * NT, ti.hi - 1, lane, pg,
                               mbar_kv0 + (s % SB) * 8);
        // Own V-slot free: own PV(s) complete (it read own smem V copy).
        mbar_wait(mbar_pv, s & 1);
        fence_after_sync();
        if (lane == 0)
          mbar_expect_tx(mbar_vv0 + (s % SB) * 8, NT * DV * 2);
        fence_proxy_async();
        __syncwarp();
        if (mc_on)
          tc_tma_kv_mc<NT, DV>(P, smem_v[s % SB], &vmap, kvh,
                               kv0 + (s + SB) * NT, last_mc, lane, pg,
                               mbar_vv0 + (s % SB) * 8, 3);
        else
          tc_tma_kv_pre<NT, DV>(P, smem_v[s % SB], &vmap, kvh,
                                kv0 + (s + SB) * NT, ti.hi - 1, lane, pg,
                                mbar_vv0 + (s % SB) * 8);
        __syncwarp();
      }
      // (6) page IDs for stage s+SB+1 (iteration s+1's refill uses them).
      pg = tc_page_pre<NT, D>(P, tabt, kv0 + (s + SB + 1) * NT, last_mc, lane);
      __syncwarp();
    }
  } else {
    // ---- softmax warps 0-7: identical to main8ws (own tile, own S) ----
    const int rg = ti.m0 + row;
    const bool valid = row < ti.mrows;
    const int q_idx = min(rg / P.group, ti.qlen - 1);
    const int head = kvh * P.group + (rg % P.group);
    const int vis = ti.kvlen - ti.qlen + rg / P.group;
    const int rhi = min(P.causal ? vis : ti.kvlen - 1, kv1 - 1);
    const int rlo = max((P.window_left >= 0) ? max(0, vis - P.window_left) : 0,
                        kv0_exact);
    float mbase = NEG_INF, lt = 0.f;
    const bool wlead = ((tid & 31) == 0);
    for (int s = 0; s < nstages; ++s) {
      mbar_wait(mbar_qk, s & 1);
      fence_after_sync();
      float sreg[HC];
#pragma unroll
      for (int c = 0; c < HC / 32; ++c)
        tmem_ld32(sreg + c * 32, tmem_s + half * HC + c * 32);
      tmem_wait_ld();
      fence_before_sync();
      if (wlead) mbar_arrive((s & 1) ? mbar_sfree1 : mbar_sfree0);
      const int pbase = kv0 + s * NT + half * HC;
      const bool interior = (pbase >= rlo) && (pbase + HC - 1 <= rhi);
      float rmax_raw = NEG_INF;
      if (interior) {
#pragma unroll
        for (int i = 0; i < HC; ++i) rmax_raw = fmaxf(rmax_raw, sreg[i]);
      } else {
#pragma unroll
        for (int i = 0; i < HC; ++i) {
          const int c = pbase + i;
          float x = sreg[i];
          if (c < rlo || c > rhi) x = NEG_INF;
          sreg[i] = x;
          rmax_raw = fmaxf(rmax_raw, x);
        }
      }
      float* const rmb = rm + (s & 1) * (NH * 128);
      rmb[half * 128 + row] = rmax_raw;
#ifdef FMHA_TC_PAIRMAX
      // Pair-scoped exchange (FMHA_TC_PAIRMAX): each row's max is shared by
      // exactly NH warps (w, w+WPH, ...; WPH = SW/32/NH = 4 for NH=2 and
      // NH=4). Sync only that group (bar id 1+w%WPH, NH*32 threads) instead
      // of the full-SW named barrier: the row group waits for its own skew,
      // not the slowest of all 8/16 warps (rmx_exch measured 553 cyc/stage
      // on c1, probe dv_fc8d4bb9a57c). Values and reduction order are
      // unchanged -> bitwise identical. Instance-mixing is impossible: a
      // warp reaches its stage-s+1 group barrier only after its siblings
      // passed the stage-s one (bar.sync releases the whole group), and the
      // parity rm buffer covers the s+1 write vs the s read.
      asm volatile("bar.sync %0, %1;" :: "r"(1 + ((tid >> 5) & 3)),
                   "n"(NH * 32) : "memory");
#else
      bar_softmax_t<SW>();
#endif
      float rmax = rmb[row];
#pragma unroll
      for (int q = 1; q < NH; ++q) rmax = fmaxf(rmax, rmb[q * 128 + row]);
      const float mn = fmaxf(mbase, rmax * P.qk_scale2);
      const float msafe = (mn == NEG_INF) ? 0.f : mn;
      const bool skip = (msafe - mbase) < P.lazy_thresh;
      const float alpha = skip ? 1.f : exp2_fast(mbase - msafe);
      const float nqk = P.qk_scale2;
      const float npb = -(skip ? mbase : msafe);
      if (!skip) { lt *= alpha; mbase = msafe; }
      float lsum = 0.f;
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        const float p = exp2_fast(fmaf(sreg[i], nqk, npb));
        lsum += p;
        sreg[i] = p;
      }
#pragma unroll
      for (int c = 0; c < HC / 32; ++c) {
        uint32_t pw[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
          pw[j] = pack_bf16(sreg[c * 32 + 2 * j], sreg[c * 32 + 2 * j + 1]);
        tmem_st16(tmem_p + (s & 1) * PH + half * (HC / 2) + c * 16, pw);
      }
      tmem_wait_st();
      if (s > 0 && __any_sync(0xffffffffu, alpha < 1.0f)) {
        mbar_wait(mbar_pv, (s - 1) & 1);
        fence_after_sync();
        constexpr int OU = DV / 32;               // 32-col units in O
        constexpr int UPC = (OU + NH - 1) / NH;   // units per group
        const int u0 = half * UPC;
#pragma unroll
        for (int c = 0; c < UPC; ++c) {
          if (u0 + c < OU) {
            float oreg[32];
            const uint32_t oa = tmem_o + (u0 + c) * 32;
            tmem_ld32(oreg, oa);
            tmem_wait_ld();
#pragma unroll
            for (int i = 0; i < 32; ++i) oreg[i] *= alpha;
            tmem_st32(oa, oreg);
          }
        }
        tmem_wait_st();
      }
      lt += lsum;
      fence_before_sync();
      if (wlead) mbar_arrive((s & 1) ? mbar_pre1 : mbar_pre0);
    }

    // ---- epilogue (half-0 warps write all 128 rows of this tile) ----
    mbar_wait(mbar_pv, (nstages - 1) & 1);
    fence_after_sync();
    float* const ltx = rm + NH * 128;    // post-loop reuse of the odd rm block
    if (half >= 1) ltx[(half - 1) * 128 + row] = lt;
    bar_softmax_t<SW>();
    if (half == 0) {
#pragma unroll
      for (int q = 1; q < NH; ++q) lt += ltx[(q - 1) * 128 + row];
      const int64_t Pb = ((int64_t)tile * P.kvh + kvh) * P.splits_ub;
      const int cu = P.cu_q[ti.b];
      const int64_t tok = cu + q_idx;
      // Host gate: splits_ub == 1 for MC2 (combine_tile sc layout is
      // m_tile<=128); the split path is unreachable here.
      float lse2 = mbase + log2f(lt);
      float inv = P.v_scale / lt;
      if (P.has_sinks) {
        float os;
        lse2 = sink_adjust(lse2, P.sinks[head * P.sink_stride] * P.log2e, &os);
        inv *= os;
      }
      __nv_bfloat16* orow =
          P_OUT + tok * P.out_tok_stride + (int64_t)head * P.out_head_stride;
#pragma unroll
      for (int c = 0; c < DV / 32; ++c) {
        float oreg[32];
        tmem_ld32(oreg, tmem_o + c * 32);
        tmem_wait_ld();
        uint32_t packed[16];
#pragma unroll
        for (int i = 0; i < 16; ++i)
          packed[i] = pack_bf16(oreg[2 * i] * inv, oreg[2 * i + 1] * inv);
        if (valid) {
#pragma unroll
          for (int u = 0; u < 4; ++u)
            *reinterpret_cast<uint4*>(orow + c * 32 + u * 8) =
                *reinterpret_cast<uint4*>(packed + u * 4);
        }
      }
      if (valid && P.has_lse) P.lse[tok * P.lse_tok_stride + head] = lse2;
      (void)Pb;
    }
  }
  fence_before_sync();
  __syncthreads();
  if (tid < 32) tmem_dealloc(tmem, TC_COLS);
}
// ---------------------------------------------------------------------------
// DTA: dual-group TMEM-alternating WS prefill (dispatch bit28, FMHA_TC_DTA).
//
// Why: probe dv_49a02c9c9fa9 measured the shipped WS equilibrium on c1 as
// period ~= sm_qkwait 1700 + softmax tail 1240 with the tensor pipe at ~52%
// duty; every reordering/scoping knob screened since (dephase, pairmax,
// maxrow, cspro2, nt128, qk-first family) failed to move it. The structural
// cause: with ONE S buffer, softmax(s) both starts and ends inside the same
// pipe round trip, so the tail and the pipe serialize per CTA.
//
// Mechanism (validated against the public CUTLASS SM100 FMHA mainloop in the
// mounted reference: two softmax warp groups alternating S stages, P aliased
// into the drained S stage, QK primed TWO stages ahead - mechanism only, all
// code here independently written in the existing raw-PTX style):
//   * TMEM 256 cols: S0 [0,64), S1 [64,128), O [128,256). P(s) is packed by
//     the softmax group INTO S[s&1] cols 0..31 (bf16 pairs, same intra-P
//     layout as the old tmem_p parity buffers, so PV's mma_ts A-addressing
//     and per-channel values are unchanged).
//   * 512 softmax threads = TWO groups of 256 (each 8 warps, NH=2 column
//     halves, same per-thread fragment shape as main8ws: 32 S cols/thread).
//     Group g processes stages s == g (mod 2) - two softmax tails in flight.
//   * One control warp: primes QK(0),QK(1) into both S buffers, then per
//     iteration: PV(s) [gated pre(s), vv(s)], QK(s+2) [gated kv(s+2)],
//     refill slot s%SB with stage s+SB [gated pv(s) COMPLETE]. Pipe order
//     ..., PV(s), QK(s+2), ... with in-order completion gives the key
//     safety: QK(s+2) overwrites S[s&1] whose tenant P(s) is read by PV(s),
//     issued BEFORE it -> PV(s) is complete before QK(s+2) executes.
//     Symmetrically QK(s)'s commit covers PV(s-2), so softmax rescale's
//     pv(s-1) parity wait cannot alias past an incomplete PV(s-2).
//   * Row state (mbase, lt-half) crosses groups through smem slots with
//     mbarrier handoff (publish after lt is final for the stage, consume
//     before the next group's max merge). The row-max exchange barrier
//     inside each group separates its S reads from its P writes.
//   * O ordering: PV(s-1) < rescale(s) < PV(s) holds via the existing
//     pre/pv gates (rescale waits pv(s-1); PV(s) waits pre(s) which trails
//     the rescale), so both groups' O RMWs stay totally ordered.
// Same MMAs, same operand values, same accumulation orders as main8ws =>
// outputs intended BITWISE identical. 544 threads, 1 CTA/SM (120-reg cap),
// TMEM 256/512, smem ~133KB. Steady-state target: period/stage ->
// (tail + 2*384)/2 ~= 1100 vs the 1461/stage the 2-CTA base achieves.
// v1 host gate: d128 NT64 m128 WS splits==1 (c1/c7/d2 class).
// v2: DUAL control warps (576 threads) - DTA v1 (dv_90ad968c6f58) was
// bitwise-perfect but x1.56-1.69 SLOW because its single control warp
// serializes mma issue backpressure (~768/stage) + the pv-gated refill
// chain (~500-600) + page prefetch into a ~2270 cyc/stage critical path.
// v2 splits it CSPLIT-style: MMA-lead (PV/QK waits+issues+commits, paced by
// pipe backpressure only) and TMA-lead (pv-gated slot refills + page
// prefetch, ~3 stages of slack). Model period: max(768 pipe, (tail+768)/2
// ~= 1004)/stage vs the base's 1461 -> c1-class x~0.69.
// ---------------------------------------------------------------------------
#define DTA_SW 512
#define DTA_SB 3

extern "C" int fmha_dta_smem() {
  // Q 32K + KV 3x2x16K + tail (mbars 112 + tmem_slot 32 + rm 2048 + state
  // 3072 + alignment slack)
  return 128 * 128 * 2 + DTA_SB * 2 * (64 * 128 * 2) + 5504;
}

__global__ void __launch_bounds__(576, 1) fmha_tc_dta(
    const FmhaParams P, const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap) {
  constexpr int D = 128, DV = 128, NT = 64, SB = DTA_SB, SW = DTA_SW;
  constexpr int HC = NT / 2;            // 32 S columns per softmax half
  constexpr int S_COLS = NT;            // TMEM columns per S buffer
  constexpr int TC_COLS = 256;
  extern __shared__ __align__(1024) char smem[];
  constexpr int Q_BYTES = 128 * D * 2;
  constexpr int KV_BYTES = NT * D * 2;
  char* smem_q = smem;
  char* smem_k[SB];
  char* smem_v[SB];
#pragma unroll
  for (int j = 0; j < SB; ++j) {
    smem_k[j] = smem + Q_BYTES + j * 2 * KV_BYTES;
    smem_v[j] = smem_k[j] + KV_BYTES;
  }
  const uint32_t MB = su32(smem + Q_BYTES + SB * 2 * KV_BYTES);
  // mbar byte offsets: qk0 0, qk1 8, pv0 16, pv1 24, pre0 32, pre1 40,
  // kv[j] 48+8j, vv[j] 72+8j, st0 96, st1 104; tmem_slot at 112 (32B);
  // rm floats at 144 (2 groups x 2 halves x 128); state at 144+2048.
  const uint32_t mbar_qk0 = MB, mbar_qk1 = MB + 8;
  const uint32_t mbar_pv0 = MB + 16, mbar_pv1 = MB + 24;
  const uint32_t mbar_pre0 = MB + 32, mbar_pre1 = MB + 40;
  const uint32_t mbar_kv0 = MB + 48, mbar_vv0 = MB + 48 + SB * 8;
  const uint32_t mbar_st0 = MB + 96, mbar_st1 = MB + 104;
  char* tail = smem + Q_BYTES + SB * 2 * KV_BYTES;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(tail + 112);
  float* rm = reinterpret_cast<float*>(tail + 144);
  float* st_mb = rm + 512;          // [2 slots][128]
  float* st_lt = st_mb + 256;       // [2 slots][2 halves][128]

  const int tid = threadIdx.x;
  const bool ctrl = (tid >= SW);
  const bool tma_lead = ctrl && (tid < SW + 32);
  const bool mma_lead = ctrl && (tid >= SW + 32);
  const int lane = (tid - SW) & 31;
  const int grp = (tid >> 8) & 1;
  const int half = (tid >> 7) & 1;
  const int row = tid & 127;

  int tile_y, split, kvh;
  grid_pos(P, kvh, tile_y, split);
  const int tile = ((((P._pad2 >> 1) & 1) && P.causal) ? (P.tiles_ub - 1 - tile_y)
                                                       : tile_y);
  TileInfo ti = map_tile(P, tile);
  if (!ti.ok) return;
  int kv0, kv1, active;
  if (!split_range(P, ti, split, kv0, kv1, active)) return;
  const int kv0_exact = kv0;
  kv0 &= ~(NT - 1);
  const int nstages = (kv1 - kv0 + NT - 1) / NT;
  if (nstages <= 0) return;
  const int32_t* const tabt = P.table + ti.b * P.tstride;

  if (tid == 0) {
    mbar_init(mbar_qk0, 1); mbar_init(mbar_qk1, 1);
    mbar_init(mbar_pv0, 1); mbar_init(mbar_pv1, 1);
    mbar_init(mbar_pre0, 8); mbar_init(mbar_pre1, 8);   // 8 wleads per group
    mbar_init(mbar_st0, 256); mbar_init(mbar_st1, 256); // all group threads
    for (int j = 0; j < SB; ++j) {
      mbar_init(mbar_kv0 + j * 8, 1);
      mbar_init(mbar_vv0 + j * 8, 1);
    }
    for (int j = 0; j < SB && j < nstages; ++j) {
      mbar_expect_tx(mbar_kv0 + j * 8, NT * D * 2);
      mbar_expect_tx(mbar_vv0 + j * 8, NT * DV * 2);
    }
  }
  if (tid < 32) tmem_alloc(su32(tmem_slot), TC_COLS);
  __syncthreads();
  const uint32_t tmem = *tmem_slot;
  if (tid < 32) tmem_relinquish();
  const uint32_t tmem_s = tmem;               // S0 = +0, S1 = +S_COLS
  const uint32_t tmem_o = tmem + 2 * S_COLS;  // 128 fp32 columns

  constexpr uint32_t IDESC_QK = tc_idesc(128, NT, false);
  constexpr uint32_t IDESC_PV = tc_idesc(128, DV, true);
  constexpr uint64_t A_ATOM = (128 * 128) >> 4;
  constexpr uint64_t B_ATOM = (uint64_t)NT * 8;
  constexpr uint32_t V_LBO = NT * 128;
  const uint64_t qd0 = tc_desc(su32(smem_q), 1024, 0);

  if (!ctrl) tc_load_q<128, D, SW>(P, ti, smem_q, kvh, tid);
  cp_async_commit();
  cp_async_wait<0>();
  if (tma_lead) {
    const int pre = nstages < SB ? nstages : SB;
    for (int j = 0; j < pre; ++j)
      tc_tma_kv<NT, D, 32>(P, smem_k[j], &kmap, kvh, kv0 + j * NT, ti.hi - 1,
                           tabt, lane, mbar_kv0 + j * 8);
    for (int j = 0; j < pre; ++j)
      tc_tma_kv<NT, DV, 32>(P, smem_v[j], &vmap, kvh, kv0 + j * NT, ti.hi - 1,
                            tabt, lane, mbar_vv0 + j * 8);
  }
  fence_proxy_async();
  __syncthreads();

  if (mma_lead) {
    // ---- MMA-lead: prime QK(0), QK(1); then PV(s) -> QK(s+2) ----
    fence_after_sync();
    mbar_wait(mbar_kv0, 0);
    fence_proxy_async();
    fence_after_sync();
    if (lane == 0) {
      uint64_t da = qd0, db = tc_desc(su32(smem_k[0]), 1024, 0);
#pragma unroll
      for (int ks = 0; ks < D / 16; ++ks) {
        mma_ss(tmem_s, da, db, IDESC_QK, ks != 0);
        da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
        db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
      }
      umma_commit(mbar_qk0);
    }
    __syncwarp();
    if (nstages > 1) {
      mbar_wait(mbar_kv0 + 8, 0);
      fence_proxy_async();
      fence_after_sync();
      if (lane == 0) {
        uint64_t da = qd0, db = tc_desc(su32(smem_k[1]), 1024, 0);
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
          mma_ss(tmem_s + S_COLS, da, db, IDESC_QK, ks != 0);
          da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
          db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
        }
        umma_commit(mbar_qk1);
      }
      __syncwarp();
    }
    for (int s = 0; s < nstages; ++s) {
      // PV(s): P(s) packed (pre), V(s) resident (vv). A = P inside S[s&1].
      mbar_wait((s & 1) ? mbar_pre1 : mbar_pre0, (s >> 1) & 1);
      fence_after_sync();
      mbar_wait(mbar_vv0 + (s % SB) * 8, (s / SB) & 1);
      fence_proxy_async();
      fence_after_sync();
      if (lane == 0) {
        uint64_t db = tc_desc(su32(smem_v[s % SB]), 1024, V_LBO);
        const uint32_t pb = tmem_s + (s & 1) * S_COLS;
#pragma unroll
        for (int st = 0; st < NT / 16; ++st) {
          mma_ts(tmem_o, pb + st * 8, db, IDESC_PV, (st != 0) || (s > 0));
          db += 128;
        }
        umma_commit((s & 1) ? mbar_pv1 : mbar_pv0);
      }
      __syncwarp();
      // QK(s+2) -> S[(s+2)&1] == S[s&1]: legal because the pipe is in-order
      // and PV(s) (the buffer's current reader) was issued first.
      if (s + 2 < nstages) {
        mbar_wait(mbar_kv0 + ((s + 2) % SB) * 8, ((s + 2) / SB) & 1);
        fence_proxy_async();
        fence_after_sync();
        if (lane == 0) {
          const int kb = (s + 2) % SB;
          uint64_t da = qd0, db = tc_desc(su32(smem_k[kb]), 1024, 0);
          const uint32_t sbuf = tmem_s + ((s + 2) & 1) * S_COLS;
#pragma unroll
          for (int ks = 0; ks < D / 16; ++ks) {
            mma_ss(sbuf, da, db, IDESC_QK, ks != 0);
            da += ((ks & 3) == 3) ? (A_ATOM - 6) : 2;
            db += ((ks & 3) == 3) ? (B_ATOM - 6) : 2;
          }
          umma_commit(((s + 2) & 1) ? mbar_qk1 : mbar_qk0);
        }
        __syncwarp();
      }
    }
  } else if (tma_lead) {
    // ---- TMA-lead: pv-gated slot refills + page prefetch (CSPLIT-style
    // split; ~3 stages of slack, off the mma issue critical path) ----
    int pg = tc_page_pre<NT, D>(P, tabt, kv0 + SB * NT, ti.hi - 1, lane);
    __syncwarp();
    for (int s = 0; s < nstages; ++s) {
      // Refill slot s%SB with stage s+SB once PV(s) completed (V(s) dead;
      // K(s) dead transitively - QK(s) precedes PV(s) in the pipe).
      if (s + SB < nstages) {
        mbar_wait((s & 1) ? mbar_pv1 : mbar_pv0, (s >> 1) & 1);
        fence_after_sync();
        const int j = s % SB;
        if (lane == 0) {
          mbar_expect_tx(mbar_kv0 + j * 8, NT * D * 2);
          mbar_expect_tx(mbar_vv0 + j * 8, NT * DV * 2);
        }
        fence_proxy_async();
        __syncwarp();
        tc_tma_kv_pre<NT, D>(P, smem_k[j], &kmap, kvh, kv0 + (s + SB) * NT,
                             ti.hi - 1, lane, pg, mbar_kv0 + j * 8);
        tc_tma_kv_pre<NT, DV>(P, smem_v[j], &vmap, kvh, kv0 + (s + SB) * NT,
                              ti.hi - 1, lane, pg, mbar_vv0 + j * 8);
        __syncwarp();
      }
      pg = tc_page_pre<NT, D>(P, tabt, kv0 + (s + SB + 1) * NT, ti.hi - 1,
                              lane);
      __syncwarp();
    }
  } else {
    // ---- softmax group `grp`: stages s == grp (mod 2) ----
    const uint32_t qk_bar = grp ? mbar_qk1 : mbar_qk0;
    const uint32_t st_pub = grp ? mbar_st1 : mbar_st0;
    const uint32_t st_wait = grp ? mbar_st0 : mbar_st1;
    float* const rmg = rm + grp * 256;
    const int rg = ti.m0 + row;
    const bool valid = row < ti.mrows;
    const int q_idx = min(rg / P.group, ti.qlen - 1);
    const int head = kvh * P.group + (rg % P.group);
    const int vis = ti.kvlen - ti.qlen + rg / P.group;
    const int rhi = min(P.causal ? vis : ti.kvlen - 1, kv1 - 1);
    const int rlo = max((P.window_left >= 0) ? max(0, vis - P.window_left) : 0,
                        kv0_exact);
    float mbase = NEG_INF, lt = 0.f;
    const bool wlead = ((tid & 31) == 0);
    for (int s = grp; s < nstages; s += 2) {
      const int sp = s & 1;              // == grp, constant per group
      mbar_wait(qk_bar, (s >> 1) & 1);
      fence_after_sync();
      float sreg[HC];
#pragma unroll
      for (int c = 0; c < HC / 32; ++c)
        tmem_ld32(sreg + c * 32, tmem_s + sp * S_COLS + half * HC + c * 32);
      tmem_wait_ld();
      fence_before_sync();
      const int pbase = kv0 + s * NT + half * HC;
      const bool interior = (pbase >= rlo) && (pbase + HC - 1 <= rhi);
      float rmax_raw = NEG_INF;
      if (interior) {
#pragma unroll
        for (int i = 0; i < HC; ++i) rmax_raw = fmaxf(rmax_raw, sreg[i]);
      } else {
#pragma unroll
        for (int i = 0; i < HC; ++i) {
          const int c = pbase + i;
          float x = sreg[i];
          if (c < rlo || c > rhi) x = NEG_INF;
          sreg[i] = x;
          rmax_raw = fmaxf(rmax_raw, x);
        }
      }
      rmg[half * 128 + row] = rmax_raw;
      asm volatile("bar.sync %0, 256;" :: "r"(1 + grp) : "memory");
      float rmax = rmg[row];
#pragma unroll
      for (int q = 1; q < 2; ++q) rmax = fmaxf(rmax, rmg[q * 128 + row]);
      if (s > 0) {
        // Row-state handoff published by the other group's stage s-1.
        // Placed AFTER the S read + rmx exchange so this group's TMEM/bar
        // work overlaps the other group's exp/pack; mbase/lt are first
        // needed at the mn computation right below.
        mbar_wait(st_wait, ((s - 1) >> 1) & 1);
        fence_after_sync();
        mbase = st_mb[((s - 1) & 1) * 128 + row];
        lt = st_lt[((s - 1) & 1) * 256 + half * 128 + row];
      }
      const float mn = fmaxf(mbase, rmax * P.qk_scale2);
      const float msafe = (mn == NEG_INF) ? 0.f : mn;
      const bool skip = (msafe - mbase) < P.lazy_thresh;
      const float alpha = skip ? 1.f : exp2_fast(mbase - msafe);
      const float nqk = P.qk_scale2;
      const float npb = -(skip ? mbase : msafe);
      if (!skip) { lt *= alpha; mbase = msafe; }
      float lsum = 0.f;
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        const float p = exp2_fast(fmaf(sreg[i], nqk, npb));
        lsum += p;
        sreg[i] = p;
      }
      lt += lsum;
      // Publish the row state for the other group (release-arrive), then
      // pack P into the drained S columns (the group barrier above already
      // proved every warp of THIS group finished reading S).
      st_lt[sp * 256 + half * 128 + row] = lt;
      if (half == 0) st_mb[sp * 128 + row] = mbase;
      mbar_arrive(st_pub);
#pragma unroll
      for (int c = 0; c < HC / 32; ++c) {
        uint32_t pw[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
          pw[j] = pack_bf16(sreg[c * 32 + 2 * j], sreg[c * 32 + 2 * j + 1]);
        tmem_st16(tmem_s + sp * S_COLS + half * (HC / 2) + c * 16, pw);
      }
      tmem_wait_st();
      if (s > 0 && __any_sync(0xffffffffu, alpha < 1.0f)) {
        mbar_wait(((s - 1) & 1) ? mbar_pv1 : mbar_pv0, ((s - 1) >> 1) & 1);
        fence_after_sync();
        constexpr int OU = DV / 32;
        constexpr int UPC = (OU + 1) / 2;
        const int u0 = half * UPC;
#pragma unroll
        for (int c = 0; c < UPC; ++c) {
          if (u0 + c < OU) {
            float oreg[32];
            const uint32_t oa = tmem_o + (u0 + c) * 32;
            tmem_ld32(oreg, oa);
            tmem_wait_ld();
#pragma unroll
            for (int i = 0; i < 32; ++i) oreg[i] *= alpha;
            tmem_st32(oa, oreg);
          }
        }
        tmem_wait_st();
      }
      fence_before_sync();
      if (wlead) mbar_arrive((s & 1) ? mbar_pre1 : mbar_pre0);
    }

    // ---- epilogue: the group owning the LAST stage writes the tile ----
    if (nstages > 0 && grp == ((nstages - 1) & 1)) {
      const int sf = nstages - 1;
      mbar_wait((sf & 1) ? mbar_pv1 : mbar_pv0, (sf >> 1) & 1);
      fence_after_sync();
      float* const ltx = rmg + 128;     // stale half-1 rm region
      if (half >= 1) ltx[(half - 1) * 128 + row] = lt;
      asm volatile("bar.sync %0, 256;" :: "r"(1 + grp) : "memory");
      if (half == 0) {
#pragma unroll
        for (int q = 1; q < 2; ++q) lt += ltx[(q - 1) * 128 + row];
        const int64_t Pb = ((int64_t)tile * P.kvh + kvh) * P.splits_ub + split;
        const int cu = P.cu_q[ti.b];
        const int64_t tok = cu + q_idx;
        if (P.splits_ub > 1) {
          float* po = P.part_o + (Pb * P.m_tile + row) * DV;
          const int64_t blk = (int64_t)tile * P.kvh + kvh;
          const int Spad = (P.splits_ub + 3) & ~3;
          float* pm = P.part_ml + ((blk * P.m_tile + row) * 2) * Spad + split;
#pragma unroll
          for (int c = 0; c < DV / 32; ++c) {
            float oreg[32];
            tmem_ld32(oreg, tmem_o + c * 32);
            tmem_wait_ld();
#pragma unroll
            for (int i = 0; i < 32; i += 4)
              *reinterpret_cast<float4*>(po + c * 32 + i) =
                  make_float4(oreg[i], oreg[i + 1], oreg[i + 2], oreg[i + 3]);
          }
          if (valid) { pm[0] = mbase; pm[Spad] = lt; }
          cudaTriggerProgrammaticLaunchCompletion();
        } else {
          float lse2 = mbase + log2f(lt);
          float inv = P.v_scale / lt;
          if (P.has_sinks) {
            float os;
            lse2 = sink_adjust(lse2, P.sinks[head * P.sink_stride] * P.log2e, &os);
            inv *= os;
          }
          __nv_bfloat16* orow =
              P_OUT + tok * P.out_tok_stride + (int64_t)head * P.out_head_stride;
#pragma unroll
          for (int c = 0; c < DV / 32; ++c) {
            float oreg[32];
            tmem_ld32(oreg, tmem_o + c * 32);
            tmem_wait_ld();
            uint32_t packed[16];
#pragma unroll
            for (int i = 0; i < 16; ++i)
              packed[i] = pack_bf16(oreg[2 * i] * inv, oreg[2 * i + 1] * inv);
            if (valid) {
#pragma unroll
              for (int u = 0; u < 4; ++u)
                *reinterpret_cast<uint4*>(orow + c * 32 + u * 8) =
                    *reinterpret_cast<uint4*>(packed + u * 4);
            }
          }
          if (valid && P.has_lse) P.lse[tok * P.lse_tok_stride + head] = lse2;
        }
      }
    }
  }
  fence_before_sync();
  __syncthreads();
  if (tid < 32) tmem_dealloc(tmem, TC_COLS);   // same warp that allocated
}
#endif  // FMHA_TMA
