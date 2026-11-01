// fmha_pre.cu - merged tcgen05 prefill (arm-b v4.12 lineage), appended by
// candidate/solution.py after fmha.cu + fmha_tc.cu into ONE translation unit
// (the build pattern fmha_tc.cu already uses: no includes here, the prologue
// of fmha.cu provides torch/CUDA headers and the DEVI macro).
//
// Provenance: verbatim port of the prefill stack promoted at tag
// v4.12-full-suite-pass (commit 5cf8370) onto the unified pair winner's
// Params/ABI after the manual join 20261022. Evidence at that tag: engine
// fixed 10/10 x 2 seeds, geomean 0.1879ms -> speedup 0.5323 (dv_38a3cfc03d08);
// engine domain 10/10, 0.1317ms (dv_e6755393cf6a); check_quick 21/21.
// Per-case vs this tree's mma.sync prefill (join receipt dv_b68636085c6f):
// c0 1643 vs 2751us, c1 135 vs 217us, c7 386 vs 646us, c9 695 vs 1092us.
//
// Mechanism/attribution: independent implementation; tcgen05 phase structure
// follows the public PTX ISA and FA3/FA4-style synchronous scheduling,
// validated via experiments/tc_probe*.py on this device. Topology per CTA
// (128 threads, 4 warps), M=128 q rows x head:
//   TMEM: S0 [0,BNT) S1 [BNT,2*BNT) (P packed bf16 reuses S cols),
//         O f32 at SOFF=2*BNT; ACOLS 128/256/512 tiers.
//   smem: Q bf16 canonical K-major image | K bf16 interleave images x2
//         | V bf16 swizzle-128B MN-major images x2 | ks/vs f32 x2 | bars.
//   Per KV tile t (stage s=t&1): convert(t) -> prefetch(t+1) -> fence+bar ->
//   QK mma + commit -> mbar wait -> softmax (tcgen05.ld S, ks scale, causal,
//   online max/sum, P=bf16(p*vs) tcgen05.st) -> bar -> correction (O *= alpha,
//   skipped when warp-uniformly 1) -> bar -> PV mma (A from TMEM P) + commit.
//   PIPE=true (D=64 only): softmax(t) overlaps PV(t-1), QK(t+1) issues before
//   the correction (v2.3b/v2.5 split; D128 measured 0.95x pipelined, D64 seq
//   measured 0.892x -- both reconfirmed at v4.11/v4.12).
//   Driver modes: static | v4.5 mirror-fold pairs | v4.8 persistent atomic
//   work-pull (heavy-first, heads-innermost; ticket in S.tmem_base[1] --
//   NEVER a static __shared__ var here: it shifts the extern dynamic smem
//   base off 16B and tc_desc's addr>>4 silently corrupts UMMA descriptors).
// Numerics: identical contract path to the v4.12 tag (per-key k scale after
// QK in the log2 domain, unrounded denominator, bf16 P*vs, exact bf16 V).
//
// Deliberate deltas from the v4.12 build (semantics-preserving):
//  * The shared build flags have no --use_fast_math (the peer kernels are
//    tuned without it), so the two fast-math-dependent spellings are made
//    explicit: softmax exp2f -> mypre::ex2f (ex2.approx.ftz.f32, the MUFU
//    instruction fast-math selected) and epilogue 1.f/l2 -> mypre::rcpa.
//  * The persistent-pull work counter is a one-time cudaMalloc'd word zeroed
//    by cudaMemsetAsync on every persist launch (graph-capturable memset node,
//    replay re-arms it) instead of a per-call torch::zeros tensor: the
//    int-only ABI carries no scratch tensor and captured graphs replay
//    without host code.
//  * mypre::Params is the prefill-only field set, filled by launch_mypre from
//    the unified ::Params; out gets its own stride field (v4.12 assumed
//    q/out token strides equal after a host TORCH_CHECK).
// Everything else (kernel text, layouts, gating formulas, launch_bounds(128),
// BNT=64, PIPE split) is the measured v4.12 source.

// ============================ device side ===================================
namespace mypre {

// TC_THREADS (=128) and NEG (=-1e30f) come from the verbatim fmha.cuh slice
// below (THREADS renamed to avoid colliding with peer constants).

// DIAGNOSTIC ONLY (FMHA_DBG_PRE, never in the candidate build): jobs
// dv_27f6edc9a3f5, dv_d3d4fd8dc3ee/p7 and dv_6aadfb1113ef threw
// cudaErrorInvalidValue from _ext.fmha_run on the merged-prefill cases, and the
// pattern across them is now narrow:
//   * the FIRST extension loaded in a process launches every prefill case fine;
//   * the SECOND one's first merged-prefill launch is rejected - unless that
//     build carries this diagnostic, in which case it also works (dv_d3d4fd8dc3ee
//     p1-p6: six processes, two of them two-module pairs, all four cases clean);
//   * dv_6aadfb1113ef reproduced it with no reference, no compare, no event loop
//     and no prior case at all - one bare forward per arm - so ab.py's flow is
//     not the trigger and the six-process bisect was aimed at the wrong thing.
// The only thing this diagnostic does that the plain build does not, on the
// launch path, is call cudaFuncSetAttribute(MaxDynamicSharedMemorySize) AGAIN
// immediately before the launch. That is a candidate cure as well as a probe, so
// the levels are separated at RUNTIME (FMHA_DBG_PRE_LVL, read once at load) and
// one build can be run at all three:
//   1 = passive only: driver mode, grid, SMEM, and the pending error before and
//       after each launch. Touches no kernel object, so it cannot mask anything.
//   2 = level 1 plus cudaFuncGetAttributes (the maxDynamicSharedSizeBytes the
//       driver actually recorded, numRegs) and a host-side occupancy probe over
//       the exact launch config. These query the kernel, which may itself force
//       a lazily-loaded module to load - so a failure that vanishes between
//       level 1 and level 2 is a module-loading effect, not a config effect.
//   3 = level 2 plus the redundant cudaFuncSetAttribute. If level 3 works where
//       level 1 fails, the attribute recorded by the per-branch static
//       initialiser is not in force for the second module's launch, and the fix
//       belongs in the candidate rather than in the harness.
// cudaPeek (not cudaGetLastError) so the diagnostic never swallows the error the
// caller's C10_CUDA_KERNEL_LAUNCH_CHECK is about to report.
#ifdef FMHA_DBG_PRE
#include <cstdio>
#include <cstdlib>
namespace dbgpre {
inline int lvl() {
    static const int l = [] {
        const char* s = getenv("FMHA_DBG_PRE_LVL");
        return s ? atoi(s) : 1;
    }();
    return l;
}
template <class F> inline const char* setattr(F* fn, int b) {
    return cudaGetErrorString(cudaFuncSetAttribute(
        (const void*)fn, cudaFuncAttributeMaxDynamicSharedMemorySize, b));
}
template <class F> inline int maxdyn(F* fn) {
    cudaFuncAttributes a;
    const cudaError_t e = cudaFuncGetAttributes(&a, (const void*)fn);
    return e == cudaSuccess ? (int)a.maxDynamicSharedSizeBytes : -(int)e;
}
template <class F> inline int regs(F* fn) {
    cudaFuncAttributes a;
    const cudaError_t e = cudaFuncGetAttributes(&a, (const void*)fn);
    return e == cudaSuccess ? (int)a.numRegs : -(int)e;
}
// Internal linkage on purpose: these have to be per-module copies, and comparing
// them across two extensions in one process is the test. If the second module
// reports the FIRST module's address for the `static const bool ok` in
// launch_mypre<128>, then that static local - and therefore its guard variable -
// is a weak symbol that got merged at load time, the second module skipped the
// initialiser entirely, and its kernels were launched with the 48 KB default
// dynamic-shared limit still in force. That is exactly cudaErrorInvalidValue.
static int attr_calls = 0;
static const void* ok_addr = nullptr;
static int attr_err0 = -1;      // cudaFuncSetAttribute on the NM=false instantiation
static int attr_err1 = -1;      // ... and on the NM=true one
template <class F> inline const char* occ(F* fn, int thr, int smem, int* nb) {
    return cudaGetErrorString(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(nb, fn, thr, (size_t)smem));
}
}  // namespace dbgpre
#define DBGP(L, ...) do { if (mypre::dbgpre::lvl() >= (L)) {         fprintf(stderr, "[dbgp] " __VA_ARGS__); fputc('\n', stderr); fflush(stderr); } } while (0)
#ifdef FMHA_DBG_PRE_NOPEEK
// Control for "is the cure the CUDA call, or merely host work between the
// attribute-setting static initialiser and the launch": at level 1 a nopeek
// build prints the geometry and makes no CUDA call at all.
#define DBGP_PEND() "n/a-nopeek"
#else
#define DBGP_PEND() cudaGetErrorString(cudaPeekAtLastError())
#endif
#define DBGP_MAXDYN(...) mypre::dbgpre::maxdyn(__VA_ARGS__)
#define DBGP_REGS(...) mypre::dbgpre::regs(__VA_ARGS__)
#define DBGP_OCC2(THR, SM, ...) mypre::dbgpre::occ(__VA_ARGS__, THR, SM, &dbg_nb)
#define DBGP_ATTR2(SM, ...) mypre::dbgpre::setattr(__VA_ARGS__, SM)
// NAME must be a string literal; the kernel id may be any expression, including
// a ternary, as long as the caller parenthesises it. KA runs BEFORE the launch
// and can change the outcome - level 1 makes a cudaPeekAtLastError call, level 3
// re-sets the attribute - so KA is the experiment. KB runs AFTER it and only
// reports: the recorded maxDynamicSharedSizeBytes is a host-side property, so
// reading it after a rejected launch is just as informative and cannot cure it.
#define DBGP_KA(NAME, THR, SM, ...) do { \
        DBGP(1, NAME " thr=%d smem=%d pending_before=%s", (THR), (SM), DBGP_PEND()); \
        DBGP(3, NAME " setattr_again=%s", DBGP_ATTR2((SM), __VA_ARGS__)); \
    } while (0)
#define DBGP_KB(NAME, THR, SM, ...) do { \
        int dbg_nb = -1; (void)dbg_nb; \
        DBGP(1, NAME " after launch: pending=%s", DBGP_PEND()); \
        DBGP(2, NAME " maxdyn=%d regs=%d occ=%s nb=%d", DBGP_MAXDYN(__VA_ARGS__), \
             DBGP_REGS(__VA_ARGS__), DBGP_OCC2((THR), (SM), __VA_ARGS__), dbg_nb); \
    } while (0)
#else
#define DBGP(L, ...) do {} while (0)
#define DBGP_PEND() ""
#define DBGP_MAXDYN(...) 0
#define DBGP_REGS(...) 0
#define DBGP_OCC2(THR, SM, ...) ""
#define DBGP_ATTR2(SM, ...) ""
#define DBGP_KA(NAME, THR, SM, ...) do {} while (0)
#define DBGP_KB(NAME, THR, SM, ...) do {} while (0)
#endif

// Fast-math spellings made explicit (no --use_fast_math in the shared flags):
// MUFU ex2/rcp with denormal flush -- the instructions exp2f and 1.f/x
// compiled to in the measured v4.12 build. ex2 maps -inf -> 0 exactly as the
// masked softmax needs; NEG-sentinel rows annihilate as before.
DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI float rcpa(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

// merge2-nm: the warp-uniform mask-free softmax fast path is a TEMPLATE
// parameter (NM), selected by launch_mypre per driver mode. Same-job A/B
// (dv_84533b62e76c, 3 alternating reps): persist c0 +3.8% / c7 +2.5%, static
// c9 +0.2% (noise), fold c1 -3.4% -> NM=true only for persist launches; the
// other modes keep the measured merge1 code path (NM=false folds the fast
// branch away at compile time).

// Prefill-only parameter block (arm-b v4.12 field set, minus the decode and
// dormant-variant fields; out_stride split out from q_out_stride).
struct Params {
    const uint16_t* q;        // (T, H, D) bf16 bits
    uint16_t* out;            // (T, H, D) bf16 bits
    const uint8_t* k;         // fp8 e4m3, logical (blocks, page, kvh, D)
    const uint8_t* v;
    const float* ks;          // (blocks, page, kvh)
    const float* vs;
    const int* block_table;   // (batch, bt_stride)
    const int* cu;            // (batch+1,)
    const int* seqk;          // (batch,)
    long long q_out_stride;   // q token stride (bf16 elements)
    long long out_stride;     // out token stride (bf16 elements)
    long long k_sb, k_sp, k_sh;   // fp8 element strides (block, slot, kvh)
    long long v_sb, v_sp, v_sh;
    long long ks_sb, ks_sp, ks_sh; // f32 element strides
    long long vs_sb, vs_sp, vs_sh;
    int bt_stride;
    int batch, heads, g;
    int D, page_shift;
    float sm_scale_log2e;
    int fold;                 // v4.5: mirror-pair q tiles (diagonal causal)
    int* counter;             // persistent-prefill work counter (nullptr else)
    int persist;              // dynamic work-pull prefill mode
    int cl_split;             // 1 = this launch's clusters split one item's KV
                              // range across TC_CL ranks and merge over DSMEM
};

// f16 K image + f16 Q for the QK MMA, SHIPPED (dv_fdd2a0699e7e: case 9 0.9737, case 7 0.9736,
// case 0 0.9786 on medians with the minima agreeing to 0.1%, case 1 1.0000, decode nulls 4/5
// clean, all six arm/case pairs bit-identical and the worst rel_l2 unchanged at
// 0.0014176992271294285). The convert's five-instruction bf16 chain becomes
// cvt.rn.f16x2.e4m3x2 alone: the K half of the convert's ~320 ALU instructions per thread per
// tile drops from 160 to 32, so the loop goes from roughly 720 to 592 - and f16 holds an e4m3
// value exactly (4 significand bits, |x| <= 448, against f16's
// 11 and 65504), so the QK scores do not move: same exact operands, both products exact in f32
// (11x11 = 22 bits, 8x8 = 16, both under 24), same kk accumulation order into the same f32
// accumulator. Q is staged as f16 once per ITEM rather than per tile, ~192 instructions of
// bf16 -> f32 -> f16x2, which is why case 1 - 32 tiles per CTA, where the per-item cost F
// dominates - reads exactly neutral instead of gaining. V and P stay bf16 because the reference
// rounds weighted_p to q.dtype before the PV product, and no instruction saving is worth changing
// the frozen arithmetic. The direct cvt.rn.bf16x2.e4m3x2 would be one instruction for V too, but
// flashinfer guards it behind CUDA >= 13.2 (csrc/msa_prefill_nvfp4_specialized.cu:486-501) and this
// platform has 13.0.
// FMHA_PRE_NO_KF16=1 restores the bf16 path as the control.
#ifndef FMHA_PRE_NO_KF16
#define MYPRE_KF16 1
#endif

// f16 V image + f16 P for the PV MMA, SHIPPED (dv_63ce554481d1, 2 arms, 3 alternating reps x 15
// event-timed iterations, correctness ON, 12 fixed + 12 domain pairs, no failures). The V half of
// MYPRE_KF16's substitution: the convert's five-instruction e4m3->f16->f32->bf16x2 chain becomes
// cvt.rn.f16x2.e4m3x2 alone, and idesc_pv's a/b format moves BF16 -> F16 in all three prefill
// kernels. Measured on MINIMA with the medians agreeing to 0.6%: case 0 0.9827 (median 0.9826),
// case 9 0.9859 (0.9799), case 7 0.9865 (0.9866), case 1 0.9994 - flat exactly as the KF16 note
// predicts for an F-dominated shape, since its items are short diagonal walks where the per-item Q
// staging dominates. Decode nulls 4 and 5 bit-identical. Under NCU the case-9 kernel is 1062.2 ->
// 1020.4/1018.4us at 183 registers, unchanged, so unlike PF2 this is not a register story; the
// stall ratios per issue rise (long_scoreboard 1.11 -> 1.27, short 0.45 -> 0.60, wait 0.83 -> 0.93)
// which is what removing non-stalling instructions from the denominator looks like.
//
// NOT BIT-IDENTICAL, and the movement is the point of the record: worst rel_l2 on the fixed suite
// 0.0014176992271294285 -> 0.0026158982468759014, on the domain suite 0.0004632264544751673 ->
// 0.002661514094541731, against the frozen max_rel_l2 of 1e-2 and max_mismatch_fraction of 1e-3 in
// problem/metadata.json - 3.8x of margin on rel_l2, and within 7% of the 2.45e-3 predicted analytically as sqrt(1.4177e-3^2 + (2^-9)^2). The
// deviation is exactly the reference's OWN rounding: reference.py:43 computes
// weighted_p = (weights * value_scale).to(q.dtype), i.e. bf16 with 8 significand bits, and this
// rounds the same f32 quantity to f16 with 11, so it moves toward exact rather than away. V stays
// exact (e4m3 carries 4 significand bits, f16 11), the products stay exact in the f32 accumulate
// (11x11 = 22 bits), and reference.py:45 accumulates denom from the UNROUNDED weights.sum(-1),
// which is what this kernel's f32 l2 does too, so the denominator is untouched. weights is
// unnormalized exp(score - max) in [0,1], so P is in [0, v_scale] and f16's 65504 ceiling is out of
// reach; its 6e-8 subnormal floor can only zero terms whose share of a row sum is below ~1e-7.
// input.py:100 bounds that independently of any sample: v_scale = max|v_bf16|/448 with a 1e-6 floor,
// so overflow would need a bf16 input above 2.93e7, and the extreme case in the file - the sentinel
// page at input.py:114, v = 448.0 with vs = 1.125/448 - gives P <= 0.00251.
//
// THE MISMATCH FRACTION DID MOVE, and it is corrected here because an earlier revision of this
// comment claimed it had not. All 24 pairs passed, but the count of elements over atol + rtol*|actual|
// rises about 70x: case 0 from 2 to 148, case 7 from 2 to 144, case 9 from 1 to 44, case 1 from 1 to
// 40. ab.py prints `mismatched=<count>/<allowed>` with allowed = numel // 1000, so the denominator is
// the BUDGET and not the element count - "148/12582" on case 0 is 148 of 12,582,000 elements = 1.18e-5
// against the 1e-3 budget, 85x inside it, and the control's 2 is the 1.59e-7 quoted elsewhere. That
// shape is what the mechanism predicts: this kernel used to round weighted_p exactly as the reference
// does, so that step contributed nothing to the count, and now differs from it by roughly the
// reference's own bf16 rounding, which puts a small population of large-magnitude elements just over
// the threshold. maxabs moving 0.125 -> 0.25 is the same effect - 0.25 only counts because that
// element's own magnitude is at least 4.8 under the rtol term.
//
// The prior in-tree position was that "no instruction saving is worth changing the frozen
// arithmetic". What changed is the price: results/prefill-next-levers-20261025.md valued f16 V at
// "~32 net instructions once P's f16 widening is paid for", costing the widening as exact
// (bf16 -> 2x cvt.f32.bf16 -> cvt.rn.f16x2.f32, 3 instructions per pair, spending 96 of the 128).
// cvt.rn.f16x2.f32 packs P for the same one instruction cvt.rn.bf16x2.f32 costs, so all 128 survive.
// FMHA_PRE_NO_VF16=1 restores the bf16 PV path as the control, and reproduces the pre-change default
// active-code hash exactly.
// DEFAULT OFF, REVERTED 2026-10-25 AFTER CERTIFICATION. dv_e5a04341327a
// (experiments/tiny_precision.sh, results/tiny-precision-defect-20261025.md) measured this path
// FAILING the frozen tolerance on distribution:'tiny' prefill shapes, which neither frozen suite
// samples: rel_l2 2.485e-02 on case 9's shape, 2.357e-02 on case 0's, 1.988e-02 on case 1's and
// 2.424e-02 on case 7's, against max_rel_l2 = 1e-2, with FMHA_PRE_NO_VF16=1 passing all four at
// 1.3e-04 to 2.2e-04. shape_range declares distribution in {normal, tiny, zero, balanced} for the whole
// supported domain, so this is a legal input, not an exotic one.
//
// MECHANISM, and why the range argument that justified shipping it was wrong. input.py:92 does
// v.mul_(2**-20) for 'tiny' while input.py:100 computes scale = (amax/448).clamp_min(1e-06), so v_scale
// collapses onto the 1e-6 clamp floor and P = weights * v_scale <= 1e-6 lands ENTIRELY inside f16's
// subnormal range - smallest normal 2^-14 = 6.1e-5, quantum 2^-24 = 6e-8 - where ~4 mantissa bits
// survive instead of 11. The argument written below bounds P against f16's CEILING (65504, unreachable
// because P <= max|v|/448) and dismisses the FLOOR as touching only terms "whose share of a row sum is
// below ~1e-7". That is the error: the floor does not bite per-term, it bites the whole vector at once,
// because every P in the row scales with the same v_scale while the denominator l sums the UNSCALED
// weights. So the loss is a uniform relative error on the output, not a negligible tail - exactly the
// ~2e-2 measured. bf16 has 8 exponent bits and never underflows here, which is why reference.py:43
// rounds weighted_p to q.dtype = bf16, and the prefill's own MYPRE_KF16 is unaffected because Q's f16
// subnormal side perturbs one term of a 128-term dot product by ~1e-9 relative.
//
// This shipped in 48510a6, was measured at -1.73%/-1.41%/-1.35% minima on cases 0/9/7, and was
// certified in tree e0ace9ae at 120.28169us. THE CERTIFICATE THEREFORE CONTAINS THIS DEFECT: it passed
// because the fixed suite's only 'tiny' case (2) is decode, where P is still packed to bf16, and the
// domain suite's single prefill config is not 'tiny'. Reverting here makes this tree correct; it does not
// un-certify e0ace9ae, which is an owner decision, and it costs the -0.45% geomean that VF16 contributed.
// FMHA_PRE_VF16=1 still selects the fast, defective path for measurement; FMHA_PRE_NO_VF16=1 is now
// redundant and kept only so the recorded hashes stay reproducible.
#if defined(FMHA_PRE_VF16) && !defined(FMHA_PRE_NO_VF16)
#define MYPRE_VF16 1
#endif

constexpr int TC_THREADS = 128; // 4 warps; tc prefill: one 32-row slice each
#define NEEDF_WARPS 8            // max warps in any prefill kernel (W8)
constexpr float NEG = -1e30f;

DEVI unsigned cvt_bf16x2_f32(float lo, float hi) {
    unsigned r;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
#ifdef MYPRE_VF16
// f16 PV: pack the softmax product straight to f16 instead of bf16. ONE instruction, the
// same count as cvt.rn.bf16x2.f32, so this is not a conversion saving - it is what lets the
// V image be f16, which is where the 4 instructions per pair go. The reference rounds
// weighted_p to q.dtype = bf16, and this rounds the SAME quantity more finely instead: f16
// carries 11 significand bits against bf16's 8, so the deviation introduced is the
// reference's own bf16 rounding, ~2^-9 relative, against a frozen max_rel_l2 of 1e-2 and a
// shipped worst of 1.4e-3. Products stay exact in the f32 accumulate (11 x 11 = 22 bits).
// P is in [0,1] times v_scale, so f16's 65504 ceiling is unreachable; its 6e-8 subnormal
// floor can only zero terms whose share of the row sum is below ~1e-7, and the row sum l is
// accumulated in f32 from the UNROUNDED p, so the denominator is untouched either way.
DEVI unsigned cvt_f16x2_f32(float lo, float hi) {
    unsigned r;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
#endif

#ifdef FMHA_PRE_VSMAXP
// ---- PRICE PROBE for the adaptive-shift f16-P design. Never shippable, changes nothing. ----
//
// WHY A MAX IS NEEDED AT ALL. f16 P fails on distribution:'tiny' because input.py:100 clamps the value
// scale to exactly 1e-6, so P = p*vs lands in f16's subnormal range (smallest normal 2^-14) and the
// absolute quantum 2^-25 is measured against an O that is itself ~1e-6: rel_l2 2e-2 against a 1e-2 bound
// (results/tiny-precision-defect-20261025.md). The unconditional fix is a per-tile power of two, folded
// into the exponent the softmax already subtracts, so that P' = p*vs*2^k is normal while k stays an exact
// f32 power of two that cancels between the O accumulator and l:
//     k = TARGET - floor(log2(vsmax_tile)),   mn' = mn - k,   alpha' = ex2(m_prev' - mn')
// and then O and l BOTH carry 2^k, so the epilogue's rcpa(l) is untouched and P' <= 2^TARGET by
// construction - no overflow for any finite input, no magnitude assumption, unlike a fixed prescale.
//
// The one ingredient that is not free is vsmax_tile. Every thread needs the SAME k (in W8 a row's O is
// split across threads, so a per-thread k would sum terms scaled differently), and each thread does not
// see every column: the tc sites read all BNT value scales in the pack loop, but W8 reads only BNT/4 of
// them. So this is priced as its own pass, BNT/4 vector loads and BNT-1 FMAX, ~79 instructions per thread
// per tile against a tile's ~712.
//
// WHAT THE NUMBER DECIDES. MYPRE_T_NOSOFT deleted ~470 softmax instructions per tile and made the kernel
// 10-11% SLOWER, so this region has slack rather than a throughput bound, and the honest expectation is
// that 79 more instructions cost much less than 11%. If the measured price is at most ~0.3% on the four
// prefill cases, implement the shift with this simple pass and take the -1.4/-1.7/-1.35% back correctly.
// If it is ~1% or more, the simple pass is dead and the max has to come from a warp reduction inside
// tc_convert (where R.vs is already in registers and step 5's __syncthreads already publishes it), which
// is ~18 instructions but touches the shared layout and every BN/NT/TPRW geometry.
//
// The result is consumed by a guard that can fire only for a NaN vsmax, which already implies a NaN
// output, so this arm MUST be bit-identical to ctl. It prices instructions, not arithmetic.
template <int BNT>
DEVI float vs_tile_max(const float* vss) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    #pragma unroll
    for (int j = 0; j < BNT; j += 4) {
        const float4 v = *reinterpret_cast<const float4*>(vss + j);
        a0 = fmaxf(a0, v.x); a1 = fmaxf(a1, v.y);
        a2 = fmaxf(a2, v.z); a3 = fmaxf(a3, v.w);
    }
    return fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
}
DEVI void vs_keepalive(float m, float* sc) { if (!(m >= 0.f)) sc[0] = 0.f; }
#endif

// v3.0: exact fp8(e4m3)x2 -> bf16x2 via f32 roundtrip: e4m3->f16 exact,
// f16->f32 exact, cvt.rn.bf16x2.f32 = RN-even == torch .to(bfloat16);
// +-0 and NaN preserved. ~4-5 SASS insts vs ~9 for the integer rebias
// trick, cutting the convert-stage instruction count ~2x.
// One-instruction e4m3x2 -> f16x2 (PTX ISA 7.8+, sm_89+). e4m3 carries 4 significand bits
// and |x| <= 448, so f16 (11 significand bits, |x| <= 65504) and bf16 (8) both hold it EXACTLY
// - the argument flashinfer writes down for its own native cvt.rn.bf16x2.e4m3x2 guard in
// csrc/msa_prefill_nvfp4_specialized.cu:486-501, which needs CUDA >= 13.2 and so is NOT
// available on this platform's 13.0. The f16 form is, and it is 1 instruction against the 5 of
// the f16->f32->bf16 roundtrip below.
// DIAGNOSTIC PRICE PROBES (MYPRE_T_KF16 / MYPRE_T_VF16), wrong numbers by construction: the K
// or V image is stored as f16 while the MMA descriptors still say bf16, so the products are
// garbage - but the instruction count, the dependency chain and the store pattern are EXACTLY
// what a real f16 K (or V) path would have. That prices the 4 instructions per pair a real
// conversion would remove (160 of the convert's ~384 instructions per thread per tile for K,
// the same again for V) without touching the Q staging or the idesc. Default off, never shippable.
#if defined(MYPRE_T_KF16) || defined(MYPRE_T_VF16) || defined(MYPRE_KF16) || defined(MYPRE_VF16)
DEVI unsigned e4m3x2_to_f16x2(unsigned short bytepair) {
    unsigned r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(bytepair));
    return r;
}
#ifdef MYPRE_KSFOLD
// One packed f16 multiply, written as PTX because this tree has no f16x2 ARITHMETIC anywhere - only
// cvt.rn.f16x2.* conversions - so there is no in-tree precedent to copy and no local nvcc to ask. If
// mul.rn.f16x2 did not lower to a single HMUL2 the fold would cost 4 instructions per pair instead of 1 and
// the arithmetic inverts from a net saving to a net loss, so the SASS instruction count of tc_convert is the
// first thing to read from any job that builds this.
DEVI unsigned f16x2_mul_rn(unsigned a, unsigned b) {
    unsigned r; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r;
}
DEVI unsigned f16x2_splat(float v) {
    unsigned r; asm("cvt.rn.f16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(v)); return r;
}
#endif
#ifdef MYPRE_KF16
// bf16x2 -> f16x2 through f32: three instructions, and exact for every value in f16's normal
// range, because bf16 carries 8 significand bits against f16's 11 so the mantissa only widens.
// Below 2^-14 f16 goes subnormal and loses mantissa bits; above 65504 it overflows. The frozen
// input distribution is normal in bf16, so the overflow side needs 65504 sigma and is
// unreachable, and the subnormal side touches ~5e-5 of samples, each in one term of a 128-term
// dot product whose other terms are O(1) - a perturbation of order 1e-9 relative against a
// frozen rel_l2 bound of 1e-2 and a shipped worst of 1.4e-3.
DEVI unsigned bf16x2_to_f16x2(unsigned b) {
    unsigned r;
    asm("{\n.reg .b16 lo, hi;\n.reg .f32 f0, f1;\n"
        "mov.b32 {lo, hi}, %1;\n"
        "cvt.f32.bf16 f0, lo;\n"
        "cvt.f32.bf16 f1, hi;\n"
        "cvt.rn.f16x2.f32 %0, f1, f0;\n}"
        : "=r"(r) : "r"(b));
    return r;
}
DEVI uint4 bf16x8_to_f16x8(uint4 v) {
    return make_uint4(bf16x2_to_f16x2(v.x), bf16x2_to_f16x2(v.y),
                      bf16x2_to_f16x2(v.z), bf16x2_to_f16x2(v.w));
}
#endif
#endif

DEVI unsigned e4m3x2_to_bf16x2(unsigned short bytepair) {
    unsigned r;
#ifdef MYPRE_VBF16
    // ---- MYPRE_VBF16: the native one-instruction form, BIT-IDENTICAL to the five-instruction chain below. ----
    // Every finite e4m3 value is exactly representable in bf16: e4m3 carries 4 exponent and 3 mantissa bits
    // against bf16's 8 and 7, and its whole range (2^-9 subnormal to 448) sits inside bf16's 2^-126..2^127,
    // so there is no rounding in either direction and no subnormal case to special-case. The chain below is
    // already exact for the same reason - e4m3->f16 exact, f16->f32 exact, f32->bf16 RN of an exactly
    // representable value is exact - so the two spellings produce the same bits and this is a pure
    // instruction-count change, testable by a bit-identity gate rather than a tolerance.
    //
    // WHY IT IS WORTH A JOB. This is the last five-instruction convert chain in the default build. MYPRE_KF16
    // already replaced the identical chain on the K side with cvt.rn.f16x2.e4m3x2 alone, taking that half of
    // tc_convert from 160 to 32 instructions per thread per tile, and it measured -3.5%/-4.5% on cases 0/9
    // (dv_fdd2a0699e7e). V is the same 32 pairs per thread per tile at BNT=64/D=128, so the same 128
    // instructions are on the table here. V could not follow K into f16 because the PV mma needs both operands
    // in one format and P must stay bf16 - reference.py:43 rounds weighted_p to q.dtype, and the f16 P that
    // MYPRE_VF16 tried underflows on 'tiny' (fmha_pre.cu:296-306). Shortening the bf16 conversion itself is
    // the only route that keeps the shipped dtype and therefore the only one that is free.
    //
    // THE RISK IS ASSEMBLY, NOT ARITHMETIC. fmha.cu:96 and the note at line 224 both record that flashinfer
    // guards the native cvt.rn.bf16x2.e4m3x2 behind CUDA >= 13.2 (csrc/msa_prefill_nvfp4_specialized.cu:486-501)
    // while this platform has 13.0 - but that is a claim inherited from a reference library's guard, never
    // tested against this ptxas, and the sibling cvt.rn.f16x2.e4m3x2 in the very next function does assemble.
    // If the mnemonic is rejected the build fails immediately and cheaply, which is itself the answer; nothing
    // shipped is affected because the flag defaults off.
    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(bytepair));
#else
    asm("{\n.reg .f16x2 h2;\n.reg .f16 h0, h1;\n.reg .f32 f0, f1;\n"
        "cvt.rn.f16x2.e4m3x2 h2, %1;\n"
        "mov.b32 {h0, h1}, h2;\n"
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "cvt.rn.bf16x2.f32 %0, f1, f0;\n}"
        : "=r"(r) : "h"(bytepair));
#endif
    return r;
}

#ifdef MYPRE_VSFOLD
// V's image carrying value_scale instead of the softmax paying for it. Same exact e4m3 -> f16 -> f32
// widening, then ONE f32 multiply by the key's value_scale, then the same single cvt.rn.bf16x2.f32 rounding -
// so the V side keeps exactly one bf16 rounding, the same count the reference spends on weighted_p at
// reference.py:43. Two f32 multiplies per pair, +64 instructions per thread per tile in the producer, against
// 16 broadcast float4 loads plus 64 FMULs deleted from the softmax.
//
// Why f32 and not a packed mul.rn.bf16x2: the packed form is one instruction per pair instead of two and would
// be the faster design, but it rounds value_scale to bf16 first, adding a THIRD rounding where the reference has
// one. dv_daaca9e60222 is the reason to buy the cheaper error budget rather than the cheaper instruction count -
// that arm's rel_l2 stayed inside the frozen 1e-2 bound and it still failed on mismatch count, because the
// frozen checker's binding constraint is the number of elements outside atol+rtol*|actual|, not rel_l2. This
// form keeps the deviation at ~1.4x of ctl's instead of ~1.7x for the same instruction budget order.
//
// Why f32 also decides the RANGE question, which is what killed MYPRE_VF16. bf16 shares f32's 8 exponent bits,
// so V*value_scale cannot underflow on 'tiny' (value_scale there is ~2^-28.8, and V*value_scale is the original
// bf16 magnitude by construction, since value_scale = max|v_bf16|/448). f16 could not do this: its 6e-8 floor
// is above what 'tiny' produces, which is the ~2e-2 defect recorded at fmha_pre.cu:296-306. No power-of-two
// shift is needed and none is used, which is what separates this from the archived f16+shift V-side design.
DEVI unsigned e4m3x2_to_bf16x2_scaled(unsigned short bytepair, float scale) {
    unsigned r;
    asm("{\n.reg .f16x2 h2;\n.reg .f16 h0, h1;\n.reg .f32 f0, f1;\n"
        "cvt.rn.f16x2.e4m3x2 h2, %2;\n"
        "mov.b32 {h0, h1}, h2;\n"
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "mul.f32 f0, f0, %1;\n"
        "mul.f32 f1, f1, %1;\n"
        "cvt.rn.bf16x2.f32 %0, f1, f0;\n}"
        : "=r"(r) : "f"(scale), "h"(bytepair));
    return r;
}
#ifdef MYPRE_VBF16P
// The packed spelling of the same fold: two instructions per pair instead of seven. e4m3 -> bf16 is EXACT
// (4 exponent and 3 mantissa bits into 8 and 7), so cvt.rn.satfinite.bf16x2.e4m3x2 loses nothing against the
// f32 round trip, and the multiply then happens in bf16x2 instead of f32.
//
// This is a SECOND attempt at the native convert. dv_845f7401393e proved that plain cvt.rn.bf16x2.e4m3x2 is
// rejected by this platform's ptxas ("Unexpected instruction types specified for 'cvt'", 35 sites, CUDA 13.0 /
// sm_103a), which is what results/prefill-vbf16-cvt-20261026.md recorded as closing the direction. What that
// record did not test is the .satfinite FORM, and the fp8 conversions in the PTX ISA are documented with
// .satfinite as the spelling for the packed 16-bit destinations - the sibling cvt.rn.f16x2.e4m3x2 that this
// tree already uses is the one fp8 convert documented WITHOUT it. So the rejection may have been a missing
// modifier rather than a missing instruction, and this arm is the test. If it fails to assemble the answer is
// definitive in the other direction and costs about two minutes, which is what dv_845f7401393e cost.
//
// WHAT IT BUYS IF IT ASSEMBLES: V's convert drops from 7 instructions per pair to 2, i.e. 160 fewer
// instructions per thread per tile at BNT=64/D=128. MYPRE_KF16 removed 128 from the K half of this same
// function for -3.5%/-4.5% on cases 0/9 (dv_fdd2a0699e7e), so 160 is worth roughly -5% at that rate. That
// would make the V convert - not the softmax - the largest block in the loop, which is a consequence of
// MYPRE_VSFOLD having just added two FMULs per pair to it.
//
// WHAT IT COSTS: value_scale gets rounded to bf16 before the multiply, so the V side carries TWO bf16
// roundings instead of one. The extra error is per-key and therefore shared across all query rows for that key,
// which is the part that does not average away as cleanly as an independent per-element rounding; the weighted
// mean of ~2^-9 relative perturbations over the effective number of dominant keys is what it adds to the
// ~3e-3 MYPRE_VSFOLD already carries against a frozen 1e-2 bound. Whether that is affordable is a measurement,
// not an argument, so stage 1 reports rel_l2 AND mismatched on all nine shapes before any timing is read.
//
// CLOSED (dv_701a3aaf0978, results/vbf16p-satfinite-20261028.md): ptxas 13.0/sm_103a rejects the satfinite
// spelling at all 35 sites ("Illegal modifier '.satfinite'" AND "Unexpected instruction types"). With the
// plain form rejected by dv_845f7401393e, neither spelling of the native fp8->bf16x2 convert exists below
// flashinfer's documented CUDA 13.2 guard. The numerical question was never reached; the arm stays wired
// for a toolchain that assembles it.
DEVI unsigned e4m3x2_to_bf16x2_sat(unsigned short bytepair) {
    unsigned r;
    asm("cvt.rn.satfinite.bf16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(bytepair));
    return r;
}
DEVI unsigned bf16x2_mul_rn(unsigned a, unsigned b) {
    unsigned r;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
DEVI unsigned bf16x2_splat(float v) {
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(r) : "f"(v));
    return r;
}
#endif
#endif

// This thread's row's block-table entry for tile start key0, clamped;
// enables one-tile-ahead bt prefetch so the bt LDG no longer gates the KV
// data loads. ROW-AWARE: row = tid/TPR must match kv_prefetch's mapping
// (v3.0 shipped a row-0-only version -> wrong page for rows >0 when
// page_size < BN; 8 correctness failures, job dv_ae699d830f94).
//
// TPRW (threads-per-row, 0 = the shipped single-pass NT/BN) selects how many
// threads cooperate on one KV row and therefore how many PASSES a tile needs:
// PASSES = ceil(BN*TPRW/NT), rows per pass RPW = NT/TPRW, and pass p owns row
// p*RPW + tid/TPRW. With TPRW=0, PASSES=1 and RPW=BN so the mapping is exactly
// the shipped one. Each pass has its own row, hence its own block-table entry.
template <int BN, int NT = TC_THREADS, int TPRW = 0>
DEVI void kv_bt_fill(const Params& P, int b, int key0, int kvlen, int* out,
                     int tbase = 0) {
    constexpr int TPR = (TPRW > 0) ? TPRW : (NT / BN);
    constexpr int RPW = NT / TPR;
    constexpr int PASSES = (TPRW > 0) ? ((BN * TPRW + NT - 1) / NT) : 1;
    const int rbase = (threadIdx.x - tbase) / TPR;
    #pragma unroll
    for (int p = 0; p < PASSES; ++p) {
        int k = min(max(key0 + p * RPW + rbase, 0), kvlen - 1);
        out[p] = __ldg(P.block_table + b * P.bt_stride + (k >> P.page_shift));
    }
}

// single-pass form kept for the W8 / rt2p call sites (TPRW = 0)
template <int BN, int NT = TC_THREADS>
DEVI int kv_bt_next(const Params& P, int b, int key0, int kvlen, int tbase = 0) {
    int one[1];
    kv_bt_fill<BN, NT, 0>(P, b, key0, kvlen, one, tbase);
    return one[0];
}

// Per-thread staging registers for one KV tile. Rows are owned as
// (row = p*RPW + tid/TPR, part = tid%TPR) for pass p < PASSES; TPR is
// NT/BN for the shipped single-pass mapping (PASSES=1) or the requested
// TPRW. KU64/VU32 are the TOTAL per-thread chunk counts over all passes,
// which for every tc instantiation (D=64/128/256, BN=64, NT=128) and both
// TPRW=4 and TPRW=8 equal the single-pass values, so the remap costs no
// registers: PASSES * D/(8*TPRW) == D/(8*(NT/BN)) when BN*TPRW == PASSES*NT.
template <int D, int BN, int NT = TC_THREADS, int TPRW = 0>
struct ConvRegs {
    static constexpr int TPR = (TPRW > 0) ? TPRW : (NT / BN);  // threads per row
    static constexpr int RPW = NT / TPR;                        // rows per pass
    static constexpr int PASSES = (TPRW > 0) ? ((BN * TPRW + NT - 1) / NT) : 1;
    static constexpr int KPP = D / (8 * TPR);    // 8B K chunks per thread per pass
    static constexpr int VPP = D / (4 * TPR);    // 4B V chunks per thread per pass
    static constexpr int KU64 = PASSES * KPP;    // 8B K chunks per thread
    static constexpr int VU32 = PASSES * VPP;    // 4B V chunks per thread
    static_assert(NT % TPR == 0, "TPR must divide the CTA thread count");
    static_assert(PASSES * RPW >= BN, "passes must cover every row of the tile");
    static_assert(D % (8 * TPR) == 0, "K row must split into whole 8B chunks");
    uint2 k[KU64];
    unsigned v[VU32];
    float ks[PASSES], vs[PASSES];   // scales for this thread's row of each pass
};

// Which D-chunk of a row this thread fetches at step i. v3.8 interleaves
// (c = part + TPR*i) so the TPR threads sharing a row issue TPR CONTIGUOUS chunks
// and one warp instruction stays inside as few 32 B sectors as the frozen record
// stride allows; the blocked alternative (c = part*nch + i) splits a row into TPR
// contiguous runs, so lanes of one instruction land nch*chunk bytes apart and each
// 8 B or 4 B piece pulls its own sector. Both visit the same chunks and store them
// to the same smem offsets, so they are bit-identical and differ ONLY in how many
// L1 wavefronts the same bytes cost - which makes the blocked form a sensitivity
// probe for L1 access pattern on a kernel whose busiest unit is L1TEX at 62-67%
// with nothing else saturated. FMHA_PRE_BLKCHUNK=1 selects it; never shipped.
// Measured by dv_542ca2846b78: blocked costs +4.2/+4.6% on the two tc cases
// (results/prefill-l1-pattern-20261024.md).
DEVI int kv_chunk(int part, int tpr, int i, int nch) {
#ifdef FMHA_PRE_BLKCHUNK
    (void)tpr;
    return part * nch + i;
#else
    (void)nch;
    return part + tpr * i;
#endif
}

// Issue global loads for tile [key0, key0+nvalid) into registers. Rows beyond
// nvalid are skipped (convert zero-fills them); addresses never go OOB.
// blks[] holds one block-table entry per pass (see kv_bt_fill).
template <int D, int BN, int NT = TC_THREADS, int TPRW = 0>
DEVI void kv_prefetch_m(ConvRegs<D, BN, NT, TPRW>& R, const Params& P, int kvh,
                        int key0, int nvalid, const int* blks, int tbase = 0) {
    using CR = ConvRegs<D, BN, NT, TPRW>;
    constexpr int TPR = CR::TPR, RPW = CR::RPW, PASSES = CR::PASSES;
    const int tid = threadIdx.x - tbase;
    const int rbase = tid / TPR;
    const int part = tid % TPR;
    const int pmask = (1 << P.page_shift) - 1;
    #pragma unroll
    for (int p = 0; p < PASSES; ++p) {
        const int row = p * RPW + rbase;
        if (row < nvalid) {
            const int blk = blks[p];
            const int key = key0 + row;
            const long long krow = (long long)blk * P.k_sb + (long long)(key & pmask) * P.k_sp + kvh * P.k_sh;
            const uint8_t* gk = P.k + krow;
            // v3.8: interleaved chunk ownership c = part + TPR*i. A warp
            // instruction now covers TPR CONTIGUOUS chunks per row: full 32B
            // sectors at TPR=4 (was 8B/sector: part*(D/TPR)+i*8 spread lanes
            // 64B apart -> 4x DRAM amplification + 32 L1 wavefronts/instr on
            // the D256 decode path; also 2x better at TPR=2). With TPRW>1 the
            // tile is walked in PASSES row-bands, so one instruction covers
            // NT/TPRW rows x TPR contiguous chunks: 8 rows x 32B at TPRW=4,
            // 4 rows x 64B at TPRW=8.
            #pragma unroll
            for (int i = 0; i < CR::KPP; ++i)
#ifdef FMHA_PRE_LDGNC
                // The paged KV cache is read-only for the whole kernel and never aliases
                // the output, which is the same assumption this function ALREADY makes two
                // lines below by loading the per-token scales through __ldg. Taking the
                // non-coherent path for the 24 data loads as well is what lets the compiler
                // treat them as free of any aliasing with the in-place `out` writes.
                R.k[p * CR::KPP + i] = __ldg(reinterpret_cast<const uint2*>(
                    gk + kv_chunk(part, TPR, i, CR::KPP) * 8));
#else
                R.k[p * CR::KPP + i] = *reinterpret_cast<const uint2*>(
                    gk + kv_chunk(part, TPR, i, CR::KPP) * 8);
#endif
            const long long vrow = (long long)blk * P.v_sb + (long long)(key & pmask) * P.v_sp + kvh * P.v_sh;
            const uint8_t* gv = P.v + vrow;
            #pragma unroll
            for (int i = 0; i < CR::VPP; ++i)
#ifdef FMHA_PRE_LDGNC
                R.v[p * CR::VPP + i] = __ldg(reinterpret_cast<const unsigned*>(
                    gv + kv_chunk(part, TPR, i, CR::VPP) * 4));
#else
                R.v[p * CR::VPP + i] = *reinterpret_cast<const unsigned*>(
                    gv + kv_chunk(part, TPR, i, CR::VPP) * 4);
#endif
            if (part == 0) {
                const int slot = key & pmask;
                R.ks[p] = __ldg(P.ks + blk * P.ks_sb + (long long)slot * P.ks_sp + kvh * P.ks_sh);
                R.vs[p] = __ldg(P.vs + blk * P.vs_sb + (long long)slot * P.vs_sp + kvh * P.vs_sh);
            }
        }
    }
}

// single-pass form kept for the W8 / rt2p call sites (TPRW = 0)
template <int D, int BN, int NT = TC_THREADS>
DEVI void kv_prefetch(ConvRegs<D, BN, NT>& R, const Params& P, int kvh,
                      int key0, int nvalid, int blk, int tbase = 0) {
    const int one[1] = {blk};
    kv_prefetch_m<D, BN, NT, 0>(R, P, kvh, key0, nvalid, one, tbase);
}

constexpr int TCBM = 128;  // q rows per tc CTA

// Threads-per-row for the tc kernel's KV staging. 0 (the shipped default) is the
// single-pass NT/BN = 2 mapping; FMHA_PRE_TPRW=4 or 8 stages each 64-key tile in
// 2 or 4 row-band passes at 4 or 8 threads per row, which cuts global-load
// sectors per warp instruction toward the W8 kernel's measured 11.31 (tc is
// 18.97) at unchanged register array sizes and unchanged smem store wavefronts
// for TPRW=4 (2x worse for TPRW=8, where a warp covers only 4 of the 8 rows that
// km_off/v_off's XOR swizzle spreads over). Bit-identical to TPRW=0 by
// construction: same chunks, same smem offsets, permuted ownership. Priced by
// dv_542ca2846b78 (results/prefill-l1-pattern-20261024.md); never shipped until
// an A/B says so. Only fa_prefill_tc consults this; W8 and rt2p stay single-pass.
#ifdef MYPRE_CLUSTER
constexpr int TC_CL = MYPRE_CLUSTER;
#else
constexpr int TC_CL = 1;
#endif
#ifdef FMHA_PRE_TPRW
constexpr int TC_TPRW = FMHA_PRE_TPRW;
static_assert(TC_TPRW == 0 || TC_TPRW == 4 || TC_TPRW == 8,
              "FMHA_PRE_TPRW must be 0, 4 or 8 at NT=128");
#else
constexpr int TC_TPRW = 0;
#endif
// Per-D staging width, from dv_b5739e5f0521 (results/tcloop3-20261028.md): FMHA_PRE_TPRW=4 cut the
// two tc-D64 shapes by -16.7%/-18.0% bit-identically (l6 static 135.2->112.6, t1 long 249.9->204.8)
// but cost +28% on the D128 cases (c0/c9) - the multi-pass staging helps when a row is only 64 fp8
// wide and drowns the D128 image in extra passes. FMHA_PRE_TPRW64=<4|8> therefore applies the width
// to the D==64 instantiation ONLY; with the flag unset TC_TPRW_D64 == TC_TPRW and every instantiation
// compiles byte-identically to the shipped build. W8 and rt2p never consult either constant.
#ifdef FMHA_PRE_TPRW64
constexpr int TC_TPRW_D64 = FMHA_PRE_TPRW64;
static_assert(TC_TPRW_D64 == 4 || TC_TPRW_D64 == 8,
              "FMHA_PRE_TPRW64 must be 4 or 8 at NT=128");
#else
constexpr int TC_TPRW_D64 = TC_TPRW;
#endif

// Null-probe knob for the DSMEM split idea (results/prefill-work-balance-20261024.md):
// launch fa_prefill_tc UNCHANGED inside an n-CTA thread-block cluster. Nothing in
// the kernel reads cluster state and no cluster barrier is added, so the arms are
// bit-identical and the only things measured are the ones that could kill the
// split before it is built - cluster launch overhead, GPC-constrained co-residency
// against the 2 CTAs/SM the TMEM column budget already imposes, and the changed
// block rasterisation. grid.x is rounded up to a multiple of n; the extra CTAs
// leave through the path an out-of-range blockIdx.x already takes (b_s < 0 ->
// return, or the persist ticket's work0 >= T_all -> return).
#ifdef MYPRE_CLUSTER
#define MYPRE_CLUSTER_ATTR __cluster_dims__(MYPRE_CLUSTER, 1, 1)
#define MYPRE_CLUSTER_GX(gx) \
    ((unsigned)((((unsigned)(gx) + MYPRE_CLUSTER - 1) / MYPRE_CLUSTER) * MYPRE_CLUSTER))
#else
#define MYPRE_CLUSTER_ATTR
#define MYPRE_CLUSTER_GX(gx) ((unsigned)(gx))
#endif

// Shared addresses this kernel passes to hardware as integers split two ways, and
// dv_c4ae20c486be + dv_9d33c5c79c6a measured both halves of the distinction:
//
//   * The instructions - tcgen05.alloc's smem destination, tcgen05.commit,
//     mbarrier.init/try_wait - all take a full 32-bit address. In a cluster
//     __cvta_generic_to_shared returns the shared::cluster form with the CTA rank
//     in bits [24:31] (measured: rank 0 raw=00008400, rank 1 raw=01008400), and
//     masking those bits off made the launch die on cudaErrorIllegalInstruction
//     even with tcgen05.commit, the one shared::cluster consumer, left unmasked.
//     So they get the address the intrinsic returns, rank bits and all.
//   * tc_desc's addr>>4 field is 14 bits wide. A rank-1 CTA's bit 24 lands in bit
//     20 of that field, which is the NaN both tc cases returned when clustered with
//     the body untouched. This is the only consumer where the rank cannot fit, and
//     the only one that needs the CTA-local offset stripped.
//
// Without MYPRE_SHARED_MASK smem_desc_u32 is the identity and the default build is
// unchanged.
DEVI uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// Not optional under a cluster: dv_2c980c231e11 ran clustered WITHOUT this mask
// and returned NaN on both tc cases, and dv_2d39f799962b's split arm repeated it
// as a cudaErrorLaunchFailure - a rank-1 CTA's bit 24 lands in bit 20 of the
// descriptor, which is the leading-byte-offset field, so the mma walks smem out of
// range. MYPRE_SHARED_MASK remains as the explicit override the two mask jobs used.
DEVI uint32_t smem_desc_u32(const void* p) {
    uint32_t a = smem_u32(p);
#if defined(MYPRE_CLUSTER) || defined(MYPRE_SHARED_MASK)
    a &= 0x00FFFFFFu;
#endif
    return a;
}
#ifdef MYPRE_CLUSTER_DBG
// Persist across launches in one process, so a launch can report how many
// clusters the PREVIOUS one retired: dbg_done is incremented once per cluster by
// rank 0 on the way out and read by block 0 on the way in. If launch N starts with
// dbg_done < N-1 clusters' worth, some cluster died inside launch N-1; if it starts
// with the full count, the fault is in launch N itself.
__device__ unsigned dbg_launch = 0;
__device__ unsigned dbg_done = 0;
#endif

// ---- thread-block-cluster primitives --------------------------------------
// Only meaningful in a clustered launch (MYPRE_CLUSTER); every call site is under
// `if constexpr (TC_CL > 1)` or a runtime cl_split that the host only sets when
// the launch is clustered, so a default build never emits them.
DEVI unsigned cluster_ctarank() {
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
// arrive.release + wait.acquire: makes one rank's generic-proxy smem writes
// visible to the others. .aligned = every thread of every CTA must execute it,
// so these are never called from inside a divergent branch.
DEVI void cl_sync() {
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
// map a CTA-local shared address into rank `r`'s window
DEVI uint32_t cl_mapa(uint32_t local, unsigned r) {
    uint32_t a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(local), "r"(r));
    return a;
}
DEVI unsigned cl_ldu32(uint32_t ca) {
    unsigned v;
    asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(ca) : "memory");
    return v;
}
DEVI float cl_ldf(uint32_t ca) {
    float v;
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ca) : "memory");
    return v;
}
// Currently unused: the merge reads O with scalar cl_ldf (see the comment there).
// Kept because it is the prime suspect for dv_a73983629636's CL=4 launch failure
// and a later pass may want to re-test it in isolation.
DEVI float4 cl_ldf4(uint32_t ca) {
    unsigned x, y, z, w;
    asm volatile("ld.shared::cluster.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(ca) : "memory");
    return make_float4(__uint_as_float(x), __uint_as_float(y),
                       __uint_as_float(z), __uint_as_float(w));
}

DEVI void tc_fence_async_smem() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
DEVI void tc_fence_before() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
DEVI void tc_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
DEVI void tc_alloc(uint32_t* smem_dst, int ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_u32(smem_dst)),
                    "r"(ncols));
}
DEVI void tc_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
DEVI void tc_dealloc(uint32_t tm, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tm),
                 "r"(ncols));
}
DEVI void tc_mma_ss(uint32_t d_tmem, uint64_t da, uint64_t db, uint32_t idesc,
                    uint32_t scale_c) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                 :: "r"(d_tmem), "l"(da), "l"(db), "r"(idesc), "r"(scale_c));
}
DEVI void tc_mma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t db, uint32_t idesc,
                    uint32_t scale_c) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                 :: "r"(d_tmem), "r"(a_tmem), "l"(db), "r"(idesc), "r"(scale_c));
}
#ifdef FMHA_PRE_F8QK
// ---- fp8-direct QK mechanism (FMHA_PRE_F8QK, D128) ----
// Probe-verified on this device by experiments/f8probe.cu across three jobs
// (results/f8probe-20261028.md): kind::f8f6f4 with idesc atype/btype = 0
// (E4M3) accumulates the dual hi/lo split BIT-EXACT against fp64 references
// built from the same bytes; subnormal e4m3 inputs are consumed exactly (no
// FTZ); the SW128 K-major descriptor convention (atom = 8 rows x 128B with
// the 16B-chunk XOR-row swizzle, LBO = 0, SBO = 1024, desc addr +2 per 32B
// K-step) follows the public DeepGEMM sm100 make_umma_desc pattern and is
// probe-verified at production geometry (M=128, N=BNT, KD=128); the
// dependent-chain rate is 0.78-0.82 of the f16 K=16 step, so the dual split
// (2 x D/32 f8 steps vs D/16 f16 steps) is rate-favorable BEFORE deleting
// the K convert. The cvt spellings below are the probe-verified PTX: the
// __nv_cvt_float_to_fp8 / operator-float round-trip corrupts the residual
// under the torch extension's -D__CUDA_NO_* conversion macros (v2 defect).
DEVI void tc_mma_f8(uint32_t d_tmem, uint64_t da, uint64_t db, uint32_t idesc,
                    uint32_t scale_c) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}\n"
                 :: "r"(d_tmem), "l"(da), "l"(db), "r"(idesc), "r"(scale_c));
}
DEVI uint16_t f32x2_to_e4m3x2(float a_hi, float b_lo) {
    uint16_t r;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
                 : "=h"(r) : "f"(a_hi), "f"(b_lo));
    return r;   // [15:8] = e4m3(a_hi), [7:0] = e4m3(b_lo)
}
DEVI float e4m3x2_lo_to_f32(uint16_t p2) {
    // Self-contained on purpose: the tree's e4m3x2_to_f16x2 is defined only
    // under the KF16/VF16/KSFOLD flag guards, and F8QK must build without them.
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(h2) : "h"(p2));
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f)
                 : "h"((unsigned short)(h2 & 0xFFFFu)));
    return f;
}
DEVI float e4m3x2_hi_to_f32(uint16_t p2) {
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(h2) : "h"(p2));
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f)
                 : "h"((unsigned short)(h2 >> 16)));
    return f;
}
DEVI float bf16u_to_f32(unsigned short b) {
    float f;
    asm volatile("cvt.f32.bf16 %0, %1;\n" : "=f"(f) : "h"(b));
    return f;
}
#endif  // FMHA_PRE_F8QK
DEVI void tc_commit(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(smem_u32(bar)) : "memory");
}
DEVI void tc_wait(uint64_t* bar, int parity) {
    asm volatile("{\n.reg .pred p;\nTCW_%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra.uni TCW_%=;\n}\n"
                 :: "r"(smem_u32(bar)),
                    "r"(parity));
}
#ifdef FMHA_PRE_WAITELECT
// Elected-lane mbarrier wait. tc_wait has all 128 threads of the CTA polling one shared address
// in a try_wait loop, and every poll round is a shared-load wavefront on L1TEX - the unit
// results/prefill-profile-v2-20261023.md puts at the top of the prefill tree, with 42.4M shared
// LOAD wavefronts against 68.3M stores for a kernel whose only shared readers are the block
// table and these polls. Case 0 is 603 tiles/CTA x 296 CTAs x 2 waits x 4 warps = 1.4M
// wavefronts if each wait polls once, and a wait that spans an MMA has no reason to poll once.
// One lane per warp polls and the other 31 wait on __syncwarp(), which is a scheduler-level sync
// with no memory traffic: 32x fewer shared reads for the same information. Ordering is unchanged
// - the mbarrier phase flip is the MMA's physical completion, not per-thread state, so once the
// elected lane has observed it the TMEM results exist for the whole warp, and __syncwarp() is a
// compiler barrier and a warp-scope fence, so no lane's tcgen05.ld can be hoisted above it.
DEVI void tc_wait_e(uint64_t* bar, int parity) {
    if ((threadIdx.x & 31) == 0) tc_wait(bar, parity);
    __syncwarp();
}
#endif
#if defined(FMHA_PRE_NBSYNC) || defined(FMHA_PRE_NBSYNC_W8) || defined(FMHA_PRE_NBSYNC_W8D128)
#if defined(FMHA_PRE_NBSYNC_W8) && defined(FMHA_PRE_NBSYNC_W8D128)
#error "FMHA_PRE_NBSYNC_W8 (ungated, measured +9.3% on c7's BNT96 kernel) and FMHA_PRE_NBSYNC_W8D128 (D128-only gate) are two arms of one question; pick one."
#endif
// DEADLOCKS IN THE 4-WARP KERNEL - REJECTED THERE, do not enable FMHA_PRE_NBSYNC. dv_dd5604989618 stage 1 hung on fixed case 9 (killed at
// 420 s after 122 s of builds, where the same shape's setup measured 33 s in dv_757c265cd495),
// and the mechanism is understood: need_cta is CTA-uniform ONLY because every warp reads
// S.needf after step 5's barrier, so weakening that barrier into arrive/sync lets warps 1-3 read
// needf before the other warps have written it. They then disagree with warp 0 about whether
// step 7 runs at all, one side executes bar.sync or bar.arrive alone, the count of 128 is never
// reached, and the CTA waits forever. The two small shapes in experiments/shapes_tclive.json ran
// it without tripping the race, which is why this needed a real geometry to find.
//
// Its performance before that is also on record and is not a reason to fix it: +0.8% on the
// static shape (241.7 -> 243.7, min 239.7 -> 241.7) and neutral on persist (182.4 -> 182.3),
// i.e. letting three warps skip the rendezvous buys nothing, which is consistent with the four
// deletion probes pricing step 5's barrier, its three fences and step 7's barrier at -0.01%,
// -0.44% and -0.02% on those same shapes.
//
// THE 8-WARP KERNEL IS DIFFERENT, and is what FMHA_PRE_NBSYNC_W8 measures: fa_prefill_w8 has no
// need_cta, so nothing outside the barrier itself decides whether the barrier runs and the deadlock
// above cannot arise. Its case 7 profile (dv_b3a58e521acc, current tree) has `barrier` 1.46 as the
// TOP stall per issued instruction against the tc kernel's 0.64, with issue_active 57.4% the
// highest of the four prefill cases; and this kernel's own j23-w8bd note records that dropping ONE
// barrier per tile was worth +6.1% on case 1 against +0.86% for the 4-warp kernel, because
// syncthreads cost and stragglers scale with warp count - 8 warps must all arrive.
//
// Split named barrier. bar.arrive is a NON-BLOCKING arrival: the arriving threads continue and
// only the threads that execute bar.sync wait, with the same publication guarantee for them as
// __syncthreads() would give, at the same instruction count. CNT is the whole CTA (arrivers +
// waiters), and the waiting warp must NOT also arrive - bar.sync is itself its arrival, so
// arriving twice would over-count the phase. Legal here because every thread of the CTA reaches
// both sites: the PIPE loop has no per-warp exit, `more` is CTA-uniform, and the early returns
// are all before the loop. IDs 1 and 2 are free; __syncthreads() uses 0.
template <int ID, int CNT>
DEVI void nb_arrive() { asm volatile("bar.arrive %0, %1;" :: "n"(ID), "n"(CNT)); }
template <int ID, int CNT>
DEVI void nb_sync() { asm volatile("bar.sync %0, %1;" :: "n"(ID), "n"(CNT) : "memory"); }
#endif  // FMHA_PRE_NBSYNC || FMHA_PRE_NBSYNC_W8

DEVI void tc_ld_32(float* dst, uint32_t addr) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                 "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
                 "[%32];"
                 : "=f"(dst[0]),"=f"(dst[1]),"=f"(dst[2]),"=f"(dst[3]),
                   "=f"(dst[4]),"=f"(dst[5]),"=f"(dst[6]),"=f"(dst[7]),
                   "=f"(dst[8]),"=f"(dst[9]),"=f"(dst[10]),"=f"(dst[11]),
                   "=f"(dst[12]),"=f"(dst[13]),"=f"(dst[14]),"=f"(dst[15]),
                   "=f"(dst[16]),"=f"(dst[17]),"=f"(dst[18]),"=f"(dst[19]),
                   "=f"(dst[20]),"=f"(dst[21]),"=f"(dst[22]),"=f"(dst[23]),
                   "=f"(dst[24]),"=f"(dst[25]),"=f"(dst[26]),"=f"(dst[27]),
                   "=f"(dst[28]),"=f"(dst[29]),"=f"(dst[30]),"=f"(dst[31])
                 : "r"(addr));
}
DEVI void tc_st_32(uint32_t addr, const unsigned* w) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
                 "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
                 "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
                 :: "r"(addr),
                    "r"(w[0]),"r"(w[1]),"r"(w[2]),"r"(w[3]),"r"(w[4]),"r"(w[5]),
                    "r"(w[6]),"r"(w[7]),"r"(w[8]),"r"(w[9]),"r"(w[10]),"r"(w[11]),
                    "r"(w[12]),"r"(w[13]),"r"(w[14]),"r"(w[15]),"r"(w[16]),"r"(w[17]),
                    "r"(w[18]),"r"(w[19]),"r"(w[20]),"r"(w[21]),"r"(w[22]),"r"(w[23]),
                    "r"(w[24]),"r"(w[25]),"r"(w[26]),"r"(w[27]),"r"(w[28]),"r"(w[29]),
                    "r"(w[30]),"r"(w[31]));
}
DEVI void tc_st_16(uint32_t addr, const unsigned* w) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
                 "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                 :: "r"(addr),
                    "r"(w[0]),"r"(w[1]),"r"(w[2]),"r"(w[3]),"r"(w[4]),"r"(w[5]),
                    "r"(w[6]),"r"(w[7]),"r"(w[8]),"r"(w[9]),"r"(w[10]),"r"(w[11]),
                    "r"(w[12]),"r"(w[13]),"r"(w[14]),"r"(w[15]) : "memory");
}
// 16-lane TMEM shapes (W8; probe-verified mappings, see tc_probe5/6):
//   x8 ld / x4 st: reg r -> row +8*((r>>1)&1), col (l&3)*2+(r&1)+8*(r>>2)
//   16x128b.x2 st: reg r' -> row +8*(r'&1), col off+(l&3)+4*(r'>>1)
DEVI void tc_ld16_x8(unsigned* dst, uint32_t addr) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x8.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                 "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
                 "[%32];"
                 : "=r"(dst[0]),"=r"(dst[1]),"=r"(dst[2]),"=r"(dst[3]),
                   "=r"(dst[4]),"=r"(dst[5]),"=r"(dst[6]),"=r"(dst[7]),
                   "=r"(dst[8]),"=r"(dst[9]),"=r"(dst[10]),"=r"(dst[11]),
                   "=r"(dst[12]),"=r"(dst[13]),"=r"(dst[14]),"=r"(dst[15]),
                   "=r"(dst[16]),"=r"(dst[17]),"=r"(dst[18]),"=r"(dst[19]),
                   "=r"(dst[20]),"=r"(dst[21]),"=r"(dst[22]),"=r"(dst[23]),
                   "=r"(dst[24]),"=r"(dst[25]),"=r"(dst[26]),"=r"(dst[27]),
                   "=r"(dst[28]),"=r"(dst[29]),"=r"(dst[30]),"=r"(dst[31])
                 : "r"(addr));
}
// j23-bnt96: 16x256b.x4 LD (16 regs = cols off..off+31 of the 16-lane map,
// same rule as x8's low half — probe_bnt96 stage C verified the bijection on
// cols 64..95, 0 mismatches / 0 dups / 4096 cells, dv_6d76a3c8c9de).
DEVI void tc_ld16_x4(unsigned* dst, uint32_t addr) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},"
                 "[%16];"
                 : "=r"(dst[0]),"=r"(dst[1]),"=r"(dst[2]),"=r"(dst[3]),
                   "=r"(dst[4]),"=r"(dst[5]),"=r"(dst[6]),"=r"(dst[7]),
                   "=r"(dst[8]),"=r"(dst[9]),"=r"(dst[10]),"=r"(dst[11]),
                   "=r"(dst[12]),"=r"(dst[13]),"=r"(dst[14]),"=r"(dst[15])
                 : "r"(addr));
}
DEVI void tc_st16_x4(uint32_t addr, const unsigned* w) {
    asm volatile("tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], "
                 "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                 :: "r"(addr),
                    "r"(w[0]),"r"(w[1]),"r"(w[2]),"r"(w[3]),"r"(w[4]),"r"(w[5]),
                    "r"(w[6]),"r"(w[7]),"r"(w[8]),"r"(w[9]),"r"(w[10]),"r"(w[11]),
                    "r"(w[12]),"r"(w[13]),"r"(w[14]),"r"(w[15]) : "memory");
}
DEVI void tc_st16_128b_x2(uint32_t addr, const unsigned* w) {
    asm volatile("tcgen05.st.sync.aligned.16x128b.x2.b32 [%0], "
                 "{%1,%2,%3,%4};"
                 :: "r"(addr),
                    "r"(w[0]),"r"(w[1]),"r"(w[2]),"r"(w[3]) : "memory");
}
DEVI void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
DEVI void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;"); }

// smem matrix descriptor: swizzle<<61 | sbo>>4<<32 | lbo>>4<<16 | addr>>4.
// Probe-verified (tc_probe3/4): version bits unnecessary; K-major interleave
// uses sw=0,lbo=128,sbo=(K/8)*128; MN-major swizzle-128B uses sw=2,lbo=0 or
// n-atom stride, sbo=1024.
DEVI uint64_t tc_desc(const void* smem, int sw, int lbo, int sbo) {
    return ((uint64_t)sw << 61) | ((uint64_t)(sbo >> 4) << 32)
         | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16)
         | (uint64_t)(smem_desc_u32(smem) >> 4);
}
DEVI uint32_t tc_idesc(int M, int N, int b_major, int atype, int btype) {
    return (uint32_t)(((uint32_t)(M >> 4) << 24) | ((uint32_t)(N >> 3) << 17)
                      | ((uint32_t)b_major << 16) | ((uint32_t)btype << 10)
                      | ((uint32_t)atype << 7) | (1u << 4));
}

// tc-prefill smem layout (canonical UMMA images, exact sizes).
// Q/K images: interleave K-major: off(m,k) = (m/8)*(K/8*128) + (k/8)*128
//   + (m%8)*16 + (k%8)*2  [sw=0, lbo=128, sbo=(K/8)*128]
// V image: swizzle-128B MN-major per 64-dim atom:
//   off(k,n) = (n/64)*(BNT*128) + (k/8)*1024 + (k%8)*128
//              + (((n%64)/8) ^ (k%8))*16 + (n%8)*2   [sw=2, lbo=BNT*128, sbo=1024]
template <int D, int BNT>
struct TcSmem {
    uint16_t* q; uint16_t* kc; uint16_t* vc; float* ksvs; float* alpha;
    uint64_t* bars; uint32_t* tmem_base;
#if !defined(FMHA_PRE_NO_NEEDCTA) || defined(FMHA_PRE_NEEDCTA_W8)
    // [2][NEEDF_WARPS] per-warp "some row's running max moved on this tile"
    // flags, parity-indexed by the KV tile so a skipped step-7 barrier cannot
    // let the next tile's write race this tile's read. NEEDF_WARPS is the
    // 8-warp kernel's count; the 4-warp one uses the first four. 64 B of the
    // 128 B tail slack bytes() already reserves, so bytes() does not move.
    int (*needf)[NEEDF_WARPS];
#endif
    DEVI TcSmem(char* base) {
        q = reinterpret_cast<uint16_t*>(base);
        kc = q + TCBM * D;
        vc = kc + 2 * BNT * D;
        ksvs = reinterpret_cast<float*>(vc + 2 * BNT * D);
        alpha = ksvs + 4 * BNT;
        bars = reinterpret_cast<uint64_t*>(alpha + TCBM);
        tmem_base = reinterpret_cast<uint32_t*>(bars + 4);
#if !defined(FMHA_PRE_NO_NEEDCTA) || defined(FMHA_PRE_NEEDCTA_W8)
        needf = reinterpret_cast<int (*)[NEEDF_WARPS]>(tmem_base + 2);
#endif
    }
    DEVI uint16_t* k(int s) const { return kc + s * BNT * D; }
    DEVI uint16_t* v(int s) const { return vc + s * BNT * D; }
    DEVI float* ks(int s) const { return ksvs + s * 2 * BNT; }
    DEVI float* vs(int s) const { return ksvs + s * 2 * BNT + BNT; }
    static constexpr int bytes() {
        return (TCBM * D + 4 * BNT * D) * 2 + 4 * BNT * 4 + TCBM * 4 + 4 * 8 + 8 + 128;
    }
    // TMEM column layout: S double buffer [0, 2*BNT), O accumulator at SOFF.
    // ACOLS = allocation granularity (power of two, >= 32).
    static constexpr int SOFF = 2 * BNT;
    static constexpr int NEED = 2 * BNT + D;
    static constexpr int ACOLS = NEED <= 128 ? 128 : (NEED <= 256 ? 256 : 512);
    static_assert(NEED <= ACOLS, "TMEM S+O exceeds allocation");
    // interleave K-major element offset (m row index, k dim index)
    //
    // THIS LAYOUT HAS NO XOR SWIZZLE, AND THAT COSTS A 2-WAY BANK CONFLICT ON EVERY K STORE. v_off below
    // XORs the 16-byte chunk index with (krow & 7); km_off does not, because the interleaved 8x8 K-major
    // atom (sw=0 in tc_desc) has no XOR in it. The consequence is measurable: dv_0922bec15bf2's two-arm ncu
    // census of case 0 puts shared store wavefronts at 67.76M against a conflict-free floor of ~41.5M for
    // ~161k CTA-tiles, and bank conflicts at 26.55M - the two agree to 1.3%, so essentially all of the
    // store-side excess is replays. They are on the K image, not V: the loads fell 44.96% in that job while
    // conflicts fell 2.35%, and the loads that left were broadcasts, which cannot conflict by construction.
    //
    // The mechanism, for TPR = NT/BN = 2 (so RPW = 64 and lane l owns row m = p*64 + w*16 + (l>>1) with
    // part = l&1): kv_chunk puts the two lanes of a row 8 k-elements apart, km_off maps (k>>3) to a 128-byte
    // stride, and (m&7)*16 is the only term that varies within those 128 bytes. A 16-byte store is serviced
    // 8 lanes (128 B) per phase, and in lanes 0-7 the pairs (0,1) (2,3) (4,5) (6,7) share m&7 - so they hit
    // the same four banks at addresses 128 B apart. Two phases of work become four.
    //
    // MYPRE_KSW2 removes the conflict at the ACCESS-PATTERN level - storing 8 bytes at 4-element
    // granularity makes the offset 4*(part + TPR*j), so (k&7)*2 = 8*part and the two lanes of a row land in
    // DIFFERENT banks of the same 128-byte block, with lanes 0-15 of a phase covering all 32 banks exactly
    // once. dv_e30b786c945a measured that it works: bank conflicts 26.55M -> 7.53M, store wavefronts
    // 67.76M -> 47.50M. It is DEFAULT OFF AND NOT SHIPPABLE, because achieving that pattern by re-tiling
    // the store index moves the data off its own addresses and scrambles the K image; a correct version
    // needs 4-byte global loads so the two lanes of a row hold interleaved 4-element chunks. It is also
    // not worth building: the same job measured the whole conflict removal at -0.35%/-0.41% on cases 0/9,
    // so store-side shared traffic is not this kernel's binding resource. TPRW cannot do the job either -
    // at TPRW=1 the phase is conflict-free but RPW = 128 > BN = 64 masks off half the threads, and at
    // TPRW=4 lanes 0-7 share (l>>2)&7 in fours, which is a 4-way conflict.
    static DEVI long long km_off(int m, int k) {
        return (m >> 3) * ((D / 8) * 128) + (k >> 3) * 128 + (m & 7) * 16 + (k & 7) * 2;
    }
    static DEVI long long v_off(int krow, int n) {
        return (n >> 6) * (BNT * 128) + (krow >> 3) * 1024 + (krow & 7) * 128
             + ((((n & 63) >> 3) ^ (krow & 7)) * 16) + (n & 7) * 2;
    }
#ifdef FMHA_PRE_F8QK
    // ---- fp8-direct QK reinterpretation of the SAME layout (bytes() and every
    // other offset are untouched; D=128 only). The q region (TCBM*D u16) holds
    // qhi + qlo byte images (TCBM*D u8 each). The f8 K images (2 stages x
    // BNT*D bytes) ride the FIRST HALF of kc's 4*BNT*D-byte region; qinv and
    // qscale (TCBM floats each) live in the 16KB slack behind them.
    DEVI uint8_t* qhi() const { return reinterpret_cast<uint8_t*>(q); }
    DEVI uint8_t* qlo() const { return reinterpret_cast<uint8_t*>(q) + TCBM * D; }
    DEVI uint8_t* k8(int s) const {
        return reinterpret_cast<uint8_t*>(kc) + s * BNT * D;
    }
    DEVI float* qinv() const {
        return reinterpret_cast<float*>(reinterpret_cast<char*>(kc) + 2 * BNT * D);
    }
    DEVI float* qscale() const { return qinv() + TCBM; }
    // SW128 K-major byte offset for a 1-byte element (row m, dim d) with
    // KD=128: atom = 8 rows x 128B; the 16B chunk index XORs the row-in-atom.
    // Probe-verified descriptor pair: tc_desc(ptr, 2, 0, 1024), addr +2 per
    // 32-byte K step.
    static DEVI long long k8_off(int m, int d) {
        return (long long)((m >> 3) * 1024 + (m & 7) * 128
                           + ((((d >> 4) ^ (m & 7)) & 7) << 4) + (d & 15));
    }
#endif
};


// f16x2_to_bf16x2_exact: defined in fmha.cuh (v3.2 moved; was duplicated here)

// tc-convert: staged registers -> canonical bf16 UMMA images (exact).
// K: interleave K-major (km_off); V: swizzle-128B MN-major (v_off).
// The (row, chunk) each register holds is exactly what kv_prefetch_m staged, so
// a change of TPRW permutes which thread writes which smem word but never the
// image contents: multi-pass staging is bit-identical to single-pass.
template <int D, int BNT, int NT = TC_THREADS, int TPRW = 0, bool F8K = false,
          typename SM = TcSmem<D, BNT>>
DEVI void tc_convert(const ConvRegs<D, BNT, NT, TPRW>& R, const SM& S,
                     int stage, int nvalid, const Params& P) {
    using CR = ConvRegs<D, BNT, NT, TPRW>;
    constexpr int TPR = CR::TPR, RPW = CR::RPW, PASSES = CR::PASSES;
    const int tid = threadIdx.x;
    const int rbase = tid / TPR;
    const int part = tid % TPR;
    char* dk = reinterpret_cast<char*>(S.k(stage));
    char* dv = reinterpret_cast<char*>(S.v(stage));
    static_assert(!F8K || D == 128, "fp8-direct K image is probe-verified for KD=128 only");
#ifdef FMHA_PRE_F8QK
    // if constexpr, not a ternary: the discarded branch must not instantiate
    // S.k8 for SM types that do not have it (TcSmem2 in fa_prefill_rt2p).
    char* dk8 = nullptr;
    if constexpr (F8K) dk8 = reinterpret_cast<char*>(S.k8(stage));
#endif
    #pragma unroll
    for (int p = 0; p < PASSES; ++p) {
    const int row = p * RPW + rbase;
#ifdef MYPRE_KSFOLD
    // Only the part==0 lane of a row loaded R.ks (kv_prefetch guards it), but every lane of the row converts a
    // different chunk of it, so the scale has to reach them all. A warp shuffle is the cheap way: rows never
    // span a warp because TPR divides 32. This MUST sit outside the `row < nvalid` branch - __shfl_sync with a
    // full mask is undefined if some lanes skip it - and the invalid-row lanes read a value they never use,
    // because that branch zero-fills the image instead.
    static_assert(TPR == 1 || TPR == 2 || TPR == 4 || TPR == 8 || TPR == 16 || TPR == 32,
                  "MYPRE_KSFOLD broadcasts R.ks within a warp, so a row must not span warps");
    const unsigned ks2 = f16x2_splat(
        __shfl_sync(0xffffffffu, R.ks[p], (int)(((unsigned)(tid & 31) / TPR) * TPR)) * P.sm_scale_log2e);
#endif
#ifdef MYPRE_VSFOLD
    // Only the part==0 lane of a row loaded R.vs (kv_prefetch guards it) but every lane of the row converts a
    // different chunk of it, so the scale reaches them by one warp shuffle. Rows never span a warp because TPR
    // divides 32. This MUST sit outside the `row < nvalid` branch: __shfl_sync with a full mask is undefined if
    // some lanes skip it. Invalid-row lanes read a value they never use, since that branch zero-fills instead.
    static_assert(TPR == 1 || TPR == 2 || TPR == 4 || TPR == 8 || TPR == 16 || TPR == 32,
                  "MYPRE_VSFOLD broadcasts R.vs within a warp, so a row must not span warps");
    const float vsb =
        __shfl_sync(0xffffffffu, R.vs[p], (int)(((unsigned)(tid & 31) / TPR) * TPR));
#ifdef MYPRE_VBF16P
    const unsigned vs2b = bf16x2_splat(vsb);   // one splat per pass, reused by every pair this thread owns
#endif
#endif
    if (row < nvalid) {
        #pragma unroll
        for (int i = 0; i < CR::KPP; ++i) {
            uint2 raw = R.k[p * CR::KPP + i];
#ifdef FMHA_PRE_F8QK
            if constexpr (F8K) {
                // fp8-direct: the raw e4m3 bytes ARE the operand image - one
                // STS.64 per 8 keys at the SW128 offset, zero convert
                // instructions, half the image bytes of the bf16 form.
                *reinterpret_cast<uint2*>(dk8 + SM::k8_off(
                    row, kv_chunk(part, TPR, i, CR::KPP) * 8)) = raw;
                continue;
            }
#endif
#ifdef MYPRE_KSFOLD
            // K' = round_f16(K_fp8 * ks): 2 instructions per pair against 1, and it deletes 16 broadcast
            // float4 loads plus 64 FMULs per thread per tile from the softmax, where they sit on the chain
            // TMEM-load -> *ks -> row max -> exp2 that gates everything downstream. dv_35d1fe1a170c priced
            // that deletion alone at -6.1% on cases 0 and 9.
            uint4 o = make_uint4(f16x2_mul_rn(e4m3x2_to_f16x2((unsigned short)(raw.x & 0xffffu)), ks2),
                                 f16x2_mul_rn(e4m3x2_to_f16x2((unsigned short)(raw.x >> 16)), ks2),
                                 f16x2_mul_rn(e4m3x2_to_f16x2((unsigned short)(raw.y & 0xffffu)), ks2),
                                 f16x2_mul_rn(e4m3x2_to_f16x2((unsigned short)(raw.y >> 16)), ks2));
#elif defined(MYPRE_T_KF16) || defined(MYPRE_KF16)
            uint4 o = make_uint4(e4m3x2_to_f16x2((unsigned short)(raw.x & 0xffffu)),
                                 e4m3x2_to_f16x2((unsigned short)(raw.x >> 16)),
                                 e4m3x2_to_f16x2((unsigned short)(raw.y & 0xffffu)),
                                 e4m3x2_to_f16x2((unsigned short)(raw.y >> 16)));
#else
            uint4 o = make_uint4(e4m3x2_to_bf16x2((unsigned short)(raw.x & 0xffffu)),
                                 e4m3x2_to_bf16x2((unsigned short)(raw.x >> 16)),
                                 e4m3x2_to_bf16x2((unsigned short)(raw.y & 0xffffu)),
                                 e4m3x2_to_bf16x2((unsigned short)(raw.y >> 16)));
#endif
#ifdef MYPRE_KSW2
            // ---- MYPRE_KSW2: split the 16-byte K image store into two 8-byte stores at 4-element
            // granularity. KEPT AS A MEASUREMENT PROBE ONLY, DEFAULT OFF, AND **NOT** BIT-IDENTICAL -
            // the commit that added it claimed otherwise and dv_e30b786c945a's stage-1 gate proved the
            // claim wrong on five of nine shapes (rel_l2 ~1.0, i.e. a scrambled K image).
            //
            // WHY THE CLAIM WAS WRONG. Permuting which thread writes which word is only safe if each
            // word still lands at the address its DATA belongs to. This does not: thread (part, i) holds
            // the 8 fp8 elements [8*(part + 2i), +8) and it must write them there, but 4*(part + TPR*j)
            // for j = 2i, 2i+1 puts them at [4*part + 16i, +4) and [4*part + 16i + 8, +4). Those two
            // tilings of [0, D) are both complete and both disjoint - which is what made the coverage
            // argument look sound - but they are NOT the same tiling, so element e stops landing at e.
            // The gate caught it precisely where the mechanism predicts: t9/t0/t1/t7/tn (which route here
            // and whose output depends on K) all broke, while tz (v = 0, so out = 0 whatever K holds),
            // tb (q = 0, so every score is 0 whatever K holds), t256 and d2 (which never reach
            // fmha_pre.cu at all) stayed BIT-IDENTICAL.
            //
            // WHY IT IS STILL IN THE TREE. Its stage-3 profile is the measurement that closed the
            // bank-conflict direction, and it is worth more than the optimization would have been: the
            // access pattern - two 8-byte stores per 16-byte chunk, (k&7)*2 = 8*part - IS conflict-free,
            // and it took shared bank conflicts from 26.55M to 7.53M (-72%) and store wavefronts from
            // 67.76M to 47.50M (-30%), for -0.35%/-0.41% on cases 0/9. Store-side shared traffic is
            // therefore NOT the prefill's binding resource despite L1TEX reading 72.51%, which is the
            // prefill's own version of the decode's "l1tex__throughput at 89% is not 89% of headroom".
            // See results/prefill-ksw2-20261026.md. A CORRECT conflict-free variant would need the two
            // lanes of a row to hold interleaved 4-element chunks, i.e. 4-byte global loads in
            // kv_prefetch_m and a ConvRegs::k of twice the entries - and that is not worth building,
            // because this job already measured what removing the conflicts is worth.
            *reinterpret_cast<uint2*>(dk + SM::km_off(
                row, kv_chunk(part, TPR, 2 * i, 2 * CR::KPP) * 4)) = make_uint2(o.x, o.y);
            *reinterpret_cast<uint2*>(dk + SM::km_off(
                row, kv_chunk(part, TPR, 2 * i + 1, 2 * CR::KPP) * 4)) = make_uint2(o.z, o.w);
#else
            *reinterpret_cast<uint4*>(dk + SM::km_off(
                row, kv_chunk(part, TPR, i, CR::KPP) * 8)) = o;
#endif
        }
        #pragma unroll
        for (int i = 0; i < CR::VPP; ++i) {
            unsigned raw = R.v[p * CR::VPP + i];
#ifdef MYPRE_VSFOLD
#ifdef MYPRE_VBF16P
            uint2 o2 = make_uint2(
                bf16x2_mul_rn(e4m3x2_to_bf16x2_sat((unsigned short)(raw & 0xffffu)), vs2b),
                bf16x2_mul_rn(e4m3x2_to_bf16x2_sat((unsigned short)(raw >> 16)), vs2b));
#else
            uint2 o2 = make_uint2(
                e4m3x2_to_bf16x2_scaled((unsigned short)(raw & 0xffffu), vsb),
                e4m3x2_to_bf16x2_scaled((unsigned short)(raw >> 16), vsb));
#endif
#elif defined(MYPRE_T_VF16) || defined(MYPRE_VF16)
            uint2 o2 = make_uint2(
                e4m3x2_to_f16x2((unsigned short)(raw & 0xffffu)),
                e4m3x2_to_f16x2((unsigned short)(raw >> 16)));
#else
            uint2 o2 = make_uint2(
                e4m3x2_to_bf16x2((unsigned short)(raw & 0xffffu)),
                e4m3x2_to_bf16x2((unsigned short)(raw >> 16)));
#endif
            *reinterpret_cast<uint2*>(dv + SM::v_off(
                row, kv_chunk(part, TPR, i, CR::VPP) * 4)) = o2;
        }
        if (part == 0) {
            // v3.5: pre-scale ks by sm_scale*log2e here (same thread that
            // stores it; identical f32 rounding) — removes 64 FMULs/lane/tile
            // from the softmax hot loop.
            S.ks(stage)[row] = R.ks[p] * P.sm_scale_log2e;
            S.vs(stage)[row] = R.vs[p];
        }
    } else if (row < BNT) {
        // j23-bnt96 fix: rows are tid/TPR with TPR = NT/BN TRUNCATED. When BN
        // does not divide NT (96 with NT=256 -> TPR=2 -> rows 0..127), rows
        // >= BN have NO smem region — the unguarded zero-fill wrote up to
        // v_off(127) ~ +27KB past the stage => illegal address (the ab_bnt96
        // arm crashes, dv_f42573cc1ca7). For every exact divisor BN this
        // predicate is provably true (rows == NT/TPR == BN) and folds away.
        // With TPRW>1 the same guard covers the passes whose row band runs past
        // BN (e.g. BN=96, NT=256, TPRW=4 -> PASSES=2 -> rows 0..127).
        #pragma unroll
        for (int i = 0; i < CR::KPP; ++i) {
#ifdef FMHA_PRE_F8QK
            if constexpr (F8K) {
                *reinterpret_cast<uint2*>(dk8 + SM::k8_off(
                    row, kv_chunk(part, TPR, i, CR::KPP) * 8)) = make_uint2(0u, 0u);
                continue;
            }
#endif
            *reinterpret_cast<uint4*>(dk + SM::km_off(
                row, kv_chunk(part, TPR, i, CR::KPP) * 8)) = make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int i = 0; i < CR::VPP; ++i)
            *reinterpret_cast<uint2*>(dv + SM::v_off(
                row, kv_chunk(part, TPR, i, CR::VPP) * 4)) = make_uint2(0, 0);
        if (part == 0) {
            S.ks(stage)[row] = 0.f;
            S.vs(stage)[row] = 0.f;
        }
    }
    }
}


// Single line so experiments/ppcheck.py can evaluate it: its lexer has no line-continuation rule.
// FMHA_PRE_KFIRST reorders the convert and the QK issue, so it composes with none of the ablation
// probes that delete or move them - NOCONV/NOMMA/NOSOFT would price a loop that is not this one, and
// NBSYNC/PFLATE/PF2 each relocate the same step 5 that this flag splits in two.
#if defined(FMHA_PRE_F8QK) && (defined(FMHA_PRE_KFIRST) || defined(MYPRE_KSFOLD) \
    || defined(MYPRE_T_NOKSMUL) || defined(MYPRE_T_NOCONV) || defined(MYPRE_T_NOMMA) \
    || defined(MYPRE_T_NOSOFT) || defined(MYPRE_T_NOPREF) || defined(MYPRE_RT2P) \
    || defined(FMHA_PRE_PF2) || defined(FMHA_PRE_PFLATE) || defined(FMHA_PRE_WAITELECT) \
    || defined(MYPRE_CLUSTER) || defined(MYPRE_CL_NOSLICE) || defined(MYPRE_NO_CLSPLIT) \
    || defined(MYPRE_KSW2) || defined(MYPRE_T_KF16) || defined(FMHA_PRE_NO_KF16) \
    || defined(MYPRE_VF16) || defined(MYPRE_T_VF16) || defined(FMHA_PRE_BLKCHUNK))
#error "FMHA_PRE_F8QK replaces the K convert, the f16 QK issue and the softmax score domain. The convert/issue reorderings and ablations price a loop that no longer exists under it, KFIRST's split convert and KSFOLD's scaled K image have no f8 branch (silently wrong numbers), and the f16-image type flags no longer describe the operands. Measure F8QK against the shipped default only."
#endif

#if defined(MYPRE_KSFOLD) && defined(MYPRE_RT2P)
#error "MYPRE_KSFOLD pre-scales the K image inside tc_convert, which fa_prefill_rt2p also calls, but rt2p's softmax still multiplies by S.ks - a silent double scale. Convert rt2p's float4 ks site before combining them. rt2p is opt-in and off by default, so nothing shipped is affected."
#endif
#if defined(MYPRE_KSFOLD) && (defined(FMHA_PRE_KFIRST) || defined(FMHA_PRE_NO_KF16))
#error "MYPRE_KSFOLD multiplies the K image with mul.rn.f16x2 inside tc_convert, so it needs the whole of tc_convert (KFIRST splits it) and an f16 K image (FMHA_PRE_NO_KF16 reverts it to bf16, where a packed f16 multiply would be wrong numbers rather than a slow path)"
#endif
#if defined(MYPRE_VBF16P) && !defined(MYPRE_VSFOLD)
#error "MYPRE_VBF16P is a packed spelling of MYPRE_VSFOLD's fold, not a change on its own: it multiplies V's image by value_scale inside tc_convert, so without MYPRE_VSFOLD the pack loops would still multiply by S.vs and every output would be scaled twice."
#endif
#if defined(MYPRE_VSFOLD) && (defined(MYPRE_VF16) || defined(MYPRE_T_VF16))
#error "MYPRE_VSFOLD scales V through e4m3x2_to_bf16x2_scaled, which ends in cvt.rn.bf16x2.f32. With an f16 V image that store is the wrong dtype for idesc_pv, and the f16 product would underflow on 'tiny' - the exact defect that made MYPRE_VF16 unshippable. The point of folding in f32 is that bf16 shares f32's exponent width."
#endif
// KFIRST x VSFOLD rework (2026-10-28): the #error that used to sit here fired because the split left the
// vs broadcast in tc_convert_k while the V scaling lives in tc_convert_v. tc_convert_v now carries its own
// broadcast (identical text, identical per-element arithmetic), so the combination builds and stays
// bit-identical: same shuffle source lane, same e4m3x2_to_bf16x2_scaled, only the ISSUE TIME of the V half
// moves. The ks/vs metadata publication stays with the K half; under VSFOLD nothing reads S.vs any more.
#if defined(MYPRE_T_MX3) && (defined(MYPRE_T_NOKSMUL) || defined(MYPRE_KSFOLD))
#error "MYPRE_T_MX3 replaces the tc softmax's masked max loop WHOLESALE, including the #if that MYPRE_T_NOKSMUL and MYPRE_KSFOLD use to drop the per-key scale. Combining them would leave the ks multiply in the softmax while MYPRE_KSFOLD has already folded it into the K image - a silent double scale, the same class of bug the W8 site had. In the nomask branch MX3 is an inert #elif behind the probe. Pick one."
#endif
#if defined(FMHA_PRE_KFIRST) && (defined(MYPRE_T_NOCONV) || defined(MYPRE_T_NOMMA) || defined(MYPRE_T_NOSOFT) || defined(FMHA_PRE_NBSYNC) || defined(FMHA_PRE_PFLATE) || defined(FMHA_PRE_PF2))
#error "FMHA_PRE_KFIRST is incompatible with the convert/MMA/softmax ablation probes and with NBSYNC, PFLATE and PF2"
#endif
#ifdef FMHA_PRE_KFIRST
// Split halves of tc_convert, derived from it by deleting the other half's stores, so the K pass and
// the V pass can sit on opposite sides of the softmax. Nothing else changes: same registers, same
// swizzle offsets, same zero-fill predicate, and the ks/vs metadata stays with the K pass because the
// softmax that reads it now follows that pass. The default build compiles neither function.
//
// WHY THE SPLIT IS FREE. Step 5's own comment says it: "convert tile t+1 -> smem[s^1] (QK(t-1) long
// done; PV(t-1) waited)". Only the V half needs that wait. K[s^1] holds K(t-1), which QK(t-1) consumed
// and step 2 of the PREVIOUS iteration already waited on, so it is free a full iteration earlier than
// V[s^1], which PV(t-1) is still reading. Converting K(t+1) and issuing QK(t+1) before the softmax
// therefore needs no third shared stage - the 168 B that results/prefill-next-levers-20261025.md §4
// treated as the blocker was priced against moving the WHOLE convert, and the premise that the convert
// waits on PV(t-1) is only true of its V half. Both MMAs gain slack, not one at the other's expense:
// QK(t+1) goes from steps 7,8,1 to the softmax plus steps 4,5b,7,8,1, and PV(t-1) goes from steps 1,2,3
// to steps 1,2,3a,6,3b. That is the correction to the analysis in results/official-ladder-20261025.md,
// which retired this reorder on the grounds that it would leave PV(t-1) with only the prefetch issue -
// true when the whole convert moves, false when only its K half does.
template <int D, int BNT, int NT = TC_THREADS, int TPRW = 0, typename SM = TcSmem<D, BNT>>
DEVI void tc_convert_k(const ConvRegs<D, BNT, NT, TPRW>& R, const SM& S,
                     int stage, int nvalid, const Params& P) {
    using CR = ConvRegs<D, BNT, NT, TPRW>;
    constexpr int TPR = CR::TPR, RPW = CR::RPW, PASSES = CR::PASSES;
    const int tid = threadIdx.x;
    const int rbase = tid / TPR;
    const int part = tid % TPR;
    char* dk = reinterpret_cast<char*>(S.k(stage));
    #pragma unroll
    for (int p = 0; p < PASSES; ++p) {
    const int row = p * RPW + rbase;
#ifdef MYPRE_VSFOLD
    // Only the part==0 lane of a row loaded R.vs (kv_prefetch guards it) but every lane of the row converts a
    // different chunk of it, so the scale reaches them by one warp shuffle. Rows never span a warp because TPR
    // divides 32. This MUST sit outside the `row < nvalid` branch: __shfl_sync with a full mask is undefined if
    // some lanes skip it. Invalid-row lanes read a value they never use, since that branch zero-fills instead.
    static_assert(TPR == 1 || TPR == 2 || TPR == 4 || TPR == 8 || TPR == 16 || TPR == 32,
                  "MYPRE_VSFOLD broadcasts R.vs within a warp, so a row must not span warps");
    const float vsb =
        __shfl_sync(0xffffffffu, R.vs[p], (int)(((unsigned)(tid & 31) / TPR) * TPR));
#ifdef MYPRE_VBF16P
    const unsigned vs2b = bf16x2_splat(vsb);   // one splat per pass, reused by every pair this thread owns
#endif
#endif
    if (row < nvalid) {
        #pragma unroll
        for (int i = 0; i < CR::KPP; ++i) {
            uint2 raw = R.k[p * CR::KPP + i];
#if defined(MYPRE_T_KF16) || defined(MYPRE_KF16)
            uint4 o = make_uint4(e4m3x2_to_f16x2((unsigned short)(raw.x & 0xffffu)),
                                 e4m3x2_to_f16x2((unsigned short)(raw.x >> 16)),
                                 e4m3x2_to_f16x2((unsigned short)(raw.y & 0xffffu)),
                                 e4m3x2_to_f16x2((unsigned short)(raw.y >> 16)));
#else
            uint4 o = make_uint4(e4m3x2_to_bf16x2((unsigned short)(raw.x & 0xffffu)),
                                 e4m3x2_to_bf16x2((unsigned short)(raw.x >> 16)),
                                 e4m3x2_to_bf16x2((unsigned short)(raw.y & 0xffffu)),
                                 e4m3x2_to_bf16x2((unsigned short)(raw.y >> 16)));
#endif
#ifdef MYPRE_KSW2
            // ---- MYPRE_KSW2: split the 16-byte K image store into two 8-byte stores at 4-element
            // granularity. KEPT AS A MEASUREMENT PROBE ONLY, DEFAULT OFF, AND **NOT** BIT-IDENTICAL -
            // the commit that added it claimed otherwise and dv_e30b786c945a's stage-1 gate proved the
            // claim wrong on five of nine shapes (rel_l2 ~1.0, i.e. a scrambled K image).
            //
            // WHY THE CLAIM WAS WRONG. Permuting which thread writes which word is only safe if each
            // word still lands at the address its DATA belongs to. This does not: thread (part, i) holds
            // the 8 fp8 elements [8*(part + 2i), +8) and it must write them there, but 4*(part + TPR*j)
            // for j = 2i, 2i+1 puts them at [4*part + 16i, +4) and [4*part + 16i + 8, +4). Those two
            // tilings of [0, D) are both complete and both disjoint - which is what made the coverage
            // argument look sound - but they are NOT the same tiling, so element e stops landing at e.
            // The gate caught it precisely where the mechanism predicts: t9/t0/t1/t7/tn (which route here
            // and whose output depends on K) all broke, while tz (v = 0, so out = 0 whatever K holds),
            // tb (q = 0, so every score is 0 whatever K holds), t256 and d2 (which never reach
            // fmha_pre.cu at all) stayed BIT-IDENTICAL.
            //
            // WHY IT IS STILL IN THE TREE. Its stage-3 profile is the measurement that closed the
            // bank-conflict direction, and it is worth more than the optimization would have been: the
            // access pattern - two 8-byte stores per 16-byte chunk, (k&7)*2 = 8*part - IS conflict-free,
            // and it took shared bank conflicts from 26.55M to 7.53M (-72%) and store wavefronts from
            // 67.76M to 47.50M (-30%), for -0.35%/-0.41% on cases 0/9. Store-side shared traffic is
            // therefore NOT the prefill's binding resource despite L1TEX reading 72.51%, which is the
            // prefill's own version of the decode's "l1tex__throughput at 89% is not 89% of headroom".
            // See results/prefill-ksw2-20261026.md. A CORRECT conflict-free variant would need the two
            // lanes of a row to hold interleaved 4-element chunks, i.e. 4-byte global loads in
            // kv_prefetch_m and a ConvRegs::k of twice the entries - and that is not worth building,
            // because this job already measured what removing the conflicts is worth.
            *reinterpret_cast<uint2*>(dk + SM::km_off(
                row, kv_chunk(part, TPR, 2 * i, 2 * CR::KPP) * 4)) = make_uint2(o.x, o.y);
            *reinterpret_cast<uint2*>(dk + SM::km_off(
                row, kv_chunk(part, TPR, 2 * i + 1, 2 * CR::KPP) * 4)) = make_uint2(o.z, o.w);
#else
            *reinterpret_cast<uint4*>(dk + SM::km_off(
                row, kv_chunk(part, TPR, i, CR::KPP) * 8)) = o;
#endif
        }
        if (part == 0) {
            // v3.5: pre-scale ks by sm_scale*log2e here (same thread that
            // stores it; identical f32 rounding) — removes 64 FMULs/lane/tile
            // from the softmax hot loop.
            S.ks(stage)[row] = R.ks[p] * P.sm_scale_log2e;
            S.vs(stage)[row] = R.vs[p];
        }
    } else if (row < BNT) {
        // j23-bnt96 fix: rows are tid/TPR with TPR = NT/BN TRUNCATED. When BN
        // does not divide NT (96 with NT=256 -> TPR=2 -> rows 0..127), rows
        // >= BN have NO smem region — the unguarded zero-fill wrote up to
        // v_off(127) ~ +27KB past the stage => illegal address (the ab_bnt96
        // arm crashes, dv_f42573cc1ca7). For every exact divisor BN this
        // predicate is provably true (rows == NT/TPR == BN) and folds away.
        // With TPRW>1 the same guard covers the passes whose row band runs past
        // BN (e.g. BN=96, NT=256, TPRW=4 -> PASSES=2 -> rows 0..127).
        #pragma unroll
        for (int i = 0; i < CR::KPP; ++i)
            *reinterpret_cast<uint4*>(dk + SM::km_off(
                row, kv_chunk(part, TPR, i, CR::KPP) * 8)) = make_uint4(0, 0, 0, 0);
        if (part == 0) {
            S.ks(stage)[row] = 0.f;
            S.vs(stage)[row] = 0.f;
        }
    }
    }
}


template <int D, int BNT, int NT = TC_THREADS, int TPRW = 0, typename SM = TcSmem<D, BNT>>
DEVI void tc_convert_v(const ConvRegs<D, BNT, NT, TPRW>& R, const SM& S,
                     int stage, int nvalid, const Params& P) {
    using CR = ConvRegs<D, BNT, NT, TPRW>;
    constexpr int TPR = CR::TPR, RPW = CR::RPW, PASSES = CR::PASSES;
    const int tid = threadIdx.x;
    const int rbase = tid / TPR;
    const int part = tid % TPR;
    char* dv = reinterpret_cast<char*>(S.v(stage));
    #pragma unroll
    for (int p = 0; p < PASSES; ++p) {
    const int row = p * RPW + rbase;
#ifdef MYPRE_VSFOLD
    // KFIRST x VSFOLD rework: the V half owns the value-scale broadcast itself - the K half's local vsb is
    // not visible here and runs a full iteration ahead. Identical text to tc_convert's: only the part==0 lane
    // of a row loaded R.vs (kv_prefetch guards it), so the scale reaches the row's other lanes by one warp
    // shuffle. Outside the `row < nvalid` branch because __shfl_sync with a full mask is undefined if some
    // lanes skip it; invalid-row lanes read a value they never use (that branch zero-fills instead).
    static_assert(TPR == 1 || TPR == 2 || TPR == 4 || TPR == 8 || TPR == 16 || TPR == 32,
                  "MYPRE_VSFOLD broadcasts R.vs within a warp, so a row must not span warps");
    const float vsb =
        __shfl_sync(0xffffffffu, R.vs[p], (int)(((unsigned)(tid & 31) / TPR) * TPR));
#ifdef MYPRE_VBF16P
    const unsigned vs2b = bf16x2_splat(vsb);   // one splat per pass, reused by every pair this thread owns
#endif
#endif
    if (row < nvalid) {
        #pragma unroll
        for (int i = 0; i < CR::VPP; ++i) {
            unsigned raw = R.v[p * CR::VPP + i];
#ifdef MYPRE_VSFOLD
#ifdef MYPRE_VBF16P
            uint2 o2 = make_uint2(
                bf16x2_mul_rn(e4m3x2_to_bf16x2_sat((unsigned short)(raw & 0xffffu)), vs2b),
                bf16x2_mul_rn(e4m3x2_to_bf16x2_sat((unsigned short)(raw >> 16)), vs2b));
#else
            uint2 o2 = make_uint2(
                e4m3x2_to_bf16x2_scaled((unsigned short)(raw & 0xffffu), vsb),
                e4m3x2_to_bf16x2_scaled((unsigned short)(raw >> 16), vsb));
#endif
#elif defined(MYPRE_T_VF16) || defined(MYPRE_VF16)
            uint2 o2 = make_uint2(
                e4m3x2_to_f16x2((unsigned short)(raw & 0xffffu)),
                e4m3x2_to_f16x2((unsigned short)(raw >> 16)));
#else
            uint2 o2 = make_uint2(
                e4m3x2_to_bf16x2((unsigned short)(raw & 0xffffu)),
                e4m3x2_to_bf16x2((unsigned short)(raw >> 16)));
#endif
            *reinterpret_cast<uint2*>(dv + SM::v_off(
                row, kv_chunk(part, TPR, i, CR::VPP) * 4)) = o2;
        }
    } else if (row < BNT) {
        // j23-bnt96 fix: rows are tid/TPR with TPR = NT/BN TRUNCATED. When BN
        // does not divide NT (96 with NT=256 -> TPR=2 -> rows 0..127), rows
        // >= BN have NO smem region — the unguarded zero-fill wrote up to
        // v_off(127) ~ +27KB past the stage => illegal address (the ab_bnt96
        // arm crashes, dv_f42573cc1ca7). For every exact divisor BN this
        // predicate is provably true (rows == NT/TPR == BN) and folds away.
        // With TPRW>1 the same guard covers the passes whose row band runs past
        // BN (e.g. BN=96, NT=256, TPRW=4 -> PASSES=2 -> rows 0..127).
        #pragma unroll
        for (int i = 0; i < CR::VPP; ++i)
            *reinterpret_cast<uint2*>(dv + SM::v_off(
                row, kv_chunk(part, TPR, i, CR::VPP) * 4)) = make_uint2(0, 0);
    }
    }
}

#endif  // FMHA_PRE_KFIRST

// ---- tcgen05 prefill kernel (D=128 first): M=128 q rows per CTA -----------
// PIPE=true (D=64): v2.3b deep pipeline — prologue convert+QK(0), then per
// tile: softmax(t) overlaps PV(t-1); QK(t+1) issues before correction.
// Measured: D64 1.14x vs v2, but D128 0.95x (TMEM contention / register
// pressure from batched corr) -> PIPE=false for D=128/256 keeps the exact
// engine-validated v2 sequential loop.
template <int D, int BNT, bool PIPE, bool NM, bool F8Q = false>
__global__ void __launch_bounds__(TC_THREADS) MYPRE_CLUSTER_ATTR
fa_prefill_tc(const __grid_constant__ Params P) {
    extern __shared__ char smem_raw[];
    TcSmem<D, BNT> S(smem_raw);
#ifdef MYPRE_CLUSTER_DBG
    // One print per cluster rank for the first two blocks: does the shared
    // address this kernel feeds to tc_desc/tc_alloc/mbarrier carry rank bits?
    if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x < 2) {
        const uint32_t raw = static_cast<uint32_t>(__cvta_generic_to_shared(S.k(0)));
        printf("CDBG bx=%u rank=%u raw=%08x masked=%08x\n",
               (unsigned)blockIdx.x, cluster_ctarank(), raw, raw & 0x00FFFFFFu);
    }
#endif
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    // ---- cluster split (results/prefill-work-balance-20261024.md) -------------
    // CL ranks of one cluster walk CL contiguous slices of the SAME item's KV
    // range and merge (m, l, O) over DSMEM in the epilogue, so an item stops being
    // the scheduling atom: case 0's granularity loss is 9.65% of its runtime at
    // CL=1 and 3.65% at CL=4. `split` is runtime because the host only turns it on
    // where the makespan model says it pays; SPLIT is compile-time so a default
    // build emits none of it. D=256 is excluded because its O staging (128 rows x
    // 260 floats) does not fit the freed q/kc/vc region.
    constexpr int CL = TC_CL;
    constexpr bool SPLIT = (CL > 1) && (D != 256);
    const bool split = SPLIT && P.cl_split;
    unsigned crank = 0;
    if constexpr (CL > 1) crank = cluster_ctarank();
#ifdef MYPRE_CLUSTER_DBG
    if (SPLIT && blockIdx.x == 0 && tid == 0 && crank == 0) {
        const unsigned L = atomicAdd(&dbg_launch, 1u);
        const unsigned done = dbg_done;
        printf("CLDBG launch=%u clusters_retired_before=%u gridx=%u\n",
               L, done, (unsigned)gridDim.x);
    }
#endif

    // ---- work assignment (v4.8): static | fold pairs | persistent pull ----
    // v4.8.3: NO static __shared__ here — a static var shifts the extern
    // dynamic smem base off 16B alignment and tc_desc's addr>>4 silently
    // truncates -> misaligned UMMA descriptors -> all-tc-prefill corruption
    // (12 FAILs in dv_1d721aba6424 with identical static-path code to the
    // green v4.8.1). Work ticket lives in the spare word of tmem_base[1]
    // (dynamic smem, 8B allotted, [0] holds the tcgen05.alloc result).
    int T_all = 0;
    int work0 = 0;
    int b_s = -1, q0_s = 0, qlen_s = 0;
    int tileA = 0, tileB = -1;
    if (P.persist) {
        int prev = 0;
        for (int i = 0; i < P.batch; ++i) {
            int c = __ldg(P.cu + i + 1);
            T_all += (c - prev + TCBM - 1) / TCBM;
            prev = c;
        }
        T_all *= P.heads;
        // v4.8.2: ONE atomic per CTA (all-thread atomicAdd inflated the
        // counter 128x and gave every lane a different item -> divergent
        // tcgen05/mbarrier ops -> hang; 2 CHILD_TIMEOUTs in dv_461c59456687).
        if (tid == 0 && (!split || crank == 0))
            S.tmem_base[1] = (unsigned)atomicAdd(P.counter, 1);
        if (split) cl_sync();
        if (tid == 0 && split && crank != 0)
            S.tmem_base[1] = cl_ldu32(cl_mapa(smem_u32(&S.tmem_base[1]), 0));
        // SECOND cluster barrier, and it is not redundant. barrier.cluster.wait
        // only tells a rank that rank 0 ARRIVED, not that it is still alive: on the
        // pull that exhausts the queue rank 0 reaches `work0 >= T_all`, returns, and
        // can retire while a peer is still reading its shared word - which is an
        // access to a dead CTA's smem. That race is why one launch of the split is
        // correct and a 20-iteration timing loop is not (dv_6c198f98eea9: case 7
        // PASS at rel_l2 7.007e-04, then cudaErrorLaunchFailure inside bench()), and
        // why CL=4 died sooner than CL=2 - three peers read that word instead of
        // one, and dv_92c84242add3's no-slice arm died too since it pulls tickets
        // identically. Nothing after the merge's second barrier reads a peer, so
        // this is the only place the lifetime needs closing.
        if (split) cl_sync();
        __syncthreads();
        work0 = (int)S.tmem_base[1];
        // one ticket per CLUSTER, so every rank sees the same work0 and the whole
        // cluster returns together - a rank leaving before a cl_sync would deadlock
        if (work0 >= T_all) return;
    } else {
        int gx = blockIdx.x;
        // under a split the cluster is the item owner, so the host scales grid.x by
        // CL; without one every CTA keeps its own item (the null-probe behaviour)
        if (split) gx /= (int)CL;
        int base = 0;
        {
            int prev = 0;
            for (int i = 0; i < P.batch; ++i) {
                int c = __ldg(P.cu + i + 1);
                int ntiles = P.fold ? (c - prev + 2 * TCBM - 1) / (2 * TCBM)
                                    : (c - prev + TCBM - 1) / TCBM;
                if (gx < base + ntiles) {
                    b_s = i; tileA = gx - base; q0_s = prev; qlen_s = c - prev; break;
                }
                base += ntiles;
                prev = c;
            }
        }
        if (b_s < 0) return;
        // v4.5 mirror pairing (P.fold): CTA handles tile j AND tile n-1-j —
        // constant work per CTA on diagonal-causal shapes.
        const int nseq_tiles = (qlen_s + TCBM - 1) / TCBM;
        tileB = P.fold ? (nseq_tiles - 1 - tileA) : -1;
        if (tileB == tileA) tileB = -1;
        if (tileA * TCBM >= qlen_s) return;
    }

    // ---- barriers + tmem ----
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(smem_u32(&S.bars[i])));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        tc_alloc(S.tmem_base, TcSmem<D, BNT>::ACOLS);
        tc_relinquish();
    }
    __syncthreads();
    const uint32_t tm = *S.tmem_base;
#ifdef FMHA_PRE_F8QK
    // fp8-direct QK (results/f8probe-20261028.md): D128 only - the SW128
    // K-major descriptor convention is probe-verified at KD=128 (one 128B
    // swizzle row). D64 needs a SWIZZLE_64B atom probe first; D256 rows are
    // two swizzle atoms (a different descriptor walk) and stay on the f16
    // path. Non-D128 instantiations compile the shipped code unchanged.
    // F8Q is the host-selected template gate: the planner routes short-walk
    // d128 shapes to cfg39 (f16) because the per-item Q split loses there
    // (r1 kv2048 +2.8% vs c0 kv8192+ -2.1%; results/f8qk-20261028.md).
    constexpr bool F8QK = F8Q && (D == 128);
#else
    constexpr bool F8QK = false;
#endif

    // descriptors / idescs (probe-verified constants)
    const uint64_t da_q = tc_desc(S.q, 0, 128, (D / 8) * 128);
#ifdef MYPRE_KF16
    // kind::f16 idesc a/b format field: 0 = F16, 1 = BF16. Q is staged as f16 and the K
    // image is converted straight to f16, so both operands of QK are f16; the PV idesc stays
    // bf16 because the reference rounds weighted_p to q.dtype = bf16 before the product.
    const uint32_t idesc_qk = tc_idesc(TCBM, BNT, 0, 0, 0);
#else
    const uint32_t idesc_qk = tc_idesc(TCBM, BNT, 0, 1, 1);
#endif
#ifdef FMHA_PRE_F8QK
    // fp8-direct QK descriptors (probe-verified): SW128 K-major byte images,
    // LBO=0, SBO=1024, desc addr +2 per 32-byte K step; idesc atype/btype
    // code 0 = E4M3 for kind::f8f6f4. Folded away for non-D128.
    const uint64_t dQhi = F8QK ? tc_desc(S.qhi(), 2, 0, 1024) : 0ull;
    const uint64_t dQlo = F8QK ? tc_desc(S.qlo(), 2, 0, 1024) : 0ull;
    const uint32_t idesc_f8 = F8QK ? tc_idesc(TCBM, BNT, 0, 0, 0) : 0u;
#endif
#ifdef MYPRE_VF16
    // kind::f16 for PV as well as QK: atype is P in TMEM, btype is V in shared. The 16-bit
    // packing and the swizzle-128B MN-major V image are unchanged - both types are 2 bytes -
    // so only the descriptor's two type fields move, exactly as MYPRE_KF16 moved idesc_qk's.
    const uint32_t idesc_pv = tc_idesc(TCBM, D, 1, 0, 0);
#else
    const uint32_t idesc_pv = tc_idesc(TCBM, D, 1, 1, 1);
#endif
    int tg = 0;                      // GLOBAL kv-tile counter (stage/parity)
    for (int which = 0; ; ++which) {
    int b, q0, qlen, tile, h;
    if (P.persist) {
        if (which > 0) {
            if (tid == 0 && (!split || crank == 0))
                S.tmem_base[1] = (unsigned)atomicAdd(P.counter, 1);
            if (split) cl_sync();
            if (tid == 0 && split && crank != 0)
                S.tmem_base[1] = cl_ldu32(cl_mapa(smem_u32(&S.tmem_base[1]), 0));
            if (split) cl_sync();   // see the initial pull: rank 0 may exit first
            __syncthreads();
            work0 = (int)S.tmem_base[1];
        }
        if (work0 >= T_all) break;
        const int work = T_all - 1 - work0;    // heavy tiles first (LPT-like)
        h = work % P.heads;                    // heads innermost: GQA shares
        const int rem = work / P.heads;        // KV pages -> L2 reuse
        b = -1;
        int base2 = 0, prev2 = 0;
        for (int i = 0; i < P.batch; ++i) {
            int c = __ldg(P.cu + i + 1);
            int ntiles = (c - prev2 + TCBM - 1) / TCBM;
            if (rem < base2 + ntiles) {
                b = i; tile = rem - base2; q0 = prev2; qlen = c - prev2; break;
            }
            base2 += ntiles;
            prev2 = c;
        }
        if (b < 0) break;
    } else {
        // static: one tile; fold: exactly two (which 0,1) — v4.8's for(;;)
        // lost the which<2 bound and spun forever on tileB (2 CHILD_TIMEOUTs).
        if (which > 0 && (!P.fold || which > 1)) break;
        b = b_s; q0 = q0_s; qlen = qlen_s; h = blockIdx.y;
        tile = (which == 0) ? tileA : tileB;
        if (tile < 0) break;
    }
    const int kvh = h / P.g;
    const int kvlen = __ldg(P.seqk + b);
    const int i0 = tile * TCBM;
    const int nrows = min(TCBM, qlen - i0);
    // ---- Q staging ----
#ifdef FMHA_PRE_F8QK
    if constexpr (F8QK) {
        // fp8-direct, SINGLE PASS (v2 of the arm; the first f8qk job measured
        // the two-pass form costing +6.1% on short-walk static shapes: r1 has
        // 32 KV tiles per item, so the second global Q read and the middle
        // barrier did not amortize). Each thread keeps its uint4 of 8 bf16
        // LIVE in registers across the row-max reduce; the 16 threads of a
        // row reduce |q|max in four width-16 shfl rounds; the group leader
        // turns it into the power-of-two scale (row max -> 2^2, the probe's
        // normalization), shfl-broadcasts it back (no smem round-trip, no
        // middle barrier) and publishes qinv for the softmax. One global
        // read, one barrier, and the split arithmetic is bit-identical to
        // the two-pass form (same scale, same PTX cvt pair per element).
        constexpr int PER_ROW = D / 8;
        for (int idx = tid; idx < TCBM * PER_ROW; idx += TC_THREADS) {
            const int row = idx / PER_ROW;
            const int col8 = (idx % PER_ROW) * 8;
            uint4 rawv = make_uint4(0u, 0u, 0u, 0u);
            if (row < nrows) {
                const uint16_t* src = P.q + (long long)(q0 + i0 + row) * P.q_out_stride
                                    + h * D + col8;
                rawv = *reinterpret_cast<const uint4*>(src);
            }
            const unsigned* w = reinterpret_cast<const unsigned*>(&rawv);
            float mx = 0.f;
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float f0 = bf16u_to_f32((unsigned short)(w[u] & 0xFFFFu));
                const float f1 = bf16u_to_f32((unsigned short)(w[u] >> 16));
                mx = fmaxf(mx, fmaxf(fabsf(f0), fabsf(f1)));
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off, 16));
            float sc = 1.f;
            if ((tid & 15) == 0) {
                int e2 = 0;
                (void)frexpf(mx, &e2);                // mx = m * 2^e2, m in [0.5,1)
                int e = (mx > 0.f) ? (e2 - 1) : 0;    // floor(log2(mx))
                e = e < -120 ? -120 : e;              // scale stays finite (<= 2^122)
                sc = exp2f((float)(2 - e));
                S.qinv()[row] = exp2f((float)(e - 2));
            }
            sc = __shfl_sync(0xffffffffu, sc, (tid & 31) & ~15);
            union { uint8_t b[8]; uint2 u2; } H, L;
            H.u2 = make_uint2(0u, 0u);
            L.u2 = make_uint2(0u, 0u);
            if (row < nrows) {
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float q0f = bf16u_to_f32((unsigned short)(w[u] & 0xFFFFu)) * sc;
                    const float q1f = bf16u_to_f32((unsigned short)(w[u] >> 16)) * sc;
                    const uint16_t h2 = f32x2_to_e4m3x2(q1f, q0f);
                    const float hf0 = e4m3x2_lo_to_f32(h2);
                    const float hf1 = e4m3x2_hi_to_f32(h2);
                    const uint16_t l2 = f32x2_to_e4m3x2(q1f - hf1, q0f - hf0);
                    H.b[2 * u] = (uint8_t)(h2 & 0xFFu);
                    H.b[2 * u + 1] = (uint8_t)(h2 >> 8);
                    L.b[2 * u] = (uint8_t)(l2 & 0xFFu);
                    L.b[2 * u + 1] = (uint8_t)(l2 >> 8);
                }
            }
            *reinterpret_cast<uint2*>(S.qhi() + TcSmem<D, BNT>::k8_off(row, col8)) = H.u2;
            *reinterpret_cast<uint2*>(S.qlo() + TcSmem<D, BNT>::k8_off(row, col8)) = L.u2;
        }
        __syncthreads();   // qinv + Q images published before first use
    } else
#endif
    // (bf16 -> padded bf16 smem, zero rows >= nrows)
    {
        constexpr int PER_ROW = D / 8;
        char* qb = reinterpret_cast<char*>(S.q);
        for (int idx = tid; idx < TCBM * PER_ROW; idx += TC_THREADS) {
            int row = idx / PER_ROW;
            int col8 = (idx % PER_ROW) * 8;
            uint4* d = reinterpret_cast<uint4*>(qb + TcSmem<D, BNT>::km_off(row, col8));
            if (row < nrows) {
                const uint16_t* src = P.q + (long long)(q0 + i0 + row) * P.q_out_stride
                                    + h * D + col8;
#ifdef MYPRE_KF16
                *d = bf16x8_to_f16x8(*reinterpret_cast<const uint4*>(src));
#else
                *d = *reinterpret_cast<const uint4*>(src);
#endif
            } else {
                *d = make_uint4(0, 0, 0, 0);
            }
        }
    }

    const int row = warp * 32 + lane;               // this thread's q row
    const int visible = (row < nrows) ? (kvlen - qlen + i0 + row) : 0;
#ifdef FMHA_PRE_F8QK
    // F8QK: QK scores arrive in device units (q was scaled by 2^k_row); invr
    // restores the true domain inside the exp2 argument (power of two => the
    // multiply is exact). row == tid owns this row's qinv entry.
    const float invr = F8QK ? S.qinv()[row] : 1.0f;
#endif
    const int kn_full = min(kvlen, kvlen - qlen + i0 + TCBM);
    // this rank's slice of the item's KV range, in whole BNT tiles. A rank past
    // the end gets an empty range, publishes l2 == 0, and the merge skips its O.
    int kn_begin = 0, kn_end = kn_full;
#ifdef MYPRE_CL_NOSLICE
    // Discriminator, never shipped: keep the cluster, the ticket sharing, the
    // staging and the whole merge, but give every rank the WHOLE KV range. All CL
    // partials are then identical, so m* = m_p, every rescale factor is exactly
    // 1.0, l* = CL*l and O* = CL*O, and rcp.approx(CL*l) == rcp.approx(l)/CL
    // exactly because scaling by a power of two only moves the exponent - i.e.
    // this arm must reproduce the unsplit output BIT FOR BIT while exercising all
    // four ranks' barriers and DSMEM traffic. If it faults, the defect is in the
    // cluster machinery at CL=4; if it is bit-identical, the defect is in the
    // slicing, which is the part CMERGE already printed as correct.
    const bool slice = false;
#else
    const bool slice = split;
#endif
    if (slice) {
        const int ntiles = (kn_full + BNT - 1) / BNT;
        const int per = (ntiles + CL - 1) / CL;
        kn_begin = min(kn_full, (int)crank * per * BNT);
        kn_end = min(kn_full, kn_begin + per * BNT);
    }
    const bool any = (kn_begin < kn_end);      // always true when !split

    // Staging width for THIS head_dim: D64 may run the multi-pass TPRW64 mapping
    // (dv_b5739e5f0521); every other D keeps the shipped single-pass TC_TPRW, so
    // their codegen is untouched by the knob. Same chunks, same smem destinations,
    // bit-identical by construction (see the ConvRegs header).
    constexpr int TPRW_E = (D == 64) ? TC_TPRW_D64 : TC_TPRW;
    ConvRegs<D, BNT, TC_THREADS, TPRW_E> R;
#ifdef FMHA_PRE_PF2
    // MEASURED AND REJECTED (dv_c993807fb085, results/prefill-next-levers-20261025.md section 1.1):
    // +13.35% on case 9 and +4.41% across six cases, correctness clean with the worst rel_l2
    // identical, so the reordering is arithmetically right and simply slower. Whether it lost the
    // second CTA/SM to registers (183 + 32 = 215 by arithmetic, but ptxas allocates what its
    // schedule needs, and MYPRE_RT2P measured +22.0% for exactly that trade) or materialised the
    // R = R2 shift as 34 movs per tile is NOT determined - the FMHA_DBG_PRE level-2 regs print did
    // not fire, so there is no register count. Settling that is the precondition for a second
    // attempt, and it decides between the unrolled generic-lambda form (no shift at all) and the
    // register-free cp.async variant in Design B.
    //
    // Two tiles of raw fp8 in flight instead of one. R holds the tile this
    // iteration converts; R2 receives the tile AFTER it, so the loads get a
    // whole extra iteration of slack before anything reads them. 32 registers
    // (uint2 k[8] + unsigned v[16] + 2 scales) against the 73 of headroom
    // between this kernel's 183 and the 256 that 2 CTAs x 128 threads allows.
    ConvRegs<D, BNT, TC_THREADS, TPRW_E> R2;
#endif
    using CR = ConvRegs<D, BNT, TC_THREADS, TPRW_E>;
    // v3.0: block-table entry one tile ahead (bt_n) so KV data LDGs issue
    // immediately; long_scoreboard was the top stall (0.60/inst, case0).
    // One entry per staging pass (TC_PASSES=1 when TC_TPRW=0, i.e. shipped).
    int bt_n[CR::PASSES];
    kv_bt_fill<BNT, TC_THREADS, TPRW_E>(P, b, kn_begin, kn_end, bt_n);
    kv_prefetch_m<D, BNT, TC_THREADS, TPRW_E>(R, P, kvh, kn_begin,
                                               min(BNT, kn_end - kn_begin), bt_n);
    if (kn_begin + BNT < kn_end)
        kv_bt_fill<BNT, TC_THREADS, TPRW_E>(P, b, kn_begin + BNT, kn_end, bt_n);
    // An empty slice must not run the prologue: it would commit bars[] with no
    // matching wait and desynchronise this CTA's stage parity for the next item,
    // which is the merge3-pf hang class.
    if constexpr (PIPE) if (!SPLIT || any) {
        // prologue: tile 0 -> smem[s0], issue QK(0); main loop keeps the
        // tensor pipe packed (see header comment).
        // merge3-pf PARITY FIX: tg is GLOBAL across driver items, so an item
        // can start at ODD parity whenever a previous item's KV-tile count
        // was odd - any shape with ceil((kvlen-qlen+128)/64) odd (e.g. fixed
        // c0 seq1: 115 tiles/item). The old stage-0 prologue committed
        // bars[0] while the loop's first wait targets bars[tg&1] -> mbarrier
        // phase mismatch -> HANG (merge3-p128 A/B CHILD_TIMEOUTs on c0,
        // dv_f1235933d079). Latent in the whole v4.x lineage: D128/256 never
        // ran PIPE, and every measured D64-PIPE shape happened to have even
        // per-item tile counts (square causal 2+2t, domain d4 66+2t). The
        // seq loop is parity-safe (commit and wait are in-iteration).
        // merge4-ev (ADOPTED, dv_0c333d7e9fa1, 3 alternating reps):
        // parity-branched prologue with two LITERAL-stage bodies. The dynamic
        // s0 = tg&1 of merge3-pf shifted ptxas scheduling even for even-tg
        // items (-1.3% c7, -0.7% c0); constant-folding both branches recovers
        // it (c7 0.3704->0.3657 = pre-defect band, c0 1.5167->1.5066, c9/c1
        // flat). The branch is CTA-uniform (tg is uniform), once per item;
        // odd-body semantics = merge3-pf (probes pf_persist_d64_odd 0/3145,
        // pf_persist_d128_odd 5/16384 green).
        // merge4-ev A/B arm: parity-branched prologue with two LITERAL-stage
        // bodies. The dynamic s0 = tg&1 of merge3-pf changed ptxas scheduling
        // even for even-tg items (~-1% on c7/d4: A/B ctl 0.3691 vs pre-fix
        // 0.3649); both branches below constant-fold stage/bar/TMEM-column,
        // and the branch is CTA-uniform (tg is uniform), taken once per item.
        if ((tg & 1) == 0) {
            tc_convert<D, BNT, TC_THREADS, TPRW_E, F8QK>(R, S, 0, min(BNT, kn_end - kn_begin), P);
            tc_fence_async_smem();
            __syncthreads();
            if (tid == 0) {
#ifdef FMHA_PRE_F8QK
                if constexpr (F8QK) {
                    uint64_t db8 = tc_desc(S.k8(0), 2, 0, 1024);
                    #pragma unroll
                    for (int kk = 0; kk < D / 32; ++kk)
                        tc_mma_f8(tm, dQhi + kk * 2, db8 + kk * 2, idesc_f8,
                                  kk > 0 ? 1u : 0u);
                    #pragma unroll
                    for (int kk = 0; kk < D / 32; ++kk)
                        tc_mma_f8(tm, dQlo + kk * 2, db8 + kk * 2, idesc_f8, 1u);
                } else
#endif
                {
                    uint64_t db = tc_desc(S.k(0), 0, 128, (D / 8) * 128);
                    #pragma unroll
                    for (int kk = 0; kk < D / 16; ++kk)
                        tc_mma_ss(tm, da_q + kk * 16, db + kk * 16, idesc_qk,
                                  kk > 0 ? 1u : 0u);
                }
                tc_commit(&S.bars[0]);
            }
        } else {
            tc_convert<D, BNT, TC_THREADS, TPRW_E, F8QK>(R, S, 1, min(BNT, kn_end - kn_begin), P);
            tc_fence_async_smem();
            __syncthreads();
            if (tid == 0) {
#ifdef FMHA_PRE_F8QK
                if constexpr (F8QK) {
                    uint64_t db8 = tc_desc(S.k8(1), 2, 0, 1024);
                    #pragma unroll
                    for (int kk = 0; kk < D / 32; ++kk)
                        tc_mma_f8(tm + BNT, dQhi + kk * 2, db8 + kk * 2, idesc_f8,
                                  kk > 0 ? 1u : 0u);
                    #pragma unroll
                    for (int kk = 0; kk < D / 32; ++kk)
                        tc_mma_f8(tm + BNT, dQlo + kk * 2, db8 + kk * 2, idesc_f8, 1u);
                } else
#endif
                {
                    uint64_t db = tc_desc(S.k(1), 0, 128, (D / 8) * 128);
                    #pragma unroll
                    for (int kk = 0; kk < D / 16; ++kk)
                        tc_mma_ss(tm + BNT, da_q + kk * 16, db + kk * 16, idesc_qk,
                                  kk > 0 ? 1u : 0u);
                }
                tc_commit(&S.bars[1]);
            }
        }
#if defined(FMHA_PRE_PF2) || defined(FMHA_PRE_PFLATE)
        // The loop's step 1 now prefetches tile t+2 into R2, so tile 1 has to be
        // sitting in R before the first iteration: bt_n was filled for tile 1
        // above, and is refilled for tile 2 here so the first step 1 can issue its
        // KV loads without waiting on a block-table load.
        if (kn_begin + BNT < kn_end) {
            kv_prefetch_m<D, BNT, TC_THREADS, TPRW_E>(R, P, kvh, kn_begin + BNT,
                                                       min(BNT, kn_end - kn_begin - BNT), bt_n);
            kv_bt_fill<BNT, TC_THREADS, TPRW_E>(P, b, kn_begin + 2 * BNT, kn_end, bt_n);
        }
#endif
    }

    float m2 = NEG, l2 = 0.f;
    int tl = 0;                      // LOCAL tile index (PV accumulate flag)
    if constexpr (PIPE) {
        for (int kn = kn_begin; kn < kn_end; kn += BNT, ++tl, ++tg) {
            const int s = tg & 1;
            const bool more = (kn + BNT < kn_end);
            // 1) prefetch raw tile t+1 (bt already resident in bt_n)
#ifndef MYPRE_T_NOPREF
#ifdef FMHA_PRE_PF2
            if (kn + 2 * BNT < kn_end) {
                kv_prefetch_m<D, BNT, TC_THREADS, TPRW_E>(R2, P, kvh, kn + 2 * BNT,
                                                           min(BNT, kn_end - kn - 2 * BNT), bt_n);
                kv_bt_fill<BNT, TC_THREADS, TPRW_E>(P, b, kn + 3 * BNT, kn_end, bt_n);
            }
#elif defined(FMHA_PRE_PFLATE)
            // Nothing here. PFLATE issues the prefetch at step 5, the instant the convert
            // has consumed R, so tile t+2's loads are read at step 5 of the NEXT iteration:
            // slack of steps 6,7,8 plus the next iteration's 1,2,3,4 instead of only 2,3,4,
            // for zero extra registers and zero extra shared memory. bt_n already holds tile
            // t+2's entries, filled at the previous step 5.
#else
            if (more) {
                kv_prefetch_m<D, BNT, TC_THREADS, TPRW_E>(R, P, kvh, kn + BNT, min(BNT, kn_end - kn - BNT), bt_n);
                kv_bt_fill<BNT, TC_THREADS, TPRW_E>(P, b, kn + 2 * BNT, kn_end, bt_n);
            }
#endif
#endif
            // 2) QK(t) done
#ifdef FMHA_PRE_WAITELECT
            tc_wait_e(&S.bars[s], (tg >> 1) & 1);
#else
            tc_wait(&S.bars[s], (tg >> 1) & 1);
#endif

#ifdef FMHA_PRE_KFIRST
            // 3a) convert K(t+1) and its ks/vs metadata, then 6a) issue QK(t+1) - both ahead of the
            //     softmax, which is the whole point: QK(t+1) then has the softmax, the PV(t-1) wait,
            //     the V convert, the correction and the next prefetch as slack instead of only the
            //     correction and the prefetch. `more` is not declared until step 5, so the same
            //     predicate is recomputed here rather than moved, which keeps the default build
            //     textually unchanged. The ks/vs metadata rides with the K pass because the softmax
            //     that reads it now follows this one.
            if (kn + BNT < kn_end) {
                tc_convert_k<D, BNT, TC_THREADS, TPRW_E>(R, S, s ^ 1,
                                                          min(BNT, kn_end - kn - BNT), P);
                tc_fence_async_smem();
                tc_fence_before();
                __syncthreads();
                tc_fence_after();
                if (tid == 0) {
                    uint64_t dbk = tc_desc(S.k(s ^ 1), 0, 128, (D / 8) * 128);
                    #pragma unroll
                    for (int kk = 0; kk < D / 16; ++kk)
                        tc_mma_ss(tm + (s ^ 1) * BNT, da_q + kk * 16, dbk + kk * 16,
                                  idesc_qk, kk > 0 ? 1u : 0u);
                    tc_commit(&S.bars[s ^ 1]);
                }
            }
#endif

#ifdef MYPRE_T_NOSOFT
            // probe: no S load, no softmax ALU, no P store. alpha stays exactly 1.0
            // so the correction's __any_sync gate goes false on its own and the O
            // rescale is skipped too - which is why (ctl - NOSOFT) - (ctl - NOCORR)
            // is the softmax-plus-TMEM-round-trip and NOCORR alone is the rescale.
            const float mn = m2;
            float alpha = 1.0f;
            const float sum = 0.f;
#else
            // ---- softmax (row per lane) ----
            float sc[BNT];
            {
                uint32_t saddr = tm + s * BNT + ((warp * 32) << 16);
                #pragma unroll
                for (int seg = 0; seg < BNT / 32; ++seg)
                    tc_ld_32(sc + seg * 32, saddr + seg * 32);
                tc_wait_ld();
            }
            const float* kss = S.ks(s);   // pre-scaled by sm_scale*log2e
            const float* vss = S.vs(s);
#ifdef FMHA_PRE_VSMAXP
            vs_keepalive(vs_tile_max<BNT>(vss), sc);   // prices the pass; see the helper's comment
#endif
            // v3.5: 4-way ILP — the serial 64-deep FMAX/FADD chains cost
            // ~256 clk each of pure latency (stalled_wait 0.59/inst).
            float mn0 = m2, mn1 = m2, mn2 = m2, mn3 = m2;
            // merge2-nm: warp-uniform mask-free fast path (NM template gate;
            // persist launches only - see the A/B note at the top of this
            // file). Lane 0 owns the warp's smallest visible (rows ascend
            // with lane); a tile whose last key is <= that masks nothing ->
            // skip 128 ISETP+SEL per tile. Inactive rows (visible=0) take the
            // slow path. Warp-uniform branch: tcgen05/mbarrier ops stay
            // converged. NM=false folds the shfl and the branch away.
            bool nomask = false;
            if (NM) nomask = (kn + BNT - 1 <= __shfl_sync(0xffffffffu, visible, 0));
            if (nomask) {
#if defined(MYPRE_T_NOKSMUL) || defined(MYPRE_KSFOLD)
                // ---- PRICE PROBE (MYPRE_T_NOKSMUL) or the real fold (MYPRE_KSFOLD), wrong numbers by construction. Never shippable; run with --nocorrect. ----
                // Drops the per-key score scale, so the output is wrong by construction. It prices the
                // softmax's own BNT/4 broadcast float4 loads of S.ks plus BNT FMULs - 16 and 64 at BNT=64 -
                // which is the quantity a K-side fold would delete, by multiplying K's f16 image by
                // (R.ks[p] * sm_scale_log2e) inside tc_convert with one mul.f16x2 per pair. tc_convert
                // already holds that exact value on the lane that stores S.ks, and MYPRE_KF16 already makes
                // the K image f16, so the packed multiply exists. The construct being priced is the same one
                // FMHA_PRE_VSMAXP ADDED for the value scale and measured at +1.17%/+0.67% on cases 0/9 (16
                // broadcast float4 loads plus 63 FMAX = 79 instructions against these 80), and the shipped
                // calibration is that instruction count converts at ~0.13 elasticity here, so ~80 of ~720
                // instructions predicts roughly -1.2%/-0.7% before counting the ~32 mul.f16x2 the fold adds
                // back in the producer phase that is already waiting on global latency.
                // Numerics if it were built: K' = round_f16(K_fp8 * ks) costs 2^-12 relative per element, and
                // with independent rounding the dot product's relative error stays ~2^-12 = 2.4e-4, so
                // delta_p/p ~ ln2 * |s| * 2.4e-4 against the current worst rel_l2 of 1.4e-4. Range is safe:
                // |K*ks| <= 448 * key_scale * sm_scale * log2e ~ 16, and the small end goes f16-subnormal
                // where the error is absolute (2^-25) against |s| ~ O(1), i.e. negligible.
                #pragma unroll
                for (int j = 0; j < BNT; j += 4) {
                    float s0 = sc[j];
                    float s1 = sc[j + 1];
                    float s2 = sc[j + 2];
                    float s3 = sc[j + 3];
                    sc[j] = s0; sc[j + 1] = s1; sc[j + 2] = s2; sc[j + 3] = s3;
                    mn0 = fmaxf(mn0, s0); mn1 = fmaxf(mn1, s1);
                    mn2 = fmaxf(mn2, s2); mn3 = fmaxf(mn3, s3);
                }
#elif defined(MYPRE_T_MX3)
                // ---- MYPRE_T_MX3: 3-input row max, BIT-IDENTICAL to the #else below. ----
                // Each of the four accumulators absorbs TWO values instead of one, so the source
                // says fmaxf(mn, fmaxf(a, b)) and ptxas can fuse that into a single FMNMX3
                // (sm_100+; this tree already relies on it - fa_prefill_w8's softmax has been
                // written as `mnA = fmaxf(mnA, fmaxf(a0, a1))` since it was split, so tc is the
                // only prefill kernel still emitting the flat form). BNT FMNMX -> BNT/2 FMNMX3,
                // i.e. 64 -> 32 instructions per thread per tile at BNT=64, ~4.4% of the ~720.
                // The step doubles to 8 so the accumulator count stays four and the chain depth
                // per accumulator halves, 16 -> 8; four independent chains already cover it (the
                // pre-join SASS dump of this loop, artifacts/20261023T110222Z/sass_g_w8_fmha_tc128,
                // shows R13/R15/R17/R19 issuing round-robin with no gap). Eight s values are live
                // at once instead of four: free at 183/255 registers and 2 CTAs/SM, which is
                // TMEM-capped rather than register-capped.
                // fmaxf is exact and, under IEEE NaN-is-identity semantics, associative and
                // commutative, so ANY grouping of the same multiset gives the same bits: this
                // changes the instruction stream and nothing else. That is why the sum below is
                // NOT widened - FADD reassociates and would move bits.
                static_assert(BNT % 8 == 0, "MYPRE_T_MX3 needs BNT divisible by 8");
                #pragma unroll
                for (int j = 0; j < BNT; j += 8) {
                    const float4 ka = *reinterpret_cast<const float4*>(kss + j);
                    const float4 kb = *reinterpret_cast<const float4*>(kss + j + 4);
                    float s0 = sc[j] * ka.x;
                    float s1 = sc[j + 1] * ka.y;
                    float s2 = sc[j + 2] * ka.z;
                    float s3 = sc[j + 3] * ka.w;
                    float s4 = sc[j + 4] * kb.x;
                    float s5 = sc[j + 5] * kb.y;
                    float s6 = sc[j + 6] * kb.z;
                    float s7 = sc[j + 7] * kb.w;
                    sc[j] = s0; sc[j + 1] = s1; sc[j + 2] = s2; sc[j + 3] = s3;
                    sc[j + 4] = s4; sc[j + 5] = s5; sc[j + 6] = s6; sc[j + 7] = s7;
                    mn0 = fmaxf(mn0, fmaxf(s0, s1));
                    mn1 = fmaxf(mn1, fmaxf(s2, s3));
                    mn2 = fmaxf(mn2, fmaxf(s4, s5));
                    mn3 = fmaxf(mn3, fmaxf(s6, s7));
                }
#else
                #pragma unroll
                for (int j = 0; j < BNT; j += 4) {
                    const float4 ks4 = *reinterpret_cast<const float4*>(kss + j);
                    float s0 = sc[j] * ks4.x;
                    float s1 = sc[j + 1] * ks4.y;
                    float s2 = sc[j + 2] * ks4.z;
                    float s3 = sc[j + 3] * ks4.w;
                    sc[j] = s0; sc[j + 1] = s1; sc[j + 2] = s2; sc[j + 3] = s3;
                    mn0 = fmaxf(mn0, s0); mn1 = fmaxf(mn1, s1);
                    mn2 = fmaxf(mn2, s2); mn3 = fmaxf(mn3, s3);
                }
#endif
            } else {
#ifdef MYPRE_T_MX3
                // MYPRE_T_MX3 masked variant - the same 3-input reduction as the nomask branch,
                // with the same per-column NEG selects, now eight of them. Bit-identical: the
                // multiset fed to the max is unchanged, only the grouping is.
                static_assert(BNT % 8 == 0, "MYPRE_T_MX3 needs BNT divisible by 8");
                #pragma unroll
                for (int j = 0; j < BNT; j += 8) {
                    const float4 ka = *reinterpret_cast<const float4*>(kss + j);
                    const float4 kb = *reinterpret_cast<const float4*>(kss + j + 4);
                    float s0 = sc[j] * ka.x;
                    float s1 = sc[j + 1] * ka.y;
                    float s2 = sc[j + 2] * ka.z;
                    float s3 = sc[j + 3] * ka.w;
                    float s4 = sc[j + 4] * kb.x;
                    float s5 = sc[j + 5] * kb.y;
                    float s6 = sc[j + 6] * kb.z;
                    float s7 = sc[j + 7] * kb.w;
                    if (kn + j > visible) s0 = NEG;
                    if (kn + j + 1 > visible) s1 = NEG;
                    if (kn + j + 2 > visible) s2 = NEG;
                    if (kn + j + 3 > visible) s3 = NEG;
                    if (kn + j + 4 > visible) s4 = NEG;
                    if (kn + j + 5 > visible) s5 = NEG;
                    if (kn + j + 6 > visible) s6 = NEG;
                    if (kn + j + 7 > visible) s7 = NEG;
                    sc[j] = s0; sc[j + 1] = s1; sc[j + 2] = s2; sc[j + 3] = s3;
                    sc[j + 4] = s4; sc[j + 5] = s5; sc[j + 6] = s6; sc[j + 7] = s7;
                    mn0 = fmaxf(mn0, fmaxf(s0, s1));
                    mn1 = fmaxf(mn1, fmaxf(s2, s3));
                    mn2 = fmaxf(mn2, fmaxf(s4, s5));
                    mn3 = fmaxf(mn3, fmaxf(s6, s7));
                }
#else
                #pragma unroll
                for (int j = 0; j < BNT; j += 4) {
                    // v3.7: vectorized scale loads (kss is 16B-aligned: ksvs
                    // base is 512B-aligned, j*4 float = 16B steps).
#if defined(MYPRE_T_NOKSMUL) || defined(MYPRE_KSFOLD)
                    float s0 = sc[j];              // PRICE PROBE, wrong numbers; see the nomask note above
                    float s1 = sc[j + 1];
                    float s2 = sc[j + 2];
                    float s3 = sc[j + 3];
#else
                    const float4 ks4 = *reinterpret_cast<const float4*>(kss + j);
                    float s0 = sc[j] * ks4.x;
                    float s1 = sc[j + 1] * ks4.y;
                    float s2 = sc[j + 2] * ks4.z;
                    float s3 = sc[j + 3] * ks4.w;
#endif
                    if (kn + j > visible) s0 = NEG;
                    if (kn + j + 1 > visible) s1 = NEG;
                    if (kn + j + 2 > visible) s2 = NEG;
                    if (kn + j + 3 > visible) s3 = NEG;
                    sc[j] = s0; sc[j + 1] = s1; sc[j + 2] = s2; sc[j + 3] = s3;
                    mn0 = fmaxf(mn0, s0); mn1 = fmaxf(mn1, s1);
                    mn2 = fmaxf(mn2, s2); mn3 = fmaxf(mn3, s3);
                }
#endif
            }
            const float mn = fmaxf(fmaxf(mn0, mn1), fmaxf(mn2, mn3));
#ifdef FMHA_PRE_F8QK
            // F8QK: mn/m2 are device-domain (Q was row-scaled by 2^k); mnt and
            // the fmaf restore the true domain INSIDE the exp2 argument. invr
            // is a power of two, so both multiplies are exact, and fma(x,1,-y)
            // == x-y makes the non-F8QK instantiations bit-identical.
            const float mnt = F8QK ? mn * invr : mn;
            float alpha = F8QK ? ex2f(fmaf(m2, invr, -mnt)) : ex2f(m2 - mn);
#else
            float alpha = ex2f(m2 - mn);
#endif
            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
            unsigned wp[BNT / 2];
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
#ifdef FMHA_PRE_F8QK
                float p0, p1, p2, p3;
                if constexpr (F8QK) {
                    p0 = ex2f(fmaf(sc[j], invr, -mnt));
                    p1 = ex2f(fmaf(sc[j + 1], invr, -mnt));
                    p2 = ex2f(fmaf(sc[j + 2], invr, -mnt));
                    p3 = ex2f(fmaf(sc[j + 3], invr, -mnt));
                } else {
                    p0 = ex2f(sc[j] - mn);
                    p1 = ex2f(sc[j + 1] - mn);
                    p2 = ex2f(sc[j + 2] - mn);
                    p3 = ex2f(sc[j + 3] - mn);
                }
#else
                float p0 = ex2f(sc[j] - mn);
                float p1 = ex2f(sc[j + 1] - mn);
                float p2 = ex2f(sc[j + 2] - mn);
                float p3 = ex2f(sc[j + 3] - mn);
#endif
                sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
                sc[j] = p0; sc[j + 1] = p1;   // reuse sc storage for p
                sc[j + 2] = p2; sc[j + 3] = p3;
            }
            const float sum = (sum0 + sum1) + (sum2 + sum3);
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
#ifdef MYPRE_VSFOLD
                // VSFOLD: V's shared image already carries value_scale, so this loop has neither the scale nor
                // its 16 broadcast float4 loads - 16 LDS.128 and 64 FMUL per thread per tile deleted, which the
                // load->use cost model priced at ~5.1% and ~1.0% respectively.
                wp[j >> 1] = cvt_bf16x2_f32(sc[j], sc[j + 1]);
                wp[(j >> 1) + 1] = cvt_bf16x2_f32(sc[j + 2], sc[j + 3]);
#else
                const float4 vs4 = *reinterpret_cast<const float4*>(vss + j);
#ifdef MYPRE_VF16
                wp[j >> 1] = cvt_f16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
                wp[(j >> 1) + 1] = cvt_f16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
#else
                wp[j >> 1] = cvt_bf16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
                wp[(j >> 1) + 1] = cvt_bf16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
#endif
#endif
            }
#endif  // MYPRE_T_NOSOFT
            m2 = mn;
            l2 = l2 * alpha + sum;
#ifndef FMHA_PRE_NO_NEEDCTA
            const bool need_w = __any_sync(0xffffffffu, alpha != 1.0f);
            if (lane == 0) S.needf[s][warp] = need_w ? 1 : 0;
#endif
#ifndef MYPRE_T_NOSOFT
            {
                uint32_t paddr = tm + s * BNT + ((warp * 32) << 16);
                if constexpr (BNT >= 64) {
                    #pragma unroll
                    for (int seg = 0; seg < BNT / 64; ++seg)
                        tc_st_32(paddr + seg * 32, wp + seg * 32);
                } else {
                    tc_st_16(paddr, wp);   // BNT=32: 16 packed-bf16 cols
                }
                tc_wait_st();
            }
#endif  // MYPRE_T_NOSOFT
            // merge9-bd (ADOPTED, dv_039b8aec8287: c0 +3.3% c7 +2.2% c9 +0.9%,
            // probes green): sync#1 dropped. P-store (TMEM) and convert(t+1)
            // (smem[s^1]) touch disjoint media; PV(t-1) is a per-thread
            // mbarrier wait; P-store publication for PV(t) rides the
            // fence_before at the remaining convert sync (or the corr sync
            // on the last tile, where BOTH syncs vanish).
            // 4) wait PV(t-1): frees V/K smem[s^1], TMEM S[s^1] (P(t-1)) and O.
#ifdef FMHA_PRE_WAITELECT
            if (tl > 0) tc_wait_e(&S.bars[2 + (s ^ 1)], ((tg - 1) >> 1) & 1);
#else
            if (tl > 0) tc_wait(&S.bars[2 + (s ^ 1)], ((tg - 1) >> 1) & 1);
#endif
            // 5) convert tile t+1 -> smem[s^1] (QK(t-1) long done; PV(t-1) waited)
            if (more) {
#ifdef FMHA_PRE_KFIRST
                // 5b) the V half only: this is the half that genuinely needs the PV(t-1) wait above,
                //     because V[s^1] is what PV(t-1) is still reading.
                tc_convert_v<D, BNT, TC_THREADS, TPRW_E>(R, S, s ^ 1, min(BNT, kn_end - kn - BNT), P);
#elif !defined(MYPRE_T_NOCONV)
                tc_convert<D, BNT, TC_THREADS, TPRW_E, F8QK>(R, S, s ^ 1, min(BNT, kn_end - kn - BNT), P);
#endif
#ifdef FMHA_PRE_PFLATE
                // R is free the instant the convert has read it, so this is the earliest tile
                // t+2 can issue and the latest tile t+1's data is needed. Placed before the
                // proxy fence: that fence orders the convert's shared stores for the async
                // proxy and has nothing to do with these global loads, and everything between
                // here and the next step 5 is slack for them. The non-PIPE loop below already
                // prefetches after its convert, so the pattern is not new to this file.
                if (kn + 2 * BNT < kn_end) {
                    kv_prefetch_m<D, BNT, TC_THREADS, TPRW_E>(R, P, kvh, kn + 2 * BNT,
                                                               min(BNT, kn_end - kn - 2 * BNT), bt_n);
                    kv_bt_fill<BNT, TC_THREADS, TPRW_E>(P, b, kn + 3 * BNT, kn_end, bt_n);
                }
#endif
#ifdef FMHA_PRE_NBSYNC
                // Only warp 0 needs the convert published: it is the warp that issues
                // QK(t+1) and PV(t). The other three arrive and run ahead into tile t+1's
                // 24-load prefetch instead of rendezvousing. tc_fence_after is warp 0's for
                // the same reason - the other warps' next tcgen05 op is tile t+1's S load,
                // already ordered by the QK(t+1) mbarrier wait at step 2.
                tc_fence_async_smem();
                tc_fence_before();
                if (warp == 0) { nb_sync<1, TC_THREADS>(); tc_fence_after(); }
                else           { nb_arrive<1, TC_THREADS>(); }
#else
#ifndef MYPRE_T_NOFENCE5
                tc_fence_async_smem();
                tc_fence_before();
#endif
#ifndef MYPRE_T_NOSYNC5
                __syncthreads();
#endif
#ifndef MYPRE_T_NOFENCE5
                tc_fence_after();
#endif
#endif
            }
#ifndef FMHA_PRE_NO_NEEDCTA
            // Did ANY warp rescale O on this tile? If not, step 7's barrier and both
            // tcgen05 fences publish nothing: P(t) was published by the convert sync
            // above and O was not touched, so PV(t) can accumulate straight into it.
            // On the last tile there is no convert sync, so P(t) is NOT yet published
            // and the barrier must stay - hence the conservative default.
            // ADOPTED for this kernel: dv_becfffe306e7 measured case 0 0.9874 and case 9
            // 0.9745, and dv_9e8043fc9eac reproduced them at 0.9860/0.9840 after the flag
            // array was widened to serve W8, both jobs bit-identical to their controls and
            // with their decode nulls inside +/-0.13%. Both cases are wide (kv >> q), which
            // is why the gate fires: after a row has seen thousands of keys its running
            // maximum stops moving, and most tiles then publish nothing.
            bool need_cta = true;
            if (more) {
                int acc = 0;
                #pragma unroll
                for (int w = 0; w < TC_THREADS / 32; ++w) acc |= S.needf[s][w];
                need_cta = (acc != 0);
            }
#endif
            // 6) issue QK(t+1) BEFORE correction: tensor pipe runs QK(t+1) while
            //    threads rescale O below (disjoint TMEM: S[s^1] vs O region).
#ifndef FMHA_PRE_KFIRST
            if (more && tid == 0) {
                uint64_t dbk = tc_desc(S.k(s ^ 1), 0, 128, (D / 8) * 128);
#ifndef MYPRE_T_NOMMA
#ifdef FMHA_PRE_F8QK
                if constexpr (F8QK) {
                    uint64_t db8 = tc_desc(S.k8(s ^ 1), 2, 0, 1024);
                    #pragma unroll
                    for (int kk = 0; kk < D / 32; ++kk)
                        tc_mma_f8(tm + (s ^ 1) * BNT, dQhi + kk * 2, db8 + kk * 2,
                                  idesc_f8, kk > 0 ? 1u : 0u);
                    #pragma unroll
                    for (int kk = 0; kk < D / 32; ++kk)
                        tc_mma_f8(tm + (s ^ 1) * BNT, dQlo + kk * 2, db8 + kk * 2,
                                  idesc_f8, 1u);
                } else
#endif
                {
                    #pragma unroll
                    for (int kk = 0; kk < D / 16; ++kk)
                        tc_mma_ss(tm + (s ^ 1) * BNT, da_q + kk * 16, dbk + kk * 16,
                                  idesc_qk, kk > 0 ? 1u : 0u);
                }
#endif
                tc_commit(&S.bars[s ^ 1]);
            }
#endif
            // 7) correction: O *= alpha. TMEM loads batched 2 segments per
            //    wait_ld (wait::ld covers all outstanding) to halve the serial
            //    latency chain.
            {
#ifdef MYPRE_T_NOCORR
                const bool need = false;
#elif !defined(FMHA_PRE_NO_NEEDCTA)
                const bool need = need_w;      // already voted, before the P store
#else
                bool need = __any_sync(0xffffffffu, alpha != 1.0f);
#endif
                if (need) {
                    uint32_t oaddr = tm + TcSmem<D, BNT>::SOFF + ((warp * 32) << 16);
                    // merge6-c1 A/B (dv_c5c84ce108a3): 1-segment retest lost
                    // (c9/c1 -0.6%, c0 flat, net 0.999x) — 2-seg batch stays.
                    float o0[32], o1[32];
                    #pragma unroll
                    for (int seg = 0; seg < D / 32; seg += 2) {
                        const bool two = (seg + 1 < D / 32);
                        tc_ld_32(o0, oaddr + seg * 32);
                        if (two) tc_ld_32(o1, oaddr + (seg + 1) * 32);
                        tc_wait_ld();
                        unsigned w[32];
                        #pragma unroll
                        for (int j = 0; j < 32; ++j) w[j] = __float_as_uint(o0[j] * alpha);
                        tc_st_32(oaddr + seg * 32, w);
                        if (two) {
                            #pragma unroll
                            for (int j = 0; j < 32; ++j) w[j] = __float_as_uint(o1[j] * alpha);
                            tc_st_32(oaddr + (seg + 1) * 32, w);
                        }
                    }
                    tc_wait_st();
                }
            }
#ifndef FMHA_PRE_NO_NEEDCTA
            if (need_cta)
#endif
            {
            tc_fence_before();
#ifdef FMHA_PRE_NBSYNC
            // Composes with need_cta: when no warp rescaled O, nobody arrives and nobody
            // waits, exactly as with the CTA barrier.
            if (warp == 0) { nb_sync<2, TC_THREADS>(); tc_fence_after(); }
            else           { nb_arrive<2, TC_THREADS>(); }
#else
#ifndef MYPRE_T_NOSYNC7
            __syncthreads();
#endif
            tc_fence_after();
#endif
            }

            // 8) PV(t): P(t) ready (sync at step-3 end), O corrected (sync above),
            //    V smem[s] intact until convert(t+2) (iter t+1 step 5, after the
            //    PV(t) wait). Pipe: QK(t+1) -> PV(t) packed.
            if (tid == 0) {
                uint64_t db = tc_desc(S.v(s), 2, BNT * 128, 1024);
                const int adv = 128; // K16 = 16 rows * 128B swizzle-atom stride >> 4
#ifndef MYPRE_T_NOMMA
                #pragma unroll
                for (int kk = 0; kk < BNT / 16; ++kk)
                    tc_mma_ts(tm + TcSmem<D, BNT>::SOFF, tm + s * BNT + kk * 8, db + kk * adv, idesc_pv,
                              (tl > 0 || kk > 0) ? 1u : 0u);
#endif
                tc_commit(&S.bars[2 + s]);
            }
#ifdef FMHA_PRE_PF2
            // Shift rather than alternate: fixed names keep the buffer index a
            // compile-time constant, which is the whole requirement - `ConvRegs
            // R[2]` indexed by `tl & 1` would put both buffers in local memory.
            // After SROA promotes the statically-indexed arrays this should be a
            // rename and not 34 movs; if the register count comes back ~34 above
            // the control's, it was materialised and the unrolled form in
            // results/prefill-next-levers-20261025.md is the one to build instead.
            // Garbage is copied on the last iteration, where `more` guards the
            // convert off.
            R = R2;
#endif
        }

    } else {
        for (int kn = kn_begin; kn < kn_end; kn += BNT, ++tl, ++tg) {
            const int s = tg & 1;
            const int nvalid = min(BNT, kn_end - kn);
            tc_convert<D, BNT, TC_THREADS, TPRW_E, F8QK>(R, S, s, nvalid, P);
            if (kn + BNT < kn_end) {
                kv_prefetch_m<D, BNT, TC_THREADS, TPRW_E>(R, P, kvh, kn + BNT, min(BNT, kn_end - kn - BNT), bt_n);
                kv_bt_fill<BNT, TC_THREADS, TPRW_E>(P, b, kn + 2 * BNT, kn_end, bt_n);
            }
            tc_fence_async_smem();
            __syncthreads();

            // ---- QK mma (elected thread) ----
            if (tid == 0) {
#ifdef FMHA_PRE_F8QK
                if constexpr (F8QK) {
                    uint64_t db8 = tc_desc(S.k8(s), 2, 0, 1024);
                    #pragma unroll
                    for (int kk = 0; kk < D / 32; ++kk)
                        tc_mma_f8(tm + s * BNT, dQhi + kk * 2, db8 + kk * 2,
                                  idesc_f8, kk > 0 ? 1u : 0u);
                    #pragma unroll
                    for (int kk = 0; kk < D / 32; ++kk)
                        tc_mma_f8(tm + s * BNT, dQlo + kk * 2, db8 + kk * 2,
                                  idesc_f8, 1u);
                } else
#endif
                {
                    uint64_t db = tc_desc(S.k(s), 0, 128, (D / 8) * 128);
                    #pragma unroll
                    for (int kk = 0; kk < D / 16; ++kk)
                        tc_mma_ss(tm + s * BNT, da_q + kk * 16, db + kk * 16, idesc_qk,
                                  kk > 0 ? 1u : 0u);
                }
                tc_commit(&S.bars[s]);
            }
            tc_wait(&S.bars[s], (tg >> 1) & 1);

            // ---- softmax (row per lane) ----
            float sc[BNT];
            {
                uint32_t saddr = tm + s * BNT + ((warp * 32) << 16);
                #pragma unroll
                for (int seg = 0; seg < BNT / 32; ++seg)
                    tc_ld_32(sc + seg * 32, saddr + seg * 32);
                tc_wait_ld();
            }
            const float* kss = S.ks(s);   // pre-scaled by sm_scale*log2e
            const float* vss = S.vs(s);
#ifdef FMHA_PRE_VSMAXP
            vs_keepalive(vs_tile_max<BNT>(vss), sc);   // prices the pass; see the helper's comment
#endif
            // v3.5: 4-way ILP — the serial 64-deep FMAX/FADD chains cost
            // ~256 clk each of pure latency (stalled_wait 0.59/inst).
            float mn0 = m2, mn1 = m2, mn2 = m2, mn3 = m2;
            // merge2-nm: warp-uniform mask-free fast path (NM template gate;
            // persist launches only - see the A/B note at the top of this
            // file). Lane 0 owns the warp's smallest visible (rows ascend
            // with lane); a tile whose last key is <= that masks nothing ->
            // skip 128 ISETP+SEL per tile. Inactive rows (visible=0) take the
            // slow path. Warp-uniform branch: tcgen05/mbarrier ops stay
            // converged. NM=false folds the shfl and the branch away.
            bool nomask = false;
            if (NM) nomask = (kn + BNT - 1 <= __shfl_sync(0xffffffffu, visible, 0));
            if (nomask) {
#if defined(MYPRE_T_NOKSMUL) || defined(MYPRE_KSFOLD)
                // ---- PRICE PROBE (MYPRE_T_NOKSMUL) or the real fold (MYPRE_KSFOLD), wrong numbers by construction. Never shippable; run with --nocorrect. ----
                // Drops the per-key score scale, so the output is wrong by construction. It prices the
                // softmax's own BNT/4 broadcast float4 loads of S.ks plus BNT FMULs - 16 and 64 at BNT=64 -
                // which is the quantity a K-side fold would delete, by multiplying K's f16 image by
                // (R.ks[p] * sm_scale_log2e) inside tc_convert with one mul.f16x2 per pair. tc_convert
                // already holds that exact value on the lane that stores S.ks, and MYPRE_KF16 already makes
                // the K image f16, so the packed multiply exists. The construct being priced is the same one
                // FMHA_PRE_VSMAXP ADDED for the value scale and measured at +1.17%/+0.67% on cases 0/9 (16
                // broadcast float4 loads plus 63 FMAX = 79 instructions against these 80), and the shipped
                // calibration is that instruction count converts at ~0.13 elasticity here, so ~80 of ~720
                // instructions predicts roughly -1.2%/-0.7% before counting the ~32 mul.f16x2 the fold adds
                // back in the producer phase that is already waiting on global latency.
                // Numerics if it were built: K' = round_f16(K_fp8 * ks) costs 2^-12 relative per element, and
                // with independent rounding the dot product's relative error stays ~2^-12 = 2.4e-4, so
                // delta_p/p ~ ln2 * |s| * 2.4e-4 against the current worst rel_l2 of 1.4e-4. Range is safe:
                // |K*ks| <= 448 * key_scale * sm_scale * log2e ~ 16, and the small end goes f16-subnormal
                // where the error is absolute (2^-25) against |s| ~ O(1), i.e. negligible.
                #pragma unroll
                for (int j = 0; j < BNT; j += 4) {
                    float s0 = sc[j];
                    float s1 = sc[j + 1];
                    float s2 = sc[j + 2];
                    float s3 = sc[j + 3];
                    sc[j] = s0; sc[j + 1] = s1; sc[j + 2] = s2; sc[j + 3] = s3;
                    mn0 = fmaxf(mn0, s0); mn1 = fmaxf(mn1, s1);
                    mn2 = fmaxf(mn2, s2); mn3 = fmaxf(mn3, s3);
                }
#elif defined(MYPRE_T_MX3)
                // ---- MYPRE_T_MX3: 3-input row max, BIT-IDENTICAL to the #else below. ----
                // Each of the four accumulators absorbs TWO values instead of one, so the source
                // says fmaxf(mn, fmaxf(a, b)) and ptxas can fuse that into a single FMNMX3
                // (sm_100+; this tree already relies on it - fa_prefill_w8's softmax has been
                // written as `mnA = fmaxf(mnA, fmaxf(a0, a1))` since it was split, so tc is the
                // only prefill kernel still emitting the flat form). BNT FMNMX -> BNT/2 FMNMX3,
                // i.e. 64 -> 32 instructions per thread per tile at BNT=64, ~4.4% of the ~720.
                // The step doubles to 8 so the accumulator count stays four and the chain depth
                // per accumulator halves, 16 -> 8; four independent chains already cover it (the
                // pre-join SASS dump of this loop, artifacts/20261023T110222Z/sass_g_w8_fmha_tc128,
                // shows R13/R15/R17/R19 issuing round-robin with no gap). Eight s values are live
                // at once instead of four: free at 183/255 registers and 2 CTAs/SM, which is
                // TMEM-capped rather than register-capped.
                // fmaxf is exact and, under IEEE NaN-is-identity semantics, associative and
                // commutative, so ANY grouping of the same multiset gives the same bits: this
                // changes the instruction stream and nothing else. That is why the sum below is
                // NOT widened - FADD reassociates and would move bits.
                static_assert(BNT % 8 == 0, "MYPRE_T_MX3 needs BNT divisible by 8");
                #pragma unroll
                for (int j = 0; j < BNT; j += 8) {
                    const float4 ka = *reinterpret_cast<const float4*>(kss + j);
                    const float4 kb = *reinterpret_cast<const float4*>(kss + j + 4);
                    float s0 = sc[j] * ka.x;
                    float s1 = sc[j + 1] * ka.y;
                    float s2 = sc[j + 2] * ka.z;
                    float s3 = sc[j + 3] * ka.w;
                    float s4 = sc[j + 4] * kb.x;
                    float s5 = sc[j + 5] * kb.y;
                    float s6 = sc[j + 6] * kb.z;
                    float s7 = sc[j + 7] * kb.w;
                    sc[j] = s0; sc[j + 1] = s1; sc[j + 2] = s2; sc[j + 3] = s3;
                    sc[j + 4] = s4; sc[j + 5] = s5; sc[j + 6] = s6; sc[j + 7] = s7;
                    mn0 = fmaxf(mn0, fmaxf(s0, s1));
                    mn1 = fmaxf(mn1, fmaxf(s2, s3));
                    mn2 = fmaxf(mn2, fmaxf(s4, s5));
                    mn3 = fmaxf(mn3, fmaxf(s6, s7));
                }
#else
                #pragma unroll
                for (int j = 0; j < BNT; j += 4) {
                    const float4 ks4 = *reinterpret_cast<const float4*>(kss + j);
                    float s0 = sc[j] * ks4.x;
                    float s1 = sc[j + 1] * ks4.y;
                    float s2 = sc[j + 2] * ks4.z;
                    float s3 = sc[j + 3] * ks4.w;
                    sc[j] = s0; sc[j + 1] = s1; sc[j + 2] = s2; sc[j + 3] = s3;
                    mn0 = fmaxf(mn0, s0); mn1 = fmaxf(mn1, s1);
                    mn2 = fmaxf(mn2, s2); mn3 = fmaxf(mn3, s3);
                }
#endif
            } else {
#ifdef MYPRE_T_MX3
                // MYPRE_T_MX3 masked variant - the same 3-input reduction as the nomask branch,
                // with the same per-column NEG selects, now eight of them. Bit-identical: the
                // multiset fed to the max is unchanged, only the grouping is.
                static_assert(BNT % 8 == 0, "MYPRE_T_MX3 needs BNT divisible by 8");
                #pragma unroll
                for (int j = 0; j < BNT; j += 8) {
                    const float4 ka = *reinterpret_cast<const float4*>(kss + j);
                    const float4 kb = *reinterpret_cast<const float4*>(kss + j + 4);
                    float s0 = sc[j] * ka.x;
                    float s1 = sc[j + 1] * ka.y;
                    float s2 = sc[j + 2] * ka.z;
                    float s3 = sc[j + 3] * ka.w;
                    float s4 = sc[j + 4] * kb.x;
                    float s5 = sc[j + 5] * kb.y;
                    float s6 = sc[j + 6] * kb.z;
                    float s7 = sc[j + 7] * kb.w;
                    if (kn + j > visible) s0 = NEG;
                    if (kn + j + 1 > visible) s1 = NEG;
                    if (kn + j + 2 > visible) s2 = NEG;
                    if (kn + j + 3 > visible) s3 = NEG;
                    if (kn + j + 4 > visible) s4 = NEG;
                    if (kn + j + 5 > visible) s5 = NEG;
                    if (kn + j + 6 > visible) s6 = NEG;
                    if (kn + j + 7 > visible) s7 = NEG;
                    sc[j] = s0; sc[j + 1] = s1; sc[j + 2] = s2; sc[j + 3] = s3;
                    sc[j + 4] = s4; sc[j + 5] = s5; sc[j + 6] = s6; sc[j + 7] = s7;
                    mn0 = fmaxf(mn0, fmaxf(s0, s1));
                    mn1 = fmaxf(mn1, fmaxf(s2, s3));
                    mn2 = fmaxf(mn2, fmaxf(s4, s5));
                    mn3 = fmaxf(mn3, fmaxf(s6, s7));
                }
#else
                #pragma unroll
                for (int j = 0; j < BNT; j += 4) {
                    // v3.7: vectorized scale loads (kss is 16B-aligned: ksvs
                    // base is 512B-aligned, j*4 float = 16B steps).
#if defined(MYPRE_T_NOKSMUL) || defined(MYPRE_KSFOLD)
                    float s0 = sc[j];              // PRICE PROBE, wrong numbers; see the nomask note above
                    float s1 = sc[j + 1];
                    float s2 = sc[j + 2];
                    float s3 = sc[j + 3];
#else
                    const float4 ks4 = *reinterpret_cast<const float4*>(kss + j);
                    float s0 = sc[j] * ks4.x;
                    float s1 = sc[j + 1] * ks4.y;
                    float s2 = sc[j + 2] * ks4.z;
                    float s3 = sc[j + 3] * ks4.w;
#endif
                    if (kn + j > visible) s0 = NEG;
                    if (kn + j + 1 > visible) s1 = NEG;
                    if (kn + j + 2 > visible) s2 = NEG;
                    if (kn + j + 3 > visible) s3 = NEG;
                    sc[j] = s0; sc[j + 1] = s1; sc[j + 2] = s2; sc[j + 3] = s3;
                    mn0 = fmaxf(mn0, s0); mn1 = fmaxf(mn1, s1);
                    mn2 = fmaxf(mn2, s2); mn3 = fmaxf(mn3, s3);
                }
#endif
            }
            const float mn = fmaxf(fmaxf(mn0, mn1), fmaxf(mn2, mn3));
#ifdef FMHA_PRE_F8QK
            // F8QK: mn/m2 are device-domain (Q was row-scaled by 2^k); mnt and
            // the fmaf restore the true domain INSIDE the exp2 argument. invr
            // is a power of two, so both multiplies are exact, and fma(x,1,-y)
            // == x-y makes the non-F8QK instantiations bit-identical.
            const float mnt = F8QK ? mn * invr : mn;
            float alpha = F8QK ? ex2f(fmaf(m2, invr, -mnt)) : ex2f(m2 - mn);
#else
            float alpha = ex2f(m2 - mn);
#endif
            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
            unsigned wp[BNT / 2];
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
#ifdef FMHA_PRE_F8QK
                float p0, p1, p2, p3;
                if constexpr (F8QK) {
                    p0 = ex2f(fmaf(sc[j], invr, -mnt));
                    p1 = ex2f(fmaf(sc[j + 1], invr, -mnt));
                    p2 = ex2f(fmaf(sc[j + 2], invr, -mnt));
                    p3 = ex2f(fmaf(sc[j + 3], invr, -mnt));
                } else {
                    p0 = ex2f(sc[j] - mn);
                    p1 = ex2f(sc[j + 1] - mn);
                    p2 = ex2f(sc[j + 2] - mn);
                    p3 = ex2f(sc[j + 3] - mn);
                }
#else
                float p0 = ex2f(sc[j] - mn);
                float p1 = ex2f(sc[j + 1] - mn);
                float p2 = ex2f(sc[j + 2] - mn);
                float p3 = ex2f(sc[j + 3] - mn);
#endif
                sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
                sc[j] = p0; sc[j + 1] = p1;   // reuse sc storage for p
                sc[j + 2] = p2; sc[j + 3] = p3;
            }
            const float sum = (sum0 + sum1) + (sum2 + sum3);
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
#ifdef MYPRE_VSFOLD
                // VSFOLD: V's shared image already carries value_scale, so this loop has neither the scale nor
                // its 16 broadcast float4 loads - 16 LDS.128 and 64 FMUL per thread per tile deleted, which the
                // load->use cost model priced at ~5.1% and ~1.0% respectively.
                wp[j >> 1] = cvt_bf16x2_f32(sc[j], sc[j + 1]);
                wp[(j >> 1) + 1] = cvt_bf16x2_f32(sc[j + 2], sc[j + 3]);
#else
                const float4 vs4 = *reinterpret_cast<const float4*>(vss + j);
#ifdef MYPRE_VF16
                wp[j >> 1] = cvt_f16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
                wp[(j >> 1) + 1] = cvt_f16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
#else
                wp[j >> 1] = cvt_bf16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
                wp[(j >> 1) + 1] = cvt_bf16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
#endif
#endif
            }
            m2 = mn;
            l2 = l2 * alpha + sum;
            {
                uint32_t paddr = tm + s * BNT + ((warp * 32) << 16);
                if constexpr (BNT >= 64) {
                    #pragma unroll
                    for (int seg = 0; seg < BNT / 64; ++seg)
                        tc_st_32(paddr + seg * 32, wp + seg * 32);
                } else {
                    tc_st_16(paddr, wp);   // BNT=32: 16 packed-bf16 cols
                }
                tc_wait_st();
            }
            // merge9-bd (ADOPTED): single sync per seq tile (P-store
            // publication folds into the corr sync below; PV(t-1) wait is
            // per-thread).
            // ---- correction: O *= alpha (needs pv(t-1) done) ----
            if (tl > 0) tc_wait(&S.bars[2 + (s ^ 1)], ((tg - 1) >> 1) & 1);
            {
                bool need = __any_sync(0xffffffffu, alpha != 1.0f);
                if (need) {
                    uint32_t oaddr = tm + TcSmem<D, BNT>::SOFF + ((warp * 32) << 16);
                    #pragma unroll
                    for (int seg = 0; seg < D / 32; ++seg) {
                        float o[32];
                        tc_ld_32(o, oaddr + seg * 32);
                        tc_wait_ld();
                        unsigned w[32];
                        #pragma unroll
                        for (int j = 0; j < 32; ++j) w[j] = __float_as_uint(o[j] * alpha);
                        tc_st_32(oaddr + seg * 32, w);
                    }
                    tc_wait_st();
                }
            }
            tc_fence_before();
            __syncthreads();
            tc_fence_after();

            // ---- PV mma (A from TMEM P) ----
            if (tid == 0) {
                uint64_t db = tc_desc(S.v(s), 2, BNT * 128, 1024);
                const int adv = 128; // K16 = 16 rows * 128B swizzle-atom stride >> 4
                #pragma unroll
                for (int kk = 0; kk < BNT / 16; ++kk)
                    tc_mma_ts(tm + TcSmem<D, BNT>::SOFF, tm + s * BNT + kk * 8, db + kk * adv, idesc_pv,
                              (tl > 0 || kk > 0) ? 1u : 0u);
                tc_commit(&S.bars[2 + s]);
            }
        }
    }

    // ---- epilogue (this half) ----
    if (!SPLIT || any) {
        tc_wait(&S.bars[2 + ((tg - 1) & 1)], ((tg - 1) >> 1) & 1);
        tc_fence_after();
    }
    if (SPLIT && split) {
        // ---- clustered merge: CL partial (m, l, O) over DSMEM ----
        // Staging lives in q+kc+vc, which the wait above has freed: the last PV mma
        // is done, and every QK mma that read q completed before it (tcgen05
        // completions are ordered per CTA). OST pads the row to D+4 floats so the
        // float4 staging stores stay 16 B aligned; that costs a 4-way bank conflict
        // on a path that is ~0.05% of an item.
        constexpr int OST = D + 4;
        float* ost = reinterpret_cast<float*>(smem_raw);
        float* mlp = ost + TCBM * OST;
        static_assert((TCBM * OST + 2 * TCBM) * 4 <= (TCBM * D + 4 * BNT * D) * 2,
                      "merge staging must fit the freed q/kc/vc region");
        {
            uint32_t oaddr = tm + TcSmem<D, BNT>::SOFF + ((warp * 32) << 16);
            #pragma unroll
            for (int seg = 0; seg < D / 32; ++seg) {
                float o[32];
                if (any) {
                    tc_ld_32(o, oaddr + seg * 32);
                    tc_wait_ld();
                } else {
                    #pragma unroll
                    for (int j = 0; j < 32; ++j) o[j] = 0.f;
                }
                #pragma unroll
                for (int j = 0; j < 32; j += 4)
                    *reinterpret_cast<float4*>(ost + row * OST + seg * 32 + j)
                        = make_float4(o[j], o[j + 1], o[j + 2], o[j + 3]);
            }
        }
        mlp[row * 2] = m2;
        mlp[row * 2 + 1] = l2;
#ifdef MYPRE_CLUSTER_DBG
        if (tid == 0 && which == 0 && blockIdx.x < 8)
            printf("CMERGE bx=%u rank=%u kn=[%d,%d) ntid=%d any=%d m2=%.4g l2=%.4g "
                   "ost=%u mlp=%u smembytes=%d\n",
                   (unsigned)blockIdx.x, crank, kn_begin, kn_end,
                   (int)((kn_end - kn_begin + BNT - 1) / BNT), (int)any, m2, l2,
                   smem_u32(ost), smem_u32(mlp), (int)TcSmem<D, BNT>::bytes());
#endif
        cl_sync();
        // rank r writes rows [r*RPC, (r+1)*RPC), reading the other ranks' O for
        // those rows. TPR_M threads share a row, so no thread holds more than
        // CPM = D/TPR_M accumulators (32 at CL=4, 16 at CL=8).
        constexpr int RPC = TCBM / CL;
        constexpr int TPR_M = TC_THREADS / RPC;
        constexpr int CPM = D / TPR_M;
        constexpr int CH = CPM < 32 ? CPM : 32;
        static_assert(RPC * CL == TCBM && TPR_M * RPC == TC_THREADS
                      && CPM * TPR_M == D && CPM % CH == 0 && CH % 4 == 0,
                      "merge row/col split must tile exactly");
        {
            const int rl = tid / TPR_M;
            const int cg = tid % TPR_M;
            const int grow = (int)crank * RPC + rl;
            float mp[CL], lp[CL];
            float mm = NEG;
            #pragma unroll
            for (int pp = 0; pp < CL; ++pp) {
                const uint32_t a = cl_mapa(smem_u32(mlp + grow * 2), (unsigned)pp);
                mp[pp] = cl_ldf(a);
                lp[pp] = cl_ldf(a + 4);
                mm = fmaxf(mm, mp[pp]);
            }
            float ll = 0.f;
            #pragma unroll
            for (int pp = 0; pp < CL; ++pp) ll += lp[pp] * ex2f(mp[pp] - mm);
            const float inv = rcpa(ll);
#ifdef MYPRE_CLUSTER_DBG
            if (tid == 0 && which == 0 && blockIdx.x < 8)
                printf("CMERGE2 bx=%u rank=%u grow=%d nrows=%d mm=%.4g ll=%.4g "
                       "mp0=%.4g lp0=%.4g mpL=%.4g lpL=%.4g obase=%u\n",
                       (unsigned)blockIdx.x, crank, grow, nrows, mm, ll,
                       mp[0], lp[0], mp[CL - 1], lp[CL - 1],
                       cl_mapa(smem_u32(ost + grow * OST + cg * CPM), (unsigned)(CL - 1)));
#endif
            if (grow < nrows) {
                uint16_t* orow =
                    P.out + (long long)(q0 + i0 + grow) * P.out_stride + h * D;
                #pragma unroll
                for (int c0 = 0; c0 < CPM; c0 += CH) {
                    float acc[CH];
                    #pragma unroll
                    for (int j = 0; j < CH; ++j) acc[j] = 0.f;
                    #pragma unroll
                    for (int pp = 0; pp < CL; ++pp) {
                        // an empty slice publishes l == 0; its O was never written,
                        // so skip it rather than multiply 0 by uninitialised TMEM
                        if (lp[pp] == 0.f) continue;
                        const float a = ex2f(mp[pp] - mm) * inv;
                        const uint32_t ob = cl_mapa(
                            smem_u32(ost + grow * OST + cg * CPM + c0), (unsigned)pp);
                        // SCALAR DSMEM reads. ld.shared::cluster.f32 is measured
                        // working at CL=2 and CL=4 (dv_f0108cce089b's CMERGE2 print
                        // shows four ranks exchanging m/l correctly through it);
                        // the ld.shared::cluster.v4.u32 form is the one change
                        // between a CL=2 merge that passes at rel_l2 6.6e-4 and a
                        // CL=4 merge that dies on cudaErrorLaunchFailure with every
                        // address it prints in range and 16 B aligned. The merge is
                        // ~0.05% of an item, so 4x the read instructions cost
                        // nothing measurable and the vector form is not worth the
                        // risk until it is understood.
                        #pragma unroll
                        for (int j = 0; j < CH; ++j)
                            acc[j] += cl_ldf(ob + (unsigned)(j * 4)) * a;
                    }
                    #pragma unroll
                    for (int j = 0; j < CH; j += 2)
                        *reinterpret_cast<unsigned*>(orow + cg * CPM + c0 + j)
                            = cvt_bf16x2_f32(acc[j], acc[j + 1]);
                }
            }
        }
        // no rank may start the next item - which restages into this region - while
        // a peer is still reading it
        cl_sync();
#ifdef MYPRE_CLUSTER_DBG
        if (tid == 0 && which < 3 && blockIdx.x < 8)
            printf("CMERGE3 bx=%u rank=%u which=%d done\n",
                   (unsigned)blockIdx.x, crank, which);
#endif
    } else {
        float inv = rcpa(l2);
        uint16_t* orow = P.out + (long long)(q0 + i0 + row) * P.out_stride + h * D;
        uint32_t oaddr = tm + TcSmem<D, BNT>::SOFF + ((warp * 32) << 16);
        #pragma unroll
        for (int seg = 0; seg < D / 32; ++seg) {
            float o[32];
            tc_ld_32(o, oaddr + seg * 32);
            tc_wait_ld();
            if (row < nrows) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    unsigned pk = cvt_bf16x2_f32(o[2 * j] * inv, o[2 * j + 1] * inv);
                    *reinterpret_cast<unsigned*>(orow + seg * 32 + 2 * j) = pk;
                }
            }
        }
    }
    }   // end driver loop (static / fold / persistent)
    __syncthreads();
    if (warp == 0) tc_dealloc(tm, TcSmem<D, BNT>::ACOLS);
#ifdef MYPRE_CLUSTER_DBG
    if (SPLIT && crank == 0) {
        __syncthreads();
        if (tid == 0) atomicAdd(&dbg_done, 1u);
    }
#endif
}

// ============ W8: 8-warp single-row-tile tc prefill (merge8-w8) ============
// The c0 stall profile (results/ncu-v412): warps_active 11.65% (8 of 64
// slots), issue_active 53%, latency stalls (wait 0.59 + long_sb 0.60 +
// short_sb 0.24 + barrier 0.22) dominating math (0.20) — the SM is
// latency-starved at the TMEM-imposed 2 CTA/SM. W8 keeps the SAME TMEM
// footprint (S 2xBNT + O D = 256 cols) and smem, but runs 256 threads:
// each warp owns 16 rows via the 16-lane tcgen05 shapes (warps 4-7 take the
// +16 lane half of each (warp%4)*32 band, addr | (16<<16)) -> 16 warps/SM.
// Fragment mappings PROBE-VERIFIED on this device (tc_probe5/6,
// results/tc-probe5-mapping-20261023.json, results/tc-probe6-20261023.json):
//   16x256b.xN ld/st: reg r of lane l -> row L = lb + (l>>2) + 8*((r>>1)&1),
//     col C = (l&3)*2 + (r&1) + 8*(r>>2)   (x8: C in 0..63; x4 st: C in 0..31)
//   16x128b.x2 st: reg r' -> row L = lb + (l>>2) + 8*(r'&1),
//     col C = off + (l&3) + 4*(r'>>1)      (8 cols per instruction)
//   Both st shapes are exact bijections (probe6 m0/m2/m4: 0 collisions).
// Softmax: each thread holds 2 rows x 16 keys per x8 chunk; row max/sum
// reduce over the 4-lane quad (shfl_xor 1,2). P-store needs ZERO shuffles:
// packing LD regs (8i..8i+7) as 4x 16x128b.x2 (col off 8i) reproduces the
// ST fragment exactly (derived + checked against probe6 m4 sample).
// Corr/epilogue: x8 chunks with alpha/inv selected by the reg's row bit.
// Register budget target <=128/thread for 2 CTA/SM (est. ~100-115: sc[32]
// peak, no o0/o1 2-seg batch, ConvRegs at NT=256 = 16).
template <int D, int BNT, bool PIPE, bool NM>
__global__ void __launch_bounds__(256) fa_prefill_w8(const __grid_constant__ Params P) {
    constexpr int NT = 256;
    extern __shared__ char smem_raw[];
    TcSmem<D, BNT> S(smem_raw);
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    // ---- work assignment: identical text to fa_prefill_tc (thread-agnostic) ----
    int T_all = 0;
    int work0 = 0;
    int b_s = -1, q0_s = 0, qlen_s = 0;
    int tileA = 0, tileB = -1;
    if (P.persist) {
        int prev = 0;
        for (int i = 0; i < P.batch; ++i) {
            int c = __ldg(P.cu + i + 1);
            T_all += (c - prev + TCBM - 1) / TCBM;
            prev = c;
        }
        T_all *= P.heads;
        if (tid == 0) S.tmem_base[1] = (unsigned)atomicAdd(P.counter, 1);
        __syncthreads();
        work0 = (int)S.tmem_base[1];
        if (work0 >= T_all) return;
    } else {
        int gx = blockIdx.x;
        int base = 0;
        {
            int prev = 0;
            for (int i = 0; i < P.batch; ++i) {
                int c = __ldg(P.cu + i + 1);
                int ntiles = P.fold ? (c - prev + 2 * TCBM - 1) / (2 * TCBM)
                                    : (c - prev + TCBM - 1) / TCBM;
                if (gx < base + ntiles) {
                    b_s = i; tileA = gx - base; q0_s = prev; qlen_s = c - prev; break;
                }
                base += ntiles;
                prev = c;
            }
        }
        if (b_s < 0) return;
        const int nseq_tiles = (qlen_s + TCBM - 1) / TCBM;
        tileB = P.fold ? (nseq_tiles - 1 - tileA) : -1;
        if (tileB == tileA) tileB = -1;
        if (tileA * TCBM >= qlen_s) return;
    }

    // ---- barriers + tmem ----
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(smem_u32(&S.bars[i])));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        tc_alloc(S.tmem_base, TcSmem<D, BNT>::ACOLS);
        tc_relinquish();
    }
    __syncthreads();
    const uint32_t tm = *S.tmem_base;

    const uint64_t da_q = tc_desc(S.q, 0, 128, (D / 8) * 128);
#ifdef MYPRE_KF16
    // kind::f16 idesc a/b format field: 0 = F16, 1 = BF16. Q is staged as f16 and the K
    // image is converted straight to f16, so both operands of QK are f16; the PV idesc stays
    // bf16 because the reference rounds weighted_p to q.dtype = bf16 before the product.
    const uint32_t idesc_qk = tc_idesc(TCBM, BNT, 0, 0, 0);
#else
    const uint32_t idesc_qk = tc_idesc(TCBM, BNT, 0, 1, 1);
#endif
#ifdef MYPRE_VF16
    // kind::f16 for PV as well as QK: atype is P in TMEM, btype is V in shared. The 16-bit
    // packing and the swizzle-128B MN-major V image are unchanged - both types are 2 bytes -
    // so only the descriptor's two type fields move, exactly as MYPRE_KF16 moved idesc_qk's.
    const uint32_t idesc_pv = tc_idesc(TCBM, D, 1, 0, 0);
#else
    const uint32_t idesc_pv = tc_idesc(TCBM, D, 1, 1, 1);
#endif
    // this warp's 16-lane group: band (warp%4)*32, half (warp/4)*16
    const uint32_t lb = (uint32_t)(((warp & 3) * 32 + (warp >> 2) * 16) << 16);
    const int lrow = lane >> 2;          // row within half-band (0..7)
    const int lcol = lane & 3;           // column-quad index (0..3)
    const int rowA = (warp & 3) * 32 + (warp >> 2) * 16 + lrow;   // low row
    int tg = 0;                          // GLOBAL kv-tile counter (parity)
    for (int which = 0; ; ++which) {
    int b, q0, qlen, tile, h;
    if (P.persist) {
        if (which > 0) {
            if (tid == 0) S.tmem_base[1] = (unsigned)atomicAdd(P.counter, 1);
            __syncthreads();
            work0 = (int)S.tmem_base[1];
        }
        if (work0 >= T_all) break;
        const int work = T_all - 1 - work0;
        h = work % P.heads;
        const int rem = work / P.heads;
        b = -1;
        int base2 = 0, prev2 = 0;
        for (int i = 0; i < P.batch; ++i) {
            int c = __ldg(P.cu + i + 1);
            int ntiles = (c - prev2 + TCBM - 1) / TCBM;
            if (rem < base2 + ntiles) {
                b = i; tile = rem - base2; q0 = prev2; qlen = c - prev2; break;
            }
            base2 += ntiles;
            prev2 = c;
        }
        if (b < 0) break;
    } else {
        if (which > 0 && (!P.fold || which > 1)) break;
        b = b_s; q0 = q0_s; qlen = qlen_s; h = blockIdx.y;
        tile = (which == 0) ? tileA : tileB;
        if (tile < 0) break;
    }
    const int kvh = h / P.g;
    const int kvlen = __ldg(P.seqk + b);
    const int i0 = tile * TCBM;
    const int nrows = min(TCBM, qlen - i0);
    // ---- Q staging (NT=256; validated mapping, v4.12 text) ----
    {
        constexpr int PER_ROW = D / 8;
        char* qb = reinterpret_cast<char*>(S.q);
        for (int idx = tid; idx < TCBM * PER_ROW; idx += NT) {
            int row = idx / PER_ROW;
            int col8 = (idx % PER_ROW) * 8;
            uint4* d = reinterpret_cast<uint4*>(qb + TcSmem<D, BNT>::km_off(row, col8));
            if (row < nrows) {
                const uint16_t* src = P.q + (long long)(q0 + i0 + row) * P.q_out_stride
                                    + h * D + col8;
#ifdef MYPRE_KF16
                *d = bf16x8_to_f16x8(*reinterpret_cast<const uint4*>(src));
#else
                *d = *reinterpret_cast<const uint4*>(src);
#endif
            } else {
                *d = make_uint4(0, 0, 0, 0);
            }
        }
    }

    // this thread's two rows and their visible bounds
    const int row0 = rowA;               // low row of the thread's pair
    const int row1 = rowA + 8;           // high row (row bit in fragments)
    const int vis0 = (row0 < nrows) ? (kvlen - qlen + i0 + row0) : 0;
    const int vis1 = (row1 < nrows) ? (kvlen - qlen + i0 + row1) : 0;
    const int kn_end = min(kvlen, kvlen - qlen + i0 + TCBM);

    ConvRegs<D, BNT, NT> R;
    int bt_n = kv_bt_next<BNT, NT>(P, b, 0, kn_end);
    kv_prefetch<D, BNT, NT>(R, P, kvh, 0, min(BNT, kn_end), bt_n);
    if (BNT < kn_end) bt_n = kv_bt_next<BNT, NT>(P, b, BNT, kn_end);
    if constexpr (PIPE) {
        // prologue (EVPRO parity-branched, merge4-validated semantics)
        if ((tg & 1) == 0) {
            tc_convert<D, BNT, NT>(R, S, 0, min(BNT, kn_end), P);
            tc_fence_async_smem();
            __syncthreads();
            if (tid == 0) {
                uint64_t db = tc_desc(S.k(0), 0, 128, (D / 8) * 128);
                #pragma unroll
                for (int kk = 0; kk < D / 16; ++kk)
                    tc_mma_ss(tm, da_q + kk * 16, db + kk * 16, idesc_qk,
                              kk > 0 ? 1u : 0u);
                tc_commit(&S.bars[0]);
            }
        } else {
            tc_convert<D, BNT, NT>(R, S, 1, min(BNT, kn_end), P);
            tc_fence_async_smem();
            __syncthreads();
            if (tid == 0) {
                uint64_t db = tc_desc(S.k(1), 0, 128, (D / 8) * 128);
                #pragma unroll
                for (int kk = 0; kk < D / 16; ++kk)
                    tc_mma_ss(tm + BNT, da_q + kk * 16, db + kk * 16, idesc_qk,
                              kk > 0 ? 1u : 0u);
                tc_commit(&S.bars[1]);
            }
        }
    }

    float m20 = NEG, m21 = NEG, l20 = 0.f, l21 = 0.f;   // rows row0 / row1
    int tl = 0;
    if constexpr (PIPE) {
        for (int kn = 0; kn < kn_end; kn += BNT, ++tl, ++tg) {
            const int s = tg & 1;
            const bool more = (kn + BNT < kn_end);
            if (more) {
                kv_prefetch<D, BNT, NT>(R, P, kvh, kn + BNT, min(BNT, kn_end - kn - BNT), bt_n);
                bt_n = kv_bt_next<BNT, NT>(P, b, kn + 2 * BNT, kn_end);
            }
            tc_wait(&S.bars[s], (tg >> 1) & 1);

            // ---- softmax W8: 16x256b.x8 ld, quad-reduced max/sum ----
            float sc[BNT / 2];
            tc_ld16_x8(reinterpret_cast<unsigned*>(sc), tm + s * BNT + lb);
            if (BNT > 64)   // cols 64..BNT via the probe-verified x4 map
                tc_ld16_x4(reinterpret_cast<unsigned*>(sc) + 32, tm + s * BNT + 64 + lb);
            tc_wait_ld();
            const float* kss = S.ks(s);
            const float* vss = S.vs(s);
#ifdef FMHA_PRE_VSMAXP
            vs_keepalive(vs_tile_max<BNT>(vss), sc);   // prices the pass; see the helper's comment
#endif
            // scale + mask; reg r = 4k + 2*bh + bl -> row bh, col 2*lcol+bl+8k
            float mnA = m20, mnB = m21;
            bool nomask = false;
            if (NM) nomask = (kn + BNT - 1 <= __shfl_sync(0xffffffffu, vis0, 0));
            #pragma unroll
            for (int k = 0; k < BNT / 8; ++k) {
                const int c0 = 2 * lcol + 8 * k;
#ifdef MYPRE_KSFOLD
                // The K image already carries ks (tc_convert is SHARED with the tc kernel), so this site must
                // NOT scale again - double-scaling cases 1 and 7 would be wrong numbers, not a slow path.
                float a0 = sc[4 * k + 0];            // row0, col c0
                float a1 = sc[4 * k + 1];            // row0, col c0+1
                float b0 = sc[4 * k + 2];            // row1, col c0
                float b1 = sc[4 * k + 3];            // row1, col c0+1
#else
                const float2 ks2a = *reinterpret_cast<const float2*>(kss + c0);
                float a0 = sc[4 * k + 0] * ks2a.x;   // row0, col c0
                float a1 = sc[4 * k + 1] * ks2a.y;   // row0, col c0+1
                float b0 = sc[4 * k + 2] * ks2a.x;   // row1, col c0
                float b1 = sc[4 * k + 3] * ks2a.y;   // row1, col c0+1
#endif
                if (!nomask) {
                    if (kn + c0 > vis0) a0 = NEG;
                    if (kn + c0 + 1 > vis0) a1 = NEG;
                    if (kn + c0 > vis1) b0 = NEG;
                    if (kn + c0 + 1 > vis1) b1 = NEG;
                }
                sc[4 * k + 0] = a0; sc[4 * k + 1] = a1;
                sc[4 * k + 2] = b0; sc[4 * k + 3] = b1;
                mnA = fmaxf(mnA, fmaxf(a0, a1));
                mnB = fmaxf(mnB, fmaxf(b0, b1));
            }
            // quad reduce (4 lanes per row)
            mnA = fmaxf(mnA, __shfl_xor_sync(0xffffffffu, mnA, 1));
            mnA = fmaxf(mnA, __shfl_xor_sync(0xffffffffu, mnA, 2));
            mnB = fmaxf(mnB, __shfl_xor_sync(0xffffffffu, mnB, 1));
            mnB = fmaxf(mnB, __shfl_xor_sync(0xffffffffu, mnB, 2));
            const float alpha0 = ex2f(m20 - mnA);
            const float alpha1 = ex2f(m21 - mnB);
            float sA0 = 0.f, sA1 = 0.f, sB0 = 0.f, sB1 = 0.f;
            #pragma unroll
            for (int k = 0; k < BNT / 8; ++k) {
                float p0 = ex2f(sc[4 * k + 0] - mnA);
                float p1 = ex2f(sc[4 * k + 1] - mnA);
                float q0 = ex2f(sc[4 * k + 2] - mnB);
                float q1 = ex2f(sc[4 * k + 3] - mnB);
                sA0 += p0; sA1 += p1; sB0 += q0; sB1 += q1;
                sc[4 * k + 0] = p0; sc[4 * k + 1] = p1;
                sc[4 * k + 2] = q0; sc[4 * k + 3] = q1;
            }
            float sumA = sA0 + sA1, sumB = sB0 + sB1;
            sumA += __shfl_xor_sync(0xffffffffu, sumA, 1);
            sumA += __shfl_xor_sync(0xffffffffu, sumA, 2);
            sumB += __shfl_xor_sync(0xffffffffu, sumB, 1);
            sumB += __shfl_xor_sync(0xffffffffu, sumB, 2);
            // P = bf16(p * vs), stored via 4x 16x128b.x2 (zero shuffles:
            // ST reg order = [row0 col g+8i, row1 col g+8i, row0 col g+8i+4,
            // row1 col g+8i+4] <- LD regs (8i,8i+1),(8i+2,8i+3),(8i+4,8i+5),
            // (8i+6,8i+7); vs cols 2g+16i / 2g+16i+8 pairs)
            {
                uint32_t paddr = tm + s * BNT + lb;
                #pragma unroll
                for (int i = 0; i < BNT / 16; ++i) {
                    unsigned w[4];
#ifdef MYPRE_VSFOLD
                    // VSFOLD, W8 form: the same deletion as the tc pack loop, 12 lane-varying float2 loads and
                    // 48 FMUL at BNT=96. tc_convert is shared, so W8's V image is pre-scaled too and leaving
                    // this multiply in place would DOUBLE-SCALE cases 1 and 7.
                    w[0] = cvt_bf16x2_f32(sc[8 * i + 0], sc[8 * i + 1]);
                    w[1] = cvt_bf16x2_f32(sc[8 * i + 2], sc[8 * i + 3]);
                    w[2] = cvt_bf16x2_f32(sc[8 * i + 4], sc[8 * i + 5]);
                    w[3] = cvt_bf16x2_f32(sc[8 * i + 6], sc[8 * i + 7]);
#else
                    const float2 v0 = *reinterpret_cast<const float2*>(vss + 2 * lcol + 16 * i);
                    const float2 v1 = *reinterpret_cast<const float2*>(vss + 2 * lcol + 16 * i + 8);
#ifdef MYPRE_VF16
                    w[0] = cvt_f16x2_f32(sc[8 * i + 0] * v0.x, sc[8 * i + 1] * v0.y);
                    w[1] = cvt_f16x2_f32(sc[8 * i + 2] * v0.x, sc[8 * i + 3] * v0.y);
                    w[2] = cvt_f16x2_f32(sc[8 * i + 4] * v1.x, sc[8 * i + 5] * v1.y);
                    w[3] = cvt_f16x2_f32(sc[8 * i + 6] * v1.x, sc[8 * i + 7] * v1.y);
#else
                    w[0] = cvt_bf16x2_f32(sc[8 * i + 0] * v0.x, sc[8 * i + 1] * v0.y);
                    w[1] = cvt_bf16x2_f32(sc[8 * i + 2] * v0.x, sc[8 * i + 3] * v0.y);
                    w[2] = cvt_bf16x2_f32(sc[8 * i + 4] * v1.x, sc[8 * i + 5] * v1.y);
                    w[3] = cvt_bf16x2_f32(sc[8 * i + 6] * v1.x, sc[8 * i + 7] * v1.y);
#endif
#endif
                    tc_st16_128b_x2(paddr + 8 * i, w);
                }
                tc_wait_st();
            }
            m20 = mnA; m21 = mnB;
            l20 = l20 * alpha0 + sumA;
            l21 = l21 * alpha1 + sumB;
#ifdef FMHA_PRE_NEEDCTA_W8
            const bool need_w = __any_sync(0xffffffffu,
                                           alpha0 != 1.0f || alpha1 != 1.0f);
            if (lane == 0) S.needf[s][warp] = need_w ? 1 : 0;
#endif
            // j23-w8bd (ADOPTED; independent replication on the unified tree,
            // receipt 20261023T083343Z: c1 0.1244->0.1172 = +6.1%, canaries
            // 1.0009/1.0003/0.9974 flat, fold probes green; prior arm-b A/B
            // dv_0cc07abe0c94 measured +6.7%): the BD barrier drop inside the
            // W8 fold kernel. P-store (TMEM) vs convert(t+1) (smem[s^1])
            // disjoint; PV(t-1) wait per-thread; publication rides the
            // remaining syncs' fence_before; last tile drops both syncs. The
            // 8-warp kernel gains ~7x the 4-warp kernel's +0.86% — syncthreads
            // cost/stragglers scale with warp count.
            if (tl > 0) tc_wait(&S.bars[2 + (s ^ 1)], ((tg - 1) >> 1) & 1);
            if (more) {
                tc_convert<D, BNT, NT>(R, S, s ^ 1, min(BNT, kn_end - kn - BNT), P);
                tc_fence_async_smem();
                tc_fence_before();
#ifdef FMHA_PRE_NBSYNC_W8D128
                // D128-gated form (dv_538ce79974a7 measured the ungated arm: -1.9% c1 /
                // -6.9% w1 on W8-D128 band persist but +9.3% on c7's W8-D64-BNT96 kernel,
                // 3/3 reps each, all bit-identical). The BNT96 kernel keeps the shipped
                // __syncthreads(); its per-tile overhead is already 1/3 smaller and the
                // run-ahead warps collide with its denser staging instead of hiding it.
                if constexpr (D == 128) {
                    if (warp == 0) { nb_sync<1, NT>(); tc_fence_after(); }
                    else           { nb_arrive<1, NT>(); }
                } else {
                    __syncthreads();
                    tc_fence_after();
                }
#elif defined(FMHA_PRE_NBSYNC_W8)
                // Only warp 0 issues the MMAs, so only warp 0 needs the convert published;
                // the other seven arrive and run ahead into the next tile's prefetch.
                if (warp == 0) { nb_sync<1, NT>(); tc_fence_after(); }
                else           { nb_arrive<1, NT>(); }
#else
                __syncthreads();
                tc_fence_after();
#endif
            }
#ifdef FMHA_PRE_NEEDCTA_W8
            // Same argument as the 4-warp kernel, and this kernel's own j23-w8bd note
            // already relies on it: "publication rides the remaining syncs'
            // fence_before". If no warp rescaled O, step 7's barrier and both tcgen05
            // fences publish nothing - P(t) rode the convert sync above and O was not
            // touched. On the last tile there is no convert sync, so the barrier must
            // stay; hence the conservative default.
            //
            // MEASURED AND REJECTED for this kernel (dv_9e8043fc9eac): case 1 1.0003,
            // case 7 1.0133 median, while the same job reproduced the tc gains at
            // 0.9860/0.9840 and both decode nulls at exactly 1.0000. j23-w8bd's 7x
            // scaling was about the cost of a barrier, but it said nothing about how
            // often the gate FIRES: both W8 cases here are diagonal (q ~= kv), where
            // nearly every tile moves some row's maximum, so need_cta is almost always
            // true and only the gate's own overhead is left. The two cases where it
            // pays are wide (kv >> q), and those run the 4-warp kernel. Opt-in only.
            bool need_cta = true;
            if (more) {
                int acc = 0;
                #pragma unroll
                for (int w = 0; w < NT / 32; ++w) acc |= S.needf[s][w];
                need_cta = (acc != 0);
            }
#endif
            // 6) QK(t+1) before correction
            if (more && tid == 0) {
                uint64_t dbk = tc_desc(S.k(s ^ 1), 0, 128, (D / 8) * 128);
                #pragma unroll
                for (int kk = 0; kk < D / 16; ++kk)
                    tc_mma_ss(tm + (s ^ 1) * BNT, da_q + kk * 16, dbk + kk * 16,
                              idesc_qk, kk > 0 ? 1u : 0u);
                tc_commit(&S.bars[s ^ 1]);
            }
            // 7) correction: O *= alpha (W8: x8 ld + 2x x4 st per 64-col chunk)
            {
#ifdef FMHA_PRE_NEEDCTA_W8
                const bool need = need_w;      // already voted, before the P store
#else
                bool need = __any_sync(0xffffffffu, alpha0 != 1.0f || alpha1 != 1.0f);
#endif
                if (need) {
                    uint32_t oaddr = tm + TcSmem<D, BNT>::SOFF + lb;
                    #pragma unroll
                    for (int ch = 0; ch < D / 64; ++ch) {
                        float o[32];
                        tc_ld16_x8(reinterpret_cast<unsigned*>(o), oaddr + ch * 64);
                        tc_wait_ld();
                        unsigned w[32];
                        #pragma unroll
                        for (int r = 0; r < 32; ++r) {
                            const float al = ((r >> 1) & 1) ? alpha1 : alpha0;
                            w[r] = __float_as_uint(o[r] * al);
                        }
                        tc_st16_x4(oaddr + ch * 64, w);
                        tc_st16_x4(oaddr + ch * 64 + 32, w + 16);
                    }
                    tc_wait_st();
                }
            }
#ifdef FMHA_PRE_NEEDCTA_W8
            if (need_cta)
#endif
            {
            tc_fence_before();
#ifdef FMHA_PRE_NBSYNC_W8D128
            if constexpr (D == 128) {
                if (warp == 0) { nb_sync<2, NT>(); tc_fence_after(); }
                else           { nb_arrive<2, NT>(); }
            } else {
                __syncthreads();
                tc_fence_after();
            }
#elif defined(FMHA_PRE_NBSYNC_W8)
            if (warp == 0) { nb_sync<2, NT>(); tc_fence_after(); }
            else           { nb_arrive<2, NT>(); }
#else
            __syncthreads();
            tc_fence_after();
#endif
            }

            // 8) PV(t)
            if (tid == 0) {
                uint64_t db = tc_desc(S.v(s), 2, BNT * 128, 1024);
                const int adv = 128;
                #pragma unroll
                for (int kk = 0; kk < BNT / 16; ++kk)
                    tc_mma_ts(tm + TcSmem<D, BNT>::SOFF, tm + s * BNT + kk * 8,
                              db + kk * adv, idesc_pv, (tl > 0 || kk > 0) ? 1u : 0u);
                tc_commit(&S.bars[2 + s]);
            }
        }
    } else {
        // sequential W8 loop (fallback instantiation; same phases, in-iteration
        // commit+wait => parity-safe for any tg start)
        for (int kn = 0; kn < kn_end; kn += BNT, ++tl, ++tg) {
            const int s = tg & 1;
            const int nvalid = min(BNT, kn_end - kn);
            tc_convert<D, BNT, NT>(R, S, s, nvalid, P);
            if (kn + BNT < kn_end) {
                kv_prefetch<D, BNT, NT>(R, P, kvh, kn + BNT, min(BNT, kn_end - kn - BNT), bt_n);
                bt_n = kv_bt_next<BNT, NT>(P, b, kn + 2 * BNT, kn_end);
            }
            tc_fence_async_smem();
            __syncthreads();
            if (tid == 0) {
                uint64_t db = tc_desc(S.k(s), 0, 128, (D / 8) * 128);
                #pragma unroll
                for (int kk = 0; kk < D / 16; ++kk)
                    tc_mma_ss(tm + s * BNT, da_q + kk * 16, db + kk * 16, idesc_qk,
                              kk > 0 ? 1u : 0u);
                tc_commit(&S.bars[s]);
            }
            tc_wait(&S.bars[s], (tg >> 1) & 1);
            float sc[BNT / 2];
            tc_ld16_x8(reinterpret_cast<unsigned*>(sc), tm + s * BNT + lb);
            if (BNT > 64)   // cols 64..BNT via the probe-verified x4 map
                tc_ld16_x4(reinterpret_cast<unsigned*>(sc) + 32, tm + s * BNT + 64 + lb);
            tc_wait_ld();
            const float* kss = S.ks(s);
            const float* vss = S.vs(s);
#ifdef FMHA_PRE_VSMAXP
            vs_keepalive(vs_tile_max<BNT>(vss), sc);   // prices the pass; see the helper's comment
#endif
            float mnA = m20, mnB = m21;
            bool nomask = false;
            if (NM) nomask = (kn + BNT - 1 <= __shfl_sync(0xffffffffu, vis0, 0));
            #pragma unroll
            for (int k = 0; k < BNT / 8; ++k) {
                const int c0 = 2 * lcol + 8 * k;
#ifdef MYPRE_KSFOLD
                // The K image already carries ks (tc_convert is SHARED with the tc kernel), so this site must
                // NOT scale again - double-scaling cases 1 and 7 would be wrong numbers, not a slow path.
                float a0 = sc[4 * k + 0];
                float a1 = sc[4 * k + 1];
                float b0 = sc[4 * k + 2];
                float b1 = sc[4 * k + 3];
#else
                const float2 ks2a = *reinterpret_cast<const float2*>(kss + c0);
                float a0 = sc[4 * k + 0] * ks2a.x;
                float a1 = sc[4 * k + 1] * ks2a.y;
                float b0 = sc[4 * k + 2] * ks2a.x;
                float b1 = sc[4 * k + 3] * ks2a.y;
#endif
                if (!nomask) {
                    if (kn + c0 > vis0) a0 = NEG;
                    if (kn + c0 + 1 > vis0) a1 = NEG;
                    if (kn + c0 > vis1) b0 = NEG;
                    if (kn + c0 + 1 > vis1) b1 = NEG;
                }
                sc[4 * k + 0] = a0; sc[4 * k + 1] = a1;
                sc[4 * k + 2] = b0; sc[4 * k + 3] = b1;
                mnA = fmaxf(mnA, fmaxf(a0, a1));
                mnB = fmaxf(mnB, fmaxf(b0, b1));
            }
            mnA = fmaxf(mnA, __shfl_xor_sync(0xffffffffu, mnA, 1));
            mnA = fmaxf(mnA, __shfl_xor_sync(0xffffffffu, mnA, 2));
            mnB = fmaxf(mnB, __shfl_xor_sync(0xffffffffu, mnB, 1));
            mnB = fmaxf(mnB, __shfl_xor_sync(0xffffffffu, mnB, 2));
            const float alpha0 = ex2f(m20 - mnA);
            const float alpha1 = ex2f(m21 - mnB);
            float sA0 = 0.f, sA1 = 0.f, sB0 = 0.f, sB1 = 0.f;
            #pragma unroll
            for (int k = 0; k < BNT / 8; ++k) {
                float p0 = ex2f(sc[4 * k + 0] - mnA);
                float p1 = ex2f(sc[4 * k + 1] - mnA);
                float q0 = ex2f(sc[4 * k + 2] - mnB);
                float q1 = ex2f(sc[4 * k + 3] - mnB);
                sA0 += p0; sA1 += p1; sB0 += q0; sB1 += q1;
                sc[4 * k + 0] = p0; sc[4 * k + 1] = p1;
                sc[4 * k + 2] = q0; sc[4 * k + 3] = q1;
            }
            float sumA = sA0 + sA1, sumB = sB0 + sB1;
            sumA += __shfl_xor_sync(0xffffffffu, sumA, 1);
            sumA += __shfl_xor_sync(0xffffffffu, sumA, 2);
            sumB += __shfl_xor_sync(0xffffffffu, sumB, 1);
            sumB += __shfl_xor_sync(0xffffffffu, sumB, 2);
            {
                uint32_t paddr = tm + s * BNT + lb;
                #pragma unroll
                for (int i = 0; i < BNT / 16; ++i) {
                    unsigned w[4];
#ifdef MYPRE_VSFOLD
                    // VSFOLD, W8 form: the same deletion as the tc pack loop, 12 lane-varying float2 loads and
                    // 48 FMUL at BNT=96. tc_convert is shared, so W8's V image is pre-scaled too and leaving
                    // this multiply in place would DOUBLE-SCALE cases 1 and 7.
                    w[0] = cvt_bf16x2_f32(sc[8 * i + 0], sc[8 * i + 1]);
                    w[1] = cvt_bf16x2_f32(sc[8 * i + 2], sc[8 * i + 3]);
                    w[2] = cvt_bf16x2_f32(sc[8 * i + 4], sc[8 * i + 5]);
                    w[3] = cvt_bf16x2_f32(sc[8 * i + 6], sc[8 * i + 7]);
#else
                    const float2 v0 = *reinterpret_cast<const float2*>(vss + 2 * lcol + 16 * i);
                    const float2 v1 = *reinterpret_cast<const float2*>(vss + 2 * lcol + 16 * i + 8);
#ifdef MYPRE_VF16
                    w[0] = cvt_f16x2_f32(sc[8 * i + 0] * v0.x, sc[8 * i + 1] * v0.y);
                    w[1] = cvt_f16x2_f32(sc[8 * i + 2] * v0.x, sc[8 * i + 3] * v0.y);
                    w[2] = cvt_f16x2_f32(sc[8 * i + 4] * v1.x, sc[8 * i + 5] * v1.y);
                    w[3] = cvt_f16x2_f32(sc[8 * i + 6] * v1.x, sc[8 * i + 7] * v1.y);
#else
                    w[0] = cvt_bf16x2_f32(sc[8 * i + 0] * v0.x, sc[8 * i + 1] * v0.y);
                    w[1] = cvt_bf16x2_f32(sc[8 * i + 2] * v0.x, sc[8 * i + 3] * v0.y);
                    w[2] = cvt_bf16x2_f32(sc[8 * i + 4] * v1.x, sc[8 * i + 5] * v1.y);
                    w[3] = cvt_bf16x2_f32(sc[8 * i + 6] * v1.x, sc[8 * i + 7] * v1.y);
#endif
#endif
                    tc_st16_128b_x2(paddr + 8 * i, w);
                }
                tc_wait_st();
            }
            m20 = mnA; m21 = mnB;
            l20 = l20 * alpha0 + sumA;
            l21 = l21 * alpha1 + sumB;
            tc_fence_before();
            __syncthreads();
            tc_fence_after();
            if (tl > 0) tc_wait(&S.bars[2 + (s ^ 1)], ((tg - 1) >> 1) & 1);
            {
                bool need = __any_sync(0xffffffffu, alpha0 != 1.0f || alpha1 != 1.0f);
                if (need) {
                    uint32_t oaddr = tm + TcSmem<D, BNT>::SOFF + lb;
                    #pragma unroll
                    for (int ch = 0; ch < D / 64; ++ch) {
                        float o[32];
                        tc_ld16_x8(reinterpret_cast<unsigned*>(o), oaddr + ch * 64);
                        tc_wait_ld();
                        unsigned w[32];
                        #pragma unroll
                        for (int r = 0; r < 32; ++r) {
                            const float al = ((r >> 1) & 1) ? alpha1 : alpha0;
                            w[r] = __float_as_uint(o[r] * al);
                        }
                        tc_st16_x4(oaddr + ch * 64, w);
                        tc_st16_x4(oaddr + ch * 64 + 32, w + 16);
                    }
                    tc_wait_st();
                }
            }
            tc_fence_before();
            __syncthreads();
            tc_fence_after();
            if (tid == 0) {
                uint64_t db = tc_desc(S.v(s), 2, BNT * 128, 1024);
                const int adv = 128;
                #pragma unroll
                for (int kk = 0; kk < BNT / 16; ++kk)
                    tc_mma_ts(tm + TcSmem<D, BNT>::SOFF, tm + s * BNT + kk * 8,
                              db + kk * adv, idesc_pv, (tl > 0 || kk > 0) ? 1u : 0u);
                tc_commit(&S.bars[2 + s]);
            }
        }
    }

    // ---- epilogue (W8: two rows per thread) ----
    tc_wait(&S.bars[2 + ((tg - 1) & 1)], ((tg - 1) >> 1) & 1);
    tc_fence_after();
    {
        const float inv0 = rcpa(l20);
        const float inv1 = rcpa(l21);
        uint16_t* orow0 = P.out + (long long)(q0 + i0 + row0) * P.out_stride + h * D;
        uint16_t* orow1 = P.out + (long long)(q0 + i0 + row1) * P.out_stride + h * D;
        const bool wr0 = row0 < nrows, wr1 = row1 < nrows;
        uint32_t oaddr = tm + TcSmem<D, BNT>::SOFF + lb;
        #pragma unroll
        for (int ch = 0; ch < D / 64; ++ch) {
            float o[32];
            tc_ld16_x8(reinterpret_cast<unsigned*>(o), oaddr + ch * 64);
            tc_wait_ld();
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                // regs 4k..4k+3: (row0 c,c+1), (row1 c,c+1), c = 2*lcol+8k
                const int c = ch * 64 + 2 * lcol + 8 * k;
                if (wr0) {
                    unsigned pk = cvt_bf16x2_f32(o[4 * k + 0] * inv0, o[4 * k + 1] * inv0);
                    *reinterpret_cast<unsigned*>(orow0 + c) = pk;
                }
                if (wr1) {
                    unsigned pk = cvt_bf16x2_f32(o[4 * k + 2] * inv1, o[4 * k + 3] * inv1);
                    *reinterpret_cast<unsigned*>(orow1 + c) = pk;
                }
            }
        }
    }
    }   // end driver loop
    __syncthreads();
    if (warp == 0) tc_dealloc(tm, TcSmem<D, BNT>::ACOLS);
}

// ============ RT2-persist: dual-row-tile tc prefill (D=128, merge7-rt2) ============
// CLEAN-ROOM rebuild on the validated live-kernel skeleton (persist driver +
// seq phase order + validated softmax/convert/epilogue texts). The dormant
// v4.0 fa_prefill_tc_rt2 failed BOTH its D128 corr configs (check-v40rt2:
// prefill_mqa_d128 55118/84 mism rel_l2 0.636; pf_d128_tiny rel_l2 0.354)
// and its bug was never localized — that text is deliberately NOT reused.
// Concept: 256 threads / 8 warps / two 128-row groups per CTA; K/V convert +
// prefetch + bt-ahead are SHARED across the two row tiles, halving the
// per-row instruction cost of the memory path on an issue-bound kernel
// (~87% of the per-tile budget is non-mma). TMEM: S_A/S_B (2 stages x BNT)
// + O_A/O_B = 4*BNT + 2*D = 512 cols -> exactly 1 CTA/SM (8 warps/SM = the
// same warp pool as 2 single-tile CTAs). PERSIST-ONLY: v4.0's static grid
// halving caused the c9 0.504x wave-quantization loss; the ticket puller
// (slots = sms, heavy-first, heads-innermost) keeps balance. Global tg
// (stage/parity, merge3-pf-safe), local tl (PV accumulate). NM=false for
// now (nomask is a separate A/B once the base is green).
template <int D, int BNT>
struct TcSmem2 {
    uint16_t* q; uint16_t* kc; uint16_t* vc; float* ksvs; float* alpha;
    uint64_t* bars; uint32_t* tmem_base;
    DEVI TcSmem2(char* base) {
        q = reinterpret_cast<uint16_t*>(base);
        kc = q + 2 * TCBM * D;
        vc = kc + 2 * BNT * D;
        ksvs = reinterpret_cast<float*>(vc + 2 * BNT * D);
        alpha = ksvs + 4 * BNT;
        bars = reinterpret_cast<uint64_t*>(alpha + 2 * TCBM);
        tmem_base = reinterpret_cast<uint32_t*>(bars + 4);
    }
    DEVI uint16_t* k(int s) const { return kc + s * BNT * D; }
    DEVI uint16_t* v(int s) const { return vc + s * BNT * D; }
    DEVI float* ks(int s) const { return ksvs + s * 2 * BNT; }
    DEVI float* vs(int s) const { return ksvs + s * 2 * BNT + BNT; }
    static constexpr int bytes() {
        return (2 * TCBM * D + 4 * BNT * D) * 2 + 4 * BNT * 4 + 2 * TCBM * 4
             + 4 * 8 + 8 + 128;
    }
    static constexpr int SOFF = 4 * BNT;        // O_A col base
    static constexpr int ACOLS = 512;
    static_assert(4 * BNT + 2 * D <= ACOLS, "RT2 TMEM overflow");
    static DEVI long long km_off(int m, int k) {   // within one 128-row image
        return (m >> 3) * ((D / 8) * 128) + (k >> 3) * 128 + (m & 7) * 16 + (k & 7) * 2;
    }
    static DEVI long long v_off(int krow, int n) {
        return (n >> 6) * (BNT * 128) + (krow >> 3) * 1024 + (krow & 7) * 128
             + ((((n & 63) >> 3) ^ (krow & 7)) * 16) + (n & 7) * 2;
    }
};

template <int D, int BNT>
__global__ void __launch_bounds__(256) fa_prefill_rt2p(const __grid_constant__ Params P) {
    constexpr int NT = 256;
    constexpr int RT = 2 * TCBM;               // 256 q rows per CTA item
    extern __shared__ char smem_raw[];
    TcSmem2<D, BNT> S(smem_raw);
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int grp = warp >> 2;                 // row-tile group (0/1)
    const int w3 = warp & 3;                   // warp within group

    // ---- persist ticket (validated live-kernel mechanism; ONE atomic/CTA) ----
    int T_all = 0;
    {
        int prev = 0;
        for (int i = 0; i < P.batch; ++i) {
            int c = __ldg(P.cu + i + 1);
            T_all += (c - prev + RT - 1) / RT;
            prev = c;
        }
    }
    T_all *= P.heads;
    if (tid == 0) S.tmem_base[1] = (unsigned)atomicAdd(P.counter, 1);
    __syncthreads();
    int work0 = (int)S.tmem_base[1];
    if (work0 >= T_all) return;

    // ---- barriers + tmem ----
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(smem_u32(&S.bars[i])));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        tc_alloc(S.tmem_base, TcSmem2<D, BNT>::ACOLS);
        tc_relinquish();
    }
    __syncthreads();
    const uint32_t tm = *S.tmem_base;

    const uint64_t da_qA = tc_desc(S.q, 0, 128, (D / 8) * 128);
    const uint64_t da_qB = tc_desc(S.q + TCBM * D, 0, 128, (D / 8) * 128);
#ifdef MYPRE_KF16
    // kind::f16 idesc a/b format field: 0 = F16, 1 = BF16. Q is staged as f16 and the K
    // image is converted straight to f16, so both operands of QK are f16; the PV idesc stays
    // bf16 because the reference rounds weighted_p to q.dtype = bf16 before the product.
    const uint32_t idesc_qk = tc_idesc(TCBM, BNT, 0, 0, 0);
#else
    const uint32_t idesc_qk = tc_idesc(TCBM, BNT, 0, 1, 1);
#endif
#ifdef MYPRE_VF16
    // kind::f16 for PV as well as QK: atype is P in TMEM, btype is V in shared. The 16-bit
    // packing and the swizzle-128B MN-major V image are unchanged - both types are 2 bytes -
    // so only the descriptor's two type fields move, exactly as MYPRE_KF16 moved idesc_qk's.
    const uint32_t idesc_pv = tc_idesc(TCBM, D, 1, 0, 0);
#else
    const uint32_t idesc_pv = tc_idesc(TCBM, D, 1, 1, 1);
#endif
    const uint32_t socl = grp * (2 * BNT);            // group's S col base
    const uint32_t oocl = TcSmem2<D, BNT>::SOFF + grp * D;  // group's O col base
    int tg = 0;                     // GLOBAL kv-tile counter (stage/parity)

    for (int which = 0; ; ++which) {
        if (which > 0) {
            if (tid == 0) S.tmem_base[1] = (unsigned)atomicAdd(P.counter, 1);
            __syncthreads();
            work0 = (int)S.tmem_base[1];
        }
        if (work0 >= T_all) break;
        const int work = T_all - 1 - work0;    // heavy items first (LPT-like)
        const int h = work % P.heads;          // heads innermost: GQA shares
        const int rem = work / P.heads;        // KV pages -> L2 reuse
        int b = -1, q0 = 0, qlen = 0, tile = 0;
        {
            int base2 = 0, prev2 = 0;
            for (int i = 0; i < P.batch; ++i) {
                int c = __ldg(P.cu + i + 1);
                int ntiles = (c - prev2 + RT - 1) / RT;
                if (rem < base2 + ntiles) {
                    b = i; tile = rem - base2; q0 = prev2; qlen = c - prev2; break;
                }
                base2 += ntiles;
                prev2 = c;
            }
        }
        if (b < 0) break;
        const int kvh = h / P.g;
        const int kvlen = __ldg(P.seqk + b);
        const int i0 = tile * RT;
        const int nrows = min(RT, qlen - i0);

        // ---- Q staging: 256 rows into two 128-row images (v4.12 mapping) ----
        {
            constexpr int PER_ROW = D / 8;
            char* qb = reinterpret_cast<char*>(S.q);
            for (int idx = tid; idx < RT * PER_ROW; idx += NT) {
                int row = idx / PER_ROW;
                int col8 = (idx % PER_ROW) * 8;
                int gimg = row >> 7;
                uint4* d = reinterpret_cast<uint4*>(qb + gimg * (TCBM * D * 2)
                                + TcSmem2<D, BNT>::km_off(row & (TCBM - 1), col8));
                if (row < nrows) {
                    const uint16_t* src = P.q + (long long)(q0 + i0 + row) * P.q_out_stride
                                        + h * D + col8;
#ifdef MYPRE_KF16
                    *d = bf16x8_to_f16x8(*reinterpret_cast<const uint4*>(src));
#else
                    *d = *reinterpret_cast<const uint4*>(src);
#endif
                } else {
                    *d = make_uint4(0, 0, 0, 0);
                }
            }
        }

        const int grow = grp * TCBM + w3 * 32 + lane;   // this thread's q row
        const int visible = (grow < nrows) ? (kvlen - qlen + i0 + grow) : 0;
        const int kn_end = min(kvlen, kvlen - qlen + i0 + RT);

        ConvRegs<D, BNT, NT> R;
        int bt_n = kv_bt_next<BNT, NT>(P, b, 0, kn_end);
        kv_prefetch<D, BNT, NT>(R, P, kvh, 0, min(BNT, kn_end), bt_n);
        if (BNT < kn_end) bt_n = kv_bt_next<BNT, NT>(P, b, BNT, kn_end);

        float m2 = NEG, l2 = 0.f;
        int tl = 0;                  // LOCAL tile index (PV accumulate flag)
        for (int kn = 0; kn < kn_end; kn += BNT, ++tl, ++tg) {
            const int s = tg & 1;
            const int nvalid = min(BNT, kn_end - kn);
            tc_convert<D, BNT, NT, 0, false, TcSmem2<D, BNT>>(R, S, s, nvalid, P);
            if (kn + BNT < kn_end) {
                kv_prefetch<D, BNT, NT>(R, P, kvh, kn + BNT, min(BNT, kn_end - kn - BNT), bt_n);
                bt_n = kv_bt_next<BNT, NT>(P, b, kn + 2 * BNT, kn_end);
            }
            tc_fence_async_smem();
            __syncthreads();

            // ---- QK x2 row tiles (shared K image), one commit ----
            if (tid == 0) {
                uint64_t db = tc_desc(S.k(s), 0, 128, (D / 8) * 128);
                #pragma unroll
                for (int kk = 0; kk < D / 16; ++kk) {
                    tc_mma_ss(tm + 0 * (2 * BNT) + s * BNT, da_qA + kk * 16, db + kk * 16,
                              idesc_qk, kk > 0 ? 1u : 0u);
                    tc_mma_ss(tm + 1 * (2 * BNT) + s * BNT, da_qB + kk * 16, db + kk * 16,
                              idesc_qk, kk > 0 ? 1u : 0u);
                }
                tc_commit(&S.bars[s]);
            }
            tc_wait(&S.bars[s], (tg >> 1) & 1);

            // ---- softmax (per group; validated single-tile text, w3 lanes) ----
            float sc[BNT];
            {
                uint32_t saddr = tm + socl + s * BNT + ((w3 * 32) << 16);
                #pragma unroll
                for (int seg = 0; seg < BNT / 32; ++seg)
                    tc_ld_32(sc + seg * 32, saddr + seg * 32);
                tc_wait_ld();
            }
            const float* kss = S.ks(s);   // pre-scaled by sm_scale*log2e
            const float* vss = S.vs(s);
#ifdef FMHA_PRE_VSMAXP
            vs_keepalive(vs_tile_max<BNT>(vss), sc);   // prices the pass; see the helper's comment
#endif
            float mn0 = m2, mn1 = m2, mn2 = m2, mn3 = m2;
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
                const float4 ks4 = *reinterpret_cast<const float4*>(kss + j);
                float s0 = sc[j] * ks4.x;
                float s1 = sc[j + 1] * ks4.y;
                float s2 = sc[j + 2] * ks4.z;
                float s3 = sc[j + 3] * ks4.w;
                if (kn + j > visible) s0 = NEG;
                if (kn + j + 1 > visible) s1 = NEG;
                if (kn + j + 2 > visible) s2 = NEG;
                if (kn + j + 3 > visible) s3 = NEG;
                sc[j] = s0; sc[j + 1] = s1; sc[j + 2] = s2; sc[j + 3] = s3;
                mn0 = fmaxf(mn0, s0); mn1 = fmaxf(mn1, s1);
                mn2 = fmaxf(mn2, s2); mn3 = fmaxf(mn3, s3);
            }
            const float mn = fmaxf(fmaxf(mn0, mn1), fmaxf(mn2, mn3));
            float alpha = ex2f(m2 - mn);
            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
            unsigned wp[BNT / 2];
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
                float p0 = ex2f(sc[j] - mn);
                float p1 = ex2f(sc[j + 1] - mn);
                float p2 = ex2f(sc[j + 2] - mn);
                float p3 = ex2f(sc[j + 3] - mn);
                sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
                sc[j] = p0; sc[j + 1] = p1;   // reuse sc storage for p
                sc[j + 2] = p2; sc[j + 3] = p3;
            }
            const float sum = (sum0 + sum1) + (sum2 + sum3);
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
#ifdef MYPRE_VSFOLD
                // VSFOLD: V's shared image already carries value_scale, so this loop has neither the scale nor
                // its 16 broadcast float4 loads - 16 LDS.128 and 64 FMUL per thread per tile deleted, which the
                // load->use cost model priced at ~5.1% and ~1.0% respectively.
                wp[j >> 1] = cvt_bf16x2_f32(sc[j], sc[j + 1]);
                wp[(j >> 1) + 1] = cvt_bf16x2_f32(sc[j + 2], sc[j + 3]);
#else
                const float4 vs4 = *reinterpret_cast<const float4*>(vss + j);
#ifdef MYPRE_VF16
                wp[j >> 1] = cvt_f16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
                wp[(j >> 1) + 1] = cvt_f16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
#else
                wp[j >> 1] = cvt_bf16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
                wp[(j >> 1) + 1] = cvt_bf16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
#endif
#endif
            }
            m2 = mn;
            l2 = l2 * alpha + sum;
            {
                uint32_t paddr = tm + socl + s * BNT + ((w3 * 32) << 16);
                #pragma unroll
                for (int seg = 0; seg < BNT / 64; ++seg)
                    tc_st_32(paddr + seg * 32, wp + seg * 32);
                tc_wait_st();
            }
            tc_fence_before();
            __syncthreads();
            tc_fence_after();

            // ---- correction per group (needs PV(t-1) done; 1-seg: 8 warps) ----
            if (tl > 0) tc_wait(&S.bars[2 + (s ^ 1)], ((tg - 1) >> 1) & 1);
            {
                bool need = __any_sync(0xffffffffu, alpha != 1.0f);
                if (need) {
                    uint32_t oaddr = tm + oocl + ((w3 * 32) << 16);
                    #pragma unroll
                    for (int seg = 0; seg < D / 32; ++seg) {
                        float o[32];
                        tc_ld_32(o, oaddr + seg * 32);
                        tc_wait_ld();
                        unsigned w[32];
                        #pragma unroll
                        for (int j = 0; j < 32; ++j) w[j] = __float_as_uint(o[j] * alpha);
                        tc_st_32(oaddr + seg * 32, w);
                    }
                    tc_wait_st();
                }
            }
            tc_fence_before();
            __syncthreads();
            tc_fence_after();

            // ---- PV x2 (shared V image; A = each group's P; O = each group) ----
            if (tid == 0) {
                uint64_t db = tc_desc(S.v(s), 2, BNT * 128, 1024);
                const int adv = 128; // K16 = 16 rows * 128B swizzle-atom stride >> 4
                #pragma unroll
                for (int kk = 0; kk < BNT / 16; ++kk) {
                    tc_mma_ts(tm + TcSmem2<D, BNT>::SOFF,
                              tm + 0 * (2 * BNT) + s * BNT + kk * 8,
                              db + kk * adv, idesc_pv, (tl > 0 || kk > 0) ? 1u : 0u);
                    tc_mma_ts(tm + TcSmem2<D, BNT>::SOFF + D,
                              tm + 1 * (2 * BNT) + s * BNT + kk * 8,
                              db + kk * adv, idesc_pv, (tl > 0 || kk > 0) ? 1u : 0u);
                }
                tc_commit(&S.bars[2 + s]);
            }
        }

        // ---- epilogue (per group, this item) ----
        tc_wait(&S.bars[2 + ((tg - 1) & 1)], ((tg - 1) >> 1) & 1);
        tc_fence_after();
        {
            float inv = rcpa(l2);
            uint16_t* orow = P.out + (long long)(q0 + i0 + grow) * P.out_stride + h * D;
            uint32_t oaddr = tm + oocl + ((w3 * 32) << 16);
            #pragma unroll
            for (int seg = 0; seg < D / 32; ++seg) {
                float o[32];
                tc_ld_32(o, oaddr + seg * 32);
                tc_wait_ld();
                if (grow < nrows) {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        unsigned pk = cvt_bf16x2_f32(o[2 * j] * inv, o[2 * j + 1] * inv);
                        *reinterpret_cast<unsigned*>(orow + seg * 32 + 2 * j) = pk;
                    }
                }
            }
        }
    }   // end persist item loop
    __syncthreads();
    if (warp == 0) tc_dealloc(tm, TcSmem2<D, BNT>::ACOLS);
}

#ifdef FMHA_PRE_ATTR_FORCE
// HARNESS ESCAPE HATCH, never in the candidate build. A second extension loaded in
// one process launches its merged prefill with maxDynamicSharedSizeBytes still at
// the 49152 default, so every opt-in-shared-memory launch is rejected with
// cudaErrorInvalidValue; re-issuing cudaFuncSetAttribute immediately before the
// launch raises it to the requested value and the identical launch then succeeds
// (dv_91d47f080294, dp4_8: maxdyn 49152 -> 100008, EXCEPTION -> OK). The per-branch
// `static const bool ok = []{...}()` initialisers are evidently not in force for the
// second module and why is not established - the probes that would settle it
// (fmha_dbg_attr_calls / ok_addr / stub_addr) are committed but unrun. What matters
// for measurement is that this makes in-process multi-arm prefill A/B possible
// again, which is a much better instrument than one arm per process: alternating
// reps over one shared input set instead of alternating processes.
//
// It costs 2-4 host-side runtime calls per prefill launch, so absolute numbers
// taken with it are inflated (up to a few percent on case 1's 118us) and only
// arm-to-arm ratios are meaningful. Every arm in a comparison must carry it.
namespace attrforce {
template <class F> inline void one(F* fn, int b) {
    cudaFuncSetAttribute((const void*)fn, cudaFuncAttributeMaxDynamicSharedMemorySize, b);
}
template <int D> inline void all() {
    constexpr int SMEM = TcSmem<D, 64>::bytes();
    if constexpr (D == 64) {
        // Only the instantiations the default build can actually reach: with
        // MYPRE_W8BNT96 on (the shipped default) fa_prefill_w8<64, 64, ...> is
        // never launched, and naming it here would instantiate a path nothing
        // else compiles.
        constexpr int SMEM96 = TcSmem<64, 96>::bytes();
        one(fa_prefill_w8<64, 96, true, true>, SMEM96);
        one(fa_prefill_tc<64, 64, true, false>, SMEM);
        one(fa_prefill_tc<64, 64, true, true>, SMEM);
    } else if constexpr (D == 128) {
        one(fa_prefill_w8<128, 64, true, true>, SMEM);
        one(fa_prefill_tc<128, 64, true, false>, SMEM);
        one(fa_prefill_tc<128, 64, true, true>, SMEM);
    } else {
        one(fa_prefill_tc<256, 64, false, false>, SMEM);
        one(fa_prefill_tc<256, 64, false, true>, SMEM);
    }
}
}  // namespace attrforce
#endif

} // namespace mypre

#ifdef FMHA_DBG_PRE
// Exported so a multi-arm process can ask each extension what IT sees. Same
// address from two extensions means the symbol was merged at load time.
int fmha_dbg_attr_calls() { return mypre::dbgpre::attr_calls; }
long long fmha_dbg_ok_addr() {
    return (long long)(uintptr_t)mypre::dbgpre::ok_addr;
}
long long fmha_dbg_stub_addr() {
    return (long long)(uintptr_t)(const void*)mypre::fa_prefill_tc<128, 64, true, false>;
}
long long fmha_dbg_attr_errs() {
    return (long long)mypre::dbgpre::attr_err0 * 1000 + mypre::dbgpre::attr_err1;
}
#endif

// ============================ host side =====================================
// Called from launch_plan (cfg 30/31/32) with the unified ::Params. Field
// mapping is positional: both solutions consume the same input tensors, so
// peer k_s0/s1/s2 are exactly arm-b k_sb/k_sp/k_sh (block/slot/head strides),
// qk_scale == sm_scale_log2e (same double constant) and ps_log2 == page_shift.
template <int D>
void launch_mypre(const Params& p, cudaStream_t st, bool f8ok) {
    static_assert(D == 64 || D == 128 || D == 256, "frozen shape_range head dims");
    TORCH_CHECK(p.n_split == 1, "merged tc prefill requires n_split == 1");
    TORCH_CHECK(p.D == D, "head_dim/cfg mismatch");
    static const int sms = [] {
        int x = 0; cudaDeviceGetAttribute(&x, cudaDevAttrMultiProcessorCount, 0); return x;
    }();
    mypre::Params P = {};
    P.q = reinterpret_cast<const uint16_t*>(p.q);
    P.out = reinterpret_cast<uint16_t*>(p.out);
    P.k = p.k; P.v = p.v; P.ks = p.ks; P.vs = p.vs;
    P.block_table = p.btab; P.cu = p.cu; P.seqk = p.slen;
    P.q_out_stride = p.q_s0; P.out_stride = p.out_s0;
    P.k_sb = p.k_s0; P.k_sp = p.k_s1; P.k_sh = p.k_s2;
    P.v_sb = p.v_s0; P.v_sp = p.v_s1; P.v_sh = p.v_s2;
    P.ks_sb = p.ks_s0; P.ks_sp = p.ks_s1; P.ks_sh = p.ks_s2;
    P.vs_sb = p.vs_s0; P.vs_sp = p.vs_s1; P.vs_sh = p.vs_s2;
    P.bt_stride = p.btab_s;
    P.batch = p.B; P.heads = p.H; P.g = p.g;
    P.D = p.D; P.page_shift = p.ps_log2;
    P.sm_scale_log2e = p.qk_scale;
    // v4.5 mirror pairing (diagonal band) / v4.8 persistent pull (>= band):
    // gating formulas verbatim from the v4.12 host (fmha_host.cu), with tq
    // (packed total-q rows) carried in Params for the loose grid bound.
    const long long grid_orig =
        ((long long)(p.tq + mypre::TCBM - 1) / mypre::TCBM + p.B) * p.H;
    const long long slots = (D == 256 ? 1 : 2) * (long long)sms;
    const bool diag = 2LL * (long long)p.msq >= (long long)p.msk;
    // D64 kernel-class predicates (stride-derived layout family + the edge-
    // swept offset boundary); computed once here, reused by the D64 launcher.
    const bool hnd_family = p.k_s2 > p.k_s1;   // head stride > slot stride
    const bool offrect_g1 = hnd_family && p.g == 1 && 4LL * p.msk > 5LL * p.msq;
#if (defined(MYPRE_W8D64_FOLD) || defined(MYPRE_W8D64_FOLDS)) && !defined(MYPRE_NO_W8D64)
    // j23-w8d64f A/B arm: D64 joins the v4.5 fold band (mirror-pair items,
    // single wave) EXCEPT the offrect_g1 tc class. The D!=64 exclusion was
    // v4.12-lineage, never a measured D64 verdict; W8 fold won D128 (+4.7%)
    // and D64 halves TMEM columns (ACOLS 256 -> 2 CTA/SM = 16 warps).
    //
    // MEASURED ungated (dv_9a145e6a6c9c, results/bm32fl64-20261028.md): -6.1%
    // on b1 (band, persist-INELIGIBLE: 4*grid < 5*slots, where the alternative
    // is static tc with a 300/296 two-wave tail) but +38.8% on b2 (band,
    // persist-ELIGIBLE, where the alternative is W8-96 persist and v4.9's
    // persist-over-fold logic holds for D64 too). The FOLDS gate below is the
    // measured-good half of the arm: fold replaces STATIC, never persist.
#if defined(MYPRE_W8D64_FOLDS)
    const bool d64_fold =
        (D == 64) && !offrect_g1 && (4LL * grid_orig < 5LL * slots);
#else
    const bool d64_fold = (D == 64) && !offrect_g1;
#endif
#else
    const bool d64_fold = false;
#endif
#if defined(MYPRE_W8D64_FOLD) && defined(MYPRE_W8D64_FOLDS)
#error "MYPRE_W8D64_FOLDS is the persist-eligibility-gated form of MYPRE_W8D64_FOLD; the ungated arm is measured at +38.8% on band-persist shapes. Pick one."
#endif
// NOTE (dv job 20261028T123857Z, the first folds64 run): these two guards were
// originally MYPRE_W8D64_FOLD-only, so a FOLDS-only build compiled d64_fold=false
// AND omitted the fold launch branch - the arm measured ctl-vs-ctl, flat and
// bit-identical everywhere. Both guards above now accept either flag. The tell
// was b1: fold CANNOT be bit-identical (fl64 moved it 4.5e-6 rel_l2).
    const bool band = diag && grid_orig >= slots && grid_orig <= 2 * slots;
    // v4.9 (ADOPTED, dv_04c60dfa2883, 3 alternating reps): the D128 band runs W8 +
    // PERSIST, not W8 + fold. Mirror pairing makes work per CTA constant but only
    // launches ceil(n/2)*H CTAs, so its makespan is 2n+2 tiles whatever the machine
    // size, while persist's LPT over the n*H singles approaches total/slots. Across
    // 24 band shapes (experiments/prefill_balance.py, the model validated to 1.34%
    // on the per-tile constant and exact on case 9's measured null) persist/fold
    // runs 0.857 to 0.990 and never exceeds 1.0, with the worst case at the top of
    // the band where fold nearly fills the machine. Case 1 measured 118.8 -> 110.6
    // median and 116.8 -> 108.4 minimum, 0.9310 / 0.9282 against the model's
    // 0.9412, and BIT-IDENTICAL - only the item->CTA mapping moves, and every q tile
    // is independent. The same job's third arm isolates why this was never found
    // before: tc + persist on case 1 is 122.9 (1.0345), so every earlier
    // fold-vs-persist comparison that used the 4-warp kernel on the persist side
    // was measuring the kernel gap (+4.7% in fold's favour, merge9-w8f) and not the
    // schedule. W8 is what makes persist win here.
#ifdef MYPRE_BAND_FOLD
    const bool band_w8p = false;   // A/B control: the pre-v4.9 W8-fold band
#else
    const bool band_w8p = (D == 128) && band;
#endif
    P.fold = ((((D != 64) || d64_fold) && band) && !band_w8p) ? 1 : 0;
#ifdef MYPRE_NO_FOLD
    // A/B arm, never shipped: the fold band runs persist instead, i.e. LPT over
    // single 128-row tiles rather than mirror pairs. The makespan model
    // (experiments/prefill_balance.py, validated to 1.3% on the per-tile constant
    // and exact on case 9's measured null result) says case 1 is the one fixed
    // case where fold leaves slots idle - 256 paired CTAs against 296 slots, a
    // makespan of 34 tiles against an ideal of 29.4 - and that persist's 512
    // single tiles schedule to 32.0, i.e. -5.88%. Against that: merge8 measured
    // W8 on a multi-wave persist shape at -12.9%, attributing it to compounding
    // per-tile TMEM-port and 8-warp barrier costs, and case 1 in persist is 1.73
    // waves. The two disagree, so it gets a job.
    P.fold = 0;
#endif
    // v4.8.4: persist only when clearly multi-wave (c9 at 1.08 waves lost 4%
    // to pull overhead; c0 2.8x / c7 2.6x waves won 15% / 35%). Reconfirmed
    // post-PIPE/nomask (merge5-pg A/B dv_223d61ae232a): relaxing to
    // grid >= slots gave c9 +0.17% = noise — its non-diag items have uniform
    // work, so pull scheduling has no variance to exploit. Gate stays 4/5.
    // the band joins persist on the strength of the same A/B; the 4/5 gate below it
    // is unchanged, so a band shape no longer has to be multi-wave to pull tickets
    P.persist = (!P.fold && (band_w8p || 4 * grid_orig >= 5 * slots)) ? 1 : 0;
    int nctas = 0;
    if (P.persist) {
        // One-time allocation, zeroed on the launch stream every call: graph
        // capture records the memset as a node and every replay re-arms the
        // ticket. fmha_capture runs only after a direct fmha_run of the same
        // binding (solution.py captures on second sighting), and the mode is
        // a pure function of the binding's scalars, so the cudaMalloc below
        // can never execute during capture.
        static int* ctr = [] {
            int* c = nullptr;
            TORCH_CHECK(cudaMalloc(&c, 4) == cudaSuccess, "persist counter alloc");
            return c;
        }();
        cudaMemsetAsync(ctr, 0, 4, st);
        P.counter = ctr;
        nctas = (int)(grid_orig < slots ? grid_orig : slots);
#ifdef MYPRE_CLUSTER
        // Cluster KV-split gate. Only the ticket path: a static grid sized
        // n_items*CL overflows co-residency, and dv_2c980c231e11 measured what that
        // costs (+78% on case 9, 12 clusters deferred to a second wave). Only
        // D != 256, whose 128x260 float O staging does not fit the freed q/kc/vc
        // region. And only when every rank still gets several tiles, so the merge
        // amortises: 8*BNT*CL keys. nctas is rounded DOWN to a whole number of
        // clusters so the launch never needs more slots than are resident.
#ifndef MYPRE_NO_CLSPLIT
        if constexpr (mypre::TC_CL > 1) {
            if (D != 256 && !P.fold
                && (long long)p.msk >= 8LL * 64 * mypre::TC_CL) {
                P.cl_split = 1;
                nctas = (nctas / mypre::TC_CL) * mypre::TC_CL;
            }
        }
#endif
#endif
    }
#ifdef MYPRE_RT2P
    // merge7-rt2 A/B arm: D128 persist routes to the dual-row-tile kernel
    // (256 threads, 256-row items, TMEM 512 cols -> 1 CTA/SM, slots = sms).
    // The counter memset above already re-arms the same ticket word; the
    // ticket space is the RT2 T_all (256-row items), computed in-kernel.
    if constexpr (D == 128) {
        if (P.persist) {
            constexpr int SMEM2 = mypre::TcSmem2<128, 64>::bytes();
            static_assert(SMEM2 <= 232448, "rt2 shared memory over budget");
            static const bool ok2 = [] {
                cudaFuncSetAttribute((const void*)mypre::fa_prefill_rt2p<128, 64>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM2);
                return true;
            }();
            (void)ok2;
            const long long grid_rt2 =
                ((long long)(p.tq + 2 * mypre::TCBM - 1) / (2 * mypre::TCBM) + p.B) * p.H;
            const int nctas2 = (int)(grid_rt2 < (long long)sms ? grid_rt2 : (long long)sms);
            DBGP_KA("D128 RT2 persist:", 256, SMEM2, mypre::fa_prefill_rt2p<128, 64>);
            DBGP(1, "D128 RT2 persist: nctas=%d grid_rt2=%lld slots=%d ACOLS=%d",
                 nctas2, grid_rt2, (int)sms, (int)mypre::TcSmem2<128, 64>::ACOLS);
            mypre::fa_prefill_rt2p<128, 64><<<dim3((unsigned)nctas2, 1), 256, SMEM2, st>>>(P);
            DBGP_KB("D128 RT2 persist:", 256, SMEM2, mypre::fa_prefill_rt2p<128, 64>);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }
#endif
    constexpr int SMEM = mypre::TcSmem<D, 64>::bytes();
    static_assert(SMEM <= 232448, "tc prefill shared memory over budget");
#ifdef FMHA_PRE_ATTR_FORCE
    {
        // Skip the re-set while the caller is capturing: the direct fmha_run that
        // always precedes a capture (solution.py captures on second sighting) has
        // already raised the limit for this module's kernels, and keeping the
        // capture path free of extra runtime calls means the recorded launch is
        // validated the same way production validates it.
        cudaStreamCaptureStatus dbg_cs = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(st, &dbg_cs);
        if (dbg_cs == cudaStreamCaptureStatusNone) mypre::attrforce::all<D>();
    }
#endif
    const int trows = P.fold ? 2 * mypre::TCBM : mypre::TCBM;
    const dim3 grid = P.persist
        ? dim3((unsigned)nctas, 1)
        : dim3((unsigned)((p.tq + trows - 1) / trows + p.B), (unsigned)p.H);
    // identical to `grid` unless MYPRE_CLUSTER is compiled in (see the macro)
    const dim3 grid_tc = dim3(MYPRE_CLUSTER_GX(grid.x), grid.y, grid.z);
    DBGP(1, "D=%d fold=%d persist=%d grid=(%u,%u,%u) SMEM=%d trows=%d tq=%lld B=%d H=%d g=%d msq=%d msk=%d n_split=%d pending=%s",
         D, (int)P.fold, (int)P.persist, grid.x, grid.y, grid.z,
         SMEM, trows,
         (long long)p.tq, (int)p.B, (int)p.H, (int)p.g, (int)p.msq, (int)p.msk,
         (int)p.n_split, DBGP_PEND());
    // merge9-w8f (ADOPTED, dv_039b8aec8287: c1 +4.7%, canaries flat; merge8
    // dv_0d3da0b90db5: persist -10% / static -9.4%): D128 FOLD launches route
    // to the 8-warp kernel — single-wave fold gets pure latency hiding from
    // 16 warps/SM, while multi-wave shapes would pay compounding per-tile
    // TMEM-port + 8-warp barrier costs. Persist/static keep the 4-warp kernel.
    if constexpr (D == 128) {
        // v4.9: fold and the band's persist both want the 8-warp kernel; only the
        // driver mode differs, and the kernel reads it from P. SMEMW == SMEM at
        // D=128 (both TcSmem<128,64>::bytes()).
        if (P.fold || band_w8p) {
            constexpr int SMEMW = mypre::TcSmem<128, 64>::bytes();
            // j23-w8nm (ADOPTED, receipt 20261023T095213Z: c1 +1.04% with
            // all 3 reps winning, canaries bit-flat, mask-edge probes green):
            // nomask ON in the W8 fold kernel. The 4-warp fold lost -3.4%
            // with nomask (merge1-nm) but W8's quad-shfl softmax mix flips
            // the sign - priors really do not transfer across structures.
            static const bool okw = [] {
                cudaFuncSetAttribute((const void*)mypre::fa_prefill_w8<128, 64, true, true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, SMEMW);
                return true;
            }();
            (void)okw;
            DBGP_KA("D128 fold W8:", 256, SMEMW, mypre::fa_prefill_w8<128, 64, true, true>);
            mypre::fa_prefill_w8<128, 64, true, true><<<grid, 256, SMEMW, st>>>(P);
            DBGP_KB("D128 fold W8:", 256, SMEMW, mypre::fa_prefill_w8<128, 64, true, true>);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }
    // PIPE split (v2.5/v4.12, merge3-p128): D=64 and D=128 pipelined,
    // D=256 sequential. NM (mask-free softmax fast path) rides only the
    // persist launches; see the dv_84533b62e76c A/B note at the top.
    if constexpr (D == 64) {
#ifndef MYPRE_NO_W8D64
        // j23-w8d64 ADOPTED + j23-w8d64s CONDITIONAL ROUTING.
        // Adoption A/B (dv_6ef8a5279598, results/ab-w8d64-20261023.json):
        // c7 +1.7% on 3/3 reps, canaries flat, probes green — W8's per-tile
        // TMEM penalty halves at D64 (S 128 + O 64 cols) while its L1
        // wavefront advantage holds (TPR=4). The D128 all-mode -10% verdict
        // was structure/scale-specific.
        // Geometry sweep (dv_d2e3e4a1ea51, results/w8d64-sweep-20261023.json,
        // 8 legal D64-persist geometries x 3 alternating reps, all 3/3
        // consistent): W8 wins square-causal at any page (sq8k p16 +2.1%,
        // p32 +1.9%, sq4k p32 +2.3%), NHD offset (+1.7%), g=4 offset (+0.6%);
        // loses -2.3% on HND-labeled g=1 offset rectangles (d4-class,
        // confirmed by the domain gate: d4 300->308). tc IS the pre-adoption
        // baseline, so this branch is >= baseline on every swept geometry.
        // Edge sweep (dv_9624f1bc3fbc, results/w8d64-edge-20261023.json,
        // q4096 HND g1 ratio axis, anchors reproduced sweep-1 within 0.1%):
        // W8 wins 3/3 up to ratio 1.25 (+1.4-1.6% at r102/r112/r125, p16 and
        // p32 alike), tc wins 3/3 from 1.5 (-1.6%). Boundary set at 4*msk >
        // 5*msq (ratio 5/4): last measured W8 win kept, unmeasured gap
        // (1.25, 1.5) falls to the tc baseline, style matches the persist
        // gate's 4/5 constants. Mild-offset W8 correctness: probes
        // mild_odd_4160 (ODD tile parity) + mild_5120, 0 mismatch.
        // Layout family is read from the runtime KV strides, never from a
        // label: the frozen generator's (block, slot, kvh) view gives the
        // HND family k_sp = 264B packed-row pitch < k_sh = page*264, and the
        // NHD family the reverse (input.py L70-75). Generic geometry/
        // stride dispatch, no case identity; kvh >= 4 whenever g == 1 here
        // (heads in {4,8,12,32}), so the strides are never degenerate.
        // MYPRE_NO_W8D64=1 disables the W8 path entirely (A/B control).
        // j23-bnt96/bnt96b ADOPTED (dv_140460d6086b + dv_78c73a264ec9,
        // evict protocol, 3/3 reps each): BNT=96 on the W8 D64-persist kernel
        // wins +17.6..18.5% on the former W8-side class AND +20.1..20.4% on
        // the offrect class that had motivated the tc carve-out (d4 twin
        // 0.3132->0.2601ms), +10.9% on NHD-offset over W8-64. The carve-out
        // and its 4*msk>5*msq boundary are RETIRED — W8-96 takes every D64
        // persist launch; tc keeps D64 static and the FMHA_PRE_NO_W8D64
        // control arm. Mechanism: 1/3 fewer per-tile overhead rounds at
        // UNCHANGED occupancy (TMEM NEED 2*96+64 = 256 = ACOLS exactly,
        // 128 regs x 256 thr x 2 CTA = exact regfile, smem 67752B x 2).
        // Prereq fix 75070d7: tc_convert zero-fill guard for BN not dividing
        // NT (rows >= BN have no smem region). FORCE knob now == default.
        const bool use_w8 = (P.persist != 0);
        (void)offrect_g1;   // dormant: only the MYPRE_W8D64_FOLD arm reads it
#if (defined(MYPRE_W8D64_FOLD) || defined(MYPRE_W8D64_FOLDS)) && !defined(MYPRE_NO_W8D64)
        if (P.fold) {
            // D64 fold arm: mirror-pair items on the 8-warp kernel (same
            // pattern as the adopted D128 W8 fold: SMEM = TcSmem<D,64>,
            // 256 threads, fold 2D grid over 2*TCBM rows, NM=true).
            static const bool okf = [] {
                cudaFuncSetAttribute((const void*)mypre::fa_prefill_w8<64, 64, true, true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
                return true;
            }();
            (void)okf;
            mypre::fa_prefill_w8<64, 64, true, true><<<grid, 256, SMEM, st>>>(P);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
#endif
        if (use_w8) {
#ifdef MYPRE_W8BNT96
            // j23-bnt96 A/B arm: W8-D64-persist with 96-col KV tiles. All
            // three probe gates green (dv_6d76a3c8c9de): 128 regs/0 spills
            // (= exactly 2 CTA/SM), mma N=96 exact on the L1 image, x4-LD map
            // bijective on cols 64..95. 1/3 fewer per-tile overhead rounds;
            // TMEM NEED = 2*96+64 = 256 = ACOLS exactly.
            constexpr int SMEM96 = mypre::TcSmem<D, 96>::bytes();
            static_assert(SMEM96 <= 232448, "bnt96 shared memory over budget");
            static const bool ok96 = [] {
                cudaFuncSetAttribute((const void*)mypre::fa_prefill_w8<64, 96, true, true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM96);
                return true;
            }();
            (void)ok96;
            DBGP_KA("D64 W8-96 persist:", 256, SMEM96, mypre::fa_prefill_w8<64, 96, true, true>);
            mypre::fa_prefill_w8<64, 96, true, true><<<grid, 256, SMEM96, st>>>(P);
            DBGP_KB("D64 W8-96 persist:", 256, SMEM96, mypre::fa_prefill_w8<64, 96, true, true>);
#else
            static const bool ok8 = [] {
                cudaFuncSetAttribute((const void*)mypre::fa_prefill_w8<64, 64, true, true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
                return true;
            }();
            (void)ok8;
            mypre::fa_prefill_w8<64, 64, true, true><<<grid, 256, SMEM, st>>>(P);
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
#endif
        static const bool ok = [] {
            cudaFuncSetAttribute((const void*)mypre::fa_prefill_tc<64, 64, true, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
            cudaFuncSetAttribute((const void*)mypre::fa_prefill_tc<64, 64, true, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
            return true;
        }();
        (void)ok;
        if (P.persist) mypre::fa_prefill_tc<64, 64, true, true><<<grid_tc, mypre::TC_THREADS, SMEM, st>>>(P);
        else mypre::fa_prefill_tc<64, 64, true, false><<<grid_tc, mypre::TC_THREADS, SMEM, st>>>(P);
    } else if constexpr (D == 128) {
        // merge3-p128: PIPE is now the D128 default. v2.3b's 0.95x verdict
        // predates bt-ahead, chunk-interleave, ILP-4, float4 scales, nomask
        // and the v4.8 drivers; retested same-job post-parity-fix
        // (dv_82f60696c22a, 3 alternating reps): c0 persist +2.1%, c9 static
        // +2.9%, c1 fold flat (1.000x), c7 D64 canary untouched; corr probes
        // pf_fold_odd + pf_persist_d128 green (fold x PIPE first ever run,
        // safe only since the merge3-pf prologue parity fix).
        constexpr bool PIPEF = true;
        static const bool ok = [] {
            const cudaError_t e0 = cudaFuncSetAttribute(
                (const void*)mypre::fa_prefill_tc<128, 64, PIPEF, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
            const cudaError_t e1 = cudaFuncSetAttribute(
                (const void*)mypre::fa_prefill_tc<128, 64, PIPEF, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
#ifdef FMHA_PRE_F8QK
            cudaFuncSetAttribute(
                (const void*)mypre::fa_prefill_tc<128, 64, PIPEF, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
            cudaFuncSetAttribute(
                (const void*)mypre::fa_prefill_tc<128, 64, PIPEF, true, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
#endif
#ifdef FMHA_DBG_PRE
            // The two questions the level reports cannot answer: did this run at
            // all in this module (attr_calls, ok_addr against the other module's),
            // and if it ran, what did the driver say (attr_err0/1).
            ++mypre::dbgpre::attr_calls;
            mypre::dbgpre::ok_addr = (const void*)&ok;
            mypre::dbgpre::attr_err0 = (int)e0;
            mypre::dbgpre::attr_err1 = (int)e1;
#else
            (void)e0; (void)e1;
#endif
            return true;
        }();
        (void)ok;
#ifdef MYPRE_W8D128M
        // j23-w8d128m re-validation arm: W8 on D128 PERSIST (NM=true) and
        // STATIC (NM=false). The warm-L2 merge8 verdicts (persist -10%,
        // static -9.4%) predate the official restore-l2-evict-event-v2
        // metric; the v2 engine shows W8-fold beating the incumbent's tc
        // fold by -22.9% on d3 (cold L2 favors 16-warp DRAM-latency hiding),
        // so both warm rejections are re-tested under an evict-variant A/B
        // protocol before standing. Default build unaffected (ifdef off).
        {
            static const bool okm = [] {
                cudaFuncSetAttribute((const void*)mypre::fa_prefill_w8<128, 64, true, true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
                cudaFuncSetAttribute((const void*)mypre::fa_prefill_w8<128, 64, true, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
                return true;
            }();
            (void)okm;
            if (P.persist) mypre::fa_prefill_w8<128, 64, true, true><<<grid, 256, SMEM, st>>>(P);
            else mypre::fa_prefill_w8<128, 64, true, false><<<grid, 256, SMEM, st>>>(P);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
#endif
#ifdef MYPRE_W8D128S
        // A/B arm (FMHA_PRE_W8D128S): the 8-warp kernel on D128 STATIC launches,
        // with nomask on, i.e. the shipped static path with 4 warps/CTA replaced by
        // 8 at the same 256 TMEM columns and the same 100008 B of shared memory -
        // 2 CTAs/SM either way, so 16 warps/SM instead of 8. The only unmeasured
        // arm left on the second-largest fixed case: case 9 is static, 636us, one
        // balanced wave of 288 real CTAs over 296 slots, at ~19% of the f16 tensor
        // roofline with nothing saturated.
        //
        // The evidence for and against is genuinely mixed, which is why it gets a
        // job rather than a verdict from analogy. Against: arm b measured W8-static
        // at -9.4% and W8-persist at -10% on warm L2, and I re-measured persist
        // first-party under v2 at -12.9% (results/prefill-w8-persist-20261024.md),
        // i.e. the warm number understated the loss. For: W8-FOLD was +4.7% warm
        // and much better again under v2, so this mechanism family does change
        // magnitude and even sign between protocols, and static is a different
        // schedule from persist - no ticket atomics, no pull loop, one item per CTA,
        // which is the case where the extra warps are pure latency hiding.
        // NM=true matches the shipped static path, so the A/B isolates warp count.
        if (!P.persist) {
            static const bool oks = [] {
                cudaFuncSetAttribute((const void*)mypre::fa_prefill_w8<128, 64, true, true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
                return true;
            }();
            (void)oks;
            DBGP_KA("D128 W8 static:", 256, SMEM, mypre::fa_prefill_w8<128, 64, true, true>);
            mypre::fa_prefill_w8<128, 64, true, true><<<grid, 256, SMEM, st>>>(P);
            DBGP_KB("D128 W8 static:", 256, SMEM, mypre::fa_prefill_w8<128, 64, true, true>);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
#endif
#ifdef MYPRE_NM_STATIC
        // EXPERIMENT (FMHA_PRE_NM_STATIC): mask-free softmax on STATIC D128 launches too,
        // not only persist. This stack ships nomask for persist on the strength of c0 +3.8%;
        // the pre-join arm-a tree instead gated it as `P.persist || !P.fold`, i.e. persist AND
        // static and never fold, and measured static separately in the same job as persist
        // (results/prefill-nomask-20261023.md, three arms ctl/nm/nma): static case 9 read
        // 662.8 -> 663.4 with persist-only (1.0008, "arm b measured nothing") and 662.8 ->
        // 644.8 with static included (0.9728), a second arm putting it at -2.64% and -2.72%.
        // Fold is untouched in both trees, which is where the 4-warp kernel measured -3.4%
        // WITH nomask and W8's quad-shuffle mix flips the sign. Manual join 20261024 then
        // adopted the persist-only winner and fixed case 9 went 654.16 -> 684.03 (arm b's own
        // certificate) / 702.35 (the unified job), i.e. +4.6% to +7.4%, of which this is the
        // part with a measured mechanism. Both instantiations already have their
        // cudaFuncSetAttribute call above, so this changes only which one is launched.
#ifdef FMHA_PRE_F8QK
        if (f8ok) {
            mypre::fa_prefill_tc<128, 64, PIPEF, true, true><<<grid_tc, mypre::TC_THREADS, SMEM, st>>>(P);
        } else
#endif
        {
            DBGP_KA("D128 tc NM-static:", (int)mypre::TC_THREADS, SMEM,
                    mypre::fa_prefill_tc<128, 64, PIPEF, true>);
            mypre::fa_prefill_tc<128, 64, PIPEF, true><<<grid_tc, mypre::TC_THREADS, SMEM, st>>>(P);
            DBGP_KB("D128 tc NM-static:", (int)mypre::TC_THREADS, SMEM,
                    mypre::fa_prefill_tc<128, 64, PIPEF, true>);
        }
#else
#ifdef FMHA_PRE_F8QK
        if (f8ok) {
            if (P.persist) mypre::fa_prefill_tc<128, 64, PIPEF, true, true><<<grid_tc, mypre::TC_THREADS, SMEM, st>>>(P);
            else mypre::fa_prefill_tc<128, 64, PIPEF, false, true><<<grid_tc, mypre::TC_THREADS, SMEM, st>>>(P);
        } else
#endif
        {
            DBGP_KA("D128 tc:", (int)mypre::TC_THREADS, SMEM,
                    (P.persist ? mypre::fa_prefill_tc<128, 64, PIPEF, true>
                               : mypre::fa_prefill_tc<128, 64, PIPEF, false>));
            if (P.persist) mypre::fa_prefill_tc<128, 64, PIPEF, true><<<grid_tc, mypre::TC_THREADS, SMEM, st>>>(P);
            else mypre::fa_prefill_tc<128, 64, PIPEF, false><<<grid_tc, mypre::TC_THREADS, SMEM, st>>>(P);
            DBGP_KB("D128 tc:", (int)mypre::TC_THREADS, SMEM,
                    (P.persist ? mypre::fa_prefill_tc<128, 64, PIPEF, true>
                               : mypre::fa_prefill_tc<128, 64, PIPEF, false>));
        }
#endif
    } else {
        static const bool ok = [] {
            cudaFuncSetAttribute((const void*)mypre::fa_prefill_tc<256, 64, false, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
            cudaFuncSetAttribute((const void*)mypre::fa_prefill_tc<256, 64, false, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
            return true;
        }();
        (void)ok;
        if (P.persist) mypre::fa_prefill_tc<256, 64, false, true><<<grid_tc, mypre::TC_THREADS, SMEM, st>>>(P);
        else mypre::fa_prefill_tc<256, 64, false, false><<<grid_tc, mypre::TC_THREADS, SMEM, st>>>(P);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef MYPRE_W8D64_CENSUS
// j23-bnt96 census-only reference (never called from production): forces
// instantiation of the BNT=96 W8-D64 candidate so ptxas --ptxas-options=-v
// reports its register/spill cost without changing any routing. TcSmem<64,96>:
// NEED = 2*96+64 = 256 = ACOLS exactly -> 2 CTA/SM kept; smem 67752B.
extern "C" void* w8_bnt96_census_addr() {
    return (void*)mypre::fa_prefill_w8<64, 96, true, true>;
}
#endif
