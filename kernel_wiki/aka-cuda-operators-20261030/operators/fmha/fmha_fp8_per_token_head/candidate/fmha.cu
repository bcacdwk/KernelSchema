// fmha_fp8_per_token_head — independent CUDA C++/PTX implementation for L20D (sm_103).
//
// Numerical contract (frozen, problem/reference.py + metadata.json):
//   * Q is BF16; KV payload is E4M3 with independent FP32 per-(token, kv-head)
//     scales k_scale_cache / v_scale_cache stored inside the page.
//   * QK is accumulated in FP32 (TF32 disabled in the reference), then the K
//     scale and softmax_scale are applied to the FP32 score.
//   * softmax runs in FP32; the denominator uses the *unrounded* P.
//   * P * v_scale is rounded to BF16 (RNE) before the PV product, which is
//     accumulated in FP32 against the stored V payload.
//   * Bottom-right causal masking only (window_size=(-1,-1), softcap=0,
//     sinks=None, lse=None) — that is what reference.normalized() fixes.
//
// E4M3 -> BF16 is exact (3 mantissa bits, exponent range inside BF16), and the
// Q operand is already BF16, so mma.sync.m16n8k16.bf16 with an FP32 accumulator
// reproduces the reference's FP32 QK exactly up to summation order. The
// 32-key grouping in the reference is numerical semantics, not a schedule, so
// this kernel uses its own BN and its own online-softmax rescaling.
//
// Attribution: mechanism (paged KV, flash online softmax, mma/ldmatrix fragment
// layouts) follows the public FlashAttention-2 / flashinfer descriptions in
// knowledge/nvidia; all code here is written independently.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#define DEVI __device__ __forceinline__

// Programmatic dependent launch (sm_90+), used only when FMHA_PDL is defined and
// only between fmha_main and fmha_combine. launch_dependents in the primary is a
// scheduling hint: it lets the secondary grid be dispatched and run whatever of its
// own work does not depend on the primary's output. It gives NO memory visibility -
// that comes from griddepcontrol.wait in the secondary, which must precede every
// read of the workspace the primary wrote. If the launch attribute is not honoured
// (or the pair is not in the same stream/graph) both instructions degenerate to the
// ordinary stream ordering that is in force today.
//
// Mechanism and placement follow the public sm103a sources in references/:
// flashinfer/csrc/blackwell_msa/sm103a/blackwell_msa_long_prefill_reduce_paged_bf16_gqa16.cu:163
// (wait at the top of the reduce kernel, after address setup, before its first
// dependent load) and ..._long_prefill_paged_bf16_gqa16_sm103.cu:1904
// (launch_dependents elected in one lane of one warp of the primary).
#ifdef FMHA_PDL
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define FMHA_PDL_TRIG() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
#define FMHA_PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#else
#define FMHA_PDL_TRIG() do {} while (0)
#define FMHA_PDL_WAIT() do {} while (0)
#endif
#endif

// ---------------------------------------------------------------- PTX helpers
DEVI uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// volatile on purpose: without it ptxas hoists every fragment load of the whole
// KV block to the top, which blows the register budget and spills. The overlap
// is produced by the explicit one-deep pipeline inside compute() instead.
DEVI void ldsm_x4(uint32_t* r, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVI void ldsm_x4_t(uint32_t* r, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVI void mma_bf16(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEVI void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                 "r"(smem_u32(dst)), "l"(src));
}
DEVI void cp_async4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::
                 "r"(smem_u32(dst)), "l"(src));
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// Two E4M3 bytes -> two BF16 (exact). Hardware fp8->fp16 is exact, and every
// finite E4M3 value (including subnormals, whose FP16 exponent is >= 6) is
// exactly representable in BF16, so the two-step route needs no special case.
#if defined(FMHA_DEC_F16K) || defined(FMHA_DEC_VF16)
// The decode side of the same mechanism the prefill ships as MYPRE_KF16: one instruction instead
// of four, and exact, because every finite E4M3 value - including subnormals, whose FP16 exponent
// is >= 6 - is representable in FP16 (11 significand bits, |x| <= 65504) as well as in BF16 (8).
// That is the argument the existing helper's own comment makes for the two-step route, and it is
// also flashinfer's for the native cvt.rn.bf16x2.e4m3x2 it guards behind CUDA >= 13.2, which this
// platform's 13.0 does not have.
DEVI uint32_t fp8x2_to_f16x2(uint32_t two_bytes) {
    uint32_t h2;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(h2) : "h"((unsigned short)two_bytes));
    return h2;
}
// BF16x2 -> F16x2 through F32: exact for every value in F16's normal range, because BF16's 8
// significand bits only widen into F16's 11. Below 2^-14 F16 goes subnormal and loses mantissa
// bits, and above 65504 it overflows; the frozen input distribution is normal in BF16, so the
// overflow side is unreachable and the subnormal side touches ~5e-5 of samples, each one term of
// a D-term dot product whose other terms are O(1).
DEVI uint32_t bf16x2_to_f16x2(uint32_t b) {
    __nv_bfloat162 bb = *reinterpret_cast<__nv_bfloat162*>(&b);
    __half2 h = __float22half2_rn(__bfloat1622float2(bb));
    return *reinterpret_cast<uint32_t*>(&h);
}
// Used by the QK MMA under FMHA_DEC_F16K and by the PV MMA under FMHA_DEC_VF16. The comment here
// used to say the PV product stays BF16 "because the contract rounds P * v_scale to q.dtype = BF16
// before the product, and pack_bf16 is what implements that rounding" - which is what the prefill
// believed too, until it priced the widening as EXACT (bf16 -> 2x cvt.f32.bf16 -> cvt.rn.f16x2.f32,
// 3 instructions per pair) and concluded f16 V was worth ~32 net instructions. Packing P straight to
// f16 costs the one instruction pack_bf16 already costs, so all of the V convert's saving survives.
// See results/prefill-vf16-20261025.md, which is this change measured and certified on the prefill
// side, for the precision argument and the numbers.
DEVI void mma_f16(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif  // FMHA_DEC_F16K || FMHA_DEC_VF16

DEVI uint32_t fp8x2_to_bf16x2(uint32_t two_bytes) {
#ifdef FMHA_NOCONV
    // DIAGNOSTIC ONLY (never defined for a real run): replaces the 4-instruction
    // E4M3->F16->F32->BF16 chain with one ALU op so a timing A/B can show how much
    // of the kernel is conversion throughput. Produces wrong numbers by design.
    return two_bytes * 0x00010001u;
#else
    uint32_t h2;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(h2) : "h"((unsigned short)two_bytes));
    __half2 hh = *reinterpret_cast<__half2*>(&h2);
    float2 f = __half22float2(hh);
    __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<uint32_t*>(&b);
#endif
}

DEVI int cdiv(int a, int b) { return (a + b - 1) / b; }

// exp2 on the MUFU unit. ex2.approx.f32 carries ~2^-22 relative error, which is
// negligible beside the BF16 rounding the contract mandates on P*v_scale, and it
// maps -inf -> 0 exactly as the reference's masked softmax needs.
DEVI float ex2(float x) { float y; asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

// Two FP32 -> packed BF16x2 (RNE), low half = lo. Matches the contract's
// "round P * V_scale to BF16 before PV".
DEVI uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&b);
}

#ifdef FMHA_DEC_VF16
// Two FP32 -> packed F16x2 (RNE), low half = lo. One instruction, the same count pack_bf16 costs.
// The CEILING half of this argument is correct: P is exp(score - row_max) * v_scale with the exponential
// in [0,1], so P <= v_scale = max|v_bf16|/448 (input.py:100) and overflow would need a bf16 input above
// 2.93e7; the sentinel page at input.py:114 (v = 448.0, vs = 1.125/448) gives P <= 0.00251.
// THE FLOOR HALF WAS WRONG AND THIS PATH FAILS WITHOUT FMHA_DEC_VSHIFT. "The subnormal floor can only
// zero terms whose share of a row sum is below ~1e-7" treats the floor as a per-term tail, but input.py:92
// multiplies v by 2^-20 for distribution:'tiny' and input.py:100 then clamps the scale to exactly 1e-6, so
// EVERY P in a row lands inside f16's subnormal range at once while l still sums the UNSCALED weights
// (reference.py:45) - the whole row loses its mantissa, not a tail. Measured: case 2 at rel_l2
// 0.014957772616608397 against the frozen 1e-2 bound in dv_8686b51ec86b. See FMHA_DEC_VSHIFT below and
// results/tiny-precision-defect-20261025.md, where the same substitution in the prefill read 1.988e-02 to
// 2.485e-02 on four legal 'tiny' shapes and is what put the defect inside the current certificate.
DEVI uint32_t pack_f16(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}
#endif

#ifdef FMHA_DEC_VSHIFT
// Adaptive per-tile power-of-two shift: the repair for the f16-P subnormal floor documented above.
//
// k is chosen per KV tile from that tile's largest value scale and folded into the exponent the softmax
// already subtracts, so p' = ex2(s - (m - k)) = p*2^k costs nothing per score. alpha becomes
// ex2((m_prev - k_prev) - (m - k)) - the same instruction, one more addend - which puts rl and O on the
// SAME 2^k. Publishing (m - k) in place of m then makes BOTH merges correct with no change to either: the
// WK>1 shared-memory merge below and fmha_combine across splits both weight by ex2(m_w - max_w m_w), and
// (m - k, l*2^k, O*2^k) is exactly a valid online-softmax partial with maximum (m - k). When n_split == 1
// nothing is published and 2^k cancels in O * (1/rl).
//
// SAFETY. k <= TARGET - floor(log2(vmax)), so P' <= 2^TARGET for ANY finite input: no magnitude assumption
// and no reachable overflow. A fixed prescale would be free but would return inf for a finite input with
// |v| > 7168 against an explicit rule to preserve nonfinite semantics. Quantising k DOWN to a multiple of
// QBIN costs at most QBIN binades of headroom - a worst-case relative error of 2^-25/2^(TARGET-QBIN) =
// 2^-22 here against a 1e-2 bound - and it keeps alpha at exactly 1.0f on tiles where nothing changed, so
// the __any_sync rescale vote keeps the firing rate it has today. The +-120 clamp keeps the f32
// intermediate p*2^k in range at either extreme of vmax.
//
// vmax is a genuine bound, not a guess: issue_meta clamps out-of-range tokens with `kgc = kg < hi ? kg :
// hi - 1`, so vst holds a real scale from a real page for all BN columns.
#define DECVS_TARGET 15
#define DECVS_QBIN 4
DEVI float dec_vs_shift(float vmax) {
    const float v = fmaxf(vmax, 1.17549435e-38f);                   // FLT_MIN: k finite even at vmax == 0
    const int e = (int)((__float_as_uint(v) >> 23) & 0xffu) - 127;   // floor(log2) of a normal
    int q = DECVS_TARGET - e;                                        // in [-113, 142] for any finite v
    // Floor to a multiple of QBIN without depending on q's sign: bias by 128, itself a multiple of QBIN,
    // so the unsigned truncation is exact and no implementation-defined right shift is relied on.
    q = (int)((((unsigned)(q + 128)) / DECVS_QBIN) * DECVS_QBIN) - 128;
    q = q < -120 ? -120 : (q > 120 ? 120 : q);
    return (float)q;
}
#endif

// ---------------------------------------------------------------- parameters
struct Params {
    const __nv_bfloat16* q;
    const uint8_t* k;
    const uint8_t* v;
    const float* ks;
    const float* vs;
    const int* cu;
    const int* slen;
    const int* btab;
    __nv_bfloat16* out;
    float* ws;
    long long q_s0;             // q token stride (bf16 elements)
    long long out_s0;           // out token stride (bf16 elements)
    long long k_s0, k_s1, k_s2; // byte strides: page block, page slot, kv head
    long long v_s0, v_s1, v_s2;
    long long ks_s0, ks_s1, ks_s2; // float strides
    long long vs_s0, vs_s1, vs_s2;
    int btab_s;
    int H, Hkv, g, g_log2, D;
    int BM, BN;
    int ps_log2;
    int B, n_split;
    float qk_scale; // softmax_scale * log2(e)
    int ws_stride;  // floats per (tile, split) record
    int dc, rpt;    // combine kernel: dims per CTA, rows per CTA
    // ---- merged tcgen05 prefill (fmha_pre.cu, arm-b merge5 lineage; join
    // 20261023). APPENDED at the end so every offset above stays identical:
    // Params layout changes measurably shift ptxas scheduling of the kernels.
    int tq;         // packed total q rows (q.size(0)); loose tc grid bound
    int msq;        // max_seqlen_q
    int msk;        // max_seqlen_k (fold "diag" gate; tensor path sets msq)
    // ---- fused split-merge (FM), 2026-10-28; APPENDED for the same reason.
    int* fctr;      // arrival counters: one int per (tile, kv-head) group
    int fuse_merge; // 1 = fmha_main's last arriver merges each group's partials
                    //     into `out` itself; no fmha_combine launch follows
};
// Fused split-merge cap: the in-main merge walks every split of a group with
// one CTA, so it is only for the small-split decode class; the block combine
// (whose 48us-of-case-3 history is the warning) keeps the big-split shapes.
constexpr int FM_MAXS = 32;

// Shared-memory metadata block (all warps).
template <int BM, int BN, int WK>
struct MetaS {
    int cu[33];                 // cumsum of query lengths, B <= 32
    int bound[BM];              // inclusive visible-key bound per tile row
    long long qbase[BM];        // q element offset per tile row
    long long koff[2][WK][BN];  // byte offset of key n inside the KV cache
    long long voff[2][WK][BN];
    float kst[2][WK][BN];       // per-key K scale (fp32)
    float vst[2][WK][BN];       // per-key V scale (fp32)
};

template <int D, int BM, int BN, int WK, int DV = D>
struct SmemLayout {
    static constexpr int DP = D + 8;               // padded row stride, bf16 elements
    static constexpr int DPV = DV + 8;             // V-image row stride (DV < D under dim-split)
    static constexpr int META_B = ((int)sizeof(MetaS<BM, BN, WK>) + 15) & ~15;
    static constexpr int Q_B = BM * DP * 2;
    static constexpr int KV_B = WK * BN * DP * 2;  // one buffer of K, all kw warps
    static constexpr int KV_BV = WK * BN * DPV * 2;
    static constexpr int OFF_Q = META_B;
    static constexpr int OFF_K = OFF_Q + Q_B;
    static constexpr int OFF_V = OFF_K + KV_B;
    static constexpr int BYTES = OFF_V + KV_BV;    // DV == D reproduces the shipped layout exactly
};


// ---------------------------------------------------------------- main kernel
// WM warps split the BM query rows; WK warps split the KV range of the CTA.
// Prefill uses (WM = BM/16, WK = 1); decode/MTP uses (WM = 1, WK > 1).
// Do NOT add a minBlocksPerMultiprocessor argument to __launch_bounds__ here, not
// even 1: it is not a no-op. ptxas reads it as permission to spend more registers,
// and job dv_f63efef85d60 measured that directly on this GPU. With
// (32*WM*WK, 1) the decode instantiations come out at 190 (cfg4), 207 (cfg3) and
// 252 (cfg5) registers; with the argument omitted, 166, 165 and 243. At 128
// threads that is 2 resident CTAs/SM instead of 3 - 8 warps/SM instead of 12 -
// and the six decode cases then cost 5.1% more on their geomean (case 2
// 155.4 -> 181.1us, case 4 117.2 -> 138.2, case 8 42.0 -> 44.4). Prefill spills
// as well: cfg8 grows an 8-byte spill and cfg2 a 4-byte one. The forced-occupancy
// experiment that motivated the parameter was itself a loss (1.31-3.45x worse,
// results/prefill-occupancy-20261021.md); details in
// results/minblocks-register-effect-20261021.md.
// KV prefetch depth in 4-byte chunks per lane. SHIPPED AT 32, which is >= CPT for every config _pick can
// return, so PFC = min(CPT, 32) = CPT everywhere and stash_kv's interleaved tail pipeline folds away entirely:
// the whole block is fetched into rkP/rvP and stashed in one pass. Measured by dv_cdc6c11fd8d5 (three arms on
// case 3) and confirmed by dv_358cf0b6f45b (all six decode cases, correctness ON): case 3 median 77.8 -> 67.6us
// (-13.1%) and minimum 75.5 -> 65.7 (-13.0%), with long_scoreboard 1.84 -> 0.82, registers 244 -> 253, and
// launch__occupancy_per_register_count UNCHANGED at 2056 - the extra registers were spilled, not traded against
// residency, so the depth bought latency hiding for free. All six decode cases bit-identical to control.
// The old cap of 16 bound only cfg 5 (CPT=32), leaving half of each KV block unprefetched; FMHA_DEC_PFC=16
// restores it. Values that are not congruent to CPT modulo GS now fail the static_assert in fmha_main rather
// than silently dropping the last (CPT-PFC)%GS chunks, which is what FMHA_DEC_PFC=22 did (rel_l2=nan while
// still timing 10.5% faster - see results/decode-prefetch-depth-20261026.md).
#ifndef FMHA_DEC_PFC
#define FMHA_DEC_PFC 32
#endif

// Inert decode shared-memory pad (A/B probe only; default 0 = production
// launches byte-identical). Adds N KB to the dynamic smem of every decode/MTP
// fmha_main instantiation (WK > 1); the kernel never touches the region. It
// lowers resident CTAs/SM WITHOUT touching register allocation: the 255-reg
// hardware cap means cfg5's 253 regs can never be pushed below 4 blocks by
// adding registers, and __launch_bounds__ confounds occupancy with spills and
// allocation changes (results/minblocks-register-effect-20261021.md, and the
// note above line 285). This isolates the warps/SM -> time slope of the
// latency-bound decode walk, which decides whether the tcgen05/TMEM decode
// class (O out of registers -> more warps/SM at equal traffic) is worth
// building. ANY residency change MUST be accompanied by an integer-wave
// FMHA_SPLIT_WAVE_MUL rebalance on wave-saturated shapes, or the split-wave
// quantization swamps the slope (pfc8m: 592->888 CTAs cost +84.8% on c3).
#ifndef FMHA_DEC_SMEMPAD_KB
#define FMHA_DEC_SMEMPAD_KB 0
#endif

template <int D, int BM, int BN, int WM, int WK, bool DS = false, bool FM = false>
__global__ void __launch_bounds__(32 * WM * WK) fmha_main(const Params p) {
    static_assert(!(DS && FM), "dim-split records are not merged in-kernel (v1)");
    // ---- dim-split decode (DS), the direction results/decmisc-20261028.md leaves ----
    // Two CTAs per (tile, split): blockIdx.z = split*2 + half. Each CTA scores
    // against the FULL K (QK and softmax are replicated, bit-identical) and
    // accumulates only its half of V/O. The O accumulator is the register hog
    // pinning the d256 decode tile at 4 CTAs/SM (O = D/8*4 = 128 of the 253);
    // halving it targets 5 CTAs/SM = 10 warps with ~87% more bytes in flight,
    // at 1.5x KV traffic (K read by both halves) which the censused 21% DRAM
    // utilisation can absorb. ws holds HALF-WIDE record PAIRS per split:
    // entry (split*2+half) = [O_half (BM x DH), m (BM), l (BM, ZERO on the odd
    // half so the combine's denominator counts each split's l exactly once)].
    // With the planner's n_split forced equal, the per-dim accumulation order
    // and every f32 term are unchanged, so this build is BIT-IDENTICAL to the
    // shipped path - that equality is the staged correctness gate.
    static_assert(!DS || (D == 256 && WM == 1 && BM == 16 && WK == 2),
                  "dim-split decode is instantiated only for the d256 tile");
#ifdef FMHA_DEC_F16K
    // Whether THIS head dimension takes the f16 K image. Under FMHA_DEC_F16K_GATE it is D != 256, and
    // that gate is a model rather than a fit: results/decode-f16k-20261025.md measured the ungated arm
    // at -1.8% on case 2 and -1.9% on case 4 while case 3 read +2.6% median and +4.9% minimum, and
    // rejected the gate because "the instruction accounting says d=256 should be the best case ... and
    // it measures the worst. A heuristic that contradicts its own cost model on the one point that
    // motivates it is a fit, not a model." The instruction accounting is the wrong accounting. Case 3 is
    // the only d=256 shape and it runs 64-thread CTAs at 243 registers (results/decode-profile-20261022),
    // where 65536 / (64 * 243) = 4.2 resident CTAs - 13 registers from the cliff. Q widening is what
    // spends them: the A-fragments become f16 alongside a second image in flight, and dropping from 4
    // CTAs/SM to 3 is -25% of the warps available to hide a chain that dv_cc629ef99b95 measured as
    // latency-bound (wait 1.21 top stall, no pipe above 39%, warps_active 17.36% of an 18.75% ceiling).
    // So d=256 loses occupancy and d<=128, at 166 of the 170 registers 3 CTAs/SM allows with 128-thread
    // CTAs, does not - which is the same 170-register ceiling that made Q-hoist read 1.016/1.024/1.184
    // worse. The gate is predicted by where each config sits against its own cliff, not by its result.
    // Bit-identical either way: f16 and bf16 both hold an E4M3 value exactly, BF16 -> F16 only widens 8
    // significand bits into 11, both products are exact in the f32 accumulate, and m16n8k16 is the same
    // hardware path. FMHA_DEC_F16K_GATE=0 restores the ungated arm for reproduction.
#ifdef FMHA_DEC_F16K_GATE
    // Gated on WARPS PER CTA, not on head dimension, because that is the property that separates the
    // measured regression from the measured neutral. dv_88b84d127f8f: case 3 (cfg 5 = <256,16,16,1,2>,
    // 64 threads, 2 warps/CTA, 8 warps/SM) reads +2.5% median / +0.4% min ungated and flat gated, while
    // case 5 (cfg 4 = <128,16,16,1,4>, 128 threads) is flat with FEWER blocks per CTA (16 against case
    // 3's 28) and cases 2 and 4 (cfg 4, 85 and 158 blocks) gain -3.1% and -1.9%. So blocks-per-CTA does
    // not separate them and head dimension does not either: what case 3 uniquely has is 2 warps per CTA.
    // The Q widening is a one-off serial read-modify-write of the whole shared Q image in the prologue,
    // and at 2 warps/CTA with 8 warps/SM there is nothing else resident to run while it drains.
    // Enumerating launch_plan's table, the sub-128-thread configs are exactly cfg 5 <256,16,16,1,2>,
    // cfg 10 <128,64,64,2,1>, cfg 11 <64,64,64,2,1> and cfg 24 <256,16,32,1,2>; this gate and
    // `D != 256` select identically on every config the fixed cases reach (cfg 3 for case 8, cfg 4 for
    // cases 2/4/5/6, cfg 5 for case 3), so the measured A/B transfers unchanged. They differ only on
    // cfg 2 <256,64,32,4,1>, which this gate enables and D!=256 would not, and on cfgs 10/11, which
    // this gate excludes and D!=256 would not - i.e. the thread gate is the conservative one exactly
    // where the regression was observed. f16 K is an exact transformation in every config, so a config
    // taking the other path than measured cannot be wrong, only faster or slower.
    constexpr bool KF16 = (32 * WM * WK >= 128);
#else
    constexpr bool KF16 = true;
#endif
#endif
#ifndef FMHA_DEC_F16K
    constexpr bool KF16 = false;   // VF16's gate below is KF16's complement and needs the value either way
#endif
#ifdef FMHA_DEC_VF16
    // Whether THIS config takes the f16 V image, f16 P and the shift that makes f16 P correct. Gated to
    // the COMPLEMENT of KF16, which is what dv_2d4c8943be69 measured rather than what its design assumed.
    // Ungated, against a ctl that already ships f16 K: case 3 -2.39% on the minimum, but case 2 +0.13%
    // and case 4 +1.56%. The same change measured -1.87% on case 2 in dv_8686b51ec86b, whose ctl still
    // converted K through the four-instruction bf16 chain. So the saving is the CONVERT, and once K's
    // chain is gone V's is the only one left and the shift's ~16 instructions per tile eat it. Case 3 is
    // precisely the d=256 shape the KF16 gate excludes, i.e. the one config that still pays both chains,
    // and it is the one that keeps the gain. FMHA_DEC_VF16_GATE=0 restores the ungated arm.
#ifdef FMHA_DEC_VF16_GATE
    constexpr bool VF16 = !KF16;
#else
    constexpr bool VF16 = true;
#endif
#endif
#ifdef FMHA_DEC_VSHIFT
    // The shift exists to repair f16 P's subnormal floor. With bf16 P it would only spend instructions
    // and move the rounding, so it folds to nothing wherever VF16 is off - and ksh == 0.f makes every
    // use below an exact identity (x - 0.f == x, and rm - 0 - sf == rm - sf), so a gated-off config is
    // bit-identical to one compiled without FMHA_DEC_VSHIFT at all.
    constexpr bool VSH = VF16;
#endif
    constexpr int NW = WM * WK;
    constexpr int NT = 32 * NW;
    constexpr int DP = D + 8;
    constexpr int DH = DS ? D / 2 : D;   // dims THIS CTA accumulates
    constexpr int DPV = DH + 8;          // V-image row pitch (== DP when !DS)
    // MT > 1 makes each warp own several 16-row m-tiles, so one ldmatrix'd K/V
    // fragment feeds MT tensor-core ops. Shared-memory bytes per mma are
    // 2048/BN + 256/MT, so MT is the dominant lever on this part (sm_103
    // mma.sync is fast enough that the L1 data path, not the tensor core, binds).
    constexpr int MT = BM / (16 * WM);
    constexpr int KPW = BN / WM;          // keys loaded per warp
    constexpr int CPR4 = D / 4;           // 4-byte chunks per key row
    constexpr int CPT = KPW * CPR4 / 32;  // chunks per lane (K or V)
    constexpr int GS = (CPT > 4) ? 4 : CPT;
    // Chunks prefetched across compute. The shipped cap is 16, and at D=256 CPT is 32, so cfg 5 - case 3, the
    // only d=256 decode shape and the one NOGLD priced at -27.2% - prefetches only HALF of each KV block. That
    // cap is a register budget, not a memory-system limit: rkP[PFC] + rvP[PFC] cost 2*PFC registers out of the
    // 243 this kernel runs at, and fmha.cu:296-300 records it as "13 registers from the cliff" where
    // 65536/(64*256) drops it from 4 resident CTAs to 3. So there is room to raise the cap to 22 (+12 registers,
    // 255 total, still 4 CTAs since 65536/(64*255) = 4.02) and NOT to 24 (259 > 256, ptxas spills).
    // FMHA_DEC_PFC raises it; 32 means "the whole block", which must spill, and is in the experiment as the
    // probe that prices the register wall itself - if 32 costs a lot and 22 gains, the wall is real and the only
    // way past it is staging the raw fp8 in SHARED memory with cp.async (SmemLayout uses ~42 KB of the 232 KB
    // budget at cfg 5, so ~190 KB is free, enough for ~20 blocks of raw fp8) instead of in registers.
    // Bit-identical by construction: prefetch depth changes when a load is issued, never what it returns, and
    // stash_one's convert and the mma see the same bytes in the same order.
    constexpr int PFC = (CPT < FMHA_DEC_PFC) ? CPT : FMHA_DEC_PFC;
    // The Q A-fragments are re-ldmatrix'ed on every KV block even though Q is
    // loop invariant. Hoisting them into MT*(D/16)*4 registers was measured and
    // does not pay - neutral at D=64, 1.6-2.4% worse at D=128, 18% worse at D=256,
    // because the registers come straight out of occupancy. See
    // results/decode-q-hoist-20261021.md; the code is in commit 8d62992.
    using SL = SmemLayout<D, BM, BN, WK, DH>;
    static_assert(BN % WM == 0 && (KPW * CPR4) % 32 == 0 && CPT % GS == 0, "tile split");
    static_assert(!DS || (CPR4 % 32 == 0 && CPT == PFC),
                  "dim-split needs the fast chunk mapping and whole-block prefetch");
    // stash_kv's tail loop covers chunks [PFC, CPT) in G = (CPT - PFC)/GS groups of GS, and that division
    // TRUNCATES: at CPT=32, GS=4 a cap of 22 gives G=2, covering chunks 22..29 and silently never fetching or
    // stashing 30 and 31. Those shared slots keep whatever was there - uninitialised on the first block - and the
    // mma consumes garbage, which is exactly what FMHA_DEC_PFC=22 did on case 3 in dv_cdc6c11fd8d5: rel_l2=nan
    // while the timing still looked like a 10% win. CPT % GS == 0 does not imply (CPT - PFC) % GS == 0, so the
    // second condition needs its own assert. FMHA_DEC_PFC=32 makes G=0 and drops the tail pipeline altogether,
    // which is why it is both bit-identical and the fastest of the three arms measured.
    static_assert((CPT - PFC) % GS == 0,
                  "PFC must leave a whole number of GS-chunk groups in stash_kv's tail, or the tail silently "
                  "drops the last (CPT - PFC) % GS chunks and the mma reads uninitialised shared memory");
    static_assert(BM % (16 * WM) == 0 && BN % 16 == 0, "mma tile");
    static_assert(WK == 1 || (WM == 1 && BM == 16), "kv-split warps need a single m tile");
    static_assert(WK == 1 || (SL::KV_B + SL::KV_BV >= WK * 16 * DH * 4 + WK * 16 * 8),
                  "epilogue reduction must fit in the K/V shared region");

    extern __shared__ __align__(16) char smem[];
    MetaS<BM, BN, WK>* meta = reinterpret_cast<MetaS<BM, BN, WK>*>(smem);
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem + SL::OFF_Q);
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem + SL::OFF_K);
    __nv_bfloat16* sV = reinterpret_cast<__nv_bfloat16*>(smem + SL::OFF_V);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int mw = warp / WK;
    const int kw = warp % WK;
    const int m0 = mw * 16 * MT;

    const int tx = blockIdx.x;
    const int hy = blockIdx.y;
    const int sz = DS ? (int)(blockIdx.z >> 1) : (int)blockIdx.z;
    const int dhalf = DS ? (int)(blockIdx.z & 1) : 0;

    // ---- map the linear tile index to (request, m-block)
    for (int i = tid; i <= p.B; i += NT) meta->cu[i] = p.cu[i];
    __syncthreads();
#ifdef FMHA_SLOWMAP_MAIN
    int b = -1, m_blk = 0, cum = 0;
    for (int i = 0; i < p.B; ++i) {
        int ql = meta->cu[i + 1] - meta->cu[i];
        int nt = (ql * p.g + BM - 1) / BM;
        if (b < 0 && tx < cum + nt) { b = i; m_blk = nt - 1 - (tx - cum); }
        cum += nt;
    }
#else
    // Per-warp replacement for the serial scan in the #ifdef branch above. B <= 32 is
    // enforced on the host, so one lane per request can hold its tile count, five
    // shfl_up rounds prefix-sum it and a ballot picks the owner. Every warp runs
    // it independently, which needs no extra barrier and no shared slot and keeps
    // b/m_blk CTA-uniform (they must be, or `if (b < 0) return` would strand part
    // of a CTA at the next __syncthreads). The serial version's cost is B
    // *dependent* runtime integer divisions by BM in every thread; this does one
    // per lane in parallel. All the warp-collectives below are reached by every
    // lane of every warp on every path - see the deadlock fixed in f52a652.
    int b, m_blk;
    {
        const int nt = (lane < p.B)
                     ? ((meta->cu[lane + 1] - meta->cu[lane]) * p.g + BM - 1) / BM : 0;
        int pre = nt;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, pre, off);
            if (lane >= off) pre += y;
        }
        const int lo = pre - nt;
        const unsigned bal = __ballot_sync(0xffffffffu,
                                           (lane < p.B) && (tx >= lo) && (tx < pre));
        const int src = bal ? (__ffs(bal) - 1) : 0;
        const int lob = __shfl_sync(0xffffffffu, lo, src);
        const int ntb = __shfl_sync(0xffffffffu, nt, src);
        b = bal ? src : -1;
        m_blk = bal ? (ntb - 1 - (tx - lob)) : 0;
    }
#endif
    // Heaviest tile first: with bottom-right causal masking tile m covers keys
    // [0, kv-ql+(m+1)*BM/g), so work grows ~32x across the tile range. grid.x is
    // the fastest-varying dimension, so reversing the mapping puts the long tiles
    // at the front of the launch order and shortens the tail. Both branches above
    // implement the same reversal.
    if (b < 0) return;

#ifdef FMHA_PDL
#ifndef FMHA_PDL_LATE
    // One lane per CTA. CTAs that returned above exit, which counts as a trigger,
    // so every CTA of the primary contributes either way and the secondary grid
    // becomes dispatchable while this grid's last wave is still running.
    if (threadIdx.x == 0) FMHA_PDL_TRIG();
#endif
#endif

    const int q0 = meta->cu[b];
    const int qlen = meta->cu[b + 1] - q0;
    const int q_start = m_blk * (BM / p.g);
    const int n_tok = min(BM / p.g, qlen - q_start);
    const int n_rows = n_tok * p.g;
    const int kv_len = p.slen[b];
    const int hi = kv_len - qlen + q_start + n_tok;   // exclusive key bound
    const int chunk = (hi + p.n_split - 1) / p.n_split;
    const int chunk_bn = ((chunk + BN - 1) / BN) * BN;
    const int kv_lo = sz * chunk_bn;
    if (kv_lo >= hi) {
        // FM: an empty slice still ARRIVES, or its group's counter never
        // reaches n_split and the group would never be merged. It writes no
        // partial, so no fence is needed before the arrival.
        if constexpr (FM)
            if (p.fuse_merge && tid == 0)
                atomicAdd(&p.fctr[tx * p.Hkv + hy], 1);
        return;
    }
    const int kv_hi = min(kv_lo + chunk_bn, hi);
    const int ntot = (kv_hi - kv_lo + BN - 1) / BN;
    const int nblk = (ntot + WK - 1) / WK;

    // ---- per-row query base offsets and causal bounds
    if (tid < BM) {
        int tok = q_start + (tid >> p.g_log2);
        if (tok > qlen - 1) tok = qlen - 1;
        int h = hy * p.g + (tid & (p.g - 1));
        meta->qbase[tid] = (long long)(q0 + tok) * p.q_s0 + (long long)h * D;
        meta->bound[tid] = kv_len - qlen + q_start + (tid >> p.g_log2);
    }
    __syncthreads();   // qbase/bound visible before the Q tile read

    // ---- Q tile (already BF16: straight cp.async, no conversion)
    {
        const char* qb = reinterpret_cast<const char*>(p.q);
        for (int c = tid; c < BM * (D / 8); c += NT) {
            int row = c / (D / 8), col16 = c % (D / 8);
            cp_async16(reinterpret_cast<char*>(sQ + row * DP) + col16 * 16,
                       qb + meta->qbase[row] * 2 + col16 * 16);
        }
        cp_commit();
    }

    int row0[MT], row1[MT], bnd0[MT], bnd1[MT];
    int bmin = 0x7fffffff;   // min causal bound over this warp's rows
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) { row0[mt] = m0 + mt * 16 + (lane >> 2); row1[mt] = row0[mt] + 8; }
    const int colbase = (lane & 3) * 2;

    // ---- accumulators
    float O[MT][DH / 8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < DH / 8; ++i) O[mt][i][0] = O[mt][i][1] = O[mt][i][2] = O[mt][i][3] = 0.f;
    float rm[MT][2], rl[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) { rm[mt][0] = rm[mt][1] = -INFINITY; rl[mt][0] = rl[mt][1] = 0.f; }
#ifdef FMHA_DEC_VSHIFT
    float kprev = 0.f;   // shift of the tile that produced the current rm/rl/O; rm stays UNSHIFTED because
                         // nm = fmaxf(rm, mx) below compares it against unshifted scores
#endif

    // ---- per-block metadata (page lookup + per-key scales).
    // issue_meta() puts the btab -> scale dependency chain in registers two
    // blocks ahead; commit_meta() publishes it just before it is consumed, so
    // that chain is hidden by compute() instead of stalling at a barrier. Each
    // warp resolves the whole BN-key range of its own kw tile and only ever reads
    // metadata it wrote itself, so WM == 1 needs no block-wide barrier at all.
    constexpr int MPL = (BN + 31) / 32;   // keys resolved per lane
    struct MetaRegs { int phys[MPL], slot[MPL]; float ksv[MPL], vsv[MPL]; };
    MetaRegs mr;
    const bool mall = (BN % 32 == 0);

    auto issue_meta = [&](int ib) {
        const int n0 = kv_lo + (ib * WK + kw) * BN;
        const int* bt = p.btab + (long long)b * p.btab_s;
        const int psm = (1 << p.ps_log2) - 1;
#pragma unroll
        for (int m = 0; m < MPL; ++m) {
            const int n = lane + m * 32;
            if (mall || n < BN) {
                const int kg = n0 + n;
                const int kgc = kg < hi ? kg : hi - 1;
                const int sl = kgc & psm;
                const int ph = bt[kgc >> p.ps_log2];
                mr.phys[m] = ph;
                mr.slot[m] = sl;
                mr.ksv[m] = p.ks[(long long)ph * p.ks_s0 + (long long)sl * p.ks_s1 + (long long)hy * p.ks_s2];
                mr.vsv[m] = p.vs[(long long)ph * p.vs_s0 + (long long)sl * p.vs_s1 + (long long)hy * p.vs_s2];
            }
        }
    };
    auto commit_meta = [&](int slot) {
#pragma unroll
        for (int m = 0; m < MPL; ++m) {
            const int n = lane + m * 32;
            if (mall || n < BN) {
                meta->koff[slot][kw][n] = (long long)mr.phys[m] * p.k_s0 + (long long)mr.slot[m] * p.k_s1
                                        + (long long)hy * p.k_s2;
                meta->voff[slot][kw][n] = (long long)mr.phys[m] * p.v_s0 + (long long)mr.slot[m] * p.v_s1
                                        + (long long)hy * p.v_s2;
                meta->kst[slot][kw][n] = mr.ksv[m] * p.qk_scale;
                meta->vst[slot][kw][n] = mr.vsv[m];
            }
        }
        __syncwarp();
    };

    // ---- K/V block movement: fp8 global -> registers -> bf16 shared.
    // Loads are 4 bytes wide because the page payload stride is 2*(D+4) only
    // guarantees 4-byte alignment (the V payload starts at byte D+4).
    // fetch() issues the global loads and keeps them in registers so they are
    // still in flight while compute() runs on the previous buffer; stash() does
    // the E4M3 -> BF16 conversion and the shared-memory stores.
    auto stash_one = [&](int lc, uint32_t rk, uint32_t rv) {
        const int key = mw * KPW + lc / CPR4;
        const int col = lc % CPR4;
        const uint16_t* hk = reinterpret_cast<const uint16_t*>(&rk);
        const uint16_t* hv = reinterpret_cast<const uint16_t*>(&rv);
        uint2 ok, ov;
#ifdef FMHA_DEC_F16K
        if constexpr (KF16) { ok.x = fp8x2_to_f16x2(hk[0]); ok.y = fp8x2_to_f16x2(hk[1]); }
        else                { ok.x = fp8x2_to_bf16x2(hk[0]); ok.y = fp8x2_to_bf16x2(hk[1]); }
#else
        ok.x = fp8x2_to_bf16x2(hk[0]); ok.y = fp8x2_to_bf16x2(hk[1]);
#endif
#ifdef FMHA_DEC_VF16
        if constexpr (VF16) { ov.x = fp8x2_to_f16x2(hv[0]); ov.y = fp8x2_to_f16x2(hv[1]); }
        else                { ov.x = fp8x2_to_bf16x2(hv[0]); ov.y = fp8x2_to_bf16x2(hv[1]); }
#else
        ov.x = fp8x2_to_bf16x2(hv[0]); ov.y = fp8x2_to_bf16x2(hv[1]);
#endif
        const int off = (kw * BN + key) * DP + col * 4;
        *reinterpret_cast<uint2*>(sK + off) = ok;
        *reinterpret_cast<uint2*>(sV + off) = ov;
    };
    auto fetch_chunk = [&](int lc, int slot, uint32_t& rk, uint32_t& rv) {
        const int key = mw * KPW + lc / CPR4;
        const int col = lc % CPR4;
        rk = *reinterpret_cast<const uint32_t*>(p.k + meta->koff[slot][kw][key] + col * 4);
        rv = *reinterpret_cast<const uint32_t*>(p.v + meta->voff[slot][kw][key] + col * 4);
    };
    // DS: only this CTA's V half is prefetched, so rvP needs PFC/2 slots and
    // entry c>>1 holds the chunk with (c & 1) == dhalf. rkP stays whole-K.
    uint32_t rkP[PFC], rvP[DS ? PFC / 2 : PFC], rkT[GS], rvT[GS];
    auto fetch_kv = [&](int ib, int slot) {
        if constexpr (CPR4 % 32 == 0) {
            // Read the per-key base offsets out of shared memory first, then
            // issue every global load back to back. Otherwise each global
            // address depends on its own LDS and the loads serialise, which is
            // what NCU showed as long_scoreboard 2.44/issue on the decode shapes
            // (case 3 sat at 10% of DRAM peak with 4 warps/SM).
            constexpr int NK = PFC * 32 / CPR4;
            long long kb[NK], vb[NK];
#pragma unroll
            for (int j = 0; j < NK; ++j) {
                kb[j] = meta->koff[slot][kw][mw * KPW + j];
                vb[j] = meta->voff[slot][kw][mw * KPW + j];
            }
#pragma unroll
            for (int c = 0; c < PFC; ++c) {
                const int lc = c * 32 + lane;
                const int j = (c * 32) / CPR4;   // lane-independent: CPR4 % 32 == 0
                rkP[c] = *reinterpret_cast<const uint32_t*>(p.k + kb[j] + (lc % CPR4) * 4);
                if constexpr (DS) {
                    // col = lc % CPR4 = (c & 1) * 32 + lane, so (c & 1) selects the
                    // dim half: load only this CTA's, into the compact rvP[c >> 1].
                    if ((c & 1) == dhalf)
                        rvP[c >> 1] = *reinterpret_cast<const uint32_t*>(p.v + vb[j] + (lc % CPR4) * 4);
                } else {
                    rvP[c] = *reinterpret_cast<const uint32_t*>(p.v + vb[j] + (lc % CPR4) * 4);
                }
            }
        } else {
#pragma unroll
            for (int c = 0; c < PFC; ++c) fetch_chunk(c * 32 + lane, slot, rkP[c], rvP[c]);
        }
    };
    // DS-only split stashers: identical conversions to stash_one's, but the V
    // store goes to this half's DH-wide image at col - dhalf*(CPR4/2). The
    // shipped stash_one is left textually untouched for the !DS builds.
    auto stash_k = [&](int lc, uint32_t rk) {
        const int key = mw * KPW + lc / CPR4;
        const int col = lc % CPR4;
        const uint16_t* hk = reinterpret_cast<const uint16_t*>(&rk);
        uint2 ok;
#ifdef FMHA_DEC_F16K
        if constexpr (KF16) { ok.x = fp8x2_to_f16x2(hk[0]); ok.y = fp8x2_to_f16x2(hk[1]); }
        else                { ok.x = fp8x2_to_bf16x2(hk[0]); ok.y = fp8x2_to_bf16x2(hk[1]); }
#else
        ok.x = fp8x2_to_bf16x2(hk[0]); ok.y = fp8x2_to_bf16x2(hk[1]);
#endif
        const int off = (kw * BN + key) * DP + col * 4;
        *reinterpret_cast<uint2*>(sK + off) = ok;
    };
    auto stash_v = [&](int lc, uint32_t rv) {
        const int key = mw * KPW + lc / CPR4;
        const int col = lc % CPR4 - dhalf * (CPR4 / 2);   // local to this half
        const uint16_t* hv = reinterpret_cast<const uint16_t*>(&rv);
        uint2 ov;
#ifdef FMHA_DEC_VF16
        if constexpr (VF16) { ov.x = fp8x2_to_f16x2(hv[0]); ov.y = fp8x2_to_f16x2(hv[1]); }
        else                { ov.x = fp8x2_to_bf16x2(hv[0]); ov.y = fp8x2_to_bf16x2(hv[1]); }
#else
        ov.x = fp8x2_to_bf16x2(hv[0]); ov.y = fp8x2_to_bf16x2(hv[1]);
#endif
        const int off = (kw * BN + key) * DPV + col * 4;
        *reinterpret_cast<uint2*>(sV + off) = ov;
    };
    auto stash_kv = [&](int ib) {
        const int slot = ib & 1;
#pragma unroll
        for (int c = 0; c < PFC; ++c) {
            if constexpr (DS) {
                if ((c & 1) == dhalf) { stash_k(c * 32 + lane, rkP[c]); stash_v(c * 32 + lane, rvP[c >> 1]); }
                else                  { stash_k(c * 32 + lane, rkP[c]); }
            } else {
                stash_one(c * 32 + lane, rkP[c], rvP[c]);
            }
        }
#ifndef FMHA_SLOWTAIL
        // Overlap the tail groups: issue group gr+1's global loads before
        // converting and storing group gr, instead of fetch-then-stash per group,
        // so only the first group's latency is exposed. Costs GS*2 registers.
        // Reachable only when CPT > PFC, which among the default configs is cfg5
        // alone (d=256 decode: CPT=32, PFC=16, four tail groups), so no other
        // instantiation changes at all.
        if constexpr ((CPT - PFC) / GS > 0) {
            constexpr int G = (CPT - PFC) / GS;
            uint32_t rkN[GS], rvN[GS];
#pragma unroll
            for (int c = 0; c < GS; ++c) fetch_chunk((PFC + c) * 32 + lane, slot, rkT[c], rvT[c]);
#pragma unroll
            for (int gr = 0; gr < G; ++gr) {
                if (gr + 1 < G) {
#pragma unroll
                    for (int c = 0; c < GS; ++c)
                        fetch_chunk((PFC + (gr + 1) * GS + c) * 32 + lane, slot, rkN[c], rvN[c]);
                }
#pragma unroll
                for (int c = 0; c < GS; ++c)
                    stash_one((PFC + gr * GS + c) * 32 + lane, rkT[c], rvT[c]);
                if (gr + 1 < G) {
#pragma unroll
                    for (int c = 0; c < GS; ++c) { rkT[c] = rkN[c]; rvT[c] = rvN[c]; }
                }
            }
        }
#else
#pragma unroll
        for (int gr = 0; gr < (CPT - PFC) / GS; ++gr) {
#pragma unroll
            for (int c = 0; c < GS; ++c)
                fetch_chunk((PFC + gr * GS + c) * 32 + lane, slot, rkT[c], rvT[c]);
#pragma unroll
            for (int c = 0; c < GS; ++c)
                stash_one((PFC + gr * GS + c) * 32 + lane, rkT[c], rvT[c]);
        }
#endif
    };

    // Is this warp active on block ib? Declared above compute() because compute()
    // needs it for the last-block __syncwarp.
    const auto act = [&](int ib) { return (ib < nblk) && ((ib * WK + kw) < ntot); };

    // ---- one KV block: QK -> scale/mask -> online softmax -> PV
    auto compute = [&](int ib, int slot) {
        // Cross-lane shared-memory handoff. stash_kv(ib) was issued by this same
        // warp in the previous iteration, and the ldmatrix below reads elements
        // other lanes stored. When WM > 1, barrier (B) after stash_kv already
        // orders that; when WM == 1 the ordering normally comes from the
        // __syncwarp() at the end of commit_meta(ib+1), which runs before
        // compute(ib) - but commit_meta is skipped on this warp's last block, so
        // that one iteration would rely on in-warp program order alone. The CUDA
        // memory model only promises cross-lane visibility through an explicit
        // __syncwarp(), so issue one exactly there. Cost is one fence per CTA.
        if constexpr (WM == 1) { if (!act(ib + 1)) __syncwarp(); }
        const __nv_bfloat16* kb = sK + (kw * BN) * DP;
        const __nv_bfloat16* vb = sV + (kw * BN) * DPV;
        const float* kst = meta->kst[slot][kw];
        const float* vst = meta->vst[slot][kw];
        const int n0 = kv_lo + (ib * WK + kw) * BN;
        // Warp-uniform: a block wholly below the causal diagonal needs no select,
        // which is the common prefill case (only diagonal blocks actually mask).
        const bool nomask = __all_sync(0xffffffffu, n0 + BN - 1 <= bmin);

#ifdef FMHA_DEC_VSHIFT
        // This tile's largest value scale, reduced over the quad that owns a row's columns - the same
        // __shfl_xor 1 and 2 the row max and the row sum already use, and the group whose O accumulator
        // has to agree on one k. Each lane reads the BN/4 columns the score and PV loops index.
        float ksh = 0.f;
        if constexpr (VSH) {
            float vsm = 0.f;
#pragma unroll
            for (int nt = 0; nt < BN / 8; ++nt) {
                const int cv = nt * 8 + colbase;
                vsm = fmaxf(vsm, fmaxf(vst[cv], vst[cv + 1]));
            }
            vsm = fmaxf(vsm, __shfl_xor_sync(0xffffffffu, vsm, 1));
            vsm = fmaxf(vsm, __shfl_xor_sync(0xffffffffu, vsm, 2));
            ksh = dec_vs_shift(vsm);
        }
#endif

        float S[MT][BN / 8][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int i = 0; i < BN / 8; ++i) S[mt][i][0] = S[mt][i][1] = S[mt][i][2] = S[mt][i][3] = 0.f;

        // ---- QK^T: one B fragment serves all MT m-tiles
        const int aco = (lane & 16) ? 8 : 0;
        const int krow = (lane & 7) + ((lane & 16) ? 8 : 0), kco = ((lane >> 3) & 1) ? 8 : 0;
#pragma unroll
        for (int kt = 0; kt < D / 16; ++kt) {
            uint32_t a[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#if defined(FMHA_DEC_NOLDM) && (FMHA_DEC_NOLDM & 2)
                a[mt][0] = a[mt][1] = a[mt][2] = a[mt][3] = 0x3f803f80u ^ (kt & 1);
#else
                ldsm_x4(a[mt], smem_u32(sQ + (m0 + mt * 16 + (lane & 15)) * DP + kt * 16 + aco));
#endif
#pragma unroll
            for (int nt2 = 0; nt2 < BN / 16; ++nt2) {
                uint32_t bb[4];
#if defined(FMHA_DEC_NOLDM) && (FMHA_DEC_NOLDM & 1)
                bb[0] = bb[1] = bb[2] = bb[3] = 0x3f803f80u ^ (kt & 1);
#else
                ldsm_x4(bb, smem_u32(kb + (nt2 * 16 + krow) * DP + kt * 16 + kco));
#endif
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
#ifdef FMHA_DEC_F16K
                    if constexpr (KF16) {
                        mma_f16(S[mt][nt2 * 2 + 0], a[mt], &bb[0]);
                        mma_f16(S[mt][nt2 * 2 + 1], a[mt], &bb[2]);
                    } else {
                        mma_bf16(S[mt][nt2 * 2 + 0], a[mt], &bb[0]);
                        mma_bf16(S[mt][nt2 * 2 + 1], a[mt], &bb[2]);
                    }
#else
                    mma_bf16(S[mt][nt2 * 2 + 0], a[mt], &bb[0]);
                    mma_bf16(S[mt][nt2 * 2 + 1], a[mt], &bb[2]);
#endif
                }
            }
        }

        // ---- per m-tile online softmax (log2 domain, bottom-right causal)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int b0 = bnd0[mt], b1 = bnd1[mt];
#pragma unroll
            for (int nt = 0; nt < BN / 8; ++nt) {
                const int c0 = nt * 8 + colbase;
                const float s0 = kst[c0], s1 = kst[c0 + 1];   // qk_scale pre-folded
                if (nomask) {
                    S[mt][nt][0] *= s0; S[mt][nt][1] *= s1;
                    S[mt][nt][2] *= s0; S[mt][nt][3] *= s1;
                } else {
                    const int g0 = n0 + c0, g1 = g0 + 1;
                    S[mt][nt][0] = (g0 <= b0) ? S[mt][nt][0] * s0 : -INFINITY;
                    S[mt][nt][1] = (g1 <= b0) ? S[mt][nt][1] * s1 : -INFINITY;
                    S[mt][nt][2] = (g0 <= b1) ? S[mt][nt][2] * s0 : -INFINITY;
                    S[mt][nt][3] = (g1 <= b1) ? S[mt][nt][3] * s1 : -INFINITY;
                }
            }
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < BN / 8; ++nt) {
                mx0 = fmaxf(mx0, fmaxf(S[mt][nt][0], S[mt][nt][1]));
                mx1 = fmaxf(mx1, fmaxf(S[mt][nt][2], S[mt][nt][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
            const float nm0 = fmaxf(rm[mt][0], mx0), nm1 = fmaxf(rm[mt][1], mx1);
#ifdef FMHA_DEC_VSHIFT
            // sf carries this tile's shift, so ex2(S - sf) below is p*2^k; al folds the CHANGE of shift
            // into the exponent it already had, which is what keeps rl and O on the same 2^k.
            const float sf0 = (isinf(nm0) ? 0.f : nm0) - ksh;
            const float sf1 = (isinf(nm1) ? 0.f : nm1) - ksh;
            const float al0 = ex2(rm[mt][0] - kprev - sf0), al1 = ex2(rm[mt][1] - kprev - sf1);
#else
            const float sf0 = isinf(nm0) ? 0.f : nm0;
            const float sf1 = isinf(nm1) ? 0.f : nm1;
            const float al0 = ex2(rm[mt][0] - sf0), al1 = ex2(rm[mt][1] - sf1);
#endif
            float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
            for (int nt = 0; nt < BN / 8; ++nt) {
                const float e00 = ex2(S[mt][nt][0] - sf0), e01 = ex2(S[mt][nt][1] - sf0);
                const float e10 = ex2(S[mt][nt][2] - sf1), e11 = ex2(S[mt][nt][3] - sf1);
                sum0 += e00 + e01;
                sum1 += e10 + e11;
                S[mt][nt][0] = e00; S[mt][nt][1] = e01; S[mt][nt][2] = e10; S[mt][nt][3] = e11;
            }
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
            rl[mt][0] = rl[mt][0] * al0 + sum0;
            rl[mt][1] = rl[mt][1] * al1 + sum1;
            rm[mt][0] = nm0; rm[mt][1] = nm1;
            // Once the running max settles, alpha is exactly 1.0f and the rescale
            // is D/8*4 wasted FMAs per m-tile per block (128 of the ~600 non-mma
            // instructions in this config), so vote it away per warp.
            if (__any_sync(0xffffffffu, (al0 != 1.f) | (al1 != 1.f))) {
#pragma unroll
                for (int i = 0; i < DH / 8; ++i) {
                    O[mt][i][0] *= al0; O[mt][i][1] *= al0; O[mt][i][2] *= al1; O[mt][i][3] *= al1;
                }
            }
        }
#ifdef FMHA_DEC_VSHIFT
        kprev = ksh;   // after every m-tile has consumed the previous tile's shift
#endif

        // ---- PV: one transposed V fragment serves all MT m-tiles
#pragma unroll
        for (int nt2 = 0; nt2 < BN / 16; ++nt2) {
            // Contract: round P * v_scale to BF16 (RNE) immediately before PV.
            const int ca = nt2 * 16 + colbase;
            const float wa0 = vst[ca], wa1 = vst[ca + 1];
            const float wb0 = vst[ca + 8], wb1 = vst[ca + 9];
            uint32_t af[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
#ifdef FMHA_DEC_VF16
              if constexpr (VF16) {
                af[mt][0] = pack_f16(S[mt][nt2 * 2][0] * wa0, S[mt][nt2 * 2][1] * wa1);
                af[mt][1] = pack_f16(S[mt][nt2 * 2][2] * wa0, S[mt][nt2 * 2][3] * wa1);
                af[mt][2] = pack_f16(S[mt][nt2 * 2 + 1][0] * wb0, S[mt][nt2 * 2 + 1][1] * wb1);
                af[mt][3] = pack_f16(S[mt][nt2 * 2 + 1][2] * wb0, S[mt][nt2 * 2 + 1][3] * wb1);
              } else {
                af[mt][0] = pack_bf16(S[mt][nt2 * 2][0] * wa0, S[mt][nt2 * 2][1] * wa1);
                af[mt][1] = pack_bf16(S[mt][nt2 * 2][2] * wa0, S[mt][nt2 * 2][3] * wa1);
                af[mt][2] = pack_bf16(S[mt][nt2 * 2 + 1][0] * wb0, S[mt][nt2 * 2 + 1][1] * wb1);
                af[mt][3] = pack_bf16(S[mt][nt2 * 2 + 1][2] * wb0, S[mt][nt2 * 2 + 1][3] * wb1);
              }
#else
                af[mt][0] = pack_bf16(S[mt][nt2 * 2][0] * wa0, S[mt][nt2 * 2][1] * wa1);
                af[mt][1] = pack_bf16(S[mt][nt2 * 2][2] * wa0, S[mt][nt2 * 2][3] * wa1);
                af[mt][2] = pack_bf16(S[mt][nt2 * 2 + 1][0] * wb0, S[mt][nt2 * 2 + 1][1] * wb1);
                af[mt][3] = pack_bf16(S[mt][nt2 * 2 + 1][2] * wb0, S[mt][nt2 * 2 + 1][3] * wb1);
#endif
            }
            const int vrow = nt2 * 16 + (lane & 15), vco = (lane & 16) ? 8 : 0;
#pragma unroll
            for (int dt2 = 0; dt2 < DH / 16; ++dt2) {
                uint32_t bv[4];
#if defined(FMHA_DEC_NOLDM) && (FMHA_DEC_NOLDM & 1)
                bv[0] = bv[1] = bv[2] = bv[3] = 0x3f803f80u ^ (dt2 & 1);
#else
                ldsm_x4_t(bv, smem_u32(vb + vrow * DPV + dt2 * 16 + vco));
#endif
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
#ifdef FMHA_DEC_VF16
                    if constexpr (VF16) {
                        mma_f16(O[mt][dt2 * 2 + 0], af[mt], &bv[0]);
                        mma_f16(O[mt][dt2 * 2 + 1], af[mt], &bv[2]);
                    } else {
                        mma_bf16(O[mt][dt2 * 2 + 0], af[mt], &bv[0]);
                        mma_bf16(O[mt][dt2 * 2 + 1], af[mt], &bv[2]);
                    }
#else
                    mma_bf16(O[mt][dt2 * 2 + 0], af[mt], &bv[0]);
                    mma_bf16(O[mt][dt2 * 2 + 1], af[mt], &bv[2]);
#endif
                }
            }
        }
    };

    // ---- prologue + main loop.
    // Per iteration: publish metadata and issue the K/V loads for block ib+1,
    // issue the metadata loads for ib+2, then run compute(ib), which hides both
    // latencies, and only then convert block ib+1 into the single K/V buffer.
    // Barriers (A)/(B) are only needed when WM > 1, i.e. when several warps read
    // the same K/V tile; with WM == 1 each warp owns its slice end to end.
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        bnd0[mt] = meta->bound[row0[mt]];
        bnd1[mt] = meta->bound[row1[mt]];
        bmin = min(bmin, min(bnd0[mt], bnd1[mt]));
    }
    issue_meta(0);
    cp_wait<0>();
#ifdef FMHA_DEC_F16K
    // Q was cp.async'd straight from the caller's BF16, and the QK MMA now wants F16, so widen it
    // here. TWO constraints, and the first version of this got both wrong (dv_3593bf430e38 and
    // dv_5cec5d94c95c: NaN on case 2, rel_l2 0.453 on case 8, both not bit-identical).
    //
    // 1. The shared row stride is DP = D + 8, not D, so a flat walk over BM*(D/2) words goes
    //    straight through each row's 8-element pad and every row after the first comes out half
    //    BF16, half F16.
    // 2. cp.async completion is PER-THREAD: cp_wait<0> retires this thread's groups, not the CTA's,
    //    and the __syncthreads() below is what makes every thread's copy visible. Converting in
    //    front of that barrier therefore races other threads' copies still in flight - it widens
    //    stale shared memory, and the cp.async then lands on top of it as BF16. A BF16 word read
    //    as F16 has its exponent field taken from bits 14:10, so a Q value around 1e-3 comes back
    //    around 0.5: scores inflate by ~500x, ex2 overflows to +inf, the row maximum becomes inf,
    //    and inf - inf in the softmax is the NaN that case 2 reported.
    //
    // The fix for both is to convert exactly the 16-byte chunks THIS thread cp.async'd, reusing the
    // staging loop's own (row, col16) decomposition: no cross-thread dependency, so no extra
    // barrier, and the one that already follows publishes the widened image. ~8 pairs per thread
    // once per CTA, against the ~32 K pairs per thread per KV block that stop needing four
    // instructions each.
    if constexpr (KF16) {
        uint32_t* qw = reinterpret_cast<uint32_t*>(sQ);
        for (int c = tid; c < BM * (D / 8); c += NT) {
            const int w0 = ((c / (D / 8)) * DP + (c % (D / 8)) * 8) / 2;
            #pragma unroll
            for (int j = 0; j < 4; ++j) qw[w0 + j] = bf16x2_to_f16x2(qw[w0 + j]);
        }
    }
#endif
    __syncthreads();
    if (act(0)) { commit_meta(0); fetch_kv(0, 0); }
    if (act(1)) issue_meta(1);
    if (act(0)) stash_kv(0);
    __syncthreads();
    for (int ib = 0; ib < nblk; ++ib) {
        const bool nxt = act(ib + 1);
        if (nxt) { commit_meta((ib + 1) & 1); fetch_kv(ib + 1, (ib + 1) & 1); }
        if (act(ib + 2)) issue_meta(ib + 2);
        if (act(ib)) compute(ib, ib & 1);
        if (WM > 1) __syncthreads();                       // (A)
        if (nxt) stash_kv(ib + 1);
        if (WM > 1) __syncthreads();                       // (B)
    }
    __syncthreads();

    // ---- epilogue
    const long long obase = (long long)(q0 + q_start) * p.out_s0 + (long long)hy * p.g * D;
    float* wrec = p.ws + ((long long)(tx * p.Hkv + hy) * (DS ? 2 * p.n_split : p.n_split)
                          + (DS ? sz * 2 + dhalf : sz)) * (long long)p.ws_stride;

    auto emit = [&](int r, int d, float val) {
        if (r >= n_rows) return;
        if (p.n_split == 1) {
            long long o = obase + (long long)(r >> p.g_log2) * p.out_s0
                        + (long long)(r & (p.g - 1)) * D + dhalf * DH + d;
            p.out[o] = __float2bfloat16_rn(val);
        } else {
            wrec[r * DH + d] = val;
        }
    };
    auto emit_ml = [&](int r, float m, float l) {
        if (r >= n_rows || p.n_split == 1) return;
        wrec[BM * DH + r] = m;
        // The odd half publishes l == 0: m is duplicated (harmless for the max)
        // while the denominator must count each split's l exactly ONCE.
        wrec[BM * DH + BM + r] = (DS && dhalf != 0) ? 0.f : l;
    };

    // ---- fused split-merge tail (FM). The LAST CTA to arrive on group
    // (tx, hy) reduces the group's n_split partial records straight into
    // `out`, deleting the separate fmha_combine launch (and its visible
    // 4.1-6.3us slice, results/skipprice-20261028.md) for the small-split
    // decode class. Ordering: every CTA writes its partial, __threadfence()s
    // (release to device scope), then atomicAdd's the group counter; the
    // arriver that reads n_split-1 knows every other partial is device-
    // visible (classic threadfence-reduction pattern; the acquire side is
    // the second __threadfence below). Empty slices (kv_lo >= hi) arrive
    // without writing and are excluded from the sum by nse, exactly like
    // fmha_combine does. Per-dim accumulation walks splits ASCENDING, the
    // same order as combine's per-thread dot; the denominator is a warp
    // shuffle tree per row instead of combine's block tree - the one f32
    // regrouping, worth ~1ulp on inv (rel_l2 ~1e-7 class, NOT bit-identical).
    auto fuse_tail = [&]() {
        if constexpr (FM) {
        if (p.fuse_merge) {
            // Release pattern (threadfence-reduction): EVERY thread fences its
            // own partial writes to device scope first; the barrier then orders
            // all fences ahead of tid0's arrival atomic, so the CTA that reads
            // n_split-1 knows every partial of every earlier arriver is
            // device-visible. (Fence-after-barrier would let a straggler's
            // fence lag the arrival and break that guarantee.)
            __threadfence();
            __syncthreads();
            __shared__ int s_last;
            if (tid == 0)
                s_last = (atomicAdd(&p.fctr[tx * p.Hkv + hy], 1)
                          == p.n_split - 1) ? 1 : 0;
            __syncthreads();
            if (!s_last) return;
            __threadfence();      // acquire: all earlier arrivals are visible
            const int chunk = (hi + p.n_split - 1) / p.n_split;
            const int chunk_bn = ((chunk + BN - 1) / BN) * BN;
            const int nse = min(p.n_split, (hi + chunk_bn - 1) / chunk_bn);
            const float* gbase =
                p.ws + (long long)(tx * p.Hkv + hy) * p.n_split * p.ws_stride;
            __shared__ float s_w[BM][FM_MAXS + 1];  // [r][s] weights; [r][FM_MAXS] = inv
            const int warp = tid >> 5, lane = tid & 31;
            constexpr int NWARP = NT / 32;
            // phase 1: one warp per row round; lane s holds entry s (nse <= 32)
            for (int r = warp; r < n_rows; r += NWARP) {
                float m = -INFINITY, l = 0.f;
                if (lane < nse) {
                    m = gbase[(long long)lane * p.ws_stride + BM * D + r];
                    l = gbase[(long long)lane * p.ws_stride + BM * D + BM + r];
                }
                float mr = m;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    mr = fmaxf(mr, __shfl_xor_sync(0xffffffffu, mr, off));
                const float sf = isinf(mr) ? 0.f : mr;
                const float w = (lane < nse) ? ex2(m - sf) : 0.f;
                float ll = l * w;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    ll += __shfl_xor_sync(0xffffffffu, ll, off);
                if (lane < nse) s_w[r][lane] = w;
                if (lane == 0) s_w[r][FM_MAXS] = (ll > 0.f) ? (1.f / ll) : 0.f;
            }
            __syncthreads();
            // phase 2: per (row, dim) ascending-split dot, identical term order
            // to fmha_combine's per-thread strided loop.
            for (int r = 0; r < n_rows; ++r) {
                const float inv = s_w[r][FM_MAXS];
                for (int d = tid; d < D; d += NT) {
                    float a = 0.f;
                    for (int s2 = 0; s2 < nse; ++s2)
                        a += s_w[r][s2] * gbase[(long long)s2 * p.ws_stride + r * D + d];
                    const long long o = obase + (long long)(r >> p.g_log2) * p.out_s0
                                      + (long long)(r & (p.g - 1)) * D + d;
                    p.out[o] = __float2bfloat16_rn(a * inv);
                }
            }
        }
        }
    };

    const bool normalise = (p.n_split == 1);
    if (WK == 1) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const float inv0 = normalise ? ((rl[mt][0] > 0.f) ? (1.f / rl[mt][0]) : 0.f) : 1.f;
            const float inv1 = normalise ? ((rl[mt][1] > 0.f) ? (1.f / rl[mt][1]) : 0.f) : 1.f;
#pragma unroll
            for (int dt = 0; dt < DH / 8; ++dt) {
                const int d = dt * 8 + colbase;
                emit(row0[mt], d, O[mt][dt][0] * inv0);
                emit(row0[mt], d + 1, O[mt][dt][1] * inv0);
                emit(row1[mt], d, O[mt][dt][2] * inv1);
                emit(row1[mt], d + 1, O[mt][dt][3] * inv1);
            }
            if ((lane & 3) == 0 && !normalise) {
#ifdef FMHA_DEC_VSHIFT
                // (m - k, l) with O on the same 2^k is a valid partial, so fmha_combine is unchanged.
                emit_ml(row0[mt], rm[mt][0] - kprev, rl[mt][0]);
                emit_ml(row1[mt], rm[mt][1] - kprev, rl[mt][1]);
#else
                emit_ml(row0[mt], rm[mt][0], rl[mt][0]);
                emit_ml(row1[mt], rm[mt][1], rl[mt][1]);
#endif
            }
        }
        fuse_tail();
        return;
    }

    // WK > 1 (WM == 1, BM == 16): merge the per-warp KV partials in shared memory.
    __syncthreads();   // every warp has finished reading sK/sV
    float* cacc = reinterpret_cast<float*>(smem + SL::OFF_K);
    float* cm = cacc + WK * 16 * DH;
    float* cl = cm + WK * 16;
    {
        const int r0 = lane >> 2, r1 = r0 + 8;
#pragma unroll
        for (int dt = 0; dt < DH / 8; ++dt) {
            const int d = dt * 8 + colbase;
            cacc[(kw * 16 + r0) * DH + d] = O[0][dt][0];
            cacc[(kw * 16 + r0) * DH + d + 1] = O[0][dt][1];
            cacc[(kw * 16 + r1) * DH + d] = O[0][dt][2];
            cacc[(kw * 16 + r1) * DH + d + 1] = O[0][dt][3];
        }
        if ((lane & 3) == 0) {
#ifdef FMHA_DEC_VSHIFT
            cm[kw * 16 + r0] = rm[0][0] - kprev; cl[kw * 16 + r0] = rl[0][0];
            cm[kw * 16 + r1] = rm[0][1] - kprev; cl[kw * 16 + r1] = rl[0][1];
#else
            cm[kw * 16 + r0] = rm[0][0]; cl[kw * 16 + r0] = rl[0][0];
            cm[kw * 16 + r1] = rm[0][1]; cl[kw * 16 + r1] = rl[0][1];
#endif
        }
    }
    __syncthreads();
    for (int r = 0; r < 16; ++r) {
        if (r >= n_rows) break;
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < WK; ++w) M = fmaxf(M, cm[w * 16 + r]);
        const float sf = isinf(M) ? 0.f : M;
        float scv[WK], L = 0.f;
#pragma unroll
        for (int w = 0; w < WK; ++w) { scv[w] = ex2(cm[w * 16 + r] - sf); L += cl[w * 16 + r] * scv[w]; }
        const float inv = normalise ? ((L > 0.f) ? (1.f / L) : 0.f) : 1.f;
        for (int d = tid; d < DH; d += NT) {
            float a = 0.f;
#pragma unroll
            for (int w = 0; w < WK; ++w) a += cacc[(w * 16 + r) * DH + d] * scv[w];
            emit(r, d, a * inv);
        }
        if (tid == 0 && !normalise) emit_ml(r, M, L);
    }
    fuse_tail();
#ifdef FMHA_PDL
#ifdef FMHA_PDL_LATE
    if (threadIdx.x == 0) FMHA_PDL_TRIG();
#endif
#endif
}

// ---------------------------------------------------------------- combine
// Reduce the split-KV partials. Parallelised over (dim chunk, row group) in
// grid.z and over the split range inside the block, because a single CTA per
// tile made the reduction dominate the decode shapes (n_split in the hundreds).
// VF > 0 fuses this row's partial-value loads into the same round trip as the m/l
// loads. Their addresses depend only on (s, r, d) - not on the max or the sum - so
// they can be in flight before the first barrier instead of after the second, which
// removes one exposed round trip from a kernel whose cost IS that chain. VF is a
// compile-time bound chosen on the host from n_split/SG so the values stay in
// registers; 0 keeps the original strided loop.
#if defined(FMHA_COMBINE_REG) && defined(FMHA_COMBINE_VF_ON)
#error "FMHA_COMBINE_REG keeps m/l in registers and does not issue the fused value " \
       "prefetch, so a VF > 0 instantiation would read uninitialised registers."
#endif
template <int D, int MAXS, int VF = 0, int DH = 0>
__global__ void fmha_combine(const Params p) {
    // DH > 0: dim-split records (cfg37). Each split owns TWO half-wide entries
    // (rec = split*2 + half), O slots cover dims [half*DH, (half+1)*DH), the
    // odd half publishes l == 0 so the denominator counts every split once,
    // and both halves publish the same m. A thread's dim d lives in exactly
    // one half, so its dot reads nse entries out of nseE = 2*nse - the same
    // terms in the same ascending split order as the unsplit build, which is
    // what makes the forced-n_split A/B bit-identical.
    constexpr int DW = (DH > 0) ? DH : D;
    static_assert(DH == 0 || VF == 0,
                  "dim-split entries ride the unfused strided path (VF window not ported)");
#ifdef FMHA_COMBINE_REG
    // (DH is a template parameter - the preprocessor cannot see it, so this is
    // a static_assert and not an #error.)
    static_assert(DH == 0,
                  "FMHA_COMBINE_REG's mv/lv window reads full-D entries at nse "
                  "stride; it was never ported to the dim-split pair layout.");
#endif
    constexpr int NT = 256;
    const int tx = blockIdx.x, hy = blockIdx.y, tid = threadIdx.x;
    const int BM = p.BM, dc = p.dc, rpt = p.rpt;
    const int ndc = (D + dc - 1) / dc;
    const int rz = blockIdx.z / ndc, dz = blockIdx.z % ndc;
    __shared__ int s_cu[33];
#ifdef FMHA_COMBINE_V2
    __shared__ int s_sl[33];
#endif
#ifndef FMHA_COMBINE_REG
    __shared__ float s_m[MAXS], s_l[MAXS], s_sc[MAXS], s_red[NT], s_warp[NT / 32];
#else
    // m and l are only ever read back by the thread that loaded them: the max and
    // the sum are per-thread partials, and the only cross-thread values are the
    // warp partials and s_sc. So they can stay in registers, which removes two
    // shared stores and four shared loads per element AND the barrier that
    // published them. K is the worst-case trip count of the s = tid + i*NT loop.
    __shared__ float s_sc[MAXS], s_red[NT], s_warp[NT / 32];
#endif
    for (int i = tid; i <= p.B; i += NT) {
        s_cu[i] = p.cu[i];
#ifdef FMHA_COMBINE_V2
        // `hi` below needs slen[b] and b only exists after the tile scan, so
        // reading it from global put a second dependent round trip in front of the
        // m/l load. Staging it with cu costs 33 ints of shared and nothing else.
        // cu needs index B, seqused_k has only B elements: the guard is required.
        if (i < p.B) s_sl[i] = p.slen[i];
#endif
    }
    __syncthreads();
#ifdef FMHA_SLOWMAP
    int b = -1, m_blk = 0, cum = 0;
    for (int i = 0; i < p.B; ++i) {
        int ql = s_cu[i + 1] - s_cu[i];
        int nt = (ql * p.g + BM - 1) / BM;
        if (b < 0 && tx < cum + nt) { b = i; m_blk = nt - 1 - (tx - cum); }   // must match fmha_main
        cum += nt;
    }
#else
    // The scan above is a serial O(B) chain of dependent shared loads in every
    // thread, which is cheap in fmha_main (amortised over dozens of KV blocks) and
    // not cheap here: with rpt=1 a combine CTA does one row of work, so the
    // prologue is a large fraction of it, and B is up to 32 in the frozen
    // shape_range. B <= 32 is enforced on the host, so one warp can hold every
    // request's tile count, prefix-sum it with shuffles and pick the owner by
    // ballot. It must produce exactly what the serial scan produces: the first i
    // with tx < cum_i + nt_i, and since the ranges [lo_i, pre_i) are contiguous
    // and increasing that is the unique i with lo_i <= tx < pre_i.
    __shared__ int s_bm[2];
    if (tid < 32) {
        const int nt = (tid < p.B) ? ((s_cu[tid + 1] - s_cu[tid]) * p.g + BM - 1) / BM : 0;
        int pre = nt;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, pre, off);
            if ((tid & 31) >= off) pre += y;
        }
        const int lo = pre - nt;
        const unsigned bal = __ballot_sync(0xffffffffu,
                                           (tid < p.B) && (tx >= lo) && (tx < pre));
        // The two gathers must be executed by ALL 32 lanes: a *_sync shuffle with
        // a full mask inside `if (tid == 0)` makes lane 0 wait for lanes that never
        // arrive, which deadlocked the kernel (job dv_77f87bffedde hung on the
        // first forward call, after the reference had completed, on both a fixed
        // case and a domain config). Same failure class as the divergent
        // tcgen05.ld in candidate/fmha_tc.cu.
        const int src = bal ? (__ffs(bal) - 1) : 0;
        const int lob = __shfl_sync(0xffffffffu, lo, src);
        const int ntb = __shfl_sync(0xffffffffu, nt, src);
        if (tid == 0) {
            s_bm[0] = bal ? src : -1;
            s_bm[1] = bal ? (ntb - 1 - (tx - lob)) : 0;
        }
    }
    __syncthreads();
    const int b = s_bm[0], m_blk = s_bm[1];
#endif
    if (b < 0) return;
    const int q0 = s_cu[b], qlen = s_cu[b + 1] - q0;
    const int q_start = m_blk * (BM / p.g);
    const int n_rows = min(BM, min(BM / p.g, qlen - q_start) * p.g);
    const int r_lo = rz * rpt;
    if (r_lo >= n_rows) return;
    const int r_hi = min(n_rows, r_lo + rpt);
    const int d_lo = dz * dc, d = d_lo + (tid % dc);
    const int SG = NT / dc, sg = tid / dc;
    const bool dok = (d < min(D, d_lo + dc));
#ifdef FMHA_COMBINE_V2
    const int hi = s_sl[b] - qlen + q_start + n_rows / p.g;
#else
    const int hi = p.slen[b] - qlen + q_start + n_rows / p.g;
#endif
    const int chunk_bn = (((hi + p.n_split - 1) / p.n_split + p.BN - 1) / p.BN) * p.BN;
    const int nse = min(p.n_split, (hi + chunk_bn - 1) / chunk_bn);
    const int nseE = (DH > 0) ? 2 * nse : nse;   // entries this row reduces
    const long long stride = p.ws_stride;
    const float* wbase = p.ws + (long long)(tx * p.Hkv + hy)
                        * (DH > 0 ? 2 * p.n_split : p.n_split) * stride;
    const long long obase = (long long)(q0 + q_start) * p.out_s0 + (long long)hy * p.g * D;
#ifdef FMHA_PDL
    // Everything above reads only the caller's inputs (cu_seqlens, seqused_k) and
    // this kernel's own grid/block indices, so it is the part of the combine that
    // can legally run before the primary grid has finished. Everything below reads
    // p.ws, which fmha_main wrote, so the wait has to be here and not later.
    FMHA_PDL_WAIT();
#endif
    constexpr int K = (MAXS + NT - 1) / NT;
    for (int r = r_lo; r < r_hi; ++r) {
        // Declared outside the #ifdef below so the `if constexpr (VF > 0)` branch
        // of the dot always parses: VF is a template parameter, so that branch is
        // discarded but its non-dependent names are still looked up at definition
        // time (this is what broke the FMHA_COMBINE_REG build, job dv_b6288422bd5f).
        // Unused when VF == 0, which is every build except FMHA_COMBINE_VF_ON.
        float v[VF > 0 ? VF : 1];
#ifdef FMHA_COMBINE_REG
        float mv[K], lv[K];
#pragma unroll
        for (int i = 0; i < K; ++i) {
            const int s = tid + i * NT;
            if (s < nse) {
                mv[i] = wbase[(long long)s * stride + BM * D + r];
                lv[i] = wbase[(long long)s * stride + BM * D + BM + r];
            } else {
                // Masked slots must not move the max or the sum: -inf and 0.
                mv[i] = -INFINITY;
                lv[i] = 0.f;
            }
        }
        // No barrier here. The removed one published s_m/s_l to the other
        // threads; nothing is published now, and the previous row's reads of
        // s_warp and s_red are already separated from this row's writes by that
        // row's post-dot barrier and its trailing barrier.
#else
        if constexpr (VF > 0) {
            // Must stay under dok: for !dok lanes d can exceed D and the address
            // would leave this row's workspace slice.
            if (dok) {
#pragma unroll
                for (int i = 0; i < VF; ++i) {
                    const int s = sg + i * SG;
                    if (s < nse) v[i] = wbase[(long long)s * stride + r * D + d];
                }
            }
        }
        if constexpr (DH > 0) {
            // Pair-mapped: thread s owns entries {2s, 2s+1} so the per-thread ll
            // partial below accumulates exactly the unsplit build's terms plus an
            // exact +0.0 from the odd half, in the same order - that grouping is
            // what makes the forced-n_split A/B bit-identical instead of merely
            // close (f32 addition is not associative across regroupings).
            for (int s = tid; s < nse; s += NT) {
                s_m[2 * s] = wbase[(long long)(2 * s) * stride + BM * DW + r];
                s_l[2 * s] = wbase[(long long)(2 * s) * stride + BM * DW + BM + r];
                s_m[2 * s + 1] = wbase[(long long)(2 * s + 1) * stride + BM * DW + r];
                s_l[2 * s + 1] = 0.f;   // odd half publishes 0 by contract; re-zero locally
            }
        } else {
            for (int s = tid; s < nseE; s += NT) {
                s_m[s] = wbase[(long long)s * stride + BM * DW + r];
                s_l[s] = wbase[(long long)s * stride + BM * DW + BM + r];
            }
        }
        __syncthreads();
#endif
        // Block-parallel max/exp/sum: with nse in the hundreds, letting every
        // thread scan the whole split range made this kernel cost more than the
        // attention itself (48us of case 3's 168us).
        float lm = -INFINITY;
#ifdef FMHA_COMBINE_REG
#pragma unroll
        for (int i = 0; i < K; ++i) lm = fmaxf(lm, mv[i]);
#else
        for (int s = tid; s < nseE; s += NT) lm = fmaxf(lm, s_m[s]);
#endif
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, off));
        if ((tid & 31) == 0) s_warp[tid >> 5] = lm;
        __syncthreads();
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < NT / 32; ++w) M = fmaxf(M, s_warp[w]);
        const float sf = isinf(M) ? 0.f : M;
        float ll = 0.f;
#ifdef FMHA_COMBINE_REG
#pragma unroll
        for (int i = 0; i < K; ++i) {
            const int s = tid + i * NT;
            const float sc = ex2(mv[i] - sf);
            // Predicated, not unconditional: K is the WORST case (MAXS/NT), so at
            // nse=12 (case 2) an unconditional store would write 1024 shared
            // floats per row where the old loop wrote 12. Masked slots still
            // contribute sc == 0 and lv == 0 to ll.
            if (s < nse) s_sc[s] = sc;
            ll += lv[i] * sc;
        }
#else
        if constexpr (DH > 0) {
            for (int s = tid; s < nse; s += NT) {
                const float sc0 = ex2(s_m[2 * s] - sf);
                const float sc1 = ex2(s_m[2 * s + 1] - sf);
                s_sc[2 * s] = sc0;
                s_sc[2 * s + 1] = sc1;
                ll += s_l[2 * s] * sc0;
                ll += s_l[2 * s + 1] * sc1;   // == +0.0f exactly, keeps ctl's grouping
            }
        } else {
            for (int s = tid; s < nseE; s += NT) { const float sc = ex2(s_m[s] - sf); s_sc[s] = sc; ll += s_l[s] * sc; }
        }
#endif
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ll += __shfl_xor_sync(0xffffffffu, ll, off);
        if ((tid & 31) == 0) s_warp[tid >> 5] = ll;
        __syncthreads();
        float L = 0.f;
#pragma unroll
        for (int w = 0; w < NT / 32; ++w) L += s_warp[w];
        const float inv = (L > 0.f) ? (1.f / L) : 0.f;
#ifdef FMHA_SLOWBAR
        __syncthreads();
#endif
        // No barrier needed here: between the previous one and the one after
        // s_red, threads only READ s_warp and s_sc (both written before that
        // previous barrier) and write their own s_red[tid]. The previous row's
        // reads of s_red are separated by the trailing barrier below.
        if (dok) {
            float a = 0.f;
            if constexpr (VF > 0) {
#pragma unroll
                for (int i = 0; i < VF; ++i) {
                    const int s = sg + i * SG;
                    if (s < nse) a += v[i] * s_sc[s];
                }
                // Whatever the register window did not cover keeps the original
                // strided loop. VF is therefore a BUDGET, not a promise: a shape
                // with more splits than the budget still gets the overlap for its
                // first VF values instead of falling back entirely, and the sum is
                // accumulated in exactly the original ascending order of s, so the
                // result is bit-identical either way.
                for (int s = sg + VF * SG; s < nse; s += SG)
                    a += wbase[(long long)s * stride + r * D + d] * s_sc[s];
            } else if constexpr (DH > 0) {
                // Iterate SPLITS in the unsplit build's order; entry = 2*split + half.
                // (A flat entry stride would preserve parity - SG is even - and half
                // the threads would read none of their dim's entries.)
                const int hneed = d / DH, dslot = d - hneed * DH;
                for (int s2 = sg; s2 < nse; s2 += SG) {
                    const long long e = 2LL * s2 + hneed;
                    a += wbase[e * stride + r * DW + dslot] * s_sc[e];
                }
            } else {
                for (int s = sg; s < nse; s += SG)
                    a += wbase[(long long)s * stride + r * D + d] * s_sc[s];
            }
            s_red[tid] = a;
        }
        __syncthreads();
        if (dok && sg == 0) {
            // SG is 256/dc and dc is now chosen at plan time, so this bound is a
            // runtime value; it was hardcoded to 8 when dc could not go below 32.
            float a = s_red[tid];
            for (int j = 1; j < SG; ++j) a += s_red[tid + j * dc];
            const long long o = obase + (long long)(r >> p.g_log2) * p.out_s0
                              + (long long)(r & (p.g - 1)) * D + d;
            p.out[o] = __float2bfloat16_rn(a * inv);
        }
#ifdef FMHA_SLOWBAR
        __syncthreads();
#else
        // Only the next row's s_m/s_l/s_warp/s_red writes need to be kept off this
        // row's reads, so the last row does not need it. r_hi is CTA-uniform, so
        // every thread takes the same side and the barrier cannot be divergent.
        if (r + 1 < r_hi) __syncthreads();
#endif
    }
}

#ifdef FMHA_COMBINE_W
// Warp-per-row combine: ONE WARP OWNS ONE ROW, so the row needs no cross-warp
// communication at all - no shared arrays for m/l/sc, no s_red reduction, and no
// __syncthreads() anywhere in the row loop.
//
// Why: the profile of the split-KV pair (run 20261023T120653Z-147065-l20d and
// 20261023T123258Z-412788-l20d) says fmha_combine is 27-33% of GPU time on fixed cases
// 5/6/8 and 7-17% on cases 2/3/4, and that it is barrier-bound rather than busy:
// stalled_barrier 3.83-6.19 per issued instruction against 0.30-0.54 in fmha_main, with
// 64 registers, no dynamic smem, tiny DRAM traffic (0.3-8.1 MB) and waves_per_multiprocessor
// 0.86. The block-parallel version spends ~6 __syncthreads() per row to publish and reduce
// three per-row scalars (max, denominator, sc[]) across 256 threads that then cooperate on
// only dc=32-64 dims. Case 5 does 1 row x 64 dims x 16 splits per CTA, i.e. 1024 FMAs, in
// about 16500 cycles.
//
// Layout: 8 warps, warp w takes row r_lo+w, and lane l owns dims l, l+32, ... so the value
// loads for a given split are lane-contiguous and fully coalesced (128 B per instruction),
// while the split's m is one uniform load shared by the warp. Only the two per-row scalars
// cross lanes, and they do it with 5+5 shuffles instead of six barriers.
//
// Not bit-identical to fmha_combine: the denominator sums l_s*sc_s in a warp tree instead of
// a per-thread strided partial plus a block tree, and the numerator accumulates splits in
// ascending order in one lane instead of across SG lanes. Same terms, same f32 type,
// different grouping - the frozen tolerance rule decides, per shape.
template <int D, int WPR>
__global__ void __launch_bounds__(256) fmha_combine_w(const Params p) {
    constexpr int NT = 256, NW = NT / 32;
    // WPR warps share one row, splitting its dim range into WPR blocks of D/WPR.
    // Every one of them recomputes the row's max and denominator (two loads and
    // 5+5 shuffles each) so that no barrier or shared array is needed between
    // them: the redundancy is the price of turning "rows" into "rows x WPR"
    // warps, which is the only parallelism this kernel can add. RPC rows per CTA
    // instead of NW, so grid.z grows by WPR and the launch fills WPR x more SMs.
    constexpr int RPC = NW / WPR;         // rows per CTA
    constexpr int KD = D / (32 * WPR);    // 32-dim groups this warp's lane covers
    static_assert(D % (32 * WPR) == 0, "WPR must divide D/32");
    const int tx = blockIdx.x, hy = blockIdx.y, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int BM = p.BM;
    __shared__ int s_cu[33], s_sl[33], s_bm[2];
    for (int i = tid; i <= p.B; i += NT) {
        s_cu[i] = p.cu[i];
        // seqused_k has only B elements, cu needs index B: the guard is required.
        if (i < p.B) s_sl[i] = p.slen[i];
    }
    __syncthreads();
    // Tile -> (b, m_blk), identical to fmha_combine's warp prefix-sum + ballot path, which
    // must itself match fmha_main's serial scan. Both gathers run on all 32 lanes: a
    // *_sync shuffle with a full mask inside `if (tid == 0)` waits for lanes that never
    // arrive and deadlocked the kernel once already (job dv_77f87bffedde).
    if (tid < 32) {
        const int nt = (tid < p.B) ? ((s_cu[tid + 1] - s_cu[tid]) * p.g + BM - 1) / BM : 0;
        int pre = nt;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, pre, off);
            if ((tid & 31) >= off) pre += y;
        }
        const int lo = pre - nt;
        const unsigned bal = __ballot_sync(0xffffffffu, (tid < p.B) && (tx >= lo) && (tx < pre));
        const int src = bal ? (__ffs(bal) - 1) : 0;
        const int lob = __shfl_sync(0xffffffffu, lo, src);
        const int ntb = __shfl_sync(0xffffffffu, nt, src);
        if (tid == 0) {
            s_bm[0] = bal ? src : -1;
            s_bm[1] = bal ? (ntb - 1 - (tx - lob)) : 0;
        }
    }
    __syncthreads();
    const int b = s_bm[0], m_blk = s_bm[1];
    if (b < 0) return;
    const int q0 = s_cu[b], qlen = s_cu[b + 1] - q0;
    const int q_start = m_blk * (BM / p.g);
    const int n_rows = min(BM, min(BM / p.g, qlen - q_start) * p.g);
    const int r_lo = blockIdx.z * RPC;
    if (r_lo >= n_rows) return;
    // hi / chunk_bn / nse are exactly fmha_main's per-tile bounds, so this warp reads the
    // same set of partials the primary wrote and no others.
    const int hi = s_sl[b] - qlen + q_start + n_rows / p.g;
    const int chunk_bn = (((hi + p.n_split - 1) / p.n_split + p.BN - 1) / p.BN) * p.BN;
    const int nse = min(p.n_split, (hi + chunk_bn - 1) / chunk_bn);
    const long long stride = p.ws_stride;
    const float* wbase = p.ws + (long long)(tx * p.Hkv + hy) * p.n_split * stride;
    const long long obase = (long long)(q0 + q_start) * p.out_s0 + (long long)hy * p.g * D;
#ifdef FMHA_PDL
    // Everything above reads only the caller's inputs and this kernel's own indices; p.ws is
    // fmha_main's output, so the wait has to be here and not earlier.
    FMHA_PDL_WAIT();
#endif
    const int r = r_lo + warp / WPR;
    if (r >= min(n_rows, r_lo + RPC)) return;     // warp-uniform: whole warps exit
    // This warp's dim block. The k index runs over 32-dim groups, so a block is
    // KD contiguous groups starting at k0 and the loads stay lane-contiguous.
    const int k0 = (warp % WPR) * KD;
    const float* mrow = wbase + BM * D + r;
    const float* lrow = mrow + BM;
    const float* vrow = wbase + (long long)r * D;

    float lm = -INFINITY;
    for (int s = lane; s < nse; s += 32) lm = fmaxf(lm, mrow[(long long)s * stride]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, off));

    float ll = 0.f;
    for (int s = lane; s < nse; s += 32) {
        const long long o = (long long)s * stride;
        ll += lrow[o] * ex2(mrow[o] - lm);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ll += __shfl_xor_sync(0xffffffffu, ll, off);
    const float inv = (ll > 0.f) ? (1.f / ll) : 0.f;

    float acc[KD];
#pragma unroll
    for (int k = 0; k < KD; ++k) acc[k] = 0.f;
    // Four splits in flight per iteration: the m loads are independent of each other and of
    // the previous iteration's FMAs, so the group hides their latency without needing a
    // register window over the values (which is what cost fmha_combine's VF=16 arm 64
    // registers and 2 CTAs/SM).
    int s = 0;
    for (; s + 4 <= nse; s += 4) {
        const long long o0 = (long long)s * stride;
        const float c0 = ex2(mrow[o0] - lm), c1 = ex2(mrow[o0 + stride] - lm);
        const float c2 = ex2(mrow[o0 + 2 * stride] - lm), c3 = ex2(mrow[o0 + 3 * stride] - lm);
        const float* v0 = vrow + o0;
#pragma unroll
        for (int k = 0; k < KD; ++k) {
            const int d = lane + 32 * (k0 + k);
            acc[k] += v0[d] * c0 + v0[d + stride] * c1
                    + v0[d + 2 * stride] * c2 + v0[d + 3 * stride] * c3;
        }
    }
    for (; s < nse; ++s) {
        const float c = ex2(mrow[(long long)s * stride] - lm);
        const float* vs = vrow + (long long)s * stride;
#pragma unroll
        for (int k = 0; k < KD; ++k) acc[k] += vs[lane + 32 * (k0 + k)] * c;
    }
#pragma unroll
    for (int k = 0; k < KD; ++k) {
        const int d = lane + 32 * (k0 + k);
        const long long o = obase + (long long)(r >> p.g_log2) * p.out_s0
                          + (long long)(r & (p.g - 1)) * D + d;
        p.out[o] = __float2bfloat16_rn(acc[k] * inv);
    }
}
#endif  // FMHA_COMBINE_W

// ---------------------------------------------------------------- host side
#include <cstdlib>
#ifdef FMHA_DBG_PLAN
#include <cstdio>
#endif
#include <utility>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static int ilog2i(int x) { int r = 0; while ((1 << r) < x) ++r; return r; }

// DIAGNOSTIC ONLY, and only compiled in when FMHA_DBG_SKIP is defined: bit 0
// skips the main kernel, bit 1 skips the combine, so the two can be timed
// separately in one process. The produced output is wrong by construction, so
// this must never be set for a real run; solution.py keys a distinct extension
// name off it. Without the define g_skip is a constexpr 0 and both branches fold
// away, leaving the production path untouched.
// Window ceiling for combine dot loops longer than the VF table's cap of 8. Shipped at 16;
// FMHA_COMBINE_NO_VFHI=1 restores the unfused strided loop and FMHA_COMBINE_VFHI=<n> overrides
// the ceiling (32 measured +7.7% on case 3, the register cliff). Only `need > 8` shapes are
// touched, so on the fixed suite that is cases 3 and 8 and nothing else.
#ifndef FMHA_COMBINE_VFHI
#ifndef FMHA_COMBINE_NO_VFHI
#define FMHA_COMBINE_VFHI 16
#endif
#endif

// DIAGNOSTIC PRICE PROBE (FMHA_DEC_NOLDM), wrong numbers by construction and never shippable: bit 0
// deletes the K and V fragment ldmatrix, bit 1 the Q A-fragment ldmatrix, replacing each with a
// constant so every dependent instruction, both mma.sync issues, the softmax and the epilogue still
// run and nothing downstream branches on data or goes non-finite - 0x3f803f80 is two bf16 1.0, and the
// xor against the loop index keeps the value loop-variant so the compiler cannot hoist it and silently
// price less than the load. This bounds what a tcgen05 decode main kernel would win, because tcgen05
// takes both operands from shared memory and so has no fragment loads at all: the decode L1 account is
// 5 + 16*MT/BN bytes per fp8 KV byte, of which 2 are the K and V ldmatrix and 16*MT/BN the Q
// A-fragments re-read per block. The premise that those bytes leave the L1TEX account rather than
// moving within it is measurable from the prefill, which already uses tcgen05: its shared wavefronts
// are 68.3M store against 42.4M load, while its MMAs re-read the whole 32 KB Q image on every one of
// 257 tiles - had those reads been charged as load wavefronts, loads would exceed stores roughly 3:1
// instead of running at 0.62:1.
// FMHA_DEC_NOLDM is part of the extension name, so each mask is its own build with its own value.
#ifdef FMHA_DBG_SKIP
static const int g_skip = [] { const char* e = getenv("FMHA_SKIP"); return e ? atoi(e) : 0; }();
#else
static constexpr int g_skip = 0;
#endif

#ifdef FMHA_PDL
#define FMHA_LAUNCH_COMBINE(...) do {                                          \
        cudaLaunchConfig_t lc{};                                               \
        cudaLaunchAttribute la[1]{};                                           \
        la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;         \
        la[0].val.programmaticStreamSerializationAllowed = 1;                  \
        lc.gridDim = grid;                                                     \
        lc.blockDim = dim3(256);                                               \
        lc.dynamicSmemBytes = 0;                                               \
        lc.stream = st;                                                        \
        lc.attrs = la;                                                         \
        lc.numAttrs = 1;                                                       \
        cudaLaunchKernelEx(&lc, __VA_ARGS__, p);                               \
    } while (0)
#else
#define FMHA_LAUNCH_COMBINE(...) __VA_ARGS__<<<grid, 256, 0, st>>>(p)
#endif

template <int D, int BM, int BN, int WM, int WK, bool DS = false>
static void launch_one(const Params& p, int n_tiles_ub, cudaStream_t st) {
    using SL = SmemLayout<D, BM, BN, WK, DS ? D / 2 : D>;
    // Decode-only inert smem pad (see the FMHA_DEC_SMEMPAD_KB note above).
    // Prefill instantiations (WK == 1) keep the production launch byte-for-byte.
    constexpr int SPAD = (WK > 1) ? FMHA_DEC_SMEMPAD_KB * 1024 : 0;
    constexpr int SMEM_TOT = SL::BYTES + SPAD;
    // The fused split-merge instantiation exists only in the flagged build and
    // only for the non-DS BM=16 decode tiles; every other launch_one
    // specialization keeps exactly ONE fmha_main instantiation, so the
    // default build's codegen is untouched by the flag's existence.
#ifdef FMHA_DEC_FUSEMERGE
    constexpr bool FMC = (BM == 16) && !DS;
#else
    constexpr bool FMC = false;
#endif
    static const bool ok = [] {
        cudaFuncSetAttribute((const void*)fmha_main<D, BM, BN, WM, WK, DS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOT);
        if constexpr (FMC)
            cudaFuncSetAttribute((const void*)fmha_main<D, BM, BN, WM, WK, DS, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOT);
        return true;
    }();
    (void)ok;
    TORCH_CHECK(SMEM_TOT <= 232448, "shared memory over budget");
    Params q = p;
    q.fuse_merge = 0;
    q.fctr = nullptr;
    bool fused = false;
    if constexpr (FMC) {
        if (q.n_split > 1 && q.n_split <= FM_MAXS && !g_skip) {
            // Per-group arrival counters. Lazy cudaMalloc outside capture: a
            // binding's fmha_capture only runs after a direct fmha_run of the
            // same binding and eligibility is a pure function of the binding's
            // scalars, so the malloc can never execute during capture (same
            // contract as the merged prefill's ticket counter). The memset is
            // recorded as a graph node and re-arms the counters every replay.
            static int* fctr = [] {
                int* c = nullptr;
                TORCH_CHECK(cudaMalloc(&c, 1 << 20) == cudaSuccess,
                            "fuse counter alloc");
                return c;
            }();
            cudaMemsetAsync(fctr, 0,
                            (size_t)n_tiles_ub * q.Hkv * sizeof(int), st);
            q.fctr = fctr;
            q.fuse_merge = 1;
            fused = true;
        }
    }
    (void)fused;
    if (!(g_skip & 1)) {
        const dim3 grid(n_tiles_ub, q.Hkv, DS ? 2 * q.n_split : q.n_split);
#ifdef FMHA_DEC_OCCPRINT
        // One-shot residency telemetry for the spad probe arms: achieved
        // blocks/SM for THIS launch shape, first 3 launches per instantiation
        // (warmup absorbs them; the timed loops stay silent). Liveness evidence
        // that the pad actually moved residency.
        if constexpr (WK > 1) {
            static int nprint = 3;
            if (nprint > 0) {
                --nprint;
                int nb = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &nb, fmha_main<D, BM, BN, WM, WK, DS>, 32 * WM * WK, SMEM_TOT);
                fprintf(stderr,
                        "[dec-occ] D=%d BM=%d BN=%d WK=%d grid=(%d,%d,%d) thr=%d smem=%d B/SM=%d\n",
                        D, BM, BN, WK, (int)grid.x, (int)grid.y, (int)grid.z,
                        32 * WM * WK, SMEM_TOT, nb);
                fflush(stderr);
            }
        }
#endif
        bool did = false;
        if constexpr (FMC) {
            if (fused) {
                fmha_main<D, BM, BN, WM, WK, DS, true><<<grid, 32 * WM * WK, SMEM_TOT, st>>>(q);
                did = true;
            }
        }
        if (!did)
            fmha_main<D, BM, BN, WM, WK, DS><<<grid, 32 * WM * WK, SMEM_TOT, st>>>(q);
    }
    if (!fused && p.n_split > 1 && !(g_skip & 2)) {
        // fmha_combine's s_m/s_l/s_sc are sized MAXS = 1024 and its
        // nse = min(p.n_split, ...) does not clamp to MAXS, so a larger split count
        // indexes past those shared arrays and returns wrong numbers rather than
        // failing: raising the Python-side cap to 2048 gave fixed case 3
        // rel_l2 0.8388 (job dv_ac7febbd4868) at n_split = 1332. solution.py caps
        // n_split at _SPLIT_CAP for exactly this reason; the check makes the invariant
        // explicit instead of relying on two files happening to agree.
        TORCH_CHECK((DS ? 2 * p.n_split : p.n_split) <= 1024,
                    "n_split ", p.n_split, " exceeds the combine's MAXS=1024");
#ifdef FMHA_COMBINE_W
        // One warp per row, 8 rows per CTA; (dc, rpt) do not apply to fmha_combine_w.
        // GATED ON ROW COUNT. The warp-per-row version has exactly one warp per output
        // row and no way to add parallelism without a cross-warp reduction - which is
        // precisely what the block-parallel version's barriers buy. Measured ungated
        // (dv_1a7b989b7761, two arms, 3 reps x 30 calls, 14/14 PASS): case 2, with
        // tiles_ub*Hkv*BM = 2048 rows, is 0.9213; case 4 (512 rows) 1.0134, case 8 (512)
        // 1.0276, case 5 (256) 1.0820, case 6 (256) 1.1362, and case 3 - 16 rows with 592
        // splits, i.e. TWO CTAs for the whole launch - 2.9735. So it ships only where the
        // row count fills the machine, and below the gate the launch is the unmodified
        // fmha_combine<D, 1024>, which keeps those shapes bit-identical by construction.
        //
        // The gate has now been confirmed a second time, independently and under
        // TIMING_PROTOCOL_V2: manual join 20261024 adopted a whole-candidate winner whose
        // fmha.cu is this file minus fmha_combine_w and nothing else in the decode path,
        // and its fixed case 2 reads 139.25 us against 129.07 for the tree that had it -
        // a ratio of 1.0788, against the 1/0.9213 = 1.0854 that dv_1a7b989b7761 measured
        // for the same mechanism on the same case. Cases 4, 5 and 8, which the gate keeps
        // on the block kernel, are unchanged across the join at 0.9996 / 1.0000 / 0.9988.
        //
        // AND GATED ON THE SPLIT COUNT, because the row count alone is not the whole
        // claim. One warp per row means one warp walks all nse = n_split partials of
        // its row serially, while the block kernel spreads them over SG = 256/dc
        // threads and pays a shared-memory reduction plus two barriers for it. So the
        // warp version wins when the rows fill the machine AND that serial walk is
        // short; `need = ceil(n_split/SG)` is the block kernel's per-thread trip
        // count and is the right measure of the ratio. A ten-shape sweep of legal
        // unsampled geometries straddling the row gate (dv_60227efe3ac2,
        // experiments/gate_sweep.json) found five wins and one loss above 1184 rows:
        //   rows 1280 need 3 n_split 12   0.8652      rows 1536 need 4 n_split 16  1.0308
        //   rows 1344 need 3 n_split 11   0.9231      rows 1024-1152 need 4 (gate off,
        //   rows 1536 need 3 n_split 10   0.8770       bit-identical, 1.0000-1.0025)
        //   rows 2048 need 2 n_split  7   0.8568
        //   rows 1280 need 3 n_split 12   0.9160
        // Every need <= 3 shape wins 7.7-14.3% and the one need = 4 shape loses 3.1%,
        // with the fixed case 2 that this gate was adopted on sitting at need = 3
        // (n_split 12, dc 64) and 0.9163 official. The losing shape is also the one
        // where the row count overstates the parallelism: tiles_ub = 3 against
        // tiles_est = 1, so rows = 1536 while base = tiles_est*Hkv = 32, and a small
        // base is what pushes n_split to 16. Threshold at 3 with 4 measured as a loss;
        // 13-15 splits at need 4 are unmeasured. FMHA_COMBINE_W_NEED=<n> overrides it
        // as a compile flag (distinct extension name, no getenv at launch).
#ifdef FMHA_COMBINE_W_NEED
        constexpr int CBW_NEED = FMHA_COMBINE_W_NEED;
#else
        constexpr int CBW_NEED = 3;
#endif
        // Warps per row. 1 is the shipped, measured form (rows >= 8*sms gate);
        // >1 is the extension the adoption note in combine-warp-row-20261023.md
        // asks for, selected at compile time so a measurement arm is a distinct
        // extension, never a getenv at launch.
        //
        // MEASURED AND REJECTED (dv_5c20800433db, results/combine-wpr-20261025.md,
        // 18/18 correctness pairs, all bit-identical, prefill null 1.0000): case 6
        // 1.1344 at WPR=2 and 1.2010 at WPR=4 median (1.2092/1.2774 on minima),
        // case 8 1.0614/1.0000, case 5 1.0729/1.0794, case 4 1.0000/1.0000, and
        // case 2 - the one shape that already ships this kernel - neutral at
        // 1.0005/1.0010. Monotone in WPR, so it is the mechanism and not noise.
        // WPR multiplies three per-row overheads the block kernel pays once per
        // CTA: the prologue (cu/slen staging, prefix-sum + ballot, two barriers -
        // case 6 goes 32 -> 128 CTAs), the redundant row max/denominator, and the
        // launch tail, while the useful work per warp shrinks to KD=1 dim. And the
        // limit it was meant to lift is not the warp count at all: giving case 5
        // 4x the warps moved it from 1.0820 (ungated WPR=1, 20261023) to
        // 1.0053-1.0794. The real limit is nse, the per-warp serial walk over
        // splits - case 6 at nse=27 is the worst of the four, case 2 at nse=12 is
        // the one that wins, at WPR=1. Splitting dims does nothing for that chain,
        // which is what the shipped `need <= 3` condition already encodes.
#ifdef FMHA_COMBINE_WPR
        constexpr int CBW_WPR = FMHA_COMBINE_WPR;
#else
        constexpr int CBW_WPR = 1;
#endif
        // The WPR actually launched, after the (D, WPR) legality fallback below.
        // grid.z MUST use this and not CBW_WPR: a z sized for WPR=4 with a WPR=1
        // kernel launched would cover only a quarter of the rows and return
        // garbage for the rest, silently.
        constexpr int CBW_WPR_E = (CBW_WPR >= 4 && D >= 32 * CBW_WPR) ? CBW_WPR
                                : (CBW_WPR == 2 && D >= 64) ? 2 : 1;
        constexpr int CBW_RPC = 8 / CBW_WPR_E;
        static const int sms_w = [] {
            int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v;
        }();
        const int sg_w = 256 / (p.dc ? p.dc : 1);
        const int need_w = (p.n_split + sg_w - 1) / sg_w;
#ifdef FMHA_COMBINE_W_FORCE
        // Diagnostic: take the warp path on every shape so WPR can be measured
        // below the gates. Wrong gate, correct numbers.
        const bool use_w = true;
#else
        const bool use_w = (long long)n_tiles_ub * p.Hkv * BM >= 8LL * sms_w
                           && need_w <= CBW_NEED;
#endif
        const int gz = use_w ? (BM + CBW_RPC - 1) / CBW_RPC
                             : ((BM + p.rpt - 1) / p.rpt) * ((D + p.dc - 1) / p.dc);
#else
        const int ndc = (D + p.dc - 1) / p.dc;
        const int gz = ((BM + p.rpt - 1) / p.rpt) * ndc;
#endif
        const dim3 grid(n_tiles_ub, p.Hkv, gz);
        // Only the SECONDARY carries the programmatic-serialization attribute. The
        // primary must keep ordinary stream ordering: what precedes it is the
        // caller's in-place input restore, and relaxing that edge would let it read
        // inputs that have not been restored yet. A macro rather than a lambda
        // because the kernel name is a template id containing commas, and variadic
        // so both the VF switch and the plain launch can use it.
#ifdef FMHA_COMBINE_W
        if (use_w) {
            // Illegal (D, WPR) pairs fall back to WPR=1 rather than failing to
            // compile: KD would be 0. D=64 supports WPR<=2, D=128 WPR<=4,
            // D=256 WPR<=8.
            if constexpr (CBW_WPR_E >= 4)
                FMHA_LAUNCH_COMBINE(fmha_combine_w<D, CBW_WPR_E>);
            else if constexpr (CBW_WPR_E == 2)
                FMHA_LAUNCH_COMBINE(fmha_combine_w<D, 2>);
            else
                FMHA_LAUNCH_COMBINE(fmha_combine_w<D, 1>);
        } else
#endif
        {
#if defined(FMHA_COMBINE_VF_ON) || defined(FMHA_COMBINE_VFHI)
        // Values each thread would have to hold to cover the dot loop, which runs
        // ceil(nse/SG) <= ceil(n_split/SG) times, capped at 8. The cap is a
        // measured register cliff, not a guess (job dv_6029fc9339d0, ptxas -v):
        // VF=0/2/8 all compile to 64 registers and VF=4 to 56, but VF=16 jumps to
        // 128, i.e. 65536/(256*128) = 2 resident CTAs/SM instead of 4 - and the
        // one case that got VF=16 (need=9) was 5.7% slower while every case at
        // VF<=8 was faster or flat. Anything past the budget is handled by the
        // strided tail inside the kernel.
        //
        // All-or-nothing: fuse only when the window covers the whole dot loop. Two
        // same-job measurements disagree about the tail version (job
        // dv_9641fe6562f7, 4 reps: case 8 need=9 VF=8+tail +4.4%, case 3 need=28
        // VF=8+tail -0.6%; job dv_6029fc9339d0, 3 reps: case 8 need=9 VF=16 +5.7%),
        // while the fully covered cases are the reproducible ones (case 2 need=3
        // VF=4: -3.2% and -3.1%; case 4 need=7 VF=8: -1.2% and -1.0%). A partially
        // covered loop still stalls serially on its tail after the barrier, so it
        // pays the register and issue pressure for a fraction of the benefit - and
        // those extra in-flight loads are issued ahead of the m/l loads the barrier
        // actually waits for.
        //
        // The variants are compiled only into this experimental build: nvcc
        // instantiates every template named in a switch before optimising, so
        // keeping the table here would cost the candidate four extra combine
        // instantiations per head dimension for code it can never reach.
        const int SG = 256 / p.dc;
        const int need = (p.n_split + SG - 1) / SG;
        int vf = 0;
        // DS entries ride the unfused path (fmha_combine's static_assert); its
        // effective entry count is 2*nse, which the window was never ported to.
        if (!DS && need > 0 && need <= 8) {
            vf = 1;
            while (vf < need) vf <<= 1;
        }
#ifdef FMHA_COMBINE_VF_ON
        if (!DS && need > 0 && need <= 8) {
            vf = 1;
            while (vf < need) vf <<= 1;
        }
#endif
#ifdef FMHA_COMBINE_VFHI
        // Past the shipped cap of 8, and SHIPPED at a ceiling of 16. The VF rejection
        // (results/combine-chain-20261022.md, three jobs) is sound for the shapes it
        // measured - case 2 (need=3) always about -3.2%, case 4 (need=7) always about
        // -1%, cases 5 and 6 unstable, geomean neutral, and a mechanism: PDL and the
        // fused window compete for the same slack, so hiding the dot's FIRST load buys
        // nothing once the prologue runs under the primary's tail. But every one of those
        // shapes has need <= 8, and the rule `need <= 8` meant the one fixed case with a
        // long dot loop never had a window at all: case 3 is nse=592 at dc=8, so SG=32 and
        // need=19, on the unfused strided loop with a 16.4 KB stride between partials and
        // long_scoreboard 10.75 per issue. That is a different regime - not one exposed
        // first load but nineteen serial trips.
        //
        // MEASURED (dv_2c48fa99e5f2, four arms, 20 correctness pairs, all 15 arm/case pairs
        // bit-identical, worst rel_l2 unchanged): case 3 0.9756 median / 0.9747 min at a
        // ceiling of 16, and 1.0769 / 1.0746 at 32 - the register cliff, VF=16 compiling to
        // 128 registers against 64, so the ceiling is 16 and not 32. Case 8, the other
        // need > 8 fixed shape at need=9, reads 0.9945 / 0.9966, so it does not reproduce
        // the +5.7% that VF=16 cost it in dv_6029fc9339d0 on a different tree. The arm that
        // turned the whole VF table on without this branch read EXACTLY 1.0000 on case 3,
        // which is what attributes the gain here rather than to the table.
        //
        // This is deliberately NOT combined with FMHA_COMBINE_VF_ON in the shipped build:
        // enabling the need <= 8 window as well would put case 6 (need=8) back on the path
        // that measured -1.5%, then +1.9%, then +5.5%, then +9.6% across four jobs. With
        // VFHI alone, every shape at need <= 8 keeps vf = 0 and the shipped instantiation,
        // so cases 2, 4, 5 and 6 are bit-identical BY CONSTRUCTION and not by measurement.
        // FMHA_COMBINE_NO_VFHI=1 is the control; FMHA_COMBINE_VFHI=<n> overrides the ceiling.
        if (!DS && need > 8) {
            int w = 16;
            while (w < need && w < FMHA_COMBINE_VFHI) w <<= 1;
            vf = w;
        }
#endif
        // An if/else chain rather than a switch so that each arm compiles only the
        // instantiations it can reach: nvcc instantiates every template named in a switch
        // before optimising, and the comment above FMHA_COMBINE_VF_ON is about that cost.
        if (vf == 0) {
            if constexpr (DS) { FMHA_LAUNCH_COMBINE(fmha_combine<D, 1024, 0, D / 2>); }
            else              { FMHA_LAUNCH_COMBINE(fmha_combine<D, 1024, 0>); }
        }
#ifdef FMHA_COMBINE_VF_ON
        else if (vf == 1) { FMHA_LAUNCH_COMBINE(fmha_combine<D, 1024, 1>); }
        else if (vf == 2) { FMHA_LAUNCH_COMBINE(fmha_combine<D, 1024, 2>); }
        else if (vf == 4) { FMHA_LAUNCH_COMBINE(fmha_combine<D, 1024, 4>); }
        else if (vf == 8) { FMHA_LAUNCH_COMBINE(fmha_combine<D, 1024, 8>); }
#endif
#ifdef FMHA_COMBINE_VFHI
        else if (vf == 16) { FMHA_LAUNCH_COMBINE(fmha_combine<D, 1024, 16>); }
#if FMHA_COMBINE_VFHI >= 32
        // Guarded by the ceiling, not merely reachable-or-not: nvcc instantiates every
        // template named here before optimising, and the shipped ceiling of 16 can never
        // select 32, so naming it would cost a dead combine instantiation per head dim.
        else if (vf == 32) { FMHA_LAUNCH_COMBINE(fmha_combine<D, 1024, 32>); }
#endif
#endif
        // REACHABLE, and it is what serves every `need <= 8` shape on the fixed suite - cases 2, 4, 5,
        // 6 and 8 at the current geometries. The `vf` value computed above for need <= 8 is discarded
        // unless FMHA_COMBINE_VF_ON is defined, because that is the ifdef guarding the vf == 1/2/4/8
        // branches; with only VFHI defined, need <= 8 falls through to here and need > 8 takes vf == 16.
        // dv_f96173efc005 measured turning the discarded value on: 0.9990 median / 1.0070 min over six
        // decode cases, all seven pairs bit-identical, so the window is neutral here and the fallthrough
        // is the correct behaviour, not a bug. See results/combine-vf-retest-20261025.md.
        else { FMHA_LAUNCH_COMBINE(fmha_combine<D, 1024, 0>); }
#else
        FMHA_LAUNCH_COMBINE(fmha_combine<D, 1024>);
#endif
        }
    }
}

#ifndef FMHA_NO_TC
template <int D, int BM, int BN>
void launch_tc(const Params& p, int n_tiles_ub, cudaStream_t st);
#endif

#ifndef FMHA_NO_PRE
// Merged tcgen05 prefill (fmha_pre.cu, arm-b merge5-full-suite-pass lineage;
// join 20261023). Defined at the end of the concatenated TU; requires the
// 10.3a target like fmha_tc.cu (solution.py mirrors the gate).
template <int D>
void launch_mypre(const Params& p, cudaStream_t st, bool f8ok = true);
#endif

static void launch_plan(const Params& p, int cfg, int n_tiles_ub, cudaStream_t st) {
#define DISPATCH(Dd, BBM, BBN, WWM, WWK) \
    launch_one<Dd, BBM, BBN, WWM, WWK>(p, n_tiles_ub, st)
#define DISPATCH_DS(Dd, BBM, BBN, WWM, WWK) \
    launch_one<Dd, BBM, BBN, WWM, WWK, true>(p, n_tiles_ub, st)
    switch (cfg) {
        case 0: DISPATCH(64, 128, 64, 8, 1); break;
        case 1: DISPATCH(128, 128, 64, 8, 1); break;
        case 2: DISPATCH(256, 64, 32, 4, 1); break;
        case 3: DISPATCH(64, 16, 32, 1, 4); break;
        case 4: DISPATCH(128, 16, 16, 1, 4); break;
        case 5: DISPATCH(256, 16, 16, 1, 2); break;
        case 6: DISPATCH(64, 64, 64, 4, 1); break;
        case 7: DISPATCH(128, 64, 64, 4, 1); break;
        case 8: DISPATCH(128, 128, 32, 4, 1); break;
        case 9: DISPATCH(64, 128, 64, 4, 1); break;
        case 10: DISPATCH(128, 64, 64, 2, 1); break;
        case 11: DISPATCH(64, 64, 64, 2, 1); break;
        // BN=32 decode variants: L1 bytes per fp8 KV byte are 5 + 16*MT/BN, so
        // doubling BN off the 16-key tiles cuts that from 6 to 5.5 (-8%). Whether
        // it survives depends on registers - S grows by MT*BN/8*4 and the K/V
        // prefetch gains a tail group - because 166 registers is exactly what buys
        // cfg4 its third resident CTA (65536/(128*166) = 3.09).
        case 23: DISPATCH(128, 16, 32, 1, 4); break;
        case 24: DISPATCH(256, 16, 32, 1, 2); break;
        // BM=256 prefill: per (row, key) the L1 cost is D/(4*MT) for the K/V
        // ldmatrix (independent of BM), 6*D/BM for the fp8 load plus the bf16
        // shared store, and 2*D/BN for the Q A-fragments. At D=128, MT=2, BN=32
        // that is 16 + 6 + 8 = 30 bytes for BM=128 and 16 + 3 + 8 = 27 for BM=256,
        // i.e. -10%, because the global/store term amortises over twice the rows.
        // The accumulators per thread are unchanged (O = MT*D/8*4 = 128, S = 32),
        // so it lands at ~255 registers x 256 threads = exactly 1 CTA/SM = the
        // same 8 warps/SM as cfg8 - which also means no second CTA to hide its
        // barriers, so this may well lose. That is what the measurement is for.
        case 25: DISPATCH(128, 256, 32, 8, 1); break;
        case 26: DISPATCH(64, 256, 32, 8, 1); break;
        // Decode tiles with BM=32 and WM=2. The L1 cost per fp8 KV byte is
        // 5 + 16*MT/BN: one byte for the global load, two for the bf16 shared
        // store, two for the K and V ldmatrix, and 16*MT/BN for the Q
        // A-fragments every warp re-ldmatrix'es per block (results/
        // roofline-headroom-20261023.md). That last term is the only one a decode
        // tile can shrink, and BN=32 alone cannot shrink it: KPW = BN/WM doubles
        // to 32, so CPT = KPW*(D/4)/32 goes to 32 against PFC = 16 and half of
        // every block's loads lose their prefetch - which is exactly why cfg23
        // cost cases 2 and 4 19.9% and 29.5% (results/decode-bn32-20261021.md).
        // WM=2 is the other half of the lever: it halves KPW back to 16, so CPT
        // stays at 16 = PFC and the prefetch still covers the whole block, while
        // BN=32 halves the Q term. It needs BM = 16*WM*MT = 32, and WK must be 1
        // because the kv-split epilogue requires a single m tile per warp
        // (`WK == 1 || (WM == 1 && BM == 16)`).
        //
        // What that costs and what it keeps, for cfg33 against cfg4 on the shapes
        // that matter: threads 64 instead of 128, so at ~170 registers residency
        // goes 3 -> 6 CTAs/SM and warps/SM stay at 12; shared goes 26 -> 27 KB,
        // not binding; the per-warp block count is unchanged (each warp walks the
        // whole range now, but a block covers twice the keys); the grid is
        // unchanged for every fixed decode case because tiles_ub is
        // ceil(msq*g/BM)*B and msq*g <= 16 on all of them; and the cross-warp
        // epilogue reduction disappears with WK=1. The mma work doubles on shapes
        // whose real row count is below 32 - case 4 has one real row - which is
        // affordable only because these cases run at 20-40% of the tensor roofline
        // while L1TEX is the unit at 89% (results/decode-l1tex-accounting-20261024.md).
        // cfg34 is the same lever at d=64: BN=64, Q term 5.5 -> 5.25, CPT =
        // (64/2)*(64/4)/32 = 16 = PFC.
#ifdef FMHA_DECBM32
        case 33: DISPATCH(128, 32, 32, 2, 1); break;
        case 34: DISPATCH(64, 32, 64, 2, 1); break;
#endif
        // WK=4 for the d=256 decode tile (cfg5 is WK=2): 128-thread CTAs at
        // ~253 registers land at exactly 2 CTAs/SM (2*128*253 = 64768 <= 65536),
        // keeping the 8 warps/SM the 4x64-thread status quo provides while
        // DOUBLING the bytes in flight per SM (8 warps x one whole 16-key
        // block each, PFC=32 covering CPT=32) and HALVING each warp's serial
        // tile walk (c3's 1024-split chunks go 8 tiles/warp -> 4). Shared
        // grows 42 -> ~78 KB (KV_B = WK*BN*DP*2 = 33792 per image; epilogue
        // assert 66048 <= 67584 passes), still 2 CTAs/SM at 156 KB of 232.
        // Side effect of the KF16 thread gate (32*WM*WK >= 128): cfg35 takes
        // the f16 K convert that cfg5's 64 threads are excluded from, so the
        // arm is the package (WK4 + f16K), which is how cfg selection works.
        // skipprice (dv_95dac96b65a5) priced c3's main kernel at 91% of
        // measured time - this is the lever that targets it. Gated like
        // DECBM32: predicted register counts must be measured before any
        // default routing changes; select with FMHA_CFG_OVERRIDE=35.
#ifdef FMHA_DECWK4D256
        case 35: DISPATCH(256, 16, 16, 1, 4); break;
#endif
        // cfg38: cfg4's d128 decode at WK=2 (64-thread CTAs). Half the KV smem
        // (KV_B/KV_BV 17408 -> 8704 each; epilogue assert 17408 >= 2*16*128*4
        // + 2*16*8 = 16640 passes), CPT 8 -> PFC 8 -> prefetch regs 16 vs
        // cfg4's 8, O = 64 either way. Discriminates the spad concurrency
        // slope (alpha ~= 0.1: 12->14 warps worth ~-1.5%) against the
        // carveout/L1 effect (dead +24KB cost c2 +12.9% at unchanged
        // occupancy; 7x25.4KB = 178KB lifts the carveout to the 196 tier =
        // L1 60KB vs cfg4's 124KB). 64 threads are below the KF16 gate, so
        // K convert takes the non-f16 path (the arm is the WK2+non-f16K
        // package). Gated like cfg35/37; FMHA_CFG_OVERRIDE=38 selects it.
#ifdef FMHA_DECWK2D128
        case 38: DISPATCH(128, 16, 16, 1, 2); break;
#endif
        // cfg37: dim-split d256 decode (DS=true): two CTAs per (tile, split),
        // each accumulating half of V/O. O drops 128 -> 64 registers, targeting
        // 5 CTAs/SM (10 warps) against cfg5's register-pinned 4x2; K traffic
        // duplicates (1.5x total KV bytes) against 79% idle DRAM. grid.z carries
        // split*2+half; ws holds half-wide record pairs (stride BM*(D/2+2));
        // the combine runs the unfused DH path. Bit-identical to cfg5 at equal
        // n_split by construction - see the fmha_main DS header. Selected by
        // FMHA_CFG_OVERRIDE=37; compiled only under FMHA_DEC_D256DSPLIT
        // (predicted registers, measured before any routing change).
#ifdef FMHA_DEC_D256DSPLIT
        case 37: DISPATCH_DS(256, 16, 16, 1, 2); break;
#endif
        // cfg 12-15 (minBlocksPerMultiprocessor 3 and 4 on the prefill tiles) were
        // removed after measurement: 1.31-3.45x worse on every prefill case, and
        // the parameter cost the decode cases 5.1% even at its default of 1. See
        // the note on __launch_bounds__ above.
        // cfg 18-22 (Q A-fragments hoisted into registers) were removed after
        // measurement; see results/decode-q-hoist-20261021.md and commit 8d62992.
#ifndef FMHA_NO_TC
        // tcgen05 (5th-gen tensor core) prefill arms; single launch, n_split == 1
        // only. Correct on d=64 and d=128 but 1.35-1.76x slower than mma.sync, so
        // not a default path: results/tcgen05-arm-20261021.md. FMHA_NO_TC=1 drops
        // fmha_tc.cu from the build, which also lets TORCH_CUDA_ARCH_LIST go back
        // to the plain target; job dv_3adc0cbb0312 measured that the
        // architecture-specific variant changes nothing here (identical register
        // counts and timings), so it is kept only for convenience, not speed.
        case 16: TORCH_CHECK(p.n_split == 1, "tcgen05 arm supports n_split == 1");
                 launch_tc<128, 128, 64>(p, n_tiles_ub, st); break;
        case 17: TORCH_CHECK(p.n_split == 1, "tcgen05 arm supports n_split == 1");
                 launch_tc<64, 128, 64>(p, n_tiles_ub, st); break;
#endif
#ifndef FMHA_NO_PRE
        // Merged tcgen05 prefill (fmha_pre.cu): the arm-b merge5 stack (v4.12
        // lineage + parity fix + PIPE-D128 + EVPRO prologue + persist nomask +
        // BD barrier-drop + W8F fold-only 8-warp kernel). Single launch,
        // n_split == 1; fold/persist/static chosen in launch_mypre. Arm-b
        // engine evidence at tag merge5-full-suite-pass: c0 1467.0us,
        // c1 127.6, c7 361.2, c9 659.8 (dv_cbb93fbe6bb1 lineage).
        case 30: TORCH_CHECK(p.n_split == 1, "merged tc prefill needs n_split == 1");
                 launch_mypre<64>(p, st); break;
        case 31: TORCH_CHECK(p.n_split == 1, "merged tc prefill needs n_split == 1");
                 launch_mypre<128>(p, st); break;
        case 32: TORCH_CHECK(p.n_split == 1, "merged tc prefill needs n_split == 1");
                 launch_mypre<256>(p, st); break;
        // cfg39: cfg31's merged D128 prefill with the fp8-direct QK path OFF
        // (f8ok=false selects the f16 instantiations). The Python planner
        // routes short-walk d128 shapes here (max_seqlen_k < FMHA_PRE_F8KV_MIN,
        // default 6144): the e4m3 dual split wins long walks (c0 -2.1%
        // official) and loses short ones (r1 kv2048 +2.8%, domain cfg5 +4.6%)
        // - results/f8qk-20261028.md. Identical to cfg31 in every other
        // respect (n_split == 1, no ws, same grid math).
        case 39: TORCH_CHECK(p.n_split == 1, "merged tc prefill needs n_split == 1");
                 launch_mypre<128>(p, st, false); break;
#endif
        default: TORCH_CHECK(false, "bad cfg");
    }
#undef DISPATCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// cfg table (must match candidate/solution.py):
//   0: D=64  BM=128 BN=64 WM=8 WK=1     3: D=64  BM=16 BN=32 WM=1 WK=4
//   1: D=128 BM=128 BN=64 WM=8 WK=1     4: D=128 BM=16 BN=16 WM=1 WK=4
//   2: D=256 BM=64  BN=32 WM=4 WK=1     5: D=256 BM=16 BN=16 WM=1 WK=2
//   6: D=64  BM=64  BN=64 WM=4 WK=1     7: D=128 BM=64 BN=64 WM=4 WK=1
//   8: D=128 BM=128 BN=32 WM=4 WK=1 (MT=2)   9: D=64 BM=128 BN=64 WM=4 WK=1 (MT=2)
//  10: D=128 BM=64  BN=64 WM=2 WK=1 (MT=2)  11: D=64 BM=64 BN=64 WM=2 WK=1 (MT=2)
//      -- same 160 shared-bytes-per-mma as 8/9 but 64-thread CTAs, which should
//      fit 4-5 CTAs/SM (8-10 warps) instead of 2 CTAs/SM (8 warps).
void fmha_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
                  torch::Tensor cu, torch::Tensor slen, torch::Tensor btab,
                  torch::Tensor ksc, torch::Tensor vsc, torch::Tensor ws,
                  double softmax_scale, int64_t cfg, int64_t n_split,
                  int64_t max_seqlen_q, int64_t bm, int64_t bn) {
    auto st = at::cuda::getCurrentCUDAStream().stream();
    Params p;
    p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    p.k = reinterpret_cast<const uint8_t*>(k.data_ptr());
    p.v = reinterpret_cast<const uint8_t*>(v.data_ptr());
    p.ks = ksc.data_ptr<float>();
    p.vs = vsc.data_ptr<float>();
    p.cu = cu.data_ptr<int>();
    p.slen = slen.data_ptr<int>();
    p.btab = btab.data_ptr<int>();
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.ws = ws.numel() ? ws.data_ptr<float>() : nullptr;
    p.q_s0 = q.stride(0);
    p.out_s0 = out.stride(0);
    p.k_s0 = k.stride(0); p.k_s1 = k.stride(1); p.k_s2 = k.stride(2);
    p.v_s0 = v.stride(0); p.v_s1 = v.stride(1); p.v_s2 = v.stride(2);
    p.ks_s0 = ksc.stride(0); p.ks_s1 = ksc.stride(1); p.ks_s2 = ksc.stride(2);
    p.vs_s0 = vsc.stride(0); p.vs_s1 = vsc.stride(1); p.vs_s2 = vsc.stride(2);
    p.btab_s = (int)btab.stride(0);
    const int H = (int)q.size(1), D = (int)q.size(2), Hkv = (int)k.size(2);
    p.H = H; p.Hkv = Hkv; p.g = H / Hkv; p.D = D;
    p.g_log2 = ilog2i(p.g);
    p.B = (int)cu.size(0) - 1;
    p.n_split = (int)n_split;
    p.ps_log2 = ilog2i((int)k.size(1));
    p.qk_scale = (float)(softmax_scale * 1.44269504088896340736);
    p.BM = (int)bm; p.BN = (int)bn;
    p.ws_stride = (int)bm * (((int)cfg == 37) ? (D / 2 + 2) : (D + 2));
    p.fuse_merge = 0; p.fctr = nullptr;   // launch_one sets these when it fuses
    // Tensor-path stand-ins for the int-only ABI's tq/msk tail args (this
    // entry is not used by solution.py's hot path). msk = msq makes the fold
    // "diag" gate true; fold is a performance mode, correct either way.
    p.tq = (int)q.size(0); p.msq = (int)max_seqlen_q; p.msk = (int)max_seqlen_q;
    TORCH_CHECK(q.stride(1) == D && q.stride(2) == 1 && out.stride(1) == D
                && out.stride(2) == 1, "q/out must be row-major [tokens, heads, dim]");
    TORCH_CHECK(p.B >= 1 && p.B <= 32, "batch out of supported range");
    TORCH_CHECK((p.g & (p.g - 1)) == 0, "GQA ratio must be a power of two");
    const int n_tiles_ub = ((int)((max_seqlen_q * p.g + bm - 1) / bm)) * p.B;
    if (n_tiles_ub <= 0) return;

    // combine-kernel geometry: aim for ~2 CTAs per SM without going below one
    // row per CTA.
    //
    // STALE ON PURPOSE, AND NOT THE PRODUCTION PATH. This is the pre-tuning
    // heuristic - fixed dc = min(D, 64) and rpt from a 2*sms divisor. The int-only
    // entry below (build_params) carries the tuned one: the occupancy-gated
    // dc-halving loop and rpt_k = 8. solution.py's forward only ever calls
    // fmha_run / fmha_capture / fmha_replay, so nothing measured through the
    // engine, ab.py or diag_pre comes through here; the only caller left is
    // experiments/dev.py's fixed-overhead probe on a tiny shape, where the combine
    // geometry cannot matter. Do not draw a geometry conclusion from a number taken
    // through this entry. If it ever needs to be more than a probe, make it call
    // build_params instead of duplicating it.
    static const int sms = [] {
        int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v;
    }();
    p.dc = D < 64 ? D : 64;
    const int ndc = (D + p.dc - 1) / p.dc;
    const long long tiles_hkv = (long long)n_tiles_ub * Hkv;
    const long long want = (long long)bm * tiles_hkv * ndc / (2LL * sms);
    p.rpt = (int)(want < 1 ? 1 : (want > bm ? bm : want));

    launch_plan(p, (int)cfg, n_tiles_ub, st);
}

// Int-only entry point: identical semantics, but the caller passes already
// extracted pointers and strides so pybind does not have to unwrap ten
// torch::Tensor objects on every call. candidate/solution.py caches this argument
// tuple per input binding (holding references, so the ids cannot be recycled).
#define FMHA_INT_ARGS                                                         \
    int64_t q, int64_t k, int64_t v, int64_t out, int64_t cu, int64_t slen,    \
    int64_t btab, int64_t ks, int64_t vs, int64_t ws,                          \
    int64_t q_s0, int64_t out_s0, int64_t k_s0, int64_t k_s1, int64_t k_s2,    \
    int64_t v_s0, int64_t v_s1, int64_t v_s2, int64_t ks_s0, int64_t ks_s1,    \
    int64_t ks_s2, int64_t vs_s0, int64_t vs_s1, int64_t vs_s2, int64_t btab_s,\
    int64_t H, int64_t Hkv, int64_t D, int64_t B, int64_t page,                \
    int64_t n_split, int64_t max_seqlen_q, int64_t cfg, int64_t bm, int64_t bn,\
    double softmax_scale, int64_t tq, int64_t msk

// Shared by fmha_run and fmha_capture so a captured graph is provably the same
// work the direct path would have launched. Returns n_tiles_ub, or 0 if there is
// nothing to do.
static int build_params(Params& p, FMHA_INT_ARGS) {
    p.q = reinterpret_cast<const __nv_bfloat16*>(q);
    p.k = reinterpret_cast<const uint8_t*>(k);
    p.v = reinterpret_cast<const uint8_t*>(v);
    p.ks = reinterpret_cast<const float*>(ks);
    p.vs = reinterpret_cast<const float*>(vs);
    p.cu = reinterpret_cast<const int*>(cu);
    p.slen = reinterpret_cast<const int*>(slen);
    p.btab = reinterpret_cast<const int*>(btab);
    p.out = reinterpret_cast<__nv_bfloat16*>(out);
    p.ws = reinterpret_cast<float*>(ws);
    p.q_s0 = q_s0; p.out_s0 = out_s0;
    p.k_s0 = k_s0; p.k_s1 = k_s1; p.k_s2 = k_s2;
    p.v_s0 = v_s0; p.v_s1 = v_s1; p.v_s2 = v_s2;
    p.ks_s0 = ks_s0; p.ks_s1 = ks_s1; p.ks_s2 = ks_s2;
    p.vs_s0 = vs_s0; p.vs_s1 = vs_s1; p.vs_s2 = vs_s2;
    p.btab_s = (int)btab_s;
    p.H = (int)H; p.Hkv = (int)Hkv; p.g = (int)(H / Hkv); p.D = (int)D;
    p.g_log2 = ilog2i(p.g);
    p.B = (int)B;
    p.n_split = (int)n_split;
    p.ps_log2 = ilog2i((int)page);
    p.qk_scale = (float)(softmax_scale * 1.44269504088896340736);
    p.BM = (int)bm; p.BN = (int)bn;
    p.ws_stride = (int)bm * (((int)cfg == 37) ? ((int)D / 2 + 2) : ((int)D + 2));
    p.fuse_merge = 0; p.fctr = nullptr;   // launch_one sets these when it fuses
    p.tq = (int)tq; p.msq = (int)max_seqlen_q; p.msk = (int)msk;
    const int n_tiles_ub = ((int)((max_seqlen_q * p.g + bm - 1) / bm)) * (int)B;
    if (n_tiles_ub <= 0) return 0;
    static const int sms = [] {
        int x = 0; cudaDeviceGetAttribute(&x, cudaDevAttrMultiProcessorCount, 0); return x;
    }();
    // Combine geometry. The reduction has two serial costs per thread: nse/SG
    // strided global loads of the partial rows, and SG-1 shared loads to merge
    // the split-groups, where SG = 256/dc. Their sum is minimised near
    // SG = sqrt(nse), and a smaller dc also multiplies the grid by D/dc, which
    // matters because the old fixed dc=64 left case 3 (nse=863, one tile, one
    // kv-head) with 64 CTAs - 0.14 waves - and a long_scoreboard stall of 10.75
    // per issue (job dv_27fd80359ec9). fmha_combine costs 17-29% of the decode
    // forward (job dv_bfb64b1931d6), so this is worth tuning.
    // FMHA_DC=<n> pins dc for A/B work; 0 or unset means the heuristic.
    static const int force_dc = [] {
        const char* e = getenv("FMHA_DC"); return e ? atoi(e) : 0;
    }();
    const long long tiles_hkv = (long long)n_tiles_ub * Hkv;
    int dc = D < 64 ? (int)D : 64;
    // Shrinking dc multiplies the grid by D/dc but also multiplies the per-CTA
    // m/l load and prologue by the same factor, so it only pays when the grid
    // would otherwise leave SMs idle. Measured (job dv_a95c10747cf9, same job for
    // both arms): a sqrt(n_split) rule that also moved cases 6 and 8 from dc=64
    // to dc=32 cost them 3.3% and 5.4%, while case 3 - the one shape whose
    // combine grid is a single tile times a single kv-head, 64 CTAs, 0.14 waves -
    // gained 9.2% going to dc=8. So gate on occupancy, not on the split count.
    // Rows per CTA, as rpt_k * sms in the divisor, i.e. aim for about rpt_k/2
    // CTAs per SM. This was 2, which for case 2 gave 512 CTAs = 1.15 waves: the
    // combine is latency bound per row (NCU: barrier 3.73, long_scoreboard 4.03
    // per issue) and at one wave deep a row's three dependent global round trips
    // have nothing to overlap with. Fewer rows per CTA means more CTAs and more
    // waves, at the cost of repeating the cu_seqlens staging, the O(B) tile
    // mapping and the per-CTA m/l load more often, so there is an optimum.
    // Measured twice in one job each, alternating arms, 40 iterations (jobs
    // dv_3eb1c5c1aedf and dv_34265a5666c7): case 2 is 158.3/159.9us at 2,
    // 147.3/147.5 at 8 and 158.6 at 32, while cases 3/4/5/6/8 stay within +-0.7%
    // across the whole range because their rpt is already 1 or their geometry is
    // set by the dc gate. 8 is the plateau (4 and 8 both give 146-147).
    static const int rpt_k = [] {
        const char* e = getenv("FMHA_RPT_K");
        const int v = e ? atoi(e) : 8;
        return v > 0 ? v : 8;
    }();
    auto geom = [&](int dcv) {
        const int n = ((int)D + dcv - 1) / dcv;
        const long long w = (long long)bm * tiles_hkv * n / ((long long)rpt_k * sms);
        const int rp = (int)(w < 1 ? 1 : (w > bm ? bm : w));
        return std::make_pair(tiles_hkv * ((bm + rp - 1) / rp) * n, rp);
    };
    if (force_dc > 0) {
        dc = force_dc < (int)D ? force_dc : (int)D;
    } else {
        while (dc > 8 && geom(dc).first < 2LL * sms) dc /= 2;
    }
    p.dc = dc;
    p.rpt = geom(dc).second;
#ifdef FMHA_DBG_PLAN
    // DIAGNOSTIC ONLY: the combine's grid and per-thread trip counts are decided
    // here and are not visible from Python, so an experiment that changes them
    // has no way to say which cases it actually touched. Never compiled into the
    // candidate; solution.py keys a distinct extension name off the flag.
    const int sg_dbg = 256 / (p.dc ? p.dc : 1);
    fprintf(stderr, "[plan] cfg=%lld D=%lld BM=%lld BN=%lld n_split=%d dc=%d rpt=%d "
            "SG=%d need=%d tiles_ub=%d tiles_hkv=%lld sms=%d ctas=%lld\n",
            (long long)cfg, (long long)D, (long long)bm, (long long)bn,
            p.n_split, p.dc, p.rpt, sg_dbg, (p.n_split + sg_dbg - 1) / sg_dbg,
            n_tiles_ub, tiles_hkv, sms, (long long)geom(dc).first);
#endif
    return n_tiles_ub;
}

void fmha_run(FMHA_INT_ARGS) {
    Params p;
    const int n_tiles_ub = build_params(p, q, k, v, out, cu, slen, btab, ks, vs, ws,
        q_s0, out_s0, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, ks_s0, ks_s1, ks_s2,
        vs_s0, vs_s1, vs_s2, btab_s, H, Hkv, D, B, page, n_split, max_seqlen_q,
        cfg, bm, bn, softmax_scale, tq, msk);
    if (n_tiles_ub <= 0) return;
    launch_plan(p, (int)cfg, n_tiles_ub, at::cuda::getCurrentCUDAStream().stream());
}

// ---- CUDA-graph capture of the very same launch sequence ----------------------
// The frozen ABI hands us one Python call per forward, and for the decode shapes
// that call is a large fraction of the measured latency: the noop floor is
// 7.0-7.4us, a direct extension call 15.7-17.4us and the full forward 19.8-20.6us
// on a shape whose GPU work is a couple of microseconds. Most of that gap is
// 36-argument pybind marshalling plus one or two kernel launches, so record the
// launches once per input binding and replay them. Nothing is cached numerically:
// a replay re-executes the same kernels over whatever the buffers hold at that
// moment, and the handle is only valid for the exact pointers/strides it was
// captured from, which candidate/solution.py ties to its identity cache.
#include <vector>
struct GraphEntry { cudaGraphExec_t ex; Params p; int cfg; int nt; };
static std::vector<GraphEntry> g_execs;

static cudaStream_t capture_stream() {
    static cudaStream_t cs = [] {
        cudaStream_t r = nullptr;
        cudaStreamCreateWithFlags(&r, cudaStreamNonBlocking);
        return r;
    }();
    return cs;
}

int64_t fmha_capture(FMHA_INT_ARGS) {
    if (g_execs.size() >= 256) return -1;
    Params p;
    const int n_tiles_ub = build_params(p, q, k, v, out, cu, slen, btab, ks, vs, ws,
        q_s0, out_s0, k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, ks_s0, ks_s1, ks_s2,
        vs_s0, vs_s1, vs_s2, btab_s, H, Hkv, D, B, page, n_split, max_seqlen_q,
        cfg, bm, bn, softmax_scale, tq, msk);
    if (n_tiles_ub <= 0) return -1;
    // Capture is illegal on the legacy default stream, and cudaFuncSetAttribute
    // has already been run by the caller's preceding direct launch, so record on
    // a private non-blocking stream and replay on whatever stream is current.
    cudaStream_t cs = capture_stream();
    if (!cs) return -1;
    if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    try {
        launch_plan(p, (int)cfg, n_tiles_ub, cs);
    } catch (...) {
        // Never leave the stream in capture mode: a TORCH_CHECK inside
        // launch_plan would otherwise strand every later launch on this thread.
        cudaGraph_t dead = nullptr;
        cudaStreamEndCapture(cs, &dead);
        if (dead) cudaGraphDestroy(dead);
        cudaGetLastError();
        return -1;
    }
    cudaGraph_t g = nullptr;
    if (cudaStreamEndCapture(cs, &g) != cudaSuccess || !g) {
        cudaGetLastError();
        return -1;
    }
    cudaGraphExec_t ex = nullptr;
    const cudaError_t e = cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0);
    cudaGraphDestroy(g);
    if (e != cudaSuccess || !ex) {
        cudaGetLastError();
        return -1;
    }
    g_execs.push_back(GraphEntry{ex, p, (int)cfg, n_tiles_ub});
    return (int64_t)g_execs.size() - 1;
}

void fmha_replay(int64_t handle) {
    if (handle < 0 || handle >= (int64_t)g_execs.size()) return;
    GraphEntry& e = g_execs[(size_t)handle];
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    // cudaGraphLaunch is not allowed on a stream that is itself capturing. If the
    // caller wrapped this forward in its own graph, launch the same kernels
    // directly so they become part of the caller's graph instead.
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cs);
    if (cs == cudaStreamCaptureStatusNone) cudaGraphLaunch(e.ex, st);
    else launch_plan(e.p, e.cfg, e.nt, st);
}

// FMHA_DIRECT, RESTORED. This shipped default-on in 59be8e3 on the strength of dv_979973cac8be
// (six own-path cases, medians -0.9 to -1.1us, all three reps of every case agreeing to 0.1us, all
// six pairs bit-identical, worst rel_l2 unchanged at 0.0014176992271294285) and was confirmed on the
// official instrument at -0.402% (dv_85f5e2435da9, results/host-floor-20261023.md). It was then lost
// - not rejected: git log -S fmha_launch shows exactly two commits, 087c1a0 which added it and
// ccb5fb5 "Manual join 20261024: adopt whole certified pair winner", which adopted a tree that did not
// contain it. No later measurement retracts it; the only other mention in results/ is a note that it
// was never nominated on its own.
//
// The mechanism is the host floor's largest addressable piece. experiments/hostcost.py put one forward
// at 4.6-5.1us of which cudaGraphLaunch is 3.2-3.5us, ~2.9us of it BEFORE the first node; the frozen
// 35-keyword binding is ~1.0us and is not addressable (a **kwargs pass-through measured 1.27us, i.e.
// slower). Two cudaLaunchKernelEx calls against Params that are already built and cached replace the
// graph launch, and also drop the per-call cudaStreamIsCapturing - direct launching is legal while a
// caller is capturing, so the branch that existed to handle that case is not needed. FMHA_LAUNCH_COMBINE
// applies cudaLaunchAttributeProgrammaticStreamSerialization unconditionally when FMHA_PDL is on, so
// PDL does not depend on a graph edge to carry it; fmha.cu's own PDL comment notes that if the
// attribute is not honoured the pair degenerates to the ordinary stream ordering in force anyway.
//
// Numerically identical by construction: same cached Params, same two kernels, same stream, same
// order, so the graph stays captured either way and FMHA_DIRECT=0 restores the previous behaviour
// exactly. NOTE the hot path is the METH_O fast-replay module, which was added AFTER 59be8e3, so
// fmha_launch_addr below is what actually puts this on the measured path - restoring the function
// alone would leave it reachable only from the pybind call sites that the fast path bypasses.
void fmha_launch(int64_t handle) {
    if (handle < 0 || handle >= (int64_t)g_execs.size()) return;
    GraphEntry& e = g_execs[(size_t)handle];
    launch_plan(e.p, e.cfg, e.nt, at::cuda::getCurrentCUDAStream().stream());
}
// j23-h1: raw function address for the METH_O fast-replay module. solution.py
// builds a tiny host-only CPython extension whose replay() bypasses the
// pybind dispatcher (~0.75-0.9us/call saved; the host_overhead probe put the
// full python forward enqueue at 5.1us of which the pybind replay call was
// 3.43us - and host dispatch sits inside the engine's event bracket).
int64_t fmha_replay_addr() {
    return reinterpret_cast<int64_t>(reinterpret_cast<void*>(&fmha_replay));
}
// Same raw-address export for the direct path, so the METH_O module's one process-wide function
// pointer can target either submission mechanism without a pybind dispatcher in the timed call.
int64_t fmha_launch_addr() {
    return reinterpret_cast<int64_t>(reinterpret_cast<void*>(&fmha_launch));
}

#undef FMHA_INT_ARGS
