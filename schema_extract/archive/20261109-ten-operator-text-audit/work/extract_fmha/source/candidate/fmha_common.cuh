// Shared device helpers for the fmha_mxfp8_qkv pure CUDA/PTX candidate.
//
// Everything here is independently authored CUDA/PTX.  The layouts reproduce the
// frozen contract exactly:
//   * Q/K scale factors are E8M0 bytes in the F8_128x4 swizzle over a
//     [ceil(S/128)*128, D/32] logical matrix, one 32-wide block along D.
//   * V scale factors are E8M0 bytes in the F8_128x4 swizzle applied to the
//     TRANSPOSED [D, B*KH*cpv] matrix, i.e. 32-wide blocks along the KEY axis.
//   * Payloads are E4M3FN.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace fmha_mxfp8 {

// --- Programmatic dependent launch for the split path's kernel chain ---------------
// The split path is a chain of dependent kernels per forward -- gm -> segmax -> scan ->
// split -> combine for the two-pass scan (s2, s5), gm -> scan_flat -> split -> combine
// for the flat one (s3, s9) -- so it pays three or four dependent-launch boundaries on
// the four shapes whose helper kernels are 22-37% of the runtime. §37 measured what a
// boundary costs and what PDL recovers, on a paired equal-work A/B of a five-node
// dependent chain inside a graph (dv_afe226c376fe): 0.572 us per boundary with nothing
// to overlap, and ~80% of a successor's independent setup when there is something to
// hide. Four boundaries is 2.29 us -- 6.5% of s2, 7.4% of s3, 5.0% of s5, 1.7% of s9.
//
// Placement IS the safety argument:
//   * fmha_pdl_wait() is the FIRST statement of a successor, before any early return and
//     before any read of its predecessor's output. There it degenerates to full
//     serialisation, so it cannot be incorrect; it is the conservative end of the design.
//     The larger prize in §37 is moving it later, past the split kernel's TMEM alloc and
//     Q/K/V staging, which read nothing the scan produced -- not done here.
//   * fmha_pdl_trigger() is the LAST statement of a primary, after its writes. A block
//     that exited early counts as triggered, and these lower to griddepcontrol
//     .launch_dependents / .wait, which are per-thread rather than .sync.aligned, so the
//     divergent `if (row >= rows) return;` in these kernels is safe. That distinction is
//     not academic here: dv_cec046852e29 HUNG on a divergent per-lane .sync.aligned test.
// Both are no-ops when the grid has no programmatic dependency, so the same kernels stay
// correct on the direct-launch path and on the six fused shapes, which launch one kernel
// and have no chain at all.
// Default 1: dv_bc20ea1065b0 measured the four-boundary form at x1.0228 whole-suite
// geomean against a null band of six byte-identical fused shapes centered at exactly
// x1.0000 (0.999-1.004), with s2 +7.2%, s3 +7.4%, s5 +8.3%, s9 +0.5%.  -DFMHA_SPLIT_PDL=0
// reproduces the plain-launch tree exactly, so both A/B arms come from one source.
#ifndef FMHA_SPLIT_PDL
#define FMHA_SPLIT_PDL 1
#endif
#if (FMHA_SPLIT_PDL != 0) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define FMHA_PDL_LIVE 1
#else
#define FMHA_PDL_LIVE 0
#endif

// The FOURTH boundary, scan -> split, gated on its own because it is the only one that
// has to change a launch mechanism rather than a launch site: fmha_tcg_kernel takes two
// __grid_constant__ CUtensorMaps BY VALUE, so it cannot go through pdl_launch's
// cudaLaunchKernelEx template without deduction against a __grid_constant__ parameter,
// and instead goes through the C entry point cudaLaunchKernelExC with an explicit array
// of argument addresses (the pattern the mounted sm_103a reference uses).  Separate
// gate, separate arm, so a build failure or a mis-measurement is attributable.
//
// This is also the boundary where §37's LARGER prize lives, and the one where early
// launch is unambiguously safe.  The scan is a SINGLE CTA running a serial recurrence
// over T groups, so it cannot be starved by its own successor -- it is already resident
// before split can be triggered, because the trigger only fires once every scan thread
// has reached it.  And split is the dominant kernel of the chain, so overlapping its
// TMEM alloc, Q payload staging and Q scale writes with the scan's tail is worth more
// than the 0.572 us launch gap alone.  fmha_pdl_scan_wait() therefore sits LATE,
// immediately above the first read of p.mstate, not at kernel entry.
// Default 1 with FMHA_SPLIT_PDL: this boundary is most of the win.  In dv_bc20ea1065b0
// the three-boundary arm read x1.0046 and the four-boundary arm x1.0228, so scan -> split
// is worth about +1.8% of the whole-suite geomean on its own -- three times what the other
// three boundaries together recovered.  The absolute gains track §37's 0.572 us per
// boundary: s2 2.2 us, s3 2.0 us, s5 3.3 us (D=256 stages twice the Q bytes, so it has
// twice the prologue to hide), s9 0.6 us.
#ifndef FMHA_PDL_SCAN_SPLIT
#define FMHA_PDL_SCAN_SPLIT 1
#endif
#if (FMHA_PDL_LIVE != 0) && (FMHA_PDL_SCAN_SPLIT != 0)
#define FMHA_PDL_SCAN_LIVE 1
#else
#define FMHA_PDL_SCAN_LIVE 0
#endif

__device__ __forceinline__ void fmha_pdl_scan_wait() {
#if FMHA_PDL_SCAN_LIVE
  cudaGridDependencySynchronize();
#endif
}

__device__ __forceinline__ void fmha_pdl_scan_trigger() {
#if FMHA_PDL_SCAN_LIVE
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__device__ __forceinline__ void fmha_pdl_wait() {
#if FMHA_PDL_LIVE
  cudaGridDependencySynchronize();
#endif
}

__device__ __forceinline__ void fmha_pdl_trigger() {
#if FMHA_PDL_LIVE
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

constexpr int GROUP_KEYS = 128;
constexpr float RESCALE_THRESHOLD = 2.7725887222397812f;  // 4 * ln(2)

struct Params {
  const uint8_t* q;
  const uint8_t* k;
  const uint8_t* v;
  const uint8_t* sfq;
  const uint8_t* sfk;
  const uint8_t* sfv;
  __nv_bfloat16* o;
  float* lse;
  int64_t q_b, q_h, q_s;
  int64_t k_b, k_h, k_s;
  int64_t v_b, v_h, v_s;
  int64_t o_b, o_h, o_s;
  int64_t l_b, l_h, l_s;
  int B, H, KH, Sq, Skv, D, group;
  int Cq, Rpq, Rpkv, cpv;
  int64_t colatoms;
  float scale, bias;
  int causal, has_window, window_left, has_lse;
  // tensor-core launch geometry (filled by the host dispatcher)
  int qph;        // query rows per CTA per query head = BM / group
  int log2_qph;   // qph is always a power of two
  // Row packing for the BM=128 tcgen05 kernel (Phase D).  Its CTA covers twice
  // as many rows as the mma.sync kernel, so it needs its own tiling.
  int qph128;
  int log2_qph128;
  // split-KV multi-pass state (Phase C).  The delayed-maximum recurrence makes
  // safe_t path dependent, so a key-range split cannot be merged naively; these
  // buffers carry the exact group-maximum sequence instead.
  float* bm;          // [rows_total][T] group maxima of the masked scaled scores
  // [nseg][rows_total] per-segment maxima of `bm`, transposed like it.  The scan
  // is two passes: this is what lets the serial one skip a whole segment of groups
  // without reading them (see fmha_segmax_kernel).
  float* segmax;
  int nseg;           // ceil(T / FMHA_SCAN_SEG)
  float* mstate;      // [rows_total][T] delayed running maximum after each group
  float* mtail;       // [rows_total]    final running maximum
  float* part_a;      // [nsplit][rows_total][D] weighted numerator partials.
                      // Raw workspace pointer: the element type is `part_t<BF16>`
                      // (FP32, or BF16 where the plan gated it on - see
                      // FMHA_PARTIALS_BF16); device code addresses it through
                      // part_ptr<>/part_load and the split epilogue's stores.
  float* part_b;      // [nsplit][rows_total]    weighted denominator partials
  float* part_m;      // [nsplit][rows_total]    fused-scan only (FSCAN): each
                      // chunk's end state m_s, from which the FWGT combine
                      // forms the normalised weight exp(m_s - mtail) <= 1.
                      // Null unless Plan::scan_fused sized and laid it out.
  int T;              // total 128-key groups = ceil(Skv/128)
  int nsplit;
  int rows_total;     // B*H*Sq, row index = (b*H + h)*Sq + i
  int qtiles;         // query tiles in gridDim.x, used for the balanced order
  // Non-zero when this launch should walk its CTAs in descending work order
  // (see cta_index).  Set by the host only for a multi-wave launch whose mask
  // makes the per-tile key span grow with blockIdx.x.
  // lpt2 pair-class modes (FMHA_LPT_PAIR_MODE; re-implemented post-join on
  // this tree — provenance: results/summaries/2026-10-24-lpt2-pair-balance.md
  // and its archived job chain on the OLD tree, where mode 2 measured
  // s1 −1.4…−1.8% same-job and −1.181% official, bit-identical, before the
  // nomination was archived by join 20261025; new-tree evidence required):
  // a launch with sms < N <= slots starts at t=0 but co-locates pairs
  // (k, k+sms) under the list-scheduling fill; 2 = rr-complement (IDs [0,P)
  // heaviest-first pair with [P,N) lightest-first head-reversed),
  // 3 = adjacent-interleave (even IDs heavy, odd light; the old fill-model
  // discriminator, kept for completeness).  Both are bijections over
  // (tile, kvflat) — experiments/lpt2bij.py; numerics untouched.
  int lpt;
  int pair_p;       // P = device_sm_count() for the lpt==2/3 pair maps
};

// BF16 split-path numerator partials (`part_a`), GATED — ported from the
// a-side lineage (best/pb2_full) where pb1/pb2 measured it same-job and
// plan-identical: s9 -2.5..-3.0% (bit-identical - zero distribution), s5
// -4.2..-4.6%, d7 -1.1..-1.8% against clean controls, engine PASS with
// failures=[] on every treated shape (jobs dv_1d784bfd5ba0, dv_f06d144a18e1,
// dv_599735458b80; results/summaries/2026-10-21-pb1-bf16-partials.md).
// pb1's ungated form lost s3 +2.2% - a 2.4MB part_a shape where halved bytes
// cannot matter and the delta was codegen jitter - so the format is gated on
// the PLAN: `part_bf16` is set only where the FP32 part_a would be at least
// FMHA_PARTIALS_BF16_MIN_MB megabytes, and only on the tcgen05 path (the tc
// fallback epilogue stays FP32).  The kernels take the format as a TEMPLATE
// parameter, not a runtime branch: a runtime branch compiles both load paths
// into the combine (REG 30->~50), and the narrow combine's occupancy is
// exactly what s9's 8200-CTA dispatch needs.
//
// Why the bytes matter on THIS tree too: its own ap21 (dv_d5779d79adcd)
// measured the s5 split pass as the most memory-pipe-loaded kernel anywhere
// (Mem Busy 42.2%, L1/TEX 43.1%, L2 42.2% against 8.7% compute) with the
// part_a stores as the cause, and agy1 grew s5's part_a 6.3 -> 14.5 MB with
// the kernel growing 2.3x - superlinear in bytes.  stg128_scaled fixed the
// store-side sector utilisation (4.0-of-32-bytes -> 16B stores); this halves
// the byte count itself, on both the epilogue stores and the combine loads,
// and stacks with PDL (which overlaps launches, not bytes).
//
// Numerics (why this stays inside atol .01 / rtol .05 / mismatch 0.0):
// part_a holds w_s * num_s with w_s = exp(m_s - mtail) <= 1, and the combine
// divides by den = sum of the w_s-weighted denominator partials, so the
// effective per-channel weights form a convex combination.  The per-element
// error is therefore bounded by 2^-9 * (weighted mean |num_s,d|), i.e.
// ~2^-9 * E|V_d| ~ 0.002-0.008 across the declared generators, against the
// .01 absolute floor plus 5% relative budget; P >= 0 means there is no
// cross-group cancellation blow-up, and the per-segment means concentrate
// (>= 450 keys per segment on every gated shape).  BF16's FP32 exponent
// range cannot overflow (|part_a| <= den_s * max|V| ~ 1e6) and its subnormals
// are far below any tolerance-relevant magnitude.  Zero-distribution shapes
// (s9, d8) have exactly-zero partials and stay bit-identical.  `part_b` (the
// denominator partials: the contract's unrounded FP32 denominator and the
// LSE path stay bit-identical) and `bm`/`segmax`/`mstate`/`mtail` (the
// delayed-maximum scan recurrence - bf16-rounded group maxima could flip a
// 4*ln(2) fire decision and shift group weights by e^delta) stay FP32.
//
// plan_geometry's COST MODEL stays on the FP32 byte constant, so the A/B is
// plan-identical apart from the gate itself (pure byte delta on the treated
// shapes).
//
// PROMOTED ON THIS TREE (pb2t, post-join): default 1.  Same-job A/B on the
// unified b-side tree (dv_3ab08a4d5381): s5 **-5.09%** (engine PASS x2,
// non-identical by design), s9 **-1.38%** (bit-identical - zero
// distribution), d3 -0.91% / d7 -0.56% (engine PASS x2, inside the small-
// shape control bias of that job: s0 -2.45%, s2 -2.19% plan-identical),
// gated-out controls bit-identical with byte-identical plans; census clean
// (STACK:0 everywhere; narrow<128,bf16> REG 30->32, wide 32->34, split
// BF16PA instantiations 198-255 like their FP32 twins).  The b-tree wins
// are smaller than the a-side originals on s9 (PDL already overlaps the
// combine there) and larger on s5.
// DISABLED 2026-10-23 on measured evidence, not on the bound alone (§65, dv_6e63ffbdc713).
// experiments/bf16_margin.py built this tree twice from one source (BF16 1 and 0) and ran the
// frozen reference over legal gated geometries. On `peaked-128` (B1 H16 KH4 Sq255 Skv8191 D128
// causal -- (16,4,128) is one of the eight legal triples and every cross_field_invariant holds),
// which the shipped plan splits (nsplit=9) and gates in (part_a fp32 18.8 MB >= 6.0), bf16 exceeded
// the frozen tolerance on 3 of 4 seeds:
//     seed900536 ratio 1.1878  mismatched 2/522240  maxabs 0.06250   fp32 control 0.1424
//     seed901536 ratio 1.8076  mismatched 1/522240  maxabs 0.06250   fp32 control 0.1292
//     seed903536 ratio 1.3976  mismatched 2/522240  maxabs 0.06250   fp32 control 0.1391
// where ratio = max over ALL elements of |actual-ref| / (atol + rtol*|actual|), and
// max_mismatch_fraction is 0.0 so one element over fails the shape. The fp32 control is clean on the
// identical inputs every time, so the cause is isolated to this format. maxabs 0.0625 implies
// E_w[|V_d|] ~ 30, which `peaked` reaches because problem/input.py scales the operands by 3 before
// quantization; `normal` gives 7.7x headroom on the same geometries.
// NO GATE CAN FIX THIS. The failure depends on the input DISTRIBUTION, which the model never sees and
// which shape_range does not tie to geometry, so any gated geometry can be presented with `peaked`.
// Raising MIN_MB does not help: the failing geometry's part_a (18.8 MB) is LARGER than sampled d3's
// (18.7 MB), so a gate that excludes it excludes d3 and s5 too. A wider-exponent 16-bit format is not
// a fix either -- fp16 overflows (part_a reaches 72k-598k against 65504, §12) and its 2^-11 only moves
// the failure threshold from E_w[|V_d|] > 5.12 to > 20.5, which peaked's ~25-30 tail still reaches.
// bf16's relative error is scale-invariant, so rescaling cannot help.
// The sampled suites cannot see this: d3 is the only sampled `peaked` gated shape and it sits at 1.27x
// headroom, so the engine's 10/10 pass there is luck rather than margin. s9 is `zero` (exactly-zero
// partials, bit-identical) and s5 is `normal` (7.7x).
// The bound in the comment above is CORRECT in form (the bias is in both numerator and denominator so
// it cancels, and P >= 0 makes the triangle inequality give a probability-WEIGHTED MEAN, not the
// maximum that §49 assumed) -- it is simply not small enough at the `peaked` tail. Cost of disabling,
// measured by same-job A/B rather than assumed; arm a's own figures put the gains at s5 -5.09% and
// s9 -1.38%, i.e. ~+0.64% on the fixed geomean. One macro, so trivially reversible if an external
// decision judges the sampled-suite pass sufficient.
#ifndef FMHA_PARTIALS_BF16
#define FMHA_PARTIALS_BF16 0
#endif
#ifndef FMHA_PARTIALS_BF16_MIN_MB
#define FMHA_PARTIALS_BF16_MIN_MB 6.0
#endif
template <bool BF16>
using part_t = std::conditional_t<BF16, __nv_bfloat16, float>;
__device__ __forceinline__ float part_load(float v) { return v; }
__device__ __forceinline__ float part_load(__nv_bfloat16 v) { return __bfloat162float(v); }
template <class T>
__device__ __forceinline__ T* part_ptr(float* base, size_t elems) {
  return reinterpret_cast<T*>(base) + elems;
}
template <class T>
__device__ __forceinline__ const T* part_ptr(const float* base, size_t elems) {
  return reinterpret_cast<const T*>(base) + elems;
}

// Fused-scan combine normalisation (a-side fs3/fs4 lineage): the split
// epilogue's shipped weight expression verbatim - guarded exp(m_s - mtail),
// bounded by 1 because the delayed-maximum state is monotone non-decreasing,
// and 0 when either side is nonfinite (fully-masked rows), exactly as the
// epilogue's isfinite guard makes it.  Evaluated ONCE per split per CTA into
// shared memory by the FWGT combines (the a-side fs3 measurement: computing
// it per thread per round instead cost +9% on the treated shapes at
// REG:32/STACK:0 - pure instruction flood, dv_c8141de5e812).
__device__ __forceinline__ float fmha_chunk_wgt(const float* __restrict__ part_m, size_t idx,
                                                float mt) {
  const float ms = __ldg(part_m + idx);
  return (isfinite(mt) && isfinite(ms)) ? expf(ms - mt) : 0.0f;
}

// Work-descending CTA dispatch order.
//
// A causal or windowed mask makes query tile x cover a key span that grows with
// x, so with the hardware's linear dispatch order (blockIdx.x fastest) the
// longest CTAs are handed out last and the machine drains through a tail of
// ever-shorter work.  ncu on frozen shape 7 (job pI5, `fmha_tcg_kernel<128,0,2>`,
// 512 CTAs over 148 one-CTA-per-SM slots) measured 475996 SM-active cycles
// against 652654 elapsed: the average SM performs 112.4 group-units of work
// while the kernel runs for 154 group-units of time, i.e. 27% of the machine
// idles in that tail.  Longest-processing-time-first is the classical fix; a
// list-scheduling simulation of this exact launch reproduces the measured 153
// units for the natural order and 119 units (1.29x) for the descending one,
// where reversing only blockIdx.x - which keeps one long CTA per head and so
// still leaves the eight longest CTAs spread across the launch - gives 144.
// That is why pH22's x-only reversal measured nothing.
//
// The map is a bijection over the grid, so every CTA still covers exactly one
// (tile, batch*kv_head, split) triple and the numerical result is untouched;
// only which SM runs which triple changes.
struct CtaIndex {
  int x, y, z;
};
__device__ __forceinline__ CtaIndex cta_index(const Params& p) {
  CtaIndex r;
  r.x = (int)blockIdx.x;
  r.y = (int)blockIdx.y;
  r.z = (int)blockIdx.z;
  if (p.lpt == 0) return r;
  const unsigned nx = gridDim.x, ny = gridDim.y, nz = gridDim.z;
  const unsigned d = blockIdx.x + nx * (blockIdx.y + ny * blockIdx.z);
  if (p.lpt == 2 || p.lpt == 3) {
    // Pair-class complementary maps (lpt2).  Bijection proof and load
    // tables: experiments/lpt2bij.py.  Heavy side walks tiles descending
    // from the last; light side walks ascending with the head rank
    // reversed, which keeps the shared boundary tile disjoint because
    // (P % nyz) + ((N-P) % nyz) is 0 or nyz whenever nyz divides N.
    const unsigned nyz = ny * nz;
    const unsigned P = (unsigned)p.pair_p;
    const bool heavy = (p.lpt == 2) ? (d < P) : ((d & 1u) == 0u);
    const unsigned rank = heavy ? ((p.lpt == 2) ? d : (d >> 1)) : ((p.lpt == 2) ? (d - P) : (d >> 1));
    unsigned t, rem;
    if (heavy) {
      t = nx - 1u - rank / nyz;
      rem = rank % nyz;
    } else {
      t = rank / nyz;
      rem = nyz - 1u - (rank % nyz);
    }
    r.x = (int)t;
    r.y = (int)(rem % ny);
    r.z = (int)(rem / ny);
    return r;
  }
  const unsigned rev = nx * ny * nz - 1u - d;
  const unsigned yz = ny * nz;
  const unsigned qx = rev / yz;
  const unsigned rem = rev - qx * yz;
  r.x = (int)qx;
  r.y = (int)(rem % ny);
  r.z = (int)(rem / ny);
  return r;
}

// ---------------------------------------------------------------- conversions
__device__ __forceinline__ float e4m3_to_float(uint32_t byte) {
  const uint32_t sign = (byte & 0x80u) << 24;
  const uint32_t em = byte & 0x7fu;
  const uint32_t exponent = em >> 3;
  const uint32_t mantissa = em & 7u;
  const float subnormal = (float)mantissa * 0x1p-9f;
  const uint32_t sub_bits = __float_as_uint(subnormal) | sign;
  const uint32_t normal_bits = sign | ((exponent + 120u) << 23) | (mantissa << 20);
  return __uint_as_float(exponent == 0u ? sub_bits : normal_bits);
}

// Exact 2^(byte-127); E8M0 byte 0 maps to the fp32 subnormal 2^-127 that
// torch.exp2 also produces.
__device__ __forceinline__ float e8m0_to_float(uint32_t byte) {
  return byte == 0u ? 0x1p-127f : __uint_as_float(byte << 23);
}

// Exact 2^(byte-127) as BF16 (a power of two always has one significant bit).
__device__ __forceinline__ unsigned short e8m0_to_bf16_bits(uint32_t byte) {
  return (unsigned short)(byte == 0u ? 0x0040u : (byte << 7));
}

// Round-to-nearest-even quantize/dequantize through E4M3FN for x in [0, 448].
// Verified bit-identical to torch's clamp(0,448).to(float8_e4m3fn).float() and
// to cvt.rn.satfinite.e4m3x2.f32 over a 270336-point sweep.
__device__ __forceinline__ float qdq_e4m3(float x) {
  const int32_t biased = (int32_t)((__float_as_uint(x) >> 23) & 0xFFu) - 127;
  const int32_t shift = (biased >= -6) ? (biased - 3) : -9;
  const float step = __int_as_float((shift + 127) << 23);
  const float reciprocal = __int_as_float((127 - shift) << 23);
  return rintf(x * reciprocal) * step;
}

// Hardware saturating RNE pair conversion; the low byte holds cvt(x).
__device__ __forceinline__ unsigned short cvt_e4m3x2(float lo, float hi) {
  unsigned short out;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(out) : "f"(lo), "f"(hi));
  return out;
}

// One E4M3 byte as BF16 bits.  Exact: an E4M3 value carries four significant
// bits and BF16 carries eight, with the same exponent width as FP32, so no
// rounding mode can matter and the fp32 decode this builds on is reused.
// E4M3FN's NaN encoding is handled explicitly because the fp32 decode maps that
// byte to 480.0 instead (it cannot occur in a clamped input tensor, but the
// hardware pair convert produced a BF16 NaN and this keeps that behaviour).
__device__ __forceinline__ unsigned short e4m3_to_bf16_bits(uint32_t byte) {
  if ((byte & 0x7fu) == 0x7fu) {
    return (unsigned short)(((byte & 0x80u) << 8) | 0x7fc0u);
  }
  return (unsigned short)__bfloat16_as_ushort(__float2bfloat16_rn(e4m3_to_float(byte)));
}

// E4M3 byte pair -> BF16 pair.  Low byte -> low half.
//
// This used to be one `cvt.rn.bf16x2.e4m3x2`, which ptxas 13.2.51 assembles but
// ptxas 13.1.115 rejects with "Unexpected instruction types specified for
// 'cvt'" - and because every kernel in the translation unit is compiled
// regardless of whether the geometry reaches it, that one form failed the whole
// build on the gateway runtime (job dv_a5ed0467eaa9, eight errors: the four
// probability-fragment sites in fmha_tc.cuh times two instantiations).  The
// arithmetic form is exact and toolchain independent.  It only affects the Phase
// C fallback - all ten frozen shapes and the tcgen05-eligible domain configs take
// the tcgen05 path, and the Phase C users (domain d0/d1/d2) have 1..129 keys.
__device__ __forceinline__ unsigned e4m3x2_to_bf16x2(unsigned short bytes) {
  const unsigned lo = e4m3_to_bf16_bits(bytes & 0xffu);
  const unsigned hi = e4m3_to_bf16_bits((unsigned)bytes >> 8);
  return lo | (hi << 16);
}

// ------------------------------------------------------------------- layouts
// F8_128x4 byte offset of logical (row, col) inside one swizzled [Rp, C] slice.
__device__ __forceinline__ int64_t sf_rowwise_offset(int row, int col, int C) {
  return (int64_t)((((row >> 7) * (C >> 2)) + (col >> 2)) << 9) +
         ((int64_t)(row & 31) << 4) + ((int64_t)((row & 127) >> 5) << 2) +
         (int64_t)(col & 3);
}

// F8_128x4 byte offset for the key-axis V scales (swizzle over [D, B*KH*cpv]).
__device__ __forceinline__ int64_t sf_v_offset(int d, int key_block, int64_t atom_base,
                                               int64_t colatoms) {
  const int64_t atom = (int64_t)(d >> 7) * colatoms + atom_base + (key_block >> 2);
  return (atom << 9) + ((int64_t)(d & 31) << 4) + ((int64_t)((d & 127) >> 5) << 2) +
         (int64_t)(key_block & 3);
}

// Offset of (d, key-block) inside one 512-byte F8_128x4 atom group already
// staged in shared memory.  The atom group for one 128-key block covers every
// (d, cb) pair of that block, so the swizzle index collapses to this form.
__device__ __forceinline__ int sf_v_local_offset(int d, int cb_local) {
  return ((d >> 7) << 9) + ((d & 31) << 4) + (((d & 127) >> 5) << 2) + (cb_local & 3);
}

// XOR-swizzled byte offset inside a shared-memory FP8 tile of row width D.
__device__ __forceinline__ int fp8_tile_offset(int row, int d, int D) {
  return row * D + ((((d >> 4) ^ (row & 7)) << 4) | (d & 15));
}

// XOR-swizzled element offset inside a shared-memory BF16 tile of row width D.
__device__ __forceinline__ int bf16_tile_offset(int row, int d, int D) {
  return row * D + ((((d >> 3) ^ (row & 7)) << 3) | (d & 7));
}

// ------------------------------------------------------------------- PTX wraps
__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

// Shared-memory window base, hoisted once per kernel so tile accesses stay in
// 32-bit offsets.  Computing __cvta_generic_to_shared per access cost one CVTA
// plus 64-bit address arithmetic each time (Nsight Compute: 13% of the kernel's
// instructions were integer address work).
__device__ __forceinline__ uint32_t smem_window(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, unsigned& r0, unsigned& r1, unsigned& r2,
                                        unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t addr, unsigned& r0, unsigned& r1,
                                              unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x2(uint32_t addr, unsigned& r0, unsigned& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

__device__ __forceinline__ void mma_fp8_16832(float* c, const unsigned* a, const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_bf16_16816(float* c, const unsigned* a, const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Wide loads/stores as explicit 128-bit PTX.
//
// `__builtin_memcpy(&uint4, p, 16)` does NOT reliably become LDG.E.128 here: the
// row strides are runtime values so nvcc cannot prove 16-byte alignment, and it
// degraded the tile loads into byte accesses (Nsight Compute showed 37.8M global
// sectors, ~1.2 GB of L1 requests, for a 6.6 MB working set, with 56% of warp
// cycles stalled on long_scoreboard).  The addresses really are 16-byte aligned
// by construction: every payload base is 256-byte aligned, the row strides are
// multiples of head_dim (128/256) and the offsets are multiples of 16.
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void sts128(uint32_t addr, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w));
}

// ------------------------------------------------------------- cp.async (SM80+)
// Global->shared without touching registers, so the load latency of the next
// key group overlaps the current group's tensor-core and softmax work.  The
// src-size form zero-fills, which is exactly what an out-of-range key row needs.
__device__ __forceinline__ void cp_async16(uint32_t smem_dst, const void* gmem_src, bool valid) {
  const int src_size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_dst), "l"(gmem_src),
               "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int PENDING>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING));
}

__device__ __forceinline__ void sts32f(uint32_t addr, float v) {
  asm volatile("st.shared.f32 [%0], %1;" ::"r"(addr), "f"(v));
}

__device__ __forceinline__ float lds32f(uint32_t addr) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr));
  return v;
}

__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr));
  return v;
}

// Strict-aliasing-safe wide load of one lane's CPT contiguous E4M3 bytes.
template <int CPT>
__device__ __forceinline__ void load_fp8_block(const uint8_t* __restrict__ ptr, float* out) {
  uint32_t packed[2];
  // Explicitly sized global accesses: memcpy with a runtime-strided base does
  // not reliably become a vector load.
  if (CPT == 4) {
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(packed[0]) : "l"(ptr));
  } else {
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];"
                 : "=r"(packed[0]), "=r"(packed[1])
                 : "l"(ptr));
  }
#pragma unroll
  for (int w = 0; w < CPT / 4; ++w) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out[w * 4 + i] = e4m3_to_float((packed[w] >> (8 * i)) & 0xFFu);
    }
  }
}

}  // namespace fmha_mxfp8
