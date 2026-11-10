// Phase B tensor-core kernel for fmha_mxfp8_qkv.
//
// Measured on this SM103 part: fp8 mma.sync m16n8k32 = 2167 TFLOP/s, bf16
// mma.sync m16n8k16 = 576 TFLOP/s, HBM 6.2 TB/s, 228 KB smem/SM, 2032 MHz.
//
//   * QK^T uses FP8 `mma.sync.aligned.m16n8k32` straight on the stored E4M3
//     payloads.  One MMA k-step of 32 is exactly one MX block, so the E8M0
//     scales fold in as a rank-1 rescale of each block's partial accumulator
//     (sq[row,blk] * sk[key,blk]).  Power-of-two scaling is exact, so the sum
//     equals the FP32 reference value up to accumulation order.
//   * The probability path is bit-faithful to reference.py: increasing 128-key
//     groups, delayed maximum rescale at 4*ln(2), bias 16 (D128) / 1 (D256),
//     unrounded FP32 probabilities into the denominator and
//     `cvt.rn.satfinite.e4m3x2.f32` (verified equal to torch's clamp+RNE cast
//     over a 270336-point sweep) into the numerator.
//   * PV uses BF16 `mma.sync.m16n8k16` with `ldmatrix.x4.trans`, so no byte
//     transpose is needed.  V is decoded to BF16 with its key-axis E8M0 scale
//     folded in; E4M3 * 2^e has four significant bits, so that decode and the
//     E4M3-rounded probability are both exact in BF16, and the C-fragment ->
//     A-fragment hand-off for the second GEMM needs no register shuffles.
//   * K, V and both scale-factor atom groups are staged with `cp.async` into a
//     two-deep pipeline, so the next group's memory latency overlaps the current
//     group's tensor-core and softmax work.
//   * Query heads sharing one KV head are packed into the M dimension, so K and
//     V are read once per (batch, kv_head, query tile) rather than per head.
#pragma once

#include "fmha_common.cuh"
#include "fmha_tcg_ptx.cuh"

// Resident CTAs per SM ptxas must allow when allocating registers.  The shared
// memory footprint already caps D=128 at 2 and D=256 at 1, and pushing the
// register cap below that only caused spills (168 regs -> 1540 B of spill).
#ifndef FMHA_TC_MIN_BLOCKS
#define FMHA_TC_MIN_BLOCKS 2
#endif

namespace fmha_mxfp8 {

// Kernel passes.  PASS_FUSED is the single-kernel path.  The split-KV path runs
// GROUP_MAX -> scan -> SPLIT -> combine, because the delayed 4*ln(2) rescale
// makes safe_t depend on the entire preceding group-maximum sequence: a key
// range split cannot merge independently computed softmax states and still
// reproduce reference.py bit-for-bit, so the exact group maxima are materialised
// and replayed instead.
enum : int { PASS_FUSED = 0, PASS_GROUP_MAX = 1, PASS_SPLIT = 2 };

// PVMODE selects how the probability-value product is computed:
//   0 - BF16 mma.sync on a decoded V tile, output accumulator in registers
//       (the validated Phase B/C path);
//   1 - tcgen05 block-scaled FP8 MMA straight on the staged E4M3 V tile, with
//       V's sequence-axis E8M0 scales as SFB and the output accumulator in
//       TMEM.  That removes the whole BF16 V decode, the ldmatrix.trans traffic
//       and the 64/128 accumulator registers, at the cost of scattering the
//       quantized probabilities into an E4M3 smem tile.  It needs BM=128
//       because the block-scaled instruction only accepts M=128 (probe pD9:
//       M=64 and M=256 both fault with `illegal instruction`).
template <int D, int WARP, int PASS, int PVMODE = 0>
struct TcLayout {
  static constexpr int BM = 16 * WARP;
  static constexpr int BN = 128;
  static constexpr int CQ = D / 32;
  static constexpr int SFK_BYTES = CQ * 128;         // raw K scale atoms / group
  static constexpr int SFV_BYTES = (D / 128) * 512;  // raw V scale atoms / group
  // The group-maximum pass never touches V, so it gets a much smaller footprint
  // and therefore far more resident CTAs, which is the whole point of splitting.
  static constexpr int WANT_V = (PASS != PASS_GROUP_MAX);
  static constexpr int NV = (WANT_V && PVMODE == 0) ? 1 : 0;
  static constexpr int NP = (WANT_V && PVMODE == 1) ? 1 : 0;
  static constexpr int NST = 2 * WANT_V;
  static constexpr size_t oQ = 0;
  static constexpr size_t oKraw = oQ + (size_t)BM * D;
  static constexpr size_t oVraw = oKraw + 2u * (size_t)BN * D;
  static constexpr size_t oV = oVraw + (size_t)NST * BN * D;
  static constexpr size_t oP = oV + (size_t)NV * BN * D * 2;
  static constexpr size_t oSfKraw = oP + (size_t)NP * BM * BN;
  static constexpr size_t oSfVraw = oSfKraw + 2u * (size_t)SFK_BYTES;
  static constexpr size_t oSfQ = oSfVraw + (size_t)NST * SFV_BYTES;
  static constexpr size_t oSfK = oSfQ + (size_t)CQ * BM * 4;
  static constexpr size_t oSfV = oSfK + (size_t)CQ * BN * 4;
  static constexpr size_t oRow = oSfV + (size_t)NV * 4u * (D / 2) * 4;
  // per-row scratch for PVMODE 1: [0]=alpha [1]=denominator [2]=maximum [3]=weight
  static constexpr size_t bytes =
      oRow + (size_t)(PVMODE == 1 ? 4 * BM * 4 + 1024 : 0) + 64;
};

// One pipeline stage: stage `stage` of the shared-memory K/V/scale buffers with
// the 128-key group starting at `grp`.  Out-of-range key rows zero-fill through
// cp.async's source-size operand.
template <int D, int WARP, int PASS, int PVMODE = 0>
__device__ __forceinline__ void tc_issue_stage(const Params& p, int tid, int stage, int grp,
                                               uint32_t sKraw_u, uint32_t sVraw_u,
                                               uint32_t sSfKraw_u, uint32_t sSfVraw_u,
                                               const uint8_t* k_base, const uint8_t* v_base,
                                               const uint8_t* sfk_slice, int64_t v_atom_base) {
  constexpr int BN = 128;
  constexpr int THR = 32 * WARP;
  constexpr int CHQ = D / 16;
  constexpr int CQ = D / 32;
  const int keyBase = grp << 7;
  const uint32_t kdst = sKraw_u + (uint32_t)(stage * BN * D);
  const uint32_t vdst = sVraw_u + (uint32_t)(stage * BN * D);
#pragma unroll 2
  for (int it = 0; it < (BN * CHQ) / THR; ++it) {
    const int c = it * THR + tid;
    const int j = c / CHQ;
    const int ci = c - j * CHQ;
    const int jg = keyBase + j;
    const bool ok = (jg < p.Skv);
    const int jc = ok ? jg : (p.Skv - 1);
    const uint32_t soff = (uint32_t)fp8_tile_offset(j, ci * 16, D);
    cp_async16(kdst + soff, k_base + (int64_t)jc * p.k_s + ci * 16, ok);
    if (PASS != PASS_GROUP_MAX) {
      // A MN-major UMMA operand wants 128-byte swizzle chunks keyed by row; for
      // D=128 that is the same tile, for D=256 it splits d into two chunks.
      const uint32_t voff = (PVMODE == 1)
                                ? (uint32_t)((ci >> 3) * (BN * 128) +
                                             fp8_tile_offset(j, (ci & 7) * 16, 128))
                                : soff;
      cp_async16(vdst + voff, v_base + (int64_t)jc * p.v_s + ci * 16, ok);
    }
  }
  {
    const uint8_t* atom = sfk_slice + (int64_t)grp * (CQ >> 2) * 512;
    const uint32_t dst = sSfKraw_u + (uint32_t)(stage * TcLayout<D, WARP, PASS, PVMODE>::SFK_BYTES);
    for (int c = tid; c < CQ * 8; c += THR) cp_async16(dst + c * 16, atom + c * 16, true);
  }
  if (PASS != PASS_GROUP_MAX) {
    const uint8_t* atom = p.sfv + ((int64_t)v_atom_base + grp) * 512;
    const uint32_t dst = sSfVraw_u + (uint32_t)(stage * TcLayout<D, WARP, PASS, PVMODE>::SFV_BYTES);
    for (int c = tid; c < (D / 128) * 32; c += THR) {
      cp_async16(dst + c * 16, atom + (int64_t)(c >> 5) * p.colatoms * 512 + (c & 31) * 16, true);
    }
  }
}

template <int D, int WARP, int PASS, int PVMODE = 0>
// The tcgen05 PV path frees the 64/128 accumulator registers, so D=128 can keep
// two resident CTAs; D=256 needs its whole 512-column TMEM and 184 KB of smem,
// where a two-block request would only force spills.
__global__ void __launch_bounds__(32 * WARP,
                                 (PVMODE == 1) ? (D == 128 ? 2 : 1) : FMHA_TC_MIN_BLOCKS)
fmha_tc_kernel(const Params p) {
  constexpr int BM = 16 * WARP;
  constexpr int BN = 128;
  constexpr int THR = 32 * WARP;
  constexpr int CQ = D / 32;
  constexpr int CHQ = D / 16;
  constexpr int NT = BN / 8;
  constexpr int KB = D / 32;
  constexpr int KK = BN / 16;
  constexpr int NTOT = D / 8;
  constexpr int NTP = D / 16;
  constexpr bool WANT_V = (PASS != PASS_GROUP_MAX);
  constexpr bool TCG = (PVMODE == 1);
  static_assert(!TCG || (WARP == 8 && PASS != PASS_GROUP_MAX),
                "the tcgen05 PV path needs BM=128 and a pass that uses V");
  using L = TcLayout<D, WARP, PASS, PVMODE>;

  extern __shared__ __align__(16) char smem_raw[];
  // One hoisted shared-window base: every tile access then stays a 32-bit
  // offset instead of a fresh CVTA plus 64-bit address arithmetic.  The UMMA
  // descriptors drop the low 4 address bits and the SW128 XOR reads bits [9:7],
  // so the tcgen05 path works from a 1024-byte aligned base -- and every
  // generic pointer into the same buffers has to use that base too, or cp.async
  // writes and the reads that consume them disagree by up to 1023 bytes.
  char* const smem_base =
      TCG ? reinterpret_cast<char*>(((uintptr_t)smem_raw + 1023u) & ~(uintptr_t)1023u) : smem_raw;
  const uint32_t smem_u = smem_window(smem_base);
  const uint32_t sQ_u = smem_u + (uint32_t)L::oQ;
  const uint32_t sKraw_u = smem_u + (uint32_t)L::oKraw;
  const uint32_t sVraw_u = smem_u + (uint32_t)L::oVraw;
  const uint32_t sV_u = smem_u + (uint32_t)L::oV;
  const uint32_t sSfKraw_u = smem_u + (uint32_t)L::oSfKraw;
  const uint32_t sSfVraw_u = smem_u + (uint32_t)L::oSfVraw;
  float* sSfQ = reinterpret_cast<float*>(smem_base + L::oSfQ);
  float* sSfK = reinterpret_cast<float*>(smem_base + L::oSfK);
  unsigned* sSfV = reinterpret_cast<unsigned*>(smem_base + L::oSfV);
  const uint32_t sP_u = smem_u + (uint32_t)L::oP;
  // per-row scratch for the tcgen05 PV path: alpha, denominator, maximum, weight
  float* sRow = reinterpret_cast<float*>(smem_base + L::oRow);
  __shared__ __align__(8) uint64_t mbar_pv;
  __shared__ __align__(16) uint32_t tmem_base_s;
  // TMEM column map for PVMODE 1: the output accumulator first, then the two
  // scale-factor groups.  Bases stay 4-word aligned (2 is the hardware minimum).
  constexpr int TC_O = 0;
  constexpr int TC_SFA_P = (D + 3) & ~3;
  constexpr int TC_SFB_V = TC_SFA_P + 4;
  constexpr int TC_ALLOC = (TC_SFB_V + 4 * (D / 128) <= 256) ? 256 : 512;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const int q = lane & 3;

  const int kvflat = blockIdx.y;
  const int b = kvflat / p.KH;
  const int kh = kvflat - b * p.KH;
  const int G = p.group;
  const int Qph = TCG ? p.qph128 : p.qph;
  const int log2Qph = TCG ? p.log2_qph128 : p.log2_qph;
  // An experiment interleaved light and heavy causal query tiles so two resident
  // CTAs on one SM would sum to equal work.  Over the frozen suite it was a wash
  // (shape 7 -6%, shape 6 +13%, rest flat) because the block->SM pairing stride
  // is not 1, so it was reverted; evidence in results/raw/20261017T152651Z-tcD1.
  const int q0 = blockIdx.x * Qph;
  if (q0 >= p.Sq) return;

  const uint8_t* k_base = p.k + b * p.k_b + kh * p.k_h;
  const uint8_t* v_base = p.v + b * p.v_b + kh * p.v_h;
  const uint8_t* sfk_slice = p.sfk + (int64_t)kvflat * p.Rpkv * p.Cq;
  const int64_t v_atom_base = (int64_t)kvflat * (p.cpv >> 2);

  const int rowA = w * 16 + (lane >> 2);
  const int rowB = rowA + 8;
  const int iA = q0 + (rowA & (Qph - 1)), iB = q0 + (rowB & (Qph - 1));
  const int hA = kh * G + (rowA >> log2Qph), hB = kh * G + (rowB >> log2Qph);
  const bool validA = iA < p.Sq, validB = iB < p.Sq;
  // Padding rows (i >= Sq, when Sq is not a multiple of qph) have no global row
  // slot; clamp their index so the split-pass workspace reads stay in bounds.
  // Their results are never written back, so the clamped state is harmless.
  const int64_t rowGA = (int64_t)((b * p.H + hA) * p.Sq) + (validA ? iA : 0);
  const int64_t rowGB = (int64_t)((b * p.H + hB) * p.Sq) + (validB ? iB : 0);

  const int off = p.Skv - p.Sq;
  const int cA = iA + off, cB = iB + off;
  int hiA = min(p.causal ? cA : (p.Skv - 1), p.Skv - 1);
  int hiB = min(p.causal ? cB : (p.Skv - 1), p.Skv - 1);
  int loA = max(p.has_window ? (cA - p.window_left) : 0, 0);
  int loB = max(p.has_window ? (cB - p.window_left) : 0, 0);
  if (!validA) { loA = 0; hiA = p.Skv - 1; }
  if (!validB) { loB = 0; hiB = p.Skv - 1; }
  const unsigned laneLo = (unsigned)max(loA, loB);
  const unsigned laneHi = (unsigned)min(hiA, hiB);

  const int iMin = q0;
  const int iMax = min(q0 + Qph - 1, p.Sq - 1);
  const int hiMax = min(p.causal ? (iMax + off) : (p.Skv - 1), p.Skv - 1);
  const int loMin = max(p.has_window ? (iMin + off - p.window_left) : 0, 0);
  const int gLoAll = loMin >> 7;
  const int gHiAll = hiMax >> 7;

  // Key-range split.  Splits tile THIS CTA's useful group span [gLoAll, gHiAll]
  // rather than [0, T): with causal or windowed masks the useful span can be a
  // small interior slice, and tiling [0, T) left one split doing almost all the
  // work (measured: shape 8 regressed 197 -> 321 us).  Groups outside the span
  // are entirely masked for every row of this CTA, so their maximum is -inf and
  // they contribute nothing; the first and last split record those entries so
  // the scan still sees a fully initialised array.
  int cLo = gLoAll, cHi = gHiAll;
  if (PASS != PASS_FUSED) {
    const int span = gHiAll - gLoAll + 1;
    const int per = (span + p.nsplit - 1) / p.nsplit;
    cLo = gLoAll + blockIdx.z * per;
    cHi = min(cLo + per - 1, gHiAll);
  }
  const bool any_compute = (cLo <= cHi);

  if (PASS == PASS_GROUP_MAX && q == 0) {
    if (blockIdx.z == 0) {
      for (int t = 0; t < gLoAll; ++t) {
        if (validA) p.bm[(size_t)t * p.rows_total + rowGA] = -INFINITY;
        if (validB) p.bm[(size_t)t * p.rows_total + rowGB] = -INFINITY;
      }
    }
    if (blockIdx.z == p.nsplit - 1) {
      for (int t = gHiAll + 1; t < p.T; ++t) {
        if (validA) p.bm[(size_t)t * p.rows_total + rowGA] = -INFINITY;
        if (validB) p.bm[(size_t)t * p.rows_total + rowGB] = -INFINITY;
      }
    }
  }

  if (any_compute) {
    tc_issue_stage<D, WARP, PASS, PVMODE>(p, tid, 0, cLo, sKraw_u, sVraw_u, sSfKraw_u,
                                          sSfVraw_u, k_base, v_base, sfk_slice, v_atom_base);
    cp_async_commit();
  }

  // ---- Q payload tile -----------------------------------------------------
#pragma unroll 2
  for (int it = 0; it < (BM * CHQ) / THR; ++it) {
    const int c = it * THR + tid;
    const int m = c / CHQ;
    const int ci = c - m * CHQ;
    const int i = q0 + (m & (Qph - 1));
    uint4 data = make_uint4(0u, 0u, 0u, 0u);
    if (i < p.Sq) {
      const int h = kh * G + (m >> log2Qph);
      data = ldg128(p.q + b * p.q_b + (int64_t)h * p.q_h + (int64_t)i * p.q_s + ci * 16);
    }
    sts128(sQ_u + (uint32_t)fp8_tile_offset(m, ci * 16, D), data);
  }
  for (int idx = tid; idx < CQ * BM; idx += THR) {
    const int kb = idx / BM;
    const int m = idx - kb * BM;
    const int i = q0 + (m & (Qph - 1));
    float sc = 0.0f;
    if (i < p.Sq) {
      const uint8_t* slice = p.sfq + (int64_t)(b * p.H + kh * G + (m >> log2Qph)) * p.Rpq * p.Cq;
      sc = e8m0_to_float(slice[sf_rowwise_offset(i, kb, p.Cq)]);
    }
    sSfQ[kb * BM + m] = sc;
  }
  __syncthreads();

  float accumulator[TCG ? 1 : NTOT][4];
  if (!TCG) {
#pragma unroll
    for (int t = 0; t < NTOT; ++t) {
      accumulator[t][0] = accumulator[t][1] = accumulator[t][2] = accumulator[t][3] = 0.0f;
    }
  }
  uint32_t tmem = 0, tlane = 0, pvWaitPhase = 0;
  if (TCG) {
    if (tid == 0) mbar_init(&mbar_pv, 1);
    if (tid < 32) tcg_alloc(&tmem_base_s, TC_ALLOC);
    __syncthreads();
    tmem = tmem_base_s;
    tlane = tmem + ((uint32_t)((w & 3) * 32) << 16);
    if (tid < 32) tcg_relinquish();
    // PV's A operand is the quantized probability itself, so its scale factor is
    // exactly 1.0: the contract's bias is already inside the E4M3 payload.
    if (w < 4) tcg_st1(tlane + TC_SFA_P + w, 0x7f7f7f7fu);
    tcg_wait_st();
  }
  float mA = -INFINITY, mB = -INFINITY;
  float denA = 0.0f, denB = 0.0f;
  if (PASS == PASS_SPLIT) {
    // Replay the exact delayed-maximum state this chunk inherits.
    if (cLo > 0) {
      mA = p.mstate[(size_t)(cLo - 1) * p.rows_total + rowGA];
      mB = p.mstate[(size_t)(cLo - 1) * p.rows_total + rowGB];
    }
  }

  for (int grp = cLo; grp <= cHi; ++grp) {
    const int stage = (grp - cLo) & 1;
    const int keyBase = grp << 7;

    if (TCG && grp > cLo) {
      // The previous group's PV MMA reads the V buffer that this iteration's
      // cp.async is about to overwrite (stage ^ 1 is the buffer group grp-1
      // used), so it has to retire first.  Waiting here still leaves that MMA
      // the whole probability/rescale tail of the previous iteration to finish
      // in, and with two resident CTAs the other CTA keeps the tensor core busy.
      mbar_wait(&mbar_pv, pvWaitPhase);
      pvWaitPhase ^= 1u;
      tcg_fence_after();
    }

    // issue the next tile, then wait for this one (at most one group in flight)
    if (grp + 1 <= cHi) {
      tc_issue_stage<D, WARP, PASS, PVMODE>(p, tid, stage ^ 1, grp + 1, sKraw_u, sVraw_u,
                                            sSfKraw_u, sSfVraw_u, k_base, v_base, sfk_slice,
                                            v_atom_base);
    }
    cp_async_commit();
    cp_async_wait<1>();
    if (TCG) fence_async_shared();  // staged K/V become UMMA operands
    __syncthreads();

    const uint32_t kstage = sKraw_u + (uint32_t)(stage * BN * D);
    const uint32_t vstage = sVraw_u + (uint32_t)(stage * BN * D);
    const uint8_t* sfkraw =
        reinterpret_cast<const uint8_t*>(smem_base + L::oSfKraw) + (size_t)stage * L::SFK_BYTES;
    const uint8_t* sfvraw =
        reinterpret_cast<const uint8_t*>(smem_base + L::oSfVraw) + (size_t)stage * L::SFV_BYTES;

#pragma unroll
    for (int it = 0; it < (CQ * BN) / THR; ++it) {
      const int idx = it * THR + tid;
      const int kb = idx / BN;
      const int j = idx - kb * BN;
      sSfK[kb * BN + j] =
          (keyBase + j < p.Skv) ? e8m0_to_float(sfkraw[sf_rowwise_offset(j, kb, CQ)]) : 0.0f;
    }
    if (WANT_V && !TCG) {
#pragma unroll
      for (int it = 0; it < (4 * (D / 2)) / THR; ++it) {
        const int idx = it * THR + tid;
        const int cb = idx / (D / 2);
        const int dh = idx - cb * (D / 2);
        const int d = dh * 2;
        const unsigned b0 = sfvraw[sf_v_local_offset(d, cb)];
        const unsigned b1 = sfvraw[sf_v_local_offset(d + 1, cb)];
        sSfV[idx] = e8m0_to_bf16_bits(b0) | (e8m0_to_bf16_bits(b1) << 16);
      }
    }
    __syncthreads();

    if (WANT_V && !TCG) {
      // ---- decode the staged V tile into BF16 with its scale folded in ----
#pragma unroll 2
      for (int it = 0; it < (BN * CHQ) / THR; ++it) {
        const int c = it * THR + tid;
        const int j = c / CHQ;
        const int ci = c - j * CHQ;
        const uint4 raw = lds128(vstage + (uint32_t)fp8_tile_offset(j, ci * 16, D));
        const unsigned* scales = sSfV + (j >> 5) * (D / 2) + ci * 8;
        const unsigned raw_words[4] = {raw.x, raw.y, raw.z, raw.w};
        unsigned words[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          const unsigned short byte_pair =
              (unsigned short)((raw_words[t >> 1] >> ((t & 1) * 16)) & 0xFFFFu);
          const unsigned bf16_pair = e4m3x2_to_bf16x2(byte_pair);
          const __nv_bfloat162 scale2 = *reinterpret_cast<const __nv_bfloat162*>(&scales[t]);
          const __nv_bfloat162 unscaled = *reinterpret_cast<const __nv_bfloat162*>(&bf16_pair);
          const __nv_bfloat162 scaled = __hmul2(unscaled, scale2);
          words[t] = *reinterpret_cast<const unsigned*>(&scaled);
        }
        uint4 lo4, hi4;
        __builtin_memcpy(&lo4, words, 16);
        __builtin_memcpy(&hi4, words + 4, 16);
        sts128(sV_u + 2u * (uint32_t)bf16_tile_offset(j, ci * 16, D), lo4);
        sts128(sV_u + 2u * (uint32_t)bf16_tile_offset(j, ci * 16 + 8, D), hi4);
      }
      __syncthreads();
    }

    // ---- QK^T with the per-MX-block rank-1 rescale ------------------------
    // The score tile lives in packed f32x2 register pairs for its whole life:
    // S2[nt][0] holds (row A, keys 2q and 2q+1) of 8-key tile nt and S2[nt][1]
    // the same pair for row B.  The rank-1 MX rescale below is this kernel's
    // biggest instruction consumer, and with the tile already paired it becomes
    // one mul.rn.f32x2 plus one fma.rn.f32x2 per pair instead of two FMULs and
    // two FFMAs.  Both packed forms round exactly like `*` and `fmaf`, so every
    // accumulated score is bit-identical to the scalar version.
    uint64_t S2[NT][2];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) S2[nt][0] = S2[nt][1] = 0ull;

#pragma unroll
    for (int kb = 0; kb < KB; ++kb) {
      unsigned a[4];
      {
        const int row = w * 16 + (lane & 15);
        const int chunk = 2 * kb + (lane >> 4);
        ldsm_x4(sQ_u + (uint32_t)fp8_tile_offset(row, chunk * 16, D), a[0], a[1], a[2], a[3]);
      }
      const float sqA = sSfQ[kb * BM + rowA];
      const float sqB = sSfQ[kb * BM + rowB];
      const uint64_t sqA2 = f32x2_pack(sqA, sqA);
      const uint64_t sqB2 = f32x2_pack(sqB, sqB);
#pragma unroll
      for (int ntp = 0; ntp < NT / 2; ++ntp) {
        unsigned bf[4];
        {
          const int keyIdx = ntp * 16 + (lane & 7) + ((lane >> 4) << 3);
          const int chunk = 2 * kb + ((lane >> 3) & 1);
          ldsm_x4(kstage + (uint32_t)fp8_tile_offset(keyIdx, chunk * 16, D), bf[0], bf[1], bf[2],
                  bf[3]);
        }
        float t0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float t1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma_fp8_16832(t0, a, bf);
        mma_fp8_16832(t1, a, bf + 2);
        const float2 sk0 = *reinterpret_cast<const float2*>(sSfK + kb * BN + ntp * 16 + 2 * q);
        const float2 sk1 =
            *reinterpret_cast<const float2*>(sSfK + kb * BN + ntp * 16 + 8 + 2 * q);
        const int n0 = 2 * ntp, n1 = 2 * ntp + 1;
        const uint64_t sk02 = f32x2_pack(sk0.x, sk0.y);
        const uint64_t sk12 = f32x2_pack(sk1.x, sk1.y);
        S2[n0][0] = f32x2_fma(f32x2_mul(f32x2_pack(t0[0], t0[1]), sqA2), sk02, S2[n0][0]);
        S2[n0][1] = f32x2_fma(f32x2_mul(f32x2_pack(t0[2], t0[3]), sqB2), sk02, S2[n0][1]);
        S2[n1][0] = f32x2_fma(f32x2_mul(f32x2_pack(t1[0], t1[1]), sqA2), sk12, S2[n1][0]);
        S2[n1][1] = f32x2_fma(f32x2_mul(f32x2_pack(t1[2], t1[3]), sqB2), sk12, S2[n1][1]);
      }
    }
    // Softmax scale as one packed multiply per row pair, in place.
    {
      const uint64_t scale2 = f32x2_pack(p.scale, p.scale);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        S2[nt][0] = f32x2_mul(S2[nt][0], scale2);
        S2[nt][1] = f32x2_mul(S2[nt][1], scale2);
      }
    }

    // ---- masking (only groups straddling a row's window edge) ------------
    const unsigned warpLo = __reduce_max_sync(0xffffffffu, laneLo);
    const unsigned warpHi = __reduce_min_sync(0xffffffffu, laneHi);
    if ((unsigned)keyBase < warpLo || (unsigned)(keyBase + 127) > warpHi) {
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        // k0 is a GLOBAL key index: lo/hi and keyBase are global too.
        const int k0 = keyBase + nt * 8 + 2 * q;
        float2 sA = f32x2_unpack(S2[nt][0]);
        float2 sB = f32x2_unpack(S2[nt][1]);
        if (k0 < loA || k0 > hiA) sA.x = -INFINITY;
        if (k0 + 1 < loA || k0 + 1 > hiA) sA.y = -INFINITY;
        if (k0 < loB || k0 > hiB) sB.x = -INFINITY;
        if (k0 + 1 < loB || k0 + 1 > hiB) sB.y = -INFINITY;
        S2[nt][0] = f32x2_pack(sA.x, sA.y);
        S2[nt][1] = f32x2_pack(sB.x, sB.y);
      }
    }

    // ---- group maximum ---------------------------------------------------
    float bmA = -INFINITY, bmB = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const float2 sA = f32x2_unpack(S2[nt][0]);
      const float2 sB = f32x2_unpack(S2[nt][1]);
      bmA = fmaxf(bmA, fmaxf(sA.x, sA.y));
      bmB = fmaxf(bmB, fmaxf(sB.x, sB.y));
    }
    bmA = fmaxf(bmA, __shfl_xor_sync(0xffffffffu, bmA, 1));
    bmA = fmaxf(bmA, __shfl_xor_sync(0xffffffffu, bmA, 2));
    bmB = fmaxf(bmB, __shfl_xor_sync(0xffffffffu, bmB, 1));
    bmB = fmaxf(bmB, __shfl_xor_sync(0xffffffffu, bmB, 2));

    if (PASS == PASS_GROUP_MAX) {
      if (q == 0) {
        if (validA) p.bm[(size_t)grp * p.rows_total + rowGA] = bmA;
        if (validB) p.bm[(size_t)grp * p.rows_total + rowGB] = bmB;
      }
      __syncthreads();
      continue;
    }
    (void)bmA;
    (void)bmB;

    // ---- delayed rescale -------------------------------------------------
    float safeA, safeB, alphaA, alphaB;
    if (PASS == PASS_SPLIT) {
      // Replay the materialised running maximum so this chunk reproduces the
      // single-pass schedule bit-for-bit.
      const float mtA = p.mstate[(size_t)grp * p.rows_total + rowGA];
      const float mtB = p.mstate[(size_t)grp * p.rows_total + rowGB];
      const float mpA = (grp > 0) ? p.mstate[(size_t)(grp - 1) * p.rows_total + rowGA] : -INFINITY;
      const float mpB = (grp > 0) ? p.mstate[(size_t)(grp - 1) * p.rows_total + rowGB] : -INFINITY;
      safeA = isfinite(mtA) ? mtA : 0.0f;
      safeB = isfinite(mtB) ? mtB : 0.0f;
      alphaA = isfinite(mpA) ? expf(mpA - safeA) : 0.0f;
      alphaB = isfinite(mpB) ? expf(mpB - safeB) : 0.0f;
      mA = mtA;
      mB = mtB;
    } else {
      const bool updA = (bmA - mA) > RESCALE_THRESHOLD;
      const bool updB = (bmB - mB) > RESCALE_THRESHOLD;
      const float mnA = updA ? bmA : mA;
      const float mnB = updB ? bmB : mB;
      safeA = isfinite(mnA) ? mnA : 0.0f;
      safeB = isfinite(mnB) ? mnB : 0.0f;
      alphaA = isfinite(mA) ? expf(mA - safeA) : 0.0f;
      alphaB = isfinite(mB) ? expf(mB - safeB) : 0.0f;
      mA = mnA;
      mB = mnB;
    }

    // alpha is exactly 1.0f whenever the maximum did not move, so the rescale
    // can be skipped bit-identically; that is the common case.
    if (TCG) {
      denA *= alphaA;
      denB *= alphaB;
    } else if (__any_sync(0xffffffffu, (alphaA != 1.0f) || (alphaB != 1.0f))) {
#pragma unroll
      for (int t = 0; t < NTOT; ++t) {
        accumulator[t][0] *= alphaA;
        accumulator[t][1] *= alphaA;
        accumulator[t][2] *= alphaB;
        accumulator[t][3] *= alphaB;
      }
      denA *= alphaA;
      denB *= alphaB;
    }

    // ---- probabilities, E4M3 QDQ, and PV ---------------------------------
    if (TCG && w < 4) {
      // V's sequence-axis E8M0 scales are SFB for this MMA: the sequence axis is
      // the PV reduction axis, in 32-key blocks, which is exactly scale_vec::1X.
      // The F8_128x4 atom already stores the four key blocks of one d as four
      // consecutive bytes, i.e. one TMEM word, and every subpartition needs it.
      const uint32_t srcV = sSfVraw_u + (uint32_t)(stage * L::SFV_BYTES);
#pragma unroll
      for (int g2 = 0; g2 < D / 128; ++g2) {
        const uint4 wv = lds128(srcV + (uint32_t)(g2 * 512 + lane * 16));
        tcg_st4(tlane + TC_SFB_V + 4 * g2, wv.x, wv.y, wv.z, wv.w);
      }
    }
    // Packed probability pipeline: the shift by -safe, the bias multiply and the
    // denominator accumulation each become one f32x2 op per row pair instead of
    // two scalar ones.  `expf` stays scalar and bit-identical - it is ~59% of
    // this loop and must match torch's `exp` exactly, because its value is what
    // the contract quantizes to E4M3.  Only the denominator's summation order
    // changes (two lanes instead of one, ~1e-7 relative against rtol 0.05).
    const uint64_t nsafeA2 = f32x2_pack(-safeA, -safeA);
    const uint64_t nsafeB2 = f32x2_pack(-safeB, -safeB);
    const uint64_t bias2 = f32x2_pack(p.bias, p.bias);
    uint64_t den2A = f32x2_pack(0.0f, 0.0f);
    uint64_t den2B = f32x2_pack(0.0f, 0.0f);
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      const float2 aA0 = f32x2_unpack(f32x2_add(S2[2 * kk][0], nsafeA2));
      const float2 aA1 = f32x2_unpack(f32x2_add(S2[2 * kk + 1][0], nsafeA2));
      const float2 aB0 = f32x2_unpack(f32x2_add(S2[2 * kk][1], nsafeB2));
      const float2 aB1 = f32x2_unpack(f32x2_add(S2[2 * kk + 1][1], nsafeB2));
      const uint64_t r0 = f32x2_mul(f32x2_pack(expf(aA0.x), expf(aA0.y)), bias2);
      const uint64_t r1 = f32x2_mul(f32x2_pack(expf(aA1.x), expf(aA1.y)), bias2);
      const uint64_t r2 = f32x2_mul(f32x2_pack(expf(aB0.x), expf(aB0.y)), bias2);
      const uint64_t r3 = f32x2_mul(f32x2_pack(expf(aB1.x), expf(aB1.y)), bias2);
      den2A = f32x2_add(f32x2_add(den2A, r0), r1);
      den2B = f32x2_add(f32x2_add(den2B, r2), r3);
      const float2 g0 = f32x2_unpack(r0);
      const float2 g1 = f32x2_unpack(r1);
      const float2 g2 = f32x2_unpack(r2);
      const float2 g3 = f32x2_unpack(r3);
      const float p0 = g0.x, p1 = g0.y, p4 = g1.x, p5 = g1.y;
      const float p2 = g2.x, p3 = g2.y, p6 = g3.x, p7 = g3.y;
      if (TCG) {
        // Scatter the quantized probabilities into the K-major E4M3 tile the PV
        // A descriptor reads: row r, 16-byte unit kk, byte 2q (and +8 for the
        // second 8-key half of the unit).
        const uint32_t offA =
            sP_u + (uint32_t)(rowA * 128 + (((kk ^ (rowA & 7)) << 4) | (2 * q)));
        const uint32_t offB =
            sP_u + (uint32_t)(rowB * 128 + (((kk ^ (rowB & 7)) << 4) | (2 * q)));
        sts16(offA, cvt_e4m3x2(p0, p1));
        sts16(offA + 8, cvt_e4m3x2(p4, p5));
        sts16(offB, cvt_e4m3x2(p2, p3));
        sts16(offB + 8, cvt_e4m3x2(p6, p7));
      } else {
        unsigned pfrag[4];
        pfrag[0] = e4m3x2_to_bf16x2(cvt_e4m3x2(p0, p1));
        pfrag[1] = e4m3x2_to_bf16x2(cvt_e4m3x2(p2, p3));
        pfrag[2] = e4m3x2_to_bf16x2(cvt_e4m3x2(p4, p5));
        pfrag[3] = e4m3x2_to_bf16x2(cvt_e4m3x2(p6, p7));
#pragma unroll
        for (int ntp = 0; ntp < NTP; ++ntp) {
          unsigned bf[4];
          {
            const int key = 16 * kk + (lane & 15);
            const int dcol = 16 * ntp + ((lane >> 4) << 3);
            ldsm_x4_trans(sV_u + 2u * (uint32_t)bf16_tile_offset(key, dcol, D), bf[0], bf[1],
                          bf[2], bf[3]);
          }
          mma_bf16_16816(accumulator[2 * ntp], pfrag, bf);
          mma_bf16_16816(accumulator[2 * ntp + 1], pfrag, bf + 2);
        }
      }
    }
    {
      const float2 dA = f32x2_unpack(den2A);
      const float2 dB = f32x2_unpack(den2B);
      denA += dA.x + dA.y;
      denB += dB.x + dB.y;
    }
    if (TCG) {
      sRow[rowA] = alphaA;
      sRow[rowB] = alphaB;
      // publish the probability tile to the async proxy that UMMA reads it with
      fence_async_shared();
      tcg_wait_st();
      tcg_fence_before();
      // one barrier publishes the probability tile and votes on the rescale
      const bool need = __syncthreads_or((alphaA != 1.0f) || (alphaB != 1.0f));
      if (need) {
        // num = num*alpha: the delayed-maximum rescale of the TMEM accumulator.
        // tcgen05.ld/st are .sync.aligned, i.e. warp-wide, so the decision has
        // to be a warp vote and the multiply has to run on every lane of a
        // participating warp; lanes whose alpha is exactly 1.0f multiply by 1.0,
        // which is bit-identical to skipping them.  Executing these inside a
        // per-lane divergent branch deadlocks the warp.
        const float aRow = sRow[(w & 3) * 32 + lane];
        if (__any_sync(0xffffffffu, aRow != 1.0f)) {
          const uint32_t half = (uint32_t)((w >> 2) * (D / 2));
#pragma unroll
          for (int c0 = 0; c0 < D / 2; c0 += 32) {
            float v[32];
            const uint32_t addr = tlane + TC_O + half + (uint32_t)c0;
            tcg_ld32(addr, v);
#pragma unroll
            for (int j = 0; j < 32; ++j) v[j] *= aRow;
            tcg_st32(addr, v);
          }
          tcg_wait_st();
        }
        tcg_fence_before();
        __syncthreads();
      }
      if (tid == 0) {
        tcg_fence_after();
        const uint32_t vstage = sVraw_u + (uint32_t)(stage * BN * D);
#pragma unroll
        for (int kb = 0; kb < BN / 32; ++kb) {
#pragma unroll
          for (int nc = 0; nc < D / 128; ++nc) {
            const uint64_t da = umma_desc(sP_u, 16, 1024) + 2 * kb;
            const uint64_t db =
                umma_desc(vstage + (uint32_t)(nc * (128 * BN)), 0, 1024) + (uint64_t)256 * kb;
            tcg_mma_bs(tmem + TC_O + (uint32_t)(nc * 128), da, db, tcg_idesc(BM, 128, 1, kb),
                       (uint32_t)(kb > 0 || grp != cLo), tmem + TC_SFA_P,
                       tmem + TC_SFB_V + nc * 4);
          }
        }
        tcg_commit(smem_addr(&mbar_pv));
      }
    } else {
      __syncthreads();
    }
  }

  // ---- epilogue ----------------------------------------------------------
  denA += __shfl_xor_sync(0xffffffffu, denA, 1);
  denA += __shfl_xor_sync(0xffffffffu, denA, 2);
  denB += __shfl_xor_sync(0xffffffffu, denB, 1);
  denB += __shfl_xor_sync(0xffffffffu, denB, 2);

  if (TCG) {
    // ---- tcgen05 epilogue: TMEM lane == row, so the accumulator is read out
    // row-wise and stored with wide coalesced writes instead of the 2-element
    // fragment scatter the register accumulator needed.
    if (cHi >= cLo) mbar_wait(&mbar_pv, pvWaitPhase);
    tcg_fence_after();
    float wA = 0.0f, wB = 0.0f;
    if (PASS == PASS_SPLIT) {
      if (any_compute) {
        const float tA = p.mtail[rowGA], tB = p.mtail[rowGB];
        wA = (isfinite(tA) && isfinite(mA)) ? expf(mA - tA) : 0.0f;
        wB = (isfinite(tB) && isfinite(mB)) ? expf(mB - tB) : 0.0f;
      }
    }
    sRow[BM + rowA] = denA;
    sRow[BM + rowB] = denB;
    sRow[2 * BM + rowA] = mA;
    sRow[2 * BM + rowB] = mB;
    sRow[3 * BM + rowA] = wA;
    sRow[3 * BM + rowB] = wB;
    __syncthreads();

    const int r = (w & 3) * 32 + lane;
    const int i2 = q0 + (r & (Qph - 1));
    const int h2 = kh * G + (r >> log2Qph);
    const bool valid2 = i2 < p.Sq;
    const int64_t rowG = (int64_t)((b * p.H + h2) * p.Sq) + (valid2 ? i2 : 0);
    const float den = sRow[BM + r];
    const float mrow = sRow[2 * BM + r];
    const float wgt = sRow[3 * BM + r];
    const bool live = den > 0.0f;
    const float inv = live ? (1.0f / den) : 0.0f;
    const int dOfs = (w >> 2) * (D / 2);
    __nv_bfloat16* oRow = p.o + b * p.o_b + (int64_t)h2 * p.o_h + (int64_t)i2 * p.o_s + dOfs;
    float* partRow =
        p.part_a + ((size_t)blockIdx.z * p.rows_total + rowG) * D + dOfs;
#pragma unroll
    for (int c = 0; c < D / 64; ++c) {
      float v[32];
      tcg_ld32(tlane + TC_O + (uint32_t)(dOfs + c * 32), v);
      if (valid2) {
        if (PASS == PASS_SPLIT) {
#pragma unroll
          for (int j = 0; j < 32; ++j) partRow[c * 32 + j] = v[j] * wgt;
        } else {
#pragma unroll
          for (int cc = 0; cc < 4; ++cc) {
            __nv_bfloat162 packed[4];
#pragma unroll
            for (int qq = 0; qq < 4; ++qq) {
              const float2 f = make_float2(live ? v[cc * 8 + 2 * qq] * inv : 0.0f,
                                         live ? v[cc * 8 + 2 * qq + 1] * inv : 0.0f);
              packed[qq] = __float22bfloat162_rn(f);
            }
            uint4 outv;
            outv.x = *reinterpret_cast<unsigned*>(&packed[0]);
            outv.y = *reinterpret_cast<unsigned*>(&packed[1]);
            outv.z = *reinterpret_cast<unsigned*>(&packed[2]);
            outv.w = *reinterpret_cast<unsigned*>(&packed[3]);
            stg128(oRow + c * 32 + cc * 8, outv);
          }
        }
      }
    }
    if ((w >> 2) == 0 && valid2) {
      if (p.has_lse) {
        float value = -INFINITY;
        if (live && isfinite(mrow)) value = mrow + logf(den / p.bias);
        p.lse[b * p.l_b + (int64_t)h2 * p.l_h + (int64_t)i2 * p.l_s] = value;
      }
      if (PASS == PASS_SPLIT) {
        p.part_b[(size_t)blockIdx.z * p.rows_total + rowG] = den * wgt;
      }
    }
    tcg_fence_before();
    __syncthreads();
    if (tid < 32) tcg_dealloc(tmem, TC_ALLOC);
    return;
  }

  if (PASS == PASS_SPLIT) {
    // Weight this chunk back to the global delayed-maximum reference so the
    // per-chunk partials sum to exactly the single-pass numerator/denominator.
    float wA = 0.0f, wB = 0.0f;
    if (any_compute) {
      const float tA = p.mtail[rowGA], tB = p.mtail[rowGB];
      wA = (isfinite(tA) && isfinite(mA)) ? expf(mA - tA) : 0.0f;
      wB = (isfinite(tB) && isfinite(mB)) ? expf(mB - tB) : 0.0f;
    }
    const int split = blockIdx.z;
    float* pa = p.part_a + ((size_t)split * p.rows_total + rowGA) * D;
    float* pb = p.part_b + (size_t)split * p.rows_total + rowGA;
    float* paB = p.part_a + ((size_t)split * p.rows_total + rowGB) * D;
    float* pbB = p.part_b + (size_t)split * p.rows_total + rowGB;
    if (validA) {
#pragma unroll
      for (int t = 0; t < NTOT; ++t) {
        pa[t * 8 + 2 * q] = accumulator[t][0] * wA;
        pa[t * 8 + 2 * q + 1] = accumulator[t][1] * wA;
      }
      if (q == 0) *pb = denA * wA;
    }
    if (validB) {
#pragma unroll
      for (int t = 0; t < NTOT; ++t) {
        paB[t * 8 + 2 * q] = accumulator[t][2] * wB;
        paB[t * 8 + 2 * q + 1] = accumulator[t][3] * wB;
      }
      if (q == 0) *pbB = denB * wB;
    }
    return;
  }

  const bool liveA = denA > 0.0f;
  const bool liveB = denB > 0.0f;
  // One reciprocal per row: a 1-2 ULP fp32 difference from true division cannot
  // survive BF16 rounding as more than one BF16 ULP (0.39%), far inside rtol .05.
  const float invA = liveA ? (1.0f / denA) : 0.0f;
  const float invB = liveB ? (1.0f / denB) : 0.0f;

  __nv_bfloat16* oA = p.o + b * p.o_b + (int64_t)hA * p.o_h + (int64_t)iA * p.o_s + 2 * q;
  __nv_bfloat16* oB = p.o + b * p.o_b + (int64_t)hB * p.o_h + (int64_t)iB * p.o_s + 2 * q;
#pragma unroll
  for (int t = 0; t < NTOT; ++t) {
    const int dstep = t * 8;
    if (validA) {
      const __nv_bfloat162 out = __float22bfloat162_rn(
          make_float2(liveA ? accumulator[t][0] * invA : 0.0f,
                      liveA ? accumulator[t][1] * invA : 0.0f));
      *reinterpret_cast<__nv_bfloat162*>(oA + dstep) = out;
    }
    if (validB) {
      const __nv_bfloat162 out = __float22bfloat162_rn(
          make_float2(liveB ? accumulator[t][2] * invB : 0.0f,
                      liveB ? accumulator[t][3] * invB : 0.0f));
      *reinterpret_cast<__nv_bfloat162*>(oB + dstep) = out;
    }
  }
  if (p.has_lse && q == 0) {
    if (validA) {
      float value = -INFINITY;
      if (liveA && isfinite(mA)) value = mA + logf(denA / p.bias);
      p.lse[b * p.l_b + (int64_t)hA * p.l_h + (int64_t)iA * p.l_s] = value;
    }
    if (validB) {
      float value = -INFINITY;
      if (liveB && isfinite(mB)) value = mB + logf(denB / p.bias);
      p.lse[b * p.l_b + (int64_t)hB * p.l_h + (int64_t)iB * p.l_s] = value;
    }
  }
}

// The single-pass scan, kept for short key ranges: at T < FMHA_SCAN_TWO_PASS_T the
// segment design's extra launch costs more than the loads it deletes (`agw4`: s3 at
// T=64 read +8.92% and d3 at T=32 +7.09%, against s2 at T=128 -1.80%/-4.03% and s5
// at T=513 -6.32%/-6.93%).  Everything below this line is that kernel, unchanged
// from the promoted `best/ao13_full`, with its macros renamed so the two files of
// scan code cannot collide.
// Replays reference.py's delayed-maximum recurrence over the materialised group
// maxima: maximum starts at -inf and moves to a group maximum only when that
// maximum exceeds it by more than 4*ln(2).  NaN comparisons are false, so an
// all-masked prefix stays at -inf exactly as torch's `>` does.
// How many groups a row must walk before the scan's loads are batched 64 deep
// instead of 16.  See the `if (T >= FMHA_SCAN_DEEP_T)` below for the two
// measurements that bracket it: 64-deep is -4.4%/-5.1% at T=512/513 and
// +1.3%/+0.9% at T=128/64.
#ifndef FMHA_SCAN_DEEP_T
#define FMHA_SCAN_DEEP_T 256
#endif

// EXPERIMENT agv7.  `agv6` batched this kernel's loads sixty-four deep and measured
// -2.2% on s5, -3.7% on d7, -3.0% on s3 and -1.1% on s2, flat on the many-row
// shapes - but the register census still reported **32 registers**, which 64 live
// floats cannot fit in, so ptxas sank the loads back towards their uses and the
// batch never actually went 64 deep.  Without `__launch_bounds__` ptxas picks a
// register target that keeps many blocks per SM resident; this kernel is launched as
// `ceil(rows/256)` blocks of 256, and for exactly the shapes where the batch matters
// that is ONE block, so occupancy is not a resource it needs to protect there.  Two
// blocks per SM caps the allocation at 128 registers, which holds all 64 loads plus
// the recurrence, while still leaving 16-24 warps per SM for the many-row launches
// that do fill the machine.
__global__ void __launch_bounds__(256, 2) fmha_scan_flat_kernel(
    const float* __restrict__ bm, float* __restrict__ mstate, float* __restrict__ mtail, int rows,
    int T) {
  fmha_pdl_wait();
  // Primary of the fourth boundary (scan -> split).  Everything below is the serial
  // recurrence over T that §30 proved not composable, so it is pure latency that split's
  // TMEM alloc, Q staging and Q scale writes can be made to overlap.  Triggered after the
  // wait because that is the point where this kernel has taken delivery of segmax's output
  // and has nothing left to do but compute; triggering before it would also be safe -- the
  // scan is one CTA and is already resident before split can be triggered at all -- but it
  // would release split early enough to queue behind a segmax that has not retired.
  fmha_pdl_scan_trigger();
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  // bm and mstate are stored TRANSPOSED, [group][row], so the 32 lanes of a warp
  // read and write 32 consecutive floats.  With the row-major layout every
  // thread's load touched its own cache line, which is why this kernel ran at
  // 0.08-0.20 instructions per cycle with 69-84% long_scoreboard stalls and was
  // 58.6% of domain d7 (job pH19), 29.9% of frozen shape 3 and 11.0% of shape 9
  // (job pH39) even after its loads were batched sixteen deep.
  const size_t stride = (size_t)rows;
  const float* src = bm + row;
  float* dst = mstate + row;
  float m = -INFINITY;
#define FMHA_FLAT_STEP64(index)                               \
  do {                                                        \
    if ((c##index - m) > RESCALE_THRESHOLD) m = c##index;      \
    dst[(size_t)(t + (index)) * stride] = m;                  \
  } while (0)
#define FMHA_FLAT_STEP(index)                                 \
  do {                                                        \
    if ((b##index - m) > RESCALE_THRESHOLD) m = b##index;      \
    dst[(size_t)(t + (index)) * stride] = m;                  \
  } while (0)
  int t = 0;
  // Sixty-four independent loads in flight before the dependent recurrence, then a
  // sixteen-deep tier, then a scalar tail.  `agv6`/`agv7`/`agv8` bracketed this:
  // 64 deep is worth -4.4%/-5.1% at T=512/513 and costs +1.3%/+0.9% at T=128/64,
  // hence the gate, and `__launch_bounds__(256,2)` is what lets the batch actually
  // stay live (without it ptxas reported 32 registers and sank the loads).
  for (; t + 64 <= T; t += 64) {
    const float c0 = __ldg(src + (size_t)(t + 0) * stride);
    const float c1 = __ldg(src + (size_t)(t + 1) * stride);
    const float c2 = __ldg(src + (size_t)(t + 2) * stride);
    const float c3 = __ldg(src + (size_t)(t + 3) * stride);
    const float c4 = __ldg(src + (size_t)(t + 4) * stride);
    const float c5 = __ldg(src + (size_t)(t + 5) * stride);
    const float c6 = __ldg(src + (size_t)(t + 6) * stride);
    const float c7 = __ldg(src + (size_t)(t + 7) * stride);
    const float c8 = __ldg(src + (size_t)(t + 8) * stride);
    const float c9 = __ldg(src + (size_t)(t + 9) * stride);
    const float c10 = __ldg(src + (size_t)(t + 10) * stride);
    const float c11 = __ldg(src + (size_t)(t + 11) * stride);
    const float c12 = __ldg(src + (size_t)(t + 12) * stride);
    const float c13 = __ldg(src + (size_t)(t + 13) * stride);
    const float c14 = __ldg(src + (size_t)(t + 14) * stride);
    const float c15 = __ldg(src + (size_t)(t + 15) * stride);
    const float c16 = __ldg(src + (size_t)(t + 16) * stride);
    const float c17 = __ldg(src + (size_t)(t + 17) * stride);
    const float c18 = __ldg(src + (size_t)(t + 18) * stride);
    const float c19 = __ldg(src + (size_t)(t + 19) * stride);
    const float c20 = __ldg(src + (size_t)(t + 20) * stride);
    const float c21 = __ldg(src + (size_t)(t + 21) * stride);
    const float c22 = __ldg(src + (size_t)(t + 22) * stride);
    const float c23 = __ldg(src + (size_t)(t + 23) * stride);
    const float c24 = __ldg(src + (size_t)(t + 24) * stride);
    const float c25 = __ldg(src + (size_t)(t + 25) * stride);
    const float c26 = __ldg(src + (size_t)(t + 26) * stride);
    const float c27 = __ldg(src + (size_t)(t + 27) * stride);
    const float c28 = __ldg(src + (size_t)(t + 28) * stride);
    const float c29 = __ldg(src + (size_t)(t + 29) * stride);
    const float c30 = __ldg(src + (size_t)(t + 30) * stride);
    const float c31 = __ldg(src + (size_t)(t + 31) * stride);
    const float c32 = __ldg(src + (size_t)(t + 32) * stride);
    const float c33 = __ldg(src + (size_t)(t + 33) * stride);
    const float c34 = __ldg(src + (size_t)(t + 34) * stride);
    const float c35 = __ldg(src + (size_t)(t + 35) * stride);
    const float c36 = __ldg(src + (size_t)(t + 36) * stride);
    const float c37 = __ldg(src + (size_t)(t + 37) * stride);
    const float c38 = __ldg(src + (size_t)(t + 38) * stride);
    const float c39 = __ldg(src + (size_t)(t + 39) * stride);
    const float c40 = __ldg(src + (size_t)(t + 40) * stride);
    const float c41 = __ldg(src + (size_t)(t + 41) * stride);
    const float c42 = __ldg(src + (size_t)(t + 42) * stride);
    const float c43 = __ldg(src + (size_t)(t + 43) * stride);
    const float c44 = __ldg(src + (size_t)(t + 44) * stride);
    const float c45 = __ldg(src + (size_t)(t + 45) * stride);
    const float c46 = __ldg(src + (size_t)(t + 46) * stride);
    const float c47 = __ldg(src + (size_t)(t + 47) * stride);
    const float c48 = __ldg(src + (size_t)(t + 48) * stride);
    const float c49 = __ldg(src + (size_t)(t + 49) * stride);
    const float c50 = __ldg(src + (size_t)(t + 50) * stride);
    const float c51 = __ldg(src + (size_t)(t + 51) * stride);
    const float c52 = __ldg(src + (size_t)(t + 52) * stride);
    const float c53 = __ldg(src + (size_t)(t + 53) * stride);
    const float c54 = __ldg(src + (size_t)(t + 54) * stride);
    const float c55 = __ldg(src + (size_t)(t + 55) * stride);
    const float c56 = __ldg(src + (size_t)(t + 56) * stride);
    const float c57 = __ldg(src + (size_t)(t + 57) * stride);
    const float c58 = __ldg(src + (size_t)(t + 58) * stride);
    const float c59 = __ldg(src + (size_t)(t + 59) * stride);
    const float c60 = __ldg(src + (size_t)(t + 60) * stride);
    const float c61 = __ldg(src + (size_t)(t + 61) * stride);
    const float c62 = __ldg(src + (size_t)(t + 62) * stride);
    const float c63 = __ldg(src + (size_t)(t + 63) * stride);
    FMHA_FLAT_STEP64(0);
    FMHA_FLAT_STEP64(1);
    FMHA_FLAT_STEP64(2);
    FMHA_FLAT_STEP64(3);
    FMHA_FLAT_STEP64(4);
    FMHA_FLAT_STEP64(5);
    FMHA_FLAT_STEP64(6);
    FMHA_FLAT_STEP64(7);
    FMHA_FLAT_STEP64(8);
    FMHA_FLAT_STEP64(9);
    FMHA_FLAT_STEP64(10);
    FMHA_FLAT_STEP64(11);
    FMHA_FLAT_STEP64(12);
    FMHA_FLAT_STEP64(13);
    FMHA_FLAT_STEP64(14);
    FMHA_FLAT_STEP64(15);
    FMHA_FLAT_STEP64(16);
    FMHA_FLAT_STEP64(17);
    FMHA_FLAT_STEP64(18);
    FMHA_FLAT_STEP64(19);
    FMHA_FLAT_STEP64(20);
    FMHA_FLAT_STEP64(21);
    FMHA_FLAT_STEP64(22);
    FMHA_FLAT_STEP64(23);
    FMHA_FLAT_STEP64(24);
    FMHA_FLAT_STEP64(25);
    FMHA_FLAT_STEP64(26);
    FMHA_FLAT_STEP64(27);
    FMHA_FLAT_STEP64(28);
    FMHA_FLAT_STEP64(29);
    FMHA_FLAT_STEP64(30);
    FMHA_FLAT_STEP64(31);
    FMHA_FLAT_STEP64(32);
    FMHA_FLAT_STEP64(33);
    FMHA_FLAT_STEP64(34);
    FMHA_FLAT_STEP64(35);
    FMHA_FLAT_STEP64(36);
    FMHA_FLAT_STEP64(37);
    FMHA_FLAT_STEP64(38);
    FMHA_FLAT_STEP64(39);
    FMHA_FLAT_STEP64(40);
    FMHA_FLAT_STEP64(41);
    FMHA_FLAT_STEP64(42);
    FMHA_FLAT_STEP64(43);
    FMHA_FLAT_STEP64(44);
    FMHA_FLAT_STEP64(45);
    FMHA_FLAT_STEP64(46);
    FMHA_FLAT_STEP64(47);
    FMHA_FLAT_STEP64(48);
    FMHA_FLAT_STEP64(49);
    FMHA_FLAT_STEP64(50);
    FMHA_FLAT_STEP64(51);
    FMHA_FLAT_STEP64(52);
    FMHA_FLAT_STEP64(53);
    FMHA_FLAT_STEP64(54);
    FMHA_FLAT_STEP64(55);
    FMHA_FLAT_STEP64(56);
    FMHA_FLAT_STEP64(57);
    FMHA_FLAT_STEP64(58);
    FMHA_FLAT_STEP64(59);
    FMHA_FLAT_STEP64(60);
    FMHA_FLAT_STEP64(61);
    FMHA_FLAT_STEP64(62);
    FMHA_FLAT_STEP64(63);
  }
  for (; t + 16 <= T; t += 16) {
    // Sixteen independent loads in named registers ahead of the dependent
    // recurrence.  An array here goes to local memory and measured slower
    // (job pH18), so the batch is spelled out.
    const float b0 = __ldg(src + (size_t)(t + 0) * stride);
    const float b1 = __ldg(src + (size_t)(t + 1) * stride);
    const float b2 = __ldg(src + (size_t)(t + 2) * stride);
    const float b3 = __ldg(src + (size_t)(t + 3) * stride);
    const float b4 = __ldg(src + (size_t)(t + 4) * stride);
    const float b5 = __ldg(src + (size_t)(t + 5) * stride);
    const float b6 = __ldg(src + (size_t)(t + 6) * stride);
    const float b7 = __ldg(src + (size_t)(t + 7) * stride);
    const float b8 = __ldg(src + (size_t)(t + 8) * stride);
    const float b9 = __ldg(src + (size_t)(t + 9) * stride);
    const float b10 = __ldg(src + (size_t)(t + 10) * stride);
    const float b11 = __ldg(src + (size_t)(t + 11) * stride);
    const float b12 = __ldg(src + (size_t)(t + 12) * stride);
    const float b13 = __ldg(src + (size_t)(t + 13) * stride);
    const float b14 = __ldg(src + (size_t)(t + 14) * stride);
    const float b15 = __ldg(src + (size_t)(t + 15) * stride);
    FMHA_FLAT_STEP(0);
    FMHA_FLAT_STEP(1);
    FMHA_FLAT_STEP(2);
    FMHA_FLAT_STEP(3);
    FMHA_FLAT_STEP(4);
    FMHA_FLAT_STEP(5);
    FMHA_FLAT_STEP(6);
    FMHA_FLAT_STEP(7);
    FMHA_FLAT_STEP(8);
    FMHA_FLAT_STEP(9);
    FMHA_FLAT_STEP(10);
    FMHA_FLAT_STEP(11);
    FMHA_FLAT_STEP(12);
    FMHA_FLAT_STEP(13);
    FMHA_FLAT_STEP(14);
    FMHA_FLAT_STEP(15);
  }
  for (; t < T; ++t) {
    const float b = __ldg(src + (size_t)t * stride);
    if ((b - m) > RESCALE_THRESHOLD) m = b;
    dst[(size_t)t * stride] = m;
  }
#undef FMHA_FLAT_STEP64
#undef FMHA_FLAT_STEP
  mtail[row] = m;
}

// Replays reference.py's delayed-maximum recurrence over the materialised group
// maxima: the maximum starts at -inf and moves to a group maximum only when that
// maximum exceeds it by more than 4*ln(2).  NaN comparisons are false, so an
// all-masked prefix stays at -inf exactly as torch's `>` does.
//
// TWO PASSES, because the recurrence is serial in the group index while its
// *inputs* are not.  `f_b(m) = (b - m > 4 ln 2) ? b : m` never returns less than
// `m`, so `m` is non-decreasing; therefore a segment whose `max(b) - m_in` is not
// greater than `4 ln 2` contains **no jump at all** - inside it `b_j <= segmax` and
// `m_{j-1} >= m_in`, and subtraction is monotone in both operands, so
// `fl(b_j - m_{j-1}) <= fl(segmax - m_in)` and every test in it is false - which
// means every `mstate` entry in that segment is the constant `m_in` and can be
// *stored without being read*.  Pass A computes the segment maxima fully parallel
// (`rows * nseg` threads over a 2-D grid); pass B stays one thread per row, loads
// its row's maxima in batched rounds, and then either streams `FMHA_SCAN_SEG`
// constants or walks the rare segment that did jump, bit-identically.
//
// Why this and not a deeper batch: `agv6`/`agv7`/`agv8` cut d7's scan from 32
// dependent rounds to 8 and bought -5.42% of the forward, i.e. only ~15% of the
// kernel, because the round *count* was never the limit.  `ap9b` shows it moving
// 401 KB in 27.3 us - 15 GB/s against `bench4`'s 6223 GB/s - with `long_scoreboard`
// 67-71% on ONE SM of 148 (`grid = ceil(rows/256)`, and `rows_total` is 96 for s5
// and 192 for d7), so what binds is how many 128-byte requests one SM can have in
// flight.  Pass A spreads the same requests over every SM the launch has, and pass
// B deletes ~15/16 of them.  Worst case - a segment maximum that jumps every
// segment, which no `shape_range` distribution produces and which the walk handles
// exactly - pass B degrades to `ceil(T/16)` batched rounds, which is precisely what
// it does today, so this is never worse than the kernel it replaces.
//
// The three conditions under which the two-pass scan is launched, each a measured
// bracket with a mechanism behind it rather than a shape list:
//
// * `T >= FMHA_SCAN_TWO_PASS_T` (128).  Pass A is a second launch, worth a couple of
//   microseconds of measured window, and only a long scan earns it back: `agw4` put
//   T=513 at -6.32%/-6.93% (s5) and T=128 at -1.80%/-4.03% (s2), against T=64 at
//   +8.92% (s3) and T=32 at +7.09% (d3).
// * `rows_total <= FMHA_SCAN_TWO_PASS_ROWS` (512).  The win is that pass B issues
//   ~16x fewer 128-byte requests, and that only matters when the flat scan was
//   request-starved on one or two SMs - `grid = ceil(rows/256)`.  s9 has 8200 rows,
//   so its flat scan already spreads over 33 blocks and the two-pass form read
//   +0.14%/+0.36% across `agw4`/`agw5`, i.e. nothing but the extra launch.
// * no left window (`FMHA_SCAN_TWO_PASS_NOWIN`).  A window makes `lo` grow with the
//   row index, so most rows carry a long prefix of entirely masked groups whose `bm`
//   is -INFINITY, and `m` stays -inf across it.  Testing all sixteen maxima of a
//   chunk against the incoming `m` and taking the LOWEST set bit did shrink the walk
//   from the whole 256-group chunk to the 32 groups from the first candidate segment
//   onward, and it lifted s5 from -5.25% to -5.91% and s2 from +0.21% to -2.15% -
//   but d7 still read **+15.14%** (`agw7b`), after **+20.91%** for `agw4`'s
//   whole-chunk walk and **+15.07%** for `agw5`'s per-segment re-test.  Four
//   dispatch designs all land in the same place on the one shape with a masked
//   prefix, so that prefix costs something this analysis has not located, and the
//   honest move is to gate the case out rather than keep guessing: windowed launches
//   take the promoted flat scan byte-identically, which `agw6` measured as airtight
//   (gated-off shapes +0.41% against a +0.32% control bias).  Finding what d7's
//   prefix actually costs is its own experiment, not a reason to hold back s5 and s2.
#ifndef FMHA_SCAN_TWO_PASS_T
#define FMHA_SCAN_TWO_PASS_T 128
#endif
#ifndef FMHA_SCAN_TWO_PASS_ROWS
#define FMHA_SCAN_TWO_PASS_ROWS 512
#endif
#ifndef FMHA_SCAN_TWO_PASS_NOWIN
#define FMHA_SCAN_TWO_PASS_NOWIN 1
#endif

// ---- fused flat scan (fs5t/fs7t port of the a-side fs-series) -------------
// 1 fuses fmha_scan_flat (flat tier) or the whole segmax+scan chain
// (two-pass tier) INTO the split pass: the split kernel (FSCAN template
// axis) derives the delayed-maximum states via a per-thread prefix walk over
// p.bm - bit-identical to the scan kernels' recurrence - stores its partials
// UNWEIGHTED plus each chunk's end state in the new part_m buffer, and its
// last chunk republishes mtail.  The combine's FWGT instantiations form
// w_s = guarded exp(m_s - mtail) <= 1 ONCE per split per CTA into shared sW
// and weight both sums in the shipped ascending order; the LSE keeps
// mt + log(total_b/bias).  a-side evidence chain (results/summaries/
// 2026-10-22-fs3-combine-normalisation-measured.md and -fs6-two-pass-
// fusion-measured.md): fs1 in-kernel mtail read RACED (dv_70e074a7276e);
// fs2' unnormalised e^{mA} OVERFLOWED the declared peaked generator
// (dv_bc379c3737ce); fs3 per-thread weights flooded the combine (+9%,
// dv_c8141de5e812); fs4/fs5 shared-sW form won s3 -6.99%/-6.91%
// bit-identical, engine PASS x2 (dv_c559dddd10d8); fs6/fs7 two-pass tier
// won s2 -8.16%/-7.77% bit-identical, with flat span-129 fusion (s9)
// excluded at +1.81% (dv_2f42aba011ed, dv_9f7feb79b0ec).  D=256 stays
// gated out: the a-side K=2 FWGT combine exposure netted only -0.8% on d3.
// PDL interaction (this tree): the fused split still calls
// fmha_pdl_scan_wait before the walk; its predecessor is then the GM pass,
// whose EXIT trigger plus the wait's completion-and-visibility semantics
// cover the walk's p.bm reads, and the combine's entry wait covers the
// split's part_m/mtail writes.  Defaults 0/48/48/0 = the shipped tree
// until this tree's own A/B promotes the tiers.
// SHIPPED on this tree (fs7t, default 1 with the two-pass tier at 160):
// same-job 3-arm A/B dv_8c718c726a23 - s2 (two-pass tier) -3.46%/-2.65%
// min/mean, s3 (flat tier) -1.55%/-1.55% in the fs7t arm and +0.18/+0.45 in
// the fs5t arm (identical device path; the spread is cross-arm allocation
// noise, pooled read ~-0.6%), engine PASS x2 seeds x3 arms on s2/s3 (max_rel
// digit-identical to base), every plan-identical control inside band
// (s9/s5/s7/s0/d1/d9/d3/d7/d0), census STACK:0 with FWGT combines at their
// FP32 twins' REG.  The wins are ~1/3 of the a-side originals (s2 -8.16,
// s3 -6.99) because this tree's PDL launches already recover the chain's
// gaps; fusion still removes the scan kernels' WORK, which PDL cannot.
#ifndef FMHA_FUSE_SCAN_FLAT
#define FMHA_FUSE_SCAN_FLAT 1
#endif
#ifndef FMHA_FUSE_SCAN_MAX_SPAN
#define FMHA_FUSE_SCAN_MAX_SPAN 48
#endif
#ifndef FMHA_FUSE_SCAN2P_MAX_SPAN
#define FMHA_FUSE_SCAN2P_MAX_SPAN 160
#endif
#ifndef FMHA_FUSE_SCAN_D256
#define FMHA_FUSE_SCAN_D256 0
#endif

// Groups per segment, and segments whose maxima pass B loads per batched round.
// Both are 16 so that a walked segment is exactly one `fmha_scan_batch16` and the
// maxima of a whole chunk fit in named registers; `nseg = ceil(T/16) <= 128` over
// the entire domain (`kv_len <= 262144`), so pass B needs at most 8 chunks.
#ifndef FMHA_SCAN_SEG
#define FMHA_SCAN_SEG 16
#endif
#ifndef FMHA_SCAN_SEGCHUNK
#define FMHA_SCAN_SEGCHUNK 16
#endif

// EXPERIMENT sn2t: 1 splits the two-pass scan into B1 (one thread per row:
// the serial masked-skip walk writing ONLY the segment-boundary states
// mstate[s*FMHA_SCAN_SEG-1] and mtail) + B2/segfill (one thread per
// (row, segment): the exact element fill from B1's boundary, fully
// parallel).  a-side evidence (sn1/sn2, promoted there): s5 -1.37%/-3.37%
// across two allocations, bit-identical (max is exact and order-free; the
// boundary states carry the serial dependence, the fill is the shipped walk
// per segment); s2 +2.45% when ungated - hence FMHA_SCAN_B1B2_MINSEG
// (nseg >= 16), which keeps small-T shapes on the single serial kernel.
// On THIS tree the win mechanism is unchanged by PDL: the serial
// scan_kernel body sits between segmax's trigger and the split's
// fmha_pdl_scan_wait, and PDL cannot shorten a serial body - B1B2
// parallelises it.  Fused shapes (FMHA_FUSE_SCAN_FLAT) skip the whole
// chain and never reach this branch.  Default 0 until this tree's A/B.
//
// THE A/B RAN: NULL — stays 0 (sn2t, dv_e73de1ed972f, 4 rounds,
// bit-identity exact on all five cases, checker PASS both seeds;
// results/summaries/2026-10-25-sn2t-scan-b1b2-prereg.md).  Treated s5
// mode-resolved +0.07% = zero effect: on this tree the scan chain no
// longer binds - the peer's wins cut s5's GM by 21.7% and gmc2's
// negative s5 residual shows PDL overlap already absorbing the chain,
// so parallelizing a hidden body buys nothing while the extra launch
// costs it back.  The old -3.37% was real on the OLD tree; per-tree
// re-measurement is mandatory for inherited latency-exposure claims
// (schedule-balance claims like lpt2's transfer robustly - see the
// FMHA_LPT_PAIR_MODE site).  Kept at 0 with this evidence.
#ifndef FMHA_SCAN_B1B2
#define FMHA_SCAN_B1B2 0
#endif
#ifndef FMHA_SCAN_B1B2_MINSEG
#define FMHA_SCAN_B1B2_MINSEG 16
#endif

#define FMHA_SCAN_STEP(index)                                 \
  do {                                                        \
    if ((b##index - m) > RESCALE_THRESHOLD) m = b##index;      \
    dst[(size_t)(t + (index)) * stride] = m;                  \
  } while (0)

#define FMHA_SCAN_STEP_NS(index)                              \
  do {                                                        \
    if ((b##index - m) > RESCALE_THRESHOLD) m = b##index;     \
  } while (0)


// One sixteen-group batch of the recurrence: sixteen independent loads in named
// registers ahead of the dependent chain, then the sixteen stores.  An array here
// goes to local memory and measured slower (job pH18), so the batch is spelled
// out.  Requires `t + FMHA_SCAN_SEG <= T` and `FMHA_SCAN_SEG == 16`.
// STORE=false is the state-only variant the B1 boundary pass walks with: same
// loads, same recurrence, no writes - the stores are what B2 parallelises.
template <bool STORE>
__device__ __forceinline__ void fmha_scan_batch16(const float* __restrict__ src,
                                                  float* __restrict__ dst, int t, size_t stride,
                                                  float& m) {
  const float b0 = __ldg(src + (size_t)(t + 0) * stride);
  const float b1 = __ldg(src + (size_t)(t + 1) * stride);
  const float b2 = __ldg(src + (size_t)(t + 2) * stride);
  const float b3 = __ldg(src + (size_t)(t + 3) * stride);
  const float b4 = __ldg(src + (size_t)(t + 4) * stride);
  const float b5 = __ldg(src + (size_t)(t + 5) * stride);
  const float b6 = __ldg(src + (size_t)(t + 6) * stride);
  const float b7 = __ldg(src + (size_t)(t + 7) * stride);
  const float b8 = __ldg(src + (size_t)(t + 8) * stride);
  const float b9 = __ldg(src + (size_t)(t + 9) * stride);
  const float b10 = __ldg(src + (size_t)(t + 10) * stride);
  const float b11 = __ldg(src + (size_t)(t + 11) * stride);
  const float b12 = __ldg(src + (size_t)(t + 12) * stride);
  const float b13 = __ldg(src + (size_t)(t + 13) * stride);
  const float b14 = __ldg(src + (size_t)(t + 14) * stride);
  const float b15 = __ldg(src + (size_t)(t + 15) * stride);
  if (STORE) {
    FMHA_SCAN_STEP(0);
    FMHA_SCAN_STEP(1);
    FMHA_SCAN_STEP(2);
    FMHA_SCAN_STEP(3);
    FMHA_SCAN_STEP(4);
    FMHA_SCAN_STEP(5);
    FMHA_SCAN_STEP(6);
    FMHA_SCAN_STEP(7);
    FMHA_SCAN_STEP(8);
    FMHA_SCAN_STEP(9);
    FMHA_SCAN_STEP(10);
    FMHA_SCAN_STEP(11);
    FMHA_SCAN_STEP(12);
    FMHA_SCAN_STEP(13);
    FMHA_SCAN_STEP(14);
    FMHA_SCAN_STEP(15);
  } else {
    FMHA_SCAN_STEP_NS(0);
    FMHA_SCAN_STEP_NS(1);
    FMHA_SCAN_STEP_NS(2);
    FMHA_SCAN_STEP_NS(3);
    FMHA_SCAN_STEP_NS(4);
    FMHA_SCAN_STEP_NS(5);
    FMHA_SCAN_STEP_NS(6);
    FMHA_SCAN_STEP_NS(7);
    FMHA_SCAN_STEP_NS(8);
    FMHA_SCAN_STEP_NS(9);
    FMHA_SCAN_STEP_NS(10);
    FMHA_SCAN_STEP_NS(11);
    FMHA_SCAN_STEP_NS(12);
    FMHA_SCAN_STEP_NS(13);
    FMHA_SCAN_STEP_NS(14);
    FMHA_SCAN_STEP_NS(15);
  }
}


// Walk the group range [t0,t1) exactly - the serial recurrence, sixteen groups per
// batched round.  Out of line so that there is ONE copy of the batch however many
// places call it; `ap10` measured what inlining it into sixteen dispatch sites does
// to a kernel that runs on a single SM (`no_instruction` 61-72%).
template <bool STORE>
__device__ __noinline__ void fmha_scan_walk(const float* __restrict__ src,
                                            float* __restrict__ dst, int t0, int t1, size_t stride,
                                            float& m) {
  int t = t0;
  for (; t + 16 <= t1; t += 16) fmha_scan_batch16<STORE>(src, dst, t, stride, m);
  for (; t < t1; ++t) {
    const float b = __ldg(src + (size_t)t * stride);
    if ((b - m) > RESCALE_THRESHOLD) m = b;
    if (STORE) dst[(size_t)t * stride] = m;
  }
}

// Pass A: one thread per (row, segment), the exact maximum of that segment's group
// maxima.  `bm` is stored transposed, [group][row], so the 32 lanes of a warp read
// 32 consecutive floats, and the 2-D grid puts `rows` on x and the segment index on
// y - which is what turns a serial 27 us kernel on ONE SM into a fully parallel one:
// for d7 that is 32 blocks instead of 1, and `ap9b` showed the flat scan was bound by
// how many 128-byte requests a single SM can have in flight, not by how it batches
// them.  Sixteen independent loads, then a reduction that costs nothing because
// `fmaxf` is exact and order independent, so the value is the segment maximum bit
// for bit.
__global__ void __launch_bounds__(256, 4) fmha_segmax_kernel(const float* __restrict__ bm,
                                                            float* __restrict__ segmax, int rows,
                                                            int T) {
  // Trigger at the ENTRY here, where fmha_tcg_kernel must not: segmax is 8-33 CTAs and its
  // successor is the scan's ONE CTA, so an early launch steals a slot from nothing.  This is
  // the one boundary where entry placement is strictly better, and it cannot let the
  // successor's wait return early -- that still waits for every segmax thread to exit.
  fmha_pdl_trigger();
  fmha_pdl_wait();
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const size_t stride = (size_t)rows;
  const int g0 = (int)blockIdx.y * FMHA_SCAN_SEG;
  const int g1 = (g0 + FMHA_SCAN_SEG < T) ? (g0 + FMHA_SCAN_SEG) : T;
  const float* src = bm + (size_t)g0 * stride + row;
  float mx = -INFINITY;
#define FMHA_SEGMAX_MAX(i) mx = fmaxf(mx, v##i);
#define FMHA_SEGMAX_LD(i) const float v##i = (g0 + (i) < g1) ? __ldg(src + (size_t)(i) * stride) : -INFINITY;
  FMHA_SEGMAX_LD(0) FMHA_SEGMAX_LD(1) FMHA_SEGMAX_LD(2) FMHA_SEGMAX_LD(3)
  FMHA_SEGMAX_LD(4) FMHA_SEGMAX_LD(5) FMHA_SEGMAX_LD(6) FMHA_SEGMAX_LD(7)
  FMHA_SEGMAX_LD(8) FMHA_SEGMAX_LD(9) FMHA_SEGMAX_LD(10) FMHA_SEGMAX_LD(11)
  FMHA_SEGMAX_LD(12) FMHA_SEGMAX_LD(13) FMHA_SEGMAX_LD(14) FMHA_SEGMAX_LD(15)
#undef FMHA_SEGMAX_LD
  FMHA_SEGMAX_MAX(0) FMHA_SEGMAX_MAX(1) FMHA_SEGMAX_MAX(2) FMHA_SEGMAX_MAX(3)
  FMHA_SEGMAX_MAX(4) FMHA_SEGMAX_MAX(5) FMHA_SEGMAX_MAX(6) FMHA_SEGMAX_MAX(7)
  FMHA_SEGMAX_MAX(8) FMHA_SEGMAX_MAX(9) FMHA_SEGMAX_MAX(10) FMHA_SEGMAX_MAX(11)
  FMHA_SEGMAX_MAX(12) FMHA_SEGMAX_MAX(13) FMHA_SEGMAX_MAX(14) FMHA_SEGMAX_MAX(15)
#undef FMHA_SEGMAX_MAX
  segmax[(size_t)blockIdx.y * rows + row] = mx;
}

// Pass B: one thread per row, the serial recurrence - but over chunks of sixteen
// segments, and a chunk it can prove contains no jump costs stores and no loads at
// all.  `__launch_bounds__(256,2)` is free here and provably so: this kernel only
// runs on the split path, which needs `base = B*KH*qtiles128 < resident`, and
// `qtiles128 >= rows_total/128`, so `rows_total < 296*128 = 37888` and the launch is
// at most 148 blocks - never two per SM, so the occupancy a register cap would
// protect does not exist to lose.
__global__ void __launch_bounds__(256, 2) fmha_scan_kernel(
    const float* __restrict__ bm, float* __restrict__ mstate, float* __restrict__ mtail,
    const float* __restrict__ segmax, int rows, int T, int nseg) {
  fmha_pdl_wait();
  fmha_pdl_scan_trigger();  // primary of scan -> split; see fmha_scan_flat_kernel
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const size_t stride = (size_t)rows;
  const float* src = bm + row;
  float* dst = mstate + row;
  const float* srow = segmax + row;
  float m = -INFINITY;
  // Segment 0 is walked unconditionally: `m` starts at -inf, so its test is
  // `+inf > 4 ln 2` (or NaN when the segment is entirely masked, which the serial
  // scan skips and leaves at -inf) and walking it gives the identical state either
  // way.  Walking is always correct; skipping is the optimisation.
  fmha_scan_walk<true>(src, dst, 0, (FMHA_SCAN_SEG < T) ? FMHA_SCAN_SEG : T, stride, m);
  for (int sbase = 1; sbase < nseg; sbase += FMHA_SCAN_SEGCHUNK) {
    const int t0 = sbase * FMHA_SCAN_SEG;
    const int t1 =
        (t0 + FMHA_SCAN_SEGCHUNK * FMHA_SCAN_SEG < T) ? (t0 + FMHA_SCAN_SEGCHUNK * FMHA_SCAN_SEG) : T;
    if (t0 >= t1) break;
    // This chunk's sixteen segment maxima: sixteen independent addresses with
    // nothing between them and their only consumer, so they issue as one round.
    // `agw3` put a function call in that window and ptxas sank every load back to
    // its use, which is why it kept `agw2`'s fetch win and lost the batching.
    const float g0 = (sbase + 0 < nseg) ? __ldg(srow + (size_t)(sbase + 0) * rows)
                                              : -INFINITY;
    const float g1 = (sbase + 1 < nseg) ? __ldg(srow + (size_t)(sbase + 1) * rows)
                                              : -INFINITY;
    const float g2 = (sbase + 2 < nseg) ? __ldg(srow + (size_t)(sbase + 2) * rows)
                                              : -INFINITY;
    const float g3 = (sbase + 3 < nseg) ? __ldg(srow + (size_t)(sbase + 3) * rows)
                                              : -INFINITY;
    const float g4 = (sbase + 4 < nseg) ? __ldg(srow + (size_t)(sbase + 4) * rows)
                                              : -INFINITY;
    const float g5 = (sbase + 5 < nseg) ? __ldg(srow + (size_t)(sbase + 5) * rows)
                                              : -INFINITY;
    const float g6 = (sbase + 6 < nseg) ? __ldg(srow + (size_t)(sbase + 6) * rows)
                                              : -INFINITY;
    const float g7 = (sbase + 7 < nseg) ? __ldg(srow + (size_t)(sbase + 7) * rows)
                                              : -INFINITY;
    const float g8 = (sbase + 8 < nseg) ? __ldg(srow + (size_t)(sbase + 8) * rows)
                                              : -INFINITY;
    const float g9 = (sbase + 9 < nseg) ? __ldg(srow + (size_t)(sbase + 9) * rows)
                                              : -INFINITY;
    const float g10 = (sbase + 10 < nseg) ? __ldg(srow + (size_t)(sbase + 10) * rows)
                                              : -INFINITY;
    const float g11 = (sbase + 11 < nseg) ? __ldg(srow + (size_t)(sbase + 11) * rows)
                                              : -INFINITY;
    const float g12 = (sbase + 12 < nseg) ? __ldg(srow + (size_t)(sbase + 12) * rows)
                                              : -INFINITY;
    const float g13 = (sbase + 13 < nseg) ? __ldg(srow + (size_t)(sbase + 13) * rows)
                                              : -INFINITY;
    const float g14 = (sbase + 14 < nseg) ? __ldg(srow + (size_t)(sbase + 14) * rows)
                                              : -INFINITY;
    const float g15 = (sbase + 15 < nseg) ? __ldg(srow + (size_t)(sbase + 15) * rows)
                                              : -INFINITY;
    // Test all sixteen against the INCOMING m and collect the results in a mask,
    // still with no call in sight.  A segment that fails this test cannot jump at
    // all - `m` is non-decreasing, so `m_(j-1) >= m` inside it and monotonicity of
    // subtraction gives `fl(b_j - m_(j-1)) <= fl(g_k - m)` - so the mask is a
    // superset of the true jump set, and its LOWEST set bit is exactly the first
    // segment that might jump.  Every segment before that one provably does not, so
    // its entries are the constant `m`; from it onward the walk is exact by
    // construction.  One call, after every loaded value has been consumed.
    unsigned jmp = 0;
    if ((g0 - m) > RESCALE_THRESHOLD) jmp |= 1u << 0;
    if ((g1 - m) > RESCALE_THRESHOLD) jmp |= 1u << 1;
    if ((g2 - m) > RESCALE_THRESHOLD) jmp |= 1u << 2;
    if ((g3 - m) > RESCALE_THRESHOLD) jmp |= 1u << 3;
    if ((g4 - m) > RESCALE_THRESHOLD) jmp |= 1u << 4;
    if ((g5 - m) > RESCALE_THRESHOLD) jmp |= 1u << 5;
    if ((g6 - m) > RESCALE_THRESHOLD) jmp |= 1u << 6;
    if ((g7 - m) > RESCALE_THRESHOLD) jmp |= 1u << 7;
    if ((g8 - m) > RESCALE_THRESHOLD) jmp |= 1u << 8;
    if ((g9 - m) > RESCALE_THRESHOLD) jmp |= 1u << 9;
    if ((g10 - m) > RESCALE_THRESHOLD) jmp |= 1u << 10;
    if ((g11 - m) > RESCALE_THRESHOLD) jmp |= 1u << 11;
    if ((g12 - m) > RESCALE_THRESHOLD) jmp |= 1u << 12;
    if ((g13 - m) > RESCALE_THRESHOLD) jmp |= 1u << 13;
    if ((g14 - m) > RESCALE_THRESHOLD) jmp |= 1u << 14;
    if ((g15 - m) > RESCALE_THRESHOLD) jmp |= 1u << 15;
    if (jmp == 0) {
      for (int g = t0; g < t1; ++g) dst[(size_t)g * stride] = m;
    } else {
      const int tp = t0 + (__ffs(jmp) - 1) * FMHA_SCAN_SEG;
      for (int g = t0; g < tp; ++g) dst[(size_t)g * stride] = m;
      fmha_scan_walk<true>(src, dst, (tp < t1) ? tp : t1, t1, stride, m);
    }
  }
  mtail[row] = m;
}


// B1: one thread per row; writes the segment-boundary states and mtail only.
__global__ void __launch_bounds__(256, 2) fmha_scan_kernel_b1(
    const float* __restrict__ bm, float* __restrict__ mstate, float* __restrict__ mtail,
    const float* __restrict__ segmax, int rows, int T, int nseg) {
  fmha_pdl_wait();  // PDL successor: of segmax (b1) / of b1 (segfill)
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const size_t stride = (size_t)rows;
  const float* src = bm + row;
  float* dst = mstate + row;
  const float* srow = segmax + row;
  float m = -INFINITY;
  // Segment 0 unconditionally (same argument as pass B): m starts at -inf.
  fmha_scan_walk<false>(src, dst, 0, (FMHA_SCAN_SEG < T) ? FMHA_SCAN_SEG : T, stride, m);
  for (int sbase = 1; sbase < nseg; sbase += FMHA_SCAN_SEGCHUNK) {
    const int send = (sbase + FMHA_SCAN_SEGCHUNK < nseg) ? (sbase + FMHA_SCAN_SEGCHUNK) : nseg;
    // Sixteen independent segmax loads, same as pass B - see the agw3 note there.
    const float g0 = (sbase + 0 < nseg) ? __ldg(srow + (size_t)(sbase + 0) * rows) : -INFINITY;
    const float g1 = (sbase + 1 < nseg) ? __ldg(srow + (size_t)(sbase + 1) * rows) : -INFINITY;
    const float g2 = (sbase + 2 < nseg) ? __ldg(srow + (size_t)(sbase + 2) * rows) : -INFINITY;
    const float g3 = (sbase + 3 < nseg) ? __ldg(srow + (size_t)(sbase + 3) * rows) : -INFINITY;
    const float g4 = (sbase + 4 < nseg) ? __ldg(srow + (size_t)(sbase + 4) * rows) : -INFINITY;
    const float g5 = (sbase + 5 < nseg) ? __ldg(srow + (size_t)(sbase + 5) * rows) : -INFINITY;
    const float g6 = (sbase + 6 < nseg) ? __ldg(srow + (size_t)(sbase + 6) * rows) : -INFINITY;
    const float g7 = (sbase + 7 < nseg) ? __ldg(srow + (size_t)(sbase + 7) * rows) : -INFINITY;
    const float g8 = (sbase + 8 < nseg) ? __ldg(srow + (size_t)(sbase + 8) * rows) : -INFINITY;
    const float g9 = (sbase + 9 < nseg) ? __ldg(srow + (size_t)(sbase + 9) * rows) : -INFINITY;
    const float g10 = (sbase + 10 < nseg) ? __ldg(srow + (size_t)(sbase + 10) * rows) : -INFINITY;
    const float g11 = (sbase + 11 < nseg) ? __ldg(srow + (size_t)(sbase + 11) * rows) : -INFINITY;
    const float g12 = (sbase + 12 < nseg) ? __ldg(srow + (size_t)(sbase + 12) * rows) : -INFINITY;
    const float g13 = (sbase + 13 < nseg) ? __ldg(srow + (size_t)(sbase + 13) * rows) : -INFINITY;
    const float g14 = (sbase + 14 < nseg) ? __ldg(srow + (size_t)(sbase + 14) * rows) : -INFINITY;
    const float g15 = (sbase + 15 < nseg) ? __ldg(srow + (size_t)(sbase + 15) * rows) : -INFINITY;
    unsigned jmp = 0;
    if ((g0 - m) > RESCALE_THRESHOLD) jmp |= 1u << 0;
    if ((g1 - m) > RESCALE_THRESHOLD) jmp |= 1u << 1;
    if ((g2 - m) > RESCALE_THRESHOLD) jmp |= 1u << 2;
    if ((g3 - m) > RESCALE_THRESHOLD) jmp |= 1u << 3;
    if ((g4 - m) > RESCALE_THRESHOLD) jmp |= 1u << 4;
    if ((g5 - m) > RESCALE_THRESHOLD) jmp |= 1u << 5;
    if ((g6 - m) > RESCALE_THRESHOLD) jmp |= 1u << 6;
    if ((g7 - m) > RESCALE_THRESHOLD) jmp |= 1u << 7;
    if ((g8 - m) > RESCALE_THRESHOLD) jmp |= 1u << 8;
    if ((g9 - m) > RESCALE_THRESHOLD) jmp |= 1u << 9;
    if ((g10 - m) > RESCALE_THRESHOLD) jmp |= 1u << 10;
    if ((g11 - m) > RESCALE_THRESHOLD) jmp |= 1u << 11;
    if ((g12 - m) > RESCALE_THRESHOLD) jmp |= 1u << 12;
    if ((g13 - m) > RESCALE_THRESHOLD) jmp |= 1u << 13;
    if ((g14 - m) > RESCALE_THRESHOLD) jmp |= 1u << 14;
    if ((g15 - m) > RESCALE_THRESHOLD) jmp |= 1u << 15;
    if (jmp == 0) {
      // No segment in this chunk can jump: every boundary in it is the constant m.
      for (int s = sbase; s < send; ++s) dst[(size_t)(s * FMHA_SCAN_SEG - 1) * stride] = m;
    } else {
      const int f = sbase + (__ffs(jmp) - 1);
      // Segments before f cannot jump, so entry(f) - written here with s == f -
      // is still the constant m; the walk below starts AT segment f.
      for (int s = sbase; s <= f && s < send; ++s)
        dst[(size_t)(s * FMHA_SCAN_SEG - 1) * stride] = m;
      for (int s = (f < send) ? f : send; s < send; ++s) {
        const int t0 = s * FMHA_SCAN_SEG;
        const int t1 = (t0 + FMHA_SCAN_SEG < T) ? (t0 + FMHA_SCAN_SEG) : T;
        fmha_scan_walk<false>(src, dst, t0, t1, stride, m);
        if (s + 1 < nseg) dst[(size_t)((s + 1) * FMHA_SCAN_SEG - 1) * stride] = m;
      }
    }
  }
  mtail[row] = m;
}


// B2: one thread per (row, segment); fills [s*16, min(s*16+16,T)) of mstate from
// the boundary state B1 published, using the shipped walk.
__global__ void __launch_bounds__(256, 4) fmha_segfill_kernel(
    const float* __restrict__ bm, float* __restrict__ mstate, int rows, int T) {
  fmha_pdl_wait();  // PDL successor: of segmax (b1) / of b1 (segfill)
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int s = blockIdx.y;
  const size_t stride = (size_t)rows;
  const int t0 = s * FMHA_SCAN_SEG;
  const int t1 = (t0 + FMHA_SCAN_SEG < T) ? (t0 + FMHA_SCAN_SEG) : T;
  float m = (s == 0) ? -INFINITY
                     : __ldg(mstate + (size_t)(t0 - 1) * stride + row);
  fmha_scan_walk<true>(bm + row, mstate + row, t0, t1, stride, m);
}

// Threads per CTA of `fmha_combine_kernel`, and therefore the largest `nsplit` its
// `__shared__ float sB[BT]` denominator buffer can hold.  Both combine kernels are
// correct for any `nsplit` only if this bound is respected, and it is now enforced at
// the launch site rather than assumed: `launch_combine` routes to the wide kernel
// whenever `nsplit > FMHA_COMBINE_NARROW_BT`.
//
// It used to be enforced by a coincidence in another file.  The comment below read
// "plan_geometry caps nsplit at 64", which was true until `agy1` deleted that cap
// (`if (ns > 64) ns = 64;`, a Phase C bring-up constant with no measurement behind it),
// after which `nsplit` is bounded only by one resident wave, `T`, `T/2` and the 384 MB
// partial cap - so it reaches 296 at D=128 with `base = 1`.  The narrow kernel is still
// safe, but only by a derivation that spans two modules: it is selected when
// `rows >= 2*sms`, `base >= rows/128` holds for every legal geometry because
// `qtiles128 = ceil(Sq/(128/G)) >= Sq*G/128`, so `rows >= 2*sms` gives `base >= sms/64`
// and hence `nsplit <= resident/base <= 2*sms/(sms/64) = 128`.  Exactly 128, with zero
// margin, on any device.  The launch-site guard below makes that local and gives it
// margin; it is unreachable today, so it changes no measurement.
#ifndef FMHA_COMBINE_NARROW_BT
#define FMHA_COMBINE_NARROW_BT 128
#endif

// Split-groups the wide combine reduces over, i.e. `blockDim.x == 128 * SP`.  Each group
// takes `chunk = ceil(nsplit/SP)` of the splits, so this looked like the knob on how much
// serial depth a thread carries: `ap12` found `fmha_combine_kernel_wide` at 11.8-14.9% of
// s5/s2/d7 with `long_scoreboard` 67.2% on s5 at IPC 0.58 and only 42% `sm_active`, and
// `agy1` made that kernel 2.3x larger by taking s5's `part_a` from 6.3 MB to 14.5 MB,
// which put `chunk` at 37.
//
// **`aga1` (`dv_61e1a5f38fee`) measured SP=8 and it is worse, so 4 stays.** Halving the
// serial depth did not buy the latency hiding the `long_scoreboard` share implied:
// bias-corrected against s9/s3/s4 (+0.528%), **d7 read +2.91% median and +2.86% min,
// reproducing at +3.82% and +3.06% in the two rounds independently**, s5 was +0.22% and
// only s2 improved, at -1.76%/-2.13%. The mechanism the profile did not show is the final
// reduction: only the `sg == 0` group of 128 threads performs it, so raising SP both
// doubles that group's serial work over `sPart[SP][D]` and idles a larger fraction of the
// CTA at the `__syncthreads` in front of it - 7/8 of 1024 threads instead of 3/4 of 512.
// A kernel whose cost is that tail does not want more split-groups, it wants fewer, and
// the shapes disagree about how many (d7's grid is 192 CTAs, s2's 128, s5's 96), so there
// is no single SP that wins and 4 remains the shipped value.
//
// Exactness under SP=8 was fine and is recorded so it is not re-derived: all 11 records
// PASS with `mismatched: 0` in all four runs, d7/s9/s3/s4 bit-identical, s2's LSE
// `rel_l2` 3.7436e-08 -> 3.8241e-08, and s5's LSE actually *improved* on every field
// (`max_abs` 1.9073e-06 -> 9.5367e-07, `max_rel` 1.4578e-07 -> 8.1399e-08, `rel_l2`
// 3.5028e-08 -> 3.3470e-08). d7 stayed bit-identical despite using this kernel because its
// left window makes most of its 74 splits contribute exactly 0.0f, and reassociating a sum
// of zeros is exact.
#ifndef FMHA_COMBINE_SP
#define FMHA_COMBINE_SP 4
#endif

// Slice the wide combine's head dimension across the GRID instead of across threads.
//
// §98(d) measured this kernel as the only one in the candidate that is GRID-bounded rather
// than resource-bounded: it launches one CTA per output row, so s5 gets 96 CTAs and s2 128 on
// a 148-SM machine at 21.75% achieved occupancy, "while Block Limit Registers (4) and Shared
// Mem (12) would allow far more".  It then dismissed the cure in one sentence -- "splitting one
// row across CTAs would need a second cross-CTA reduction" -- which is true for splitting along
// the SPLIT INDEX and false for splitting along D: each slice writes a DISJOINT output range,
// and the only quantities shared between slices are `total_b` and the weights `w_{r,s}`, which
// are scalars over `s` alone and do not depend on `d0` or `K` at all, so every slice-CTA simply
// recomputes the identical value.  No reduction, no communication, no second kernel.
//
// This is NOT the `CT = D` variant §107 priced and closed.  That one keeps the grid at one CTA
// per row and doubles threads per CTA (96 x 1024), which is why it came out arithmetically
// identical to the SP=8 that `aga1` measured neutral.  Keeping CT=128 and slicing the grid
// gives D/CT CTAs per row at UNCHANGED threads per CTA, so on s5 96x512 becomes 192x512:
// per-thread loads 74 -> 37, the sg==0 tail 8 -> 4 adds, the barrier still spans 16 warps
// rather than SP=8's 32, the idle fraction stays 3/4 rather than 7/8, sPart[SP][D] shrinks to
// sPart[SP][CT] so SMEM 6.0 -> 4.0 KB, and all 148 SMs are used instead of 96.  Every term
// `aga1` blamed for SP=8 improves.  The price is that the row's weights and denominator are
// computed once per slice-CTA instead of once per row -- D/CT = 2x redundant at D=256, against
// fs4's measured +8.9%/+9.4% for 128x PER-THREAD redundancy (dv_c8141de5e812).
//
// At D=128 D/CT is 1, so the slice count is 1 and this is a NO-OP: only the D=256 wide combine
// changes, i.e. s5 in the fixed suite and nothing else in either suite (d3 is D=256 but takes
// the NARROW combine at rows 2032 >= 2*148, so SL=1 there too).
//
// **MEASURED: A RESOLVED NULL, AND THE ABLATION IS VERIFIED TO HAVE FIRED.** Job
// `dv_176b582f330e`, same-job ABBA under v2, `--suite both --ids s5,s2,s9,s4 --passes 16
// --l2-evict 1`, control against `-DFMHA_COMBINE_DSLICE=1`:
//
//     s5 (the only treated shape)  45.064 -> 45.056 us   -0.018%   0.1 sigma
//     s2 (wide, D=128, SL=1)       32.768 -> 32.768       0.000%   control
//     s4 (D=256 fused, no combine) 86.016 -> 86.016       0.000%   control
//     s9 (narrow combine)         133.064 -> 133.056      -0.006%   control
//
// against a predicted -3.7..-5.5% on s5 and §80's per-shape null sd of 0.2086%. §97's rule was
// applied before accepting the null -- verify that an ablation actually removed what it was built
// to remove -- and the ptxas census confirms it did: exactly **4 of 80** kernels differ between
// the arms and they are precisely the four `fmha_combine_kernel_wide<D=256,...>` instantiations,
// with static smem **4160 -> 2112** bytes (FWGT=0) and **6208 -> 4160** (FWGT=1), i.e.
// `sPart[4][256]` -> `sPart[4][128]` exactly as designed; all four D=128 wide instantiations are
// unchanged, which is the SL=1 no-op proven at the codegen level rather than argued.
//
// **So §98(d)'s "grid bounded by rows" is a true description with no cure.** Halving the
// per-thread load chain (74 -> 37), doubling the CTA count (96 -> 192), filling all 148 SMs
// instead of 96, halving the tail and halving SMEM does not move this kernel at all. Its 21.75%
// occupancy is not a starvation it can be relieved of, which retires the last unexplained
// low-occupancy kernel in the candidate. Kept at 0 with its evidence here rather than reverted,
// in the same style as FMHA_SCAN_B1B2 and FMHA_SPLIT_MAX: the mechanism is sound, the code is
// inert at the default, and the next reader should not have to re-derive or re-run it.
#ifndef FMHA_COMBINE_DSLICE
#define FMHA_COMBINE_DSLICE 0
#endif
// Slice count for a given head dimension, shared by the kernel and its launch site so they
// cannot disagree.  Always a power of two (1 or 2 here), which is what lets the kernel recover
// `b` and the slice index from blockIdx.z with a shift and a mask rather than the runtime
// division that job pI1 measured at 293 warp-instructions for ~30 of real work.
#define FMHA_COMBINE_SLICES(D) ((FMHA_COMBINE_DSLICE != 0) ? ((D) / 128) : 1)

// The largest `nsplit` the wide combine can reduce correctly, which is the bound its
// one-lane-per-split denominator load assumes: `chunk = ceil(nsplit/SP)` must not exceed
// `CT`.  plan_geometry clamps to this, so the kernel's assumption is enforced where it is
// free rather than paid for in the hot loop.  At `SP=4` and `CT=128` that is 512, against
// a maximum reachable `nsplit` of `resident/base <= 2*148 = 296` on this device, so the
// clamp never binds here - it exists so that a device with more SMs degrades to slightly
// less split parallelism instead of silently summing a prefix of the denominator.
#ifndef FMHA_COMBINE_MAX_NSPLIT
#define FMHA_COMBINE_MAX_NSPLIT (128 * FMHA_COMBINE_SP)
#endif

// Sums the per-split weighted partials and finishes the output conversion.
template <int D, bool BF16, bool FWGT = false>
__global__ void fmha_combine_kernel(const Params p) {
  fmha_pdl_wait();  // b-side PDL: first statement of a successor kernel
  // One block per output row, addressed as (query, head, batch) by the grid so the
  // row index never needs a runtime division.  ncu (job pI1) measured 293
  // warp-instructions per warp here for about 30 instructions of actual work on
  // frozen shape 9 - the rest was `row % Sq`, `row / Sq`, `bh % H` and `bh / H`
  // by runtime values, 17.7 us of a 219.6 us forward at 2.31 instructions per
  // cycle.  The grid is exactly B*H*Sq blocks, so no bounds check is needed.
  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int row = (b * p.H + h) * p.Sq + i;
  const int tid = threadIdx.x;

  // nsplit reaches 64 on the long-KV shapes and each partial is rows_total floats
  // from the next, so ncu showed this kernel at 0.16 instructions per cycle with
  // 86% long_scoreboard stalls (16.4 of domain d7's 190.8 us).  Eight independent
  // loads per round, summed in the original order, so the result is
  // bit-identical.  Named scalars, not an array: pH18 measured an array version
  // slower because ptxas spilled it to local memory.
  //
  // Two more fixes on top of that batching.  (1) The denominator total used to be
  // reduced by *every one of the 128 threads* straight out of global memory,
  // which for nsplit=64 is eight serialized rounds of global latency per thread
  // ahead of the numerator loop - pI18 measured 51.3% long_scoreboard here.  One
  // thread per split now loads it into shared memory (the planner caps nsplit at
  // 64, see plan_geometry) and the reduction reads it from there in the same
  // ascending order, so the sum is bit-identical.  (2) The numerator loads are
  // issued before that barrier, so a thread's whole memory-level parallelism is in
  // flight at once instead of denominator-then-numerator.
  // blockDim.x is FMHA_COMBINE_NARROW_BT for every launch of this kernel and
  // launch_combine refuses to send a larger nsplit here, so one round covers the whole
  // denominator.  See FMHA_COMBINE_NARROW_BT for why that is now checked rather than
  // assumed.
  constexpr int BT = FMHA_COMBINE_NARROW_BT;
  __shared__ float sB[BT];
  __shared__ float sW[FWGT ? BT : 1];
  // fs3: the row's final state, from which every chunk weight exp(m_s - mtail)
  // is formed.  A dead constant 0.0f for every non-fused instantiation.
  const float fmt = FWGT ? p.mtail[row] : 0.0f;
  for (int s = tid; s < p.nsplit; s += BT) sB[s] = p.part_b[(size_t)s * p.rows_total + row];
  if constexpr (FWGT) {
    // fs4: ONE expf + one part_m load per split per CTA, published by the same
    // barrier as sB.  fs3 computed the same weight per split per thread per
    // round instead (32 expf + 32 part_m loads per thread at nsplit=9); that
    // instruction flood - not registers, census said REG:32/STACK:0 in both
    // arms - consumed the whole scan-fusion win: s3 +8.9%, d3 +9.4% with
    // bit-identical / engine-PASS outputs (dv_c8141de5e812).
    for (int s = tid; s < p.nsplit; s += BT)
      sW[s] = fmha_chunk_wgt(p.part_m, (size_t)s * p.rows_total + row, fmt);
  }

  __nv_bfloat16* o_row = p.o + b * p.o_b + (int64_t)h * p.o_h + (int64_t)i * p.o_s;
  constexpr int K = D / BT;
  float acc[K];
#pragma unroll
  for (int k = 0; k < K; ++k) acc[k] = 0.0f;
  if constexpr (FWGT) {
    // fs4/fs5: channel-0 round-0 numerator loads are issued BEFORE the
    // barrier that publishes sB/sW, so fix (2)'s memory-level parallelism
    // survives; the weighted adds and any later rounds run after it (at K=2
    // channel 1 runs entirely after it - see its loop below).  Every product
    // is guarded by the split count, so an unfilled sW slot is never
    // multiplied (0 * garbage could be a NaN).  Values and ascending add
    // order are exactly fs3's, so s3-class FP32 partials stay bit-identical
    // to the shipped chain.
    const size_t stride = (size_t)p.rows_total * D;
    const int n0 = p.nsplit;
    const part_t<BF16>* pr =
        part_ptr<part_t<BF16>>(p.part_a, ((size_t)row) * D + tid);
    const float x0 = (n0 > 0) ? part_load(pr[0]) : 0.0f;
    const float x1 = (n0 > 1) ? part_load(pr[stride]) : 0.0f;
    const float x2 = (n0 > 2) ? part_load(pr[2 * stride]) : 0.0f;
    const float x3 = (n0 > 3) ? part_load(pr[3 * stride]) : 0.0f;
    const float x4 = (n0 > 4) ? part_load(pr[4 * stride]) : 0.0f;
    const float x5 = (n0 > 5) ? part_load(pr[5 * stride]) : 0.0f;
    const float x6 = (n0 > 6) ? part_load(pr[6 * stride]) : 0.0f;
    const float x7 = (n0 > 7) ? part_load(pr[7 * stride]) : 0.0f;
    __syncthreads();
    float a = 0.0f;
    a += (n0 > 0) ? x0 * sW[0] : 0.0f;
    a += (n0 > 1) ? x1 * sW[1] : 0.0f;
    a += (n0 > 2) ? x2 * sW[2] : 0.0f;
    a += (n0 > 3) ? x3 * sW[3] : 0.0f;
    a += (n0 > 4) ? x4 * sW[4] : 0.0f;
    a += (n0 > 5) ? x5 * sW[5] : 0.0f;
    a += (n0 > 6) ? x6 * sW[6] : 0.0f;
    a += (n0 > 7) ? x7 * sW[7] : 0.0f;
    acc[0] = a;
    for (int s0 = 8; s0 < p.nsplit; s0 += 8) {
      const int n8 = p.nsplit - s0;
      const part_t<BF16>* pr8 =
          part_ptr<part_t<BF16>>(p.part_a, ((size_t)s0 * p.rows_total + row) * D + tid);
      const float u0 = (n8 > 0) ? part_load(pr8[0]) : 0.0f;
      const float u1 = (n8 > 1) ? part_load(pr8[stride]) : 0.0f;
      const float u2 = (n8 > 2) ? part_load(pr8[2 * stride]) : 0.0f;
      const float u3 = (n8 > 3) ? part_load(pr8[3 * stride]) : 0.0f;
      const float u4 = (n8 > 4) ? part_load(pr8[4 * stride]) : 0.0f;
      const float u5 = (n8 > 5) ? part_load(pr8[5 * stride]) : 0.0f;
      const float u6 = (n8 > 6) ? part_load(pr8[6 * stride]) : 0.0f;
      const float u7 = (n8 > 7) ? part_load(pr8[7 * stride]) : 0.0f;
      acc[0] += (n8 > 0) ? u0 * sW[s0 + 0] : 0.0f;
      acc[0] += (n8 > 1) ? u1 * sW[s0 + 1] : 0.0f;
      acc[0] += (n8 > 2) ? u2 * sW[s0 + 2] : 0.0f;
      acc[0] += (n8 > 3) ? u3 * sW[s0 + 3] : 0.0f;
      acc[0] += (n8 > 4) ? u4 * sW[s0 + 4] : 0.0f;
      acc[0] += (n8 > 5) ? u5 * sW[s0 + 5] : 0.0f;
      acc[0] += (n8 > 6) ? u6 * sW[s0 + 6] : 0.0f;
      acc[0] += (n8 > 7) ? u7 * sW[s0 + 7] : 0.0f;
    }
    if constexpr (K > 1) {
      // fs5: channel 1 runs ALL of its rounds after the barrier.  fs4's
      // 16-load pre-issue did not survive ptxas at REG:35 - kt6 showed d3's
      // FWGT narrow at 14.61us against the shipped 8.30us while s3's 8-load
      // K=1 form cost only +0.37us - the sunk loads serialised fill -> all
      // loads (the pre-pI18 pathology).  An 8-load pre-issue is the proven
      // depth; channel 1 pays one exposed chain here but keeps fs3's flood
      // gone (every round still reads only the shared sW).
      float c = 0.0f;
      for (int s0 = 0; s0 < p.nsplit; s0 += 8) {
        const int n8 = p.nsplit - s0;
        const part_t<BF16>* prk =
            part_ptr<part_t<BF16>>(p.part_a, ((size_t)s0 * p.rows_total + row) * D + tid + BT);
        const float v0 = (n8 > 0) ? part_load(prk[0]) : 0.0f;
        const float v1 = (n8 > 1) ? part_load(prk[stride]) : 0.0f;
        const float v2 = (n8 > 2) ? part_load(prk[2 * stride]) : 0.0f;
        const float v3 = (n8 > 3) ? part_load(prk[3 * stride]) : 0.0f;
        const float v4 = (n8 > 4) ? part_load(prk[4 * stride]) : 0.0f;
        const float v5 = (n8 > 5) ? part_load(prk[5 * stride]) : 0.0f;
        const float v6 = (n8 > 6) ? part_load(prk[6 * stride]) : 0.0f;
        const float v7 = (n8 > 7) ? part_load(prk[7 * stride]) : 0.0f;
        c += (n8 > 0) ? v0 * sW[s0 + 0] : 0.0f;
        c += (n8 > 1) ? v1 * sW[s0 + 1] : 0.0f;
        c += (n8 > 2) ? v2 * sW[s0 + 2] : 0.0f;
        c += (n8 > 3) ? v3 * sW[s0 + 3] : 0.0f;
        c += (n8 > 4) ? v4 * sW[s0 + 4] : 0.0f;
        c += (n8 > 5) ? v5 * sW[s0 + 5] : 0.0f;
        c += (n8 > 6) ? v6 * sW[s0 + 6] : 0.0f;
        c += (n8 > 7) ? v7 * sW[s0 + 7] : 0.0f;
      }
      acc[1] = c;
    }
  } else {
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int d = tid + k * BT;
      float a = 0.0f;
      for (int s0 = 0; s0 < p.nsplit; s0 += 8) {
        const part_t<BF16>* pa =
            part_ptr<part_t<BF16>>(p.part_a, ((size_t)s0 * p.rows_total + row) * D + d);
        const int n8 = p.nsplit - s0;
        const size_t stride = (size_t)p.rows_total * D;
        const float x0 = (n8 > 0) ? part_load(pa[0]) : 0.0f;
        const float x1 = (n8 > 1) ? part_load(pa[stride]) : 0.0f;
        const float x2 = (n8 > 2) ? part_load(pa[2 * stride]) : 0.0f;
        const float x3 = (n8 > 3) ? part_load(pa[3 * stride]) : 0.0f;
        const float x4 = (n8 > 4) ? part_load(pa[4 * stride]) : 0.0f;
        const float x5 = (n8 > 5) ? part_load(pa[5 * stride]) : 0.0f;
        const float x6 = (n8 > 6) ? part_load(pa[6 * stride]) : 0.0f;
        const float x7 = (n8 > 7) ? part_load(pa[7 * stride]) : 0.0f;
        a += x0;
        a += x1;
        a += x2;
        a += x3;
        a += x4;
        a += x5;
        a += x6;
        a += x7;
      }
      acc[k] = a;
    }
    __syncthreads();
  }
  float total_b = 0.0f;
  for (int s0 = 0; s0 < p.nsplit; s0 += 8) {
    const int n8 = p.nsplit - s0;
    if constexpr (FWGT) {
      // fs4: both operands come from shared - the same single-rounded product
      // the shipped chain computes in the split epilogue (den * exp(m_s -
      // mtail), identical expf inputs), summed in the same ascending order.
      const float x0 = (n8 > 0) ? sB[s0 + 0] * sW[s0 + 0] : 0.0f;
      const float x1 = (n8 > 1) ? sB[s0 + 1] * sW[s0 + 1] : 0.0f;
      const float x2 = (n8 > 2) ? sB[s0 + 2] * sW[s0 + 2] : 0.0f;
      const float x3 = (n8 > 3) ? sB[s0 + 3] * sW[s0 + 3] : 0.0f;
      const float x4 = (n8 > 4) ? sB[s0 + 4] * sW[s0 + 4] : 0.0f;
      const float x5 = (n8 > 5) ? sB[s0 + 5] * sW[s0 + 5] : 0.0f;
      const float x6 = (n8 > 6) ? sB[s0 + 6] * sW[s0 + 6] : 0.0f;
      const float x7 = (n8 > 7) ? sB[s0 + 7] * sW[s0 + 7] : 0.0f;
      total_b += x0;
      total_b += x1;
      total_b += x2;
      total_b += x3;
      total_b += x4;
      total_b += x5;
      total_b += x6;
      total_b += x7;
    } else {
      const float x0 = (n8 > 0) ? sB[s0] : 0.0f;
      const float x1 = (n8 > 1) ? sB[s0 + 1] : 0.0f;
      const float x2 = (n8 > 2) ? sB[s0 + 2] : 0.0f;
      const float x3 = (n8 > 3) ? sB[s0 + 3] : 0.0f;
      const float x4 = (n8 > 4) ? sB[s0 + 4] : 0.0f;
      const float x5 = (n8 > 5) ? sB[s0 + 5] : 0.0f;
      const float x6 = (n8 > 6) ? sB[s0 + 6] : 0.0f;
      const float x7 = (n8 > 7) ? sB[s0 + 7] : 0.0f;
      total_b += x0;
      total_b += x1;
      total_b += x2;
      total_b += x3;
      total_b += x4;
      total_b += x5;
      total_b += x6;
      total_b += x7;
    }
  }
  const bool live = total_b > 0.0f;
  const float inv = live ? (1.0f / total_b) : 0.0f;
#pragma unroll
  for (int k = 0; k < D / BT; ++k) {
    o_row[tid + k * BT] = __float2bfloat16_rn(live ? acc[k] * inv : 0.0f);
  }
  if (p.has_lse && tid == 0) {
    const float mt = p.mtail[row];
    float value = -INFINITY;
    if (live && isfinite(mt)) value = mt + logf(total_b / p.bias);
    p.lse[b * p.l_b + (int64_t)h * p.l_h + (int64_t)i * p.l_s] = value;
  }
}

// Sums the per-split weighted partials and finishes the output conversion.
//
// One block per output row, addressed as (query, head, batch) by the grid so the
// row index never needs a runtime division (ncu, job pI1, measured 293
// warp-instructions per warp here for about 30 instructions of real work when it
// was `row % Sq`, `row / Sq`, `bh % H` and `bh / H`).
//
// The four-group variant, launched with blockDim.x == CT*4 only where the row
// count cannot fill the machine on its own; every other shape keeps
// fmha_combine_kernel below, byte for byte as validated at best/pI22_full, because
// rewriting it to share this body measured 2-5% slower on shapes 8 and 9 (pI26,
// pI27) for reasons the source does not show.  pI25 measured the wide form at -5.2% on shape 5 and -3.4% on domain d7, whose
// rows_total of 96 and 192 give 96 and 192 *blocks* - nowhere near enough to fill
// 148 SMs twice over - but at +42% on shape 8 and +20% on shape 9, whose 4096 and
// 8200 rows already fill the machine with 128-thread blocks and where the wider
// block only adds a sixteen-warp barrier and a shared-memory round trip per row.
// The host picks SP=4 only where the row count cannot fill the machine.
//
// The denominator follows the same split: at SP=1 every thread reduces the
// per-split values straight out of shared memory in ascending order, because a
// warp butterfly would cost five dependent shuffles per block and pI26 measured
// that as a net loss (-4.6% on shape 8, -1.9% on shape 9) when nsplit is small
// enough for the shared walk to be one or two rounds.  At SP=4 each thread owns
// one split, so the butterfly is the only way to avoid a serial walk.  Both forms
// reassociate by ~1e-7 relative at most, which is the same order of difference the
// key-split denominator exchange already makes and is invisible in a bf16 output
// at rtol 0.05.
template <int D, bool BF16, bool FWGT = false>
__global__ void fmha_combine_kernel_wide(const Params p) {
  fmha_pdl_wait();  // b-side PDL: first statement of a successor kernel
  constexpr int SP = FMHA_COMBINE_SP;
  constexpr int CT = 128;  // channels one group covers; blockDim.x == CT * SP
  // See FMHA_COMBINE_DSLICE.  SL == 1 makes every expression below reduce to the shipped one
  // exactly, so the default build is unchanged rather than merely equivalent.
  constexpr int SL = FMHA_COMBINE_SLICES(D);
  constexpr int K = (D / SL) / CT;   // channels per thread: 2 only for unsliced D=256
  const int i = blockIdx.x;
  const int h = blockIdx.y;
  // SL is a compile-time power of two, so this is a shift and a mask, not the runtime division
  // pI1 measured at 293 warp-instructions per warp for ~30 of actual work.
  const int b = (SL == 1) ? blockIdx.z : (int)(blockIdx.z / (unsigned)SL);
  const int sl = (SL == 1) ? 0 : ((int)blockIdx.z - b * SL);
  const int row = (b * p.H + h) * p.Sq + i;
  const int tid = threadIdx.x;
  const int sg = tid >> 7;
  const int d0 = tid & (CT - 1);
  const int cbase = sl * CT;   // this slice's first channel; 0 when SL == 1

  const int chunk = (p.nsplit + SP - 1) / SP;
  const int sLo = min(sg * chunk, p.nsplit);
  const int sHi = min(sLo + chunk, p.nsplit);

  __shared__ float sPart[SP][D / SL];   // this slice's channels only; [SP][D] when SL == 1
  __shared__ float sBw[SP][CT / 32];   // one denominator partial per warp
  __shared__ float sW[FWGT ? SP * CT : 1];  // fs4: per-group chunk weights

  // This thread's slice of the numerator: channels cbase + d0 (+CT when K == 2, i.e. only in
  // the unsliced D=256 form), splits [sLo, sHi).  Eight independent loads per round, named
  // scalars rather than an array because pH18 measured an array version slower (ptxas puts it
  // in local memory).
  float a[K];
#pragma unroll
  for (int k = 0; k < K; ++k) a[k] = 0.0f;
  const size_t stride = (size_t)p.rows_total * D;
  const part_t<BF16>* pa =
      part_ptr<part_t<BF16>>(p.part_a, ((size_t)sLo * p.rows_total + row) * D + cbase + d0);
  // fs3: dead constant 0.0f for every non-fused instantiation.
  const float fmt = FWGT ? p.mtail[row] : 0.0f;
  int s = sLo;
  if constexpr (FWGT) {
    // fs4: one expf + one part_m load per split per GROUP, on the lane that
    // owns the split, published by an early barrier.  The round-0 numerator
    // loads are issued before that barrier so the group's MLP still overlaps
    // the fill (fs3 recomputed the weight per thread per round instead:
    // dv_c8141de5e812 priced that flood at +8.9%/+9.4% on the narrow-treated
    // s3/d3).  Guards keep unfilled/garbage sW slots out of every product.
    const int q = sLo + d0;
    sW[sg * CT + d0] =
        (q < sHi) ? fmha_chunk_wgt(p.part_m, (size_t)q * p.rows_total + row, fmt) : 0.0f;
    const int n0 = sHi - sLo;
    const float x0 = (n0 > 0) ? part_load(pa[0]) : 0.0f;
    const float x1 = (n0 > 1) ? part_load(pa[stride]) : 0.0f;
    const float x2 = (n0 > 2) ? part_load(pa[2 * stride]) : 0.0f;
    const float x3 = (n0 > 3) ? part_load(pa[3 * stride]) : 0.0f;
    const float x4 = (n0 > 4) ? part_load(pa[4 * stride]) : 0.0f;
    const float x5 = (n0 > 5) ? part_load(pa[5 * stride]) : 0.0f;
    const float x6 = (n0 > 6) ? part_load(pa[6 * stride]) : 0.0f;
    const float x7 = (n0 > 7) ? part_load(pa[7 * stride]) : 0.0f;
    float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
    float y4 = 0.0f, y5 = 0.0f, y6 = 0.0f, y7 = 0.0f;
    if constexpr (K > 1) {
      y0 = (n0 > 0) ? part_load(pa[CT]) : 0.0f;
      y1 = (n0 > 1) ? part_load(pa[stride + CT]) : 0.0f;
      y2 = (n0 > 2) ? part_load(pa[2 * stride + CT]) : 0.0f;
      y3 = (n0 > 3) ? part_load(pa[3 * stride + CT]) : 0.0f;
      y4 = (n0 > 4) ? part_load(pa[4 * stride + CT]) : 0.0f;
      y5 = (n0 > 5) ? part_load(pa[5 * stride + CT]) : 0.0f;
      y6 = (n0 > 6) ? part_load(pa[6 * stride + CT]) : 0.0f;
      y7 = (n0 > 7) ? part_load(pa[7 * stride + CT]) : 0.0f;
    }
    __syncthreads();
    a[0] += (n0 > 0) ? x0 * sW[sg * CT + 0] : 0.0f;
    a[0] += (n0 > 1) ? x1 * sW[sg * CT + 1] : 0.0f;
    a[0] += (n0 > 2) ? x2 * sW[sg * CT + 2] : 0.0f;
    a[0] += (n0 > 3) ? x3 * sW[sg * CT + 3] : 0.0f;
    a[0] += (n0 > 4) ? x4 * sW[sg * CT + 4] : 0.0f;
    a[0] += (n0 > 5) ? x5 * sW[sg * CT + 5] : 0.0f;
    a[0] += (n0 > 6) ? x6 * sW[sg * CT + 6] : 0.0f;
    a[0] += (n0 > 7) ? x7 * sW[sg * CT + 7] : 0.0f;
    if constexpr (K > 1) {
      a[1] += (n0 > 0) ? y0 * sW[sg * CT + 0] : 0.0f;
      a[1] += (n0 > 1) ? y1 * sW[sg * CT + 1] : 0.0f;
      a[1] += (n0 > 2) ? y2 * sW[sg * CT + 2] : 0.0f;
      a[1] += (n0 > 3) ? y3 * sW[sg * CT + 3] : 0.0f;
      a[1] += (n0 > 4) ? y4 * sW[sg * CT + 4] : 0.0f;
      a[1] += (n0 > 5) ? y5 * sW[sg * CT + 5] : 0.0f;
      a[1] += (n0 > 6) ? y6 * sW[sg * CT + 6] : 0.0f;
      a[1] += (n0 > 7) ? y7 * sW[sg * CT + 7] : 0.0f;
    }
    s = sLo + 8;
    for (; s + 8 <= sHi; s += 8) {
      const part_t<BF16>* q8 = pa + (size_t)(s - sLo) * stride;
      const float u0 = part_load(q8[0]);
      const float u1 = part_load(q8[stride]);
      const float u2 = part_load(q8[2 * stride]);
      const float u3 = part_load(q8[3 * stride]);
      const float u4 = part_load(q8[4 * stride]);
      const float u5 = part_load(q8[5 * stride]);
      const float u6 = part_load(q8[6 * stride]);
      const float u7 = part_load(q8[7 * stride]);
      const int w8 = s - sLo;
      a[0] += u0 * sW[sg * CT + w8 + 0];
      a[0] += u1 * sW[sg * CT + w8 + 1];
      a[0] += u2 * sW[sg * CT + w8 + 2];
      a[0] += u3 * sW[sg * CT + w8 + 3];
      a[0] += u4 * sW[sg * CT + w8 + 4];
      a[0] += u5 * sW[sg * CT + w8 + 5];
      a[0] += u6 * sW[sg * CT + w8 + 6];
      a[0] += u7 * sW[sg * CT + w8 + 7];
      if constexpr (K > 1) {
        const float v0 = part_load(q8[CT]);
        const float v1 = part_load(q8[stride + CT]);
        const float v2 = part_load(q8[2 * stride + CT]);
        const float v3 = part_load(q8[3 * stride + CT]);
        const float v4 = part_load(q8[4 * stride + CT]);
        const float v5 = part_load(q8[5 * stride + CT]);
        const float v6 = part_load(q8[6 * stride + CT]);
        const float v7 = part_load(q8[7 * stride + CT]);
        a[1] += v0 * sW[sg * CT + w8 + 0];
        a[1] += v1 * sW[sg * CT + w8 + 1];
        a[1] += v2 * sW[sg * CT + w8 + 2];
        a[1] += v3 * sW[sg * CT + w8 + 3];
        a[1] += v4 * sW[sg * CT + w8 + 4];
        a[1] += v5 * sW[sg * CT + w8 + 5];
        a[1] += v6 * sW[sg * CT + w8 + 6];
        a[1] += v7 * sW[sg * CT + w8 + 7];
      }
    }
    for (; s < sHi; ++s) {
      const part_t<BF16>* q1 = pa + (size_t)(s - sLo) * stride;
      const float w = sW[sg * CT + (s - sLo)];
      a[0] += part_load(q1[0]) * w;
      if constexpr (K > 1) a[1] += part_load(q1[CT]) * w;
    }
  } else {
    for (; s + 8 <= sHi; s += 8) {
      const part_t<BF16>* q = pa + (size_t)(s - sLo) * stride;
      const float x0 = part_load(q[0]);
      const float x1 = part_load(q[stride]);
      const float x2 = part_load(q[2 * stride]);
      const float x3 = part_load(q[3 * stride]);
      const float x4 = part_load(q[4 * stride]);
      const float x5 = part_load(q[5 * stride]);
      const float x6 = part_load(q[6 * stride]);
      const float x7 = part_load(q[7 * stride]);
      a[0] += x0; a[0] += x1; a[0] += x2; a[0] += x3;
      a[0] += x4; a[0] += x5; a[0] += x6; a[0] += x7;
      if (K > 1) {
        const float y0 = part_load(q[CT]);
        const float y1 = part_load(q[stride + CT]);
        const float y2 = part_load(q[2 * stride + CT]);
        const float y3 = part_load(q[3 * stride + CT]);
        const float y4 = part_load(q[4 * stride + CT]);
        const float y5 = part_load(q[5 * stride + CT]);
        const float y6 = part_load(q[6 * stride + CT]);
        const float y7 = part_load(q[7 * stride + CT]);
        a[1] += y0; a[1] += y1; a[1] += y2; a[1] += y3;
        a[1] += y4; a[1] += y5; a[1] += y6; a[1] += y7;
      }
    }
    for (; s < sHi; ++s) {
      const part_t<BF16>* q = pa + (size_t)(s - sLo) * stride;
      a[0] += part_load(q[0]);
      if (K > 1) a[1] += part_load(q[CT]);
    }
  }
#pragma unroll
  for (int k = 0; k < K; ++k) sPart[sg][d0 + k * CT] = a[k];

  {
    // One lane per split, so this covers the group's whole chunk ONLY when
    // `chunk = ceil(nsplit/SP) <= CT`, i.e. when `nsplit <= SP * CT`.  Past that it would
    // silently drop every split after the first CT of each group and corrupt `total_b`,
    // `inv`, the output AND the LSE with no diagnostic.  The bound is enforced on the host
    // by `FMHA_COMBINE_MAX_NSPLIT` in plan_geometry rather than by looping here, because
    // looping costs real time: this load is deliberately placed ahead of the `__syncthreads`
    // so that it is in flight together with the numerator loads above - see fix (2) in this
    // kernel's header comment - and `aga2` (`dv_e99c059fca41`) measured wrapping it in a
    // runtime-trip-count loop at **+2.22% on s5, +1.82% on s2 and +1.01% on d7**, exactly
    // the three shapes that select this kernel, against +0.08%/-0.65%/-0.71% on the three
    // that do not, with all six bit-identical.  Exposing that one load's latency costs more
    // than a bound that can never bind is worth.
    const int q = sLo + d0;
    float v =
        (q < sHi) ? p.part_b[(size_t)q * p.rows_total + row] : 0.0f;
    // fs4: the weight this lane published to sW before the early barrier; the
    // dead branch folds away for every non-fused instantiation.
    if constexpr (FWGT) v *= sW[sg * CT + d0];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if ((tid & 31) == 0) sBw[sg][d0 >> 5] = v;
  }
  __syncthreads();

  if (sg == 0) {
    float total_b = 0.0f;
#pragma unroll
    for (int g = 0; g < SP; ++g) {
#pragma unroll
      for (int w = 0; w < CT / 32; ++w) total_b += sBw[g][w];
    }
    const bool live = total_b > 0.0f;
    const float inv = live ? (1.0f / total_b) : 0.0f;
    __nv_bfloat16* o_row = p.o + b * p.o_b + (int64_t)h * p.o_h + (int64_t)i * p.o_s;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float v = 0.0f;
#pragma unroll
      for (int g = 0; g < SP; ++g) v += sPart[g][d0 + k * CT];
      // cbase is 0 when SL == 1.  Slices write DISJOINT channel ranges, which is why no
      // cross-CTA reduction is needed -- the only shared values are total_b and the weights,
      // and both are scalars over the split index that every slice recomputes identically.
      o_row[cbase + d0 + k * CT] = __float2bfloat16_rn(live ? v * inv : 0.0f);
    }
    // One writer for the row's LSE: every slice computes the same total_b, so any of them
    // could, and letting all of them would be a benign but pointless duplicate store.
    if (p.has_lse && d0 == 0 && sl == 0) {
      const float mt = p.mtail[row];
      float value = -INFINITY;
      if (live && isfinite(mt)) value = mt + logf(total_b / p.bias);
      p.lse[b * p.l_b + (int64_t)h * p.l_h + (int64_t)i * p.l_s] = value;
    }
  }
}

// Shared-memory bytes required by fmha_tc_kernel<D, WARP, PASS>.
template <int D, int WARP, int PASS, int PVMODE = 0>
struct TcSmem {
  static constexpr size_t bytes = TcLayout<D, WARP, PASS, PVMODE>::bytes;
};

}  // namespace fmha_mxfp8
