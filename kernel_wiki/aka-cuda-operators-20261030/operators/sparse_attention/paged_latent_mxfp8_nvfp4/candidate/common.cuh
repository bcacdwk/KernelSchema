#pragma once
// Exact byte/nibble -> fp16 bit decoding for the two frozen paged KV formats of
// paged_latent_mxfp8_nvfp4:
//   * MXFP8 SWA page record 528B = 512 x float8_e4m3 values + 16 ue8m0 group
//     scales (2^(s-127)), one scale per 32 values.
//   * NVFP4 compressed page record 288B = 256 bytes holding 512 e2m1 nibbles
//     (low nibble = even element, as problem/reference.py::read_rows stacks
//     them) + 32 float8_e4m3 group scales, one per 16 values.
//
// These helpers are pure integer/float bit math with no CUDA dependency so that
// experiments/local/decode_math.py can exhaustively verify every encoding on the
// host against torch's float8_e4m3fn view and the reference e2m1 LUT.
#if defined(__CUDACC__)
#define PLMN_FN __host__ __device__ inline
#else
#define PLMN_FN static inline
#endif
#include <cstdint>

// float8_e4m3fn -> IEEE half bit pattern. Exact for all 256 encodings,
// including the e4m3 subnormals (e==0), signed zeros and the 0x7f/0xff NaN.
PLMN_FN uint32_t plmn_e4m3_to_half_bits(uint32_t x) {
  uint32_t sign = (x & 0x80u) << 8;
  uint32_t e = (x >> 3) & 0xFu;
  uint32_t m = x & 0x7u;
  if (e == 0u) {
    if (m == 0u) return sign;                        // +-0
    uint32_t sh = (m >= 4u) ? 0u : ((m >= 2u) ? 1u : 2u);
    uint32_t mm = m << sh;                           // now in [4,7]
    return sign | ((8u - sh) << 10) | ((mm - 4u) << 8);
  }
  if (e == 15u && m == 7u) return sign | 0x7E00u;     // NaN (e4m3fn has no Inf)
  return sign | ((e + 8u) << 10) | (m << 7);
}

// e2m1 nibble -> IEEE half bit pattern. Exact for all 16 encodings.
PLMN_FN uint32_t plmn_e2m1_to_half_bits(uint32_t n) {
  uint32_t sign = (n & 0x8u) << 12;
  uint32_t mag = n & 0x7u;
  if (mag == 0u) return sign;                        // +-0
  uint32_t e = (mag >> 1) & 0x3u;
  uint32_t m = mag & 0x1u;
  return sign | (((e + 14u) << 10) | (e ? (m << 9) : 0u));
}

// ue8m0 MXFP8 group scale byte -> IEEE half bit pattern for 2^(s-127).
// Exact for s in [103,142]; below that the scale underflows to zero and above it
// saturates to Inf, both matching an fp32->fp16 round. The frozen packer only
// emits s = ceil(log2(amax/448)) + 127, measured 105..119 in-domain.
PLMN_FN uint32_t plmn_ue8m0_to_half_bits(uint32_t s) {
  if (s >= 113u && s <= 142u) return (s - 112u) << 10;
  if (s >= 103u && s < 113u) return 1u << (s - 103u);
  if (s > 142u) return 0x7C00u;
  return 0u;
}

// ue8m0 MXFP8 group scale byte -> IEEE bf16 bit pattern for 2^(s-127). bf16's
// exponent field covers every encoding exactly, so unlike the fp16 form this needs
// no underflow/saturation split: s in [1,254] are normals with a zero mantissa,
// s == 0 is the subnormal 2^-127 and s == 255 is ue8m0's NaN.
PLMN_FN uint32_t plmn_ue8m0_to_bf16_bits(uint32_t s) {
  if (s >= 1u && s <= 254u) return s << 7;
  if (s == 0u) return 0x0040u;
  return 0x7FC0u;
}

// half bit pattern -> float, host testable, exact.
PLMN_FN float plmn_half_bits_to_float(uint32_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1Fu;
  uint32_t m = h & 0x3FFu;
  uint32_t f;
  if (e == 0u) {
    if (m == 0u) {
      f = 0u;
    } else {
      uint32_t sh = 0u;
      while (!(m & 0x400u)) { m <<= 1; ++sh; }
      m &= 0x3FFu;
      f = ((113u - sh) << 23) | (m << 13);
    }
  } else if (e == 31u) {
    f = (0xFFu << 23) | (m << 13);
  } else {
    f = ((e + 112u) << 23) | (m << 13);
  }
  f |= sign;
  union { uint32_t u; float v; } c;
  c.u = f;
  return c.v;
}

#pragma once
// Shared host-side context for the paged_latent_mxfp8_nvfp4 candidate.
//
// plmn.cu compiles the SAME kernel body twice (HB=32/NT=256 and HB=64/NT=512)
// and picks one per call from declared shape metadata alone. This struct is what
// crosses that boundary: it carries raw pointers and frozen shape integers only,
// so it is configuration-independent and pulls in no Torch type.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace plmn {

// frozen latent head dim (problem/metadata.json: D512/V512)
constexpr int FwdCtx_D = 512;

struct FwdCtx {
  // frozen inputs (validated by plmn_forward before this is filled)
  const __nv_bfloat16* q;
  const unsigned char* k1;   // SWA cache bytes
  const int* idx1;           // SWA selectors
  const unsigned char* k2;   // compressed cache bytes, nullptr when absent
  const int* idx2;           // compressed selectors, nullptr when absent
  const float* sink;         // per-head attn_sink
  float sm_scale;
  // outputs: the caller's own bf16 out, and the fresh [B,S,H] lse base
  __nv_bfloat16* out;
  float* lse;
  // shape / page metadata
  int H;
  int BS;
  int budget1;
  int budget2;
  int page1;
  int rec1;
  int page2;
  int rec2;
  int listn;
  // Upper bound on the tokens one request can actually select, derived from the
  // cache page counts (tensor shapes, so free to read). The selector WIDTH is
  // only a bound on it - input.py fills the rest with -1 - and the split rule
  // needs the real chunk count. See the comment in plmn_forward.
  int nest;
  // device properties, queried once per process
  int nsm;
  int smem_per_sm;
};

}  // namespace plmn
