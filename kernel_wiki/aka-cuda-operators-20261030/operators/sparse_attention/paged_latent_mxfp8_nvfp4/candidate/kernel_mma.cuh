// Configuration-parameterised body of the paged_latent_mxfp8_nvfp4 candidate.
//
// This file is NOT a header: launch.cu includes it TWICE, once per launch
// geometry, and the includer must define PLMN_CFG (the namespace), PLMN_HB
// (heads per CTA) and PLMN_NT (threads per CTA) first. It must therefore
// carry no include guard and no #pragma once, and every include it needs is
// already provided by launch.cu.
//
// Nothing below depends on which configuration is being compiled except
// through those three macros, so both instantiations compute bit-identical
// results: grouping heads into CTAs is a launch geometry, not an arithmetic
// change.
namespace PLMN_CFG {

// The three __global__ entry points are named through macros so that the two
// configurations this file is compiled into never share a device symbol, even
// though both live in an unnamed namespace. Everything else here has internal
// linkage or is inlined, so no other name needs this treatment.

namespace {

// PLMN_HB / PLMN_NT select the head-block and CTA width; everything else is
// derived so the two configurations can be A/B measured from one source.
//   HB=32, NT=256: 2 CTAs/SM (16 warps/SM), dequant repeated H/32 times.
//   HB=64, NT=512: 1 CTA/SM (16 warps/SM), dequant repeated H/64 times, i.e. half
//     the redundant dequantisation at identical total shared-memory traffic.
#ifndef PLMN_HB
#define PLMN_HB 32
#endif
#ifndef PLMN_NT
#define PLMN_NT 256
#endif
constexpr int HB = PLMN_HB;    // heads per CTA (grid.x = H / HB)
constexpr int NC = 64;         // selected tokens per streamed chunk
constexpr int NT = PLMN_NT;    // threads per CTA
constexpr int NW = NT / 32;    // warps per CTA
constexpr int NWN = 4;         // warps along tokens (QK) / output dims (PV)
constexpr int NBG = NT / NC;   // dequant slices per token row
constexpr int MINCTA = (NT <= 256) ? 2 : 1;  // CTAs/SM the register budget allows
constexpr int D = 512;         // frozen latent head dim
constexpr int SLICE = D / NBG;    // dims per dequant slice
constexpr int BLKS = SLICE / 32;  // 32-value blocks per thread per chunk
static_assert(NT == NBG * NC, "dequant mapping needs NT == NC * slices");
static_assert(D % NBG == 0 && SLICE % 32 == 0, "dequant slice must be 32-value blocks");
static_assert(HB % (16 * (NW / NWN)) == 0, "head tiles must cover HB exactly");
static_assert(NC % (8 * NWN) == 0, "token warps need whole 8-token mma n-tiles");
constexpr int KS = D + 8;      // padded fp16 row stride for Q and K
constexpr int PS = NC + 8;     // padded fp16 row stride for P
constexpr int MAXENT = 4;      // selector entries per thread during compaction
constexpr int MAXSPLIT = 16;   // cap on the adaptive token-split fan-out
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

constexpr int SMEM_Q = HB * KS * 2;
constexpr int SMEM_K = NC * KS * 2;
constexpr int SMEM_P = HB * PS * 2;
constexpr int SMEM_RED = NWN * HB * 4;  // cross-warp row maxima
constexpr int SMEM_FIXED = SMEM_Q + SMEM_K + SMEM_P + SMEM_RED + 64;
// meta is one int2 per candidate selector slot; SMEM_FIXED already covers the
// fixed regions and the launcher adds 8 bytes per slot.

__device__ __forceinline__ float pinf() { return __int_as_float(0x7f800000); }
__device__ __forceinline__ float ninf() { return __int_as_float(0xff800000); }

union H2 {
  uint32_t u;
  __half2 h;
};

// The 16-bit element format of one MMA tile. Both formats hold every decoded cache
// value exactly (e4m3 x ue8m0 needs 4 significand bits, e2m1 x e4m3 needs 7; f16
// has 11 and bf16 8), so this choice is about the operand *pair*, not about decode
// accuracy: SM103 traps on a mixed f16/bf16 pair (micro_fmt: bf16f16 -> illegal
// instruction), so one kernel's Q, K and probability tiles all share one format.
//   * the tcgen05 path builds BF16, which makes its A operand the caller's own bf16
//     Q exactly as the TMA staged it - the whole in-shared-memory conversion pass
//     disappears (64 KB read + 64 KB written back per CTA for M64, twice that for
//     M128, on the fixed cost of every shape);
//   * the mma.sync path keeps F16, where it has always converted Q.
// The frozen 0.01/0.05 tolerance is met by both: the production library this
// question is measured against also decodes to bf16 and applies the group scale in
// bf16 (csrc/kernels/sm100/helpers.h: cvt.rn.bf16x2.e4m3x2 then __hmul2), and its
// fixed/domain receipts carry the same worst-case error as this candidate's.
template<bool BF16> struct Tile16;
template<> struct Tile16<false> { using elem = __half; using pair = __half2; };
template<> struct Tile16<true>  { using elem = __nv_bfloat16; using pair = __nv_bfloat162; };

// Packed 16-bit helpers in the tile's own format. They take and return the raw
// 32-bit word on purpose: H2 above must never hold a __nv_bfloat162, because a
// variant member with a non-trivial copy constructor deletes the union's copy
// constructor and with it every H2 the decoders pass around.
template<bool BF16>
__device__ __forceinline__ unsigned splat16(unsigned short bits) {
  return ((unsigned)bits << 16) | bits;
}
template<bool BF16>
__device__ __forceinline__ unsigned mul16x2(unsigned x, unsigned s) {
  if constexpr (BF16) {
    const __nv_bfloat162 v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&x),
                                     *reinterpret_cast<const __nv_bfloat162*>(&s));
    return *reinterpret_cast<const unsigned*>(&v);
  }
  const __half2 v = __hmul2(*reinterpret_cast<const __half2*>(&x),
                            *reinterpret_cast<const __half2*>(&s));
  return *reinterpret_cast<const unsigned*>(&v);
}
template<bool BF16>
__device__ __forceinline__ unsigned pack16x2(float lo, float hi) {
  if constexpr (BF16) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&v);
  }
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&v);
}
template<bool BF16>
__device__ __forceinline__ float2 unpack16x2(unsigned x) {
  if constexpr (BF16) return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&x));
  return __half22float2(*reinterpret_cast<const __half2*>(&x));
}
// The group scale as one 16-bit value of the tile's format, replicated below.
template<bool BF16>
__device__ __forceinline__ unsigned short scale_bits16(unsigned sbyte) {
  return (unsigned short)(BF16 ? plmn_ue8m0_to_bf16_bits(sbyte)
                               : plmn_ue8m0_to_half_bits(sbyte));
}

__device__ __forceinline__ unsigned sa(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldm_x4(const __half* p, unsigned& r0, unsigned& r1, unsigned& r2,
                                       unsigned& r3) {
  unsigned addr = sa(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void ldm_x4t(const __half* p, unsigned& r0, unsigned& r1, unsigned& r2,
                                        unsigned& r3) {
  unsigned addr = sa(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// D += A * B for m16n8k16 row.col with fp32 accumulation. Scalar operands keep
// every fragment in registers; no local-memory addressing is possible here.
__device__ __forceinline__ void mma16816(float* d, unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm volatile("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

struct Smem {
  __half* q;
  __half* k;
  __half* p;
  float* red;
  int2* meta;        // per selected token: (data offset | src<<31, scale offset)
};

__device__ __forceinline__ Smem smem_of(char* base, int listn) {
  Smem s;
  s.q = reinterpret_cast<__half*>(base);
  s.k = s.q + HB * KS;
  s.p = s.k + NC * KS;
  s.red = reinterpret_cast<float*>(s.p + HB * PS);
  s.meta = reinterpret_cast<int2*>(s.red + NWN * HB);
  return s;
}

// Ordered block-wide exclusive prefix sum; every thread also receives the total.
__device__ __forceinline__ int block_scan(int local, int* scratch, int* total) {
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  int v = local;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    int n = __shfl_up_sync(0xffffffffu, v, d);
    if (lane >= d) v += n;
  }
  if (lane == 31) scratch[w] = v;
  __syncthreads();
  if (w == 0) {
    int x = (lane < NW) ? scratch[lane] : 0;
#pragma unroll
    for (int d = 1; d < NW; d <<= 1) {
      int n = __shfl_up_sync(0xffffffffu, x, d);
      if (lane >= d) x += n;
    }
    if (lane < NW) scratch[lane] = x;
  }
  __syncthreads();
  int before = (w > 0) ? scratch[w - 1] : 0;
  *total = scratch[NW - 1];
  return v - local + before;
}

// Hardware FP8 decode preserves zeros/subnormals as well as normal payloads.
// Input pairs are adjacent bytes, unlike the legacy bit-spreading approximation.
// SM100 has a native bf16 form of the same instruction, so the two formats cost the
// same single F2FP per pair.
template<bool BF16>
__device__ __forceinline__ void e4m3x4_to_tile4(unsigned w, unsigned& lo, unsigned& hi) {
 const unsigned short a=static_cast<unsigned short>(w), b=static_cast<unsigned short>(w>>16);
 if constexpr (BF16) {
  asm("cvt.rn.bf16x2.e4m3x2 %0,%1;" : "=r"(lo) : "h"(a));
  asm("cvt.rn.bf16x2.e4m3x2 %0,%1;" : "=r"(hi) : "h"(b));
 } else {
  asm("cvt.rn.f16x2.e4m3x2 %0,%1;" : "=r"(lo) : "h"(a));
  asm("cvt.rn.f16x2.e4m3x2 %0,%1;" : "=r"(hi) : "h"(b));
 }
}

// Exact 2^(s-127) as fp32 for every ue8m0 byte.
__device__ __forceinline__ float ue8m0_to_float(unsigned sbyte) {
  return __uint_as_float((sbyte == 0u) ? 0x00400000u : (sbyte << 23));
}

// 32 e4m3 bytes + one ue8m0 scale -> 32 fp16. The fast path multiplies packed
// half2 by the exact fp16 scale (representable for s in [103,142], which covers
// the measured in-domain 105..119); a rare out-of-range scale byte falls back to
// an exact fp32 multiply.
// N is BLKS (MXFP8) or 2*BLKS (NVFP4), i.e. 2, 4 or 8 for every supported
// (NT, NC) config. The caller's offset is a multiple of N inside the record's
// 16 B / 32 B scale area, so the load is naturally aligned and reads exactly the
// bytes this slice needs - never past the record end, which a fixed 8-byte load
// would for MXFP8 (16 B scale area, offsets 0/4/8/12). N is a template parameter
// so each instantiation emits exactly one load of exactly one width.
template <int N>
__device__ __forceinline__ uint64_t load_scale_run(const unsigned char* p) {
  static_assert(N == 2 || N == 4 || N == 8, "scale-run width");
  if (N == 8) return *reinterpret_cast<const uint64_t*>(p);
  if (N == 4) return *reinterpret_cast<const uint32_t*>(p);
  return *reinterpret_cast<const uint16_t*>(p);
}

template<bool BF16>
__device__ __forceinline__ void decode_mxfp8_words(uint4 w0, uint4 w1, unsigned sbyte,
                                                   typename Tile16<BF16>::elem* dst) {
  const unsigned wd[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
  // In-format fast path: the scale is an exact power of two in both formats for the
  // measured in-domain bytes (fp16 for s in [103,142], bf16 for s in [1,254], and
  // the frozen packer emits 105..119), so the packed multiply is exact. A byte
  // outside that range falls back to an exact fp32 multiply.
  const bool fast = BF16 ? (sbyte >= 1u && sbyte <= 254u) : (sbyte >= 103u && sbyte <= 142u);
  if (fast) {
    const unsigned sc = splat16<BF16>(scale_bits16<BF16>(sbyte));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      unsigned pk[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        unsigned lo, hi;
        e4m3x4_to_tile4<BF16>(wd[2 * j + h], lo, hi);
        pk[2 * h] = mul16x2<BF16>(lo, sc);
        pk[2 * h + 1] = mul16x2<BF16>(hi, sc);
      }
      reinterpret_cast<uint4*>(dst)[j] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
  } else {
    const float scale = ue8m0_to_float(sbyte);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      unsigned pk[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned b = wd[2 * j + h];
        unsigned lo, hi;
        e4m3x4_to_tile4<BF16>(b, lo, hi);
        const float2 f01 = unpack16x2<BF16>(lo), f23 = unpack16x2<BF16>(hi);
        pk[2 * h] = pack16x2<BF16>(f01.x * scale, f01.y * scale);
        pk[2 * h + 1] = pack16x2<BF16>(f23.x * scale, f23.y * scale);
      }
      reinterpret_cast<uint4*>(dst)[j] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
  }
}

template<bool BF16>
__device__ __forceinline__ void decode_mxfp8_32(const unsigned char* data, unsigned scale,
                                                typename Tile16<BF16>::elem* dst) {
 decode_mxfp8_words<BF16>(*(const uint4*)data,*(const uint4*)(data+16),scale,dst);
}

// Exact hardware FP4 conversion on SM103, including signed zeros: eight nibble
// pairs, i.e. one whole 32-bit register, in one go. Unpacking the register into
// its four bytes once removes the shift/mask the byte-at-a-time form needed per
// pair, which is a third of this decoder's instructions - and the decoder is the
// hottest straight-line code in the KV path (8 of every 10 chunks are NVFP4).
template<bool BF16>
__device__ __forceinline__ void e2m1x8_to_tile8(unsigned w, unsigned* pk) {
  if constexpr (BF16) {
  asm("{.reg .b8 b0,b1,b2,b3;\n\t"
      " mov.b32 {b0,b1,b2,b3},%4;\n\t"
      " cvt.rn.bf16x2.e2m1x2 %0,b0;\n\t"
      " cvt.rn.bf16x2.e2m1x2 %1,b1;\n\t"
      " cvt.rn.bf16x2.e2m1x2 %2,b2;\n\t"
      " cvt.rn.bf16x2.e2m1x2 %3,b3;}"
      : "=r"(pk[0]), "=r"(pk[1]), "=r"(pk[2]), "=r"(pk[3])
      : "r"(w));
  } else {
  asm("{.reg .b8 b0,b1,b2,b3;\n\t"
      " mov.b32 {b0,b1,b2,b3},%4;\n\t"
      " cvt.rn.f16x2.e2m1x2 %0,b0;\n\t"
      " cvt.rn.f16x2.e2m1x2 %1,b1;\n\t"
      " cvt.rn.f16x2.e2m1x2 %2,b2;\n\t"
      " cvt.rn.f16x2.e2m1x2 %3,b3;}"
      : "=r"(pk[0]), "=r"(pk[1]), "=r"(pk[2]), "=r"(pk[3])
      : "r"(w));
  }
}

// 32 e2m1 nibbles (16 packed bytes) + 2 e4m3 group scales -> 32 tile elements.
template<bool BF16>
__device__ __forceinline__ void decode_nvfp4_words(uint4 w, unsigned s0, unsigned s1,
                                                   typename Tile16<BF16>::elem* dst) {
  const unsigned wd[4] = {w.x, w.y, w.z, w.w};
  unsigned packed_scale;
  const unsigned short scale_bytes=static_cast<unsigned short>(s0 | (s1<<8));
  if constexpr (BF16) asm("cvt.rn.bf16x2.e4m3x2 %0,%1;" : "=r"(packed_scale) : "h"(scale_bytes));
  else asm("cvt.rn.f16x2.e4m3x2 %0,%1;" : "=r"(packed_scale) : "h"(scale_bytes));
  const unsigned sc0 = splat16<BF16>((unsigned short)(packed_scale & 0xffffu));
  const unsigned sc1 = splat16<BF16>((unsigned short)(packed_scale >> 16));
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const unsigned sc = (j < 2) ? sc0 : sc1;
    unsigned pk[4];
    e2m1x8_to_tile8<BF16>(wd[j], pk);
#pragma unroll
    for (int h = 0; h < 4; ++h)
      pk[h] = mul16x2<BF16>(pk[h], sc);
    reinterpret_cast<uint4*>(dst)[j] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  }
}

template<bool BF16>
__device__ __forceinline__ void decode_nvfp4_32(const unsigned char* data, unsigned s0, unsigned s1,
                                                typename Tile16<BF16>::elem* dst) {
 decode_nvfp4_words<BF16>(*(const uint4*)data,s0,s1,dst);
}

__global__ void __launch_bounds__(NT, MINCTA)
PLMN_KERNEL(const __nv_bfloat16* __restrict__ q, const unsigned char* __restrict__ k1,
            const int* __restrict__ idx1, const unsigned char* __restrict__ k2,
            const int* __restrict__ idx2, const float* __restrict__ sink, float sm_scale,
            __nv_bfloat16* __restrict__ out, float* __restrict__ lse, __half* __restrict__ part_o,
            float* __restrict__ part_ml, int H, int BS, int budget1, int budget2, int page1,
            int rec1, int page2, int rec2, int listn, int splits) {
  extern __shared__ __align__(16) char smem_raw[];
  Smem sm = smem_of(smem_raw, listn);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int w = tid >> 5;
  const int wm = w / NWN;   // head tile (0..1)
  const int wn = w % NWN;   // token tile for QK, output-dim tile for PV (0..3)
  const int g = lane >> 2;  // fragment row inside the 16-row mma tile
  const int tg = lane & 3;  // fragment column pair

  const int bs = blockIdx.y;          // b*S + s
  const int hbase = blockIdx.x * HB;  // first physical head of this CTA
  const int sp = blockIdx.z;          // token-split index
  const int row0 = wm * 16 + g;
  const int row1 = row0 + 8;
  const bool fp4_1 = (rec1 == 288);
  const bool fp4_2 = (rec2 == 288);

  int n1 = 0, n2 = 0;
  {
    int per1 = (budget1 + NT - 1) / NT;
    if (per1 > MAXENT) per1 = MAXENT;
    int per2 = (budget2 + NT - 1) / NT;
    if (per2 > MAXENT) per2 = MAXENT;
    int sl1[MAXENT], sl2[MAXENT];
    int c1 = 0, c2 = 0;
#pragma unroll
    for (int i = 0; i < MAXENT; ++i) {
      sl1[i] = -1;
      sl2[i] = -1;
      if (i < per1) {
        const int e = tid + i * NT;
        if (e < budget1) {
          const int v = __ldg(idx1 + (long)bs * budget1 + e);
          sl1[i] = v;
          c1 += (v >= 0);
        }
      }
      if (k2 != nullptr && i < per2) {
        const int e = tid + i * NT;
        if (e < budget2) {
          const int v = __ldg(idx2 + (long)bs * budget2 + e);
          sl2[i] = v;
          c2 += (v >= 0);
        }
      }
    }
    // SEL_HOIST_BEGIN - the two scans and their writes, in the original order and
    // with the original barrier between them; only the loads above moved.
    const int excl1 = block_scan(c1, reinterpret_cast<int*>(sm.red), &n1);
    int pos = excl1;
#pragma unroll
    for (int i = 0; i < MAXENT; ++i) {
      if (sl1[i] >= 0) sm.meta[pos++].x = sl1[i];
    }
    __syncthreads();
    const int excl2 = block_scan(c2, reinterpret_cast<int*>(sm.red), &n2);
    pos = excl2;
#pragma unroll
    for (int i = 0; i < MAXENT; ++i) {
      if (sl2[i] >= 0) sm.meta[n1 + pos++].x = sl2[i];
    }
    // SEL_HOIST_END
  }
  const int N = n1 + n2;
  __syncthreads();

  // ---- slots -> (source, data offset, scale offset) exactly as read_rows ----
  {
    const int w1 = fp4_1 ? 256 : 512;
    const int ns1 = fp4_1 ? 32 : 16;
    const int w2 = fp4_2 ? 256 : 512;
    const int ns2 = fp4_2 ? 32 : 16;
    for (int i = tid; i < N; i += NT) {
      const int slot = sm.meta[i].x;
      const bool second = (i >= n1);
      const int page = second ? page2 : page1;
      const int rec = second ? rec2 : rec1;
      const int wid = second ? w2 : w1;
      const int ns = second ? ns2 : ns1;
      const int p = slot / page;
      const int r = slot - p * page;
      const int rowbase = p * page * rec;
      sm.meta[i] = make_int2((int)((unsigned)(rowbase + r * wid) | ((unsigned)second << 31)),
                             rowbase + page * wid + r * ns);
    }
  }
  __syncthreads();

  // ---- q for this head block: bf16 -> exact fp16 ----
  {
    const __nv_bfloat16* qbase = q + (((long)bs * H) + hbase) * D;
    for (int idx = tid; idx < HB * (D / 8); idx += NT) {
      const int r = idx >> 6;
      const int dim = (idx & 63) << 3;
      const uint4 raw = *reinterpret_cast<const uint4*>(qbase + r * D + dim);
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
      uint4 packed;
      H2* h2 = reinterpret_cast<H2*>(&packed);
#pragma unroll
      for (int t = 0; t < 4; ++t) h2[t].h = __float22half2_rn(__bfloat1622float2(b2[t]));
      *reinterpret_cast<uint4*>(sm.q + r * KS + dim) = packed;
    }
  }
  __syncthreads();

  const int nchunk = (N + NC - 1) / NC;
  const int per_split = (nchunk + splits - 1) / splits;
  const int c_first = sp * per_split;
  const int c_last = min(nchunk, c_first + per_split);
  // Row-major partials: the `splits` contributions to one (request row, head) are
  // adjacent, so the combine reads one contiguous run per head instead of `splits`
  // runs strided by the whole partial buffer.
  const long pbase = ((long)bs * H + hbase) * splits + sp;

  // No token for this request (or none in this split): publish the identity
  // element of the split combine, or the reference's zero/+Inf answer directly.
  if (c_first >= c_last) {
    if (splits == 1) {
      __nv_bfloat16* obase = out + (((long)bs * H) + hbase) * D;
      for (int r = 0; r < HB; ++r)
        for (int i = tid; i < D / 8; i += NT)
          reinterpret_cast<uint4*>(obase + r * D)[i] = make_uint4(0u, 0u, 0u, 0u);
      if (tid < HB) lse[(long)bs * H + hbase + tid] = pinf();
    } else {
      if (wn == 0 && tg == 0) {
        part_ml[(pbase + (long)row0 * splits) * 2] = ninf();
        part_ml[(pbase + (long)row0 * splits) * 2 + 1] = 0.f;
        part_ml[(pbase + (long)row1 * splits) * 2] = ninf();
        part_ml[(pbase + (long)row1 * splits) * 2 + 1] = 0.f;
      }
      // Store the zero accumulator as well. A split with no tokens contributes
      // nothing whatever its stored value, so this changes no result; it does make
      // every row of part_o defined, which lets the combine weight a split by its
      // -Inf row maximum instead of branching around it - a branch there would
      // hold each partial load until its own weight had arrived.
#pragma unroll
      for (int dp = 0; dp < 16; ++dp) {
        const int dim = wn * 128 + dp * 8 + 2 * tg;
        *reinterpret_cast<uint32_t*>(part_o + (pbase + (long)row0 * splits) * D + dim) = 0u;
        *reinterpret_cast<uint32_t*>(part_o + (pbase + (long)row1 * splits) * D + dim) = 0u;
      }
    }
    return;
  }

  float m2[2] = {ninf(), ninf()};
  // Per-warp partial row sums. Rescaling them with the same alpha the
  // accumulator uses keeps l_final = sum_wn lpart_wn * 2^(m_c - m_final) exact,
  // so the loop needs no cross-warp sum exchange at all.
  float lpart[2] = {0.f, 0.f};
  float acc[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  const float qk_scale = sm_scale * LOG2E;
  // Dequant mapping: the LANE selects the token row and tid>>6 selects the
  // 128-dim slice, so one metadata load still feeds four 32-value blocks while
  // every 16 B store phase (8 lanes) touches 8 consecutive rows at one unit, so
  // 4*(row+unit) mod 32 is distinct inside the phase and the store is
  // conflict-free. The previous mapping (row = tid>>2) placed the four slice
  // lanes of a row 64 words apart, which is 0 mod 32, and cost 3.04 M extra
  // shared store wavefronts on case 0 (NCU: 4.19 M measured vs 1.19 M ideal).
  const int drow = tid & (NC - 1);
  const int bg = tid / NC;

  for (int ci = c_first; ci < c_last; ++ci) {
    const int c0 = ci * NC;
    const int nc = min(NC, N - c0);

    // -- dequantise NC token rows into fp16 shared memory --
    {
      __half* dst = sm.k + drow * KS + bg * SLICE;
      if (drow < nc) {
        const int2 meta = sm.meta[c0 + drow];
        const bool second = (meta.x < 0);
        const unsigned char* base = second ? k2 : k1;
        const bool fp4 = second ? fp4_2 : fp4_1;
        const unsigned char* data =
            base + (meta.x & 0x7fffffff) + bg * (fp4 ? (SLICE / 2) : SLICE);
        const unsigned char* scl = base + meta.y + bg * (fp4 ? (SLICE / 16) : (SLICE / 32));
        // ONE naturally aligned load for this slice's whole group-scale run
        // (BLKS bytes for MXFP8, 2*BLKS for NVFP4, both inside the record's
        // 16 B / 32 B scale area) instead of BLKS separate byte loads. Rows are
        // 528 B / 288 B apart, so every one of those byte loads touched its own
        // 32 B sector per lane: 4 instructions x 32 sectors per warp-chunk
        // become 1 x 32, which matters because L1/TEX is the busiest pipe in
        // this kernel (NCU on case 0: 66 % throughput).
        uint64_t sc;
        if (fp4) {
          sc = load_scale_run<2 * BLKS>(scl);
        } else {
          sc = load_scale_run<BLKS>(scl);
        }
#pragma unroll
        for (int i = 0; i < BLKS; ++i) {
          if (fp4) {
            decode_nvfp4_32<false>(data + i * 16, (unsigned)(sc >> (16 * i)) & 0xffu,
                            (unsigned)(sc >> (16 * i + 8)) & 0xffu, dst + i * 32);
          } else {
            decode_mxfp8_32<false>(data + i * 32, (unsigned)(sc >> (8 * i)) & 0xffu, dst + i * 32);
          }
        }
      } else {
        // Pad the tail so ldmatrix never reads uninitialised shared memory;
        // those columns are masked to -Inf anyway and get P == 0.
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int i = 0; i < SLICE / 8; ++i) reinterpret_cast<uint4*>(dst)[i] = z;
      }
    }
    __syncthreads();

    // -- QK^T: S[HB][NC] --
    float c[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int u = 0; u < 4; ++u) c[j][u] = 0.f;
    {
      const __half* qtile = sm.q + wm * 16 * KS;
      const __half* ktile = sm.k + wn * 16 * KS;
      const int btoken = (lane & 7) + ((lane >> 4) << 3);
      const int bcol = ((lane >> 3) & 1) << 3;
      const int arow = lane & 15;
      const int acol = (lane >> 4) << 3;
#pragma unroll 2
      for (int k0 = 0; k0 < D; k0 += 16) {
        unsigned a[4], b[4];
        ldm_x4(qtile + arow * KS + k0 + acol, a[0], a[1], a[2], a[3]);
        ldm_x4(ktile + btoken * KS + k0 + bcol, b[0], b[1], b[2], b[3]);
        mma16816(c[0], a[0], a[1], a[2], a[3], b[0], b[1]);
        mma16816(c[1], a[0], a[1], a[2], a[3], b[2], b[3]);
      }
    }

    // -- base-2 logits, invalid token columns masked to -Inf --
    float t[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int col = c0 + wn * 16 + j * 8 + 2 * tg;
#pragma unroll
      for (int u = 0; u < 4; ++u) t[j][u] = c[j][u] * qk_scale;
      if (col >= N) {
        t[j][0] = ninf();
        t[j][2] = ninf();
      }
      if (col + 1 >= N) {
        t[j][1] = ninf();
        t[j][3] = ninf();
      }
    }

    // -- running row maximum across the four token warps --
    float mx0 = fmaxf(fmaxf(t[0][0], t[0][1]), fmaxf(t[1][0], t[1][1]));
    float mx1 = fmaxf(fmaxf(t[0][2], t[0][3]), fmaxf(t[1][2], t[1][3]));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    if (tg == 0) {
      sm.red[wn * HB + row0] = mx0;
      sm.red[wn * HB + row1] = mx1;
    }
    __syncthreads();
    float mc0 = sm.red[row0], mc1 = sm.red[row1];
#pragma unroll
    for (int u = 1; u < NWN; ++u) {
      mc0 = fmaxf(mc0, sm.red[u * HB + row0]);
      mc1 = fmaxf(mc1, sm.red[u * HB + row1]);
    }
    const float mn0 = fmaxf(m2[0], mc0);
    const float mn1 = fmaxf(m2[1], mc1);
    const float al0 = ex2(m2[0] - mn0);
    const float al1 = ex2(m2[1] - mn1);

    // -- P = 2^(T - m) and this warp's partial row sums --
    float p[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      p[j][0] = ex2(t[j][0] - mn0);
      p[j][1] = ex2(t[j][1] - mn0);
      p[j][2] = ex2(t[j][2] - mn1);
      p[j][3] = ex2(t[j][3] - mn1);
    }
    float s0 = (p[0][0] + p[0][1]) + (p[1][0] + p[1][1]);
    float s1 = (p[0][2] + p[0][3]) + (p[1][2] + p[1][3]);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
    lpart[0] = lpart[0] * al0 + s0;
    lpart[1] = lpart[1] * al1 + s1;
    m2[0] = mn0;
    m2[1] = mn1;

    // -- rescale the running PV accumulator; skipped exactly when m did not move --
    if (al0 != 1.f) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        acc[i][0] *= al0;
        acc[i][1] *= al0;
      }
    }
    if (al1 != 1.f) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        acc[i][2] *= al1;
        acc[i][3] *= al1;
      }
    }

    // -- publish P as fp16 in the mma A-operand layout --
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int col = wn * 16 + j * 8 + 2 * tg;
      H2 h0, h1;
      h0.h = __floats2half2_rn(p[j][0], p[j][1]);
      h1.h = __floats2half2_rn(p[j][2], p[j][3]);
      *reinterpret_cast<uint32_t*>(sm.p + row0 * PS + col) = h0.u;
      *reinterpret_cast<uint32_t*>(sm.p + row1 * PS + col) = h1.u;
    }
    __syncthreads();

    // -- PV: O[HB][512] += P[HB][NC] * K[NC][512] --
    {
      const __half* ptile = sm.p + wm * 16 * PS;
      const int arow = lane & 15;
      const int acol = (lane >> 4) << 3;
      const int btok = lane & 15;
      const int bcol = (lane >> 4) << 3;
#pragma unroll
      for (int kk = 0; kk < NC; kk += 16) {
        unsigned a[4];
        ldm_x4(ptile + arow * PS + kk + acol, a[0], a[1], a[2], a[3]);
#pragma unroll
        for (int dp = 0; dp < 8; ++dp) {
          unsigned b[4];
          ldm_x4t(sm.k + (kk + btok) * KS + wn * 128 + dp * 16 + bcol, b[0], b[1], b[2], b[3]);
          mma16816(acc[dp * 2], a[0], a[1], a[2], a[3], b[0], b[1]);
          mma16816(acc[dp * 2 + 1], a[0], a[1], a[2], a[3], b[2], b[3]);
        }
      }
    }
    __syncthreads();
  }

  // ---- epilogue: cross-warp row sums, then sink normalisation or a split partial
  if (tg == 0) {
    sm.red[wn * HB + row0] = lpart[0];
    sm.red[wn * HB + row1] = lpart[1];
  }
  __syncthreads();
  float lacc[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int row = (u == 0) ? row0 : row1;
    float sum = sm.red[row];
#pragma unroll
    for (int q = 1; q < NWN; ++q) sum += sm.red[q * HB + row];
    lacc[u] = sum;
  }

  if (splits > 1) {
    if (wn == 0 && tg == 0) {
      part_ml[(pbase + (long)row0 * splits) * 2] = m2[0];
      part_ml[(pbase + (long)row0 * splits) * 2 + 1] = lacc[0];
      part_ml[(pbase + (long)row1 * splits) * 2] = m2[1];
      part_ml[(pbase + (long)row1 * splits) * 2 + 1] = lacc[1];
    }
    // Only a split that actually saw tokens publishes an accumulator; the
    // combine skips the rest on its -Inf row maximum.
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      const int dim = wn * 128 + dp * 8 + 2 * tg;
      H2 v0, v1;
      v0.h = __floats2half2_rn(acc[dp][0], acc[dp][1]);
      v1.h = __floats2half2_rn(acc[dp][2], acc[dp][3]);
      *reinterpret_cast<uint32_t*>(part_o + (pbase + (long)row0 * splits) * D + dim) = v0.u;
      *reinterpret_cast<uint32_t*>(part_o + (pbase + (long)row1 * splits) * D + dim) = v1.u;
    }
    return;
  }

  float f[2], lsen[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const float lse2 = m2[u] + log2f(lacc[u]);
    const float sk2 = __ldg(sink + hbase + ((u == 0) ? row0 : row1)) * LOG2E;
    const float lae2 = fmaxf(lse2, sk2) + log2f(1.f + ex2(-fabsf(lse2 - sk2)));
    f[u] = ex2(m2[u] - lae2);
    lsen[u] = lse2 * LN2;
  }
  if (wn == 0 && tg == 0) {
    lse[(long)bs * H + hbase + row0] = lsen[0];
    lse[(long)bs * H + hbase + row1] = lsen[1];
  }
  __nv_bfloat16* obase = out + (((long)bs * H) + hbase) * D;
#pragma unroll
  for (int dp = 0; dp < 16; ++dp) {
    const int dim = wn * 128 + dp * 8 + 2 * tg;
    const __nv_bfloat162 v0 = __floats2bfloat162_rn(acc[dp][0] * f[0], acc[dp][1] * f[0]);
    const __nv_bfloat162 v1 = __floats2bfloat162_rn(acc[dp][2] * f[1], acc[dp][3] * f[1]);
    *reinterpret_cast<__nv_bfloat162*>(obase + row0 * D + dim) = v0;
    *reinterpret_cast<__nv_bfloat162*>(obase + row1 * D + dim) = v1;
  }
}

constexpr int CH = 4;        // heads merged per combine CTA (one warp each)
constexpr int CT = 32 * CH;  // combine threads

// o[0..15] += fs * one 16-dim slice of a split partial, arriving in two 16 B loads.
__device__ __forceinline__ void comb_acc16(float* o, uint4 r0, uint4 r1, float fs) {
  const H2* a = reinterpret_cast<const H2*>(&r0);
  const H2* b = reinterpret_cast<const H2*>(&r1);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 x = __half22float2(a[i].h);
    const float2 y = __half22float2(b[i].h);
    o[2 * i] += fs * x.x;
    o[2 * i + 1] += fs * x.y;
    o[8 + 2 * i] += fs * y.x;
    o[8 + 2 * i + 1] += fs * y.y;
  }
}

__global__ void __launch_bounds__(CT)
PLMN_COMB_WIDE(const __half* __restrict__ part_o, const float* __restrict__ part_ml,
             const float* __restrict__ sink, __nv_bfloat16* __restrict__ out,
             float* __restrict__ lse, int splits, int H, int BS) {
  const int tid = threadIdx.x;
  const int hg = tid >> 5;   // head slot inside this CTA
  const int lane = tid & 31;
  const int total = BS * H;
  const int rh = blockIdx.x * CH + hg;  // (request row) * H + head
  // One warp owns one head, so the tail warps of the last CTA leave as a whole:
  // every reduction below is warp-internal and needs no in-bounds stand-in.
  if (rh >= total) return;
  const int h = rh - (rh / H) * H;
  const long mlstride = 2;
  const float* ml = part_ml + (long)rh * splits * 2;

  // ---- per-split statistics: lane s owns split s ----
  float m = ninf(), l = 0.f;
  if (lane < splits) {
    m = __ldg(ml + (long)lane * mlstride);
    l = __ldg(ml + (long)lane * mlstride + 1);
  }
  float mmax = m;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) mmax = fmaxf(mmax, __shfl_xor_sync(0xffffffffu, mmax, d));
  const float f = (lane < splits && m != ninf()) ? ex2(m - mmax) : 0.f;
  float lsum = f * l;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, d);
  const float lse2 = mmax + log2f(lsum);
  const float sk2 = __ldg(sink + h) * LOG2E;
  const float lae2 = fmaxf(lse2, sk2) + log2f(1.f + ex2(-fabsf(lse2 - sk2)));
  // No split saw a token: the reference answer is a zero output and +Inf lse.
  const float scale = (mmax == ninf()) ? 0.f : ex2(mmax - lae2);
  if (lane == 0) lse[rh] = (mmax == ninf()) ? pinf() : lse2 * LN2;

  const long ostride = D;
  const __half* po = part_o + (long)rh * splits * D + lane * 16;
  float o[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i] = 0.f;
  // Each partial element is read exactly once, so stream it: evict-first keeps a
  // whole row of split partials from displacing anything else. The per-split
  // weight stays in the lane that owns it and is broadcast by shuffle - no shared
  // memory round trip - and every split is loaded and weighted rather than
  // skipped when its weight is zero, because a skip makes each load wait for its
  // own weight. A zero-weight split's partial is the zeros its producer stored, so
  // the product is exactly the nothing the skipped iteration would have added.
  // Four splits are loaded before any of them is consumed: one load-accumulate pair
  // per iteration keeps only two 16 B requests per lane in flight, and this combine
  // then reads its partials at ~1.6 TB/s whatever the grid.
  int s = 0;
  for (; s + 4 <= splits; s += 4) {
    uint4 v[8];
    float w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      v[2 * k] = __ldcs(reinterpret_cast<const uint4*>(po + (long)(s + k) * ostride));
      v[2 * k + 1] = __ldcs(reinterpret_cast<const uint4*>(po + (long)(s + k) * ostride + 8));
      w[k] = __shfl_sync(0xffffffffu, f, s + k);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) comb_acc16(o, v[2 * k], v[2 * k + 1], w[k]);
  }
  for (; s < splits; ++s) {
    comb_acc16(o, __ldcs(reinterpret_cast<const uint4*>(po + (long)s * ostride)),
               __ldcs(reinterpret_cast<const uint4*>(po + (long)s * ostride + 8)),
               __shfl_sync(0xffffffffu, f, s));
  }
  uint4 w0, w1;
  __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(&w0);
  __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(&w1);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p0[i] = __floats2bfloat162_rn(o[2 * i] * scale, o[2 * i + 1] * scale);
    p1[i] = __floats2bfloat162_rn(o[8 + 2 * i] * scale, o[8 + 2 * i + 1] * scale);
  }
  __nv_bfloat16* obase = out + (long)rh * D + lane * 16;
  *reinterpret_cast<uint4*>(obase) = w0;
  *reinterpret_cast<uint4*>(obase + 8) = w1;
}

__global__ void __launch_bounds__(128)
PLMN_COMB_NARROW(const __half* __restrict__ part_o, const float* __restrict__ part_ml,
             const float* __restrict__ sink, __nv_bfloat16* __restrict__ out,
             float* __restrict__ lse, int splits, int H, int BS) {
  const int rh = blockIdx.x;
  const int row = rh / H;
  const int h = rh - row * H;
  const int tid = threadIdx.x;
  const long mlstride = 2;
  const float* ml = part_ml + ((long)row * H + h) * splits * 2;

  float mmax = ninf();
  for (int s = 0; s < splits; ++s) mmax = fmaxf(mmax, ml[s * mlstride]);
  __nv_bfloat16* obase = out + ((long)row * H + h) * D;
  if (mmax == ninf()) {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (int i = tid; i < D / 8; i += 128) reinterpret_cast<uint4*>(obase)[i] = z;
    if (tid == 0) lse[(long)row * H + h] = pinf();
    return;
  }
  float lsum = 0.f;
  for (int s = 0; s < splits; ++s) {
    const float m = ml[s * mlstride];
    if (m != ninf()) lsum += ex2(m - mmax) * ml[s * mlstride + 1];
  }
  const float lse2 = mmax + log2f(lsum);
  const float sk2 = __ldg(sink + h) * LOG2E;
  const float lae2 = fmaxf(lse2, sk2) + log2f(1.f + ex2(-fabsf(lse2 - sk2)));
  const float scale = ex2(mmax - lae2);
  if (tid == 0) lse[(long)row * H + h] = lse2 * LN2;

  const int d0 = tid * 4;
  const long ostride = D;
  const __half* po = part_o + ((long)row * H + h) * splits * D + d0;
  float o[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < splits; ++s) {
    const float m = ml[s * mlstride];
    if (m == ninf()) continue;
    const float f = ex2(m - mmax);
    const uint2 raw = __ldcs(reinterpret_cast<const uint2*>(po + s * ostride));
    H2 a, b;
    a.u = raw.x;
    b.u = raw.y;
    const float2 fa = __half22float2(a.h);
    const float2 fb = __half22float2(b.h);
    o[0] += f * fa.x;
    o[1] += f * fa.y;
    o[2] += f * fb.x;
    o[3] += f * fb.y;
  }
  const __nv_bfloat162 v0 = __floats2bfloat162_rn(o[0] * scale, o[1] * scale);
  const __nv_bfloat162 v1 = __floats2bfloat162_rn(o[2] * scale, o[3] * scale);
  *reinterpret_cast<uint2*>(obase + d0) = make_uint2(
      *reinterpret_cast<const unsigned*>(&v0), *reinterpret_cast<const unsigned*>(&v1));
}

int smem_bytes(int listn) { return SMEM_FIXED + 8 * listn; }

}  // namespace

// ---------------------------------------------------------------------------
// Host side of THIS configuration. It lives inside the configuration namespace
// because the split rule, the shared-memory footprint and the launch geometry
// are all derived from HB/NT. plmn_forward() picks one configuration per call,
// in C++, from declared shape metadata alone - so the two geometries coexist in
// a single extension and the choice costs no Python and no per-call work beyond
// one integer comparison.
//
// The rule and the launch below are the shipped V1.10.1 host code verbatim; the
// only change is where the shape integers come from (the shared plmn::FwdCtx
// instead of local variables), bound to the same names in the first line of each
// function so that experiments/variants.py's textual anchors keep matching.
// ---------------------------------------------------------------------------

constexpr int CFG_D = D;
constexpr int CFG_HB = HB;
constexpr int CFG_NT = NT;
constexpr int CFG_MAXENT = MAXENT;
constexpr int CFG_NC = NC;

int cfg_smem_bytes(int listn) { return smem_bytes(listn); }

// What this geometry's launch schedule looks like for one shape. Everything here
// is derived from declared metadata, so plmn_forward can compare two geometries
// without launching either.
struct Schedule {
  int splits;   // token-split fan-out
  int grid;     // CTAs launched
  int sms;      // SMs that ever receive a CTA
  int depth;    // chunk-passes on the deepest SM = rounds * chunks per CTA
};

int resident_ctas(const plmn::FwdCtx& c) {
  const int smem = smem_bytes(c.listn);
  const int spsm = (c.smem_per_sm > 0) ? c.smem_per_sm : 2 * smem;
  return std::max(1, std::min(MINCTA, spsm / std::max(1, smem)));
}

int plan_splits(const plmn::FwdCtx& c) {
  const int H = c.H, BS = c.BS, listn = c.listn, nsm = c.nsm;
  // Chunk count from the ESTIMATED selectable tokens, not from the selector
  // width: a short context fills only the first few of its 640 slots and the
  // rest are -1, so width-based planning hands out splits that only publish
  // identity partials. Over-estimating can only over-split, which is what the
  // width did anyway; under-estimating is impossible (see plmn_forward).
  const int maxchunk = (c.nest + NC - 1) / NC;
  const int smem_per_sm = (c.smem_per_sm > 0) ? c.smem_per_sm : 2 * smem_bytes(listn);

  // Adaptive token split: one CTA per (head block, request row) leaves most SMs
  // idle when B*S*H/HB is small, and the per-request chain of NC-token chunks is
  // then pure latency. Splitting the chunk range adds CTAs without duplicating
  // any dequantisation; the partials are combined afterwards.
  const int items = BS * (H / HB);
  const int smem = smem_bytes(listn);
  // How many CTAs of this shape are resident at once: the register budget allows
  // MINCTA, the shared-memory budget allows smem_per_sm/smem, and the truth is
  // the smaller of the two (a wide selector list pushes this to 1).
  const int cpa = std::max(1, std::min(MINCTA, smem_per_sm / std::max(1, smem)));

  int splits = 1;
  if (items < nsm) {
    splits = (nsm + items - 1) / items;
    splits = std::min(splits, std::min(maxchunk, MAXSPLIT));
  } else {

    constexpr int MINCHUNK = 3;
    const int smax = std::min(maxchunk, MAXSPLIT);
    long best = -1;
    for (int sp = 1; sp <= smax; ++sp) {
      const int per = (maxchunk + sp - 1) / sp;
      if (sp > 1 && per < MINCHUNK) continue;
      const int rounds = ((items * sp + nsm - 1) / nsm + cpa - 1) / cpa;
      const long cost = (long)rounds * (per * 64 + 39);
      if (best < 0 || cost < best) {
        best = cost;
        splits = sp;
      }
    }
  }
  return splits;
}

Schedule schedule(const plmn::FwdCtx& c) {
  const int splits = plan_splits(c);
  const int maxchunk = (c.nest + NC - 1) / NC;
  const int grid = c.BS * (c.H / HB) * splits;
  const int slots = c.nsm * resident_ctas(c);
  const int rounds = (grid + slots - 1) / slots;
  Schedule s;
  s.splits = splits;
  s.grid = grid;
  s.sms = std::min(grid, c.nsm);
  s.depth = rounds * ((maxchunk + splits - 1) / splits);
  return s;
}

Schedule best_one_round(const plmn::FwdCtx& c) {
  const int maxchunk = (c.nest + NC - 1) / NC;
  const int items = c.BS * (c.H / HB);
  const int cap = c.nsm * MINCTA;
  Schedule s;
  s.splits = 0;
  s.grid = 0;
  s.sms = 0;
  s.depth = 0;
  for (int sp = std::min(maxchunk, MAXSPLIT); sp >= 1; --sp) {
    if (items * sp > cap) continue;
    s.splits = sp;
    s.grid = items * sp;
    s.sms = s.grid;                     // one round, so every CTA is on its own SM
    s.depth = (maxchunk + sp - 1) / sp;
    return s;
  }
  return s;
}

void launch(const plmn::FwdCtx& c, int splits, __half* __restrict__ pop,
            float* __restrict__ pml) {
  const int H = c.H, BS = c.BS, listn = c.listn, nsm = c.nsm;
  const int smem = smem_bytes(listn);
  static int configured = -1;
  if (configured < smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(PLMN_KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem));
    configured = smem;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(H / HB, (unsigned)BS, (unsigned)splits);
  PLMN_KERNEL<<<grid, NT, smem, stream>>>(
      c.q, c.k1, c.idx1, c.k2, c.idx2, c.sink, c.sm_scale, c.out, c.lse, pop, pml, H, BS,
      c.budget1, c.budget2, c.page1, c.rec1, c.page2, c.rec2, listn, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (splits > 1) {

    const int total = BS * H;
    const int wgrid = (total + CH - 1) / CH;
    if (wgrid >= nsm) {
      PLMN_COMB_WIDE<<<(unsigned)wgrid, CT, 0, stream>>>(
          pop, pml, c.sink, c.out, c.lse, splits, H, BS);
    } else {
      PLMN_COMB_NARROW<<<(unsigned)total, 128, 0, stream>>>(
          pop, pml, c.sink, c.out, c.lse, splits, H, BS);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace PLMN_CFG
