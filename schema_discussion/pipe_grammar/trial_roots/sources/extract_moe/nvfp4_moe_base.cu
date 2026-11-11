/* candidate/nvfp4_moe.cu -- independent CUDA/PTX implementation of the frozen
 * nvfp4_fused_moe contract (L20D, cc 10.3 / sm_103).
 *
 * Every numerical step is implemented here in CUDA C++/PTX: NVFP4 (E2M1
 * payload + E4M3 block-16 scale) decoding of activations and of the
 * row-permuted / scale-swizzled prepared weights, grouped GEMMs with FP32
 * accumulation through bf16 warp tensor cores (an E2M1 x E4M3 product needs
 * at most 6 significand bits, so bf16 carries it exactly), the declared clamp
 * and activation, post-activation NVFP4 block-16 quantize/dequantize, the
 * per-route BF16 materialization of FC2 and the FP32 routing-weighted
 * reduction into the caller-owned output tensor.
 *
 * No GEMM/MoE library, CUTLASS/CuTe, Triton, cubin or torch numerics are
 * used. Python only allocates, binds the ABI and launches this object.
 *
 * The physical layout conventions (32-row block shuffle, 128x4 scale swizzle,
 * gate/up row interleave) are the published production ABI described by the
 * frozen contract (problem/reference.py, problem/input.py; upstream
 * flashinfer d849eca4 / TRT-LLM). The index algebra below was re-derived from
 * those frozen files, not copied from any library source.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>      /* CUtensorMap only; reached through cudart's entry point */
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#define DINLINE __device__ __forceinline__

namespace nvfp4moe {

/* --------------------------------------------------------------- parameters */
struct Params {
  const int32_t* ids;        /* [T,K] routed expert ids (-1 = no route)        */
  const float*   scores;     /* [T,K] routing weights                          */
  const uint8_t* hid_q;      /* [T,H/2] E2M1 payload, low nibble = even index  */
  const uint8_t* hid_sf;     /* [T,H/16] E4M3 bytes, logical row-major         */
  const uint8_t* w1_q;       /* [E,2I,H/2] prepared (permuted) payload         */
  const uint8_t* w1_sf;      /* [E,2I,H/16] prepared swizzled E4M3 bytes       */
  const uint8_t* w2_q;       /* [E,H,I/2]                                      */
  const uint8_t* w2_sf;      /* [E,H,I/16]                                     */
  const float*   alpha;      /* [E]                                            */
  const float*   beta;       /* [E]                                            */
  const float*   limit;      /* [E] or nullptr                                 */
  __nv_bfloat16* out;        /* [T,H] caller-owned, the only writable input    */
  /* scratch (owned by this implementation, sized for R_max = T*K routes)      */
  int32_t* counts;           /* [E]                                            */
  int32_t* fill;             /* [E]                                            */
  int32_t* offs;             /* [E+1] route offsets per expert                 */
  int32_t* toffs;            /* [E+1] m-tile offsets per expert                */
  int32_t* ptok;             /* [R_max] route -> token                         */
  int32_t* pslot;            /* [R_max] route -> topk slot                     */
  int32_t* pexp;             /* [R_max] route -> expert                        */
  int32_t* route_of;         /* [T*K] (t,k) -> route index, -1 if invalid      */
  int32_t* texp;             /* [max_tiles] tile -> expert                     */
  int32_t* tm0;              /* [max_tiles] tile -> first route inside expert  */
  __nv_bfloat16* hid_bf;     /* [T,H] decoded activations (exact bf16)         */
  float*         fc1;        /* [R_max,2I] FP32 pre-activation                 */
  __nv_bfloat16* act;        /* [R_max,I] QDQ'd activation (exact bf16)        */
  /* The FP4 rung (7) keeps the intermediate packed instead of materializing it
   * as bf16; both views alias the same `act` workspace (payload first, then the
   * E4M3 scale bytes), which needs R*I*9/16 bytes of the R*I*2 that are
   * already allocated. */
  uint8_t*       act_q;      /* [R_max,I/2] E2M1 payload, low nibble = even idx */
  uint8_t*       act_sf;     /* [R_max,I/16] E4M3 bytes, row-major             */
  __nv_bfloat16* rout;       /* [R_max,H] per-route FC2 materialized as BF16   */
  /* REDF rungs (46/47): [T,H] fp32 routing accumulator. GEMM2's epilogue
   * rounds each route to BF16 exactly as the rout store does, scales by the
   * routing weight and atomically adds (red.global.add.v4.f32) into acc32;
   * k_convert performs the single final BF16 rounding of the fp32 sum, so
   * the declared numerical boundary (per-route BF16 materialization, fp32
   * route sum) is preserved -- only the fp32 sum ORDER differs from the
   * sequential k_finalize form, exactly as the reference's own atomic
   * index_add_ does. Zeroed on a side-stream branch each call. */
  float*         acc32;
  /* Split-K GEMM1 (small-batch rungs): raw FP32 partial sums, one contiguous
   * [sk][R_max][2I] slab; k_act_packed reduces them in slice order, so the
   * result is deterministic for a given sk. */
  float*         part;
  /* Pre-gathered GEMM1 activation operand (rungs 27-29): k_gather_a copies
   * the routed token rows of hid_q/hid_sf into contiguous route order ONCE,
   * so every GEMM1 stage can box its A tile with one 2-D TMA request instead
   * of 1024+ 16-byte cp.async gathers. Rows >= offs[E] are never written;
   * tiles read them as stale padding whose lanes the epilogue discards (the
   * same contract GEMM2's act operand already runs under). */
  uint8_t*       hidq_g;     /* [R_max,H/2] E2M1 payload in route order      */
  uint8_t*       hidsf_g;    /* [R_max,H/16] E4M3 scales in route order      */
  /* Fused split-K activation (rungs 31/32): per-(m-tile,n-tile) arrival
   * counters, zeroed by the routing prep of the same call; the CTA whose
   * atomicAdd returns SK-1 owns the reduce+quantize of that group. */
  int32_t*       cnt;
  int T, K, E, H, I;
  int cnt_n;
  int Rmax, max_tiles, act_type, has_limit, bm, cfg2;
  int sk;
};

/* --------------------------------------------------- format tables (constants) */
/* Filled once by k_init_tables at module load. These encode only the NVFP4 /
 * E4M3 number formats, never problem data. */
__device__ unsigned int g_e2m1_pair[256];  /* byte -> bf16x2(lo nibble, hi nibble) */
__device__ float        g_e4m3_f32[256];   /* byte -> float                        */
__device__ unsigned int g_e4m3_bf2[256];   /* byte -> bf16 bits replicated twice   */

DINLINE float e2m1_mag(int k) {
  /* k in 0..7 -> 0, .5, 1, 1.5, 2, 3, 4, 6 */
  if (k == 0) return 0.0f;
  if (k == 1) return 0.5f;
  return __int_as_float(((((k >> 1) - 1) + 127) << 23) | ((k & 1) << 22));
}

DINLINE float e2m1_val(int c) {
  float v = e2m1_mag(c & 7);
  return (c & 8) ? -v : v;
}

DINLINE float e4m3_val(int b) {
  int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
  float v;
  if (e == 0xF && m == 7) v = __int_as_float(0x7FC00000);        /* NaN */
  else if (e == 0) v = (float)m * 0.001953125f;                 /* subnormal: m * 2^-9 */
  else v = __int_as_float(((e - 7 + 127) << 23) | (m << 20));
  return s ? -v : v;
}

DINLINE unsigned int bf16_bits(float f) {
  return (unsigned int)__bfloat16_as_ushort(__float2bfloat16_rn(f));
}

__global__ void k_init_tables() {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < 256) {
    int lo = i & 15, hi = i >> 4;
    unsigned int plo = bf16_bits(e2m1_val(lo)), phi = bf16_bits(e2m1_val(hi));
    g_e2m1_pair[i] = plo | (phi << 16);
    float s = e4m3_val(i);
    g_e4m3_f32[i] = s;
    unsigned int sb = bf16_bits(s);
    g_e4m3_bf2[i] = sb | (sb << 16);
  }
}

/* ------------------------------------------------------------- small helpers */
DINLINE void cp_async16(void* smem, const void* gmem) {
  unsigned int s = (unsigned int)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
DINLINE void cp_async4(void* smem, const void* gmem) {
  unsigned int s = (unsigned int)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(gmem));
}
DINLINE void cp_async8(void* smem, const void* gmem) {
  unsigned int s = (unsigned int)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(gmem));
}
DINLINE void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> DINLINE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

/* Programmatic Dependent Launch primitives. pdl_wait blocks until every
 * predecessor on the stream has completed and its writes are visible; when
 * the kernel was launched WITHOUT the programmatic-serialization attribute
 * (or no predecessor is pending) it returns immediately, and pdl_trigger is
 * a no-op when no dependent grid was launched with the attribute -- so both
 * sit unconditionally in the bodies and only the host-side launch attribute
 * (NVFP4_MOE_PDL) decides whether the overlap is armed. Production idiom
 * (alphamoe_nvfp4_sm100 load role, cake_grouped tma role): trigger EARLY --
 * it only releases the dependent grid's LAUNCH (its CTAs then block in their
 * own pdl_wait); visibility is guaranteed by the waiter, never by the
 * trigger's placement. One warp per CTA executes the trigger; the grid-wide
 * signal fires when every CTA has executed it (or the grid completes). */
DINLINE void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DINLINE void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

/* Tiled TMA. One box per operand tile, completion through an mbarrier armed with
 * the box's byte count. The tensor map carries the swizzle (SWIZZLE_128B), so the
 * image it lands is the SW128 layout the cp.async staging loop produces and the
 * MMA descriptors are unchanged. */
DINLINE void tma_expect_tx(unsigned bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(bar), "r"(bytes));
}
DINLINE void tma_box_2d(const CUtensorMap* m, unsigned dst, unsigned bar, int c0,
                        int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile"
               ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
DINLINE void tma_box_4d(const CUtensorMap* m, unsigned dst, unsigned bar, int c0,
                        int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.tile"
               ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
               : "memory");
}
DINLINE void tma_box_5d(const CUtensorMap* m, unsigned dst, unsigned bar, int c0,
                        int c1, int c2, int c3, int c4) {
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile"
               ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
                  "r"(bar) : "memory");
}
/* TMA tile::gather4: one instruction fetches four arbitrary rows x 128 bytes
 * into four consecutive 128-byte shared slots, with the map's SWIZZLE_128B
 * applied per destination row. This is the async replacement for the GEMM1
 * activation gather (1024 sixteen-byte cp.async requests per stage). The
 * canonical non-multicast .shared::cta form follows the PTX ISA qualifier
 * order; the row coordinates are tensor dim-1 indices (token ids here). */
DINLINE void tma_gather4(const CUtensorMap* m, unsigned dst, unsigned bar, int c0,
                         int r0, int r1, int r2, int r3) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
               ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
               :: "r"(dst), "l"(m), "r"(c0), "r"(r0), "r"(r1), "r"(r2), "r"(r3),
                  "r"(bar) : "memory");
}

DINLINE void ldmatrix_x4(unsigned int& r0, unsigned int& r1, unsigned int& r2,
                         unsigned int& r3, const void* addr) {
  unsigned int s = (unsigned int)__cvta_generic_to_shared(addr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}

DINLINE void mma_bf16(float& d0, float& d1, float& d2, float& d3,
                      unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
                      unsigned int b0, unsigned int b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/* Correctly-rounded x/d given inv = rcp.rn(d): q0 = x*inv, then the Markstein
 * refinement with two fused multiply-adds. This is the same arithmetic
 * div.rn.f32 performs internally, with the reciprocal hoisted out of the
 * 16-element scale block, so 16 divisions cost 1 reciprocal + 48 FMA-class
 * instructions instead of ~160. experiments/div_exactness.py sweeps it against
 * div.rn.f32 (all 255 nonzero E4M3 divisors x structured and random dividends,
 * including every rounding tie boundary) before it is relied on. */
DINLINE float div_by_recip(float x, float d, float inv) {
  float q0 = x * inv;
  float r = fmaf(-d, q0, x);
  return fmaf(r, inv, q0);
}

/* E2M1 quantize of one value given a positive block scale, returning the 4-bit
 * code (magnitude with the reference's tie rules, sign in bit 3) instead of the
 * dequantized value. The FP4 rung stores these codes as the GEMM2 operand, so
 * the tensor core applies the scale itself; code*sf is then bit-identical to the
 * value qdq_e2m1 returns, which is what the bf16 rungs store. */
DINLINE int q_e2m1_code(float x, float sf, float inv_sf) {
  float q = div_by_recip(x, sf, inv_sf);   /* == x/sf, see div_by_recip */
  float m = fabsf(q);
  int code = 0;
  code += (m > 0.25f);
  code += (m >= 0.75f);
  code += (m > 1.25f);
  code += (m >= 1.75f);
  code += (m > 2.5f);
  code += (m >= 3.5f);
  code += (m > 5.0f);
  return code | (signbit(q) ? 8 : 0);
}

/* E4M3 encode: round-to-nearest-even with saturation at 448, exactly the
 * reference's clamp(min,448).to(float8_e4m3fn). Input is >= 0 here. */
DINLINE int f32_to_e4m3(float v) {
  unsigned short bits;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
               : "=h"(bits) : "f"(v), "f"(v));
  return (int)(bits & 0xFF);
}

/* E2M1 quantize/dequantize of one value given a positive block scale, with the
 * reference's tie rules (strict > at .25/1.25/2.5/5, >= at .75/1.75/3.5). */
DINLINE float qdq_e2m1(float x, float sf, float inv_sf) {
  float q = div_by_recip(x, sf, inv_sf);   /* == x/sf, see div_by_recip */
  float m = fabsf(q);
  int code = 0;
  code += (m > 0.25f);
  code += (m >= 0.75f);
  code += (m > 1.25f);
  code += (m >= 1.75f);
  code += (m > 2.5f);
  code += (m >= 3.5f);
  code += (m > 5.0f);
  float dq = e2m1_mag(code);
  return copysignf(dq, q) * sf;
}

/* Hardware E2M1 pair cast: cvt.rn.satfinite.e2m1x2 rounds to nearest with
 * ties-to-even and saturates at 6 -- exactly the reference's tie/saturation
 * rules -- so for the SAME correctly-rounded quotient (div_by_recip is kept)
 * the produced codes are bit-identical to the seven-compare chain, at one
 * instruction per PAIR instead of ~14 per element. The destination is a .b8
 * register and the FIRST source operand lands in the HIGH nibble (production
 * form: flashinfer alphamoe_nvfp4_sm100 / cake_minimax_h3_fc1_swiglu use
 * `cvt b0, %odd, %even` so the even element takes the low nibble, matching
 * the payload packing of problem/input.py). q_e2m1_pack8 packs eight
 * quotients into one payload word with the same 4-cvt + mov.b32 sequence.
 * experiments/probe_e2m1.py verifies order/ties/saturation on the actual
 * device (moe_selftest which=4) before the epilogues rely on it. */
DINLINE unsigned int q_e2m1_pack8(float v0, float v1, float v2, float v3,
                                  float v4, float v5, float v6, float v7) {
  unsigned int out;
  asm volatile("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
               "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n\t"
               "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n\t"
               "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n\t"
               "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n\t"
               "mov.b32 %0, {b0, b1, b2, b3};\n\t}"
               : "=r"(out)
               : "f"(v0), "f"(v1), "f"(v2), "f"(v3),
                 "f"(v4), "f"(v5), "f"(v6), "f"(v7));
  return out;
}

/* Fast activation arithmetic for the hot fused epilogues. The frozen contract
 * (problem/metadata.json numerical contract) "does not prescribe approximate
 * activation or reciprocal instructions": precision is tolerance-governed.
 * sigmoid = rcp(1 + ex2.approx) has ~3e-7 relative error; tanh uses the
 * production identity tanh(x) = 2*sigmoid(2x) - 1 (the DeepGEMM/FlashInfer
 * situ formulation), ~1e-7 absolute. A perturbed activation value can only
 * change the output through an NVFP4 code flip when it lands within ~1e-6 of
 * a quantization midpoint (p ~ 2e-6 per element); one flip perturbs one token
 * row by ~sf*0.5*w2 << atol=0.01, so mismatch_fraction stays 0 and rel_l2
 * stays ~1e-3 scale against the 0.01 bound (measured on the check shapes).
 * The E2M1/E4M3 casts themselves stay hardware-RNE on the same
 * correctly-rounded quotients, so quantization is NOT approximated. */
DINLINE float sigmoid_fast(float x) {
  float d = 1.0f + __expf(-x);
  float r;
  asm volatile("rcp.approx.f32 %0, %1;\n" : "=f"(r) : "f"(d));
  return r;
}
DINLINE float tanh_fast(float x) {
  return fmaf(sigmoid_fast(2.0f * x), 2.0f, -1.0f);
}
/* alpha/beta reciprocals are hoisted per item by the caller (the reference
 * divides per element; gate*rcp_rn(alpha) differs from gate/alpha by <=1 ulp). */
DINLINE float activation_fast(const Params& p, float gate, float up,
                              float alpha, float beta,
                              float inv_alpha, float inv_beta) {
  if (p.act_type == 10) {
    float t0 = alpha * tanh_fast(gate * inv_alpha);
    float t1 = beta * tanh_fast(up * inv_beta);
    return (t0 * sigmoid_fast(gate)) * t1;
  }
  return (gate * sigmoid_fast(alpha * gate)) * (up + beta);
}

/* Blackwell packed FP32 arithmetic (f32x2): the epilogue activation/quantize
 * chain is the dominant GEMM1 instruction consumer (job Q NCU: tensor pipe
 * only ~13% busy while the FMUL/FMNMX epilogue lines top the stall samples),
 * so the pair-parallel forms below halve its ALU issue. The instruction forms
 * are the production ones (flashinfer cake_grouped_fp8_fused_silu_quant
 * sm_100a: mul.f32x2 / add.f32x2 / fma.rn.f32x2 over 64-bit register pairs).
 * Quantization switches from the Markstein-corrected div_by_recip to a plain
 * packed multiply by the correctly-rounded reciprocal: a <=1.5-ulp quotient
 * perturbation flips an E2M1 code only within ~1e-6 of a tie (p ~ 2e-6 per
 * element), the same tolerance-governed class the fast activation already
 * relies on (measured worst-case rel_l2 6.6e-4 << 0.01, 0 mismatched). */
DINLINE float2 f2_mul(float2 a, float2 b) {
  unsigned long long r;
  asm("mul.f32x2 %0, %1, %2;" : "=l"(r)
      : "l"(*reinterpret_cast<unsigned long long*>(&a)),
        "l"(*reinterpret_cast<unsigned long long*>(&b)));
  return *reinterpret_cast<float2*>(&r);
}
DINLINE float2 f2_add(float2 a, float2 b) {
  unsigned long long r;
  asm("add.f32x2 %0, %1, %2;" : "=l"(r)
      : "l"(*reinterpret_cast<unsigned long long*>(&a)),
        "l"(*reinterpret_cast<unsigned long long*>(&b)));
  return *reinterpret_cast<float2*>(&r);
}
DINLINE float2 f2_fma(float2 a, float2 b, float2 c) {
  unsigned long long r;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(r)
      : "l"(*reinterpret_cast<unsigned long long*>(&a)),
        "l"(*reinterpret_cast<unsigned long long*>(&b)),
        "l"(*reinterpret_cast<unsigned long long*>(&c)));
  return *reinterpret_cast<float2*>(&r);
}
DINLINE float2 ex2_2(float2 x) {
  float r0, r1;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r0) : "f"(x.x));
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r1) : "f"(x.y));
  return make_float2(r0, r1);
}
DINLINE float2 rcp2(float2 x) {
  float r0, r1;
  asm("rcp.approx.f32 %0, %1;\n" : "=f"(r0) : "f"(x.x));
  asm("rcp.approx.f32 %0, %1;\n" : "=f"(r1) : "f"(x.y));
  return make_float2(r0, r1);
}
/* C11a: sigmoid with the ex2 argument prescale FOLDED into a per-item
 * constant: sigmoid(v) = rcp(1 + ex2(v * k)), k = -(log2 e) * (whatever
 * scales v). __expf does the -log2e multiply per element (job X SASS: the
 * scalar FMUL prescale is 10.8% of GEMM1's non-issued stall samples); the
 * fold turns (f2_mul + 2 scalar FMUL) into ONE f2_mul per pair. The folded
 * argument differs from the two-step scaling by <=1.5 ulp, so sigmoid moves
 * <=~1e-7 relative -- the same tolerance-governed class as the fast
 * activation itself (bounded by probe selftest 8 and the frozen checker). */
DINLINE float2 sigmoid_pre(float2 v2, float2 k2) {
  return rcp2(f2_add(ex2_2(f2_mul(v2, k2)), make_float2(1.0f, 1.0f)));
}
/* Packed activation over one float2 (gate, up) pair. k_sw prescales the
 * gate-sigmoid argument (swiglu: -alpha*log2e; situ: -log2e for
 * sigmoid(gate)); k_gu/k_uu prescale the situ tanh arguments
 * (inv_alpha|inv_beta * -2*log2e; tanh(x) = 2*sigmoid(2x) - 1). */
DINLINE float2 activation2(int act_type, float2 g2, float2 u2,
                           float alpha, float beta,
                           float2 k_sw, float2 k_gu, float2 k_uu) {
  if (act_type == 10) {
    const float2 two = make_float2(2.0f, 2.0f);
    const float2 m1 = make_float2(-1.0f, -1.0f);
    const float2 t0 = f2_mul(make_float2(alpha, alpha),
                             f2_fma(sigmoid_pre(g2, k_gu), two, m1));
    const float2 t1 = f2_mul(make_float2(beta, beta),
                             f2_fma(sigmoid_pre(u2, k_uu), two, m1));
    return f2_mul(f2_mul(t0, sigmoid_pre(g2, k_sw)), t1);
  }
  return f2_mul(f2_mul(g2, sigmoid_pre(g2, k_sw)),
                f2_add(u2, make_float2(beta, beta)));
}
/* Non-folded packed activation (the C10 form) for latency-bound kernels:
 * bit-identical to activation_fast (per-lane mul.rn/add.rn/ex2/rcp.approx,
 * same order). k_act_packed does only ~40 instructions per thread, so the
 * folded prescale's two saved FMULs per pair are worthless there while its
 * extra live constants pushed the kernel 72 -> 90 registers, dropping it
 * from 3 to 2 blocks/SM at 256 threads and costing ~0.7% end-to-end on the
 * SK band (job Z: fixed_2 0.9926, fixed_4 0.9919, domain_5 0.9919,
 * domain_7 0.9934 with tight block medians -- real, not drift). */
DINLINE float2 sigmoid2(float2 x) {
  const float2 n = make_float2(-1.44269504f, -1.44269504f);
  return rcp2(f2_add(ex2_2(f2_mul(x, n)), make_float2(1.0f, 1.0f)));
}
DINLINE float2 tanh2(float2 x) {
  return f2_fma(sigmoid2(f2_add(x, x)), make_float2(2.0f, 2.0f),
                make_float2(-1.0f, -1.0f));
}
DINLINE float2 activation2_nf(int act_type, float2 g2, float2 u2,
                              float alpha, float beta,
                              float inv_alpha, float inv_beta) {
  if (act_type == 10) {
    const float2 two = make_float2(2.0f, 2.0f);
    const float2 m1 = make_float2(-1.0f, -1.0f);
    const float2 t0 = f2_mul(make_float2(alpha, alpha),
        tanh2(f2_mul(g2, make_float2(inv_alpha, inv_alpha))));
    const float2 t1 = f2_mul(make_float2(beta, beta),
        tanh2(f2_mul(u2, make_float2(inv_beta, inv_beta))));
    return f2_mul(f2_mul(t0, sigmoid2(g2)), t1);
  }
  return f2_mul(f2_mul(g2, sigmoid2(f2_mul(g2, make_float2(alpha, alpha)))),
                f2_add(u2, make_float2(beta, beta)));
}

/* Per-item prescale constants for activation2 (mirrors activation_fast's
 * alpha/beta/inv_alpha/inv_beta semantics with the log2e fold). */
struct ActK { float2 k_sw, k_gu, k_uu; };
DINLINE ActK act_k_constants(int act_type, float alpha,
                             float inv_alpha, float inv_beta) {
  const float L2E = -1.44269504f;           /* -(log2 e), the __expf prescale */
  ActK k;
  if (act_type == 10) {
    k.k_sw = make_float2(L2E, L2E);
    k.k_gu = make_float2(inv_alpha * (2.0f * L2E), inv_alpha * (2.0f * L2E));
    k.k_uu = make_float2(inv_beta * (2.0f * L2E), inv_beta * (2.0f * L2E));
  } else {
    k.k_sw = make_float2(alpha * L2E, alpha * L2E);
    k.k_gu = k.k_sw;
    k.k_uu = k.k_sw;
  }
  return k;
}

/* --------------------------------------------------------------- routing ---- */
/* Routing prep measured 10-45 us on every shape in the job-A stage profile
 * (dv_7d07d264c41d): the single-block small path spent most of it in shared
 * atomic contention (32 warps x 128 hot counters) and a serial tid-0 scan of E
 * experts; the big path's k_scatter serialized on global atomic round-trips.
 * The rewrites below remove both: warp-private fill tables, warp-aggregated
 * global atomics (one round-trip per warp per distinct expert, rank via
 * ballot), and a two-level parallel scan. Route ORDER inside an expert is an
 * arbitrary permutation in every version (V0's was atomic-order), and all
 * downstream consumers only read the (ptok, pslot, pexp, route_of) tuple. */
__global__ void k_count(const Params p) {
  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  /* Two-level histogram: routes are counted in shared memory and each block
   * publishes one atomic per expert it saw, instead of one global atomic per
   * route on E counters. The counts are a sum, so the result is unchanged. */
  __shared__ int s_cnt[256];
  const int n = p.T * p.K;
  for (int e = threadIdx.x; e < p.E; e += blockDim.x) s_cnt[e] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    const int e = p.ids[i];
    if (e >= 0 && e < p.E) atomicAdd(&s_cnt[e], 1);
  }
  __syncthreads();
  for (int e = threadIdx.x; e < p.E; e += blockDim.x)
    if (s_cnt[e]) atomicAdd(&p.counts[e], s_cnt[e]);
}

/* Inclusive scan of (c, tl) over NW*32 threads of one block, returning each
 * thread's EXCLUSIVE prefixes and the block totals. Every thread of the block
 * must call it (two block-wide barriers inside). s_wtot/s_wpre are 2*NW-int
 * scratch arrays; lanes >= NW feed zeros into the second-level warp scan. */
template <int NW>
DINLINE void scanNW(int t, int c0, int tl0, int& c_excl, int& tl_excl,
                    int* s_wtot, int* s_wpre, int& c_total, int& tl_total) {
  const int lane = t & 31, w = t >> 5;
  int c = c0, tl = tl0;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    int c2 = __shfl_up_sync(0xffffffffu, c, d);
    int t2 = __shfl_up_sync(0xffffffffu, tl, d);
    if (lane >= d) { c += c2; tl += t2; }
  }
  if (lane == 31) { s_wtot[w] = c; s_wtot[NW + w] = tl; }
  __syncthreads();
  if (w == 0) {
    int a = (lane < NW) ? s_wtot[lane] : 0;
    int b = (lane < NW) ? s_wtot[NW + lane] : 0;
    const int ea = a, eb = b;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int a2 = __shfl_up_sync(0xffffffffu, a, d);
      int b2 = __shfl_up_sync(0xffffffffu, b, d);
      if (lane >= d) { a += a2; b += b2; }
    }
    if (lane < NW) { s_wpre[lane] = a - ea; s_wpre[NW + lane] = b - eb; }
  }
  __syncthreads();
  c_excl = s_wpre[w] + (c - c0);
  tl_excl = s_wpre[NW + w] + (tl - tl0);
  c_total = s_wpre[NW - 1] + s_wtot[NW - 1];
  tl_total = s_wpre[2 * NW - 1] + s_wtot[2 * NW - 1];
}

__global__ void k_scan(const Params p, int bm) {
  __shared__ int s_wtot[16];
  __shared__ int s_wpre[16];
  const int t = threadIdx.x;               /* 256 threads = NW 8 */
  const int c0 = (t < p.E) ? p.counts[t] : 0;
  const int tl0 = (c0 + bm - 1) / bm;
  int c_excl, tl_excl, c_total, tl_total;
  scanNW<8>(t, c0, tl0, c_excl, tl_excl, s_wtot, s_wpre, c_total, tl_total);
  if (t < p.E) {
    p.offs[t] = c_excl;
    p.toffs[t] = tl_excl;
  }
  if (t == 0) {
    p.offs[p.E] = c_total;
    p.toffs[p.E] = tl_total;
  }
}

__global__ void k_scatter(const Params p) {
  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  const int n = p.T * p.K;
  const int lane = threadIdx.x & 31;
  for (int base = blockIdx.x * blockDim.x; base < n;
       base += blockDim.x * gridDim.x) {
    const int i = base + threadIdx.x;
    int e = (i < n) ? p.ids[i] : -1;
    const bool ok = (e >= 0 && e < p.E);
    if (!ok) e = -1;
    /* One global atomic round-trip per warp per distinct expert: the leader
     * reserves popc(mask) consecutive slots, the others take leader+rank. */
    const unsigned mask = __match_any_sync(0xffffffffu, e);
    const int leader = __ffs(mask) - 1;
    const int rank = __popc(mask & ((1u << lane) - 1));
    int pos = 0;
    if (lane == leader && e >= 0)
      pos = p.offs[e] + atomicAdd(&p.fill[e], __popc(mask));
    pos = __shfl_sync(0xffffffffu, pos, leader);
    if (i >= n) continue;
    if (!ok) { p.route_of[i] = -1; continue; }
    pos += rank;
    p.ptok[pos] = i / p.K;
    p.pslot[pos] = i - (i / p.K) * p.K;
    p.pexp[pos] = e;
    p.route_of[i] = pos;
  }
}

__global__ void k_tiles(const Params p, int bm) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= p.max_tiles) return;
  int total = p.toffs[p.E];
  if (j >= total) return;
  int lo = 0, hi = p.E;                    /* toffs[E] > j >= toffs[0] = 0 */
  while (hi - lo > 1) {
    int mid = (lo + hi) >> 1;
    if (p.toffs[mid] <= j) lo = mid; else hi = mid;
  }
  p.texp[j] = lo;
  p.tm0[j] = (j - p.toffs[lo]) * bm;
}

/* Fused scan + tile map for the big routing path: both are tiny single-block
 * kernels (256 threads each) whose separate launches cost ~3 us of graph-node
 * floor each; the tile binary searches run from the scan's shared toffs right
 * after the scan, with no global round-trip in between. max_tiles is bounded
 * by R/128 + E <= 640 in the declared domain, so one 1024-thread block covers
 * every tile in a single pass. */
__global__ void __launch_bounds__(1024) k_scan_tiles(const Params p, int bm) {
  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  __shared__ int s_toffs[257];
  __shared__ int s_wtot[64];
  __shared__ int s_wpre[64];
  __shared__ int s_total;
  const int t = threadIdx.x;
  const int c0 = (t < p.E) ? p.counts[t] : 0;
  const int tl0 = (c0 + bm - 1) / bm;
  int c_excl, tl_excl, c_total, tl_total;
  scanNW<32>(t, c0, tl0, c_excl, tl_excl, s_wtot, s_wpre, c_total, tl_total);
  if (t < p.E) {
    p.offs[t] = c_excl;
    s_toffs[t] = tl_excl;
    p.toffs[t] = tl_excl;
  }
  if (t == 0) {
    p.offs[p.E] = c_total;
    s_toffs[p.E] = tl_total;
    p.toffs[p.E] = tl_total;
    s_total = tl_total;
  }
  __syncthreads();
  for (int j = t; j < s_total; j += 1024) {
    int lo = 0, hi = p.E;
    while (hi - lo > 1) {
      int mid = (lo + hi) >> 1;
      if (s_toffs[mid] <= j) lo = mid; else hi = mid;
    }
    p.texp[j] = lo;
    p.tm0[j] = (j - s_toffs[lo]) * bm;
  }
}

union Bf2Word { unsigned int u; __nv_bfloat162 v; };
DINLINE __nv_bfloat162 as_bf2(unsigned int u) { Bf2Word w; w.u = u; return w.v; }
DINLINE unsigned int as_u32(__nv_bfloat162 v) { Bf2Word w; w.v = v; return w.u; }

/* ------------------------------------------------------- fused small prepare */
/* For small batches the routing pipeline is pure launch overhead: five tiny
 * kernels (memset, histogram, scan, scatter, tile map) plus the activation
 * decode each cost several microseconds of device time regardless of work.
 * One block does all of it from shared memory when the shapes are small enough
 * that a single block is not the slower choice. Shape-selected, never data
 * selected, so both paths stay exercised by the fixed and domain suites.
 *
 * Restructured after the job-A profile showed 12-44 us here: each of the 32
 * warps histograms and scatters its own contiguous route range through a
 * warp-PRIVATE fill table (the V0 version had all 32 warps contending on the
 * same E shared counters), the expert scan is the parallel scanNW instead of a
 * serial tid-0 loop, and the per-warp bases are the running prefix of the
 * private tables. Route order inside an expert is a permutation, so the
 * private-order positions are as valid as V0's atomic-order ones. */
__global__ void __launch_bounds__(1024) k_prepare_small(const Params p) {
  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  __shared__ int s_fillw[32 * 256];   /* warp-private fill/counters -> bases */
  __shared__ int s_offs[257];
  __shared__ int s_toffs[257];
  __shared__ int s_wtot[64];
  __shared__ int s_wpre[64];
  __shared__ int s_total;
  extern __shared__ __align__(16) int s_ids[];   /* [nroutes] staged expert ids */
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int nroutes = p.T * p.K;
  /* Contiguous per-warp route ranges (multiples of 32 keep warps aligned). */
  const int chunk = ((nroutes + 31) / 32 + 31) & ~31;
  const int wlo = min(warp * chunk, nroutes);
  const int whi = min(wlo + chunk, nroutes);
  /* Zero the fused split-K arrival counters (rungs 31/32; cnt_n == 0
   * elsewhere). GEMM1 runs after this kernel completes (PDL wait), so the
   * counters are never read stale. */
  for (int i = tid; i < p.cnt_n; i += 1024) p.cnt[i] = 0;
  /* 0. stage every routed expert id into shared with 16-byte loads: the
   * histogram and the scatter below then read shared memory instead of each
   * running a latency-serialized pass of dependent global loads (job A
   * measured 40 us for this kernel at R=16K; the ids are read twice). */
  if ((reinterpret_cast<uintptr_t>(p.ids) & 15u) == 0 && (nroutes & 3) == 0) {
    const int n4 = nroutes >> 2;
    for (int i4 = tid; i4 < n4; i4 += 1024)
      reinterpret_cast<int4*>(s_ids)[i4] = reinterpret_cast<const int4*>(p.ids)[i4];
  } else {
    for (int i = tid; i < nroutes; i += 1024) s_ids[i] = p.ids[i];
  }
  /* Zero the FULL warp-private table: the row stride is 256 regardless of E,
   * so warps >= 16 with E=128 index entries above 32*E. Leaving those
   * un-zeroed corrupts the step-2 prefix bases -> OOB scatter positions
   * (this was the fixed-suite IMA in job dv_37f63420c67d). */
  for (int i = tid; i < 32 * 256; i += 1024) s_fillw[i] = 0;
  __syncthreads();
  /* 1. warp-private histogram */
  for (int i = wlo + lane; i < whi; i += 32) {
    const int e = s_ids[i];
    if (e >= 0 && e < p.E) atomicAdd(&s_fillw[warp * 256 + e], 1);
  }
  __syncthreads();
  /* 2. per-expert cross-warp prefix in place; totals feed the expert scan */
  int c0 = 0, tl0 = 0;
  if (tid < p.E) {
    int acc = 0;
    for (int w = 0; w < 32; ++w) {
      const int c = s_fillw[w * 256 + tid];
      s_fillw[w * 256 + tid] = acc;
      acc += c;
    }
    c0 = acc;
    tl0 = (acc + p.bm - 1) / p.bm;
    p.counts[tid] = acc;
  }
  __syncthreads();
  /* 3. exclusive scan over experts (threads >= 256 feed zeros) */
  int c_excl, tl_excl, c_total, tl_total;
  scanNW<32>(tid, c0, tl0, c_excl, tl_excl, s_wtot, s_wpre, c_total, tl_total);
  if (tid < p.E) {
    s_offs[tid] = c_excl;
    p.offs[tid] = c_excl;
    s_toffs[tid] = tl_excl;
    p.toffs[tid] = tl_excl;
  }
  if (tid == 0) {
    s_offs[p.E] = c_total;
    p.offs[p.E] = c_total;
    s_toffs[p.E] = tl_total;
    p.toffs[p.E] = tl_total;
    s_total = tl_total;
  }
  __syncthreads();
  /* 4. scatter through the private bases (uncontended shared atomics) */
  for (int i = wlo + lane; i < whi; i += 32) {
    const int e = s_ids[i];
    if (e < 0 || e >= p.E) { p.route_of[i] = -1; continue; }
    const int pos = s_offs[e] + atomicAdd(&s_fillw[warp * 256 + e], 1);
    p.ptok[pos] = i / p.K;
    p.pslot[pos] = i - (i / p.K) * p.K;
    p.pexp[pos] = e;
    p.route_of[i] = pos;
  }
  /* 5. tile map */
  for (int j = tid; j < s_total; j += 1024) {
    int lo = 0, hi = p.E;
    while (hi - lo > 1) {
      int mid = (lo + hi) >> 1;
      if (s_toffs[mid] <= j) lo = mid; else hi = mid;
    }
    p.texp[j] = lo;
    p.tm0[j] = (j - s_toffs[lo]) * p.bm;
  }
}

/* ------------------------------------------------- activation payload decode */
/* Pre-gathered GEMM1 activation operand (rungs 27-29). One block per route
 * row copies the routed token's payload and scale bytes into contiguous
 * route order, so every GEMM1 stage boxes its 128-row A tile with ONE 2-D
 * TMA request instead of 1024+ 16-byte cp.async gathers (which NCU shows as
 * the top ALU/LSU consumer of the mainloop: 8 requests + address algebra per
 * thread per stage, twice per SM-stage-pair). Rows >= offs[E] are not
 * written; GEMM1 tiles read them as stale padding whose accumulator lanes
 * the epilogue discards -- the same contract GEMM2's act operand runs under.
 * H is a multiple of 2048 in the declared domain, so both row strides
 * (H/2, H/16 bytes) and every row base are 16-byte aligned. */
__global__ void k_gather_a(const Params p) {
  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  const int n = p.offs[p.E];               /* valid route count (device-side) */
  const int hb = p.H >> 1;                 /* payload bytes per row */
  const int hs = p.H >> 4;                 /* scale bytes per row   */
  /* Grid-stride over 4-row chunks under a 592-CTA cap: empties/short routings
   * pay one small launch (domain_4 lost 7 us to the per-row grid), while the
   * 4-row unroll keeps ~4 independent 16B load streams per thread in flight,
   * which is what saturates HBM at big R (the 1-row stride measured 78 us on
   * fixed_6 against a ~38 us bandwidth floor: 111 serial rows per block). */
  for (int r0 = blockIdx.x * 4; r0 < n; r0 += gridDim.x * 4) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int r = r0 + u;
      if (r >= n) break;
      const int t = p.ptok[r];
      const uint8_t* sq = p.hid_q + (size_t)t * hb;
      uint8_t* const dq = p.hidq_g + (size_t)r * hb;
      for (int off = threadIdx.x * 16; off < hb; off += blockDim.x * 16)
        *reinterpret_cast<uint4*>(dq + off) =
            *reinterpret_cast<const uint4*>(sq + off);
      const uint8_t* ss = p.hid_sf + (size_t)t * hs;
      uint8_t* const ds = p.hidsf_g + (size_t)r * hs;
      for (int off = threadIdx.x * 16; off < hs; off += blockDim.x * 16)
        *reinterpret_cast<uint4*>(ds + off) =
            *reinterpret_cast<const uint4*>(ss + off);
    }
  }
}

/* C17 (experiment, NVFP4_MOE_GATHERT=1): token-major pre-gather. k_gather_a
 * reads each token's payload+scale row K times (once per route); this form
 * reads a row ONCE into a register and writes it to the token's K sorted
 * route positions (route_of, the same indices k_finalize consumes), cutting
 * gather read traffic K-fold (fixed_6: 150 MB -> 84 MB total; the stage
 * measures 68 us against a ~38 us write floor). Bytes and destinations are
 * identical to k_gather_a, so the hidq_g/hidsf_g images -- and every
 * downstream output -- are bit-identical. Invalid slots (route_of < 0) are
 * skipped; positions past offs[E] never appear in route_of. */
__global__ void k_gather_t(const Params p) {
  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  /* Empty-routing bail: k_gather_a does zero work when offs[E] == 0; the
   * token-major form must match that contract (job AH: domain_4 0.961 from
   * scanning route_of rows with no valid routes). */
  if (p.offs[p.E] <= 0) return;
  const int hb = p.H >> 1;                 /* payload bytes per row */
  const int hs = p.H >> 4;                 /* scale bytes per row   */
  const int kb = hb >> 4;                  /* 16B chunks per payload row */
  const int ks = hs >> 4;                  /* 16B chunks per scale row */
  for (int t = blockIdx.x; t < p.T; t += gridDim.x) {
    const int32_t* ro = p.route_of + (size_t)t * p.K;
    const uint8_t* sq = p.hid_q + (size_t)t * hb;
    const uint8_t* ss = p.hid_sf + (size_t)t * hs;
    if (p.K == 8) {
      int rr[8];
      bool any = false;
#pragma unroll
      for (int k = 0; k < 8; ++k) { rr[k] = ro[k]; any |= (rr[k] >= 0); }
      /* Empty-routing protection: skip the row read entirely when the token
       * has no valid route (k_gather_a does zero work there; the token-major
       * form would otherwise stream every row of an empty routing). */
      if (!any) continue;
      for (int off = threadIdx.x; off < kb; off += blockDim.x) {
        const uint4 v = *reinterpret_cast<const uint4*>(sq + (size_t)off * 16);
#pragma unroll
        for (int k = 0; k < 8; ++k)
          if (rr[k] >= 0)
            *reinterpret_cast<uint4*>(p.hidq_g + (size_t)rr[k] * hb
                                      + (size_t)off * 16) = v;
      }
      for (int off = threadIdx.x; off < ks; off += blockDim.x) {
        const uint4 v = *reinterpret_cast<const uint4*>(ss + (size_t)off * 16);
#pragma unroll
        for (int k = 0; k < 8; ++k)
          if (rr[k] >= 0)
            *reinterpret_cast<uint4*>(p.hidsf_g + (size_t)rr[k] * hs
                                      + (size_t)off * 16) = v;
      }
    } else {
      bool any = false;
      for (int k = 0; k < p.K; ++k) any |= (ro[k] >= 0);
      if (!any) continue;
      for (int off = threadIdx.x; off < kb; off += blockDim.x) {
        const uint4 v = *reinterpret_cast<const uint4*>(sq + (size_t)off * 16);
        for (int k = 0; k < p.K; ++k) {
          const int r = ro[k];
          if (r >= 0)
            *reinterpret_cast<uint4*>(p.hidq_g + (size_t)r * hb
                                      + (size_t)off * 16) = v;
        }
      }
      for (int off = threadIdx.x; off < ks; off += blockDim.x) {
        const uint4 v = *reinterpret_cast<const uint4*>(ss + (size_t)off * 16);
        for (int k = 0; k < p.K; ++k) {
          const int r = ro[k];
          if (r >= 0)
            *reinterpret_cast<uint4*>(p.hidsf_g + (size_t)r * hs
                                      + (size_t)off * 16) = v;
        }
      }
    }
  }
}

/* ------------------------------------------------- activation payload decode */
/* hidden_states: [T,H/2] E2M1 payload + [T,H/16] row-major E4M3 scales ->
 * exact bf16. One thread per 16-element scale block. */
__global__ void k_decode_hidden(const Params p) {
  const int nb = p.H >> 4;                 /* 16-element scale blocks per token */
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nb) return;
  const size_t idx = (size_t)blockIdx.y * nb + b;
  const uint2 payload = *reinterpret_cast<const uint2*>(p.hid_q + idx * 8);
  unsigned int sc2 = g_e4m3_bf2[p.hid_sf[idx]];
  __nv_bfloat162 w = as_bf2(sc2);
  unsigned int out[8];
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&payload);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    /* byte j holds elements 2j (low nibble) and 2j+1 (high nibble) */
    out[j] = as_u32(__hmul2(as_bf2(g_e2m1_pair[bytes[j]]), w));
  }
  *reinterpret_cast<uint4*>(p.hid_bf + idx * 16) =
      make_uint4(out[0], out[1], out[2], out[3]);
  *reinterpret_cast<uint4*>(p.hid_bf + idx * 16 + 8) =
      make_uint4(out[4], out[5], out[6], out[7]);
}

/* ------------------------------------------ clamp + activation + NVFP4 QDQ --- */
/* 1/(1+e) with the correctly-rounded reciprocal instruction: the numerator is
 * exactly 1.0, so rcp.rn IS the correctly-rounded quotient and this is
 * bit-identical to the division while costing one instruction instead of the
 * ~10-instruction div.rn.f32 sequence. The GEMM1 epilogue evaluates this once
 * per output element, so it was a measurable share of the stage: the pipeline
 * ablation (job dv_6ceeece75093) puts the packed-FP4 pipeline floor at 2114
 * cycles per stage while the real GEMM1 spends ~52k cycles per 8-stage item,
 * i.e. two thirds of GEMM1 is the epilogue, not the tensor core. */
DINLINE float sigmoid_f32(float x) { return __frcp_rn(1.0f + expf(-x)); }

DINLINE float activation(const Params& p, float gate, float up, float alpha, float beta) {
  if (p.act_type == 10) {
    /* SiTU: alpha*tanh(gate/alpha)*sigmoid(gate)*(beta*tanh(up/beta)) */
    float t0 = alpha * tanhf(gate / alpha);
    float t1 = beta * tanhf(up / beta);
    return (t0 * sigmoid_f32(gate)) * t1;
  }
  /* gate * sigmoid(alpha*gate) * (up + beta) */
  return (gate * sigmoid_f32(alpha * gate)) * (up + beta);
}

__global__ void k_act(const Params p) {
  int nb = p.I >> 4;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int r = idx / nb;
  if (r >= p.offs[p.E]) return;
  int b = idx - r * nb;
  int e = p.pexp[r];
  float alpha = p.alpha[e], beta = p.beta[e];
  const float* src = p.fc1 + (size_t)r * (2 * p.I) + ((size_t)b << 4);
  __nv_bfloat16* dst = p.act + (size_t)r * p.I + ((size_t)b << 4);
  float vals[16];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    float up = src[j];
    float gate = src[p.I + j];
    if (p.has_limit) {
      float lim = p.limit[e];
      up = fminf(fmaxf(up, -lim), lim);
      gate = fminf(gate, lim);
    }
    float a = activation(p, gate, up, alpha, beta);
    vals[j] = a;
    amax = fmaxf(amax, fabsf(a));
  }
  float sf_pre = fminf(amax * (1.0f / 6.0f), 448.0f);
  float sf = e4m3_val(f32_to_e4m3(sf_pre));
  unsigned int out[8];
  if (sf > 0.0f) {
    const float inv_sf = __frcp_rn(sf);        /* one reciprocal per block */
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float lo = qdq_e2m1(vals[2 * j], sf, inv_sf);
      float hi = qdq_e2m1(vals[2 * j + 1], sf, inv_sf);
      out[j] = as_u32(__floats2bfloat162_rn(lo, hi));
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = 0u;
  }
  *reinterpret_cast<uint4*>(dst) = make_uint4(out[0], out[1], out[2], out[3]);
  *reinterpret_cast<uint4*>(dst + 8) = make_uint4(out[4], out[5], out[6], out[7]);
}

/* Split-K GEMM1 reduction (rungs 20/21/28/29/34/35): FOUR threads per
 * 16-element activation block (one quarter each), a 4x widening of the
 * original one-thread version whose 2-block grid made the stage a ~8 us
 * latency floor on tiny shapes (job A stage profile). Each lane sums its
 * four elements over the sk fp32 slices in FIXED slice order, the block amax
 * completes over the aligned 4-lane group via shuffle, and the eight payload
 * bytes reassemble with one shfl_down + two 4-byte stores. The per-element
 * arithmetic chain is unchanged, so the packed activation bytes match the
 * original kernel (the fmax tree can only differ from the serial max on a
 * mixed +-0 block, where both forms emit an all-zero payload and the sign
 * bit of a zero scale cannot change any product). */
/* Occupancy note: 3 blocks/SM at 256 threads need <= 85 registers. The
 * folded-prescale form hit 90 (job Z: 3->2 blocks, -0.7% on the SK band) and
 * an explicit __launch_bounds__(256, 3) squeezed the non-folded form into
 * 80 regs WITH 40/68-byte spills (job AB: fixed_0 0.990 -- spills hurt this
 * latency-bound kernel more than the occupancy floor helps). The non-folded
 * chain alone is the fix: no bounds annotation, ptxas lands <= 85 naturally
 * (72 at C10), keeping 3 blocks/SM and spill-free. */
template <bool BRLESS = false>
__global__ void k_act_packed(const Params p) {
  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  const int nb = p.I >> 4;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;   /* (r, b, quarter) */
  /* Bound by Rmax (a parameter register), not by the device-side offs[E]
   * read: rows past the valid routes never enter a GEMM2 tile. pexp of such
   * a row is uninitialized (scatter only writes valid positions), so the
   * expert id MUST be range-checked before indexing alpha/beta/limit. */
  if (idx >= p.Rmax * nb * 4) return;
  const int q4 = idx & 3;
  const int b = (idx >> 2) % nb;
  const int r = idx / (nb * 4);
  const int e = p.pexp[r];
  if ((unsigned)e >= (unsigned)p.E) return;
  const float alpha = p.alpha[e], beta = p.beta[e];
  const float inv_alpha = __frcp_rn(alpha), inv_beta = __frcp_rn(beta);
  const float lim = p.has_limit ? p.limit[e] : 0.0f;
  const size_t slice_stride = (size_t)p.Rmax * (2 * p.I);
  const int col = (b << 4) + (q4 << 2);
  float up[4], gate[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) { up[q] = 0.0f; gate[q] = 0.0f; }
  for (int sl = 0; sl < p.sk; ++sl) {
    const float* ps = p.part + sl * slice_stride + (size_t)r * (2 * p.I);
    const float4 u4 = *reinterpret_cast<const float4*>(ps + col);
    const float4 g4 = *reinterpret_cast<const float4*>(ps + p.I + col);
    up[0] += u4.x; up[1] += u4.y; up[2] += u4.z; up[3] += u4.w;
    gate[0] += g4.x; gate[1] += g4.y; gate[2] += g4.z; gate[3] += g4.w;
  }
  /* C10: pair-packed activation + packed quantize multiply (bit-identical to
   * activation_fast; the quantize change is the ~1.5-ulp class documented at
   * the f2_mul helpers). The 4-element amax tree equals the serial max on
   * non-NaN inputs (activation outputs are bounded; see the k_act_packed
   * header comment on +-0). */
  float vals[4];
  if (p.has_limit) {
    const float u0 = fminf(fmaxf(up[0], -lim), lim);
    const float u1 = fminf(fmaxf(up[1], -lim), lim);
    const float u2 = fminf(fmaxf(up[2], -lim), lim);
    const float u3 = fminf(fmaxf(up[3], -lim), lim);
    const float2 a0 = activation2_nf(p.act_type,
        make_float2(fminf(gate[0], lim), fminf(gate[1], lim)),
        make_float2(u0, u1), alpha, beta, inv_alpha, inv_beta);
    const float2 a1 = activation2_nf(p.act_type,
        make_float2(fminf(gate[2], lim), fminf(gate[3], lim)),
        make_float2(u2, u3), alpha, beta, inv_alpha, inv_beta);
    vals[0] = a0.x; vals[1] = a0.y; vals[2] = a1.x; vals[3] = a1.y;
  } else {
    const float2 a0 = activation2_nf(p.act_type, make_float2(gate[0], gate[1]),
                                     make_float2(up[0], up[1]), alpha, beta,
                                     inv_alpha, inv_beta);
    const float2 a1 = activation2_nf(p.act_type, make_float2(gate[2], gate[3]),
                                     make_float2(up[2], up[3]), alpha, beta,
                                     inv_alpha, inv_beta);
    vals[0] = a0.x; vals[1] = a0.y; vals[2] = a1.x; vals[3] = a1.y;
  }
  float amax = fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])),
                     fmaxf(fabsf(vals[2]), fabsf(vals[3])));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  const int sf_code = f32_to_e4m3(fminf(amax * (1.0f / 6.0f), 448.0f));
  const float sf = e4m3_val(sf_code);
  unsigned int hw = 0u;
  /* C11c: branchless form -- see the f4_body epilogue comment. */
  if (BRLESS || sf > 0.0f) {
    const float inv_sf = BRLESS ? __frcp_rn(fmaxf(sf, 1.17549435e-38f))
                                : __frcp_rn(sf);
    const float2 inv2 = make_float2(inv_sf, inv_sf);
    const float2 q0 = f2_mul(make_float2(vals[0], vals[1]), inv2);
    const float2 q1 = f2_mul(make_float2(vals[2], vals[3]), inv2);
    hw = q_e2m1_pack8(q0.x, q0.y, q1.x, q1.y,
                      q1.x, q1.y, q0.x, q0.y) & 0xFFFFu;
  }
  const unsigned int other = __shfl_down_sync(0xffffffffu, hw, 1);
  if ((q4 & 1) == 0) {
    *reinterpret_cast<unsigned int*>(p.act_q + (size_t)r * (p.I >> 1)
                                     + ((size_t)b << 3) + (q4 << 1)) =
        hw | (other << 16);
  }
  if (q4 == 0) p.act_sf[(size_t)r * nb + b] = (unsigned char)sf_code;
}

/* -------------------------------------- routing-weighted reduction to output */
/* 8 output elements per thread: one uint4 (16 B) load per route instead of four
 * 4-byte loads, a quarter of the threads and four times the memory-level
 * parallelism per thread; the K-route FP32 accumulation order per element is
 * unchanged from the 2-element version (sequential over k, one FMA per term). */
__global__ void k_finalize(const Params p) {
  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  const int nh = p.H >> 3;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nh) return;
  const int t = blockIdx.y;
  const int h8 = i << 3;
  const int32_t* ro = p.route_of + (size_t)t * p.K;
  const float* sc = p.scores + (size_t)t * p.K;
  float acc[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) acc[q] = 0.0f;
  if (p.K == 8) {
    /* Batch the route/score reads (two 32-byte sectors) so the eight rout
     * loads issue back-to-back instead of each waiting on its route id. */
    int rr[8];
    float ss[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { rr[k] = ro[k]; ss[k] = sc[k]; }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (rr[k] < 0) continue;
      const float s = ss[k];
      const uint4 raw =
          *reinterpret_cast<const uint4*>(p.rout + (size_t)rr[k] * p.H + h8);
      const unsigned int* w = reinterpret_cast<const unsigned int*>(&raw);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&w[q]);
        acc[2 * q] = fmaf(__bfloat162float(v.x), s, acc[2 * q]);
        acc[2 * q + 1] = fmaf(__bfloat162float(v.y), s, acc[2 * q + 1]);
      }
    }
  } else {
    /* General fallback (the declared contract fixes topk = 8). */
    for (int k = 0; k < p.K; ++k) {
      int r = ro[k];
      if (r < 0) continue;
      float s = sc[k];
      const uint4 raw =
          *reinterpret_cast<const uint4*>(p.rout + (size_t)r * p.H + h8);
      const unsigned int* w = reinterpret_cast<const unsigned int*>(&raw);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&w[q]);
        acc[2 * q] = fmaf(__bfloat162float(v.x), s, acc[2 * q]);
        acc[2 * q + 1] = fmaf(__bfloat162float(v.y), s, acc[2 * q + 1]);
      }
    }
  }
  unsigned int outw[4];
#pragma unroll
  for (int q = 0; q < 4; ++q)
    outw[q] = as_u32(__floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]));
  *reinterpret_cast<uint4*>(p.out + (size_t)t * p.H + h8) =
      make_uint4(outw[0], outw[1], outw[2], outw[3]);
}

/* REDF rungs (46/47): the routing-weighted fp32 sum already lives in acc32
 * (the GEMM2 epilogue reduced it there); this pass performs the single final
 * BF16 rounding and writes the caller's output for EVERY token, including
 * tokens without a valid route (whose accumulator stayed at zero, matching
 * the reference's zero-initialized fp32 output). One thread per 8 elements:
 * two float4 loads + one uint4 store, fully coalesced and bandwidth-bound at
 * ~copy speed (measured 6.2 TB/s for contiguous 256 MB streams). */
/* Streaming variant of k_finalize for rout >> L2 (host-gated at
 * R*H*2 >= 256 MB): 16 elements/thread + evict-first .cs loads/stores.
 * Measured +10us fixed_6 / +9us domain_3; the .cs policy LOSES when rout
 * is L2-resident (it kills the GEMM2-write -> finalize-read handoff), so
 * the 8-element default form stays for everything below the gate. */
__global__ void k_finalize16(const Params p) {
  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  /* 16 output elements per thread (two adjacent uint4 per route): halves
   * the thread count again and doubles the per-thread MLP; rout (268 MB on
   * fixed_6) streams through L2 once and out is never re-read, so both
   * sides use the evict-first .cs policy. The K-route FP32 accumulation
   * order per element is unchanged (sequential over k, one FMA per term),
   * so results are bit-identical to the 8-element form. H % 16 == 0 holds
   * for every declared hidden size (2048/4096/7168). */
  const int nh = p.H >> 4;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nh) return;
  const int t = blockIdx.y;
  const int h16 = i << 4;
  const int32_t* ro = p.route_of + (size_t)t * p.K;
  const float* sc = p.scores + (size_t)t * p.K;
  float acc[16];
#pragma unroll
  for (int q = 0; q < 16; ++q) acc[q] = 0.0f;
  if (p.K == 8) {
    /* Batch the route/score reads (two 32-byte sectors) so the sixteen
     * rout loads issue back-to-back instead of each waiting on its id. */
    int rr[8];
    float ss[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { rr[k] = ro[k]; ss[k] = sc[k]; }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (rr[k] < 0) continue;
      const float s = ss[k];
      const uint4* rp =
          reinterpret_cast<const uint4*>(p.rout + (size_t)rr[k] * p.H + h16);
      const uint4 raw0 = __ldcs(rp);
      const uint4 raw1 = __ldcs(rp + 1);
      const unsigned int* w0 = reinterpret_cast<const unsigned int*>(&raw0);
      const unsigned int* w1 = reinterpret_cast<const unsigned int*>(&raw1);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&w0[q]);
        acc[2 * q] = fmaf(__bfloat162float(v.x), s, acc[2 * q]);
        acc[2 * q + 1] = fmaf(__bfloat162float(v.y), s, acc[2 * q + 1]);
        __nv_bfloat162 u = *reinterpret_cast<const __nv_bfloat162*>(&w1[q]);
        acc[8 + 2 * q] = fmaf(__bfloat162float(u.x), s, acc[8 + 2 * q]);
        acc[8 + 2 * q + 1] = fmaf(__bfloat162float(u.y), s, acc[8 + 2 * q + 1]);
      }
    }
  } else {
    /* General fallback (the declared contract fixes topk = 8). */
    for (int k = 0; k < p.K; ++k) {
      int r = ro[k];
      if (r < 0) continue;
      float s = sc[k];
      const uint4* rp =
          reinterpret_cast<const uint4*>(p.rout + (size_t)r * p.H + h16);
      const uint4 raw0 = __ldcs(rp);
      const uint4 raw1 = __ldcs(rp + 1);
      const unsigned int* w0 = reinterpret_cast<const unsigned int*>(&raw0);
      const unsigned int* w1 = reinterpret_cast<const unsigned int*>(&raw1);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&w0[q]);
        acc[2 * q] = fmaf(__bfloat162float(v.x), s, acc[2 * q]);
        acc[2 * q + 1] = fmaf(__bfloat162float(v.y), s, acc[2 * q + 1]);
        __nv_bfloat162 u = *reinterpret_cast<const __nv_bfloat162*>(&w1[q]);
        acc[8 + 2 * q] = fmaf(__bfloat162float(u.x), s, acc[8 + 2 * q]);
        acc[8 + 2 * q + 1] = fmaf(__bfloat162float(u.y), s, acc[8 + 2 * q + 1]);
      }
    }
  }
  uint4 out0, out1;
  unsigned int* o0 = reinterpret_cast<unsigned int*>(&out0);
  unsigned int* o1 = reinterpret_cast<unsigned int*>(&out1);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    o0[q] = as_u32(__floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]));
    o1[q] = as_u32(__floats2bfloat162_rn(acc[8 + 2 * q], acc[8 + 2 * q + 1]));
  }
  __stcs(reinterpret_cast<uint4*>(p.out + (size_t)t * p.H + h16), out0);
  __stcs(reinterpret_cast<uint4*>(p.out + (size_t)t * p.H + h16) + 1, out1);
}

__global__ void k_convert(const Params p) {
  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  const size_t n8 = ((size_t)p.T * (size_t)p.H) >> 3;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8;
       i += stride) {
    const float4 a = *reinterpret_cast<const float4*>(p.acc32 + (i << 3));
    const float4 b = *reinterpret_cast<const float4*>(p.acc32 + (i << 3) + 4);
    const unsigned int w0 = as_u32(__floats2bfloat162_rn(a.x, a.y));
    const unsigned int w1 = as_u32(__floats2bfloat162_rn(a.z, a.w));
    const unsigned int w2 = as_u32(__floats2bfloat162_rn(b.x, b.y));
    const unsigned int w3 = as_u32(__floats2bfloat162_rn(b.z, b.w));
    *reinterpret_cast<uint4*>(p.out + (i << 3)) =
        make_uint4(w0, w1, w2, w3);
  }
}

/* ---------------------------------------------------------------- GEMM core */
/* Grouped GEMM over the per-expert route lists.
 *   GEMM1: A = decoded hidden bf16 rows gathered by token id, B = prepared
 *          gemm1 weights (gate/up interleave + 32-row shuffle + SF swizzle),
 *          N = 2I, K = H, C = FP32 pre-activation.
 *   GEMM2: A = QDQ'd activation bf16 rows (route order, contiguous),
 *          B = prepared gemm2 weights, N = H, K = I,
 *          C = BF16 (the reference materializes every route as BF16).
 * Weights are decoded E2M1*E4M3 -> bf16 in shared memory, which is exact
 * (at most 6 significand bits), so tensor-core products equal the reference's
 * FP32 dequantized GEMM up to accumulation order.
 */
DINLINE int phys_row(int rl, int half_n, bool gated) {
  int ii = gated ? ((rl < half_n) ? (rl << 1) : (((rl - half_n) << 1) + 1)) : rl;
  return (ii & ~31) + ((ii & 3) << 3) + ((ii >> 2) & 7);
}

DINLINE size_t sf_offset(int pr, int c, int c4) {
  return ((size_t)(pr >> 7) * c4 + (size_t)(c >> 2)) * 512 +
         (size_t)(pr & 31) * 16 + (size_t)((pr >> 5) & 3) * 4 + (size_t)(c & 3);
}

template <int BM, int BN, int BK, int WM, int WN, bool GEMM1>
__global__ void __launch_bounds__(WM * WN * 32) k_gemm(const Params p) {
  constexpr int THREADS = WM * WN * 32;
  constexpr int WTM = BM / WM;             /* rows per warp      */
  constexpr int WTN = BN / WN;             /* cols per warp      */
  constexpr int MT = WTM / 16;             /* m16 tiles per warp */
  constexpr int NT = WTN / 8;              /* n8 tiles per warp  */
  constexpr int ACH = BK / 8;              /* 16B chunks per A row */
  constexpr int BCH = BK / 32;             /* 16B payload tasks per B row */
  constexpr int A_BYTES = BM * BK * 2;
  constexpr int B_BYTES = BN * BK * 2;
  constexpr int BTASKS = BN * BCH;
  constexpr int BT = (BTASKS + THREADS - 1) / THREADS;   /* B tasks per thread */
  constexpr int SFSTRIDE = (BK / 64) * 512;  /* scale bytes per k-tile */
  constexpr int STAGES = 2;

  const int total_tiles = p.toffs[p.E];
  /* This rung tiles the full logical row space (N = 2I for GEMM1, H for
   * GEMM2), so its item space uses N directly. */
  const int N = GEMM1 ? (2 * p.I) : p.H;
  const int n_tiles = N / BN;   /* full logical row space: 2I for GEMM1 */
  const long long total_items = (long long)total_tiles * n_tiles;
  if (total_items <= 0) return;
  const int K = GEMM1 ? p.H : p.I;
  const int half_n = p.I;
  const int c4 = K >> 6;
  const int nk = K / BK;
  const __nv_bfloat16* A = GEMM1 ? p.hid_bf : p.act;
  const uint8_t* Wq = GEMM1 ? p.w1_q : p.w2_q;
  const uint8_t* Wsf = GEMM1 ? p.w1_sf : p.w2_sf;
  const size_t eq_stride = (size_t)N * (K >> 1);
  const size_t esf_stride = (size_t)N * (K >> 4);

  extern __shared__ __align__(16) unsigned char smem_raw[];
  unsigned char* const a_base = smem_raw;
  unsigned char* const b_base = smem_raw + STAGES * A_BYTES;
  int32_t* s_row = reinterpret_cast<int32_t*>(smem_raw + STAGES * (A_BYTES + B_BYTES));
  /* Format tables in shared memory: the decode does 16 table reads per thread
   * per k-tile and divergent indices make global/L1 reads far more expensive
   * than shared reads, where they were co-limiting with the tensor cores. */
  unsigned int* s_e2m1 = reinterpret_cast<unsigned int*>(
      smem_raw + STAGES * (A_BYTES + B_BYTES) + BM * 4);
  unsigned int* s_e4m3 = s_e2m1 + 256;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp % WM;
  const int wn = warp / WM;

  for (int i = tid; i < 256; i += THREADS) {
    s_e2m1[i] = g_e2m1_pair[i];
    s_e4m3[i] = g_e4m3_bf2[i];
  }

  /* Persistent item index: this grid is sized by the host from an upper bound,
   * so blocks simply walk the (m-tile, n-tile) item space in expert order,
   * which keeps one expert's weights hot in L2 across its blocks. */
  for (int item = blockIdx.y * gridDim.x + blockIdx.x; item < total_items;
       item += gridDim.x * gridDim.y) {
    const int j = item / n_tiles;
    const int n0 = (item - j * n_tiles) * BN;
    const int e = p.texp[j];
    const int m0 = p.tm0[j];   /* p.bm == BM for this rung */
    const int row_base = p.offs[e] + m0;
    const int rows_valid = p.counts[e] - m0;
    const uint8_t* Bq = Wq + (size_t)e * eq_stride;
    const uint8_t* Bsf = Wsf + (size_t)e * esf_stride;

    for (int i = tid; i < BM; i += THREADS) {
      int ii = (i < rows_valid) ? i : 0;
      s_row[i] = GEMM1 ? p.ptok[row_base + ii] : (row_base + ii);
    }

    /* Per-thread weight tasks: addresses and swizzled destinations are
     * loop-invariant across k, only the payload/scale loads move. */
    const uint8_t* qsrc[BT];
    const uint8_t* sfsrc[BT];
    unsigned char* bdst[BT];
    int bchunk[BT][4];
    int bscale[BT];
    bool bvalid[BT];
#pragma unroll
    for (int t = 0; t < BT; ++t) {
      const int task = tid + t * THREADS;
      bvalid[t] = (task < BTASKS);
      if (bvalid[t]) {
        const int n_row = task / BCH;
        const int half = task - n_row * BCH;
        const int pr = phys_row(n0 + n_row, half_n, GEMM1);
        qsrc[t] = Bq + (size_t)pr * (K >> 1) + half * 16;
        sfsrc[t] = Bsf + sf_offset(pr, 0, c4) + (half >> 1) * 512;
        bscale[t] = half * 2 - (half >> 1) * 4;   /* byte index inside the word */
        bdst[t] = b_base + n_row * (BK * 2);
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          const int chunk = half * 4 + cc;
          bchunk[t][cc] = (chunk ^ (n_row & 7)) * 16;
        }
      }
    }
    __syncthreads();

#define LOAD_A(LKT, DST)                                                        \
    do {                                                                        \
      const int k0_ = (LKT) * BK;                                               \
      for (int task = tid; task < BM * ACH; task += THREADS) {                  \
        int r_ = task / ACH;                                                    \
        int c_ = task - r_ * ACH;                                               \
        cp_async16((DST) + r_ * (BK * 2) + ((c_ ^ (r_ & 7)) * 16),               \
                   A + (size_t)s_row[r_] * K + k0_ + c_ * 8);                   \
      }                                                                         \
    } while (0)

    /* Issue the weight loads for k-tile LKT into registers (latency is then
     * hidden by the current tile's tensor-core work). */
#define FETCH_B(LKT, PAY, SFW_)                                                 \
    do {                                                                        \
      const int off_ = (LKT) * (BK >> 1);                                       \
      _Pragma("unroll")                                                         \
      for (int t = 0; t < BT; ++t) {                                            \
        if (bvalid[t]) {                                                        \
          (PAY)[t] = *reinterpret_cast<const uint4*>(qsrc[t] + off_);           \
          (SFW_)[t] = *reinterpret_cast<const unsigned int*>(                   \
              sfsrc[t] + (size_t)(LKT) * SFSTRIDE);                             \
        }                                                                       \
      }                                                                         \
    } while (0)

#define STORE_B(STAGE, PAY, SFW_)                                               \
    do {                                                                        \
      _Pragma("unroll")                                                         \
      for (int t = 0; t < BT; ++t) {                                            \
        if (bvalid[t]) {                                                        \
          const unsigned char* bytes_ =                                         \
              reinterpret_cast<const unsigned char*>(&(PAY)[t]);                \
          _Pragma("unroll")                                                     \
          for (int cc = 0; cc < 4; ++cc) {                                      \
            const int li_ = bscale[t] + (cc >> 1);                              \
            const unsigned int code_ = ((SFW_)[t] >> (8 * li_)) & 0xFF;         \
            unsigned int sb_ = s_e4m3[code_];                               \
            __nv_bfloat162 w_ = *reinterpret_cast<__nv_bfloat162*>(&sb_);       \
            unsigned int words_[4];                                             \
            _Pragma("unroll")                                                   \
            for (int q = 0; q < 4; ++q) {                                       \
              unsigned int pair_ = s_e2m1[bytes_[cc * 4 + q]];             \
              words_[q] = as_u32(__hmul2(                                       \
                  *reinterpret_cast<__nv_bfloat162*>(&pair_), w_));             \
            }                                                                   \
            *reinterpret_cast<uint4*>((STAGE) + (bdst[t] - b_base) +            \
                                      bchunk[t][cc]) =                          \
                make_uint4(words_[0], words_[1], words_[2], words_[3]);         \
          }                                                                     \
        }                                                                       \
      }                                                                         \
    } while (0)

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[mi][nt][q] = 0.0f;

    /* Two register buffers so a weight tile's global loads are issued one
     * k-tile before they are decoded: the decode of tile kt+1 must not stall
     * on its own load, otherwise the ~600 ns memory latency is serialised
     * against the tensor-core work of every tile (measured as ~45% of the
     * mma.sync ceiling). */
    uint4 pay0[BT], pay1[BT];
    unsigned int sfw0[BT], sfw1[BT];
    FETCH_B(0, pay0, sfw0);
    STORE_B(b_base, pay0, sfw0);
    LOAD_A(0, a_base);
    cp_async_commit();

    for (int kt = 0; kt < nk; ++kt) {
      const int st = kt & 1;
      if (kt + 1 < nk) {
        FETCH_B(kt + 1, pay1, sfw1);          /* in flight during this tile */
        LOAD_A(kt + 1, a_base + ((kt + 1) & 1) * A_BYTES);
      }
      cp_async_commit();
      cp_async_wait<1>();
      __syncthreads();

      unsigned char* aS = a_base + st * A_BYTES;
      unsigned char* bS = b_base + st * B_BYTES;
      unsigned int a[MT][4], b[NT / 2][4];
#pragma unroll
      for (int ks = 0; ks < BK / 16; ++ks) {
        const int arow = wm * WTM + ((lane < 16) ? lane : (lane - 16));
        const int achunk = ks * 2 + ((lane < 16) ? 0 : 1);
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
          int row = arow + mi * 16;
          ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3],
                      aS + row * (BK * 2) + ((achunk ^ (row & 7)) * 16));
        }
        const int brow = wn * WTN + ((lane >> 4) & 1) * 8 + (lane & 7);
        const int bchunk_ = ks * 2 + ((lane >> 3) & 1);
#pragma unroll
        for (int np = 0; np < NT / 2; ++np) {
          int row = brow + np * 16;
          ldmatrix_x4(b[np][0], b[np][1], b[np][2], b[np][3],
                      bS + row * (BK * 2) + ((bchunk_ ^ (row & 7)) * 16));
        }
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma_bf16(acc[mi][nt][0], acc[mi][nt][1], acc[mi][nt][2], acc[mi][nt][3],
                     a[mi][0], a[mi][1], a[mi][2], a[mi][3],
                     b[nt >> 1][(nt & 1) * 2], b[nt >> 1][(nt & 1) * 2 + 1]);
      }
      __syncthreads();
      /* Decode the tile fetched above into the stage the next iteration uses. */
      STORE_B(b_base + ((kt + 1) & 1) * B_BYTES, pay1, sfw1);
#pragma unroll
      for (int t = 0; t < BT; ++t) { pay0[t] = pay1[t]; sfw0[t] = sfw1[t]; }
    }
#undef LOAD_A
#undef FETCH_B
#undef STORE_B

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
      const int row_local = wm * WTM + mi * 16 + (lane >> 2);
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        const int rl = row_local + hr * 8;
        if (rl >= rows_valid) continue;
        const size_t r = (size_t)(row_base + rl);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int col = n0 + wn * WTN + nt * 8 + ((lane & 3) << 1);
          float v0 = acc[mi][nt][hr * 2 + 0];
          float v1 = acc[mi][nt][hr * 2 + 1];
          if (GEMM1) {
            *reinterpret_cast<float2*>(p.fc1 + r * (2 * (size_t)p.I) + col) =
                make_float2(v0, v1);
          } else {
            *reinterpret_cast<__nv_bfloat162*>(p.rout + r * p.H + col) =
                __floats2bfloat162_rn(v0, v1);
          }
        }
      }
    }
    __syncthreads();
  }
}

/* --------------------------------------------------- tcgen05 (5th gen TC) --- */
/* Verified on L20D: `-gencode arch=compute_103a,code=sm_103a` accepts
 * tcgen05.alloc/ld/mma/commit and kind::mxf4nvf4.block_scale.scale_vec::4X,
 * while plain sm_103 rejects all of them. These wrappers are the plumbing for
 * a tensor-core path whose operands are read directly from shared memory (no
 * ldmatrix traffic) and whose accumulators live in TMEM (no register
 * accumulators), which is what the warp-mma path is limited by. */
DINLINE unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

/* One warp allocates `ncols` TMEM columns; the address lands in shared memory. */
DINLINE void tmem_alloc(unsigned* slot_smem, int ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :: "r"(smem_u32(slot_smem)), "r"(ncols));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
DINLINE void tmem_dealloc(unsigned tmem, int ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
               :: "r"(tmem), "r"(ncols));
}

DINLINE void mbar_init(unsigned bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
DINLINE void mbar_wait(unsigned bar, unsigned phase) {
  unsigned done = 0;
  while (!done) {
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.u32 %0, 1, 0, p;\n\t}"
                 : "=r"(done) : "r"(bar), "r"(phase));
  }
}

/* Shared-memory matrix descriptor: start address (14 bits, 16-byte granular),
 * leading/stride byte offsets, and the swizzle mode. The operand tiles below
 * use 128-byte rows, so SBO = 8 rows * 128 B and the 128-byte swizzle mode;
 * this is the same encoding shape the published production kernels use. */
DINLINE unsigned long long make_smem_desc(unsigned addr, unsigned sbo, unsigned lbo,
                                          unsigned mode) {
  return (unsigned long long)((addr >> 4) & 0x3FFF)
       | ((unsigned long long)((lbo >> 4) & 0x3FFF) << 16)
       | ((unsigned long long)((sbo >> 4) & 0x3FFF) << 32)
       | (1ULL << 46)
       | ((unsigned long long)(mode & 7) << 61);
}

/* Instruction descriptor for tcgen05.mma: c/a/b formats, operand majors and
 * the M/N sizes. Field positions follow the published 32-bit layout. */
DINLINE unsigned make_instr_desc(unsigned c_fmt, unsigned a_fmt, unsigned b_fmt,
                                 unsigned n_dim, unsigned m_dim, unsigned m_shift,
                                 unsigned extra) {
  return (c_fmt << 4) | (a_fmt << 7) | (b_fmt << 10) | (n_dim << 17)
       | (m_dim << m_shift) | extra;
}

DINLINE void mma_f16_ss(unsigned tmem_d, unsigned long long a_desc,
                        unsigned long long b_desc, unsigned idesc, unsigned enable_d) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
               :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_d));
}

DINLINE void mma_mxf4nvf4_ss(unsigned tmem_d, unsigned long long a_desc,
                             unsigned long long b_desc, unsigned idesc,
                             unsigned sfa_tmem, unsigned sfb_tmem, unsigned enable_d) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
               "[%0], %1, %2, %3, [%4], [%5], p;\n\t}\n"
               :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(idesc),
                  "r"(sfa_tmem), "r"(sfb_tmem), "r"(enable_d));
}

/* Signal that all MMAs issued so far by this CTA have retired. */
DINLINE void mma_commit(unsigned bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(bar));
}

/* Copy scale factors from shared memory into TMEM (32 lanes x 128 b per warp,
 * four warps). */
DINLINE void tmem_cp_32x128b_warpx4(unsigned tmem_dst, unsigned long long s_desc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
               :: "r"(tmem_dst), "l"(s_desc));
}

DINLINE void tc_fence_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;"); }
DINLINE void tc_fence_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;"); }

/* Read 32 TMEM columns for this warp's 32 lanes. */
DINLINE void tmem_ld_x32(unsigned taddr, float* v) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                 "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]),
                 "=f"(v[8]), "=f"(v[9]), "=f"(v[10]), "=f"(v[11]),
                 "=f"(v[12]), "=f"(v[13]), "=f"(v[14]), "=f"(v[15]),
                 "=f"(v[16]), "=f"(v[17]), "=f"(v[18]), "=f"(v[19]),
                 "=f"(v[20]), "=f"(v[21]), "=f"(v[22]), "=f"(v[23]),
                 "=f"(v[24]), "=f"(v[25]), "=f"(v[26]), "=f"(v[27]),
                 "=f"(v[28]), "=f"(v[29]), "=f"(v[30]), "=f"(v[31])
               : "r"(taddr));
}
DINLINE void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }

/* Elect one lane of the warp to issue the (single-threaded) MMA sequence. */
DINLINE unsigned elect_one() {
  unsigned leader = 0;
  asm volatile("{\n\t.reg .pred p;\n\telect.sync _|p, 0xffffffff;\n\t"
               "selp.u32 %0, 1, 0, p;\n\t}" : "=r"(leader));
  return leader;
}

/* --------------------------------------- tcgen05 grouped GEMM (5th gen TC) --- */
/* Option: tokens are the M operand (TMEM lanes) and weight features are the N
 * operand (TMEM columns), so D[token][feature] lands in TMEM and the whole
 * post-GEMM chain (clamp, activation, NVFP4 block-16 QDQ for GEMM1; the BF16
 * route materialization for GEMM2) is intra-thread work with no cross-lane
 * reduction and no transpose.
 *
 * Compared with the warp-mma rung this removes the ldmatrix operand traffic
 * (the tensor core reads shared memory directly), removes the register
 * accumulators (they live in TMEM) and doubles the issue rate; with
 * kind::mxf4nvf4 the weights could additionally stay packed as FP4.
 *
 * Descriptor/instruction encodings are produced by make_smem_desc/make_instr_desc
 * so the exact field values stay in one place; experiments/tc_probe.py
 * determines them on the actual device.
 */
/* Encodings verified on L20D by experiments/tc_probe.py (job dv_e035fdf23714):
 * with M=128, N=32, K=64 bf16 operands in SW128 K-major shared memory, the
 * instruction descriptor 0x08080490 (= c_format F32, a/b format BF16,
 * n_dim=N>>3 at bits 17-22, m_dim=M>>4 at bits 24-29) together with the
 * shared-memory descriptor 0x4000404000000000 (= SBO 1024, bit 46, swizzle
 * mode 2, LBO 0) reproduced the reference product with max|diff| = 0.0.
 * 41 other field combinations either mismatched or raised
 * cudaErrorIllegalInstruction, which is why each candidate ran in its own
 * process. */
#ifndef TC_SBO
#define TC_SBO 1024u
#define TC_MODE 2u
#define TC_LBO 0u
#define TC_C_FMT 1u
#define TC_A_FMT 1u
#define TC_B_FMT 1u
#define TC_M_SHIFT 24u
#define TC_EXTRA 0u
#endif

template <int BN, bool GEMM1, int NSUB = 1, int BK = 64>
__global__ void __launch_bounds__(128) k_gemm_tc(const Params p) {
  /* NSUB token sub-tiles share one decoded weight tile: the fp4->bf16 decode
   * and the weight traffic are then amortised over NSUB*128 tokens instead of
   * 128, which the ablation showed to be the dominant pipeline cost. Each
   * sub-tile gets its own TMEM accumulator column range (128 lanes each). */
  constexpr int BM = 128 * NSUB;       /* tokens per CTA                  */
  constexpr int NROW = GEMM1 ? (2 * BN) : BN;   /* weight rows per tile    */
  constexpr int A_BYTES = BM * BK * 2;
  constexpr int B_BYTES = NROW * BK * 2;
  constexpr int STAGES = 2;
  constexpr int TSUB = GEMM1 ? (2 * BN) : BN;   /* TMEM columns per sub-tile */
  constexpr int TCOLS = TSUB * NSUB;

  const int total_tiles = p.toffs[p.E];
  const int N = GEMM1 ? (2 * p.I) : p.H;
  /* GEMM1 covers BN *intermediate* columns per CTA (up and gate are two
   * weight tiles of BN rows each), so its item space is over I, not 2I. */
  const int n_tiles = (GEMM1 ? p.I : p.H) / BN;
  const long long total_items = (long long)total_tiles * n_tiles;
  if (total_items <= 0) return;
  const int K = GEMM1 ? p.H : p.I;
  const int half_n = p.I;
  const int c4 = K >> 6;
  const int nk = K / BK;
  const __nv_bfloat16* A = GEMM1 ? p.hid_bf : p.act;
  const uint8_t* Wq = GEMM1 ? p.w1_q : p.w2_q;
  const uint8_t* Wsf = GEMM1 ? p.w1_sf : p.w2_sf;
  const size_t eq_stride = (size_t)N * (K >> 1);
  const size_t esf_stride = (size_t)N * (K >> 4);

  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* const a_base = smem_raw;                       /* 2 x A_BYTES */
  unsigned char* const b_base = smem_raw + STAGES * A_BYTES;    /* 2 x B_BYTES */
  unsigned* const s_tmem = reinterpret_cast<unsigned*>(b_base + STAGES * B_BYTES);
  unsigned long long* s_bar = reinterpret_cast<unsigned long long*>(s_tmem + 4);
  int32_t* s_row = reinterpret_cast<int32_t*>(s_bar + STAGES);
  unsigned int* s_e2m1 = reinterpret_cast<unsigned int*>(s_row + BM);
  unsigned int* s_e4m3 = s_e2m1 + 256;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  for (int i = tid; i < 256; i += 128) {
    s_e2m1[i] = g_e2m1_pair[i];
    s_e4m3[i] = g_e4m3_bf2[i];
  }

  if (tid == 0) mbar_init(smem_u32(s_bar), 1);
  if (tid < 32) tmem_alloc(s_tmem, TCOLS < 32 ? 32 : TCOLS);
  __syncthreads();
  const unsigned tmem = *s_tmem;
  const unsigned bar = smem_u32(s_bar);
  /* One commit per k-tile, and nk may be odd, so the barrier phase has to be
   * carried across items of the persistent loop instead of restarting at 0. */
  unsigned phase = 0;

  /* Weight rows of this tile, pre-resolved to physical rows. */
  constexpr int BCH = BK / 32;                 /* 16 B payload tasks per row */
  constexpr int BTASKS = NROW * BCH;
  constexpr int BT = (BTASKS + 127) / 128;
  constexpr int SFSTRIDE = (BK / 64) * 512;

  for (int item = blockIdx.y * gridDim.x + blockIdx.x; item < total_items;
       item += gridDim.x * gridDim.y) {
    const int j = item / n_tiles;
    const int n0 = (item - j * n_tiles) * BN;
    const int e = p.texp[j];
    const int m0 = p.tm0[j];   /* p.bm == BM for this rung */
    const int row_base = p.offs[e] + m0;
    const int rows_valid = p.counts[e] - m0;
    const uint8_t* Bq = Wq + (size_t)e * eq_stride;
    const uint8_t* Bsf = Wsf + (size_t)e * esf_stride;
    const float alpha = p.alpha[e], beta = p.beta[e];
    /* Per-item hoisted reciprocals for the fast activation (see
     * activation_fast): the reference divides per element. */
    const float inv_alpha = __frcp_rn(alpha), inv_beta = __frcp_rn(beta);
    const float lim = p.has_limit ? p.limit[e] : 0.0f;

    for (int i = tid; i < BM; i += 128) {
      int ii = (i < rows_valid) ? i : 0;
      s_row[i] = GEMM1 ? p.ptok[row_base + ii] : (row_base + ii);
    }
    __syncthreads();

#define TC_LOAD_A(LKT, DST)                                                       \
    do {                                                                          \
      const int k0_ = (LKT) * BK;                                                 \
      for (int task = tid; task < BM * (BK / 8); task += 128) {                   \
        int r_ = task / (BK / 8);                                                 \
        int c_ = task - r_ * (BK / 8);                                            \
        cp_async16((DST) + r_ * (BK * 2) + ((c_ ^ (r_ & 7)) * 16),                \
                   A + (size_t)s_row[r_] * K + k0_ + c_ * 8);                     \
      }                                                                           \
    } while (0)

    /* Decode one weight tile (NROW rows x BK) from packed FP4 into bf16 smem. */
/* Per-thread weight tasks: addresses, swizzled destinations and the scale byte
   * index are loop invariant, so only the payload/scale loads move per k-tile. */
    const uint8_t* qsrc[BT];
    const uint8_t* sfsrc[BT];
    unsigned char* bdst[BT];
    int bchunk[BT][4];
    int bscale[BT];
    bool bvalid[BT];
#pragma unroll
    for (int t = 0; t < BT; ++t) {
      const int task = tid + t * 128;
      bvalid[t] = (task < BTASKS);
      if (bvalid[t]) {
        const int n_row = task / BCH;
        const int half = task - n_row * BCH;
        const int rl = (GEMM1 && n_row >= BN) ? (n0 + n_row - BN + half_n)
                                              : (n0 + n_row);
        const int pr = phys_row(rl, half_n, GEMM1);
        qsrc[t] = Bq + (size_t)pr * (K >> 1) + half * 16;
        sfsrc[t] = Bsf + sf_offset(pr, 0, c4) + (half >> 1) * 512;
        bscale[t] = half * 2 - (half >> 1) * 4;
        bdst[t] = b_base + n_row * (BK * 2);
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          const int chunk = half * 4 + cc;
          bchunk[t][cc] = (chunk ^ (n_row & 7)) * 16;
        }
      }
    }

/* Issue one k-tile's weight loads into registers (latency is then covered by
   * the asynchronous tensor-core work of the current tile). */
#define TC_FETCH_B(LKT, PAY, SFW)                                               \
    do {                                                                        \
      const int off_ = (LKT) * (BK >> 1);                                       \
      _Pragma("unroll")                                                         \
      for (int t = 0; t < BT; ++t) {                                            \
        if (bvalid[t]) {                                                        \
          (PAY)[t] = *reinterpret_cast<const uint4*>(qsrc[t] + off_);           \
          (SFW)[t] = *reinterpret_cast<const unsigned int*>(                    \
              sfsrc[t] + (size_t)(LKT) * SFSTRIDE);                             \
        }                                                                       \
      }                                                                         \
    } while (0)

/* Decode the registers held above into the bf16 operand tile (exact: an E2M1
   * value times an E4M3 scale needs at most 6 significand bits). */
#define TC_STORE_B(STAGE, PAY, SFW)                                             \
    do {                                                                        \
      _Pragma("unroll")                                                         \
      for (int t = 0; t < BT; ++t) {                                            \
        if (bvalid[t]) {                                                        \
          const unsigned char* bytes_ =                                         \
              reinterpret_cast<const unsigned char*>(&(PAY)[t]);                \
          _Pragma("unroll")                                                     \
          for (int cc = 0; cc < 4; ++cc) {                                      \
            const int li_ = bscale[t] + (cc >> 1);                              \
            unsigned int sb_ = s_e4m3[((SFW)[t] >> (8 * li_)) & 0xFF];          \
            __nv_bfloat162 w_ = *reinterpret_cast<__nv_bfloat162*>(&sb_);       \
            unsigned int words_[4];                                             \
            _Pragma("unroll")                                                   \
            for (int q = 0; q < 4; ++q) {                                       \
              unsigned int pair_ = s_e2m1[bytes_[cc * 4 + q]];                  \
              words_[q] = as_u32(__hmul2(                                       \
                  *reinterpret_cast<__nv_bfloat162*>(&pair_), w_));             \
            }                                                                   \
            *reinterpret_cast<uint4*>((STAGE) + (bdst[t] - b_base) +            \
                                      bchunk[t][cc]) =                          \
                make_uint4(words_[0], words_[1], words_[2], words_[3]);         \
          }                                                                     \
        }                                                                       \
      }                                                                         \
    } while (0)

    uint4 pay0[BT], pay1[BT];
    unsigned int sfw0[BT], sfw1[BT];
    TC_FETCH_B(0, pay0, sfw0);
    TC_STORE_B(b_base, pay0, sfw0);
    TC_LOAD_A(0, a_base);
    cp_async_commit();

    for (int kt = 0; kt < nk; ++kt) {
      const int st = kt & 1;
      /* Iteration kt decodes into stage (kt+1)&1, which the MMA of iteration
       * kt-1 was reading, so that MMA must have retired first. */
      if (kt >= 1) {
        if (warp == 0 && lane == 0) mbar_wait(bar, phase);
        phase ^= 1;
        __syncthreads();
      }
      if (kt + 1 < nk) {
        TC_FETCH_B(kt + 1, pay1, sfw1);      /* in flight during this tile */
        TC_LOAD_A(kt + 1, a_base + ((kt + 1) & 1) * A_BYTES);
      }
      cp_async_commit();
      cp_async_wait<1>();
      __syncthreads();
      if (warp == 0 && elect_one()) {
        tc_fence_after_sync();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const unsigned a_stage = smem_u32(a_base + st * A_BYTES);
        const unsigned long long b_up =
            make_smem_desc(smem_u32(b_base + st * B_BYTES), TC_SBO, TC_LBO, TC_MODE);
        const unsigned long long b_gate =
            make_smem_desc(smem_u32(b_base + st * B_BYTES + BN * BK * 2),
                           TC_SBO, TC_LBO, TC_MODE);
        const unsigned idesc = make_instr_desc(
            TC_C_FMT, TC_A_FMT, TC_B_FMT, (BN >> 3), (128 >> 4), TC_M_SHIFT,
            TC_EXTRA);
        for (int sub = 0; sub < NSUB; ++sub) {
          const unsigned long long a_desc = make_smem_desc(
              a_stage + sub * (128 * BK * 2), TC_SBO, TC_LBO, TC_MODE);
          const unsigned acc_base = tmem + sub * TSUB;
          for (int ks = 0; ks < BK / 16; ++ks) {
            const unsigned enable = (kt == 0 && ks == 0) ? 0u : 1u;
            mma_f16_ss(acc_base, a_desc + 2 * ks, b_up + 2 * ks, idesc, enable);
            if (GEMM1)
              mma_f16_ss(acc_base + BN, a_desc + 2 * ks, b_gate + 2 * ks, idesc,
                         enable);
          }
        }
        mma_commit(bar);
      }
      /* Decode the tile fetched above while the tensor core is still busy. */
      if (kt + 1 < nk) {
        TC_STORE_B(b_base + ((kt + 1) & 1) * B_BYTES, pay1, sfw1);
#pragma unroll
        for (int t = 0; t < BT; ++t) { pay0[t] = pay1[t]; sfw0[t] = sfw1[t]; }
      }
    }
    if (warp == 0 && lane == 0) mbar_wait(bar, phase);
    /* Flip after the final wait too, so `phase` tracks (commits consumed) % 2
     * across items; see the same comment in k_gemm_f4. */
    phase ^= 1;
    tc_fence_after_sync();
    __syncthreads();
#undef TC_LOAD_A
#undef TC_FETCH_B
#undef TC_STORE_B

    /* ------------------------------- epilogue: TMEM -> activation/output ---- */
#pragma unroll 1
    for (int sub = 0; sub < NSUB; ++sub) {
    const unsigned taddr0 = tmem + sub * TSUB + ((unsigned)(warp * 32) << 16);
    const int token_local = sub * 128 + warp * 32 + lane;
    const bool tok_ok = (token_local < rows_valid);
    const size_t route = (size_t)(row_base + token_local);
#pragma unroll 1
    for (int c = 0; c < BN / 32; ++c) {
      float up[32];
      tmem_ld_x32(taddr0 + c * 32, up);
      tmem_ld_wait();
      if (GEMM1) {
        float gate[32];
        tmem_ld_x32(taddr0 + BN + c * 32, gate);
        tmem_ld_wait();
        __nv_bfloat16* dst = p.act + route * p.I + n0 + c * 32;
        if (tok_ok) {
          for (int blk = 0; blk < 2; ++blk) {
            float vals[16];
            float amax = 0.0f;
            for (int q = 0; q < 16; ++q) {
              float g = gate[blk * 16 + q], u = up[blk * 16 + q];
              if (p.has_limit) {
                u = fminf(fmaxf(u, -lim), lim);
                g = fminf(g, lim);
              }
              float a = activation(p, g, u, alpha, beta);
              vals[q] = a;
              amax = fmaxf(amax, fabsf(a));
            }
            float sf_pre = fminf(amax * (1.0f / 6.0f), 448.0f);
            float sf = e4m3_val(f32_to_e4m3(sf_pre));
            unsigned int outw[8];
            if (sf > 0.0f) {
              const float inv_sf = __frcp_rn(sf);   /* one reciprocal per block */
              for (int q = 0; q < 8; ++q)
                outw[q] = as_u32(__floats2bfloat162_rn(
                    qdq_e2m1(vals[2 * q], sf, inv_sf),
                    qdq_e2m1(vals[2 * q + 1], sf, inv_sf)));
            } else {
              for (int q = 0; q < 8; ++q) outw[q] = 0u;
            }
            *reinterpret_cast<uint4*>(dst + blk * 16) =
                make_uint4(outw[0], outw[1], outw[2], outw[3]);
            *reinterpret_cast<uint4*>(dst + blk * 16 + 8) =
                make_uint4(outw[4], outw[5], outw[6], outw[7]);
          }
        }
      } else {
        __nv_bfloat16* dst = p.rout + route * p.H + n0 + c * 32;
        if (tok_ok) {
          unsigned int outw[16];
          for (int q = 0; q < 16; ++q)
            outw[q] = as_u32(__floats2bfloat162_rn(up[2 * q], up[2 * q + 1]));
          for (int q = 0; q < 4; ++q)
            *reinterpret_cast<uint4*>(dst + q * 8) =
                make_uint4(outw[4 * q], outw[4 * q + 1], outw[4 * q + 2], outw[4 * q + 3]);
        }
      }
    }
    if (sub + 1 < NSUB) __syncthreads();
    }
    __syncthreads();
  }
  if (tid < 32) tmem_dealloc(tmem, TCOLS < 32 ? 32 : TCOLS);
}

/* ------------------------- tcgen05 block-scaled NVFP4 grouped GEMM (rung 7) --- */
/* Both operands stay packed as NVFP4 (E2M1 payload + E4M3 block-16 scale), so
 * the fp4->bf16 decode disappears -- the ablation (job dv_a2ac5b2e9d68)
 * attributed 1830 of 2663 cycles per k-tile to it and ~0 to the global loads --
 * and the same accumulator shape runs ~2.5x faster per K (measured ceilings,
 * job dv_bce0bdab86e1: 5262 TFLOPS block-scaled FP4 vs 2131 TFLOPS bf16 at two
 * CTAs/SM). GEMM1 also stops needing the separate hidden-state decode kernel,
 * because its A operand is the packed input tensor itself.
 *
 * Scale-factor plumbing, measured on this device by experiments/tc_sf_map.py
 * (job dv_53dee76b9f7a) and cross-checked against the published SM100 NVFP4 MoE
 * source under references/nvidia-sources (read for the layout only; all code
 * here is my own):
 *   - one `tcgen05.mma...kind::mxf4nvf4.block_scale.scale_vec::4X` consumes
 *     K=64 FP4 = four 16-element scale blocks and reads four TMEM columns
 *     [sf+4j .. sf+4j+3], one column per 32-lane quadrant, each holding the four
 *     E4M3 bytes of that quadrant's rows. The probe wrote 16 distinct known
 *     values with tcgen05.st and made the payload hot in exactly one scale
 *     block, so D[m]/16 identified that block's (column, byte) slot: block%4 is
 *     the byte inside the column and the column is the lane quadrant. Blocks
 *     whose columns were left unwritten contributed exactly zero, which is what
 *     pinned the four-column-per-instruction stride down.
 *   - the shared-memory staging `tcgen05.cp.cta_group::1.32x128b.warpx4`
 *     expects for tile row r, kset j (= block/4) of an NK-kset stage is
 *         off = ((sf_c*NK + j)*8 + sf_d)*16 + sf_g*4,  byte = block%4
 *         sf_c = (r%32)/8,  sf_d = r%8,  sf_g = r/32
 *     with descriptor SBO = NK*128, no swizzle mode, and one copy per kset whose
 *     source is 128*j bytes into the region and whose destination is TMEM
 *     columns sf+4*j. The 4-byte word of a row and kset is exactly what both
 *     scale layouts provide contiguously: the frozen weight swizzle keeps
 *     c%4 = 0..3 adjacent, and the row-major token/intermediate scales keep
 *     blocks 4j..4j+3 adjacent.
 *   - payload descriptors are unchanged from the bf16 rung (SBO 1024, swizzle
 *     mode 2, 128-byte SW128 rows): a stage row is 256 FP4 = 128 bytes, and
 *     K=64 FP4 = 32 bytes, so the per-instruction descriptor step is again +2.
 */
#define F4_ROWB 128   /* payload bytes per operand row per stage (K = 256 FP4)  */
#define F4_NK 4       /* ksets (4-block scale groups) per stage                 */
#define F4_SFB 2048   /* staged scale bytes per 128-row operand per stage       */

DINLINE int f4_sf_off(int r, int j) {
  return ((((r & 31) >> 3) * F4_NK + j) * 8 + (r & 7)) * 16 + ((r >> 5) & 3) * 4;
}
DINLINE unsigned long long make_sf_desc(unsigned addr, unsigned sbo) {
  return (unsigned long long)((addr >> 4) & 0x3FFF)
       | ((unsigned long long)((sbo >> 4) & 0x3FFF) << 32) | (1ULL << 46);
}

/* GEMM1 allocates all 512 TMEM columns (up and gate accumulators plus the scale
 * ring), so only one CTA is resident per SM and ptxas may budget registers
 * accordingly; GEMM2 needs 256 and keeps two. Without the minimum-blocks hint
 * ptxas capped GEMM1 at 72 registers and spilled a 368-byte stack frame (job
 * dv_6467e9dc7740 build log), and the spills sit in the fused epilogue, which
 * the pipeline ablation shows is two thirds of GEMM1. */
template <int BN, bool GEMM1, int STAGES, bool TMA, int AMODE = 0,
          int SFS = STAGES, int SFPRE = 0, int SK = 1, bool FUSE = false,
          bool BRLESS = false, bool SFCP = false, bool REDF = false,
          bool PIPE = false>
DINLINE void f4_body(const Params& p, const CUtensorMap* tmb, const CUtensorMap* tma) {
  /* PIPE (rungs 50+): pipelined-commit mainloop. The base loop waits for the
   * MMA of stage ks-1 to RETIRE before it may issue the MMA of stage ks (the
   * retire wait sits ahead of the SF store + TMA issue + bytes wait + sync
   * chain), so the tensor pipe drains completely once per stage: the measured
   * per-stage period is ~retire + TMA + issue latencies summed (~1.4 us on
   * fixed_6), while the MMA burst itself is only ~0.2-0.6 us (historical NCU:
   * tensor pipe ~13-33% busy). PIPE reorders the iteration to
   *   A. wait TMA bytes of stage ks        (issued STAGES-1 iterations ago)
   *   B. issue SF cp + MMAs of stage ks, commit -> s_bar[mcom & 1], mcom++
   *   C. wait retire of commit mcom-2 (= stage ks-1)   [ks >= 1]
   *   D. store SF registers for stage ks+SFS-1 (slot freed by C's retire)
   *   E. issue TMA of stage ks+STAGES-1 into the buffer freed by C
   *   F. prefetch SF registers for stage ks+SFS+1
   * so consecutive stages' MMAs overlap in the tensor pipe and the retire
   * wait only gates BUFFER REUSE, never the next MMA issue. Two ping-pong
   * commit barriers live in the existing STAGES-deep s_bar allocation
   * (STAGES >= 2 always); commit #m lands on s_bar[m&1] and its kth arrival
   * (k = m>>1) is waited with parity k&1. tcgen05 completions are ordered per
   * CTA, so the item-end wait on the LAST commit retires all earlier MMAs.
   * Buffer/SF-slot safety is identical to the base loop: with any ring depth
   * D the buffer staged at iteration ks was last read by MMA(ks-1), and the
   * SF slot written at D was last read by tcgen05.cp(ks-1), which the
   * in-order retire of MMA(ks-1) implies. Requires TMA && SFPRE >= 2 &&
   * !SFCP (register SF prefetch, no cp.async SF groups in the reorder). */
  static_assert(!PIPE || (TMA && SFPRE >= 2 && !SFCP),
                "PIPE requires the TMA + register-prefetched-SF mainloop");
  /* C15 (SFCP): scale factors stage via cp.async.ca 4B writes straight into
   * the SMEM ring -- no LDG->register->STS leg, so no long_scoreboard on the
   * SF path and no prefetch registers. The cp.async joins the SAME per-
   * iteration commit group as the stage's payload copies, and consumption at
   * iteration g is after cp_async_wait<STAGES-1>, which retires the group of
   * iteration g-(STAGES-1) == g-(SFS-1) exactly when SFS >= STAGES. */
  static_assert(!SFCP || (SFS >= STAGES), "SFCP needs SFS >= STAGES");
  constexpr int BM = 128;                        /* route rows per CTA         */
  constexpr int NROW = GEMM1 ? (2 * BN) : BN;    /* weight rows per stage tile  */
  constexpr int NSFB = GEMM1 ? 2 : 1;            /* up and gate, or one operand */
  constexpr int A_BYTES = BM * F4_ROWB;
  constexpr int B_BYTES = NROW * F4_ROWB;
  constexpr int ACC = GEMM1 ? (2 * BN) : BN;     /* TMEM accumulator columns    */
  constexpr int SFC = 4 * F4_NK;                 /* SF columns per operand/slot */
  /* SFS decouples the scale-factor ring depth from the payload ring depth.
   * A scale slot only has to survive from its tcgen05.cp until the commit of
   * the MMA that consumed it retires, so SFS=2 is safe under a STAGES=3
   * payload ring (the retire barrier of iteration ks frees slot (gst-1)%SFS
   * before the top-of-iteration store into slot (gst+1)%SFS == (gst-1)%SFS).
   * That is what brings the BN=64 GEMM1 three-buffer rung back to 224 TMEM
   * columns -> two CTAs per SM (rung 11 at SFS=STAGES=3 needed 272). */
  constexpr int TCOLS = ACC + SFS * SFC * (NSFB + 1);
  constexpr int TALLOC = TCOLS <= 256 ? 256 : 512;   /* power of two, >= 32     */
  constexpr int BCH = F4_ROWB / 16;              /* 16-byte chunks per row      */
  constexpr int BT = NROW * BCH / 128;           /* payload tasks per thread    */

  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  const int total_tiles = p.toffs[p.E];
  /* GEMM1 covers BN *intermediate* columns per CTA (up and gate are two weight
   * tiles of BN rows each), so its item space is over I, not 2I. With SK > 1
   * (GEMM1 small-batch split-K) the item space becomes (m-tile, k-slice,
   * n-tile) with the n-tile fastest, so neighbouring CTAs share the gathered
   * A slice of one k-range in L2; each item then runs nk/SK stages and stores
   * RAW fp32 partials into part[ksplit][route][col], which k_act_packed
   * reduces in fixed slice order before the fused clamp/activation/QDQ. */
  const int n_tiles = (GEMM1 ? p.I : p.H) / BN;
  const long long total_items = (long long)total_tiles * n_tiles * SK;
  if (total_items <= 0) return;
  const int K = GEMM1 ? p.H : p.I;
  const int half_n = p.I;
  const int c4 = K >> 6;
  const int nk_full = K >> 8;                    /* stages of 256 FP4 values   */
  const int nk = (SK > 1) ? (nk_full / SK) : nk_full;   /* stages per item     */
  const int N = GEMM1 ? (2 * p.I) : p.H;
  /* GEMM1 A-operand modes: 0 = cp.async gather from hid_q by ptok,
   * 1 = TMA tile::gather4 from hid_q, 2 = plain 2-D TMA boxes over the
   * pre-gathered route-order buffers hidq_g/hidsf_g (k_gather_a). */
  const uint8_t* Aq = GEMM1 ? ((AMODE == 2) ? p.hidq_g : p.hid_q) : p.act_q;
  const uint8_t* Asf = GEMM1 ? ((AMODE == 2) ? p.hidsf_g : p.hid_sf) : p.act_sf;
  const size_t aq_stride = (size_t)(K >> 1);
  const size_t asf_stride = (size_t)(K >> 4);
  const uint8_t* Wq = GEMM1 ? p.w1_q : p.w2_q;
  const uint8_t* Wsf = GEMM1 ? p.w1_sf : p.w2_sf;
  const size_t eq_stride = (size_t)N * (size_t)(K >> 1);
  const size_t esf_stride = (size_t)N * (size_t)(K >> 4);

  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* const a_base = smem_raw;
  unsigned char* const b_base = smem_raw + STAGES * A_BYTES;
  unsigned char* const sfa_base = b_base + STAGES * B_BYTES;
  unsigned char* const sfb_base = sfa_base + SFS * F4_SFB;
  unsigned* const s_tmem =
      reinterpret_cast<unsigned*>(sfb_base + SFS * NSFB * F4_SFB);
  unsigned long long* s_bar = reinterpret_cast<unsigned long long*>(s_tmem + 4);
  /* One box-completion barrier per buffer, reserved only by the TMA rung so the
   * cp.async rungs keep their exact shared-memory image. */
  unsigned long long* s_lbar = s_bar + STAGES;
  int32_t* s_row = reinterpret_cast<int32_t*>(s_lbar + (TMA ? STAGES : 0));

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  if (tid == 0) {
    mbar_init(smem_u32(s_bar), 1);
    if constexpr (PIPE) mbar_init(smem_u32(s_bar + 1), 1);
    if constexpr (TMA) {
      for (int i = 0; i < STAGES; ++i) mbar_init(smem_u32(s_lbar + i), 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  if (tid < 32) tmem_alloc(s_tmem, TALLOC);
  __syncthreads();
  const unsigned tmem = *s_tmem;
  const unsigned bar = smem_u32(s_bar);
  /* One commit per stage and nk may be odd, so the barrier phase is carried
   * across items of the persistent loop instead of restarting at 0. */
  unsigned phase = 0;
  /* Box barriers flip once per use of their buffer, so with TMA the buffer index
   * and the parity come from a stage counter that runs across items of the
   * persistent loop instead of restarting at zero for each one. */
  int gstage = 0;
  /* PIPE: global MMA-commit counter (ping-pong barrier index = mcom & 1,
   * per-barrier completion parity = (mcom >> 1) & 1), carried across items. */
  long long mcom = 0;
  const unsigned idesc = make_instr_desc(0u, 1u, 1u, (unsigned)(BN >> 3),
                                         (unsigned)(128 >> 4), TC_M_SHIFT, 0u);

  for (int item = blockIdx.y * gridDim.x + blockIdx.x; item < total_items;
       item += gridDim.x * gridDim.y) {
    int j, n0, ksplit;
    if constexpr (SK > 1) {
      const int inner = n_tiles * SK;
      j = item / inner;
      const int rem = item - j * inner;
      ksplit = rem / n_tiles;
      n0 = (rem - ksplit * n_tiles) * BN;
    } else {
      j = item / n_tiles;
      n0 = (item - j * n_tiles) * BN;
      ksplit = 0;
    }
    const int ks0 = ksplit * nk;                 /* absolute first stage of the
                                                  * slice; copies/SF addressing
                                                  * stay absolute while slots,
                                                  * parity and commits run on
                                                  * the local stage counter. */
    const int e = p.texp[j];
    const int m0 = p.tm0[j];   /* p.bm == BM for this rung */
    const int row_base = p.offs[e] + m0;
    const int rows_valid = p.counts[e] - m0;
    const uint8_t* Bq = Wq + (size_t)e * eq_stride;
    const uint8_t* Bsf = Wsf + (size_t)e * esf_stride;
    const float alpha = p.alpha[e], beta = p.beta[e];
    /* Per-item hoisted reciprocals for the fast activation (see
     * activation_fast): the reference divides per element. */
    const float inv_alpha = __frcp_rn(alpha), inv_beta = __frcp_rn(beta);
    /* C11a: per-item ex2 prescale constants for the packed activation. */
    const ActK ak = act_k_constants(p.act_type, alpha, inv_alpha, inv_beta);
    const float lim = p.has_limit ? p.limit[e] : 0.0f;

    for (int i = tid; i < BM; i += 128) {
      int ii = (i < rows_valid) ? i : 0;
      /* AMODE 2 indexes the pre-gathered route-order buffers directly. */
      s_row[i] = (GEMM1 && AMODE != 2) ? p.ptok[row_base + ii] : (row_base + ii);
    }
    __syncthreads();

    /* Loop-invariant per-thread addresses. Task decomposition: task = tid +
     * t*128, chunk c = task%8 = tid%8 (constant), row = task/8 = (tid>>3)+16t,
     * so only the source pointer needs to be kept per task; the swizzled
     * destination offset is recomputed from (row, c) in three instructions. */
    /* Offsets, not pointers: 32 bits each, which halves the register cost of the
     * per-task address table that stays live across the whole item (the largest
     * weight tensor in the domain is 470 MB, so a row offset always fits). */
    unsigned int qoff[BT];
#pragma unroll
    for (int t = 0; t < BT; ++t) {
      const int n_row = (tid >> 3) + 16 * t;
      const int rl = (GEMM1 && n_row >= BN) ? (n0 + n_row - BN + half_n)
                                            : (n0 + n_row);
      const int pr = phys_row(rl, half_n, GEMM1);
      qoff[t] = (unsigned int)((size_t)pr * aq_stride) + (unsigned)((tid & 7) * 16);
    }
    /* Scale rows: the staging layout is defined for 128 operand rows, so with
     * BN < 128 only the first BN threads have a row to stage (the remaining
     * slots are never read by an N=BN MMA). */
    const uint8_t* bsf[NSFB];
#pragma unroll
    for (int g = 0; g < NSFB; ++g) {
      const int rr = (tid < BN) ? tid : 0;
      const int rl = (GEMM1 && g) ? (n0 + rr + half_n) : (n0 + rr);
      bsf[g] = Bsf + sf_offset(phys_row(rl, half_n, GEMM1), 0, c4);
    }

#define F4_LOAD_A(KS, DST)                                                       \
    do {                                                                         \
      const size_t off_ = (size_t)(KS) * F4_ROWB;                                \
      for (int task = tid; task < BM * BCH; task += 128) {                       \
        const int r_ = task >> 3;                                                \
        const int c_ = task & 7;                                                 \
        cp_async16((DST) + r_ * F4_ROWB + ((c_ ^ (r_ & 7)) * 16),                 \
                   Aq + (size_t)s_row[r_] * aq_stride + off_ + c_ * 16);          \
      }                                                                          \
    } while (0)

#define F4_LOAD_B(KS, DST)                                                       \
    do {                                                                         \
      const size_t off_ = (size_t)(KS) * F4_ROWB;                                \
      _Pragma("unroll")                                                          \
      for (int t = 0; t < BT; ++t) {                                             \
        const int n_ = (tid >> 3) + 16 * t;                                      \
        cp_async16((DST) + n_ * F4_ROWB + (((tid & 7) ^ (n_ & 7)) * 16),         \
                   Bq + qoff[t] + off_);                                          \
      }                                                                          \
    } while (0)

    /* Stage the scale factors of stage KS into the copy layout above. The A side
     * is one 16-byte row-major (or word-packed) read per thread; the B side is
     * four 4-byte reads at 512-byte stride, which is how the frozen 128x4
     * swizzle stores the four columns of one scale group. */
#define F4_LOAD_SF(KS, BUF)                                                      \
    do {                                                                         \
      unsigned char* sfa_ = sfa_base + (BUF) * F4_SFB;                            \
      const uint4 aw_ = *reinterpret_cast<const uint4*>(                         \
          Asf + (size_t)s_row[tid] * asf_stride + (size_t)(KS) * 16);             \
      const unsigned int* awp_ = reinterpret_cast<const unsigned int*>(&aw_);     \
      _Pragma("unroll")                                                          \
      for (int j_ = 0; j_ < F4_NK; ++j_)                                         \
        *reinterpret_cast<unsigned int*>(sfa_ + f4_sf_off(tid, j_)) = awp_[j_];   \
      if (tid < BN) {                                                            \
        _Pragma("unroll")                                                        \
        for (int g_ = 0; g_ < NSFB; ++g_) {                                      \
          unsigned char* sfb_ =                                                  \
              sfb_base + (g_ * SFS + (BUF)) * F4_SFB;                            \
          const uint8_t* src_ = bsf[g_] + (size_t)(KS) * (F4_NK * 512);          \
          _Pragma("unroll")                                                      \
          for (int j_ = 0; j_ < F4_NK; ++j_)                                     \
            *reinterpret_cast<unsigned int*>(sfb_ + f4_sf_off(tid, j_)) =         \
                *reinterpret_cast<const unsigned int*>(src_ + (size_t)j_ * 512);  \
        }                                                                        \
      }                                                                          \
      /* Publish these generic-proxy stores to the async proxy that tcgen05.cp   \
       * reads through; the issuing thread fences again after the barrier. */    \
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");               \
    } while (0)

    /* Register-prefetched scale staging (SFPRE): F4_FETCH_SF issues the global
     * reads for one stage into per-thread registers; F4_PUT_SF lands them in
     * the shared copy layout one iteration later. The combined F4_LOAD_SF
     * stalls the issuing thread on the scale-load latency at the END of every
     * iteration, directly on the mainloop critical path (~600-800 ns per stage
     * from L2/HBM); splitting fetch/store lets the next iteration's
     * MMA-retire barrier wait cover it. Slot safety: the PUT at the top of
     * iteration ks targets slot (gst+SFS-1)%SFS == (gst-1)%SFS, whose previous
     * reader (the tcgen05.cp of stage gst-1) retired at this iteration's top
     * barrier. */
    /* SFPRE levels: 1 = single register set, fetched one iteration ahead
     * (the original rung-17 scheme); 2 = TWO ping-pong sets with the leading
     * prime fetches BATCHED (one exposed latency round for the whole prime)
     * and the loop-tail fetch aimed TWO iterations ahead. NCU on rung 19/27
     * (job dv_98e07b3d1234): GEMM2 warps stall 10.4 of 18.4 cycles/instr on
     * long_scoreboard -- the per-item serial SF prime rounds and the
     * one-iteration tail cover expose L2/HBM scale-load latency on the
     * critical path; SFPRE 2 removes both. Buffer index = stage parity, so a
     * tail fetch never clobbers the set the next PUT consumes. */
    constexpr int NRB = (SFPRE >= 2) ? 2 : 1;
    uint4 sfa_r_[NRB];
    unsigned int sfb_r_[NRB][NSFB][F4_NK];
#define F4_FETCH_SF(KS, B)                                                       \
    do {                                                                         \
      sfa_r_[B] = *reinterpret_cast<const uint4*>(                               \
          Asf + (size_t)s_row[tid] * asf_stride + (size_t)(KS) * 16);            \
      if (tid < BN) {                                                            \
        _Pragma("unroll")                                                        \
        for (int g_ = 0; g_ < NSFB; ++g_) {                                      \
          const uint8_t* src_ = bsf[g_] + (size_t)(KS) * (F4_NK * 512);          \
          _Pragma("unroll")                                                      \
          for (int j_ = 0; j_ < F4_NK; ++j_)                                     \
            sfb_r_[B][g_][j_] = *reinterpret_cast<const unsigned int*>(          \
                src_ + (size_t)j_ * 512);                                        \
        }                                                                        \
      }                                                                          \
    } while (0)
#define F4_CPSF(KS, BUF)                                                          \
    do {                                                                          \
      unsigned char* sfa_ = sfa_base + (BUF) * F4_SFB;                            \
      const uint8_t* asrc_ = Asf + (size_t)s_row[tid] * asf_stride                \
                           + (size_t)(KS) * 16;                                   \
      _Pragma("unroll")                                                           \
      for (int j_ = 0; j_ < F4_NK; ++j_)                                          \
        cp_async4(sfa_ + f4_sf_off(tid, j_), asrc_ + 4 * j_);                     \
      if (tid < BN) {                                                             \
        _Pragma("unroll")                                                         \
        for (int g_ = 0; g_ < NSFB; ++g_) {                                       \
          unsigned char* sfb_ = sfb_base + (g_ * SFS + (BUF)) * F4_SFB;           \
          const uint8_t* src_ = bsf[g_] + (size_t)(KS) * (F4_NK * 512);           \
          _Pragma("unroll")                                                       \
          for (int j_ = 0; j_ < F4_NK; ++j_)                                      \
            cp_async4(sfb_ + f4_sf_off(tid, j_), src_ + (size_t)j_ * 512);        \
        }                                                                         \
      }                                                                           \
    } while (0)
#define F4_PUT_SF(B, BUF)                                                        \
    do {                                                                         \
      unsigned char* sfa_ = sfa_base + (BUF) * F4_SFB;                           \
      const unsigned int* awp_ =                                                 \
          reinterpret_cast<const unsigned int*>(&sfa_r_[B]);                     \
      _Pragma("unroll")                                                          \
      for (int j_ = 0; j_ < F4_NK; ++j_)                                         \
        *reinterpret_cast<unsigned int*>(sfa_ + f4_sf_off(tid, j_)) = awp_[j_];  \
      if (tid < BN) {                                                            \
        _Pragma("unroll")                                                        \
        for (int g_ = 0; g_ < NSFB; ++g_) {                                      \
          unsigned char* sfb_ = sfb_base + (g_ * SFS + (BUF)) * F4_SFB;          \
          _Pragma("unroll")                                                      \
          for (int j_ = 0; j_ < F4_NK; ++j_)                                     \
            *reinterpret_cast<unsigned int*>(sfb_ + f4_sf_off(tid, j_)) =        \
                sfb_r_[B][g_][j_];                                               \
        }                                                                        \
      }                                                                          \
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");               \
    } while (0)

    /* One TMA box per operand. The tensor map's dimensions express the frozen
     * 32-row block shuffle as a transpose, so a single box lands the same SW128
     * image that 2048 (GEMM2) or 3072 (GEMM1, both halves) cp.async16 requests
     * produce -- verified byte-for-byte against this kernel's own staging loop by
     * experiments/tma_box_probe.py (job dv_816ecb04c15a). The block coordinate
     * carries the expert offset, because one map covers all E experts: dim 3 has
     * a 32-row stride, so expert e starts at block e * (N / 32). GEMM2's A operand
     * is already contiguous in routed-row order, so it is one plain 2D box; rows
     * past the expert's count are out of bounds and TMA fills them with zeros,
     * which the epilogue discards exactly as it discards the clamped rows the
     * cp.async path stages. GEMM1's A operand is gathered per token: with TMAA it
     * becomes BM/4 gather4 boxes (four rows x 128 bytes each, issued across
     * warp 0) instead of 1024 sixteen-byte cp.async requests per stage; the map
     * is the plain 2D hidden-state byte matrix, and a gathered row lands at
     * dst+128*i with the same SW128 phase (dst/128+i)%8 the cp.async loop
     * writes, because A_BYTES is a multiple of 1024. */
#define F4_LOAD_TMA(KS, BUF, ADST, BDST)                                          \
    do {                                                                          \
      const unsigned lb_ = smem_u32(s_lbar + (BUF));                              \
      const int c0_ = (KS) * F4_ROWB;                                             \
      const int g3_ = e * (N >> 5) + (GEMM1 ? (n0 >> 4) : (n0 >> 5));             \
      if constexpr (GEMM1 && AMODE == 1) {                                          \
        if (tid < 32) {                                                           \
          if (tid == 0)                                                           \
            tma_expect_tx(lb_, (2u * BN + (unsigned)BM) * F4_ROWB);               \
          __syncwarp();                                                           \
          const int r0_ = tid * 4;                                                \
          tma_gather4(tma, smem_u32(ADST) + 512u * (unsigned)tid, lb_, c0_,       \
                      s_row[r0_], s_row[r0_ + 1], s_row[r0_ + 2], s_row[r0_ + 3]);\
          if (tid == 0) {                                                         \
            tma_box_5d(tmb, smem_u32(BDST), lb_, c0_, 0, 0, g3_, 0);              \
            tma_box_5d(tmb, smem_u32(BDST) + BN * F4_ROWB, lb_, c0_, 0, 0, g3_, 1);\
          }                                                                       \
        }                                                                         \
      } else if (tid == 0) {                                                      \
        if constexpr (GEMM1) {                                                    \
          if constexpr (AMODE == 2) {                                             \
            /* One plain 2-D box over the pre-gathered route-order rows; rows   \
             * past Rmax are OOB and TMA zero-fills them. */                      \
            tma_expect_tx(lb_, (2u * BN + (unsigned)BM) * F4_ROWB);               \
            tma_box_2d(tma, smem_u32(ADST), lb_, c0_, row_base);                  \
          } else {                                                                \
            tma_expect_tx(lb_, 2u * BN * F4_ROWB);                                \
          }                                                                       \
          tma_box_5d(tmb, smem_u32(BDST), lb_, c0_, 0, 0, g3_, 0);                \
          tma_box_5d(tmb, smem_u32(BDST) + BN * F4_ROWB, lb_, c0_, 0, 0, g3_, 1);  \
        } else {                                                                  \
          tma_expect_tx(lb_, (unsigned)(BN + BM) * F4_ROWB);                      \
          tma_box_2d(tma, smem_u32(ADST), lb_, c0_, row_base);                    \
          tma_box_4d(tmb, smem_u32(BDST), lb_, c0_, 0, 0, g3_);                   \
        }                                                                         \
      }                                                                           \
    } while (0)

    /* Prime STAGES-1 buffers so the copy feeding the stage about to be consumed
     * is always a full MMA old. Short K loops still commit one group per slot
     * (empty ones after the last stage), which keeps `cp_async_wait<STAGES-1>`
     * waiting for exactly the group of stage ks even when nk < STAGES. The SF
     * ring primes only its own SFS-1 leading slots; the loop tail below keeps
     * it exactly one iteration ahead of the MMA that consumes each slot. */
    const int g0 = TMA ? gstage : 0;
    if constexpr (SFCP) {
      /* SF lands with the payload prime loop below (same commit groups). */
    } else if constexpr (SFPRE >= 2) {
      /* Batch the leading loads, then land them: ONE exposed global-latency
       * round for the whole prime regardless of SFS. */
      for (int pre = 0; pre + 1 < SFS && pre < nk; ++pre)
        F4_FETCH_SF(ks0 + pre, pre & 1);
      for (int pre = 0; pre + 1 < SFS && pre < nk; ++pre)
        F4_PUT_SF(pre & 1, (g0 + pre) % SFS);
    } else if constexpr (SFPRE) {
      /* Land SF stages 0..SFS-2 now (one exposed global-latency round per
       * item), and issue the reads for stage SFS-1 into registers; the loop
       * stores them at the top of iteration 0. */
      for (int pre = 0; pre + 1 < SFS && pre < nk; ++pre) {
        F4_FETCH_SF(ks0 + pre, 0);
        F4_PUT_SF(0, (g0 + pre) % SFS);
      }
    }
#pragma unroll
    for (int pre = 0; pre < STAGES - 1; ++pre) {
      if (pre < nk) {
        const int pb_ = (g0 + pre) % STAGES;
        if constexpr (SFCP) {
          if (pre < SFS) F4_CPSF(ks0 + pre, (g0 + pre) % SFS);
        } else if constexpr (!SFPRE) {
          if (pre < SFS) F4_LOAD_SF(ks0 + pre, (g0 + pre) % SFS);
        }
        if constexpr (TMA) {
          F4_LOAD_TMA(ks0 + pre, pb_, a_base + pb_ * A_BYTES,
                      b_base + pb_ * B_BYTES);
          if constexpr (GEMM1 && AMODE == 0)
            F4_LOAD_A(ks0 + pre, a_base + pb_ * A_BYTES);
        } else {
          F4_LOAD_A(ks0 + pre, a_base + (pre % STAGES) * A_BYTES);
          F4_LOAD_B(ks0 + pre, b_base + (pre % STAGES) * B_BYTES);
        }
      }
      cp_async_commit();
    }
    if constexpr (SFCP) {
      /* Stage SFS-1's SF issues in iteration 0 of the loop below. */
    } else if constexpr (SFPRE >= 2) {
      /* Fill both ping-pong sets: stages SFS-1 and SFS. */
      if (SFS - 1 < nk) F4_FETCH_SF(ks0 + SFS - 1, (SFS - 1) & 1);
      if (SFS < nk) F4_FETCH_SF(ks0 + SFS, SFS & 1);
    } else if constexpr (SFPRE) {
      if (SFS - 1 < nk) F4_FETCH_SF(ks0 + SFS - 1, 0);
    }

    if constexpr (PIPE) {
      for (int ks = 0; ks < nk; ++ks) {
        const int gst = g0 + ks;
        const int st = gst % STAGES;
        /* A. Stage ks's payload bytes were issued STAGES-1 iterations ago. */
        if (tid == 0)
          mbar_wait(smem_u32(s_lbar + st), (unsigned)((gst / STAGES) & 1));
        __syncthreads();
        /* B. Issue this stage's SF copies + MMAs and commit BEFORE waiting
         * for the previous stage's retire, so two stages' MMA bursts can be
         * in flight in the tensor pipe across the iteration boundary. */
        const long long m = mcom++;
        if (warp == 0 && elect_one()) {
          tc_fence_after_sync();
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          const int sfs_ = gst % SFS;
          const unsigned sfa_t = tmem + ACC + sfs_ * (SFC * (NSFB + 1));
          const unsigned sfa_s = smem_u32(sfa_base + sfs_ * F4_SFB);
          for (int j = 0; j < F4_NK; ++j)
            tmem_cp_32x128b_warpx4(sfa_t + 4 * j,
                                   make_sf_desc(sfa_s + 128 * j, F4_NK * 128));
          unsigned sfb_t[NSFB];
#pragma unroll
          for (int g = 0; g < NSFB; ++g) {
            sfb_t[g] = sfa_t + SFC * (g + 1);
            const unsigned s_ = smem_u32(sfb_base + (g * SFS + sfs_) * F4_SFB);
            for (int j = 0; j < F4_NK; ++j)
              tmem_cp_32x128b_warpx4(sfb_t[g] + 4 * j,
                                     make_sf_desc(s_ + 128 * j, F4_NK * 128));
          }
          const unsigned long long a_d = make_smem_desc(
              smem_u32(a_base + st * A_BYTES), TC_SBO, TC_LBO, TC_MODE);
          const unsigned long long b_d = make_smem_desc(
              smem_u32(b_base + st * B_BYTES), TC_SBO, TC_LBO, TC_MODE);
          const unsigned long long bg_d =
              GEMM1 ? make_smem_desc(smem_u32(b_base + st * B_BYTES
                                              + BN * F4_ROWB),
                                     TC_SBO, TC_LBO, TC_MODE)
                    : 0ULL;
          for (int j = 0; j < F4_NK; ++j) {
            const unsigned en = (ks == 0 && j == 0) ? 0u : 1u;
            mma_mxf4nvf4_ss(tmem, a_d + 2 * j, b_d + 2 * j, idesc,
                            sfa_t + 4 * j, sfb_t[0] + 4 * j, en);
            if (GEMM1)
              mma_mxf4nvf4_ss(tmem + BN, a_d + 2 * j, bg_d + 2 * j, idesc,
                              sfa_t + 4 * j, sfb_t[GEMM1 ? 1 : 0] + 4 * j, en);
          }
          mma_commit(smem_u32(s_bar + (m & 1)));
        }
        /* C. Retire of stage ks-1 (commit m-1) frees the payload buffer and
         * the SF slot that D/E are about to overwrite. tcgen05 completions
         * are issued-order, so this also retires stage ks-1's SF copies. */
        if (ks >= 1) {
          if (warp == 0 && lane == 0)
            mbar_wait(smem_u32(s_bar + ((m - 1) & 1)),
                      (unsigned)(((m - 1) >> 1) & 1));
          __syncthreads();
        }
        /* D. Land the register-prefetched scales of stage ks+SFS-1. */
        if (ks + SFS - 1 < nk)
          F4_PUT_SF((ks + SFS - 1) & 1, (gst + SFS - 1) % SFS);
        /* E. One TMA box set for stage ks+STAGES-1 into the freed buffer. */
        if (ks + STAGES - 1 < nk) {
          const int pb_ = (gst + STAGES - 1) % STAGES;
          F4_LOAD_TMA(ks0 + ks + STAGES - 1, pb_, a_base + pb_ * A_BYTES,
                      b_base + pb_ * B_BYTES);
        }
        /* F. Issue the SF reads two iterations ahead (SFPRE >= 2). */
        if (ks + SFS + 1 < nk)
          F4_FETCH_SF(ks0 + ks + SFS + 1, (ks + SFS + 1) & 1);
      }
    } else {
    for (int ks = 0; ks < nk; ++ks) {
      const int gst = g0 + ks;
      const int st = gst % STAGES;
      const int pre = ks + STAGES - 1;
      const int pre_buf = (gst + STAGES - 1) % STAGES;
      /* Iteration ks stages into buffer (ks+STAGES-1)%STAGES, which the MMA of
       * iteration ks-1 was reading, so that MMA (and the scale copies feeding
       * it) must have retired first. */
      if (ks >= 1) {
        if (warp == 0 && lane == 0) mbar_wait(bar, phase);
        phase ^= 1;
        __syncthreads();
      }
      if constexpr (SFCP) {
        /* Issue stage ks+SFS-1's SF cp.asyncs into THIS iteration's commit
         * group; the slot's previous reader retired at the barrier above. */
        if (ks + SFS - 1 < nk)
          F4_CPSF(ks0 + ks + SFS - 1, (gst + SFS - 1) % SFS);
      } else if constexpr (SFPRE) {
        /* Store the scales fetched at the end of the previous iteration; the
         * slot's previous reader retired at the barrier above. */
        if (ks + SFS - 1 < nk)
          F4_PUT_SF((SFPRE >= 2) ? ((ks + SFS - 1) & 1) : 0,
                    (gst + SFS - 1) % SFS);
      }
      if (pre < nk) {
        if constexpr (TMA) {
          F4_LOAD_TMA(ks0 + pre, pre_buf, a_base + pre_buf * A_BYTES,
                      b_base + pre_buf * B_BYTES);
          if constexpr (GEMM1 && AMODE == 0)
            F4_LOAD_A(ks0 + pre, a_base + pre_buf * A_BYTES);
        } else {
          F4_LOAD_A(ks0 + pre, a_base + (pre % STAGES) * A_BYTES);
          F4_LOAD_B(ks0 + pre, b_base + (pre % STAGES) * B_BYTES);
        }
      }
      cp_async_commit();
      cp_async_wait<STAGES - 1>();
      if constexpr (TMA) {
        /* The boxes feeding this stage were issued STAGES-1 iterations ago; one
         * thread waits for their bytes and the barrier below publishes them. */
        if (tid == 0)
          mbar_wait(smem_u32(s_lbar + st), (unsigned)((gst / STAGES) & 1));
      }
      __syncthreads();
      if (warp == 0 && elect_one()) {
        tc_fence_after_sync();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const int sfs_ = gst % SFS;   /* SF ring slot (== st when SFS==STAGES) */
        const unsigned sfa_t = tmem + ACC + sfs_ * (SFC * (NSFB + 1));
        const unsigned sfa_s = smem_u32(sfa_base + sfs_ * F4_SFB);
        for (int j = 0; j < F4_NK; ++j)
          tmem_cp_32x128b_warpx4(sfa_t + 4 * j,
                                 make_sf_desc(sfa_s + 128 * j, F4_NK * 128));
        unsigned sfb_t[NSFB];
#pragma unroll
        for (int g = 0; g < NSFB; ++g) {
          sfb_t[g] = sfa_t + SFC * (g + 1);
          const unsigned s_ = smem_u32(sfb_base + (g * SFS + sfs_) * F4_SFB);
          for (int j = 0; j < F4_NK; ++j)
            tmem_cp_32x128b_warpx4(sfb_t[g] + 4 * j,
                                   make_sf_desc(s_ + 128 * j, F4_NK * 128));
        }
        const unsigned long long a_d = make_smem_desc(
            smem_u32(a_base + st * A_BYTES), TC_SBO, TC_LBO, TC_MODE);
        const unsigned long long b_d = make_smem_desc(
            smem_u32(b_base + st * B_BYTES), TC_SBO, TC_LBO, TC_MODE);
        const unsigned long long bg_d =
            GEMM1 ? make_smem_desc(smem_u32(b_base + st * B_BYTES + BN * F4_ROWB),
                                   TC_SBO, TC_LBO, TC_MODE)
                  : 0ULL;
        for (int j = 0; j < F4_NK; ++j) {
          const unsigned en = (ks == 0 && j == 0) ? 0u : 1u;
          mma_mxf4nvf4_ss(tmem, a_d + 2 * j, b_d + 2 * j, idesc, sfa_t + 4 * j,
                          sfb_t[0] + 4 * j, en);
          if (GEMM1)
            mma_mxf4nvf4_ss(tmem + BN, a_d + 2 * j, bg_d + 2 * j, idesc,
                            sfa_t + 4 * j, sfb_t[GEMM1 ? 1 : 0] + 4 * j, en);
        }
        mma_commit(bar);
      }
      /* Stage the scales of the SF slot being refilled while the tensor core is
       * still busy; their global latency is then covered by asynchronous MMA
       * work. With SFS slots the tail of iteration ks refills the slot for
       * stage ks+SFS-1, whose previous reader (the tcgen05.cp of stage ks-1)
       * retired at this iteration's top barrier. For SFS==STAGES this is the
       * original "refill the payload slot being re-staged" schedule. SFPRE
       * instead only ISSUES the reads here (into registers) and stores them at
       * the top of the next iteration, so the scale-load latency never sits on
       * the mainloop critical path. */
      if constexpr (SFCP) {
        /* The SF cp.asyncs for stage ks+SFS-1 were issued at the top. */
      } else if constexpr (SFPRE >= 2) {
        if (ks + SFS + 1 < nk)
          F4_FETCH_SF(ks0 + ks + SFS + 1, (ks + SFS + 1) & 1);
      } else if constexpr (SFPRE) {
        if (ks + SFS < nk) F4_FETCH_SF(ks0 + ks + SFS, 0);
      } else {
        const int sfx = ks + SFS - 1;
        if (sfx < nk) F4_LOAD_SF(ks0 + sfx, (gst + SFS - 1) % SFS);
      }
    }
    }  /* !PIPE */
    if constexpr (TMA) gstage = g0 + nk;
    if constexpr (PIPE) {
      /* Waiting on the LAST commit retires every earlier MMA: tcgen05
       * completions of one CTA are ordered by issue. */
      if (warp == 0 && lane == 0) {
        const long long lm = mcom - 1;
        mbar_wait(smem_u32(s_bar + (lm & 1)), (unsigned)((lm >> 1) & 1));
      }
    } else {
      if (warp == 0 && lane == 0) mbar_wait(bar, phase);
      /* One commit per stage, so the barrier parity consumed by an item is
       * nk-aligned: flipping after the final wait as well keeps `phase` equal to
       * (commits consumed so far) % 2, which is what the next item's first wait
       * needs. With the current grid (n_tiles x max_tiles >= total_items) a CTA
       * only ever runs one item, but the persistent loop must not depend on that. */
      phase ^= 1;
    }
    tc_fence_after_sync();
    __syncthreads();
#undef F4_LOAD_A
#undef F4_LOAD_B
#undef F4_LOAD_SF
#undef F4_LOAD_TMA
#undef F4_FETCH_SF
#undef F4_PUT_SF

    /* ------------------------- epilogue: TMEM -> packed activation / output --- */
    const unsigned taddr0 = tmem + ((unsigned)(warp * 32) << 16);
    const int token_local = warp * 32 + lane;
    const bool tok_ok = (token_local < rows_valid);
    const size_t route = (size_t)(row_base + token_local);
    if constexpr (SK > 1) {
      /* Split-K GEMM1: store the RAW fp32 slice partials; k_act_packed sums
       * the slices in fixed order and applies the fused clamp/activation/QDQ.
       * part layout [sk][Rmax][2I]: up at columns [n0, n0+BN), gate at
       * [I+n0, I+n0+BN). Padding rows are never stored and never read. */
      float* const pp = p.part + ((size_t)ksplit * p.Rmax + route)
                                 * (size_t)(2 * p.I);
#pragma unroll 1
      for (int c = 0; c < BN / 32; ++c) {
        float up[32], gate[32];
        tmem_ld_x32(taddr0 + c * 32, up);
        tmem_ld_x32(taddr0 + BN + c * 32, gate);
        tmem_ld_wait();
        if (tok_ok) {
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            *reinterpret_cast<float4*>(pp + n0 + c * 32 + 4 * q) =
                make_float4(up[4 * q], up[4 * q + 1], up[4 * q + 2],
                            up[4 * q + 3]);
            *reinterpret_cast<float4*>(pp + p.I + n0 + c * 32 + 4 * q) =
                make_float4(gate[4 * q], gate[4 * q + 1], gate[4 * q + 2],
                            gate[4 * q + 3]);
          }
        }
      }
      if constexpr (FUSE) {
        /* Rungs 31/32: the CTA whose arrival completes this (m-tile, n-tile)
         * group reduces the SK partial slices and writes the packed activation
         * itself, replacing the separate k_act_packed stage (a 2-block kernel
         * at tiny R: ~8 us of pure latency). Release/acquire via threadfence +
         * device atomic is the classic last-arriver pattern; the reduce runs
         * in FIXED slice order with the same arithmetic as k_act_packed, so
         * the stored activation bytes are identical whichever CTA arrives
         * last (only the arrival order varies). One thread per route row;
         * BN/16 sixteen-element blocks per row. */
        __threadfence();
        __shared__ int s_last;
        __syncthreads();
        if (tid == 0)
          s_last = (atomicAdd(&p.cnt[j * n_tiles + (n0 / BN)], 1) == SK - 1);
        __syncthreads();
        if (s_last && tok_ok) {
          const size_t slice_stride = (size_t)p.Rmax * (2 * p.I);
          const float* const pr = p.part + (size_t)route * (2 * p.I);
          const int nb = p.I >> 4;
#pragma unroll 1
          for (int blk = 0; blk < BN / 16; ++blk) {
            float up[16], gate[16];
#pragma unroll
            for (int q = 0; q < 16; ++q) { up[q] = 0.0f; gate[q] = 0.0f; }
            for (int sl = 0; sl < SK; ++sl) {
              const float* ps = pr + sl * slice_stride + n0 + blk * 16;
#pragma unroll
              for (int q4 = 0; q4 < 4; ++q4) {
                const float4 u4 = *reinterpret_cast<const float4*>(ps + 4 * q4);
                const float4 g4 =
                    *reinterpret_cast<const float4*>(ps + p.I + 4 * q4);
                up[4 * q4 + 0] += u4.x; up[4 * q4 + 1] += u4.y;
                up[4 * q4 + 2] += u4.z; up[4 * q4 + 3] += u4.w;
                gate[4 * q4 + 0] += g4.x; gate[4 * q4 + 1] += g4.y;
                gate[4 * q4 + 2] += g4.z; gate[4 * q4 + 3] += g4.w;
              }
            }
            /* C10: same pair-packed form as k_act_packed (rung 31/32 must
             * keep producing identical stored bytes to the split act stage). */
            float vals[16];
            float amax = 0.0f;
            if (p.has_limit) {
#pragma unroll
              for (int q = 0; q < 8; ++q) {
                const float g0 = fminf(gate[2 * q], lim);
                const float g1 = fminf(gate[2 * q + 1], lim);
                const float u0 = fminf(fmaxf(up[2 * q], -lim), lim);
                const float u1 = fminf(fmaxf(up[2 * q + 1], -lim), lim);
                const float2 a2 = activation2(p.act_type, make_float2(g0, g1),
                                              make_float2(u0, u1), alpha, beta,
                                              ak.k_sw, ak.k_gu, ak.k_uu);
                vals[2 * q] = a2.x; vals[2 * q + 1] = a2.y;
                amax = fmaxf(amax, fmaxf(fabsf(a2.x), fabsf(a2.y)));
              }
            } else {
#pragma unroll
              for (int q = 0; q < 8; ++q) {
                const float2 a2 = activation2(p.act_type,
                    make_float2(gate[2 * q], gate[2 * q + 1]),
                    make_float2(up[2 * q], up[2 * q + 1]), alpha, beta,
                    ak.k_sw, ak.k_gu, ak.k_uu);
                vals[2 * q] = a2.x; vals[2 * q + 1] = a2.y;
                amax = fmaxf(amax, fmaxf(fabsf(a2.x), fabsf(a2.y)));
              }
            }
            const int sf_code = f32_to_e4m3(fminf(amax * (1.0f / 6.0f), 448.0f));
            const float sf = e4m3_val(sf_code);
            unsigned int words[2] = {0u, 0u};
            if (sf > 0.0f) {
              const float inv_sf = __frcp_rn(sf);
              const float2 inv2 = make_float2(inv_sf, inv_sf);
#pragma unroll
              for (int q = 0; q < 8; ++q) {
                const float2 v2 = f2_mul(make_float2(vals[2 * q], vals[2 * q + 1]),
                                         inv2);
                vals[2 * q] = v2.x; vals[2 * q + 1] = v2.y;
              }
              words[0] = q_e2m1_pack8(vals[0], vals[1], vals[2], vals[3],
                                      vals[4], vals[5], vals[6], vals[7]);
              words[1] = q_e2m1_pack8(vals[8], vals[9], vals[10], vals[11],
                                      vals[12], vals[13], vals[14], vals[15]);
            }
            const int bg = (n0 >> 4) + blk;          /* global 16-block index */
            *reinterpret_cast<uint2*>(p.act_q + (size_t)route * (p.I >> 1)
                                      + ((size_t)bg << 3)) =
                make_uint2(words[0], words[1]);
            p.act_sf[(size_t)route * nb + bg] = (unsigned char)sf_code;
          }
        }
      }
    } else if (GEMM1) {
      uint8_t* const pd = p.act_q + route * (size_t)(p.I >> 1) + (size_t)(n0 >> 1);
      unsigned int* const sd = reinterpret_cast<unsigned int*>(p.act_sf)
                             + route * (size_t)(p.I >> 6) + (size_t)(n0 >> 6);
      /* Everything the epilogue holds is a named scalar. Any array indexed by the
       * column loop -- which stays rolled (`unroll 1`) so that the 32-float TMEM
       * reads do not all have to be live at once -- is placed in local memory,
       * and ptxas had been reporting a 288-336 byte stack frame for this kernel
       * (job dv_6467e9dc7740). The scale word covering four consecutive blocks is
       * accumulated in `sfacc` and stored as soon as it is complete. */
      unsigned int sfacc = 0u;
#pragma unroll 1
      for (int c = 0; c < BN / 32; ++c) {
        float up[32], gate[32];
        /* Issue both reads before waiting: tcgen05.wait::ld retires every
         * outstanding load, so one wait covers the pair and the two TMEM
         * round trips overlap instead of serialising. */
        tmem_ld_x32(taddr0 + c * 32, up);
        tmem_ld_x32(taddr0 + BN + c * 32, gate);
        tmem_ld_wait();
        unsigned int wa = 0u, wb = 0u, wc = 0u, wd = 0u;
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
          /* C10: pair-packed activation. The warp-uniform has_limit clamp is
           * unswitched out of the element loop so the common no-limit path
           * pays no FMNMX; the packed chain itself is bit-identical to
           * activation_fast (same per-lane mul.rn/add.rn/ex2/rcp.approx
           * sequence and order, see activation2). gate values are dead from
           * here on, so the block doubles as activation scratch. */
          float* gv = &gate[blk * 16];
          float amax = 0.0f;
          if (p.has_limit) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const float g0 = fminf(gv[2 * q], lim);
              const float g1 = fminf(gv[2 * q + 1], lim);
              const float u0 = fminf(fmaxf(up[blk * 16 + 2 * q], -lim), lim);
              const float u1 = fminf(fmaxf(up[blk * 16 + 2 * q + 1], -lim), lim);
              const float2 a2 = activation2(p.act_type, make_float2(g0, g1),
                                            make_float2(u0, u1), alpha, beta,
                                            ak.k_sw, ak.k_gu, ak.k_uu);
              gv[2 * q] = a2.x; gv[2 * q + 1] = a2.y;
              amax = fmaxf(amax, fmaxf(fabsf(a2.x), fabsf(a2.y)));
            }
          } else {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const float2 g2 = make_float2(gv[2 * q], gv[2 * q + 1]);
              const float2 u2 = make_float2(up[blk * 16 + 2 * q],
                                            up[blk * 16 + 2 * q + 1]);
              const float2 a2 = activation2(p.act_type, g2, u2, alpha, beta,
                                            ak.k_sw, ak.k_gu, ak.k_uu);
              gv[2 * q] = a2.x; gv[2 * q + 1] = a2.y;
              amax = fmaxf(amax, fmaxf(fabsf(a2.x), fabsf(a2.y)));
            }
          }
          const int sf_code = f32_to_e4m3(fminf(amax * (1.0f / 6.0f), 448.0f));
          const float sf = e4m3_val(sf_code);
          const int bidx = c * 2 + blk;          /* scale block inside the tile */
          sfacc |= (unsigned)sf_code << (8 * (bidx & 3));
          unsigned int w0 = 0u, w1 = 0u;
          /* C11c (BRLESS rungs 41-43): the per-block sf>0 guard is the hot
           * FSETP+BRA+BSSY/BSYNC control cluster (job X SASS sampling).
           * Clamping sf to FLT_MIN makes the guard unnecessary: rcp stays
           * finite, the near-zero vals of a code-0 block quantize to finite
           * saturating codes, and the STORED sf byte (0) re-zeroes them in
           * GEMM2's block-scaled MMA, so outputs are unchanged. Bit-identical
           * to the guarded form on every sf>0 block (fmaxf is exact there). */
          if (BRLESS || sf > 0.0f) {
            /* Packed multiply by the correctly-rounded block reciprocal
             * replaces the Markstein div_by_recip (3 instr/elem -> 0.5):
             * an e2m1 code flips only when the true quotient lies within
             * ~1.5 ulp of a rounding boundary (~2e-6 of elements; bound
             * derived at the f2_mul helpers). One rcp per 16-block. */
            const float inv_sf =
                BRLESS ? __frcp_rn(fmaxf(sf, 1.17549435e-38f))
                       : __frcp_rn(sf);
            const float2 inv2 = make_float2(inv_sf, inv_sf);
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const float2 v2 = f2_mul(make_float2(gv[2 * q], gv[2 * q + 1]),
                                       inv2);
              gv[2 * q] = v2.x; gv[2 * q + 1] = v2.y;
            }
            w0 = q_e2m1_pack8(gv[0], gv[1], gv[2], gv[3],
                              gv[4], gv[5], gv[6], gv[7]);
            w1 = q_e2m1_pack8(gv[8], gv[9], gv[10], gv[11],
                              gv[12], gv[13], gv[14], gv[15]);
          }
          if (blk == 0) { wa = w0; wb = w1; } else { wc = w0; wd = w1; }
        }
        /* BN features = BN/2 payload bytes = BN/32 uint4 stores, one per
         * iteration, each 16-byte aligned because n0 is a multiple of BN and I/2
         * is a multiple of 16 in every legal geometry. */
        if (tok_ok)
          *reinterpret_cast<uint4*>(pd + c * 16) = make_uint4(wa, wb, wc, wd);
        if ((c & 1) == 1 || c == (BN / 32) - 1) {
          if (tok_ok) sd[c >> 1] = sfacc;
          sfacc = 0u;
        }
      }
    } else if constexpr (REDF) {
      /* Fused routing reduction: per route, round the FP32 accumulator to
       * BF16 (bit-identical to the rout store), scale by the routing weight
       * and reduce into acc32[token] with vectorized fire-and-forget fp32
       * atomics (red.global.add.v4.f32, measured 5.6 TB/s L2-resident on
       * this device -- probe job dv_f68e4d60e34b). The scattered-read
       * k_finalize pass (0.9-1.5 TB/s effective) and the entire [R,H] rout
       * buffer disappear; k_convert does the single final BF16 rounding.
       * The multiply-then-add split matches the reference exactly (it
       * materializes BF16, multiplies in fp32, then sums in fp32); only the
       * fp32 sum ORDER differs, exactly as the reference's own atomic
       * index_add_ does. Per-lane route metadata is loaded once per item. */
      int tok = 0;
      float sc = 0.0f;
      if (tok_ok) {
        tok = p.ptok[route];
        sc = p.scores[(size_t)tok * p.K + p.pslot[route]];
      }
#pragma unroll 1
      for (int c = 0; c < BN / 32; ++c) {
        float up[32];
        tmem_ld_x32(taddr0 + c * 32, up);
        tmem_ld_wait();
        if (tok_ok) {
          float* dst = p.acc32 + (size_t)tok * p.H + n0 + c * 32;
#pragma unroll
          for (int v = 0; v < 8; ++v) {
            const float2 f0 = __bfloat1622float2(
                __floats2bfloat162_rn(up[4 * v], up[4 * v + 1]));
            const float2 f1 = __bfloat1622float2(
                __floats2bfloat162_rn(up[4 * v + 2], up[4 * v + 3]));
            asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n"
                         :: "l"(dst + 4 * v), "f"(f0.x * sc), "f"(f0.y * sc),
                            "f"(f1.x * sc), "f"(f1.y * sc)
                         : "memory");
          }
        }
      }
    } else {
#pragma unroll 1
      for (int c = 0; c < BN / 32; ++c) {
        float up[32];
        tmem_ld_x32(taddr0 + c * 32, up);
        tmem_ld_wait();
        __nv_bfloat16* dst = p.rout + route * p.H + n0 + c * 32;
        if (tok_ok) {
          unsigned int outw[16];
          for (int q = 0; q < 16; ++q)
            outw[q] = as_u32(__floats2bfloat162_rn(up[2 * q], up[2 * q + 1]));
          for (int q = 0; q < 4; ++q)
            *reinterpret_cast<uint4*>(dst + q * 8) =
                make_uint4(outw[4 * q], outw[4 * q + 1], outw[4 * q + 2],
                           outw[4 * q + 3]);
        }
      }
    }
    __syncthreads();
  }
  if (tid < 32) tmem_dealloc(tmem, TALLOC);
}

/* Rungs 7-11: payload staged by 16-byte cp.async requests. */
template <int BN, bool GEMM1, int STAGES = 2>
__global__ void __launch_bounds__(128, GEMM1 ? 1 : 2) k_gemm_f4(const Params p) {
  f4_body<BN, GEMM1, STAGES, false>(p, nullptr, nullptr);
}

/* Rung 12: the same arithmetic and the same shared-memory image, with the weight
 * tile (and GEMM2's intermediate tile) staged by TMA. Rungs 13-16 reuse this
 * entry with BN=64 (the TMEM budget for two CTAs per SM) and, for GEMM1 with
 * TMAA, the activation gather staged by TMA tile::gather4. Rung 16 additionally
 * decouples the scale-factor ring (SFS=2) from the payload ring (STAGES=3).
 * The maps are `__grid_constant__` kernel parameters so that their address stays
 * in parameter space, which is what the copy instruction requires. */
template <int BN, bool GEMM1, int STAGES, int AMODE = 0, int SFS = STAGES,
          int SFPRE = 0, int SK = 1, bool FUSE = false, bool BRLESS = false,
          bool SFCP = false, bool REDF = false, bool PIPE = false>
__global__ void __launch_bounds__(128, (GEMM1 && BN > 64) ? 1 : 2)
k_gemm_f4_tma(const __grid_constant__ Params p,
              const __grid_constant__ CUtensorMap tm_b,
              const __grid_constant__ CUtensorMap tm_a) {
  f4_body<BN, GEMM1, STAGES, true, AMODE, SFS, SFPRE, SK, FUSE, BRLESS, SFCP,
          REDF, PIPE>(p, &tm_b, &tm_a);
}


/* --------------- warp-specialized fam-2 body (WS2, rung 56+) ---------------- */
/* NCU on rung 41 GEMM1 / rung 44 GEMM2 (job dv_cea324bddbfe, fixed_6): the
 * tensor pipe runs at ~20% of elapsed while the schedulers see 78-88% no-
 * eligible cycles with 1.94 active warps each; GEMM1's top stall is the CTA
 * barrier (38.8%) and GEMM2's is long-scoreboard on the 4-byte SF cp.async
 * storm (47.7%, L1TEX 64% busy). The monolithic loop serializes every role
 * through two __syncthreads per stage.
 *
 * WS3 ports the production choreography (cake_minimax_h3_fc1_swiglu_sm103a /
 * alphamoe_nvfp4_sm100 role split) to this kernel's tiling at SEVEN warps
 * (224 threads) and TWO CTAs per SM (TMEM 224 -> 256 columns, smem ~108 KiB):
 *   warp 0    producer A: activation scales -- LDG.128 ping-pong registers
 *             (one-stage lookahead, batched prime per item) + 4x STS.32 into
 *             the ring slot, fence.proxy.async, arrive on full[s].
 *   warp 1    producer B: TMA payload boxes (expect_tx on full[s]) + weight
 *             scales as 4-byte cp.asyncs landing directly in the ring slot,
 *             cp.async.mbarrier.arrive.noinc on full[s]. Both producers gate
 *             on empty[s] = commit(g-SFS), the cumulative MMA snapshot that
 *             frees payload buffer AND SF slot together. No syncthreads.
 *   warp 2    MMA: per stage, waits full[s], issues the 12 tcgen05.cp + 8
 *             block-scaled MMAs and commits empty[s]; commits epi_full at
 *             the item's end. MMAs of consecutive stages are in flight
 *             together -- the retire commit gates the PRODUCERS' restage,
 *             never the next MMA issue.
 *   warps 3-6 epilogue: wait epi_full, drain TMEM (lane quadrant = warp & 3),
 *             run the identical clamp/activation/QDQ, store, arrive epi_done.
 *             The MMA warp waits epi_done before the NEXT item's first MMA
 *             (accumulator reuse), so staging of the next item overlaps the
 *             drain. (WS2 v1 ran all SF staging on one producer warp: 1024
 *             4B cp.asyncs per stage = ~540ns issue time -- measured 0.92x,
 *             job dv_b5791d4d99ba; WS3 splits A/B scale staging and uses
 *             16B loads for the contiguous A scales.)
 * The staged shared-memory images, the MMA sequence and the epilogue
 * arithmetic are those of f4_body rung 41, so outputs are bit-identical.
 * v1 scope: GEMM1, SK=1 fused epilogue, pre-gathered A (AMODE 2). */
template <int BN, bool GEMM1, int STAGES, int SFS, int AMODE,
          bool BRLESS = true, int SK = 1, bool PP = false, int EPI = 4>
DINLINE void f4_body_ws(const Params& p, const CUtensorMap* tmb,
                        const CUtensorMap* tma) {
  static_assert(SFS <= STAGES &&
                    (GEMM1 ? (AMODE == 2 || AMODE == 1) : (AMODE == 0 && SK == 1)),
                "WS3: GEMM1 (pre-gathered or gather4 A, SK 1/2/4) or GEMM2");
  static_assert(EPI == 4 || EPI == 8, "WS: 4 or 8 epilogue warps");
  static_assert(!PP || (SK == 1 && (GEMM1 ? EPI == 8
                                    : (2 * BN + SFS * 16 * 2 <= 256))),
                "PP: GEMM2 BN=64 in 256 TMEM cols, or GEMM1 EPI=8 form");
  constexpr int BM = 128;
  constexpr int NROW = GEMM1 ? (2 * BN) : BN;
  constexpr int NSFB = GEMM1 ? 2 : 1;
  constexpr int A_BYTES = BM * F4_ROWB;
  constexpr int B_BYTES = NROW * F4_ROWB;
  constexpr int ACC = GEMM1 ? (2 * BN) : BN;
  constexpr int SFC = 4 * F4_NK;
  /* PP: two accumulator buffers (item parity) so the MMA warp never waits
   * for the previous item's TMEM drain -- it fills buffer (j&1) while the
   * epilogue drains buffer ((j-1)&1); the wait moves two items back. */
  constexpr int NACC = PP ? 2 : 1;
  constexpr int TCOLS = NACC * ACC + SFS * SFC * (NSFB + 1);
  constexpr int TALLOC = TCOLS <= 256 ? 256 : 512;

  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  const int total_tiles = p.toffs[p.E];
  const int n_tiles = (GEMM1 ? p.I : p.H) / BN;
  /* The SK item space is (m-tile, k-slice, n-tile) with the n-tile fastest,
   * exactly as in f4_body -- the SK port originally dropped the * SK factor
   * here, silently skipping (SK-1)/SK of the m-tiles (job dv_bfed8f476561:
   * garbage split-K partials -> every official SK-band case failed while the
   * fused band passed). */
  const long long total_items = (long long)total_tiles * n_tiles * SK;
  if (total_items <= 0) return;
  const int K = GEMM1 ? p.H : p.I;
  const int half_n = p.I;
  const int c4 = K >> 6;
  const int nk_full = K >> 8;
  const int nk = (SK > 1) ? (nk_full / SK) : nk_full;   /* stages per item */
  const int N = GEMM1 ? (2 * p.I) : p.H;
  /* AMODE 1 reads the A scales straight from the row-major input tensor with
   * ptok indirection; AMODE 2 from the pre-gathered route-order buffer. */
  const uint8_t* Asf = GEMM1 ? ((AMODE == 1) ? p.hid_sf : p.hidsf_g)
                             : p.act_sf;
  const size_t asf_stride = (size_t)(K >> 4);
  const uint8_t* Wsf = GEMM1 ? p.w1_sf : p.w2_sf;
  const size_t esf_stride = (size_t)N * (size_t)(K >> 4);

  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* const a_base = smem_raw;
  unsigned char* const b_base = smem_raw + STAGES * A_BYTES;
  unsigned char* const sfa_base = b_base + STAGES * B_BYTES;
  unsigned char* const sfb_base = sfa_base + SFS * F4_SFB;
  unsigned* const s_tmem =
      reinterpret_cast<unsigned*>(sfb_base + SFS * NSFB * F4_SFB);
  unsigned long long* const s_full =
      reinterpret_cast<unsigned long long*>(s_tmem + 4);
  unsigned long long* const s_empty = s_full + STAGES;
  unsigned long long* const s_epifull = s_empty + STAGES;
  unsigned long long* const s_epidone = s_epifull + (PP ? 2 : 1);

  const int tid = threadIdx.x;          /* 0..191 */
  const int warp = tid >> 5;            /* 0..5   */
  const int lane = tid & 31;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(smem_u32(s_full + i), 34);   /* tx + w0 arrive + 32 w1 cp.async */
      mbar_init(smem_u32(s_empty + i), 1);
    }
    mbar_init(smem_u32(s_epifull), 1);
    mbar_init(smem_u32(s_epidone), EPI);
    if constexpr (PP) {
      mbar_init(smem_u32(s_epifull + 1), 1);
      mbar_init(smem_u32(s_epidone + 1), EPI);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < 32) tmem_alloc(s_tmem, TALLOC);
  __syncthreads();                       /* one-time init rendezvous */
  const unsigned tmem = *s_tmem;
  const unsigned idesc = make_instr_desc(0u, 1u, 1u, (unsigned)(BN >> 3),
                                         (unsigned)(128 >> 4), TC_M_SHIFT, 0u);
  const long long istride = (long long)gridDim.x * gridDim.y;
  const long long start = (long long)blockIdx.y * gridDim.x + blockIdx.x;

  if (warp == 0) {
    /* -------------------- producer A: activation scales -------------------- */
    /* A-SF rows are 16B-contiguous in global: LDG.128 with a ping-pong
     * register set (one-stage lookahead, both sets primed batched at the
     * item start), then 4x STS.32 scattered into the ring slot -- rung 41's
     * proven SFPRE mechanism, moved off the critical warps. WS2 v1 drove
     * ALL SF staging as 1024 4-byte cp.asyncs per stage from this single
     * warp (~540ns of pure issue time per stage), which is why it measured
     * 0.92x against rung 41 (job dv_b5791d4d99ba). */
    long long g = 0;
    for (long long item = start; item < total_items; item += istride) {
      int j, ks0;
      if constexpr (SK > 1) {
        const int inner = n_tiles * SK;
        j = (int)(item / inner);
        const int ksplit = (int)((item - (long long)j * inner) / n_tiles);
        ks0 = ksplit * nk;
      } else {
        j = (int)(item / n_tiles);
        ks0 = 0;
      }
      const int e = p.texp[j];
      const int m0 = p.tm0[j];
      const int row_base = p.offs[e] + m0;
      const int rows_valid = p.counts[e] - m0;
      const uint8_t* asrc[BM / 32];
#pragma unroll
      for (int t = 0; t < BM / 32; ++t) {
        const int r = lane + 32 * t;
        const int ii = (r < rows_valid) ? r : 0;
        const size_t arow = (AMODE == 1)
            ? (size_t)p.ptok[row_base + ii]
            : (size_t)(row_base + ii);
        asrc[t] = Asf + arow * asf_stride;
      }
      uint4 sfa_r[2][BM / 32];
#pragma unroll
      for (int t = 0; t < BM / 32; ++t) {
        sfa_r[0][t] = *reinterpret_cast<const uint4*>(
            asrc[t] + (size_t)ks0 * 16);
        if (nk > 1)
          sfa_r[1][t] = *reinterpret_cast<const uint4*>(
              asrc[t] + (size_t)(ks0 + 1) * 16);
      }
      for (int ks = 0; ks < nk; ++ks, ++g) {
        const int s = (int)(g % STAGES);
        /* Ring safety: slot g%SFS was last read by tcgen05.cp(g-SFS) and the
         * payload buffer by MMA(g-STAGES); commit(g-SFS) covers both
         * (cumulative snapshot; SFS <= STAGES). */
        if (g >= SFS) {
          const long long ge = g - SFS;
          if (lane == 0)
            mbar_wait(smem_u32(s_empty + (int)(ge % STAGES)),
                      (unsigned)((ge / STAGES) & 1));
          __syncwarp();
        }
        unsigned char* sfa = sfa_base + (int)(g % SFS) * F4_SFB;
        const int cur = ks & 1;
#pragma unroll
        for (int t = 0; t < BM / 32; ++t) {
          const int r = lane + 32 * t;
          const unsigned int* w4 =
              reinterpret_cast<const unsigned int*>(&sfa_r[cur][t]);
#pragma unroll
          for (int w = 0; w < F4_NK; ++w)
            *reinterpret_cast<unsigned int*>(sfa + f4_sf_off(r, w)) = w4[w];
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (ks + 1 < nk) {
          const int nxt = (ks + 1) & 1;
#pragma unroll
          for (int t = 0; t < BM / 32; ++t)
            sfa_r[nxt][t] = *reinterpret_cast<const uint4*>(
                asrc[t] + (size_t)(ks0 + ks + 1) * 16);
        }
        __syncwarp();
        if (lane == 0)
          asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
                       :: "r"(smem_u32(s_full + s)) : "memory");
      }
    }
  } else if (warp == 1) {
    /* ------------------ producer B: TMA payloads + weight scales ---------- */
    long long g = 0;
    for (long long item = start; item < total_items; item += istride) {
      int j, n0, ks0;
      if constexpr (SK > 1) {
        const int inner = n_tiles * SK;
        j = (int)(item / inner);
        const int rem = (int)(item - (long long)j * inner);
        const int ksplit = rem / n_tiles;
        n0 = (rem - ksplit * n_tiles) * BN;
        ks0 = ksplit * nk;
      } else {
        j = (int)(item / n_tiles);
        n0 = (int)(item - (long long)j * n_tiles) * BN;
        ks0 = 0;
      }
      const int e = p.texp[j];
      const int m0 = p.tm0[j];
      const int row_base = p.offs[e] + m0;
      /* AMODE 1: this warp's four gather4 token rows, clamped to the tile's
       * valid range exactly like the base kernel's s_row fill (padding rows
       * re-read row 0; the epilogue discards them). */
      int tok4[4];
      if constexpr (GEMM1 && AMODE == 1) {
        const int rows_valid = p.counts[e] - m0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int r = lane * 4 + i;
          const int ii = (r < rows_valid) ? r : 0;
          tok4[i] = p.ptok[row_base + ii];
        }
      }
      const uint8_t* Bsf = Wsf + (size_t)e * esf_stride;
      /* B-scale staging, two forms.
       *
       * GEMM2 (gated=false): phys_row is a 32-row-group-local shuffle
       * (pr = (rl & ~31) + sigma(rl & 31)), so the sixteen production bytes
       * at (512B kset block of pr>>7) + (pr&31)*16 hold tile rows rho+32m
       * (m=0..3) with word m at +4m -- byte-congruent with the f4_sf_off
       * smem chunk for (rho, j) whenever n0 % 64 == 0: BN=128 stages a whole
       * chunk with one cp.async16 (4x fewer L1 requests than the 4B storm
       * NCU flagged: GEMM2 L1TEX 64%, short-scoreboard 40.8%,
       * dv_616bb3921e2f); BN=64 takes the half's two row groups as the 8
       * bytes at +0 or +8 selected by (n0 & 64), one cp.async8 per (rho, j).
       *
       * GEMM1 keeps the per-row cp.async4 form: its phys_row interleave
       * (ii = 2*rl gate / 2*(rl-half_n)+1 up) makes production chunk q hold
       * logical rows {s, s+16, s+32, s+48} (s = f(q), word m = r>>4), which
       * maps to NO contiguous 8/16B smem span. Chunked GEMM1 staging
       * corrupted every WS3-GEMM1 output on real-data cases (job
       * dv_2ea26323a41d: bitexact=False on fixed_2..7/domain_3; fixed_8/9
       * matched only because empty-routing / zero-input data hide any
       * B-scale error). */
      const uint8_t* bsfp[NSFB][BN / 32];
      unsigned dbs = 0;
      const uint8_t* bchunk0 = nullptr;
      if constexpr (GEMM1) {
#pragma unroll
        for (int h = 0; h < NSFB; ++h)
#pragma unroll
          for (int t = 0; t < BN / 32; ++t) {
            const int rl = n0 + lane + 32 * t + (h ? half_n : 0);
            bsfp[h][t] = Bsf + sf_offset(phys_row(rl, half_n, true), 0, c4);
          }
      } else {
        const int rho = ((lane & 7) << 2) | (lane >> 3); /* sigma^-1: pr&31=lane */
        dbs = (unsigned)((((rho >> 3) * 4) * 8 + (rho & 7)) * 16);
        bchunk0 = Bsf + (size_t)(n0 >> 7) * c4 * 512 + lane * 16
                + ((n0 & 64) ? 8 : 0);
      }
      const int g3 = e * (N >> 5) + (GEMM1 ? (n0 >> 4) : (n0 >> 5));
      for (int ks = 0; ks < nk; ++ks, ++g) {
        const int s = (int)(g % STAGES);
        if (g >= SFS) {
          const long long ge = g - SFS;
          if (lane == 0)
            mbar_wait(smem_u32(s_empty + (int)(ge % STAGES)),
                      (unsigned)((ge / STAGES) & 1));
          __syncwarp();
        }
        const unsigned lb = smem_u32(s_full + s);
        const int c0 = (ks0 + ks) * F4_ROWB;
        if constexpr (GEMM1 && AMODE == 1) {
          /* One tile::gather4 per lane (four rows x 128B into 512B), armed
           * by lane 0 exactly like the base kernel's AMODE-1 path. */
          if (lane == 0)
            tma_expect_tx(lb, (2u * BN + (unsigned)BM) * F4_ROWB);
          __syncwarp();
          tma_gather4(tma, smem_u32(a_base + s * A_BYTES) + 512u * lane, lb,
                      c0, tok4[0], tok4[1], tok4[2], tok4[3]);
          if (lane == 0) {
            tma_box_5d(tmb, smem_u32(b_base + s * B_BYTES), lb, c0, 0, 0,
                       g3, 0);
            tma_box_5d(tmb, smem_u32(b_base + s * B_BYTES) + BN * F4_ROWB, lb,
                       c0, 0, 0, g3, 1);
          }
        } else if (lane == 0) {
          if constexpr (GEMM1) {
            tma_expect_tx(lb, (2u * BN + (unsigned)BM) * F4_ROWB);
            tma_box_2d(tma, smem_u32(a_base + s * A_BYTES), lb, c0, row_base);
            tma_box_5d(tmb, smem_u32(b_base + s * B_BYTES), lb, c0, 0, 0, g3, 0);
            tma_box_5d(tmb, smem_u32(b_base + s * B_BYTES) + BN * F4_ROWB, lb,
                       c0, 0, 0, g3, 1);
          } else {
            tma_expect_tx(lb, ((unsigned)BN + BM) * F4_ROWB);
            tma_box_2d(tma, smem_u32(a_base + s * A_BYTES), lb, c0, row_base);
            tma_box_4d(tmb, smem_u32(b_base + s * B_BYTES), lb, c0, 0, 0, g3);
          }
        }
        {
          const int slot = (int)(g % SFS);
          const size_t ko = (size_t)(ks0 + ks) * (F4_NK * 512);
          if constexpr (GEMM1) {
#pragma unroll
            for (int h = 0; h < NSFB; ++h) {
              unsigned char* sb = sfb_base + (h * SFS + slot) * F4_SFB;
#pragma unroll
              for (int t = 0; t < BN / 32; ++t)
#pragma unroll
                for (int w = 0; w < F4_NK; ++w)
                  cp_async4(sb + f4_sf_off(lane + 32 * t, w),
                            bsfp[h][t] + ko + 512 * w);
            }
          } else {
            unsigned char* sb = sfb_base + slot * F4_SFB + dbs;
            const uint8_t* sc = bchunk0 + ko;
#pragma unroll
            for (int j = 0; j < F4_NK; ++j) {
              if constexpr (BN % 128 == 0) {
                cp_async16(sb + 128 * j, sc + 512 * j);
              } else {
                cp_async8(sb + 128 * j, sc + 512 * j);
              }
            }
          }
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n"
                     :: "r"(lb) : "memory");
      }
    }
  } else if (warp == 2) {
    /* ------------------------------- MMA -------------------------------- */
    long long g = 0;
    long long icnt = 0;
    for (long long item = start; item < total_items; item += istride, ++icnt) {
      if constexpr (PP) {
        if (icnt >= 2) {
          /* Buffer (icnt&1) was last drained by item icnt-2; that drain's
           * arrival is completion #(icnt/2 - 1) of epidone[icnt&1]. */
          if (lane == 0)
            mbar_wait(smem_u32(s_epidone + (icnt & 1)),
                      (unsigned)(((icnt - 2) >> 1) & 1));
          __syncwarp();
        }
      } else if (icnt > 0) {
        if (lane == 0)
          mbar_wait(smem_u32(s_epidone), (unsigned)((icnt - 1) & 1));
        __syncwarp();
      }
      for (int ks = 0; ks < nk; ++ks, ++g) {
        const int s = (int)(g % STAGES);
        if (lane == 0)
          mbar_wait(smem_u32(s_full + s), (unsigned)((g / STAGES) & 1));
        __syncwarp();
        if (elect_one()) {
          tc_fence_after_sync();
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          const int sfs = (int)(g % SFS);
          const unsigned accj = tmem + (unsigned)(PP ? (icnt & 1) * ACC : 0);
          const unsigned sfa_t =
              tmem + NACC * ACC + sfs * (SFC * (NSFB + 1));
          const unsigned sfa_s = smem_u32(sfa_base + sfs * F4_SFB);
#pragma unroll
          for (int q = 0; q < F4_NK; ++q)
            tmem_cp_32x128b_warpx4(sfa_t + 4 * q,
                                   make_sf_desc(sfa_s + 128 * q, F4_NK * 128));
          unsigned sfb_t[NSFB];
#pragma unroll
          for (int h = 0; h < NSFB; ++h) {
            sfb_t[h] = sfa_t + SFC * (h + 1);
            const unsigned s_ =
                smem_u32(sfb_base + (h * SFS + sfs) * F4_SFB);
#pragma unroll
            for (int q = 0; q < F4_NK; ++q)
              tmem_cp_32x128b_warpx4(sfb_t[h] + 4 * q,
                                     make_sf_desc(s_ + 128 * q, F4_NK * 128));
          }
          const unsigned long long a_d = make_smem_desc(
              smem_u32(a_base + s * A_BYTES), TC_SBO, TC_LBO, TC_MODE);
          const unsigned long long b_d = make_smem_desc(
              smem_u32(b_base + s * B_BYTES), TC_SBO, TC_LBO, TC_MODE);
          const unsigned long long bg_d =
              GEMM1 ? make_smem_desc(
                          smem_u32(b_base + s * B_BYTES + BN * F4_ROWB),
                          TC_SBO, TC_LBO, TC_MODE)
                    : 0ULL;
#pragma unroll
          for (int q = 0; q < F4_NK; ++q) {
            const unsigned en = (ks == 0 && q == 0) ? 0u : 1u;
            mma_mxf4nvf4_ss(accj, a_d + 2 * q, b_d + 2 * q, idesc,
                            sfa_t + 4 * q, sfb_t[0] + 4 * q, en);
            if constexpr (GEMM1)
              mma_mxf4nvf4_ss(accj + BN, a_d + 2 * q, bg_d + 2 * q, idesc,
                              sfa_t + 4 * q, sfb_t[1] + 4 * q, en);
          }
          mma_commit(smem_u32(s_empty + s));
        }
      }
      if (elect_one())
        mma_commit(smem_u32(s_epifull + (PP ? (icnt & 1) : 0)));
    }
  } else {
    /* ----------------------------- epilogue ------------------------------ */
    /* TMEM lane quadrants are bound to (warp % 4), NOT (warp - 2): warps
     * 2..5 cover quadrants 2,3,0,1. The address lane field must equal the
     * executing warp's hardware quadrant (production form: alphamoe/
     * cake use `consumer_warp = warp % 4`). Using warp-2 here read the
     * wrong 32-row groups and corrupted every output row (job
     * dv_d2536361dc8b: 97% mismatch, rel_l2 1.35). */
    const int wl = warp & 3;                     /* TMEM lane quadrant */
    /* EPI=8: two warps share each quadrant and split the 32-col
     * c-chunks by parity ((warp-3)>>2), halving the per-warp drain
     * chain; the pair's 4-byte sf word becomes two independent
     * 2-byte stores (bytes 2*(c&1) of word c>>1), bit-identical
     * to the EPI=4 single uint32 store. */
    const int par = (EPI == 8) ? ((warp - 3) >> 2) : 0;
    long long icnt = 0;
    const unsigned tq = tmem + ((unsigned)(wl * 32) << 16);
    const int token_local = wl * 32 + lane;
    for (long long item = start; item < total_items; item += istride, ++icnt) {
      int j, n0, ksplit;
      if constexpr (SK > 1) {
        const int inner = n_tiles * SK;
        j = (int)(item / inner);
        const int rem = (int)(item - (long long)j * inner);
        ksplit = rem / n_tiles;
        n0 = (rem - ksplit * n_tiles) * BN;
      } else {
        j = (int)(item / n_tiles);
        n0 = (int)(item - (long long)j * n_tiles) * BN;
        ksplit = 0;
      }
      const int e = p.texp[j];
      const int m0 = p.tm0[j];
      const int row_base = p.offs[e] + m0;
      const int rows_valid = p.counts[e] - m0;
      const float alpha = p.alpha[e], beta = p.beta[e];
      const float inv_alpha = __frcp_rn(alpha), inv_beta = __frcp_rn(beta);
      const ActK ak = act_k_constants(p.act_type, alpha, inv_alpha, inv_beta);
      const float lim = p.has_limit ? p.limit[e] : 0.0f;
      const bool tok_ok = (token_local < rows_valid);
      const size_t route = (size_t)(row_base + token_local);
      if (lane == 0)
        mbar_wait(smem_u32(s_epifull + (PP ? (icnt & 1) : 0)),
                  (unsigned)(PP ? ((icnt >> 1) & 1) : (icnt & 1)));
      __syncwarp();
      tc_fence_after_sync();
      const unsigned taddr0 =
          tq + (unsigned)(PP ? (icnt & 1) * ACC : 0);
      if constexpr (SK > 1) {
        /* Split-K GEMM1: store the RAW fp32 slice partials; k_act_packed
         * reduces the slices in fixed order (identical to f4_body). */
        float* const pp = p.part + ((size_t)ksplit * p.Rmax + route)
                                   * (size_t)(2 * p.I);
#pragma unroll 1
        for (int c = par; c < BN / 32; c += (EPI == 8 ? 2 : 1)) {
          float up[32], gate[32];
          tmem_ld_x32(taddr0 + c * 32, up);
          tmem_ld_x32(taddr0 + BN + c * 32, gate);
          tmem_ld_wait();
          if (tok_ok) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              *reinterpret_cast<float4*>(pp + n0 + c * 32 + 4 * q) =
                  make_float4(up[4 * q], up[4 * q + 1], up[4 * q + 2],
                              up[4 * q + 3]);
              *reinterpret_cast<float4*>(pp + p.I + n0 + c * 32 + 4 * q) =
                  make_float4(gate[4 * q], gate[4 * q + 1], gate[4 * q + 2],
                              gate[4 * q + 3]);
            }
          }
        }
      } else if constexpr (GEMM1) {
      uint8_t* const pd = p.act_q + route * (size_t)(p.I >> 1) + (size_t)(n0 >> 1);
      unsigned int* const sd = reinterpret_cast<unsigned int*>(p.act_sf)
                             + route * (size_t)(p.I >> 6) + (size_t)(n0 >> 6);
      unsigned int sfacc = 0u;
#pragma unroll 1
      for (int c = par; c < BN / 32; c += (EPI == 8 ? 2 : 1)) {
        float up[32], gate[32];
        tmem_ld_x32(taddr0 + c * 32, up);
        tmem_ld_x32(taddr0 + BN + c * 32, gate);
        tmem_ld_wait();
        unsigned int wa = 0u, wb = 0u, wc = 0u, wd = 0u;
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
          float* gv = &gate[blk * 16];
          float pm[8];   /* per-pair maxima, tree-reduced below */
          if (p.has_limit) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const float g0 = fminf(gv[2 * q], lim);
              const float g1 = fminf(gv[2 * q + 1], lim);
              const float u0 = fminf(fmaxf(up[blk * 16 + 2 * q], -lim), lim);
              const float u1 = fminf(fmaxf(up[blk * 16 + 2 * q + 1], -lim), lim);
              const float2 a2 = activation2(p.act_type, make_float2(g0, g1),
                                            make_float2(u0, u1), alpha, beta,
                                            ak.k_sw, ak.k_gu, ak.k_uu);
              gv[2 * q] = a2.x; gv[2 * q + 1] = a2.y;
              pm[q] = fmaxf(fabsf(a2.x), fabsf(a2.y));
            }
          } else {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const float2 g2 = make_float2(gv[2 * q], gv[2 * q + 1]);
              const float2 u2 = make_float2(up[blk * 16 + 2 * q],
                                            up[blk * 16 + 2 * q + 1]);
              const float2 a2 = activation2(p.act_type, g2, u2, alpha, beta,
                                            ak.k_sw, ak.k_gu, ak.k_uu);
              gv[2 * q] = a2.x; gv[2 * q + 1] = a2.y;
              pm[q] = fmaxf(fabsf(a2.x), fabsf(a2.y));
            }
          }
          /* max is order-independent (fmaxf ignores NaN, +0 max -0 =
           * +0): the tree equals the old serial chain bit-exactly. */
          const float t0 = fmaxf(fmaxf(pm[0], pm[1]), fmaxf(pm[2], pm[3]));
          const float t1 = fmaxf(fmaxf(pm[4], pm[5]), fmaxf(pm[6], pm[7]));
          const float amax = fmaxf(t0, t1);
          const int sf_code = f32_to_e4m3(fminf(amax * (1.0f / 6.0f), 448.0f));
          const float sf = e4m3_val(sf_code);
          const int bidx = c * 2 + blk;
          sfacc |= (unsigned)sf_code << (8 * (bidx & 3));
          unsigned int w0 = 0u, w1 = 0u;
          if (BRLESS || sf > 0.0f) {
            const float inv_sf =
                BRLESS ? __frcp_rn(fmaxf(sf, 1.17549435e-38f))
                       : __frcp_rn(sf);
            const float2 inv2 = make_float2(inv_sf, inv_sf);
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const float2 v2 = f2_mul(make_float2(gv[2 * q], gv[2 * q + 1]),
                                       inv2);
              gv[2 * q] = v2.x; gv[2 * q + 1] = v2.y;
            }
            w0 = q_e2m1_pack8(gv[0], gv[1], gv[2], gv[3],
                              gv[4], gv[5], gv[6], gv[7]);
            w1 = q_e2m1_pack8(gv[8], gv[9], gv[10], gv[11],
                              gv[12], gv[13], gv[14], gv[15]);
          }
          if (blk == 0) { wa = w0; wb = w1; } else { wc = w0; wd = w1; }
        }
        if (tok_ok)
          *reinterpret_cast<uint4*>(pd + c * 16) = make_uint4(wa, wb, wc, wd);
        if constexpr (EPI == 8) {
          if (tok_ok)
            *reinterpret_cast<unsigned short*>(
                reinterpret_cast<unsigned char*>(sd) + 2 * (c & 1)) =
                (unsigned short)(sfacc >> (16 * (c & 1)));
        } else if ((c & 1) == 1 || c == (BN / 32) - 1) {
          if (tok_ok) sd[c >> 1] = sfacc;
          sfacc = 0u;
        }
      }
      } else {
        /* GEMM2: materialize each route's FC2 result as BF16 into rout
         * (identical rounding and stores to the f4_body epilogue). */
#pragma unroll 1
        for (int c = par; c < BN / 32; c += (EPI == 8 ? 2 : 1)) {
          float up[32];
          tmem_ld_x32(taddr0 + c * 32, up);
          tmem_ld_wait();
          __nv_bfloat16* dst = p.rout + route * p.H + n0 + c * 32;
          if (tok_ok) {
            unsigned int outw[16];
            for (int q = 0; q < 16; ++q)
              outw[q] = as_u32(__floats2bfloat162_rn(up[2 * q], up[2 * q + 1]));
            for (int q = 0; q < 4; ++q)
              *reinterpret_cast<uint4*>(dst + q * 8) =
                  make_uint4(outw[4 * q], outw[4 * q + 1], outw[4 * q + 2],
                             outw[4 * q + 3]);
          }
        }
      }
      /* TMEM is released once this warp's loads retired (tmem_ld_wait above);
       * the global stores may still be in flight. */
      __syncwarp();
      if (lane == 0)
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
                     :: "r"(smem_u32(s_epidone + (PP ? (icnt & 1) : 0)))
                     : "memory");
    }
  }
  __syncthreads();
  if (tid < 32) tmem_dealloc(tmem, TALLOC);
}

template <int BN, bool GEMM1, int STAGES, int SFS, int AMODE,
          bool BRLESS = true, int SK = 1, bool PP = false, int EPI = 4>
__global__ void __launch_bounds__((3 + EPI) * 32, EPI == 4 ? 2 : 1)
k_gemm_f4_ws2(const __grid_constant__ Params p,
              const __grid_constant__ CUtensorMap tm_b,
              const __grid_constant__ CUtensorMap tm_a) {
  f4_body_ws<BN, GEMM1, STAGES, SFS, AMODE, BRLESS, SK, PP, EPI>(p, &tm_b,
                                                                 &tm_a);
}

/* --------------- single-warp mainloop fam-2 body (rungs 33-36) -------------- */
/* NCU on the post-AGA kernels (job dv_360517988586, fixed_6 + domain_3):
 * GEMM1 rung 27 runs 9.4 warp-cycles per issued instruction with the CTA
 * BARRIER as the top stall (34% on domain_3) and only 1.94 active warps per
 * scheduler; GEMM2 rung 19 is worse at 19.2 cycles/instruction (54%
 * short-scoreboard). Both mainloops serialize four warps through two
 * __syncthreads per stage around work that a single warp can do: the SF
 * register prefetch/staging, the TMA box issues, and the one-thread
 * tcgen05.cp + MMA issue section. This body gives the ENTIRE mainloop to
 * warp 0 -- no barrier between stages at all -- while warps 1-3 park at one
 * syncthreads per item and then run the usual four-warp epilogue (the TMEM
 * quadrant reads and the activation/QDQ or BF16 stores stay parallel).
 * 2 CTAs per SM are kept, so the SM still interleaves two independent
 * stage streams; what disappears is the per-stage all-warp rendezvous.
 * Requires AMODE 2 for GEMM1 (A payload arrives by TMA box, so no cp.async
 * work is left for the other warps); GEMM2's A was always a box. The staged
 * shared-memory images (payload boxes, SF f4_sf_off layout, ring slots,
 * phase/gstage carry) are byte-identical to f4_body, so results are
 * bit-identical by construction -- verified by the cmp33-36 probes. */
template <int BN, bool GEMM1, int STAGES, int SFS, int SK>
DINLINE void f4_body_w1(const Params& p, const CUtensorMap* tmb,
                        const CUtensorMap* tma) {
  constexpr int BM = 128;
  constexpr int NROW = GEMM1 ? (2 * BN) : BN;
  constexpr int NSFB = GEMM1 ? 2 : 1;
  constexpr int A_BYTES = BM * F4_ROWB;
  constexpr int B_BYTES = NROW * F4_ROWB;
  constexpr int ACC = GEMM1 ? (2 * BN) : BN;
  constexpr int SFC = 4 * F4_NK;
  constexpr int TCOLS = ACC + SFS * SFC * (NSFB + 1);
  constexpr int TALLOC = TCOLS <= 256 ? 256 : 512;
  constexpr int BH = BN / 32;            /* B-SF row slots per lane (1 or 2... ) */

  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  const int total_tiles = p.toffs[p.E];
  const int n_tiles = (GEMM1 ? p.I : p.H) / BN;
  const long long total_items = (long long)total_tiles * n_tiles * SK;
  if (total_items <= 0) return;
  const int K = GEMM1 ? p.H : p.I;
  const int half_n = p.I;
  const int c4 = K >> 6;
  const int nk_full = K >> 8;
  const int nk = (SK > 1) ? (nk_full / SK) : nk_full;
  const int N = GEMM1 ? (2 * p.I) : p.H;
  const uint8_t* Asf = GEMM1 ? p.hidsf_g : p.act_sf;
  const size_t asf_stride = (size_t)(K >> 4);
  const uint8_t* Wsf = GEMM1 ? p.w1_sf : p.w2_sf;
  const size_t esf_stride = (size_t)N * (size_t)(K >> 4);

  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* const a_base = smem_raw;
  unsigned char* const b_base = smem_raw + STAGES * A_BYTES;
  unsigned char* const sfa_base = b_base + STAGES * B_BYTES;
  unsigned char* const sfb_base = sfa_base + SFS * F4_SFB;
  unsigned* const s_tmem =
      reinterpret_cast<unsigned*>(sfb_base + SFS * NSFB * F4_SFB);
  unsigned long long* s_bar = reinterpret_cast<unsigned long long*>(s_tmem + 4);
  unsigned long long* s_lbar = s_bar + STAGES;
  int32_t* s_row = reinterpret_cast<int32_t*>(s_lbar + STAGES);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  if (tid == 0) {
    mbar_init(smem_u32(s_bar), 1);
    for (int i = 0; i < STAGES; ++i) mbar_init(smem_u32(s_lbar + i), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < 32) tmem_alloc(s_tmem, TALLOC);
  __syncthreads();
  const unsigned tmem = *s_tmem;
  const unsigned bar = smem_u32(s_bar);
  unsigned phase = 0;                    /* warp0-private from here on */
  int gstage = 0;
  const unsigned idesc = make_instr_desc(0u, 1u, 1u, (unsigned)(BN >> 3),
                                         (unsigned)(128 >> 4), TC_M_SHIFT, 0u);

  for (int item = blockIdx.y * gridDim.x + blockIdx.x; item < total_items;
       item += gridDim.x * gridDim.y) {
    int j, n0, ksplit;
    if constexpr (SK > 1) {
      const int inner = n_tiles * SK;
      j = item / inner;
      const int rem = item - j * inner;
      ksplit = rem / n_tiles;
      n0 = (rem - ksplit * n_tiles) * BN;
    } else {
      j = item / n_tiles;
      n0 = (item - j * n_tiles) * BN;
      ksplit = 0;
    }
    const int ks0 = ksplit * nk;
    const int e = p.texp[j];
    const int m0 = p.tm0[j];
    const int row_base = p.offs[e] + m0;
    const int rows_valid = p.counts[e] - m0;
    const uint8_t* Bsf = Wsf + (size_t)e * esf_stride;
    const float alpha = p.alpha[e], beta = p.beta[e];
    const float inv_alpha = __frcp_rn(alpha), inv_beta = __frcp_rn(beta);
    const float lim = p.has_limit ? p.limit[e] : 0.0f;

    for (int i = tid; i < BM; i += 128) {
      int ii = (i < rows_valid) ? i : 0;
      s_row[i] = GEMM1 ? (row_base + ii) : (row_base + ii);
    }
    __syncthreads();

    if (warp == 0) {
      /* Per-item B scale-row bases: lane covers rows lane+32h (h < BH) of each
       * B half (up/gate for GEMM1). */
      unsigned int bsfo[BH][NSFB];
#pragma unroll
      for (int h = 0; h < BH; ++h) {
        const int rr = lane + 32 * h;
#pragma unroll
        for (int g = 0; g < NSFB; ++g) {
          const int rl = (GEMM1 && g) ? (n0 + rr + half_n) : (n0 + rr);
          bsfo[h][g] = (unsigned int)sf_offset(phys_row(rl, half_n, GEMM1), 0, c4);
        }
      }
      uint4 sfa_r[4];
      unsigned int sfb_r[BH][NSFB][F4_NK];

#define W1_FETCH_SF(KS)                                                          \
      do {                                                                       \
        _Pragma("unroll")                                                        \
        for (int i = 0; i < 4; ++i)                                              \
          sfa_r[i] = *reinterpret_cast<const uint4*>(                            \
              Asf + (size_t)s_row[lane + 32 * i] * asf_stride                    \
                  + (size_t)(KS) * 16);                                          \
        _Pragma("unroll")                                                        \
        for (int h = 0; h < BH; ++h)                                             \
          _Pragma("unroll")                                                      \
          for (int g = 0; g < NSFB; ++g) {                                       \
            const uint8_t* src_ = Bsf + bsfo[h][g] + (size_t)(KS) * (F4_NK * 512); \
            _Pragma("unroll")                                                    \
            for (int q = 0; q < F4_NK; ++q)                                      \
              sfb_r[h][g][q] =                                                   \
                  *reinterpret_cast<const unsigned int*>(src_ + (size_t)q * 512);\
          }                                                                      \
      } while (0)

#define W1_PUT_SF(BUF)                                                           \
      do {                                                                       \
        unsigned char* sfa_ = sfa_base + (BUF) * F4_SFB;                         \
        _Pragma("unroll")                                                        \
        for (int i = 0; i < 4; ++i) {                                            \
          const unsigned int* awp_ =                                             \
              reinterpret_cast<const unsigned int*>(&sfa_r[i]);                  \
          const int row_ = lane + 32 * i;                                        \
          _Pragma("unroll")                                                      \
          for (int q = 0; q < F4_NK; ++q)                                        \
            *reinterpret_cast<unsigned int*>(sfa_ + f4_sf_off(row_, q)) = awp_[q]; \
        }                                                                        \
        _Pragma("unroll")                                                        \
        for (int h = 0; h < BH; ++h) {                                           \
          const int rr = lane + 32 * h;                                          \
          _Pragma("unroll")                                                      \
          for (int g = 0; g < NSFB; ++g) {                                       \
            unsigned char* sfb_ = sfb_base + (g * SFS + (BUF)) * F4_SFB;         \
            _Pragma("unroll")                                                    \
            for (int q = 0; q < F4_NK; ++q)                                      \
              *reinterpret_cast<unsigned int*>(sfb_ + f4_sf_off(rr, q)) =        \
                  sfb_r[h][g][q];                                                \
          }                                                                      \
        }                                                                        \
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");             \
      } while (0)

#define W1_LOAD_TMA(KS, BUF)                                                     \
      do {                                                                       \
        const unsigned lb_ = smem_u32(s_lbar + (BUF));                           \
        const int c0_ = (KS) * F4_ROWB;                                          \
        if constexpr (GEMM1) {                                                   \
          const int g3_ = e * (N >> 5) + (n0 >> 4);                              \
          tma_expect_tx(lb_, (2u * BN + (unsigned)BM) * F4_ROWB);                \
          tma_box_2d(tma, smem_u32(a_base + (BUF) * A_BYTES), lb_, c0_,          \
                     row_base);                                                  \
          tma_box_5d(tmb, smem_u32(b_base + (BUF) * B_BYTES), lb_, c0_, 0, 0,    \
                     g3_, 0);                                                    \
          tma_box_5d(tmb, smem_u32(b_base + (BUF) * B_BYTES) + BN * F4_ROWB,     \
                     lb_, c0_, 0, 0, g3_, 1);                                    \
        } else {                                                                 \
          const int g3_ = e * (N >> 5) + (n0 >> 5);                              \
          tma_expect_tx(lb_, (unsigned)(BN + BM) * F4_ROWB);                     \
          tma_box_2d(tma, smem_u32(a_base + (BUF) * A_BYTES), lb_, c0_,          \
                     row_base);                                                  \
          tma_box_4d(tmb, smem_u32(b_base + (BUF) * B_BYTES), lb_, c0_, 0, 0,    \
                     g3_);                                                       \
        }                                                                        \
      } while (0)

      const int g0 = gstage;
      for (int pre = 0; pre + 1 < SFS && pre < nk; ++pre) {
        W1_FETCH_SF(ks0 + pre);
        W1_PUT_SF((g0 + pre) % SFS);
      }
      for (int pre = 0; pre < STAGES - 1; ++pre) {
        if (pre < nk && lane == 0) W1_LOAD_TMA(ks0 + pre, (g0 + pre) % STAGES);
      }
      if (SFS - 1 < nk) W1_FETCH_SF(ks0 + SFS - 1);

      for (int ks = 0; ks < nk; ++ks) {
        const int gst = g0 + ks;
        const int st = gst % STAGES;
        const int pre = ks + STAGES - 1;
        const int pre_buf = (gst + STAGES - 1) % STAGES;
        /* Retire barrier for the MMA of stage ks-1: frees both the payload
         * slot about to be re-staged and the SF slot PUT_SF overwrites. */
        if (ks >= 1) {
          if (lane == 0) mbar_wait(bar, phase);
          phase ^= 1;
          __syncwarp();
        }
        if (ks + SFS - 1 < nk) W1_PUT_SF((gst + SFS - 1) % SFS);
        if (pre < nk && lane == 0) W1_LOAD_TMA(ks0 + pre, pre_buf);
        if (lane == 0)
          mbar_wait(smem_u32(s_lbar + st), (unsigned)((gst / STAGES) & 1));
        __syncwarp();
        if (elect_one()) {
          tc_fence_after_sync();
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          const int sfs_ = gst % SFS;
          const unsigned sfa_t = tmem + ACC + sfs_ * (SFC * (NSFB + 1));
          const unsigned sfa_s = smem_u32(sfa_base + sfs_ * F4_SFB);
#pragma unroll
          for (int q = 0; q < F4_NK; ++q)
            tmem_cp_32x128b_warpx4(sfa_t + 4 * q,
                                   make_sf_desc(sfa_s + 128 * q, F4_NK * 128));
          unsigned sfb_t[NSFB];
#pragma unroll
          for (int g = 0; g < NSFB; ++g) {
            sfb_t[g] = sfa_t + SFC * (g + 1);
            const unsigned s_ = smem_u32(sfb_base + (g * SFS + sfs_) * F4_SFB);
#pragma unroll
            for (int q = 0; q < F4_NK; ++q)
              tmem_cp_32x128b_warpx4(sfb_t[g] + 4 * q,
                                     make_sf_desc(s_ + 128 * q, F4_NK * 128));
          }
          const unsigned long long a_d = make_smem_desc(
              smem_u32(a_base + st * A_BYTES), TC_SBO, TC_LBO, TC_MODE);
          const unsigned long long b_d = make_smem_desc(
              smem_u32(b_base + st * B_BYTES), TC_SBO, TC_LBO, TC_MODE);
          const unsigned long long bg_d =
              GEMM1 ? make_smem_desc(smem_u32(b_base + st * B_BYTES
                                              + BN * F4_ROWB),
                                     TC_SBO, TC_LBO, TC_MODE)
                    : 0ULL;
#pragma unroll
          for (int q = 0; q < F4_NK; ++q) {
            const unsigned en = (ks == 0 && q == 0) ? 0u : 1u;
            mma_mxf4nvf4_ss(tmem, a_d + 2 * q, b_d + 2 * q, idesc,
                            sfa_t + 4 * q, sfb_t[0] + 4 * q, en);
            if (GEMM1)
              mma_mxf4nvf4_ss(tmem + BN, a_d + 2 * q, bg_d + 2 * q, idesc,
                              sfa_t + 4 * q, sfb_t[GEMM1 ? 1 : 0] + 4 * q, en);
          }
          mma_commit(bar);
        }
        __syncwarp();
        if (ks + SFS < nk) W1_FETCH_SF(ks0 + ks + SFS);
      }
      gstage = g0 + nk;
      if (lane == 0) mbar_wait(bar, phase);
      phase ^= 1;
#undef W1_FETCH_SF
#undef W1_PUT_SF
#undef W1_LOAD_TMA
    }
    /* All warps (not just warp 0) order the retired async TC work before the
     * epilogue's tcgen05.ld, mirroring f4_body's fence/sync sequence. */
    tc_fence_after_sync();
    __syncthreads();

    /* ---------------- epilogue: identical arithmetic to f4_body ------------ */
    const unsigned taddr0 = tmem + ((unsigned)(warp * 32) << 16);
    const int token_local = warp * 32 + lane;
    const bool tok_ok = (token_local < rows_valid);
    const size_t route = (size_t)(row_base + token_local);
    if constexpr (SK > 1) {
      float* const pp = p.part + ((size_t)ksplit * p.Rmax + route)
                                 * (size_t)(2 * p.I);
#pragma unroll 1
      for (int c = 0; c < BN / 32; ++c) {
        float up[32], gate[32];
        tmem_ld_x32(taddr0 + c * 32, up);
        tmem_ld_x32(taddr0 + BN + c * 32, gate);
        tmem_ld_wait();
        if (tok_ok) {
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            *reinterpret_cast<float4*>(pp + n0 + c * 32 + 4 * q) =
                make_float4(up[4 * q], up[4 * q + 1], up[4 * q + 2],
                            up[4 * q + 3]);
            *reinterpret_cast<float4*>(pp + p.I + n0 + c * 32 + 4 * q) =
                make_float4(gate[4 * q], gate[4 * q + 1], gate[4 * q + 2],
                            gate[4 * q + 3]);
          }
        }
      }
    } else if (GEMM1) {
      uint8_t* const pd = p.act_q + route * (size_t)(p.I >> 1) + (size_t)(n0 >> 1);
      unsigned int* const sd = reinterpret_cast<unsigned int*>(p.act_sf)
                             + route * (size_t)(p.I >> 6) + (size_t)(n0 >> 6);
      unsigned int sfacc = 0u;
#pragma unroll 1
      for (int c = 0; c < BN / 32; ++c) {
        float up[32], gate[32];
        tmem_ld_x32(taddr0 + c * 32, up);
        tmem_ld_x32(taddr0 + BN + c * 32, gate);
        tmem_ld_wait();
        unsigned int wa = 0u, wb = 0u, wc = 0u, wd = 0u;
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
          float amax = 0.0f;
          for (int q = 0; q < 16; ++q) {
            float g = gate[blk * 16 + q], u = up[blk * 16 + q];
            if (p.has_limit) {
              u = fminf(fmaxf(u, -lim), lim);
              g = fminf(g, lim);
            }
            float a = activation_fast(p, g, u, alpha, beta, inv_alpha, inv_beta);
            gate[blk * 16 + q] = a;
            amax = fmaxf(amax, fabsf(a));
          }
          const int sf_code = f32_to_e4m3(fminf(amax * (1.0f / 6.0f), 448.0f));
          const float sf = e4m3_val(sf_code);
          const int bidx = c * 2 + blk;
          sfacc |= (unsigned)sf_code << (8 * (bidx & 3));
          unsigned int w0 = 0u, w1 = 0u;
          if (sf > 0.0f) {
            const float inv_sf = __frcp_rn(sf);
            float* gv = &gate[blk * 16];
#pragma unroll
            for (int q = 0; q < 16; ++q)
              gv[q] = div_by_recip(gv[q], sf, inv_sf);
            w0 = q_e2m1_pack8(gv[0], gv[1], gv[2], gv[3],
                              gv[4], gv[5], gv[6], gv[7]);
            w1 = q_e2m1_pack8(gv[8], gv[9], gv[10], gv[11],
                              gv[12], gv[13], gv[14], gv[15]);
          }
          if (blk == 0) { wa = w0; wb = w1; } else { wc = w0; wd = w1; }
        }
        if (tok_ok)
          *reinterpret_cast<uint4*>(pd + c * 16) = make_uint4(wa, wb, wc, wd);
        if ((c & 1) == 1 || c == (BN / 32) - 1) {
          if (tok_ok) sd[c >> 1] = sfacc;
          sfacc = 0u;
        }
      }
    } else {
#pragma unroll 1
      for (int c = 0; c < BN / 32; ++c) {
        float up[32];
        tmem_ld_x32(taddr0 + c * 32, up);
        tmem_ld_wait();
        __nv_bfloat16* dst = p.rout + route * p.H + n0 + c * 32;
        if (tok_ok) {
          unsigned int outw[16];
          for (int q = 0; q < 16; ++q)
            outw[q] = as_u32(__floats2bfloat162_rn(up[2 * q], up[2 * q + 1]));
          for (int q = 0; q < 4; ++q)
            *reinterpret_cast<uint4*>(dst + q * 8) =
                make_uint4(outw[4 * q], outw[4 * q + 1], outw[4 * q + 2],
                           outw[4 * q + 3]);
        }
      }
    }
    __syncthreads();
  }
  if (tid < 32) tmem_dealloc(tmem, TALLOC);
}

template <int BN, bool GEMM1, int STAGES, int SFS, int SK>
__global__ void __launch_bounds__(128, GEMM1 ? 2 : 2)
k_gemm_f4_w1(const __grid_constant__ Params p,
             const __grid_constant__ CUtensorMap tm_b,
             const __grid_constant__ CUtensorMap tm_a) {
  f4_body_w1<BN, GEMM1, STAGES, SFS, SK>(p, &tm_b, &tm_a);
}

/* --------------------------- flat-stream GEMM2 (rung 23) --------------------- */
/* The item-structured loop re-primes its ring at every item and drains it at
 * the epilogue: with nk=1 (deepseek I=256 GEMM2) EVERY stage is a boundary,
 * so each item serialises prime-latency -> MMA -> epilogue (shape 4 GEMM2
 * measured 172-177 us against an 85 us traffic floor). This body instead runs
 * ONE stage stream across item boundaries:
 *   - copies for stage g+2 are issued at iteration g regardless of the item
 *     they belong to, so two stages of TMA latency stay in flight always;
 *   - item metadata lives in a 3-slot register ring resolved by identical
 *     broadcast loads on every thread (no shared state, no extra syncs); the
 *     resolve for item (g+3)/nk is issued at the END of iteration g and
 *     committed at the top of g+1 -- AFTER the boundary epilogue, whose slot
 *     it may alias;
 *   - the epilogue of item li-1 runs at the top of the first iteration of
 *     item li, right after the retire barrier for that item's last MMA and
 *     before the next MMA issue, so TMEM reuse is ordered exactly as in the
 *     item loop;
 *   - GEMM2 needs no s_row table: the A box coordinate is row_base and the
 *     A-scale row is row_base + clamp(tid, rows_valid), both from the ring.
 * Scale factors use the rung-17/19 register prefetch (SFS=2). */
template <int BN, int STAGES, int SFS, bool NK1 = false>
DINLINE void f4_flat_g2(const Params& p, const CUtensorMap* tmb,
                        const CUtensorMap* tma) {
  constexpr int BM = 128;
  constexpr int NROW = BN;
  constexpr int NSFB = 1;
  constexpr int A_BYTES = BM * F4_ROWB;
  constexpr int B_BYTES = NROW * F4_ROWB;
  constexpr int ACC = BN;
  constexpr int SFC = 4 * F4_NK;
  constexpr int TCOLS = ACC + SFS * SFC * (NSFB + 1);
  constexpr int TALLOC = TCOLS <= 256 ? 256 : 512;

  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  const int total_tiles = p.toffs[p.E];
  const int n_tiles = p.H / BN;
  const long long total_items = (long long)total_tiles * n_tiles;
  const long long my_start = blockIdx.y * (long long)gridDim.x + blockIdx.x;
  const long long stride = (long long)gridDim.x * gridDim.y;
  if (total_items <= 0 || my_start >= total_items) return;
  const int my_items = (int)((total_items - my_start + stride - 1) / stride);
  const int K = p.I;
  const int half_n = p.I;
  const int c4 = K >> 6;
  const int nk = K >> 8;
  const int N = p.H;
  /* C14: incremental item cursor. Resolves used to re-derive the item index
   * from the absolute stage counter with a 64-bit divmod (FLAT_RESOLVE) plus
   * 32-bit divmods per stage access; job X2 SASS sampling on fixed_4 shows
   * the integer families (IMAD 31%, ISETP 12%, BRA 12%, VIADD 7%) dominating
   * this kernel's stall profile. Resolves happen in strictly ascending
   * consecutive item order (prime 0..2, then every new LI at its first
   * appearance), so a stepped (jj_c, nidx_c) cursor reproduces them exactly
   * with ONE divmod pair per kernel launch. NK1 additionally folds li = S,
   * ks = 0 for the I == 256 band (the flat kernel's only production
   * consumer), eliminating the per-stage divisions outright. All values fit
   * int32 with orders of magnitude to spare across the declared domain
   * (total_items <= 2304 m-tiles x 56 n-tiles = 129024). */
  int jj_c = (int)(my_start / n_tiles);
  int nidx_c = (int)(my_start - (long long)jj_c * n_tiles);
  const int sq_ = (int)(stride / n_tiles);
  const int sr_ = (int)(stride - (long long)sq_ * n_tiles);
  const uint8_t* Asf = p.act_sf;
  const size_t asf_stride = (size_t)(K >> 4);
  const size_t esf_stride = (size_t)N * (size_t)(K >> 4);

  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* const a_base = smem_raw;
  unsigned char* const b_base = smem_raw + STAGES * A_BYTES;
  unsigned char* const sfa_base = b_base + STAGES * B_BYTES;
  unsigned char* const sfb_base = sfa_base + SFS * F4_SFB;
  unsigned* const s_tmem =
      reinterpret_cast<unsigned*>(sfb_base + SFS * NSFB * F4_SFB);
  unsigned long long* s_bar = reinterpret_cast<unsigned long long*>(s_tmem + 4);
  unsigned long long* s_lbar = s_bar + STAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  if (tid == 0) {
    mbar_init(smem_u32(s_bar), 1);
    for (int i = 0; i < STAGES; ++i) mbar_init(smem_u32(s_lbar + i), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < 32) tmem_alloc(s_tmem, TALLOC);
  __syncthreads();
  const unsigned tmem = *s_tmem;
  const unsigned bar = smem_u32(s_bar);
  unsigned phase = 0;
  const unsigned idesc = make_instr_desc(0u, 1u, 1u, (unsigned)(BN >> 3),
                                         (unsigned)(128 >> 4), TC_M_SHIFT, 0u);

  struct Meta { int row_base, rows_valid, n0, g3b; unsigned int bsfo; };
  Meta m0, m1, m2, mp;
  int pend_li = -1;
#define FLAT_META(SLOT) ((SLOT) == 0 ? m0 : ((SLOT) == 1 ? m1 : m2))
/* C14b: the generic (nk > 1) path keeps the ORIGINAL stateless LI-based
 * resolve. With nk > 1 the in-loop resolve can re-request an LI the prime
 * already resolved (e.g. nk=3, g=0 -> L=1), which the divmod form answers
 * idempotently but a stepping cursor would double-advance (job AE: forced
 * cfg2=23 on I=768 shapes hit an illegal access). Production only launches
 * this kernel through the NK1=true specialization (I == 256), where the
 * resolve sequence is strictly consecutive (proven + bit-exact, job AD). */
#define FLAT_RESOLVE(LI, DST)                                                    \
  do {                                                                           \
    const long long it_ = my_start + (long long)(LI) * stride;                   \
    const int jj_ = (int)(it_ / n_tiles);                                        \
    const int e_ = p.texp[jj_];                                                  \
    const int m0v_ = p.tm0[jj_];                                                 \
    const int n0v_ = (int)(it_ - (long long)jj_ * n_tiles) * BN;                 \
    const int rr_ = (tid < BN) ? tid : 0;                                        \
    DST.row_base = p.offs[e_] + m0v_;                                            \
    DST.rows_valid = p.counts[e_] - m0v_;                                        \
    DST.n0 = n0v_;                                                               \
    DST.g3b = e_ * (N >> 5);                                                     \
    DST.bsfo = (unsigned int)(e_ * esf_stride +                                  \
        sf_offset(phys_row(n0v_ + rr_, half_n, false), 0, c4));                  \
  } while (0)
#define FLAT_RESOLVE_NEXT(DST)                                                   \
  do {                                                                           \
    const int jj_ = jj_c;                                                        \
    const int e_ = p.texp[jj_];                                                  \
    const int m0v_ = p.tm0[jj_];                                                 \
    const int n0v_ = nidx_c * BN;                                                \
    const int rr_ = (tid < BN) ? tid : 0;                                        \
    DST.row_base = p.offs[e_] + m0v_;                                            \
    DST.rows_valid = p.counts[e_] - m0v_;                                        \
    DST.n0 = n0v_;                                                               \
    DST.g3b = e_ * (N >> 5);                                                     \
    DST.bsfo = (unsigned int)(e_ * esf_stride +                                  \
        sf_offset(phys_row(n0v_ + rr_, half_n, false), 0, c4));                  \
    nidx_c += sr_;                                                               \
    if (nidx_c >= n_tiles) { nidx_c -= n_tiles; ++jj_c; }                        \
    jj_c += sq_;                                                                 \
  } while (0)
#define FLAT_LOAD(S)                                                             \
  do {                                                                           \
    const int li_ = NK1 ? (int)(S) : (int)((S) / nk);                            \
    const int ks_ = NK1 ? 0 : (int)((S) - (long long)li_ * nk);                  \
    const Meta mm_ = FLAT_META(li_ % 3);                                         \
    const int pb_ = (int)((S) % STAGES);                                         \
    if (tid == 0) {                                                              \
      const unsigned lb_ = smem_u32(s_lbar + pb_);                               \
      const int c0_ = ks_ * F4_ROWB;                                             \
      tma_expect_tx(lb_, (unsigned)(BN + BM) * F4_ROWB);                         \
      tma_box_2d(tma, smem_u32(a_base + pb_ * A_BYTES), lb_, c0_, mm_.row_base); \
      tma_box_4d(tmb, smem_u32(b_base + pb_ * B_BYTES), lb_, c0_, 0, 0,          \
                 mm_.g3b + (mm_.n0 >> 5));                                       \
    }                                                                            \
  } while (0)
  uint4 sfa_r_;
  unsigned int sfb_r_[F4_NK];
#define FLAT_FETCH_SF(S)                                                         \
  do {                                                                           \
    const int li_ = NK1 ? (int)(S) : (int)((S) / nk);                            \
    const int ks_ = NK1 ? 0 : (int)((S) - (long long)li_ * nk);                  \
    const Meta mm_ = FLAT_META(li_ % 3);                                         \
    const int ii_ = (tid < mm_.rows_valid) ? tid : 0;                            \
    sfa_r_ = *reinterpret_cast<const uint4*>(                                    \
        Asf + (size_t)(mm_.row_base + ii_) * asf_stride + (size_t)ks_ * 16);     \
    if (tid < BN) {                                                              \
      const uint8_t* src_ = p.w2_sf + mm_.bsfo + (size_t)ks_ * (F4_NK * 512);    \
      _Pragma("unroll")                                                          \
      for (int j_ = 0; j_ < F4_NK; ++j_)                                         \
        sfb_r_[j_] = *reinterpret_cast<const unsigned int*>(src_ + (size_t)j_ * 512); \
    }                                                                            \
  } while (0)
#define FLAT_PUT_SF(BUF)                                                         \
  do {                                                                           \
    unsigned char* sfa_ = sfa_base + (BUF) * F4_SFB;                             \
    const unsigned int* awp_ = reinterpret_cast<const unsigned int*>(&sfa_r_);   \
    _Pragma("unroll")                                                            \
    for (int j_ = 0; j_ < F4_NK; ++j_)                                           \
      *reinterpret_cast<unsigned int*>(sfa_ + f4_sf_off(tid, j_)) = awp_[j_];    \
    if (tid < BN) {                                                              \
      unsigned char* sfb_ = sfb_base + ((BUF)) * F4_SFB;                         \
      _Pragma("unroll")                                                          \
      for (int j_ = 0; j_ < F4_NK; ++j_)                                         \
        *reinterpret_cast<unsigned int*>(sfb_ + f4_sf_off(tid, j_)) = sfb_r_[j_];\
    }                                                                            \
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");                 \
  } while (0)
#define FLAT_EPILOGUE(LI)                                                        \
  do {                                                                           \
    const Meta mm_ = FLAT_META((LI) % 3);                                        \
    const unsigned taddr0_ = tmem + ((unsigned)(warp * 32) << 16);               \
    const int token_local_ = warp * 32 + lane;                                   \
    const bool tok_ok_ = (token_local_ < mm_.rows_valid);                        \
    const size_t route_ = (size_t)(mm_.row_base + token_local_);                 \
    _Pragma("unroll 1")                                                          \
    for (int c = 0; c < BN / 32; ++c) {                                          \
      float up_[32];                                                             \
      tmem_ld_x32(taddr0_ + c * 32, up_);                                        \
      tmem_ld_wait();                                                            \
      __nv_bfloat16* dst_ = p.rout + route_ * p.H + mm_.n0 + c * 32;             \
      if (tok_ok_) {                                                             \
        unsigned int outw_[16];                                                  \
        for (int q = 0; q < 16; ++q)                                             \
          outw_[q] = as_u32(__floats2bfloat162_rn(up_[2 * q], up_[2 * q + 1]));  \
        for (int q = 0; q < 4; ++q)                                              \
          *reinterpret_cast<uint4*>(dst_ + q * 8) =                              \
              make_uint4(outw_[4 * q], outw_[4 * q + 1], outw_[4 * q + 2],       \
                         outw_[4 * q + 3]);                                      \
      }                                                                          \
    }                                                                            \
  } while (0)

  const int G = my_items * nk;
  for (int li = 0; li < 3; ++li) {
    if (li < my_items) {
      if (li == 0) { if constexpr (NK1) { FLAT_RESOLVE_NEXT(m0); } else { FLAT_RESOLVE(li, m0); } }
      else if (li == 1) { if constexpr (NK1) { FLAT_RESOLVE_NEXT(m1); } else { FLAT_RESOLVE(li, m1); } }
      else { if constexpr (NK1) { FLAT_RESOLVE_NEXT(m2); } else { FLAT_RESOLVE(li, m2); } }
    }
  }
  /* Prime: SF slot for stage 0 lands now; stage 1's goes to registers. */
  FLAT_FETCH_SF(0);
  FLAT_PUT_SF(0);
  for (int gp = 0; gp < STAGES - 1; ++gp)
    if (gp < G) FLAT_LOAD(gp);
  if (1 < G) FLAT_FETCH_SF(1);

  for (int g = 0; g < G; ++g) {
    const int li = NK1 ? g : (int)(g / nk);
    const int ks = NK1 ? 0 : (int)(g - (long long)li * nk);
    /* 1. retire barrier for MMA(g-1); at an item boundary this also frees
     *    TMEM for the epilogue below. */
    if (g >= 1) {
      if (warp == 0 && lane == 0) mbar_wait(bar, phase);
      phase ^= 1;
      __syncthreads();
    }
    /* 2. boundary: epilogue of the previous item BEFORE its metadata slot is
     *    overwritten by the pending resolve. */
    if (ks == 0 && g > 0) {
      FLAT_EPILOGUE(li - 1);
      __syncthreads();
    }
    /* 3. commit pending metadata (all threads resolved identical values). */
    if (pend_li >= 0) {
      const int slot_ = pend_li % 3;
      if (slot_ == 0) { m0 = mp; } else if (slot_ == 1) { m1 = mp; }
      else { m2 = mp; }
      pend_li = -1;
    }
    /* 4. SF store for stage g+1, then the payload copy for stage g+2. */
    if (g + 1 < G) FLAT_PUT_SF((g + 1) % SFS);
    if (g + 2 < G) FLAT_LOAD(g + 2);
    /* 5. wait for stage g's boxes and issue its MMAs. */
    if (tid == 0)
      mbar_wait(smem_u32(s_lbar + (g % STAGES)), (unsigned)((g / STAGES) & 1));
    __syncthreads();
    if (warp == 0 && elect_one()) {
      tc_fence_after_sync();
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      const int sfs_ = g % SFS;
      const unsigned sfa_t = tmem + ACC + sfs_ * (SFC * (NSFB + 1));
      const unsigned sfa_s = smem_u32(sfa_base + sfs_ * F4_SFB);
      for (int j = 0; j < F4_NK; ++j)
        tmem_cp_32x128b_warpx4(sfa_t + 4 * j,
                               make_sf_desc(sfa_s + 128 * j, F4_NK * 128));
      const unsigned sfb_t = sfa_t + SFC;
      const unsigned sfb_s = smem_u32(sfb_base + sfs_ * F4_SFB);
      for (int j = 0; j < F4_NK; ++j)
        tmem_cp_32x128b_warpx4(sfb_t + 4 * j,
                               make_sf_desc(sfb_s + 128 * j, F4_NK * 128));
      const unsigned long long a_d = make_smem_desc(
          smem_u32(a_base + (g % STAGES) * A_BYTES), TC_SBO, TC_LBO, TC_MODE);
      const unsigned long long b_d = make_smem_desc(
          smem_u32(b_base + (g % STAGES) * B_BYTES), TC_SBO, TC_LBO, TC_MODE);
      for (int j = 0; j < F4_NK; ++j) {
        const unsigned en = (ks == 0 && j == 0) ? 0u : 1u;
        mma_mxf4nvf4_ss(tmem, a_d + 2 * j, b_d + 2 * j, idesc, sfa_t + 4 * j,
                        sfb_t + 4 * j, en);
      }
      mma_commit(bar);
    }
    /* 6. tail: fetch SF for stage g+2 and resolve the metadata the NEXT
     *    iteration's copy needs (item of stage g+3), both into registers. */
    if (g + 2 < G) FLAT_FETCH_SF(g + 2);
    {
      const int L = (g + 3 < G)
                        ? (NK1 ? (g + 3) : (int)((long long)(g + 3) / nk))
                        : -1;
      if (L >= 0 &&
          (L % 3) != (NK1 ? ((g + 1) % 3) : (((g + 1) / nk) % 3))) {
        if constexpr (NK1) { FLAT_RESOLVE_NEXT(mp); }
        else { FLAT_RESOLVE(L, mp); }
        pend_li = L;
      }
    }
  }
  if (warp == 0 && lane == 0) mbar_wait(bar, phase);
  tc_fence_after_sync();
  __syncthreads();
  FLAT_EPILOGUE(my_items - 1);
  __syncthreads();
  if (tid < 32) tmem_dealloc(tmem, TALLOC);
#undef FLAT_META
#undef FLAT_RESOLVE
#undef FLAT_RESOLVE_NEXT
#undef FLAT_LOAD
#undef FLAT_FETCH_SF
#undef FLAT_PUT_SF
#undef FLAT_EPILOGUE
}

template <int BN, int STAGES, int SFS = 2, bool NK1 = false>
__global__ void __launch_bounds__(128, 2)
k_gemm_f4_flat(const __grid_constant__ Params p,
               const __grid_constant__ CUtensorMap tm_b,
               const __grid_constant__ CUtensorMap tm_a) {
  f4_flat_g2<BN, STAGES, SFS, NK1>(p, &tm_b, &tm_a);
}

/* ------------------- warp-specialized GEMM1 (rung 26) ---------------------- */
/* NCU on rung 17 (diagnostics job, shape 6): DRAM 4.4%, L2 11.7%, tensor
 * pipe ~33%, 0.34 issued warps/scheduler, 66% of cycles with no eligible
 * warp -- the monolithic 4-warp item loop serialises copy issue, barrier
 * waits, SF staging, MMA issue and the activation epilogue on the same
 * warps. This body ports the published DeepGEMM SM100 role choreography to
 * the NVFP4 block-scaled GEMM1: 384 threads, ONE CTA per SM, and
 *   warp 0    : B TMA boxes (up+gate) + A scale-factor staging,
 *   warps 1,3 : the gathered A payload halves via cp.async (the 1024
 *               requests per stage stay spread over 64 threads -- the C1
 *               A/B re-rejected single-warp TMA gather4: 572 vs 480 us),
 *   warp 2    : MMA issuer (12 tcgen05.cp + 8 MMAs + commit per stage),
 *   warps 4-7 : the epilogue (lane quadrant x both BN=64 column halves)
 *               running clamp/activation/QDQ for item j while the mainloop
 *               already stages item j+1 -- enabled by the TMEM accumulator
 *               ping-pong (2 x 128 columns) that the single-CTA TMEM budget
 *               pays for. Warps 8-11 are idle: production references never
 *               issue tcgen05.ld outside warp groups 0/1, and a WG2 epilogue
 *               half produced garbage + intermittent launch failures here.
 * Rings: payload STAGES=4 with SF slots aliased to payload slots (SFS=4),
 * so the producer's empty[s] wait transitively retires the previous
 * tcgen05.cp of that SF slot. TMEM: 2*128 ACC + 4*48 SF = 448 <= 512.
 * Barriers: full[s] (count 4 = arrive.expect_tx + warp0 SF-store arrive +
 * two cp.async half arrives), empty[s] (one commit),
 * tmem_full/empty[2] (commit / 128 epilogue threads)
 * with the DeepGEMM parity discipline: producers wait empty[s] with
 * (phase^1), the MMA warp waits full[s] with phase, phase flips on slot
 * wrap; the TMEM pair uses (j&1, (j>>1)&1). */
DINLINE void mbar_arrive(unsigned bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
               :: "r"(bar) : "memory");
}

/* Rung-26 bounded barrier waits with sticky timeout diagnostics. Legitimate
 * waits here are microseconds; 2^28 cycles (~150 ms at boost clock) is never
 * reached in a healthy run. If a site DOES time out, the sticky flag makes
 * every ws wait in the grid drain immediately -- the kernel finishes with
 * garbage instead of hanging the job -- and g_ws26_dbg records which site
 * stalled with which context. Read back with moe_ws26_readdbg (see
 * experiments/ws26_debug.py). Sites: 1 warp0 empty[s], 2 warp1/3 empty[s],
 * 3 warp2 full[s], 4 warp2 tmem_empty[j&1], 5 epilogue tmem_full[j&1].
 * ctx = (j << 20) | (ks << 8) | (s << 1) | phase. */
__device__ volatile unsigned int g_ws26_dbg[32];
/* Optional host-mapped mirror (cudaHostAlloc Mapped; device pointer set via
 * moe_ws26_sethostdbg): zero-copy stores land in HOST RAM, so the records
 * survive the cudaErrorLaunchFailure context teardown that killed the
 * device-side readback path on fixed:6. */
__device__ unsigned int* volatile g_ws26_hostdbg = nullptr;
DINLINE void ws_dbg_record(unsigned site, unsigned ctx, unsigned block) {
  atomicAdd((unsigned int*)&g_ws26_dbg[0], 1u);
  atomicAdd((unsigned int*)&g_ws26_dbg[site], 1u);
  atomicMax((unsigned int*)&g_ws26_dbg[8 + site], ctx);
  atomicMax((unsigned int*)&g_ws26_dbg[16 + site], block);
  volatile unsigned int* const h = g_ws26_hostdbg;
  if (h != nullptr) {
    /* Plain stores over PCIe (no host-memory atomics: native-atomic support
     * is not guaranteed and a faulting diagnostic would mask the real one).
     * Racy across CTAs, which is acceptable: any record beats none. */
    if (h[site] == 0u) {
      h[site] = ctx + 1u;              /* first-wins context per site */
      h[16 + site] = block + 1u;
    }
    h[0] = 1u;                         /* fired flag */
  }
}
DINLINE void ws_mbar_wait(unsigned bar, unsigned phase, unsigned site,
                          unsigned ctx) {
  if (g_ws26_dbg[0]) return;                     /* sticky: drain the grid */
  const unsigned long long t0 = clock64();
  unsigned done = 0;
  while (!done) {
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.u32 %0, 1, 0, p;\n\t}"
                 : "=r"(done) : "r"(bar), "r"(phase));
    if (!done && clock64() - t0 > (1ull << 28)) {
      ws_dbg_record(site, ctx, blockIdx.y * gridDim.x + blockIdx.x);
      return;
    }
  }
}

template <int BN, int STAGES, int SFS>
DINLINE void f4_ws_g1(const Params& p, const CUtensorMap* tmb) {
  constexpr int BM = 128;
  constexpr int NROW = 2 * BN;
  constexpr int A_BYTES = BM * F4_ROWB;
  constexpr int B_BYTES = NROW * F4_ROWB;
  constexpr int ACC = 2 * BN;                      /* up+gate cols per buffer */
  constexpr int SFC = 4 * F4_NK;                   /* SF cols per operand/slot */
  constexpr int TCOLS = 2 * ACC + SFS * SFC * 3;   /* ping-pong + SF ring */
  constexpr int TALLOC = TCOLS <= 256 ? 256 : 512;
  static_assert(SFS == STAGES, "SF slot must alias the payload slot here");

  pdl_wait();
  if (threadIdx.x < 32) pdl_trigger();
  const int total_tiles = p.toffs[p.E];
  const int n_tiles = p.I / BN;
  const long long total_items = (long long)total_tiles * n_tiles;
  const long long my_start = blockIdx.y * (long long)gridDim.x + blockIdx.x;
  const long long stride = (long long)gridDim.x * gridDim.y;
  if (total_items <= 0 || my_start >= total_items) return;
  const int my_items = (int)((total_items - my_start + stride - 1) / stride);
  const int K = p.H;
  const int half_n = p.I;
  const int c4 = K >> 6;
  const int nk = K >> 8;
  const int N = 2 * p.I;
  const size_t aq_stride = (size_t)(K >> 1);
  const size_t asf_stride = (size_t)(K >> 4);
  const size_t esf_stride = (size_t)N * (size_t)(K >> 4);

  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* const a_base = smem_raw;
  unsigned char* const b_base = smem_raw + STAGES * A_BYTES;
  unsigned char* const sfa_base = b_base + STAGES * B_BYTES;
  unsigned char* const sfb_base = sfa_base + SFS * F4_SFB;
  unsigned* const s_tmem =
      reinterpret_cast<unsigned*>(sfb_base + SFS * 2 * F4_SFB);
  unsigned long long* s_full = reinterpret_cast<unsigned long long*>(s_tmem + 4);
  unsigned long long* s_empty = s_full + STAGES;
  unsigned long long* s_tfull = s_empty + STAGES;
  unsigned long long* s_tempty = s_tfull + 2;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      /* FOUR arrivals per phase: warp0's arrive.expect_tx (the TMA arming
       * instruction IS one arrival), warp0's post-SF-store arrive, and the
       * two cp.async half arrives from warps 1/3. The first cut used 3 --
       * the fourth arrival leaked into the next phase, completing it before
       * the data was staged (nondeterministic garbage) and eventually
       * tripping the mbarrier into an illegal state (the intermittent
       * cudaErrorLaunchFailure; under unbounded waits it presented as the
       * original fixed:0 hang because a trapping warp stops arriving). */
      mbar_init(smem_u32(s_full + i), 4);
      mbar_init(smem_u32(s_empty + i), 1);
    }
    mbar_init(smem_u32(s_tfull + 0), 1);
    mbar_init(smem_u32(s_tfull + 1), 1);
    mbar_init(smem_u32(s_tempty + 0), 128);
    mbar_init(smem_u32(s_tempty + 1), 128);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < 32) tmem_alloc(s_tmem, TALLOC);
  __syncthreads();
  const unsigned tmem = *s_tmem;
  const unsigned idesc = make_instr_desc(0u, 1u, 1u, (unsigned)(BN >> 3),
                                         (unsigned)(128 >> 4), TC_M_SHIFT, 0u);

  if (warp == 0) {
    /* ---- B TMA producer + A scale factors ---- */
    int s = 0;
    unsigned phase = 0;
    for (int j = 0; j < my_items; ++j) {
      const long long it_ = my_start + (long long)j * stride;
      const int jj_ = (int)(it_ / n_tiles);
      const int n0 = (int)(it_ - (long long)jj_ * n_tiles) * BN;
      const int e = p.texp[jj_];
      const int m0 = p.tm0[jj_];
      const int row_base = p.offs[e] + m0;
      const int rows_valid = p.counts[e] - m0;
      const int g3_ = e * (N >> 5) + (n0 >> 4);
      int srow[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = lane + 32 * i;
        const int ii = (r < rows_valid) ? r : 0;
        srow[i] = p.ptok[row_base + ii];
      }
      for (int ks = 0; ks < nk; ++ks) {
        ws_mbar_wait(smem_u32(s_empty + s), phase ^ 1u, 1u,
                     ((unsigned)j << 20) | ((unsigned)ks << 8)
                         | ((unsigned)s << 1) | phase);
        if (lane == 0) {
          const unsigned lb_ = smem_u32(s_full + s);
          tma_expect_tx(lb_, (unsigned)(2 * BN) * F4_ROWB);
          const int c0_ = ks * F4_ROWB;
          tma_box_5d(tmb, smem_u32(b_base + s * B_BYTES), lb_, c0_, 0, 0, g3_, 0);
          tma_box_5d(tmb, smem_u32(b_base + s * B_BYTES) + BN * F4_ROWB, lb_,
                     c0_, 0, 0, g3_, 1);
        }
        unsigned char* sfa_ = sfa_base + s * F4_SFB;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 v = *reinterpret_cast<const uint4*>(
              p.hid_sf + (size_t)srow[i] * asf_stride + (size_t)ks * 16);
          const unsigned int* vw = reinterpret_cast<const unsigned int*>(&v);
          const int r = lane + 32 * i;
#pragma unroll
          for (int q = 0; q < F4_NK; ++q)
            *reinterpret_cast<unsigned int*>(sfa_ + f4_sf_off(r, q)) = vw[q];
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(s_full + s));
        s = (s + 1) % STAGES;
        phase ^= (unsigned)(s == 0);
      }
    }
  } else if (warp == 1 || warp == 3) {
    /* ---- gathered A halves (warp 1: rows 0-63, warp 3: rows 64-127);
       warp 1 additionally stages the B scale factors ---- */
    const int rbase = (warp == 1) ? 0 : 64;
    int s = 0;
    unsigned phase = 0;
    for (int j = 0; j < my_items; ++j) {
      const long long it_ = my_start + (long long)j * stride;
      const int jj_ = (int)(it_ / n_tiles);
      const int n0 = (int)(it_ - (long long)jj_ * n_tiles) * BN;
      const int e = p.texp[jj_];
      const int m0 = p.tm0[jj_];
      const int row_base = p.offs[e] + m0;
      const int rows_valid = p.counts[e] - m0;
      int tok[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = rbase + 2 * lane + i;
        const int ii = (r < rows_valid) ? r : 0;
        tok[i] = p.ptok[row_base + ii];
      }
      unsigned int bsfo[4];
      if (warp == 1) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int g = u >> 1, hh = u & 1;
          const int rr = lane + 32 * hh;            /* row inside the operand */
          const int rl = n0 + rr + (g ? half_n : 0);
          bsfo[u] = (unsigned int)((size_t)e * esf_stride
                      + sf_offset(phys_row(rl, half_n, true), 0, c4));
        }
      }
      for (int ks = 0; ks < nk; ++ks) {
        ws_mbar_wait(smem_u32(s_empty + s), phase ^ 1u, 2u,
                     ((unsigned)j << 20) | ((unsigned)ks << 8)
                         | ((unsigned)s << 1) | phase);
        unsigned char* adst = a_base + s * A_BYTES;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int r = rbase + 2 * lane + i;
          const uint8_t* src = p.hid_q + (size_t)tok[i] * aq_stride
                             + (size_t)ks * F4_ROWB;
#pragma unroll
          for (int c = 0; c < 8; ++c)
            cp_async16(adst + r * F4_ROWB + ((c ^ (r & 7)) * 16), src + c * 16);
        }
        cp_async_commit();
        unsigned int bsw[4][4];
        if (warp == 1) {
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const uint8_t* src_ = p.w1_sf + bsfo[u] + (size_t)ks * (F4_NK * 512);
#pragma unroll
            for (int q = 0; q < F4_NK; ++q)
              bsw[u][q] =
                  *reinterpret_cast<const unsigned int*>(src_ + (size_t)q * 512);
          }
        }
        cp_async_wait<0>();
        if (warp == 1) {
          unsigned char* sfb0 = sfb_base + (0 * SFS + s) * F4_SFB;
          unsigned char* sfb1 = sfb_base + (1 * SFS + s) * F4_SFB;
#pragma unroll
          for (int q = 0; q < F4_NK; ++q) {
            *reinterpret_cast<unsigned int*>(sfb0 + f4_sf_off(lane, q)) = bsw[0][q];
            *reinterpret_cast<unsigned int*>(sfb0 + f4_sf_off(lane + 32, q)) = bsw[1][q];
            *reinterpret_cast<unsigned int*>(sfb1 + f4_sf_off(lane, q)) = bsw[2][q];
            *reinterpret_cast<unsigned int*>(sfb1 + f4_sf_off(lane + 32, q)) = bsw[3][q];
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(s_full + s));
        s = (s + 1) % STAGES;
        phase ^= (unsigned)(s == 0);
      }
    }
  } else if (warp == 2) {
    /* ---- MMA issuer ---- */
    int s = 0;
    unsigned phase = 0;
    for (int j = 0; j < my_items; ++j) {
      ws_mbar_wait(smem_u32(s_tempty + (j & 1)),
                   (unsigned)(((j >> 1) & 1) ^ 1), 4u, ((unsigned)j << 20));
      tc_fence_after_sync();
      const unsigned acc = tmem + (unsigned)(j & 1) * ACC;
      for (int ks = 0; ks < nk; ++ks) {
        ws_mbar_wait(smem_u32(s_full + s), phase, 3u,
                     ((unsigned)j << 20) | ((unsigned)ks << 8)
                         | ((unsigned)s << 1) | phase);
        if (elect_one()) {
          tc_fence_after_sync();
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          const unsigned sfa_t = tmem + 2 * ACC + (unsigned)s * (SFC * 3);
          const unsigned sfa_s = smem_u32(sfa_base + s * F4_SFB);
          const unsigned sfb0_t = sfa_t + SFC;
          const unsigned sfb1_t = sfa_t + 2 * SFC;
          const unsigned sfb0_s = smem_u32(sfb_base + (0 * SFS + s) * F4_SFB);
          const unsigned sfb1_s = smem_u32(sfb_base + (1 * SFS + s) * F4_SFB);
#pragma unroll
          for (int q = 0; q < F4_NK; ++q) {
            tmem_cp_32x128b_warpx4(sfa_t + 4 * q,
                                   make_sf_desc(sfa_s + 128 * q, F4_NK * 128));
            tmem_cp_32x128b_warpx4(sfb0_t + 4 * q,
                                   make_sf_desc(sfb0_s + 128 * q, F4_NK * 128));
            tmem_cp_32x128b_warpx4(sfb1_t + 4 * q,
                                   make_sf_desc(sfb1_s + 128 * q, F4_NK * 128));
          }
          const unsigned long long a_d = make_smem_desc(
              smem_u32(a_base + s * A_BYTES), TC_SBO, TC_LBO, TC_MODE);
          const unsigned long long b_d = make_smem_desc(
              smem_u32(b_base + s * B_BYTES), TC_SBO, TC_LBO, TC_MODE);
          const unsigned long long bg_d = make_smem_desc(
              smem_u32(b_base + s * B_BYTES) + BN * F4_ROWB, TC_SBO, TC_LBO,
              TC_MODE);
#pragma unroll
          for (int q = 0; q < F4_NK; ++q) {
            const unsigned en = (ks == 0 && q == 0) ? 0u : 1u;
            mma_mxf4nvf4_ss(acc, a_d + 2 * q, b_d + 2 * q, idesc,
                            sfa_t + 4 * q, sfb0_t + 4 * q, en);
            mma_mxf4nvf4_ss(acc + BN, a_d + 2 * q, bg_d + 2 * q, idesc,
                            sfa_t + 4 * q, sfb1_t + 4 * q, en);
          }
          mma_commit(smem_u32(s_empty + s));
        }
        __syncwarp();
        s = (s + 1) % STAGES;
        phase ^= (unsigned)(s == 0);
      }
      if (elect_one()) {
        tc_fence_after_sync();
        mma_commit(smem_u32(s_tfull + (j & 1)));
      }
      __syncwarp();
    }
  } else if (warp >= 4 && warp < 8) {
    /* ---- epilogue: four WG1 warps x both column halves ----
     * Warps 8-11 are deliberately idle. The production references never
     * issue tcgen05.ld from outside warp groups 0/1 (cake_kimi_k3_situ
     * keeps TMEM reads on warps 0-3 and gives warps 8-11 the cp/mma
     * issue roles; alphamoe reads on warps 2-5), and the first cut of
     * this kernel -- with an 8-warp epilogue spanning warps 4-11 --
     * produced systematically wrong output on shapes that completed
     * (fixed:0: frac 1.0 vs reference, cfg 17 bit-exact on the same
     * bytes, zero barrier timeouts) and intermittent
     * cudaErrorLaunchFailure on others (fixed:6, and the second cfg-26
     * model in one process). Undefined lane mapping for WG2 readers is
     * the single common cause; the epilogue stays inside WG1 and takes
     * both BN=64 column halves serially, which the ACC ping-pong hides.
     * The 384-thread shape is kept so producer/MMA warp ids stay put. */
    const int quad = warp & 3;                     /* TMEM lane quadrant */
    for (int j = 0; j < my_items; ++j) {
      const long long it_ = my_start + (long long)j * stride;
      const int jj_ = (int)(it_ / n_tiles);
      const int n0 = (int)(it_ - (long long)jj_ * n_tiles) * BN;
      const int e = p.texp[jj_];
      const int m0 = p.tm0[jj_];
      const int row_base = p.offs[e] + m0;
      const int rows_valid = p.counts[e] - m0;
      const float alpha = p.alpha[e], beta = p.beta[e];
      const float inv_alpha = __frcp_rn(alpha), inv_beta = __frcp_rn(beta);
      const float lim = p.has_limit ? p.limit[e] : 0.0f;
      ws_mbar_wait(smem_u32(s_tfull + (j & 1)), (unsigned)((j >> 1) & 1), 5u,
                   ((unsigned)j << 20));
      tc_fence_after_sync();
      const int token_local = quad * 32 + lane;
      const bool tok_ok = (token_local < rows_valid);
      const size_t route = (size_t)(row_base + token_local);
#pragma unroll
      for (int c = 0; c < BN / 32; ++c) {
        const unsigned taddr = tmem + ((unsigned)(quad * 32) << 16)
                             + (unsigned)(j & 1) * ACC + (unsigned)c * 32;
        float up[32], gate[32];
        tmem_ld_x32(taddr, up);
        tmem_ld_x32(taddr + BN, gate);
        tmem_ld_wait();
        uint8_t* const pd = p.act_q + route * (size_t)(p.I >> 1)
                          + (size_t)(n0 >> 1) + (size_t)c * 16;
        unsigned char* const sd = p.act_sf + route * (size_t)(p.I >> 4)
                                + (size_t)(n0 >> 4) + (size_t)c * 2;
        unsigned int wa = 0u, wb = 0u, wc = 0u, wd = 0u;
        unsigned int sfacc = 0u;
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
          float vals[16];
          float amax = 0.0f;
#pragma unroll
          for (int q = 0; q < 16; ++q) {
            float g = gate[blk * 16 + q], u = up[blk * 16 + q];
            if (p.has_limit) {
              u = fminf(fmaxf(u, -lim), lim);
              g = fminf(g, lim);
            }
            const float a = activation_fast(p, g, u, alpha, beta, inv_alpha,
                                            inv_beta);
            vals[q] = a;
            amax = fmaxf(amax, fabsf(a));
          }
          const int sf_code = f32_to_e4m3(fminf(amax * (1.0f / 6.0f), 448.0f));
          const float sf = e4m3_val(sf_code);
          sfacc |= (unsigned)sf_code << (8 * blk);
          unsigned int w0 = 0u, w1 = 0u;
          if (sf > 0.0f) {
            const float inv_sf = __frcp_rn(sf);
#pragma unroll
            for (int q = 0; q < 16; ++q)
              vals[q] = div_by_recip(vals[q], sf, inv_sf);
            w0 = q_e2m1_pack8(vals[0], vals[1], vals[2], vals[3],
                              vals[4], vals[5], vals[6], vals[7]);
            w1 = q_e2m1_pack8(vals[8], vals[9], vals[10], vals[11],
                              vals[12], vals[13], vals[14], vals[15]);
          }
          if (blk == 0) { wa = w0; wb = w1; } else { wc = w0; wd = w1; }
        }
        if (tok_ok) {
          *reinterpret_cast<uint4*>(pd) = make_uint4(wa, wb, wc, wd);
          *reinterpret_cast<unsigned short*>(sd) = (unsigned short)sfacc;
        }
      }
      mbar_arrive(smem_u32(s_tempty + (j & 1)));
    }
  }
  __syncthreads();
  if (tid < 32) tmem_dealloc(tmem, TALLOC);
}

template <int BN, int STAGES, int SFS = STAGES>
__global__ void __launch_bounds__(384, 1)
k_gemm_f4_ws(const __grid_constant__ Params p,
             const __grid_constant__ CUtensorMap tm_b) {
  f4_ws_g1<BN, STAGES, SFS>(p, &tm_b);
}
/* Compile-only check hook: experiments/tc_probe.py builds the file with
 * -DTC_FORCE_INSTANTIATE so the tcgen05 template is fully instantiated (and
 * therefore type-checked and ptxas-assembled) without the production build or
 * the launch ladder depending on it yet. */
#ifdef TC_FORCE_INSTANTIATE
template __global__ void k_gemm_tc<128, true, 1, 64>(const Params);
template __global__ void k_gemm_tc<256, false, 1, 64>(const Params);
template __global__ void k_gemm_tc<128, true, 2, 64>(const Params);
template __global__ void k_gemm_tc<256, false, 2, 64>(const Params);
template __global__ void k_gemm_f4<128, true, 2>(const Params);
template __global__ void k_gemm_f4<128, false, 2>(const Params);
template __global__ void k_gemm_f4<128, true, 3>(const Params);
template __global__ void k_gemm_f4<128, false, 3>(const Params);
template __global__ void k_gemm_f4<128, true, 4>(const Params);
template __global__ void k_gemm_f4<128, false, 4>(const Params);
template __global__ void k_gemm_f4<64, true, 2>(const Params);
template __global__ void k_gemm_f4<64, false, 2>(const Params);
template __global__ void k_gemm_f4<64, true, 3>(const Params);
template __global__ void k_gemm_f4<64, false, 3>(const Params);
/* Taking the addresses forces both TMA instantiations to be compiled and
 * assembled without restating the __grid_constant__ parameter attributes in an
 * explicit instantiation. */
static void* const g_force_tma[] = {(void*)&k_gemm_f4_tma<128, true, 3>,
                                    (void*)&k_gemm_f4_tma<128, false, 3>,
                                    (void*)&k_gemm_f4_tma<64, true, 2, false>,
                                    (void*)&k_gemm_f4_tma<64, true, 2, true>,
                                    (void*)&k_gemm_f4_tma<128, true, 3, true>,
                                    (void*)&k_gemm_f4_tma<64, true, 3, false, 2, true>,
                                    (void*)&k_gemm_f4_tma<64, true, 2, false, 2, true>,
                                    (void*)&k_gemm_f4_tma<128, true, 3, false, 3, true>,
                                    (void*)&k_gemm_f4_tma<128, false, 3, false, 3, true>,
                                    (void*)&k_gemm_f4_tma<64, true, 2, false, 2, true, 4>,
                                    (void*)&k_gemm_f4_tma<64, true, 2, false, 2, true, 2>,
                                    (void*)&k_gemm_f4_tma<64, true, 2, true, 2, true>,
                                    (void*)&k_gemm_f4_tma<64, true, 3, false, 1, true>,
                                    (void*)&k_gemm_f4_tma<64, false, 2, false>,
                                    (void*)&k_gemm_f4_ws<64, 4, 4>};
#endif

/* ------------------------------------------------------------ host launcher */
}  /* namespace nvfp4moe */

/* Flat C ABI: every field is 64 bit so the ctypes mirror needs no padding. */
struct MoEABI {
  int64_t ids, scores, hid_q, hid_sf, w1_q, w1_sf, w2_q, w2_sf;
  int64_t alpha, beta, limit, out;
  int64_t counts, fill, offs, toffs, ptok, pslot, pexp, route_of, texp, tm0;
  int64_t hid_bf, fc1, act, rout;
  int64_t T, K, E, H, I, Rmax, max_tiles, act_type, has_limit, bm, cfg, stream;
  int64_t cfg2;   /* rung for GEMM2; appended so the ABI order stays stable */
  int64_t part;   /* split-K fp32 partials [sk][Rmax][2I] or 0 */
  int64_t sk;     /* split-K slice count (1 = fused epilogue path) */
  int64_t hidq_g;  /* pre-gathered activation payload [Rmax,H/2] or 0 */
  int64_t hidsf_g; /* pre-gathered activation scales  [Rmax,H/16] or 0 */
  int64_t cnt;     /* fused split-K arrival counters [cnt_n] or 0 */
  int64_t cnt_n;   /* counter count (0 when the rung does not fuse) */
  int64_t acc32;   /* REDF: [T,H] fp32 route accumulator or 0 */
};

extern "C" int moe_init(int64_t stream) {
  cudaStream_t s = (cudaStream_t)stream;
  nvfp4moe::k_init_tables<<<1, 256, 0, s>>>();
  return (int)cudaGetLastError();
}

namespace tmal {

using namespace nvfp4moe;

/* Persistent-grid cap. Every GEMM kernel here maps work as
 *   item = blockIdx.y * gridDim.x + blockIdx.x;  item += gridDim.x * gridDim.y
 * so the grid shape only sets how many CTAs share the item space, not which
 * items run. The host knows only an upper bound on the tile count (experts that
 * received no route produce none, and the bound is only available on device), so
 * a shape with few or no valid routes launches thousands of CTAs that each read
 * toffs[E], find no items and return. That cost is visible: with the packed-FP4
 * rung's 128-column N tile, domain case 4 (`empty` routing, 777 tokens) launches
 * 5664 + 1062 such CTAs and measured 89.0 us against 78.4 us for the
 * 256-column bf16 rung (jobs dv_b790fd5e280a / dv_344d0b8834a5). Capping the
 * grid at a multiple of the resident CTA count removes it. 0 (the default) keeps
 * the uncapped grid; experiments/profile_kernels.py A/Bs the values. */
static int g_grid_cap = -3;          /* -3 unread, -1 auto, 0 off, >0 explicit */
static unsigned g_grid_cap_gen = 0;  /* bumped on change; part of the graph key */

static int grid_cap_raw() {
  if (g_grid_cap == -3) {
    const char* e = getenv("NVFP4_MOE_GRID_CAP");
    g_grid_cap = e ? atoi(e) : -1;               /* default: automatic */
    if (g_grid_cap < -1) g_grid_cap = -1;
  }
  return g_grid_cap;
}

/* Automatic cap: two waves of resident CTAs. A tcgen05 CTA of either family
 * occupies all (bf16 rung) or half (packed FP4 GEMM2) of the 512 TMEM columns,
 * so at most one or two are resident per SM and extra CTAs only add launch and
 * retire cost. Measured on L20D (job dv_d874b979eba7, same tensors, back to
 * back): 2x148 = 296 beat both the uncapped grid and 592/1184 on every shape,
 * most on the ones that route no work at all -- domain case 4 (`empty` routing)
 * 82.1 -> 55.8 us and fixed case 8 59.2 -> 47.2 us -- while the largest shape
 * (fixed_6) stayed at 1271.9 vs 1280.1 us. The warp-mma ladder is NOT capped:
 * its CTAs are small and the grid is its only parallelism. */
static int grid_cap() {
  const int c = grid_cap_raw();
  if (c != -1) return c;
  static int auto_cap = -2;
  if (auto_cap == -2) {
    int dev = 0, sms = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        sms <= 0)
      auto_cap = 0;                              /* cannot tell: stay uncapped */
    else
      auto_cap = 2 * sms;
  }
  return auto_cap;
}
static inline unsigned grid_cap_gen() { return g_grid_cap_gen; }

static inline dim3 capped_grid_tc(int gx, int gy) {
  const int cap = grid_cap();
  if (cap > 0 && (long long)gx * (long long)gy > (long long)cap) {
    int ny = cap / gx;
    if (ny < 1) ny = 1;
    return dim3((unsigned)gx, (unsigned)ny);
  }
  return dim3((unsigned)gx, (unsigned)gy);
}

/* Rung 26 is 384 threads, ~152 KiB of dynamic smem and a 512-column TMEM
 * allocation: strictly ONE CTA per SM, so the two-wave cap would only queue a
 * second wave of idle-until-retire CTAs. Cap at one SM-wave; the persistent
 * item loop inside covers the rest of the item space. */
static inline dim3 capped_grid_1sm(int gx, int gy) {
  int cap = grid_cap();
  cap = (cap > 0) ? (cap + 1) / 2 : 0;
  if (cap <= 0) {
    int dev = 0, sms = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) ==
            cudaSuccess &&
        sms > 0)
      cap = sms;
  }
  if (cap > 0 && (long long)gx * (long long)gy > (long long)cap) {
    int ny = cap / gx;
    if (ny < 1) ny = 1;
    return dim3((unsigned)gx, (unsigned)ny);
  }
  return dim3((unsigned)gx, (unsigned)gy);
}

/* Programmatic Dependent Launch knob + launch helper. With the attribute a
 * kernel may begin launching while its stream predecessor still runs; the
 * in-kernel griddepcontrol.wait/trigger pair keeps memory ordering exact
 * (wait blocks until the predecessor has FULLY completed and flushed), so
 * results are bit-identical either way. Env NVFP4_MOE_PDL, read per eager
 * call like small_threshold and mixed into the graph key because the
 * attribute bakes into captured nodes. Kernels launched without the
 * attribute treat wait as an already-satisfied dependency and trigger as a
 * no-op, so the cold ladder rungs (k_gemm, k_gemm_tc, k_decode_hidden,
 * k_act) stay plain launches and remain correct in either mode.
 * Default ON, promoted by same-process 20-shape measurement (job
 * dv_c4e040f0f141, experiments/v0bench.py): geomean 1.053x fixed / 1.057x
 * domain, every shape >= 1.001x, largest on launch-bound shapes (domain_4
 * 1.197x, fixed_1 1.175x, fixed_8 1.151x, domain_0 1.149x); fixed_0 in
 * graph-replay mode still gained 1.043x, so the win is device-side overlap,
 * not host launch cost. Results are bit-identical by the wait/trigger
 * contract; the paired A/B re-verifies bit-exactness on every shape.
 * NVFP4_MOE_PDL=0 restores the serialized order for A/B evidence. */
static int pdl_on() {
  const char* e = getenv("NVFP4_MOE_PDL");
  if (e == nullptr) return 1;                  /* promoted default: ON */
  return atoi(e) != 0 ? 1 : 0;
}

template <typename F, typename... Args>
static inline cudaError_t launch_pdl(F kernel, dim3 grid, dim3 block,
                                     size_t smem, cudaStream_t s,
                                     Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[1];
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  if (pdl_on()) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&cfg, kernel, static_cast<Args&&>(args)...);
}

/* ---------------------------------------------------- TMA descriptor cache --
 * cuTensorMapEncodeTiled is reached through cudart's driver entry point, so this
 * object keeps linking only against cudart. A map describes geometry and never
 * contents, so a cache keyed on the pointers and dimensions stays valid when the
 * caller rewrites the tensors in place between calls -- which is what the frozen
 * checker does -- and the few microseconds of encoding are not paid per launch.
 *
 * One map covers all E experts of a weight tensor: dim 3 strides 32 rows, so the
 * kernel adds e * (rows_per_expert / 32) to the block coordinate. The remaining
 * dimensions express the frozen 32-row block shuffle as the 4x8 (GEMM2) or
 * 2x8-per-half (GEMM1) transpose that it is, with TMA's dim1-fastest shared write
 * order making shared row == logical row, which is what the scale staging and the
 * epilogue already assume. Those strides are therefore not ascending;
 * experiments/tma_box_probe.py measured that the driver accepts them and that the
 * resulting image is byte-identical to the cp.async staging loop in all ten
 * geometries tried (job dv_816ecb04c15a). */
struct TmaSet { CUtensorMap b; CUtensorMap a; };

typedef CUresult (*encode_tiled_fn)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static encode_tiled_fn encode_entry() {
  static encode_tiled_fn fn = nullptr;
  static bool tried = false;
  if (!tried) {
    tried = true;
    void* sym = nullptr;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &sym, 12000,
                                         cudaEnableDefault, nullptr) == cudaSuccess)
      fn = reinterpret_cast<encode_tiled_fn>(sym);
  }
  return fn;
}

static bool encode_common(CUtensorMap* tm, void* base, unsigned rank,
                          const cuuint64_t* dim, const cuuint64_t* str,
                          const cuuint32_t* box) {
  encode_tiled_fn fn = encode_entry();
  if (!fn || !base) return false;
  const cuuint32_t estr[5] = {1u, 1u, 1u, 1u, 1u};
  /* Experiment knob (NVFP4_MOE_L2P): 128 = L2_PROMOTION_L2_128B (production
   * default), 256 = L2_256B, 0 = none. Read once per process; maps are cached
   * per geometry afterwards. */
  static int l2p = -1;
  if (l2p < 0) {
    const char* e = getenv("NVFP4_MOE_L2P");
    l2p = e ? atoi(e) : 128;
    if (l2p != 0 && l2p != 128 && l2p != 256) l2p = 128;
  }
  const CUtensorMapL2promotion promo =
      (l2p == 0) ? CU_TENSOR_MAP_L2_PROMOTION_NONE
                 : ((l2p == 256) ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B
                                 : CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  return fn(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, base, dim, str, box, estr,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            promo,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

/* Weight tile: one box is `boxrows` logical rows x 128 payload bytes.
 * boxrows=128 (rungs 12/15) stages 16 KiB per half; boxrows=64 (rungs 13/14)
 * halves the box so a BN=64 tile is still exactly one box per half. Box-row
 * algebra (c1/c2 fastest dims, c3' = block index inside the box):
 *   gated:    shared row r_s = c1 + 2*c2 + 16*c3',  prepared row =
 *             32*(c3_base+c3') + 16*c1 + c2 + 8*c4  ->  logical n0 + r_s + c4*I
 *   ungated:  shared row r_s = c1 + 4*c2 + 32*c3',  prepared row =
 *             32*(c3_base+c3') + 8*c1 + c2          ->  logical n0 + r_s
 * so box[3] (blocks per box) is boxrows/16 gated, boxrows/32 ungated, and the
 * kernel's block coordinate c3_base = e*(N/32) + n0/16 (gated) or n0/32. */
static bool encode_weight(CUtensorMap* tm, void* base, unsigned rowbytes,
                          unsigned rows_total, bool gated, unsigned boxrows) {
  if (!base || (rowbytes & 15u) || rowbytes < 128u) return false;
  if (!rows_total || (rows_total % 32u)) return false;
  if (boxrows != 64u && boxrows != 128u) return false;
  cuuint64_t dim[5], str[4];
  cuuint32_t box[5];
  unsigned rank;
  dim[0] = rowbytes;
  box[0] = 128u;
  if (gated) {
    rank = 5u;
    dim[1] = 2; dim[2] = 8; dim[3] = rows_total / 32u; dim[4] = 2;
    str[0] = 16ull * rowbytes; str[1] = rowbytes;
    str[2] = 32ull * rowbytes; str[3] = 8ull * rowbytes;
    box[1] = 2u; box[2] = 8u; box[3] = boxrows / 16u; box[4] = 1u;
  } else {
    rank = 4u;
    dim[1] = 4; dim[2] = 8; dim[3] = rows_total / 32u;
    str[0] = 8ull * rowbytes; str[1] = rowbytes; str[2] = 32ull * rowbytes;
    box[1] = 4u; box[2] = 8u; box[3] = boxrows / 32u;
  }
  return encode_common(tm, base, rank, dim, str, box);
}

/* Contiguous rows (GEMM2's intermediate operand): one box is 128 x 128 bytes. */
static bool encode_rows(CUtensorMap* tm, void* base, unsigned rowbytes,
                        unsigned rows_total) {
  if (!base || (rowbytes & 15u) || rowbytes < 128u || !rows_total) return false;
  const cuuint64_t dim[2] = {rowbytes, rows_total};
  const cuuint64_t str[1] = {rowbytes};
  const cuuint32_t box[2] = {128u, 128u};
  return encode_common(tm, base, 2u, dim, str, box);
}

/* Hidden-state byte matrix for tile::gather4 (GEMM1's activation operand):
 * a plain 2D UINT8 map, box {128, 1}; the gather instruction itself supplies
 * four row coordinates per box. */
static bool encode_gather(CUtensorMap* tm, void* base, unsigned rowbytes,
                          unsigned rows_total) {
  if (!base || (rowbytes & 15u) || rowbytes < 128u || !rows_total) return false;
  const cuuint64_t dim[2] = {rowbytes, rows_total};
  const cuuint64_t str[1] = {rowbytes};
  const cuuint32_t box[2] = {128u, 1u};
  return encode_common(tm, base, 2u, dim, str, box);
}

struct TmaCache {
  const void* wb; const void* wa;
  int rowb, rowsb, rowa, rowsa;
  int boxrows; int gather_a; const void* hidq; int hidrow, hidrows;
  TmaSet set; bool valid;
  /* A geometry the driver refused is cached too. Without this, every launch of a
   * shape whose rows are not describable -- an unaligned weight view, a row
   * shorter than a box, rows not a multiple of 32 -- would pay two driver calls
   * before falling back to the cp.async rung, which is a per-call cost invisible
   * in a correctness run and pure loss in a timed one. */
  bool refused;
};
static TmaCache g_tma[2];

/* Returns the maps for this GEMM, or nullptr when the geometry cannot be
 * described (unaligned row, fewer than 128 payload bytes per row, no driver
 * entry point); the caller then keeps the cp.async rung. */
static const TmaSet* tma_set(const Params& p, bool g1, int boxrows, int amode) {
  TmaCache& c = g_tma[g1 ? 0 : 1];
  const void* wb = g1 ? (const void*)p.w1_q : (const void*)p.w2_q;
  const void* wa = g1 ? ((amode == 2) ? (const void*)p.hidq_g : nullptr)
                      : (const void*)p.act_q;
  const int rowb = g1 ? (p.H >> 1) : (p.I >> 1);
  const int rowsb = (g1 ? (2 * p.I) : p.H) * p.E;
  const int rowa = (g1 && amode == 2) ? (p.H >> 1) : (p.I >> 1);
  const int rowsa = p.Rmax;
  const bool same_key = c.wb == wb && c.wa == wa && c.rowb == rowb
                     && c.rowsb == rowsb && c.rowa == rowa && c.rowsa == rowsa
                     && c.boxrows == boxrows && c.gather_a == amode
                     && c.hidq == (const void*)p.hid_q && c.hidrow == (p.H >> 1)
                     && c.hidrows == p.T;
  if (same_key && (c.valid || c.refused)) return c.valid ? &c.set : nullptr;
  c.valid = false;
  c.refused = true;                 /* every early return below is a refusal */
  c.wb = wb; c.wa = wa; c.rowb = rowb; c.rowsb = rowsb;
  c.rowa = rowa; c.rowsa = rowsa;
  c.boxrows = boxrows; c.gather_a = amode;
  c.hidq = (const void*)p.hid_q; c.hidrow = p.H >> 1; c.hidrows = p.T;
  if (rowb <= 0 || rowsb <= 0) return nullptr;
  if (!encode_weight(&c.set.b, const_cast<void*>(wb), (unsigned)rowb,
                     (unsigned)rowsb, g1, (unsigned)boxrows))
    return nullptr;
  if (g1) {
    if (amode == 1) {
      if (!encode_gather(&c.set.a, const_cast<void*>((const void*)p.hid_q),
                         (unsigned)(p.H >> 1), (unsigned)p.T))
        return nullptr;
    } else if (amode == 2) {
      /* Contiguous-rows map over the pre-gathered activation payload. */
      if (!wa || !encode_rows(&c.set.a, const_cast<void*>(wa), (unsigned)rowa,
                              (unsigned)(rowsa > 0 ? rowsa : 1)))
        return nullptr;
    } else {
      memset(&c.set.a, 0, sizeof(c.set.a));
    }
  } else if (!encode_rows(&c.set.a, const_cast<void*>(wa), (unsigned)rowa,
                          (unsigned)(rowsa > 0 ? rowsa : 1))) {
    return nullptr;
  }
  c.valid = true;
  c.refused = false;
  return &c.set;
}

/* Tile ladder. Larger tiles raise the FLOP-per-shared-byte ratio of the
 * ldmatrix operand traffic (the measured limiter): a warp tile of 32x64
 * (MT=2, NT=8) reaches about 10.7 FLOP/byte versus 5.3 for 16x32. Small-batch
 * shapes instead need many blocks, so the last rung keeps BN small.
 * p.bm must match the BM the host used to size the tile map. */
template <bool G1>
int launch_gemm(const Params& p, int cfg, cudaStream_t s) {
  const int N = G1 ? (2 * p.I) : p.H;
  if (cfg == 23) {
    /* Flat-stream GEMM2 (f4_flat_g2): one stage stream across item
     * boundaries. GEMM1 at cfg 23 behaves as rung 17. */
    if constexpr (!G1) {
      const TmaSet* tset = tma_set(p, false, 128, false);
      if (tset != nullptr) {
        constexpr int SMEM_ = 3 * (128 * F4_ROWB + 128 * F4_ROWB)
                            + 2 * 2 * F4_SFB + 128 * 4 + 64 + 3 * 8;
        static bool attr_done_flat = false;
        if (!attr_done_flat) {
          if (cudaFuncSetAttribute(k_gemm_f4_flat<128, 3, 2, false>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   SMEM_) != cudaSuccess) return 740;
          if (cudaFuncSetAttribute(k_gemm_f4_flat<128, 3, 2, true>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   SMEM_) != cudaSuccess) return 740;
          attr_done_flat = true;
        }
        /* C14: the NK1 specialization is exact only for nk == 1 (I == 256),
         * which is precisely the band the policy pairs with rung 23. */
        if (p.I == 256) {
          launch_pdl(k_gemm_f4_flat<128, 3, 2, true>,
                     capped_grid_tc(p.H / 128, p.max_tiles), dim3(128),
                     (size_t)SMEM_, s, p, tset->b, tset->a);
        } else {
          launch_pdl(k_gemm_f4_flat<128, 3, 2, false>,
                     capped_grid_tc(p.H / 128, p.max_tiles), dim3(128),
                     (size_t)SMEM_, s, p, tset->b, tset->a);
        }
        return (int)cudaGetLastError();
      }
      cfg = 12;   /* map refused: fall back through the rung-12 path (-> 8) */
    } else {
      cfg = 17;
    }
  }
#define LAUNCH(BM, BN, BK, WM, WN)                                              \
  do {                                                                          \
    constexpr int THREADS_ = (WM) * (WN) * 32;                                  \
    constexpr int SMEM_ =                                                       \
        2 * ((BM) * (BK) * 2 + (BN) * (BK) * 2) + (BM) * 4 + 2048;              \
    if (SMEM_ > 48 * 1024) {                                                    \
      static bool configured_ = false;                                          \
      if (!configured_) {                                                       \
        if (cudaFuncSetAttribute(k_gemm<BM, BN, BK, WM, WN, G1>,                \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                                 SMEM_) != cudaSuccess) return 90;              \
        configured_ = true;                                                     \
      }                                                                         \
    }                                                                           \
    k_gemm<BM, BN, BK, WM, WN, G1>                                             \
        <<<dim3(N / (BN), p.max_tiles), dim3(THREADS_), SMEM_, s>>>(p);         \
  } while (0)
  if (cfg >= 7 && cfg <= 70) {
    /* Block-scaled NVFP4 rungs: both operands stay packed, the E4M3 scales are
     * staged in shared memory and copied into TMEM by tcgen05.cp, and GEMM1
     * writes the intermediate as (payload, scale) so GEMM2 needs no decode
     * either. N is 128 because that is the operand width whose scale-factor TMEM
     * slot mapping is measured (experiments/tc_sf_map.py, and independently
     * verified bit-exact by experiments/tc_f4_verify.py). Rungs 8 and 9 deepen
     * the copy pipeline to 3 and 4 buffers: one CTA per SM here (512 and 448 of
     * the 512 TMEM columns), so the only way to overlap more HBM latency is a
     * longer ring, and shared memory allows 162 KiB / 216 KiB of the 227 KiB. */
    constexpr int BM_ = 128;
#define F4_LAUNCH_S(BNV, G1V, STV, TAG)                                         \
    do {                                                                        \
      constexpr int NROW_ = (G1V) ? 2 * (BNV) : (BNV);                          \
      constexpr int NSFB_ = (G1V) ? 2 : 1;                                      \
      constexpr int SMEM_ = (STV) * (BM_ * F4_ROWB + NROW_ * F4_ROWB            \
                                     + (NSFB_ + 1) * F4_SFB) + BM_ * 4 + 64;    \
      static bool attr_done_f4_##TAG = false;                                   \
      if (!attr_done_f4_##TAG) {                                                \
        if (cudaFuncSetAttribute(k_gemm_f4<BNV, G1V, STV>,                      \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                                 SMEM_) != cudaSuccess) return 700 + (BNV);     \
        attr_done_f4_##TAG = true;                                              \
      }                                                                         \
      launch_pdl(k_gemm_f4<BNV, G1V, STV>,                                      \
                 capped_grid_tc((G1V ? p.I : p.H) / (BNV), p.max_tiles),        \
                 dim3(128), (size_t)SMEM_, s, p);                               \
      { cudaError_t le2_ = cudaGetLastError();                                  \
        if (le2_ != cudaSuccess) return 4000 + (int)le2_; }                     \
    } while (0)
#define F4_LAUNCH(BNV, G1V, STV) F4_LAUNCH_S(BNV, G1V, STV, s##STV##n##BNV##g##G1V)
    /* Rung 12: rung 8's three-buffer ring with the operand tiles staged by TMA,
     * which replaces 3072 (GEMM1) or 2048 (GEMM2) sixteen-byte cp.async requests
     * per stage with two or one 16-KiB box.
     * Rungs 13/14: BN=64 with two buffers -- ACC 128 + SF ring 2*3*16 = 224 of
     * the 256-column TMEM budget, so TWO CTAs are resident per SM, which the E1
     * override sweep measured as the dominant GEMM1 factor on filled shapes
     * (rung 10 vs 12: fixed_6 822 vs 936 us, geomean 1.046x). Rung 14 stages the
     * gathered activation operand with TMA tile::gather4 (BM/4 four-row boxes)
     * instead of 1024 cp.async requests per stage; rung 13 keeps cp.async A.
     * Rung 15: rung 12 plus gather4 A (isolates gather4 at BN=128).
     * Every TMA rung falls back to the matching cp.async rung (BN=64 -> 10,
     * BN=128 -> 8) when the tensor maps cannot be encoded for this geometry,
     * so selecting one is always safe. */
    const bool wide = (cfg == 12 || cfg == 15 || cfg == 18 || cfg == 19 ||
                       cfg == 36 || cfg == 38 || cfg == 44 || cfg == 46 ||
                       cfg == 47 || cfg == 48 || cfg == 49 || cfg == 52 ||
                       cfg == 53 || cfg == 57 || cfg == 66 ||
                       cfg == 67 || cfg == 68);
    const bool ga = (cfg == 14 || cfg == 15 || cfg == 24 || cfg == 63 ||
                     cfg == 64);
    const bool r16 = (cfg == 16);
    /* GEMM1 A-operand mode: 1 = tile::gather4 from hid_q, 2 = plain 2-D boxes
     * over the pre-gathered route-order buffer (k_gather_a ran first). */
    const int amode = !G1 ? 0 : (ga ? 1
                      : (((cfg >= 27 && cfg <= 35) || cfg == 37 || cfg == 39
                          || (cfg >= 41 && cfg <= 43) || cfg == 45
                          || cfg == 48 || cfg == 49 || cfg == 50 || cfg == 51
                          || cfg == 54 || cfg == 55 || cfg == 56
                          || cfg == 58 || cfg == 59 || cfg == 65
                          || cfg == 69 || cfg == 70)
                         ? 2 : 0));
    /* Only the TMA rungs may touch the map cache: tma_set RE-ENCODES on a key
     * change, and a cp.async rung querying it with a different boxrows would
     * replace the cached maps out from under a later TMA launch (this shipped
     * once as an illegal access: rung 12 ran on 64-row boxes). */
    const TmaSet* tset = (cfg >= 12) ? tma_set(p, G1, wide ? 128 : 64, amode)
                                     : nullptr;
    if (cfg >= 12 && tset != nullptr) {
#define F4_TMA_LAUNCH_S(BNV, G1V, STV, TAV, SFV, PV, SKV, TAG)                    \
      do {                                                                       \
        constexpr int NROW_ = (G1V) ? 2 * (BNV) : (BNV);                          \
        constexpr int NSFB_ = (G1V) ? 2 : 1;                                     \
        constexpr int SMEM_ = (STV) * (BM_ * F4_ROWB + NROW_ * F4_ROWB)           \
                              + (SFV) * (NSFB_ + 1) * F4_SFB                      \
                              + BM_ * 4 + 64 + (STV) * 8;                        \
        static bool attr_done_t_##TAG = false;                                    \
        if (!attr_done_t_##TAG) {                                                 \
          if (cudaFuncSetAttribute(k_gemm_f4_tma<BNV, G1V, STV, TAV, SFV, PV, SKV>, \
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,  \
                                   SMEM_) != cudaSuccess) return 720 + (BNV);    \
          attr_done_t_##TAG = true;                                               \
        }                                                                        \
        cudaError_t le_ = launch_pdl(k_gemm_f4_tma<BNV, G1V, STV, TAV, SFV, PV, SKV>, \
                   capped_grid_tc(((G1V ? p.I : p.H) / (BNV)) * (SKV),          \
                                  p.max_tiles),                                 \
                   dim3(128), (size_t)SMEM_, s, p, tset->b, tset->a);           \
        if (le_ != cudaSuccess) return 3000 + (int)le_;                         \
      } while (0)
#define F4_TMA_LAUNCH(BNV, G1V, STV, TAV) \
      F4_TMA_LAUNCH_S(BNV, G1V, STV, TAV, STV, false, 1, t##STV##n##BNV##g##G1V##a##TAV)
#define F4_TMA_LAUNCH_C(BNV, G1V, STV, TAV, SFV, PV, SKV, BRV, TAG)             \
      do {                                                                       \
        constexpr int NROW_ = (G1V) ? 2 * (BNV) : (BNV);                          \
        constexpr int NSFB_ = (G1V) ? 2 : 1;                                     \
        constexpr int SMEM_ = (STV) * (BM_ * F4_ROWB + NROW_ * F4_ROWB)           \
                              + (SFV) * (NSFB_ + 1) * F4_SFB                      \
                              + BM_ * 4 + 64 + (STV) * 8;                        \
        static bool attr_done_c_##TAG = false;                                    \
        if (!attr_done_c_##TAG) {                                                 \
          if (cudaFuncSetAttribute(                                               \
                  k_gemm_f4_tma<BNV, G1V, STV, TAV, SFV, PV, SKV, false, BRV,     \
                                true>,                                            \
                  cudaFuncAttributeMaxDynamicSharedMemorySize,                    \
                  SMEM_) != cudaSuccess) return 790;                              \
          attr_done_c_##TAG = true;                                               \
        }                                                                        \
        cudaError_t lec_ = launch_pdl(                                            \
            k_gemm_f4_tma<BNV, G1V, STV, TAV, SFV, PV, SKV, false, BRV, true>,    \
            capped_grid_tc(((G1V ? p.I : p.H) / (BNV)) * (SKV), p.max_tiles),     \
            dim3(128), (size_t)SMEM_, s, p, tset->b, tset->a);                    \
        if (lec_ != cudaSuccess) return 3800 + (int)lec_;                         \
      } while (0)
      if constexpr (G1) {
        if (cfg == 26) {
          /* Warp-specialized GEMM1: 12 warps, one CTA per SM, BN=64,
           * STAGES=SFS=4, B via 64-row TMA boxes (the rung-17 maps), A via
           * cp.async gather on warps 1/3. Grid capped at one SM-wave. */
          constexpr int BN_ = 64, STV_ = 4, SFV_ = 4;
          constexpr int SMEM_ = STV_ * (BM_ * F4_ROWB + 2 * BN_ * F4_ROWB)
                              + SFV_ * 3 * F4_SFB + 4 * 4 + 12 * 8;
          static bool attr_done_ws26 = false;
          if (!attr_done_ws26) {
            if (cudaFuncSetAttribute(k_gemm_f4_ws<BN_, STV_, SFV_>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     SMEM_) != cudaSuccess) return 760;
            attr_done_ws26 = true;
          }
          launch_pdl(k_gemm_f4_ws<BN_, STV_, SFV_>,
                     capped_grid_1sm(p.I / BN_, p.max_tiles), dim3(384),
                     (size_t)SMEM_, s, p, tset->b);
        } else if (cfg == 20 || cfg == 21) {
          /* Split-K small-batch GEMM1 (rung 17 tiling, K sliced into SK
           * contiguous stage groups; raw fp32 partials + k_act_packed). */
          if (cfg == 20) { F4_TMA_LAUNCH_S(64, true, 2, 0, 2, true, 4, t2n64g1s2pk4); }
          else { F4_TMA_LAUNCH_S(64, true, 2, 0, 2, true, 2, t2n64g1s2pk2); }
        } else if (cfg == 33 || cfg == 34 || cfg == 35) {
          /* Single-warp mainloop (rungs 27/28/29 tiling with WMODE). */
          constexpr int SMEM_ = 2 * (BM_ * F4_ROWB + 2 * 64 * F4_ROWB)
                              + 2 * 3 * F4_SFB + BM_ * 4 + 64 + 2 * 8;
#define W1_LAUNCH(SKV, TAG)                                                      \
      do {                                                                       \
        static bool attr_done_w_##TAG = false;                                   \
        if (!attr_done_w_##TAG) {                                                \
          if (cudaFuncSetAttribute(k_gemm_f4_w1<64, true, 2, 2, SKV>,            \
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,  \
                                   SMEM_) != cudaSuccess) return 770;            \
          attr_done_w_##TAG = true;                                              \
        }                                                                        \
        cudaError_t lew_ = launch_pdl(k_gemm_f4_w1<64, true, 2, 2, SKV>,         \
                     capped_grid_tc((p.I / 64) * (SKV), p.max_tiles),            \
                     dim3(128), (size_t)SMEM_, s, p, tset->b, tset->a);          \
        if (lew_ != cudaSuccess) return 3600 + (int)lew_;                        \
      } while (0)
          if (cfg == 35) { W1_LAUNCH(4, sk4); }
          else if (cfg == 34) { W1_LAUNCH(2, sk2); }
          else { W1_LAUNCH(1, sk1); }
#undef W1_LAUNCH
        } else if (cfg == 39) {
          /* Rung 27 + SFPRE2 (depth-2 batched SF register prefetch). */
          F4_TMA_LAUNCH_S(64, true, 2, 2, 2, 2, 1, t2n64g1s2p2ag);
        } else if (cfg == 40) {
          /* Rung 17 + SFPRE2 (cp.async-gather A; deepseek-large band). */
          F4_TMA_LAUNCH_S(64, true, 2, 0, 2, 2, 1, t2n64g1s2p2);
        } else if (cfg == 37) {
          /* Rung 29 tiling with an eight-way K split (nk=1 per item; legal
           * when H/256 % 8 == 0, i.e. H in {2048, 4096}): the tiny-batch
           * GEMM1 chain shrinks from 2 stages x 2.6 items to 1 stage x 5.2. */
          F4_TMA_LAUNCH_S(64, true, 2, 2, 2, true, 8, t2n64g1s2pagk8);
        } else if (cfg == 27 || cfg == 28 || cfg == 29) {
          /* Rungs 17/21/20 with the pre-gathered A operand (AMODE 2): one
           * 2-D TMA box replaces the 1024 cp.async gathers per stage. */
          if (cfg == 27) { F4_TMA_LAUNCH_S(64, true, 2, 2, 2, true, 1, t2n64g1s2pag); }
          else if (cfg == 28) { F4_TMA_LAUNCH_S(64, true, 2, 2, 2, true, 2, t2n64g1s2pagk2); }
          else { F4_TMA_LAUNCH_S(64, true, 2, 2, 2, true, 4, t2n64g1s2pagk4); }
        } else if (cfg == 41 || cfg == 42 || cfg == 43) {
          /* C11c experiment: rungs 27/28/29 with the branchless quantize
           * guard (41: fused GEMM1 epilogue; 42/43: k_act_packed<true>,
           * selected in run_pipeline). */
#define F4_TMA_LAUNCH_B(BNV, G1V, STV, TAV, SFV, PV, SKV, TAG)                  \
      do {                                                                      \
        constexpr int NROW_ = (G1V) ? 2 * (BNV) : (BNV);                        \
        constexpr int NSFB_ = (G1V) ? 2 : 1;                                    \
        constexpr int SMEM_ = (STV) * (BM_ * F4_ROWB + NROW_ * F4_ROWB)         \
                            + (SFV) * (NSFB_ + 1) * F4_SFB                      \
                            + BM_ * 4 + 64 + (STV) * 8;                         \
        static bool attr_done_b_##TAG = false;                                  \
        if (!attr_done_b_##TAG) {                                               \
          if (cudaFuncSetAttribute(                                             \
                  k_gemm_f4_tma<BNV, G1V, STV, TAV, SFV, PV, SKV, false, true>, \
                  cudaFuncAttributeMaxDynamicSharedMemorySize,                  \
                  SMEM_) != cudaSuccess) return 780;                            \
          attr_done_b_##TAG = true;                                             \
        }                                                                       \
        cudaError_t leb_ = launch_pdl(                                          \
            k_gemm_f4_tma<BNV, G1V, STV, TAV, SFV, PV, SKV, false, true>,       \
            capped_grid_tc(((G1V ? p.I : p.H) / (BNV)) * (SKV), p.max_tiles),  \
            dim3(128), (size_t)SMEM_, s, p, tset->b, tset->a);                  \
        if (leb_ != cudaSuccess) return 3700 + (int)leb_;                       \
      } while (0)
          if (cfg == 41) { F4_TMA_LAUNCH_B(64, true, 2, 2, 2, 1, 1, t2n64g1s2pagb); }
          else if (cfg == 42) { F4_TMA_LAUNCH_B(64, true, 2, 2, 2, 1, 2, t2n64g1s2pagk2b); }
          else { F4_TMA_LAUNCH_B(64, true, 2, 2, 2, 1, 4, t2n64g1s2pagk4b); }
#undef F4_TMA_LAUNCH_B
        } else if (cfg == 45) {
          /* C15: rung 41 with SF staged by cp.async (SFCP), no register
           * prefetch leg. */
          F4_TMA_LAUNCH_C(64, true, 2, 2, 2, 0, 1, true, t2n64g1s2pagc);
        } else if (cfg == 31 || cfg == 32) {
          /* Rungs 29/28 with the split-K activation reduce fused into the
           * last-arriving CTA of each (m-tile, n-tile) group. */
#define F4_TMA_LAUNCH_F(BNV, G1V, STV, TAV, SFV, PV, SKV, TAG)                  \
      do {                                                                      \
        constexpr int NROW_ = (G1V) ? 2 * (BNV) : (BNV);                        \
        constexpr int NSFB_ = (G1V) ? 2 : 1;                                    \
        constexpr int SMEM_ = (STV) * (BM_ * F4_ROWB + NROW_ * F4_ROWB)         \
                            + (SFV) * (NSFB_ + 1) * F4_SFB                      \
                            + BM_ * 4 + 64 + (STV) * 8 + 16;                    \
        static bool attr_done_u_##TAG = false;                                  \
        if (!attr_done_u_##TAG) {                                               \
          if (cudaFuncSetAttribute(                                             \
                  k_gemm_f4_tma<BNV, G1V, STV, TAV, SFV, PV, SKV, true>,        \
                  cudaFuncAttributeMaxDynamicSharedMemorySize,                  \
                  SMEM_) != cudaSuccess) return 750 + (BNV);                    \
          attr_done_u_##TAG = true;                                             \
        }                                                                       \
        cudaError_t lef_ = launch_pdl(                                          \
            k_gemm_f4_tma<BNV, G1V, STV, TAV, SFV, PV, SKV, true>,              \
            capped_grid_tc(((G1V ? p.I : p.H) / (BNV)) * (SKV), p.max_tiles),   \
            dim3(128), (size_t)SMEM_, s, p, tset->b, tset->a);                  \
        if (lef_ != cudaSuccess) return 3500 + (int)lef_;                       \
      } while (0)
          if (cfg == 31) { F4_TMA_LAUNCH_F(64, true, 2, 2, 2, true, 4, t2n64g1s2pk4f); }
          else { F4_TMA_LAUNCH_F(64, true, 2, 2, 2, true, 2, t2n64g1s2pk2f); }
#undef F4_TMA_LAUNCH_F
        } else if (cfg == 30) {
          /* Rung 16 (three payload buffers, two-slot SF ring) + pre-gathered
           * A: with the A gather gone the deeper B ring may finally pay
           * (pre-AGA it measured 0.98-1.01x). smem 111.2 KiB keeps 2 CTAs/SM
           * (2 x 111.2 <= 227). */
          F4_TMA_LAUNCH_S(64, true, 3, 2, 2, true, 1, t3n64g1s2pag);
        } else if (cfg == 24) {
          /* Rung 17 + TMA gather4 activation operand: the B2-era gather4 loss
           * was measured before SF register prefetch existed; with prefetch
           * the 1024 per-stage cp.async issues (all four warps; NCU shows the
           * mainloop short-scoreboard/issue bound at 0.34 issued warps per
           * scheduler) become 32 boxes on warp 0. */
          F4_TMA_LAUNCH_S(64, true, 2, 1, 2, true, 1, t2n64g1a1s2p);
        } else if (cfg == 25) {
          /* Three payload buffers with a SINGLE SF slot: TMEM 128+48=176 and
           * smem 102 KiB keep two CTAs per SM while the payload ring runs
           * three deep. SFS=1 is safe: the top-of-iteration retire barrier
           * already orders the slot refill after the consuming MMA. */
          F4_TMA_LAUNCH_S(64, true, 3, 0, 1, true, 1, t3n64g1s1p);
        } else if (r16) {
          /* BN=64, three payload buffers, two-slot SF ring with register
           * prefetch: TMEM 128+2*48=224, smem 111 KiB -> two CTAs per SM
           * while the payload ring keeps two stages in flight. */
          F4_TMA_LAUNCH_S(64, true, 3, 0, 2, true, 1, t3n64g1s2p);
        } else if (cfg == 17) {
          /* Rung 13's two-buffer ring plus SF register prefetch. */
          F4_TMA_LAUNCH_S(64, true, 2, 0, 2, true, 1, t2n64g1s2p);
        } else if (cfg == 18) {
          /* Rung 12 plus SF register prefetch (BN=128, one CTA per SM). */
          F4_TMA_LAUNCH_S(128, true, 3, 0, 3, true, 1, t3n128g1s3p);
        } else if (cfg == 48 || cfg == 49) {
          /* BN=128 GEMM1 WITH the pre-gathered A operand (AMODE 2), SF
           * register prefetch depth 2 and the branchless quantize guard --
           * the BN=128 operand-traffic halving (items and A refetches both
           * halve) was only ever measured with the 1024-request cp.async A
           * gather (rung 18: 0.85-0.90x), never with the A box. TMEM 256+96
           * -> one CTA per SM, so the ring runs three deep (48) or two (49).
           * Experiment rungs; override-selected only. */
          constexpr int SMEM48_3 = 3 * (BM_ * F4_ROWB + 2 * 128 * F4_ROWB)
              + 2 * 3 * F4_SFB + BM_ * 4 + 64 + 3 * 8;
          constexpr int SMEM48_2 = 2 * (BM_ * F4_ROWB + 2 * 128 * F4_ROWB)
              + 2 * 3 * F4_SFB + BM_ * 4 + 64 + 2 * 8;
          if (cfg == 48) {
            static bool attr_done_r48 = false;
            if (!attr_done_r48) {
              if (cudaFuncSetAttribute(
                      k_gemm_f4_tma<128, true, 3, 2, 2, 2, 1, false, true>,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      SMEM48_3) != cudaSuccess) return 796;
              attr_done_r48 = true;
            }
            cudaError_t le48_ = launch_pdl(
                k_gemm_f4_tma<128, true, 3, 2, 2, 2, 1, false, true>,
                capped_grid_tc(p.I / 128, p.max_tiles), dim3(128),
                (size_t)SMEM48_3, s, p, tset->b, tset->a);
            if (le48_ != cudaSuccess) return 3960 + (int)le48_;
          } else {
            static bool attr_done_r49 = false;
            if (!attr_done_r49) {
              if (cudaFuncSetAttribute(
                      k_gemm_f4_tma<128, true, 2, 2, 2, 2, 1, false, true>,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      SMEM48_2) != cudaSuccess) return 797;
              attr_done_r49 = true;
            }
            cudaError_t le49_ = launch_pdl(
                k_gemm_f4_tma<128, true, 2, 2, 2, 2, 1, false, true>,
                capped_grid_tc(p.I / 128, p.max_tiles), dim3(128),
                (size_t)SMEM48_2, s, p, tset->b, tset->a);
            if (le49_ != cudaSuccess) return 3970 + (int)le49_;
          }
        } else if (cfg == 50 || cfg == 51) {
          /* PIPE GEMM1: rung 41's tiling (BN=64, AMODE 2, SFS=2, SFPRE2,
           * BRLESS) with the pipelined-commit mainloop; 51 runs the payload
           * ring three deep (smem 108.6 KiB keeps two CTAs per SM). */
#define F4_PIPE_LAUNCH(BNV, G1V, STV, TAV, BRV, TAG)                             \
  do {                                                                           \
    constexpr int NROW_ = (G1V) ? 2 * (BNV) : (BNV);                             \
    constexpr int NSFB_ = (G1V) ? 2 : 1;                                         \
    constexpr int SMEM_ = (STV) * (BM_ * F4_ROWB + NROW_ * F4_ROWB)              \
                        + 2 * (NSFB_ + 1) * F4_SFB                               \
                        + BM_ * 4 + 64 + (STV) * 8;                              \
    static bool attr_done_p_##TAG = false;                                       \
    if (!attr_done_p_##TAG) {                                                    \
      if (cudaFuncSetAttribute(                                                  \
              k_gemm_f4_tma<BNV, G1V, STV, TAV, 2, 2, 1, false, BRV, false,      \
                            false, true>,                                        \
              cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
              SMEM_) != cudaSuccess) return 798;                                 \
      attr_done_p_##TAG = true;                                                  \
    }                                                                            \
    cudaError_t lep_ = launch_pdl(                                               \
        k_gemm_f4_tma<BNV, G1V, STV, TAV, 2, 2, 1, false, BRV, false, false,     \
                      true>,                                                     \
        capped_grid_tc(((G1V ? p.I : p.H) / (BNV)), p.max_tiles),                \
        dim3(128), (size_t)SMEM_, s, p, tset->b, tset->a);                       \
    if (lep_ != cudaSuccess) return 3980 + (int)lep_;                            \
  } while (0)
          if (cfg == 50) { F4_PIPE_LAUNCH(64, true, 2, 2, true, g1s2); }
          else { F4_PIPE_LAUNCH(64, true, 3, 2, true, g1s3); }
        } else if (cfg == 54 || cfg == 55) {
          /* PIPE split-K GEMM1: rungs 42 (SK2) / 43 (SK4) with the
           * pipelined-commit mainloop; raw fp32 partials + k_act_packed. */
#define F4_PIPE_SK(BNV, G1V, STV, TAV, SKV, TAG)                                 \
  do {                                                                           \
    constexpr int NROW_ = (G1V) ? 2 * (BNV) : (BNV);                             \
    constexpr int NSFB_ = (G1V) ? 2 : 1;                                         \
    constexpr int SMEM_ = (STV) * (BM_ * F4_ROWB + NROW_ * F4_ROWB)              \
                        + 2 * (NSFB_ + 1) * F4_SFB                               \
                        + BM_ * 4 + 64 + (STV) * 8;                              \
    static bool attr_done_k_##TAG = false;                                       \
    if (!attr_done_k_##TAG) {                                                    \
      if (cudaFuncSetAttribute(                                                  \
              k_gemm_f4_tma<BNV, G1V, STV, TAV, 2, 2, SKV, false, true, false,   \
                            false, true>,                                        \
              cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
              SMEM_) != cudaSuccess) return 799;                                 \
      attr_done_k_##TAG = true;                                                  \
    }                                                                            \
    cudaError_t lek_ = launch_pdl(                                               \
        k_gemm_f4_tma<BNV, G1V, STV, TAV, 2, 2, SKV, false, true, false,         \
                      false, true>,                                              \
        capped_grid_tc(((G1V ? p.I : p.H) / (BNV)) * (SKV), p.max_tiles),        \
        dim3(128), (size_t)SMEM_, s, p, tset->b, tset->a);                       \
    if (lek_ != cudaSuccess) return 3990 + (int)lek_;                            \
  } while (0)
          if (cfg == 54) { F4_PIPE_SK(64, true, 2, 2, 2, g1sk2); }
          else { F4_PIPE_SK(64, true, 2, 2, 4, g1sk4); }
#undef F4_PIPE_SK
        } else if (cfg == 56) {
          /* WS2: warp-specialized GEMM1 -- producer/MMA/epilogue roles across
           * six warps, mbarrier-only mainloop choreography, 2 CTAs per SM.
           * Bit-identical staged images and MMA sequence to rung 41. */
          constexpr int SMEM_ = 3 * (BM_ * F4_ROWB + 2 * 64 * F4_ROWB)
                              + 2 * 3 * F4_SFB + 16 + 64 + 64;
          static bool attr_done_ws56 = false;
          if (!attr_done_ws56) {
            if (cudaFuncSetAttribute(k_gemm_f4_ws2<64, true, 3, 2, 2>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     SMEM_) != cudaSuccess) return 793;
            attr_done_ws56 = true;
          }
          cudaError_t le56_ = launch_pdl(k_gemm_f4_ws2<64, true, 3, 2, 2>,
                 capped_grid_tc(p.I / 64, p.max_tiles), dim3(224),
                 (size_t)SMEM_, s, p, tset->b, tset->a);
          if (le56_ != cudaSuccess) return 3930 + (int)le56_;
        } else if (cfg == 65) {
          /* DIAGNOSTIC: rung 56 with a 4-stage payload ring (SFS stays 2 --
           * GEMM1's SF TMEM cost is 48 cols/slot, so SFS=3 would force
           * TALLOC 512 and one CTA per SM). smem ~78KB, 2 CTAs/SM. */
          constexpr int SMEM_ = 4 * (BM_ * F4_ROWB + 2 * 64 * F4_ROWB)
                              + 2 * 3 * F4_SFB + 16 + 64 + 64;
          static bool attr_done_ws65 = false;
          if (!attr_done_ws65) {
            if (cudaFuncSetAttribute(k_gemm_f4_ws2<64, true, 4, 2, 2>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     SMEM_) != cudaSuccess) return 793;
            attr_done_ws65 = true;
          }
          cudaError_t le65_ = launch_pdl(k_gemm_f4_ws2<64, true, 4, 2, 2>,
                 capped_grid_tc(p.I / 64, p.max_tiles), dim3(224),
                 (size_t)SMEM_, s, p, tset->b, tset->a);
          if (le65_ != cudaSuccess) return 3930 + (int)le65_;
        } else if (cfg == 69 || cfg == 70) {
          /* WS4 DIAGNOSTIC: 352 threads / 1 CTA per SM / 8 epilogue warps
           * (two per TMEM quadrant, c-chunks split by parity; per-parity
           * STG.16 sf stores keep the act_sf word bit-identical). 69:
           * single acc -- isolates the drain split. 70: PP double acc
           * (TMEM 2*128 + 2*48 = 352 -> TALLOC 512) -- the MMA warp never
           * waits for the drain. Ring sizes as rung 56 (S=3/SFS=2): the
           * ring-depth sweep (dv_4768f4b8af40) rejected deeper rings. */
          constexpr int SMEM_ = 3 * (BM_ * F4_ROWB + 2 * 64 * F4_ROWB)
                              + 2 * 3 * F4_SFB + 16 + 64 + 64;
#define WS4V(PP_, TAG)                                                         \
          do {                                                                 \
            static bool attr_done_##TAG = false;                               \
            if (!attr_done_##TAG) {                                            \
              if (cudaFuncSetAttribute(                                        \
                      k_gemm_f4_ws2<64, true, 3, 2, 2, true, 1, PP_, 8>,       \
                      cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_)      \
                  != cudaSuccess) return 793;                                  \
              attr_done_##TAG = true;                                          \
            }                                                                  \
            cudaError_t le_##TAG = launch_pdl(                                 \
                k_gemm_f4_ws2<64, true, 3, 2, 2, true, 1, PP_, 8>,             \
                capped_grid_tc(p.I / 64, p.max_tiles), dim3(352),              \
                (size_t)SMEM_, s, p, tset->b, tset->a);                        \
            if (le_##TAG != cudaSuccess) return 3930 + (int)le_##TAG;          \
          } while (0)
          if (cfg == 69) { WS4V(false, a); } else { WS4V(true, b); }
#undef WS4V
        } else if (cfg == 58 || cfg == 59) {
          /* WS3 split-K GEMM1 (58: SK2 = pipelined rung 42; 59: SK4 = rung
           * 43): raw fp32 partials + k_act_packed<true>. */
          constexpr int SMEM_ = 3 * (BM_ * F4_ROWB + 2 * 64 * F4_ROWB)
                              + 2 * 3 * F4_SFB + 16 + 64 + 64;
          if (cfg == 58) {
            static bool attr_done_ws58 = false;
            if (!attr_done_ws58) {
              if (cudaFuncSetAttribute(k_gemm_f4_ws2<64, true, 3, 2, 2, true, 2>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       SMEM_) != cudaSuccess) return 790;
              attr_done_ws58 = true;
            }
            cudaError_t le58_ = launch_pdl(
                k_gemm_f4_ws2<64, true, 3, 2, 2, true, 2>,
                capped_grid_tc((p.I / 64) * 2, p.max_tiles), dim3(224),
                (size_t)SMEM_, s, p, tset->b, tset->a);
            if (le58_ != cudaSuccess) return 3900 + (int)le58_;
          } else {
            static bool attr_done_ws59 = false;
            if (!attr_done_ws59) {
              if (cudaFuncSetAttribute(k_gemm_f4_ws2<64, true, 3, 2, 2, true, 4>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       SMEM_) != cudaSuccess) return 791;
              attr_done_ws59 = true;
            }
            cudaError_t le59_ = launch_pdl(
                k_gemm_f4_ws2<64, true, 3, 2, 2, true, 4>,
                capped_grid_tc((p.I / 64) * 4, p.max_tiles), dim3(224),
                (size_t)SMEM_, s, p, tset->b, tset->a);
            if (le59_ != cudaSuccess) return 3910 + (int)le59_;
          }
        } else if (cfg == 63 || cfg == 64) {
          /* WS3 GEMM1 with TMA tile::gather4 A operand (AMODE 1): drops the
           * k_gather_t pre-pass and the [R,H*9/16] route-order buffer; the
           * A scales come straight from hid_sf via ptok. 63 = fused (rung 56
           * tiling), 64 = SK2 (rung 58 tiling). */
          constexpr int SMEM_ = 3 * (BM_ * F4_ROWB + 2 * 64 * F4_ROWB)
                              + 2 * 3 * F4_SFB + 16 + 64 + 64;
          if (cfg == 63) {
            static bool attr_done_ws63 = false;
            if (!attr_done_ws63) {
              if (cudaFuncSetAttribute(k_gemm_f4_ws2<64, true, 3, 2, 1>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       SMEM_) != cudaSuccess) return 786;
              attr_done_ws63 = true;
            }
            cudaError_t le63_ = launch_pdl(k_gemm_f4_ws2<64, true, 3, 2, 1>,
                   capped_grid_tc(p.I / 64, p.max_tiles), dim3(224),
                   (size_t)SMEM_, s, p, tset->b, tset->a);
            if (le63_ != cudaSuccess) return 3860 + (int)le63_;
          } else {
            static bool attr_done_ws64 = false;
            if (!attr_done_ws64) {
              if (cudaFuncSetAttribute(k_gemm_f4_ws2<64, true, 3, 2, 1, true, 2>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       SMEM_) != cudaSuccess) return 787;
              attr_done_ws64 = true;
            }
            cudaError_t le64_ = launch_pdl(
                k_gemm_f4_ws2<64, true, 3, 2, 1, true, 2>,
                capped_grid_tc((p.I / 64) * 2, p.max_tiles), dim3(224),
                (size_t)SMEM_, s, p, tset->b, tset->a);
            if (le64_ != cudaSuccess) return 3870 + (int)le64_;
          }
        } else if (wide) {
          if (ga) { F4_TMA_LAUNCH(128, true, 3, 1); }
          else { F4_TMA_LAUNCH(128, true, 3, 0); }
        } else {
          if (ga) { F4_TMA_LAUNCH(64, true, 2, 1); }
          else { F4_TMA_LAUNCH(64, true, 2, 0); }
        }
      } else {
        if (cfg == 38) {
          /* Rung 19 with SFS=2 + SFPRE2: batched one-round prime and a
           * two-iteration tail cover kill the long-scoreboard stalls (56% of
           * GEMM2 warp cycles, NCU job dv_98e07b3d1234). */
          F4_TMA_LAUNCH_S(128, false, 3, 0, 2, 2, 1, t3n128g0s2p2);
        } else if (cfg == 36) {
          /* Single-warp mainloop GEMM2 (rung 19 tiling). */
          constexpr int SMEM_ = 3 * (BM_ * F4_ROWB + 128 * F4_ROWB)
                              + 3 * 2 * F4_SFB + BM_ * 4 + 64 + 3 * 8;
          static bool attr_done_w1g2 = false;
          if (!attr_done_w1g2) {
            if (cudaFuncSetAttribute(k_gemm_f4_w1<128, false, 3, 3, 1>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     SMEM_) != cudaSuccess) return 771;
            attr_done_w1g2 = true;
          }
          cudaError_t lew_ = launch_pdl(k_gemm_f4_w1<128, false, 3, 3, 1>,
                       capped_grid_tc(p.H / 128, p.max_tiles),
                       dim3(128), (size_t)SMEM_, s, p, tset->b, tset->a);
          if (lew_ != cudaSuccess) return 3600 + (int)lew_;
        } else if (cfg == 44) {
          /* C15: rung 19 with SF staged by cp.async (SFCP). */
          F4_TMA_LAUNCH_C(128, false, 3, 0, 3, 0, 1, false, t3n128g0s3c);
        } else if (cfg == 46 || cfg == 47) {
          /* REDF: rung 44 (SFCP) or rung 19 (SFPRE) plus the fused routing
           * reduction -- the epilogue atomically accumulates score-scaled
           * BF16-rounded route outputs into acc32 instead of storing rout,
           * and run_pipeline replaces k_finalize with the k_convert final
           * rounding. No cp.async fallback for these rungs: run_pipeline
           * selects the finalize stage by cfg2, so a refused map must fail
           * loudly instead of dropping to a kernel that writes rout. */
          constexpr int SMEM_ = 3 * (BM_ * F4_ROWB + 128 * F4_ROWB)
                              + 3 * 2 * F4_SFB + BM_ * 4 + 64 + 3 * 8;
          if (cfg == 46) {
            static bool attr_done_r46 = false;
            if (!attr_done_r46) {
              if (cudaFuncSetAttribute(
                      k_gemm_f4_tma<128, false, 3, 0, 3, 0, 1, false, false,
                                    true, true>,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      SMEM_) != cudaSuccess) return 794;
              attr_done_r46 = true;
            }
            cudaError_t le46_ = launch_pdl(
                k_gemm_f4_tma<128, false, 3, 0, 3, 0, 1, false, false, true,
                              true>,
                capped_grid_tc(p.H / 128, p.max_tiles), dim3(128),
                (size_t)SMEM_, s, p, tset->b, tset->a);
            if (le46_ != cudaSuccess) return 3900 + (int)le46_;
          } else {
            static bool attr_done_r47 = false;
            if (!attr_done_r47) {
              if (cudaFuncSetAttribute(
                      k_gemm_f4_tma<128, false, 3, 0, 3, 1, 1, false, false,
                                    false, true>,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      SMEM_) != cudaSuccess) return 795;
              attr_done_r47 = true;
            }
            cudaError_t le47_ = launch_pdl(
                k_gemm_f4_tma<128, false, 3, 0, 3, 1, 1, false, false, false,
                              true>,
                capped_grid_tc(p.H / 128, p.max_tiles), dim3(128),
                (size_t)SMEM_, s, p, tset->b, tset->a);
            if (le47_ != cudaSuccess) return 3950 + (int)le47_;
          }
        } else if (cfg == 57) {
          /* WS3 GEMM2: the 7-warp role split at BN=128 (w0 A-scales, w1 TMA
           * + B-scales, w2 MMA, w3-6 epilogue -> rout). Bit-identical to
           * rung 44 by construction. */
          constexpr int SMEM_ = 3 * (BM_ * F4_ROWB + 128 * F4_ROWB)
                              + 2 * 2 * F4_SFB + 16 + 64 + 64;
          static bool attr_done_ws57 = false;
          if (!attr_done_ws57) {
            if (cudaFuncSetAttribute(k_gemm_f4_ws2<128, false, 3, 2, 0>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     SMEM_) != cudaSuccess) return 792;
            attr_done_ws57 = true;
          }
          cudaError_t le57_ = launch_pdl(k_gemm_f4_ws2<128, false, 3, 2, 0>,
                 capped_grid_tc(p.H / 128, p.max_tiles), dim3(224),
                 (size_t)SMEM_, s, p, tset->b, tset->a);
          if (le57_ != cudaSuccess) return 3920 + (int)le57_;
        } else if (cfg >= 66 && cfg <= 68) {
          /* DIAGNOSTIC: rung 57 with deeper rings. GEMM2's SF TMEM cost is
           * 32 cols/slot, so SFS=3 still fits TALLOC 256 (2 CTAs/SM); at
           * nk=3 (I=768) a 5-6 stage ring spans two whole items, letting
           * the producers free-run across the item boundary while the
           * single accumulator drains. 66: S=4/SFS=2 (~74KB), 67: S=5/
           * SFS=3 (~94KB), 68: S=6/SFS=3 (~111KB). */
#define WS57V(S_, SFS_, TAG)                                                   \
          do {                                                                 \
            constexpr int SMEM_ = (S_) * (BM_ * F4_ROWB + 128 * F4_ROWB)       \
                                + (SFS_) * 2 * F4_SFB + 16 + 64 + 64;          \
            static bool attr_done_##TAG = false;                               \
            if (!attr_done_##TAG) {                                            \
              if (cudaFuncSetAttribute(                                        \
                      k_gemm_f4_ws2<128, false, S_, SFS_, 0>,                  \
                      cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_)      \
                  != cudaSuccess) return 792;                                  \
              attr_done_##TAG = true;                                          \
            }                                                                  \
            cudaError_t le_##TAG = launch_pdl(                                 \
                k_gemm_f4_ws2<128, false, S_, SFS_, 0>,                        \
                capped_grid_tc(p.H / 128, p.max_tiles), dim3(224),             \
                (size_t)SMEM_, s, p, tset->b, tset->a);                        \
            if (le_##TAG != cudaSuccess) return 3920 + (int)le_##TAG;          \
          } while (0)
          if (cfg == 66) { WS57V(4, 2, a); }
          else if (cfg == 67) { WS57V(5, 3, b); }
          else { WS57V(6, 3, c); }
#undef WS57V
        } else if (cfg == 61 || cfg == 62) {
          /* WS3 GEMM2 + TMEM accumulator PING-PONG at BN=64: the MMA warp
           * fills buffer (j&1) while the epilogue drains buffer ((j-1)&1),
           * so the per-item epi_done bubble disappears (it moves two items
           * back). 2*64 ACC + 2*2*16 SF = 192 TMEM columns keeps two CTAs
           * per SM. 61: 3-stage ring; 62: 4-stage ring. */
          constexpr int SMEM61 = 3 * (BM_ * F4_ROWB + 64 * F4_ROWB)
                               + 2 * 2 * F4_SFB + 16 + 64 + 64;
          constexpr int SMEM62 = 4 * (BM_ * F4_ROWB + 64 * F4_ROWB)
                               + 2 * 2 * F4_SFB + 16 + 64 + 64;
          if (cfg == 61) {
            static bool attr_done_ws61 = false;
            if (!attr_done_ws61) {
              if (cudaFuncSetAttribute(
                      k_gemm_f4_ws2<64, false, 3, 2, 0, true, 1, true>,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      SMEM61) != cudaSuccess) return 788;
              attr_done_ws61 = true;
            }
            cudaError_t le61_ = launch_pdl(
                k_gemm_f4_ws2<64, false, 3, 2, 0, true, 1, true>,
                capped_grid_tc(p.H / 64, p.max_tiles), dim3(224),
                (size_t)SMEM61, s, p, tset->b, tset->a);
            if (le61_ != cudaSuccess) return 3880 + (int)le61_;
          } else {
            static bool attr_done_ws62 = false;
            if (!attr_done_ws62) {
              if (cudaFuncSetAttribute(
                      k_gemm_f4_ws2<64, false, 4, 2, 0, true, 1, true>,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      SMEM62) != cudaSuccess) return 789;
              attr_done_ws62 = true;
            }
            cudaError_t le62_ = launch_pdl(
                k_gemm_f4_ws2<64, false, 4, 2, 0, true, 1, true>,
                capped_grid_tc(p.H / 64, p.max_tiles), dim3(224),
                (size_t)SMEM62, s, p, tset->b, tset->a);
            if (le62_ != cudaSuccess) return 3890 + (int)le62_;
          }
        } else if (cfg == 52 || cfg == 53) {
          /* PIPE GEMM2: rung 38's tiling (BN=128, SFS=2, SFPRE2) with the
           * pipelined-commit mainloop; 53 runs the payload ring two deep. */
          if (cfg == 52) { F4_PIPE_LAUNCH(128, false, 3, 0, false, g2s3); }
          else { F4_PIPE_LAUNCH(128, false, 2, 0, false, g2s2); }
#undef F4_PIPE_LAUNCH
        } else if ((cfg >= 19 && cfg <= 35 && cfg != 22) || cfg == 37 ||
                   cfg == 39 || cfg == 40) {
          /* GEMM2 rung 12 plus SF register prefetch. */
          F4_TMA_LAUNCH_S(128, false, 3, 0, 3, true, 1, t3n128g0s3p);
        } else if (wide || r16 || cfg == 17 || cfg == 18) {
          F4_TMA_LAUNCH(128, false, 3, 0);
        } else {
          F4_TMA_LAUNCH(64, false, 2, 0);
        }
      }
#undef F4_TMA_LAUNCH
#undef F4_TMA_LAUNCH_S
#undef F4_TMA_LAUNCH_C
      return (int)cudaGetLastError();
    }
    /* The REDF rungs must never silently fall back to a rout-writing kernel:
     * run_pipeline picks k_convert (no rout read) purely from cfg2. */
    if (cfg == 46 || cfg == 47) return 732;
    /* WS3 has no cp.async fallback form. */
    if (cfg >= 56 && cfg <= 70) return 733;
    /* The split-K rungs have no cp.async fallback: their pipeline contract
     * includes the fp32 partials k_act_packed consumes. Refusal here is
     * unreachable for every legal geometry (the maps are rung 17's), but a
     * clean error beats silently skipping the partials. */
    if (cfg == 20 || cfg == 21 || cfg == 28 || cfg == 29 ||
        cfg == 31 || cfg == 32 || cfg == 34 || cfg == 35 || cfg == 37 ||
        cfg == 42 || cfg == 43 || cfg == 54 || cfg == 55) return 731;
    const int fc = (cfg >= 12) ? (wide ? 8 : 10) : cfg;
    if (fc == 11) {
      /* N=64 halves the accumulator (2*64 columns for up+gate) so the SF ring
       * still fits in 256 TMEM columns and two CTAs share an SM: twice the
       * resident CTAs and twice the item count, which is what the tiny-batch
       * shapes are short of (fixed_0 GEMM1 runs 48 CTAs and reaches 0.34 TB/s). */
      if constexpr (G1) { F4_LAUNCH(64, true, 3); } else { F4_LAUNCH(64, false, 3); }
    } else if (fc == 10) {
      if constexpr (G1) { F4_LAUNCH(64, true, 2); } else { F4_LAUNCH(64, false, 2); }
    } else if (fc == 9) {
      if constexpr (G1) { F4_LAUNCH(128, true, 4); } else { F4_LAUNCH(128, false, 4); }
    } else if (fc == 8) {
      if constexpr (G1) { F4_LAUNCH(128, true, 3); } else { F4_LAUNCH(128, false, 3); }
    } else {
      if constexpr (G1) { F4_LAUNCH(128, true, 2); } else { F4_LAUNCH(128, false, 2); }
    }
#undef F4_LAUNCH
#undef F4_LAUNCH_S
    return (int)cudaGetLastError();
  }
  if (cfg == 4 || cfg == 5 || cfg == 6) {
    /* tcgen05 rungs: operands are read from shared memory by the tensor core
     * and accumulators live in TMEM, so there is no ldmatrix traffic, and the
     * clamp/activation/QDQ (GEMM1) or BF16 materialization (GEMM2) is fused
     * into the epilogue. Rung 4 keeps two CTAs per SM (256 TMEM columns each,
     * N=128 for GEMM1); rung 5 uses all 512 columns so every MMA runs at
     * N=256, which measured 291 TFLOPS against 196 TFLOPS at N=128. */
    constexpr int BM = 128, BK = 64;
#define TC_LAUNCH_S(BNV, G1V, NSUBV, TAG)                                       \
    do {                                                                        \
      constexpr int BM_ = 128 * (NSUBV);                                        \
      constexpr int SMEM_ = 2 * (BM_ * BK * 2                                   \
                          + (G1V ? 2 : 1) * (BNV) * BK * 2)                     \
                          + BM_ * 4 + 64 + 2048;                                \
      static bool attr_done_##TAG = false;                                      \
      if (!attr_done_##TAG) {                                                   \
        if (cudaFuncSetAttribute(k_gemm_tc<BNV, G1V, NSUBV, BK>,                \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                                 SMEM_) != cudaSuccess) return 90 + (BNV);      \
        attr_done_##TAG = true;                                                 \
      }                                                                         \
      k_gemm_tc<BNV, G1V, NSUBV, BK>                                            \
          <<<capped_grid_tc((G1V ? p.I : p.H) / (BNV), p.max_tiles),            \
                            dim3(128), SMEM_, s>>>(p);                          \
    } while (0)
#define TC_LAUNCH(BNV, G1V) TC_LAUNCH_S(BNV, G1V, 1, n##BNV##g##G1V)
#define TC_LAUNCH_SUB(BNV, G1V, NSUBV) TC_LAUNCH_S(BNV, G1V, NSUBV, s##BNV##g##G1V##n##NSUBV)
    if constexpr (G1) {
      if (cfg == 6) { TC_LAUNCH_SUB(128, true, 2); }
      else if (cfg == 5) { TC_LAUNCH(256, true); }
      else { TC_LAUNCH(128, true); }
    } else {
      if (cfg == 6) { TC_LAUNCH_SUB(256, false, 2); }
      else { TC_LAUNCH(256, false); }
    }
#undef TC_LAUNCH
#undef TC_LAUNCH_S
#undef TC_LAUNCH_SUB
    return (int)cudaGetLastError();
  }
  switch (cfg) {
    case 0: LAUNCH(128, 128, 64, 4, 2); break;
    case 1: LAUNCH(64, 256, 64, 2, 4); break;
    case 2: LAUNCH(32, 256, 64, 1, 4); break;
    default: LAUNCH(16, 32, 256, 1, 2); break;
  }
#undef LAUNCH
  return (int)cudaGetLastError();
}

}  /* namespace tmal */
using namespace tmal;

static void fill_params(const MoEABI* abi, nvfp4moe::Params& p) {
  p.ids = (const int32_t*)abi->ids;
  p.scores = (const float*)abi->scores;
  p.hid_q = (const uint8_t*)abi->hid_q;
  p.hid_sf = (const uint8_t*)abi->hid_sf;
  p.w1_q = (const uint8_t*)abi->w1_q;
  p.w1_sf = (const uint8_t*)abi->w1_sf;
  p.w2_q = (const uint8_t*)abi->w2_q;
  p.w2_sf = (const uint8_t*)abi->w2_sf;
  p.alpha = (const float*)abi->alpha;
  p.beta = (const float*)abi->beta;
  p.limit = (const float*)abi->limit;
  p.out = (__nv_bfloat16*)abi->out;
  p.counts = (int32_t*)abi->counts;
  p.fill = (int32_t*)abi->fill;
  p.offs = (int32_t*)abi->offs;
  p.toffs = (int32_t*)abi->toffs;
  p.ptok = (int32_t*)abi->ptok;
  p.pslot = (int32_t*)abi->pslot;
  p.pexp = (int32_t*)abi->pexp;
  p.route_of = (int32_t*)abi->route_of;
  p.texp = (int32_t*)abi->texp;
  p.tm0 = (int32_t*)abi->tm0;
  p.hid_bf = (__nv_bfloat16*)abi->hid_bf;
  p.fc1 = (float*)abi->fc1;
  p.act = (__nv_bfloat16*)abi->act;
  /* Rung 7 aliases the same workspace as packed NVFP4: payload first, then the
   * E4M3 scale bytes (R*I/2 + R*I/16 <= R*I*2 bytes, so no extra allocation). */
  p.act_q = reinterpret_cast<uint8_t*>(abi->act);
  p.act_sf = p.act_q + (size_t)abi->Rmax * (size_t)(abi->I >> 1);
  p.rout = (__nv_bfloat16*)abi->rout;
  p.T = (int)abi->T; p.K = (int)abi->K; p.E = (int)abi->E;
  p.H = (int)abi->H; p.I = (int)abi->I;
  p.Rmax = (int)abi->Rmax; p.max_tiles = (int)abi->max_tiles;
  p.act_type = (int)abi->act_type; p.has_limit = (int)abi->has_limit;
  p.bm = (int)abi->bm;
  p.cfg2 = (int)abi->cfg2;
  p.part = (float*)abi->part;
  p.sk = (int)abi->sk;
  p.hidq_g = (uint8_t*)abi->hidq_g;
  p.hidsf_g = (uint8_t*)abi->hidsf_g;
  p.cnt = (int32_t*)abi->cnt;
  p.cnt_n = (int)abi->cnt_n;
  p.acc32 = (float*)abi->acc32;
}

/* Small-path route threshold. Re-tuned from 8192 to 6500 on the AGA bytes
 * (jobs T2 dv_6c27fd41182a + U dv_be343423c20f, same-process sweeps): the
 * five-node big path beats the single-block small path decisively at R=8192
 * (domain_8: 209.2 -> 194.0 us = +7.8%) and marginally at R=4096 (fixed_9
 * +1.7%, fixed_3 +0.6%), but the small path still wins for EMPTY routing at
 * R=6216 (domain_4: big path adds ~2.5 us of node floors for zero work,
 * -4.3%). 6500 sits between the last measured small-path win and the first
 * measured big-path win, trading the marginal R=4096 gains for removing the
 * empty-routing regression. The env knob (read per eager call, mixed into
 * the graph key) exists for A/B evidence; both paths compute identical
 * outputs, so any value is correctness-safe. */
static long small_threshold() {
  const char* e = getenv("NVFP4_MOE_SMALL_THRESH");
  return e ? atol(e) : 6500;
}

/* C18 (promoted from the C17 experiment): token-major pre-gather
 * (k_gather_t) instead of the route-major k_gather_a. Job AG same-run
 * sweep: fixed_6 -6.5%, domain_2 -6.7%, domain_3 -2.3%, fixed_3 -1.4%,
 * domain_7 -0.8%, fixed_2 -0.2% (6/6 wins). Read per eager call and mixed
 * into the graph key (the choice bakes into the captured launch). Both
 * forms produce identical hidq_g/hidsf_g images, so any value is
 * correctness-safe; NVFP4_MOE_GATHERT=0 restores route-major for A/B
 * evidence. */
static int gather_tokenwise() {
  const char* e = getenv("NVFP4_MOE_GATHERT");
  if (e == nullptr) return 1;                  /* promoted default: ON */
  return atoi(e) != 0 ? 1 : 0;
}

/* Stage indices used by moe_profile(); the forward path passes ev == nullptr. */
enum Stage { ST_MEMSET = 0, ST_COUNT, ST_SCAN, ST_SCATTER, ST_TILES, ST_DECODE,
             ST_GEMM1, ST_ACT, ST_GEMM2, ST_FINALIZE, STAGES };

static int run_pipeline(const nvfp4moe::Params& p, int cfg, cudaStream_t s,
                        cudaEvent_t* ev) {
  using namespace nvfp4moe;
#define EV(i) do { if (ev) cudaEventRecord(ev[i], s); } while (0)
  /* Launch failures return a site-tagged code (3x routing/act/finalize,
   * 1xx GEMM1, 2xx GEMM2) instead of being swallowed until the sticky
   * cudaGetLastError at the end -- an InvalidValue from one rejected launch
   * is otherwise indistinguishable from every other site (job dv_01231f85ea97
   * debugging). */
#define LP(expr, tag) do { cudaError_t le_ = (expr); \
    if (le_ != cudaSuccess) return (tag); } while (0)
  const int nroutes = p.T * p.K;
  const int bm = p.bm;
  const int nb_hid = p.H >> 4;
  /* One block covers the whole routing pipeline (histogram, scan, scatter, tile
   * map) whenever the route count is small enough that a single block is not
   * the slower choice; the activation decode stays parallel because it scales
   * with T*H and a single block measured ~20x slower there. The threshold is
   * an experiment knob (NVFP4_MOE_SMALL_THRESH, read per eager call, never
   * during graph replay); both paths compute identical outputs, so any value
   * is correctness-safe. Job-A/B4 stage data: the small path costs ~9 us at
   * R=8 and ~36 us at R=16K (shared-atomic issue bound), while the big path
   * costs ~26 us real at R=65K across five launches (~2.8 us event/launch
   * floor each), which puts the expected crossover near R ~ 8-16K. */
  const bool small = (nroutes <= small_threshold());
  /* REDF rungs: zero the [T,H] fp32 route accumulator on a SIDE-STREAM branch
   * that rejoins immediately before GEMM2, so the memset (up to ~120 MB on the
   * largest shapes) overlaps prep + gather + GEMM1 instead of serializing.
   * Under graph capture the fork/join becomes a parallel branch of the graph;
   * the events are DisableTiming and snapshotted by cudaStreamWaitEvent, so
   * re-recording them on the next call is safe. The branch always rejoins
   * before EndCapture. Creation happens on the first eager call, which always
   * precedes any capture of this launch set. */
  const bool redf = (p.cfg2 == 46 || p.cfg2 == 47);
  cudaEvent_t ze_join = nullptr;
  if (redf) {
    static cudaStream_t zs = nullptr;
    static cudaEvent_t ze0 = nullptr, ze1 = nullptr;
    if (zs == nullptr) {
      if (cudaStreamCreateWithFlags(&zs, cudaStreamNonBlocking) != cudaSuccess) {
        zs = nullptr;
        return 46;
      }
      if (cudaEventCreateWithFlags(&ze0, cudaEventDisableTiming) != cudaSuccess ||
          cudaEventCreateWithFlags(&ze1, cudaEventDisableTiming) != cudaSuccess)
        return 47;
    }
    LP(cudaEventRecord(ze0, s), 48);
    LP(cudaStreamWaitEvent(zs, ze0, 0), 49);
    LP(cudaMemsetAsync(p.acc32, 0,
                       (size_t)p.T * (size_t)p.H * sizeof(float), zs), 50);
    LP(cudaEventRecord(ze1, zs), 51);
    ze_join = ze1;
  }
  EV(ST_MEMSET);
  if (small) {
    /* k_prepare_small stages all nroutes expert ids in dynamic shared memory
     * (4 B each, <= 128 KiB at the 32768-route threshold); the attribute is
     * raised once per process, on the first eager call, which always precedes
     * any graph capture of this launch. */
    static bool prep_attr = false;
    if (!prep_attr) {
      if (cudaFuncSetAttribute(k_prepare_small,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               32768 * 4) != cudaSuccess)
        return 91;
      prep_attr = true;
    }
    LP(launch_pdl(k_prepare_small, dim3(1), dim3(1024), (size_t)nroutes * 4, s, p), 31);
    EV(ST_COUNT); EV(ST_SCAN); EV(ST_SCATTER); EV(ST_TILES);
  } else {
    /* counts, fill (and for the fused split-K rungs the arrival counters)
     * are ONE adjacent allocation (solution.py sizes them as a single
     * tensor), so one memset node zeroes all of them. */
    cudaError_t err = cudaMemsetAsync(
        p.counts, 0, (size_t)p.E * 8 + (size_t)(p.cnt_n > 0 ? p.cnt_n : 0) * 4, s);
    if (err) return (int)err;
    EV(ST_COUNT);
    LP(launch_pdl(k_count, dim3((nroutes + 255) / 256), dim3(256), 0, s, p), 32);
    EV(ST_SCAN);
    LP(launch_pdl(k_scan_tiles, dim3(1), dim3(1024), 0, s, p, bm), 33);
    EV(ST_SCATTER);
    LP(launch_pdl(k_scatter, dim3((nroutes + 255) / 256), dim3(256), 0, s, p), 34);
    EV(ST_TILES);
  }
  EV(ST_DECODE);
  /* Pre-gathered-A rungs: materialize GEMM1's activation operand in route
   * order once (see k_gather_a), before the GEMM boxes it per stage. */
  if ((cfg >= 27 && cfg <= 35) || cfg == 37 || cfg == 39
      || (cfg >= 41 && cfg <= 43) || cfg == 45 || cfg == 48 || cfg == 49
      || cfg == 50 || cfg == 51 || cfg == 54 || cfg == 55 || cfg == 56
      || cfg == 58 || cfg == 59 || cfg == 65 || cfg == 69 || cfg == 70) {
    if (gather_tokenwise()) {
      const unsigned gblocks = (unsigned)(p.T < 592 ? p.T : 592);
      LP(launch_pdl(k_gather_t, dim3(gblocks), dim3(128), 0, s, p), 41);
    } else {
      const int chunks = (p.Rmax + 3) / 4;
      const unsigned gblocks = (unsigned)(chunks < 592 ? chunks : 592);
      LP(launch_pdl(k_gather_a, dim3(gblocks), dim3(128), 0, s, p), 39);
    }
  }
  /* The packed-FP4 rungs (7 and its deeper-pipeline variants 8/9) read the
   * hidden states straight from the input tensors -- payload plus E4M3 scales --
   * as their A operand, so the decode stage does not exist in that pipeline: one
   * launch and one T*H-byte bf16 buffer less. The test is a family test, not a
   * single-rung test: gating it on `cfg != 7` left the decode running (and
   * measured 15.0 us on fixed_6, 6.6-7.0 us on the small shapes) for rungs 8/9,
   * which never read the buffer it writes. */
  if (cfg < 7) {
    k_decode_hidden<<<dim3((nb_hid + 127) / 128, p.T), 128, 0, s>>>(p);
    cudaError_t de = cudaGetLastError();
    if (de != cudaSuccess) return 37;
  }
  EV(ST_GEMM1);
  int gerr = launch_gemm<true>(p, cfg, s);
  if (gerr) return 100 + gerr;
  EV(ST_ACT);
  /* The tcgen05 rung fuses clamp + activation + NVFP4 QDQ into the GEMM1
   * epilogue and writes `act` itself; running k_act afterwards would reread the
   * (never written) fc1 buffer and clobber it. fc1 belongs to the warp-mma
   * rungs only. The split-K rungs store raw fp32 partials instead, which
   * k_act_packed reduces (fixed slice order) and quantizes. */
  if (cfg < 4) {
    const int nb_act = p.Rmax * (p.I >> 4);
    k_act<<<(nb_act + 127) / 128, 128, 0, s>>>(p);
    cudaError_t ae = cudaGetLastError();
    if (ae != cudaSuccess) return 38;
  } else if (p.sk > 1 && cfg != 31 && cfg != 32) {
    const int nb_act = p.Rmax * (p.I >> 4) * 4;   /* quarter-widened */
    if (cfg == 42 || cfg == 43 || cfg == 54 || cfg == 55 ||
        cfg == 58 || cfg == 59 || cfg == 64) {
      LP(launch_pdl(k_act_packed<true>, dim3((nb_act + 255) / 256), dim3(256),
                    0, s, p), 35);
    } else {
      LP(launch_pdl(k_act_packed<false>, dim3((nb_act + 255) / 256), dim3(256),
                    0, s, p), 35);
    }
  }
  EV(ST_GEMM2);
  if (redf) LP(cudaStreamWaitEvent(s, ze_join, 0), 52);
  gerr = launch_gemm<false>(p, p.cfg2, s);
  if (gerr) return 200 + gerr;
  EV(ST_FINALIZE);
  if (redf) {
    /* The route sum is already reduced; this is the final BF16 rounding. */
    const size_t n8 = ((size_t)p.T * (size_t)p.H) >> 3;
    const unsigned nbz = (unsigned)((n8 + 127) / 128);
    LP(launch_pdl(k_convert, dim3(nbz), dim3(128), 0, s, p), 36);
  } else {
    if ((size_t)p.Rmax * (size_t)p.H * 2 >= ((size_t)256 << 20)) {
      const int nfin = p.H >> 4;
      LP(launch_pdl(k_finalize16, dim3((nfin + 127) / 128, p.T), dim3(128),
                    0, s, p), 36);
    } else {
      const int nfin = p.H >> 3;
      LP(launch_pdl(k_finalize, dim3((nfin + 127) / 128, p.T), dim3(128),
                    0, s, p), 36);
    }
  }
  EV(STAGES);
#undef EV
#undef LP
  return (int)cudaGetLastError();
}

/* ------------------------------------------------------- graph replay cache */
/* The launch sequence is pointer-stable for a given call signature, so after
 * seeing the same ABI twice the sequence is captured once into a CUDA graph
 * and replayed. This removes the per-launch host cost (~6 us measured per
 * stage on this host, which dominated every small-token shape). The graph
 * replays the very same kernels on the very same buffers, so every forward
 * still computes the current input values; nothing is memoized. Any pointer,
 * dimension or configuration change produces a new key and a new capture.
 * Set NVFP4_MOE_NO_GRAPH=1 to force the eager path (used for A/B evidence).
 */
namespace {

struct GraphCache {
  cudaGraphExec_t exec;
  uint64_t key;
  uint64_t pending_key;
  int pending_hits;
  cudaStream_t capture_stream;
  bool disabled;
};

GraphCache g_graph = {nullptr, 0, 0, 0, nullptr, false};
bool g_graph_env_checked = false;
int g_graph_status[5] = {0, 0, 0, 0, 0};  /* begin, pipeline, end, instantiate, launch */

uint64_t abi_hash(const MoEABI* abi) {
  const int64_t* w = reinterpret_cast<const int64_t*>(abi);
  const int n = (int)(sizeof(MoEABI) / sizeof(int64_t));
  /* Hash every field that changes the launch set, and only those. The stream is
   * excluded because a captured graph is replayed onto whatever stream the caller
   * passes; `cfg2` must NOT be excluded -- it selects the GEMM2 kernel. This used
   * to skip the last field of the struct on the assumption that it was `stream`,
   * but `cfg2` was appended later, so the hash covered the stream and ignored the
   * GEMM2 rung. */
  const int skip = (int)(offsetof(MoEABI, stream) / sizeof(int64_t));
  uint64_t h = 1469598103934665603ULL;
  /* The launch grid is baked into a captured graph, so a change of the
   * persistent-grid cap must produce a new key even when the ABI is identical.
   * The small-path route threshold likewise selects which kernels are
   * captured. */
  h ^= (uint64_t)grid_cap_gen();
  h *= 1099511628211ULL;
  h ^= (uint64_t)small_threshold();
  h *= 1099511628211ULL;
  h ^= (uint64_t)pdl_on();
  h *= 1099511628211ULL;
  h ^= (uint64_t)gather_tokenwise();
  h *= 1099511628211ULL;
  for (int i = 0; i < n; ++i) {
    if (i == skip) continue;
    h ^= (uint64_t)w[i];
    h *= 1099511628211ULL;
  }
  return h;
}

bool graph_disabled() {
  if (!g_graph_env_checked) {
    const char* v = getenv("NVFP4_MOE_NO_GRAPH");
    g_graph.disabled = (v != nullptr && v[0] == '1');
    g_graph_env_checked = true;
  }
  return g_graph.disabled;
}

}  /* namespace */

extern "C" int moe_run(const MoEABI* abi) {
  nvfp4moe::Params p;
  fill_params(abi, p);
  cudaStream_t s = (cudaStream_t)abi->stream;
  const int cfg = (int)abi->cfg;

  if (graph_disabled() || g_graph.pending_hits < 0) return run_pipeline(p, cfg, s, nullptr);
  const uint64_t key = abi_hash(abi);
  if (g_graph.exec != nullptr && g_graph.key == key)
    return (int)cudaGraphLaunch(g_graph.exec, s);
  if (key == g_graph.pending_key) ++g_graph.pending_hits;
  else { g_graph.pending_key = key; g_graph.pending_hits = 1; }
  if (g_graph.pending_hits < 2) return run_pipeline(p, cfg, s, nullptr);

  /* Second call with an identical signature: capture, then replay. */
  if (g_graph.exec != nullptr) {
    cudaGraphExecDestroy(g_graph.exec);
    g_graph.exec = nullptr;
    g_graph.key = 0;
  }
  if (g_graph.capture_stream == nullptr &&
      cudaStreamCreateWithFlags(&g_graph.capture_stream, cudaStreamNonBlocking) != cudaSuccess) {
    g_graph.pending_hits = -1;
    return run_pipeline(p, cfg, s, nullptr);
  }
  cudaGraph_t captured = nullptr;
  g_graph_status[0] = (int)cudaStreamBeginCapture(g_graph.capture_stream,
                                                 cudaStreamCaptureModeThreadLocal);
  bool ok = (g_graph_status[0] == 0);
  int code = run_pipeline(p, cfg, g_graph.capture_stream, nullptr);
  g_graph_status[1] = code;
  g_graph_status[2] = (int)cudaStreamEndCapture(g_graph.capture_stream, &captured);
  if (g_graph_status[2] != 0) ok = false;
  g_graph_status[3] = (ok && code == 0 && captured != nullptr)
      ? (int)cudaGraphInstantiate(&g_graph.exec, captured, nullptr, nullptr, 0) : -1;
  if (g_graph_status[3] == 0 && ok && code == 0) {
    g_graph.key = key;
  } else {
    if (g_graph.exec != nullptr) { cudaGraphExecDestroy(g_graph.exec); g_graph.exec = nullptr; }
    g_graph.pending_hits = -1;      /* capture unsupported here: stay eager */
  }
  if (captured != nullptr) cudaGraphDestroy(captured);
  if (g_graph.exec != nullptr && g_graph.key == key)
    return (int)cudaGraphLaunch(g_graph.exec, s);
  return run_pipeline(p, cfg, s, nullptr);
}

/* --------------------------------------------- diagnostics / profiling ABI */
extern "C" void moe_set_graph(int64_t enable) {
  g_graph_env_checked = true;
  g_graph.disabled = (enable == 0);
  if (enable == 0) {
    if (g_graph.exec != nullptr) {
      cudaGraphExecDestroy(g_graph.exec);
      g_graph.exec = nullptr;
    }
    g_graph.key = 0;
    g_graph.pending_key = 0;
    g_graph.pending_hits = 0;
  } else if (g_graph.pending_hits < 0) {
    g_graph.pending_hits = 0;
  }
}

extern "C" int64_t moe_graph_active() {
  return (g_graph.exec != nullptr && !g_graph.disabled) ? 1 : 0;
}

/* [begin_capture, pipeline, end_capture, instantiate, pending_hits] */
extern "C" void moe_graph_status(int64_t* out5) {
  for (int i = 0; i < 4; ++i) out5[i] = g_graph_status[i];
  out5[4] = g_graph.pending_hits;
}

/* Persistent-grid cap (diagnostic/experiment knob; the default stays 0 = off
 * until a measurement promotes a value). Changing it bumps a generation counter
 * that is part of the graph key, so a captured sequence is never replayed with
 * a grid it was not captured with. */
extern "C" int64_t moe_set_grid_cap(int64_t cap) {
  const int c = (cap < -1) ? -1 : (int)cap;      /* -1 auto, 0 off, >0 explicit */
  if (c != grid_cap_raw()) {
    g_grid_cap = c;
    ++g_grid_cap_gen;
  }
  return (int64_t)grid_cap();
}

extern "C" int64_t moe_grid_cap() { return (int64_t)grid_cap(); }

/* Diagnostic for rung 12: bit 0 = the driver entry point resolved, bit 1 = the
 * GEMM1 weight map is encoded and cached, bit 2 = the GEMM2 weight map, bit 3 =
 * the GEMM2 intermediate map. A zero in bits 1-3 after a rung-12 launch means
 * launch_gemm silently fell back to rung 8 for that geometry, which is what makes
 * an A/B of the two rungs trustworthy instead of merely intended. */
extern "C" int64_t moe_tma_status() {
  int64_t v = (encode_entry() != nullptr) ? 1 : 0;
  if (g_tma[0].valid) v |= 2;
  if (g_tma[1].valid) v |= 4;
  if (g_tma[1].valid && g_tma[1].wa != nullptr) v |= 8;
  if (g_tma[0].refused) v |= 16;    /* GEMM1 geometry refused, negatively cached */
  if (g_tma[1].refused) v |= 32;    /* GEMM2 geometry refused, negatively cached */
  if (g_tma[0].valid && g_tma[0].gather_a) v |= 64;  /* GEMM1 gather4 A map */
  return v;
}

/* Per-stage device timing (diagnostic only): the same launches as moe_run with
 * cudaEvents between stages. out_ns has STAGES entries, each the minimum over
 * `reps` passes, in nanoseconds. */
extern "C" int moe_profile(const MoEABI* abi, int64_t* out_ns, int reps) {
  nvfp4moe::Params p;
  fill_params(abi, p);
  cudaStream_t s = (cudaStream_t)abi->stream;
  cudaEvent_t ev[STAGES + 1];
  for (int i = 0; i <= STAGES; ++i)
    if (cudaEventCreate(&ev[i]) != cudaSuccess) return -1;
  double best[STAGES];
  for (int i = 0; i < STAGES; ++i) best[i] = 1e30;
  int code = 0;
  for (int r = 0; r < reps; ++r) {
    code = run_pipeline(p, (int)abi->cfg, s, ev);
    if (code) break;
    if (cudaStreamSynchronize(s) != cudaSuccess) { code = -2; break; }
    for (int i = 0; i < STAGES; ++i) {
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, ev[i], ev[i + 1]);
      if (ms < best[i]) best[i] = ms;
    }
  }
  for (int i = 0; i < STAGES; ++i)
    out_ns[i] = (best[i] > 1e29) ? -1 : (int64_t)(best[i] * 1e6);
  for (int i = 0; i <= STAGES; ++i) cudaEventDestroy(ev[i]);
  return code;
}

/* ------------------------------------------------- numerics self-test hooks */
/* Development entry points used by experiments/dev_check.py to compare the
 * format conversions and the activation QDQ against the frozen reference on
 * the device. They are not part of the forward path. */
namespace nvfp4moe {

__global__ void k_dump_tables(unsigned int* e2m1_pair, float* e4m3_f32,
                              unsigned int* e4m3_bf2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 256) {
    e2m1_pair[i] = g_e2m1_pair[i];
    e4m3_f32[i] = g_e4m3_f32[i];
    e4m3_bf2[i] = g_e4m3_bf2[i];
  }
}

/* out[i] = e4m3 RNE value of clamp(in[i], 0, 448), i.e. the reference's
 * sf.clamp(min,448).to(float8_e4m3fn).float() */
__global__ void k_e4m3_roundtrip(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = fminf(fmaxf(in[i], 0.0f), 448.0f);
  out[i] = g_e4m3_f32[f32_to_e4m3(v)];
}

/* reference nvfp4() over 16-element blocks: in/out are [nblocks, 16] fp32 */
__global__ void k_qdq_ref(const float* in, float* out, int nblocks) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  const float* src = in + (size_t)b * 16;
  float* dst = out + (size_t)b * 16;
  float amax = 0.0f;
  for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(src[j]));
  float sf_pre = fminf(amax * (1.0f / 6.0f), 448.0f);
  float sf = g_e4m3_f32[f32_to_e4m3(sf_pre)];
  const float inv_sf = (sf > 0.0f) ? __frcp_rn(sf) : 0.0f;
  for (int j = 0; j < 16; ++j)
    dst[j] = (sf > 0.0f) ? qdq_e2m1(src[j], sf, inv_sf) : 0.0f;
}

/* decode path: payload byte + E4M3 scale byte -> two exact bf16 values,
 * returned as float pairs, mirroring k_decode_hidden's arithmetic */
__global__ void k_decode_pair(const unsigned char* payload, const unsigned char* sf,
                              float* out, int nbytes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nbytes) return;
  unsigned int pair = g_e2m1_pair[payload[i]];
  unsigned int sb = g_e4m3_bf2[sf[i]];
  __nv_bfloat162 w = *reinterpret_cast<__nv_bfloat162*>(&sb);
  unsigned int res = as_u32(__hmul2(*reinterpret_cast<__nv_bfloat162*>(&pair), w));
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&res);
  out[2 * i + 0] = __bfloat162float(v.x);
  out[2 * i + 1] = __bfloat162float(v.y);
}

/* cvt.rn.satfinite.e2m1x2.f32 probe: compares BOTH nibble orders of the
 * hardware pair cast against the scalar seven-compare chain over the same
 * (already-divided) quotients. xy holds n (lo,hi) pairs; bad[0] counts
 * mismatches of order A = cvt(d, hi, lo) -- the one q_e2m1_pack8 ships --
 * and bad[1] counts order B = cvt(d, lo, hi). The first 8 mismatching
 * samples of each order land in samp (3 u64 words each). Driven by
 * experiments/probe_e2m1.py over tie neighborhoods, saturation extremes
 * and pseudo-random vectors before the epilogues rely on the cast. */
__global__ void k_probe_e2m1x2(const float* xy, int* bad,
                               unsigned long long* samp, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float lo = xy[2 * i], hi = xy[2 * i + 1];
  const unsigned int ref = (unsigned int)q_e2m1_code(lo, 1.0f, 1.0f)
                         | ((unsigned int)q_e2m1_code(hi, 1.0f, 1.0f) << 4);
  /* Both nibble orders, via the shipped pack8 instruction sequence. */
  const unsigned int pa = q_e2m1_pack8(lo, hi, lo, hi, lo, hi, lo, hi) & 0xFFu;
  const unsigned int pb = q_e2m1_pack8(hi, lo, hi, lo, hi, lo, hi, lo) & 0xFFu;
  if (pa != ref) {
    const int k = atomicAdd(&bad[0], 1);
    if (k < 8) {
      samp[3 * k + 0] = ((unsigned long long)__float_as_uint(lo) << 32)
                      | (unsigned long long)__float_as_uint(hi);
      samp[3 * k + 1] = ((unsigned long long)ref << 32) | (unsigned long long)pa;
      samp[3 * k + 2] = (unsigned long long)pb;
    }
  }
  if (pb != ref) {
    const int k = atomicAdd(&bad[1], 1);
    if (k < 8) {
      samp[24 + 3 * k + 0] = ((unsigned long long)__float_as_uint(lo) << 32)
                           | (unsigned long long)__float_as_uint(hi);
      samp[24 + 3 * k + 1] = ((unsigned long long)ref << 32) | (unsigned long long)pb;
      samp[24 + 3 * k + 2] = (unsigned long long)pa;
    }
  }
}

/* tcgen05.cp shape probe: fills SMEM with a known byte pattern, issues ONE
 * cp of the selected shape/variant into TMEM columns 0..15, reads the 128
 * lanes x 16 columns back with tcgen05.ld and dumps them so python can
 * decode the exact (smem byte -> lane, column) mapping. Variants:
 *   0: 32x128b.warpx4 (control -- must reproduce the shipped SF staging)
 *   1: 16x256b.warpx4      2: 16x256b.warpx2
 *   3: 16x128b.warpx4      4: 32x128b.warpx2
 *   5: 16x64b.warpx4       6: 32x128b (warpx0, warp 0 only)
 * Completion: cp is an async TC-pipe op with no mbarrier hook; the readback
 * is separated by a conservative clock spin plus the tcgen05 fences, which
 * is sufficient for a diagnostic (results cross-checked against the control
 * variant whose mapping is already validated by the shipping kernels). */
__global__ void k_probe_tmemcp(unsigned* out, int variant) {
  __shared__ __align__(1024) unsigned char buf[4096];
  __shared__ unsigned s_tmem[4];
  const int tid = threadIdx.x;
  for (int i = tid; i < 4096; i += 128)
    buf[i] = (unsigned char)((i * 37 + 11) & 0xFF);
  if (tid < 32) tmem_alloc(s_tmem, 64);
  __syncthreads();
  const unsigned tmem = *s_tmem;
  if (tid < 32) {
    if (elect_one()) {
      tc_fence_after_sync();
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      const unsigned long long d = make_sf_desc(smem_u32(buf), 512);
      /* ptxas 13.1/sm_103a rejects every alternative shape: .16x64b/.16x128b/
       * .16x256b are "Illegal matrix shape" for tcgen05.cp, .warpx2 is an
       * "Unknown modifier", and .32x128b REQUIRES a warptype -- so
       * 32x128b.warpx4 (the one the shipping SF staging already uses) is the
       * only legal form and the SF-copy instruction count is hardware-minimal
       * (job T build log, nvcc-9f94949b6b49c6b2). */
      if (variant == 0) {
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                     :: "r"(tmem), "l"(d));
      }
    }
  }
  /* Conservative completion wait, then fenced readback. */
  const long long t0 = clock64();
  while (clock64() - t0 < 40000) { }
  tc_fence_before_sync();
  __syncthreads();
  tc_fence_after_sync();
  const unsigned taddr = tmem + ((unsigned)(threadIdx.x & 96) << 16);
  /* warp w reads lanes 32w..32w+31, columns 0..15 in four x32 loads */
  const int warp = tid >> 5;
  for (int c0 = 0; c0 < 16; c0 += 32) { /* single pass; 16 cols < 32 */ }
  float v[32];
  tmem_ld_x32(taddr + ((unsigned)(threadIdx.x & 31) << 16) * 0 + 0, v);
  (void)warp;
  tmem_ld_wait();
  for (int j = 0; j < 16; ++j)
    out[tid * 16 + j] = __float_as_uint(v[j]);
  __syncthreads();
  if (tid < 32) tmem_dealloc(tmem, 64);
}

/* C10 probe: bit-exactness of the packed epilogue primitives. Each thread
 * takes ONE (alpha, beta) pair plus two (gate, up) tuples and evaluates them
 * as the two lanes (.x, .y) of a single activation2 call, so a swapped lane
 * mapping in the f32x2 asm shows up as a mismatch (the driver makes the two
 * tuples distinct). alpha/beta are shared by both lanes because activation2
 * broadcasts them -- exactly the production form. Classes: bad[0] swiglu
 * (act_type 0), bad[1] situ (act_type 10) -- both tolerance-checked against
 * activation_fast since C11a folds the ex2 prescale (max rel err per class
 * lands in samp[224+class]); bad[2]/bad[3]/bad[4] f2_mul / f2_add / f2_fma
 * and bad[5]/bad[6] activation2_nf (swiglu/situ, k_act_packed's non-folded
 * form) vs scalar mul/add/fma and activation_fast (bitwise, so NaN
 * canonicalization and +-0.0 signs are checked too). First 8 samples per
 * class land in samp
 * (4 u64 each: {g0,u0} bits, {g1,u1} bits, {scalar0,scalar1} bits,
 * {packed.x,packed.y} bits). Input layout per thread: 8 floats
 * [alpha, beta, g0, u0, g1, u1, c0, c1]; c is the fma addend. */
__global__ void k_probe_act2(const float* in8, int* bad,
                             unsigned long long* samp, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* t = in8 + (size_t)8 * i;
  const float al = t[0], be = t[1];
  const float g0 = t[2], u0 = t[3], g1 = t[4], u1 = t[5];
  const float c0 = t[6], c1 = t[7];
  const float ia = __frcp_rn(al), ib = __frcp_rn(be);
  const float2 g2 = make_float2(g0, g1), u2 = make_float2(u0, u1);
  const float2 c2 = make_float2(c0, c1);
  Params p;
#pragma unroll
  for (int at = 0; at < 2; ++at) {
    const int act = at ? 10 : 0;
    p.act_type = act;
    const ActK ak = act_k_constants(act, al, ia, ib);
    const float s0 = activation_fast(p, g0, u0, al, be, ia, ib);
    const float s1 = activation_fast(p, g1, u1, al, be, ia, ib);
    const float2 pk = activation2(act, g2, u2, al, be,
                                  ak.k_sw, ak.k_gu, ak.k_uu);
    /* C11a folds the ex2 argument prescale per item, so activation2 is no
     * longer bit-identical to activation_fast: the bound is |diff| <=
     * 1e-5 * (1 + |scalar|) per lane (the fold perturbs the exponent
     * argument by <=1.5 ulp -> sigmoid relative error <= ~1e-7; 1e-5 leaves
     * a 100x margin). The max observed relative error lands in samp[160+at]
     * (float bits, non-negative -> uint ordering is monotonic). */
    bool ok = true;
    float err = 0.0f;
#pragma unroll
    for (int lane = 0; lane < 2; ++lane) {
      const float sk = lane ? s1 : s0;
      const float kv = lane ? pk.y : pk.x;
      if (__float_as_uint(kv) != __float_as_uint(sk)) {
        const float d = fabsf(kv - sk);
        const float bound = 1e-5f + 1e-5f * fabsf(sk);
        if (!(isfinite(kv) && isfinite(sk) && d <= bound)) ok = false;
        if (isfinite(sk)) {
          const float rel = d / fmaxf(1.0f, fabsf(sk));
          err = fmaxf(err, rel);
        }
      }
    }
    const unsigned int errbits = (unsigned int)__float_as_int(err);
    atomicMax(reinterpret_cast<unsigned int*>(&samp[224 + at]), errbits);
    if (!ok) {
      const int k = atomicAdd(&bad[at], 1);
      if (k < 8) {
        unsigned long long* sp = samp + (size_t)32 * at + 4 * k;
        sp[0] = ((unsigned long long)__float_as_uint(g0) << 32)
              | (unsigned long long)__float_as_uint(u0);
        sp[1] = ((unsigned long long)__float_as_uint(g1) << 32)
              | (unsigned long long)__float_as_uint(u1);
        sp[2] = ((unsigned long long)__float_as_uint(s0) << 32)
              | (unsigned long long)__float_as_uint(s1);
        sp[3] = ((unsigned long long)__float_as_uint(pk.x) << 32)
              | (unsigned long long)__float_as_uint(pk.y);
      }
    }
    /* Classes 5/6: the non-folded packed form (k_act_packed's) must stay
     * BIT-identical to activation_fast. */
    const float2 nf = activation2_nf(act, g2, u2, al, be, ia, ib);
    if (__float_as_uint(nf.x) != __float_as_uint(s0) ||
        __float_as_uint(nf.y) != __float_as_uint(s1)) {
      const int k = atomicAdd(&bad[5 + at], 1);
      if (k < 8) {
        unsigned long long* sp = samp + (size_t)32 * (5 + at) + 4 * k;
        sp[0] = ((unsigned long long)__float_as_uint(g0) << 32)
              | (unsigned long long)__float_as_uint(u0);
        sp[1] = ((unsigned long long)__float_as_uint(g1) << 32)
              | (unsigned long long)__float_as_uint(u1);
        sp[2] = ((unsigned long long)__float_as_uint(s0) << 32)
              | (unsigned long long)__float_as_uint(s1);
        sp[3] = ((unsigned long long)__float_as_uint(nf.x) << 32)
              | (unsigned long long)__float_as_uint(nf.y);
      }
    }
  }
  const unsigned int chk[3] = {
      (__float_as_uint(f2_mul(g2, u2).x) == __float_as_uint(g0 * u0) &&
       __float_as_uint(f2_mul(g2, u2).y) == __float_as_uint(g1 * u1)) ? 0u : 1u,
      (__float_as_uint(f2_add(g2, u2).x) == __float_as_uint(g0 + u0) &&
       __float_as_uint(f2_add(g2, u2).y) == __float_as_uint(g1 + u1)) ? 0u : 1u,
      (__float_as_uint(f2_fma(g2, u2, c2).x) == __float_as_uint(fmaf(g0, u0, c0)) &&
       __float_as_uint(f2_fma(g2, u2, c2).y) == __float_as_uint(fmaf(g1, u1, c1)))
          ? 0u : 1u};
#pragma unroll
  for (int cls = 0; cls < 3; ++cls) {
    if (chk[cls]) {
      const int k = atomicAdd(&bad[2 + cls], 1);
      if (k < 8) {
        unsigned long long* sp = samp + (size_t)32 * (2 + cls) + 4 * k;
        sp[0] = ((unsigned long long)__float_as_uint(g0) << 32)
              | (unsigned long long)__float_as_uint(u0);
        sp[1] = ((unsigned long long)__float_as_uint(g1) << 32)
              | (unsigned long long)__float_as_uint(u1);
        sp[2] = (unsigned long long)__float_as_uint(c0);
        sp[3] = 0ull;
      }
    }
  }
}

}  /* namespace nvfp4moe */

extern "C" int moe_selftest(int64_t which, int64_t a, int64_t b, int64_t c, int64_t n,
                            int64_t stream) {
  using namespace nvfp4moe;
  cudaStream_t s = (cudaStream_t)stream;
  switch (which) {
    case 0:
      k_dump_tables<<<1, 256, 0, s>>>((unsigned int*)a, (float*)b, (unsigned int*)c);
      break;
    case 1:
      k_e4m3_roundtrip<<<((int)n + 255) / 256, 256, 0, s>>>((const float*)a, (float*)b, (int)n);
      break;
    case 2:
      k_qdq_ref<<<((int)n + 255) / 256, 256, 0, s>>>((const float*)a, (float*)b, (int)n);
      break;
    case 3:
      k_decode_pair<<<((int)n + 255) / 256, 256, 0, s>>>((const unsigned char*)a,
                                                         (const unsigned char*)b,
                                                         (float*)c, (int)n);
      break;
    case 4:
      k_probe_e2m1x2<<<((int)n + 255) / 256, 256, 0, s>>>((const float*)a,
                                                          (int*)b,
                                                          (unsigned long long*)c,
                                                          (int)n);
      break;
    case 7:
      nvfp4moe::k_probe_tmemcp<<<1, 128, 0, s>>>((unsigned*)a, (int)b);
      break;
    case 8:
      k_probe_act2<<<((int)n + 255) / 256, 256, 0, s>>>((const float*)a,
                                                        (int*)b,
                                                        (unsigned long long*)c,
                                                        (int)n);
      break;
    default:
      return -1;
  }
  return (int)cudaGetLastError();
}

/* Development probe for the rung-14/15 activation gather: stages ONE 128-row
 * x 128-byte A-operand tile twice -- once with the exact cp.async/swizzle
 * pattern of F4_LOAD_A, once with 32 TMA tile::gather4 boxes -- compares the
 * images byte-for-byte on device and exports both. A nonzero mismatch count
 * means the gather4 shared-memory image (row order or SW128 phase) differs
 * from what the MMA descriptors expect, and the TMAA rungs must not be used. */
namespace nvfp4moe {

__global__ void k_probe_gather4(const __grid_constant__ CUtensorMap tm,
                                const uint8_t* Aq, const int32_t* rows,
                                int64_t aq_stride, int ks,
                                unsigned char* img_cp, unsigned char* img_tma,
                                int* mismatch) {
  __shared__ __align__(1024) unsigned char buf[2][128 * 128];
  __shared__ __align__(8) unsigned long long bar;
  const int tid = threadIdx.x;   /* 128 */
  if (tid == 0) {
    mbar_init(smem_u32(&bar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  for (int task = tid; task < 128 * 8; task += 128) {
    const int r = task >> 3, c = task & 7;
    cp_async16(buf[0] + r * 128 + ((c ^ (r & 7)) * 16),
               Aq + (size_t)rows[r] * (size_t)aq_stride + (size_t)ks * 128 + c * 16);
  }
  cp_async_commit();
  if (tid == 0) tma_expect_tx(smem_u32(&bar), 128u * 128u);
  __syncthreads();
  if (tid < 32) {
    tma_gather4(&tm, smem_u32(buf[1]) + 512u * (unsigned)tid, smem_u32(&bar),
                ks * 128, rows[tid * 4], rows[tid * 4 + 1], rows[tid * 4 + 2],
                rows[tid * 4 + 3]);
  }
  cp_async_wait<0>();
  __syncthreads();
  if (tid == 0) mbar_wait(smem_u32(&bar), 0);
  __syncthreads();
  for (int i = tid * 16; i < 128 * 128; i += 128 * 16) {
    const uint4 a = *reinterpret_cast<const uint4*>(&buf[0][i]);
    const uint4 b = *reinterpret_cast<const uint4*>(&buf[1][i]);
    if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) atomicAdd(mismatch, 1);
    *reinterpret_cast<uint4*>(img_cp + i) = a;
    *reinterpret_cast<uint4*>(img_tma + i) = b;
  }
}

}  /* namespace nvfp4moe */

/* Development probes for the batch-2 (atomic finalize) and GEMM1 L2-ceiling
 * decisions. Both run on an L2-resident working set, mimicking the stg buffer
 * (64 MB on the largest shape) and the GEMM operand re-read stream. */
namespace nvfp4moe {

__global__ void k_probe_red(float* dst, int n_vec4, int reps) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthr = gridDim.x * blockDim.x;
  for (int r = 0; r < reps; ++r) {
    for (int i = tid; i < n_vec4; i += nthr) {
      float* a = dst + 4 * (size_t)i;
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n"
                   :: "l"(a), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f)
                   : "memory");
    }
  }
}

__global__ void k_probe_read(const float4* __restrict__ src, float4* sink,
                             int n_vec4, int passes) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthr = gridDim.x * blockDim.x;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int r = 0; r < passes; ++r) {
    for (int i = tid; i < n_vec4; i += nthr) {
      const float4 v = src[i];
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
  }
  if (acc.x == 1234.5678f && acc.y == -1.f) *sink = acc;  /* keep loads live */
}

}  /* namespace nvfp4moe */

extern "C" int moe_probe_bw(int64_t which, int64_t buf, int64_t sink,
                            int64_t n_vec4, int64_t reps, int64_t blocks,
                            int64_t stream) {
  using namespace nvfp4moe;
  cudaStream_t s = (cudaStream_t)stream;
  const int nb = (int)(blocks > 0 ? blocks : 296);
  switch (which) {
    case 0:  /* red.global.add.v4.f32 throughput */
      k_probe_red<<<nb, 128, 0, s>>>((float*)buf, (int)n_vec4, (int)reps);
      break;
    case 1:  /* L2-resident vectorized read bandwidth */
      k_probe_read<<<nb, 128, 0, s>>>((const float4*)buf, (float4*)sink,
                                      (int)n_vec4, (int)reps);
      break;
    default:
      return -1;
  }
  return (int)cudaGetLastError();
}

extern "C" int moe_probe_gather4(int64_t aq, int64_t rowbytes, int64_t rows_total,
                                 int64_t rows, int64_t ks, int64_t img_cp,
                                 int64_t img_tma, int64_t mismatch, int64_t stream) {
  CUtensorMap tm;
  if (!encode_gather(&tm, (void*)aq, (unsigned)rowbytes, (unsigned)rows_total))
    return -1;
  nvfp4moe::k_probe_gather4<<<1, 128, 0, (cudaStream_t)stream>>>(
      tm, (const uint8_t*)aq, (const int32_t*)rows, rowbytes, (int)ks,
      (unsigned char*)img_cp, (unsigned char*)img_tma, (int*)mismatch);
  return (int)cudaGetLastError();
}

/* Rung-26 debug readback: copy the first n words of g_ws26_dbg into the
 * caller-provided device buffer (see ws_mbar_wait and experiments/
 * ws26_debug.py). */
extern "C" int moe_ws26_readdbg(int64_t out, int64_t n) {
  size_t bytes = (size_t)n * sizeof(unsigned int);
  if (bytes > sizeof(nvfp4moe::g_ws26_dbg)) bytes = sizeof(nvfp4moe::g_ws26_dbg);
  return (int)cudaMemcpy((void*)out, (const void*)nvfp4moe::g_ws26_dbg, bytes,
                         cudaMemcpyDeviceToDevice);
}

/* Register the DEVICE alias of a cudaHostAlloc(Mapped) buffer so ws_dbg_record
 * mirrors timeout records into host RAM (survives context teardown). */
extern "C" int moe_ws26_sethostdbg(int64_t dev_ptr) {
  unsigned int* p = reinterpret_cast<unsigned int*>(dev_ptr);
  return (int)cudaMemcpyToSymbol(nvfp4moe::g_ws26_hostdbg, &p, sizeof(p));
}

/* Dual-library launch-failure diagnostic (job dv_3168e5ebc91a: the SECOND
 * loaded module's >48KB-dynamic-smem GEMM launch fails with InvalidValue while
 * every single-module topology passes). mode 0: read back the opt-in attribute
 * of the rung-17 GEMM1 instantiation; mode 1: set it to 78416 B first, then
 * read back. Returns maxDynamicSharedSizeBytes, or -(errno) / -(1000+errno) /
 * -(2000+errno) for get/set/readback failures. If a "successful" set reads
 * back below the requested value, the opt-in never landed on this module's
 * kernel, which is exactly what makes the later launch reject its smem. */
extern "C" int64_t moe_dbg_f4attr(int64_t mode) {
  cudaFuncAttributes a = {};
  cudaError_t e0 = cudaFuncGetAttributes(
      &a, (const void*)nvfp4moe::k_gemm_f4_tma<64, true, 2, false, 2, true, 1>);
  if (e0 != cudaSuccess) return -(int64_t)e0;
  if (mode == 0) return (int64_t)a.maxDynamicSharedSizeBytes;
  cudaError_t e1 = cudaFuncSetAttribute(
      nvfp4moe::k_gemm_f4_tma<64, true, 2, false, 2, true, 1>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 78416);
  if (e1 != cudaSuccess) return -(1000 + (int64_t)e1);
  cudaFuncAttributes b = {};
  cudaError_t e2 = cudaFuncGetAttributes(
      &b, (const void*)nvfp4moe::k_gemm_f4_tma<64, true, 2, false, 2, true, 1>);
  if (e2 != cudaSuccess) return -(2000 + (int64_t)e2);
  return (int64_t)b.maxDynamicSharedSizeBytes;
}
