// fmha_fp8_per_token_head — independent CUDA implementation for L20D (sm_103).
//
// Numerical contract (reference-materialization-v1-20261013):
//   score = (Q . K_fp8) * (softmax_scale * k_scale[key, kv_head])   (FP32)
//   p     = exp(score - running_max)                                 (FP32)
//   denominator uses unrounded p; numerator uses round(p * v_scale)
//   O     = accum / denom -> bf16, bottom-right causal, paged KV.
// Implementation: FlashAttention-2 style, mma.sync.m16n8k16, FP32
// accumulation and softmax in the log2 domain. QK uses f16 operands
// (bf16 Q -> f16 and fp8 K -> f16 are both exact in-domain). PV uses bf16
// operands: P = bf16(p * v_scale) reproduces the reference rounding exactly
// and V = bf16(V_payload) is exact, so PV products are bit-identical to the
// reference; only f32 accumulation order differs. K/V rows are gathered from
// the paged cache (HND or NHD via runtime strides) into smem tiles.
// Softmax uses a -1e30f sentinel instead of -inf so fully-masked split-K
// chunks can never produce NaN (stale weight mass is annihilated by the
// rescale factor once any real key is seen).
// Attribution: mechanisms follow the public FlashAttention-2 algorithm and
// vLLM unified_attention semantics; all code here is an independent
// implementation for this contract.
#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <vector>
#include <utility>

#define DEVI __device__ __forceinline__

namespace fmha {

constexpr int BM = 64;   // query rows per CTA tile
constexpr int BN = 64;   // keys per KV tile
constexpr int THREADS = 128; // 4 warps, one 16-row m-tile each
constexpr float NEG = -1e30f;

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
    float* scratch;           // decode split partials (null iff nchunks==1)
    long long q_out_stride;   // H*D
    long long k_sb, k_sp, k_sh;   // fp8 element strides (block, slot, kvh)
    long long v_sb, v_sp, v_sh;
    long long ks_sb, ks_sp, ks_sh; // f32 element strides
    long long vs_sb, vs_sp, vs_sh;
    int bt_stride;
    int batch, heads, g;
    int D, page_shift;
    float sm_scale_log2e;
    int raw16;                // v4.3: K/V stream bases+strides all 16B-aligned
    int fold;                 // v4.5: mirror-pair q tiles (diagonal causal)
    int nchunks;              // decode chunks per (b,kvh); prefill ignores
    long long chunk_stride;   // f32 elements per (b,kvh,chunk) partial
    int Rmax;                 // decode rows upper bound per request
    int chunk_size;           // decode keys per chunk
    float* dbg;               // FMHA_DBG builds: row-0 state dump (else null)
    // v4.8 (APPENDED: keep earlier offsets stable — launch_bounds/Params layout
    // changes measurably shift ptxas scheduling of the decode kernels)
    int* counter;             // persistent-prefill work counter (nullptr else)
    int persist;              // dynamic work-pull prefill mode
    int nctas;                // persistent grid.x
};

DEVI void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* addr) {
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(addr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEVI void ldmatrix_x4_trans(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* addr) {
    unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(addr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEVI void mma_16816(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                    unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI void mma_16816_bf16(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                         unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI unsigned cvt_f16x2_f32(float lo, float hi) {
    unsigned r;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
DEVI unsigned cvt_bf16x2_f32(float lo, float hi) {
    unsigned r;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
// exact f16x2 -> bf16x2 for <=4-significand-bit inputs; +-0 preserved.
DEVI unsigned f16x2_to_bf16x2_exact(unsigned f) {
    unsigned t = (f & 0x7FFF7FFFu) >> 3;
    unsigned nz = (((t & 0xFFFFu) + 0xFFFFu) >> 16)
                | ((((t >> 16) + 0xFFFFu) >> 16) << 16);
    return ((t + 0x38003800u) & (nz * 0xFFFFu)) | (f & 0x80008000u);
}
DEVI unsigned cvt_e4m3x2(unsigned short bytepair) { // 2 fp8 bytes -> f16x2, exact
    unsigned r;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(bytepair));
    return r;
}

// v3.0: exact fp8(e4m3)x2 -> bf16x2 via f32 roundtrip: e4m3->f16 exact,
// f16->f32 exact, cvt.rn.bf16x2.f32 = RN-even == torch .to(bfloat16);
// +-0 and NaN preserved. ~4-5 SASS insts vs ~9 for the integer rebias
// trick, cutting the convert-stage instruction count ~2x.
DEVI unsigned e4m3x2_to_bf16x2(unsigned short bytepair) {
#ifdef PF_NOCONV
    // DIAGNOSTIC ONLY, never defined for a real run, and a local modification to the
    // otherwise verbatim peer copy (recorded in PROVENANCE.md). Replaces the
    // four-instruction E4M3->F16->F32->BF16 chain with one integer multiply so a
    // timing A/B can bound what the whole conversion stage costs this prefill path:
    // tc_convert calls it 2 * D / TPR times per thread per KV tile (at D=128, BNT=64,
    // 128 threads that is 32 K pairs + 32 V pairs = 256 calls, i.e. 1024 instructions
    // per thread per tile), so if the bound is small the direction closes and nothing
    // in the kernel needs to change. Produces wrong numbers by design.
    return (unsigned)bytepair * 0x00010001u;
#else
    unsigned r;
    asm("{\n.reg .f16x2 h2;\n.reg .f16 h0, h1;\n.reg .f32 f0, f1;\n"
        "cvt.rn.f16x2.e4m3x2 h2, %1;\n"
        "mov.b32 {h0, h1}, h2;\n"
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "cvt.rn.bf16x2.f32 %0, f1, f0;\n}"
        : "=r"(r) : "h"(bytepair));
    return r;
#endif
}

// This thread's row's block-table entry for tile start key0, clamped;
// enables one-tile-ahead bt prefetch so the bt LDG no longer gates the KV
// data loads. ROW-AWARE: row = tid/TPR must match kv_prefetch's mapping
// (v3.0 shipped a row-0-only version -> wrong page for rows >0 when
// page_size < BN; 8 correctness failures, job dv_ae699d830f94).
template <int BN, int NT = THREADS>
DEVI int kv_bt_next(const Params& P, int b, int key0, int kvlen, int tbase = 0) {
    const int row = (threadIdx.x - tbase) / (NT / BN);
    int k = min(max(key0 + row, 0), kvlen - 1);
    return __ldg(P.block_table + b * P.bt_stride + (k >> P.page_shift));
}
// Precise exp2 for the softmax weights: degree-8 Taylor of 2^f on [-0.5,0.5]
// (rel err ~3e-10, far below the reference expf's ~1e-7), exponent patched via
// integer add. x <= 0 always in our call sites (weights and rescale factors).
// --use_fast_math maps exp2f to ex2.approx whose ~2^-22 noise is amplified by
// output cancellation in tiny-scale distributions (domain shape 6 evidence).
DEVI float exp2_precise(float x) {
    if (x < -100.f) return 0.f;
    float n = rintf(x);
    float f = x - n;
    float p = 1.3212e-6f;
    p = fmaf(p, f, 1.52527e-5f);
    p = fmaf(p, f, 1.54035e-4f);
    p = fmaf(p, f, 1.3333558e-3f);
    p = fmaf(p, f, 9.618129e-3f);
    p = fmaf(p, f, 5.55041087e-2f);
    p = fmaf(p, f, 0.2402265069f);
    p = fmaf(p, f, 0.69314718056f);
    p = fmaf(p, f, 1.0f);
    return __int_as_float(__float_as_int(p) + ((int)n << 23));
}

DEVI float bf16_bits_to_f32(unsigned b16) { return __uint_as_float(b16 << 16); }
DEVI unsigned short f32_to_bf16_bits(float f) {
    __nv_bfloat16 b = __float2bfloat16_rn(f);
    return *reinterpret_cast<unsigned short*>(&b);
}
// cp.async primitives (sm_80+): 4/8/16B global->shared, no register staging.
DEVI void cp_async4(void* smem_dst, const void* gmem_src) {
    unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d), "l"(gmem_src));
}
DEVI void cp_async8(void* smem_dst, const void* gmem_src) {
    unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(d), "l"(gmem_src));
}
DEVI void cp_async16(void* smem_dst, const void* gmem_src) {
    unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(gmem_src));
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
DEVI void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// smem layout (pipelined): [Q f16 BM x SS]
//   [conv K f16 2 x BN x SS][conv V bf16 2 x BN x SS]
//   [scales f32 2 x 2*BN][row table BM x 8B]
// SS = D + 8: row stride ≡ 4 words (mod 32) keeps ldmatrix phases on 32
// distinct banks. Conv tiles are double-buffered; global->register prefetch
// for tile t+1 is issued during tile t's compute (see attention_loop).
template <int D, int BN, int QR = BM>
struct SmemT {
    static constexpr int SS = D + 8;
    uint16_t* q; uint16_t* kc; uint16_t* vc; float* ksvs;
    DEVI SmemT(char* base) {
        q = reinterpret_cast<uint16_t*>(base);
        kc = q + QR * SS;
        vc = kc + 2 * BN * SS;
        ksvs = reinterpret_cast<float*>(vc + 2 * BN * SS);
    }
    DEVI uint16_t* k(int stage) const { return kc + stage * BN * SS; }
    DEVI uint16_t* v(int stage) const { return vc + stage * BN * SS; }
    DEVI float* ks(int stage) const { return ksvs + stage * 2 * BN; }
    DEVI float* vs(int stage) const { return ksvs + stage * 2 * BN + BN; }
    DEVI float* ra() const { return ksvs + 2 * 2 * BN; }
    static constexpr int bytes() {
        return (QR * SS + 4 * BN * SS) * 2 + 4 * BN * 4 + QR * 8 + 16;
    }
};

// Per-thread staging registers for one KV tile: rows are owned as
// (row = tid>>1, half = tid&1); each half-row is D/2 bytes.
template <int D, int BN, int NT = THREADS>
struct ConvRegs {
    static constexpr int TPR = NT / BN;           // threads per row
    static constexpr int KU64 = D / (8 * TPR);    // 8B K chunks per thread
    static constexpr int VU32 = D / (4 * TPR);    // 4B V chunks per thread
    uint2 k[KU64];
    unsigned v[VU32];
    float ks, vs;      // scales for this thread's row (part==0 only)
};

// Issue global loads for tile [key0, key0+nvalid) into registers. Rows beyond
// nvalid are skipped (convert zero-fills them); addresses never go OOB.
template <int D, int BN, int NT = THREADS>
DEVI void kv_prefetch(ConvRegs<D, BN, NT>& R, const Params& P, int kvh,
                      int key0, int nvalid, int blk, int tbase = 0) {
    constexpr int TPR = ConvRegs<D, BN, NT>::TPR;
    const int tid = threadIdx.x - tbase;
    const int row = tid / TPR;
    const int part = tid % TPR;
    const int pmask = (1 << P.page_shift) - 1;
    if (row < nvalid) {
        const int key = key0 + row;
        const long long krow = (long long)blk * P.k_sb + (long long)(key & pmask) * P.k_sp + kvh * P.k_sh;
        const uint8_t* gk = P.k + krow;
        // v3.8: interleaved chunk ownership c = part + TPR*i. A warp
        // instruction now covers TPR CONTIGUOUS chunks per row: full 32B
        // sectors at TPR=4 (was 8B/sector: part*(D/TPR)+i*8 spread lanes
        // 64B apart -> 4x DRAM amplification + 32 L1 wavefronts/instr on
        // the D256 decode path; also 2x better at TPR=2).
        #pragma unroll
        for (int i = 0; i < ConvRegs<D, BN, NT>::KU64; ++i)
            R.k[i] = *reinterpret_cast<const uint2*>(gk + (part + TPR * i) * 8);
        const long long vrow = (long long)blk * P.v_sb + (long long)(key & pmask) * P.v_sp + kvh * P.v_sh;
        const uint8_t* gv = P.v + vrow;
        #pragma unroll
        for (int i = 0; i < ConvRegs<D, BN, NT>::VU32; ++i)
            R.v[i] = *reinterpret_cast<const unsigned*>(gv + (part + TPR * i) * 4);
        if (part == 0) {
            const int slot = key & pmask;
            R.ks = __ldg(P.ks + blk * P.ks_sb + (long long)slot * P.ks_sp + kvh * P.ks_sh);
            R.vs = __ldg(P.vs + blk * P.vs_sb + (long long)slot * P.vs_sp + kvh * P.vs_sh);
        }
    }
}

// Convert prefetched registers into the staged conv tile (exact conversions:
// K fp8->f16 hw cvt; V fp8->f16->bf16 packed rebias preserving +-0).
template <int D, int BN, int NT = THREADS, int QR = BM>
DEVI void kv_convert(const ConvRegs<D, BN, NT>& R, const SmemT<D, BN, QR>& S, int stage, int nvalid, int tbase = 0) {
    constexpr int SS = D + 8;
    constexpr int TPR = ConvRegs<D, BN, NT>::TPR;
    const int tid = threadIdx.x - tbase;
    const int row = tid / TPR;
    const int part = tid % TPR;
    uint16_t* dk = S.k(stage) + row * SS;   // v3.8: chunk-interleaved stores
    uint16_t* dv = S.v(stage) + row * SS;
    if (row < nvalid) {
        #pragma unroll
        for (int i = 0; i < ConvRegs<D, BN, NT>::KU64; ++i) {
            uint2 raw = R.k[i];
            uint4 o = make_uint4(cvt_e4m3x2((unsigned short)(raw.x & 0xffffu)),
                                 cvt_e4m3x2((unsigned short)(raw.x >> 16)),
                                 cvt_e4m3x2((unsigned short)(raw.y & 0xffffu)),
                                 cvt_e4m3x2((unsigned short)(raw.y >> 16)));
            *reinterpret_cast<uint4*>(dk + (part + TPR * i) * 8) = o;
        }
        #pragma unroll
        for (int i = 0; i < ConvRegs<D, BN, NT>::VU32; ++i) {
            unsigned raw = R.v[i];
            // exact fp8 -> bf16 via f32 roundtrip (uniform; v3.6's D>=256
            // trick experiment measured 0.974x on c3 -> CVT theory dead).
            *reinterpret_cast<uint2*>(dv + (part + TPR * i) * 4) = make_uint2(
                e4m3x2_to_bf16x2((unsigned short)(raw & 0xffffu)),
                e4m3x2_to_bf16x2((unsigned short)(raw >> 16)));
        }
        if (part == 0) {
            S.ks(stage)[row] = R.ks;
            S.vs(stage)[row] = R.vs;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < ConvRegs<D, BN, NT>::KU64; ++i)
            *reinterpret_cast<uint4*>(dk + (part + TPR * i) * 8) = make_uint4(0, 0, 0, 0);
        #pragma unroll
        for (int i = 0; i < ConvRegs<D, BN, NT>::VU32; ++i)
            *reinterpret_cast<uint2*>(dv + (part + TPR * i) * 4) = make_uint2(0, 0);
        if (part == 0) {
            S.ks(stage)[row] = 0.f;
            S.vs(stage)[row] = 0.f;
        }
    }
}

// One KV tile of FA2 math on a staged conv buffer: QK^T (f16 mma), scale by
// sm_scale*log2e*ks_j, bottom-right causal mask via row_visible, online
// softmax in the log2 domain with the -1e30 sentinel, P=bf16(p*vs) exact
// reference rounding, PV (bf16 mma) into O.
#ifdef FMHA_DBG
// three dumped threads: tid0 -> slot0 (row0, cols 0 mod 8), tid1 -> slot1
// (row0, cols 2 mod 8), tid4 -> slot2 (row1, cols 0 mod 8); 96 floats each.
DEVI int dbg_slot() {
    return threadIdx.x == 0 ? 0 : (threadIdx.x == 1 ? 1 : (threadIdx.x == 4 ? 2 : -1));
}
// dump layout (floats): [0..3] sc raw after mma (nt0), [4..7] sc*c, [8..11] post-mask,
// [12] mn0 [13] alpha0 [14] sum0 [15..18] p(nt0..1 pair), [19..22] pf as f32 (nt0/1 lo/hi),
// [23] m2_new [24] l2_new, [25..28] ks[0..3], [29..32] vs[0..3], [33] kn
#endif

template <int D, int BN, typename SM>
DEVI void attention_tile(const Params& P, const SM& S, int stage, int kn,
                         const int row_visible[2], float (*O)[4],
                         float m2[2], float l2[2]) {
    constexpr int SS = D + 8;
    const int lane = threadIdx.x & 31;
    const int t = lane & 3;
    const uint16_t* kbuf = S.k(stage);
    const uint16_t* vbuf = S.v(stage);
    const float* kss = S.ks(stage);
    const float* vss = S.vs(stage);

    float sc[BN / 8][4];
    #pragma unroll
    for (int nt = 0; nt < BN / 8; ++nt)
        sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;

    const int warp_m0 = (threadIdx.x >> 5) * 16;
    for (int kk = 0; kk < D / 16; ++kk) {
        unsigned qa[4];
        ldmatrix_x4(qa[0], qa[1], qa[2], qa[3],
                    S.q + (warp_m0 + (lane & 15)) * SS + kk * 16 + (lane >> 4) * 8);
        #pragma unroll
        for (int nn = 0; nn < BN / 16; ++nn) {
            unsigned r0, r1, r2, r3;
            // B[k=dim][n=key] for row-major K[key][dim] is a NON-trans
            // ldmatrix; n8(key) tiles pair as {r0,r2} and {r1,r3}.
            ldmatrix_x4(r0, r1, r2, r3,
                        kbuf + (nn * 16 + (lane & 15)) * SS + kk * 16 + (lane >> 4) * 8);
            mma_16816(sc[nn * 2], qa[0], qa[1], qa[2], qa[3], r0, r2);
            mma_16816(sc[nn * 2 + 1], qa[0], qa[1], qa[2], qa[3], r1, r3);
        }
    }

#ifdef FMHA_DBG
    if (P.dbg && kn == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        int sl = dbg_slot();
        if (sl >= 0) {
            float* db = P.dbg + 96 * sl;
            db[0] = sc[0][0]; db[1] = sc[0][1]; db[2] = sc[0][2]; db[3] = sc[0][3];
            db[33] = (float)kn;
            for (int j = 0; j < 4; ++j) { db[25 + j] = kss[j]; db[29 + j] = vss[j]; }
            if (sl == 0) {
                for (int j = 0; j < 8; ++j) {
                    db[56 + j] = (float)(int)S.v(0)[j];
                    db[64 + j] = (float)(int)S.k(0)[j];
                    db[72 + j] = (float)(int)S.q[j];
                }
            }
        }
    }
#endif
    float mn0 = m2[0], mn1 = m2[1];
    #pragma unroll
    for (int nt = 0; nt < BN / 8; ++nt) {
        int col0 = nt * 8 + 2 * t;
        float c0 = kss[col0] * P.sm_scale_log2e;
        float c1 = kss[col0 + 1] * P.sm_scale_log2e;
        sc[nt][0] *= c0; sc[nt][1] *= c1;
        sc[nt][2] *= c0; sc[nt][3] *= c1;
        if (kn + col0     > row_visible[0]) sc[nt][0] = NEG;
        if (kn + col0 + 1 > row_visible[0]) sc[nt][1] = NEG;
        if (kn + col0     > row_visible[1]) sc[nt][2] = NEG;
        if (kn + col0 + 1 > row_visible[1]) sc[nt][3] = NEG;
        mn0 = fmaxf(mn0, fmaxf(sc[nt][0], sc[nt][1]));
        mn1 = fmaxf(mn1, fmaxf(sc[nt][2], sc[nt][3]));
    }
    #pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        mn0 = fmaxf(mn0, __shfl_xor_sync(0xffffffffu, mn0, off, 4));
        mn1 = fmaxf(mn1, __shfl_xor_sync(0xffffffffu, mn1, off, 4));
    }
#ifdef FMHA_DBG
    if (P.dbg && kn == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        int sl = dbg_slot();
        if (sl >= 0) {
            float* db = P.dbg + 96 * sl;
            db[4] = sc[0][0]; db[5] = sc[0][1]; db[8] = sc[0][2]; db[9] = sc[0][3];
            db[12] = mn0;
        }
    }
#endif
    float alpha0 = exp2f(m2[0] - mn0);
    float alpha1 = exp2f(m2[1] - mn1);
    float sum0 = 0.f, sum1 = 0.f;
    unsigned pf[BN / 16][4];
    #pragma unroll
    for (int nt = 0; nt < BN / 8; ++nt) {
        float p0 = exp2f(sc[nt][0] - mn0);
        float p1 = exp2f(sc[nt][1] - mn0);
        float p2 = exp2f(sc[nt][2] - mn1);
        float p3 = exp2f(sc[nt][3] - mn1);
        sum0 += p0 + p1; sum1 += p2 + p3;
        int col0 = nt * 8 + 2 * t;
        // weighted_p = bf16(p * v_scale): exact reference rounding; the
        // staged bf16 V payload is exact too, so PV products match the
        // reference bit-for-bit (only f32 accumulation order differs).
        pf[nt >> 1][(nt & 1) * 2 + 0] = cvt_bf16x2_f32(p0 * vss[col0], p1 * vss[col0 + 1]);
        pf[nt >> 1][(nt & 1) * 2 + 1] = cvt_bf16x2_f32(p2 * vss[col0], p3 * vss[col0 + 1]);
    }
    #pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, off, 4);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, off, 4);
    }
#ifdef FMHA_DBG
    if (P.dbg && kn == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        int sl = dbg_slot();
        if (sl >= 0) {
            float* db = P.dbg + 96 * sl;
            db[13] = alpha0; db[14] = sum0;
            db[19] = __uint_as_float(pf[0][0]); db[20] = __uint_as_float(pf[0][1]);
            db[21] = __uint_as_float(pf[1][0]); db[22] = __uint_as_float(pf[1][1]);
            db[23] = mn0; db[24] = l2[0] * alpha0 + sum0;
        }
    }
#endif
    m2[0] = mn0; m2[1] = mn1;
    l2[0] = l2[0] * alpha0 + sum0;
    l2[1] = l2[1] * alpha1 + sum1;
    #pragma unroll
    for (int j = 0; j < D / 8; ++j) {
        O[j][0] *= alpha0; O[j][1] *= alpha0;
        O[j][2] *= alpha1; O[j][3] *= alpha1;
    }
    #pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
        #pragma unroll
        for (int nn = 0; nn < D / 16; ++nn) {
            unsigned r0, r1, r2, r3;
            ldmatrix_x4_trans(r0, r1, r2, r3,
                              vbuf + (kk * 16 + (lane & 15)) * SS + nn * 16 + (lane >> 4) * 8);
            mma_16816_bf16(O[nn * 2], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], r0, r1);
            mma_16816_bf16(O[nn * 2 + 1], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], r2, r3);
        }
    }
}

// Pipelined FA2 driver over KV tiles [kn0, kn_end): global->register
// prefetch of tile t+1 is issued between the convert and compute of tile t,
// so load latency overlaps the mma phase. Converted tiles double-buffered.
template <int D, int BNT, int NT = THREADS, int QR = BM>
DEVI void attention_loop(const Params& P, const SmemT<D, BNT, QR>& S, int b, int kvh,
                         int kn0, int kn_end, const int row_visible[2],
                         float (*O)[4], float m2[2], float l2[2], int mtiles) {
    ConvRegs<D, BNT, NT> R;
    int nvalid0 = min(BNT, kn_end - kn0);
    // v3.0: bt_n always holds the block-table entry for the NEXT prefetch,
    // loaded a full tile earlier so data LDGs issue without a bt roundtrip.
    int bt_n = kv_bt_next<BNT, NT>(P, b, kn0, kn_end);
    kv_prefetch<D, BNT, NT>(R, P, kvh, kn0, nvalid0, bt_n);
    if (kn0 + BNT < kn_end) bt_n = kv_bt_next<BNT, NT>(P, b, kn0 + BNT, kn_end);
    int stage = 0;
    for (int kn = kn0; kn < kn_end; kn += BNT, stage ^= 1) {
        int nvalid = min(BNT, kn_end - kn);
        kv_convert<D, BNT, NT, QR>(R, S, stage, nvalid);
        if (kn + BNT < kn_end) {
            kv_prefetch<D, BNT, NT>(R, P, kvh, kn + BNT, min(BNT, kn_end - kn - BNT), bt_n);
            bt_n = kv_bt_next<BNT, NT>(P, b, kn + 2 * BNT, kn_end);
        }
        __syncthreads();
        // v2.6: warp-granular M-tile skip. attention_tile has NO internal
        // barriers; warps owning rows >= mtiles*16 (decode: R = qlen*g often
        // << BM) skip all QK/softmax/PV mma work -> tensor throughput scales
        // with real rows instead of the 64-row padding (c4: 4x waste at R=1).
        if ((threadIdx.x >> 5) < mtiles)
            attention_tile<D, BNT, SmemT<D, BNT, QR>>(P, S, stage, kn, row_visible, O, m2, l2);
        __syncthreads();
    }
}

// v4.9: warp-split decode loop. Warps [0, mtiles) run attention_tile(t)
// WHILE warps [NW-CONV_W, NW) prefetch+convert tile t+1 into the other conv
// stage — tile time = max(attn, conv) instead of their sum, and ONE
// __syncthreads per tile (was two). Targets c2/c4 (mtiles=1: single-warp
// attention ~3000clk serial + barrier-3.02/inst idle) with conv on the
// upper half of the CTA. Requires mtiles <= NW-CONV_W = NT/64.
template <int D, int BNT, int NT = THREADS, int QR = BM>
DEVI void attention_loop_ws(const Params& P, const SmemT<D, BNT, QR>& S, int b, int kvh,
                            int kn0, int kn_end, const int row_visible[2],
                            float (*O)[4], float m2[2], float l2[2], int mtiles) {
    constexpr int NW = NT / 32;
    constexpr int CONV_W = NT / 64;          // convert warps (2 @NT=128, 4 @256)
    constexpr int NT_C = NT / 2;             // convert threads
    constexpr int TB = NT - NT_C;            // their tbase
    const int warp = threadIdx.x >> 5;
    const bool is_attn = (warp < mtiles);
    const bool is_conv = (warp >= NW - CONV_W);
    const int ntiles = (kn_end - kn0 + BNT - 1) / BNT;

    ConvRegs<D, BNT, NT_C> R[2];
    int bt_n = 0;
    if (is_conv) {
        int bt0 = kv_bt_next<BNT, NT_C>(P, b, kn0, kn_end, TB);
        kv_prefetch<D, BNT, NT_C>(R[0], P, kvh, kn0, min(BNT, kn_end - kn0), bt0, TB);
        if (ntiles > 1) {
            bt_n = kv_bt_next<BNT, NT_C>(P, b, kn0 + BNT, kn_end, TB);
            kv_prefetch<D, BNT, NT_C>(R[1], P, kvh, kn0 + BNT, min(BNT, kn_end - kn0 - BNT), bt_n, TB);
            bt_n = kv_bt_next<BNT, NT_C>(P, b, kn0 + 2 * BNT, kn_end, TB);
        }
        kv_convert<D, BNT, NT_C, QR>(R[0], S, 0, min(BNT, kn_end - kn0), TB);
    }
    __syncthreads();
    int stage = 0;
    for (int t = 0; t < ntiles; ++t, stage ^= 1) {
        const int kn = kn0 + t * BNT;
        if (is_attn)
            attention_tile<D, BNT, SmemT<D, BNT, QR>>(P, S, stage, kn, row_visible, O, m2, l2);
        if (is_conv && t + 1 < ntiles) {
            // convert(t+1) from regs loaded last iter; refill freed buffer.
            // Explicit parity branches: R[] indexed by runtime t would spill
            // the whole payload to local memory.
            const int nv1 = min(BNT, kn_end - kn - BNT);
            if (t & 1) {
                kv_convert<D, BNT, NT_C, QR>(R[0], S, stage ^ 1, nv1, TB);
                if (t + 2 < ntiles) {
                    kv_prefetch<D, BNT, NT_C>(R[1], P, kvh, kn + 2 * BNT,
                                              min(BNT, kn_end - kn - 2 * BNT), bt_n, TB);
                    bt_n = kv_bt_next<BNT, NT_C>(P, b, kn + 3 * BNT, kn_end, TB);
                }
            } else {
                kv_convert<D, BNT, NT_C, QR>(R[1], S, stage ^ 1, nv1, TB);
                if (t + 2 < ntiles) {
                    kv_prefetch<D, BNT, NT_C>(R[0], P, kvh, kn + 2 * BNT,
                                              min(BNT, kn_end - kn - 2 * BNT), bt_n, TB);
                    bt_n = kv_bt_next<BNT, NT_C>(P, b, kn + 3 * BNT, kn_end, TB);
                }
            }
        }
        __syncthreads();
    }
}

// Stage Q (bf16 global -> f16 smem); row_base[r] < 0 zero-fills the row.
template <int D, int NT = THREADS>
DEVI void stage_q_d(uint16_t* dst, const uint16_t* qsrc,
                    const long long* row_base, int nrows, int qrows) {
    constexpr int SS = D + 8;
    constexpr int PER_ROW = D / 8; // 8 bf16 (16B) per chunk
    for (int idx = threadIdx.x; idx < qrows * PER_ROW; idx += NT) {
        int row = idx / PER_ROW;
        int col8 = (idx % PER_ROW) * 8;
        uint16_t* d = dst + row * SS + col8;
        if (row >= nrows || row_base[row] < 0) {
            *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
            continue;
        }
        uint4 raw = *reinterpret_cast<const uint4*>(qsrc + row_base[row] + col8);
        unsigned short* bb = reinterpret_cast<unsigned short*>(&raw);
        uint4 o;
        unsigned* oo = reinterpret_cast<unsigned*>(&o);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            oo[i] = cvt_f16x2_f32(bf16_bits_to_f32(bb[2 * i]), bf16_bits_to_f32(bb[2 * i + 1]));
        *reinterpret_cast<uint4*>(d) = o;
    }
}

DEVI void epilogue_out(uint16_t* out, const long long row_base[2],
                       int D, const float (*O)[4], const float l2[2]) {
    const int lane = threadIdx.x & 31;
    const int t = lane & 3;
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (row_base[r] < 0) continue;
        uint16_t* orow = out + row_base[r];
        float inv = 1.f / l2[r];
        #pragma unroll
        for (int nt = 0; nt < D / 8; ++nt) {
            // DIRECT f32 -> bf16 (RN), matching the reference out.copy_().
            // The previous f32->f16->bf16 double conversion corrupted tiny
            // outputs: f16 subnormals (< 6.1e-5) quantize on a 5.96e-8 grid
            // (~1.4% rel_l2 on the tiny distribution; domain shape 6 failure).
            unsigned pk = cvt_bf16x2_f32(O[nt][2 * r] * inv, O[nt][2 * r + 1] * inv);
            *reinterpret_cast<unsigned*>(orow + nt * 8 + 2 * t) = pk;
        }
    }
}

// ---- prefill kernel: CTA = (q tile, head) ---------------------------------
template <int D, int BN>
__global__ void __launch_bounds__(THREADS) fa_prefill(const __grid_constant__ Params P) {
    extern __shared__ char smem_raw[];
    SmemT<D, BN> S(smem_raw);
    int gx = blockIdx.x;
    int b = -1, tile_local = 0, base = 0, q0 = 0, qlen = 0;
    {
        int prev = 0;
        for (int i = 0; i < P.batch; ++i) {
            int c = __ldg(P.cu + i + 1);
            int ntiles = (c - prev + BM - 1) / BM;
            if (gx < base + ntiles) {
                b = i; tile_local = gx - base; q0 = prev; qlen = c - prev; break;
            }
            base += ntiles;
            prev = c;
        }
    }
    if (b < 0) return;
    int i0 = tile_local * BM;
    if (i0 >= qlen) return;
    int nrows = min(BM, qlen - i0);
    int h = blockIdx.y;
    int kvh = h / P.g;
    int kvlen = __ldg(P.seqk + b);

    long long row_base[2]; // this lane's two rows (g, g+8)
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int g = lane >> 2;
    int rr0 = warp * 16 + g, rr1 = rr0 + 8;
    row_base[0] = rr0 < nrows ? (long long)(q0 + i0 + rr0) * P.q_out_stride + h * D : -1;
    row_base[1] = rr1 < nrows ? (long long)(q0 + i0 + rr1) * P.q_out_stride + h * D : -1;

    // stage all 64 Q rows: row r address = (q0+i0+r)*stride + h*D (r<nrows)
    {
        // per-thread temporary row table via recomputation in stage_q_d:
        // build a smem-free lambda: stage_q_d needs row_base per row; compute
        // arithmetic addresses into a small smem region first.
        long long* ra = reinterpret_cast<long long*>(S.ra());
        for (int r = threadIdx.x; r < BM; r += THREADS)
            ra[r] = (r < nrows) ? (long long)(q0 + i0 + r) * P.q_out_stride + h * D : -1;
        __syncthreads();
        stage_q_d<D>(S.q, P.q, ra, nrows, BM);
    }
    __syncthreads();

    int row_visible[2];
    row_visible[0] = rr0 < nrows ? (kvlen - qlen + i0 + rr0) : 0;
    row_visible[1] = rr1 < nrows ? (kvlen - qlen + i0 + rr1) : 0;

    float O[D / 8][4];
    #pragma unroll
    for (int j = 0; j < D / 8; ++j) O[j][0] = O[j][1] = O[j][2] = O[j][3] = 0.f;
    float m2[2] = {NEG, NEG};
    float l2[2] = {0.f, 0.f};

    int kn_end = min(kvlen, kvlen - qlen + i0 + BM);
#ifdef FMHA_DBG
    if (P.dbg && blockIdx.x == 0 && blockIdx.y == 0) {
        int sl = dbg_slot();
        if (sl >= 0) {
            float* db = P.dbg + 96 * sl;
            db[80] = (float)kn_end; db[81] = (float)nrows;
            db[82] = (float)kvlen; db[83] = (float)qlen;
            db[84] = (float)row_visible[0]; db[85] = (float)min(BN, kn_end);
            db[86] = (float)row_visible[1];
        }
    }
#endif
    attention_loop<D, BN>(P, S, b, kvh, 0, kn_end, row_visible, O, m2, l2,
                          (nrows + 15) >> 4);

#ifdef FMHA_DBG
    if (P.dbg && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 4) {
        // full row-0 final state: lanes 0..3 (t=0..3) cover all D columns.
        // layout: dbg[288] = l2 (tid0); per lane L: O cols at 289+1.. :
        //   dbg[289 + L*(2*D/8) + nt]      = O[nt][0] (col nt*8 + 2L)
        //   dbg[289 + L*(2*D/8) + D/8 + nt]= O[nt][1] (col nt*8 + 2L + 1)
        int L = threadIdx.x;
        if (L == 0) P.dbg[288] = l2[0];
        #pragma unroll
        for (int nt = 0; nt < D / 8; ++nt) {
            P.dbg[289 + L * (2 * D / 8) + nt] = O[nt][0];
            P.dbg[289 + L * (2 * D / 8) + D / 8 + nt] = O[nt][1];
        }
    }
#endif
    epilogue_out(P.out, row_base, D, O, l2);
#ifdef FMHA_DBG
    if (P.dbg && blockIdx.x == 0 && blockIdx.y == 0) {
        __syncthreads();  // block-uniform (P.dbg uniform across the block)
        if (threadIdx.x < D / 2) {
            uint16_t* orow = P.out + (long long)(q0 + i0) * P.q_out_stride + h * D;
            P.dbg[289 + 4 * (2 * D / 8) + threadIdx.x] =
                __uint_as_float(reinterpret_cast<unsigned*>(orow)[threadIdx.x]);
        }
    }
#endif
}

// ================= cp.async pipelined decode path (v1c) ====================
// Raw fp8 ring (depth RING) fed by cp.async; convert-from-smem into the
// double-buffered f16/bf16 conv tiles consumed by attention_tile. Global
// latency is hidden by up to RING-1 outstanding tile transfers with zero
// register payload, allowing 2-3 CTAs/SM.
template <int D, int BNT, int RING>
struct SmemD {
    static constexpr int SS = D + 8;
    static constexpr int RAWROW = D + 16;           // padded raw row bytes
    static constexpr int RAWSLOT = BNT * RAWROW * 2 + 2 * BNT * 4; // K+V+scales
    uint16_t* q; uint16_t* kc; uint16_t* vc; float* ksvs;
    unsigned char* raw; uint64_t* pad;
    DEVI SmemD(char* base) {
        q = reinterpret_cast<uint16_t*>(base);
        kc = q + BM * SS;
        vc = kc + 2 * BNT * SS;
        ksvs = reinterpret_cast<float*>(vc + 2 * BNT * SS);
        raw = reinterpret_cast<unsigned char*>(ksvs + 4 * BNT);
    }
    DEVI uint16_t* k(int s) const { return kc + s * BNT * SS; }
    DEVI uint16_t* v(int s) const { return vc + s * BNT * SS; }
    DEVI float* ks(int s) const { return ksvs + s * 2 * BNT; }
    DEVI float* vs(int s) const { return ksvs + s * 2 * BNT + BNT; }
    DEVI unsigned char* rawslot(int i) const { return raw + (i % RING) * RAWSLOT; }
    DEVI float* rawsc(int i) const {
        return reinterpret_cast<float*>(raw + (i % RING) * RAWSLOT + 2 * BNT * RAWROW);
    }
    DEVI long long* ra() const {
        return reinterpret_cast<long long*>(raw + RING * RAWSLOT);
    }
    static constexpr int bytes() {
        return BM * SS * 2 + 4 * BNT * SS * 2 + 4 * BNT * 4
             + RING * RAWSLOT + BM * 8 + 16;
    }
};

// Issue cp.async for one KV tile into raw ring slot `slot`.
template <int D, int BNT, int RING>
DEVI void kv_cpasync(const SmemD<D, BNT, RING>& S, int slot, const Params& P,
                     int b, int kvh, int key0, int nvalid) {
    constexpr int TPR = THREADS / BNT;             // threads per row
    constexpr int KU64 = D / (8 * TPR);            // 8B K chunks per thread
    constexpr int VU32 = D / (4 * TPR);            // 4B V chunks per thread
    const int tid = threadIdx.x;
    const int row = tid / TPR;
    const int part = tid % TPR;
    const int pmask = (1 << P.page_shift) - 1;
    unsigned char* rk = S.rawslot(slot);
    unsigned char* rv = rk + BNT * SmemD<D, BNT, RING>::RAWROW;
    if (row < BNT) {
        if (row < nvalid) {
            const int key = key0 + row;
            const int blk = __ldg(P.block_table + b * P.bt_stride + (key >> P.page_shift));
            const int slotk = key & pmask;
            const uint8_t* gk = P.k + blk * P.k_sb + (long long)slotk * P.k_sp + kvh * P.k_sh
                              + part * (D / TPR);
            unsigned char* dk = rk + row * SmemD<D, BNT, RING>::RAWROW + part * (D / TPR);
            #pragma unroll
            for (int i = 0; i < KU64; ++i) cp_async8(dk + i * 8, gk + i * 8);
            const uint8_t* gv = P.v + blk * P.v_sb + (long long)slotk * P.v_sp + kvh * P.v_sh
                              + part * (D / TPR);
            unsigned char* dv = rv + row * SmemD<D, BNT, RING>::RAWROW + part * (D / TPR);
            #pragma unroll
            for (int i = 0; i < VU32; ++i) cp_async4(dv + i * 4, gv + i * 4);
            if (part == 0) {
                const float* gks = P.ks + blk * P.ks_sb + (long long)slotk * P.ks_sp + kvh * P.ks_sh;
                const float* gvs = P.vs + blk * P.vs_sb + (long long)slotk * P.vs_sp + kvh * P.vs_sh;
                float* sc = reinterpret_cast<float*>(S.rawsc(slot));
                cp_async4(sc + row, gks);
                cp_async4(sc + BNT + row, gvs);
            }
        } else {
            // invalid row: zero-fill synchronously (tiny, tail only)
            unsigned char* dk = rk + row * SmemD<D, BNT, RING>::RAWROW + part * (D / TPR);
            unsigned char* dv = rv + row * SmemD<D, BNT, RING>::RAWROW + part * (D / TPR);
            #pragma unroll
            for (int i = 0; i < KU64; ++i)
                *reinterpret_cast<uint2*>(dk + i * 8) = make_uint2(0u, 0u);
            #pragma unroll
            for (int i = 0; i < VU32; ++i)
                *reinterpret_cast<unsigned*>(dv + i * 4) = 0u;
            if (part == 0) {
                float* sc = reinterpret_cast<float*>(S.rawsc(slot));
                sc[row] = 0.f;
                sc[BNT + row] = 0.f;
            }
        }
    }
    cp_async_commit();
}

// Convert raw ring slot -> conv stage (K f16, V exact bf16), scales -> smem.
template <int D, int BNT, int RING>
DEVI void kv_convert_raw(const SmemD<D, BNT, RING>& S, int slot, int stage) {
    constexpr int SS = D + 8;
    constexpr int TPR = THREADS / BNT;
    constexpr int KU64 = D / (8 * TPR);
    constexpr int VU32 = D / (4 * TPR);
    constexpr int RAWROW = SmemD<D, BNT, RING>::RAWROW;
    const int tid = threadIdx.x;
    const int row = tid / TPR;
    const int part = tid % TPR;
    const unsigned char* rk = S.rawslot(slot);
    const unsigned char* rv = rk + BNT * RAWROW;
    uint16_t* dk = S.k(stage) + row * SS + part * (D / TPR);
    uint16_t* dv = S.v(stage) + row * SS + part * (D / TPR);
    #pragma unroll
    for (int i = 0; i < KU64; ++i) {
        uint2 rawv = *reinterpret_cast<const uint2*>(rk + row * RAWROW + part * (D / TPR) + i * 8);
        uint4 o = make_uint4(cvt_e4m3x2((unsigned short)(rawv.x & 0xffffu)),
                             cvt_e4m3x2((unsigned short)(rawv.x >> 16)),
                             cvt_e4m3x2((unsigned short)(rawv.y & 0xffffu)),
                             cvt_e4m3x2((unsigned short)(rawv.y >> 16)));
        *reinterpret_cast<uint4*>(dk + i * 8) = o;
    }
    #pragma unroll
    for (int i = 0; i < VU32; ++i) {
        unsigned rawv = *reinterpret_cast<const unsigned*>(rv + row * RAWROW + part * (D / TPR) + i * 4);
        unsigned f0 = cvt_e4m3x2((unsigned short)(rawv & 0xffffu));
        unsigned f1 = cvt_e4m3x2((unsigned short)(rawv >> 16));
        unsigned b0, b1;
        {
            unsigned t = (f0 & 0x7FFF7FFFu) >> 3;
            unsigned nz = (((t & 0xFFFFu) + 0xFFFFu) >> 16)
                        | ((((t >> 16) + 0xFFFFu) >> 16) << 16);
            b0 = ((t + 0x38003800u) & (nz * 0xFFFFu)) | (f0 & 0x80008000u);
        }
        {
            unsigned t = (f1 & 0x7FFF7FFFu) >> 3;
            unsigned nz = (((t & 0xFFFFu) + 0xFFFFu) >> 16)
                        | ((((t >> 16) + 0xFFFFu) >> 16) << 16);
            b1 = ((t + 0x38003800u) & (nz * 0xFFFFu)) | (f1 & 0x80008000u);
        }
        *reinterpret_cast<uint2*>(dv + i * 4) = make_uint2(b0, b1);
    }
    if (part == 0) {
        const float* sc = reinterpret_cast<const float*>(S.rawsc(slot));
        S.ks(stage)[row] = sc[row];
        S.vs(stage)[row] = sc[BNT + row];
    }
}

// Pipelined decode driver: cp.async ring + conv double buffer.
template <int D, int BNT, int RING>
DEVI void attention_loop_async(const Params& P, const SmemD<D, BNT, RING>& S,
                               int b, int kvh, int kn0, int kn_end,
                               const int row_visible[2], float (*O)[4],
                               float m2[2], float l2[2]) {
    const int ntiles = (kn_end - kn0 + BNT - 1) / BNT;
    // prologue: issue first min(RING-1, ntiles) tiles
    #pragma unroll
    for (int i = 0; i < RING - 1; ++i) {
        if (i < ntiles) {
            int kn = kn0 + i * BNT;
            kv_cpasync<D, BNT, RING>(S, i, P, b, kvh, kn, min(BNT, kn_end - kn));
        } else {
            cp_async_commit();  // keep group accounting uniform
        }
    }
    int stage = 0;
    for (int t = 0; t < ntiles; ++t, stage ^= 1) {
        if (t + RING - 1 < ntiles) {
            int kn = kn0 + (t + RING - 1) * BNT;
            kv_cpasync<D, BNT, RING>(S, t + RING - 1, P, b, kvh, kn,
                                     min(BNT, kn_end - kn));
        } else {
            cp_async_commit();
        }
        cp_async_wait<RING - 2>();   // tile t completed
        __syncthreads();
        kv_convert_raw<D, BNT, RING>(S, t, stage);
        __syncthreads();
        attention_tile<D, BNT, SmemD<D, BNT, RING>>(P, S, stage,
                               kn0 + t * BNT, row_visible, O, m2, l2);
    }
}

// ---- decode kernel: CTA = (chunk, b*kvh); rows = qlen*g (tokens x group) --
// grid.x = nchunks, grid.y = batch*kv_heads. Row r: token i = r/g, head
// kvh*g + r%g. If nchunks==1 writes out directly; else writes f32 partials
// [m2 Rmax][l2 Rmax][O Rmax*D] at scratch[(bkvh*nchunks + chunk)*stride].
// v3.0: NT = decode thread count. D128 decode runs NT=256 (2 CTAs/SM x 8
// warps = 16 warps vs 8: case2/3 profiles show latency starvation). Warps
// >= mtiles never enter attention_tile (v2.6 guard); their O/m2/l2 stay
// garbage and every write path is row-guarded (rr < R / row_base < 0).
// v4.6: D256 decode runs NT=256 (8 warps, still 1 CTA/SM by smem/regs —
// attacks c3's 17.9-cyc/inst latency starvation at zero occupancy cost).
// Plain launch_bounds(NT): minBlocks=2 would cap regs at 128 and spill
// D256's O[128].
template <int D, int BN, int NT = THREADS, int QR = BM>
__global__ void __launch_bounds__(NT, (NT == 256 && D < 256) ? 2 : 1) fa_decode(const __grid_constant__ Params P) {
    extern __shared__ char smem_raw[];
    SmemT<D, BN, QR> S(smem_raw);
    int bkvh = blockIdx.y;
    int kv_heads = P.heads / P.g;
    int b = bkvh / kv_heads;
    int kvh = bkvh % kv_heads;
    int q0 = __ldg(P.cu + b);
    int qlen = __ldg(P.cu + b + 1) - q0;
    int kvlen = __ldg(P.seqk + b);
    int R = qlen * P.g;
    int c = blockIdx.x;
    int kn0 = c * P.chunk_size;
    long long pbase = ((long long)bkvh * P.nchunks + c) * P.chunk_stride;

    if (kn0 >= kvlen) { // empty chunk: mark partial inactive
        if (P.nchunks > 1 && threadIdx.x < P.Rmax) {
            P.scratch[pbase + threadIdx.x] = NEG;
            P.scratch[pbase + P.Rmax + threadIdx.x] = 0.f;
        }
        return;
    }

    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int g = lane >> 2;
    int rr0 = warp * 16 + g, rr1 = rr0 + 8;
    long long row_base[2];
    #pragma unroll
    for (int x = 0; x < 2; ++x) {
        int rr = x == 0 ? rr0 : rr1;
        row_base[x] = rr < R ? (long long)(q0 + rr / P.g) * P.q_out_stride
                                 + (kvh * P.g + rr % P.g) * D
                             : -1;
    }
    {
        long long* ra = reinterpret_cast<long long*>(S.ra());
        for (int r = threadIdx.x; r < QR; r += NT)
            ra[r] = r < R ? (long long)(q0 + r / P.g) * P.q_out_stride
                            + (kvh * P.g + r % P.g) * D
                          : -1;
        __syncthreads();
        stage_q_d<D, NT>(S.q, P.q, ra, R, QR);
    }
    __syncthreads();

    int row_visible[2];
    row_visible[0] = rr0 < R ? (kvlen - qlen + rr0 / P.g) : 0;
    row_visible[1] = rr1 < R ? (kvlen - qlen + rr1 / P.g) : 0;

    float O[D / 8][4];
    #pragma unroll
    for (int j = 0; j < D / 8; ++j) O[j][0] = O[j][1] = O[j][2] = O[j][3] = 0.f;
    float m2[2] = {NEG, NEG};
    float l2[2] = {0.f, 0.f};

    int kn_end = min(kvlen, kn0 + P.chunk_size);
    const int mtiles = (R + 15) >> 4;
    // v4.10 gate: WS wins where attention dominates (D256 256-mma tiles:
    // c3 1.189x; D64: c8 1.127x) but LOSES at D128 (c4 0.860x, c5/c6 ~0.94x:
    // conv and attn are comparable and the 2-warp TPR=1 convert doubles
    // per-thread work). D128 keeps the v2.6 serial loop.
    if (mtiles <= NT / 64 && D != 128)
        attention_loop_ws<D, BN, NT, QR>(P, S, b, kvh, kn0, kn_end, row_visible, O, m2, l2, mtiles);
    else
        attention_loop<D, BN, NT, QR>(P, S, b, kvh, kn0, kn_end, row_visible, O, m2, l2, mtiles);

    if (P.nchunks == 1) {
        epilogue_out(P.out, row_base, D, O, l2);
        return;
    }
    float* dst = P.scratch + pbase;
    int t = lane & 3;
    long long ooff = (2LL * P.Rmax + 3) & ~3LL;
    #pragma unroll
    for (int x = 0; x < 2; ++x) {
        int rr = x == 0 ? rr0 : rr1;
        if (rr >= R) continue;
        dst[rr] = m2[x];
        dst[P.Rmax + rr] = l2[x];
        float* orow = dst + ooff + (long long)rr * D;
        #pragma unroll
        for (int nt = 0; nt < D / 8; ++nt)
            *reinterpret_cast<float2*>(orow + nt * 8 + 2 * t) =
                make_float2(O[nt][2 * x], O[nt][2 * x + 1]);
    }
}


// ---- per-TU launcher plumbing ----------------------------------------------
// Kernel templates are defined above; each head_dim is instantiated in exactly
// one TU (fmha_p1.cu: 64/128, fmha_p2.cu: 256) so ptxas work per file stays
// small and ninja builds the TUs in parallel.
static inline void set_smem(const void* fn, int bytes) {
    static std::vector<std::pair<const void*, int>> done;
    for (auto& p : done)
        if (p.first == fn) return;
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    done.push_back({fn, bytes});
}

#define DEFINE_FMHA_LAUNCHERS(D, BNV, DNT)                                       \
    void launch_prefill_##D(const fmha::Params& P, int Tq, int smem, cudaStream_t st) { \
        fmha::set_smem(reinterpret_cast<const void*>(fmha::fa_prefill<D, BNV>), smem); \
        fmha::fa_prefill<D, BNV><<<dim3((Tq + fmha::BM - 1) / fmha::BM + P.batch, P.heads), \
                               fmha::THREADS, smem, st>>>(P);                    \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    }                                                                            \
    void launch_decode_##D(const fmha::Params& P, int nchunks, int base_ctas, int smem, cudaStream_t st, int qr) { \
        if (qr <= 16) {                                                          \
            fmha::set_smem(reinterpret_cast<const void*>(fmha::fa_decode<D, BNV, DNT, 16>), smem); \
            fmha::fa_decode<D, BNV, DNT, 16><<<dim3(nchunks, base_ctas), DNT, smem, st>>>(P); \
        } else {                                                                 \
            fmha::set_smem(reinterpret_cast<const void*>(fmha::fa_decode<D, BNV, DNT, fmha::BM>), smem); \
            fmha::fa_decode<D, BNV, DNT, fmha::BM><<<dim3(nchunks, base_ctas), DNT, smem, st>>>(P); \
        }                                                                        \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
    }                                                                            \
    int decode_occ_##D(int smem, int qr) {                                       \
        int n = 0;                                                               \
        if (qr <= 16)                                                            \
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(                       \
                &n, fmha::fa_decode<D, BNV, DNT, 16>, DNT, smem);                \
        else                                                                     \
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(                       \
                &n, fmha::fa_decode<D, BNV, DNT, fmha::BM>, DNT, smem);          \
        return n;                                                                \
    }

} // namespace fmha

#define DECLARE_FMHA_LAUNCHERS(D)                                              \
    void launch_prefill_##D(const fmha::Params& P, int Tq, int smem, cudaStream_t st); \
    void launch_decode_##D(const fmha::Params& P, int nchunks, int base_ctas, int smem, cudaStream_t st, int qr); \
    int decode_occ_##D(int smem, int qr);

DECLARE_FMHA_LAUNCHERS(64)
DECLARE_FMHA_LAUNCHERS(128)
DECLARE_FMHA_LAUNCHERS(256)
