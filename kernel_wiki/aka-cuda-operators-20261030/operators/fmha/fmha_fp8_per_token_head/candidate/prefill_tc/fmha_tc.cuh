// tcgen05 (SM103/Blackwell-class) prefill kernel for fmha_fp8_per_token_head.
// Independent implementation; mechanisms follow the public PTX ISA and the
// FA3/FA4-style synchronous phase structure validated via experiments/
// tc_probe.py on this device.
//
// Topology per CTA (128 threads, 4 warps), M=128 q rows x head:
//   TMEM (256 cols): S0 [0,64) S1 [64,128)  (P packed bf16 reuses S cols)
//                    O [128, 128+D) f32
//   smem: Q bf16 canonical K-major image | K bf16 interleave images x2
//         | V bf16 swizzle-128B MN-major images x2 | ks/vs f32 x2 | bars
// Per KV tile t (stage s=t&1), synchronous phases with elected-thread mma:
//   convert(t) from prefetched regs -> prefetch(t+1) -> fence+bar
//   -> QK mma issue + commit -> mbar wait -> softmax (tcgen05.ld S, scale by
//   ks_j, causal mask, online max/sum, P=bf16(p*vs) tcgen05.st) -> bar
//   -> correction (wait pv(t-1); O *= alpha, skipped when alpha==1) -> bar
//   -> PV mma issue (A from TMEM P) + commit.
// Numerics: identical contract path to the mma.sync kernels (per-key k scale
// after QK in log2 domain, unrounded denom, bf16 P*vs, exact bf16 V).
#pragma once

#include "fmha.cuh"

namespace fmha {

constexpr int TCBM = 128;  // q rows per tc CTA

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
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst))),
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
DEVI void tc_commit(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))) : "memory");
}
DEVI void tc_wait(uint64_t* bar, int parity) {
    asm volatile("{\n.reg .pred p;\nTCW_%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra.uni TCW_%=;\n}\n"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))),
                    "r"(parity));
}
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
DEVI void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
DEVI void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;"); }

// smem matrix descriptor: swizzle<<61 | sbo>>4<<32 | lbo>>4<<16 | addr>>4.
// Probe-verified (tc_probe3/4): version bits unnecessary; K-major interleave
// uses sw=0,lbo=128,sbo=(K/8)*128; MN-major swizzle-128B uses sw=2,lbo=0 or
// n-atom stride, sbo=1024.
DEVI uint64_t tc_desc(const void* smem, int sw, int lbo, int sbo) {
    return ((uint64_t)sw << 61) | ((uint64_t)(sbo >> 4) << 32)
         | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16)
         | (uint64_t)(static_cast<uint32_t>(__cvta_generic_to_shared(smem)) >> 4);
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
    DEVI TcSmem(char* base) {
        q = reinterpret_cast<uint16_t*>(base);
        kc = q + TCBM * D;
        vc = kc + 2 * BNT * D;
        ksvs = reinterpret_cast<float*>(vc + 2 * BNT * D);
        alpha = ksvs + 4 * BNT;
        bars = reinterpret_cast<uint64_t*>(alpha + TCBM);
        tmem_base = reinterpret_cast<uint32_t*>(bars + 4);
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
    static DEVI long long km_off(int m, int k) {
        return (m >> 3) * ((D / 8) * 128) + (k >> 3) * 128 + (m & 7) * 16 + (k & 7) * 2;
    }
    static DEVI long long v_off(int krow, int n) {
        return (n >> 6) * (BNT * 128) + (krow >> 3) * 1024 + (krow & 7) * 128
             + ((((n & 63) >> 3) ^ (krow & 7)) * 16) + (n & 7) * 2;
    }
};


// f16x2_to_bf16x2_exact: defined in fmha.cuh (v3.2 moved; was duplicated here)

// tc-convert: staged registers -> canonical bf16 UMMA images (exact).
// K: interleave K-major (km_off); V: swizzle-128B MN-major (v_off).
template <int D, int BNT, int NT = THREADS, typename SM = TcSmem<D, BNT>>
DEVI void tc_convert(const ConvRegs<D, BNT, NT>& R, const SM& S,
                     int stage, int nvalid, const Params& P) {
    const int tid = threadIdx.x;
    const int row = tid / ConvRegs<D, BNT, NT>::TPR;
    const int part = tid % ConvRegs<D, BNT, NT>::TPR;
    char* dk = reinterpret_cast<char*>(S.k(stage));
    char* dv = reinterpret_cast<char*>(S.v(stage));
    constexpr int TPR = ConvRegs<D, BNT, NT>::TPR;   // v3.8: chunk-interleaved
    if (row < nvalid) {
        #pragma unroll
        for (int i = 0; i < ConvRegs<D, BNT, NT>::KU64; ++i) {
            uint2 raw = R.k[i];
#ifdef PF_F16QK
            // E4M3 -> F16 is EXACT (e4m3's 3 mantissa bits fit in f16's 10 and its
            // 2^-6..2^8 exponent range fits in f16's) and costs ONE instruction, where
            // E4M3 -> BF16 costs four because this ptxas has no direct form
            // (cvt.rn.bf16x2.e4m3x2 is rejected - see
            // results/conversion-and-noconv-20261022.md). The QK mma then needs f16 on
            // both sides, so Q is converted during staging below; this file's own
            // mma.sync path already runs QK in f16 for the same reason ("bf16 Q -> f16
            // and fp8 K -> f16 are both exact in-domain"). Products and f32
            // accumulation are unchanged, so scores must be bit-identical.
            uint4 o = make_uint4(cvt_e4m3x2((unsigned short)(raw.x & 0xffffu)),
                                 cvt_e4m3x2((unsigned short)(raw.x >> 16)),
                                 cvt_e4m3x2((unsigned short)(raw.y & 0xffffu)),
                                 cvt_e4m3x2((unsigned short)(raw.y >> 16)));
#else
            uint4 o = make_uint4(e4m3x2_to_bf16x2((unsigned short)(raw.x & 0xffffu)),
                                 e4m3x2_to_bf16x2((unsigned short)(raw.x >> 16)),
                                 e4m3x2_to_bf16x2((unsigned short)(raw.y & 0xffffu)),
                                 e4m3x2_to_bf16x2((unsigned short)(raw.y >> 16)));
#endif
            *reinterpret_cast<uint4*>(dk + SM::km_off(row, (part + TPR * i) * 8)) = o;
        }
        #pragma unroll
        for (int i = 0; i < ConvRegs<D, BNT, NT>::VU32; ++i) {
            unsigned raw = R.v[i];
#ifdef PF_F16PV
            // Same exact one-instruction E4M3 -> F16 for V. Unlike the QK side this
            // is NOT numerically neutral: it forces P to f16 as well, so p*v_scale is
            // rounded to a FINER grid than the reference's bf16. That deviates from the
            // reference by at most bf16's half-ulp (2^-8 relative) per term, and the
            // peer's own mma.sync path shipped exactly this rounding ("the kernel
            // rounds to the finer FP16 grid, inside the declared tolerance"). Measured,
            // not assumed: the A/B checks rel_l2 and the mismatch fraction per case.
            uint2 o2 = make_uint2(
                cvt_e4m3x2((unsigned short)(raw & 0xffffu)),
                cvt_e4m3x2((unsigned short)(raw >> 16)));
#else
            uint2 o2 = make_uint2(
                e4m3x2_to_bf16x2((unsigned short)(raw & 0xffffu)),
                e4m3x2_to_bf16x2((unsigned short)(raw >> 16)));
#endif
            *reinterpret_cast<uint2*>(dv + SM::v_off(row, (part + TPR * i) * 4)) = o2;
        }
        if (part == 0) {
            // v3.5: pre-scale ks by sm_scale*log2e here (same thread that
            // stores it; identical f32 rounding) — removes 64 FMULs/lane/tile
            // from the softmax hot loop.
            S.ks(stage)[row] = R.ks * P.sm_scale_log2e;
            S.vs(stage)[row] = R.vs;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < ConvRegs<D, BNT, NT>::KU64; ++i)
            *reinterpret_cast<uint4*>(dk + SM::km_off(row, (part + TPR * i) * 8)) = make_uint4(0, 0, 0, 0);
        #pragma unroll
        for (int i = 0; i < ConvRegs<D, BNT, NT>::VU32; ++i)
            *reinterpret_cast<uint2*>(dv + SM::v_off(row, (part + TPR * i) * 4)) = make_uint2(0, 0);
        if (part == 0) {
            S.ks(stage)[row] = 0.f;
            S.vs(stage)[row] = 0.f;
        }
    }
}

// ---- tcgen05 prefill kernel (D=128 first): M=128 q rows per CTA -----------
// PIPE=true (D=64): v2.3b deep pipeline — prologue convert+QK(0), then per
// tile: softmax(t) overlaps PV(t-1); QK(t+1) issues before correction.
// Measured: D64 1.14x vs v2, but D128 0.95x (TMEM contention / register
// pressure from batched corr) -> PIPE=false for D=128/256 keeps the exact
// engine-validated v2 sequential loop.
template <int D, int BNT, bool PIPE>
__global__ void __launch_bounds__(THREADS) fa_prefill_tc(const __grid_constant__ Params P) {
    extern __shared__ char smem_raw[];
    TcSmem<D, BNT> S(smem_raw);
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

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
                         :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&S.bars[i]))));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        tc_alloc(S.tmem_base, TcSmem<D, BNT>::ACOLS);
        tc_relinquish();
    }
    __syncthreads();
    const uint32_t tm = *S.tmem_base;

    // descriptors / idescs (probe-verified constants)
    const uint64_t da_q = tc_desc(S.q, 0, 128, (D / 8) * 128);
    // Operand type field of the instruction descriptor: 1 = bf16, 0 = f16. The
    // smem/TMEM images are 16 bits wide either way, so only the values and this field
    // change - the descriptors, swizzles and column offsets are untouched.
#ifdef PF_F16QK
    const uint32_t idesc_qk = tc_idesc(TCBM, BNT, 0, 0, 0);
#else
    const uint32_t idesc_qk = tc_idesc(TCBM, BNT, 0, 1, 1);
#endif
#ifdef PF_F16PV
    const uint32_t idesc_pv = tc_idesc(TCBM, D, 1, 0, 0);
#else
    const uint32_t idesc_pv = tc_idesc(TCBM, D, 1, 1, 1);
#endif
    int tg = 0;                      // GLOBAL kv-tile counter (stage/parity)
    for (int which = 0; ; ++which) {
    int b, q0, qlen, tile, h;
    if (P.persist) {
        if (which > 0) {
            if (tid == 0) S.tmem_base[1] = (unsigned)atomicAdd(P.counter, 1);
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
    // ---- Q staging (bf16 -> padded bf16 smem, zero rows >= nrows) ----
    {
        constexpr int PER_ROW = D / 8;
        char* qb = reinterpret_cast<char*>(S.q);
        for (int idx = tid; idx < TCBM * PER_ROW; idx += THREADS) {
            int row = idx / PER_ROW;
            int col8 = (idx % PER_ROW) * 8;
            uint4* d = reinterpret_cast<uint4*>(qb + TcSmem<D, BNT>::km_off(row, col8));
            if (row < nrows) {
                const uint16_t* src = P.q + (long long)(q0 + i0 + row) * P.q_out_stride
                                    + h * D + col8;
#ifdef PF_F16QK
                // bf16 -> f16 for the f16 QK operands: exact for in-domain Q (bf16's 7
                // stored mantissa bits fit in f16's 10, and f16's normal range covers
                // it). ptxas here rejects cvt.rn.f16x2.bf16x2, so each pair goes
                // through f32 at 3 instructions per 2 values. This runs once per CTA
                // over TCBM*D/8 uint4s, against D/4 conversion calls per thread per KV
                // tile, so it amortises over the whole KV walk - but a one-tile walk
                // pays it all, which is why the small-shape routing thresholds matter.
                const uint4 bv = *reinterpret_cast<const uint4*>(src);
                const unsigned* bp = reinterpret_cast<const unsigned*>(&bv);
                uint4 fv;
                unsigned* fp = reinterpret_cast<unsigned*>(&fv);
                #pragma unroll
                for (int t = 0; t < 4; ++t)
                    fp[t] = cvt_f16x2_f32(bf16_bits_to_f32(bp[t] & 0xffffu),
                                          bf16_bits_to_f32(bp[t] >> 16));
                *d = fv;
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
    const int kn_end = min(kvlen, kvlen - qlen + i0 + TCBM);

    ConvRegs<D, BNT> R;
    // v3.0: block-table entry one tile ahead (bt_n) so KV data LDGs issue
    // immediately; long_scoreboard was the top stall (0.60/inst, case0).
    int bt_n = kv_bt_next<BNT>(P, b, 0, kn_end);
    kv_prefetch<D, BNT>(R, P, kvh, 0, min(BNT, kn_end), bt_n);
    if (BNT < kn_end) bt_n = kv_bt_next<BNT>(P, b, BNT, kn_end);
    if constexpr (PIPE) {
        // prologue: tile 0 -> smem[0], issue QK(0); main loop keeps the
        // tensor pipe packed (see header comment).
        tc_convert<D, BNT>(R, S, 0, min(BNT, kn_end), P);
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
    }

    float m2 = NEG, l2 = 0.f;
    int tl = 0;                      // LOCAL tile index (PV accumulate flag)
    if constexpr (PIPE) {
        for (int kn = 0; kn < kn_end; kn += BNT, ++tl, ++tg) {
            const int s = tg & 1;
            const bool more = (kn + BNT < kn_end);
            // 1) prefetch raw tile t+1 (bt already resident in bt_n)
            if (more) {
                kv_prefetch<D, BNT>(R, P, kvh, kn + BNT, min(BNT, kn_end - kn - BNT), bt_n);
                bt_n = kv_bt_next<BNT>(P, b, kn + 2 * BNT, kn_end);
            }
            // 2) QK(t) done
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
            // v3.5: 4-way ILP — the serial 64-deep FMAX/FADD chains cost
            // ~256 clk each of pure latency (stalled_wait 0.59/inst).
            float mn0 = m2, mn1 = m2, mn2 = m2, mn3 = m2;
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
                // v3.7: vectorized scale loads (kss is 16B-aligned: ksvs
                // base is 512B-aligned, j*4 float = 16B steps).
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
            float alpha = exp2f(m2 - mn);
            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
            unsigned wp[BNT / 2];
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
                float p0 = exp2f(sc[j] - mn);
                float p1 = exp2f(sc[j + 1] - mn);
                float p2 = exp2f(sc[j + 2] - mn);
                float p3 = exp2f(sc[j + 3] - mn);
                sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
                sc[j] = p0; sc[j + 1] = p1;   // reuse sc storage for p
                sc[j + 2] = p2; sc[j + 3] = p3;
            }
            const float sum = (sum0 + sum1) + (sum2 + sum3);
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
                const float4 vs4 = *reinterpret_cast<const float4*>(vss + j);
#ifdef PF_F16PV
                wp[j >> 1] = cvt_f16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
                wp[(j >> 1) + 1] = cvt_f16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
#else
                wp[j >> 1] = cvt_bf16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
                wp[(j >> 1) + 1] = cvt_bf16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
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
            tc_fence_before();
            __syncthreads();
            tc_fence_after();

            // 4) wait PV(t-1): frees V/K smem[s^1], TMEM S[s^1] (P(t-1)) and O.
            if (tl > 0) tc_wait(&S.bars[2 + (s ^ 1)], ((tg - 1) >> 1) & 1);
            // 5) convert tile t+1 -> smem[s^1] (QK(t-1) long done; PV(t-1) waited)
            if (more) {
                tc_convert<D, BNT>(R, S, s ^ 1, min(BNT, kn_end - kn - BNT), P);
                tc_fence_async_smem();
                __syncthreads();
            }
            // 6) issue QK(t+1) BEFORE correction: tensor pipe runs QK(t+1) while
            //    threads rescale O below (disjoint TMEM: S[s^1] vs O region).
            if (more && tid == 0) {
                uint64_t dbk = tc_desc(S.k(s ^ 1), 0, 128, (D / 8) * 128);
                #pragma unroll
                for (int kk = 0; kk < D / 16; ++kk)
                    tc_mma_ss(tm + (s ^ 1) * BNT, da_q + kk * 16, dbk + kk * 16,
                              idesc_qk, kk > 0 ? 1u : 0u);
                tc_commit(&S.bars[s ^ 1]);
            }
            // 7) correction: O *= alpha. TMEM loads batched 2 segments per
            //    wait_ld (wait::ld covers all outstanding) to halve the serial
            //    latency chain.
            {
                bool need = __any_sync(0xffffffffu, alpha != 1.0f);
                if (need) {
                    uint32_t oaddr = tm + TcSmem<D, BNT>::SOFF + ((warp * 32) << 16);
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
            tc_fence_before();
            __syncthreads();
            tc_fence_after();

            // 8) PV(t): P(t) ready (sync at step-3 end), O corrected (sync above),
            //    V smem[s] intact until convert(t+2) (iter t+1 step 5, after the
            //    PV(t) wait). Pipe: QK(t+1) -> PV(t) packed.
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

    } else {
        for (int kn = 0; kn < kn_end; kn += BNT, ++tl, ++tg) {
            const int s = tg & 1;
            const int nvalid = min(BNT, kn_end - kn);
            tc_convert<D, BNT>(R, S, s, nvalid, P);
            if (kn + BNT < kn_end) {
                kv_prefetch<D, BNT>(R, P, kvh, kn + BNT, min(BNT, kn_end - kn - BNT), bt_n);
                bt_n = kv_bt_next<BNT>(P, b, kn + 2 * BNT, kn_end);
            }
            tc_fence_async_smem();
            __syncthreads();

            // ---- QK mma (elected thread) ----
            if (tid == 0) {
                uint64_t db = tc_desc(S.k(s), 0, 128, (D / 8) * 128);
                #pragma unroll
                for (int kk = 0; kk < D / 16; ++kk)
                    tc_mma_ss(tm + s * BNT, da_q + kk * 16, db + kk * 16, idesc_qk,
                              kk > 0 ? 1u : 0u);
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
            // v3.5: 4-way ILP — the serial 64-deep FMAX/FADD chains cost
            // ~256 clk each of pure latency (stalled_wait 0.59/inst).
            float mn0 = m2, mn1 = m2, mn2 = m2, mn3 = m2;
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
                // v3.7: vectorized scale loads (kss is 16B-aligned: ksvs
                // base is 512B-aligned, j*4 float = 16B steps).
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
            float alpha = exp2f(m2 - mn);
            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
            unsigned wp[BNT / 2];
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
                float p0 = exp2f(sc[j] - mn);
                float p1 = exp2f(sc[j + 1] - mn);
                float p2 = exp2f(sc[j + 2] - mn);
                float p3 = exp2f(sc[j + 3] - mn);
                sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
                sc[j] = p0; sc[j + 1] = p1;   // reuse sc storage for p
                sc[j + 2] = p2; sc[j + 3] = p3;
            }
            const float sum = (sum0 + sum1) + (sum2 + sum3);
            #pragma unroll
            for (int j = 0; j < BNT; j += 4) {
                const float4 vs4 = *reinterpret_cast<const float4*>(vss + j);
#ifdef PF_F16PV
                wp[j >> 1] = cvt_f16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
                wp[(j >> 1) + 1] = cvt_f16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
#else
                wp[j >> 1] = cvt_bf16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
                wp[(j >> 1) + 1] = cvt_bf16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
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
            tc_fence_before();
            __syncthreads();
            tc_fence_after();


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
    tc_wait(&S.bars[2 + ((tg - 1) & 1)], ((tg - 1) >> 1) & 1);
    tc_fence_after();
    {
        float inv = 1.f / l2;
        uint16_t* orow = P.out + (long long)(q0 + i0 + row) * P.q_out_stride + h * D;
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
}


// ================= RT2: dual-row-tile tc prefill (D=128) ==================
// 8 warps / 256 q-rows per CTA: warp groups g=0/1 (warps 0-3 / 4-7) own
// 128-row tiles A/B. K/V convert+prefetch are SHARED between the two row
// tiles (per-row instruction cost of the memory path halves — kernel is
// instruction-issue bound). TMEM: S_A/S_B (2 stages x BNT cols each) +
// O_A/O_B = 4*BNT + 2*D = 512 cols exactly -> 1 CTA/SM with 8 warps
// (same warps/SM as 2 CTAs x 4 today). Sequential (v2-proven) loop order.
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
    static constexpr int SOFF = 4 * BNT;        // O_A col base (S uses 4*BNT)
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
__global__ void __launch_bounds__(256) fa_prefill_tc_rt2(const __grid_constant__ Params P) {
    constexpr int NT = 256;
    constexpr int RT = 2 * TCBM;               // 256 q rows per CTA
    extern __shared__ char smem_raw[];
    TcSmem2<D, BNT> S(smem_raw);
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int grp = warp >> 2;                 // row-tile group (0/1)
    const int w3 = warp & 3;                   // warp within group

    // ---- locate (batch, 256-row tile) ----
    int gx = blockIdx.x;
    int b = -1, tile_local = 0, base = 0, q0 = 0, qlen = 0;
    {
        int prev = 0;
        for (int i = 0; i < P.batch; ++i) {
            int c = __ldg(P.cu + i + 1);
            int ntiles = (c - prev + RT - 1) / RT;
            if (gx < base + ntiles) {
                b = i; tile_local = gx - base; q0 = prev; qlen = c - prev; break;
            }
            base += ntiles;
            prev = c;
        }
    }
    if (b < 0) return;
    const int i0 = tile_local * RT;
    if (i0 >= qlen) return;
    const int nrows = min(RT, qlen - i0);
    const int h = blockIdx.y;
    const int kvh = h / P.g;
    const int kvlen = __ldg(P.seqk + b);

    // ---- barriers + tmem ----
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&S.bars[i]))));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        tc_alloc(S.tmem_base, TcSmem2<D, BNT>::ACOLS);
        tc_relinquish();
    }
    __syncthreads();
    const uint32_t tm = *S.tmem_base;

    // ---- Q staging: 256 rows, image g at q + g*TCBM*D ----
    {
        constexpr int PER_ROW = D / 8;
        char* qb = reinterpret_cast<char*>(S.q);
        for (int idx = tid; idx < RT * PER_ROW; idx += NT) {
            int row = idx / PER_ROW;
            int col8 = (idx % PER_ROW) * 8;
            int g = row >> 7;
            uint4* d = reinterpret_cast<uint4*>(qb + g * (TCBM * D * 2)
                            + TcSmem2<D, BNT>::km_off(row & (TCBM - 1), col8));
            if (row < nrows) {
                const uint16_t* src = P.q + (long long)(q0 + i0 + row) * P.q_out_stride
                                    + h * D + col8;
                *d = *reinterpret_cast<const uint4*>(src);
            } else {
                *d = make_uint4(0, 0, 0, 0);
            }
        }
    }

    const int grow = grp * TCBM + w3 * 32 + lane;      // global row 0..255
    const int visible = (grow < nrows) ? (kvlen - qlen + i0 + grow) : 0;
    const int kn_end = min(kvlen, kvlen - qlen + i0 + RT);

    const uint64_t da_qA = tc_desc(S.q, 0, 128, (D / 8) * 128);
    const uint64_t da_qB = tc_desc(S.q + TCBM * D, 0, 128, (D / 8) * 128);
    const uint32_t idesc_qk = tc_idesc(TCBM, BNT, 0, 1, 1);
    const uint32_t idesc_pv = tc_idesc(TCBM, D, 1, 1, 1);
    const uint32_t socl = grp * (2 * BNT);             // this group's S base col
    const uint32_t oocl = TcSmem2<D, BNT>::SOFF + grp * D;  // this group's O base

    ConvRegs<D, BNT, NT> R;
    int bt_n = kv_bt_next<BNT, NT>(P, b, 0, kn_end);
    kv_prefetch<D, BNT, NT>(R, P, kvh, 0, min(BNT, kn_end), bt_n);

    float m2 = NEG, l2 = 0.f;
    int t = 0;
    for (int kn = 0; kn < kn_end; kn += BNT, ++t) {
        const int s = t & 1;
        const int nvalid = min(BNT, kn_end - kn);
        tc_convert<D, BNT, NT, TcSmem2<D, BNT>>(R, S, s, nvalid, P);
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
        tc_wait(&S.bars[s], (t >> 1) & 1);

        // ---- softmax (per group; rows w3*32+lane of the group's S region) ----
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
        float mn0 = m2, mn1 = m2, mn2 = m2, mn3 = m2;
        // v4.11: warp-uniform mask-free fast path. Lane 0 owns the warp's
        // smallest visible (rows ascend with lane); if the tile's last key
        // is <= that, no element masks -> skip 128 ISETP+SEL per tile.
        // Inactive rows (visible=0) force the slow path (correct).
        const bool nomask = (kn + BNT - 1 <= __shfl_sync(0xffffffffu, visible, 0));
        if (nomask) {
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
        } else {
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
        }
        const float mn = fmaxf(fmaxf(mn0, mn1), fmaxf(mn2, mn3));
        float alpha = exp2f(m2 - mn);
        float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
        unsigned wp[BNT / 2];
        #pragma unroll
        for (int j = 0; j < BNT; j += 4) {
            float p0 = exp2f(sc[j] - mn);
            float p1 = exp2f(sc[j + 1] - mn);
            float p2 = exp2f(sc[j + 2] - mn);
            float p3 = exp2f(sc[j + 3] - mn);
            sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
            sc[j] = p0; sc[j + 1] = p1; sc[j + 2] = p2; sc[j + 3] = p3;
        }
        const float sum = (sum0 + sum1) + (sum2 + sum3);
        #pragma unroll
        for (int j = 0; j < BNT; j += 4) {
            const float4 vs4 = *reinterpret_cast<const float4*>(vss + j);
            wp[j >> 1] = cvt_bf16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
            wp[(j >> 1) + 1] = cvt_bf16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
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

        // ---- correction per group (needs PV(t-1) done) ----
        if (t > 0) tc_wait(&S.bars[2 + (s ^ 1)], ((t - 1) >> 1) & 1);
        {
            bool need = __any_sync(0xffffffffu, alpha != 1.0f);
            if (need) {
                uint32_t oaddr = tm + oocl + ((w3 * 32) << 16);
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
        tc_fence_before();
        __syncthreads();
        tc_fence_after();

        // ---- PV x2 (shared V image; A = each group's P; O = each group) ----
        if (tid == 0) {
            uint64_t db = tc_desc(S.v(s), 2, BNT * 128, 1024);
            const int adv = 128;
            #pragma unroll
            for (int kk = 0; kk < BNT / 16; ++kk) {
                tc_mma_ts(tm + TcSmem2<D, BNT>::SOFF, tm + 0 * (2 * BNT) + s * BNT + kk * 8,
                          db + kk * adv, idesc_pv, (t > 0 || kk > 0) ? 1u : 0u);
                tc_mma_ts(tm + TcSmem2<D, BNT>::SOFF + D, tm + 1 * (2 * BNT) + s * BNT + kk * 8,
                          db + kk * adv, idesc_pv, (t > 0 || kk > 0) ? 1u : 0u);
            }
            tc_commit(&S.bars[2 + s]);
        }
    }

    // ---- epilogue (per group) ----
    tc_wait(&S.bars[2 + ((t - 1) & 1)], ((t - 1) >> 1) & 1);
    tc_fence_after();
    {
        float inv = 1.f / l2;
        uint16_t* orow = P.out + (long long)(q0 + i0 + grow) * P.q_out_stride + h * D;
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
    __syncthreads();
    if (warp == 0) tc_dealloc(tm, TcSmem2<D, BNT>::ACOLS);
}


// ====== v4.4: raw-staged tc prefill (D=128) — padded mirrors, rowbase ======
// History: v4.2/v4.3 were 3x SLOW despite fixing global sectors — ncu showed
// 100M shared-load requests at 3.6-way conflict (237M wasted wavefronts):
// dense 128B raw rows alias every bank (128B == 32 words == 0 mod 32).
// v4.4: padded raw rows (D+8 = 136B -> +2 banks/row -> 16 rows distinct);
// separate K/V/scale mirrors staged in 3 power-of-2 chunk passes (K 8B
// chunks: rows always 8-aligned since content=2d+8; V 4B chunks: bases are
// 4-mod-8 so 4B is the max, but 32 lanes now cover one row's full 128B =
// 4 full sectors vs 12.5% before); per-row base table (rowbase, 64 ull x2
// streams, double-buffered) replaces the pb/run machinery — built from ONE
// bt LDG per row issued after the QK mma, published by the softmax-end
// sync. K canonical single-stage (QK(t) done before convert-K(t+1)); V
// canonical two-stage. Staging runs post-softmax (~half-tile lookahead,
// cp.async so no register payload). smem ~103KB -> 2 CTAs/SM kept.
template <int D, int BNT>
struct TcSmemR {
    static constexpr int RR = D + 8;             // padded raw row bytes
    uint16_t* q; uint16_t* kc; uint16_t* vc; float* ksvs;
    unsigned char* rawk; unsigned char* rawv; unsigned long long* smir;
    unsigned long long* rowb;                    // 2 slots x 2 streams x BNT
    uint64_t* bars; uint32_t* tmem_base;
    DEVI TcSmemR(char* base) {
        q = reinterpret_cast<uint16_t*>(base);
        kc = q + TCBM * D;
        vc = kc + BNT * D;
        ksvs = reinterpret_cast<float*>(vc + 2 * BNT * D);
        rawk = reinterpret_cast<unsigned char*>(ksvs + 4 * BNT);
        rawv = rawk + BNT * RR;
        smir = reinterpret_cast<unsigned long long*>(rawv + BNT * RR);
        rowb = smir + BNT;                       // 8B per row: [ks|vs] bits
        bars = reinterpret_cast<uint64_t*>(rowb + 4 * BNT);
        tmem_base = reinterpret_cast<uint32_t*>(bars + 4);
    }
    DEVI uint16_t* k() const { return kc; }
    DEVI uint16_t* v(int s) const { return vc + s * BNT * D; }
    DEVI float* ks(int s) const { return ksvs + s * 2 * BNT; }
    DEVI float* vs(int s) const { return ksvs + s * 2 * BNT + BNT; }
    DEVI unsigned long long* rowb_s(int s) const { return rowb + s * 2 * BNT; }
    DEVI unsigned char* smir_u8() const { return reinterpret_cast<unsigned char*>(smir); }
    static constexpr int bytes() {
        return (TCBM * D + 3 * BNT * D) * 2 + 4 * BNT * 4 + 2 * BNT * RR
             + BNT * 8 + 4 * BNT * 8 + 4 * 8 + 8 + 128;
    }
    static constexpr int SOFF = 2 * BNT;
    static constexpr int ACOLS = (2 * BNT + D) <= 256 ? 256 : 512;
    static_assert(2 * BNT + D <= ACOLS, "TMEM overflow");
    static DEVI long long km_off(int m, int k) {
        return (m >> 3) * ((D / 8) * 128) + (k >> 3) * 128 + (m & 7) * 16 + (k & 7) * 2;
    }
    static DEVI long long v_off(int krow, int n) {
        return (n >> 6) * (BNT * 128) + (krow >> 3) * 1024 + (krow & 7) * 128
             + ((((n & 63) >> 3) ^ (krow & 7)) * 16) + (n & 7) * 2;
    }
};

template <int D, int BNT>
__global__ void __launch_bounds__(THREADS) fa_prefill_tc_r(const __grid_constant__ Params P) {
    extern __shared__ char smem_raw[];
    TcSmemR<D, BNT> S(smem_raw);
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    // ---- locate (batch, 128-row tile) ----
    int gx = blockIdx.x;
    int b = -1, tile_local = 0, base = 0, q0 = 0, qlen = 0;
    {
        int prev = 0;
        for (int i = 0; i < P.batch; ++i) {
            int c = __ldg(P.cu + i + 1);
            int ntiles = (c - prev + TCBM - 1) / TCBM;
            if (gx < base + ntiles) {
                b = i; tile_local = gx - base; q0 = prev; qlen = c - prev; break;
            }
            base += ntiles;
            prev = c;
        }
    }
    if (b < 0) return;
    const int i0 = tile_local * TCBM;
    if (i0 >= qlen) return;
    const int nrows = min(TCBM, qlen - i0);
    const int h = blockIdx.y;
    const int kvh = h / P.g;
    const int kvlen = __ldg(P.seqk + b);

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&S.bars[i]))));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        tc_alloc(S.tmem_base, TcSmemR<D, BNT>::ACOLS);
        tc_relinquish();
    }
    __syncthreads();
    const uint32_t tm = *S.tmem_base;

    // ---- Q staging ----
    {
        constexpr int PER_ROW = D / 8;
        char* qb = reinterpret_cast<char*>(S.q);
        for (int idx = tid; idx < TCBM * PER_ROW; idx += THREADS) {
            int row = idx / PER_ROW;
            int col8 = (idx % PER_ROW) * 8;
            uint4* d = reinterpret_cast<uint4*>(qb + TcSmemR<D, BNT>::km_off(row, col8));
            if (row < nrows) {
                const uint16_t* src = P.q + (long long)(q0 + i0 + row) * P.q_out_stride
                                    + h * D + col8;
                *d = *reinterpret_cast<const uint4*>(src);
            } else {
                *d = make_uint4(0, 0, 0, 0);
            }
        }
    }

    const int row = warp * 32 + lane;
    const int visible = (row < nrows) ? (kvlen - qlen + i0 + row) : 0;
    const int kn_end = min(kvlen, kvlen - qlen + i0 + TCBM);

    const uint64_t da_q = tc_desc(S.q, 0, 128, (D / 8) * 128);
    const uint32_t idesc_qk = tc_idesc(TCBM, BNT, 0, 1, 1);
    const uint32_t idesc_pv = tc_idesc(TCBM, D, 1, 1, 1);
    constexpr int TPR = THREADS / BNT;      // 2
    constexpr int RR = TcSmemR<D, BNT>::RR;
    const int pmask = (1 << P.page_shift) - 1;

    // ---- prologue: rowbase(0) + scales(0) LDG, stage raw(0) ----
    int blk_r = 0;    // pending bt for rowbase(t+1) build (tid<BNT)
    {
        const int nv0 = min(BNT, kn_end);
        if (tid < BNT && tid < nv0) {
            const int key = tid;
            blk_r = __ldg(P.block_table + b * P.bt_stride + (key >> P.page_shift));
        }
        __syncthreads();
        if (tid < BNT && tid < nv0) {
            const long long off = (long long)(tid & pmask);
            unsigned long long* rb = S.rowb_s(0);
            rb[tid] = (unsigned long long)(P.k + (long long)blk_r * P.k_sb + off * P.k_sp + kvh * P.k_sh);
            rb[BNT + tid] = (unsigned long long)(P.v + (long long)blk_r * P.v_sb + off * P.v_sp + kvh * P.v_sh);
        }
        __syncthreads();
        // scales(0): staged via smir together with K/V below
    }
    // stage raw(0): K chunks (8B), V chunks (4B), scales (2x4B per row)
    {
        const int nv0 = min(BNT, kn_end);
        const unsigned long long* rb = S.rowb_s(0);
        constexpr int CK = BNT * (D / 8);
        for (int c = tid; c < CK; c += THREADS) {
            const int r = c >> 4, o = (c & 15) * 8;          // D=128: 16 chunks/row
            if (r < nv0)
                cp_async8(S.rawk + r * RR + o, reinterpret_cast<const void*>(rb[r] + o));
        }
        constexpr int CV = BNT * (D / 4);
        for (int c = tid; c < CV; c += THREADS) {
            const int r = c >> 5, o = (c & 31) * 4;          // 32 chunks/row
            if (r < nv0)
                cp_async4(S.rawv + r * RR + o, reinterpret_cast<const void*>(rb[BNT + r] + o));
        }
        if (tid < BNT && tid < nv0) {
            // scales live inline in the cache row: ks at +D, vs at +2D+4
            const unsigned long long krow = rb[tid];
            cp_async4(S.smir_u8() + tid * 8 + 0, reinterpret_cast<const void*>(krow + D));
            cp_async4(S.smir_u8() + tid * 8 + 4, reinterpret_cast<const void*>(krow + 2 * D + 4));
        }
        cp_async_commit();
    }

    float m2 = NEG, l2 = 0.f;
    int t = 0;
    for (int kn = 0; kn < kn_end; kn += BNT, ++t) {
        const int s = t & 1;
        const int nvalid = min(BNT, kn_end - kn);
        const bool more = (kn + BNT < kn_end);
        // 1) raw(t) landed
        cp_async_wait<0>();
        __syncthreads();
        // 2) convert K(t)+V(t)+scales(t) from padded mirrors
        {
            const int rrow = tid / TPR;
            const int part = tid % TPR;
            constexpr int KU64 = D / (8 * TPR);
            constexpr int VU32 = D / (4 * TPR);
            char* dk = reinterpret_cast<char*>(S.k());
            char* dv = reinterpret_cast<char*>(S.v(s));
            if (rrow < nvalid) {
                const unsigned char* srk = S.rawk + rrow * RR;   // linear mirror
                #pragma unroll
                for (int i = 0; i < KU64; ++i) {
                    uint2 raw = *reinterpret_cast<const uint2*>(srk + (part + TPR * i) * 8);
                    uint4 o = make_uint4(e4m3x2_to_bf16x2((unsigned short)(raw.x & 0xffffu)),
                                         e4m3x2_to_bf16x2((unsigned short)(raw.x >> 16)),
                                         e4m3x2_to_bf16x2((unsigned short)(raw.y & 0xffffu)),
                                         e4m3x2_to_bf16x2((unsigned short)(raw.y >> 16)));
                    *reinterpret_cast<uint4*>(dk + TcSmemR<D, BNT>::km_off(rrow, (part + TPR * i) * 8)) = o;
                }
                const unsigned char* srv = S.rawv + rrow * RR;   // linear mirror
                #pragma unroll
                for (int i = 0; i < VU32; ++i) {
                    unsigned raw = *reinterpret_cast<const unsigned*>(srv + (part + TPR * i) * 4);
                    *reinterpret_cast<uint2*>(dv + TcSmemR<D, BNT>::v_off(rrow, (part + TPR * i) * 4)) = make_uint2(
                        e4m3x2_to_bf16x2((unsigned short)(raw & 0xffffu)),
                        e4m3x2_to_bf16x2((unsigned short)(raw >> 16)));
                }
            } else {
                #pragma unroll
                for (int i = 0; i < KU64; ++i)
                    *reinterpret_cast<uint4*>(dk + TcSmemR<D, BNT>::km_off(rrow, (part + TPR * i) * 8)) = make_uint4(0, 0, 0, 0);
                #pragma unroll
                for (int i = 0; i < VU32; ++i)
                    *reinterpret_cast<uint2*>(dv + TcSmemR<D, BNT>::v_off(rrow, (part + TPR * i) * 4)) = make_uint2(0, 0);
            }
            if (part == 0) {
                // scales from smir (staged with the row): ks pre-scaled
                float ks, vs;
                if (rrow < nvalid) {
                    ks = *reinterpret_cast<const float*>(S.smir_u8() + rrow * 8);
                    vs = *reinterpret_cast<const float*>(S.smir_u8() + rrow * 8 + 4);
                } else { ks = 0.f; vs = 0.f; }
                S.ks(s)[rrow] = ks * P.sm_scale_log2e;
                S.vs(s)[rrow] = vs;
            }
        }
        tc_fence_async_smem();
        __syncthreads();
        // 4) QK(t) issue; bt(t+1) LDG issue (lands during softmax)
        if (tid == 0) {
            uint64_t db = tc_desc(S.k(), 0, 128, (D / 8) * 128);
            #pragma unroll
            for (int kk = 0; kk < D / 16; ++kk)
                tc_mma_ss(tm + s * BNT, da_q + kk * 16, db + kk * 16, idesc_qk,
                          kk > 0 ? 1u : 0u);
            tc_commit(&S.bars[s]);
        }
        if (more && tid < BNT && kn + BNT + tid < kn_end)
            blk_r = __ldg(P.block_table + b * P.bt_stride + ((kn + BNT + tid) >> P.page_shift));
        // 5) wait QK(t)
        tc_wait(&S.bars[s], (t >> 1) & 1);
        // 6) softmax(t)
        float sc[BNT];
        {
            uint32_t saddr = tm + s * BNT + ((warp * 32) << 16);
            #pragma unroll
            for (int seg = 0; seg < BNT / 32; ++seg)
                tc_ld_32(sc + seg * 32, saddr + seg * 32);
            tc_wait_ld();
        }
        const float* kss = S.ks(s);
        const float* vss = S.vs(s);
        float mn0 = m2, mn1 = m2, mn2 = m2, mn3 = m2;
        // v4.11: warp-uniform mask-free fast path. Lane 0 owns the warp's
        // smallest visible (rows ascend with lane); if the tile's last key
        // is <= that, no element masks -> skip 128 ISETP+SEL per tile.
        // Inactive rows (visible=0) force the slow path (correct).
        const bool nomask = (kn + BNT - 1 <= __shfl_sync(0xffffffffu, visible, 0));
        if (nomask) {
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
        } else {
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
        }
        const float mn = fmaxf(fmaxf(mn0, mn1), fmaxf(mn2, mn3));
        float alpha = exp2f(m2 - mn);
        float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
        unsigned wp[BNT / 2];
        #pragma unroll
        for (int j = 0; j < BNT; j += 4) {
            float p0 = exp2f(sc[j] - mn);
            float p1 = exp2f(sc[j + 1] - mn);
            float p2 = exp2f(sc[j + 2] - mn);
            float p3 = exp2f(sc[j + 3] - mn);
            sum0 += p0; sum1 += p1; sum2 += p2; sum3 += p3;
            sc[j] = p0; sc[j + 1] = p1; sc[j + 2] = p2; sc[j + 3] = p3;
        }
        const float sum = (sum0 + sum1) + (sum2 + sum3);
        #pragma unroll
        for (int j = 0; j < BNT; j += 4) {
            const float4 vs4 = *reinterpret_cast<const float4*>(vss + j);
            wp[j >> 1] = cvt_bf16x2_f32(sc[j] * vs4.x, sc[j + 1] * vs4.y);
            wp[(j >> 1) + 1] = cvt_bf16x2_f32(sc[j + 2] * vs4.z, sc[j + 3] * vs4.w);
        }
        m2 = mn;
        l2 = l2 * alpha + sum;
        {
            uint32_t paddr = tm + s * BNT + ((warp * 32) << 16);
            #pragma unroll
            for (int seg = 0; seg < BNT / 64; ++seg)
                tc_st_32(paddr + seg * 32, wp + seg * 32);
            tc_wait_st();
        }
        tc_fence_before();
        __syncthreads();
        tc_fence_after();
        // 7) rowbase(t+1) publish + stage raw(t+1)
        if (more) {
            if (tid < BNT && kn + BNT + tid < kn_end) {
                const int key = kn + BNT + tid;
                const long long off = (long long)(key & pmask);
                unsigned long long* rb = S.rowb_s(s ^ 1);
                rb[tid] = (unsigned long long)(P.k + (long long)blk_r * P.k_sb + off * P.k_sp + kvh * P.k_sh);
                rb[BNT + tid] = (unsigned long long)(P.v + (long long)blk_r * P.v_sb + off * P.v_sp + kvh * P.v_sh);
            }
            __syncthreads();
            const int nv1 = min(BNT, kn_end - kn - BNT);
            const unsigned long long* rb = S.rowb_s(s ^ 1);
            constexpr int CK = BNT * (D / 8);
            for (int c = tid; c < CK; c += THREADS) {
                const int r = c >> 4, o = (c & 15) * 8;
                if (r < nv1)
                    cp_async8(S.rawk + r * RR + o, reinterpret_cast<const void*>(rb[r] + o));
            }
            constexpr int CV = BNT * (D / 4);
            for (int c = tid; c < CV; c += THREADS) {
                const int r = c >> 5, o = (c & 31) * 4;
                if (r < nv1)
                    cp_async4(S.rawv + r * RR + o, reinterpret_cast<const void*>(rb[BNT + r] + o));
            }
            if (tid < BNT && tid < nv1) {
                const unsigned long long krow = rb[tid];
                cp_async4(S.smir_u8() + tid * 8 + 0, reinterpret_cast<const void*>(krow + D));
                cp_async4(S.smir_u8() + tid * 8 + 4, reinterpret_cast<const void*>(krow + 2 * D + 4));
            }
        }
        cp_async_commit();
        // 8) PV(t-1) done; correction
        if (t > 0) tc_wait(&S.bars[2 + (s ^ 1)], ((t - 1) >> 1) & 1);
        {
            bool need = __any_sync(0xffffffffu, alpha != 1.0f);
            if (need) {
                uint32_t oaddr = tm + TcSmemR<D, BNT>::SOFF + ((warp * 32) << 16);
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
        tc_fence_before();
        __syncthreads();
        tc_fence_after();
        // 9) PV(t)
        if (tid == 0) {
            uint64_t db = tc_desc(S.v(s), 2, BNT * 128, 1024);
            const int adv = 128;
            #pragma unroll
            for (int kk = 0; kk < BNT / 16; ++kk)
                tc_mma_ts(tm + TcSmemR<D, BNT>::SOFF, tm + s * BNT + kk * 8, db + kk * adv, idesc_pv,
                          (t > 0 || kk > 0) ? 1u : 0u);
            tc_commit(&S.bars[2 + s]);
        }
    }

    // ---- epilogue ----
    tc_wait(&S.bars[2 + ((t - 1) & 1)], ((t - 1) >> 1) & 1);
    tc_fence_after();
    {
        float inv = 1.f / l2;
        uint16_t* orow = P.out + (long long)(q0 + i0 + row) * P.q_out_stride + h * D;
        uint32_t oaddr = tm + TcSmemR<D, BNT>::SOFF + ((warp * 32) << 16);
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
    __syncthreads();
    if (warp == 0) tc_dealloc(tm, TcSmemR<D, BNT>::ACOLS);
}

#define DEFINE_TCR_LAUNCHER(D, BNV)                                             \
    void launch_prefill_tcr_##D(const fmha::Params& P, int Tq, int smem, cudaStream_t st) { \
        fmha::set_smem(reinterpret_cast<const void*>(fmha::fa_prefill_tc_r<D, BNV>), smem); \
        fmha::fa_prefill_tc_r<D, BNV><<<dim3((Tq + fmha::TCBM - 1) / fmha::TCBM + P.batch, P.heads), \
                                        fmha::THREADS, smem, st>>>(P);          \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
    }
#define DECLARE_TCR_LAUNCHER(D) \
    void launch_prefill_tcr_##D(const fmha::Params& P, int Tq, int smem, cudaStream_t st);

#define DEFINE_TC2_LAUNCHER(D, BNV)                                             \
    void launch_prefill_tc2_##D(const fmha::Params& P, int Tq, int smem, cudaStream_t st) { \
        fmha::set_smem(reinterpret_cast<const void*>(fmha::fa_prefill_tc_rt2<D, BNV>), smem); \
        fmha::fa_prefill_tc_rt2<D, BNV><<<dim3((Tq + 2 * fmha::TCBM - 1) / (2 * fmha::TCBM) + P.batch, P.heads), \
                                           256, smem, st>>>(P);                 \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                         \
    }
#define DECLARE_TC2_LAUNCHER(D) \
    void launch_prefill_tc2_##D(const fmha::Params& P, int Tq, int smem, cudaStream_t st);

#define DEFINE_TC_LAUNCHER(D, BNV, PIPEF)                                       \
    void launch_prefill_tc_##D(const fmha::Params& P, int Tq, int smem, cudaStream_t st) { \
        fmha::set_smem(reinterpret_cast<const void*>(fmha::fa_prefill_tc<D, BNV, PIPEF>), smem); \
        const int trows = P.fold ? 2 * fmha::TCBM : fmha::TCBM;                 \
        const dim3 grid = P.persist ? dim3(P.nctas, 1)                          \
                          : dim3((Tq + trows - 1) / trows + P.batch, P.heads);  \
        fmha::fa_prefill_tc<D, BNV, PIPEF><<<grid, fmha::THREADS, smem, st>>>(P); \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    }

} // namespace fmha

#define DECLARE_TC_LAUNCHER(D) \
    void launch_prefill_tc_##D(const fmha::Params& P, int Tq, int smem, cudaStream_t st);

DECLARE_TC_LAUNCHER(64)
DECLARE_TC_LAUNCHER(128)
DECLARE_TC_LAUNCHER(256)
DECLARE_TC2_LAUNCHER(128)
DECLARE_TCR_LAUNCHER(128)
