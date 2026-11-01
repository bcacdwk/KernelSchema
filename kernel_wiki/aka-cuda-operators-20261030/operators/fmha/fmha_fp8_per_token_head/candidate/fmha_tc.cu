// ============================================================================
// tcgen05 (5th-generation tensor core) prefill mainloop for sm_103a.
//
// Appended to candidate/fmha.cu by candidate/solution.py, so Params, ex2,
// pack_bf16, fp8x2_to_bf16x2, cp_async*, MetaS and SmemLayout are all visible.
//
// Why: warp-level mma.sync peaks at 568.8 TFLOPS on this part and prefill stalls
// at 25-43% of it, because ldmatrix costs 192 shared bytes per mma and the
// 128-register O accumulator caps occupancy at 8 warps/SM (NCU: tensor pipe
// 41.66%, issue_active 45.11%, max occupancy 12.5%). tcgen05 reads both operands
// straight from shared memory via descriptors, keeps the accumulator in tensor
// memory, and is issued by a single thread - so there is no ldmatrix traffic, no
// O registers, and each thread ends up owning one whole row of S, which makes the
// online softmax completely thread-local (no shuffles at all).
//
// Descriptor / layout / tmem recipe validated empirically on this GPU by
// experiments/tc_probe.py (job dv_11e44abed648); see
// results/tcgen05-recipe-20261021.md. Mechanism references, read for the encoding
// only, with nothing linked or copied:
//   knowledge/nvidia/atrex-fa/src/kernels/sm100/fp8_e4m3/pure_cuda/d256/sm100_primitives.cuh
//   knowledge/nvidia/deepgemm/deep_gemm/include/deep_gemm/ptx/tcgen05.cuh
// ============================================================================

// Shared-memory arrangement of the PV B operand (V, MN-major). Values come from
// experiments/tc_probe2.py; TC_V_LAYOUT selects the index map used at stash time.
#ifndef TC_V_LAYOUT
#define TC_V_LAYOUT 2
#define TC_V_LBO 0
#define TC_V_SBO 128
#endif

// Core-matrix atom slot size, in elements and bytes. An 8x8 tile of BF16 is 64
// elements / 128 bytes, and 128 bytes is exactly 32 shared-memory words, so
// consecutive atoms along the contiguous index all land in the same bank. In
// stash_kv one warp writes the 16 dim-atoms of a single key, i.e. 16 addresses
// 32 words apart: an 8-way conflict on the K uint2 store and a 16-way conflict on
// each of the 64 2-byte V-transpose stores per thread per block. Padding the slot
// to 72 elements / 144 bytes makes those 16 atoms 36 words apart, so a 16-lane
// phase covers words {j*36 + 4q + t}, j=0..7, t=0..3 - all 32 banks exactly once.
// LBO and SBO are documented as multiples of 16 bytes, so 144 is encodable;
// experiments/tc_probe4.py measures whether the hardware honours it.
#ifdef FMHA_TC_PAD
#define TC_ATOM_E 72
#define TC_ATOM_B 144
#else
#define TC_ATOM_E 64
#define TC_ATOM_B 128
#endif
// One K=16 MMA step spans two atoms along the contiguous index.
#define TC_KSTEP_B (2 * TC_ATOM_B)

// Shared-memory budget for the tcgen05 kernel: Q, K and V^T tiles (tiled into 8x8
// core matrices) plus the tmem base address and one mbarrier.
template <int D, int BM, int BN>
struct TcLayout {
    static constexpr int META_B = ((int)sizeof(MetaS<BM, BN, 1>) + 15) & ~15;
    // 1 KiB alignment for every operand base: the descriptor recipe in
    // results/tcgen05-recipe-20261021.md was validated against __align__(1024)
    // static shared arrays, so do not hand the tensor core a base it was never
    // proven to accept.  Costs <= 3 KiB of shared memory per CTA.
    // (rows/8) * (cols/8) atom slots of TC_ATOM_E elements each.
    static constexpr int QB = (BM * D / 64) * TC_ATOM_E * 2;
    static constexpr int KB = (BN * D / 64) * TC_ATOM_E * 2;
    static constexpr int VB = (D * BN / 64) * TC_ATOM_E * 2;
    static constexpr int OFF_Q = (META_B + 1023) & ~1023;
    static constexpr int OFF_K = (OFF_Q + QB + 1023) & ~1023;
    static constexpr int OFF_VT = (OFF_K + KB + 1023) & ~1023;
    static constexpr int OFF_TM = (OFF_VT + VB + 15) & ~15;
    static constexpr int OFF_MBAR = OFF_TM + 16;
    static constexpr int BYTES = OFF_MBAR + 16;
};

// ---------------------------------------------------------------- PTX wrappers
DEVI uint32_t sa_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI uint64_t umma_desc(uint32_t addr, int lbo, int sbo, int swizzle) {
    return ((uint64_t)swizzle << 61) | (1ULL << 46) | (((uint64_t)sbo >> 4) << 32)
         | (((uint64_t)lbo >> 4) << 16) | ((uint64_t)(addr & 0xFFFFFFU) >> 4);
}
DEVI uint64_t desc_add(uint64_t d, int bytes) {
    return d + (uint64_t)(bytes >> 4);   // start_address field is addr>>4
}
// M and N as encoded in the validated instruction descriptor:
//   (M>>5)<<25 | (N>>3)<<17 | b_major<<16 | c_format(1=F32)<<4 | a_format<<7 | b_format<<10
// a_format = b_format = 1 selects BF16 for kind::f16; mixing 0/1 is an illegal
// instruction (measured), so both operands must be BF16.
DEVI constexpr uint32_t umma_idesc(int M, int N, bool b_major) {
    return ((uint32_t)(M >> 5) << 25) | ((uint32_t)(N >> 3) << 17)
         | ((uint32_t)b_major << 16) | (1u << 4) | (1u << 7) | (1u << 10);
}
DEVI void umma_ss(uint32_t dt, uint64_t ad, uint64_t bd, uint32_t idesc, bool acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                 :: "r"(dt), "l"(ad), "l"(bd), "r"(idesc), "r"((uint32_t)acc));
}
DEVI void umma_ts(uint32_t dt, uint32_t at, uint64_t bd, uint32_t idesc, bool acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                 :: "r"(dt), "r"(at), "l"(bd), "r"(idesc), "r"((uint32_t)acc));
}
DEVI void umma_commit(uint32_t mbar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
                 :: "r"(mbar));
}
// Non-blocking phase test. `mbarrier.try_wait.parity` is NOT a poll: it may
// suspend the thread for an implementation-defined duration, so any spin bound
// built on it is not a wall-clock bound. Job dv_3641782fd3db proved that - the
// %globaltimer deadline added in f5ac8ec never got a chance to be re-tested, the
// kernel still ran past a 300 s timeout, and only the pre-wait printf was ever
// flushed. `test_wait` returns immediately.
DEVI bool mbar_test(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    asm volatile("{\n.reg .pred P;\n"
                 "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2;\n"
                 "selp.b32 %0, 1, 0, P;\n}\n"
                 : "=r"(done) : "r"(mbar), "r"(phase));
    return done != 0;
}
// Bounded wait. An mbarrier that never flips would otherwise hang the GPU until
// the job budget expires; a spin-count bound is not enough because
// mbarrier.try_wait may suspend for an implementation-defined time, so bound it
// against %globaltimer instead. Job dv_b0a603cf4fb1: a 2e6-iteration bound still
// let case 1 run past a 400s timeout, i.e. the barrier really never flipped.
#ifndef FMHA_TC_TIMEOUT_NS
#ifdef FMHA_TC_DEBUG
#define FMHA_TC_TIMEOUT_NS 5000000ULL      // 5 ms
#else
#define FMHA_TC_TIMEOUT_NS 50000000ULL     // 50 ms
#endif
#endif
DEVI unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
DEVI bool mbar_wait(uint32_t mbar, uint32_t phase, int tag) {
    (void)tag;
    if (mbar_test(mbar, phase)) return true;          // common case: already done
    const unsigned long long t0 = gtimer();
    for (int i = 0;; ++i) {
        if (i > 256) __nanosleep(128);                // keep the spin off the LSU
        if (mbar_test(mbar, phase)) return true;
        if (gtimer() - t0 > FMHA_TC_TIMEOUT_NS) break;
    }
#ifdef FMHA_TC_DEBUG
    if (threadIdx.x == 0) {
        unsigned long long st = 0;
        asm volatile("ld.shared.u64 %0, [%1];\n" : "=l"(st) : "r"(mbar));
        printf("TC timeout blk=(%d,%d,%d) tag=%d phase=%u mbar=0x%016llx\n",
               (int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, tag, phase, st);
    }
#endif
    return false;
}
DEVI void mbar_init(uint32_t mbar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(mbar), "r"(count));
}
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n"); }
DEVI void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
DEVI void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n"); }
DEVI void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n"); }

#define TCGEN05_LD32(DST, ADDR)                                                            \
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "                                 \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"                          \
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"       \
        : "=f"(DST[0]),"=f"(DST[1]),"=f"(DST[2]),"=f"(DST[3]),"=f"(DST[4]),"=f"(DST[5]),   \
          "=f"(DST[6]),"=f"(DST[7]),"=f"(DST[8]),"=f"(DST[9]),"=f"(DST[10]),"=f"(DST[11]), \
          "=f"(DST[12]),"=f"(DST[13]),"=f"(DST[14]),"=f"(DST[15]),"=f"(DST[16]),           \
          "=f"(DST[17]),"=f"(DST[18]),"=f"(DST[19]),"=f"(DST[20]),"=f"(DST[21]),           \
          "=f"(DST[22]),"=f"(DST[23]),"=f"(DST[24]),"=f"(DST[25]),"=f"(DST[26]),           \
          "=f"(DST[27]),"=f"(DST[28]),"=f"(DST[29]),"=f"(DST[30]),"=f"(DST[31])            \
        : "r"(ADDR))

#define TCGEN05_ST32(ADDR, SRC)                                                            \
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "                          \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"                          \
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};\n"              \
        :: "f"(SRC[0]),"f"(SRC[1]),"f"(SRC[2]),"f"(SRC[3]),"f"(SRC[4]),"f"(SRC[5]),         \
           "f"(SRC[6]),"f"(SRC[7]),"f"(SRC[8]),"f"(SRC[9]),"f"(SRC[10]),"f"(SRC[11]),       \
           "f"(SRC[12]),"f"(SRC[13]),"f"(SRC[14]),"f"(SRC[15]),"f"(SRC[16]),                \
           "f"(SRC[17]),"f"(SRC[18]),"f"(SRC[19]),"f"(SRC[20]),"f"(SRC[21]),                \
           "f"(SRC[22]),"f"(SRC[23]),"f"(SRC[24]),"f"(SRC[25]),"f"(SRC[26]),                \
           "f"(SRC[27]),"f"(SRC[28]),"f"(SRC[29]),"f"(SRC[30]),"f"(SRC[31]), "r"(ADDR))

#define TCGEN05_ST8(ADDR, SRC)                                                             \
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], "                            \
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                            \
                 :: "r"(SRC[0]),"r"(SRC[1]),"r"(SRC[2]),"r"(SRC[3]),                        \
                    "r"(SRC[4]),"r"(SRC[5]),"r"(SRC[6]),"r"(SRC[7]), "r"(ADDR))

// K-major operand element (r, c) inside an 8x8-core-matrix tiled block.
DEVI int km_off(int r, int c, int atoms) {
    return (r >> 3) * (atoms * TC_ATOM_E) + (c >> 3) * TC_ATOM_E
         + ((r & 7) << 3) + (c & 7);
}
// MN-major operand element (k, n).
DEVI int mn_off(int k, int n) {
#if TC_V_LAYOUT == 0
    (void)k; (void)n; return 0;
#elif TC_V_LAYOUT == 1
    return (k >> 3) * 8 * 64 + (n >> 3) * 64 + (k & 7) * 8 + (n & 7);
#else
    return (n >> 3) * 2 * 64 + (k >> 3) * 64 + (n & 7) * 8 + (k & 7);
#endif
}

// ---------------------------------------------------------------- the kernel
// BM must be 128: tcgen05 with cta_group::1 has M in {64,128} and the four warps
// map onto the four 32-lane tmem groups, so warp w, lane l owns row 32w+l.
template <int D, int BM, int BN>
__global__ void __launch_bounds__(128) fmha_tc(const Params p) {
    constexpr int NT = 128;
    constexpr int KA = D / 8;                  // 8-element atoms along K for Q/K
    constexpr int VA = BN / 8;                 // atoms along K(=keys) for V^T
    constexpr int QK_SBO = KA * TC_ATOM_B, PV_SBO = VA * TC_ATOM_B;
    constexpr uint32_t ID_QK = ((BM >> 5) << 25) | ((BN >> 3) << 17) | (1u << 4) | (1u << 7) | (1u << 10);
    constexpr uint32_t ID_PV = ((BM >> 5) << 25) | ((D >> 3) << 17) | (1u << 4) | (1u << 7) | (1u << 10);
    constexpr int TC_O = 0, TC_S = D, TC_P = D + BN;
    constexpr int TCOLS = (D + BN + BN / 2 <= 256) ? 256 : 512;
    constexpr int CPR4 = D / 4;
    constexpr int CPT = BN * CPR4 / NT;        // 4-byte chunks per lane (K or V)
    constexpr int MPL = BN / 32;               // keys resolved per lane
    static_assert(BM == 128 && BN % 16 == 0 && D % 16 == 0 && CPT > 0, "tc tile");

    extern __shared__ __align__(1024) char smem[];
    MetaS<BM, BN, 1>* meta = reinterpret_cast<MetaS<BM, BN, 1>*>(smem);
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem + TcLayout<D, BM, BN>::OFF_Q);
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem + TcLayout<D, BM, BN>::OFF_K);
    __nv_bfloat16* sVT = reinterpret_cast<__nv_bfloat16*>(smem + TcLayout<D, BM, BN>::OFF_VT);
    uint32_t* s_tmem = reinterpret_cast<uint32_t*>(smem + TcLayout<D, BM, BN>::OFF_TM);
    uint64_t* s_mbar = reinterpret_cast<uint64_t*>(smem + TcLayout<D, BM, BN>::OFF_MBAR);

    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int row = w * 32 + lane;             // the tile row this thread owns
    const int tx = blockIdx.x, hy = blockIdx.y;

    for (int i = tid; i <= p.B; i += NT) meta->cu[i] = p.cu[i];
    __syncthreads();
    int b = -1, m_blk = 0, cum = 0;
    for (int i = 0; i < p.B; ++i) {
        int ql = meta->cu[i + 1] - meta->cu[i];
        int nt = (ql * p.g + BM - 1) / BM;
        if (b < 0 && tx < cum + nt) { b = i; m_blk = nt - 1 - (tx - cum); }
        cum += nt;
    }
    if (b < 0) return;
    const int q0 = meta->cu[b];
    const int qlen = meta->cu[b + 1] - q0;
    const int q_start = m_blk * (BM / p.g);
    const int n_tok = min(BM / p.g, qlen - q_start);
    const int n_rows = n_tok * p.g;
    const int kv_len = p.slen[b];
    const int hi = kv_len - qlen + q_start + n_tok;
    const int nblk = (hi + BN - 1) / BN;
    const int bound = kv_len - qlen + q_start + (row >> p.g_log2);

    if (tid < BM) {
        int tok = q_start + (tid >> p.g_log2);
        if (tok > qlen - 1) tok = qlen - 1;
        int h = hy * p.g + (tid & (p.g - 1));
        meta->qbase[tid] = (long long)(q0 + tok) * p.q_s0 + (long long)h * D;
    }
    // Every thread later reads qbase[r] for r it did not write, so the Q cp.async
    // must not start before those writes are visible. fmha_main has this barrier;
    // fmha_tc did not, which is the cudaErrorIllegalAddress that killed the
    // case 7 benchmark in job dv_840f14045d88 after the first call had passed.
    __syncthreads();
    if (tid == 0) mbar_init(sa_u32(s_mbar), 1);
    if (tid < 32) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                     :: "r"(sa_u32(s_tmem)), "r"((uint32_t)TCOLS));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
    // Q tile: already BF16, and an 8-element atom is 16 contiguous bytes in the
    // tiled layout, so cp.async still applies.
    {
        const char* qb = reinterpret_cast<const char*>(p.q);
        for (int c = tid; c < BM * (D / 8); c += NT) {
            int r = c / (D / 8), a = c % (D / 8);
            cp_async16(reinterpret_cast<char*>(sQ + km_off(r, a * 8, KA)),
                       qb + meta->qbase[r] * 2 + a * 16);
        }
        cp_commit();
    }
    cp_wait<0>();
    __syncthreads();
    const uint32_t tmem = *s_tmem;
    const uint32_t mbar = sa_u32(s_mbar);
    const uint64_t qd0 = umma_desc(sa_u32(sQ), TC_ATOM_B, QK_SBO, 0);
    uint32_t phase = 0;
#ifdef FMHA_TC_DEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0)
        printf("TC enter D=%d BM=%d BN=%d nblk=%d hi=%d n_rows=%d tmem=0x%x qd0=0x%llx "
               "smem=%d ID_QK=0x%x ID_PV=0x%x\n",
               D, BM, BN, nblk, hi, n_rows, tmem, (unsigned long long)qd0,
               (int)TcLayout<D, BM, BN>::BYTES, ID_QK, ID_PV);
#endif
    bool tc_ok = true;   // set false if an MMA barrier never flips

    // ---- metadata + K/V movement (register prefetch, as in fmha_main)
    struct MetaRegs { int phys[MPL], slot[MPL]; float ksv[MPL], vsv[MPL]; };
    MetaRegs mr;
    uint32_t rk[CPT], rv[CPT];
    auto issue_meta = [&](int ib) {
        const int n0 = ib * BN;
        const int* bt = p.btab + (long long)b * p.btab_s;
        const int psm = (1 << p.ps_log2) - 1;
#pragma unroll
        for (int m = 0; m < MPL; ++m) {
            const int n = lane + m * 32;
            const int kg = n0 + n;
            const int kgc = kg < hi ? kg : hi - 1;
            const int sl = kgc & psm;
            const int ph = bt[kgc >> p.ps_log2];
            mr.phys[m] = ph; mr.slot[m] = sl;
            mr.ksv[m] = p.ks[(long long)ph * p.ks_s0 + (long long)sl * p.ks_s1 + (long long)hy * p.ks_s2];
            mr.vsv[m] = p.vs[(long long)ph * p.vs_s0 + (long long)sl * p.vs_s1 + (long long)hy * p.vs_s2];
        }
    };
    auto commit_meta = [&](int slot) {
#pragma unroll
        for (int m = 0; m < MPL; ++m) {
            const int n = lane + m * 32;
            meta->koff[slot][0][n] = (long long)mr.phys[m] * p.k_s0 + (long long)mr.slot[m] * p.k_s1
                                   + (long long)hy * p.k_s2;
            meta->voff[slot][0][n] = (long long)mr.phys[m] * p.v_s0 + (long long)mr.slot[m] * p.v_s1
                                   + (long long)hy * p.v_s2;
            meta->kst[slot][0][n] = mr.ksv[m] * p.qk_scale;
            meta->vst[slot][0][n] = mr.vsv[m];
        }
        __syncwarp();
    };
    auto fetch_kv = [&](int ib, int slot) {
        const long long* ko = meta->koff[slot][0];
        const long long* vo = meta->voff[slot][0];
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            const int lc = c * NT + tid;
            const int key = lc / CPR4, col = (lc % CPR4) * 4;
            rk[c] = *reinterpret_cast<const uint32_t*>(p.k + ko[key] + col);
            rv[c] = *reinterpret_cast<const uint32_t*>(p.v + vo[key] + col);
        }
    };
    auto stash_kv = [&](int ib) {
        (void)ib;
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            const int lc = c * NT + tid;
            const int key = lc / CPR4, col = (lc % CPR4) * 4;
            const uint16_t* hk = reinterpret_cast<const uint16_t*>(&rk[c]);
            const uint16_t* hv = reinterpret_cast<const uint16_t*>(&rv[c]);
            uint2 kk;
            kk.x = fp8x2_to_bf16x2(hk[0]); kk.y = fp8x2_to_bf16x2(hk[1]);
            *reinterpret_cast<uint2*>(sK + km_off(key, col, KA)) = kk;
            // V is stored transposed so that the PV B operand is K-major, which
            // reuses the descriptor recipe validated for QK instead of needing the
            // MN-major encoding (which experiments/tc_probe2.py did not resolve).
            const uint32_t v01 = fp8x2_to_bf16x2(hv[0]);
            const uint32_t v23 = fp8x2_to_bf16x2(hv[1]);
            const uint16_t bits[4] = {(uint16_t)(v01 & 0xFFFFu), (uint16_t)(v01 >> 16),
                                      (uint16_t)(v23 & 0xFFFFu), (uint16_t)(v23 >> 16)};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint16_t u = bits[j];
                sVT[km_off(col + j, key, VA)] = *reinterpret_cast<const __nv_bfloat16*>(&u);
            }
        }
    };

    const auto taddr = [&](int col) { return tmem + (uint32_t)col + ((uint32_t)(w * 32) << 16); };
    float sv[BN];
    float rmax = -INFINITY, rsum = 0.f;

    issue_meta(0);
    commit_meta(0);
    fetch_kv(0, 0);
    if (nblk > 1) issue_meta(1);
    for (int ib = 0; ib < nblk; ++ib) {
        const int slot = ib & 1;
        stash_kv(ib);
        fence_async_shared();
        __syncthreads();
        tc_fence_before();
#ifdef FMHA_TC_DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && ib < 2)
            printf("TC ib=%d pre-qk\n", ib);
#endif
        if (tid == 0) {
            uint64_t qd = qd0, kd = umma_desc(sa_u32(sK), TC_ATOM_B, QK_SBO, 0);
#pragma unroll
            for (int kt = 0; kt < D / 16; ++kt) {
                umma_ss(tmem + TC_S, qd, kd, ID_QK, kt > 0);
                qd = desc_add(qd, TC_KSTEP_B); kd = desc_add(kd, TC_KSTEP_B);
            }
            umma_commit(mbar);
        }
        if (!mbar_wait(mbar, phase, ib * 2 + 0)) { tc_ok = false; }
        phase ^= 1;
        tc_fence_after();
#ifdef FMHA_TC_DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && ib < 2)
            printf("TC ib=%d post-qk ok=%d phase=%u\n", ib, (int)tc_ok, phase);
#endif

        if (ib + 1 < nblk) { commit_meta((ib + 1) & 1); fetch_kv(ib + 1, (ib + 1) & 1); }
        if (ib + 2 < nblk) issue_meta(ib + 2);

        // ---- softmax, entirely thread-local: this thread owns row `row`
#pragma unroll
        for (int h = 0; h < BN / 32; ++h) {
            float t[32];
            TCGEN05_LD32(t, taddr(TC_S + h * 32));
            tmem_wait_ld();
#pragma unroll
            for (int j = 0; j < 32; ++j) sv[h * 32 + j] = t[j];
        }
        const float* kst = meta->kst[slot][0];
        const float* vst = meta->vst[slot][0];
        const int n0 = ib * BN;
        float mx = -INFINITY;
#pragma unroll
        for (int n = 0; n < BN; ++n) {
            const int gk = n0 + n;
            sv[n] = (gk <= bound) ? (sv[n] * kst[n]) : -INFINITY;
            mx = fmaxf(mx, sv[n]);
        }
        const float nm = fmaxf(rmax, mx);
        const float sf = isinf(nm) ? 0.f : nm;
        const float alpha = ex2(rmax - sf);
        float sum = 0.f;
        uint32_t pk[BN / 2];
#pragma unroll
        for (int n = 0; n < BN; n += 2) {
            const float e0 = ex2(sv[n] - sf), e1 = ex2(sv[n + 1] - sf);
            sum += e0 + e1;
            pk[n >> 1] = pack_bf16(e0 * vst[n], e1 * vst[n + 1]);
        }
        rsum = rsum * alpha + sum;
        rmax = nm;
        // tcgen05.ld / tcgen05.st / tcgen05.wait::* are .sync.aligned
        // warp-collective: a lane that skips them deadlocks the whole warp.
        // `alpha` is per row and rows are per lane, so the branch below WAS
        // divergent, and that is the stall: job dv_ebadea32cc6d hung between
        // "ib=1 post-qk" and "ib=1 post-pv", i.e. exactly at the first block
        // where the running max moves for some rows but not others (on block 0
        // every lane has rmax=-inf so alpha is uniformly 0 and it survived).
        // Vote instead; lanes that did not need it multiply by 1.0.
        // ib==0 is skipped outright: its PV MMA is issued with accumulate=false
        // and overwrites all D columns, and scaling uninitialised tensor memory
        // by alpha==0 could otherwise produce 0*Inf = NaN.
        if (ib != 0 && __any_sync(0xffffffffu, alpha != 1.f)) {
#pragma unroll
            for (int h = 0; h < D / 32; ++h) {
                float t[32];
                TCGEN05_LD32(t, taddr(TC_O + h * 32));
                tmem_wait_ld();
#pragma unroll
                for (int j = 0; j < 32; ++j) t[j] *= alpha;
                TCGEN05_ST32(taddr(TC_O + h * 32), t);
                tmem_wait_st();
            }
        }
        asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
                     "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                     "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};\n"
                     :: "r"(pk[0]),"r"(pk[1]),"r"(pk[2]),"r"(pk[3]),"r"(pk[4]),"r"(pk[5]),
                        "r"(pk[6]),"r"(pk[7]),"r"(pk[8]),"r"(pk[9]),"r"(pk[10]),"r"(pk[11]),
                        "r"(pk[12]),"r"(pk[13]),"r"(pk[14]),"r"(pk[15]),"r"(pk[16]),"r"(pk[17]),
                        "r"(pk[18]),"r"(pk[19]),"r"(pk[20]),"r"(pk[21]),"r"(pk[22]),"r"(pk[23]),
                        "r"(pk[24]),"r"(pk[25]),"r"(pk[26]),"r"(pk[27]),"r"(pk[28]),"r"(pk[29]),
                        "r"(pk[30]),"r"(pk[31]), "r"(taddr(TC_P)));
        tmem_wait_st();
        tc_fence_before();
        __syncthreads();
        if (tid == 0) {
            uint64_t vd = umma_desc(sa_u32(sVT), TC_ATOM_B, PV_SBO, 0);
#pragma unroll
            for (int nt = 0; nt < BN / 16; ++nt) {
                umma_ts(tmem + TC_O, tmem + TC_P + nt * 8, vd, ID_PV, !(ib == 0 && nt == 0));
                vd = desc_add(vd, TC_KSTEP_B);
            }
            umma_commit(mbar);
        }
        if (!mbar_wait(mbar, phase, ib * 2 + 1)) { tc_ok = false; }
        phase ^= 1;
        tc_fence_after();
#ifdef FMHA_TC_DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && ib < 2)
            printf("TC ib=%d post-pv ok=%d\n", ib, (int)tc_ok);
#endif
        __syncthreads();
    }

    // ---- epilogue: one whole row per thread.  The tcgen05.ld below is
    // warp-collective, so it must be executed by every lane; only the store to
    // `out` is guarded by the row bound.  n_rows is a multiple of p.g but not of
    // 32, so a partially-filled tile would otherwise diverge inside the load.
    {
        const bool wr = (row < n_rows);
        // A timed-out barrier must fail loudly, never silently.
        const float inv = tc_ok ? ((rsum > 0.f) ? (1.f / rsum) : 0.f)
                                : __int_as_float(0x7fc00000);
        const long long o = (long long)(q0 + q_start) * p.out_s0
                          + (long long)(row >> p.g_log2) * p.out_s0
                          + (long long)(hy * p.g + (row & (p.g - 1))) * D;
        for (int h = 0; h < D / 32; ++h) {
            float t[32];
            TCGEN05_LD32(t, taddr(TC_O + h * 32));
            tmem_wait_ld();
            if (wr) {
#pragma unroll
                for (int j = 0; j < 32; j += 2)
                    *reinterpret_cast<uint32_t*>(p.out + o + h * 32 + j) =
                        pack_bf16(t[j] * inv, t[j + 1] * inv);
            }
        }
    }
#ifdef FMHA_TC_DEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0)
        printf("TC done tc_ok=%d nblk=%d rmax=%f rsum=%f\n", (int)tc_ok, nblk, rmax, rsum);
#endif
    __syncthreads();
    if (tid < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                     :: "r"(tmem), "r"((uint32_t)TCOLS));
}

template <int D, int BM, int BN>
static void launch_tc(const Params& p, int n_tiles_ub, cudaStream_t st) {
    using TL = TcLayout<D, BM, BN>;
    // Tensor memory is 512 columns per SM and each CTA takes TCOLS, so at most
    // 512/TCOLS CTAs can hold it. Nothing in the frozen contract limits how many
    // CTAs the scheduler makes resident, and tc_probe validated tcgen05.alloc
    // with exactly one CTA per SM, so do not rely on alloc blocking when tensor
    // memory is oversubscribed: inflate the dynamic shared request until the
    // per-SM shared budget also admits no more than 512/TCOLS CTAs. The padding
    // is never touched.
    constexpr int TC_MAX_CTA = 512 / ((D + BN + BN / 2 <= 256) ? 256 : 512);
    constexpr int SMEM_BUDGET = 232448;
    constexpr int SMEM_MIN = SMEM_BUDGET / (TC_MAX_CTA + 1) + 1;
    constexpr int SMEM = TL::BYTES > SMEM_MIN ? TL::BYTES : SMEM_MIN;
    static_assert(SMEM <= SMEM_BUDGET, "tc shared memory over budget");
    static const bool ok = [] {
        cudaFuncSetAttribute((const void*)fmha_tc<D, BM, BN>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        return true;
    }();
    (void)ok;
    fmha_tc<D, BM, BN><<<dim3(n_tiles_ub, p.Hkv, 1), 128, SMEM, st>>>(p);
}
