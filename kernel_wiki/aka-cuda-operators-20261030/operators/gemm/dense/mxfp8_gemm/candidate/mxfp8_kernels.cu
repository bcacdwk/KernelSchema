// mxfp8_kernels.cu — independent MXFP8 block-scaled GEMM for L20D (sm_103a).
//
// Operator: D = alpha * dequant(A) @ dequant(B)^T + C   (C optional, may alias D)
//   A = (aq e4m3 [m,k] K-major, sa fp32 [m,k/32])   B = (bq e4m3 [n,k], sb fp32 [n,k/32])
//   Scales are exact powers of two (UE8M0 domain). D is bf16 or fp32, N-major.
//
// V3.20 = BK256 3-stage fallback arm (single-wave BN128 class: c3/c0/c1/
// dom1/dom2): probe dv_cdfdd5bc78a8 measured 747ns/BK256-stage = 373.5ns
// per K128 vs 476 for the old 6st-BK128 arm (-21.5% chain; per-stage
// overhead amortized over 2x work). Same probe: clean BN256 full stage =
// 486ns (historical 544 was inflated ~58ns by the old probe's volatile
// marker writes) => heavy-arm per-tile X revised to ~9.2us. BK is now a
// gemm_umma template param (KG=BK/128 TMA groups, NW=BK/32 SF words/row,
// SF tmem col groups sfa+4h / sfb+(BN/32)h, desc bases step h*BM|BN*128);
// BK=128 instantiations fold to the identical instruction stream.
//
// V3.21 = gemm_splitk BK256 3-stage conversion (d0/dom8 class): split
// chains -21% (24x476 -> 12x747ns per split) via the probe-verified V3.20
// mechanism; fixed suite structurally untouched (no fixed case splits).
// Split rule now in BK256 units (>= 4 stages/split); k%256==128 shapes
// stay exact via TMA OOB zero-fill + ue8m0(0)=2^-127 zero-scales.
//
// V3.17 = ABOX on mt==1 arms: A TMA box shrunk to ceil8(m) rows (c0-class
// stages were 44% OOB zero-fill; split class same). MMA still reads the
// full 128-row smem region: rows >= abox accumulate stale garbage only
// into D rows >= m (independent GEMM rows), never stored. Template-isolated
// (ABOX flag) so classic instantiations keep byte-identical codegen.
//
// V3.16 = isolated deep-k split-K (gemm_splitk): tiles128 <= nsm/2 && sk>=3
// (d0 class: m<=128-scale latency shapes with k>=3072) split K across grid.y
// into raw fp32 partials; per-tile counter elects the last arrival to reduce
// inline (alpha/C/cast, deterministic p-order); cached self-cleaning ws.
// Evidence: d0 82->27.4us (-67%, dv_28d28e884bda); shallow-k sk=2 rejected
// (no gain); shared-body implementation rejected (V3.15: +20 regs/+1024B
// static smem leaked ~1.5% onto identical heavy paths; scalar ws stores
// cost c1 0.536x). Classic arms remain byte-identical to join-best V3.10.
//
// Implementation (V3.6): 5th-gen tensor core block-scaled UMMA with TMA feeds.
// V3.6 = warp-specialized MMA_FIRST pipeline on all cg1 arms: tid0 issues
// cp1+MMA(s) into SF tmem slot BN+16*(s&1) AHEAD of buffer-reuse waits (MMA(s)
// queues behind in-flight MMA(s-1), zero tensor-core drain), tid32 re-issues
// TMA(s+PRE) right after commit(s-1) (early prefetch cadence). A/B vs V3.5:
// geomean 1.083x (c5 1.215x c3 1.192x c7 1.123x); c6 0.979x only regression.
// Mechanisms verified numerically-exact on this GPU (experiments/probe_kernels.cu
// sweeps v3..v12 + TMA byte-exactness probes):
//   * tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale, M=128, N=128|256,
//     K=32/inst; idesc = (1<<23 UE8M0) | ((N>>3)<<17) | ((M>>4)<<24) |
//     (a_sf_id<<29) | (b_sf_id<<4); sf_id = byte selector of packed SF words.
//   * TMA 2D box loads (SWIZZLE_128B for operands, NONE for fp32 scale rows)
//     with mbarrier expect_tx; OOB zero-fill covers m/n tails.
//   * scales: exact fp32->UE8M0 pack (byte = exponent bits) into the UTCCP
//     interleave (word w of a 128-row block = row 32*(w%4)+w/4), published via
//     fence.proxy.async; tcgen05.cp warpx4.32x128b to TMEM (SFA 4 cols, SFB
//     4 per 128-row block).
//   * fp32 accumulation in TMEM; epilogue tcgen05.ld 32x32b.x4 (warp w = rows
//     32w..), alpha/C fusion, float2/bfloat162 pair stores, alias-safe.
//   * buffer reuse guarded by per-stage tcgen05.commit mbarriers.
// Mechanism conventions studied from the public PTX ISA and NVIDIA DeepGEMM
// e1f418c2 / CUTLASS as read-only references; independent implementation here.
// No GEMM library, DSL or Torch numerical fallback is used.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <tuple>
#include <cstring>

#define DEVI __device__ __forceinline__

namespace mxfp8 {

// ---------------------------------------------------------------- PTX helpers
DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// --- mbarriers / fences
DEVI void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
DEVI void mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(tx));
}
DEVI void mbar_wait(uint64_t* bar, uint32_t phase) {
    uint32_t addr = smem_u32(bar);
    asm volatile(
        "{\n\t.reg .pred P;\n\t"
        "LAB_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t"
        "bra LAB_WAIT_%=;\n\t"
        "DONE_%=:\n\t}"
        ::"r"(addr), "r"(phase));
}
// Cluster-acquire wait: REQUIRED on the leader for barriers that collect
// peer-CTA cta_group::2 TMA transactions (or remote arrives) before the
// waiting thread issues tcgen05 ops that read the peer's smem. Mechanism
// reference: atrex-fa pure-CUDA sm100 mbarrier_try_wait_acquire_cluster;
// DeepGEMM sm100 (cutlass ClusterBarrier::wait). Plain .shared::cta parity
// waits do NOT synchronize-with cluster-scope completions (V37.0-37.2 lesson:
// deterministic ~47-60% element mismatch).
DEVI void mbar_wait_acq_cluster(uint64_t* bar, uint32_t phase) {
    uint32_t addr = smem_u32(bar);
    asm volatile(
        "{\n\t.reg .pred P;\n\t"
        "LAB_WAITC_%=:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONEC_%=;\n\t"
        "bra LAB_WAITC_%=;\n\t"
        "DONEC_%=:\n\t}"
        ::"r"(addr), "r"(phase) : "memory");
}
DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;"); }
DEVI void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;"); }

// --- TMA 2D load (probe-verified byte-exact incl. swizzle + OOB zero-fill)
DEVI void tma_load_2d(const CUtensorMap* map, uint64_t* bar, void* smem, int x, int y) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        ::"r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(smem_u32(bar))
        : "memory");
}

// --- cta_group::2 TMA (SM100 two-CTA contract): each CTA issues its own load
// into its OWN smem, but BOTH transactions complete the LEADER CTA's barrier
// (peer bit 24 of the shared::cluster address cleared). Mechanism reference:
// DeepGEMM sm100 SM100_TMA_2SM_LOAD / SM100 two-CTA ISA contract. Lets the
// rank0 MMA issuer gate on a purely LOCAL wait for the pair's stage data.
DEVI void tma_load_2d_cg2(const CUtensorMap* map, uint64_t* leader_bar, void* smem, int x, int y) {
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y),
          "r"(smem_u32(leader_bar) & 0xFEFFFFFFu)
        : "memory");
}

// --- multicast TMA (sm90-era form): one issue delivers the box to ALL CTAs in
// the mask; each receiver's barrier at the SAME smem offset collects the tx.
// Used by V38 pdbmc for the pair-shared B stage (rank0 issues once; both CTAs'
// full barriers receive the bytes => B's L2 read traffic halves per SM).
DEVI void tma_load_2d_mc(const CUtensorMap* map, uint64_t* bar, void* smem, int x, int y,
                         uint16_t mask) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes."
        "multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
        ::"r"(smem_u32(smem)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y),
          "r"(smem_u32(bar)), "h"(mask)
        : "memory");
}

// --- tensor memory
DEVI void tmem_alloc1(uint32_t* dst_smem, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)), "r"(ncols));
}
DEVI void tmem_relinq1() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
DEVI void tmem_dealloc1(uint32_t base, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(base), "r"(ncols));
}
DEVI void tmem_ld_4x(uint32_t taddr, uint32_t* v) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                 : "r"(taddr));
}
DEVI void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
// 32-col block load: one wait per 32 cols (vs 8 x4 loads+waits); words v[0..31]
// = cols c..c+31 in order (x4-concatenation semantics).
DEVI void tmem_ld_32x(uint32_t taddr, uint32_t* v) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                 "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]),
                   "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]),
                   "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]),
                   "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]),
                   "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]),
                   "=r"(v[30]), "=r"(v[31])
                 : "r"(taddr));
}

// --- UMMA (tcgen05.mma) block-scaled mxf8f6f4, cta_group::1
DEVI uint64_t make_desc(uint32_t smem_addr, uint32_t sbo16, uint32_t lbo16, uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((smem_addr >> 4) & 0x3FFF);
    d |= (uint64_t)(lbo16 & 0x3FFF) << 16;
    d |= (uint64_t)(sbo16 & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;                 // version (SM100)
    d |= (uint64_t)(layout & 0x7) << 61;    // 0=NONE 2=128B 4=64B 6=32B
    return d;
}
DEVI void umma_mxf8_1sm(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc,
                        uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t acc) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}"
        ::"r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(acc), "r"(tmem_sfa), "r"(tmem_sfb));
}
DEVI void umma_commit1(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)));
}
DEVI void utccp_4x32dp128bit_1cta(uint32_t tmem_col, uint64_t desc) {
    asm volatile("tcgen05.cp.cta_group::1.warpx4.32x128b [%0], %1;" ::"r"(tmem_col), "l"(desc));
}

// Output-type helpers
template <typename T>
DEVI float cd2f(T v);
template <>
DEVI float cd2f<float>(float v) { return v; }
template <>
DEVI float cd2f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T>
DEVI T cd_from_f32(float v);
template <>
DEVI float cd_from_f32<float>(float v) { return v; }
template <>
DEVI __nv_bfloat16 cd_from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// exact fp32 power-of-two -> UE8M0 byte (contract: scales are exp2(byte-127))
DEVI uint32_t ue8m0(float v) { return (__float_as_uint(v) >> 23) & 0xFFu; }
DEVI void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)));
}
DEVI void bar_sync128() { asm volatile("bar.sync 1, 128;" ::: "memory"); }

// --- cluster / cta_group::2 helpers (verified by bisect2 var15/var19, job
// dv_45ef3fd10762: exact numerics with per-CTA-divergent SF)
DEVI uint32_t cluster_rank() {
    uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
DEVI uint32_t mapa_rank0(uint32_t saddr) {
    uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(r) : "r"(saddr)); return r;
}
DEVI void mbar_arrive_remote(uint32_t cluster_bar_addr) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_bar_addr));
}
DEVI void cluster_sync() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
DEVI void umma_mxf8_2sm(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc,
                        uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t acc) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}"
        ::"r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(acc), "r"(tmem_sfa), "r"(tmem_sfb));
}
DEVI void umma_commit2_mc(uint64_t* bar, uint16_t mask) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                 ::"r"(smem_u32(bar)), "h"(mask));
}
DEVI void utccp_2cta(uint32_t tmem_col, uint64_t desc) {
    asm volatile("tcgen05.cp.cta_group::2.warpx4.32x128b [%0], %1;" ::"r"(tmem_col), "l"(desc));
}
DEVI void tmem_alloc2(uint32_t* dst_smem, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)), "r"(ncols));
}
DEVI void tmem_relinq2() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"); }
DEVI void tmem_dealloc2(uint32_t base, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(base), "r"(ncols));
}

// Compute+store 32 columns for one row from x32 words (fast epilogue block).
// Caller guarantees col+31 < n. wide_ok: 16B-aligned rows => 16B C loads and
// D stores (float4 / uint4); else 2B/4B elements. Element math identical to
// the group-4 path: y = alpha*acc (+C). C fully read before D written
// (c_alias_d safe).
// CS=true (V39.4): D stores become st.global.cs (streaming, evict-first) -
// heavy-class d output (c7: 201MB) otherwise streams through L2 evicting the
// A/B working set and inflating stage-feed miss latency inside the ring.
// CS=false default => every existing caller keeps byte-identical codegen.
DEVI void stcs16(void* p, uint32_t a, uint32_t b, uint32_t c, uint32_t d2) {
    asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};"
                 ::"l"(p), "r"(a), "r"(b), "r"(c), "r"(d2) : "memory");
}
template <typename CD, bool CS = false>
DEVI void epi_store32(CD* drow, const CD* crow, int col, float alpha, int has_c,
                      bool wide_ok, const uint32_t* v) {
    float y[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) y[i] = __uint_as_float(v[i]) * alpha;
    if (has_c) {
        if (wide_ok) {
            if constexpr (sizeof(CD) == 4) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 c4 = *reinterpret_cast<const float4*>(crow + col + 4 * i);
                    y[4 * i] += c4.x; y[4 * i + 1] += c4.y;
                    y[4 * i + 2] += c4.z; y[4 * i + 3] += c4.w;
                }
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 q = *reinterpret_cast<const uint4*>(crow + col + 8 * i);
                    const __nv_bfloat162 h0 = *reinterpret_cast<const __nv_bfloat162*>(&q.x);
                    const __nv_bfloat162 h1 = *reinterpret_cast<const __nv_bfloat162*>(&q.y);
                    const __nv_bfloat162 h2 = *reinterpret_cast<const __nv_bfloat162*>(&q.z);
                    const __nv_bfloat162 h3 = *reinterpret_cast<const __nv_bfloat162*>(&q.w);
                    y[8 * i]     += __bfloat162float(__low2bfloat16(h0));
                    y[8 * i + 1] += __bfloat162float(__high2bfloat16(h0));
                    y[8 * i + 2] += __bfloat162float(__low2bfloat16(h1));
                    y[8 * i + 3] += __bfloat162float(__high2bfloat16(h1));
                    y[8 * i + 4] += __bfloat162float(__low2bfloat16(h2));
                    y[8 * i + 5] += __bfloat162float(__high2bfloat16(h2));
                    y[8 * i + 6] += __bfloat162float(__low2bfloat16(h3));
                    y[8 * i + 7] += __bfloat162float(__high2bfloat16(h3));
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 32; ++i) y[i] += cd2f<CD>(crow[col + i]);
        }
    }
    if (wide_ok) {
        if constexpr (sizeof(CD) == 4) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 q = make_float4(y[4 * i], y[4 * i + 1], y[4 * i + 2], y[4 * i + 3]);
                if constexpr (CS)
                    stcs16(drow + col + 4 * i, __float_as_uint(q.x), __float_as_uint(q.y),
                           __float_as_uint(q.z), __float_as_uint(q.w));
                else
                    *reinterpret_cast<float4*>(drow + col + 4 * i) = q;
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint4 q;
                const __nv_bfloat162 h0 = {__float2bfloat16_rn(y[8 * i]),     __float2bfloat16_rn(y[8 * i + 1])};
                const __nv_bfloat162 h1 = {__float2bfloat16_rn(y[8 * i + 2]), __float2bfloat16_rn(y[8 * i + 3])};
                const __nv_bfloat162 h2 = {__float2bfloat16_rn(y[8 * i + 4]), __float2bfloat16_rn(y[8 * i + 5])};
                const __nv_bfloat162 h3 = {__float2bfloat16_rn(y[8 * i + 6]), __float2bfloat16_rn(y[8 * i + 7])};
                q.x = *reinterpret_cast<const uint32_t*>(&h0);
                q.y = *reinterpret_cast<const uint32_t*>(&h1);
                q.z = *reinterpret_cast<const uint32_t*>(&h2);
                q.w = *reinterpret_cast<const uint32_t*>(&h3);
                if constexpr (CS)
                    stcs16(drow + col + 8 * i, q.x, q.y, q.z, q.w);
                else
                    *reinterpret_cast<uint4*>(drow + col + 8 * i) = q;
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) drow[col + i] = cd_from_f32<CD>(y[i]);
    }
}

// V41 (J28 mechanism-refined): dedicated store for the pipelined drain
// (bf16 wide, C preloaded into registers). Arithmetic order IDENTICAL to
// epi_store32's bf16-wide path => bitwise-identical results. epi_store32
// itself is untouched => all serial callers keep byte-identical codegen.
template <typename CD, bool CS = false>
DEVI void epi_store32_cq(CD* drow, int col, float alpha,
                         const uint32_t* v, const uint4* cq) {
    float y[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) y[i] = __uint_as_float(v[i]) * alpha;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 q = cq[i];
        const __nv_bfloat162 h0 = *reinterpret_cast<const __nv_bfloat162*>(&q.x);
        const __nv_bfloat162 h1 = *reinterpret_cast<const __nv_bfloat162*>(&q.y);
        const __nv_bfloat162 h2 = *reinterpret_cast<const __nv_bfloat162*>(&q.z);
        const __nv_bfloat162 h3 = *reinterpret_cast<const __nv_bfloat162*>(&q.w);
        y[8 * i]     += __bfloat162float(__low2bfloat16(h0));
        y[8 * i + 1] += __bfloat162float(__high2bfloat16(h0));
        y[8 * i + 2] += __bfloat162float(__low2bfloat16(h1));
        y[8 * i + 3] += __bfloat162float(__high2bfloat16(h1));
        y[8 * i + 4] += __bfloat162float(__low2bfloat16(h2));
        y[8 * i + 5] += __bfloat162float(__high2bfloat16(h2));
        y[8 * i + 6] += __bfloat162float(__low2bfloat16(h3));
        y[8 * i + 7] += __bfloat162float(__high2bfloat16(h3));
    }
    if constexpr (sizeof(CD) == 4) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 q = make_float4(y[4 * i], y[4 * i + 1], y[4 * i + 2], y[4 * i + 3]);
            if constexpr (CS)
                stcs16(drow + col + 4 * i, __float_as_uint(q.x), __float_as_uint(q.y),
                       __float_as_uint(q.z), __float_as_uint(q.w));
            else
                *reinterpret_cast<float4*>(drow + col + 4 * i) = q;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint4 q;
            const __nv_bfloat162 h0 = {__float2bfloat16_rn(y[8 * i]),     __float2bfloat16_rn(y[8 * i + 1])};
            const __nv_bfloat162 h1 = {__float2bfloat16_rn(y[8 * i + 2]), __float2bfloat16_rn(y[8 * i + 3])};
            const __nv_bfloat162 h2 = {__float2bfloat16_rn(y[8 * i + 4]), __float2bfloat16_rn(y[8 * i + 5])};
            const __nv_bfloat162 h3 = {__float2bfloat16_rn(y[8 * i + 6]), __float2bfloat16_rn(y[8 * i + 7])};
            q.x = *reinterpret_cast<const uint32_t*>(&h0);
            q.y = *reinterpret_cast<const uint32_t*>(&h1);
            q.z = *reinterpret_cast<const uint32_t*>(&h2);
            q.w = *reinterpret_cast<const uint32_t*>(&h3);
            if constexpr (CS)
                stcs16(drow + col + 8 * i, q.x, q.y, q.z, q.w);
            else
                *reinterpret_cast<uint4*>(drow + col + 8 * i) = q;
        }
    }
}

// ---------------------------------------------------------------- GEMM kernel
// BM=128, BN in {128,256}, BK=128 per stage (4x K32 UMMA steps), 128 threads /
// 4 warps, TMA-fed multi-stage pipeline, fp32 accumulation in TMEM.
template <int STAGES, int BN, typename CD, bool MMA_FIRST = false, bool ABOX = false,
          int BK = 128>
__global__ void __launch_bounds__(128, (BN == 256 || BN == 64) ? 1 : 2)   // V3.22: BN224 -> 2 (2-CTA residency, regs <= 256)
gemm_umma(int m, int n, int k, float alpha, int group,
          const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
          const __grid_constant__ CUtensorMap tmSA, const __grid_constant__ CUtensorMap tmSB,
          CD* __restrict__ d, int64_t d_stride0,
          const CD* __restrict__ cptr, int64_t c_stride0, int has_c) {
    constexpr int BM = 128;
    // V3.20: BK is templated (128 classic, 256 for the 3-stage fallback arm:
    // probe dv_cdfdd5bc78a8 cfg1 vs cfg2 — 747ns/BK256-stage = 373.5/K128
    // vs 476 for BK128-6st => -21.5% chain on single-wave long-k shapes).
    constexpr int KG = BK / 128;                 // 128B TMA k-groups per stage
    constexpr int NW = BK / 32;                  // k32 blocks (= SF bytes/row)
    constexpr int SFB_BLOCKS = (BN + 127) / 128;  // UTCCP blocks, ceil (256->2 224->2 128->1 64->1; rows>=BN don't-care padding)
    // pad to 1024B: SWIZZLE_128B atoms are 8x128B; stage bases must stay atom-aligned
    constexpr int ST_RAW = BM * BK + BN * BK + (BM + BN) * NW * 4 + (BM + BN) * NW;
    constexpr int ST_BYTES = (ST_RAW + 1023) & ~1023;
    constexpr int A_OFF = 0;
    constexpr int B_OFF = BM * BK;
    constexpr int SAF_OFF = B_OFF + BN * BK;
    constexpr int SBF_OFF = SAF_OFF + BM * NW * 4;
    constexpr int SFA_OFF = SBF_OFF + BN * NW * 4;
    constexpr int SFB_OFF = SFA_OFF + BM * NW;
    constexpr int TMEM_COLS = (BN == 256) ? 512 : (BN == 64) ? 128 : 256;  // BN224: D 224 + SF 2x16 slots = 256 exact; 2 CTAs = 512
    constexpr uint32_t TX_BYTES = BM * BK + BN * BK + (BM + BN) * NW * 4;
    // V3.17 ABOX (mt==1 arms only; host pairs this flag with an A tensor map
    // whose box is ceil8(min(m,BM)) rows): TX must match the reduced A
    // delivery. Rows >= abox keep stale smem; the MMA accumulates their
    // garbage only into D rows >= m (GEMM rows are independent) which the
    // epilogue never stores. ABOX=false folds to the constexpr => classic
    // instantiations keep byte-identical codegen (V3.15 isolation lesson).
    // V27 SAF-ABOX (mechanism reference: arm-a N4, join 20261025): the SAF
    // fp32 box shrinks with the A box; rows >= abox keep stale smem (garbage
    // scales confined to never-stored D rows >= m - same class as ABOX-A).
    const uint32_t tx_bytes = ABOX
        ? (uint32_t)((((min(m, BM) + 7) & ~7) * (BK + NW * 4)) + BN * (BK + NW * 4))
        : TX_BYTES;

    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + STAGES * ST_BYTES);   // A/B TMA arrivals
    uint64_t* empty = full + STAGES;                                          // MMA commits
    uint64_t* fullsf = empty + STAGES;                                        // SF TMA arrivals
    uint64_t* sfready = fullsf + STAGES;                                      // packer arrivals (64)
    uint32_t* tbp = reinterpret_cast<uint32_t*>(sfready + STAGES);
    // V5: split TX so SF lands on its own barrier (packers start while A/B
    // are still in flight); ABOX only ever shrinks the A/B part.
    const uint32_t sf_tx = ABOX
        ? (uint32_t)((((min(m, BM) + 7) & ~7) + BN) * NW * 4)   // V27 SAF-ABOX
        : (uint32_t)((BM + BN) * NW * 4);
    const uint32_t ab_tx = tx_bytes - sf_tx;

    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;

    // ---- tile mapping with M-grouped swizzle for L2 locality
    const int num_m_blocks = (m + BM - 1) / BM;
    const int num_n_blocks = (n + BN - 1) / BN;
    const int total = num_m_blocks * num_n_blocks;
    const int tile = blockIdx.x;
    if (tile >= total) return;
    const int per_group = num_n_blocks * group;
    const int group_idx = tile / per_group;
    const int first = group_idx * group;
    const int in_group = tile % per_group;
    const int span = min(group, num_m_blocks - first);
    const int m_block = first + in_group % span;
    const int n_block = in_group / span;
    const int m0 = m_block * BM, n0 = n_block * BN;

    const int nk = (k + BK - 1) / BK;

    // ---- init barriers (V3.10: TMEM allocation moved AFTER the prologue TMA
    // issue below — the first TMAs need no tmem, so alloc latency overlaps
    // TMA flight. Producer is wid1 (tid32) on MMA_FIRST arms => wid0's alloc
    // runs truly concurrently. Shaves the per-CTA cold path: paid per wave on
    // c7 (21x) and over the whole lifetime of single-tile c0/c1.)
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], 1); mbar_init(&empty[s], 1);
            mbar_init(&fullsf[s], 1); mbar_init(&sfready[s], 64);
        }
        fence_mbar_init();
    }
    __syncthreads();   // barrier init visible to the producer thread

    // ---- TMA stage issue (producer thread; SF first on its own barrier so
    // the packers overlap the A/B flight - DeepGEMM e1f418c2 pattern)
    auto issue_stage = [&](int s) {
        const int buf = s % STAGES;
        uint8_t* stg = smem + buf * ST_BYTES;
        mbar_arrive_expect_tx(&fullsf[buf], sf_tx);
#pragma unroll
        for (int h = 0; h < KG; ++h) {
            tma_load_2d(&tmSA, &fullsf[buf], stg + SAF_OFF + h * (BM * 16), s * NW + h * 4, m0);
            tma_load_2d(&tmSB, &fullsf[buf], stg + SBF_OFF + h * (BN * 16), s * NW + h * 4, n0);
        }
        mbar_arrive_expect_tx(&full[buf], ab_tx);
#pragma unroll
        for (int h = 0; h < KG; ++h) {
            tma_load_2d(&tmA, &full[buf], stg + A_OFF + h * (BM * 128), s * BK + h * 128, m0);
            tma_load_2d(&tmB, &full[buf], stg + B_OFF + h * (BN * 128), s * BK + h * 128, n0);
        }
    };

    // ---- pack fp32 scale rows into UE8M0 words, UTCCP-interleaved:
    // word w of block b holds row 128b + 32*(w%4) + w/4  <=>  row r -> w = 4*(r%32) + r/32
    // V5: 64 packer threads (warps 2-3), row-mapped: r in {ptid, ptid+64}.
    auto pack_sf = [&](int stage, int ptid) {
        uint8_t* stg = smem + stage * ST_BYTES;
        uint32_t* sfa_p = reinterpret_cast<uint32_t*>(stg + SFA_OFF);
        uint32_t* sfb_p = reinterpret_cast<uint32_t*>(stg + SFB_OFF);
#pragma unroll
        for (int kg = 0; kg < KG; ++kg) {
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int r = ptid + rr * 64;
                const float4 fa = *reinterpret_cast<const float4*>(stg + SAF_OFF + kg * (BM * 16) + r * 16);
                sfa_p[kg * 128 + 4 * (r & 31) + (r >> 5)] =
                    ue8m0(fa.x) | (ue8m0(fa.y) << 8) | (ue8m0(fa.z) << 16) | (ue8m0(fa.w) << 24);
            }
#pragma unroll
            for (int b = 0; b < SFB_BLOCKS; ++b) {
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    const int r = ptid + rr * 64;
                    const int row = b * 128 + r;
                    uint32_t wb = 0u;
                    if (row < BN) {
                        const float4 fb = *reinterpret_cast<const float4*>(stg + SBF_OFF + kg * (BN * 16) + row * 16);
                        wb = ue8m0(fb.x) | (ue8m0(fb.y) << 8) | (ue8m0(fb.z) << 16) | (ue8m0(fb.w) << 24);
                    }
                    sfb_p[(b * KG + kg) * 128 + 4 * (r & 31) + (r >> 5)] = wb;
                }
            }
        }
    };

    const uint32_t idesc_base = (1u << 23) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)(BM >> 4) << 24);

    // ---- main pipelined loop, V5 role-split (DeepGEMM e1f418c2 cadence):
    // tid0 = MMA issuer, tid32 = TMA producer (free-run, empty-gated),
    // warps 2-3 = SF packers on the fullsf/sfready barrier pair. No
    // __syncthreads and no issuer-participated pack on the chain path; the
    // pack overlaps the A/B flight. MMA_FIRST SF-slot double buffering is
    // preserved; slot safety now rests purely on the issuer thread's
    // in-order TC pipe (utccp(s) executes after MMA(s-2), the slot's
    // previous reader) - strictly stronger than the old cross-thread
    // commit-observation argument.
    const int PRE = STAGES - 1 < nk ? STAGES - 1 : nk;
    if (tid == 32) {
        for (int i = 0; i < PRE; ++i) issue_stage(i);
    }
    // TMEM alloc AFTER first TMAs are in flight (V3.10); tb needed by the
    // first cp1/MMA only.
    if (wid == 0) {
        tmem_alloc1(tbp, TMEM_COLS);
        tmem_relinq1();
    }
    __syncthreads();
    const uint32_t tb = *tbp;
    if (tid == 0) {
        // ---------------- issuer ----------------
        for (int s = 0; s < nk; ++s) {
            const int buf = s % STAGES;
            mbar_wait(&full[buf], (s / STAGES) & 1);
            mbar_wait(&sfready[buf], (s / STAGES) & 1);
            tc_fence_after();
            uint8_t* stg = smem + buf * ST_BYTES;
            const uint32_t sfa_col = MMA_FIRST ? (tb + BN + 16 * (s & 1)) : (tb + BN);
            const uint32_t sfb_col = sfa_col + 4 * KG;
#pragma unroll
            for (int kg = 0; kg < KG; ++kg)
                utccp_4x32dp128bit_1cta(sfa_col + 4 * kg,
                                        make_desc(smem_u32(stg + SFA_OFF + 512 * kg), 8, 0, 0));
#pragma unroll
            for (int b = 0; b < SFB_BLOCKS; ++b)
#pragma unroll
                for (int kg = 0; kg < KG; ++kg)
                    utccp_4x32dp128bit_1cta(sfb_col + (BN / 32) * kg + 4 * b,
                                            make_desc(smem_u32(stg + SFB_OFF + (b * KG + kg) * 512), 8, 0, 0));
#pragma unroll
            for (int kk = 0; kk < NW; ++kk) {
                const int h = kk >> 2, sel = kk & 3;
                const uint64_t dah = make_desc(smem_u32(stg + A_OFF + h * (BM * 128)), 64, 0, 2);
                const uint64_t dbh = make_desc(smem_u32(stg + B_OFF + h * (BN * 128)), 64, 0, 2);
                const uint32_t id2 = idesc_base | ((uint32_t)sel << 29) | ((uint32_t)sel << 4);
                umma_mxf8_1sm(tb, dah + sel * 2, dbh + sel * 2, id2,
                              sfa_col + 4 * h, sfb_col + (BN / 32) * h,
                              (s > 0 || kk > 0) ? 1u : 0u);
            }
            umma_commit1(&empty[buf]);
        }
    } else if (tid == 32) {
        // ---------------- producer (free-run, commit-gated) ----------------
        for (int sn0 = PRE; sn0 < nk; ++sn0) {
            const int bn = sn0 % STAGES;
            // buffer bn last fed stage sn0-STAGES; its MMA commit must land first
            if (sn0 >= STAGES) mbar_wait(&empty[bn], ((sn0 - STAGES) / STAGES) & 1);
            issue_stage(sn0);
        }
    } else if (wid >= 2) {
        // ---------------- packers (warps 2-3, 64 threads) ----------------
        const int ptid = tid - 64;
        for (int s = 0; s < nk; ++s) {
            const int buf = s % STAGES;
            mbar_wait(&fullsf[buf], (s / STAGES) & 1);
            pack_sf(buf, ptid);
            fence_proxy_async();      // publish SF words to the async proxy
            mbar_arrive(&sfready[buf]);
        }
    }

    // ---- epilogue: wait final commit, read TMEM, fuse alpha + C, store D
    mbar_wait(&empty[(nk - 1) % STAGES], ((nk - 1) / STAGES) & 1);
    tc_fence_after();

    const int row = m0 + wid * 32 + lane;
    CD* drow = d + (size_t)row * d_stride0;
    const CD* crow = has_c ? cptr + (size_t)row * c_stride0 : nullptr;
    // 16B vector-store eligibility (alignment only; n-tail handled per block).
    // col bases are multiples of 32 => 16B-aligned offsets when the row base is.
    const bool wide_ok = ((d_stride0 * (int64_t)sizeof(CD)) % 16 == 0) &&
                         ((uintptr_t)d % 16 == 0) &&
                         (!has_c || ((c_stride0 * (int64_t)sizeof(CD)) % 16 == 0 &&
                                     (uintptr_t)cptr % 16 == 0));
    for (int c0 = 0; c0 < BN; c0 += 32) {
        uint32_t v[32];
        // tcgen05.ld is warp-sync: every lane participates regardless of row guard
        tmem_ld_32x(tb + c0, v);
        tmem_ld_wait();
        if (row < m) {
            const int col = n0 + c0;
            if (col + 31 < n) {
                epi_store32<CD>(drow, crow, col, alpha, has_c, wide_ok, v);
            } else {
                // guarded group-4 tail (n edge)
                uint32_t (*vg)[4] = reinterpret_cast<uint32_t (*)[4]>(v);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int cg = col + j * 4;
                    if (cg + 3 < n) {
                        float y0 = __uint_as_float(vg[j][0]) * alpha;
                        float y1 = __uint_as_float(vg[j][1]) * alpha;
                        float y2 = __uint_as_float(vg[j][2]) * alpha;
                        float y3 = __uint_as_float(vg[j][3]) * alpha;
                        if (has_c) {
                            if constexpr (sizeof(CD) == 4) {
                                const float2 c01 = *reinterpret_cast<const float2*>(crow + cg);
                                const float2 c23 = *reinterpret_cast<const float2*>(crow + cg + 2);
                                y0 += c01.x; y1 += c01.y; y2 += c23.x; y3 += c23.y;
                            } else {
                                const __nv_bfloat162 c01 = *reinterpret_cast<const __nv_bfloat162*>(crow + cg);
                                const __nv_bfloat162 c23 = *reinterpret_cast<const __nv_bfloat162*>(crow + cg + 2);
                                y0 += __bfloat162float(__low2bfloat16(c01));
                                y1 += __bfloat162float(__high2bfloat16(c01));
                                y2 += __bfloat162float(__low2bfloat16(c23));
                                y3 += __bfloat162float(__high2bfloat16(c23));
                            }
                        }
                        if constexpr (sizeof(CD) == 4) {
                            *reinterpret_cast<float2*>(drow + cg) = make_float2(y0, y1);
                            *reinterpret_cast<float2*>(drow + cg + 2) = make_float2(y2, y3);
                        } else {
                            __nv_bfloat162 p01 = {__float2bfloat16_rn(y0), __float2bfloat16_rn(y1)};
                            __nv_bfloat162 p23 = {__float2bfloat16_rn(y2), __float2bfloat16_rn(y3)};
                            *reinterpret_cast<__nv_bfloat162*>(drow + cg) = p01;
                            *reinterpret_cast<__nv_bfloat162*>(drow + cg + 2) = p23;
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            if (cg + i < n) {
                                float y = __uint_as_float(vg[j][i]) * alpha;
                                if (has_c) y += cd2f<CD>(crow[cg + i]);
                                drow[cg + i] = cd_from_f32<CD>(y);
                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    if (wid == 0) tmem_dealloc1(tb, TMEM_COLS);
}

// ------------------------------------------------------ V3.16 split-K kernel
// Standalone 6-stage BN128 MMA_FIRST kernel for the DEEP-K latency class
// (tiles128 <= nsm/2 && sk >= 3): grid (tiles128, sk); each CTA accumulates
// its k-window (blockIdx.y*kchunk stages, absolute TMA coords, LOCAL barrier
// parities), stores RAW fp32 partials to ws[y] (float4-vectorized), and the
// per-tile LAST arrival (device-scope counter) reduces all sk slices inline
// with alpha/C/cast. Counter self-resets => cached ws/counters zeroed once
// (allocation elision only; every forward recomputes all values).
// ISOLATED from gemm_umma on purpose (V3.15 lesson: sharing one body lifted
// the classic arms 76->96 regs + 1024B static smem => ~1.5% drag on
// identical heavy paths; scalar wid0-only ws stores cost c1 0.536x).
template <typename CD>
__global__ void __launch_bounds__(128, 2)
gemm_splitk(int m, int n, int k, float alpha, int group,
            const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
            const __grid_constant__ CUtensorMap tmSA, const __grid_constant__ CUtensorMap tmSB,
            CD* __restrict__ d, int64_t d_stride0,
            const CD* __restrict__ cptr, int64_t c_stride0, int has_c,
            int kchunk, float* __restrict__ ws, int* __restrict__ counter, int sk) {
    // V3.21: BK256 3-stage (V3.20 mechanism, probe dv_cdfdd5bc78a8:
    // 747ns/BK256-stage = 373.5/K128 vs 476 => split chains -21%).
    constexpr int STAGES = 3, BN = 128, BM = 128, BK = 256;
    constexpr int KG = BK / 128, NW = BK / 32;
    constexpr int ST_BYTES = BM * BK + BN * BK + (BM + BN) * NW * 4 + (BM + BN) * NW;  // 75776
    constexpr int A_OFF = 0;
    constexpr int B_OFF = BM * BK;
    constexpr int SAF_OFF = B_OFF + BN * BK;
    constexpr int SBF_OFF = SAF_OFF + BM * NW * 4;
    constexpr int SFA_OFF = SBF_OFF + BN * NW * 4;
    constexpr int SFB_OFF = SFA_OFF + BM * NW;
    constexpr int TMEM_COLS = 256;
    // V3.17: the split class is always mt==1 => A box shrunk to ceil8(m)
    // rows (host passes the matching small-box tensor map).
    // V27 SAF-ABOX: split-class SAF box shrinks with abox (host passes
    // tmSA_use); rows >= abox stale => confined as in the ABOX arms.
    const uint32_t tx_bytes =
        (uint32_t)((((min(m, BM) + 7) & ~7) * (BK + NW * 4)) + BN * (BK + NW * 4));

    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + STAGES * ST_BYTES);
    uint64_t* empty = full + STAGES;
    uint32_t* tbp = reinterpret_cast<uint32_t*>(empty + STAGES);

    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;

    const int num_m_blocks = (m + BM - 1) / BM;
    const int num_n_blocks = (n + BN - 1) / BN;
    const int total = num_m_blocks * num_n_blocks;
    const int tile = blockIdx.x;
    if (tile >= total) return;
    const int per_group = num_n_blocks * group;
    const int group_idx = tile / per_group;
    const int first = group_idx * group;
    const int in_group = tile % per_group;
    const int span = min(group, num_m_blocks - first);
    const int m_block = first + in_group % span;
    const int n_block = in_group / span;
    const int m0 = m_block * BM, n0 = n_block * BN;

    const int nk_total = (k + BK - 1) / BK;
    const int s0 = (int)blockIdx.y * kchunk;
    const int nk = min(kchunk, nk_total - s0);
    if (nk <= 0) return;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 1); }
        fence_mbar_init();
    }
    __syncthreads();

    auto issue_stage = [&](int s) {
        const int buf = s % STAGES;
        uint8_t* stg = smem + buf * ST_BYTES;
        mbar_arrive_expect_tx(&full[buf], tx_bytes);
#pragma unroll
        for (int h = 0; h < KG; ++h) {
            tma_load_2d(&tmA, &full[buf], stg + A_OFF + h * (BM * 128), s * BK + h * 128, m0);
            tma_load_2d(&tmB, &full[buf], stg + B_OFF + h * (BN * 128), s * BK + h * 128, n0);
            tma_load_2d(&tmSA, &full[buf], stg + SAF_OFF + h * (BM * 16), s * NW + h * 4, m0);
            tma_load_2d(&tmSB, &full[buf], stg + SBF_OFF + h * (BN * 16), s * NW + h * 4, n0);
        }
    };

    const int pw = 4 * (tid & 31) + (tid >> 5);
    auto pack_sf = [&](int stage) {
        uint8_t* stg = smem + stage * ST_BYTES;
        uint32_t* sfa_p = reinterpret_cast<uint32_t*>(stg + SFA_OFF);
        uint32_t* sfb_p = reinterpret_cast<uint32_t*>(stg + SFB_OFF);
#pragma unroll
        for (int kg = 0; kg < KG; ++kg) {
            const float4 fa = *reinterpret_cast<const float4*>(stg + SAF_OFF + kg * (BM * 16) + tid * 16);
            sfa_p[kg * 128 + pw] = ue8m0(fa.x) | (ue8m0(fa.y) << 8) | (ue8m0(fa.z) << 16) | (ue8m0(fa.w) << 24);
            const float4 fb = *reinterpret_cast<const float4*>(stg + SBF_OFF + kg * (BN * 16) + tid * 16);
            sfb_p[kg * 128 + pw] = ue8m0(fb.x) | (ue8m0(fb.y) << 8) | (ue8m0(fb.z) << 16) | (ue8m0(fb.w) << 24);
        }
    };

    const uint32_t idesc_base = (1u << 23) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)(BM >> 4) << 24);

    const int PRE = STAGES - 1 < nk ? STAGES - 1 : nk;
    if (tid == 32) {
        for (int i = 0; i < PRE; ++i) issue_stage(s0 + i);
    }
    if (wid == 0) {
        tmem_alloc1(tbp, TMEM_COLS);
        tmem_relinq1();
    }
    __syncthreads();
    const uint32_t tb = *tbp;
    for (int s = s0; s < s0 + nk; ++s) {
        const int buf = s % STAGES;
        mbar_wait(&full[buf], ((s - s0) / STAGES) & 1);
        pack_sf(buf);
        fence_proxy_async();
        __syncthreads();
        if (wid == 0 && lane == 0) {
            tc_fence_after();
            uint8_t* stg = smem + buf * ST_BYTES;
            const uint32_t sfa_col = tb + BN + 16 * (s & 1);
            const uint32_t sfb_col = sfa_col + 4 * KG;
#pragma unroll
            for (int kg = 0; kg < KG; ++kg)
                utccp_4x32dp128bit_1cta(sfa_col + 4 * kg,
                                        make_desc(smem_u32(stg + SFA_OFF + 512 * kg), 8, 0, 0));
#pragma unroll
            for (int kg = 0; kg < KG; ++kg)
                utccp_4x32dp128bit_1cta(sfb_col + (BN / 32) * kg,
                                        make_desc(smem_u32(stg + SFB_OFF + 512 * kg), 8, 0, 0));
#pragma unroll
            for (int kk = 0; kk < NW; ++kk) {
                const int h = kk >> 2, sel = kk & 3;
                const uint64_t dah = make_desc(smem_u32(stg + A_OFF + h * (BM * 128)), 64, 0, 2);
                const uint64_t dbh = make_desc(smem_u32(stg + B_OFF + h * (BN * 128)), 64, 0, 2);
                const uint32_t id2 = idesc_base | ((uint32_t)sel << 29) | ((uint32_t)sel << 4);
                umma_mxf8_1sm(tb, dah + sel * 2, dbh + sel * 2, id2,
                              sfa_col + 4 * h, sfb_col + (BN / 32) * h,
                              (s > s0 || kk > 0) ? 1u : 0u);
            }
        }
        if (tid == 32) {
            const int sn0 = s + PRE;
            if (sn0 < s0 + nk) {
                const int bn = sn0 % STAGES;
                if (sn0 >= s0 + STAGES) mbar_wait(&empty[bn], ((sn0 - s0 - STAGES) / STAGES) & 1);
                issue_stage(sn0);
            }
        }
        if (tid == 0) umma_commit1(&empty[buf]);
    }

    // ---- fused split epilogue: raw partials -> ws, counter-elected reduce
    const int slast = s0 + nk - 1;
    mbar_wait(&empty[slast % STAGES], ((slast - s0) / STAGES) & 1);
    tc_fence_after();

    const int row = m0 + wid * 32 + lane;
    float* wsl = ws + (size_t)blockIdx.y * ((size_t)m * n);
    for (int c0 = 0; c0 < BN; c0 += 32) {
        uint32_t v[32];
        tmem_ld_32x(tb + c0, v);   // warp-collective: all lanes participate
        tmem_ld_wait();
        if (row < m) {
            const int col = n0 + c0;
            float* wrow = wsl + (size_t)row * n;
            if (col + 31 < n) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    *reinterpret_cast<float4*>(wrow + col + 4 * i) =
                        make_float4(__uint_as_float(v[4 * i]), __uint_as_float(v[4 * i + 1]),
                                    __uint_as_float(v[4 * i + 2]), __uint_as_float(v[4 * i + 3]));
            } else {
#pragma unroll
                for (int i = 0; i < 32; ++i)
                    if (col + i < n) wrow[col + i] = __uint_as_float(v[i]);
            }
        }
    }
    __shared__ int is_last;
    __threadfence();               // release this thread's ws stores
    __syncthreads();               // all releases precede the atomic
    if (tid == 0) {
        const int old = atomicAdd(&counter[tile], 1);
        is_last = (old == sk - 1);
        if (is_last) counter[tile] = 0;   // self-reset for cached reuse
    }
    __syncthreads();
    if (is_last) {
        __threadfence();           // acquire all splits' ws stores
        const int vrows = min(BM, m - m0);
        const int vcols = min(BN, n - n0);
        const size_t mn = (size_t)m * n;
        for (int e = tid; e < vrows * vcols; e += 128) {
            const int r = e / vcols, cc = e % vcols;
            const size_t lin = (size_t)(m0 + r) * n + (n0 + cc);
            float sum = ws[lin];
            for (int p = 1; p < sk; ++p) sum += ws[(size_t)p * mn + lin];
            float y = alpha * sum;
            if (has_c) y += cd2f<CD>(cptr[(size_t)(m0 + r) * c_stride0 + (n0 + cc)]);
            d[(size_t)(m0 + r) * d_stride0 + (n0 + cc)] = cd_from_f32<CD>(y);
        }
    }
    __syncthreads();
    if (wid == 0) tmem_dealloc1(tb, TMEM_COLS);
}

// ---------------------------------------------------------------- cg2 kernel
// 2-CTA pairs (cta_group::2): M=256 per pair (each CTA 128 A rows), BN=240
// N-split (each CTA 120 B rows), SF-A own-half per CTA (collective-source cp2,
// M2-local 4-col read), SF-B full-N identical on both CTAs (8 cols).
// Recipe verified exact by bisect2 var15/var19 (dv_45ef3fd10762).
template <int STAGES, int BN, typename CD>
__global__ void __launch_bounds__(128, 1) __cluster_dims__(2, 1, 1)
gemm_umma_cg2(int m, int n, int k, float alpha, int group,
              const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmBh,
              const __grid_constant__ CUtensorMap tmSA, const __grid_constant__ CUtensorMap tmSB,
              CD* __restrict__ d, int64_t d_stride0,
              const CD* __restrict__ cptr, int64_t c_stride0, int has_c) {
    constexpr int BM = 128, BK = 128;          // per CTA
    constexpr int BH = BN / 2;                  // B rows per CTA (120)
    // packed-SFA 1024B (own+dup); packed-SFB 2048B: block0@0, block1@512,
    // zero pad @1024..2047 so BOTH cp2 1024B footprints stay NaN-free.
    constexpr int ST_RAW = BM * BK + BH * BK + (BM + BN) * 16 + 1024 + 2048;
    constexpr int ST_BYTES = (ST_RAW + 1023) & ~1023;
    constexpr int A_OFF = 0;
    constexpr int B_OFF = BM * BK;                                   // 16384
    constexpr int SAF_OFF = B_OFF + BH * BK;                         // +15360
    constexpr int SBF_OFF = SAF_OFF + BM * 16;                       // +2048
    constexpr int SFA_OFF = SBF_OFF + BN * 16;                       // +3840
    constexpr int SFB_OFF = SFA_OFF + 1024;                          // +1024 (SFA real+dup)
    constexpr int SFA_COL = BN;   // tmem: D 0..BN-1; SF slot p at BN+16p:
    // SFA@+0 (footprint +0..7), SFB blk0@+4 (fp +4..11), blk1@+8 (fp +8..15)
    constexpr uint32_t TX_BYTES = BM * BK + BH * BK + BM * 16 + BN * 16;

    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + STAGES * ST_BYTES);
    uint64_t* empty = full + STAGES;
    uint64_t* pair_full = empty + STAGES;      // leader-only: CTA1 data-ready
    uint32_t* tbp = reinterpret_cast<uint32_t*>(pair_full + STAGES);

    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    const uint32_t rank = cluster_rank();

    const int num_pm = (m + 2 * BM - 1) / (2 * BM);
    const int num_pn = (n + BN - 1) / BN;
    const int pair_total = num_pm * num_pn;
    const int pair = blockIdx.x >> 1;
    if (pair >= pair_total) return;
    const int per_group = num_pn * group;
    const int group_idx = pair / per_group;
    const int first = group_idx * group;
    const int in_group = pair % per_group;
    const int span = min(group, num_pm - first);
    const int pm0 = (first + in_group % span) * (2 * BM);
    const int pn0 = (in_group / span) * BN;
    const int m0 = pm0 + (int)rank * BM;        // own A rows
    const int nh0 = pn0 + (int)rank * BH;       // own B half rows

    const int nk = (k + BK - 1) / BK;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], 1);
            mbar_init(&pair_full[s], 1);
        }
        fence_mbar_init();
    }
    if (wid == 0) {
        tmem_alloc2(tbp, 512);
        tmem_relinq2();
    }
    cluster_sync();
    __syncthreads();
    const uint32_t tb = *tbp;

    auto issue_stage = [&](int g) {
        const int buf = g % STAGES;
        uint8_t* stg = smem + buf * ST_BYTES;
        mbar_arrive_expect_tx(&full[buf], TX_BYTES);
        tma_load_2d(&tmA, &full[buf], stg + A_OFF, (g % nk) * BK, m0);
        tma_load_2d(&tmBh, &full[buf], stg + B_OFF, (g % nk) * BK, nh0);
        tma_load_2d(&tmSA, &full[buf], stg + SAF_OFF, (g % nk) * 4, m0);
        tma_load_2d(&tmSB, &full[buf], stg + SBF_OFF, (g % nk) * 4, pn0);
    };

    const int pw = 4 * (tid & 31) + (tid >> 5);
    auto pack_sf = [&](int stage) {
        uint8_t* stg = smem + stage * ST_BYTES;
        const float4 fa = *reinterpret_cast<const float4*>(stg + SAF_OFF + tid * 16);
        uint32_t wa = ue8m0(fa.x) | (ue8m0(fa.y) << 8) | (ue8m0(fa.z) << 16) | (ue8m0(fa.w) << 24);
        uint32_t* sfa_p = reinterpret_cast<uint32_t*>(stg + SFA_OFF);
        sfa_p[pw] = wa;
        sfa_p[128 + pw] = wa;                    // dup: 1024B cp2 footprint
        uint32_t* sfb_p = reinterpret_cast<uint32_t*>(stg + SFB_OFF);
        const float4 fb0 = *reinterpret_cast<const float4*>(stg + SBF_OFF + tid * 16);
        sfb_p[pw] = ue8m0(fb0.x) | (ue8m0(fb0.y) << 8) | (ue8m0(fb0.z) << 16) | (ue8m0(fb0.w) << 24);
        if (128 + tid < BN) {
            const float4 fb1 = *reinterpret_cast<const float4*>(stg + SBF_OFF + (128 + tid) * 16);
            sfb_p[128 + pw] = ue8m0(fb1.x) | (ue8m0(fb1.y) << 8) | (ue8m0(fb1.z) << 16) | (ue8m0(fb1.w) << 24);
        } else {
            sfb_p[128 + pw] = 0u;                // rows >= BN: zero tail (valid UE8M0)
        }
        sfb_p[256 + tid] = 0u;                   // pad for copy2 footprint (NaN-free)
        sfb_p[384 + tid] = 0u;
    };

    const uint32_t idesc_base = (1u << 23) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)((2 * BM) >> 4) << 24);
    const int PRE = STAGES - 1 < nk ? STAGES - 1 : nk;

    if (tid == 0) {
        for (int g = 0; g < PRE; ++g) issue_stage(g);
    }
    for (int s = 0; s < nk; ++s) {
        const int buf = s % STAGES;
        mbar_wait(&full[buf], (s / STAGES) & 1);
        // ORDER IS LOAD-BEARING: pack+arrive BEFORE the empty-gated TMA
        // re-issue. arrive(s) then only depends on full(s) (prefetched ~PRE
        // stages early) instead of commit(s-1) — otherwise the leader's MMA(s)
        // issue serializes behind commit(s-1)->CTA1 pack->arrive and the tensor core
        // idles one handshake chain per stage (~2x slowdown, dv_ee498134ed6a).
        pack_sf(buf);
        fence_proxy_async();          // publish SF words to the async proxy
        __syncthreads();
        // CTA1 signals the leader AFTER its SF words are packed+fenced, so the
        // leader's collective-source cp2 sees valid per-CTA SF smem.
        if (tid == 0 && rank == 1) {
            mbar_arrive_remote(mapa_rank0(smem_u32(&pair_full[buf])));
        }
        if (tid == 0) {
            if (rank == 0) {
                // V4.3: MMA issue BEFORE the empty-gated re-issue, into SF slot
                // s&1 (double-buffered, 16 cols/slot, footprints slot-local).
                // cp2(s) slot safety: slot s&1 was last read by MMA(s-2), whose
                // commit was OBSERVED at iteration s-1's empty-wait below.
                // MMA(s) queues behind in-flight MMA(s-1) in the tensor core
                // (in-order, same D) => zero drain gap per stage.
                mbar_wait(&pair_full[buf], (s / STAGES) & 1);
                tc_fence_after();
                uint8_t* stg = smem + buf * ST_BYTES;
                const uint32_t sfa_col = tb + SFA_COL + 16 * (s & 1);
                const uint32_t sfb_col = sfa_col + 4;
                utccp_2cta(sfa_col, make_desc(smem_u32(stg + SFA_OFF), 8, 0, 0));
                utccp_2cta(sfb_col, make_desc(smem_u32(stg + SFB_OFF), 8, 0, 0));
                // block1 via its own copy at +512B source: lands at footprint
                // cols 0..3 of sfb_col+4 (proven mapping), overwriting copy0's
                // UNTESTED second half. Pad words keep the tail NaN-free.
                utccp_2cta(sfb_col + 4, make_desc(smem_u32(stg + SFB_OFF + 512), 8, 0, 0));
                const uint64_t da = make_desc(smem_u32(stg + A_OFF), 64, 0, 2);
                const uint64_t db = make_desc(smem_u32(stg + B_OFF), 64, 0, 2);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint32_t id2 = idesc_base | ((uint32_t)kk << 29) | ((uint32_t)kk << 4);
                    umma_mxf8_2sm(tb, da + kk * 2, db + kk * 2, id2, sfa_col, sfb_col,
                                  (s > 0 || kk > 0) ? 1u : 0u);
                }
            }
            const int sn = s + PRE;
            if (sn < nk) {
                const int bn = sn % STAGES;
                if (sn >= STAGES) mbar_wait(&empty[bn], ((sn - STAGES) / STAGES) & 1);
                issue_stage(sn);
            }
            if (rank == 0) umma_commit2_mc(&empty[buf], 3);
        }
    }

    // ---- epilogue: own 128 rows x BN cols
    mbar_wait(&empty[(nk - 1) % STAGES], ((nk - 1) / STAGES) & 1);
    tc_fence_after();
    const int row = m0 + wid * 32 + lane;
    CD* drow = d + (size_t)row * d_stride0;
    const CD* crow = has_c ? cptr + (size_t)row * c_stride0 : nullptr;
    for (int c0 = 0; c0 < BN / 4; c0 += 8) {
        uint32_t v[8][4];
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) tmem_ld_4x(tb + (c0 + jj) * 4, v[jj]);
        tmem_ld_wait();
        if (row < m) {
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const int col = pn0 + (c0 + jj) * 4;
                if ((c0 + jj) * 4 >= BN) continue;   // BN%32!=0 tail groups: tmem past D
                if (col + 3 < n) {
                    float y0 = __uint_as_float(v[jj][0]) * alpha;
                    float y1 = __uint_as_float(v[jj][1]) * alpha;
                    float y2 = __uint_as_float(v[jj][2]) * alpha;
                    float y3 = __uint_as_float(v[jj][3]) * alpha;
                    if (has_c) {
                        if constexpr (sizeof(CD) == 4) {
                            const float2 c01 = *reinterpret_cast<const float2*>(crow + col);
                            const float2 c23 = *reinterpret_cast<const float2*>(crow + col + 2);
                            y0 += c01.x; y1 += c01.y; y2 += c23.x; y3 += c23.y;
                        } else {
                            const __nv_bfloat162 c01 = *reinterpret_cast<const __nv_bfloat162*>(crow + col);
                            const __nv_bfloat162 c23 = *reinterpret_cast<const __nv_bfloat162*>(crow + col + 2);
                            y0 += __bfloat162float(__low2bfloat16(c01));
                            y1 += __bfloat162float(__high2bfloat16(c01));
                            y2 += __bfloat162float(__low2bfloat16(c23));
                            y3 += __bfloat162float(__high2bfloat16(c23));
                        }
                    }
                    if constexpr (sizeof(CD) == 4) {
                        *reinterpret_cast<float2*>(drow + col) = make_float2(y0, y1);
                        *reinterpret_cast<float2*>(drow + col + 2) = make_float2(y2, y3);
                    } else {
                        __nv_bfloat162 p01 = {__float2bfloat16_rn(y0), __float2bfloat16_rn(y1)};
                        __nv_bfloat162 p23 = {__float2bfloat16_rn(y2), __float2bfloat16_rn(y3)};
                        *reinterpret_cast<__nv_bfloat162*>(drow + col) = p01;
                        *reinterpret_cast<__nv_bfloat162*>(drow + col + 2) = p23;
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (col + i < n) {
                            float y = __uint_as_float(v[jj][i]) * alpha;
                            if (has_c) y += cd2f<CD>(crow[col + i]);
                            drow[col + i] = cd_from_f32<CD>(y);
                        }
                    }
                }
            }
        }
    }
    cluster_sync();
    __syncthreads();
    if (wid == 0) tmem_dealloc2(tb, 512);
}

// ------------------------------------------------------------ V37 cg2pp kernel
// Prepacked-SF role-split cta_group::2 arm (M=256 pair). Targets the heavy/mid
// classes whose 1-SM BN256 stages are L2-byte bound (probe dv_b9ac363977dd:
// 457ns/stage prod vs 356ns L2 floor vs 269ns MMA floor; dv_ec4b644b135e: 2-SM
// pair stage runs at FULL TC rate, 268.7ns BN256 / 251.9ns BN240 => per-SM
// operand bytes 49KB -> ~35-38KB for the same per-SM FLOPs).
// V37.4 pipeline = DeepGEMM sm100 two-CTA split, exactly:
// * payload (A + B-half): cta_group::2 TMA - both CTAs' tx complete the
//   LEADER's local full barrier (peer-bit-24 masked addr; atrex-fa pure-CUDA
//   sm100 contract); leader producer registers ONE arrive_expect_tx(2*PAY).
// * SF images: PLAIN TMA onto a per-CTA LOCAL sf_full barrier; a relay thread
//   (tid64) on EACH CTA observes it, fence.proxy.async, then release-arrives
//   on the leader's full (count 3 = leader expect + 2 relays). DeepGEMM never
//   gates collective-cp2 sources by tx-only peer accounting (their SF path is
//   local barrier + thread arrivals); P8 probe showed the tx-only SF path
//   (V37.3) races ring reuse: zero-scale rows 64+ from the first reuse stage,
//   smem correct at flip, +1/+2-stage lagged gates inert.
// * leader issuer: ONE acquire.cluster wait, then cp2 x3 + 4x UMMA + commit2mc.
// History: V37.0/37.1 pair_full relay + plain waits: deterministically wrong
// (~47% uniform); V37.2 (+cg2 TMA, plain wait): wrong (~60%); V37.3
// (acquire.cluster, tx-only SF): wrong (~59%, rows-64+ zero-scale signature,
// 3050 TF c5 = feed works). sig_mode 3 = 4x-group epilogue bisect reference.
// Stage 1 = non-persistent (one pair-tile per cluster); persistence is stage 2
// if wave prologue/drain exposure warrants.
template <int STAGES, int BN, typename CD>
__global__ void __launch_bounds__(128, 1) __cluster_dims__(2, 1, 1)
gemm_cg2pp(int m, int n, int k, float alpha, int group,
           const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmBh,
           const __grid_constant__ CUtensorMap tmSA, const __grid_constant__ CUtensorMap tmSB,
           CD* __restrict__ d, int64_t d_stride0,
           const CD* __restrict__ cptr, int64_t c_stride0, int has_c,
           int sig_mode) {
    constexpr int BM = 128, BK = 128, BH = BN / 2;
    constexpr int ST_RAW = BM * BK + BH * BK + 1024 + 2048;   // A + B-half + SFA(img+dup) + SFB(img2+pad)
    constexpr int ST_BYTES = (ST_RAW + 1023) & ~1023;
    constexpr int A_OFF = 0;
    constexpr int B_OFF = BM * BK;
    constexpr int SFA_OFF = B_OFF + BH * BK;
    constexpr int SFB_OFF = SFA_OFF + 1024;
    // V37.4: payload tx (A + B-half, both CTAs -> leader full) vs SF tx
    // (2048B per CTA -> LOCAL sf_full, thread-relayed to the leader: the
    // DeepGEMM sm100 contract - cp2 collective sources are never gated by
    // tx-only peer accounting).
    constexpr uint32_t PAY_TX = BM * BK + BH * BK;
    constexpr uint32_t SF_TX = 1024 + 1024;
    constexpr int SFA_COL = BN;   // tmem SF slots at BN + 16*(g&1)
    constexpr int SFB_BLOCKS2 = (BN + 127) / 128;   // 2 for both 240/256

    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + STAGES * ST_BYTES);
    uint64_t* empty = full + STAGES;             // V37.2: full lives on rank0 only
    uint64_t* sf_full = empty + STAGES;          // V37.4: per-CTA SF landing
    uint32_t* tbp = reinterpret_cast<uint32_t*>(sf_full + STAGES);

    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    const uint32_t rank = cluster_rank();

    const int num_pm = (m + 2 * BM - 1) / (2 * BM);
    const int num_pn = (n + BN - 1) / BN;
    const int pair_total = num_pm * num_pn;
    const int pair = blockIdx.x >> 1;
    if (pair >= pair_total) return;
    const int per_group = num_pn * group;
    const int group_idx = pair / per_group;
    const int first = group_idx * group;
    const int in_group = pair % per_group;
    const int span = min(group, num_pm - first);
    const int pm0 = (first + in_group % span) * (2 * BM);
    const int pn0 = (in_group / span) * BN;
    const int m0 = pm0 + (int)rank * BM;        // own A rows / D rows
    const int nh0 = pn0 + (int)rank * BH;       // own B half rows

    const int nk = (k + BK - 1) / BK;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            // V37.4: leader full[s] count 3 = leader producer's expect-tx
            // arrive + BOTH CTAs' SF relay arrives; tx = both CTAs' payload
            // bytes (2*PAY_TX) via the cta_group::2 loads. Local sf_full[s]
            // (count 1) tracks this CTA's 2048B SF image landing.
            mbar_init(&full[s], 3);
            mbar_init(&empty[s], 1);
            mbar_init(&sf_full[s], 1);
        }
        fence_mbar_init();
    }
    // prologue: zero the SFB cp2 tail-pad (bytes 1024..2047) in every stage;
    // TMA only ever writes [0,1024) => stays zero (prod pack_sf parity).
    for (int s = 0; s < STAGES; ++s) {
        uint32_t* pad = reinterpret_cast<uint32_t*>(smem + s * ST_BYTES + SFB_OFF + 1024);
        pad[tid] = 0u;
        pad[tid + 128] = 0u;
    }
    fence_proxy_async();   // publish generic-proxy pad writes to the async proxy
    if (wid == 0) {
        tmem_alloc2(tbp, 512);
        tmem_relinq2();
    }
    cluster_sync();
    __syncthreads();
    const uint32_t tb = *tbp;

    auto issue_stage = [&](int s) {
        const int buf = s % STAGES;
        uint8_t* stg = smem + buf * ST_BYTES;
        // V37.4: SF FIRST onto the LOCAL sf_full barrier (plain TMA). The
        // leader's collective cp2 reads BOTH CTAs' SF smem; DeepGEMM sm100
        // never gates cp2 sources by tx-only peer accounting - a peer thread
        // observes the local SF barrier and release-arrives on the leader.
        mbar_arrive_expect_tx(&sf_full[buf], SF_TX);
        tma_load_2d(&tmSA, &sf_full[buf], stg + SFA_OFF, m0, s);
        tma_load_2d(&tmSA, &sf_full[buf], stg + SFA_OFF + 512, m0, s);   // dup
        tma_load_2d(&tmSB, &sf_full[buf], stg + SFB_OFF,
                    (BN == 256) ? pn0 : (pn0 / BN) * (SFB_BLOCKS2 * 128), s);
        // payload: cta_group::2 loads; BOTH CTAs' tx complete rank0's full.
        tma_load_2d_cg2(&tmA, &full[buf], stg + A_OFF, s * BK, m0);
        tma_load_2d_cg2(&tmBh, &full[buf], stg + B_OFF, s * BK, nh0);
        if (rank == 0) mbar_arrive_expect_tx(&full[buf], 2 * PAY_TX);
    };

    const uint32_t idesc_base = (1u << 23) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)((2 * BM) >> 4) << 24);
    const int PRE = STAGES - 1 < nk ? STAGES - 1 : nk;
    // rank0 tid0 only: SF cp2 + 4x UMMA for stage s (shared by both cadences)
    auto issue_mma = [&](int s, int buf) {
        tc_fence_after();
        uint8_t* stg = smem + buf * ST_BYTES;
        const uint32_t sfa_col = tb + SFA_COL + 16 * (s & 1);
        const uint32_t sfb_col = sfa_col + 4;
        utccp_2cta(sfa_col, make_desc(smem_u32(stg + SFA_OFF), 8, 0, 0));
        utccp_2cta(sfb_col, make_desc(smem_u32(stg + SFB_OFF), 8, 0, 0));
        utccp_2cta(sfb_col + 4, make_desc(smem_u32(stg + SFB_OFF + 512), 8, 0, 0));
        const uint64_t da = make_desc(smem_u32(stg + A_OFF), 64, 0, 2);
        const uint64_t db = make_desc(smem_u32(stg + B_OFF), 64, 0, 2);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const uint32_t id2 = idesc_base | ((uint32_t)kk << 29) | ((uint32_t)kk << 4);
            umma_mxf8_2sm(tb, da + kk * 2, db + kk * 2, id2, sfa_col, sfb_col,
                          (s > 0 || kk > 0) ? 1u : 0u);
        }
    };
    if (tid == 32) {
        for (int i = 0; i < PRE; ++i) issue_stage(i);
    }
    if (rank == 0 && tid == 0) {
        // ---------------- issuer (rank0) -------------------------------------
        // V37.4: full[buf] (acquire.cluster) collects: both CTAs' payload tx
        // (cta_group::2 contract, atrex-fa/DeepGEMM) + BOTH relays' arrivals
        // (thread-released local SF observations). This is the DeepGEMM sm100
        // split: payload gated by tx accounting, cp2 sources gated by thread
        // relays. The tx-only SF path (V37.3) raced the ring reuse
        // deterministically (P8 probe: zero-scale rows from the first reuse
        // stage, smem correct at flip, lagged gates inert).
        for (int s = 0; s < nk; ++s) {
            const int buf = s % STAGES;
            mbar_wait_acq_cluster(&full[buf], (s / STAGES) & 1);
            issue_mma(s, buf);
            umma_commit2_mc(&empty[buf], 3);
        }
    } else if (tid == 64) {
        // ---------------- SF relays (both CTAs) ------------------------------
        // Observe the local SF landing, publish generic->async, release-arrive
        // on the leader's full[buf] (count 3 = leader expect + both relays).
        for (int s = 0; s < nk; ++s) {
            const int buf = s % STAGES;
            mbar_wait(&sf_full[buf], (s / STAGES) & 1);
            fence_proxy_async();
            if (rank == 0) mbar_arrive(&full[buf]);
            else mbar_arrive_remote(mapa_rank0(smem_u32(&full[buf])));
        }
    } else if (tid == 32) {
        // ---------------- producers (both CTAs, free-run) --------------------
        for (int sn0 = PRE; sn0 < nk; ++sn0) {
            const int bn = sn0 % STAGES;
            if (sn0 >= STAGES) mbar_wait(&empty[bn], ((sn0 - STAGES) / STAGES) & 1);
            issue_stage(sn0);
        }
    }

    // ---- epilogue: own 128 rows x BN cols (32-col blocks, guarded tails)
    const int row = m0 + wid * 32 + lane;
    CD* drow = d + (size_t)row * d_stride0;
    const CD* crow = has_c ? cptr + (size_t)row * c_stride0 : nullptr;
    // V31-style gated C prefetch while the last commit drains (B-footprint
    // gate: n*k <= 32MB, J8/J9 lesson).
    if (crow != nullptr && row < m && (int64_t)n * k <= 33554432) {
        const char* cbase = reinterpret_cast<const char*>(crow) + (size_t)pn0 * sizeof(CD);
        const int64_t row_bytes = (int64_t)min(BN, n - pn0) * sizeof(CD);
#pragma unroll
        for (int off = 0; off < BN * (int)sizeof(CD); off += 128) {
            if (off < row_bytes) asm volatile("prefetch.global.L2 [%0];" ::"l"(cbase + off));
        }
    }
    mbar_wait(&empty[(nk - 1) % STAGES], ((nk - 1) / STAGES) & 1);
    tc_fence_after();
    if (sig_mode == 3) {
        // BISECT reference: verified prod cg2 4x-group epilogue.
        for (int c0 = 0; c0 < BN / 4; c0 += 8) {
            uint32_t v[8][4];
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) tmem_ld_4x(tb + (c0 + jj) * 4, v[jj]);
            tmem_ld_wait();
            if (row < m) {
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    const int col = pn0 + (c0 + jj) * 4;
                    if ((c0 + jj) * 4 >= BN) continue;   // BN%32!=0 tail groups
                    if (col + 3 < n) {
                        float y0 = __uint_as_float(v[jj][0]) * alpha;
                        float y1 = __uint_as_float(v[jj][1]) * alpha;
                        float y2 = __uint_as_float(v[jj][2]) * alpha;
                        float y3 = __uint_as_float(v[jj][3]) * alpha;
                        if (has_c) {
                            if constexpr (sizeof(CD) == 4) {
                                const float2 c01 = *reinterpret_cast<const float2*>(crow + col);
                                const float2 c23 = *reinterpret_cast<const float2*>(crow + col + 2);
                                y0 += c01.x; y1 += c01.y; y2 += c23.x; y3 += c23.y;
                            } else {
                                const __nv_bfloat162 c01 = *reinterpret_cast<const __nv_bfloat162*>(crow + col);
                                const __nv_bfloat162 c23 = *reinterpret_cast<const __nv_bfloat162*>(crow + col + 2);
                                y0 += __bfloat162float(__low2bfloat16(c01));
                                y1 += __bfloat162float(__high2bfloat16(c01));
                                y2 += __bfloat162float(__low2bfloat16(c23));
                                y3 += __bfloat162float(__high2bfloat16(c23));
                            }
                        }
                        if constexpr (sizeof(CD) == 4) {
                            *reinterpret_cast<float2*>(drow + col) = make_float2(y0, y1);
                            *reinterpret_cast<float2*>(drow + col + 2) = make_float2(y2, y3);
                        } else {
                            __nv_bfloat162 p01 = {__float2bfloat16_rn(y0), __float2bfloat16_rn(y1)};
                            __nv_bfloat162 p23 = {__float2bfloat16_rn(y2), __float2bfloat16_rn(y3)};
                            *reinterpret_cast<__nv_bfloat162*>(drow + col) = p01;
                            *reinterpret_cast<__nv_bfloat162*>(drow + col + 2) = p23;
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            if (col + i < n) {
                                float y = __uint_as_float(v[jj][i]) * alpha;
                                if (has_c) y += cd2f<CD>(crow[col + i]);
                                drow[col + i] = cd_from_f32<CD>(y);
                            }
                        }
                    }
                }
            }
        }
    } else {
    const bool wide_ok = ((d_stride0 * (int64_t)sizeof(CD)) % 16 == 0) &&
                         ((uintptr_t)d % 16 == 0) &&
                         (!has_c || ((c_stride0 * (int64_t)sizeof(CD)) % 16 == 0 &&
                                     (uintptr_t)cptr % 16 == 0));
    const int ntot = BN / 32 + ((BN % 32) ? 1 : 0);
    for (int ii = 0; ii < ntot; ++ii) {
        const int c0 = ii * 32;
        const bool tail_blk = (c0 + 32 > BN);   // BN=240 remainder (16 cols)
        uint32_t v[32];
        if (!tail_blk) {
            tmem_ld_32x(tb + c0, v);
        } else {
            uint32_t (*vg)[4] = reinterpret_cast<uint32_t (*)[4]>(v);
            for (int j = 0; 4 * j < BN - c0; ++j) tmem_ld_4x(tb + c0 + 4 * j, vg[j]);
        }
        tmem_ld_wait();
        if (row < m) {
            const int col = pn0 + c0;
            if (!tail_blk && col + 31 < n) {
                epi_store32<CD>(drow, crow, col, alpha, has_c, wide_ok, v);
            } else {
                uint32_t (*vg)[4] = reinterpret_cast<uint32_t (*)[4]>(v);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int cg = col + j * 4;
                    if (c0 + 4 * j >= BN) break;   // past BN: tmem SF cols
                    if (cg + 3 < n) {
                        float y0 = __uint_as_float(vg[j][0]) * alpha;
                        float y1 = __uint_as_float(vg[j][1]) * alpha;
                        float y2 = __uint_as_float(vg[j][2]) * alpha;
                        float y3 = __uint_as_float(vg[j][3]) * alpha;
                        if (has_c) {
                            if constexpr (sizeof(CD) == 4) {
                                const float2 c01 = *reinterpret_cast<const float2*>(crow + cg);
                                const float2 c23 = *reinterpret_cast<const float2*>(crow + cg + 2);
                                y0 += c01.x; y1 += c01.y; y2 += c23.x; y3 += c23.y;
                            } else {
                                const __nv_bfloat162 c01 = *reinterpret_cast<const __nv_bfloat162*>(crow + cg);
                                const __nv_bfloat162 c23 = *reinterpret_cast<const __nv_bfloat162*>(crow + cg + 2);
                                y0 += __bfloat162float(__low2bfloat16(c01));
                                y1 += __bfloat162float(__high2bfloat16(c01));
                                y2 += __bfloat162float(__low2bfloat16(c23));
                                y3 += __bfloat162float(__high2bfloat16(c23));
                            }
                        }
                        if constexpr (sizeof(CD) == 4) {
                            *reinterpret_cast<float2*>(drow + cg) = make_float2(y0, y1);
                            *reinterpret_cast<float2*>(drow + cg + 2) = make_float2(y2, y3);
                        } else {
                            __nv_bfloat162 p01 = {__float2bfloat16_rn(y0), __float2bfloat16_rn(y1)};
                            __nv_bfloat162 p23 = {__float2bfloat16_rn(y2), __float2bfloat16_rn(y3)};
                            *reinterpret_cast<__nv_bfloat162*>(drow + cg) = p01;
                            *reinterpret_cast<__nv_bfloat162*>(drow + cg + 2) = p23;
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            if (cg + i < n) {
                                float y = __uint_as_float(vg[j][i]) * alpha;
                                if (has_c) y += cd2f<CD>(crow[cg + i]);
                                drow[cg + i] = cd_from_f32<CD>(y);
                            }
                        }
                    }
                }
            }
        }
    }
    }   // end 32x-block epilogue (sig_mode 0)
    cluster_sync();
    __syncthreads();
    if (wid == 0) tmem_dealloc2(tb, 512);
}

// ---------------------------------------------------------------- pdb kernel
// V3.9 arm: persistent double-buffer (pdb240). 256 threads, 1 CTA/SM.
// Stage crew (tid<128) runs the V3.6 pipeline (MMA-ahead issuer tid0, TMA
// producer tid32, SF tmem slots) over a GLOBALLY CONTINUOUS stage index that
// crosses tile boundaries — the pipeline never drains between tiles. Epilogue
// crew (wids 4-7) drains finished tiles from alternating tmem D buffers
// (D0 cols 0..BN-1, D1 BN..2BN-1, SF slots at 2BN+16*(g&1)) concurrently with
// the next tile's MMAs. Handshakes: commit1(epi_ready[t&1]) at tile end;
// 4-warp arrive epi_free[t&1]; issuer waits epi_free[t&1] at tile t+2 start.
// Mechanism provenance: arm-b reference architecture (join 20261022) showed
// 1.53x on c2 via tile-overlap; this is an independent TMA/UTCCP realization.
template <int STAGES, int BN, typename CD, bool DBL_D = true, bool PP = false, bool EE = false>
__global__ void __launch_bounds__(256, 1)
gemm_pdb(int m, int n, int k, float alpha, int group,
         const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
         const __grid_constant__ CUtensorMap tmSA, const __grid_constant__ CUtensorMap tmSB,
         CD* __restrict__ d, int64_t d_stride0,
         const CD* __restrict__ cptr, int64_t c_stride0, int has_c) {
    constexpr int BM = 128, BK = 128;
    constexpr int SFB_BLOCKS = (BN + 127) / 128;   // ceil: BN=240 -> 2 blocks
    constexpr int ST_RAW = BM * BK + BN * BK + (BM + BN) * 16 + (BM + BN) * 4;
    constexpr int ST_BYTES = (ST_RAW + 1023) & ~1023;
    constexpr int A_OFF = 0;
    constexpr int B_OFF = BM * BK;
    constexpr int SAF_OFF = B_OFF + BN * BK;
    constexpr int SBF_OFF = SAF_OFF + BM * 16;
    constexpr int SFA_OFF = SBF_OFF + BN * 16;
    constexpr int SFB_OFF = SFA_OFF + BM * 4;
    constexpr uint32_t TX_BYTES = BM * BK + BN * BK + BM * 16 + BN * 16;
    constexpr uint32_t SF_TX = BM * 16 + BN * 16;          // V5 split barrier
    constexpr uint32_t AB_TX = TX_BYTES - SF_TX;
    // V25 PP: merged TMA batch = A + B + prepacked SF images (512B SFA +
    // 512B*SFB_BLOCKS SFB, landing byte-identical to pack_sf output).
    // ST_BYTES unchanged (fp32 SF regions dead but reserved => identical
    // smem footprint/occupancy/launch attrs).
    constexpr uint32_t TX_PP = BM * BK + BN * BK + 512 + SFB_BLOCKS * 512;
    // V10 overlap mode (OVL): DBL_D BN256 in 512 cols via the DeepGEMM
    // compressed-stride trick - D buffers at 0..255 / 240..495 (shared cols
    // 240..255), SF single 16-col set at 496..511 (same-thread in-order TC
    // pipe makes the single set safe: utccp(g) executes after MMA(g-1);
    // DeepGEMM e1f418c2 runs exactly this pattern in production).
    constexpr bool OVL = (DBL_D && BN == 256);
    constexpr int DSTRIDE = OVL ? 240 : BN;
    constexpr int SF_BASE = OVL ? 496 : (DBL_D ? 2 * BN : BN);   // SFA@+0, SFB@+4..
    constexpr int TMEM_COLS = 512;             // 1 CTA/SM; 2*BN+16 <= 288 (BN128)

    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + STAGES * ST_BYTES);
    uint64_t* empty = full + STAGES;
    uint64_t* epi_ready = empty + STAGES;      // [2] commit-arrived
    uint64_t* epi_free = epi_ready + 2;        // [2] count 4 (epilogue warps)
    uint64_t* fullsf = epi_free + 2;           // SF TMA arrivals (V5)
    uint64_t* sfready = fullsf + STAGES;       // packer arrivals, 64 (V5)
    uint64_t* ovl = sfready + STAGES;          // [2] V10 overlap-block release
    uint32_t* tbp = reinterpret_cast<uint32_t*>(ovl + 2);

    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    const bool crew = (tid < 128);

    const int num_m_blocks = (m + BM - 1) / BM;
    const int num_n_blocks = (n + BN - 1) / BN;
    const int total = num_m_blocks * num_n_blocks;
    const int G = gridDim.x;
    const int TL = (total > G) ? (int)((total - (int)blockIdx.x + G - 1) / G)
                               : ((int)blockIdx.x < total ? 1 : 0);
    const int nk = (k + BK - 1) / BK;
    const int GT = nk * TL;                    // global stage count for this CTA

    // raster map: tile_id -> (m0, n0), same group-M swizzle as non-persistent
    auto tile_geom = [&](int tile_id, int& m0, int& n0) {
        const int per_group = num_n_blocks * group;
        const int group_idx = tile_id / per_group;
        const int first = group_idx * group;
        const int in_group = tile_id % per_group;
        const int span = min(group, num_m_blocks - first);
        m0 = (first + in_group % span) * BM;
        n0 = (in_group / span) * BN;
    };

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], 1); mbar_init(&empty[s], 1);
            if constexpr (!PP) { mbar_init(&fullsf[s], 1); mbar_init(&sfready[s], 64); }
        }
        mbar_init(&epi_ready[0], 1); mbar_init(&epi_ready[1], 1);
        mbar_init(&epi_free[0], 4); mbar_init(&epi_free[1], 4);
        if (OVL) { mbar_init(&ovl[0], 4); mbar_init(&ovl[1], 4); }
        fence_mbar_init();
    }
    if (wid == 0) {
        tmem_alloc1(tbp, TMEM_COLS);
        tmem_relinq1();
    }
    __syncthreads();
    const uint32_t tb = *tbp;
    const uint32_t idesc_base = (1u << 23) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)(BM >> 4) << 24);

    if (crew) {
        // ---------------- stage crew (V5: 64-thread packers, row-mapped) --
        auto pack_sf = [&](int buf, int ptid) {
            uint8_t* stg = smem + buf * ST_BYTES;
            uint32_t* sfa_p = reinterpret_cast<uint32_t*>(stg + SFA_OFF);
            uint32_t* sfb_p = reinterpret_cast<uint32_t*>(stg + SFB_OFF);
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int r = ptid + rr * 64;
                const float4 fa = *reinterpret_cast<const float4*>(stg + SAF_OFF + r * 16);
                sfa_p[4 * (r & 31) + (r >> 5)] =
                    ue8m0(fa.x) | (ue8m0(fa.y) << 8) | (ue8m0(fa.z) << 16) | (ue8m0(fa.w) << 24);
#pragma unroll
                for (int b = 0; b < SFB_BLOCKS; ++b) {
                    const int row = b * 128 + r;
                    uint32_t wb = 0u;
                    if (row < BN) {
                        const float4 fb = *reinterpret_cast<const float4*>(stg + SBF_OFF + row * 16);
                        wb = ue8m0(fb.x) | (ue8m0(fb.y) << 8) | (ue8m0(fb.z) << 16) | (ue8m0(fb.w) << 24);
                    }
                    sfb_p[b * 128 + 4 * (r & 31) + (r >> 5)] = wb;   // rows >= BN: zero tail
                }
            }
        };
        auto issue_stage = [&](int g_stage, int s_tile, int m0, int n0) {
            const int buf = g_stage % STAGES;   // GLOBAL stage index for buffer
            uint8_t* stg = smem + buf * ST_BYTES;
            if constexpr (PP) {
                // V25: one merged batch; SF arrives as ready UTCCP images
                // (coords: word-row m0/n0, kb-row s_tile).
                mbar_arrive_expect_tx(&full[buf], TX_PP);
                tma_load_2d(&tmA, &full[buf], stg + A_OFF, s_tile * BK, m0);
                tma_load_2d(&tmB, &full[buf], stg + B_OFF, s_tile * BK, n0);
                tma_load_2d(&tmSA, &full[buf], stg + SFA_OFF, m0, s_tile);
                // J26: BN256 = absolute 128-block prepack (x = n0 words);
                // BN224 = tile-relative prepack (nt224 pitch, 256 words/tile)
                // => x = tile_idx * 256. Constexpr fold: BN256 unchanged.
                tma_load_2d(&tmSB, &full[buf], stg + SFB_OFF,
                            (BN == 256) ? n0 : (n0 / BN) * (SFB_BLOCKS * 128), s_tile);
            } else {
                mbar_arrive_expect_tx(&fullsf[buf], SF_TX);   // V5: SF first
                tma_load_2d(&tmSA, &fullsf[buf], stg + SAF_OFF, s_tile * 4, m0);
                tma_load_2d(&tmSB, &fullsf[buf], stg + SBF_OFF, s_tile * 4, n0);
                mbar_arrive_expect_tx(&full[buf], AB_TX);
                tma_load_2d(&tmA, &full[buf], stg + A_OFF, s_tile * BK, m0);
                tma_load_2d(&tmB, &full[buf], stg + B_OFF, s_tile * BK, n0);
            }
        };
        // producer state (tid32): next-to-issue global stage g2
        int g2 = 0, t2 = 0, s2 = 0, m0_2 = 0, n0_2 = 0;
        if (tid == 32 && GT > 0) tile_geom(blockIdx.x, m0_2, n0_2);
        const int PRE = STAGES - 1 < GT ? STAGES - 1 : GT;
        if (tid == 32) {
            for (; g2 < PRE; ++g2) {
                issue_stage(g2, s2, m0_2, n0_2);
                if (++s2 == nk) { s2 = 0; ++t2; tile_geom(blockIdx.x + t2 * G, m0_2, n0_2); }
            }
        }
        // V5 role-split (DeepGEMM e1f418c2 cadence): tid0 = issuer (keeps
        // the DBL_D epi_free gate and s1/t1 tile tracking), tid32 =
        // free-run producer, warps 2-3 = 64 SF packers on fullsf/sfready.
        // bar_sync128 removed from the chain path.
        if (tid == 0) {
            // ---------------- issuer ----------------
            int t1 = 0, s1 = 0;
            for (int g = 0; g < GT; ++g) {
                const int buf = g % STAGES;
                mbar_wait(&full[buf], (g / STAGES) & 1);
                if constexpr (!PP) mbar_wait(&sfready[buf], (g / STAGES) & 1);  // V25
                if (DBL_D) {
                    if (OVL && s1 == 0 && t1 >= 1) {
                        // V10 early-start gate: tile t1-1 (other buffer) shares
                        // cols 240..255; its FIRST-stored block released ovl
                        // (tmem values already in the epi warps' registers).
                        // 0-based phase index of tile t1-1 on that buffer =
                        // ((t1-1)>>1), matching the epi_ready convention.
                        mbar_wait(&ovl[(t1 - 1) & 1], ((t1 - 1) >> 1) & 1);
                    }
                    if (s1 == 0 && t1 >= 2) {
                        // D[t1&1] last drained by epilogue(t1-2)
                        mbar_wait(&epi_free[t1 & 1], ((t1 >> 1) - 1) & 1);
                    }
                } else if (s1 == 0 && t1 >= 1) {
                    // single D: epilogue(t1-1) must have drained D. It arrives
                    // on epi_free[(t1-1)&1] (barrier follows the EPILOGUE's tile
                    // parity, not the issuer's) as arrival #((t1-1)>>1)+1 =>
                    // phase parity ((t1-1)>>1)&1. (Wrong index = t1&1 deadlocks
                    // at t=1: dv_3a09e963eacd hung case2 to budget kill.)
                    mbar_wait(&epi_free[(t1 - 1) & 1], ((t1 - 1) >> 1) & 1);
                }
                tc_fence_after();
                uint8_t* stg = smem + buf * ST_BYTES;
                const uint32_t dcol = tb + (DBL_D ? DSTRIDE * (t1 & 1) : 0);
                const uint32_t sfa_col = tb + SF_BASE + (OVL ? 0 : 16 * (g & 1));
                const uint32_t sfb_col = sfa_col + 4;
                utccp_4x32dp128bit_1cta(sfa_col, make_desc(smem_u32(stg + SFA_OFF), 8, 0, 0));
#pragma unroll
                for (int b = 0; b < SFB_BLOCKS; ++b)
                    utccp_4x32dp128bit_1cta(sfb_col + 4 * b,
                                            make_desc(smem_u32(stg + SFB_OFF + 512 * b), 8, 0, 0));
                const uint64_t da = make_desc(smem_u32(stg + A_OFF), 64, 0, 2);
                const uint64_t db = make_desc(smem_u32(stg + B_OFF), 64, 0, 2);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint32_t id2 = idesc_base | ((uint32_t)kk << 29) | ((uint32_t)kk << 4);
                    umma_mxf8_1sm(dcol, da + kk * 2, db + kk * 2, id2, sfa_col, sfb_col,
                                  (s1 > 0 || kk > 0) ? 1u : 0u);
                }
                if (s1 == nk - 1) umma_commit1(&epi_ready[t1 & 1]);
                umma_commit1(&empty[buf]);
                if (++s1 == nk) { s1 = 0; ++t1; }
            }
        } else if (tid == 32) {
            // ---------------- producer (free-run, commit-gated) -------------
            for (; g2 < GT; ++g2) {
                if (g2 >= STAGES) {
                    const int bn = g2 % STAGES;
                    mbar_wait(&empty[bn], ((g2 - STAGES) / STAGES) & 1);
                }
                issue_stage(g2, s2, m0_2, n0_2);
                if (++s2 == nk) { s2 = 0; ++t2; if (t2 < TL) tile_geom(blockIdx.x + t2 * G, m0_2, n0_2); }
            }
        } else if (wid == 2 || wid == 3) {
            // ---------------- packers (64 threads) -------------------------
            if constexpr (!PP) {   // V25: prepacked SF - no in-kernel packing
                const int ptid = tid - 64;
                for (int g = 0; g < GT; ++g) {
                    const int buf = g % STAGES;
                    mbar_wait(&fullsf[buf], (g / STAGES) & 1);
                    pack_sf(buf, ptid);
                    fence_proxy_async();
                    mbar_arrive(&sfready[buf]);
                }
            }
        }
    } else {
        // ---------------- epilogue crew (wids 4-7) ----------------
        const int ews = wid - 4;                       // 0..3
        const int row_in_tile = ews * 32 + lane;
        const bool wide_ok = ((d_stride0 * (int64_t)sizeof(CD)) % 16 == 0) &&
                             ((uintptr_t)d % 16 == 0) &&
                             (!has_c || ((c_stride0 * (int64_t)sizeof(CD)) % 16 == 0 &&
                                         (uintptr_t)cptr % 16 == 0));
        for (int t = 0; t < TL; ++t) {
            const int b = t & 1;
            // V31 gated C-prefetch (J8/J9): warm this tile's C row segment
            // into L2 during the epi_ready spin so drain C reads are L2
            // hits. GATED on the B footprint (n*k <= 32MB): J8 measured the
            // ungated form winning c2 (B=21MB) -2% but LOSING route-acc
            // (B=50MB) +2.2% - C prefetch evicts the B stream from L2 when
            // B is large. Non-binding hint; alias-safe; no-C cells inert.
            int m0, n0;
            tile_geom(blockIdx.x + t * G, m0, n0);
            const int row = m0 + row_in_tile;
            CD* drow = d + (size_t)row * d_stride0;
            const CD* crow = has_c ? cptr + (size_t)row * c_stride0 : nullptr;
            if (crow != nullptr && row < m && (int64_t)n * k <= 33554432) {
                const char* cbase =
                    reinterpret_cast<const char*>(crow) + (size_t)n0 * sizeof(CD);
                const int64_t row_bytes = (int64_t)min(BN, n - n0) * sizeof(CD);
#pragma unroll
                for (int off = 0; off < BN * (int)sizeof(CD); off += 128) {
                    if (off < row_bytes) {
                        asm volatile("prefetch.global.L2 [%0];" ::"l"(cbase + off));
                    }
                }
            }
            mbar_wait(&epi_ready[b], (t >> 1) & 1);
            tc_fence_after();
            const uint32_t dbase = tb + (DBL_D ? DSTRIDE * b : 0);
            const int nblk = BN / 32;                  // full 32-col blocks
            const int ntot = nblk + ((BN % 32) ? 1 : 0);   // + remainder block (240)
            // V10 (OVL): buf0 drains REVERSED, buf1 forward => the shared
            // cols 240..255 are the FIRST block handled in both cases; ovl[b]
            // releases after that block's tmem_ld_wait so the next tile's MMA
            // starts while the remaining drain overlaps the chain. ntot loop
            // is break-equivalent for 224/240 (audited).
            const bool rev = (OVL && (b == 0));
            // V41: pipelined drain, HAS_C-GATED (J28: the win source is the
            // exposed C-read latency; no-C cells regressed +1.1% under the
            // lookahead-only path). bf16: tmem_ld + C-register lookahead
            // (dedicated helper); f32-C: tmem-lookahead only, stores via the
            // UNTOUCHED epi_store32. WARP-UNIFORM gate (tmem_ld_32x is
            // warp-collective); boundary warps / non-OVL / no-C arms take the
            // serial loop below (byte-identical semantics; OVL constexpr
            // folds this path away for non-OVL instantiations).
            // V41b (J29: f32-C tmem-lookahead measured INERT +-0.05% over 4
            // passes; restricted to bf16-C => every non-winning cell is
            // byte-identical to v36 by construction).
            const bool v41_fast = OVL && has_c && (sizeof(CD) == 2) &&
                                  (ntot >= 2) && wide_ok &&
                                  (m0 + ews * 32 + 31 < m) && (n0 + BN <= n);
            if (v41_fast) {
                const int nb = ntot;
                constexpr bool cq_on = (sizeof(CD) == 2);   // true in fast path (V41b)
                uint32_t vv[2][32];
                uint4 cq[2][4];
                auto blk = [&](int ii) { return rev ? (nb - 1 - ii) : ii; };
                tmem_ld_32x(dbase + blk(0) * 32, vv[0]);
                tmem_ld_wait();
                __syncwarp();
                if (lane == 0) mbar_arrive(&ovl[b]);
                if (cq_on) {
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        cq[0][i] = *reinterpret_cast<const uint4*>(
                            crow + n0 + blk(0) * 32 + 8 * i);
                }
                int par = 0;
                for (int ii = 0; ii < nb; ++ii) {
                    if (ii + 1 < nb) {
                        tmem_ld_32x(dbase + blk(ii + 1) * 32, vv[par ^ 1]);
                        if (cq_on) {
#pragma unroll
                            for (int i = 0; i < 4; ++i)
                                cq[par ^ 1][i] = *reinterpret_cast<const uint4*>(
                                    crow + n0 + blk(ii + 1) * 32 + 8 * i);
                        }
                    }
                    // V39.6 EE (prod port): tmem fully read after the last
                    // tmem_ld_wait; release D before the final store block.
                    if constexpr (EE) {
                        if (ii == nb - 1) {
                            __syncwarp();
                            if (lane == 0) mbar_arrive(&epi_free[b]);
                        }
                    }
                    epi_store32_cq<CD>(drow, n0 + blk(ii) * 32, alpha,
                                       vv[par], cq[par]);
                    if (ii + 1 < nb) {
                        tmem_ld_wait();
                        par ^= 1;
                    }
                }
            }
            if (!v41_fast)
            for (int ii = 0; ii < ntot; ++ii) {
                const int bi = rev ? (ntot - 1 - ii) : ii;
                const int c0 = bi * 32;
                const bool tail_blk = (c0 + 32 > BN);  // BN%32 remainder (240)
                uint32_t v[32];
                if (!tail_blk) {
                    tmem_ld_32x(dbase + c0, v);
                } else {
                    uint32_t (*vg)[4] = reinterpret_cast<uint32_t (*)[4]>(v);
                    for (int j = 0; 4 * j < BN - c0; ++j) tmem_ld_4x(dbase + c0 + 4 * j, vg[j]);
                }
                tmem_ld_wait();
                if (OVL && ii == 0) {
                    __syncwarp();
                    if (lane == 0) mbar_arrive(&ovl[b]);
                }
                if constexpr (EE) {
                    if (ii == ntot - 1) {
                        __syncwarp();
                        if (lane == 0) mbar_arrive(&epi_free[b]);
                    }
                }
                if (row < m) {
                    const int col = n0 + c0;
                    if (!tail_blk && col + 31 < n) {
                        epi_store32<CD>(drow, crow, col, alpha, has_c, wide_ok, v);
                    } else {
                        uint32_t (*vg)[4] = reinterpret_cast<uint32_t (*)[4]>(v);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int cg = col + j * 4;
                            if (c0 + 4 * j >= BN) break;   // past BN: tmem SF cols
                            if (cg + 3 < n) {
                                float y0 = __uint_as_float(vg[j][0]) * alpha;
                                float y1 = __uint_as_float(vg[j][1]) * alpha;
                                float y2 = __uint_as_float(vg[j][2]) * alpha;
                                float y3 = __uint_as_float(vg[j][3]) * alpha;
                                if (has_c) {
                                    if constexpr (sizeof(CD) == 4) {
                                        const float2 c01 = *reinterpret_cast<const float2*>(crow + cg);
                                        const float2 c23 = *reinterpret_cast<const float2*>(crow + cg + 2);
                                        y0 += c01.x; y1 += c01.y; y2 += c23.x; y3 += c23.y;
                                    } else {
                                        const __nv_bfloat162 c01 = *reinterpret_cast<const __nv_bfloat162*>(crow + cg);
                                        const __nv_bfloat162 c23 = *reinterpret_cast<const __nv_bfloat162*>(crow + cg + 2);
                                        y0 += __bfloat162float(__low2bfloat16(c01));
                                        y1 += __bfloat162float(__high2bfloat16(c01));
                                        y2 += __bfloat162float(__low2bfloat16(c23));
                                        y3 += __bfloat162float(__high2bfloat16(c23));
                                    }
                                }
                                if constexpr (sizeof(CD) == 4) {
                                    *reinterpret_cast<float2*>(drow + cg) = make_float2(y0, y1);
                                    *reinterpret_cast<float2*>(drow + cg + 2) = make_float2(y2, y3);
                                } else {
                                    __nv_bfloat162 p01 = {__float2bfloat16_rn(y0), __float2bfloat16_rn(y1)};
                                    __nv_bfloat162 p23 = {__float2bfloat16_rn(y2), __float2bfloat16_rn(y3)};
                                    *reinterpret_cast<__nv_bfloat162*>(drow + cg) = p01;
                                    *reinterpret_cast<__nv_bfloat162*>(drow + cg + 2) = p23;
                                }
                            } else {
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    if (cg + i < n) {
                                        float y = __uint_as_float(vg[j][i]) * alpha;
                                        if (has_c) y += cd2f<CD>(crow[cg + i]);
                                        drow[cg + i] = cd_from_f32<CD>(y);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if constexpr (!EE) __syncwarp();
            if constexpr (!EE) if (lane == 0) mbar_arrive(&epi_free[b]);
        }
    }
    __syncthreads();
    if (wid == 0) tmem_dealloc1(tb, TMEM_COLS);
}

// ------------------------------------------------------------- V38 pdbmc kernel
// Persistent double-buffer BN256 (OVL) with CLUSTER(2,1) B-MULTICAST feed.
// Motivation (probe evidence): heavy classes run ~457ns/stage on PP-pdb256
// (49.5KB/CTA-stage vs 356ns L2 floor / 268.8ns TC floor); the 2-SM MMA route
// (V37.x cg2pp) hit device-level TC parity but never passed the ring-reuse
// numerics (P8/P9 bisects: rows-64+ zero-scale corruption from the first
// reuse stage even in minimal 1-CTA skeletons - parked with evidence).
// pdbmc keeps EVERY production-proven element of gemm_pdb<4,256,CD,true,true>
// (256 threads, persistent global stage index, OVL dual-D at 240 stride, SF
// single-slot at 496, padded 57344 stage layout, V41 fast drain, V31 gated C
// prefetch, per-CTA LOCAL cp1/UMMA/commit) and adds only sm90-mature cluster
// mechanics: a pair of CTAs processes adjacent m-blocks of the SAME n-block;
// rank0 issues the shared B stage ONCE via TMA multicast (mask 3) so each
// receiver's local full barrier collects the tx (B L2 bytes per SM halve:
// 49.5 -> ~33.5KB/stage); B-buffer reuse is gated on rank0's bempty barrier
// (count 2 = rank0 local commit + rank1 relay of its local commit mirror);
// rank1's A/SF reuse stays gated by its own local commit. All per-CTA tmem,
// SF cp1 sources and D drains remain CTA-local: no collective cp2, no 2-SM
// UMMA, no tx-only peer accounting on cp-consumed data (the exact structure
// the P8/P9 bisects showed production relies on).
template <int STAGES, int BN, typename CD>
__global__ void __launch_bounds__(256, 1) __cluster_dims__(2, 1, 1)
gemm_pdbmc(int m, int n, int k, float alpha, int group,
           const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
           const __grid_constant__ CUtensorMap tmSA, const __grid_constant__ CUtensorMap tmSB,
           CD* __restrict__ d, int64_t d_stride0,
           const CD* __restrict__ cptr, int64_t c_stride0, int has_c) {
    constexpr int BM = 128, BK = 128;
    constexpr int SFB_BLOCKS = (BN + 127) / 128;
    constexpr int ST_RAW = BM * BK + BN * BK + (BM + BN) * 16 + (BM + BN) * 4;
    constexpr int ST_BYTES = (ST_RAW + 1023) & ~1023;      // 57344 for BN256
    constexpr int A_OFF = 0;
    constexpr int B_OFF = BM * BK;
    constexpr int SAF_OFF = B_OFF + BN * BK;               // dead (PP), reserved
    constexpr int SBF_OFF = SAF_OFF + BM * 16;             // dead (PP), reserved
    constexpr int SFA_OFF = SBF_OFF + BN * 16;
    constexpr int SFB_OFF = SFA_OFF + BM * 4;
    constexpr uint32_t TX_PP = BM * BK + BN * BK + 512 + SFB_BLOCKS * 512;
    constexpr int DSTRIDE = 240;                            // OVL compressed stride
    constexpr int SF_BASE = 496;                            // single 16-col set
    constexpr int TMEM_COLS = 512;

    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + STAGES * ST_BYTES);
    uint64_t* empty = full + STAGES;          // rank1-local commit gate
    uint64_t* bempty = empty + STAGES;        // rank0: B-reuse gate (count 2)
    uint64_t* bmirror = bempty + STAGES;      // rank1 commit mirror (count 1)
    uint64_t* epi_ready = bmirror + STAGES;   // [2]
    uint64_t* epi_free = epi_ready + 2;       // [2] count 4
    uint64_t* ovl = epi_free + 2;             // [2] count 4
    uint32_t* tbp = reinterpret_cast<uint32_t*>(ovl + 2);

    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    const bool crew = (tid < 128);
    const uint32_t rank = cluster_rank();

    const int num_m_blocks = (m + BM - 1) / BM;
    const int num_n_blocks = (n + BN - 1) / BN;
    const int num_m_pairs = (num_m_blocks + 1) / 2;
    const int pair_total = num_m_pairs * num_n_blocks;
    const int Gp = gridDim.x >> 1;
    const int pair = blockIdx.x >> 1;
    if (pair >= pair_total) return;
    const int TL = (pair_total > Gp) ? (int)((pair_total - pair + Gp - 1) / Gp)
                                     : (pair < pair_total ? 1 : 0);
    const int nk = (k + BK - 1) / BK;
    const int GT = nk * TL;

    // pair raster: tile pt -> own (m0, n0); m-pair adjacent halves per rank.
    auto pair_tile_geom = [&](int pt, int& m0, int& n0) {
        const int per_group = num_n_blocks * group;
        const int group_idx = pt / per_group;
        const int first = group_idx * group;
        const int in_group = pt % per_group;
        const int span = min(group, num_m_pairs - first);
        const int pm = first + in_group % span;
        m0 = (2 * pm + (int)rank) * BM;
        n0 = (in_group / span) * BN;
    };

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], 1); mbar_init(&empty[s], 1);
            mbar_init(&bempty[s], 2); mbar_init(&bmirror[s], 1);
        }
        mbar_init(&epi_ready[0], 1); mbar_init(&epi_ready[1], 1);
        mbar_init(&epi_free[0], 4); mbar_init(&epi_free[1], 4);
        mbar_init(&ovl[0], 4); mbar_init(&ovl[1], 4);
        fence_mbar_init();
    }
    if (wid == 0) {
        tmem_alloc1(tbp, TMEM_COLS);
        tmem_relinq1();
    }
    cluster_sync();     // pair-wide barrier-init visibility BEFORE any TMA
    __syncthreads();
    const uint32_t tb = *tbp;
    const uint32_t idesc_base = (1u << 23) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)(BM >> 4) << 24);

    if (crew) {
        auto issue_stage = [&](int g_stage, int s_tile, int m0, int n0) {
            const int buf = g_stage % STAGES;   // GLOBAL stage index for buffer
            uint8_t* stg = smem + buf * ST_BYTES;
            // PP batch: own A + SF images local; B shared via rank0 multicast
            // whose tx lands on BOTH CTAs' full[buf] at the same offset.
            mbar_arrive_expect_tx(&full[buf], TX_PP);
            tma_load_2d(&tmA, &full[buf], stg + A_OFF, s_tile * BK, m0);
            tma_load_2d(&tmSA, &full[buf], stg + SFA_OFF, m0, s_tile);
            tma_load_2d(&tmSB, &full[buf], stg + SFB_OFF, n0, s_tile);
            if (rank == 0)
                tma_load_2d_mc(&tmB, &full[buf], stg + B_OFF, s_tile * BK, n0, 3);
        };
        int g2 = 0, t2 = 0, s2 = 0, m0_2 = 0, n0_2 = 0;
        if (tid == 32 && GT > 0) pair_tile_geom(pair, m0_2, n0_2);
        const int PRE = STAGES - 1 < GT ? STAGES - 1 : GT;
        if (tid == 32) {
            for (; g2 < PRE; ++g2) {
                issue_stage(g2, s2, m0_2, n0_2);
                if (++s2 == nk) { s2 = 0; ++t2; pair_tile_geom(pair + t2 * Gp, m0_2, n0_2); }
            }
        }
        if (tid == 0) {
            // ---------------- issuer (per CTA, own tile) ----------------------
            int t1 = 0, s1 = 0;
            for (int g = 0; g < GT; ++g) {
                const int buf = g % STAGES;
                mbar_wait(&full[buf], (g / STAGES) & 1);
                if (s1 == 0 && t1 >= 1) {
                    // OVL early-start gate (shared cols 240..255 of tile t1-1)
                    mbar_wait(&ovl[(t1 - 1) & 1], ((t1 - 1) >> 1) & 1);
                }
                if (s1 == 0 && t1 >= 2) {
                    mbar_wait(&epi_free[t1 & 1], ((t1 >> 1) - 1) & 1);
                }
                tc_fence_after();
                uint8_t* stg = smem + buf * ST_BYTES;
                const uint32_t dcol = tb + DSTRIDE * (t1 & 1);
                const uint32_t sfa_col = tb + SF_BASE;
                const uint32_t sfb_col = sfa_col + 4;
                utccp_4x32dp128bit_1cta(sfa_col, make_desc(smem_u32(stg + SFA_OFF), 8, 0, 0));
#pragma unroll
                for (int b = 0; b < SFB_BLOCKS; ++b)
                    utccp_4x32dp128bit_1cta(sfb_col + 4 * b,
                                            make_desc(smem_u32(stg + SFB_OFF + 512 * b), 8, 0, 0));
                const uint64_t da = make_desc(smem_u32(stg + A_OFF), 64, 0, 2);
                const uint64_t db = make_desc(smem_u32(stg + B_OFF), 64, 0, 2);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint32_t id2 = idesc_base | ((uint32_t)kk << 29) | ((uint32_t)kk << 4);
                    umma_mxf8_1sm(dcol, da + kk * 2, db + kk * 2, id2, sfa_col, sfb_col,
                                  (s1 > 0 || kk > 0) ? 1u : 0u);
                }
                if (s1 == nk - 1) umma_commit1(&epi_ready[t1 & 1]);
                if (rank == 0) umma_commit1(&bempty[buf]);
                else { umma_commit1(&empty[buf]); umma_commit1(&bmirror[buf]); }
                if (++s1 == nk) { s1 = 0; ++t1; }
            }
        } else if (tid == 32) {
            // ---------------- producer (free-run) -----------------------------
            for (; g2 < GT; ++g2) {
                if (g2 >= STAGES) {
                    const int bn = g2 % STAGES;
                    if (rank == 0) mbar_wait(&bempty[bn], ((g2 - STAGES) / STAGES) & 1);
                    else mbar_wait(&empty[bn], ((g2 - STAGES) / STAGES) & 1);
                }
                issue_stage(g2, s2, m0_2, n0_2);
                if (++s2 == nk) { s2 = 0; ++t2; if (t2 < TL) pair_tile_geom(pair + t2 * Gp, m0_2, n0_2); }
            }
        } else if (rank == 1 && tid == 64) {
            // ---------------- rank1 commit relay ------------------------------
            // Local commit observation -> remote arrive on rank0's bempty:
            // gates the multicast B overwrite of rank1's buf (thread-released,
            // the structure DeepGEMM uses for peer-consumed smem; tx-only
            // peer accounting was P8/P9's corruption source).
            for (int g = 0; g < GT; ++g) {
                const int buf = g % STAGES;
                mbar_wait(&bmirror[buf], (g / STAGES) & 1);
                mbar_arrive_remote(mapa_rank0(smem_u32(&bempty[buf])));
            }
        }
    } else {
        // ---------------- epilogue crew (wids 4-7, own rows) ------------------
        const int ews = wid - 4;
        const int row_in_tile = ews * 32 + lane;
        const bool wide_ok = ((d_stride0 * (int64_t)sizeof(CD)) % 16 == 0) &&
                             ((uintptr_t)d % 16 == 0) &&
                             (!has_c || ((c_stride0 * (int64_t)sizeof(CD)) % 16 == 0 &&
                                         (uintptr_t)cptr % 16 == 0));
        for (int t = 0; t < TL; ++t) {
            const int b = t & 1;
            int m0, n0;
            pair_tile_geom(pair + t * Gp, m0, n0);
            const int row = m0 + row_in_tile;
            CD* drow = d + (size_t)row * d_stride0;
            const CD* crow = has_c ? cptr + (size_t)row * c_stride0 : nullptr;
            if (crow != nullptr && row < m && (int64_t)n * k <= 33554432) {
                const char* cbase =
                    reinterpret_cast<const char*>(crow) + (size_t)n0 * sizeof(CD);
                const int64_t row_bytes = (int64_t)min(BN, n - n0) * sizeof(CD);
#pragma unroll
                for (int off = 0; off < BN * (int)sizeof(CD); off += 128) {
                    if (off < row_bytes) {
                        asm volatile("prefetch.global.L2 [%0];" ::"l"(cbase + off));
                    }
                }
            }
            mbar_wait(&epi_ready[b], (t >> 1) & 1);
            tc_fence_after();
            const uint32_t dbase = tb + DSTRIDE * b;
            const int nblk = BN / 32;
            const int ntot = nblk + ((BN % 32) ? 1 : 0);
            const bool rev = (b == 0);   // OVL: drain shared cols first
            const bool v41_fast = has_c && (sizeof(CD) == 2) &&
                                  (ntot >= 2) && wide_ok &&
                                  (m0 + ews * 32 + 31 < m) && (n0 + BN <= n);
            if (v41_fast) {
                constexpr bool cq_on = (sizeof(CD) == 2);
                const int nb = ntot;
                uint32_t vv[2][32];
                uint4 cq[2][4];
                auto blk = [&](int ii) { return rev ? (nb - 1 - ii) : ii; };
                tmem_ld_32x(dbase + blk(0) * 32, vv[0]);
                tmem_ld_wait();
                __syncwarp();
                if (lane == 0) mbar_arrive(&ovl[b]);
                if (cq_on) {
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        cq[0][i] = *reinterpret_cast<const uint4*>(
                            crow + n0 + blk(0) * 32 + 8 * i);
                }
                int par = 0;
                for (int ii = 0; ii < nb; ++ii) {
                    if (ii + 1 < nb) {
                        tmem_ld_32x(dbase + blk(ii + 1) * 32, vv[par ^ 1]);
                        if (cq_on) {
#pragma unroll
                            for (int i = 0; i < 4; ++i)
                                cq[par ^ 1][i] = *reinterpret_cast<const uint4*>(
                                    crow + n0 + blk(ii + 1) * 32 + 8 * i);
                        }
                    }
                    epi_store32_cq<CD>(drow, n0 + blk(ii) * 32, alpha,
                                       vv[par], cq[par]);
                    if (ii + 1 < nb) {
                        tmem_ld_wait();
                        par ^= 1;
                    }
                }
            }
            if (!v41_fast)
            for (int ii = 0; ii < ntot; ++ii) {
                const int bi = rev ? (ntot - 1 - ii) : ii;
                const int c0 = bi * 32;
                const bool tail_blk = (c0 + 32 > BN);
                uint32_t v[32];
                if (!tail_blk) {
                    tmem_ld_32x(dbase + c0, v);
                } else {
                    uint32_t (*vg)[4] = reinterpret_cast<uint32_t (*)[4]>(v);
                    for (int j = 0; 4 * j < BN - c0; ++j) tmem_ld_4x(dbase + c0 + 4 * j, vg[j]);
                }
                tmem_ld_wait();
                if (ii == 0) {
                    __syncwarp();
                    if (lane == 0) mbar_arrive(&ovl[b]);
                }
                if (row < m) {
                    const int col = n0 + c0;
                    if (!tail_blk && col + 31 < n) {
                        epi_store32<CD>(drow, crow, col, alpha, has_c, wide_ok, v);
                    } else {
                        uint32_t (*vg)[4] = reinterpret_cast<uint32_t (*)[4]>(v);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int cg = col + j * 4;
                            if (c0 + 4 * j >= BN) break;
                            if (cg + 3 < n) {
                                float y0 = __uint_as_float(vg[j][0]) * alpha;
                                float y1 = __uint_as_float(vg[j][1]) * alpha;
                                float y2 = __uint_as_float(vg[j][2]) * alpha;
                                float y3 = __uint_as_float(vg[j][3]) * alpha;
                                if (has_c) {
                                    if constexpr (sizeof(CD) == 4) {
                                        const float2 c01 = *reinterpret_cast<const float2*>(crow + cg);
                                        const float2 c23 = *reinterpret_cast<const float2*>(crow + cg + 2);
                                        y0 += c01.x; y1 += c01.y; y2 += c23.x; y3 += c23.y;
                                    } else {
                                        const __nv_bfloat162 c01 = *reinterpret_cast<const __nv_bfloat162*>(crow + cg);
                                        const __nv_bfloat162 c23 = *reinterpret_cast<const __nv_bfloat162*>(crow + cg + 2);
                                        y0 += __bfloat162float(__low2bfloat16(c01));
                                        y1 += __bfloat162float(__high2bfloat16(c01));
                                        y2 += __bfloat162float(__low2bfloat16(c23));
                                        y3 += __bfloat162float(__high2bfloat16(c23));
                                    }
                                }
                                if constexpr (sizeof(CD) == 4) {
                                    *reinterpret_cast<float2*>(drow + cg) = make_float2(y0, y1);
                                    *reinterpret_cast<float2*>(drow + cg + 2) = make_float2(y2, y3);
                                } else {
                                    __nv_bfloat162 p01 = {__float2bfloat16_rn(y0), __float2bfloat16_rn(y1)};
                                    __nv_bfloat162 p23 = {__float2bfloat16_rn(y2), __float2bfloat16_rn(y3)};
                                    *reinterpret_cast<__nv_bfloat162*>(drow + cg) = p01;
                                    *reinterpret_cast<__nv_bfloat162*>(drow + cg + 2) = p23;
                                }
                            } else {
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    if (cg + i < n) {
                                        float y = __uint_as_float(vg[j][i]) * alpha;
                                        if (has_c) y += cd2f<CD>(crow[cg + i]);
                                        drow[cg + i] = cd_from_f32<CD>(y);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(&epi_free[b]);
        }
    }
    __syncthreads();
    cluster_sync();     // pair-wide exit: no peer multicast in flight
    if (wid == 0) tmem_dealloc1(tb, TMEM_COLS);
}

// ------------------------------------------------------- V39 gemm_pdbcg2 arm
// Persistent pair-tile cta_group::2 arm: pdbmc's production-attribute context
// (EXACT numerics incl. graph replay, sweep dv_...183491) + the bisect2-
// verified raw gemm_umma_cg2 MMA mechanics (2-SM UMMA, collective cp2 SF from
// per-CTA smem images, commit2 multicast) + a DEEP 6-stage ring (35840B/CTA
// per stage vs 49.5KB for 1-SM BN256: same per-SM FLOPs, ~30% fewer L2 bytes).
// Why not pdbmc-with-faster-relay: pdbmc measured ~725ns/stage vs prod 457;
// the per-stage bempty thread relay sits INSIDE the producer reuse circle
// (+~270ns/stage). pdbcg2 has NO per-stage thread hop: B is per-CTA (own
// half), reuse is gated by commit2_mc multicast (hardware arrives on BOTH
// CTAs' empty), and SF is tx-only (SFMODE 0) or relayed (SFMODE 1) with the
// relay off the reuse circle. SF images are the PP prepack (production host
// path): SFA = [img512|dup512] via two identical TMA loads; SFB = 1024B
// [blk0|blk1] + one-time zero pad to 2048B (raw-cg2 NaN-free footprint rule).
// Model: bytes/stage/CTA 34.8KB -> ~300-330ns/stage byte floor at prod mixed
// L2 (108 GB/s/CTA), chain ~1600ns/6 = ~270ns => target ~330-380 vs 457.
// SFMODE 0 full[buf]: count 1, leader expect 2*(A+Bh+SFA2+SFB) = 69632 tx,
// ALL loads cg2-form (peer-bit-masked barrier -> leader). SFMODE 1: payload
// tx-only (65536), SF plain-local onto sf_full (2048/CTA) + tid64 relays
// (wait, fence.proxy.async, arrive leader full; count 3) = V37.4/DeepGEMM
// class, kept as the numerical fallback if tx-only cp2-source races.
// tmem: OVL dual-D (D0 0..255, D1 240..495) + single SF set @496 (in-order
// tcgen05 pipe safety = production PP-pdb256 argument, extended to the pair
// pipe: leader issues cp2/UMMA strictly in order).
// V39.4: PW (plain leader wait) and CS (streaming st.global.cs D stores) are
// TEMPLATE params - V39.1's runtime params coincided with a 309->338us c7
// codegen regression (sweep 7 vs 8/15, identical prod controls); templating
// restores branch-free default paths. Split-producer removed (sweep 8: inert).
template <int STAGES, int BN, int SFMODE, int PW, int CS, typename CD>
__global__ void __launch_bounds__(256, 1) __cluster_dims__(2, 1, 1)
gemm_pdbcg2(int m, int n, int k, float alpha, int group,
            const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmBh,
            const __grid_constant__ CUtensorMap tmSA, const __grid_constant__ CUtensorMap tmSB,
            CD* __restrict__ d, int64_t d_stride0,
            const CD* __restrict__ cptr, int64_t c_stride0, int has_c,
            int pre_cap, int ee) {
    constexpr int BM = 128, BK = 128;
    constexpr int BH = BN / 2;                  // own B rows (128)
    constexpr int ST_BYTES = (BM * BK + BH * BK + 1024 + 2048 + 1023) & ~1023;  // 35840 BN256
    constexpr int A_OFF = 0;
    constexpr int B_OFF = BM * BK;                                   // 16384
    constexpr int SFA_OFF = B_OFF + BH * BK;                         // 32768: [img|dup] 1024B
    constexpr int SFB_OFF = SFA_OFF + 1024;                          // 33792: img 1024 + zero pad 1024
    constexpr uint32_t TX_ALL = 2u * (BM * BK + BH * BK + 512 + 512 + 1024);   // 69632
    constexpr uint32_t TX_PAY = 2u * (BM * BK + BH * BK);                      // 65536
    constexpr uint32_t SF_TX = 512 + 512 + 1024;                               // 2048 per CTA
    constexpr int DSTRIDE = 240;
    constexpr int SF_BASE = 496;
    constexpr int TMEM_COLS = 512;

    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + STAGES * ST_BYTES);
    uint64_t* empty = full + STAGES;          // commit2_mc multicast, count 1
    uint64_t* sf_full = empty + STAGES;       // SFMODE 1: local SF tx, count 1
    uint64_t* epi_ready = sf_full + STAGES;   // [2] count 1 (commit2_mc)
    uint64_t* epi_free = epi_ready + 2;       // [2] count 8 (4 local + 4 peer crew)
    uint64_t* ovl = epi_free + 2;             // [2] count 8
    uint32_t* tbp = reinterpret_cast<uint32_t*>(ovl + 2);

    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    const bool crew = (tid < 128);
    const uint32_t rank = cluster_rank();

    const int num_m_blocks = (m + BM - 1) / BM;
    const int num_n_blocks = (n + BN - 1) / BN;
    const int num_m_pairs = (num_m_blocks + 1) / 2;
    const int pair_total = num_m_pairs * num_n_blocks;
    const int Gp = gridDim.x >> 1;
    const int pair = blockIdx.x >> 1;
    if (pair >= pair_total) return;
    const int TL = (pair_total > Gp) ? (int)((pair_total - pair + Gp - 1) / Gp)
                                     : (pair < pair_total ? 1 : 0);
    const int nk = (k + BK - 1) / BK;
    const int GT = nk * TL;

    // pair raster: tile pt -> own (m0, n0); m-pair adjacent halves per rank.
    auto pair_tile_geom = [&](int pt, int& m0, int& n0) {
        const int per_group = num_n_blocks * group;
        const int group_idx = pt / per_group;
        const int first = group_idx * group;
        const int in_group = pt % per_group;
        const int span = min(group, num_m_pairs - first);
        const int pm = first + in_group % span;
        m0 = (2 * pm + (int)rank) * BM;
        n0 = (in_group / span) * BN;
    };

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], SFMODE == 0 ? 1 : 3);
            mbar_init(&empty[s], 1);
            mbar_init(&sf_full[s], 1);
        }
        mbar_init(&epi_ready[0], 1); mbar_init(&epi_ready[1], 1);
        mbar_init(&epi_free[0], 8); mbar_init(&epi_free[1], 8);
        mbar_init(&ovl[0], 8); mbar_init(&ovl[1], 8);
        fence_mbar_init();
    }
    // one-time zero pad for the 3rd cp2 footprint (SFB_OFF+1024..2047);
    // generic-proxy stores must be published to the async proxy (cp2 reads
    // it) before the first stage: fence.proxy.async = raw-cg2 pack pattern.
#pragma unroll
    for (int s = 0; s < STAGES; ++s)
        reinterpret_cast<uint32_t*>(smem + s * ST_BYTES + SFB_OFF + 1024)[tid] = 0u;
    fence_proxy_async();
    if (wid == 0) {
        tmem_alloc2(tbp, TMEM_COLS);
        tmem_relinq2();
    }
    cluster_sync();     // pair-wide barrier-init + pad visibility BEFORE any TMA
    __syncthreads();
    const uint32_t tb = *tbp;
    const uint32_t idesc_base = (1u << 23) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)((2 * BM) >> 4) << 24);

    if (crew) {
        auto issue_stage = [&](int g_stage, int s_tile, int m0, int n0) {
            const int buf = g_stage % STAGES;   // GLOBAL stage index for buffer
            uint8_t* stg = smem + buf * ST_BYTES;
            if (SFMODE == 0) {
                if (rank == 0) mbar_arrive_expect_tx(&full[buf], TX_ALL);
                tma_load_2d_cg2(&tmA, &full[buf], stg + A_OFF, s_tile * BK, m0);
                tma_load_2d_cg2(&tmBh, &full[buf], stg + B_OFF, s_tile * BK, n0 + (int)rank * BH);
                tma_load_2d_cg2(&tmSA, &full[buf], stg + SFA_OFF, m0, s_tile);
                tma_load_2d_cg2(&tmSA, &full[buf], stg + SFA_OFF + 512, m0, s_tile);
                tma_load_2d_cg2(&tmSB, &full[buf], stg + SFB_OFF, n0, s_tile);
            } else {
                if (rank == 0) mbar_arrive_expect_tx(&full[buf], TX_PAY);
                tma_load_2d_cg2(&tmA, &full[buf], stg + A_OFF, s_tile * BK, m0);
                tma_load_2d_cg2(&tmBh, &full[buf], stg + B_OFF, s_tile * BK, n0 + (int)rank * BH);
                mbar_arrive_expect_tx(&sf_full[buf], SF_TX);
                tma_load_2d(&tmSA, &sf_full[buf], stg + SFA_OFF, m0, s_tile);
                tma_load_2d(&tmSA, &sf_full[buf], stg + SFA_OFF + 512, m0, s_tile);
                tma_load_2d(&tmSB, &sf_full[buf], stg + SFB_OFF, n0, s_tile);
            }
        };
        int g2 = 0, t2 = 0, s2 = 0, m0_2 = 0, n0_2 = 0;
        if (tid == 32 && GT > 0) pair_tile_geom(pair, m0_2, n0_2);
        // pre_cap (V39.3): sweep 8 ST4 (PRE=3) outran ST5/ST6 (PRE=4/5) -
        // shallower run-ahead may reduce TMA/TC fabric contention.
        int PRE = STAGES - 1 < GT ? STAGES - 1 : GT;
        if (pre_cap > 0 && pre_cap < PRE) PRE = pre_cap;
        if (tid == 32) {
            for (; g2 < PRE; ++g2) {
                issue_stage(g2, s2, m0_2, n0_2);
                if (++s2 == nk) { s2 = 0; ++t2; pair_tile_geom(pair + t2 * Gp, m0_2, n0_2); }
            }
        }
        if (tid == 0 && rank == 0) {
            // ---------------- pair issuer (leader only) ----------------------
            int t1 = 0, s1 = 0;
            for (int g = 0; g < GT; ++g) {
                const int buf = g % STAGES;
                if constexpr (PW) mbar_wait(&full[buf], (g / STAGES) & 1);
                else mbar_wait_acq_cluster(&full[buf], (g / STAGES) & 1);
                if (s1 == 0 && t1 >= 1) {
                    // OVL early-start gate (shared cols 240..255 of tile t1-1,
                    // BOTH CTAs' crews: peer arrives are release.cluster =>
                    // acquire.cluster wait required for the scope pairing).
                    mbar_wait_acq_cluster(&ovl[(t1 - 1) & 1], ((t1 - 1) >> 1) & 1);
                }
                if (s1 == 0 && t1 >= 2) {
                    mbar_wait_acq_cluster(&epi_free[t1 & 1], ((t1 >> 1) - 1) & 1);
                }
                tc_fence_after();
                uint8_t* stg = smem + buf * ST_BYTES;
                const uint32_t dcol = tb + DSTRIDE * (t1 & 1);
                const uint32_t sfa_col = tb + SF_BASE;
                const uint32_t sfb_col = sfa_col + 4;
                utccp_2cta(sfa_col, make_desc(smem_u32(stg + SFA_OFF), 8, 0, 0));
                utccp_2cta(sfb_col, make_desc(smem_u32(stg + SFB_OFF), 8, 0, 0));
                utccp_2cta(sfb_col + 4, make_desc(smem_u32(stg + SFB_OFF + 512), 8, 0, 0));
                const uint64_t da = make_desc(smem_u32(stg + A_OFF), 64, 0, 2);
                const uint64_t db = make_desc(smem_u32(stg + B_OFF), 64, 0, 2);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint32_t id2 = idesc_base | ((uint32_t)kk << 29) | ((uint32_t)kk << 4);
                    umma_mxf8_2sm(dcol, da + kk * 2, db + kk * 2, id2, sfa_col, sfb_col,
                                  (s1 > 0 || kk > 0) ? 1u : 0u);
                }
                if (s1 == nk - 1) umma_commit2_mc(&epi_ready[t1 & 1], 3);
                umma_commit2_mc(&empty[buf], 3);
                if (++s1 == nk) { s1 = 0; ++t1; }
            }
        } else if (tid == 32) {
            // ---------------- producer (free-run, both ranks) -----------------
            for (; g2 < GT; ++g2) {
                if (g2 >= STAGES) {
                    const int bn = g2 % STAGES;
                    mbar_wait(&empty[bn], ((g2 - STAGES) / STAGES) & 1);
                }
                issue_stage(g2, s2, m0_2, n0_2);
                if (++s2 == nk) { s2 = 0; ++t2; if (t2 < TL) pair_tile_geom(pair + t2 * Gp, m0_2, n0_2); }
            }
        } else if (SFMODE == 1 && tid == 64) {
            // ---------------- SF relay (both ranks) ---------------------------
            // Local SF tx observation -> arrive on the LEADER's full: cp2
            // collective source release, off the producer reuse circle (the
            // structure DeepGEMM uses for peer-consumed SF smem).
            for (int g = 0; g < GT; ++g) {
                const int buf = g % STAGES;
                mbar_wait(&sf_full[buf], (g / STAGES) & 1);
                fence_proxy_async();
                if (rank == 0) mbar_arrive(&full[buf]);
                else mbar_arrive_remote(mapa_rank0(smem_u32(&full[buf])));
            }
        }
    } else {
        // ---------------- epilogue crew (wids 4-7, own rows) ------------------
        const int ews = wid - 4;
        const int row_in_tile = ews * 32 + lane;
        const bool wide_ok = ((d_stride0 * (int64_t)sizeof(CD)) % 16 == 0) &&
                             ((uintptr_t)d % 16 == 0) &&
                             (!has_c || ((c_stride0 * (int64_t)sizeof(CD)) % 16 == 0 &&
                                         (uintptr_t)cptr % 16 == 0));
        for (int t = 0; t < TL; ++t) {
            const int b = t & 1;
            int m0, n0;
            pair_tile_geom(pair + t * Gp, m0, n0);
            const int row = m0 + row_in_tile;
            CD* drow = d + (size_t)row * d_stride0;
            const CD* crow = has_c ? cptr + (size_t)row * c_stride0 : nullptr;
            if (crow != nullptr && row < m && (int64_t)n * k <= 33554432) {
                const char* cbase =
                    reinterpret_cast<const char*>(crow) + (size_t)n0 * sizeof(CD);
                const int64_t row_bytes = (int64_t)min(BN, n - n0) * sizeof(CD);
#pragma unroll
                for (int off = 0; off < BN * (int)sizeof(CD); off += 128) {
                    if (off < row_bytes) {
                        asm volatile("prefetch.global.L2 [%0];" ::"l"(cbase + off));
                    }
                }
            }
            // epi_ready is a commit2_mc MULTICAST arrival (cluster scope):
            // acquire.cluster wait for the scope pairing.
            mbar_wait_acq_cluster(&epi_ready[b], (t >> 1) & 1);
            tc_fence_after();
            const uint32_t dbase = tb + DSTRIDE * b;
            const int nblk = BN / 32;
            const int ntot = nblk + ((BN % 32) ? 1 : 0);
            const bool rev = (b == 0);   // OVL: drain shared cols first
            const bool v41_fast = has_c && (sizeof(CD) == 2) &&
                                  (ntot >= 2) && wide_ok &&
                                  (m0 + ews * 32 + 31 < m) && (n0 + BN <= n);
            if (v41_fast) {
                constexpr bool cq_on = (sizeof(CD) == 2);
                const int nb = ntot;
                uint32_t vv[2][32];
                uint4 cq[2][4];
                auto blk = [&](int ii) { return rev ? (nb - 1 - ii) : ii; };
                tmem_ld_32x(dbase + blk(0) * 32, vv[0]);
                tmem_ld_wait();
                __syncwarp();
                if (lane == 0) {
                    if (rank == 0) mbar_arrive(&ovl[b]);
                    else mbar_arrive_remote(mapa_rank0(smem_u32(&ovl[b])));
                }
                if (cq_on) {
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        cq[0][i] = *reinterpret_cast<const uint4*>(
                            crow + n0 + blk(0) * 32 + 8 * i);
                }
                int par = 0;
                for (int ii = 0; ii < nb; ++ii) {
                    if (ii + 1 < nb) {
                        tmem_ld_32x(dbase + blk(ii + 1) * 32, vv[par ^ 1]);
                        if (cq_on) {
#pragma unroll
                            for (int i = 0; i < 4; ++i)
                                cq[par ^ 1][i] = *reinterpret_cast<const uint4*>(
                                    crow + n0 + blk(ii + 1) * 32 + 8 * i);
                        }
                    }
                    // V39.5 EE: tmem is fully read after the last tmem_ld_wait
                    // (end of ii==nb-2); release the D buffer BEFORE the final
                    // store block so the issuer's epi_free gate stops paying
                    // the last-store tail (~300-600ns/tile).
                    if (ee && ii == nb - 1) {
                        __syncwarp();
                        if (lane == 0) {
                            if (rank == 0) mbar_arrive(&epi_free[b]);
                            else mbar_arrive_remote(mapa_rank0(smem_u32(&epi_free[b])));
                        }
                    }
                    epi_store32_cq<CD, CS != 0>(drow, n0 + blk(ii) * 32, alpha,
                                       vv[par], cq[par]);
                    if (ii + 1 < nb) {
                        tmem_ld_wait();
                        par ^= 1;
                    }
                }
            }
            if (!v41_fast)
            for (int ii = 0; ii < ntot; ++ii) {
                const int bi = rev ? (ntot - 1 - ii) : ii;
                const int c0 = bi * 32;
                const bool tail_blk = (c0 + 32 > BN);
                uint32_t v[32];
                if (!tail_blk) {
                    tmem_ld_32x(dbase + c0, v);
                } else {
                    uint32_t (*vg)[4] = reinterpret_cast<uint32_t (*)[4]>(v);
                    for (int j = 0; 4 * j < BN - c0; ++j) tmem_ld_4x(dbase + c0 + 4 * j, vg[j]);
                }
                tmem_ld_wait();
                if (ii == 0) {
                    __syncwarp();
                    if (lane == 0) {
                        if (rank == 0) mbar_arrive(&ovl[b]);
                        else mbar_arrive_remote(mapa_rank0(smem_u32(&ovl[b])));
                    }
                }
                if (ee && ii == ntot - 1) {
                    __syncwarp();
                    if (lane == 0) {
                        if (rank == 0) mbar_arrive(&epi_free[b]);
                        else mbar_arrive_remote(mapa_rank0(smem_u32(&epi_free[b])));
                    }
                }
                if (row < m) {
                    const int col = n0 + c0;
                    if (!tail_blk && col + 31 < n) {
                        epi_store32<CD, CS != 0>(drow, crow, col, alpha, has_c, wide_ok, v);
                    } else {
                        uint32_t (*vg)[4] = reinterpret_cast<uint32_t (*)[4]>(v);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int cg = col + j * 4;
                            if (c0 + 4 * j >= BN) break;
                            if (cg + 3 < n) {
                                float y0 = __uint_as_float(vg[j][0]) * alpha;
                                float y1 = __uint_as_float(vg[j][1]) * alpha;
                                float y2 = __uint_as_float(vg[j][2]) * alpha;
                                float y3 = __uint_as_float(vg[j][3]) * alpha;
                                if (has_c) {
                                    if constexpr (sizeof(CD) == 4) {
                                        const float2 c01 = *reinterpret_cast<const float2*>(crow + cg);
                                        const float2 c23 = *reinterpret_cast<const float2*>(crow + cg + 2);
                                        y0 += c01.x; y1 += c01.y; y2 += c23.x; y3 += c23.y;
                                    } else {
                                        const __nv_bfloat162 c01 = *reinterpret_cast<const __nv_bfloat162*>(crow + cg);
                                        const __nv_bfloat162 c23 = *reinterpret_cast<const __nv_bfloat162*>(crow + cg + 2);
                                        y0 += __bfloat162float(__low2bfloat16(c01));
                                        y1 += __bfloat162float(__high2bfloat16(c01));
                                        y2 += __bfloat162float(__low2bfloat16(c23));
                                        y3 += __bfloat162float(__high2bfloat16(c23));
                                    }
                                }
                                if constexpr (sizeof(CD) == 4) {
                                    *reinterpret_cast<float2*>(drow + cg) = make_float2(y0, y1);
                                    *reinterpret_cast<float2*>(drow + cg + 2) = make_float2(y2, y3);
                                } else {
                                    __nv_bfloat162 p01 = {__float2bfloat16_rn(y0), __float2bfloat16_rn(y1)};
                                    __nv_bfloat162 p23 = {__float2bfloat16_rn(y2), __float2bfloat16_rn(y3)};
                                    *reinterpret_cast<__nv_bfloat162*>(drow + cg) = p01;
                                    *reinterpret_cast<__nv_bfloat162*>(drow + cg + 2) = p23;
                                }
                            } else {
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    if (cg + i < n) {
                                        float y = __uint_as_float(vg[j][i]) * alpha;
                                        if (has_c) y += cd2f<CD>(crow[cg + i]);
                                        drow[cg + i] = cd_from_f32<CD>(y);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (!ee) {
                __syncwarp();
                if (lane == 0) {
                    if (rank == 0) mbar_arrive(&epi_free[b]);
                    else mbar_arrive_remote(mapa_rank0(smem_u32(&epi_free[b])));
                }
            }
        }
    }
    __syncthreads();
    cluster_sync();     // pair-wide exit: no peer commit/arrive in flight
    if (wid == 0) tmem_dealloc2(tb, TMEM_COLS);
}

// V25 prepack (phase join-20261025): device-side ue8m0 SF prepack into the
// exact per-kb permuted 512B images that pack_sf produces in smem (word w of
// a 128-row block = row 32*(w%4) + w/4; index math taken from pack_sf above).
// Mechanism reference: arm-a N6/N17/N19 (join 20261025 exchange); independent
// implementation. Padded rows (r >= m/n) store 0x00 = ue8m0(0.0), matching
// the classic TMA-OOB-zero-fill + pack path byte-for-byte (garbage-scale
// confinement unchanged: those D rows are never stored). One float4 load +
// one word store per thread; runs in the measured path on EVERY forward.
__global__ void prepack_sf(const float* __restrict__ sa, int m, int sa_stride,
                           const float* __restrict__ sb, int n, int sb_stride,
                           uint8_t* __restrict__ out_a, int pad_m,
                           uint8_t* __restrict__ out_b, int pad_n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pad_m + pad_n) return;
    const int kb = blockIdx.y;
    const bool isA = (tid < pad_m);
    const int r = isA ? tid : tid - pad_m;
    const int rows = isA ? m : n;
    const int padrows = isA ? pad_m : pad_n;
    const int rin = r & 127;
    uint32_t* dst = reinterpret_cast<uint32_t*>(
        (isA ? out_a : out_b) + (size_t)kb * padrows * 4 + (size_t)(r >> 7) * 512)
        + 4 * (rin & 31) + (rin >> 5);
    uint32_t w = 0u;
    if (r < rows) {
        const float4 f4 = *reinterpret_cast<const float4*>(
            (isA ? sa : sb) + (size_t)r * (isA ? sa_stride : sb_stride) + kb * 4);
        w = ue8m0(f4.x) | (ue8m0(f4.y) << 8) | (ue8m0(f4.z) << 16) | (ue8m0(f4.w) << 24);
    }
    *dst = w;
}

// J26 probe (mechanism: arm-a N30-v2, archived ae62d01; certified c2 0.9016
// on that lineage): bn_tile-generalized prepack. bn_tile==0 == prepack_sf;
// bn_tile==224 packs the B side TILE-RELATIVE (a 224-row tile's 2 x 512B
// block images contiguous => one 1024B TMA box; in-tile rows >= 224 and
// rows >= n zero-filled = packer zero-tail parity). Cold unless MXFP8_PP224.
__global__ void prepack_sf2(const float* __restrict__ sa, int m, int sa_stride,
                            const float* __restrict__ sb, int n, int sb_stride,
                            uint8_t* __restrict__ out_a, int pad_m,
                            uint8_t* __restrict__ out_b, int pad_n, int bn_tile) {
    // bn_tile == 0: absolute 128-row blocks (BN256-class, layout as before).
    // bn_tile == 224 (N30-v2): B side is TILE-RELATIVE — block index within
    // the 224-row tile (b = (r%224)/128), so a tile's 2 x 512B images are
    // contiguous (1024B merged TMA box); in-tile rows >= 224 zero-filled =
    // packer zero-tail parity. A side always absolute (BM=128 aligned).
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_tile = bn_tile ? ((bn_tile + 127) / 128) : 1;
    const int bthreads = bn_tile ? (pad_n / bn_tile) * (blocks_per_tile * 128) : pad_n;
    if (tid >= pad_m + bthreads) return;
    const int kb = blockIdx.y;
    const bool isA = tid < pad_m;
    const int r = isA ? tid : tid - pad_m;
    uint32_t w = 0u;   // invalid rows: 0x00 = 2^-127 (classic TMA-OOB parity)
    if (isA) {
        const int blk = r >> 7, rin = r & 127;
        uint8_t* dst = out_a + (size_t)kb * pad_m * 4 + (size_t)blk * 512 +
                       (4 * (rin & 31) + (rin >> 5)) * 4;
        if (r < m) {
            const float4 f4 = *reinterpret_cast<const float4*>(sa + (size_t)r * sa_stride + kb * 4);
            w = ue8m0(f4.x) | (ue8m0(f4.y) << 8) | (ue8m0(f4.z) << 16) | (ue8m0(f4.w) << 24);
        }
        *reinterpret_cast<uint32_t*>(dst) = w;
        return;
    }
    if (bn_tile == 0) {
        const int rows = n;
        const int blk = r >> 7, rin = r & 127;
        uint8_t* dst = out_b + (size_t)kb * pad_n * 4 + (size_t)blk * 512 +
                       (4 * (rin & 31) + (rin >> 5)) * 4;
        if (r < rows) {
            const float4 f4 = *reinterpret_cast<const float4*>(sb + (size_t)r * sb_stride + kb * 4);
            w = ue8m0(f4.x) | (ue8m0(f4.y) << 8) | (ue8m0(f4.z) << 16) | (ue8m0(f4.w) << 24);
        }
        *reinterpret_cast<uint32_t*>(dst) = w;
        return;
    }
    {
        const int t = r / (blocks_per_tile * 128);
        const int ril = r % (blocks_per_tile * 128);
        const int b = ril >> 7, q = ril & 127;
        const int rin_tile = b * 128 + q;                 // 0..255 for bn224
        const int rabs = t * bn_tile + rin_tile;
        uint8_t* dst = out_b + (size_t)kb * (size_t)(pad_n / bn_tile) * (blocks_per_tile * 512) +
                       (size_t)t * (blocks_per_tile * 512) + (size_t)b * 512 +
                       (4 * (q & 31) + (q >> 5)) * 4;
        if (rin_tile < bn_tile && rabs < n) {
            const float4 f4 = *reinterpret_cast<const float4*>(sb + (size_t)rabs * sb_stride + kb * 4);
            w = ue8m0(f4.x) | (ue8m0(f4.y) << 8) | (ue8m0(f4.z) << 16) | (ue8m0(f4.w) << 24);
        }
        *reinterpret_cast<uint32_t*>(dst) = w;
    }
}

// ------------------------------------------------------- tensor-map cache (host)
namespace {
using MapKey = std::tuple<uintptr_t, int64_t, int64_t, int64_t, uint32_t, uint32_t, int, int>;
std::map<MapKey, CUtensorMap> g_map_cache;

const CUtensorMap* get_map(const void* ptr, int64_t inner, int64_t outer, int64_t outer_stride_bytes,
                           uint32_t box_inner, uint32_t box_outer, int dtype_id /*0=u8,2=f32*/,
                           int swizzle_bytes) {
    MapKey key{(uintptr_t)ptr, inner, outer, outer_stride_bytes, box_inner, box_outer, dtype_id, swizzle_bytes};
    auto it = g_map_cache.find(key);
    if (it != g_map_cache.end()) return &it->second;
    CUtensorMap tm;
    cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
    cuuint64_t strides[1] = {(cuuint64_t)outer_stride_bytes};
    cuuint32_t box[2] = {box_inner, box_outer};
    cuuint32_t estr[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(
        &tm, dtype_id == 0 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2, const_cast<void*>(ptr), dims, strides, box, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "mxfp8: cuTensorMapEncodeTiled failed ", (int)r);
    auto res = g_map_cache.emplace(key, tm);
    return &res.first->second;
}

// ---- V35 (J18): C++-side CUDA-graph cache for the pp two-kernel chain ----
// Launch mechanism only (peer N9/N9b graph-replay precedent, owner-certified
// on arm-a; J17 dv_40b777a60b39 measured the GPU-side win on THIS stack:
// c5 -2.9us, c7 -1.0, d8 -1.1, c2-implied ~-1.3..-1.6; J17's rejection was
// the PYTHON keying tax ~+5us/call, which does not exist here: the key is
// built from dispatch metadata already at hand). Replay RE-EXECUTES both
// kernels on the CURRENT buffer contents (params recorded by value; ws/map
// caches are insert-only => sap/sbp and map values are pointer-stable, and
// both are key fields anyway). Shape-gated to pp_pdb chains (the measured-
// win class). Bounded 64 keys; any capture/instantiate error => permanently
// eager for that key; outer stream capture => eager pass-through.
struct PPGKey {
    uint64_t aq, sa, bq, sb, d, c, sap, sbp;
    int64_t m, n, k, sa_s0, sb_s0, aq_s0, bq_s0, d_s0, c_s0;
    int32_t dt_f32, has_c, dev;
    uint32_t alpha_bits;
    bool operator<(const PPGKey& o) const {
        return std::tie(aq, sa, bq, sb, d, c, sap, sbp, m, n, k, sa_s0, sb_s0,
                        aq_s0, bq_s0, d_s0, c_s0, dt_f32, has_c, dev, alpha_bits) <
               std::tie(o.aq, o.sa, o.bq, o.sb, o.d, o.c, o.sap, o.sbp, o.m, o.n, o.k,
                        o.sa_s0, o.sb_s0, o.aq_s0, o.bq_s0, o.d_s0, o.c_s0,
                        o.dt_f32, o.has_c, o.dev, o.alpha_bits);
    }
};
struct PPGEnt { cudaGraphExec_t exec = nullptr; int count = 0; bool failed = false; };
static std::map<PPGKey, PPGEnt> g_ppg;
static long g_ppg_caps = 0, g_ppg_launches = 0, g_ppg_fails = 0;
static cudaStream_t g_ppg_cs = nullptr;
static bool g_ppg_cs_bad = false;
static const bool g_ppg_on = [] {
    const char* e = getenv("MXFP8_GRAPHPP");
    return !(e && atoi(e) == 0);
}();
// V36: cold-path graph helpers (noinline => dispatch layout ~v33).
struct PPGArgs {
    uintptr_t aq, sa, bq, sb, d, c, sap, sbp;
    int64_t m, n, k, sa_s0, sb_s0, aq_s0, bq_s0, d_s0, c_s0;
    int32_t f32, has_c, dev;
    uint32_t abits;
};
enum { PPG_EAGER = 0, PPG_REPLAY = 1, PPG_CAPTURE = 2 };
__attribute__((noinline)) static int ppg_begin(const PPGArgs& a, cudaStream_t cur,
                                               PPGEnt** ent, cudaGraphExec_t* exec,
                                               cudaStream_t* st) {
    *ent = nullptr; *exec = nullptr; *st = cur;
    if (!g_ppg_on) return PPG_EAGER;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(cur, &cs);
    if (cs != cudaStreamCaptureStatusNone) return PPG_EAGER;
    PPGKey key{a.aq, a.sa, a.bq, a.sb, a.d, a.c, a.sap, a.sbp,
               a.m, a.n, a.k, a.sa_s0, a.sb_s0, a.aq_s0, a.bq_s0, a.d_s0, a.c_s0,
               a.f32, a.has_c, a.dev, a.abits};
    auto it = g_ppg.find(key);
    if (it != g_ppg.end() && it->second.exec != nullptr) {
        *exec = it->second.exec;
        return PPG_REPLAY;
    }
    if (it == g_ppg.end() && g_ppg.size() < 64)
        it = g_ppg.emplace(key, PPGEnt()).first;
    if (it == g_ppg.end()) return PPG_EAGER;
    *ent = &it->second;
    if (!it->second.failed && it->second.count >= 1) {
        if (!g_ppg_cs && !g_ppg_cs_bad) {
            if (cudaStreamCreateWithFlags(&g_ppg_cs, cudaStreamNonBlocking) != cudaSuccess) {
                g_ppg_cs_bad = true;
                g_ppg_cs = nullptr;
                (void)cudaGetLastError();   // consume: handled, eager path
            }
        }
        if (g_ppg_cs &&
            cudaStreamBeginCapture(g_ppg_cs, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
            *st = g_ppg_cs;
            return PPG_CAPTURE;
        }
        it->second.failed = true;
        ++g_ppg_fails;
        (void)cudaGetLastError();   // consume: handled, eager path
    } else if (!it->second.failed) {
        ++it->second.count;
    }
    return PPG_EAGER;
}
__attribute__((noinline)) static cudaGraphExec_t ppg_end(PPGEnt* ent) {
    cudaGraph_t gr = nullptr;
    bool ok = (cudaStreamEndCapture(g_ppg_cs, &gr) == cudaSuccess) && gr != nullptr;
    cudaGraphExec_t ex = nullptr;
    if (ok && cudaGraphInstantiate(&ex, gr, 0) == cudaSuccess) {
        cudaGraphDestroy(gr);
        ent->exec = ex;
        ++g_ppg_caps;
        return ex;
    }
    if (gr) cudaGraphDestroy(gr);
    ent->failed = true;
    ++g_ppg_fails;
    (void)cudaGetLastError();   // consume: handled, caller re-launches eagerly
    return nullptr;
}
static inline uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
}  // namespace

}  // namespace mxfp8

// ---------------------------------------------------------------- entry point
void mxfp8_gemm(const torch::Tensor& aq, const torch::Tensor& sa,
                const torch::Tensor& bq, const torch::Tensor& sb,
                torch::Tensor& d, const c10::optional<torch::Tensor>& c, double alpha) {
    using namespace mxfp8;
    const int m = (int)aq.size(0), k = (int)aq.size(1), n = (int)bq.size(0);
    if (m == 0 || n == 0) return;
    TORCH_CHECK(k % 32 == 0 && sa.size(0) == m && sa.size(1) == k / 32, "mxfp8: A scale shape");
    TORCH_CHECK(sb.size(0) == n && sb.size(1) == k / 32, "mxfp8: B scale shape");
    TORCH_CHECK(aq.stride(1) == 1 && bq.stride(1) == 1, "mxfp8: A/B must be K-major");
    TORCH_CHECK(sa.stride(1) == 1 && sb.stride(1) == 1, "mxfp8: scales must be K-major");
    TORCH_CHECK(d.stride(1) == 1 && d.size(0) == m && d.size(1) == n, "mxfp8: D must be N-major (m,n)");
    const bool is_f32 = d.scalar_type() == torch::kFloat32;
    TORCH_CHECK(is_f32 || d.scalar_type() == torch::kBFloat16, "mxfp8: D dtype must be f32/bf16");
    TORCH_CHECK(aq.scalar_type() == torch::kFloat8_e4m3fn && bq.scalar_type() == torch::kFloat8_e4m3fn,
                "mxfp8: payloads must be float8_e4m3fn");
    TORCH_CHECK(sa.scalar_type() == torch::kFloat32 && sb.scalar_type() == torch::kFloat32,
                "mxfp8: scales must be float32");
    TORCH_CHECK(k % 128 == 0, "mxfp8: V2.3 requires k%128==0 (frozen domain guarantees this)");
    TORCH_CHECK(sa.stride(0) % 4 == 0 && sb.stride(0) % 4 == 0, "mxfp8: scale rows must be 16B aligned");
    TORCH_CHECK(((uint64_t)d.stride(0) * d.element_size()) % 8 == 0 && ((uintptr_t)d.data_ptr() % 8 == 0),
                "mxfp8: D rows must be 8B aligned");
    TORCH_CHECK(((uintptr_t)aq.data_ptr() | (uintptr_t)bq.data_ptr() | (uintptr_t)sa.data_ptr() |
                 (uintptr_t)sb.data_ptr()) % 16 == 0, "mxfp8: operands must be 16B aligned");
    TORCH_CHECK(aq.stride(0) % 16 == 0 && bq.stride(0) % 16 == 0, "mxfp8: A/B rows must be 16B aligned");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (k == 0) {
        if (c.has_value()) d.copy_(c.value()); else d.zero_();
        return;
    }

    const void* cptr = nullptr;
    int64_t c_stride0 = n;
    bool has_c = c.has_value();
    if (has_c) {
        const auto& ct = c.value();
        TORCH_CHECK(ct.size(0) == m && ct.size(1) == n && ct.scalar_type() == d.scalar_type() &&
                    ct.stride(1) == 1, "mxfp8: C must match D shape/dtype, N-major");
        cptr = ct.data_ptr();
        c_stride0 = ct.stride(0);
    }

    // Hybrid BN: 256 unless halving the tile count costs >7% wave efficiency;
    // 64 (8 deep stages) when even BN=128 leaves half the SMs idle.
    int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int mt = (m + 127) / 128;
    const int tiles256 = mt * ((n + 255) / 256);
    const int tiles128 = mt * ((n + 127) / 128);
    const int tiles64 = mt * ((n + 63) / 64);
    auto eff = [nsm](int t) { return (double)t / (((t + nsm - 1) / nsm) * nsm); };
    // BN=64 dev knob (MXFP8_BN64=1): latency-class single-m-row shapes only
    // (mt==1): doubles SM coverage (c0: 128 vs 64 tiles) with 8-stage depth
    // and smaller per-stage ingest (27648B). Rejected for general dispatch
    // (dv_d92a3d871b42: c0 0.909x under the OLD pre-fast-epilogue balance);
    // retest post-V3.7 pending.
    static const int bn64_force = [] {
        const char* e = getenv("MXFP8_BN64");
        return (e && atoi(e) > 0) ? atoi(e) : 0;
    }();
    const bool use64 = (bn64_force >= 1) && ((m + 127) / 128 == 1) && (n >= 64);
    // MXFP8_FORCE256=1: bypass the 0.93 wave-efficiency threshold (tuned in
    // the V2.3 era, before the fast epilogue shrank per-tile cost — retest).
    // MXFP8_FORCE256=2 (dev): HARD np-BN256 route - also bypasses every
    // pdb/pp/224c2/cg2 arm (tests classic 4-stage BN256 on pdb-routed cells).
    static const int force256 = [] {
        const char* e = getenv("MXFP8_FORCE256");
        return (e && atoi(e) > 0) ? atoi(e) : 0;
    }();
    const bool hard256 = (force256 == 2);
    // Threshold 0.93 -> 0.85 post-fast-epilogue (T1 dv_...0605 job: forcing
    // BN256 on the 2sm class won c9 1.11x (ratio .875) and c2 1.03x (1.0) —
    // per-tile epilogue cost no longer scales with tile count the old way).
    // pdb240: persistent double-buffer BN240 (2 D buffers + SF slots fill the
    // 512-col tmem exactly). Measured (dv_340d17381e51 A/B + domain): wins
    // ONLY the short-k sub-0.93-wave-ratio class (c2 1.17x vs BN256-np, d4
    // 1.19x vs 2sm) — persistence removes relaunch/cold-prologue and hides
    // the epilogue, but the BN240 TMA box runs ~709ns/stage vs BN256's ~575,
    // so ratio>=0.93 (c4/c7/c8) and deep-k (c9/d2) shapes stay on the
    // non-persistent BN256 / 2sm / 6-stage arms.
    const int tiles240 = mt * ((n + 239) / 240);
    // V3.19 pdb224 DEFAULT: same persistent double-buffer mechanism as
    // pdb240, box 224. Measured (dv_c2bb3cedecef, same-node 4-pass, 40/40
    // exact): c2 76.5->72.1 (-5.8%), d4 105.7->99.9 (-5.5%), all other
    // cases noise. Model note: pdb240's "709ns box penalty" was CONFOUNDED
    // with persistent-mode cost (np224 = 568ns 128-thread; pdb224 derived
    // ~664ns 256-thread dual-D tmem contention) — in-mode box gain -45ns.
    // Class extension re-checked with corrected stage (664ns): c7/c8/c4 all
    // lose => class unchanged. tmem 2x224+32 = 480 <= 512; epilogue 7 clean
    // 32-col blocks (240 needs the tmem_ld_4x remainder path).
    // MXFP8_PDB240=1 = measured fallback knob.
    static const int pdb240_force = [] {
        const char* e = getenv("MXFP8_PDB240");
        return (e && atoi(e) > 0) ? atoi(e) : 0;
    }();
    const int tiles224 = mt * ((n + 223) / 224);
    // V3.22 BN224-BK128-2st at 2 CTAs/SM. PRODUCTION WINDOW (sweep
    // dv_b6b7ca302782, same-node 4-pass): tiles224 in (nsm, 2*nsm] converts
    // a 2-wave BN256 schedule into ONE dual-resident wave: c4 0.898x,
    // c8 0.887x. Excluded by measurement: mt==1 (needs ABOX/BK256 arms:
    // full-A-box + 1-CTA residency = 1.47-1.60x), tiles224 > 2*nsm
    // (production DRAM-mixed stage ~1300ns >> probe's L2-warm 888: c2 1.056
    // c5 1.349 c7 1.035 c9 1.373), tiles224 <= nsm (1-CTA residency exposes
    // PRE=1 DRAM latency per stage: c3 2.320x). Window implies !splitk
    // (tiles128 >= tiles224 > nsm > nsm/2) and dominates pdb224 there
    // (dom9-class: 1 dual wave vs persistent 1-CTA/SM waves).
    // MXFP8_BN224C2=1 = diagnostic force (any n>=224 shape).
    static const int bn224c2_force = [] {
        const char* e = getenv("MXFP8_BN224C2");
        return (e && atoi(e) > 0) ? atoi(e) : 0;
    }();
    // Dev-only experiment knobs (sweep evidence gathering). NOTE: defaults
    // keep production behavior byte-identical EXCEPT MXFP8_FB6ST, which is
    // now default-ON (V40a, validated below) - it changes the fallback arm.
    // MXFP8_SKMIN: minimum split count for the deep-k splitk gate (prod: 3).
    //   Re-opens the V3.16 sk=2 closure for the shallow-k mt==1 class (c0/c1/
    //   d1: 64 CTAs pull only ~1.3-1.7 TB/s; dv_b9ac363977dd measured plain
    //   64-CTA streaming at 0.57 TB/s vs 4.52 at 592 CTAs => CTA-count bound).
    // MXFP8_SKF=N: force N-way split for single-wave BN128 fallback shapes
    //   (tiles128 <= nsm, e.g. c3 128 CTAs x 48 BK256 stages) when feasible.
    // MXFP8_BN64=2: BN64 arm ALSO builds the ABOX small maps (prod knob =1
    //   runs full 128-row A boxes; m<=32 wastes 75-99% of the A ingest).
    static const int skmin_gate = [] {
        const char* e = getenv("MXFP8_SKMIN");
        return (e && atoi(e) > 0) ? atoi(e) : 3;
    }();
    static const int skf_force = [] {
        const char* e = getenv("MXFP8_SKF");
        return (e && atoi(e) >= 2) ? atoi(e) : 0;
    }();
    // MXFP8_FB6ST: fallback arm as 6-stage BK128 (V40a DEFAULT ON; prod was
    // 3-stage BK256, V3.20 L2-warm chain win). Validated same-job 3x: c3
    // 46.08 vs 48.10 (+4.4%, dv_5641ec0c2926 + dv_10ea1fa957d3, mm=0 eager
    // +replay); c0/c1/d1/d2 identical to 0.01us (DRAM-per-CTA bound, prefetch
    // depth inert); d3 neutral. MXFP8_FB6ST=0 restores the old arm.
    static const int fb6_force = [] {
        const char* e = getenv("MXFP8_FB6ST");
        return (e && atoi(e) >= 0) ? atoi(e) : 1;
    }();
    // V39.6 MXFP8_EE: early epi_free release in the PP-pdb256 arm (template
    // flag; default off = byte-identical). Port of the pdbcg2 EE knob: sweep
    // 17 measured EE -3.4% on c2 (nk=20 boundary-dominated) with mm=0 incl.
    // replay on c9 (alias+acc); long-nk heavies neutral-positive.
    static const int ee_force = [] {
        const char* e = getenv("MXFP8_EE");
        return (e && atoi(e) > 0) ? 1 : 0;
    }();
    // V38 pdbmc dev arm: persistent OVL BN256 with cluster(2,1) B-multicast
    // (MXFP8_PDBMC=1). Heavy/mid m>=256 classes; overrides other arms.
    static const int pdbmc_force = [] {
        const char* e = getenv("MXFP8_PDBMC");
        return (e && atoi(e) > 0) ? atoi(e) : 0;
    }();
    // V40b: MXFP8_PDBCG2 modes: 0 = off, 1 = force (dev sweeps), 2 = AUTO
    // production routing for the validated deep-k heavy class (V40c DEFAULT).
    // Sweep 19 full-suite rehearsal (20261028T202027Z, same-job A/B, mm=0
    // incl replay on 6 checked): c5 1.018x, c7 1.020x, c9 1.043x, d4 1.066x,
    // d6 1.017x; ALL non-gated cases identical to 0.2% (no leaks). Losers
    // stay excluded: c2/c4/c8 (k2560), short-chain (2048,8192,4096) 0.971x,
    // odd-mt tails, d9-class 224c2 window (pdbcg2 wins there too - 49.15 vs
    // 52.2 forced - but untested in-gate neighbors; post-promotion follow-up).
    // Gate = k>=4096 && mt>=2 && mt%2==0 && n>=256 && ((pdb256-class &&
    // tl256*nk>=288) || pdb224-class). MXFP8_PDBCG2=0 restores certified
    // routing.
    static const int pdbcg2_force = [] {
        const char* e = getenv("MXFP8_PDBCG2");
        return (e && atoi(e) >= 0) ? atoi(e) : 2;
    }();
    static const int cg2sf_mode = [] {
        const char* e = getenv("MXFP8_CG2SF");
        return (e && atoi(e) >= 0) ? atoi(e) : 0;
    }();
    // V39.1/39.4 chain-cut diagnostics: MXFP8_CG2ST = ring depth (4/5/6,
    // default 6), MXFP8_CG2PW=1 = plain leader wait (acquire.cluster proved
    // numerically unnecessary for tx-only barriers, sweep 8), MXFP8_CG2CS=1 =
    // streaming st.global.cs D stores (d-output L2 pollution countermeasure;
    // sweep 15 decomposition: prod = ~270ns/stage mechanism + ~4-5.4us/tile
    // boundary + WS-miss penalty), MXFP8_CG2PRE = PRE cap. V39.4: PW/CS are
    // template params (V39.1 runtime params cost ~9% via codegen).
    static const int cg2st_env = [] {
        const char* e = getenv("MXFP8_CG2ST");
        return (e && atoi(e) >= 4) ? atoi(e) : 6;
    }();
    static const int cg2_pw = [] {
        const char* e = getenv("MXFP8_CG2PW");
        return (e && atoi(e) > 0) ? 1 : 0;
    }();
    static const int cg2_cs = [] {
        const char* e = getenv("MXFP8_CG2CS");
        return (e && atoi(e) > 0) ? 1 : 0;
    }();
    static const int cg2_pre_cap = [] {
        const char* e = getenv("MXFP8_CG2PRE");
        return (e && atoi(e) > 0) ? atoi(e) : 0;
    }();
    // V39.5 EE: early epi_free release (tmem freed at last tmem_ld_wait,
    // before the final store block) - attacks the ~4-5.4us/tile boundary
    // cost from the sweep-15 decomposition. Per-TILE branch (no per-stage
    // codegen risk).
    static const int cg2_ee = [] {
        const char* e = getenv("MXFP8_CG2EE");
        return (e && atoi(e) > 0) ? 1 : 0;
    }();
    // V40b: auto mode bakes in the validated config (PW=1, EE=1, CS=0, ST6,
    // SFM0) regardless of the dev knobs.
    const int cg2_pw_sel = (pdbcg2_force == 2) ? 1 : cg2_pw;
    const int cg2_ee_sel = (pdbcg2_force == 2) ? 1 : cg2_ee;
    const int cg2st_sel = (cg2st_env == 4 || cg2st_env == 5) ? cg2st_env : 6;
    // V40b: MXFP8_PDBCG2 modes: 0 = off (default), 1 = force (dev sweeps),
    // 2 = AUTO production routing for the validated deep-k heavy class
    // (sweeps 7/16/17/18: PW+EE pair arm wins c5 1.014x, c7 1.014x, c9
    // 1.038x, d4 1.082x, d6 1.014x, d9 1.060x; loses c2/c4/c8 (k2560) and
    // short-chain (2048,8192,4096) 0.971x => gate = k>=4096 && mt even &&
    // ((pdb256-class && tl256*nk>=288) || pdb224-class)).
    const bool use_pdbcg2_forced = (pdbcg2_force == 1) && (mt >= 2) && (n >= 256);
    const bool use_pdbmc = (pdbmc_force == 1) && (mt >= 2) && (n >= 256) && !use_pdbcg2_forced;
    // tiles256 > nsm required: if BN256 is itself single-wave it runs
    // nk x 486 + X, which beats the dual-resident makespan nk x 888.
    const bool win224c2 = (mt >= 2) && (tiles224 > nsm) && (tiles224 <= 2 * nsm) &&
                          (tiles256 > nsm);
    const bool use224c2 = (win224c2 || bn224c2_force == 1) && (n >= 224) && !use64 && !hard256 && !use_pdbmc && !use_pdbcg2_forced;
    const bool pdb_gate = (k <= 6144) && (eff(tiles256) < 0.93 * eff(tiles128));
    // V29 (J6, dv_4a9eb64db08f): the post-PP arm equilibrium flips for the
    // pdb_gate class when prepack economics hold (tl256*nk >= 80: measured
    // c2-acc -1.8%med/-3.1%best, c2-plain -7.3%, tl6-acc -2.4%/-4.7%;
    // routing WITHOUT pp was +5-7% worse = the old equilibrium). pdb224
    // cannot take PP: BN224 tile n0 is not 128-aligned, so the per-128-
    // block prepacked permutation cannot feed a 224-wide box. d3-class
    // (tl*nk = 64 < 80) stays pdb224 (measured noisy-neutral).
    const int tl256 = (tiles256 + nsm - 1) / nsm;
    const bool pp_route = pdb_gate && (tiles256 >= nsm) &&
                          ((int64_t)tl256 * (k / 128) >= 80);
    // V37 cg2pp dev arm: prepacked role-split cta_group::2 (M256 pair),
    // forced via MXFP8_CG2PP: 1 = BN240, 2 = BN256. Attacks the L2 byte wall
    // of the 1-SM BN256-class stages (49KB/stage vs ~35-38KB per CTA at the
    // same per-SM FLOPs); evidence in the gemm_cg2pp header. Overrides every
    // other arm when eligible (m >= 256); dev-only until A/B-verified.
    static const int cg2pp_force = [] {
        const char* e = getenv("MXFP8_CG2PP");
        return (e && atoi(e) > 0) ? atoi(e) : 0;
    }();
    // MXFP8_CG2SIG: cg2pp epilogue bisect (dev): 0 = 32-col fast epilogue
    // (production target), 3 = verified prod cg2 4x-group epilogue reference.
    static const int cg2_sig = [] {
        const char* e = getenv("MXFP8_CG2SIG");
        return (e && atoi(e) >= 0) ? atoi(e) : 0;
    }();
    const int cg2pp_bn = (cg2pp_force == 1) ? 240 : 256;
    const bool use_cg2pp = (cg2pp_force >= 1) && (m >= 256) && (n >= cg2pp_bn) && !use_pdbmc && !use_pdbcg2_forced;
    // J26: PP-pdb224 for the pp_route class (arm-a N30-v2 mechanism; probe
    // dv_aa0e2e2e1722 GREEN, all pre-registered criteria met): tile-relative
    // 224 prepack beats the V29 PP-pdb256 route when 224's wave packing does
    // not exceed 256's — c2acc 0.949/0.952, c2plain 0.948/0.960, m-tail(2433)
    // 0.936/0.944, all mm=0; c9-class (ceil(t224/nsm) 5 > ceil(t256/nsm) 4)
    // measured 1.011/1.014 => excluded by the packing discriminator; c5/c7
    // route-invariant controls flat (0.998/1.001). MXFP8_PP224=0 disables.
    static const int pp224_force = [] {
        const char* e = getenv("MXFP8_PP224");
        return (e && atoi(e) >= 0) ? atoi(e) : 1;
    }();
    const bool use_pp224 = (pp224_force == 1) && pp_route && (n >= 224) &&
                           (tiles224 >= nsm) && !use224c2 && !use_cg2pp && !hard256 &&
                           !use_pdbmc && !use_pdbcg2_forced &&
                           ((tiles224 + nsm - 1) / nsm <= (tiles256 + nsm - 1) / nsm);
    const bool use_pdb224_pre = (pdb240_force != 1) && !use224c2 && (n >= 224) && (tiles224 >= nsm) && pdb_gate && !pp_route && !hard256 && !use_pdbmc && !use_pdbcg2_forced;
    // V6 heavy window (job 13 dv_756ecb746cb5 / job 14): persistent
    // BN240 double-buffer for multi-wave heavies. mt>=2 (ABOX mt1
    // stays np), tiles240 > 2*nsm (strict: keeps c8-class
    // tiles224==2nsm on 224c2; <=nsm single-wave stays np),
    // k <= 6144 (deep-k c5 LOSES: 1.033/1.103 measured), !pdb_gate
    // (pdb_gate shapes keep the pdb224 default: c2 stolen by the
    // unfixed window cost +9.8%). Fires fixed {c7, c9}.
    // V6.1 refinement (job 17 dv_480e3275c8b2 official calibration):
    // pdb240-heavy wins ONLY with long chains AND many tiles/CTA
    // (c7 22.5x32 -3.7%, d8 44.6x32 -6.4%); short-chain/few-tile shapes
    // LOSE (d4 2.13x20 +14.1%, d6 6.05x20 +2.4%) - real heavy-pdb240
    // per-stage ~585ns gives drain margin ~2.2x at nk=32 but ~1.4x at
    // nk=20, and ceil(tiles/G) quantization bites below ~15 tiles/CTA.
    // V9 (cycle 10): heavy class -> PERSISTENT SINGLE-D BN256 (pdb256).
    // The "BN256 persistence tmem-infeasible" closure conflated DOUBLE-D
    // (2x256+32 = 544 > 512) with persistence itself: single-D BN256 =
    // 256+2x16 = 288 <= 512 always fits, and gemm_pdb<4,256,CD,false>
    // needs no kernel-body change. The +38-53% single-D verdict
    // (dv_619bb62963e0) is PRE-V5 evidence: post-V5 span ~357-370 (not
    // 559), free-run producer prefetches across tile boundaries, gate ~=
    // tmem_ld phase (~2us/tile) vs ~10us/wave np overhead; single-D also
    // removes the dual-D tmem contention inflating pdb224/240 spans.
    // Window: mt>=2 (ABOX mt1 stays np), tiles256 >= 3*nsm (quantization:
    // fires c5 6.9x / c7 20.7x / c9 3.46x; excludes c4/c8 1.6-1.7x, d4
    // 1.95x, d9 2.7x), k >= 1024 (chain/tile >> gate; has_c raises it to
    // 2048: C-fusion drain vs chain balance), !pdb_gate (c2-class keeps
    // proven pdb224 DBL_D). SUPERSET of the v6.1 pdb240-heavy window
    // (t240 >= 12nsm => t256 >= 11.25nsm) => pdb240 reverts to knob-only.
    // V33 (J15 dv_0cdfb19430ea): sub-floor band [1.5nsm, 3nsm) routed to
    // pdb256-PP when pp-eligible (tl256*nk >= 80). Measured: d5 0.8819,
    // d9alias 0.8436, zone 0.7961 (pp-routed); d4-class (tl*nk=40,
    // classic-routed) LOST 1.6420 => pp-eligibility is part of the gate
    // (fresh measurement overturned the archived v16 classic-band result:
    // classic short-chain persistence pays per-tile X without the PP
    // compensation; np256's wave structure wins there). 224c2-window
    // (c4/c8) and pdb_gate cells keep production routing; fixed-neutral
    // (no fixed cell in band). pp gate follows via the unified clause.
    const bool band256 = !use224c2 && !use_pdb224_pre && (n >= 256) && (mt >= 2) &&
                         !pdb_gate && (tiles256 * 2 >= 3 * nsm) &&
                         (tiles256 < 3 * nsm) &&
                         ((int64_t)tl256 * (k / 128) >= 80);
    const bool use_pdb256_pre = !use224c2 && !use_pp224 && !hard256 && !use_pdbmc && !use_pdbcg2_forced && (n >= 256) && (mt >= 2) &&
                            ((((tiles256 >= 3 * nsm) && !pdb_gate) || pp_route) || band256) &&
                            (has_c ? (k >= 2048) : (k >= 1024));
    const bool use_pdb240 = (pdb240_force == 1) && !use_pdb256_pre && !use224c2 && !hard256 && !use_pdbmc && !use_pdbcg2_forced &&
                            (n >= 240) && (tiles240 >= nsm) && pdb_gate;
    // V40b auto-steal: deep-k heavies from the 1-SM persistent arms. Chain-
    // depth condition tl256*nk >= 288 keeps the validated winners (c5/c7 672,
    // c9 384, d4 288, d6 1344) and excludes the measured short-chain loser
    // (2048,8192,4096: tl256*nk = 128, 0.971x). mt%2==0 excludes pair-tail
    // waste (m1025-class 42%). pdb224-class (d9) has no depth condition (the
    // pair arm beat it 1.060x outright).
    const bool pdbcg2_auto = (pdbcg2_force == 2) && !use_pdbmc && !use_pdbcg2_forced &&
                             (k >= 4096) && (mt >= 2) && (mt % 2 == 0) && (n >= 256) &&
                             ((use_pdb256_pre && ((int64_t)tl256 * (k / 128) >= 288)) ||
                              use_pdb224_pre);
    const bool use_pdbcg2 = use_pdbcg2_forced || pdbcg2_auto;
    const bool use_pdb256 = use_pdb256_pre && !pdbcg2_auto;
    const bool use_pdb224 = use_pdb224_pre && !pdbcg2_auto;
    const bool use_pdb_any = use_pdb256 || use_pdb240 || use_pdb224 || use_pp224;
    const bool use256 = !use64 && !use_pdb_any && (n >= 256) &&
                        (force256 >= 1 || eff(tiles256) >= 0.85 * eff(tiles128));
    // BN128 dual mode: 3-stage/2-CTA-per-SM wins when tiles fill both CTA slots
    // wave after wave (A/B dv_d722ae1cef52: c9 1.41x, c2 1.13x, d9 1.08x);
    // single-wave shapes keep the 6-stage deep prefetch (c3/d2/d3/d8 regress
    // ~6% under 2-CTA).
    const bool use2sm = !use256 && !use64 && !use_pdb_any && (tiles128 >= 256);

// cg2 (2-CTA M=256, BN=240) dev gate: forced via MXFP8_CG2=1 until A/B-verified
    static const int cg2_force = [] {
        const char* e = getenv("MXFP8_CG2");
        return (e && atoi(e) > 0) ? atoi(e) : 0;
    }();
    const int cg2_bn = (cg2_force == 2) ? 256 : 240;
    const bool use_cg2 = (cg2_force >= 1) && (m >= 256) && (n >= cg2_bn);

    // V3.16 deep-k split-K (runtime geometry only): 6-stage BN128 class with
    // tiles128 <= nsm/2 AND >= 3 feasible splits of >= 8 stages each. Evidence
    // (dv_28d28e884bda): d0 (m2 k12288, sk=4) 82->27.4us = -67%; sk=2 shallow-k
    // (c0/c1 nk=20) gained nothing => gated to sk>=3. Runs in the ISOLATED
    // gemm_splitk kernel; classic arms stay byte-identical to join-best.
    int splitk = 1, kchunk = 0;
    if (!use_pdb_any && !use256 && !use64 && !use2sm && !use_cg2 && !use_cg2pp &&
        !use_pdbmc && !use_pdbcg2_forced && tiles128 <= nsm / 2 && (n % 4) == 0) {
        const int nk_st = (k / 128 + 1) / 2;    // V3.21: BK256 stages (k%256==128
        int sk = nsm / tiles128;                // tail handled by TMA OOB fill +
        if (sk > 4) sk = 4;                     // 2^-127 zero-scales => exact)
        const int max_sk = nk_st / 4;           // >= 4 BK256 stages per split
        if (sk > max_sk) sk = max_sk;
        if (sk >= skmin_gate) {
            kchunk = (nk_st + sk - 1) / sk;
            splitk = (nk_st + kchunk - 1) / kchunk;   // all splits non-empty
        }
    }
    // MXFP8_SKF dev arm: forced split for the SINGLE-WAVE BN128 fallback class
    // (tiles128 <= nsm; c3 128x48-stage chain ~36us of the measured 47-48).
    if (splitk < 2 && skf_force >= 2 && !use_pdb_any && !use256 && !use64 &&
        !use2sm && !use_cg2 && !use_cg2pp && !use_pdbmc && !use_pdbcg2_forced && tiles128 <= nsm && (n % 4) == 0) {
        const int nk_st = (k / 128 + 1) / 2;
        int sk = skf_force;
        const int max_sk = nk_st / 4;
        if (sk > max_sk) sk = max_sk;
        if (sk >= 2 && tiles128 * sk <= 2 * nsm) {
            kchunk = (nk_st + sk - 1) / sk;
            splitk = (nk_st + kchunk - 1) / kchunk;
        }
    }
    const bool fb6 = (fb6_force == 1) && (splitk < 2);

    constexpr int SM256 = 4 * 57344 + 4 * 32 + 8;    // 229512 <= 232448, 1 CTA/SM (V5: 4 barrier arrays)
    constexpr int SM128_2 = 3 * 37888 + 3 * 32 + 8;  // 113768: 2 CTAs/SM = 227536 (V5)
    constexpr int SM128_1 = 6 * 37888 + 6 * 16 + 8;  // 227432: gemm_splitk (6st BK128)
    // MXFP8_FB6ST dev arm: gemm_umma<6,128,..,BK128> has FOUR barrier arrays
    // (full/empty/fullsf/sfready = 6*32B) unlike splitk's two => own sizing.
    constexpr int SM128_6 = 6 * 37888 + 6 * 32 + 8;  // 227528
    // V3.20 fallback arm: 3-stage BK256 (75776B stages; probe cfg2: 747ns/stage
    // = 373.5/K128 vs 476 for 6st-BK128 => -21.5% chain on single-wave long-k)
    constexpr int SM128_3K = 3 * 75776 + 3 * 32 + 8;  // 227432: 1 CTA/SM (V5; splitk over-allocated, harmless)
    // V3.22: BN224 2-stage at 2 CTAs/SM: 2x104488 = 208976 <= 232448 smem;
    // tmem 2x256 = 512 exact. Probe cfg3 (dv_459839d7ed76): shared-rate
    // full stage 888ns < every break-even (c5 972 / c7 1155 / c9 1197);
    // co-residency hides E/churn (X_exposed ~3.5 vs 9.2). PRE=1 validated
    // by the probe itself (VAR5 ran exactly this config).
    constexpr int ST224 = (16384 + 224 * 128 + 352 * 16 + 352 * 4 + 1023) & ~1023;  // 52224
    constexpr int SM224_2 = 2 * ST224 + 2 * 32 + 8;   // 104520 per CTA (V5)
    constexpr int ST64_RAW = 128 * 128 + 64 * 128 + (128 + 64) * 16 + (128 + 64) * 4;
    constexpr int SM64 = 8 * ((ST64_RAW + 1023) & ~1023) + 8 * 32 + 8;  // 229640 (V5)
    // splitk branch has launch priority and is BN128: its maps must not
    // follow use224c2 (knob-force + split shape => box/TX mismatch => the
    // dv_b6b7ca302782 d0 illegal-access).
    const int bnsel = (use_pdbmc || use_pdbcg2) ? 256 : ((splitk >= 2) ? 128 : (use224c2 ? 224 : ((use256 || use_pdb256) ? 256 : (use64 ? 64 : 128))));
    const int smem_bytes = use224c2 ? SM224_2 : (use256 ? SM256 : (use64 ? SM64 : (use2sm ? SM128_2 : (fb6 ? SM128_6 : SM128_3K))));
    const int tiles = use224c2 ? tiles224 : (use256 ? tiles256 : (use64 ? tiles64 : tiles128));
    dim3 grid(tiles), block(128);
    const long u8 = 8L * 128 + ((long)nsm + 7) / 8 * 128;
    const long u16 = 16L * 128 + ((long)nsm + 15) / 16 * 128;
    int group = (u16 < u8) ? 16 : 8;
    // N39 (arm-a lineage, 6x replicated; J26 probe-1 dv_15c4b29f0c82
    // CONFIRMED ON THIS BASE: c7 0.9831 med / 0.9796 min, nt48-transfer
    // 0.988/0.990, nt40 0.985/0.988 single-measurement note only; nt16
    // control 1.062 regression => gate holds at nt>=48): wide-n pdb256-class
    // switches to M-major L2 ordering.
    if ((use_pdb256 || use_pdbmc || use_pdbcg2) && ((n + 255) / 256) >= 48) group = 64;
    // dev-only tuning knob (unset in production; default keeps the heuristic)
    static const int group_override = [] {
        const char* e = getenv("MXFP8_GROUP");
        return (e && atoi(e) > 0) ? atoi(e) : 0;
    }();
    if (group_override > 0) group = group_override;

    // tensor maps (cached by pointer+geometry; metadata-only, no output caching)
    const CUtensorMap* tmA = get_map(aq.data_ptr(), k, m, aq.stride(0), 128, 128, 0, 128);
    const CUtensorMap* tmB = get_map(bq.data_ptr(), k, n, bq.stride(0), 128, bnsel, 0, 128);
    const CUtensorMap* tmSA = get_map(sa.data_ptr(), k / 32, m, sa.stride(0) * 4, 4, 128, 2, 0);
    const CUtensorMap* tmSB = get_map(sb.data_ptr(), k / 32, n, sb.stride(0) * 4, 4, bnsel, 2, 0);
    const CUtensorMap* tmB240 = use_pdb240 ? get_map(bq.data_ptr(), k, n, bq.stride(0), 128, 240, 0, 128) : nullptr;
    const CUtensorMap* tmSB240 = use_pdb240 ? get_map(sb.data_ptr(), k / 32, n, sb.stride(0) * 4, 4, 240, 2, 0) : nullptr;
    const CUtensorMap* tmB224 = (use_pdb224 || use_pp224) ? get_map(bq.data_ptr(), k, n, bq.stride(0), 128, 224, 0, 128) : nullptr;
    const CUtensorMap* tmSB224 = (use_pdb224 || use_pp224) ? get_map(sb.data_ptr(), k / 32, n, sb.stride(0) * 4, 4, 224, 2, 0) : nullptr;

    // V3.17 ABOX: mt==1 shapes haul 44-99% OOB zero-fill in every A stage
    // tile (c0 m=1: 15.9KB of 16KB). Shrink the A TMA box to ceil8(m) rows;
    // kernel TX matches via the ABOX template flag (umma arms) or in-body
    // (gemm_splitk, whose class is always mt==1). Map/instantiation pairing
    // is strict: a classic instantiation must NEVER receive the small map
    // (TX mismatch => hang).
    const bool mt1 = (mt == 1);
    const int abox = mt1 ? ((m < 128 ? m : 128) + 7) & ~7 : 128;
    const CUtensorMap* tmA_small = nullptr;
    const CUtensorMap* tmSA_small = nullptr;   // V27 SAF-ABOX
    if (mt1 && !use_pdb_any && !use_cg2 && (!use64 || bn64_force >= 2)) {
        tmA_small = get_map(aq.data_ptr(), k, m, aq.stride(0), 128, abox, 0, 128);
        tmSA_small = get_map(sa.data_ptr(), k / 32, m, sa.stride(0) * 4, 4, abox, 2, 0);
    }
    const bool abox_on = (tmA_small != nullptr) && !use2sm && !use224c2 && (splitk < 2);
    const CUtensorMap* tmA_use = ((abox_on || splitk >= 2) && tmA_small) ? tmA_small : tmA;
    // V27: null-safe pair of tmA_use (strict pairing: ABOX/splitk
    // instantiations only ever see the small map when it was built).
    const CUtensorMap* tmSA_use = ((abox_on || splitk >= 2) && tmSA_small) ? tmSA_small : tmSA;

    // V25 PREPACK gate (mechanism reference: arm-a N6/N17/N19, join
    // 20261025 exchange; peer official A/B: c5/c7 -8.9%, c9 -6.6% on their
    // non-persistent arm; cycle-31 fit independently located ~80-95
    // cycles/stage of SF-path overhead in THIS arm). Gate mirrors the
    // peer's measured economics: deep chains (nk>=64) or many waves
    // (>=10) amortize the ~2-5us prepack kernel; short/low-wave pdb256
    // cells stay classic (peer's d1-class +18% lesson). Fires fixed
    // {c5,c7,c9} + domain {d8}. Buffer cache = ALLOCATION ELISION ONLY
    // (splitk ws_cache precedent): every forward recomputes all bytes.
    const int waves256 = tl256;   // V29: tl256 hoisted above the arm gates
    const bool pp_pdb = use_pdb256 && ((k / 128) >= 64 || waves256 >= 10 ||
                                       (int64_t)waves256 * (k / 128) >= 80);   // V29 unified economics
    const bool pp_chain_here = pp_pdb && !use_cg2 && !use_cg2pp;   // V35: chain hoisted into the pp block
    // V36: PP launch macro (stream-parameterized) hoisted above the pp block
    // so the two-kernel chain is contiguous inside the launch lambda.
#define SM_PDB_OF(ST, BN) ((ST) * ((16384 + (BN) * 128 + (128 + (BN)) * 16 +                   \
                                   (128 + (BN)) * 4 + 1023) & ~1023) + ((ST) * 4 + 6) * 8 + 8)
#define MXFP8_LAUNCH_PDB_PP(ST, BN, CD_T, CD_P, TILES, EE, LST)                              \
    {                                                                                           \
        constexpr int SM_PDB = SM_PDB_OF(ST, BN);                                               \
        static bool attr_done_pp = false;                                                       \
        if (!attr_done_pp) {                                                                    \
            cudaFuncSetAttribute(gemm_pdb<ST, BN, CD_T, true, true, EE>,                        \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SM_PDB);          \
            attr_done_pp = true;                                                                \
        }                                                                                       \
        dim3 gridp((TILES) < nsm ? (TILES) : nsm);                                              \
        dim3 blockp(256);                                                                       \
        gemm_pdb<ST, BN, CD_T, true, true, EE><<<gridp, blockp, SM_PDB, LST>>>(                \
            m, n, k, (float)alpha, group, *tmA, *tmB, *tmSA_pp, *tmSB_pp,                       \
            CD_P, d.stride(0), reinterpret_cast<const CD_T*>(cptr), c_stride0, has_c ? 1 : 0);  \
    }
    const CUtensorMap* tmSA_pp = nullptr;
    const CUtensorMap* tmSB_pp = nullptr;
    if (pp_pdb && !use_cg2pp && !use_pdbmc) {
        const int nk128 = (int)(k / 128);
        const int pad_m = mt * 128;
        const int pad_n = ((n + 255) / 256) * 256;
        const int64_t ba = (int64_t)nk128 * pad_m * 4;
        const int64_t bb = (int64_t)nk128 * pad_n * 4;
        static std::map<std::pair<int64_t, int64_t>,
                        std::pair<at::Tensor, at::Tensor>> pp_cache;
        auto it = pp_cache.find({ba, bb});
        if (it == pp_cache.end()) {
            auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(aq.device());
            it = pp_cache.emplace(std::make_pair(ba, bb),
                                  std::make_pair(at::empty({ba}, opts),
                                                 at::empty({bb}, opts))).first;
        }
        uint8_t* sap = it->second.first.data_ptr<uint8_t>();
        uint8_t* sbp = it->second.second.data_ptr<uint8_t>();
        dim3 pgrid((unsigned)((pad_m + pad_n + 255) / 256), (unsigned)nk128), pblock(256);
        // V36: two-kernel chain as one lambda; graph mechanics in noinline
        // helpers (cold path). Semantics identical to v35fix.
        auto pp_launch = [&](cudaStream_t st) {
            prepack_sf<<<pgrid, pblock, 0, st>>>(
                reinterpret_cast<const float*>(sa.data_ptr()), (int)m, (int)sa.stride(0),
                reinterpret_cast<const float*>(sb.data_ptr()), (int)n, (int)sb.stride(0),
                sap, pad_m, sbp, pad_n);
            tmSA_pp = get_map(sap, pad_m, nk128, (int64_t)pad_m * 4, 128, 1, 2, 0);
            tmSB_pp = get_map(sbp, pad_n, nk128, (int64_t)pad_n * 4, 256, 1, 2, 0);
            if (is_f32) { float* dp = reinterpret_cast<float*>(d.data_ptr());
                if (ee_force) { MXFP8_LAUNCH_PDB_PP(4, 256, float, dp, tiles256, true, st) }
                else { MXFP8_LAUNCH_PDB_PP(4, 256, float, dp, tiles256, false, st) } }
            else { __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(d.data_ptr());
                if (ee_force) { MXFP8_LAUNCH_PDB_PP(4, 256, __nv_bfloat16, dp, tiles256, true, st) }
                else { MXFP8_LAUNCH_PDB_PP(4, 256, __nv_bfloat16, dp, tiles256, false, st) } }
        };
        int pp_mode = mxfp8::PPG_EAGER;
        if (pp_chain_here) {
            mxfp8::PPGEnt* pp_ent = nullptr;
            cudaGraphExec_t pp_exec = nullptr;
            cudaStream_t pp_st = stream;
            mxfp8::PPGArgs ga{(uintptr_t)aq.data_ptr(), (uintptr_t)sa.data_ptr(),
                              (uintptr_t)bq.data_ptr(), (uintptr_t)sb.data_ptr(),
                              (uintptr_t)d.data_ptr(), has_c ? (uintptr_t)cptr : 0ull,
                              (uintptr_t)sap, (uintptr_t)sbp,
                              (int64_t)m, (int64_t)n, (int64_t)k,
                              (int64_t)sa.stride(0), (int64_t)sb.stride(0),
                              (int64_t)aq.stride(0), (int64_t)bq.stride(0),
                              (int64_t)d.stride(0), (int64_t)c_stride0,
                              is_f32 ? 1 : 0, has_c ? 1 : 0,
                              (int32_t)d.device().index(), mxfp8::f2bits((float)alpha)};
            pp_mode = mxfp8::ppg_begin(ga, stream, &pp_ent, &pp_exec, &pp_st);
            if (pp_mode != mxfp8::PPG_REPLAY) {
                pp_launch(pp_st);
                if (pp_mode == mxfp8::PPG_CAPTURE) {
                    pp_exec = mxfp8::ppg_end(pp_ent);
                    if (pp_exec) { cudaGraphLaunch(pp_exec, stream); ++mxfp8::g_ppg_launches; }
                    else { pp_launch(stream); }   // capture ran nothing: eager result
                }
            } else {
                cudaGraphLaunch(pp_exec, stream);
                ++mxfp8::g_ppg_launches;
            }
        } else {
            // pp_pdb under the diagnostic cg2 force: v33-faithful behavior -
            // prepack eager here; the ladder's use_cg2 branch issues the gemm.
            prepack_sf<<<pgrid, pblock, 0, stream>>>(
                reinterpret_cast<const float*>(sa.data_ptr()), (int)m, (int)sa.stride(0),
                reinterpret_cast<const float*>(sb.data_ptr()), (int)n, (int)sb.stride(0),
                sap, pad_m, sbp, pad_n);
            tmSA_pp = get_map(sap, pad_m, nk128, (int64_t)pad_m * 4, 128, 1, 2, 0);
            tmSB_pp = get_map(sbp, pad_n, nk128, (int64_t)pad_n * 4, 256, 1, 2, 0);
        }
    }

// J26 probe: PP-pdb224 launch macro + chain (mirrors the V36 pp pattern so
// both A/B arms carry the same launch-overhead class; additive/cold).
#define MXFP8_LAUNCH_PDB_PP224(ST, BN, CD_T, CD_P, TILES, LST)                                \
    {                                                                                           \
        constexpr int SM_PDB224 = SM_PDB_OF(ST, BN);                                            \
        static bool attr_done_pp224 = false;                                                    \
        if (!attr_done_pp224) {                                                                 \
            cudaFuncSetAttribute(gemm_pdb<ST, BN, CD_T, true, true>,                            \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SM_PDB224);       \
            attr_done_pp224 = true;                                                             \
        }                                                                                       \
        dim3 gridp((TILES) < nsm ? (TILES) : nsm);                                              \
        dim3 blockp(256);                                                                       \
        gemm_pdb<ST, BN, CD_T, true, true><<<gridp, blockp, SM_PDB224, LST>>>(                  \
            m, n, k, (float)alpha, group, *tmA, *tmB224, *tmSA_pp224, *tmSB_pp224,              \
            CD_P, d.stride(0), reinterpret_cast<const CD_T*>(cptr), c_stride0, has_c ? 1 : 0);  \
    }
    if (use_pp224) {
        const int nk128 = (int)(k / 128);
        const int pad_m = mt * 128;
        const int nt224 = (n + 223) / 224;
        const int pad_n = nt224 * 224;
        const int64_t ba = (int64_t)nk128 * pad_m * 4;
        const int64_t bb = (int64_t)nk128 * nt224 * 1024;
        static std::map<std::pair<int64_t, int64_t>,
                        std::pair<at::Tensor, at::Tensor>> pp224_cache;   // ALLOCATION ELISION ONLY
        auto it = pp224_cache.find({ba, bb});
        if (it == pp224_cache.end()) {
            auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(aq.device());
            it = pp224_cache.emplace(std::make_pair(ba, bb),
                                  std::make_pair(at::empty({ba}, opts),
                                                 at::empty({bb}, opts))).first;
        }
        uint8_t* sap2 = it->second.first.data_ptr<uint8_t>();
        uint8_t* sbp2 = it->second.second.data_ptr<uint8_t>();
        const CUtensorMap* tmSA_pp224 = nullptr;
        const CUtensorMap* tmSB_pp224 = nullptr;
        const int bthreads = nt224 * 256;
        dim3 pgrid((unsigned)((pad_m + bthreads + 255) / 256), (unsigned)nk128), pblock(256);
        auto pp224_launch = [&](cudaStream_t st) {
            prepack_sf2<<<pgrid, pblock, 0, st>>>(
                reinterpret_cast<const float*>(sa.data_ptr()), (int)m, (int)sa.stride(0),
                reinterpret_cast<const float*>(sb.data_ptr()), (int)n, (int)sb.stride(0),
                sap2, pad_m, sbp2, pad_n, 224);
            tmSA_pp224 = get_map(sap2, pad_m, nk128, (int64_t)pad_m * 4, 128, 1, 2, 0);
            tmSB_pp224 = get_map(sbp2, nt224 * 256, nk128, (int64_t)nt224 * 1024, 256, 1, 2, 0);
            if (is_f32) { float* dp = reinterpret_cast<float*>(d.data_ptr());
                MXFP8_LAUNCH_PDB_PP224(4, 224, float, dp, tiles224, st) }
            else { __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(d.data_ptr());
                MXFP8_LAUNCH_PDB_PP224(4, 224, __nv_bfloat16, dp, tiles224, st) }
        };
        int pp224_mode = mxfp8::PPG_EAGER;
        mxfp8::PPGEnt* pp224_ent = nullptr;
        cudaGraphExec_t pp224_exec = nullptr;
        cudaStream_t pp224_st = stream;
        mxfp8::PPGArgs ga{(uintptr_t)aq.data_ptr(), (uintptr_t)sa.data_ptr(),
                          (uintptr_t)bq.data_ptr(), (uintptr_t)sb.data_ptr(),
                          (uintptr_t)d.data_ptr(), has_c ? (uintptr_t)cptr : 0ull,
                          (uintptr_t)sap2, (uintptr_t)sbp2,
                          (int64_t)m, (int64_t)n, (int64_t)k,
                          (int64_t)sa.stride(0), (int64_t)sb.stride(0),
                          (int64_t)aq.stride(0), (int64_t)bq.stride(0),
                          (int64_t)d.stride(0), c_stride0,
                          is_f32 ? 1 : 0, has_c ? 1 : 0,
                          (int32_t)d.device().index(), mxfp8::f2bits((float)alpha)};
        pp224_mode = mxfp8::ppg_begin(ga, stream, &pp224_ent, &pp224_exec, &pp224_st);
        if (pp224_mode != mxfp8::PPG_REPLAY) {
            pp224_launch(pp224_st);
            if (pp224_mode == mxfp8::PPG_CAPTURE) {
                pp224_exec = mxfp8::ppg_end(pp224_ent);
                if (pp224_exec) { cudaGraphLaunch(pp224_exec, stream); ++mxfp8::g_ppg_launches; }
                else { pp224_launch(stream); }   // capture ran nothing: eager result
            }
        } else {
            cudaGraphLaunch(pp224_exec, stream);
            ++mxfp8::g_ppg_launches;
        }
    }

// V37: cg2pp launch macro + prepack/graph chain (mirrors the V36 pp pattern so
// both A/B arms carry the same launch-overhead class; additive/cold).
#define MXFP8_LAUNCH_CG2PP(ST, BN, CD_T, CD_P, TMBH, TMSB, LST)                                \
    {                                                                                           \
        constexpr int STB_PP2 = (16384 + (BN / 2) * 128 + 3072 + 1023) & ~1023;                 \
        constexpr int SM_CG2PP = (ST) * STB_PP2 + (ST) * 24 + 136;                              \
        static bool attr_done_cg2pp = false;                                                    \
        if (!attr_done_cg2pp) {                                                                 \
            cudaFuncSetAttribute(gemm_cg2pp<ST, BN, CD_T>,                                      \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SM_CG2PP);        \
            attr_done_cg2pp = true;                                                             \
        }                                                                                       \
        dim3 grid2p(2 * ((m + 255) / 256) * ((n + BN - 1) / BN));                               \
        gemm_cg2pp<ST, BN, CD_T><<<grid2p, block, SM_CG2PP, LST>>>(                             \
            m, n, k, (float)alpha, group, *tmA, *(TMBH), *tmSA_pp2, *(TMSB),                    \
            CD_P, d.stride(0), reinterpret_cast<const CD_T*>(cptr), c_stride0, has_c ? 1 : 0,   \
            cg2_sig);                                                                           \
    }
    if (use_cg2pp) {
        const int nk128 = (int)(k / 128);
        const int pad_m = mt * 128;
        const int nt240 = (n + 239) / 240;
        const int pad_n = (cg2pp_bn == 256) ? (((n + 255) / 256) * 256) : (nt240 * 240);
        const int64_t ba = (int64_t)nk128 * pad_m * 4;
        const int64_t bb = (cg2pp_bn == 256) ? (int64_t)nk128 * pad_n * 4
                                             : (int64_t)nk128 * nt240 * 1024;
        static std::map<std::pair<int64_t, int64_t>,
                        std::pair<at::Tensor, at::Tensor>> cg2pp_cache;   // ALLOCATION ELISION ONLY
        auto it = cg2pp_cache.find({ba, bb});
        if (it == cg2pp_cache.end()) {
            auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(aq.device());
            it = cg2pp_cache.emplace(std::make_pair(ba, bb),
                                     std::make_pair(at::empty({ba}, opts),
                                                    at::empty({bb}, opts))).first;
        }
        uint8_t* sap3 = it->second.first.data_ptr<uint8_t>();
        uint8_t* sbp3 = it->second.second.data_ptr<uint8_t>();
        const CUtensorMap* tmSA_pp2 = nullptr;
        const CUtensorMap* tmSB_pp2 = nullptr;
        const CUtensorMap* tmBh2 = get_map(bq.data_ptr(), k, n, bq.stride(0),
                                           128, cg2pp_bn / 2, 0, 128);
        dim3 pblock(256);
        auto cg2pp_launch = [&](cudaStream_t st) {
            if (cg2pp_bn == 256) {
                dim3 pg((unsigned)((pad_m + pad_n + 255) / 256), (unsigned)nk128);
                prepack_sf<<<pg, pblock, 0, st>>>(
                    reinterpret_cast<const float*>(sa.data_ptr()), (int)m, (int)sa.stride(0),
                    reinterpret_cast<const float*>(sb.data_ptr()), (int)n, (int)sb.stride(0),
                    sap3, pad_m, sbp3, pad_n);
                tmSA_pp2 = get_map(sap3, pad_m, nk128, (int64_t)pad_m * 4, 128, 1, 2, 0);
                tmSB_pp2 = get_map(sbp3, pad_n, nk128, (int64_t)pad_n * 4, 256, 1, 2, 0);
                if (is_f32) { float* dp = reinterpret_cast<float*>(d.data_ptr());
                    MXFP8_LAUNCH_CG2PP(6, 256, float, dp, tmBh2, tmSB_pp2, st) }
                else { __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(d.data_ptr());
                    MXFP8_LAUNCH_CG2PP(6, 256, __nv_bfloat16, dp, tmBh2, tmSB_pp2, st) }
            } else {
                const int bthreads = nt240 * 256;
                dim3 pg((unsigned)((pad_m + bthreads + 255) / 256), (unsigned)nk128);
                prepack_sf2<<<pg, pblock, 0, st>>>(
                    reinterpret_cast<const float*>(sa.data_ptr()), (int)m, (int)sa.stride(0),
                    reinterpret_cast<const float*>(sb.data_ptr()), (int)n, (int)sb.stride(0),
                    sap3, pad_m, sbp3, pad_n, 240);
                tmSA_pp2 = get_map(sap3, pad_m, nk128, (int64_t)pad_m * 4, 128, 1, 2, 0);
                tmSB_pp2 = get_map(sbp3, nt240 * 256, nk128, (int64_t)nt240 * 1024, 256, 1, 2, 0);
                if (is_f32) { float* dp = reinterpret_cast<float*>(d.data_ptr());
                    MXFP8_LAUNCH_CG2PP(6, 240, float, dp, tmBh2, tmSB_pp2, st) }
                else { __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(d.data_ptr());
                    MXFP8_LAUNCH_CG2PP(6, 240, __nv_bfloat16, dp, tmBh2, tmSB_pp2, st) }
            }
        };
        int cg2pp_mode = mxfp8::PPG_EAGER;
        mxfp8::PPGEnt* cg2pp_ent = nullptr;
        cudaGraphExec_t cg2pp_exec = nullptr;
        cudaStream_t cg2pp_st = stream;
        mxfp8::PPGArgs ga{(uintptr_t)aq.data_ptr(), (uintptr_t)sa.data_ptr(),
                          (uintptr_t)bq.data_ptr(), (uintptr_t)sb.data_ptr(),
                          (uintptr_t)d.data_ptr(), has_c ? (uintptr_t)cptr : 0ull,
                          (uintptr_t)sap3, (uintptr_t)sbp3,
                          (int64_t)m, (int64_t)n, (int64_t)k,
                          (int64_t)sa.stride(0), (int64_t)sb.stride(0),
                          (int64_t)aq.stride(0), (int64_t)bq.stride(0),
                          (int64_t)d.stride(0), c_stride0,
                          is_f32 ? 1 : 0, has_c ? 1 : 0,
                          (int32_t)d.device().index(), mxfp8::f2bits((float)alpha)};
        cg2pp_mode = mxfp8::ppg_begin(ga, stream, &cg2pp_ent, &cg2pp_exec, &cg2pp_st);
        if (cg2pp_mode != mxfp8::PPG_REPLAY) {
            cg2pp_launch(cg2pp_st);
            if (cg2pp_mode == mxfp8::PPG_CAPTURE) {
                cg2pp_exec = mxfp8::ppg_end(cg2pp_ent);
                if (cg2pp_exec) { cudaGraphLaunch(cg2pp_exec, stream); ++mxfp8::g_ppg_launches; }
                else { cg2pp_launch(stream); }   // capture ran nothing: eager result
            }
        } else {
            cudaGraphLaunch(cg2pp_exec, stream);
            ++mxfp8::g_ppg_launches;
        }
    }

// V38: pdbmc launch macro + prepack/graph chain (same launch-overhead class
// as the pp chains; BN256-only, PP prepack required).
#define MXFP8_LAUNCH_PDBMC(ST, BN, CD_T, CD_P, LST)                                              \
    {                                                                                             \
        constexpr int STB_MC = (16384 + (BN) * 128 + (128 + (BN)) * 16 +                          \
                                (128 + (BN)) * 4 + 1023) & ~1023;                                 \
        constexpr int SM_PDBMC = (ST) * STB_MC + 320;                                             \
        static bool attr_done_mc = false;                                                         \
        if (!attr_done_mc) {                                                                      \
            cudaFuncSetAttribute(gemm_pdbmc<ST, BN, CD_T>,                                        \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SM_PDBMC);          \
            attr_done_mc = true;                                                                  \
        }                                                                                         \
        const int nmp = (m + 511) / 512;                                                          \
        const int ptot = nmp * ((n + (BN) - 1) / (BN));                                           \
        const int gpc = (ptot < (nsm / 2)) ? ptot : (nsm / 2);                                    \
        dim3 gridmc(2 * gpc);                                                                     \
        dim3 blockmc(256);                                                                        \
        gemm_pdbmc<ST, BN, CD_T><<<gridmc, blockmc, SM_PDBMC, LST>>>(                             \
            m, n, k, (float)alpha, group, *tmA, *tmB, *tmSA_mc, *tmSB_mc,                         \
            CD_P, d.stride(0), reinterpret_cast<const CD_T*>(cptr), c_stride0, has_c ? 1 : 0);    \
    }
    if (use_pdbmc) {
        const int nk128 = (int)(k / 128);
        const int pad_m = mt * 128;
        const int pad_n = ((n + 255) / 256) * 256;
        const int64_t ba = (int64_t)nk128 * pad_m * 4;
        const int64_t bb = (int64_t)nk128 * pad_n * 4;
        static std::map<std::pair<int64_t, int64_t>,
                        std::pair<at::Tensor, at::Tensor>> pdbmc_cache;   // ALLOCATION ELISION ONLY
        auto it = pdbmc_cache.find({ba, bb});
        if (it == pdbmc_cache.end()) {
            auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(aq.device());
            it = pdbmc_cache.emplace(std::make_pair(ba, bb),
                                     std::make_pair(at::empty({ba}, opts),
                                                    at::empty({bb}, opts))).first;
        }
        uint8_t* sap4 = it->second.first.data_ptr<uint8_t>();
        uint8_t* sbp4 = it->second.second.data_ptr<uint8_t>();
        const CUtensorMap* tmSA_mc = nullptr;
        const CUtensorMap* tmSB_mc = nullptr;
        dim3 pblock(256);
        auto pdbmc_launch = [&](cudaStream_t st) {
            dim3 pg((unsigned)((pad_m + pad_n + 255) / 256), (unsigned)nk128);
            prepack_sf<<<pg, pblock, 0, st>>>(
                reinterpret_cast<const float*>(sa.data_ptr()), (int)m, (int)sa.stride(0),
                reinterpret_cast<const float*>(sb.data_ptr()), (int)n, (int)sb.stride(0),
                sap4, pad_m, sbp4, pad_n);
            tmSA_mc = get_map(sap4, pad_m, nk128, (int64_t)pad_m * 4, 128, 1, 2, 0);
            tmSB_mc = get_map(sbp4, pad_n, nk128, (int64_t)pad_n * 4, 256, 1, 2, 0);
            if (is_f32) { float* dp = reinterpret_cast<float*>(d.data_ptr());
                MXFP8_LAUNCH_PDBMC(4, 256, float, dp, st) }
            else { __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(d.data_ptr());
                MXFP8_LAUNCH_PDBMC(4, 256, __nv_bfloat16, dp, st) }
        };
        int pdbmc_mode = mxfp8::PPG_EAGER;
        mxfp8::PPGEnt* pdbmc_ent = nullptr;
        cudaGraphExec_t pdbmc_exec = nullptr;
        cudaStream_t pdbmc_st = stream;
        mxfp8::PPGArgs ga{(uintptr_t)aq.data_ptr(), (uintptr_t)sa.data_ptr(),
                          (uintptr_t)bq.data_ptr(), (uintptr_t)sb.data_ptr(),
                          (uintptr_t)d.data_ptr(), has_c ? (uintptr_t)cptr : 0ull,
                          (uintptr_t)sap4, (uintptr_t)sbp4,
                          (int64_t)m, (int64_t)n, (int64_t)k,
                          (int64_t)sa.stride(0), (int64_t)sb.stride(0),
                          (int64_t)aq.stride(0), (int64_t)bq.stride(0),
                          (int64_t)d.stride(0), c_stride0,
                          is_f32 ? 1 : 0, has_c ? 1 : 0,
                          (int32_t)d.device().index(), mxfp8::f2bits((float)alpha)};
        pdbmc_mode = mxfp8::ppg_begin(ga, stream, &pdbmc_ent, &pdbmc_exec, &pdbmc_st);
        if (pdbmc_mode != mxfp8::PPG_REPLAY) {
            pdbmc_launch(pdbmc_st);
            if (pdbmc_mode == mxfp8::PPG_CAPTURE) {
                pdbmc_exec = mxfp8::ppg_end(pdbmc_ent);
                if (pdbmc_exec) { cudaGraphLaunch(pdbmc_exec, stream); ++mxfp8::g_ppg_launches; }
                else { pdbmc_launch(stream); }   // capture ran nothing: eager result
            }
        } else {
            cudaGraphLaunch(pdbmc_exec, stream);
            ++mxfp8::g_ppg_launches;
        }
    }

// V39: pdbcg2 launch macro + prepack/graph chain (same launch-overhead class
// as the pp/pdbmc chains; BN256-only, PP prepack required, ST=6 ring).
#define MXFP8_LAUNCH_PDBCG2(ST, SFM, PW, CS, BN, CD_T, CD_P, LST)                                    \
    {                                                                                              \
        constexpr int STB_C2 = (16384 + ((BN) / 2) * 128 + 1024 + 2048 + 1023) & ~1023;            \
        constexpr int SM_PDBCG2 = (ST) * STB_C2 + 320;                                             \
        static bool attr_done_c2 = false;                                                          \
        if (!attr_done_c2) {                                                                       \
            cudaFuncSetAttribute(gemm_pdbcg2<ST, BN, SFM, PW, CS, CD_T>,                           \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SM_PDBCG2);          \
            attr_done_c2 = true;                                                                   \
        }                                                                                          \
        const int nmp = (m + 255) / 256;                /* true m-pair count */                    \
        const int ptot = nmp * ((n + (BN) - 1) / (BN));                                            \
        const int gpc = (ptot < (nsm / 2)) ? ptot : (nsm / 2);                                     \
        dim3 gridc2(2 * gpc);                                                                      \
        dim3 blockc2(256);                                                                         \
        gemm_pdbcg2<ST, BN, SFM, PW, CS, CD_T><<<gridc2, blockc2, SM_PDBCG2, LST>>>(               \
            m, n, k, (float)alpha, group, *tmA, *tmBh_c2, *tmSA_c2, *tmSB_c2,                      \
            CD_P, d.stride(0), reinterpret_cast<const CD_T*>(cptr), c_stride0, has_c ? 1 : 0,      \
            cg2_pre_cap, cg2_ee_sel);                                                                  \
    }
    if (use_pdbcg2) {
        const int nk128 = (int)(k / 128);
        const int pad_m = mt * 128;
        const int pad_n = ((n + 255) / 256) * 256;
        const int64_t ba = (int64_t)nk128 * pad_m * 4;
        const int64_t bb = (int64_t)nk128 * pad_n * 4;
        static std::map<std::pair<int64_t, int64_t>,
                        std::pair<at::Tensor, at::Tensor>> pdbcg2_cache;   // ALLOCATION ELISION ONLY
        auto it = pdbcg2_cache.find({ba, bb});
        if (it == pdbcg2_cache.end()) {
            auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(aq.device());
            it = pdbcg2_cache.emplace(std::make_pair(ba, bb),
                                     std::make_pair(at::empty({ba}, opts),
                                                    at::empty({bb}, opts))).first;
        }
        uint8_t* sap5 = it->second.first.data_ptr<uint8_t>();
        uint8_t* sbp5 = it->second.second.data_ptr<uint8_t>();
        const CUtensorMap* tmSA_c2 = nullptr;
        const CUtensorMap* tmSB_c2 = nullptr;
        const CUtensorMap* tmBh_c2 = get_map(bq.data_ptr(), k, n, bq.stride(0), 128, 128, 0, 128);
        dim3 pblock(256);
        auto pdbcg2_launch = [&](cudaStream_t st) {
            dim3 pg((unsigned)((pad_m + pad_n + 255) / 256), (unsigned)nk128);
            prepack_sf<<<pg, pblock, 0, st>>>(
                reinterpret_cast<const float*>(sa.data_ptr()), (int)m, (int)sa.stride(0),
                reinterpret_cast<const float*>(sb.data_ptr()), (int)n, (int)sb.stride(0),
                sap5, pad_m, sbp5, pad_n);
            tmSA_c2 = get_map(sap5, pad_m, nk128, (int64_t)pad_m * 4, 128, 1, 2, 0);
            tmSB_c2 = get_map(sbp5, pad_n, nk128, (int64_t)pad_n * 4, 256, 1, 2, 0);
            // V39.4 instantiation tree (14): ST6 SFM0 full (PW,CS) grid; ST4
            // {(0,0),(0,1)}; SFM1 (0,0). CG2ST=5 routes to ST6 (retired).
            const int pw = cg2_pw_sel, cs = cg2_cs;
            if (is_f32) { float* dp = reinterpret_cast<float*>(d.data_ptr());
                if (cg2sf_mode == 1) { MXFP8_LAUNCH_PDBCG2(6, 1, 0, 0, 256, float, dp, st) }
                else if (cg2st_sel == 4) {
                    if (cs) { MXFP8_LAUNCH_PDBCG2(4, 0, 0, 1, 256, float, dp, st) }
                    else { MXFP8_LAUNCH_PDBCG2(4, 0, 0, 0, 256, float, dp, st) }
                }
                else if (pw && cs) { MXFP8_LAUNCH_PDBCG2(6, 0, 1, 1, 256, float, dp, st) }
                else if (cs) { MXFP8_LAUNCH_PDBCG2(6, 0, 0, 1, 256, float, dp, st) }
                else if (pw) { MXFP8_LAUNCH_PDBCG2(6, 0, 1, 0, 256, float, dp, st) }
                else { MXFP8_LAUNCH_PDBCG2(6, 0, 0, 0, 256, float, dp, st) }
            }
            else { __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(d.data_ptr());
                if (cg2sf_mode == 1) { MXFP8_LAUNCH_PDBCG2(6, 1, 0, 0, 256, __nv_bfloat16, dp, st) }
                else if (cg2st_sel == 4) {
                    if (cs) { MXFP8_LAUNCH_PDBCG2(4, 0, 0, 1, 256, __nv_bfloat16, dp, st) }
                    else { MXFP8_LAUNCH_PDBCG2(4, 0, 0, 0, 256, __nv_bfloat16, dp, st) }
                }
                else if (pw && cs) { MXFP8_LAUNCH_PDBCG2(6, 0, 1, 1, 256, __nv_bfloat16, dp, st) }
                else if (cs) { MXFP8_LAUNCH_PDBCG2(6, 0, 0, 1, 256, __nv_bfloat16, dp, st) }
                else if (pw) { MXFP8_LAUNCH_PDBCG2(6, 0, 1, 0, 256, __nv_bfloat16, dp, st) }
                else { MXFP8_LAUNCH_PDBCG2(6, 0, 0, 0, 256, __nv_bfloat16, dp, st) }
            }
        };
        int pdbcg2_mode = mxfp8::PPG_EAGER;
        mxfp8::PPGEnt* pdbcg2_ent = nullptr;
        cudaGraphExec_t pdbcg2_exec = nullptr;
        cudaStream_t pdbcg2_st = stream;
        mxfp8::PPGArgs ga{(uintptr_t)aq.data_ptr(), (uintptr_t)sa.data_ptr(),
                          (uintptr_t)bq.data_ptr(), (uintptr_t)sb.data_ptr(),
                          (uintptr_t)d.data_ptr(), has_c ? (uintptr_t)cptr : 0ull,
                          (uintptr_t)sap5, (uintptr_t)sbp5,
                          (int64_t)m, (int64_t)n, (int64_t)k,
                          (int64_t)sa.stride(0), (int64_t)sb.stride(0),
                          (int64_t)aq.stride(0), (int64_t)bq.stride(0),
                          (int64_t)d.stride(0), c_stride0,
                          is_f32 ? 1 : 0, has_c ? 1 : 0,
                          (int32_t)d.device().index(), mxfp8::f2bits((float)alpha)};
        pdbcg2_mode = mxfp8::ppg_begin(ga, stream, &pdbcg2_ent, &pdbcg2_exec, &pdbcg2_st);
        if (pdbcg2_mode != mxfp8::PPG_REPLAY) {
            pdbcg2_launch(pdbcg2_st);
            if (pdbcg2_mode == mxfp8::PPG_CAPTURE) {
                pdbcg2_exec = mxfp8::ppg_end(pdbcg2_ent);
                if (pdbcg2_exec) { cudaGraphLaunch(pdbcg2_exec, stream); ++mxfp8::g_ppg_launches; }
                else { pdbcg2_launch(stream); }   // capture ran nothing: eager result
            }
        } else {
            cudaGraphLaunch(pdbcg2_exec, stream);
            ++mxfp8::g_ppg_launches;
        }
    }

#define MXFP8_LAUNCH_PDB(ST, BN, CD_T, CD_P, TMB, TMSB, TILES, DBLD)                           \
    {                                                                                           \
        constexpr int SM_PDB = SM_PDB_OF(ST, BN);                                               \
        static bool attr_done = false;                                                          \
        if (!attr_done) {                                                                       \
            cudaFuncSetAttribute(gemm_pdb<ST, BN, CD_T, DBLD>,                                  \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SM_PDB);          \
            attr_done = true;                                                                   \
        }                                                                                       \
        dim3 gridp((TILES) < nsm ? (TILES) : nsm);                                              \
        dim3 blockp(256);                                                                       \
        gemm_pdb<ST, BN, CD_T, DBLD><<<gridp, blockp, SM_PDB, stream>>>(                        \
            m, n, k, (float)alpha, group, *tmA, *(TMB), *tmSA, *(TMSB),                         \
            CD_P, d.stride(0), reinterpret_cast<const CD_T*>(cptr), c_stride0, has_c ? 1 : 0);  \
    }

// V35: PP macro moved above the pp block (contiguous capturable chain)

#define MXFP8_LAUNCH(ST, BN, CD_T, CD_P, MF, AT, BKv)                                           \
    {                                                                                           \
        static bool attr_done = false;                                                          \
        if (!attr_done) {                                                                       \
            cudaFuncSetAttribute(gemm_umma<ST, BN, CD_T, MF, AT, BKv>,                          \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);      \
            attr_done = true;                                                                   \
        }                                                                                       \
        gemm_umma<ST, BN, CD_T, MF, AT, BKv><<<grid, block, smem_bytes, stream>>>(              \
            m, n, k, (float)alpha, group, *tmA_use, *tmB, *tmSA_use, *tmSB,                     \
            CD_P, d.stride(0), reinterpret_cast<const CD_T*>(cptr), c_stride0, has_c ? 1 : 0);  \
    }

#define MXFP8_LAUNCH_CG2(ST, BN, CD_T, CD_P)                                                    \
    {                                                                                           \
        constexpr int SM_CG2 =                                                                  \
            ST * ((16384 + (BN / 2) * 128 + (128 + BN) * 16 + 3072 + 1023) & ~1023) + 128;      \
        static bool attr_done = false;                                                          \
        if (!attr_done) {                                                                       \
            cudaFuncSetAttribute(gemm_umma_cg2<ST, BN, CD_T>,                                   \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SM_CG2);          \
            attr_done = true;                                                                   \
        }                                                                                       \
        dim3 grid2(2 * ((m + 255) / 256) * ((n + BN - 1) / BN));                                \
        gemm_umma_cg2<ST, BN, CD_T><<<grid2, block, SM_CG2, stream>>>(                          \
            m, n, k, (float)alpha, group, *tmA, *tmBh, *tmSA, *tmSBh,                           \
            CD_P, d.stride(0), reinterpret_cast<const CD_T*>(cptr), c_stride0, has_c ? 1 : 0);  \
    }

    // V39.2 PPG-all prep: splitk workspace hoist. The ws/counter allocation
    // (and the once-per-allocation zero_()) must execute OUTSIDE any graph
    // capture; the cached pointers are insert-only and pointer-stable, and
    // (sk,m,n) is a deterministic function of the PPG key fields, so replay
    // re-uses the recorded pointers safely.
    float* sk_wp = nullptr;
    int* sk_cp = nullptr;
    if (splitk >= 2) {
        // cached ws + counters per (sk,m,n); kernel leaves counters zeroed
        struct WsKey { int sk; int64_t m, n; bool operator<(const WsKey& o) const {
            return sk != o.sk ? sk < o.sk : (m != o.m ? m < o.m : n < o.n); } };
        static std::map<WsKey, std::pair<at::Tensor, at::Tensor>> ws_cache;
        const WsKey wkey{splitk, (int64_t)m, (int64_t)n};
        auto it = ws_cache.find(wkey);
        if (it == ws_cache.end()) {
            auto opts = torch::TensorOptions().dtype(torch::kFloat).device(aq.device());
            at::Tensor wsT = at::empty({(int64_t)splitk * (int64_t)m * n}, opts);
            at::Tensor cntT = at::empty({(int64_t)tiles128}, opts.dtype(torch::kInt));
            cntT.zero_();   // once per allocation; kernel self-resets thereafter
            it = ws_cache.emplace(wkey, std::make_pair(wsT, cntT)).first;
        }
        sk_wp = it->second.first.data_ptr<float>();
        sk_cp = it->second.second.data_ptr<int>();
    }
    // Classic-ladder launcher. The lambda parameter deliberately SHADOWS the
    // outer `stream` so every launch macro below targets the passed stream
    // (capture stream under PPG-all, real stream otherwise) without touching
    // the macro bodies.
    auto classic_launch = [&](cudaStream_t stream) {
    if (use_pdbmc || use_pdbcg2) {
        // V38/V39: the pdbmc/pdbcg2 chain already launched above (graph or eager).
    } else if (use_cg2pp) {
        // V37: the cg2pp chain already launched above (graph or eager).
    } else if (pp_chain_here) {
        // V35: the pp two-kernel chain already launched (eager/captured) or
        // graph-replayed inside the pp block above; the ladder's pp sites
        // below are unreachable in production (kept for the cg2-force combo).
    } else if (use_pp224) {
        // J26: the pp224 chain already launched above (graph or eager).
    } else if (use_cg2) {
        const CUtensorMap* tmBh = get_map(bq.data_ptr(), k, n, bq.stride(0), 128, cg2_bn / 2, 0, 128);
        const CUtensorMap* tmSBh = get_map(sb.data_ptr(), k / 32, n, sb.stride(0) * 4, 4, cg2_bn, 2, 0);
        if (cg2_bn == 256) {
            if (is_f32) { float* dp = reinterpret_cast<float*>(d.data_ptr());
                MXFP8_LAUNCH_CG2(5, 256, float, dp) }
            else { __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(d.data_ptr());
                MXFP8_LAUNCH_CG2(5, 256, __nv_bfloat16, dp) }
        } else {
            if (is_f32) { float* dp = reinterpret_cast<float*>(d.data_ptr());
                MXFP8_LAUNCH_CG2(5, 240, float, dp) }
            else { __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(d.data_ptr());
                MXFP8_LAUNCH_CG2(5, 240, __nv_bfloat16, dp) }
        }
    } else if (splitk >= 2) {
        dim3 gridS(tiles128, splitk);
        static bool attr_done_s = false;
        if (!attr_done_s) {
            cudaFuncSetAttribute(gemm_splitk<float>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SM128_3K);
            cudaFuncSetAttribute(gemm_splitk<__nv_bfloat16>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SM128_3K);
            attr_done_s = true;
        }
        if (is_f32) {
            float* dp = reinterpret_cast<float*>(d.data_ptr());
            gemm_splitk<float><<<gridS, block, SM128_3K, stream>>>(
                m, n, k, (float)alpha, group, *tmA_use, *tmB, *tmSA_use, *tmSB,
                dp, d.stride(0), reinterpret_cast<const float*>(cptr), c_stride0,
                has_c ? 1 : 0, kchunk, sk_wp, sk_cp, splitk);
        } else {
            __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(d.data_ptr());
            gemm_splitk<__nv_bfloat16><<<gridS, block, SM128_3K, stream>>>(
                m, n, k, (float)alpha, group, *tmA_use, *tmB, *tmSA_use, *tmSB,
                dp, d.stride(0), reinterpret_cast<const __nv_bfloat16*>(cptr), c_stride0,
                has_c ? 1 : 0, kchunk, sk_wp, sk_cp, splitk);
        }
    } else if (is_f32) {
        float* dp = reinterpret_cast<float*>(d.data_ptr());
        if (use_pdb256)  { if (pp_pdb) { MXFP8_LAUNCH_PDB_PP(4, 256, float, dp, tiles256, false, stream) }
                           else       { MXFP8_LAUNCH_PDB(4, 256, float, dp, tmB, tmSB, tiles256, true) } }
        else if (use_pdb240)  { MXFP8_LAUNCH_PDB(4, 240, float, dp, tmB240, tmSB240, tiles240, true) }
        else if (use_pdb224) { MXFP8_LAUNCH_PDB(4, 224, float, dp, tmB224, tmSB224, tiles224, true) }
        else if (use224c2) { MXFP8_LAUNCH(2, 224, float, dp, true, false, 128) }
        else if (use256) { if (abox_on) { MXFP8_LAUNCH(4, 256, float, dp, true, true, 128) }
                           else         { MXFP8_LAUNCH(4, 256, float, dp, true, false, 128) } }
        else if (use64)  { if (abox_on) { MXFP8_LAUNCH(8, 64, float, dp, true, true, 128) }
                           else         { MXFP8_LAUNCH(8, 64, float, dp, true, false, 128) } }
        else if (use2sm) { MXFP8_LAUNCH(3, 128, float, dp, true, false, 128) }
        else if (fb6)    { if (abox_on) { MXFP8_LAUNCH(6, 128, float, dp, true, true, 128) }
                           else         { MXFP8_LAUNCH(6, 128, float, dp, true, false, 128) } }
        else             { if (abox_on) { MXFP8_LAUNCH(3, 128, float, dp, true, true, 256) }
                           else         { MXFP8_LAUNCH(3, 128, float, dp, true, false, 256) } }
    } else {
        __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(d.data_ptr());
        if (use_pdb256)  { if (pp_pdb) { MXFP8_LAUNCH_PDB_PP(4, 256, __nv_bfloat16, dp, tiles256, false, stream) }
                           else       { MXFP8_LAUNCH_PDB(4, 256, __nv_bfloat16, dp, tmB, tmSB, tiles256, true) } }
        else if (use_pdb240)  { MXFP8_LAUNCH_PDB(4, 240, __nv_bfloat16, dp, tmB240, tmSB240, tiles240, true) }
        else if (use_pdb224) { MXFP8_LAUNCH_PDB(4, 224, __nv_bfloat16, dp, tmB224, tmSB224, tiles224, true) }
        else if (use224c2) { MXFP8_LAUNCH(2, 224, __nv_bfloat16, dp, true, false, 128) }
        else if (use256) { if (abox_on) { MXFP8_LAUNCH(4, 256, __nv_bfloat16, dp, true, true, 128) }
                           else         { MXFP8_LAUNCH(4, 256, __nv_bfloat16, dp, true, false, 128) } }
        else if (use64)  { if (abox_on) { MXFP8_LAUNCH(8, 64, __nv_bfloat16, dp, true, true, 128) }
                           else         { MXFP8_LAUNCH(8, 64, __nv_bfloat16, dp, true, false, 128) } }
        else if (use2sm) { MXFP8_LAUNCH(3, 128, __nv_bfloat16, dp, true, false, 128) }
        else if (fb6)    { if (abox_on) { MXFP8_LAUNCH(6, 128, __nv_bfloat16, dp, true, true, 128) }
                           else         { MXFP8_LAUNCH(6, 128, __nv_bfloat16, dp, true, false, 128) } }
        else             { if (abox_on) { MXFP8_LAUNCH(3, 128, __nv_bfloat16, dp, true, true, 256) }
                           else         { MXFP8_LAUNCH(3, 128, __nv_bfloat16, dp, true, false, 256) } }
    }
    };  // end classic_launch

    // V39.2 PPG-all (MXFP8_PPGALL=1, dev): extend the V35 C++-side CUDA-graph
    // cache to the CLASSIC single-kernel arms (splitk / 224c2 / np256 / BN64 /
    // 2sm / fb6 / fallback / classic pdb). Small-case forwards are launch-tax
    // dominated (launch+event floor ~5.6us prod / ~2.7-2.9us per launch; J17
    // measured -1.0..-2.9us GPU-side for the 2-kernel pp chains). Same key
    // space (sap/sbp = 0 here, disjoint from the pp-chain keys); arm choice is
    // a deterministic function of key fields; all allocations/first-use map
    // builds happen on call 1 (eager) before any capture. Default OFF:
    // production behavior byte-identical.
    static const int ppgall_on = [] {
        const char* e = getenv("MXFP8_PPGALL");
        return (e && atoi(e) > 0) ? 1 : 0;
    }();
    const bool pre_launched = use_pdbmc || use_pdbcg2 || use_cg2pp ||
                              pp_chain_here || use_pp224;
    if (ppgall_on && !pre_launched) {
        int ppgall_mode = mxfp8::PPG_EAGER;
        mxfp8::PPGEnt* ppgall_ent = nullptr;
        cudaGraphExec_t ppgall_exec = nullptr;
        cudaStream_t ppgall_st = stream;
        mxfp8::PPGArgs ga{(uintptr_t)aq.data_ptr(), (uintptr_t)sa.data_ptr(),
                          (uintptr_t)bq.data_ptr(), (uintptr_t)sb.data_ptr(),
                          (uintptr_t)d.data_ptr(), has_c ? (uintptr_t)cptr : 0ull,
                          0ull, 0ull,
                          (int64_t)m, (int64_t)n, (int64_t)k,
                          (int64_t)sa.stride(0), (int64_t)sb.stride(0),
                          (int64_t)aq.stride(0), (int64_t)bq.stride(0),
                          (int64_t)d.stride(0), c_stride0,
                          is_f32 ? 1 : 0, has_c ? 1 : 0,
                          (int32_t)d.device().index(), mxfp8::f2bits((float)alpha)};
        ppgall_mode = mxfp8::ppg_begin(ga, stream, &ppgall_ent, &ppgall_exec, &ppgall_st);
        if (ppgall_mode != mxfp8::PPG_REPLAY) {
            classic_launch(ppgall_st);
            if (ppgall_mode == mxfp8::PPG_CAPTURE) {
                ppgall_exec = mxfp8::ppg_end(ppgall_ent);
                if (ppgall_exec) { cudaGraphLaunch(ppgall_exec, stream); ++mxfp8::g_ppg_launches; }
                else { classic_launch(stream); }   // capture ran nothing: eager result
            }
        } else {
            cudaGraphLaunch(ppgall_exec, stream);
            ++mxfp8::g_ppg_launches;
        }
    } else {
        classic_launch(stream);
    }
#undef MXFP8_LAUNCH
#undef MXFP8_LAUNCH_PDB
#undef MXFP8_LAUNCH_PDB_PP
#undef MXFP8_LAUNCH_PDB_PP224
#undef MXFP8_LAUNCH_CG2PP
#undef MXFP8_LAUNCH_PDBMC
#undef MXFP8_LAUNCH_PDBCG2
#undef SM_PDB_OF
#undef MXFP8_LAUNCH_CG2
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "mxfp8: launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp8_gemm", &mxfp8_gemm, "MXFP8 block-scaled GEMM (L20D tcgen05 UMMA + TMA)",
          py::arg("aq"), py::arg("sa"), py::arg("bq"), py::arg("sb"), py::arg("d"),
          py::arg("c"), py::arg("alpha"));
    m.def("mxfp8_graph_stats",
          []() { return std::vector<int64_t>{mxfp8::g_ppg_caps, mxfp8::g_ppg_launches,
                                             mxfp8::g_ppg_fails, (int64_t)mxfp8::g_ppg.size()}; },
          "V35 pp-graph telemetry: caps, launches, fails, live keys");
}
