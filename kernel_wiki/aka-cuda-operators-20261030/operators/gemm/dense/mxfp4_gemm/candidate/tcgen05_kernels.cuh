// Stage 2 (v4): native tcgen05 MXFP4 block-scaled UMMA GEMM (sm_100+/sm_103).
// Consumes packed e2m1 payload + ue8m0 scales directly (no bf16 dequant pass).
// Mechanism conventions informed by DeepGEMM (Apache-2.0) sm100 fp8_fp4 gemm,
// flashinfer cutedsl block-scaled idesc layout, and alphamoe nvfp4 sm100 PTX;
// verified bit-exact on the target device (probe2 P0, probe4, check_native):
//   kind::mxf4.block_scale.block32, idesc: a/b_format=1 (e2m1),
//   scale_format bit23=1 (ue8m0), n_dim=(N>>3)<<17, m_dim=(M>>4)<<24,
//   sf_id in {0,2} at bits 29/4; smem desc: SWIZZLE_128B(2), SBO=1024,
//   bit46=1; SF words in UTCCP order word w = row (w%4)*32 + w/4.
// All code here written independently for this candidate.
//
// v4: templated BN in {128, 224} (224 -> 1.75x flops per SM-ingress byte;
// TMEM 2*224+8+16 = 472 <= 512 keeps two overlapped accumulator stages),
// runtime n_dim for partial n tiles, wave-estimate dispatch.
// Warp roles: 5 = TMA/bulk producer; 4 = UTCCP+UMMA issuer; 0-3 = epilogue.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <map>
#include <tuple>
#include <unordered_map>

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------
DEVI uint64_t make_desc(uint32_t smem_addr_bytes, uint32_t sbo_bytes, uint32_t layout) {
  return (static_cast<uint64_t>((smem_addr_bytes & 0x3FFFFu) >> 4)) |
         (static_cast<uint64_t>((sbo_bytes >> 4) & 0x3FFF) << 32) | (1ULL << 46) |
         (static_cast<uint64_t>(layout) << 61);
}
DEVI void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
DEVI void tc_fence_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
DEVI void tc_alloc(uint32_t slot_smem_addr, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
               "r"(slot_smem_addr), "r"(ncols));
}
DEVI void tc_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
DEVI void tc_dealloc(uint32_t base, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(base), "r"(ncols));
}
DEVI void utccp_4x32dp128b(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(sdesc));
}
DEVI void umma_mxf4(uint32_t tacc, uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                    uint32_t tsfa, uint32_t tsfb, uint32_t accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
      :: "r"(tacc), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(tsfa), "r"(tsfb), "r"(accum));
}
DEVI void umma_commit(uint32_t mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::
               "r"(mbar_addr));
}
DEVI void mbar_init(uint32_t bar_addr, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar_addr), "r"(cnt));
}
DEVI void mbar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEVI void mbar_arrive(uint32_t bar_addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar_addr));
}
DEVI void mbar_arrive_expect_tx(uint32_t bar_addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar_addr), "r"(bytes));
}
DEVI void mbar_wait(uint32_t bar_addr, uint32_t phase) {
  asm volatile(
      "{\n\t.reg .pred P;\n\tLAB_WAIT%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
      "@!P bra LAB_WAIT%=;\n\t}" :: "r"(bar_addr), "r"(phase));
}
// step-8 cluster-pair resync primitives (probe11/12 evidence: co-scheduled
// 2-way B-tile sharing cuts stage ingress 362->239ns; us-scale inter-item
// skew kills it; a per-item DSMEM pair barrier fully recovers it).
DEVI void mbar_arrive_cluster(uint32_t bar_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::
               "r"(bar_addr) : "memory");
}
DEVI uint32_t cluster_rank() {
  uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
DEVI uint32_t mapa_mate(uint32_t addr) {  // cluster-2: mate = rank ^ 1
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                           : "=r"(r) : "r"(addr), "r"(cluster_rank() ^ 1u));
  return r;
}
DEVI void cluster_arrive_wait() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// ---------------------------------------------------------------------------
// step-12: cta_group::2 (2-CTA cluster pair, UMMA M256) primitives.
// Mechanism conventions per the arm-a join exchange (.join-exchange/
// 20261023/a/candidate, bit-exact validated there): pair UMMA M256 with each
// CTA's TMEM holding its own 128 rows x full N; A M-split per CTA; B data
// N-split (each CTA loads BN/2 columns); SFA per-CTA own half; SFB FULL-N
// mirrored per CTA; TMA .cta_group::2 credits the pair LEADER's barrier;
// commit multicast (mask 3) releases both CTAs' barriers - no per-stage
// software remote arrives. Independent implementation on this tree.
// ---------------------------------------------------------------------------
DEVI uint32_t cluster_leader_bar(uint32_t local_bar_addr) {
  return local_bar_addr & 0xFEFFFFFFu;  // clear rank bit -> pair leader window
}
DEVI void tma_load_2d_cg2(const void* tmap, uint32_t leader_bar, uint32_t dst_smem,
                          int x, int y) {
  // Data lands in THIS CTA's smem; complete_tx credits the leader's barrier.
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
      " [%0], [%1, {%2, %3}], [%4];"
      :: "r"(dst_smem), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(x), "r"(y),
         "r"(leader_bar)
      : "memory");
}
DEVI void tc_alloc_cg2(uint32_t slot_smem_addr, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::
               "r"(slot_smem_addr), "r"(ncols));
}
DEVI void tc_relinquish_cg2() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}
DEVI void tc_dealloc_cg2(uint32_t base, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(base),
               "r"(ncols));
}
DEVI void utccp_cg2(uint32_t taddr, uint64_t sdesc) {
  // Mirrored: reads EACH CTA's smem at the same offset, writes EACH CTA's
  // TMEM columns (leader-issued).
  asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(taddr),
               "l"(sdesc));
}
DEVI void umma_mxf4_cg2(uint32_t tacc, uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                        uint32_t tsfa, uint32_t tsfb, uint32_t accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n\t}"
      :: "r"(tacc), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(tsfa), "r"(tsfb),
         "r"(accum));
}
DEVI void umma_commit_cg2_mc(uint32_t mbar_addr, uint16_t mask) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
      " [%0], %1;" :: "r"(mbar_addr), "h"(mask));
}
DEVI void tma_load_2d(const void* tmap, uint32_t dst_smem, int x, int y, uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"
      :: "r"(dst_smem), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(x), "r"(y), "r"(mbar)
      : "memory");
}
// join3-E1(rev2): advisory L2 line prefetch (classic form). ptxas sm_103a
// rejects the tensormap form cp.async.bulk.prefetch.L2.global.tile ("Illegal
// modifier '.tile'", job dv_d9eb2b26ad87), so E1 prefetches the raw fp4 row
// buffers per 128B line: fire-and-forget, no destination, no barrier, no
// completion tracking. Addresses are 128B box corners, clamped in-bounds by
// the caller.
DEVI void prefetch_l2_line(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
DEVI void cp_async_bulk(uint32_t dst_smem, const void* src_global, uint32_t bytes, uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(dst_smem), "l"(src_global), "r"(bytes), "r"(mbar) : "memory");
}
DEVI void tmem_load_x32(uint32_t* d, uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]),
        "=r"(d[7]), "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]),
        "=r"(d[14]), "=r"(d[15]), "=r"(d[16]), "=r"(d[17]), "=r"(d[18]), "=r"(d[19]), "=r"(d[20]),
        "=r"(d[21]), "=r"(d[22]), "=r"(d[23]), "=r"(d[24]), "=r"(d[25]), "=r"(d[26]), "=r"(d[27]),
        "=r"(d[28]), "=r"(d[29]), "=r"(d[30]), "=r"(d[31])
      : "r"(taddr));
}
DEVI void tmem_load_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
DEVI bool elect_one() {
  uint32_t pred = 0;
  asm volatile("{\n\t.reg .pred P;\n\telect.sync _|P, 0xFFFFFFFF;\n\tselp.b32 %0, 1, 0, P;\n\t}"
               : "=r"(pred));
  return pred != 0;
}
DEVI void st_shared_v4(uint32_t* p, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
               "r"(smem_u32(p)), "r"(v0), "r"(v1), "r"(v2), "r"(v3));
}
// 256-bit global chunk helpers (32B/lane): floors P3 (dv_9da845c73411)
// measured the epilogue's scattered 16B/lane STG.128 pattern at 2.12 TB/s
// vs 5.62 TB/s for STG.256 on a 201MB D-shaped buffer (2.65x). One chunk =
// 64B per lane-row (32 bf16 cols), 32B-aligned by construction (N%128==0,
// n0/cc multiples of 32 cols; guarded/alloc bases keep >=32B alignment).
DEVI void ld_c_chunk(const __nv_bfloat16* p, uint32_t* v) {
  asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]),
                 "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
               : "l"(p));
  asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]),
                 "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
               : "l"(p + 16));
}
DEVI void st_d_chunk(__nv_bfloat16* p, const uint32_t* v) {
  asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::
               "l"(p), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]),
               "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]) : "memory");
  asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::
               "l"(p + 16), "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]),
               "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]) : "memory");
}

// ---------------------------------------------------------------------------
// Prep kernel (fully coalesced):
//  blocks [0, num_m*4): zero-flag quarters of A tiles (32 rows each);
//    flags[tile*4+q] = 1 iff that quarter's payload codes are all {+0,-0}.
//  remaining blocks: SF pair chunks: A tiles (128 rows) then B tiles
//  (SFBN rows). Per (tile, kb-pair): sector-aligned 32B loads, pack ue8m0
//  words, smem-stage, linear writeout in UTCCP order (word w of a 128-row
//  chunk = row (w%4)*32 + w/4).
// ---------------------------------------------------------------------------
DEVI uint32_t pack_sf4(float4 sc) {
  const uint32_t e0 = (__float_as_uint(sc.x) >> 23) & 0xFFu;
  const uint32_t e1 = (__float_as_uint(sc.y) >> 23) & 0xFFu;
  const uint32_t e2 = (__float_as_uint(sc.z) >> 23) & 0xFFu;
  const uint32_t e3 = (__float_as_uint(sc.w) >> 23) & 0xFFu;
  return e0 | (e1 << 8) | (e2 << 16) | (e3 << 24);
}

// Phase-2 step-3: prep-v2 variant (archived design, 1024 threads, CPB=4,
// single block sync). Selected by P.prep_v2 (provisional rule num_m>=4);
// per-shape winner decided by same-job A/B.
template <int SFBN>
__global__ void __launch_bounds__(1024) prep_kernel2(
    const uint8_t* __restrict__ A, int row_bytes,
    const float* __restrict__ SA, int rows_a,
    const float* __restrict__ SB, int rows_b,
    int gpr, int nk, int num_m, int num_m_chunk, int num_n, int BROWSPAN,
    uint8_t* __restrict__ flags,
    uint32_t* __restrict__ out_a, uint32_t* __restrict__ out_b) {
  constexpr int CPB = 4;                       // SF chunks per block
  constexpr int WCH = (SFBN == 256) ? 512 : 256;  // words per kb side... A:512
  // step-19 (PDL prep->gemm, pdlnsw dv_7c1dcaad2fd4): early programmatic-
  // completion trigger. Lets the dependent gemm grid (launched with PSS on
  // the cl1 plain path) start its prep-independent prologue while prep CTAs
  // still occupy SMs. No-op for followers without PSS (cg2/fallback paths).
  if (threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int flag_blocks = num_m * 4;
  if (b < flag_blocks) {
    const int mt = b >> 2;
    const int q = b & 3;
    const int r0 = mt * 128 + q * 32;
    const int rows = min(32, rows_a - r0);
    if (rows <= 0) {
      if (tid == 0) flags[b] = 1;
      return;
    }
    const uint32_t chunks = static_cast<uint32_t>(rows) * (row_bytes / 16);
    const uint8_t* base = A + static_cast<size_t>(r0) * row_bytes;
    __shared__ uint32_t red[32];
    // step-6a sampled pre-pass (see prep_kernel): nonzero sample -> flags=0.
    uint32_t acc = 0;
    for (uint32_t i = tid; i < chunks; i += 1024 * 16) {
      const uint4 v = *reinterpret_cast<const uint4*>(base + static_cast<size_t>(i) * 16);
      acc |= v.x | v.y | v.z | v.w;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc |= __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    uint32_t all = 0;
#pragma unroll
    for (int w = 0; w < 32; ++w) all |= red[w];
    if ((all & 0x77777777u) != 0) {
      if (tid == 0) flags[b] = 0;
      return;
    }
    __syncthreads();  // red[] reuse guard
    acc = 0;
    for (uint32_t i = tid; i < chunks; i += 1024) {
      const uint4 v = *reinterpret_cast<const uint4*>(base + static_cast<size_t>(i) * 16);
      acc |= v.x | v.y | v.z | v.w;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc |= __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
      all = 0;
#pragma unroll
      for (int w = 0; w < 32; ++w) all |= red[w];
      flags[b] = (all & 0x77777777u) == 0 ? 1 : 0;
    }
    return;
  }
  const int pairs = (nk + 1) >> 1;
  const int cnt_a = num_m_chunk * pairs;
  const int total_chunks = cnt_a + num_n * pairs;
  const int c0 = (b - flag_blocks) * CPB;
  const int sub = tid >> 8;          // chunk slot in block
  const int t = tid & 255;           // thread within chunk
  const int c = c0 + sub;
  __shared__ uint32_t sw[CPB * (SFBN == 256 ? 1024 : 512)];
  uint32_t* mysw = sw + sub * (SFBN == 256 ? 1024 : 512);

  const bool active = (c < total_chunks);
  const bool isA = active && c < cnt_a;
  int tile = 0, kp = 0;
  if (active) {
    const int idx = isA ? c : c - cnt_a;
    tile = idx / pairs;
    kp = idx - tile * pairs;
  }
  const float* S = isA ? SA : SB;
  // ---- phase 1: load 32B/row sector-aligned, pack ue8m0 words into smem ----
  if (active) {
    if (isA || SFBN == 128) {
      const int rows = isA ? rows_a : rows_b;
      const int r = t & 127;
      const int kbh = t >> 7;
      int gr = tile * 128 + r;
      gr = min(gr, rows - 1);
      const float* srow = S + static_cast<size_t>(gr) * gpr;
      const int kb = kp * 2 + kbh;
      if (kb < nk) {
        const float4 lo = *reinterpret_cast<const float4*>(srow + kb * 8);
        const float4 hi = *reinterpret_cast<const float4*>(srow + kb * 8 + 4);
        mysw[kbh * 256 + r] = pack_sf4(lo);
        mysw[kbh * 256 + 128 + r] = pack_sf4(hi);
      }
    } else {
      const int r128 = t & 127;
      const int c2 = t >> 7;
      const int r = c2 * 128 + r128;
      int gr = tile * BROWSPAN + r;
      gr = min(gr, rows_b - 1);
      const float* srow = S + static_cast<size_t>(gr) * gpr;
#pragma unroll
      for (int kbh = 0; kbh < 2; ++kbh) {
        const int kb = kp * 2 + kbh;
        if (kb < nk) {
          const float4 lo = *reinterpret_cast<const float4*>(srow + kb * 8);
          const float4 hi = *reinterpret_cast<const float4*>(srow + kb * 8 + 4);
          mysw[kbh * 512 + r] = pack_sf4(lo);
          mysw[kbh * 512 + 256 + r] = pack_sf4(hi);
        }
      }
    }
  }
  __syncthreads();
  // ---- phase 2: transposed linear writeout (UTCCP word order) ----
  if (active) {
    if (isA || SFBN == 128) {
      uint32_t* out = isA ? (out_a + static_cast<size_t>(tile) * nk * 256)
                          : (out_b + static_cast<size_t>(tile) * nk * 256);
#pragma unroll
      for (int i0 = 0; i0 < 2; ++i0) {
        const int i = t + i0 * 256;
        const int kbh2 = i >> 8;
        const int kb2 = kp * 2 + kbh2;
        if (kb2 >= nk) break;
        const int j = i & 255;
        const int sb = j >> 7;
        const int w = j & 127;
        const int rr = (w & 3) * 32 + (w >> 2);
        out[kb2 * 256 + sb * 128 + w] = mysw[kbh2 * 256 + sb * 128 + rr];
      }
    } else {
      uint32_t* out = out_b + static_cast<size_t>(tile) * nk * (2 * SFBN);
#pragma unroll
      for (int i0 = 0; i0 < 4; ++i0) {
        const int i = t + i0 * 256;
        const int kbh2 = i >> 9;
        const int kb2 = kp * 2 + kbh2;
        if (kb2 >= nk) break;
        const int j = i & 511;
        const int sb = j >> 8;
        const int rr = j & 255;
        const int cc2 = rr >> 7;
        const int r127 = rr & 127;
        const int w = 4 * (r127 & 31) + (r127 >> 5);
        out[kb2 * 512 + sb * 256 + cc2 * 128 + w] = mysw[kbh2 * 512 + sb * 256 + rr];
      }
    }
  }
}
template <int SFBN>
__global__ void prep_kernel(const uint8_t* __restrict__ A, int row_bytes,
                            const float* __restrict__ SA, int rows_a,
                            const float* __restrict__ SB, int rows_b,
                            int gpr, int nk, int num_m, int BN,
                            uint8_t* __restrict__ flags,
                            uint32_t* __restrict__ out_a, uint32_t* __restrict__ out_b) {
  constexpr int WKB = 2 * SFBN;      // words per kb for B side
  // step-19: early PDL trigger (see prep_kernel2).
  if (threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int flag_blocks = num_m * 4;
  if (b < flag_blocks) {
    const int mt = b >> 2;
    const int q = b & 3;
    const int r0 = mt * 128 + q * 32;
    const int rows = min(32, rows_a - r0);
    if (rows <= 0) {
      if (tid == 0) flags[b] = 1;
      return;
    }
    const uint32_t chunks = static_cast<uint32_t>(rows) * (row_bytes / 16);
    const uint8_t* base = A + static_cast<size_t>(r0) * row_bytes;
    __shared__ uint32_t red[8];
    // step-6a: sampled pre-pass (1/16 of words). Any nonzero sample proves the
    // block nonzero -> flags=0 without the full scan (flags scan dominates
    // prep on big-A shapes: case5 ~50MB). flags=1 still requires the FULL
    // scan below, so zf semantics are unchanged; a sampled-miss nonzero block
    // simply pays the old cost.
    uint32_t acc = 0;
    for (uint32_t i = tid; i < chunks; i += 256 * 16) {
      const uint4 v = *reinterpret_cast<const uint4*>(base + static_cast<size_t>(i) * 16);
      acc |= v.x | v.y | v.z | v.w;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc |= __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    uint32_t all = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) all |= red[w];
    if ((all & 0x77777777u) != 0) {
      if (tid == 0) flags[b] = 0;
      return;
    }
    __syncthreads();  // red[] reuse guard
    acc = 0;
    for (uint32_t i = tid; i < chunks; i += 256) {
      const uint4 v = *reinterpret_cast<const uint4*>(base + static_cast<size_t>(i) * 16);
      acc |= v.x | v.y | v.z | v.w;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc |= __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
      all = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) all |= red[w];
      flags[b] = (all & 0x77777777u) == 0 ? 1 : 0;
    }
    return;
  }
  const int c = b - flag_blocks;
  const int pairs = (nk + 1) >> 1;
  const int cnt_a = num_m * pairs;
  const bool isA = c < cnt_a;
  const int idx = isA ? c : c - cnt_a;
  const int tile = idx / pairs;
  const int kp = idx - tile * pairs;

  __shared__ uint32_t sw[SFBN == 256 ? 1024 : 512];
  if (isA) {
    const float* S = SA;
    uint32_t* out = out_a + static_cast<size_t>(tile) * nk * 256;
    const int r = tid & 127;
    const int kbh = tid >> 7;
    int gr = tile * 128 + r;
    gr = min(gr, rows_a - 1);
    const float* srow = S + static_cast<size_t>(gr) * gpr;
    const int kb = kp * 2 + kbh;
    if (kb < nk) {
      const float4 lo = *reinterpret_cast<const float4*>(srow + kb * 8);
      const float4 hi = *reinterpret_cast<const float4*>(srow + kb * 8 + 4);
      sw[kbh * 256 + r] = pack_sf4(lo);
      sw[kbh * 256 + 128 + r] = pack_sf4(hi);
    }
    __syncthreads();
#pragma unroll
    for (int i0 = 0; i0 < 2; ++i0) {
      const int i = tid + i0 * 256;
      const int kbh2 = i >> 8;
      const int kb2 = kp * 2 + kbh2;
      if (kb2 >= nk) break;
      const int j = i & 255;
      const int sb = j >> 7;
      const int w = j & 127;
      const int rr = (w & 3) * 32 + (w >> 2);
      out[kb2 * 256 + sb * 128 + w] = sw[kbh2 * 256 + sb * 128 + rr];
    }
    return;
  }
  // B side: SFBN rows per tile chunk
  {
    const float* S = SB;
    uint32_t* out = out_b + static_cast<size_t>(tile) * nk * WKB;
    if (SFBN == 128) {
      const int r = tid & 127;
      const int kbh = tid >> 7;
      int gr = tile * BN + r;
      gr = min(gr, rows_b - 1);
      const float* srow = S + static_cast<size_t>(gr) * gpr;
      const int kb = kp * 2 + kbh;
      if (kb < nk) {
        const float4 lo = *reinterpret_cast<const float4*>(srow + kb * 8);
        const float4 hi = *reinterpret_cast<const float4*>(srow + kb * 8 + 4);
        sw[kbh * 256 + r] = pack_sf4(lo);
        sw[kbh * 256 + 128 + r] = pack_sf4(hi);
      }
      __syncthreads();
#pragma unroll
      for (int i0 = 0; i0 < 2; ++i0) {
        const int i = tid + i0 * 256;
        const int kbh2 = i >> 8;
        const int kb2 = kp * 2 + kbh2;
        if (kb2 >= nk) break;
        const int j = i & 255;
        const int sb = j >> 7;
        const int w = j & 127;
        const int rr = (w & 3) * 32 + (w >> 2);
        out[kb2 * 256 + sb * 128 + w] = sw[kbh2 * 256 + sb * 128 + rr];
      }
    } else {
      const int r128 = tid & 127;
      const int c2 = tid >> 7;
      const int r = c2 * 128 + r128;
      int gr = tile * BN + r;
      gr = min(gr, rows_b - 1);
      const float* srow = S + static_cast<size_t>(gr) * gpr;
#pragma unroll
      for (int kbh = 0; kbh < 2; ++kbh) {
        const int kb = kp * 2 + kbh;
        if (kb < nk) {
          const float4 lo = *reinterpret_cast<const float4*>(srow + kb * 8);
          const float4 hi = *reinterpret_cast<const float4*>(srow + kb * 8 + 4);
          sw[kbh * 512 + r] = pack_sf4(lo);
          sw[kbh * 512 + 256 + r] = pack_sf4(hi);
        }
      }
      __syncthreads();
#pragma unroll
      for (int i0 = 0; i0 < 4; ++i0) {
        const int i = tid + i0 * 256;
        const int kbh2 = i >> 9;
        const int kb2 = kp * 2 + kbh2;
        if (kb2 >= nk) break;
        const int j = i & 511;
        const int sb = j >> 8;
        const int rr = j & 255;
        const int cc2 = rr >> 7;
        const int r127 = rr & 127;
        const int w = 4 * (r127 & 31) + (r127 >> 5);
        out[kb2 * 512 + sb * 256 + cc2 * 128 + w] = sw[kbh2 * 512 + sb * 256 + rr];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// prep-v3: coalesced SF pack (arm after NCU case5 prep = 22MB reads @
// 18.37us base-clock = 1.2TB/s vs 5.4TB/s read floor; v7 reads 32B-per-row
// strided (16 rows/instr) and stores scattered 4B words).
//  blocks [0, num_m*4): flag scan, IDENTICAL to prep_kernel (same sampled
//    pre-pass + full scan + semantics).
//  rest: quad blocks (tile, qa): 4 kb groups x ROWS (128 A / SFBN B) fp32
//    scales read as 4 float4/thread with lane-pair row sharing (each 128B
//    L1 line fully consumed by 4 consecutive j-instructions), packed to
//    ue8m0 words staged in smem in the EXACT v7 out word order, then
//    linear uint4 writeout (16B/thread contiguous).
// Word-order equivalence: A/SFBN128: kb*256 + half*128 + w, w = 4*(r&31)
//   + (r>>5) (inverse of v7's rr = (w&3)*32 + (w>>2)); SFBN256: kb*512 +
//   half*256 + (r>>7)*128 + w(r&127). half = (col>>2)&1 (lo/hi 4-scale
//   groups of the 8-scale kb). Bit-identical output buffers vs v7.
// ---------------------------------------------------------------------------
template <int SFBN>
__global__ void __launch_bounds__(256) prep_kernel3(
    const uint8_t* __restrict__ A, int row_bytes,
    const float* __restrict__ SA, int rows_a,
    const float* __restrict__ SB, int rows_b,
    int gpr, int nk, int num_m, int num_n, int nqa, int BN,
    uint8_t* __restrict__ flags,
    uint32_t* __restrict__ out_a, uint32_t* __restrict__ out_b) {
  if (threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int flag_blocks = num_m * 4;
  if (b < flag_blocks) {
    const int mt = b >> 2;
    const int q = b & 3;
    const int r0 = mt * 128 + q * 32;
    const int rows = min(32, rows_a - r0);
    if (rows <= 0) {
      if (tid == 0) flags[b] = 1;
      return;
    }
    const uint32_t chunks = static_cast<uint32_t>(rows) * (row_bytes / 16);
    const uint8_t* base = A + static_cast<size_t>(r0) * row_bytes;
    __shared__ uint32_t red[8];
    uint32_t acc = 0;
    for (uint32_t i = tid; i < chunks; i += 256 * 16) {
      const uint4 v = *reinterpret_cast<const uint4*>(base + static_cast<size_t>(i) * 16);
      acc |= v.x | v.y | v.z | v.w;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc |= __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    uint32_t all = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) all |= red[w];
    if ((all & 0x77777777u) != 0) {
      if (tid == 0) flags[b] = 0;
      return;
    }
    __syncthreads();  // red[] reuse guard
    acc = 0;
    for (uint32_t i = tid; i < chunks; i += 256) {
      const uint4 v = *reinterpret_cast<const uint4*>(base + static_cast<size_t>(i) * 16);
      acc |= v.x | v.y | v.z | v.w;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) acc |= __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
      all = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) all |= red[w];
      flags[b] = (all & 0x77777777u) == 0 ? 1 : 0;
    }
    return;
  }
  const int c = b - flag_blocks;
  const int cnt_a = num_m * nqa;
  const bool isA = c < cnt_a;
  const int idx = isA ? c : c - cnt_a;
  const int tile = idx / nqa;
  const int qa = idx - tile * nqa;
  const int kv = min(4, nk - qa * 4);            // valid kb in this quad
  constexpr int HALVES = (SFBN == 256) ? 2 : 1;  // B-side 128-row halves
  __shared__ uint32_t stg[2048];                 // <= 4 kb x 512 words

  const float* S = isA ? SA : SB;
  const int rows = isA ? rows_a : rows_b;
  const int row0 = isA ? (tile * 128) : (tile * BN);
  // load+pack: per 128-row half h, thread t owns float4 x = t*4+j of the
  // half's 4096-float block: e = 4x floats -> r = e>>5, col = e&31. Lanes
  // 2k/2k+1 share row k (cols 4j / 16+4j); j=0..3 completes both 128B lines.
#pragma unroll
  for (int h = 0; h < (isA ? 1 : HALVES); ++h) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int x = tid * 4 + j;
      const int e = x * 4;
      const int r = h * 128 + (e >> 5);
      const int col = e & 31;
      const int kb = col >> 3;
      if (qa * 4 + kb < nk) {
        int gr = row0 + r;
        gr = min(gr, rows - 1);
        const float4 v = *reinterpret_cast<const float4*>(
            S + static_cast<size_t>(gr) * gpr + qa * 32 + col);
        const uint32_t word = pack_sf4(v);
        const int halfsc = (col >> 2) & 1;
        const int rr = r & 127;
        const int w = 4 * (rr & 31) + (rr >> 5);
        if (isA || SFBN == 128) {
          stg[kb * 256 + halfsc * 128 + w] = word;
        } else {
          stg[kb * 512 + halfsc * 256 + (r >> 7) * 128 + w] = word;
        }
      }
    }
  }
  __syncthreads();
  const int words_per_kb = (isA || SFBN == 128) ? 256 : 512;
  if (isA) {
    uint4* dst = reinterpret_cast<uint4*>(
        out_a + (static_cast<size_t>(tile) * nk + qa * 4) * 256);
    const int n4 = kv * 64;
    for (int i = tid; i < n4; i += 256) dst[i] = reinterpret_cast<const uint4*>(stg)[i];
  } else {
    uint4* dst = reinterpret_cast<uint4*>(
        out_b + (static_cast<size_t>(tile) * nk + qa * 4) * words_per_kb);
    const int n4 = kv * (words_per_kb / 4);
    for (int i = tid; i < n4; i += 256) dst[i] = reinterpret_cast<const uint4*>(stg)[i];
  }
}

// ---------------------------------------------------------------------------
// Persistent block-scaled MXFP4 GEMM: D = alpha*A@B^T + C
//   BM=128, BN=128|224, BK=256 fp4 (128B rows), UMMA M128 x N(BN) x K64,
//   4 per stage. Stage image: A 16KB | B BN*128B | SFA 1KB | SFB SFBN*8B.
// ---------------------------------------------------------------------------
template <int BN, int SFBN, int MX_STAGES, int ACC_STRIDE, int OVL, int FUSED, int CLU, int SFS>
__global__ void __launch_bounds__(256, 1) mxf4_gemm_kernel(
    const __grid_constant__ CUtensorMap tmap_a,
    const __grid_constant__ CUtensorMap tmap_b,
    const uint32_t* __restrict__ SFAW, const uint32_t* __restrict__ SFBW,
    __nv_bfloat16* __restrict__ D, const __nv_bfloat16* __restrict__ C,
    const uint8_t* __restrict__ aflags, float alpha, int has_c,
    int M, int N, int K, int num_m, int num_n, int total_tiles,
    const float* __restrict__ SAg, const float* __restrict__ SBg, int gpr,
    const __grid_constant__ CUtensorMap tmap_at, int a_tail_bytes,
    const __grid_constant__ CUtensorMap tmap_sfap,
    const __grid_constant__ CUtensorMap tmap_sfbp,
    const __grid_constant__ CUtensorMap tmap_bh, int rskn_R,
    const uint8_t* __restrict__ Aq, const uint8_t* __restrict__ Bq) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // step-12: cg2 pair kernel (CLU==2) halves the per-CTA B tile (N-split);
  // A and SF regions unchanged. SFP_OFF = packed-SF region offset in stage.
  constexpr bool kC2 = (CLU == 2);
  constexpr int CL_EFF = kC2 ? 2 : 1;
  constexpr int B_BYTES = (BN / CL_EFF) * 128;
  // fd (FUSED==2): per-stage raw fp32 SF staging (TMA-loaded via tmap_sfap/
  // tsfbp into SFRAW_OFF), packed into the existing UTCCP word region by
  // warps 6/7. Removes the prep kernel and its global packed-SF round trip
  // for single-wave shapes; DeepGEMM-style SF transposer mechanism, adapted
  // to the frozen fp32-scale ABI (pack = exponent extract, same word map).
  constexpr int SFRAW_BYTES = (FUSED == 2) ? (4096 + SFBN * 32) : 0;
  // sf-split ring (SFS>0): packed-SF words live in their own SFS-slot ring
  // after the payload stages, freeing smem for a deeper payload ring
  // (BN=224: S 4->5; latency-chain model says period ~ chain/S, A/B#6 case7
  // -10% at 4->6). Slot reuse safety: producer waits esf[z] (armed by a
  // second umma_commit per stage) before re-issuing SF bulk loads.
  constexpr int PAY_BYTES = 16384 + B_BYTES;
  constexpr int SFSLOT = 1024 + SFBN * 8;
  constexpr int SF_OFF = MX_STAGES * PAY_BYTES;
  constexpr int MX_STAGE_BYTES = (SFS > 0)
                                     ? PAY_BYTES
                                     : (PAY_BYTES + 1024 + SFBN * 8 + SFRAW_BYTES);
  constexpr int SFP_OFF = 16384 + B_BYTES;
  constexpr int SFRAW_OFF = SFP_OFF + 1024 + SFBN * 8;
  constexpr int NCC = BN / 32;                 // epilogue 32-col chunks
  constexpr int SFB_SUB_COLS = SFBN / 32;      // tmem cols per SFB sub-block

  extern __shared__ char smem[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int nk = K / 256;

  // join3-F1: fused-shape SF-slice L2 prefetch. Fused (small-grid) shapes
  // stage SF words from fp32 SAg/SBg INSIDE the ring (warps 6/7 below): the
  // stage's full barrier waits on 3 arrivals, so each stage's cold fp32
  // scale loads (~500-700ns DRAM latency, issued at most one stage ahead)
  // serialize into the chain - the floor map's ~4us case0/1 residual
  // ("fused per-stage fp32-SF latency", rank-stable in the peer fusedab
  // fit; scales with nk). The fused gate (tiles<=148) means single-item
  // workers, so the CTA's whole SF slice is known at entry: SAg rows
  // [m0, min(m0+128,M)) and SBg rows [n0, min(n0+SFBN,N)) are contiguous
  // row-major fp32 spans - sweep their 128B lines from warp 6/7 at kernel
  // entry (before barrier init, ~1.5-2us before the first SF load issues)
  // so the ring's SF loads hit L2 (~200ns). Advisory lines only: no
  // barriers, no state, bitexact; the last line's <=127B overhang stays
  // inside the 512B-rounded allocator block. Non-fused instantiations
  // compile this to nothing (FUSED is constexpr).
  if (FUSED && (warp == 6 || warp == 7)) {
    const int ftile = (int)blockIdx.x;
    if (ftile < total_tiles) {
      int fmt, fnt;
      {
        const int per_group = 8 * num_n;
        const int gid = ftile / per_group;
        const int first_m = gid * 8;
        const int gsz = min(num_m - first_m, 8);
        const int rem = ftile - gid * per_group;
        fnt = rem / gsz;
        fmt = first_m + rem % gsz;
      }
      const size_t gpr4 = (size_t)gpr * 4;  // scale-row bytes (fp32)
      if (warp == 6) {
        const int rows = min(128, M - fmt * 128);
        const uint8_t* base =
            reinterpret_cast<const uint8_t*>(SAg) + (size_t)fmt * 128 * gpr4;
        const int lines = (int)((rows * gpr4 + 127) / 128);
        for (int i = lane; i < lines; i += 32)
          prefetch_l2_line(base + (size_t)i * 128);
      } else {
        const int cols = min(SFBN, N - fnt * BN);
        const uint8_t* base =
            reinterpret_cast<const uint8_t*>(SBg) + (size_t)fnt * BN * gpr4;
        const int lines = (int)((cols * gpr4 + 127) / 128);
        for (int i = lane; i < lines; i += 32)
          prefetch_l2_line(base + (size_t)i * 128);
      }
    }
  }

  char* const bar_base = (SFS > 0) ? (smem + SF_OFF + SFS * SFSLOT)
                                   : (smem + MX_STAGES * MX_STAGE_BYTES);
  const uint32_t full_b0 = smem_u32(bar_base);              // full[S]
  const uint32_t empty_b0 = full_b0 + MX_STAGES * 8;         // empty[S]
  const uint32_t tmemfull_b0 = empty_b0 + MX_STAGES * 8;     // tmem_full[2]
  const uint32_t tmemempty_b0 = tmemfull_b0 + 16;            // tmem_empty[2]
  const uint32_t overlap_b0 = tmemempty_b0 + 16;             // overlap[2]
  const uint32_t sfraw_b0 = overlap_b0 + 16;                 // fd: sfraw[S]
  const uint32_t esf_b0 = sfraw_b0 + ((FUSED == 2) ? MX_STAGES * 8 : 0);
  // tbase sits past all barrier arrays (2*S*8 + 48 + fd sfraw + esf);
  // launch tail reserves >= that for S<=8, SFS<=2.
  uint32_t* tbase = reinterpret_cast<uint32_t*>(
      bar_base + 2 * MX_STAGES * 8 + 48 + ((FUSED == 2) ? MX_STAGES * 8 : 0) +
      ((SFS > 0) ? SFS * 8 : 0));
  volatile uint32_t* zflag = tbase + 2;
  // step-8: pair-resync barrier at tbase+4 (bytes 160..167 of the 208B
  // barrier region; tbase 8B + zflag 8B end at 160) - no smem resize.
  const uint32_t ps_b0 = smem_u32(tbase + 4);

  if (tid == 0) {
    for (int s = 0; s < MX_STAGES; ++s) {
      mbar_init(full_b0 + s * 8, FUSED ? 3 : 1);
      mbar_init(empty_b0 + s * 8, 1);
      if (FUSED == 2) mbar_init(sfraw_b0 + s * 8, 1);
    }
    if (SFS > 0)
      for (int z = 0; z < SFS; ++z) mbar_init(esf_b0 + z * 8, 1);
    mbar_init(tmemfull_b0, 1);
    mbar_init(tmemfull_b0 + 8, 1);
    mbar_init(tmemempty_b0, kC2 ? 8 : 4);      // cg2: + 4 remote peer arrives
    mbar_init(tmemempty_b0 + 8, kC2 ? 8 : 4);
    mbar_init(overlap_b0, kC2 ? 8 : 4);
    mbar_init(overlap_b0 + 8, kC2 ? 8 : 4);
    if (CLU == 1) mbar_init(ps_b0, 1);
    mbar_init_fence();
  }
  __syncthreads();
  if (CLU) cluster_arrive_wait();  // barrier init visible cluster-wide
  if (warp == 4) {
    if (kC2) { tc_alloc_cg2(smem_u32(tbase), 512); tc_relinquish_cg2(); }
    else { tc_alloc(smem_u32(tbase), 512); tc_relinquish(); }
  }
  __syncthreads();
  const uint32_t tmem0 = *tbase;
  if (kC2) { cluster_arrive_wait(); tc_fence_after_sync(); }
  const uint32_t tsfa = tmem0 + 2 * ACC_STRIDE + OVL;
  const uint32_t tsfb = tsfa + 8;
  if (tid == 0) {
    for (int s = 0; s < MX_STAGES; ++s) mbar_arrive(empty_b0 + s * 8);
    if (SFS > 0)
      for (int z = 0; z < SFS; ++z) mbar_arrive(esf_b0 + z * 8);
    for (int j = 0; j < 2; ++j)
      for (int r = 0; r < (kC2 ? 8 : 4); ++r) mbar_arrive(tmemempty_b0 + j * 8);
  }
  __syncthreads();
  // join3-E1: every scalar down to the item loop is prep-independent (kernel
  // params + blockIdx/gridDim only), so the block is hoisted above the
  // step-19 dep-sync: the prefetch below decodes this CTA's first tile while
  // the prep predecessor is still running.
  const uint32_t idesc_base =
      (1u << 7) | (1u << 10) | (1u << 23) | ((kC2 ? 16u : 8u) << 24);
  const int grid = gridDim.x;

  int grun = 0;  // running stage index; zf tiles consume NO stages (step-7)
  const int tile0 = kC2 ? ((int)blockIdx.x >> 1) : (int)blockIdx.x;
  const int tile_step = kC2 ? (grid >> 1) : grid;
  const uint32_t crank = kC2 ? cluster_rank() : 0u;
  const bool leader = !kC2 || crank == 0u;
  const int num_m128 = (M + 127) / 128;  // 128-row tiles (kernel num_m = pairs under cg2)
  // step-14 rskn: last-round N-split. When ceil-quantization wastes >=8% of
  // pair slots (case9: 256 pair-tiles/74 pairs -> 4 waves vs 3.46 ideal),
  // the final round's R tiles are split into 128-col halves: pair p<R runs
  // the low half (halfm=1) of tile (I-1)*P+p, idle pair P-R+r runs the high
  // half (halfm=2) of tile (I-1)*P+r. Makespan (I-1)+~0.62 items. Bitexact:
  // per-element k-chain identical (UMMA n_dim 16 vs 32 only narrows N).
  // step-18: rskn generalized to cl1 (CLU==0) - case6-class BN=128 shapes
  // carry 35% last-round wave-quantization loss (T=192/148 -> I=2, R=44,
  // idle=104). cl1 half items: BN/2-col B box (tmap_bh), full SF mirror,
  // high half offsets the SF-B tmem column by BN/64 (= mirror rows BN/2).
  const int rR = (CLU != 1 && rskn_R > 0) ? rskn_R : 0;
  const int rI = (total_tiles + tile_step - 1) / tile_step;
  // join3-E1(rev2): pre-dep-sync advisory L2 prefetch of this CTA's
  // first-item A/B boxes (first 2 k stages). Under v2 all inputs are
  // DRAM-cold at the timed start; prep touches A (flag scan) and SA/SB but
  // never B, so first-stage B ingress is a guaranteed cold DRAM round trip
  // sitting behind cudaGridDependencySynchronize(). Step-19/20 PSS parks
  // this grid during prep, so issuing fire-and-forget L2 line prefetches
  // before the dep-sync overlaps that ingress (~10-14MB grid-wide) with the
  // latency-bound prep window. rev1 (cp.async.bulk.prefetch...tile via the
  // tensormaps) was rejected by ptxas sm_103a (dv_d9eb2b26ad87); rev2 walks
  // the raw fp4 row buffers: A/B are (rows, K/2) and the producer's kb=0/1
  // boxes are exactly bytes [0,256) of each box row, so 2x128B lines per
  // row, striped over the producer warp's lanes. No barriers, no state, no
  // data-path change (bitexact); addresses clamped in-bounds (arows/bcols),
  // and the nk>=16 gate guarantees 256B valid bytes per row. The first item
  // is always a FULL tile (rskn halfm splits only the last round, rI>=2),
  // so no halfm decode is needed.
  // Gate !FUSED && nk>=16: fused shapes (case0/1, nk=10 anyway) have no prep
  // predecessor to overlap with, and every nk=10 suite shape is excluded
  // because case8 is zero-A (zf skips ALL A ingress, so prefetch there is
  // pure DRAM waste) while case2/4/8 preps are the most bandwidth-tight.
  // zf is unknowable here: aflags is a prep output, and reading it pre-
  // dep-sync would race - hence the shape-level gate. Off-suite zero-A
  // shapes with nk>=16 pay a bounded <=2-stage waste per CTA; no correctness
  // impact.
  if (!FUSED && nk >= 16 && tile0 < total_tiles && warp == 5) {
    int pmt, pnt;
    {
      const int per_group = 8 * num_n;
      const int gid = tile0 / per_group;
      const int first_m = gid * 8;
      const int gsz = min(num_m - first_m, 8);
      const int rem = tile0 - gid * per_group;
      pnt = rem / gsz;
      pmt = first_m + rem % gsz;
    }
    const int pm0 = pmt * (128 * CL_EFF) + (int)crank * 128;
    const int pn0 = pnt * BN + (kC2 ? (int)crank * (BN / 2) : 0);
    const size_t rs = (size_t)(K / 2);  // fp4-packed row stride (bytes)
    const int arows = min(128, M - pm0);
    const int bcols = min(kC2 ? (BN / 2) : BN, N - pn0);
    const uint8_t* Ab = Aq + (size_t)pm0 * rs;
    const uint8_t* Bb = Bq + (size_t)pn0 * rs;
    for (int i = lane; i < arows * 2; i += 32)
      prefetch_l2_line(Ab + (size_t)(i >> 1) * rs + (size_t)(i & 1) * 128);
    for (int i = lane; i < bcols * 2; i += 32)
      prefetch_l2_line(Bb + (size_t)(i >> 1) * rs + (size_t)(i & 1) * 128);
  }
  // step-19: PDL dependent-side wait. The prologue above (barrier init,
  // TMEM alloc, ring seeding) reads no prep output; first prep-output reads
  // (aflags, SFAW/SFBW) are in the item loop below. No-op when launched
  // without PSS (cg2 CLU path, fused shapes' memcpy predecessor).
  cudaGridDependencySynchronize();

  for (int tloc = 0;; ++tloc) {
    int t; int halfm = 0;
    if (rR == 0) {
      t = tile0 + tloc * tile_step;
      if (t >= total_tiles) break;
    } else if (tloc < rI - 1) {
      t = tile0 + tloc * tile_step;
    } else if (tloc == rI - 1) {
      if (tile0 < rR) {
        t = (rI - 1) * tile_step + tile0; halfm = 1;
      } else if (tile0 >= tile_step - rR) {
        t = (rI - 1) * tile_step + (tile0 - (tile_step - rR)); halfm = 2;
      } else break;
    } else break;
    // grouped rasterization (GROUP_M=8) for L2 locality
    int mt, nt;
    {
      const int per_group = 8 * num_n;
      const int gid = t / per_group;
      const int first_m = gid * 8;
      const int gsz = min(num_m - first_m, 8);
      const int rem = t - gid * per_group;
      nt = rem / gsz;
      mt = first_m + rem % gsz;
    }
    const int m0 = mt * (128 * CL_EFF) + (int)crank * 128;
    const int n0 = nt * BN + (halfm == 2 ? BN / 2 : 0);
    const int n_eff = (halfm != 0) ? (BN / 2) : min(BN, N - nt * BN);
    const int bcols_it = (halfm != 0) ? (kC2 ? BN / 4 : BN / 2)
                                      : (BN / CL_EFF);
    const int bbytes_it = bcols_it * 128;
    const int j = tloc & 1;
    const int g0 = grun;
    // step-12: under cg2, mt indexes 256-row PAIR tiles; flags/SFA stay per
    // 128-row half (mt_half). Stage-skip zf needs BOTH halves zero; the
    // epilogue's per-CTA `zero` uses the own-half flag only (a zero own half
    // accumulates exact fp32 zeros through the pair UMMA anyway).
    const int mt_half = kC2 ? (mt * 2 + (int)crank) : mt;
    const uint32_t fword_own =
        *reinterpret_cast<const uint32_t*>(aflags + mt_half * 4);
    const int zf_own = FUSED ? 0 : ((fword_own == 0x01010101u) ? 1 : 0);
    int zf = zf_own;
    if (kC2) {
      const uint32_t fw0 = *reinterpret_cast<const uint32_t*>(aflags + mt * 8);
      const uint32_t fw1 = *reinterpret_cast<const uint32_t*>(aflags + mt * 8 + 4);
      zf = (fw0 == 0x01010101u && fw1 == 0x01010101u) ? 1 : 0;
    }
    const uint32_t* sfaw = SFAW + static_cast<size_t>(mt_half) * nk * 256;
    const uint32_t* sfbw = SFBW + static_cast<size_t>(nt) * nk * (2 * SFBN);
    // cg2 pair tx total: rank1's A may be an SMA tail box (v1 gate keeps
    // num_m128 even -> only rank1 can hold the tail half).
    const int abytes1 = kC2 ? (((mt * 2 + 1) == num_m128 - 1) ? a_tail_bytes : 16384)
                            : 16384;

    if (warp == 5) {
      if (CLU == 1) {
        // step-8: item-boundary pair resync. Cluster-mates (blockIdx 2p, 2p+1)
        // process rasterization-adjacent tiles (t, t+1) which share nt under
        // the num_m%8==0 gate -> their B/SFB TMA reads hit the same L2 lines.
        // Natural inter-item skew (us-scale) defeats L2 broadcast coalescing;
        // arrive-then-wait on the mate's ps barrier re-aligns the pair each
        // item (probe12: recovers 239ns from 352ns skewed). Unconditional
        // (also for zf items) so the mate never starves; arrive-before-wait
        // on both ranks = deadlock-free; item counts are equal (total even).
        if (elect_one()) mbar_arrive_cluster(mapa_mate(ps_b0));
        __syncwarp();
        mbar_wait(ps_b0, tloc & 1);
      }
      // step-7: zf tiles consume no stages; the MMA warp commits tmemfull
      // directly, so the producer must not touch full/empty for them.
      // step-9 (SMA): the last mt tile transfers only its valid A rows
      // (tail tmap box = M-(num_m-1)*128 rows); smem rows beyond the box
      // keep stale stage data - harmless: e2m1 x ue8m0 is always finite,
      // per-row UMMA accumulation never mixes rows, and the epilogue never
      // stores rows >= M. num_m==1 shapes (case0/1/6, d0/d4/d5/d9-class)
      // cut per-stage A ingress from 16KB to M*128B.
      const uint32_t abytes = (mt_half == num_m128 - 1) ? (uint32_t)a_tail_bytes : 16384u;
      for (int kb = zf ? nk : 0; kb < nk; ++kb) {
        const int g = g0 + kb;
        const int s = g - (g / MX_STAGES) * MX_STAGES;
        const uint32_t ph = (g / MX_STAGES) & 1;
        mbar_wait(empty_b0 + s * 8, ph);
        if (elect_one()) {
          if (kC2) {
            // pair stage: both CTAs load (A own half, B own N half, SFA own
            // half, SFB full-N mirror); ALL tx credits the leader's barrier,
            // which the leader registers once for the pair.
            const uint32_t lfull = cluster_leader_bar(full_b0 + s * 8);
            if (leader)
              mbar_arrive_expect_tx(
                  lfull,
                  (uint32_t)(2 * (MX_STAGE_BYTES - B_BYTES + bbytes_it) - 16384 +
                             abytes1));
            char* st = smem + s * MX_STAGE_BYTES;
            tma_load_2d_cg2((mt_half == num_m128 - 1) ? &tmap_at : &tmap_a,
                            lfull, smem_u32(st), kb * 128, m0);
            // cg2 B N-split is N_DIM-RELATIVE: CTA0's smem B feeds effective
            // cols [0, n_eff/2), CTA1's feeds [n_eff/2, n_eff). Full tiles:
            // n_eff/2 == BN/2 (validated BN=256); rskn halfm: (BN/2)/2 ==
            // BN/4 (validated step-14); BN=224 tail (n_eff=128): rank offset
            // MUST be 64 — the old BN/2=112 offset corrupted cols 64..127 of
            // every tail tile (build_check 20261028T112435Z bad_frac 7.6e-3,
            // A/B#7 dv_60e73427d58f case2 correctness fail — preserved).
            tma_load_2d_cg2((halfm != 0) ? &tmap_bh : &tmap_b, lfull,
                            smem_u32(st + 16384), kb * 128,
                            n0 + (int)crank * (n_eff / 2));
            tma_load_2d_cg2(&tmap_sfap, lfull, smem_u32(st + SFP_OFF), 0,
                            mt_half * nk + kb);
            tma_load_2d_cg2(&tmap_sfbp, lfull, smem_u32(st + SFP_OFF + 1024), 0,
                            (nt * nk + kb) * (SFBN / 128));
          } else if (zf) {
            mbar_arrive(full_b0 + s * 8);
          } else {
            // sf-split ring: wait for slot z=g%SFS release (second commit of
            // stage g-SFS) before re-issuing its SF bulk loads.
            if (SFS > 0)
              mbar_wait(esf_b0 + (g - (g / SFS) * SFS) * 8, (g / SFS) & 1);
            constexpr int SF_BYTES_NF = 1024 + SFBN * 8;
            mbar_arrive_expect_tx(full_b0 + s * 8,
                                  FUSED ? (abytes + B_BYTES)
                                        : (SF_BYTES_NF + bbytes_it + abytes));
            char* st = smem + s * MX_STAGE_BYTES;
            tma_load_2d((mt_half == num_m128 - 1) ? &tmap_at : &tmap_a, smem_u32(st),
                        kb * 128, m0, full_b0 + s * 8);
            tma_load_2d((halfm != 0) ? &tmap_bh : &tmap_b, smem_u32(st + 16384),
                        kb * 128, n0, full_b0 + s * 8);
            if (FUSED == 2) {
              // fd: raw fp32 SF boxes on their own barrier; warps 6/7 pack
              // them into the UTCCP word region and complete full[s].
              mbar_arrive_expect_tx(sfraw_b0 + s * 8, SFRAW_BYTES);
              tma_load_2d(&tmap_sfap, smem_u32(st + SFRAW_OFF), kb * 8, m0,
                          sfraw_b0 + s * 8);
              tma_load_2d(&tmap_sfbp, smem_u32(st + SFRAW_OFF + 4096), kb * 8, n0,
                          sfraw_b0 + s * 8);
            }
            if (!FUSED) {
              char* sfd = (SFS > 0)
                              ? (smem + SF_OFF + (g - (g / SFS) * SFS) * SFSLOT)
                              : (st + SFP_OFF);
              cp_async_bulk(smem_u32(sfd), sfaw + kb * 256, 1024,
                            full_b0 + s * 8);
              cp_async_bulk(smem_u32(sfd + 1024),
                            sfbw + kb * (2 * SFBN), SFBN * 8, full_b0 + s * 8);
            }
          }
        }
        __syncwarp();
      }
    } else if (warp == 4) {
      const uint32_t idesc_tile = idesc_base | ((n_eff >> 3) << 17);
      if (kC2 && !leader) {
        // step-12: peer MMA warp idle - the pair UMMA (leader-issued) writes
        // this CTA's TMEM; multicast commits release this CTA's barriers.
      } else if (zf) {
        // step-7 fast path: whole tile is zero — no stages, no UMMAs. Signal
        // the epilogue directly (zflag store fenced before the commit; the
        // commit with no outstanding tcgen05 ops arrives promptly).
        if (elect_one()) {
          tc_fence_after_sync();
          mbar_wait(tmemempty_b0 + j * 8, (tloc >> 1) & 1);
          if (OVL > 0 && tloc >= 1) {
            const uint32_t ovl_par = (j == 1) ? ((tloc >> 1) & 1)
                                              : (((tloc >> 1) - 1) & 1);
            mbar_wait(overlap_b0 + (1 - j) * 8, ovl_par);
          }
          zflag[j] = 1u;
          __threadfence_block();
          if (kC2) umma_commit_cg2_mc(tmemfull_b0 + j * 8, 3);
          else     umma_commit(tmemfull_b0 + j * 8);
        }
        __syncwarp();
      } else
      for (int kb = 0; kb < nk; ++kb) {
        const int g = g0 + kb;
        const int s = g - (g / MX_STAGES) * MX_STAGES;
        const uint32_t ph = (g / MX_STAGES) & 1;
        mbar_wait(full_b0 + s * 8, ph);
        if (elect_one()) {
          tc_fence_after_sync();
          if (kb == 0) {
            mbar_wait(tmemempty_b0 + j * 8, (tloc >> 1) & 1);
            if (OVL > 0 && tloc >= 1) {
              const uint32_t ovl_par = (j == 1) ? ((tloc >> 1) & 1)
                                                : (((tloc >> 1) - 1) & 1);
              mbar_wait(overlap_b0 + (1 - j) * 8, ovl_par);
            }
            zflag[j] = static_cast<uint32_t>(zf);
          }
          if (!zf) {
            const int hsel = (kC2 && halfm != 0) ? 1 : 0;
            // step-18: cl1 high-half SF-B column offset: mirror rows BN/2..
            // live at tmem col (BN/2)/32 = BN/64 within each sub-block.
            const uint32_t sfbsel =
                (!kC2 && halfm == 2) ? (uint32_t)(BN / 64) : 0u;
            const uint32_t sf0 = (SFS > 0)
                ? smem_u32(smem + SF_OFF + (g - (g / SFS) * SFS) * SFSLOT)
                : smem_u32(smem + s * MX_STAGE_BYTES + SFP_OFF);
            if (kC2) utccp_cg2(tsfa + 0, make_desc(sf0, 128, 0));
            else     utccp_4x32dp128b(tsfa + 0, make_desc(sf0, 128, 0));
            if (kC2) utccp_cg2(tsfa + 4, make_desc(sf0 + 512, 128, 0));
            else     utccp_4x32dp128b(tsfa + 4, make_desc(sf0 + 512, 128, 0));
#pragma unroll
            for (int sb = 0; sb < 2; ++sb)
#pragma unroll
              for (int c2 = 0; c2 < (hsel ? 1 : SFBN / 128); ++c2) {
                // cg2 rskn half items (step-14): stage only the tile's 128
                // mirror rows (c2sel = high ? 128 : 0 words) into compact
                // 4-col blocks. cl1 half items (step-18): hsel=0 -> stage the
                // FULL mirror; the high half selects rows BN/2.. via sfbsel.
                const uint32_t c2sel = (hsel && halfm == 2 ? 128 : 0) * hsel
                                       + c2 * 128 * (1 - hsel);
                const uint32_t ta = tsfb + sb * (hsel ? 4 : SFB_SUB_COLS) + c2 * 4;
                const uint64_t sd =
                    make_desc(sf0 + 1024 + (sb * SFBN + c2sel) * 4, 128, 0);
                if (kC2) utccp_cg2(ta, sd);
                else     utccp_4x32dp128b(ta, sd);
              }
            const uint64_t adesc = make_desc(smem_u32(smem + s * MX_STAGE_BYTES), 1024, 2);
            const uint64_t bdesc = make_desc(smem_u32(smem + s * MX_STAGE_BYTES + 16384), 1024, 2);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
              const uint32_t sfi = (u & 1) * 2;
              const uint32_t iu = idesc_tile | (sfi << 29) | (sfi << 4);
              const uint32_t sfb_stride = hsel ? 4u : (uint32_t)SFB_SUB_COLS;
              if (kC2)
                umma_mxf4_cg2(tmem0 + j * ACC_STRIDE, adesc + u * 2, bdesc + u * 2,
                              iu, tsfa + (u >> 1) * 4,
                              tsfb + sfbsel + (u >> 1) * sfb_stride,
                              (kb > 0 || u > 0) ? 1u : 0u);
              else
                umma_mxf4(tmem0 + j * ACC_STRIDE, adesc + u * 2, bdesc + u * 2, iu,
                          tsfa + (u >> 1) * 4,
                          tsfb + sfbsel + (u >> 1) * sfb_stride,
                          (kb > 0 || u > 0) ? 1u : 0u);
            }
          }
          if (kC2) {
            umma_commit_cg2_mc(empty_b0 + s * 8, 3);
            if (kb == nk - 1) umma_commit_cg2_mc(tmemfull_b0 + j * 8, 3);
          } else {
            umma_commit(empty_b0 + s * 8);
            // sf-split ring: second commit (same tracked op set) releases the
            // SF slot for the producer's stage g+SFS bulk loads.
            if (SFS > 0) umma_commit(esf_b0 + (g - (g / SFS) * SFS) * 8);
            if (kb == nk - 1) umma_commit(tmemfull_b0 + j * 8);
          }
        }
        __syncwarp();
      }
    } else if (FUSED == 1 && warp == 6) {
      // Phase-2 fused SF prep (small gated grids): warp 6 stages SFA words
      // from fp32 scales into the stage smem SF region — identical word
      // layout to prep_kernel (word w <- row (w&3)*32+(w>>2); sb half +128),
      // so UTCCP/UMMA consumption is unchanged. Loads issue BEFORE the
      // empty[s] wait (L2 latency hidden); release via fence.proxy.async.
      for (int kb = 0; kb < nk; ++kb) {
        const int g = g0 + kb;
        const int s = g - (g / MX_STAGES) * MX_STAGES;
        const uint32_t ph = (g / MX_STAGES) & 1;
        float4 lo[4], hi[4];
#pragma unroll
        for (int ri = 0; ri < 4; ++ri) {
          const int gr = min(m0 + ri * 32 + lane, M - 1);
          const float* srow = SAg + static_cast<size_t>(gr) * gpr + kb * 8;
          lo[ri] = *reinterpret_cast<const float4*>(srow);
          hi[ri] = *reinterpret_cast<const float4*>(srow + 4);
        }
        mbar_wait(empty_b0 + s * 8, ph);
        uint32_t* sfa =
            reinterpret_cast<uint32_t*>(smem + s * MX_STAGE_BYTES + SFP_OFF);
#pragma unroll
        for (int ri = 0; ri < 4; ++ri) {
          const int w = 4 * lane + ri;  // row = ri*32+lane -> w = 4*(row&31)+ri
          sfa[w] = pack_sf4(lo[ri]);
          sfa[128 + w] = pack_sf4(hi[ri]);
        }
        __syncwarp();
        if (elect_one()) {
          fence_async_shared();
          mbar_arrive(full_b0 + s * 8);
        }
      }
    } else if (FUSED == 1 && warp == 7) {
      // warp 7 stages SFB words (SFBN rows; 256 -> two 128-row halves).
      for (int kb = 0; kb < nk; ++kb) {
        const int g = g0 + kb;
        const int s = g - (g / MX_STAGES) * MX_STAGES;
        const uint32_t ph = (g / MX_STAGES) & 1;
        float4 lo[SFBN / 32], hi[SFBN / 32];
#pragma unroll
        for (int ri = 0; ri < SFBN / 32; ++ri) {
          const int gr = min(n0 + ri * 32 + lane, N - 1);
          const float* srow = SBg + static_cast<size_t>(gr) * gpr + kb * 8;
          lo[ri] = *reinterpret_cast<const float4*>(srow);
          hi[ri] = *reinterpret_cast<const float4*>(srow + 4);
        }
        mbar_wait(empty_b0 + s * 8, ph);
        uint32_t* sfb = reinterpret_cast<uint32_t*>(
            smem + s * MX_STAGE_BYTES + SFP_OFF + 1024);
#pragma unroll
        for (int ri = 0; ri < SFBN / 32; ++ri) {
          if (SFBN == 128) {
            const int w = 4 * lane + ri;
            sfb[w] = pack_sf4(lo[ri]);
            sfb[128 + w] = pack_sf4(hi[ri]);
          } else {
            const int w = 4 * lane + (ri & 3);
            const int cc2 = ri >> 2;
            sfb[cc2 * 128 + w] = pack_sf4(lo[ri]);
            sfb[256 + cc2 * 128 + w] = pack_sf4(hi[ri]);
          }
        }
        __syncwarp();
        if (elect_one()) {
          fence_async_shared();
          mbar_arrive(full_b0 + s * 8);
        }
      }
    } else if (FUSED == 2 && warp == 6) {
      // fd: pack SFA words from the TMA-staged fp32 SF box (smem->smem).
      // sfraw[s] implies the producer already passed empty[s] for this
      // stage, so writing the packed region is race-free; visibility to
      // UTCCP (async proxy) via fence.proxy.async before arriving full[s].
      for (int kb = 0; kb < nk; ++kb) {
        const int g = g0 + kb;
        const int s = g - (g / MX_STAGES) * MX_STAGES;
        const uint32_t ph = (g / MX_STAGES) & 1;
        mbar_wait(sfraw_b0 + s * 8, ph);
        const float* src =
            reinterpret_cast<const float*>(smem + s * MX_STAGE_BYTES + SFRAW_OFF);
        uint32_t* sfa =
            reinterpret_cast<uint32_t*>(smem + s * MX_STAGE_BYTES + SFP_OFF);
#pragma unroll
        for (int ri = 0; ri < 4; ++ri) {
          const int r = ri * 32 + lane;
          const float4 lo = *reinterpret_cast<const float4*>(src + r * 8);
          const float4 hi = *reinterpret_cast<const float4*>(src + r * 8 + 4);
          const int w = 4 * lane + ri;  // row = ri*32+lane -> w = 4*(row&31)+ri
          sfa[w] = pack_sf4(lo);
          sfa[128 + w] = pack_sf4(hi);
        }
        __syncwarp();
        if (elect_one()) {
          fence_async_shared();
          mbar_arrive(full_b0 + s * 8);
        }
      }
    } else if (FUSED == 2 && warp == 7) {
      // fd: pack SFB words (SFBN rows; 256 -> two 128-col sub-blocks).
      for (int kb = 0; kb < nk; ++kb) {
        const int g = g0 + kb;
        const int s = g - (g / MX_STAGES) * MX_STAGES;
        const uint32_t ph = (g / MX_STAGES) & 1;
        mbar_wait(sfraw_b0 + s * 8, ph);
        const float* src = reinterpret_cast<const float*>(
            smem + s * MX_STAGE_BYTES + SFRAW_OFF + 4096);
        uint32_t* sfb = reinterpret_cast<uint32_t*>(
            smem + s * MX_STAGE_BYTES + SFP_OFF + 1024);
#pragma unroll
        for (int ri = 0; ri < SFBN / 32; ++ri) {
          const int r = ri * 32 + lane;
          const float4 lo = *reinterpret_cast<const float4*>(src + r * 8);
          const float4 hi = *reinterpret_cast<const float4*>(src + r * 8 + 4);
          if (SFBN == 128) {
            const int w = 4 * lane + ri;
            sfb[w] = pack_sf4(lo);
            sfb[128 + w] = pack_sf4(hi);
          } else {
            const int w = 4 * lane + (ri & 3);
            const int cc2 = ri >> 2;
            sfb[cc2 * 128 + w] = pack_sf4(lo);
            sfb[256 + cc2 * 128 + w] = pack_sf4(hi);
          }
        }
        __syncwarp();
        if (elect_one()) {
          fence_async_shared();
          mbar_arrive(full_b0 + s * 8);
        }
      }
    } else if (warp < 4) {
      // epilogue: TMEM lanes 32*warp..+31 <-> tile rows
      mbar_wait(tmemfull_b0 + j * 8, (tloc >> 1) & 1);
      tc_fence_after_sync();
      const bool zero = kC2 ? (zf_own != 0) : (zflag[j] != 0);
      const int row = m0 + warp * 32 + lane;
      const bool row_ok = row < M;
      const int ncc = n_eff >> 5;
      const bool rev = (OVL > 0) && (j == 0);
      // Software-pipelined C prefetch, depth 2: cc0/cc1 loaded in the
      // prologue; iteration ci issues cc(ci+2) into the freed third buffer
      // and rotates. Covers ~2 chunk-iterations of compute/store against
      // cold-DRAM C latency (1-ahead left ~1000cy/chunk exposed on
      // C-accumulate shapes; epilogue is the per-item critical path on
      // case2/9/d2/d5/d8-class short-nk tiles).
      uint32_t cv[16], cvn[16], cv2[16];
      const size_t crow = static_cast<size_t>(row) * N;
      if (has_c && row_ok && ncc > 0) {
        const int cc0 = rev ? (ncc - 1) : 0;
        ld_c_chunk(C + crow + n0 + cc0 * 32, cv);
        if (ncc > 1) {
          const int cc1 = rev ? (ncc - 2) : 1;
          ld_c_chunk(C + crow + n0 + cc1 * 32, cvn);
        }
      }
      uint32_t vbuf[2][32];
      if (!zero && ncc > 0) {
        const int cc0 = rev ? (ncc - 1) : 0;
        tmem_load_x32(vbuf[0], tmem0 + j * ACC_STRIDE + cc0 * 32);
        tmem_load_wait();
      }
      for (int ci = 0; ci < NCC; ++ci) {
        if (ci >= ncc) break;
        const int cc = rev ? (ncc - 1 - ci) : ci;
        if (has_c && row_ok && ci + 2 < ncc) {
          const int cc2i = rev ? (ncc - 3 - ci) : (ci + 2);
          ld_c_chunk(C + crow + n0 + cc2i * 32, cv2);
        }
        if (!zero && ci + 1 < ncc) {
          const int ccn = rev ? (ncc - 2 - ci) : (ci + 1);
          tmem_load_x32(vbuf[(ci + 1) & 1], tmem0 + j * ACC_STRIDE + ccn * 32);
        }
        if (OVL > 0 && ci == 0 && elect_one()) {
          if (kC2 && !leader)
            mbar_arrive_cluster(cluster_leader_bar(overlap_b0 + j * 8));
          else
            mbar_arrive(overlap_b0 + j * 8);
        }
        uint32_t* v = vbuf[ci & 1];
        if (row_ok) {
          const int col = n0 + cc * 32;
          __nv_bfloat16* dp = D + crow + col;
          uint32_t packed[16];
          if (has_c) {
#pragma unroll
            for (int q4 = 0; q4 < 4; ++q4) {
              const uint32_t cw[4] = {cv[q4 * 4], cv[q4 * 4 + 1],
                                      cv[q4 * 4 + 2], cv[q4 * 4 + 3]};
#pragma unroll
              for (int h = 0; h < 4; ++h) {
                const int q = q4 * 4 + h;
                const __nv_bfloat162 cb = *reinterpret_cast<const __nv_bfloat162*>(&cw[h]);
                const float2 cf = __bfloat1622float2(cb);
                const float x = (zero ? 0.f : __uint_as_float(v[2 * q])) * alpha + cf.x;
                const float y = (zero ? 0.f : __uint_as_float(v[2 * q + 1])) * alpha + cf.y;
                const __nv_bfloat162 pv = __float22bfloat162_rn(make_float2(x, y));
                packed[q] = *reinterpret_cast<const uint32_t*>(&pv);
              }
            }
          } else {
#pragma unroll
            for (int q = 0; q < 16; ++q) {
              const float x = (zero ? 0.f : __uint_as_float(v[2 * q])) * alpha;
              const float y = (zero ? 0.f : __uint_as_float(v[2 * q + 1])) * alpha;
              const __nv_bfloat162 pv = __float22bfloat162_rn(make_float2(x, y));
              packed[q] = *reinterpret_cast<const uint32_t*>(&pv);
            }
          }
          st_d_chunk(dp, packed);
#pragma unroll
          for (int q = 0; q < 16; ++q) { cv[q] = cvn[q]; cvn[q] = cv2[q]; }
        }
        if (!zero && ci + 1 < ncc) tmem_load_wait();
      }
      if (elect_one()) {
        if (kC2 && !leader)
          mbar_arrive_cluster(cluster_leader_bar(tmemempty_b0 + j * 8));
        else
          mbar_arrive(tmemempty_b0 + j * 8);
      }
    }
    grun += zf ? 0 : nk;
  }
  __syncthreads();
  if (kC2) cluster_arrive_wait();  // both CTAs done before the pair dealloc
  if (warp == 4) {
    if (kC2) tc_dealloc_cg2(tmem0, 512);
    else     tc_dealloc(tmem0, 512);
  }
#else
  (void)tmap_a; (void)tmap_b; (void)SFAW; (void)SFBW; (void)D; (void)C; (void)aflags;
  (void)alpha; (void)has_c; (void)M; (void)N; (void)K; (void)num_m; (void)num_n;
  (void)total_tiles; (void)SAg; (void)SBg; (void)gpr; (void)tmap_at; (void)a_tail_bytes;
  (void)tmap_sfap; (void)tmap_sfbp; (void)tmap_bh; (void)rskn_R;
#endif
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------
namespace mxfp4_host {

inline CUtensorMap build_tmap(const void* ptr, uint64_t rows, uint64_t row_bytes, uint32_t box_rows) {
  CUtensorMap tm;
  cuuint64_t gdim[2] = {row_bytes, rows};
  cuuint64_t gstride[1] = {row_bytes};
  cuuint32_t box[2] = {128, box_rows};
  cuuint32_t estr[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), gdim, gstride,
      box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return tm;
}

inline CUtensorMap get_tmap(const void* ptr, uint64_t rows, uint64_t row_bytes, uint32_t box_rows) {
  using TKey = std::tuple<const void*, uint64_t, uint64_t, uint64_t>;
  static std::map<TKey, CUtensorMap> cache;
  TKey key{ptr, rows, row_bytes, box_rows};
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap tm = build_tmap(ptr, rows, row_bytes, box_rows);
  cache.emplace(key, tm);
  return tm;
}

// step-12: UINT32 tensor maps over the packed SF word buffers (cg2 loads SF
// via TMA so complete_tx credits the pair leader's barrier; cp.async.bulk has
// no cta_group::2 credit path). SWIZZLE_NONE, box = {cols, box_rows}.
inline CUtensorMap build_tmap_u32(const void* ptr, uint64_t rows, uint64_t cols,
                                  uint32_t box_rows) {
  CUtensorMap tm;
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstride[1] = {cols * 4};
  cuuint32_t box[2] = {(uint32_t)cols, box_rows};
  cuuint32_t estr[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, const_cast<void*>(ptr), gdim, gstride,
      box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(u32) failed: ",
              static_cast<int>(r));
  return tm;
}

inline CUtensorMap get_tmap_u32(const void* ptr, uint64_t rows, uint64_t cols,
                                uint32_t box_rows) {
  using TKey = std::tuple<const void*, uint64_t, uint64_t, uint64_t>;
  static std::map<TKey, CUtensorMap> cache;
  TKey key{ptr, rows, cols, box_rows};
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap tm = build_tmap_u32(ptr, rows, cols, box_rows);
  cache.emplace(key, tm);
  return tm;
}

// fd (FUSED==2): fp32-scale maps with an 8-word inner box (one kb group of
// 8 scales per row). SWIZZLE_NONE; row-major fp32 (gpr cols); OOB rows/cols
// zero-fill (junk scale lanes are never stored: rows>=M / cols>=N).
inline CUtensorMap build_tmap_u32_box8(const void* ptr, uint64_t rows, uint64_t cols,
                                       uint32_t box_rows) {
  CUtensorMap tm;
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstride[1] = {cols * 4};
  cuuint32_t box[2] = {8, box_rows};
  cuuint32_t estr[2] = {1, 1};
  const CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, const_cast<void*>(ptr), gdim, gstride,
      box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(u32box8) failed: ",
              static_cast<int>(r));
  return tm;
}

inline CUtensorMap get_tmap_u32_box8(const void* ptr, uint64_t rows, uint64_t cols,
                                     uint32_t box_rows) {
  using TKey = std::tuple<const void*, uint64_t, uint64_t, uint64_t>;
  static std::map<TKey, CUtensorMap> cache;
  TKey key{ptr, rows, cols, box_rows};
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap tm = build_tmap_u32_box8(ptr, rows, cols, box_rows);
  cache.emplace(key, tm);
  return tm;
}

inline void* grow_buffer(void*& buf, int& cap, size_t need) {
  if (need > static_cast<size_t>(cap)) {
    // Old buffer intentionally leaked: captured CUDA graphs may reference it.
    void* nb = nullptr;
    cap = static_cast<int>(need * 2);
    TORCH_CHECK(cudaMalloc(&nb, cap) == cudaSuccess, "prep buffer alloc failed");
    buf = nb;
  }
  return buf;
}

template <int BN, int SFBN, int STAGES, int ACC_STRIDE, int OVL, int FUSED, int CLU, int SFS>
inline void launch_mxf4(const CUtensorMap& tma, const CUtensorMap& tmb,
                        const CUtensorMap& tmat, int a_tail_bytes,
                        const CUtensorMap& tsfap, const CUtensorMap& tsfbp,
                        const CUtensorMap& tmb_h, int rskn_R,
                        const uint32_t* sfaw, const uint32_t* sfbw,
                        __nv_bfloat16* D, const __nv_bfloat16* C,
                        const uint8_t* flags, float alpha, int has_c,
                        int M, int N, int K, int num_m, int num_n,
                        const float* sag, const float* sbg, int gpr,
                        const uint8_t* aq, const uint8_t* bq,
                        cudaStream_t stream) {
  const int total = num_m * num_n;
  static const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = (CLU == 2)
                       ? 2 * (total < sm_count / 2 ? total : sm_count / 2)
                       : (total < sm_count ? total : sm_count);
  constexpr int PAY_ = 16384 + (BN / (CLU == 2 ? 2 : 1)) * 128;
  constexpr int SFL_ = 1024 + SFBN * 8;
  constexpr int smem =
      (SFS > 0)
          ? (STAGES * PAY_ + SFS * SFL_ + 16 * 8 + 4 * 8 + 16 + 32 + SFS * 8)
          : (STAGES * (PAY_ + SFL_ + ((FUSED == 2) ? (4096 + SFBN * 32) : 0)) +
             16 * 8 + 4 * 8 + 16 + 32 + ((FUSED == 2) ? STAGES * 8 : 0));
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(mxf4_gemm_kernel<BN, SFBN, STAGES, ACC_STRIDE, OVL, FUSED, CLU, SFS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_set = true;
  }
  if (CLU) {
    // cluster dim 2: consecutive block pairs co-resident (probe12-validated
    // config: 74 clusters x 2 CTAs, full-smem persistent kernel, grid 148).
    TORCH_CHECK((grid & 1) == 0, "cluster launch needs even grid");
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((unsigned)grid, 1, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    // step-20 (pdln2sw dv_68dc1d967f10): cluster+PSS is a valid attribute
    // combination on this stack and lands the cg2 prep-tail overlap: case5
    // -2.8% (tight drift bracket), case9 -1.0%, nm48cls -1.8%, cl1 controls
    // null, all bitexact. Trigger + grid-dep-sync already unconditional;
    // depsync sits after ALL cluster prologue syncs (protocol never split).
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs; cfg.numAttrs = 2;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg,
                                   mxf4_gemm_kernel<BN, SFBN, STAGES, ACC_STRIDE, OVL, FUSED, CLU, SFS>,
                                   tma, tmb, sfaw, sfbw, D, C, flags, alpha, has_c,
                                   M, N, K, num_m, num_n, total, sag, sbg, gpr,
                                   tmat, a_tail_bytes, tsfap, tsfbp,
                                   tmb_h, rskn_R, aq, bq) == cudaSuccess,
                "mxf4 cluster launch failed");
  } else {
    // step-19: cl1 plain launch carries ProgrammaticStreamSerialization so
    // the gemm prologue overlaps the prep tail (pdlnsw: case8 -8.8%, case6
    // -3.9%, case2 -2.4%, case3 -1.3%, null elsewhere, all bitexact).
    // Graph capture bakes the attribute onto the kernel node (pdlsw proved
    // capture works). Without a prep predecessor (fused shapes) the memcpy
    // predecessor can't trigger early -> plain stream order, harmless.
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((unsigned)grid, 1, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs; cfg.numAttrs = 1;
    TORCH_CHECK(cudaLaunchKernelEx(
                    &cfg,
                    mxf4_gemm_kernel<BN, SFBN, STAGES, ACC_STRIDE, OVL, FUSED, CLU, SFS>,
                    tma, tmb, sfaw, sfbw, D, C, flags, alpha, has_c, M, N, K,
                    num_m, num_n, total, sag, sbg, gpr, tmat, a_tail_bytes,
                    tsfap, tsfbp, tmb_h, rskn_R, aq, bq) == cudaSuccess,
                "mxf4 plain launch failed");
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "mxf4 gemm launch failed");
}

}  // namespace mxfp4_host

struct Mxf4Params {
  CUtensorMap tma, tmb;
  CUtensorMap tmat;          // step-9: tail-tile A map (box = tail_rows)
  int a_tail_bytes;          // tail_rows * 128
  CUtensorMap tsfap, tsfbp;  // step-12: packed-SF u32 maps (cg2; = tma else)
  CUtensorMap tmb_h;         // step-14: half-N B map (box BN/4) for rskn
  int rskn_R;                // last-round tiles to N-split (0 = off)
  int num_m_k;               // tile-M count the kernel decodes (pairs under cg2)
  int cg2;
  const uint8_t* A;
  const uint8_t* B;          // join3-E1: raw fp4 row ptrs for L2 prefetch
  const float* SA;
  const float* SB;
  uint8_t* flags;
  const uint32_t* sfaw;
  const uint32_t* sfbw;
  __nv_bfloat16* D;
  const __nv_bfloat16* C;
  float alpha;
  int has_c, M, N, K, num_m, num_n, nk, BN, SFBN;
  const float* SAg;
  const float* SBg;
  int gpr;
  int fused;
  int prep_v2;
  int prep_v3;   // coalesced SF pack (arm vs v7; see prep_kernel3)
  int sf_split;  // BN=224 cl1: SF 2-slot ring, payload S=5 (see SFS template)
  int clu;
  int cg2st;   // join4-H39 probe (peer step-27): cg2 ring depth (nk<24 -> 4, else 6)
};

inline Mxf4Params mxf4_prepare_p(const void* aq_p, const void* asf_p,
                                 const void* bq_p, const void* bsf_p,
                                 void* d_p, const void* c_p, float alpha,
                                 int M, int N, int K) {
  TORCH_CHECK(K % 256 == 0 && N % 128 == 0 && M >= 1);
  Mxf4Params P;
  const int nk = K / 256;
  const int num_m = (M + 127) / 128;

  // BN choice: makespan = waves * (nk*cadence + per-tile overhead), with
  // constants refitted from on-device A/B data (jobs dv_140239Z, dv_41aa6f):
  //   BN=128 (368, 850) | BN=192 (400, 1100) | BN=224 (389, 1372)
  //   BN=256 (457, 1178), eligible only when nk>=16 && waves256>=4.
  auto waves_of = [&](int bn) {
    return ((long)num_m * ((N + bn - 1) / bn) + 147) / 148;
  };
  int BN = 128;
  {
    long best = waves_of(128) * (nk * 368L + 850);
    const long e192 = waves_of(192) * (nk * 400L + 1100);
    if (e192 < best) { best = e192; BN = 192; }
    if (M > 128) {
      const long e224 = waves_of(224) * (nk * 389L + 1372);
      if (e224 < best) { best = e224; BN = 224; }
      if (nk >= 16 && waves_of(256) >= 4) {
        const long e256 = waves_of(256) * (nk * 457L + 1178);
        if (e256 < best) { BN = 256; }
      }
    }
  }
  const int num_n = (N + BN - 1) / BN;
  const int SFBN = (BN >= 192) ? 256 : 128;

  uint32_t* sfaw_p;
  uint32_t* sfbw_p;
  uint8_t* flags_p;
  {
    static void* flagbuf = nullptr; static int flagcap = 0;
    static void* sfabuf = nullptr; static int sfaacap = 0;
    static void* sfbbuf = nullptr; static int sfbcap = 0;
    flags_p = static_cast<uint8_t*>(mxfp4_host::grow_buffer(flagbuf, flagcap, num_m * 4));
    sfaw_p = static_cast<uint32_t*>(
        mxfp4_host::grow_buffer(sfabuf, sfaacap, (size_t)num_m * nk * 1024));
    sfbw_p = static_cast<uint32_t*>(
        mxfp4_host::grow_buffer(sfbbuf, sfbcap, (size_t)num_n * nk * 2 * SFBN * 4));
  }

  P.A = reinterpret_cast<const uint8_t*>(aq_p);
  P.B = reinterpret_cast<const uint8_t*>(bq_p);
  P.SA = reinterpret_cast<const float*>(asf_p);
  P.SB = reinterpret_cast<const float*>(bsf_p);
  P.flags = flags_p;
  P.sfaw = sfaw_p;
  P.sfbw = sfbw_p;
  P.tma = mxfp4_host::get_tmap(aq_p, (uint64_t)M, (uint64_t)(K / 2), 128);
  // step-9 (SMA): tail-tile A tmap with box = valid rows of the last mt tile.
  // tail_rows==128 hits the same get_tmap cache entry as P.tma (neutral).
  // GATE tail_rows<=32: p2ab5 (dv_cd9669eacf18) measured wins at boxes
  // {1,3,16,17,32} (-2.8..-24.1%) but mid-size box 66 (d5cls, single-wave
  // grid=64) at +3.0% -> mid boxes unproven, excluded (step-7 behavior).
  const int tail_rows = M - (num_m - 1) * 128;   // 1..128
  if (tail_rows <= 32) {
    P.a_tail_bytes = tail_rows * 128;
    P.tmat = mxfp4_host::get_tmap(aq_p, (uint64_t)M, (uint64_t)(K / 2),
                                  (uint32_t)tail_rows);
  } else {
    P.a_tail_bytes = 16384;
    P.tmat = P.tma;
  }
  P.tmb = mxfp4_host::get_tmap(bq_p, (uint64_t)N, (uint64_t)(K / 2), BN);
  P.D = reinterpret_cast<__nv_bfloat16*>(d_p);
  P.C = reinterpret_cast<const __nv_bfloat16*>(c_p);
  P.alpha = alpha;
  P.has_c = c_p ? 1 : 0;
  P.M = M; P.N = N; P.K = K;
  P.num_m = num_m; P.num_n = num_n; P.nk = nk; P.BN = BN; P.SFBN = SFBN;
  // Phase-2 fused SF prep gate (archived fusedab single-ext evidence:
  // case0 -1.7% / case1 -5.6% fused; case3 +37% / edge6 +10% / case6 +3.4%
  // when SF fp32 re-reads exceed the small-shape L2 budget):
  //   fused iff BN<256 && tiles<=148 && tiles*nk*(4+SFBN/32)KB <= 8MB.
  const long tiles_f = static_cast<long>(num_m) * num_n;
  const long sf_kb = tiles_f * nk * (4 + SFBN / 32);
  P.fused = (BN < 256 && tiles_f <= 148 && sf_kb <= 8192) ? 1 : 0;
  // fd gate (FUSED==2, TMA-staged fp32 SF + in-kernel pack warps, no prep
  // kernel, zf off): single-wave BN<=192, nk<=24. A/B dv_9fcadd04fd24 (same
  // job vs certified): case6 -7.53%, case1 -1.97%, case0 -0.88%, case3
  // +7.14% -> nk>24 (k=12288) BN=128-class EXCLUDED: its mainloop is
  // ingress-bound and the +6KB/stage fp32 SF staging (~+23% stage bytes)
  // outweighs the (cheap, partly PDL-overlapped) prep removal. nk<=24 keeps
  // every measured win; case0/1 (legacy LDG-fused) also move to fd.
  // Supersedes LDG-fused: per-stage fp32 LDG latency chain -> async TMA.
  if (BN <= 192 && tiles_f <= 148 && nk <= 24) P.fused = 2;
  // step-8 cluster-pair resync gate: DISABLED after measurement (p2ab4,
  // dv_d8802be6d580): item-boundary resync alone does NOT realize B sharing
  // in production - all CLU cases regressed (case2 +2.4%, case8 +4.4% zf-rank
  // mismatch, case5 +1.3%, case7 +2.4%, case9 +2.2%, d3cls +11.3% short-item
  // barrier overhead; controls flat; all bitexact). probe12's recovery held
  // only under immediate-drain consumers; production's MMA-gated empty[]
  // lets the pair re-skew within an item (S=4 buffering ~1.5-2us of lead).
  // Code retained (failed arm); probe14 decides if per-stage coupling can
  // realize sharing under an UMMA-gated consumer before any re-enable.
  P.clu = 0;
  // step-12 cg2 auto-gate (2-CTA pair UMMA M256, B N-split): mechanism per
  // the arm-a join exchange (.join-exchange/20261023/a), their group-d
  // evidence: gemm -16.3..-17.5% vs cl1 BN256 at nk>=24 multi-wave, tie at
  // nk=16, LOSS at nk=10 and 1-wave shapes. v1 gate: auto-BN256 shapes only,
  // nk>=24 (=> k=12288 within the frozen (n,k) set: case5/case9 class),
  // num_m even (no odd-pair OOB A half), N%256==0 (no n_eff<BN tail: B
  // N-split halves must be full 128-col slabs). case5/case9 fixed + legal
  // domain class m>=~4736 n=4096 k=12288; every other shape untouched.
  // join4-H39 ADOPTED (peer arm-b step-27 mechanism, .join-exchange/
  // 20261025/b; p229 dv_530c5b232dbb + p229b provenance): cg2 gate
  // EXTENDED nk>=24 -> nk>=16 with nk-conditional ring depth (nk<24 ->
  // 4 stages). Peer same-job 4-arm @nk16/case7: cg4 169.95 < cl1-auto
  // 174.05 < cg5 176.13 < cg6 184.22 (+5.85% == step-24's +5.86% exact
  // replication => my u8-era "cg2 nk16 CONFIRMED LOSS (188.01 vs
  // 168.18, dv_f36c2a3f759d)" was a DEPTH-6 artifact; ring depth had
  // never been tested on my side). nk48 keeps depth 6 (peer: cg4
  // +10.79% case5, cg5 +2.98%). MY v2 VERIFICATION (H39 probe
  // dv_950ae275fe6e, knob-gated replication, engine-faithful same-job
  // A/B): case7cls level mean -6.1us (176/178 mix -> 170/172 mix,
  // dmed -7.904; 170.0 level == peer join-validation draw 170.048);
  // flipped-domain reps drep20cls -2.016 / drep24cls -2.048 (true
  // level shifts); controls clean (case5 dmed -0.144 identical-code,
  // case3 0.000, case6 +0.032, case2 -0.016, case0real -0.016 with F1
  // LOW-pin intact); BX=True everywhere (pair retile K-order
  // preserved). E1 (mine, absent from peer p229) fires on the new cg2
  // path (decode is CL_EFF/kC2-aware) and does not cancel the win.
  P.cg2 = (BN == 256 && nk >= 16 && num_m >= 2 &&
           (num_m % 2) == 0 && (N % 256) == 0 && !P.fused) ? 1 : 0;
  // cg224 arm (A/B#7): extend cg2 pair UMMA M256 to BN=224 (case2/d8 class).
  // Per-CTA stage 44KB(cl1)->33.8KB for the same pair flops; regions 2x224
  // cols are non-overlapping (OVL=0, tmemempty-only boundary like cl1-224);
  // n_eff tail tiles ride the existing runtime-n_dim + OOB-zero-fill rank
  // split; SFBN=256 mirror/tmem/UTCCP layout identical to BN=256 cg2.
  // Legal-domain note: N%448 never holds for n in {4096,6656,8192,12288,
  // 24576} -> rskn halfm stays unreachable at BN=224.
  // cg224 arm REJECTED (A/B#8 dv_3039d4329867, same job vs 98eb246): case2
  // +5.04% (43.02 vs 40.96) — at nk=10 the cg2 pair overhead (leader-barrier
  // TMA credits, multicast commits, cluster resync per short item) beats the
  // stage-ingress reduction; case8 +12.5% (zf-pair overhead or draw). Gate
  // reverted; the n_dim-relative tail fix (n_eff/2) and SFBN-general tsfbp
  // are kept as hardening. Instantiations kept as dead arms.
  // if (BN == 224 && num_m >= 2 && (num_m % 2) == 0 && !P.fused) P.cg2 = 1;
  // cg128 arm rev2 (S=8 deep ring): A/B#8 showed cg128-S6 == cl1-S6 EXACTLY
  // on case3 (28.672 both) despite 26.6 vs 34.8KB/stage ingress -> the
  // mainloop is latency-chain-bound (period ~ chain/S), not byte-bound
  // (corroborated by A/B#6 cg2st 4->6 = case7 -10%). S=8 (213KB smem) cuts
  // the predicted period ~25% for case3-class. Gate: BN=128 even num_m,
  // non-fused; rskn excluded at BN=128.
  if (BN == 128 && num_m >= 2 && (num_m % 2) == 0 && !P.fused) P.cg2 = 1;
  P.num_m_k = P.cg2 ? (num_m / 2) : num_m;
  // cg2 ring-depth arm (A/B#6): depth 6 for ALL cg2 nk. The peer's nk<24->4
  // rule (join4-H39) was fitted PRE-v8-stores; the boundary-stall model says
  // a deeper ring cushions the ~2us/item overlap-barrier serialization that
  // dominates case7 (21 items x ~2us of its 161.8us). If this loses, the
  // peer rule stands and is restored.
  P.cg2st = 6;
  // sf-split ring (SFS mechanism) PARKED as dead arm before measurement:
  // analysis showed the producer-side esf wait (slot freed by commit(g-SFS))
  // caps payload lookahead at SFS stages (~2P cushion) — WORSE than the
  // in-stage SF S=4 ring's 4P, defeating the latency-chain goal. A viable
  // variant needs a decoupled SF producer warp (warp 6) issuing SF bulks
  // ahead of the payload producer; deferred (design in /tmp/sf_split_plan).
  P.sf_split = 0;
  if (P.cg2) {
    P.tmb = mxfp4_host::get_tmap(bq_p, (uint64_t)N, (uint64_t)(K / 2), BN / 2);
    P.tsfap = mxfp4_host::get_tmap_u32(sfaw_p, (uint64_t)num_m * nk, 256, 1);
    // SFBN-general packed-SFB map: SFBN/128 rows of 256 words per kb.
    P.tsfbp = mxfp4_host::get_tmap_u32(sfbw_p,
        (uint64_t)num_n * nk * (SFBN / 128), 256, (uint32_t)(SFBN / 128));
  } else if (P.fused == 2) {
    // fd: fp32-scale maps, 8-word inner box per kb group.
    P.tsfap = mxfp4_host::get_tmap_u32_box8(asf_p, (uint64_t)M, (uint64_t)(K / 32), 128);
    P.tsfbp = mxfp4_host::get_tmap_u32_box8(bsf_p, (uint64_t)N, (uint64_t)(K / 32),
                                            (uint32_t)SFBN);
  } else {
    P.tsfap = P.tma;
    P.tsfbp = P.tma;
  }
  // step-14 rskn gate: cg2 only; last-round wave-quantization loss >= 8%;
  // enough idle pairs to host every high half (R <= idle); N divisible by
  // 2*BN (full half tiles). case9 (T=256, I=4, R=34, idle=40, loss 13.5%)
  // is the fixed-suite member; case5 (R=68 > idle=6) stays off.
  P.rskn_R = 0;
  P.tmb_h = P.tmb;
  // step-18: gate generalized to cl1 (worker cap = SM count, half box =
  // BN/2). Exhaustive arithmetic scan of the legal domain (all (n,k) x
  // num_m 1..64, results/p2s18-rskn-cl1-scan.txt): fires ONLY on odd-num_m
  // BN=256 windows that cg2 cannot pair - (nm,n,k): (39|41|47|49, 4096,
  // 12288), (19|22, 12288, 4096), (11, 24576, 4096); losses 10.8-15.7%,
  // all N%512==0 so halves are even 128|128 (n_dim=16, no tail tiles).
  // EVERY fixed-suite and domain-suite shape is a proven no-op (case2/8:
  // R=148 idle=0 zero-loss; case4/d3: R>idle or loss<8%; case7: R>idle;
  // d8: N%448=32; all BN=224 legal n blocked by N%448!=0; case6/d9:
  // single-wave BN=192). Fused excluded (SF staging path differs).
  if ((P.cg2 && P.BN != 128) || (!P.fused && P.clu == 0)) {
    static const int sm_cnt =
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int pairs_cap = P.cg2 ? (sm_cnt / 2) : sm_cnt;
    const int T = P.num_m_k * num_n;
    const int pairs = T < pairs_cap ? T : pairs_cap;
    if (T > pairs) {
      const int Ifull = (T + pairs - 1) / pairs;
      const int R = T - (Ifull - 1) * pairs;
      const int idle = pairs - R;
      const double loss = ((double)Ifull * pairs - (double)T) / ((double)Ifull * pairs);
      if (R >= 1 && R <= idle && (N % (2 * BN)) == 0 && loss >= 0.08) {
        P.rskn_R = R;
        P.tmb_h = mxfp4_host::get_tmap(bq_p, (uint64_t)N, (uint64_t)(K / 2),
                                       P.cg2 ? (uint32_t)(BN / 4)
                                             : (uint32_t)(BN / 2));
      }
    }
  }
  P.SAg = reinterpret_cast<const float*>(asf_p);
  P.SBg = reinterpret_cast<const float*>(bsf_p);
  P.gpr = K / 32;
  // prep-variant rule fitted to kbreak3 step-3 profiler rows (dv bdoqseceu):
  // prep-v2 wins at nk>=24 (case9 13.29->10.65, case3 4.97->4.09) and loses
  // at nk<=16 (case7 +4.1, case2/4/8 +1.6 each); case5 (nk=48, BN=256,
  // num_m=64) is the one nk>=24 loser (+1.0) -> excluded explicitly.
  // num_m>=4 guard: tiny-num_m shapes prefer the v7 256-thread prep
  // (case6 lesson; also protects d4-class domain shapes m<=3, nk>=24).
  // step-21 (v2prepsw dv_7d2e8b6a74e7): prep_v2 DISABLED under
  // TIMING_PROTOCOL_V2. The old rule (nk>=24 && num_m>=4 && !(BN==256 &&
  // num_m>=48) -> v2) was fitted from kbreak3 OLD-protocol profiler rows;
  // its v2-kernel preference was a WARM-fp32-read artifact. Under v2 cold
  // L2 the 256-thr wide-grid v7 wins everywhere the rule fired: case3
  // -5.5%, case9 -4.6% (forced-arm A/B, drift-controlled via identical-
  // path arm replication <=0.1%); no shape prefers the v2 kernel (case5
  // +1.6%, case7 +3.4%, case2 +4.4%, case6 +7.1% when forced). prep_kernel2
  // retained as a dead arm per failed-arm policy.
  P.prep_v2 = 0;
  // prep-v3 (coalesced quad SF pack) DISABLED after measurement (A/B#5
  // dv_462efe6d7592, same job vs v7 tree fcb6ca7): geomean +0.52%, case2
  // +4.92%, case4 +1.15%, and NO gain on the prep-heavy shapes it targeted
  // (case5 -0.12%, case7 +0.06%, case9 +0.06%). Conclusion: v7 prep is not
  // DRAM-pattern-bound at boost clock (its 1.2TB/s NCU figure was base-clock
  // 1.03GHz); the quad restructure only cut block count. Retained as a dead
  // arm per failed-arm policy; prep_kernel (v7) restored.
  P.prep_v3 = 0;
  return P;
}

inline void mxf4_launch(const Mxf4Params& P, cudaStream_t stream) {
  const int SFBN = P.SFBN;  // must match mxf4_prepare's sizing/prep template
  const int pairs = (P.nk + 1) / 2;
  const int grid_prep = P.num_m * 4 + (P.num_m + P.num_n) * pairs;
  if (P.fused) {
    // SF words staged in-kernel by warps 6/7; aflags unused (zf disabled).
  } else if (P.prep_v3) {
    const int nqa = (P.nk + 3) / 4;
    const int grid3 = P.num_m * 4 + (P.num_m + P.num_n) * nqa;
    if (SFBN == 256) {
      prep_kernel3<256><<<grid3, 256, 0, stream>>>(
          P.A, P.K / 2, P.SA, P.M, P.SB, P.N, P.K / 32, P.nk, P.num_m, P.num_n,
          nqa, P.BN, P.flags, const_cast<uint32_t*>(P.sfaw),
          const_cast<uint32_t*>(P.sfbw));
    } else {
      prep_kernel3<128><<<grid3, 256, 0, stream>>>(
          P.A, P.K / 2, P.SA, P.M, P.SB, P.N, P.K / 32, P.nk, P.num_m, P.num_n,
          nqa, P.BN, P.flags, const_cast<uint32_t*>(P.sfaw),
          const_cast<uint32_t*>(P.sfbw));
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "prep3 launch failed");
  } else if (P.prep_v2) {
    const int chunks2 = (P.num_m + P.num_n) * pairs;
    const int grid2 = P.num_m * 4 + (chunks2 + 3) / 4;
    if (SFBN == 256) {
      prep_kernel2<256><<<grid2, 1024, 0, stream>>>(
          P.A, P.K / 2, P.SA, P.M, P.SB, P.N, P.K / 32, P.nk, P.num_m, P.num_m,
          P.num_n, P.BN, P.flags, const_cast<uint32_t*>(P.sfaw),
          const_cast<uint32_t*>(P.sfbw));
    } else {
      prep_kernel2<128><<<grid2, 1024, 0, stream>>>(
          P.A, P.K / 2, P.SA, P.M, P.SB, P.N, P.K / 32, P.nk, P.num_m, P.num_m,
          P.num_n, P.BN, P.flags, const_cast<uint32_t*>(P.sfaw),
          const_cast<uint32_t*>(P.sfbw));
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "prep2 launch failed");
  } else if (SFBN == 256) {
    prep_kernel<256><<<grid_prep, 256, 0, stream>>>(
        P.A, P.K / 2, P.SA, P.M, P.SB, P.N, P.K / 32, P.nk, P.num_m, P.BN,
        P.flags, const_cast<uint32_t*>(P.sfaw), const_cast<uint32_t*>(P.sfbw));
  } else {
    prep_kernel<128><<<grid_prep, 256, 0, stream>>>(
        P.A, P.K / 2, P.SA, P.M, P.SB, P.N, P.K / 32, P.nk, P.num_m, P.BN,
        P.flags, const_cast<uint32_t*>(P.sfaw), const_cast<uint32_t*>(P.sfbw));
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "prep launch failed");
#define MXF4_LAUNCH(BN_, SFBN_, ST_, ACC_, OVL_, FU_, CL_, SFS_)                         \
  mxfp4_host::launch_mxf4<BN_, SFBN_, ST_, ACC_, OVL_, FU_, CL_, SFS_>(                  \
      P.tma, P.tmb, P.tmat, P.a_tail_bytes, P.tsfap, P.tsfbp, P.tmb_h, P.rskn_R,         \
      P.sfaw, P.sfbw, P.D, P.C, P.flags,                                                 \
      P.alpha, P.has_c,                                                                  \
      P.M, P.N, P.K, P.num_m_k, P.num_n, P.SAg, P.SBg, P.gpr, P.A, P.B, stream)
  if (P.cg2) {
    // join4-H39 (peer step-27): nk-conditional cg2 ring depth. nk>=24 ⇒
    // cg2st==6 ⇒ case5/case9 dispatch unchanged from certified.
    if (P.BN == 224) {
      if (P.cg2st == 4) MXF4_LAUNCH(224, 256, 4, 224, 0, 0, 2, 0);
      else MXF4_LAUNCH(224, 256, 6, 224, 0, 0, 2, 0);
    } else if (P.BN == 128) {
      MXF4_LAUNCH(128, 128, 8, 128, 0, 0, 2, 0);
    } else if (P.cg2st == 4) MXF4_LAUNCH(256, 256, 4, 224, 32, 0, 2, 0);
    else MXF4_LAUNCH(256, 256, 6, 224, 32, 0, 2, 0);
  } else if (P.clu) {
    if (P.BN == 256) MXF4_LAUNCH(256, 256, 4, 224, 32, 0, 1, 0);
    else if (P.BN == 224) MXF4_LAUNCH(224, 256, 4, 224, 0, 0, 1, 0);
    else MXF4_LAUNCH(192, 256, 5, 192, 0, 0, 1, 0);
  } else if (P.BN == 256) {
    MXF4_LAUNCH(256, 256, 4, 224, 32, 0, 0, 0);
  } else if (P.BN == 192) {
    // fd ring depths: BN=128 stage 43,008B -> 5; BN=192 stage 56,320B -> 4
    // (227KB dyn-smem cap; raw fp32 SF staging costs +8/+12KB per stage).
    if (P.fused == 2) MXF4_LAUNCH(192, 256, 4, 192, 0, 2, 0, 0);
    else if (P.fused) MXF4_LAUNCH(192, 256, 5, 192, 0, 1, 0, 0);
    else MXF4_LAUNCH(192, 256, 5, 192, 0, 0, 0, 0);
  } else if (P.BN == 224) {
    // sf-split ring arm: payload ring 4->5 with SF in its own 2-slot ring
    // (latency-chain model: period ~ chain/S; A/B#6 case7 -10% at S 4->6).
    if (P.fused) MXF4_LAUNCH(224, 256, 4, 224, 0, 1, 0, 0);
    else if (P.sf_split) MXF4_LAUNCH(224, 256, 5, 224, 0, 0, 0, 2);
    else MXF4_LAUNCH(224, 256, 4, 224, 0, 0, 0, 0);
  } else {
    if (P.fused == 2) MXF4_LAUNCH(128, 128, 5, 128, 0, 2, 0, 0);
    else if (P.fused) MXF4_LAUNCH(128, 128, 6, 128, 0, 1, 0, 0);
    else MXF4_LAUNCH(128, 128, 6, 128, 0, 0, 0, 0);
  }
#undef MXF4_LAUNCH
}

// C++-side CUDA-graph cache: one cudaGraphExec per distinct (pointers, shape,
// alpha, has_c). Replay recomputes everything from current buffer contents;
// capture happens host-side (no allocations inside capture).
struct Mxf4GraphKey {
  uintptr_t p[6];
  long m, n, k2;
  float alpha;
  int has_c;
  bool operator==(const Mxf4GraphKey& o) const {
    for (int i = 0; i < 6; ++i) if (p[i] != o.p[i]) return false;
    return m == o.m && n == o.n && k2 == o.k2 && alpha == o.alpha && has_c == o.has_c;
  }
};
struct Mxf4GraphKeyHash {
  size_t operator()(const Mxf4GraphKey& k) const {
    size_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    for (int i = 0; i < 6; ++i) mix(k.p[i]);
    mix((uint64_t)k.m * 1315423911u + (uint64_t)k.n * 2654435761u + (uint64_t)k.k2);
    mix((uint64_t)k.has_c * 97u + (uint32_t)k.alpha);
    return h;
  }
};

inline void mxfp4_graph_run(const void* aq, const void* asf, const void* bq,
                             const void* bsf, void* d, const void* c, float alpha,
                             int M, int N, int K, cudaStream_t stream) {
  static std::unordered_map<Mxf4GraphKey, cudaGraphExec_t, Mxf4GraphKeyHash> graphs;
  static cudaStream_t cap_stream = nullptr;
  static bool graphs_disabled = false;

  Mxf4GraphKey key;
  key.p[0] = (uintptr_t)aq;
  key.p[1] = (uintptr_t)asf;
  key.p[2] = (uintptr_t)bq;
  key.p[3] = (uintptr_t)bsf;
  key.p[4] = (uintptr_t)d;
  key.p[5] = c ? (uintptr_t)c : 0;
  key.m = M; key.n = N; key.k2 = K / 2;
  key.alpha = alpha; key.has_c = c ? 1 : 0;

  if (!graphs_disabled) {
    auto it = graphs.find(key);
    if (it != graphs.end()) {
      TORCH_CHECK(cudaGraphLaunch(it->second, stream) == cudaSuccess, "graph launch failed");
      return;
    }
  }
  Mxf4Params P = mxf4_prepare_p(aq, asf, bq, bsf, d, c, alpha, M, N, K);
  if (graphs_disabled || graphs.size() >= 128) {
    mxf4_launch(P, stream);
    return;
  }
  if (!cap_stream) {
    if (cudaStreamCreateWithFlags(&cap_stream, cudaStreamNonBlocking) != cudaSuccess) {
      graphs_disabled = true;
      mxf4_launch(P, stream);
      return;
    }
  }
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  bool ok = cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
  if (ok) {
    mxf4_launch(P, cap_stream);
    if (cudaStreamEndCapture(cap_stream, &graph) != cudaSuccess) ok = false;
  } else {
    cudaGetLastError();
  }
  if (ok && graph) {
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
      ok = false;
      cudaGetLastError();
    }
    cudaGraphDestroy(graph);
  }
  if (ok && exec) {
    cudaGraphUpload(exec, stream);  // pre-load onto device: cheaper replays
    graphs.emplace(key, exec);
    TORCH_CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess, "graph launch failed");
    return;
  }
  graphs_disabled = true;
  mxf4_launch(P, stream);
}

inline void mxfp4_run_tcgen05(const at::Tensor& aq, const at::Tensor& asf,
                              const at::Tensor& bq, const at::Tensor& bsf,
                              at::Tensor& d, const at::Tensor* c, float alpha,
                              int M, int N, int K, cudaStream_t stream) {
  mxfp4_graph_run(aq.data_ptr(), asf.data_ptr(), bq.data_ptr(), bsf.data_ptr(),
                  d.data_ptr(), c ? c->data_ptr() : nullptr, alpha, M, N, K, stream);
}

// Pointer-level entry (phase-2 step 1): skips pybind Tensor conversion on the
// hot path. Python gates this on a validated (shape/dtype/contiguity-checked
// once per signature) fast path; graph keying is identical to the tensor path.
void mxfp4_run_ptr(int64_t aq, int64_t asf, int64_t bq, int64_t bsf, int64_t d,
                   int64_t c, int64_t m, int64_t n, int64_t k2, double alpha,
                   bool has_c) {
  const int64_t K = k2 * 2;
  TORCH_CHECK((K % 256) == 0 && (n % 128) == 0 && m >= 1, "ptr path domain");
  static const int cc_major = at::cuda::getCurrentDeviceProperties()->major;
  TORCH_CHECK(cc_major >= 10, "ptr path requires the native tcgen05 domain");
  auto stream = at::cuda::getCurrentCUDAStream();
  mxfp4_graph_run((const void*)(uintptr_t)aq, (const void*)(uintptr_t)asf,
                  (const void*)(uintptr_t)bq, (const void*)(uintptr_t)bsf,
                  (void*)(uintptr_t)d,
                  has_c ? (const void*)(uintptr_t)c : nullptr,
                  static_cast<float>(alpha), static_cast<int>(m), static_cast<int>(n),
                  static_cast<int>(K), stream);
}
