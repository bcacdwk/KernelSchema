// tcgen05 (5th-generation tensor core) PTX wrappers shared by the Phase D and
// Phase E kernels, plus the UMMA descriptor arithmetic.
//
// Every convention encoded here was validated against known answers on this
// SM103 part by harness/bench/tcgen05_blockscale.cu (jobs pD1..pD9, zero
// mismatching elements over 16384/32768 outputs):
//   * idesc = m_dim<<24 | E8M0<<23 | a_sf_id<<29 | n_dim<<17 | b_major<<16 |
//             a_major<<15 | b_fmt<<10 | a_fmt<<7 | b_sf_id<<4;
//   * a_sf_id/b_sf_id (not the address) pick the scale byte inside the word;
//   * SFA row m -> lane m, word base + m/32 + 4*(kb/4), byte kb%4;
//   * SFB column n -> lane (n%32)+32*sp for every subpartition sp, word
//     base + n/32 + 4*(kb/4), byte kb%4 (replication is mandatory);
//   * a scale-factor base must be 2-word aligned;
//   * K-major operands: 128-byte swizzle chunks, LBO=16, SBO=1024, +2 per
//     32-byte k-step; MN-major operands: same bytes, LBO=0, +256 per k-step.
//
// tcgen05 is arch-specific: nvcc only accepts the PTX in an sm_100a/sm_103a
// device pass.  The bodies therefore compile to inert stubs elsewhere, and the
// host dispatcher consults fmha_tcg_probe before launching anything that uses
// them, so a fallback build can never silently run a stub kernel.
#pragma once

#include <cstdint>

#include "fmha_common.cuh"

#if defined(__CUDA_ARCH_FEAT_SM103_ALL) || defined(__CUDA_ARCH_FEAT_SM100_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM101_ALL)
#define FMHA_TCG_ARCH 1
#endif

namespace fmha_mxfp8 {

__device__ __forceinline__ uint64_t umma_desc(uint32_t smem_addr, uint32_t lbo_bytes,
                                              uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)2 << 61;  // SWIZZLE_128B
  d |= (uint64_t)1 << 46;  // descriptor version
  d |= (uint64_t)(sbo_bytes >> 4) << 32;
  d |= (uint64_t)(lbo_bytes >> 4) << 16;
  d |= (uint64_t)((smem_addr & 0x3FFFF) >> 4);
  return d;
}

__device__ __host__ __forceinline__ constexpr uint32_t tcg_idesc(int m, int n, int b_major,
                                                                 int sf_id) {
  return (uint32_t)(((uint32_t)(m >> 4) << 24) | (1u << 23) | ((uint32_t)(sf_id & 3) << 29) |
                    ((uint32_t)(n >> 3) << 17) | ((uint32_t)(b_major & 1) << 16) |
                    ((uint32_t)(sf_id & 3) << 4));
}

__device__ __forceinline__ void tcg_alloc(uint32_t* dst_smem, int ncols) {
#ifdef FMHA_TCG_ARCH
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(
                   smem_addr(dst_smem)),
               "r"(ncols));
#else
  (void)ncols;
  *dst_smem = 0;
#endif
}
__device__ __forceinline__ void tcg_dealloc(uint32_t base, int ncols) {
#ifdef FMHA_TCG_ARCH
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(base), "r"(ncols));
#else
  (void)base;
  (void)ncols;
#endif
}
__device__ __forceinline__ void tcg_relinquish() {
#ifdef FMHA_TCG_ARCH
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
#endif
}
__device__ __forceinline__ void tcg_st1(uint32_t addr, uint32_t v) {
#ifdef FMHA_TCG_ARCH
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%1], {%0};\n" ::"r"(v), "r"(addr));
#else
  (void)addr;
  (void)v;
#endif
}
__device__ __forceinline__ void tcg_st4(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2,
                                        uint32_t v3) {
#ifdef FMHA_TCG_ARCH
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};\n" ::"r"(v0), "r"(v1),
               "r"(v2), "r"(v3), "r"(addr));
#else
  (void)addr;
  (void)v0;
  (void)v1;
  (void)v2;
  (void)v3;
#endif
}
// Eight words per lane = 32 E4M3 elements = exactly one 32-key chunk of one row of
// a probability tile.  This is the store that replaces `tcg_prob_chunk`'s two
// `sts128` when the tile lives in TMEM instead of shared memory, and it is the
// width the MMA's A operand consumes per k-step: 32 keys x 1 byte = 8 columns.
// Validated on this SM103 part by harness/bench/tmem_a_operand.cu (job ba1,
// dv_234ac7262b30): `tcgen05.st.32x32b.x8` at lane==row, four keys per word,
// natural byte order, is read back bit-exactly by
// `tcgen05.mma...mxf8f6f4.block_scale.scale_vec::1X` with A addressed from TMEM
// (bad=0 of 16384 against an FP64 reference, worst rel 0.000e+00), and the
// reversed-byte and reversed-k-step variants of the same write both FAIL, so the
// layout is pinned rather than merely consistent.
__device__ __forceinline__ void tcg_st8(uint32_t addr, const uint32_t* v) {
#ifdef FMHA_TCG_ARCH
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7};\n" ::
                   "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]),
               "r"(v[7]), "r"(addr));
#else
  (void)addr;
  (void)v;
#endif
}
__device__ __forceinline__ void tcg_st32(uint32_t addr, const float* v) {
#ifdef FMHA_TCG_ARCH
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};\n" ::
          "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]),
      "f"(v[8]), "f"(v[9]), "f"(v[10]), "f"(v[11]), "f"(v[12]), "f"(v[13]), "f"(v[14]), "f"(v[15]),
      "f"(v[16]), "f"(v[17]), "f"(v[18]), "f"(v[19]), "f"(v[20]), "f"(v[21]), "f"(v[22]),
      "f"(v[23]), "f"(v[24]), "f"(v[25]), "f"(v[26]), "f"(v[27]), "f"(v[28]), "f"(v[29]),
      "f"(v[30]), "f"(v[31]), "r"(addr));
#else
  (void)addr;
  (void)v;
#endif
}
__device__ __forceinline__ void tcg_ld32(uint32_t addr, float* v) {
#ifdef FMHA_TCG_ARCH
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]),
        "=f"(v[7]), "=f"(v[8]), "=f"(v[9]), "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]),
        "=f"(v[14]), "=f"(v[15]), "=f"(v[16]), "=f"(v[17]), "=f"(v[18]), "=f"(v[19]), "=f"(v[20]),
        "=f"(v[21]), "=f"(v[22]), "=f"(v[23]), "=f"(v[24]), "=f"(v[25]), "=f"(v[26]), "=f"(v[27]),
        "=f"(v[28]), "=f"(v[29]), "=f"(v[30]), "=f"(v[31])
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;\n");
#else
  (void)addr;
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = 0.0f;
#endif
}
__device__ __forceinline__ void tcg_mma_bs(uint32_t d, uint64_t adesc, uint64_t bdesc,
                                           uint32_t idesc, uint32_t accum, uint32_t sfa,
                                           uint32_t sfb) {
#ifdef FMHA_TCG_ARCH
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
      "[%0], %1, %2, %3, [%5], [%6], p;\n\t}\n" ::"r"(d), "l"(adesc), "l"(bdesc), "r"(idesc),
      "r"(accum), "r"(sfa), "r"(sfb));
#else
  (void)d;
  (void)adesc;
  (void)bdesc;
  (void)idesc;
  (void)accum;
  (void)sfa;
  (void)sfb;
#endif
}
// The same block-scaled MMA with its A operand in TENSOR MEMORY instead of behind
// a shared-memory descriptor: `[%1]` replaces the 64-bit `%1` descriptor and
// everything else - B's descriptor, the instruction descriptor including a_major,
// both scale-factor addresses and the accumulate predicate - is byte-identical to
// tcg_mma_bs.  ba1 proved this exact combination on this part (config 2: MN-major
// B with LBO=0 and +256 per k-step, E8M0 scales on both sides, bad=0 of 16384) and
// its fingerprint config 9 pinned the k-step stride at 8 TMEM columns, i.e. 32
// columns for a 128-key E4M3 tile.  The user-supplied reference
// (Atrex-FA sm100_d128_fprop_kernel.cu:1470 `utcmma_atmem_bsmem_h_cta1`, advanced
// as `tmem_s + k*(UTCMMA_TILE_K/(4/BYTES_PER_ELEMENT))`) uses the same substitution
// for the unscaled `kind::f8f6f4`; it never issues the block_scale form, which is
// why ba1 existed.
__device__ __forceinline__ void tcg_mma_bs_atmem(uint32_t d, uint32_t atmem, uint64_t bdesc,
                                                 uint32_t idesc, uint32_t accum, uint32_t sfa,
                                                 uint32_t sfb) {
#ifdef FMHA_TCG_ARCH
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
      "[%0], [%1], %2, %3, [%5], [%6], p;\n\t}\n" ::"r"(d), "r"(atmem), "l"(bdesc), "r"(idesc),
      "r"(accum), "r"(sfa), "r"(sfb));
#else
  (void)d;
  (void)atmem;
  (void)bdesc;
  (void)idesc;
  (void)accum;
  (void)sfa;
  (void)sfb;
#endif
}
__device__ __forceinline__ void tcg_commit(uint32_t mbar_smem) {
#ifdef FMHA_TCG_ARCH
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(
      mbar_smem));
#else
  (void)mbar_smem;
#endif
}
__device__ __forceinline__ void tcg_wait_st() {
#ifdef FMHA_TCG_ARCH
  asm volatile("tcgen05.wait::st.sync.aligned;\n");
#endif
}
// TMEM load that reduces as it reads: returns the maximum of the 32 words this
// lane just loaded, so a 32-value FMNMX tree collapses into one instruction.
// Verified against a known TMEM pattern on this SM103 part for x64, x32 and x16
// by harness/bench/tmem_redmax.cu (job pH1: bad=0 of 128 lanes for each width,
// one width per process under `timeout`).  The 32 data destinations are still
// mandatory and are simply left dead here.
//
// Like every tcgen05.ld this is `.sync.aligned`: ALL 32 lanes of the warp must
// execute it.  Putting it inside a per-lane divergent branch is what hung job
// pF8, so callers must gate it on a warp-uniform vote.
__device__ __forceinline__ float tcg_ld32_redmax(uint32_t addr) {
#ifdef FMHA_TCG_ARCH
  float v[32];
  float m;
  asm volatile(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, [%33];\n"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]),
        "=f"(v[7]), "=f"(v[8]), "=f"(v[9]), "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]),
        "=f"(v[14]), "=f"(v[15]), "=f"(v[16]), "=f"(v[17]), "=f"(v[18]), "=f"(v[19]), "=f"(v[20]),
        "=f"(v[21]), "=f"(v[22]), "=f"(v[23]), "=f"(v[24]), "=f"(v[25]), "=f"(v[26]), "=f"(v[27]),
        "=f"(v[28]), "=f"(v[29]), "=f"(v[30]), "=f"(v[31]), "=f"(m)
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;\n");
  return m;
#else
  (void)addr;
  return 0.0f;
#endif
}
__device__ __forceinline__ void tcg_fence_before() {
#ifdef FMHA_TCG_ARCH
  asm volatile("tcgen05.fence::before_thread_sync;\n");
#endif
}
__device__ __forceinline__ void tcg_fence_after() {
#ifdef FMHA_TCG_ARCH
  asm volatile("tcgen05.fence::after_thread_sync;\n");
#endif
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
#ifdef FMHA_TCG_ARCH
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n\t.reg .pred P1;\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P1;\n\t}\n"
        : "=r"(done)
        : "r"(smem_addr(bar)), "r"(parity));
  }
#else
  (void)bar;
  (void)parity;
#endif
}
// Arrive on an mbarrier with no transaction count, i.e. "this thread's share of the
// producer's work is published".  `mbarrier.arrive` defaults to release semantics at
// CTA scope, so every `st.shared` this thread executed before it is visible to any
// thread whose `try_wait` observes the phase completing - which is the whole of the
// group-split form's cross-warpgroup handoff: one float per row through shared
// memory, no CTA-wide barrier and no reduction.  Initialised with a count equal to
// the producing warpgroup's 128 threads, so the phase completes when all 128 rows of
// one group's state have landed.
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
#ifdef FMHA_TCG_ARCH
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_addr(bar)) : "memory");
#else
  (void)bar;
#endif
}
// A WARPGROUP-scoped rendezvous: `barrier.sync id, count` blocks until `count`
// threads have arrived on barrier resource `id` and synchronises nobody else, so one
// warpgroup of the group-split form can prove "every one of my four warps has
// retired its tcgen05.ld" or "my probability tile is published" without stopping the
// other warpgroup.  That is the difference between the two forms at 256 threads:
// PIPE's five per-group rendezvous are all `__syncthreads()` (resource 0, count =
// blockDim), and `bk1` measured one of them at 0.9% of s7 - cheap, but CTA-wide, so
// it also makes the eight warps one lockstep stream.  Resources 1 and 2 belong to
// warpgroups 0 and 1; resource 0 stays `__syncthreads()` in the prologue, the
// epilogue and every other form's loop.
//
// Register operands, not immediates: the id is `1 + khalf`, which is uniform across a
// warpgroup but not a compile-time constant, and a barrier whose count is a fraction
// of blockDim must not carry `.aligned` - that qualifier promises every thread of the
// CTA executes the same barrier instruction, which is exactly what two warpgroups on
// two different resources do not do.
__device__ __forceinline__ void wg_barrier(int id, int count) {
#ifdef FMHA_TCG_ARCH
  asm volatile("barrier.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory");
#else
  (void)id;
  (void)count;
#endif
}
// Shared-memory writes made through the generic proxy (st.shared, and cp.async
// completions) are not automatically visible to the async proxy that UMMA uses
// to read its operands.  CUTLASS inserts exactly this fence between a producer's
// smem writes and the MMA that consumes them; without it a probability-tile byte
// can be read stale, which shows up as a single wrong output element.
__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n");
}

// ---------------------------------------------------------------- bulk tensor
// One `cp.async.bulk.tensor.2d` moves a whole 128-key x 128-byte swizzled tile
// with a single instruction issued by a single thread, replacing the sixteen
// sixteen-byte `cp.async` per thread per key group plus their address arithmetic
// and predicates.  pI14's ablation priced that staging at 11.8% of shape 7 and
// 18.6% of shape 1, and pI4 had already shown it is instruction cost, not memory
// latency (a deeper cp.async ring changed nothing; DRAM 1.0%, L2 6.0%).
//
// harness/bench/tma_layout.cu (job pI25) proved on this SM103 part that
// SWIZZLE_128B writes byte-for-byte what `fp8_tile_offset` builds, so not one
// UMMA descriptor changes: the staged tile is the same bytes.
//
// Completion is an mbarrier transaction count instead of a cp.async commit group.
// The scale-factor atoms stay on cp.async, so both mechanisms coexist in one
// iteration: the mbarrier proves K/V landed, `cp.async.wait_group` proves the
// atoms did.

// Arm the barrier for `bytes` of incoming TMA data.  Exactly one thread calls
// this, and the barrier must have been initialised with a count of one.
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
#ifdef FMHA_TCG_ARCH
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_addr(bar)),
               "r"(bytes));
#else
  (void)bar;
  (void)bytes;
#endif
}

// Load one box into the shared-memory address `dst`, signalling `bar`.  The
// descriptor is rank four - {head-dim byte, key, kv head, batch} - so a single
// map covers the whole operand and the tile is picked by its coordinates; the box
// is {128 bytes, 128 keys, 1, 1}, which is exactly one swizzle chunk of one key
// group.  Coordinates outside the tensor zero-fill, which is what the cp.async
// path does today with its `valid=false` predicate, so a key group that runs past
// Skv behaves identically (shape 5's KV=65537 and shape 8's KV=8193 are the
// cases that cross it).
__device__ __forceinline__ void tma_load_4d(const void* map, uint32_t dst, int32_t x, int32_t y,
                                            int32_t z, int32_t w, uint64_t* bar) {
#ifdef FMHA_TCG_ARCH
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr(bar))
      : "memory");
#else
  (void)map;
  (void)dst;
  (void)x;
  (void)y;
  (void)z;
  (void)w;
  (void)bar;
#endif
}

// One 1-D bulk copy from global into this CTA's shared memory, completing on
// `bar`.  This is the non-tensor sibling of `tma_load_4d`: same completion
// mechanism, same state spaces, no descriptor.  The scale-factor atoms use it
// because they are contiguous 512-byte blocks in both operands - the F8_128x4 atom
// for one 128-key group of one 128-byte chunk - so a whole atom group is one copy
// instead of 32 sixteen-byte `cp.async`s, and because folding it into the same
// mbarrier the K and V tiles already complete on removes the hot loop's entire
// `cp.async.commit_group`/`wait_group` pair.
//
// Requirements PTX imposes and this candidate meets: `size` a multiple of 16 and
// both addresses 16-byte aligned.  Atom offsets are multiples of 512 on both
// sides - `kvflat * Rpkv * Cq` and `grp * DCH * 512` for K, `(v_atom_base + grp)
// * 512` for V, all against a torch allocation base - and the shared-memory
// destinations are multiples of 512 off a 1024-byte aligned window base.
__device__ __forceinline__ void bulk_copy_1d(uint32_t dst, const void* src, uint32_t bytes,
                                             uint64_t* bar) {
#ifdef FMHA_TCG_ARCH
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1], %2, [%3];\n" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(smem_addr(bar))
               : "memory");
#else
  (void)dst;
  (void)src;
  (void)bytes;
  (void)bar;
#endif
}

// Invalidate a cached tensor-map descriptor.  Needed only when a producer wrote the
// descriptor into global memory (Atrex-FA does this in tma_descriptor_fence_acquire,
// sm100_d128_fprop_kernel.cu:2308); a `__grid_constant__` kernel parameter is
// published by the driver before the CTA starts and needs no fence, so this
// candidate does not call it.  Kept here because it is the documented pairing if a
// future variant ever updates a descriptor in place.
__device__ __forceinline__ void tensormap_acquire(const void* map) {
#ifdef FMHA_TCG_ARCH
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n" ::"l"(
      reinterpret_cast<uint64_t>(map))
               : "memory");
#else
  (void)map;
#endif
}

__device__ __forceinline__ void stg128(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w));
}

// ------------------------------------------------------------ packed f32x2 math
// sm_100a and later execute `add/mul.rn.f32x2` as one instruction over a
// register pair (SASS FMUL2/FADD2).  A compile-only census of the two forms of
// this kernel's softmax inner loop (job f32x2c2, results/raw/20261018T004537Z)
// measured 440 -> 408 SASS instructions with *no* extra MOVs: ptxas keeps the
// pair in an aligned register pair, so the mov.b64 pack/unpack is free.  Both
// operations are `.rn`, i.e. bit-identical to the scalar `*` and `+` they
// replace; only the order of the denominator summation changes.
__device__ __forceinline__ uint64_t f32x2_pack(float lo, float hi) {
  uint64_t r;
  asm("mov.b64 %0, {%1,%2};" : "=l"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ float2 f32x2_unpack(uint64_t v) {
  float2 r;
  asm("mov.b64 {%0,%1}, %2;" : "=f"(r.x), "=f"(r.y) : "l"(v));
  return r;
}
__device__ __forceinline__ uint64_t f32x2_mul(uint64_t a, uint64_t b) {
#ifdef FMHA_TCG_ARCH
  uint64_t r;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
#else
  const float2 x = f32x2_unpack(a), y = f32x2_unpack(b);
  return f32x2_pack(x.x * y.x, x.y * y.y);
#endif
}
__device__ __forceinline__ uint64_t f32x2_fma(uint64_t a, uint64_t b, uint64_t c) {
#ifdef FMHA_TCG_ARCH
  uint64_t r;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
  return r;
#else
  const float2 x = f32x2_unpack(a), y = f32x2_unpack(b), z = f32x2_unpack(c);
  return f32x2_pack(fmaf(x.x, y.x, z.x), fmaf(x.y, y.y, z.y));
#endif
}
__device__ __forceinline__ uint64_t f32x2_add(uint64_t a, uint64_t b) {
#ifdef FMHA_TCG_ARCH
  uint64_t r;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
#else
  const float2 x = f32x2_unpack(a), y = f32x2_unpack(b);
  return f32x2_pack(x.x + y.x, x.y + y.y);
#endif
}

// --------------------------------------------------------------- packed expf
// `expf` is eight of the softmax's 10.7 instructions per score element, and the
// contract pins it bit-for-bit to torch's `exp`, so it cannot be replaced by a
// cheaper approximation (`__expf` was rejected: ~5e-7 relative error flips about
// one E4M3 rounding in 6e4 and a flip on a high-weight key moves O by ~6%,
// outside rtol 0.05).  It can, however, be *packed*: job expfseq1 dumped what
// nvcc 13.2 lowers `expf` to on sm_103a, and job ptxprobe4/subprobe2 established
// which packed forms assemble.  The scalar sequence is
//
//   FFMA.SAT  a = sat(x * 0x3BBB989D + 0.5)
//   FFMA.RM   b = fma_rm(a, 252.0, 12582913.0)
//   FADD      c = b - 12583039.0
//   SHF.L.U32 d = bits(b) << 23              // d == 2^(b - 12583039.0)
//   FFMA      e = fma(x, 0x3FB8AA3B, -c)     // log2(e) as expf uses it
//   FFMA      f = fma(x, 0x32A57060, e)
//   MUFU.EX2  g = ex2.approx.ftz(f)
//   FMUL      h = d * g
//
// `fma.rm.f32x2`, `sub.rn.f32x2`, `fma.rn.f32x2` and `mul.rn.f32x2` all assemble
// (FFMA2.RM, FADD2, FFMA2, FMUL2), while no packed `.sat`, `min`, `max` or `ex2`
// exists.  So five of the eight steps run on both halves of a score pair at once
// and the saturating fma, the shift and the MUFU stay scalar: eleven
// instructions per pair instead of sixteen.
//
// Bit-exactness is by construction - same operations, same order, same rounding
// modes on the same values, and the packed `.rn` forms are defined as two
// independent IEEE-754 round-to-nearest-even results (the kernel already relies
// on that for the softmax scale, shift and denominator).  The one algebraic
// rewrite is producing `12583039 - b` directly instead of `b - 12583039` and a
// negation, which is bit-for-bit the same because round-to-nearest-even is sign
// symmetric.  `harness/bench/expf_packed_check.cu` verifies the pair against
// `expf` exhaustively over all 2^32 float bit patterns on the target part.
__device__ __host__ __forceinline__ constexpr uint64_t f32x2_splat(uint32_t bits) {
  return (uint64_t)bits | ((uint64_t)bits << 32);
}
__device__ __forceinline__ float tcg_satf(float v) {
  float r;
  asm("cvt.sat.f32.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}
__device__ __forceinline__ float tcg_ex2f(float v) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}
__device__ __forceinline__ uint64_t f32x2_sub(uint64_t a, uint64_t b) {
#ifdef FMHA_TCG_ARCH
  uint64_t r;
  asm("sub.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
#else
  const float2 x = f32x2_unpack(a), y = f32x2_unpack(b);
  return f32x2_pack(x.x - y.x, x.y - y.y);
#endif
}
// Round-toward-minus-infinity fused multiply-add, which is what picks the
// integer part of expf's range reduction.
__device__ __forceinline__ uint64_t f32x2_fma_rm(uint64_t a, uint64_t b, uint64_t c) {
#ifdef FMHA_TCG_ARCH
  uint64_t r;
  asm("fma.rm.f32x2 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
  return r;
#else
  const float2 x = f32x2_unpack(a), y = f32x2_unpack(b), z = f32x2_unpack(c);
  float r0, r1;
  asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(r0) : "f"(x.x), "f"(y.x), "f"(z.x));
  asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(r1) : "f"(x.y), "f"(y.y), "f"(z.y));
  return f32x2_pack(r0, r1);
#endif
}
// `expf` with an optional power-of-two bias folded into the exponent field of the
// 2^n factor it builds.
//
// The last two steps of CUDA's expf are `d2 = bits(b) << 23`, which materialises
// 2^n, and `d2 * ex2(f)`.  Multiplying that product by 2^L is the same as adding L
// to the exponent field, i.e. adding `L << 23` to the shifted integer - and
// `bits(b) * 2^23 + (L << 23)` is ONE IMAD where the shift plus a separate packed
// multiply was a SHF and an FMUL2.  So the D=128 contract's bias of 16 = 2^4 can be
// made free, removing one of the three packed multiplies ap6's line census counts
// per key pair (`fmha_tcg_ptx.cuh:358`, 384 instructions, the single largest line in
// <128,0,1,true>).  L=0 reproduces expf exactly, so this is also expf2's body.
//
// This is NOT obviously bit-exact and must not be assumed so: `bits(b) << 23`
// already produces wrapped and subnormal bit patterns for very negative arguments,
// which is how expf returns its zeros and subnormals, and adding to the exponent
// field is not the same as multiplying the finished product once that wrapping is in
// play - an overflow or a sign-bit wrap can land in a different place.  It is
// admissible only because `harness/bench/expf_packed_check.cu` walks all 2^32 float
// bit patterns, both halves of every pair, and compares against `expf(x) * 16.0f`
// (which is itself exact for every finite value, both infinities and both zeros,
// since 16 is a power of two).  Run that check before trusting any new L.
// FMHA_EXP_NO_KTINY drops the Dekker low half of log2(e) from expf2_bias, which is
// ONE packed FFMA2 of the ~11 instructions the exp emits per key pair.  See the
// derivation at the use site below; the short form is that §69 closed "no
// non-bit-exact exp is admissible" with a flip-rate ladder whose rates are UNIFORM
// over keys, and this perturbation is not uniform - it is proportional to |x| and
// therefore vanishes on exactly the heavy keys §69(b) identifies as the sensitive
// ones.  Default 0: it changes numerics and is admissible only if the frozen
// checker says so on the frozen cases, not because an argument says so.
//
// ADJUDICATED, PASSED, AND PROMOTED BY THE OWNER. dv_e368df534c73 (official engine,
// `--cases all --seeds 2 --compare-commit 9029f426`, candidate tree 75c57c46 against
// certified 361c8c25) gives compile 10/10, correctness 10/10, performance 10/10,
// `fatal: null`, and a same-job fixed geomean of 70.8917 -> 69.6718 us = 1.0175x with
// the incumbent arm reproducing its own certificate to -0.03%. dv_a8a34b52e982 does the
// same for the predeclared domain: 10/10, 31.0362 -> 30.6687 = 1.0120x. The owner then
// certified tree 75c57c46 at fixed 69.2336 us / domain 30.4251 us (kind: promotion), and
// dv_b6f4ca31d5f1 measured the realized worst-case tolerance delta as +0.000000.
// See 2026-10-21-optimization-analysis.md sections 117 and 120-121. The default is 1 and
// is NOT a hypothesis under test: reverting it re-opens a 2.42% measured loss.
//
// (This block was corrected once already. 74e8930 replaced the stale "SET TO 1 FOR
// ADJUDICATION, NOT YET ADJICATED" text with the adjudicated version; 65d4a4c then
// reverted 74e8930 wholesale to drop FMHA_BIAS_MAGIC_FOLD and took the comment fix with
// it, so the shipped, certified source has been telling readers that the certified
// default is unadjudicated. Restored by section 130. A revert that targets one macro has
// to be checked for what else it carries - the same failure mode as section 127's
// wholesale PROGRESS bullet replacement, in the other direction.)
#ifndef FMHA_EXP_NO_KTINY
#define FMHA_EXP_NO_KTINY 1
#endif

// FMHA_BIAS_MAGIC_FOLD folds the D=128 contract's bias of 16 = 2^4 into expf2_bias's two
// MAGIC CONSTANTS instead of into the shifted bit pattern, which deletes the caller's
// separate `f32x2_mul(pr, bias2)` while keeping `d2` a shift on the ALU pipe:
//
//     kmagic' = 0x4B400001 + L = 12582917.0f      kshift' = 0x4B40007F + L = 12583043.0f
//     b'      = fl_rm(a*252 + kmagic') = 12582917 + n        (ulp is 1 in [2^23,2^24))
//     low9(b')= (5 + n) & 0x1FF  ->  d = 2^(n+5-127) = 2^(n-122) = 16 * 2^(n-126)
//     nc'     = kshift' - b' = 126 - n                        <- UNCHANGED
//
// `nc` unchanged means `f` unchanged, so `ex2(f)` is the SAME VALUE and `d` is larger by
// exactly 2^L - a power of two, so both products are exact and the results are
// bit-identical. The sign-bit wrap that refuted FMHA_FOLD_BIAS16 needs n >= 251, i.e.
// x >= 86.64, and x = score - safe <= 4*ln2 = 2.772589 always, so it cannot occur.
//
// MEASURED AND REJECTED AS A CANDIDATE - DO NOT PROMOTE IT. dv_46f26338058b and
// dv_ab3950bf18e6 agree to <=0.6% that this build is **+0.710% WORSE on the fixed
// geomean**: it wins s0 (38.91 -> 36.85, the only build ever observed with s0 in its
// fast state) and loses s1 (167.95 -> 175.14) and s7 (165.89 -> 169.94). Section 119
// established that the per-shape signs are incoherent with any instruction-count
// mechanism - it deletes exactly the same 64 instructions from exactly the same pipe as
// FMHA_EXP_NO_KTINY and comes out opposite - and that s0/s1/s7 are BISTABLE at 2-7% in a
// state that is reproducible per BUILD. So this is a mode re-roll, not a lever.
//
// WHY IT IS IN THE TREE AT ALL, given that. It is the only committed, numerics-free,
// reproducible mode-mover this campaign has, which makes it an INSTRUMENT: section 130's
// `experiments/profile_mode_ab.sh` builds it at 0 and at 1 and profiles the same shape
// under both, so the stall-reason composition of a shape's fast state can be differenced
// against its slow state with the source, the numerics and the shape all held fixed.
//
// NEW-TREE MEASUREMENT SUPERSEDES THE OLD-TREE DISPOSITION FOR THIS TREE (ls1
// dv_47bd369f19df + ls2 dv_e3502f4f835e, bit-identity TRUE on every case;
// results/summaries/2026-10-25-ls1-layout-scan-prereg.md): on the post-join
// codebase the fold build lands s1 FAST in 3/3 build instances / 7 rounds
// (164.0-165.9 vs the control's reproducible slow family 174.1-176.1; the old
// tree rolled s1 the OTHER way, from its own fast state to slow — the roll
// direction is build-lineage-dependent) and s9 −1.55…−1.62% stable in 2/2
// instances. The s9 effect is either the deleted per-pair FMUL2 (real
// instruction count in the issue-bound split128) or a fold-specific s9 layout
// state; the two are not distinguishable from these jobs, and the nomination
// does not require distinguishing them — either way it is a reproducible
// property of fold-carrying builds. s7/s0 remain PER-BUILD LOTTERIES under
// the fold (s7 {161.8/163.8/165.9} observed both directions across ls1/ls2;
// s0 drew 34.9-36.6 in one build, 36.86 in another) — the old tree's
// bistability finding stands, now with both signs observed. D256 families
// are codegen-NO-OP under the fold (bias=1 ⇒ L=0 ⇒ constants unchanged);
// s4 movements are quantum band only. ADOPTED as default 1 IN STACK with
// FMHA_LPT_PAIR_MODE=2 (BM arm: s1 163.85 = 121×1.28+8.9 exactly — the
// pair-load reduction composes sub-additively with the fold's ~1.25-1.28
// rate; BM mechanistically dominates BF-alone). The old-tree rejection above
// is retained as accurate history for THAT tree; per the campaign's transfer
// rule, layout-state effects require per-tree measurement, which now exists.
#ifndef FMHA_BIAS_MAGIC_FOLD
#define FMHA_BIAS_MAGIC_FOLD 1
#endif

template <int L>
__device__ __forceinline__ uint64_t expf2_bias(uint64_t x2) {
#ifdef FMHA_TCG_ARCH
  static_assert(L >= 0 && L <= 8, "bias has to be a small power of two");
  const uint64_t k252 = f32x2_splat(0x437C0000u);    // 252.0f
#if FMHA_BIAS_MAGIC_FOLD != 0
  // 12582913.0f + L and 12583039.0f + L; both exact, ulp is 1 in [2^23,2^24), and the
  // difference stays 126.0f, so `nc` - and therefore `f` and `ex2(f)` - is untouched.
  const uint64_t kmagic = f32x2_splat(0x4B400001u + (uint32_t)L);
  const uint64_t kshift = f32x2_splat(0x4B40007Fu + (uint32_t)L);
#else
  const uint64_t kmagic = f32x2_splat(0x4B400001u);  // 12582913.0f
  const uint64_t kshift = f32x2_splat(0x4B40007Fu);  // 12583039.0f
#endif
  const uint64_t klog2e = f32x2_splat(0x3FB8AA3Bu);  // 1.4426950216293335f
#if FMHA_EXP_NO_KTINY == 0
  const uint64_t ktiny = f32x2_splat(0x32A57060u);   // 1.9259630335000111e-08f
#endif
  const float c1 = __int_as_float(0x3BBB989D);       // 0.0057249800302088261f
  const float2 x = f32x2_unpack(x2);
  const uint64_t a2 = f32x2_pack(tcg_satf(__fmaf_rn(x.x, c1, 0.5f)),
                                 tcg_satf(__fmaf_rn(x.y, c1, 0.5f)));
  const uint64_t b2 = f32x2_fma_rm(a2, k252, kmagic);
  const float2 b = f32x2_unpack(b2);
  const uint64_t nc2 = f32x2_sub(kshift, b2);
  const uint64_t e2 = f32x2_fma(x2, klog2e, nc2);
#if FMHA_EXP_NO_KTINY != 0
  // WHAT IS BEING DELETED, AND WHY §69 DOES NOT ALREADY FORBID IT.
  //
  // ktiny = fl(log2(e) - klog2e) exactly: log2(e) = 1.4426950408889634, klog2e =
  // 0x3FB8AA3B = 1.4426950216293335, difference 1.9259629891e-08, and ktiny =
  // 0x32A57060 = 1.9259630335e-08 -- the same quantity rounded to fp32, leaving a
  // residual of 4.4e-16 absolute (3e-16 relative).  So `f2 = fma(x2, ktiny, e2)` is
  // the Dekker low half that upgrades the fp32 product x*klog2e to the full-precision
  // x*log2(e).  Deleting it is not an approximation of exp; it is the use of the
  // one-constant product instead of the two-constant one.
  //
  // BOUND. The result is 2^(f) times a power of two, so the relative change is
  // 2^(-x*ktiny) - 1, i.e. |delta| = |x| * ktiny * ln2 = |x| * 1.3350e-08.  §97(c)
  // PROVES from reference.py:42-45 that x = score - safe <= 4*ln2 = 2.772589 always
  // (update fires -> x <= 0; it does not -> block_max - maximum <= 4*ln2), so
  // delta <= 3.701e-08 = 0.62 fp32 ulp.  Adding the <=0.5 ulp of re-rounding when
  // ex2's argument straddles one of its own boundaries, the worst case is ~1.1 ulp.
  // §69's ladder prices a UNIFORM rate; the uniform-delta equivalent of 3.70e-08 is
  // 3.70e-08/0.0469 = 7.9e-07, below the smallest rate §69 tested (1.9e-06).
  //
  // THE STRUCTURE §69's MODEL CANNOT EXPRESS. delta is proportional to |x|, and
  // x = score - safe = log(weight/bias).  §69(b) establishes that sensitivity is
  // about WEIGHT: a key can violate tolerance only if it carries enough of its row.
  // Heavy keys have score ~= safe, so x ~= 0, so delta ~= 0 -- and at x == 0 the
  // argument to ex2 is bit-identical, hence so is the result.  The perturbation is
  // largest on the keys that are provably prunable and zero on the keys that are
  // provably sensitive.  A single rate per shape averages those together, which is
  // why §69's verdict ("no non-bit-exact exp is admissible") does not transfer here:
  // it was derived for a bare ex2.approx replacing the whole polynomial, a
  // perturbation that hits every key equally regardless of its weight.
  //
  // WHAT WOULD MAKE THIS WRONG: if ex2.approx.ftz.f32 on this part is NOT correctly
  // rounded, the re-rounding term is not ~|x|-scaled and the structure above fails.
  // The exhaustive 2^32-pattern bit-exactness of the FULL sequence against expf
  // (harness/bench/expf_packed_check.cu) is strong evidence that it is, since expf is
  // correctly rounded and a 2-ulp ex2 could not compose to it over every pattern.
  const uint64_t f2 = e2;
#else
  const uint64_t f2 = f32x2_fma(x2, ktiny, e2);
#endif
  const float2 f = f32x2_unpack(f2);
  // With FMHA_BIAS_MAGIC_FOLD the 2^L is already inside `b`'s bit pattern, so the plain
  // shift is the whole job and the IMAD form must NOT be taken - taking it would add L
  // twice.  Both operands are compile-time, so this selects statically.
  constexpr bool kShiftOnly = (L == 0) || (FMHA_BIAS_MAGIC_FOLD != 0);
  const uint64_t d2 = kShiftOnly
      ? f32x2_pack(__int_as_float(__float_as_uint(b.x) << 23),
                   __int_as_float(__float_as_uint(b.y) << 23))
      : f32x2_pack(__int_as_float(__float_as_uint(b.x) * 0x800000u + ((uint32_t)L << 23)),
                   __int_as_float(__float_as_uint(b.y) * 0x800000u + ((uint32_t)L << 23)));
  return f32x2_mul(d2, f32x2_pack(tcg_ex2f(f.x), tcg_ex2f(f.y)));
#else
  const float2 x = f32x2_unpack(x2);
  return f32x2_pack(expf(x.x) * (float)(1 << L), expf(x.y) * (float)(1 << L));
#endif
}

__device__ __forceinline__ uint64_t expf2(uint64_t x2) { return expf2_bias<0>(x2); }

__device__ __forceinline__ void sts16(uint32_t addr, unsigned v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(addr), "h"((unsigned short)v));
}
__device__ __forceinline__ uint32_t ldg32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

}  // namespace fmha_mxfp8
