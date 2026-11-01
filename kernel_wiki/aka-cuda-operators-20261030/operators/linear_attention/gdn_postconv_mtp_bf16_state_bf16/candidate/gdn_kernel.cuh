/* Device code for gdn_postconv_mtp_bf16_state_bf16 (v17); included by
 * candidate/gdn_postconv.cu and type-checked locally by
 * experiments/syntax_check.py against experiments/cudastub, with the element->lane
 * map itself proved by experiments/map_check.py.
 *
 * Target: NVIDIA L20D (SM103, 148 SMs, 233472 B shared/SM, 65536 regs/SM,
 * 2.02 GHz measured under load).
 *
 * Structure: one block per (request, value head) owns that head's whole 128x128
 * state tile for the entire <=8-step recurrence, so the only mandatory per-step
 * DRAM traffic is the 32 KiB BF16 checkpoint store.  A prologue in which warp s
 * materialises step s prefetches q/k/v/gate, decay, beta and the three QK-norm
 * scalars into shared memory; the recurrence loop then contains no global loads
 * and NO barrier at all, because the gated RMSNorm (the only block-wide
 * reduction) is deferred: every step publishes its raw row and its warp's
 * row-square partial to shared memory and one barrier at the end feeds an
 * epilogue that writes all `count` output rows.
 *
 * PROMOTED MAP (v17, job dv_fa6a8d312731): gdn_kernel_w with LPR=8 - eight lanes
 * cooperate on one state row, so a thread owns 4 rows x 2 chunks and one warp-wide
 * 16-byte access writes 4 complete 128-byte lines instead of 16 quarter-written
 * ones, and one shared k/q chunk load feeds all four of the thread's rows.  Against
 * the narrow map that scored 2.330x officially this is -21.2% on the fixed ten
 * (31.23 vs 39.62 us geomean, faster on all ten) and -14.1% on the frozen domain
 * ten (41.83 vs 48.72 us), with 0 mismatched elements everywhere, 127 registers and
 * ZERO spills instead of 166 wanted / ~38 spilled, and 32.97M -> 27.84M executed
 * instructions on case 1.  gdn_kernel (the narrow map, LPR == P) is kept unchanged
 * as variant slots 0-8 so every historical measurement stays reproducible.
 *
 * Why the tile lives in registers, and why the block is 256 threads:
 *  - One FP32 128x128 tile costs 16384 registers, so at most two tiles fit per
 *    SM: 2 blocks x 256 threads = 25% occupancy is the hard ceiling for a
 *    register-resident tile.  The narrow map wanted 166 registers per thread
 *    against the 128 that __launch_bounds__(256,2) allows and spilled ~38 of them;
 *    the wide map fits in 127 with zero spills at the same occupancy, because a
 *    shared chunk load now serves LPR/P rows and the per-step address induction
 *    shrinks with CR = 16/LPR chunks per row.
 *  - A shared-memory-resident FP32 tile (3 blocks/SM, 50% occupancy) measured 25%
 *    SLOWER: it adds 128 KiB of smem traffic per tile-step. Removed.
 *  - P=4 (512-thread block, 32 state elements per thread, 97 registers, ZERO
 *    spills) and P=8 (1024 threads, 61 registers, zero spills, 32 warps/SM) were
 *    both measured repeatedly: P=4 is 1.2-5.6% slower overall and P=8 12-18%
 *    slower, because the per-step fixed cost (five scalar shared loads, six
 *    shuffles, the address induction, two barriers) is amortised over 32 or 16
 *    state elements instead of 64, which costs more instructions per tile than
 *    the extra warps and the missing spills are worth.  P=4 does win individual
 *    shapes by up to 13%, but no host-visible rule separating those from the
 *    shapes where it loses by up to 26% survived both frozen suites, so the
 *    launch pick stayed shape-independent.  All of that was measured with the
 *    narrow map (LPR == P): the wide map below beats both P=4 and P=8 on every
 *    shape of both frozen suites at 256 threads, so the "more threads per tile"
 *    axis is closed and the lanes-per-row axis replaced it.
 *  - A 32-row split with a second kernel for the RMSNorm measured 5-22% slower
 *    per shape: the extra launch, the four-fold duplicated q/k prologue and the
 *    scratch round trip cost more than the finer tail granularity buys.  Removed.
 *  - MINB=3 at P=2 spills heavily (65-73 us); MINB=1 at P=2 removes every spill
 *    but halves occupancy and measures 6.4% slower, which is the cleanest
 *    evidence that occupancy is worth more than spill-freedom here.
 *  - Packed Blackwell fma.rn.f32x2 / mul.rn.f32x2 is worth ~4% and is kept
 *    (guarded so it is only assembled for the arch-accelerated target); the
 *    scalar FP32 control measures 4-6% slower.
 *  - A 100% shared carveout always loses to the auto-sized request.
 *
 * The scheduling of the prologue and of the step body is the result of NCU PC
 * sampling (see the comments at each of the three sites) and is documented with
 * its measurements in PROGRESS.md; every rejected alternative - pre-widened k or
 * q in shared, one-deep shared prefetch, a packed per-step scalar record, a
 * shuffle-selected source slot, interleaved stores at MINB=2 with hoisted
 * epilogue addresses - is listed there with the numbers that rejected it.
 *
 * Precision is exactly the frozen contract's: the tile stays FP32 across steps and
 * is rounded to BF16 (round-to-nearest-even) only at positive checkpoint slots,
 * k/q/v/gate stay the exact BF16 input bits until use, the QK rsqrt factors are
 * folded into per-row scalars (pred, delta*rks, raw) with the reference's 1e-6
 * epsilon, and the RMSNorm uses the caller's norm_eps.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...) do { if (!(cond)) { } } while (0)
#endif

/* Packed FP32x2 arithmetic is an SM100/SM103 arch-accelerated feature, so it is
 * only assembled when the binding was built for the `a` target and explicitly
 * enabled.  Otherwise these helpers fall back to two scalar FP32 ops with
 * identical rounding, and the numerics do not change. */
#if defined(GDN_ENABLE_F32X2)
#define GDN_HAS_F32X2 1
#else
#define GDN_HAS_F32X2 0
#endif

namespace gdn {
namespace {

constexpr int D = 128;              // head dim / state tile edge
constexpr int SM_COUNT = 148;       // L20D: defines "one wave" for the launch pick
constexpr int MAXS = 8;             // contract: at most 8 MTP tokens per request
constexpr float QK_EPS = 1e-6f;     // reference hard-codes 1e-6 for the QK norm
using bf16 = __nv_bfloat16;

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float sigf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* BF16 is the top half of FP32: widen four packed pairs exactly with shifts. */
__device__ __forceinline__ void widen4(const unsigned* w, float* d) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    d[2 * i] = __uint_as_float(w[i] << 16);
    d[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ void up8(uint4 u, float* d) {
  widen4(reinterpret_cast<const unsigned*>(&u), d);
}

/* Read eight raw BF16 values of one shared chunk and widen them.  The caller
 * passes an already-offset BYTE pointer so the chunk index folds into the
 * immediate offset field of the shared access. */
/* RPERM: the position, inside one warp's RPW rows, at which true row `rr` is kept
 * in the vsh and rawsh shared arrays.  A thread owns rows i*RPG + rs for
 * i = 0..R-1, which are RPG elements apart and so cost R separate scalar shared
 * accesses; rowperm maps them to rs*R + i, i.e. R CONTIGUOUS positions, so the step
 * loop's R scalar reads become one vector read and its R scalar publishes become one
 * vector store.  It is a bijection on 0..RPW-1 (rr <-> (i, rs) <-> rs*R + i, and
 * RPW == R*RPG is static_asserted in the kernel); its inverse is
 * (q % R)*RPG + q / R.
 *
 * Only the shared layout moves.  The element->lane map, the global row each thread
 * owns, the tile addressing and the output addresses are untouched, so this is the
 * shared-memory half of CROW isolated from CROW's global-mapping half: CROW made the
 * rows consecutive in the *map* and measured -1.75% fixed / -1.57% domain with a
 * static SASS wash (LDS 33 -> 30), which leaves it open whether what it lost on was
 * the global access pattern or nothing at all.
 *
 * Bank behaviour is unchanged where it matters: the flat epilogue reads 16
 * consecutive true rows, and a permutation of 16 consecutive positions covers the
 * same 16 banks, so no new conflicts appear. */
template <int R, int RPG>
__device__ __forceinline__ int rowperm(int rr) {
  return (rr % RPG) * R + (rr / RPG);
}

__device__ __forceinline__ void ld8sh(const char* p, float* d) {
  up8(*reinterpret_cast<const uint4*>(p), d);
}

/* ALD: read eight contiguous BF16 values through the two 16-byte-ALIGNED blocks that
 * cover the window, then pick the window out of the eight words with four funnel
 * shifts.  Only used when the address is not already 16-byte aligned, i.e. exactly the
 * odd-state_slot_padding path that the portable ladder serves with five 2-/4-byte
 * accesses.
 *
 * Why: NCU on domain d2 (4096 blocks, one step, padding 7) counts 18,690,048 load
 * sectors and 18,317,312 store sectors for 4.19M sectors of logical volume each way -
 * every sector is touched ~4.5 times, because a 2-byte-misaligned 16-byte window goes
 * out as {2,4,4,4,2} bytes and each piece is a separate sector access.  DRAM sits at
 * 17.9% of peak and long_scoreboard reaches 11.50 stalled cycles per issued
 * instruction against 2.14 on the aligned case 1, so that sector traffic, not
 * bandwidth, is what the misaligned path waits on.  Two lane-contiguous aligned
 * accesses replace five scattered partial ones: one instruction now covers 128 whole
 * bytes per row group, and the second block is the neighbouring lane's first, so L1
 * serves it.
 *
 * The word offset t and the bit shift r are uniform across the block: the window
 * address is base + 16p + 128cc, so ((a & 15) >> 2) depends only on the slot/head
 * alignment and the parity of p and (a & 3) only on the slot/head alignment.  That is
 * what makes the switch cheap - one uniform branch, every array index inside it a
 * compile-time constant, so nothing spills to local memory.  For r == 0 the funnel
 * shift is the identity and this reduces to reading words t..t+3 of the pair. */
template <bool CS>
__device__ __forceinline__ void ld8_al(const bf16* __restrict__ p, float* d) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  const unsigned* const A = reinterpret_cast<const unsigned*>(a & ~(uintptr_t)15u);
  const unsigned r = ((unsigned)(a & 3u)) * 8u;   /* 0 or 16 bits */
  const int t = (int)((a >> 2) & 3u);             /* word offset, 0..3 */
  const uint4 u0 = CS ? __ldcs(reinterpret_cast<const uint4*>(A))
                      : *reinterpret_cast<const uint4*>(A);
  const uint4 u1 = CS ? __ldcs(reinterpret_cast<const uint4*>(A + 4))
                      : *reinterpret_cast<const uint4*>(A + 4);
  unsigned w[4];
  switch (t) {
    case 0:
      w[0] = __funnelshift_r(u0.x, u0.y, r); w[1] = __funnelshift_r(u0.y, u0.z, r);
      w[2] = __funnelshift_r(u0.z, u0.w, r); w[3] = __funnelshift_r(u0.w, u1.x, r);
      break;
    case 1:
      w[0] = __funnelshift_r(u0.y, u0.z, r); w[1] = __funnelshift_r(u0.z, u0.w, r);
      w[2] = __funnelshift_r(u0.w, u1.x, r); w[3] = __funnelshift_r(u1.x, u1.y, r);
      break;
    case 2:
      w[0] = __funnelshift_r(u0.z, u0.w, r); w[1] = __funnelshift_r(u0.w, u1.x, r);
      w[2] = __funnelshift_r(u1.x, u1.y, r); w[3] = __funnelshift_r(u1.y, u1.z, r);
      break;
    default:
      w[0] = __funnelshift_r(u0.w, u1.x, r); w[1] = __funnelshift_r(u1.x, u1.y, r);
      w[2] = __funnelshift_r(u1.y, u1.z, r); w[3] = __funnelshift_r(u1.z, u1.w, r);
      break;
  }
  widen4(w, d);
}

/* Load 8 contiguous state elements and widen to FP32.
 *
 * VEC is chosen from the slot stride, but with an odd state_slot_padding (a legal
 * shape_range value) the element offset of slot s is s*(hv*16384+pad), so the
 * alignment varies per slot.  The portable path therefore picks the widest access
 * the actual address allows instead of degrading every slot to eight 2-byte
 * accesses, which measured 6x slower on the domain shapes with padding 1/3/7. */
template <bool VEC, bool CS, bool ALD = false>
__device__ __forceinline__ void ld8(const bf16* __restrict__ p, float* d,
                                    bool ald_ok = true) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  if (VEC || (a & 15u) == 0u) {
    const uint4 u = CS ? __ldcs(reinterpret_cast<const uint4*>(p))
                       : *reinterpret_cast<const uint4*>(p);
    up8(u, d);
  } else if (ALD && ald_ok) {
    ld8_al<CS>(p, d);
  } else if ((a & 7u) == 0u) {
    const uint2* q = reinterpret_cast<const uint2*>(p);
    const uint2 u0 = q[0], u1 = q[1];
    const unsigned w[4] = {u0.x, u0.y, u1.x, u1.y};
    widen4(w, d);
  } else if ((a & 3u) == 0u) {
    const unsigned* q = reinterpret_cast<const unsigned*>(p);
    const unsigned w[4] = {q[0], q[1], q[2], q[3]};
    widen4(w, d);
  } else {
    /* Only 2-byte aligned. Peel the first and last element and take the six in
     * between as three 4-byte accesses; __funnelshift_r re-pairs them in one
     * instruction each. Five memory operations instead of eight matters because
     * with an odd state_slot_padding this is the whole checkpoint path. */
    const unsigned short* e = reinterpret_cast<const unsigned short*>(p);
    const unsigned* q = reinterpret_cast<const unsigned*>(p + 1);
    const unsigned q0 = q[0], q1 = q[1], q2 = q[2];
    const unsigned e0 = e[0], e7 = e[7];
    const unsigned w[4] = {(q0 << 16) | e0,
                           __funnelshift_r(q0, q1, 16),
                           __funnelshift_r(q1, q2, 16),
                           (e7 << 16) | (q2 >> 16)};
    widen4(w, d);
  }
}

/* Pack eight FP32 into exactly the BF16 uint4 that a 16-byte store would write.
 * Shared by st8 below and by the ASA store paths, which need the packed words
 * before (ASAL) or alongside (ASA) the aligned-block assembly. */
__device__ __forceinline__ uint4 pack8(const float* d) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    h[i] = __float22bfloat162_rn(make_float2(d[2 * i], d[2 * i + 1]));
  return u;
}

/* Round 8 FP32 values to BF16 (round-to-nearest-even, like torch) and store with
 * the widest access the actual address allows.
 *
 * ST4 extends the ladder below 16-byte alignment from "as many 4-byte accesses as
 * fit" to "the widest aligned access at each step", which is what the odd-padding
 * path needs: with an odd state_slot_padding the element offset of a window is 0, 4
 * or 2 mod 8 depending only on the slot and head, so the three cases below are
 * block-uniform and cost four, three and four memory operations instead of the
 * ladder's four, four and five.  NCU on domain d2 counts 18,317,312 store sectors for
 * 4.19M sectors of logical volume - every sector touched ~4.4 times - because each of
 * those five pieces is a separate sector access, and the parity edge pair (identical
 * shapes, padding 8 against 7) prices the whole misaligned path at 2.11x and 2.26x.
 * The packed words are the same and every byte is written exactly once; experiments/
 * align_check.py asserts the round trip bit-exactly for every alignment class.
 *
 * st8u is st8 with the pack already done.  The split is textually exact - st8(p, d)
 * is st8u(p, pack8(d)) with the same four __float22bfloat162_rn calls in the same
 * order - and it lets the ASA paths hand the words they have ALREADY packed to the
 * edge ladders instead of packing them a second time, which matters because ASAL
 * overwrites that pack in place. */
template <bool VEC, bool CS, bool ST4 = false>
__device__ __forceinline__ void st8u(bf16* __restrict__ p, uint4 u) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  if (VEC || (a & 15u) == 0u) {
    if (CS) {
      __stcs(reinterpret_cast<uint4*>(p), u);
    } else {
      *reinterpret_cast<uint4*>(p) = u;
    }
  } else if constexpr (ST4) {
    const unsigned* w = reinterpret_cast<const unsigned*>(&u);
    if ((a & 7u) == 0u) {
      /* 8-byte aligned: two 8-byte stores cover the window. */
      uint2* q = reinterpret_cast<uint2*>(p);
      q[0] = make_uint2(w[0], w[1]);
      q[1] = make_uint2(w[2], w[3]);
    } else if ((a & 3u) == 0u) {
      /* 4- but not 8-byte aligned: 4 + 8 + 4. */
      unsigned* q = reinterpret_cast<unsigned*>(p);
      q[0] = w[0];
      *reinterpret_cast<uint2*>(q + 1) = make_uint2(w[1], w[2]);
      q[3] = w[3];
    } else if ((a & 4u) == 0u) {
      /* 2 mod 8: the twelve middle bytes are 8-byte aligned at a+6, so 2+4+8+2. */
      unsigned short* e = reinterpret_cast<unsigned short*>(p);
      e[0] = static_cast<unsigned short>(w[0]);
      reinterpret_cast<unsigned*>(p + 1)[0] = __funnelshift_r(w[0], w[1], 16);
      *reinterpret_cast<uint2*>(p + 3) =
          make_uint2(__funnelshift_r(w[1], w[2], 16),
                     __funnelshift_r(w[2], w[3], 16));
      e[7] = static_cast<unsigned short>(w[3] >> 16);
    } else {
      /* 6 mod 8: the twelve middle bytes are 8-byte aligned at a+2, so 2+8+4+2. */
      unsigned short* e = reinterpret_cast<unsigned short*>(p);
      e[0] = static_cast<unsigned short>(w[0]);
      *reinterpret_cast<uint2*>(p + 1) =
          make_uint2(__funnelshift_r(w[0], w[1], 16),
                     __funnelshift_r(w[1], w[2], 16));
      reinterpret_cast<unsigned*>(p + 5)[0] = __funnelshift_r(w[2], w[3], 16);
      e[7] = static_cast<unsigned short>(w[3] >> 16);
    }
  } else {
    const unsigned* w = reinterpret_cast<const unsigned*>(&u);
    if ((a & 7u) == 0u) {
      uint2* q = reinterpret_cast<uint2*>(p);
      q[0] = make_uint2(w[0], w[1]);
      q[1] = make_uint2(w[2], w[3]);
    } else if ((a & 3u) == 0u) {
      unsigned* q = reinterpret_cast<unsigned*>(p);
#pragma unroll
      for (int i = 0; i < 4; ++i) q[i] = w[i];
    } else {
      /* Only 2-byte aligned: five memory operations instead of eight, with the
       * packed pairs re-formed by one funnel shift each (see ld8). */
      unsigned short* e = reinterpret_cast<unsigned short*>(p);
      unsigned* q = reinterpret_cast<unsigned*>(p + 1);
      e[0] = static_cast<unsigned short>(w[0]);
      q[0] = __funnelshift_r(w[0], w[1], 16);
      q[1] = __funnelshift_r(w[1], w[2], 16);
      q[2] = __funnelshift_r(w[2], w[3], 16);
      e[7] = static_cast<unsigned short>(w[3] >> 16);
    }
  }
}

template <bool VEC, bool CS, bool ST4 = false>
__device__ __forceinline__ void st8(bf16* __restrict__ p, const float* d) {
  st8u<VEC, CS, ST4>(p, pack8(d));
}

/* TRIM: the same pieces as st8u's ST4 ladder, but only those on the kept side of
 * one block-uniform cut.
 *
 * Why that is legal.  CCSEAM leaves exactly two ladders per row and both
 * over-write bytes an aligned block already writes: the entry ladder's window is
 * row bytes [0,16) while block m=1 covers [16-b16, 32-b16), and the exit ladder's
 * is [16*LPR*(CR-1)+16*(LPR-1), +16) while block m=15 covers the same range
 * shifted down by b16.  Only [0,16-b16) of the entry window and [16-b16,16) of the
 * exit one are uniquely theirs.  Dropping the rest cannot change a byte, because
 * every overlapped byte comes from the SAME source element - map_check's COVER
 * property 5, the same fact that already makes the ladder/assembly overlap
 * order-independent - and map_check proves the trimmed set still covers every row
 * for every LPR in {2,4,8} and every b16, while align_check replays a whole 32 KiB
 * checkpoint that way into a guarded pool.
 *
 * Why it is cheap.  The kept pieces are not a new decomposition: 16-b16 falls
 * exactly on a piece boundary of the arm b16 selects (a&7 of 2/4/6 give boundaries
 * {0,2,6,14}, {0,4,12} and {0,2,10,14}), so the kept set is a PREFIX of that list
 * for the entry ladder and a SUFFIX for the exit one, and both are the minimal legal
 * decomposition of the range they cover - align_check asserts all three of those
 * rather than trusting them.  What is left is one comparison per piece against
 * `cut`, which is block-uniform, and ENTRY is a template parameter at a site whose
 * `cc` is a compile-time constant, so the two ladders specialize separately.
 *
 * What it buys, in the two currencies align_check counts: 32 to 64 committed memory
 * operations per warp out of the 112/80/48 that separate the promoted store phase
 * from the aligned one - 57 to 67% of what CCSEAM left - for between 0 and 24
 * issued instructions, depending on whether ptxas hoists comparisons whose only
 * input is loop-invariant.  The archive prices that exchange: 44 issued instructions
 * bought 112 committed ops at CCSEAM and won (d2 -2.1%, d7 -2.3%), and bought 84 at
 * ROWSEAM and lost (d2 +4.7%).
 *
 * Plain stores, not __stcs: st8u's ladder arms have never carried the hint (only
 * its 16-byte arm does), so this keeps the misaligned path's cache behaviour
 * identical to the slot it is A/B'd against. */
template <bool ENTRY>
__device__ __forceinline__ void st8edge(bf16* __restrict__ p, uint4 u,
                                       unsigned b16) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  const unsigned* w = reinterpret_cast<const unsigned*>(&u);
  const unsigned cut = 16u - b16;   /* in 2..14, and a piece boundary */
#define GDN_EDGE(off) (ENTRY ? ((off) < cut) : ((off) >= cut))
  if ((a & 7u) == 0u) {
    /* pieces (0,8) and (8,8) */
    if (GDN_EDGE(0u)) {
      *reinterpret_cast<uint2*>(p) = make_uint2(w[0], w[1]);
    }
    if (GDN_EDGE(8u)) {
      *reinterpret_cast<uint2*>(p + 4) = make_uint2(w[2], w[3]);
    }
  } else if ((a & 3u) == 0u) {
    /* pieces (0,4), (4,8), (12,4) */
    if (GDN_EDGE(0u)) {
      *reinterpret_cast<unsigned*>(p) = w[0];
    }
    if (GDN_EDGE(4u)) {
      *reinterpret_cast<uint2*>(p + 2) = make_uint2(w[1], w[2]);
    }
    if (GDN_EDGE(12u)) {
      *reinterpret_cast<unsigned*>(p + 6) = w[3];
    }
  } else if ((a & 4u) == 0u) {
    /* a & 7 == 2: pieces (0,2), (2,4), (6,8), (14,2) */
    unsigned short* e = reinterpret_cast<unsigned short*>(p);
    if (GDN_EDGE(0u)) {
      e[0] = static_cast<unsigned short>(w[0]);
    }
    if (GDN_EDGE(2u)) {
      reinterpret_cast<unsigned*>(p + 1)[0] = __funnelshift_r(w[0], w[1], 16);
    }
    if (GDN_EDGE(6u)) {
      *reinterpret_cast<uint2*>(p + 3) =
          make_uint2(__funnelshift_r(w[1], w[2], 16),
                     __funnelshift_r(w[2], w[3], 16));
    }
    if (GDN_EDGE(14u)) {
      e[7] = static_cast<unsigned short>(w[3] >> 16);
    }
  } else {
    /* a & 7 == 6: pieces (0,2), (2,8), (10,4), (14,2) */
    unsigned short* e = reinterpret_cast<unsigned short*>(p);
    if (GDN_EDGE(0u)) {
      e[0] = static_cast<unsigned short>(w[0]);
    }
    if (GDN_EDGE(2u)) {
      *reinterpret_cast<uint2*>(p + 1) =
          make_uint2(__funnelshift_r(w[0], w[1], 16),
                     __funnelshift_r(w[1], w[2], 16));
    }
    if (GDN_EDGE(10u)) {
      reinterpret_cast<unsigned*>(p + 5)[0] = __funnelshift_r(w[2], w[3], 16);
    }
    if (GDN_EDGE(14u)) {
      e[7] = static_cast<unsigned short>(w[3] >> 16);
    }
  }
#undef GDN_EDGE
}

/* Broadcast the previous lane's uint4.  Four shuffles, because a 128-bit value is
 * four 32-bit shuffle operands; lane 0 receives its own value, which the caller
 * never uses (lane 0 keeps the ladder). */
__device__ __forceinline__ uint4 shfl_up1(uint4 v) {
  uint4 r;
  r.x = __shfl_up_sync(0xffffffffu, v.x, 1);
  r.y = __shfl_up_sync(0xffffffffu, v.y, 1);
  r.z = __shfl_up_sync(0xffffffffu, v.z, 1);
  r.w = __shfl_up_sync(0xffffffffu, v.w, 1);
  return r;
}

/* ASA: the 16-byte-ALIGNED block whose first byte sits `b16` bytes before this
 * lane's window - i.e. the top b16 bytes of the previous lane's window `n`
 * followed by the bottom 16-b16 bytes of this lane's `u`.  Treating concat(n, u)
 * as a 256-bit little-endian value with words W[0..7], the block is that value
 * shifted right by 8*(16-b16) = 32*k + sb bits, so block word j is
 * funnelshift_r(W[k+j], W[k+j+1], sb).  __funnelshift_r with sb == 0 returns its
 * low operand, which is exactly the b16 == 4 or 12 case, so one form covers both.
 *
 * k and sb are BLOCK-UNIFORM: b16 = (row-run base) mod 16, the row base is
 * 2*(j*sh + mr*sr + dst*ss) with sh and sr multiples of 32 bytes, so b16 depends
 * only on the slot stride and this step's destination slot - never on the lane, the
 * row or the chunk.  That is what makes the switch uniform and every W[] index
 * inside it a compile-time constant, so nothing spills to local memory.  This is
 * the same trick ALD already uses on the read side, applied to the write side.
 *
 * Only k in 0..3 occurs: b16 != 0 is a precondition (b16 == 0 means the window is
 * already aligned and st8's vector path handles it) so 16-b16 is in 2..14. */
__device__ __forceinline__ uint4 asa_block(uint4 n, uint4 u, unsigned k,
                                           unsigned sb) {
  const unsigned W[8] = {n.x, n.y, n.z, n.w, u.x, u.y, u.z, u.w};
  uint4 o;
  switch (k) {
    case 0:
      o.x = __funnelshift_r(W[0], W[1], sb); o.y = __funnelshift_r(W[1], W[2], sb);
      o.z = __funnelshift_r(W[2], W[3], sb); o.w = __funnelshift_r(W[3], W[4], sb);
      break;
    case 1:
      o.x = __funnelshift_r(W[1], W[2], sb); o.y = __funnelshift_r(W[2], W[3], sb);
      o.z = __funnelshift_r(W[3], W[4], sb); o.w = __funnelshift_r(W[4], W[5], sb);
      break;
    case 2:
      o.x = __funnelshift_r(W[2], W[3], sb); o.y = __funnelshift_r(W[3], W[4], sb);
      o.z = __funnelshift_r(W[4], W[5], sb); o.w = __funnelshift_r(W[5], W[6], sb);
      break;
    default:
      o.x = __funnelshift_r(W[3], W[4], sb); o.y = __funnelshift_r(W[4], W[5], sb);
      o.z = __funnelshift_r(W[5], W[6], sb); o.w = __funnelshift_r(W[6], W[7], sb);
      break;
  }
  return o;
}

/* ASAL: one neighbour word.  Splitting the uint4 broadcast into single words is what
 * lets each arm below fetch only the words it consumes, so at most three are live at
 * once instead of four plus a separate output. */
__device__ __forceinline__ unsigned shfl_up_w(unsigned v) {
  return __shfl_up_sync(0xffffffffu, v, 1);
}

/* ASAL: assemble exactly the block asa_block computes, but IN PLACE over the lane's
 * own packed words, with the shift amount a compile-time immediate in every arm.
 *
 * Why it exists.  ASA works - it cut d2's store sectors 28.4% and d1/d2/d7 by
 * 6.7/2.5/7.3% officially - but it costs six registers (slot 23 = CSB+RPERM is 122,
 * slot 24 = CSB+RPERM+ASA is 128, which is the MINB=2 cap), and slot 22 has two
 * free, so the only way to build it was on top of CSB.  CSB is not free: in
 * dv_c9b6cf80b9e2's minima slot 23, which contains no ASA at all, is +5.6% on d4 and
 * +1.7..+2.4% on d3/d8/d9 against slot 22, which is exactly where slot 24 lost the
 * official domain geomean.  So the registers, not the mechanism, are what kept a
 * measured -28.4% sector reduction out of the default.  ASA spends them on two things
 * this removes:
 *
 *   - the output uint4.  The block is concat(n, u) shifted right by 128 - 8*b16 bits,
 *     so writing block word j over u's word j from j = 3 DOWN to j = 0 clobbers only
 *     words every earlier step has already consumed.  No output register exists.
 *   - the neighbour uint4.  Switching on b16 itself rather than on k = (16-b16)>>2
 *     makes sb a literal 0 or 16, so no register holds it, and each arm fetches only
 *     the words it uses: one for b16 in {2,4}, two for {6,8} and {14}, three for
 *     {10,12}.  Peak three live values instead of eight.
 *
 * HAZARD, and the reason the arms are written out long: __shfl_up_sync reads the
 * neighbour's register AT the shuffle, so word m of the neighbour must be fetched
 * before the neighbour overwrites its own word m.  b16 is block-uniform, so every
 * lane of the warp runs the same arm in the same order and the constraint is purely
 * textual - in each arm below, the fetch of n<m> precedes the write to u.<m>.  The
 * caller must also have finished the two edge ladders before calling, because they
 * store the UNSHIFTED window and the assembly destroys it.
 *
 * b16 is even (the run base is 2 * an element offset) and non-zero (the caller's
 * guard), so the seven arms are exhaustive and `default` is exactly b16 == 14.
 * map_check.check_asal proves every arm against the byte model and proves the
 * fetch-before-overwrite order; align_check replays the whole store phase, ladders
 * first, over a guarded pool. */
__device__ __forceinline__ uint4 asa_assemble(uint4 u, unsigned b16) {
  switch (b16) {
    case 2: { /* k=3, sb=16: block = n.w's top 2 bytes then u[0..13] */
      const unsigned n3 = shfl_up_w(u.w);
      u.w = __funnelshift_r(u.z, u.w, 16);
      u.z = __funnelshift_r(u.y, u.z, 16);
      u.y = __funnelshift_r(u.x, u.y, 16);
      u.x = __funnelshift_r(n3, u.x, 16);
      break;
    }
    case 4: { /* k=3, sb=0: block = n.w then u[0..11] */
      const unsigned n3 = shfl_up_w(u.w);
      u.w = u.z;
      u.z = u.y;
      u.y = u.x;
      u.x = n3;
      break;
    }
    case 6: { /* k=2, sb=16 */
      const unsigned n3 = shfl_up_w(u.w);
      const unsigned n2 = shfl_up_w(u.z);
      u.w = __funnelshift_r(u.y, u.z, 16);
      u.z = __funnelshift_r(u.x, u.y, 16);
      u.y = __funnelshift_r(n3, u.x, 16);
      u.x = __funnelshift_r(n2, n3, 16);
      break;
    }
    case 8: { /* k=2, sb=0 */
      const unsigned n3 = shfl_up_w(u.w);
      const unsigned n2 = shfl_up_w(u.z);
      u.w = u.y;
      u.z = u.x;
      u.y = n3;
      u.x = n2;
      break;
    }
    case 10: { /* k=1, sb=16: three neighbour words, n3 spans two steps */
      const unsigned n3 = shfl_up_w(u.w);
      const unsigned n2 = shfl_up_w(u.z);
      u.w = __funnelshift_r(u.x, u.y, 16);
      u.z = __funnelshift_r(n3, u.x, 16);
      const unsigned n1 = shfl_up_w(u.y);
      u.y = __funnelshift_r(n2, n3, 16);
      u.x = __funnelshift_r(n1, n2, 16);
      break;
    }
    case 12: { /* k=1, sb=0 */
      const unsigned n3 = shfl_up_w(u.w);
      const unsigned n2 = shfl_up_w(u.z);
      u.w = u.x;
      u.z = n3;
      const unsigned n1 = shfl_up_w(u.y);
      u.y = n2;
      u.x = n1;
      break;
    }
    default: { /* b16 == 14: k=0, sb=16, all four neighbour words, two live at a time */
      const unsigned n3 = shfl_up_w(u.w);
      u.w = __funnelshift_r(n3, u.x, 16);
      const unsigned n2 = shfl_up_w(u.z);
      u.z = __funnelshift_r(n2, n3, 16);
      const unsigned n1 = shfl_up_w(u.y);
      u.y = __funnelshift_r(n1, n2, 16);
      const unsigned n0 = shfl_up_w(u.x);
      u.x = __funnelshift_r(n0, n1, 16);
      break;
    }
  }
  return u;
}

/* One word of an eight-float chunk, packed.  `m` is a compile-time constant at every
 * call site below, so an arm that needs one word materialises one word: this is what
 * keeps the seam's neighbour inside the register budget instead of adding a uint4. */
__device__ __forceinline__ unsigned packw(const float* d, int m) {
  __nv_bfloat162 h = __float22bfloat162_rn(make_float2(d[2 * m], d[2 * m + 1]));
  return *reinterpret_cast<unsigned*>(&h);
}

/* Broadcast one word from an explicit source lane.  Every lane of the warp executes
 * it; lanes whose result is unused still have to participate, which is why the call
 * sites below are outside every predicate. */
__device__ __forceinline__ unsigned shfl_lane_w(unsigned v, int src) {
  return __shfl_sync(0xffffffffu, v, src);
}

/* CCSEAM: the one aligned block per chunk boundary that no lane's own window can
 * produce, which is the block the two interior edge ladders exist to cover.
 *
 * Geometry, in row bytes (a row is 16*LPR*CR = 256 bytes at LPR=8, CR=2, and the
 * chunk runs are 16*LPR = 128 bytes each, all congruent mod 16 so b16 is the same for
 * every chunk of the row).  ASAL gives lane p of chunk cc the aligned block
 * [16*LPR*cc + 16p - b16, +16) for p = 1..LPR-1, so the blocks it writes are m = 1..7
 * and m = 9..15 of the row's sixteen, and m = 8 - the block straddling the chunk
 * boundary at byte 128 - belongs to nobody.  Its first b16 bytes are the top of lane
 * LPR-1's chunk-0 window and its remaining 16-b16 bytes are the bottom of lane 0's
 * chunk-1 window, so today two ladders write it, and each ladder is three or four
 * partial-sector memory operations for sixteen bytes that one aligned block could
 * cover in one.  Writing that block removes two of the row's four ladders.
 *
 * The writer is lane p == LPR-1, and the two sources are obtained differently, which
 * is the whole design:
 *   - its own previous-chunk window is RE-PACKED from the register-resident tile by
 *     packw() rather than held across the chunk loop.  Holding it would cost 1-4
 *     registers live across an iteration boundary; re-packing costs one cvt per word
 *     and nothing live, and the tile floats are already in registers.
 *   - the row's p == 0 window for the current chunk is fetched by shfl_lane_w from
 *     lane l-(LPR-1), and MUST be fetched here, before the collective asa_assemble
 *     below destroys it in place.
 * Peak liveness is therefore the lane's own u (4) plus (k+1) fetched words plus (4-k)
 * packed words = 9 in every arm, against 7 without the seam, and nothing is held
 * across the loop.  Each arm materialises only the words it reads, which is why the
 * arms are written out rather than sharing one uint4 pair.
 *
 * The block is inside the row for every cc in 1..CR-1: it starts at 16*LPR*cc - b16 >=
 * 16*LPR - 14 > 0 and ends at 16*LPR*cc + 16 - b16 <= 16*LPR*(CR-1) + 14 < 256, so it
 * cannot reach the previous row, the previous head or a neighbouring slot - the hazard
 * that bounds every store mechanism here.  map_check.check_ccseam proves that, the
 * disjointness of the extended block set, its exact coverage with only the two row-edge
 * ladders left, and the benignity of every overlap; align_check replays the store phase.
 *
 * The arms are asa_assemble's, word for word, with the neighbour read from `n` and the
 * value assembled fetched from `src` instead of from lane l-1. */
__device__ __forceinline__ uint4 asa_seam(uint4 own, const float* prev, int src,
                                          unsigned b16) {
  uint4 v;
  switch (b16) {
    case 2: {                       /* k=3, sb=16: one packed word, four fetched */
      const unsigned nw = packw(prev, 3);
      v.x = shfl_lane_w(own.x, src); v.y = shfl_lane_w(own.y, src);
      v.z = shfl_lane_w(own.z, src); v.w = shfl_lane_w(own.w, src);
      v.w = __funnelshift_r(v.z, v.w, 16);
      v.z = __funnelshift_r(v.y, v.z, 16);
      v.y = __funnelshift_r(v.x, v.y, 16);
      v.x = __funnelshift_r(nw, v.x, 16);
      break;
    }
    case 4: {                       /* k=3, sb=0: one packed, three fetched */
      const unsigned nw = packw(prev, 3);
      v.x = shfl_lane_w(own.x, src); v.y = shfl_lane_w(own.y, src);
      v.z = shfl_lane_w(own.z, src);
      v.w = v.z; v.z = v.y; v.y = v.x; v.x = nw;
      break;
    }
    case 6: {                       /* k=2, sb=16: two packed, three fetched */
      const unsigned nz = packw(prev, 2), nw = packw(prev, 3);
      v.x = shfl_lane_w(own.x, src); v.y = shfl_lane_w(own.y, src);
      v.z = shfl_lane_w(own.z, src);
      v.w = __funnelshift_r(v.y, v.z, 16);
      v.z = __funnelshift_r(v.x, v.y, 16);
      v.y = __funnelshift_r(nw, v.x, 16);
      v.x = __funnelshift_r(nz, nw, 16);
      break;
    }
    case 8: {                       /* k=2, sb=0: two packed, two fetched */
      const unsigned nz = packw(prev, 2), nw = packw(prev, 3);
      v.x = shfl_lane_w(own.x, src); v.y = shfl_lane_w(own.y, src);
      v.w = v.y; v.z = v.x; v.y = nw; v.x = nz;
      break;
    }
    case 10: {                      /* k=1, sb=16: three packed, two fetched */
      const unsigned ny = packw(prev, 1), nz = packw(prev, 2), nw = packw(prev, 3);
      v.x = shfl_lane_w(own.x, src); v.y = shfl_lane_w(own.y, src);
      v.w = __funnelshift_r(v.x, v.y, 16);
      v.z = __funnelshift_r(nw, v.x, 16);
      v.y = __funnelshift_r(nz, nw, 16);
      v.x = __funnelshift_r(ny, nz, 16);
      break;
    }
    case 12: {                      /* k=1, sb=0: three packed, one fetched */
      const unsigned ny = packw(prev, 1), nz = packw(prev, 2), nw = packw(prev, 3);
      v.x = shfl_lane_w(own.x, src);
      v.w = v.x; v.z = nw; v.y = nz; v.x = ny;
      break;
    }
    default: {                      /* b16 == 14: k=0, sb=16, four packed, one fetched */
      const unsigned nx = packw(prev, 0), ny = packw(prev, 1);
      const unsigned nz = packw(prev, 2), nw = packw(prev, 3);
      v.x = shfl_lane_w(own.x, src);
      v.w = __funnelshift_r(nw, v.x, 16);
      v.z = __funnelshift_r(nz, nw, 16);
      v.y = __funnelshift_r(ny, nz, 16);
      v.x = __funnelshift_r(nx, ny, 16);
      break;
    }
  }
  return v;
}

/* A 16-byte store to an address the caller has proven 16-byte aligned. */
template <bool CS>
__device__ __forceinline__ void st16al(void* p, uint4 v) {
  if (CS) {
    __stcs(reinterpret_cast<uint4*>(p), v);
  } else {
    *reinterpret_cast<uint4*>(p) = v;
  }
}

__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
#if GDN_HAS_F32X2 && defined(__CUDA_ARCH__)
  float2 r;
  asm("fma.rn.f32x2 %0, %1, %2, %3;"
      : "=l"(*reinterpret_cast<unsigned long long*>(&r))
      : "l"(*reinterpret_cast<const unsigned long long*>(&a)),
        "l"(*reinterpret_cast<const unsigned long long*>(&b)),
        "l"(*reinterpret_cast<const unsigned long long*>(&c)));
  return r;
#else
  return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
#endif
}

__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
#if GDN_HAS_F32X2 && defined(__CUDA_ARCH__)
  float2 r;
  asm("mul.rn.f32x2 %0, %1, %2;"
      : "=l"(*reinterpret_cast<unsigned long long*>(&r))
      : "l"(*reinterpret_cast<const unsigned long long*>(&a)),
        "l"(*reinterpret_cast<const unsigned long long*>(&b)));
  return r;
#else
  return make_float2(a.x * b.x, a.y * b.y);
#endif
}

/* Ask L2 for a line without consuming a register or a scoreboard entry.  A
 * prefetch has no destination, so it cannot change a result: whatever it warms,
 * the real load later reads the same bytes.  Empty in the host pass
 * (__CUDA_ARCH__ undefined), which keeps the header compilable off-device.
 *
 * "No destination, so it is free" is true of the VALUE and false of the SITE, and
 * the price of getting the site wrong is measured rather than argued.  A
 * destinationless two-instruction warm placed BEFORE the block's first loads, as
 * `asm volatile` the compiler may not move, and with NO thread guard so all 256
 * threads issued it, cost **-10.44% fixed / -7.75% domain** (dv_116f832c73c0) at
 * 78 registers with zero spills - byte-for-byte the promoted pick's own allocation,
 * so it was not a register effect.  Two prefetches cannot cost 7.6 us on a 15.7 us
 * cell; 512 duplicate requests for one or two 32-byte sectors per block, 196,640 of
 * them across a 384-block grid, queued ahead of the flight that actually matters,
 * can.  The rule every warm in this kernel that ever measured positive obeys:
 *
 *   1. guard it, so the threads that issue it cover the target bytes EXACTLY once
 *      (`tid < 8*ntok` for the staged q/k/v/gate row, `tid < (D*P)/2` for a
 *      half-tile, all D*P threads for a whole 32 KiB tile at 2*D/P bytes each);
 *   2. issue it where a flight is ALREADY in the air, never ahead of the block's
 *      first loads, and never as a barrier the compiler cannot move.
 *
 * A warm cannot stall on its own result.  It can stall everything behind it. */
__device__ __forceinline__ void prefetch_l2(const void* p) {
#if defined(__CUDA_ARCH__)
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
#else
  (void)p;
#endif
}

/* One 32-byte per-step scalar record: {rqs, rks, kqd, decay, beta, dst, -, -}.
 *
 * PACK replaces the three separate per-step arrays (nrm[MAXS][3], dcy[MAXS][2],
 * dst[MAXS]) with this record so the step loop reads all six values with two
 * broadcast 128-bit shared loads instead of six 4-byte ones, and computes one
 * address instead of three.  The slot index travels as its own bits rather than
 * through a float, so it round-trips exactly; the last two words are padding and
 * are never written or read.  Both halves are 16-byte aligned because the scalar
 * block starts at a multiple of 32 and each record is 32 bytes (Arena asserts it).
 * Factored into functions so experiments/align_check.py can exercise the layout on
 * the host instead of only on a device job. */
__device__ __forceinline__ void rec_store(char* rec, float rqs, float rks, float kqd,
                                          float dec, float bta, int dst) {
  float* const r = reinterpret_cast<float*>(rec);
  r[0] = rqs;
  r[1] = rks;
  r[2] = kqd;
  r[3] = dec;
  r[4] = bta;
  *reinterpret_cast<int*>(r + 5) = dst;
}

__device__ __forceinline__ void rec_load(const char* rec, float& rqs, float& rks,
                                         float& kqd, float& dec, float& bta, int& dst) {
  const float4 a = *reinterpret_cast<const float4*>(rec);
  const float4 b = *reinterpret_cast<const float4*>(rec + 16);
  rqs = a.x;
  rks = a.y;
  kqd = a.z;
  dec = a.w;
  bta = b.x;
  dst = *reinterpret_cast<const int*>(&b.y);
}

/* Read eight already-widened FP32 values from shared: two 16-byte groups, the
 * second 256 bytes after the first because of the perfect-shuffle group order. */
__device__ __forceinline__ void ld8f(const char* p, float* d) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  const float4 b = *reinterpret_cast<const float4*>(p + 256);
  d[0] = a.x; d[1] = a.y; d[2] = a.z; d[3] = a.w;
  d[4] = b.x; d[5] = b.y; d[6] = b.z; d[7] = b.w;
}

/* Fold one 8-element chunk of a BF16-derived vector into one dot-product
 * accumulator pair: even elements into .x, odd into .y, in ascending index
 * order.  This is exactly the association the validated v9 baseline used, so
 * factoring it out into a helper does not change a single rounded result; it
 * only lets the accumulate phase be split into a k pass and a q pass (LEAN)
 * without duplicating the arithmetic. */
template <bool X2>
__device__ __forceinline__ void acc8(const float* s, const float* v, float2& a) {
  if constexpr (X2) {
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      a = fma2(make_float2(s[i], s[i + 1]), make_float2(v[i], v[i + 1]), a);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      a.x = fmaf(s[i], v[i], a.x);
      a.y = fmaf(s[i + 1], v[i + 1], a.y);
    }
  }
}

/* One POD launch parameter block: a single kernel argument instead of 28. */
struct Params {
  const bf16* qkv; int64_t qs0;
  const bf16* ap; int64_t as0;
  const bf16* bp; int64_t bs0;
  const float* alogp;
  const bf16* dtbp;
  const int* idxp; int64_t is0;
  const int* cup;
  const int* accp;
  bf16* state; int64_t ss; int64_t sh; int64_t sr;
  const bf16* gatep; int64_t gs0;
  const bf16* nwp;
  bf16* outp; int64_t os0;
  int h; int hv; int width;
  float scale; float norm_eps;
  int gate_mode; int cs;
  int pool;   /* state.size(0): lets the kernel recognise the tensor's last tile */
  int nreq;   /* state.size(0) of the index table: the LPT dispatch remap needs it */
  int pfd;    /* PFETCH distance in requests: ceil(SM_COUNT*MINB / hv), so block
               * (j, r) warms the tile of (j, r+pfd), which is the block about
               * SM_COUNT*MINB linear positions later in the dispatch order. */
  int pfvg;   /* 1 when the staged-vector prefetch can reach at least half the
               * grid, i.e. nreq >= 2*pfd, which is coverage >= 50%.  The host
               * computes it from the same two integers it already has; see
               * MULTI_WAVE_VARIANT. */
};

/* Shared arena: all four prefetched vectors stay raw BF16.  Pre-widening k to
 * FP32 here was measured twice (QK32, then KF32 in job dv_95cd3adaed0a: 44.41 vs
 * 41.27 us geomean, spills 104/172 -> 136/204 bytes, long_scoreboard 2.23 -> 3.50,
 * shared bank conflicts +90%) and lost both times.  Packing the per-step scalars
 * into one 32-byte record read with two wide accesses (PACK) was also measured
 * and bought nothing: the P=2 kernel needs 166 registers against a 128 cap at
 * MINB=2, so whatever a change frees, the allocator spills somewhere else. */
/* QK32 stores the q and k vectors pre-widened to FP32 and in a permuted 16-byte
 * group order; everything else is unchanged, and with QK32=false the layout and
 * BYTES are bit-for-bit what they have always been.
 *
 * Why the permutation: a reader at (p, cc) wants the 8 columns [cc*8*LPR + 8p, +8),
 * which are two consecutive 4-float groups with logical indices cc*2*LPR + 2p and
 * +1.  Storing logical group g at physical group phi(g) = (g & 1)*16 + (g >> 1) -
 * the perfect shuffle, a bijection on the 32 groups of a 128-element step - maps
 * those to physical groups p + cc*LPR and 16 + p + cc*LPR, i.e. to byte offsets
 * 16*p + 16*LPR*cc and that plus 256.  Two consequences: the reader's offsets are
 * the SAME immediate expressions the BF16 layout uses (16*p in the base, 16*LPR per
 * chunk), so no address registers are added, and the LPR lanes of a row group read
 * 16 bytes at 16*p, which is banks 4p..4p+3 mod 32 - distinct for p < 8, so both
 * LDS.128 phases are conflict-free.  The previous attempt at FP32 k/q without this
 * permutation measured -7.6% with shared bank conflicts +90%, because lane p and
 * p+4 then hit the same four banks. */
/* PACK replaces the three separate per-step scalar arrays - nrm[MAXS][3],
 * dcy[MAXS][2] and dst[MAXS] - with one 32-byte record per step holding all six
 * values, so the step loop reads them with two broadcast LDS.128 instead of six
 * 4-byte loads.  With PACK=false the offsets and BYTES are bit-for-bit what they
 * have always been (SCL_OFF + 160 + 256 + 4096 + 32); with PACK=true the scalar
 * block is 256 bytes and the separate dst array disappears, so BYTES grows by 64.
 * SCL_OFF is 8192 for the BF16 q/k layout, i.e. 16-byte aligned, and each record
 * is 32 bytes, so both float4 halves of a record are aligned. */
template <int P, bool QK32 = false, bool PACK = false, bool LPT = false,
          bool LPT2 = false, bool RSPLIT = false, bool CPA = false,
          bool TSP = false>
struct Arena {
  /* RSPLIT walks the state's rows in NH passes and each pass publishes its own
   * row-square partials, so the rmsp block grows by NH.  Everything else - q/k/v/
   * gate staging, rawsh, the scalar records - is indexed by the TRUE row and is
   * unchanged, which is why BYTES moves by 256 B and not by a factor. */
  static constexpr int NH = RSPLIT ? 2 : 1;
  static constexpr int NT = D * P;
  static constexpr int NW = NT / 32;
  static constexpr int QKB = QK32 ? (MAXS * D * 4) : (MAXS * D * 2);
  static constexpr int Q_OFF = 0;
  static constexpr int K_OFF = Q_OFF + QKB;
  static constexpr int V_OFF = K_OFF + QKB;
  static constexpr int G_OFF = V_OFF + MAXS * D * 2;
  static constexpr int SCL_OFF = G_OFF + MAXS * D * 2;
  static constexpr int SCB = PACK ? (MAXS * 8 * 4) : (MAXS * 3 * 4 + MAXS * 2 * 4);
  static constexpr int NRM_OFF = SCL_OFF;                 /* [MAXS][3] f32 */
  static constexpr int DCY_OFF = NRM_OFF + MAXS * 3 * 4;  /* [MAXS][2] f32 */
  static constexpr int REC_OFF = SCL_OFF;                 /* [MAXS][8] f32, PACK */
  static constexpr int RMSP_OFF = SCL_OFF + SCB;     /* [MAXS][NH*NW] f32 */
  static constexpr int RAW_OFF = RMSP_OFF + MAXS * NW * NH * 4;/* [MAXS][D] f32 */
  static constexpr int DST_OFF = RAW_OFF + MAXS * D * 4;  /* [MAXS] int */
  static constexpr int CORE = PACK ? (RAW_OFF + MAXS * D * 4) : (DST_OFF + MAXS * 4);
  /* A record is read as two float4 halves, so its base must be 16-byte aligned;
   * SCL_OFF is 8192 for the BF16 q/k layout and 12288 for the FP32 one. */
  static_assert(SCL_OFF % 32 == 0, "per-step scalar block must keep records aligned");
  static_assert(RAW_OFF % 16 == 0 && RMSP_OFF % 4 == 0, "shared array alignment");
  /* LPT appends its dispatch-remap scratch AFTER the historical arena, so a slot
   * without LPT has bit-for-bit the BYTES it has always had: a 9-bin histogram of
   * token-count classes, the descending base of each class, one ballot total per
   * warp, and the request the scan selected.  MAXS+1 bins cover counts 0..8, and
   * counts outside that range are inactive, which the remap files in bin 0. */
  static constexpr int LPT_OFF = (CORE + 15) & ~15;
  static constexpr int HIST_OFF = LPT_OFF;
  static constexpr int BASE_OFF = LPT_OFF + (MAXS + 1) * 4;
  static constexpr int WTOT_OFF = BASE_OFF + (MAXS + 1) * 4;
  static constexpr int FOUND_OFF = WTOT_OFF + NW * 4;
  static constexpr int LPT_BYTES = (FOUND_OFF + 4 + 15) & ~15;
  /* LPT2 needs only a [classes][warps] count table and the selected request, so
   * its scratch is 292 B where LPT's is 1 KB - and it is laid out after the same
   * aligned end of the historical arena. */
  static constexpr int TAB_OFF = (CORE + 15) & ~15;
  static constexpr int FOUND2_OFF = TAB_OFF + (MAXS + 1) * NW * 4;
  static constexpr int LPT2_BYTES = (FOUND2_OFF + 4 + 15) & ~15;
  /* CPA (pass-2 cp.async pre-staging, 2026-10-26) appends a thread-private
   * staging image of the second pass's state half AFTER the historical arena
   * end, exactly like LPT/LPT2 append theirs: a slot without CPA has
   * bit-for-bit the BYTES it has always had.  The image is the pass-2 half
   * tile in BF16 - (D/2)*D*2 = 16 KiB - laid out [chunk][thread]*16B so one
   * warp touches a contiguous 512B per chunk index and every consumption
   * LDS.128 is conflict-free (the archived TSTG2 layout).  At MINB=3 the
   * three resident images cost 48 KiB of the 233472 B per-SM budget. */
  static constexpr int BASE_BYTES = LPT2 ? LPT2_BYTES : (LPT ? LPT_BYTES : CORE);
  /* TSP (pass-1 tile cp.async staging, 2026-10-26, prereg
   * results/20261026T2300Z-tsp-p1-prereg.json) appends its own half-tile
   * image AFTER the CPA one, same [chunk][thread]*16B layout.  A slot with
   * neither staging flag has bit-for-bit the BASE_BYTES it has always had,
   * and a CPA-only slot's BYTES is unchanged by this refactor
   * (STAGING_OFF == the old CPA_OFF, TSP_LEN == 0). */
  static constexpr int STAGING_OFF = (BASE_BYTES + 15) & ~15;
  static constexpr int CPA_LEN = CPA ? ((D / 2) * D * 2) : 0;
  static constexpr int CPA_OFF = STAGING_OFF;
  static constexpr int TSP_LEN = TSP ? ((D / 2) * D * 2) : 0;
  static constexpr int TSP_OFF = STAGING_OFF + CPA_LEN;
  static constexpr int BYTES = (CPA || TSP) ? (TSP_OFF + TSP_LEN) : BASE_BYTES;
  static_assert(!CPA || RSPLIT, "CPA stages the split kernel's pass-2 reload");
  static_assert(!TSP || RSPLIT,
                "TSP stages the split kernel's pass-1 half (DR = D/2 rows)");
};

/* P: threads per state row, so the block is D*P threads and each thread owns
 * C = 16/P chunks of 8 state elements (one row, 128/P columns).  NA: independent
 * dot-product accumulator pairs.  F32X2: packed Blackwell fma.rn.f32x2 /
 * mul.rn.f32x2.  CS: evict-first on the checkpoint stream.  MINB: blocks per SM
 * requested through __launch_bounds__.  MERGE: fold the checkpoint store into the
 * update loop instead of running it as a second pass over the chunks.  VEC: the
 * slot stride keeps every state access 16-byte aligned.
 *
 * Everything the 2026-10-19 measurement campaign proved out is now unconditional
 * rather than a knob, because each of these was A/B'd against the alternative in
 * the same job and the alternatives are recorded in PROGRESS.md:
 *  - LEAN live set: state + j*sh + m*sr + p*8 folded into one base pointer, the
 *    two epilogue-only scalars re-read in the epilogue, and the accumulate phase
 *    split into a k pass and a q pass so one 8-float widened transient is live
 *    instead of two.  Spill bytes 104/172 -> 72/80 at NA=2.
 *  - PLOD prologue: one pass (NW >= MAXS >= count, asserted below) that issues
 *    every global load the block needs before consuming any, puts the 32 KiB tile
 *    in the same flight, and defers expf(A_log) so the softplus/exp/sigmoid chain
 *    overlaps the pq/pk/pd shuffle reduction.  PC sampling: the prologue went from
 *    25.9% to 16.4% of all stall samples on case 1.
 *  - SCHED step body: the tile is decayed (S *= g) right after the accumulate loop
 *    so 32 independent packed multiplies issue under the reduction's shuffle
 *    latency (fma(dk, k, g*S) rounds g*S first and fuses afterwards, so this is the
 *    same rounded result), and the four-step rr butterfly runs above the update and
 *    checkpoint work instead of after it, where nothing was left to overlap.
 * Together those three took the ten-shape geomean from 41.07 to 39.73 us and were
 * faster than the baseline on all ten shapes; separately SCHED alone is worse than
 * nothing and PLOD alone is -1.3%. */
template <int P, int NA, bool F32X2, bool CS, int MINB, bool MERGE, bool EPI2,
          bool VEC>
__global__ void __launch_bounds__(D * P, MINB) gdn_kernel(const Params prm) {
  using A = Arena<P>;
  constexpr int C = 16 / P;      // 8-element chunks held per thread
  constexpr int NW = A::NW;      // warps per block
  constexpr bool X2 = F32X2 && (GDN_HAS_F32X2 != 0);
  static_assert(NA == 1 || NA == 2 || NA == 4, "accumulator count must be 1, 2 or 4");
  static_assert(C >= NA, "need at least as many chunks as accumulator pairs");
  static_assert(NW >= MAXS, "the one-pass prologue needs at least MAXS warps");

  const bf16* const qkv = prm.qkv; const int64_t qs0 = prm.qs0;
  const int* const idxp = prm.idxp; const int64_t is0 = prm.is0;
  const int* const cup = prm.cup; const int* const accp = prm.accp;
  bf16* const state = prm.state;
  const int64_t ss = prm.ss, sh = prm.sh, sr = prm.sr;
  const int hv = prm.hv, width = prm.width;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int j = bid % hv;        // value head fastest: keeps one slot local
  const int r = bid / hv;        // request
  const int p = tid % P;         // column part inside a state row
  const int m = tid / P;         // state row
  const int warp = tid >> 5;
  const int lane = tid & 31;

  extern __shared__ __align__(16) char smem[];
  /* Byte pointers: the per-chunk index then folds into the access' immediate
   * offset field instead of costing an IMAD/LEA per chunk. */
  char* const qsh = smem + A::Q_OFF;
  char* const ksh = smem + A::K_OFF;
  bf16* const vsh = reinterpret_cast<bf16*>(smem + A::V_OFF);
  bf16* const gsh = reinterpret_cast<bf16*>(smem + A::G_OFF);
  int* const dstsh = reinterpret_cast<int*>(smem + A::DST_OFF);
  float (*const nrm)[3] = reinterpret_cast<float (*)[3]>(smem + A::NRM_OFF);
  float (*const dcy)[2] = reinterpret_cast<float (*)[2]>(smem + A::DCY_OFF);
  float (*const rmsp)[NW] = reinterpret_cast<float (*)[NW]>(smem + A::RMSP_OFF);
  float* const rawsh = reinterpret_cast<float*>(smem + A::RAW_OFF);

  const int c0 = cup[r];
  const int count = cup[r + 1] - c0;
  const int acc = accp[r];
  bool active = (count > 0) && (count <= MAXS) && (acc > 0) && (acc <= width);
  int src = 0;
  if (active) {
    src = idxp[(int64_t)r * is0 + (acc - 1)];
    active = (src > 0);
  }
  if (!active) {
    /* out must read back zero for every token of an inactive request. */
    for (int t = c0; t < c0 + count; ++t) {
      bf16* o = prm.outp + (int64_t)t * prm.os0 + (int64_t)j * D;
      if (tid < D) o[tid] = __float2bfloat16(0.0f);
    }
    return;
  }

  const int ratio = hv / prm.h;
  const int hq = j / ratio;
  /* One folded base pointer: the recurrence carries this and ss (4 registers)
   * instead of state, sh, ss and mrow (8). */
  bf16* const sjb = state + (int64_t)j * sh + (int64_t)m * sr + p * 8;
  const int64_t irow = (int64_t)r * is0;

  /* ---- prologue: warp s materialises step s and reduces its own norms ----
   * NW >= MAXS >= count, so this is a single pass and warp `warp` owns step
   * `warp`.  Every global load this block needs is issued first - the four BF16
   * vectors, a, b, this step's destination slot, A_log and dt_bias - then the
   * 32 KiB source tile joins the same flight, and only then does anything get
   * consumed.  PC sampling on case 1 put 25.9% of all stall samples in this
   * prologue against ~6% of its instructions, the hottest single instruction of
   * the whole kernel being the first BF16 widen after the q/k loads. */
  const int s0 = warp;
  const bool have = (s0 < count);
  const int t0 = c0 + s0;
  const int d0 = lane * 4;
  const bf16* const qr = qkv + (int64_t)hq * D + (int64_t)t0 * qs0 + d0;
  const bf16* const kr = qkv + (int64_t)(prm.h + hq) * D + (int64_t)t0 * qs0 + d0;
  const bf16* const vr = qkv + (int64_t)(2 * prm.h + j) * D + (int64_t)t0 * qs0 + d0;
  const bf16* const gr = prm.gatep + (int64_t)j * D + (int64_t)t0 * prm.gs0 + d0;
  const float alog = prm.alogp[j];
  const float dtbf = b2f(prm.dtbp[j]);
  uint2 qu = make_uint2(0u, 0u), ku = make_uint2(0u, 0u);
  uint2 vu = make_uint2(0u, 0u), gu = make_uint2(0u, 0u);
  float av = 0.0f, bv = 0.0f;
  int dstv = 0;
  if (have) {
    qu = *reinterpret_cast<const uint2*>(qr);
    ku = *reinterpret_cast<const uint2*>(kr);
    vu = *reinterpret_cast<const uint2*>(vr);
    gu = *reinterpret_cast<const uint2*>(gr);
    if (lane == 0) {
      av = b2f(prm.ap[(int64_t)t0 * prm.as0 + j]);
      bv = b2f(prm.bp[(int64_t)t0 * prm.bs0 + j]);
      dstv = idxp[irow + s0];
    }
  }
  const bf16* sp = sjb + (int64_t)src * ss;
  float S[C][8];
#pragma unroll
  for (int c = 0; c < C; ++c) ld8<VEC, CS>(sp + c * 8 * P, S[c]);

  const float ealog = expf(alog);
  if (have) {
    *reinterpret_cast<uint2*>(vsh + s0 * D + d0) = vu;
    *reinterpret_cast<uint2*>(gsh + s0 * D + d0) = gu;
    *reinterpret_cast<uint2*>(qsh + (s0 * D + d0) * 2) = qu;
    *reinterpret_cast<uint2*>(ksh + (s0 * D + d0) * 2) = ku;
    float qf[4], kf[4];
    widen4(&qu.x, qf);
    widen4(&ku.x, kf);
    float pq = 0.0f, pk = 0.0f, pd = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      pq += qf[i] * qf[i];
      pk += kf[i] * kf[i];
      pd += qf[i] * kf[i];
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      pq += __shfl_xor_sync(0xffffffffu, pq, off);
      pk += __shfl_xor_sync(0xffffffffu, pk, off);
      pd += __shfl_xor_sync(0xffffffffu, pd, off);
    }
    if (lane == 0) {
      /* The warp reduction already produced this step's totals in this lane, so
       * the QK-norm scalars need no second cross-warp pass and no second
       * barrier: one barrier publishes vectors, scalars and the checkpoint slot
       * together. Splitting these three independent groups over lanes 0/1/2 was
       * measured as PTAIL and lost 1.4% on the fixed suite: the two rsqrts and the
       * softplus/exp/sigmoid chain are short next to the vector-load latency that
       * dominates this region, and spreading them adds divergence. */
      const float rqs = rsqrtf(pq + QK_EPS) * prm.scale;
      const float rks = rsqrtf(pk + QK_EPS);
      nrm[s0][0] = rqs;
      nrm[s0][1] = rks;
      nrm[s0][2] = pd * rqs * rks;
      const float x = av + dtbf;
      const float spl = (x > 20.0f) ? x : log1pf(expf(x));
      dcy[s0][0] = expf(-ealog * spl);
      dcy[s0][1] = sigf(bv);
      dstsh[s0] = dstv;
    }
  }
  __syncthreads();

  for (int step = 0; step < count; ++step) {
    const float rqs = nrm[step][0];
    const float rks = nrm[step][1];
    const float kqd = nrm[step][2];
    const float g = dcy[step][0];
    const float beta = dcy[step][1];
    const char* qstep = qsh + (step * D + p * 8) * 2;
    const char* kstep = ksh + (step * D + p * 8) * 2;
    const float vm = b2f(vsh[step * D + m]);

    float2 ak[NA], aq[NA];
#pragma unroll
    for (int n = 0; n < NA; ++n) {
      ak[n] = make_float2(0.f, 0.f);
      aq[n] = make_float2(0.f, 0.f);
    }
    /* k pass then q pass: the eight widened floats of a k chunk are dead before
     * the matching q chunk is loaded, so one transient is live instead of two.
     * ak[n] still sees chunks n, n+NA, ... in ascending order and aq[n] likewise,
     * so every accumulator's summation order is the validated one. */
#pragma unroll
    for (int c = 0; c < C; ++c) {
      float kc[8];
      ld8sh(kstep + c * 16 * P, kc);
      acc8<X2>(S[c], kc, ak[c % NA]);
    }
#pragma unroll
    for (int c = 0; c < C; ++c) {
      float qv[8];
      ld8sh(qstep + c * 16 * P, qv);
      acc8<X2>(S[c], qv, aq[c % NA]);
    }
    /* Serial left-to-right reduction, exactly the association the validated v9
     * tree used. */
    float acc_k = ak[0].x + ak[0].y;
    float acc_q = aq[0].x + aq[0].y;
#pragma unroll
    for (int n = 1; n < NA; ++n) {
      acc_k += ak[n].x + ak[n].y;
      acc_q += aq[n].x + aq[n].y;
    }
    /* Decay the tile before the cross-lane reduction is consumed: 32 independent
     * packed multiplies for the scheduler to issue under the shuffle latency,
     * and the multiply comes off the dk-dependent critical path. */
    if constexpr (X2) {
      const float2 gg = make_float2(g, g);
#pragma unroll
      for (int c = 0; c < C; ++c) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const float2 ns = mul2(make_float2(S[c][i], S[c][i + 1]), gg);
          S[c][i] = ns.x;
          S[c][i + 1] = ns.y;
        }
      }
    } else {
#pragma unroll
      for (int c = 0; c < C; ++c) {
#pragma unroll
        for (int i = 0; i < 8; ++i) S[c][i] = g * S[c][i];
      }
    }
#pragma unroll
    for (int off = P / 2; off >= 1; off >>= 1) {
      acc_k += __shfl_xor_sync(0xffffffffu, acc_k, off);
      acc_q += __shfl_xor_sync(0xffffffffu, acc_q, off);
    }
    const float pred = g * acc_k * rks;
    const float delta = (vm - pred) * beta;
    const float raw = fmaf(delta, kqd, g * acc_q * rqs);
    const float dk = delta * rks;

    /* Publish, do not reduce: the recurrence of step+1 does not depend on the
     * gated RMSNorm of step, so the block-wide part happens once at the end.  The
     * butterfly runs here, above the update and checkpoint work, which now issue
     * underneath its four shuffle latencies; the shared publish stays at the
     * bottom of the body. */
    float rr = raw * raw;
#pragma unroll
    for (int off = P; off < 32; off <<= 1) rr += __shfl_xor_sync(0xffffffffu, rr, off);

    if constexpr (MERGE) {
      /* Folded form: each chunk is packed and stored as soon as it is updated,
       * which is what commit 85ff963 did.  That commit also hoisted the
       * epilogue's addresses into the recurrence and changed the accumulator
       * count at the same time, and the combination measured 18-26% slower on
       * the two largest shapes, so the store fold is re-measured here on its own
       * against a tree that already carries the prologue and reduction fixes. */
      const int dst = dstsh[step];
#pragma unroll
      for (int c = 0; c < C; ++c) {
        float kc[8];
        ld8sh(kstep + c * 16 * P, kc);
        if constexpr (X2) {
          const float2 dkd = make_float2(dk, dk);
#pragma unroll
          for (int i = 0; i < 8; i += 2) {
            const float2 ns = fma2(dkd, make_float2(kc[i], kc[i + 1]),
                                   make_float2(S[c][i], S[c][i + 1]));
            S[c][i] = ns.x;
            S[c][i + 1] = ns.y;
          }
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) S[c][i] = fmaf(dk, kc[i], S[c][i]);
        }
        if (dst > 0) {
          st8<VEC, CS>(sjb + (int64_t)dst * ss + c * 8 * P, S[c]);
        }
      }
    } else {
#pragma unroll
      for (int c = 0; c < C; ++c) {
        float kc[8];
        ld8sh(kstep + c * 16 * P, kc);
        if constexpr (X2) {
          const float2 dkd = make_float2(dk, dk);
#pragma unroll
          for (int i = 0; i < 8; i += 2) {
            const float2 ns = fma2(dkd, make_float2(kc[i], kc[i + 1]),
                                   make_float2(S[c][i], S[c][i + 1]));
            S[c][i] = ns.x;
            S[c][i + 1] = ns.y;
          }
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) S[c][i] = fmaf(dk, kc[i], S[c][i]);
        }
      }

      const int dst = dstsh[step];
      if (dst > 0) {
        bf16* dp = sjb + (int64_t)dst * ss;
#pragma unroll
        for (int c = 0; c < C; ++c) st8<VEC, CS>(dp + c * 8 * P, S[c]);
      }
    }

    if (p == 0) {
      rmsp[step][warp] = rr;
      rawsh[step * D + m] = raw;
    }
  }

  __syncthreads();
  /* EPI2 (the promoted behaviour): the epilogue writes one output element per
   * (step, row) and every value it needs (rmsp, rawsh, gsh, norm_weight) is in
   * shared or global memory, so nothing ties it to the P==0 threads. Letting
   * thread (m, p) take the steps congruent to p covers each (step, m) pair exactly
   * once and shortens the serial chain from count iterations to ceil(count/P), at
   * the same total work. The epilogue is 7.7% of all stall samples on case 1 and is
   * pure latency: each iteration's rsqrt waits on that iteration's eight shared
   * loads. Measured in job dv_81b41f40cae6 against the p==0-only form in the same
   * job: 40.46 vs 40.82 us fixed-suite geomean, better or equal on 8 of the ten
   * shapes (case 1 -2.3%, case 5 -2.3%, case 4 -1.7%, case 6 -1.4%, case 3 -1.1%)
   * and never worse by more than 1%. The flag is kept only so that A/B stays
   * reproducible; EPI2=0 is variant 1 of the table. */
  if (EPI2 || p == 0) {
    const int gm_mode = prm.gate_mode;
    /* The two epilogue-only scalars are re-read here rather than held live across
     * the whole recurrence; both are L1-resident and this runs once per block. */
    const float wm = b2f(prm.nwp[m]);
    const int cb = cup[r];
    for (int step = (EPI2 ? p : 0); step < count; step += (EPI2 ? P : 1)) {
      /* Batch the NW partial loads before summing them: they are independent, so
       * one shared-load latency is exposed per step instead of NW, and the
       * accumulation order is unchanged. */
      float part[NW];
#pragma unroll
      for (int w = 0; w < NW; ++w) part[w] = rmsp[step][w];
      float tot = 0.0f;
#pragma unroll
      for (int w = 0; w < NW; ++w) tot += part[w];
      const float rms = rsqrtf(tot * (1.0f / (float)D) + prm.norm_eps);
      const float gg = b2f(gsh[step * D + m]);
      const float gate = (gm_mode == 1) ? sigf(gg) : gg * sigf(gg);
      prm.outp[(int64_t)(cb + step) * prm.os0 + (int64_t)j * D + m] =
          __float2bfloat16(((rawsh[step * D + m] * rms) * wm) * gate);
    }
  }
}

/* ------------------------------------------------------------------ */
/* WIDE lane mapping (v17 experiment).
 *
 * Same 256-thread block, same register-resident FP32 128x128 tile, same
 * C = 16/P chunks and therefore the same 64 state floats per thread and the same
 * arithmetic instruction count - only the element->lane map changes.  Today
 * (LPR = P = 2) two consecutive lanes hold two column halves of ONE row, so a
 * warp-wide 16-byte access covers 16 different rows and puts 32 contiguous bytes
 * into each of 16 different 128-byte L2 lines.  With LPR lanes per row a thread
 * owns R = LPR/P rows x CR = 16/LPR chunks, and one warp access covers
 * 32/LPR rows x 16*LPR contiguous bytes: at LPR=4 that is 8 rows x 64 bytes, so
 * each STG.128 touches half as many lines with twice the contiguous bytes, and
 * at LPR=8 it is 4 rows x a full 128-byte line.
 *
 * Why this is worth a job: the development store probe measured this exact
 * checkpoint pattern at 51.4 us with the P=2 map and 44.1 us with the P=4 map
 * (case 1, 86 us for the whole kernel, 39.5% of DRAM peak, no pipe above 40%),
 * but that comparison also doubled the thread count, so pattern and memory-level
 * parallelism were confounded.  This kernel holds the thread count, the block
 * count, the occupancy (2 blocks/SM) and the instruction mix fixed and changes
 * only the address pattern; the probe in the same job repeats the isolation at
 * the memory-ceiling level.
 *
 * Second, independent effect: the k/q chunk read from shared now serves all R
 * rows of the thread, so the three per-step shared passes issue CR loads instead
 * of C - 12 LDS.128 per thread-step at LPR=4 and 6 at LPR=8, against 24 today
 * (the step loop is 65.4% of all PC samples, of which the shared loads and their
 * address arithmetic are a fixed part).
 *
 * Cost: the row dot-product butterfly needs log2(LPR) steps per row instead of
 * log2(P), i.e. R*log2(LPR)*2 shuffles per thread-step (2 today, 8 at LPR=4, 24
 * at LPR=8), and the thread carries R sets of the per-row scalars (vm, dk, raw,
 * acc) plus R state base pointers, which is ~8 more registers on top of a kernel
 * that already wants 166 against the 128 that __launch_bounds__(256,2) allows.
 * NA=1 keeps the accumulator pairs at today's count and, because each row's
 * accumulator then sees CR=4 chunks, exactly today's accumulation chain depth.
 *
 * Numerics: the per-row dot product still sums this thread's own chunks in
 * ascending column order into NA accumulators and then the LPR lanes of the row,
 * and the RMSNorm denominator is still one warp partial per (step, warp) summed
 * left-to-right in the epilogue.  Only the association inside those two sums
 * changes (fewer terms per lane, more lanes), which is a re-association of FP32
 * sums at 1e-7 relative against a 5e-2 relative tolerance; the state carry stays
 * FP32 and the checkpoints stay BF16 round-to-nearest-even, so the contract's
 * precision rules are untouched. */
/* The flat epilogue's row index, spelled ONCE.  RPW = DR/NW is rows-per-warp
 * within one tile pass; RPE = D/NW is rows-per-warp in the epilogue, which covers
 * all D rows.  The two are equal only when NH == 1, and NH = RSPLIT ? 2 : 1 - so
 * at RSPLIT (NH=2, D=128, NW=8) RPW is 8 and RPE is 16.  Two spellings of "the
 * epilogue's row" therefore compile to different addresses on the promoted
 * package, and no static_assert can see it because both typecheck.  NWE's hoisted
 * norm_weight load had exactly that defect and its own comment asserted the two
 * forms "cannot differ"; every site that needs the epilogue's row now uses this
 * macro, and experiments/launch_check.py section 10 pins the macro's expansion by
 * value so the two sites cannot drift apart or drift together into being wrong. */
#define GDN_EPI_ROW(w, l) ((w)*RPE + ((l) % RPE))
template <int P, int LPR, int NA, bool F32X2, bool CS, int MINB, int EPIM,
          bool ALD, bool ST4, bool QK32, bool PACK, bool PUB1, bool VEC,
          bool CSB = false, bool RPERM = false, bool ASA = false,
          bool ASAL = false, bool CSBODD = false, bool CCSEAM = false,
          bool ROWSEAM = false, bool GRID2D = false, bool IDXPAR = false,
          bool NOGATE = false, bool HQFAST = false, int LDB = 0,
          bool NWE = false, bool KQREG = false, bool PLOV = false,
          bool HOIST = false, bool NOUNROLL = false, int CM = -1,
          bool LPT = false, int LPTW = 0, int LPT2W = 0, int PFW = 0,
          bool PFV = false, bool TRIM = false, bool RSPLIT = false,
          bool PFT = false, int PF2 = 0, bool CPA = false, bool TSP = false,
          bool PFL = false>
__global__ void __launch_bounds__(D * P, MINB) gdn_kernel_w(const Params prm) {
  /* NH row-halves walked in sequence by ONE block.  The recurrence is row-local -
   * with S[i,k], pred[i] = sum_k S[i,k]*k[k] and the rank-1 update both stay inside
   * row i - so the only thing coupling the halves is the gated RMSNorm's mean over
   * i, and the promoted epilogue already defers that to a shared partial plus one
   * barrier at the end.  Splitting therefore needs no second kernel, no global
   * scratch and no cross-block communication: it needs one more column in rmsp.
   *
   * Why: the tile is 64 of this kernel's 123-128 registers, which is what pins it
   * at 2 blocks/SM and 24% occupancy, and occupancy is measured to be worth about
   * 21% per doubling (slot 55 against 45, spill-free, dv_746c8213c04e).  Halving the
   * tile to 32 floats per thread is the only way to reach 3 blocks/SM without
   * spilling it - MINB=3 at the full tile fits 80 registers with 312-328 B of stack
   * and is 38% SLOWER on every one of the twenty cells (dv_62285ff2da47).
   *
   * This is not the archived 32-row split, which measured 5-22% slower for three
   * named reasons: an extra kernel launch for the RMSNorm, a four-fold duplicated
   * q/k prologue, and a scratch round trip.  None of the three is here - one block,
   * one launch, one staging pass shared by both halves, rmsp in shared - and the
   * split is two-way, not four-way.  What it does cost is the genuinely per-step
   * work paid twice per tile: the k/q chunk loads and widening, the scalar record
   * load, the rr butterfly and the loop/address induction, roughly +40 of 253
   * warp-instructions per step.  The per-row work (the four f32x2 passes, the
   * butterflies, the store phase, the tile load) totals the same as before. */
  constexpr int NH = RSPLIT ? 2 : 1;
  using A = Arena<P, QK32, PACK, LPT, (LPT2W > 0), RSPLIT, CPA, TSP>;
  static_assert(PFW == 0 || GRID2D, "PFETCH walks the 2-D grid's request axis");
  static_assert(PFW == 0 || !PLOV, "the prefetch loads ride the non-PLOV flight");
  static_assert(PFW >= 0 && PFW <= 4, "PFETCH distance in waves");
  static_assert(PFW == 0 || (2 * D) % P == 0,
                "the per-thread prefetch stride must divide the tile");
  static_assert(!PFV || PFW > 0, "the vector prefetch rides the tile prefetch's target");
  static_assert(!PFT || (PFW > 0 && RSPLIT && GRID2D),
                "the tail self-prefetch lives in the PFW branch's else case and "
                "needs a second pass whose flight it warms");
  static_assert(PF2 >= 0 && PF2 <= 2,
                "PF2: 0 = off, 1 = late self pass-2 warm at pass-1 step 0, "
                "2 = prologue self pass-2 warm on the has-target branch");
  static_assert(PF2 == 0 || (RSPLIT && PFT),
                "PF2 warms the second pass's own reload and reuses PFT's "
                "coverage contract: tail blocks stay on PFT's early issue, "
                "PF2 only extends the warm to the blocks PFT cannot reach");
  static_assert(!CPA || (RSPLIT && !CSB && !CSBODD),
                "CPA-P2 stages the pass-2 reload through the direct mr[]-based "
                "address form the non-CSBE reload uses; the CSB odd-row-base "
                "variants keep the global reload path");
  static_assert(!TSP || (RSPLIT && !CSB && !CSBODD),
                "TSP-P1 stages the pass-1 tile through rowbase(), whose "
                "non-CSBE form is the direct mr[]-based address the issue "
                "loop mirrors; CSB variants keep the global load path");
  static_assert(!TSP || (!PLOV && LDB == 0),
                "TSP consumes into S[] on the DEFAULT pass-1 load branch; "
                "PLOV and LDB own that branch in their instantiations and "
                "would bypass the consumption, leaving S[] unfilled");
  /* TRIM narrows the two ladders CCSEAM leaves to the bytes no aligned block
   * reaches, so it needs CCSEAM's boundary blocks to exist (without them the
   * interior chunk boundaries are written ONLY by those ladders), ASAL's loop to
   * host them, and the ST4 piece list whose prefix/suffix the kept pieces are. */
  static_assert(!TRIM || (ASAL && CCSEAM && ST4 && (16 / LPR) >= 2),
                "TRIM trims CCSEAM's two remaining ST4 ladders");
  constexpr int DR = D / NH;      // state rows the register tile holds at once
  /* PFL: cover every 128-byte LINE of the prefetched region, not one line per
   * thread at a region_bytes/threads stride.
   *
   * This is a defect fix, and it is invisible at P=2.  `prefetch.global.L2 [p]`
   * warms the single 128-byte line containing p, so a prefetch sweep covers the
   * region only if consecutive threads are 128 bytes apart.  The three sweeps
   * below were written as `tid * (2*D/P)` - the region's byte count divided by
   * the block's thread count - which is 128 at P=2 and 256 at P=1.  At P=1 the
   * sweep therefore touches every OTHER line and warms half of what it claims:
   * PFW's next-wave tile (16 KiB of the 32 KiB) and PFT's own pass-2 rows (8 KiB
   * of 16 KiB).  Every P=1 row in the table - 66, 70, 71 and 77, which is the
   * whole band the 2026-10-28 dispatch rewrite just widened to fixed:3/4/6/7/8 -
   * has been running with half a prefetch.  The comment at the PFW site still
   * says "256 threads x 128 bytes is exactly one tile", which is true and is
   * exactly the assumption that does not survive P=1.
   *
   * PFL replaces the stride with a compile-time-trip-count loop over 128-byte
   * lines.  At P=2 the loop runs once per participating thread and the addresses
   * are IDENTICAL to the old expression (TILE_LINES/(D*P) == 1 and t*128 ==
   * tid*(2*D/P)), so a P=2 slot with PFL is an identical-code control rather than
   * a mechanism - which is what slot 80 is for.  At P=1 it runs twice and covers
   * the whole region.  A prefetch has no destination and writes no byte, and the
   * real loads still perform their own coherency, so results stay bit-identical
   * either way; the arms are fingerprint-gated regardless. */
  constexpr int TILE_LINES = (D * D * 2) / 128;         /* a whole 128x128 tile */
  constexpr int PASS_LINES = ((D - DR) * D * 2) / 128;  /* rows [DR, D) of it */
  static_assert(TILE_LINES * 128 == D * D * 2 && PASS_LINES * 128 == (D - DR) * D * 2,
                "the prefetch sweeps are counted in whole 128-byte lines");
  static_assert(!(PFT || PF2) || PASS_LINES > 0,
                "the tail/self pass prefetch warms rows [DR, D); with no row "
                "split that region is empty, which is why PFT and PF2 both "
                "already require RSPLIT");
  constexpr int C = DR / (8 * P);  // 8-element chunks per thread
  constexpr int R = C * LPR / 16;  // state rows per thread
  constexpr int CR = 16 / LPR;    // chunks of one row per thread
  constexpr int NW = A::NW;       // warps per block

  /* Staging rounds.  The prologue gives warp `warp` the steps warp, warp+NW,
   * warp+2*NW, ... so any NW covers the contract's MAXS steps: SROUND == 1 is
   * the one-step-per-warp prologue every dispatched slot uses (NW == MAXS == 8),
   * and SROUND == 2 is P=1 (NW == 4), which is what makes four resident streams
   * per SM reachable at 128 threads per block.  See
   * results/p1-four-streams-20261026/. */
  constexpr int SROUND = (MAXS + NW - 1) / NW;
  constexpr int RPW = DR / NW;    // rows per warp, within one pass
  constexpr int RPE = D / NW;     // rows per warp in the epilogue, which covers
                                  // all D rows across both passes
  constexpr int RPG = 32 / LPR;   // distinct rows in one warp-wide access
  constexpr bool X2 = F32X2 && (GDN_HAS_F32X2 != 0);
  /* CM splits the single streaming hint into a load hint (bit 0) and a store
   * hint (bit 1).  CM < 0 - every historical slot - leaves both at CS, so those
   * instantiations are byte-identical to what they have always been.  The split
   * exists because the two directions want opposite policies at opposite ends of
   * the working-set range: the source tile is dead the moment it is widened, so
   * evict-first never costs it anything, while the checkpoints are the one thing
   * a later call re-reads, and when the whole touched region fits in this
   * device's 126.5 MiB of L2 an evict-first store pays a DRAM write the caller
   * did not ask for. */
  constexpr bool CSLD = (CM < 0) ? CS : ((CM & 1) != 0);
  constexpr bool CSST = (CM < 0) ? CS : ((CM & 2) != 0);
  constexpr int SINC = 32 / RPW;  // steps one lane covers per EPIM=2 iteration
  constexpr int SINCE = 32 / RPE; // ... in the epilogue's own row geometry
  static_assert(LPR >= P && LPR <= 32 && (LPR & (LPR - 1)) == 0,
                "lanes per row must be a power of two and at least P");
  static_assert(R * CR == C, "row/chunk bookkeeping must tile the thread's share");
  static_assert(CR >= NA, "need at least as many chunks per row as accumulators");
  static_assert(RPW == R * RPG, "the warp's rows must split into R groups");
  /* Weaker than `NW >= MAXS` and sufficient: the prologue is round-strided, so
   * SROUND rounds of NW warps cover MAXS steps at any NW.  The narrow kernel's
   * copy of this assert is deliberately left as `NW >= MAXS` - its prologue was
   * not strided, and it is not dispatched. */
  static_assert(SROUND * NW >= MAXS,
                "the prologue must be able to stage all MAXS steps: warp w takes "
                "steps w, w+NW, ... so SROUND rounds of NW warps must cover MAXS");
  static_assert(EPIM >= 0 && EPIM <= 4, "epilogue mode");
  static_assert(EPIM < 2 || (RPE <= 32 && SINCE >= 1), "flat epilogue geometry");
  /* RSPLIT is wired into the promoted flat epilogue and the plain (non-PLOV,
   * non-LDB) tile load only, and PUB1 must be off because its rawsh publish uses a
   * lane->row assignment that carries no pass offset.  Keeping the surface this
   * small is deliberate: every path it does not touch stays textually identical, so
   * the promoted slots' codegen cannot move. */
  /* CSBE is not in scope this far up - it is derived below, next to the row-base
   * accessors it selects - so the condition is spelled out.  It excludes the
   * constant-stride row base because the second pass advances sjb[] with the same
   * expression the first pass filled it with, and CSBE would make rowbase() read
   * sjb0 instead. */
  static_assert(!RSPLIT || (EPIM == 2 && !PUB1 && !PLOV && LDB == 0 &&
                            !(CSB || (CSBODD && !VEC))),
                "RSPLIT is built for the promoted package's paths only");
  static_assert(!RSPLIT || (D % NH == 0 && DR % RPW == 0 && RPW % RPG == 0),
                "the row halves must tile into warps and groups");
  /* EPIM 3/4 choose between the two proven epilogues at run time, per block, from
   * the token count: below EPT the flat (step, row) form, at or above it the
   * steps-split form.  EPT is folded into the mode value so neither the template
   * signature nor the Variant table grows a field for it. */
  constexpr int EPT = (EPIM == 3) ? 6 : (EPIM == 4) ? 8 : MAXS + 1;
  static_assert(EPIM < 3 || (EPT >= 1 && EPT <= MAXS), "epilogue threshold");
  /* PUB1 publishes through a different lane->row assignment that rowperm does not
   * describe; PUB1 is rejected anyway, so the combination is simply not built. */
  static_assert(!(PUB1 && RPERM), "PUB1's publish layout is not row-permuted");
  /* ASA and ASAL are the same mechanism in two register budgets: ASA broadcasts the
   * neighbour's whole uint4 and builds a separate output, ASAL assembles in place
   * from single-word broadcasts.  Building both would be two switches on one b16. */
  static_assert(!(ASA && ASAL), "pick one aligned-block store form");
  static_assert(!(CSB && CSBODD), "CSBODD is CSB restricted to one instantiation");
  /* NOGATE is wired into the promoted flat epilogue only; the other modes still
   * read the staged vector, so combining them would silently drop the gate. */
  static_assert(!NOGATE || EPIM == 2, "NOGATE needs the flat epilogue");
  static_assert(!CCSEAM || ASAL, "the chunk-boundary seam is written inside ASAL's loop");
  /* ASAL's own ladders fire only at cc == 0 and cc == CR-1, and its aligned blocks
   * are m = LPR*cc + p for p = 1..LPR-1, so the boundary blocks m = LPR*cc for
   * cc = 1..CR-2 have no writer at all unless CCSEAM supplies them.  At CR == 2
   * there is no interior cc, and the two ladders that do fire are exactly the two
   * that cover the single boundary, which is why the seam-less form was complete
   * for every slot ever built with it - all of them LPR == 8.  experiments/
   * map_check.py proves the coverage both ways and counts the holes: 32 to 96
   * unwritten bytes per row at (LPR=4, CR=4) and (LPR=2, CR=8), i.e. a silently
   * wrong checkpoint rather than a slow one.  This makes the invariant that was
   * only ever incidental to the table explicit. */
  static_assert(!ASAL || CCSEAM || (16 / LPR) == 2,
                "ASAL without CCSEAM covers a row's interior chunk boundaries "
                "only at CR == 2 (LPR == 8)");
  /* ROWSEAM writes the block that straddles a ROW boundary, which needs both a
   * previous row's last window and the next row's first; CCSEAM already owns the
   * chunk-boundary block and its ladder bookkeeping, and the row seam removes the
   * two ladders CCSEAM leaves, so one without the other is not a legal pair.
   * CR >= 2 is what makes the two seam sites distinct iterations of the chunk
   * loop; at CR == 1 a row is one chunk and there is no interior to hand over. */
  static_assert(!ROWSEAM || (CCSEAM && ASAL), "the row seam extends CCSEAM's coverage");
  static_assert(!ROWSEAM || (16 / LPR) >= 2, "the row seam needs two chunks per row");
  /* LDB batches the tile load, and the tile ladder's wide arm is the only one
   * that reads a whole chunk as one uint4 - so the batch exists only in the
   * VEC=true instantiation.  VEC=false keeps the ladder, which means an LDB
   * slot's odd-padding code is byte-identical to its control's. */
  /* The divisor is spelled `LDB ? LDB : 1` rather than `LDB`, which looks
   * redundant next to the `LDB == 0 ||` guard and is not.  `||` short-circuits at
   * run time, but nvcc folds the constant subexpression to form its diagnostic and
   * so every build logged `warning #179-D: right operand of "%" is zero` for the
   * LDB=0 instantiations -- which is every dispatched slot, since `launch_slot`
   * passes LDB as a literal 0.  The assert's meaning is unchanged (at LDB=0 the
   * left disjunct already decides it) and there is no codegen, this being a
   * compile-time check; the point is the same as the `if constexpr` below it, that
   * a build log carrying a permanent warning cannot show a new one. */
  static_assert(LDB == 0 || (LDB <= C && C % (LDB ? LDB : 1) == 0),
                "the tile batch must divide the thread's chunk count");
  static_assert(!NWE || EPIM == 2, "NWE is wired into the flat epilogue only");
  /* The LPT remap reassigns the request index taken from blockIdx.y, so it needs
   * the 2-D launch that makes that index the request and nothing else. */
  static_assert(!LPT || GRID2D, "LPT needs the 2-D grid");
  static_assert(LPT2W == 0 || GRID2D, "LPT2 needs the 2-D grid");
  static_assert(!(LPT && LPT2W) && !(LPTW && LPT2W), "one dispatch remap");
  static_assert(LPTW == 0 || GRID2D, "LPTW needs the 2-D grid");
  static_assert(!(LPT && LPTW), "pick one dispatch-remap form");
  static_assert(CM < 4, "CM is two hint bits, or -1 to follow CS");
  static_assert(!KQREG || !QK32, "KQREG caches packed BF16 chunk words; QK32 has none");
  static_assert(!KQREG || CR >= 1, "KQREG batches the step's chunk loads");
  /* RPERM's two vector ladders cover R = 1, 2 and 4 and no other value, and both
   * terminate in an `else` that handles exactly R = 1: the v load does
   * `vm[0] = b2f(*vp)` and the rawsh store does `rp[0] = raw[0]`.  At R > 4 those
   * else-branches compile, run, and silently leave vm[1..R-1] uninitialised and
   * raw[1..R-1] unwritten -- a wrong `out` with no diagnostic, which is the same
   * failure shape as the nwe/RPW index defect fixed by GDN_EPI_ROW.
   *
   * R = C*LPR/16 with C = DR/(8*P) and DR = D/NH, so R > 4 is reachable at exactly
   * one point in the legal grid: NH=1 (RSPLIT off), P=2 and LPR=16 give C=8 and R=8.
   * No slot in the table has that combination today -- the widest R instantiated is
   * 4, at slot 61 -- so this is latent rather than live, and it is pinned because the
   * wiring that keeps it unreachable is a coincidence of the table, not a constraint
   * anyone wrote down.  The non-RPERM path needs no bound: it loops `for i < R`. */
  static_assert(!RPERM || R <= 4,
                "RPERM's vector ladders cover R = 1, 2, 4 only; R = 8 (RSPLIT off, P=2, "
                "LPR=16) would silently drop vm[1..7] and raw[1..7]");
  static_assert(!PLOV || LDB == 0, "PLOV batches the tile itself; LDB is dead");
  /* CSBODD: the constant-stride row base, but ONLY in the VEC=false instantiation.
   *
   * CSB is rejected as a global change and the rejection is now mechanistic rather
   * than statistical.  `vec` is true exactly when the state pointer and stride(0)
   * and stride(1) are 16-byte aligned, so VEC=true is what all ten fixed shapes and
   * the seven aligned domain configurations run - and in that instantiation CSB
   * CHANGES the code (2,048 static instructions against slot 22's 2,080, with a
   * different IMAD/IADD3/LEA/SHF mix, because the folded row base is recomputed per
   * access) while ASA and ASAL leave it byte-for-byte identical.  That is why slot
   * 23, which contains no assembly at all, is +5.6% on d4 and +1.7..+2.4% on
   * d3/d8/d9: the seven aligned configurations pay for a fold they get nothing from.
   *
   * In the VEC=false instantiation the same fold is worth having, because that is
   * the only one where the misaligned store path exists and its registers are the
   * binding constraint - slot 26 is at 128 of the 128 MINB=2 allows with 0 spills,
   * and the seam work in the handoff needs 3-5 more.  CSB frees 4 there (slot 22
   * 126 -> slot 23 122, both VEC=false).  VEC is a template parameter, so gating on
   * it is not a runtime branch: CSBE is a compile-time constant and the VEC=true
   * instantiation of a CSBODD slot is the SAME CODE as without it, which the SASS
   * dump checks directly (opcode_delta prev default must come out empty).
   *
   * Legality is unchanged from CSB: ROWOFF = RPG*D is the element distance between
   * this thread's rows because the host entry point TORCH_CHECKs
   * state.stride(3) == 1 && state.stride(2) == D, so it holds for every legal input
   * and not merely for the frozen ones. */
  constexpr bool CSBE = CSB || (CSBODD && !VEC);
  /* ASAL needs no other flag: its edge ladders are st8u, which is correct (if
   * narrower) with or without ST4, and the read side is independent.  It is only
   * ever interesting alongside ALD and ST4, which is what the table builds. */

  const bf16* const qkv = prm.qkv; const int64_t qs0 = prm.qs0;
  const int* const idxp = prm.idxp; const int64_t is0 = prm.is0;
  const int* const cup = prm.cup; const int* const accp = prm.accp;
  bf16* const state = prm.state;
  const int64_t ss = prm.ss, sh = prm.sh, sr = prm.sr;
  const int hv = prm.hv, width = prm.width;

  const int tid = threadIdx.x;
  /* GRID2D launches dim3(hv, nreq) instead of dim3(nreq*hv).  The linear dispatch
   * order is identical - block (x, y) is linearised as x + y*gridDim.x, which is
   * exactly bid = r*hv + j with j fastest - so nothing about locality, the tail or
   * which slot a block touches moves.  What disappears is the signed division and
   * remainder by the runtime value hv at the top of every block: PC sampling on
   * fixed case 0 puts the IABS of that sequence alone at 5.2% of all stall samples
   * and the whole chain is on the critical path to the block's first global load,
   * because r indexes cu_seqlens, num_accepted_tokens and state_indices. */
  const int j = GRID2D ? (int)blockIdx.x : (int)(blockIdx.x % (unsigned)hv);
  /* LPT reassigns this below; every other slot keeps the dispatch identity. */
  int r = GRID2D ? (int)blockIdx.y : (int)(blockIdx.x / (unsigned)hv);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int p = lane % LPR;      // column part inside a state row
  const int rs = lane / LPR;     // which of the warp's RPG rows this lane takes

  /* NWE: norm_weight[m] is the one value the flat epilogue reads that does not
   * depend on anything the block computes - not on the request, the slot, the
   * token count or the recurrence.  Left where the epilogue reads it, it is a
   * global load issued AFTER the last step, i.e. one exposed round trip with
   * nothing left to overlap it with, at the very end of every block's life.
   * Issued here it joins the block's first flight (cu_seqlens, num_accepted,
   * A_log, dt_bias) and costs one register.  Hoisting cannot change the VALUE, but
   * the ADDRESS is the one thing that has to be spelled right, and it was not: this
   * read `warp*RPW + lane%RPW` and the comment asserted that was "the flat
   * epilogue's row".  It is not - the flat epilogue's row is `warp*RPE + lane%RPE`,
   * and RPW == RPE only when NH == 1.  Under RSPLIT (NH=2) RPW is 8 and RPE is 16,
   * so this read half the norm_weight rows and repeated them: a wrong `out` on
   * every cell, silently, with no static_assert able to see it because both
   * spellings typecheck.  It was unreachable only by accident of wiring -
   * launch_slot passes NWE as a literal `false`, so the wide table cannot express
   * the field - and turning that literal into a table field would have made it
   * live.  Both sites now use GDN_EPI_ROW, which is the whole point of the macro. */
  float nwe_wm = 0.0f;
  if constexpr (NWE) {
    nwe_wm = b2f(prm.nwp[GDN_EPI_ROW(warp, lane)]);
  }

  extern __shared__ __align__(16) char smem[];
  char* const qsh = smem + A::Q_OFF;
  char* const ksh = smem + A::K_OFF;
  bf16* const vsh = reinterpret_cast<bf16*>(smem + A::V_OFF);
  bf16* const gsh = reinterpret_cast<bf16*>(smem + A::G_OFF);
  int* const dstsh = reinterpret_cast<int*>(smem + A::DST_OFF);
  float (*const nrm)[3] = reinterpret_cast<float (*)[3]>(smem + A::NRM_OFF);
  float (*const dcy)[2] = reinterpret_cast<float (*)[2]>(smem + A::DCY_OFF);
  char* const recsh = smem + A::REC_OFF;
  float (*const rmsp)[NW * NH] =
      reinterpret_cast<float (*)[NW * NH]>(smem + A::RMSP_OFF);
  float* const rawsh = reinterpret_cast<float*>(smem + A::RAW_OFF);

  /* LPT: dispatch the requests longest-first.
   *
   * The block scheduler hands slots the next block in linear order, so with one
   * block per (request, head) the grid runs as a greedy list schedule in request
   * order.  The frozen range allows per-request token counts anywhere in 0..8, and
   * when they differ the schedule is badly balanced: NCU on fixed case 4 (counts
   * 8/1/2/5, 512 blocks, 1.73 waves) reports SM Active Cycles 32631 of 53095
   * elapsed, i.e. 39% of the machine idle in the tail, and a greedy-schedule
   * simulation fitted to the same job's wave counts (F=3.68 us per block, S=1.64
   * us per step, from cases 1 and 2) prices longest-first at -29.6% on case 4,
   * -21.1% on cases 6 and 8, -11.1% on case 7, -5.6% on case 3 and -6.1% on case 9.
   *
   * The remap is computed here, per block, from cu_seqlens alone - no host read
   * back, no second kernel, no scratch buffer and nothing persisted between
   * calls.  Counts are bounded by MAXS, so a nine-bin counting sort is enough:
   * one thread per request classifies it, a shared histogram gives each class a
   * descending base, and one ballot scan finds which request owns this block's
   * position.  The order is stable (requests keep their relative order inside a
   * class), so every block derives the same permutation and the map stays a
   * bijection; the work of a block is unchanged, only which slot runs it, so the
   * results are bit-identical to the identity dispatch.
   *
   * It is skipped, without any barrier, when it cannot pay: a grid that fits in
   * one wave starts every block together, so no order changes the makespan, and a
   * single request has nothing to reorder.  A uniform count histogram - cases 0,
   * 1 and 2 of the fixed ten - also short-circuits before the scan, because the
   * sorted order then IS the dispatch order.  nreq above one thread per request
   * keeps the identity mapping: outside the frozen range (which bounds `lengths`
   * at 256 entries and the block at 256 threads) and still correct. */
  if constexpr (LPT) {
    const int nreq = prm.nreq;
    if (nreq > 1 && nreq <= A::NT &&
        (int64_t)nreq * hv > (int64_t)SM_COUNT * MINB) {
      int* const hist = reinterpret_cast<int*>(smem + A::HIST_OFF);
      int* const base = reinterpret_cast<int*>(smem + A::BASE_OFF);
      int* const wtot = reinterpret_cast<int*>(smem + A::WTOT_OFF);
      int* const found = reinterpret_cast<int*>(smem + A::FOUND_OFF);
      const int treq = tid;
      /* -1, not 0: threads past the end of the request table own no request and
       * must not be mistaken for members of the inactive class, which is the one
       * class a scan can legitimately select. */
      int myc = -1;
      if (treq < nreq) {
        const int c = cup[treq + 1] - cup[treq];
        myc = (c > 0 && c <= MAXS) ? c : 0;
      }
      if (tid <= MAXS) hist[tid] = 0;
      if (tid == 0) *found = -1;
      __syncthreads();
      if (treq < nreq) atomicAdd(&hist[myc], 1);
      __syncthreads();
      if (tid == 0) {
        int acc = 0;
#pragma unroll
        for (int c = MAXS; c >= 0; --c) { base[c] = acc; acc += hist[c]; }
      }
      __syncthreads();
      const int pos = r;
      int cls = -1, want = 0, ncls = 0;
#pragma unroll
      for (int c = MAXS; c >= 0; --c) {
        if (hist[c] > 0) ++ncls;
        if (cls < 0 && pos >= base[c] && pos < base[c] + hist[c]) {
          cls = c;
          want = pos - base[c];
        }
      }
      if (ncls > 1 && cls >= 0) {
        const bool m = (treq < nreq) && (myc == cls);
        const unsigned bal = __ballot_sync(0xffffffffu, m);
        if (lane == 0) wtot[warp] = (int)__popc(bal);
        __syncthreads();
        int pre = (int)__popc(bal & ((1u << lane) - 1u));
        for (int w = 0; w < warp; ++w) pre += wtot[w];
        if (m && pre == want) *found = treq;
        __syncthreads();
        if (*found >= 0) r = *found;
      }
    }
  }

  /* LPTW: the same longest-first remap as LPT, but warp-redundant - every warp
   * derives the permutation entry it needs from registers, with no shared memory
   * and no barrier at all.
   *
   * Why a second implementation.  The barrier form measured a per-block cost of
   * about 0.64 us (slot 227 on domain d2: -6.2% over 13.8 waves, and d2's counts
   * are uniform so the scan short-circuits and only the histogram path runs), and
   * that cost is what turned a modelled +29.6% on fixed case 4 into a measured
   * +9.2% and a modelled +21.1% on cases 6/8 into +3.6%/-1.4%.  Five barriers and
   * a serial nine-step prefix in one thread are expensive at the very point the
   * block is trying to issue its first global loads.  Here the count array load
   * REPLACES the cup[r]/cup[r+1] load the identity path already issues - same
   * round trip, 1 KB instead of 8 B, and the L1 serves the other seven warps - and
   * the ranking is 45 shuffle adds and eight ballots, about 300 issue cycles with
   * nothing to wait for.
   *
   * LPTW is the maximum number of waves the remap is allowed to run at, because
   * the benefit and the cost scale differently: the tail imbalance it removes is a
   * fraction of ONE wave, while the ranking is paid by EVERY wave.  The measured
   * crossover on this device is between 1.73 waves (fixed 4/6/8, +3.6..+9.2%) and
   * 3.46 waves (fixed 3/7, -1.9%/-2.9%), so one slot carries the bound and one
   * does not, and the measurement decides.  Zero disables it. */
  if constexpr (LPTW > 0) {
    const int nreq = prm.nreq;
    const int64_t cap = (int64_t)SM_COUNT * MINB;
    const int64_t blocks = (int64_t)nreq * hv;
    if (nreq > 1 && nreq <= 8 * 32 && blocks > cap &&
        blocks <= (int64_t)LPTW * cap) {
      int lo[8], hi[8], cls[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int t = lane + 32 * k;
        lo[k] = (t < nreq) ? cup[t] : 0;
        hi[k] = (t < nreq) ? cup[t + 1] : 0;
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int t = lane + 32 * k;
        if (t < nreq) {
          const int c = hi[k] - lo[k];
          cls[k] = (c > 0 && c <= MAXS) ? c : 0;
        } else {
          cls[k] = -1;
        }
      }
      int hist[MAXS + 1];
      int ncls = 0;
#pragma unroll
      for (int c = 0; c <= MAXS; ++c) {
        int local = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) local += (cls[k] == c);
#pragma unroll
        for (int off = 16; off; off >>= 1)
          local += __shfl_xor_sync(0xffffffffu, local, off);
        hist[c] = local;
        ncls += (local > 0);
      }
      /* One non-empty class: the sorted order is the dispatch order. */
      if (ncls > 1) {
        const int pos = r;
        int run_base = 0, sel_cls = -1, want = 0;
#pragma unroll
        for (int c = MAXS; c >= 0; --c) {
          if (sel_cls < 0 && pos < run_base + hist[c]) {
            sel_cls = c;
            want = pos - run_base;
          }
          run_base += hist[c];
        }
        if (sel_cls >= 0) {
          int run = 0, sel = -1;
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            /* Warp-collective by construction: sel_cls, want, run and bal are the
             * same in every lane, so all 32 lanes run all eight ballots. */
            const unsigned bal = __ballot_sync(0xffffffffu, cls[k] == sel_cls);
            const int cnt = (int)__popc(bal);
            if (sel < 0 && run + cnt > want) {
              unsigned m = bal;
              for (int i = run; i < want; ++i) m &= m - 1;
              const int b = __ffs(m) - 1;
              if (b >= 0) sel = b + 32 * k;
            }
            run += cnt;
          }
          if (sel >= 0 && sel < nreq) r = sel;
        }
      }
    }
  }

  /* LPT2: the longest-first remap with two barriers instead of five.
   *
   * LPT works where the schedule model says it should - five interleaved rounds
   * put it at +10.2% on fixed case 4, +5.5% on case 8 and +2.7% on case 6, and
   * +7.4% on domain d1 - and loses everywhere else for one reason: it costs about
   * 1.0 us per block, which domain d4 and d8 (uniform counts, so the scan never
   * even runs) pay as -5.6% and -4.8%.  The cost is five block barriers and a
   * shared-atomic histogram, not the sort.
   *
   * This form keeps the identical permutation and replaces both:
   *   - nine ballots give a warp its per-class counts, so the histogram needs no
   *     atomics and no zeroing pass - lanes 0..8 simply store one int each;
   *   - one barrier publishes the [class][warp] table, nine shuffles broadcast
   *     its column sums to every lane, and the rank of this thread's own request
   *     is the table's row prefix plus one more ballot, so a second barrier is
   *     enough to hand the winner to the block.
   * Ten ballots, nine shuffles, eight shared reads and two barriers - about 400
   * cycles against LPT's five barriers and its serialized atomic histogram.
   *
   * The wave bound is part of the mechanism, not a tuning knob: the imbalance it
   * removes is a fraction of ONE wave's makespan while the ranking is paid by
   * EVERY wave, and the measurement puts the crossover between 1.73 waves (fixed
   * 4/6/8 win) and 3.46 (fixed 3/7 lose).  Below one wave every block starts
   * together and no order changes the makespan; above two, greedy scheduling has
   * already averaged the imbalance out.  `nreq` beyond one request per thread
   * keeps the identity mapping, outside the frozen range (which bounds `lengths`
   * at 256 entries) and still correct.  experiments/lpt_check.py proves the
   * permutation and every short-circuit.
   *
   * LOWER BOUND: `blocks > cap`, i.e. more than one wave of THIS slot's resident
   * capacity.  On 2026-10-28 this was briefly lowered to `blocks > SM_COUNT` on
   * the argument that an SM hosting two or more blocks in a single wave is still
   * ordered by the dispatch, so the remap should reach single-wave grids too.
   * Measured, that argument is wrong in the only direction that matters
   * (dv_e2d8c1ba2b29, 30 samples x 3 interleaved rounds, arms sign-consistent):
   * at 512 blocks on slot 71's P=1 tuple the remap fires under the lowered guard
   * and costs +1.97 us on fixed:4 (28.624 against 26.656, 0/3 rounds better) and
   * +0.03 us on fixed:6, while at 1024 blocks on the same tuple it GAINS -1.14 us
   * on fixed:3 (30.720 against 31.856, 3/3 rounds) and -2.00 us on fixed:7
   * (30.768 against 32.768, 3/3).  512 blocks is 0.86 waves at cap 592 and 1024 is
   * 1.73, so the crossover really is one wave: below it the hardware has every
   * block resident at once and re-ordering them only adds the histogram, above it
   * the order decides which requests share the second wave.  The original bound is
   * restored and slot 65's historical meaning with it.
   *
   * The cost being recovered is now priced directly rather than argued: on a shape
   * whose counts are UNIFORM, so the permutation short-circuits at ncls==1 and only
   * the cu_seqlens histogram and its two barriers execute, slot 77 against slot 71
   * is +2.048 us exactly (fixed:0, 17.408 against 15.360, three rounds identical).
   * That is an order of magnitude above the ~400 cycles this block's own comment
   * estimated, because the histogram is a cold DRAM read of all of cu_seqlens
   * placed in series BEFORE the block may issue its own metadata loads.  Any shape
   * the remap is asked to fix therefore has to be worth more than 2 us of
   * imbalance, which is why the bound is a wave count and not a shape test. */
  if constexpr (LPT2W > 0) {
    const int nreq = prm.nreq;
    const int64_t cap = (int64_t)SM_COUNT * MINB;
    const int64_t blocks = (int64_t)nreq * hv;
    if (nreq > 1 && nreq <= A::NT && blocks > cap &&
        blocks <= (int64_t)LPT2W * cap) {
      int* const tab = reinterpret_cast<int*>(smem + A::TAB_OFF);
      int* const found = reinterpret_cast<int*>(smem + A::FOUND2_OFF);
      int myc = -1;
      if (tid < nreq) {
        const int c = cup[tid + 1] - cup[tid];
        myc = (c > 0 && c <= MAXS) ? c : 0;
      }
      int mine = 0;
#pragma unroll
      for (int c = 0; c <= MAXS; ++c) {
        const int pc = (int)__popc(__ballot_sync(0xffffffffu, myc == c));
        if (lane == c) mine = pc;
      }
      if (lane <= MAXS) tab[lane * NW + warp] = mine;
      if (tid == 0) *found = -1;
      __syncthreads();
      int col = 0;
      if (lane <= MAXS) {
#pragma unroll
        for (int w = 0; w < NW; ++w) col += tab[lane * NW + w];
      }
      int hist[MAXS + 1];
#pragma unroll
      for (int c = 0; c <= MAXS; ++c)
        hist[c] = __shfl_sync(0xffffffffu, col, c);
      const int pos = r;
      int run = 0, cls = -1, want = 0, ncls = 0;
#pragma unroll
      for (int c = MAXS; c >= 0; --c) {
        ncls += (hist[c] > 0);
        if (cls < 0 && pos < run + hist[c]) {
          cls = c;
          want = pos - run;
        }
        run += hist[c];
      }
      if (ncls > 1 && cls >= 0) {
        const bool m = (myc == cls);
        const unsigned bal = __ballot_sync(0xffffffffu, m);
        int pre = (int)__popc(bal & ((1u << lane) - 1u));
        if (m) {
#pragma unroll
          for (int w = 0; w < NW; ++w)
            pre += (w < warp) ? tab[cls * NW + w] : 0;
        }
        if (m && pre == want) *found = tid;
        __syncthreads();
        if (*found >= 0) r = *found;
      }
    }
  }

  /* IDXPAR issues the whole state_indices row alongside cu_seqlens and
   * num_accepted_tokens.  Without it the block's first THREE memory round trips
   * are strictly serial - cu/acc, then the source slot (its index comes from acc),
   * then the 32 KiB source tile (its address comes from the slot) - and PC
   * sampling on fixed case 0 (job dv_d7db602e8e4b, warm caches) puts that startup
   * chain, not the arithmetic, among the hottest stall sites of a kernel that sits
   * at 20% of every throughput ceiling.  The row is at most MAXS ints inside one
   * or two cache lines and every lane wants the same values, so the speculation
   * costs MAXS uniform loads once per block and removes a whole round trip from
   * the critical path.  A row wider than MAXS is outside the frozen range but
   * stays correct: the accepted index then falls back to the dependent load. */
  int row[MAXS];
  if constexpr (IDXPAR) {
#pragma unroll
    for (int k = 0; k < MAXS; ++k)
      row[k] = (k < width) ? idxp[(int64_t)r * is0 + k] : 0;
  }
  const int c0 = cup[r];
  const int count = cup[r + 1] - c0;
  const int acc = accp[r];
  bool active = (count > 0) && (count <= MAXS) && (acc > 0) && (acc <= width);
  int src = 0;
  if (active) {
    if constexpr (IDXPAR) {
      src = (acc <= MAXS) ? row[acc - 1] : idxp[(int64_t)r * is0 + (acc - 1)];
    } else {
      src = idxp[(int64_t)r * is0 + (acc - 1)];
    }
    active = (src > 0);
  }
  /* The caller-visible inactive path: zero this request's out rows and touch
   * nothing else.  A lambda so HOIST can move the branch without duplicating it.
   * NOUNROLL keeps it a real loop: `count <= MAXS` bounds it, and ptxas turned the
   * unrolled form into 382 static instructions of reconvergence blocks that no
   * scored shape executes but that sit between the kernel entry and the first
   * tile load in the instruction stream. */
  auto zero_out = [&]() {
    if constexpr (NOUNROLL) {
#pragma unroll 1
      for (int t = c0; t < c0 + count; ++t) {
        bf16* o = prm.outp + (int64_t)t * prm.os0 + (int64_t)j * D;
        if (tid < D) o[tid] = __float2bfloat16(0.0f);
      }
    } else {
      for (int t = c0; t < c0 + count; ++t) {
        bf16* o = prm.outp + (int64_t)t * prm.os0 + (int64_t)j * D;
        if (tid < D) o[tid] = __float2bfloat16(0.0f);
      }
    }
  };
  if constexpr (!HOIST) {
    if (!active) {
      zero_out();
      return;
    }
  }
  /* HOIST defers the inactive early-out until the source tile is already in the
   * air.  PC sampling on fixed case 0 (dv_8af4fd041617) puts 83 executed
   * instructions between the ISETP that consumes state_indices[...] and the eight
   * LDG.E.EF.128s that fetch the tile - the head index, the four row bases, the
   * four projection pointers, A_log and dt_bias - none of which depends on the
   * slot, but all of which the branch to the inactive path sits textually in
   * front of.  So the warp stalls on the index load, then issues eighty-odd
   * instructions, and only then asks for the 32 KiB it actually waits for: the
   * first consumer of that tile holds 103 long_scoreboard samples, 15% of every
   * stall sample in the kernel.  Moving the branch behind the load issue lets the
   * address setup run under the index load's latency instead of after it.
   *
   * An inactive block then issues one speculative tile read, which is why the slot
   * is clamped to 0 rather than used raw: slot 0 always exists (the frozen range
   * bounds pool_size at 2 and the contract reserves slot 0), so the read is inside
   * the caller's storage for every legal input, a negative slot included, and the
   * widened result is discarded by the branch below.  An active block has
   * src > 0, so sc == src and the addresses are unchanged. */
  const int sc = HOIST ? (src > 0 ? src : 0) : src;

  /* HQFAST: the q/k head that this value head reads is j / (hv/h), and both
   * divisions are by runtime values, so the compiler emits its full signed-division
   * sequence - about twenty instructions, IABS included, on the block's startup
   * path before any address can be formed.  The value-head ratio is 2, 3 or 8
   * across the frozen range, so those three are matched by comparison and shifted
   * or divided directly; anything else keeps the general form, which is what makes
   * this legal for every hv that is a multiple of h and not just the sampled ones. */
  int hq;
  if constexpr (HQFAST) {
    const int h = prm.h;
    if (hv == 2 * h) {
      hq = j >> 1;
    } else if (hv == 8 * h) {
      hq = j >> 3;
    } else if (hv == 4 * h) {
      hq = j >> 2;
    } else if (hv == 3 * h) {
      hq = j / 3;
    } else if (hv == h) {
      hq = j;
    } else {
      hq = j / (hv / h);
    }
  } else {
    const int ratio = hv / prm.h;
    hq = j / ratio;
  }
  /* R folded base pointers, one per state row this thread owns; chunk cc of row
   * i is then a pure immediate offset from that row's base.
   *
   * CSB folds one level further.  This thread's rows are mr0 + i*RPG, and the host
   * entry point TORCH_CHECKs state.stride(2) == D, so under CSB row i's base is
   * sjb0 + i*RPG*D with RPG*D a COMPILE-TIME constant (512 elements = 1024 bytes at
   * LPR=8).  One base pointer plus constant offsets therefore replaces R base
   * pointers and R row indices, and every (row, chunk) pair becomes an immediate
   * offset of a single address.  That matters because in slot 12's SASS the chunk
   * offset IS already folded - STG and LDG immediates run 0x0..0xe and 0x80..0x8e
   * from one base register - but the row stride is NOT: the largest immediate in
   * the whole store phase is 142 bytes, not 1024, so each of the R=4 rows gets its
   * own base recomputed into a register on every step.  Address arithmetic is
   * 43.5% of case 0's PC samples and ~36% of case 1's and carries almost all of
   * the long_scoreboard; module-wide the opcode histogram has ~29,000 integer
   * address/logic instructions against ~17,500 floating-point ones.  The offsets
   * are multiples of 1024 bytes and hence of 16, so they cannot change the
   * alignment class ld8/st8 branch on and the ladder's switch stays block-uniform.
   * It should also free six registers (three 64-bit row bases) in a kernel that
   * sits at 126 of the 128 MINB=2 allows, which is now the binding constraint.
   *
   * With CSB=false the two accessors return exactly the array entries below and the
   * array is filled by exactly the expression it always was, so slots 0-20 are
   * unchanged in both semantics and code. */
  /* Not const: RSPLIT's second pass advances it by DR, along with mr[] and sjb[].
   * With NH == 1 nothing below ever runs and this is the same value it was. */
  int mr0 = warp * RPW + rs;
  bf16* const sjb0 = state + (int64_t)j * sh + (int64_t)mr0 * sr + p * 8;
  constexpr int64_t ROWOFF = (int64_t)RPG * (int64_t)D;  /* elements between rows */
  bf16* sjb[R];
  int mr[R];
#pragma unroll
  for (int i = 0; i < R; ++i) {
    /* Rows i*RPG + rs of the warp's RPW rows; for LPR == P this is the narrow
     * map's tid/P.  CROW (rs*R + i, consecutive rows so the per-row accesses
     * vectorise) was measured and rejected: -1.75% fixed / -1.57% domain against
     * its control, with the static SASS a wash. */
    mr[i] = mr0 + i * RPG;
    if constexpr (CSBE) sjb[i] = sjb0 + (int64_t)i * ROWOFF;
    else sjb[i] = state + (int64_t)j * sh + (int64_t)mr[i] * sr + p * 8;
  }
  auto rowbase = [&](int i) -> bf16* {
    if constexpr (CSBE) return sjb0 + (int64_t)i * ROWOFF;
    else return sjb[i];
  };
  auto rowidx = [&](int i) -> int {
    if constexpr (CSBE) return mr0 + i * RPG;
    else return mr[i];
  };
  const int64_t irow = (int64_t)r * is0;
  /* ld8_al reads the aligned 16-byte block that FOLLOWS the window, so a window
   * ending at the very last byte of the state storage would read up to 14 bytes
   * past it.  Only the final head of the final slot can be that window, and with an
   * odd state_slot_padding its address really is 2 mod 4 - so those blocks keep the
   * ladder.  One (slot, head) pair in the grid, decided from data the kernel already
   * holds, never from the shape, the case id or the benchmark phase. */
  const bool ald_ok = !(ALD && j == hv - 1 && sc == prm.pool - 1);

  /* ---- prologue: identical to the promoted kernel; the element->lane map does
   * not touch it, because warp `warp` materialises step `warp` whole. ---- */
  /* Per-round staging state, and why the split below is a duplication and
   * not a fold.
   *
   * SROUND == 1 at every dispatched slot (NW == MAXS == 8), so the `else` is the
   * live code at slots 39 and 58 and the strided branch is not instantiated
   * there at all: `if constexpr` discards it, which is a language guarantee and
   * not an optimiser's fold.  That distinction is the entire reason for the
   * duplication.  A uniform rewrite -- size-1 arrays, a one-iteration loop,
   * constant indices -- *should* give the same SASS, but "should" is not
   * checkable in a workspace with no compiler, and a shift in slot 58's codegen
   * would be invisible inside a dev sweep because ctrl_dup is the same binary.
   * It would also spend the property that lets the archived full-shape_range
   * screen transfer to HEAD (results/prologue-dupstore-20261026/ section 2:
   * nothing reachable at slot 39 or 58 changed against the screened tree
   * a78b6c17), and that property is worth more than sixty lines.
   *
   * The `else` is the pristine staging verbatim except for nine write targets
   * (`qu` -> `quv[0]` and so on).  The state cannot live inside the branch
   * because stage_step is called ~120 lines later down two different control
   * paths and a block-scoped copy would reach neither.  At SROUND == 1 every
   * array has exactly one element and every index is the constant 0, so the
   * rename moves no storage and no value.
   *
   * All SROUND rounds' loads are issued here, ahead of the source tile's flight,
   * so the staging overlaps it exactly as PLOV's comment describes at any
   * SROUND. */
  const int d0 = lane * 4;
  const float alog = prm.alogp[j];
  const float dtbf = b2f(prm.dtbp[j]);
  int s0v[SROUND], dstvv[SROUND];
  bool havev[SROUND];
  uint2 quv[SROUND], kuv[SROUND], vuv[SROUND], guv[SROUND];
  float avv[SROUND], bvv[SROUND];
  if constexpr (SROUND > 1) {
    /* Round-strided staging: warp `warp` takes steps warp, warp+NW, ... so
     * any NW covers MAXS.  Instantiated only where NW < MAXS, which today is
     * P=1 (slot 66) and nothing else. */
#pragma unroll
    for (int rd = 0; rd < SROUND; ++rd) {
      const int s0 = warp + rd * NW;
      const bool have = (s0 < count);
      const int t0 = c0 + s0;
      const bf16* const qr = qkv + (int64_t)hq * D + (int64_t)t0 * qs0 + d0;
      const bf16* const kr = qkv + (int64_t)(prm.h + hq) * D + (int64_t)t0 * qs0 + d0;
      const bf16* const vr = qkv + (int64_t)(2 * prm.h + j) * D + (int64_t)t0 * qs0 + d0;
      /* NOGATE drops the output-gate vector from the shared arena.  It is the one
       * staged vector the recurrence never touches - only the epilogue reads it, once
       * per (step, row) - so staging it costs a 16-byte global load and a 16-byte
       * shared store per warp on the block's critical path, to buy back a 2-byte
       * L2-warm load per output element after the recurrence is over. */
      const bf16* const gr = NOGATE
          ? prm.gatep
          : (prm.gatep + (int64_t)j * D + (int64_t)t0 * prm.gs0 + d0);
      s0v[rd] = s0;
      havev[rd] = have;
      quv[rd] = make_uint2(0u, 0u);
      kuv[rd] = make_uint2(0u, 0u);
      vuv[rd] = make_uint2(0u, 0u);
      guv[rd] = make_uint2(0u, 0u);
      avv[rd] = 0.0f;
      bvv[rd] = 0.0f;
      dstvv[rd] = 0;
      if (have) {
        quv[rd] = *reinterpret_cast<const uint2*>(qr);
        kuv[rd] = *reinterpret_cast<const uint2*>(kr);
        vuv[rd] = *reinterpret_cast<const uint2*>(vr);
        if constexpr (!NOGATE) guv[rd] = *reinterpret_cast<const uint2*>(gr);
        if (lane == 0) {
          avv[rd] = b2f(prm.ap[(int64_t)t0 * prm.as0 + j]);
          bvv[rd] = b2f(prm.bp[(int64_t)t0 * prm.bs0 + j]);
          /* `have` implies s0 < count <= MAXS, so with IDXPAR the prefetched row
           * already holds this step's destination slot.
           *
           * `if constexpr`, not the ternary this replaced.  `row` is declared
           * uninitialised a little above (`int row[MAXS];`) and filled only under
           * `if constexpr (IDXPAR)`,
           * so with IDXPAR off -- which is every dispatched slot, since `launch_slot`
           * passes it as a literal `false` -- the array is never written.  A runtime
           * ternary on a template constant folds correctly and reads nothing, but the
           * compiler still sees `row[s0]` and every build logged
           * `warning #549-D: variable "row" is used before its value is set`.  The
           * identical choice in the `src` computation above already used `if constexpr`;
           * this site was the one that did not, and the inconsistency is the defect.
           * Codegen is unchanged either way: the discarded branch is not instantiated
           * now, and was folded away before.  The reason to fix it is that a build log
           * with a permanent warning in it is a build log that cannot show a new one,
           * and this phase found a silent wrong-`out` defect that nothing flagged. */
          if constexpr (IDXPAR) { dstvv[rd] = row[s0]; }
          else { dstvv[rd] = idxp[irow + s0]; }
        }
      }
    }
  } else {
    const int s0 = warp;
    const bool have = (s0 < count);
    const int t0 = c0 + s0;
    const bf16* const qr = qkv + (int64_t)hq * D + (int64_t)t0 * qs0 + d0;
    const bf16* const kr = qkv + (int64_t)(prm.h + hq) * D + (int64_t)t0 * qs0 + d0;
    const bf16* const vr = qkv + (int64_t)(2 * prm.h + j) * D + (int64_t)t0 * qs0 + d0;
    /* NOGATE drops the output-gate vector from the shared arena.  It is the one
     * staged vector the recurrence never touches - only the epilogue reads it, once
     * per (step, row) - so staging it costs a 16-byte global load and a 16-byte
     * shared store per warp on the block's critical path, to buy back a 2-byte
     * L2-warm load per output element after the recurrence is over. */
    const bf16* const gr = NOGATE
        ? prm.gatep
        : (prm.gatep + (int64_t)j * D + (int64_t)t0 * prm.gs0 + d0);
    s0v[0] = s0;
    havev[0] = have;
    quv[0] = make_uint2(0u, 0u);
    kuv[0] = make_uint2(0u, 0u);
    vuv[0] = make_uint2(0u, 0u);
    guv[0] = make_uint2(0u, 0u);
    avv[0] = 0.0f;
    bvv[0] = 0.0f;
    dstvv[0] = 0;
    if (have) {
      quv[0] = *reinterpret_cast<const uint2*>(qr);
      kuv[0] = *reinterpret_cast<const uint2*>(kr);
      vuv[0] = *reinterpret_cast<const uint2*>(vr);
      if constexpr (!NOGATE) guv[0] = *reinterpret_cast<const uint2*>(gr);
      if (lane == 0) {
        avv[0] = b2f(prm.ap[(int64_t)t0 * prm.as0 + j]);
        bvv[0] = b2f(prm.bp[(int64_t)t0 * prm.bs0 + j]);
        /* `have` implies s0 < count <= MAXS, so with IDXPAR the prefetched row
         * already holds this step's destination slot.
         *
         * `if constexpr`, not the ternary this replaced.  `row` is declared
         * uninitialised a little above (`int row[MAXS];`) and filled only under
         * `if constexpr (IDXPAR)`,
         * so with IDXPAR off -- which is every dispatched slot, since `launch_slot`
         * passes it as a literal `false` -- the array is never written.  A runtime
         * ternary on a template constant folds correctly and reads nothing, but the
         * compiler still sees `row[s0]` and every build logged
         * `warning #549-D: variable "row" is used before its value is set`.  The
         * identical choice in the `src` computation above already used `if constexpr`;
         * this site was the one that did not, and the inconsistency is the defect.
         * Codegen is unchanged either way: the discarded branch is not instantiated
         * now, and was folded away before.  The reason to fix it is that a build log
         * with a permanent warning in it is a build log that cannot show a new one,
         * and this phase found a silent wrong-`out` defect that nothing flagged. */
        if constexpr (IDXPAR) { dstvv[0] = row[s0]; }
        else { dstvv[0] = idxp[irow + s0]; }
      }
    }
  }
  /* PLOV needs this as a callable: the step-staging work is what overlaps the
   * source tile's memory flight, so where it runs is the mechanism.  Textually
   * it is the promoted prologue verbatim, expf(A_log) included, so a slot
   * without PLOV calls it at exactly the point it has always been. */
  /* The nine parameters are exactly this round's staging state, named as the
   * captured locals were, so the body below is verbatim: it was not edited when
   * SROUND was introduced, only re-fed.  At SROUND == 1 the call site passes the
   * `[0]` elements, which are the pristine scalars under another name. */
  auto stage_step = [&](const int s0, const bool have, const uint2 qu,
                        const uint2 ku, const uint2 vu, const uint2 gu,
                        const float av, const float bv, const int dstv) {
    const float ealog = expf(alog);
    if (have) {
      if constexpr (!NOGATE) {
        *reinterpret_cast<uint2*>(gsh + s0 * D + d0) = gu;
      }
      if constexpr (RPERM) {
        /* Element d0+t of this step goes to permuted position rowperm(d0+t), so the
         * four half-words this lane holds scatter instead of forming one uint2: four
         * shared stores instead of one, ONCE per block, against six wavefronts saved
         * on every step. */
        const bf16* const ve = reinterpret_cast<const bf16*>(&vu);
  #pragma unroll
        for (int t = 0; t < 4; ++t) {
          /* d0+t is a GLOBAL row index in 0..D-1, but rowperm permutes only WITHIN one
           * warp's RPW rows - the step loop reads `vsh + step*D + warp*RPW + rs*R + i`,
           * so the warp-block index has to be carried through untouched.  Applying
           * rowperm to the global index instead maps all 128 rows into 0..RPG*R+D/RPW
           * and collides; dv_3c389b3acbb5's spot check caught exactly that, with
           * slots 22/23 failing 26/30 and 28/30 cells and all 22 other slots clean. */
          const int m = d0 + t;
          vsh[s0 * D + (m / RPW) * RPW + rowperm<R, RPG>(m % RPW)] = ve[t];
        }
      } else {
        *reinterpret_cast<uint2*>(vsh + s0 * D + d0) = vu;
      }
      *reinterpret_cast<uint2*>(gsh + s0 * D + d0) = gu;
      float qf[4], kf[4];
      widen4(&qu.x, qf);
      widen4(&ku.x, kf);
      if constexpr (QK32) {
        /* The four widened floats this lane already needed for the QK norms are the
         * four floats of logical group `lane`, so pre-widening costs no arithmetic
         * here - only a 16-byte store instead of an 8-byte one, at the swizzled
         * physical group offset. */
        const int phi = ((lane & 1) << 4) | (lane >> 1);
        *reinterpret_cast<float4*>(qsh + s0 * (D * 4) + 16 * phi) =
            make_float4(qf[0], qf[1], qf[2], qf[3]);
        *reinterpret_cast<float4*>(ksh + s0 * (D * 4) + 16 * phi) =
            make_float4(kf[0], kf[1], kf[2], kf[3]);
      } else {
        *reinterpret_cast<uint2*>(qsh + (s0 * D + d0) * 2) = qu;
        *reinterpret_cast<uint2*>(ksh + (s0 * D + d0) * 2) = ku;
      }
      float pq = 0.0f, pk = 0.0f, pd = 0.0f;
  #pragma unroll
      for (int i = 0; i < 4; ++i) {
        pq += qf[i] * qf[i];
        pk += kf[i] * kf[i];
        pd += qf[i] * kf[i];
      }
  #pragma unroll
      for (int off = 16; off; off >>= 1) {
        pq += __shfl_xor_sync(0xffffffffu, pq, off);
        pk += __shfl_xor_sync(0xffffffffu, pk, off);
        pd += __shfl_xor_sync(0xffffffffu, pd, off);
      }
      if (lane == 0) {
        const float rqs = rsqrtf(pq + QK_EPS) * prm.scale;
        const float rks = rsqrtf(pk + QK_EPS);
        const float kqd = pd * rqs * rks;
        const float x = av + dtbf;
        const float spl = (x > 20.0f) ? x : log1pf(expf(x));
        const float dec = expf(-ealog * spl);
        const float bta = sigf(bv);
        if constexpr (PACK) {
          rec_store(recsh + s0 * 32, rqs, rks, kqd, dec, bta, dstv);
        } else {
          nrm[s0][0] = rqs;
          nrm[s0][1] = rks;
          nrm[s0][2] = kqd;
          dcy[s0][0] = dec;
          dcy[s0][1] = bta;
          dstsh[s0] = dstv;
        }
      }
    }
  };

  int pf_acc = 0, pf_slot = 0;   /* PFW: next wave's accepted count and index row */
  int pf_c0 = 0, pf_cnt = 0;     /* PFV: next wave's first token and token count */
  float S[C][8];
  if constexpr (PLOV && VEC) {
    /* PLOV: issue the whole tile, stage the step, THEN widen.
     *
     * PC sampling on fixed case 0 (job dv_e8918ed3ffbf, 586 stall samples, slot
     * 207 == the promoted code) puts a third of every stall sample in the
     * prologue, on two adjacent instructions: `LOP3.LUT R75, R55, 0xffff0000`
     * with 103 long_scoreboard samples - the first consumer of the source tile,
     * i.e. the tile's memory flight - and `UMOV UR4, 0x400` with 89 barrier
     * samples, the first instruction after the prologue's __syncthreads(), i.e.
     * the warps waiting for the `count` warps that stage a step.  They are
     * SERIAL today: the ladder widens each uint4 as it lands, so the block pays
     * the tile round trip and then the staging compute and then the skew.
     *
     * Batching the loads alone (LDB8, slot 209) measured exactly nothing - the
     * stall profile is identical to the control's, 4.29 against 4.20
     * long_scoreboard, and the pipelined geomean over 40 cells is 1.001 - so
     * ptxas already had the eight loads in one flight.  What it cannot do on its
     * own is move the STAGING across them, because the staging is textually
     * after the widening.  Holding the eight uint4s and staging in between turns
     * tile_latency + staging into max(tile_latency, staging), which on case 0 is
     * worth up to the 89 barrier samples plus most of the 103.
     *
     * The uint4s are dead before the step loop's own peak, so the allocation
     * should not move; VEC=false keeps the ladder, so an odd-padding
     * instantiation of a PLOV slot is byte-identical to its control's. */
    uint4 u[C];
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const bf16* pa = rowbase(c / CR) + (int64_t)sc * ss + (c % CR) * 8 * LPR;
      const uint4* pu = reinterpret_cast<const uint4*>(pa);
      u[c] = CSLD ? __ldcs(pu) : *pu;
    }
    if constexpr (HOIST) {
      if (!active) {
        zero_out();
        return;
      }
    }
    if constexpr (SROUND > 1) {
#pragma unroll
      for (int rd = 0; rd < SROUND; ++rd)
        stage_step(s0v[rd], havev[rd], quv[rd], kuv[rd], vuv[rd], guv[rd],
                   avv[rd], bvv[rd], dstvv[rd]);
    } else {
      stage_step(s0v[0], havev[0], quv[0], kuv[0], vuv[0], guv[0],
                 avv[0], bvv[0], dstvv[0]);
    }
#pragma unroll
    for (int c = 0; c < C; ++c) up8(u[c], S[c]);
  } else {
    if constexpr (LDB != 0 && VEC) {
      /* LDB: issue the thread's whole share of the 32 KiB source tile as one
       * flight, then widen.  ld8() is a load/consume pair - it widens the uint4
       * it just read - so the scheduler is free to serialise the eight accesses
       * behind each other.  MEASURED (slot 209, job dv_e8918ed3ffbf): it does
       * not, and batching them changes nothing - 4.29 long_scoreboard against
       * the control's 4.20 and a pipelined geomean of 1.001 over 40 cells.
       * ptxas already had all eight in one flight; what it cannot do on its own
       * is move the step staging across them, which is what PLOV above does.
       * Kept as the control for that claim. */
#pragma unroll
      for (int g = 0; g < C / LDB; ++g) {
        uint4 u[LDB];
#pragma unroll
        for (int t = 0; t < LDB; ++t) {
          const int c = g * LDB + t;
          const bf16* pa =
              rowbase(c / CR) + (int64_t)sc * ss + (c % CR) * 8 * LPR;
          const uint4* pu = reinterpret_cast<const uint4*>(pa);
          u[t] = CSLD ? __ldcs(pu) : *pu;
        }
#pragma unroll
        for (int t = 0; t < LDB; ++t) up8(u[t], S[g * LDB + t]);
      }
    } else if constexpr (TSP && VEC) {
      /* TSP-P1 issue (prereg results/20261026T2300Z-tsp-p1-prereg.json):
       * replace the serial ld8-load/widen flight with C fire-and-forget
       * cp.async.cg copies of the SAME 16-byte windows into the thread's
       * private slice of the pass-1 staging image ([chunk][thread]*16B -
       * the archived TSTG2 layout: a warp touches a contiguous 512B per
       * chunk, so the consumption LDS.128 is conflict-free).  No register
       * holds the in-flight data - the property that makes this viable at
       * slot 66's 125/128 register cap where register-PLOV (+32) cannot
       * fit.  The qkv/scalar staging work below now overlaps the cold
       * tile flight instead of following it (today's order is SERIAL:
       * tile round-trip, then staging - the PLOV comment's own reading,
       * and the case6 profile prices the exposure at long_sb 1.69 top
       * class with 18.8% warp occupancy).  Unconditional issue mirrors
       * the ld8 loop it replaces. */
      char* const stage1 = smem + A::TSP_OFF;
      constexpr int NT = D * P;
#pragma unroll
      for (int c = 0; c < C; ++c)
        __pipeline_memcpy_async(stage1 + (c * NT + tid) * 16,
                                rowbase(c / CR) + (int64_t)sc * ss +
                                    (c % CR) * 8 * LPR,
                                16);
      __pipeline_commit();
    } else {
#pragma unroll
      for (int i = 0; i < R; ++i) {
        const bf16* sp = rowbase(i) + (int64_t)sc * ss;
#pragma unroll
        for (int cc = 0; cc < CR; ++cc)
          ld8<VEC, CSLD, ALD>(sp + cc * 8 * LPR, S[i * CR + cc], ald_ok);
      }
    }
    if constexpr (HOIST) {
      if (!active) {
        zero_out();
        return;
      }
    }
    if constexpr (PFW > 0) {
      /* Issue the two loads the next wave's prefetch address needs here, where a
       * flight is already in the air, instead of after the barrier where their
       * latency would be exposed with nothing left to overlap it.  `accp[pfr]` is
       * warp-uniform; the index row is one lane per entry, so both are a single
       * sector each and neither waits on the other. */
      const int pfr = r + PFW * prm.pfd;
      if (pfr < prm.nreq && width <= 32) {
        pf_acc = accp[pfr];
        if (pf_acc > 0 && pf_acc <= width && lane < pf_acc)
          pf_slot = idxp[(int64_t)pfr * is0 + lane];
        if constexpr (PFV) {
          /* Coverage fence.  The vector stage's benefit is proportional to the
           * fraction of blocks whose target exists - (nreq-pfd)/nreq, which the
           * host reports as pfvg >= 50% - while its cost is paid by every block,
           * so below that it is a net loss: three interleaved sweeps put the cells
           * at <=42% coverage between -0.4% and +1.8% and the cells at >=68%
           * between +1.4% and +4.9%, and the scored window measured the two
           * lowest-coverage multi-wave grids (d4 at 25%, d8 at 37.5%) 14-15%
           * slower.  Leaving pf_cnt at 0 makes the consumer below a no-op, since
           * ntok = 0 and no thread satisfies `tid < 8*ntok`, so this is the only
           * fence the mechanism needs.  cu_seqlens has nreq+1 entries and
           * pfr < nreq, so both reads are in bounds, and both arrays are a few
           * hundred bytes that every block of a wave re-reads - they are L2-warm
           * by the second wave, which is why the target's startup chain is not
           * worth warming and its staged vectors are. */
          if (prm.pfvg != 0) {
            pf_c0 = cup[pfr];
            pf_cnt = cup[pfr + 1] - pf_c0;
          }
        }
      }
    }
    if constexpr (SROUND > 1) {
#pragma unroll
      for (int rd = 0; rd < SROUND; ++rd)
        stage_step(s0v[rd], havev[rd], quv[rd], kuv[rd], vuv[rd], guv[rd],
                   avv[rd], bvv[rd], dstvv[rd]);
    } else {
      stage_step(s0v[0], havev[0], quv[0], kuv[0], vuv[0], guv[0],
                 avv[0], bvv[0], dstvv[0]);
    }
    if constexpr (TSP && VEC) {
      /* TSP-P1 consumption: the staging above (qkv loads, smem writes, the
       * norm/decay scalars) ran UNDER the cp.async flight; wait this
       * thread's own commit (private layout - no barrier needed for
       * visibility) and widen the SAME uint4 bytes through the SAME up8
       * the ld8 VEC path applies - bit-identical by construction, spot-
       * gated.  Placed BEFORE the prologue __syncthreads() so an early
       * thread converts its barrier wait into the unpack instead of
       * idling.  c = i*CR + cc is the issue loop's own bijection. */
      __pipeline_wait_prior(0);
      const char* const stage1 = smem + A::TSP_OFF;
      constexpr int NT = D * P;
#pragma unroll
      for (int c = 0; c < C; ++c)
        up8(*reinterpret_cast<const uint4*>(stage1 + (c * NT + tid) * 16),
            S[c]);
    }
  }

  __syncthreads();

  if constexpr (PFW > 0) {
    /* Warm L2 with the source tile of the block the scheduler starts about PFW
     * waves from now, so that block's 32 KiB read is an L2 hit instead of a DRAM
     * round trip.  256 threads x 128 bytes is exactly one 128x128 BF16 tile, and
     * with GRID2D the dispatch order is (head fastest, request slowest), so
     * block (j, r) warms (j, r + PFW*pfd) and every tile is warmed by exactly one
     * block: this moves reads earlier, it does not add any.
     *
     * Why it is worth trying where the persistent kernel was not: NCU on fixed
     * case 2 puts 33.0% of warp cycles on long_scoreboard and 21.1% on barrier
     * against 40.6% of DRAM and 24% occupancy, i.e. the block waits for a tile it
     * asked for one round trip ago and there are not enough warps to hide it,
     * while a third of the memory system is idle.  A prefetch needs no extra
     * registers to hold data, no shared buffer, no grid change and no host policy,
     * and it cannot change a result, so it is bit-identical by construction.
     * `scn > 0` skips the requests that take the inactive path and read nothing;
     * `scn < pool` keeps the address inside the caller's storage, so the prefetch
     * is in bounds for every legal input. */
    const int pfr = r + PFW * prm.pfd;
    if (pfr < prm.nreq && pf_acc > 0 && pf_acc <= 32) {
      const int scn = __shfl_sync(0xffffffffu, pf_slot, pf_acc - 1);
      if (scn > 0 && scn < prm.pool) {
        /* 2*D/P bytes per thread is exactly the D*D*2-byte tile over the D*P
         * threads of the block, so the warp covers the tile and nothing after
         * it: no prefetch can leave this (slot, head) pair's own storage.
         * (True at P=2.  At P=1 that stride is 256 bytes and covers half the
         * tile, which is what PFL's line loop below replaces.) */
        if constexpr (PFL) {
          const char* const pfb = reinterpret_cast<const char*>(
              state + (int64_t)j * sh + (int64_t)scn * ss);
#pragma unroll
          for (int t = tid; t < TILE_LINES; t += D * P) prefetch_l2(pfb + t * 128);
        } else {
          prefetch_l2(reinterpret_cast<const char*>(
                          state + (int64_t)j * sh + (int64_t)scn * ss) +
                      (int64_t)tid * (2 * D / P));
        }
        if constexpr (PFV) {
          /* The target's tile is not the only cold read in its prologue: the
           * `count` warps that stage a step each pull four 256-byte row slices
           * (q, k, v and the output gate) that no other block touches, and the
           * prologue barrier waits for them.  Warming the tile makes the tile
           * flight short, which is what exposes that wait.  8 lines of 128 bytes
           * per token is exactly the four slices, so the first 8*count threads
           * cover the target's whole staged input and nothing else.  `hq` is a
           * function of `j` alone, so this block already has the target's. */
          const int ntok = (pf_cnt > 0 && pf_cnt <= MAXS) ? pf_cnt : 0;
          if (tid < 8 * ntok) {
            const int sel = (tid >> 1) & 3;
            const int64_t t0 = (int64_t)(pf_c0 + (tid >> 3));
            const char* base;
            int64_t stride;
            if (sel == 0) {
              base = reinterpret_cast<const char*>(qkv + (int64_t)hq * D);
              stride = qs0 * 2;
            } else if (sel == 1) {
              base = reinterpret_cast<const char*>(qkv + (int64_t)(prm.h + hq) * D);
              stride = qs0 * 2;
            } else if (sel == 2) {
              base = reinterpret_cast<const char*>(qkv + (int64_t)(2 * prm.h + j) * D);
              stride = qs0 * 2;
            } else {
              base = reinterpret_cast<const char*>(prm.gatep + (int64_t)j * D);
              stride = prm.gs0 * 2;
            }
            prefetch_l2(base + t0 * stride + (int64_t)(tid & 1) * 128);
          }
        }
      }
      if constexpr (PF2 == 2) {
        /* Prologue-issued self pass-2 warm for the blocks PFT cannot reach:
         * a block WITH a next-wave target spends its prologue prefetch slot on
         * that target (PFW tile + PFV vectors), so its OWN rows [D/2, D) -
         * read back only after the whole pass-1 loop - stay cold unless some
         * other block's PFW happened to cover them.  Ported from arm a's
         * pre-join lineage (arm-7 dv_3d5cce982dfb: the f2 barrier-structure
         * win; f1 58.336/58.400 vs 58's 60.38 same-job), re-measured on this
         * base by the slot 58-vs-67 A/B.  Same bound and same bit-identity as
         * PFT: destinationless, own storage only, the reload still performs
         * its own coherency. */
        if constexpr (PFL) {
          if (active && sc > 0 && sc < prm.pool) {
            const char* const sfb = reinterpret_cast<const char*>(
                state + (int64_t)j * sh + (int64_t)sc * ss);
#pragma unroll
            for (int t = tid; t < PASS_LINES; t += D * P)
              prefetch_l2(sfb + (int64_t)(DR * D * 2) + t * 128);
          }
        } else if (active && sc > 0 && sc < prm.pool && tid < (D * P) / 2) {
          prefetch_l2(reinterpret_cast<const char*>(
                          state + (int64_t)j * sh + (int64_t)sc * ss) +
                      (int64_t)(D * D) + (int64_t)tid * (2 * D / P));
        }
      }
    } else if constexpr (PFT) {
      /* Tail self-prefetch: this block has no next-wave target to warm - the
       * last wave of a grid, or EVERY block of a single-wave grid, where
       * pfd > nreq makes the branch above dead - so it warms its own RSPLIT
       * pass-2 half-tile instead, turning pass 2's cold DRAM round trip (the
       * largest single long_sb site on the v2-cold f2/f0 pcsamp, dv_d331ce30238e)
       * into an L2 hit with the whole pass-1 loop as lead time.  Rows [D/2, D)
       * of the source slot are bytes [D*D, 2*D*D) of its tile; D*P/2 threads
       * x 2*D/P bytes cover exactly that half and stay inside this (slot,
       * head) pair's own storage, the same bound PFW's warm respects.  A
       * prefetch has no destination and writes no byte - the pass-2 load still
       * performs its own coherency - so results are bit-identical by
       * construction, the guarantee PFW itself rides.  Single-wave grids are
       * where this is wall-clock and not accounting: their resident blocks
       * advance in near-lockstep on uniform counts, so the MINB=3 absorption
       * that nulled the prologue-overlap arms has no phase diversity here. */
      if constexpr (PFL) {
        if (active && sc > 0 && sc < prm.pool) {
          const char* const sfb = reinterpret_cast<const char*>(
              state + (int64_t)j * sh + (int64_t)sc * ss);
#pragma unroll
          for (int t = tid; t < PASS_LINES; t += D * P)
            prefetch_l2(sfb + (int64_t)(DR * D * 2) + t * 128);
        }
      } else if (active && sc > 0 && sc < prm.pool && tid < (D * P) / 2) {
        prefetch_l2(reinterpret_cast<const char*>(
                        state + (int64_t)j * sh + (int64_t)sc * ss) +
                    (int64_t)(D * D) + (int64_t)tid * (2 * D / P));
      }
    }
  }

  /* RSPLIT: walk the state's rows in NH passes.  The step loop is inside, so each
   * pass runs the whole recurrence over its own DR rows and stores its own
   * checkpoints; the two passes share one staging pass, one prologue barrier and one
   * epilogue, and they are independent because the recurrence is row-local.  The
   * second pass reads rows [DR, D) of the source slot AFTER the first pass has
   * stored rows [0, DR) of the destination slots, which is safe: the row ranges are
   * disjoint, so no byte the second pass reads is a byte the first pass wrote, even
   * when a destination slot is the source slot.  (ALD's aligned-block read does touch
   * up to 14 bytes of the previous row, but those bytes are shifted out and never
   * used, which is what ld8_al's funnel shifts guarantee.) */
  if constexpr (CPA && VEC) {
    /* CPA-P2 issue (prereg results/20261026T1930Z-cpa-p2-prereg.json): every
     * thread copies the FOUR 16-byte global windows its own pass-2 reload will
     * read - the identical addresses the hb>0 ld8 flight computes, formed here
     * from the still-unadvanced mr[] plus DR - into its private slice of the
     * arena-appended staging image, [chunk][thread]*16B (the archived TSTG2
     * layout: a warp touches a contiguous 512B per chunk, so the consumption
     * LDS.128 is conflict-free and one base register plus immediates address
     * it).  Fire-and-forget: cp.async.cg needs no register for the in-flight
     * data, the wait happens at pass-2 entry with the whole first pass as lead
     * time, and the staging is per-thread so completion needs NO barrier.
     * Issued unconditionally - including by inactive blocks and invalid sc -
     * exactly mirroring the reload it replaces, whose reads the existing code
     * already proves addressable.  The VEC contract gives cp.async.cg its
     * required 16-byte source alignment (base, stride(0), stride(1) aligned;
     * row/chunk/lane offsets are multiples of 16B); the scalar twin keeps the
     * global reload path untouched. */
    char* const stage = smem + A::CPA_OFF;
    constexpr int NT = D * P;
#pragma unroll
    for (int c = 0; c < R * CR; ++c) {
      const bf16* const pa = state + (int64_t)j * sh +
                             (int64_t)(mr[c / CR] + DR) * sr + p * 8 +
                             (int64_t)sc * ss + (c % CR) * 8 * LPR;
      __pipeline_memcpy_async(stage + (c * NT + tid) * 16, pa, 16);
    }
    __pipeline_commit();
  }
  for (int hb = 0; hb < NH; ++hb) {
    if (hb > 0) {
      mr0 += DR;
#pragma unroll
      for (int i = 0; i < R; ++i) {
        mr[i] += DR;
        sjb[i] = state + (int64_t)j * sh + (int64_t)mr[i] * sr + p * 8;
      }
      if constexpr (CPA && VEC) {
        /* CPA-P2 consumption: wait this thread's own commit (per-thread, no
         * barrier) and widen the SAME uint4 bytes through the SAME up8 the
         * ld8 VEC path applies - bit-identical by construction, spot-gated.
         * The mr[]/sjb[] advance above stays: the checkpoint stores and the
         * epilogue read them exactly as before. */
        __pipeline_wait_prior(0);
        const char* const stage = smem + A::CPA_OFF;
        constexpr int NT = D * P;
#pragma unroll
        for (int c = 0; c < R * CR; ++c)
          up8(*reinterpret_cast<const uint4*>(stage + (c * NT + tid) * 16),
              S[c]);
      } else {
#pragma unroll
        for (int i = 0; i < R; ++i) {
          const bf16* sp = rowbase(i) + (int64_t)sc * ss;
#pragma unroll
          for (int cc = 0; cc < CR; ++cc)
            ld8<VEC, CSLD, ALD>(sp + cc * 8 * LPR, S[i * CR + cc], ald_ok);
        }
      }
    }
  for (int step = 0; step < count; ++step) {
    if constexpr (PF2 == 1) {
      /* Late self pass-2 warm, issued at pass-1's step 0 for EVERY active
       * block (tail blocks' PFT prologue warm is left alone; on them this
       * re-touches a line that is already resident and costs nothing).  Step 0
       * leaves the remaining count-1 steps, the pass-1 stores and the reload's
       * own address arithmetic as lead time.  Ported from arm a's pre-join
       * lineage with the PF2 axis (prereg
       * results/20261026T1445Z-pf2-port-to-peer-base-prereg.json); the
       * prologue form (PF2=2) is what slot 67 runs.  Destinationless and
       * own-storage-only like PFT/PFW: results stay bit-identical. */
      if constexpr (PFL) {
        if (hb == 0 && step == 0 && active && sc > 0 && sc < prm.pool) {
          const char* const sfb = reinterpret_cast<const char*>(
              state + (int64_t)j * sh + (int64_t)sc * ss);
#pragma unroll
          for (int t = tid; t < PASS_LINES; t += D * P)
            prefetch_l2(sfb + (int64_t)(DR * D * 2) + t * 128);
        }
      } else if (hb == 0 && step == 0 && active && sc > 0 && sc < prm.pool &&
                 tid < (D * P) / 2) {
        prefetch_l2(reinterpret_cast<const char*>(
                        state + (int64_t)j * sh + (int64_t)sc * ss) +
                    (int64_t)(D * D) + (int64_t)tid * (2 * D / P));
      }
    }
    float rqs, rks, kqd, g, beta;
    int dst_rec;
    if constexpr (PACK) {
      rec_load(recsh + step * 32, rqs, rks, kqd, g, beta, dst_rec);
    } else {
      rqs = nrm[step][0];
      rks = nrm[step][1];
      kqd = nrm[step][2];
      g = dcy[step][0];
      beta = dcy[step][1];
      dst_rec = dstsh[step];
    }
    /* With QK32 the step's FP32 vectors are 512 bytes and this lane's column part
     * starts at 16*p inside them; the chunk offsets below are the same immediates
     * either way, which is what the swizzle buys. */
    const char* const qstep = qsh + (QK32 ? (step * (D * 4) + 16 * p)
                                          : ((step * D + p * 8) * 2));
    const char* const kstep = ksh + (QK32 ? (step * (D * 4) + 16 * p)
                                          : ((step * D + p * 8) * 2));
    float vm[R];
    if constexpr (RPERM) {
      /* This thread's R rows are contiguous at rs*R.  The element offset
       * step*D + warp*RPW + rs*R times two bytes is a multiple of 2*R, so the
       * vector form is naturally aligned for every R this kernel instantiates. */
      const bf16* const vp = vsh + step * D + hb * DR + warp * RPW + rs * R;
      if constexpr (R == 4) {
        const uint2 vv = *reinterpret_cast<const uint2*>(vp);
        const bf16* const ve = reinterpret_cast<const bf16*>(&vv);
#pragma unroll
        for (int i = 0; i < 4; ++i) vm[i] = b2f(ve[i]);
      } else if constexpr (R == 2) {
        const unsigned vw = *reinterpret_cast<const unsigned*>(vp);
        const bf16* const ve = reinterpret_cast<const bf16*>(&vw);
#pragma unroll
        for (int i = 0; i < 2; ++i) vm[i] = b2f(ve[i]);
      } else {
        vm[0] = b2f(*vp);
      }
    } else {
#pragma unroll
      for (int i = 0; i < R; ++i) vm[i] = b2f(vsh[step * D + rowidx(i)]);
    }

    float2 ak[R][NA], aq[R][NA];
#pragma unroll
    for (int i = 0; i < R; ++i)
#pragma unroll
      for (int n = 0; n < NA; ++n) {
        ak[i][n] = make_float2(0.f, 0.f);
        aq[i][n] = make_float2(0.f, 0.f);
      }
    /* One shared chunk load feeds all R rows: CR loads instead of C. */
    /* KQREG: this step's k and q chunk words are read as ONE flight of 2*CR
     * shared accesses instead of CR load/use pairs, and the k words stay packed
     * in registers until the rank-1 update widens them again - the update's own
     * CR shared round trips disappear.  short_scoreboard is 1.53 of the 11.61
     * stalled warp-cycles per issued instruction on fixed case 0, and the six
     * chunk loads a step issues are what it waits for.  Cost is 4*CR registers
     * held across the reduction, which is why this is a slot and not a default.
     * The values are ld8sh's: up8() of the same uint4 from the same address. */
    uint4 kw[CR], qw[CR];
    if constexpr (KQREG) {
#pragma unroll
      for (int cc = 0; cc < CR; ++cc) {
        kw[cc] = *reinterpret_cast<const uint4*>(kstep + cc * 16 * LPR);
        qw[cc] = *reinterpret_cast<const uint4*>(qstep + cc * 16 * LPR);
      }
    }
#pragma unroll
    for (int cc = 0; cc < CR; ++cc) {
      float kc[8];
      if constexpr (KQREG) { up8(kw[cc], kc); }
      else if constexpr (QK32) { ld8f(kstep + cc * 16 * LPR, kc); }
      else { ld8sh(kstep + cc * 16 * LPR, kc); }
#pragma unroll
      for (int i = 0; i < R; ++i) acc8<X2>(S[i * CR + cc], kc, ak[i][cc % NA]);
    }
#pragma unroll
    for (int cc = 0; cc < CR; ++cc) {
      float qv[8];
      if constexpr (KQREG) { up8(qw[cc], qv); }
      else if constexpr (QK32) { ld8f(qstep + cc * 16 * LPR, qv); }
      else { ld8sh(qstep + cc * 16 * LPR, qv); }
#pragma unroll
      for (int i = 0; i < R; ++i) acc8<X2>(S[i * CR + cc], qv, aq[i][cc % NA]);
    }
    float acc_k[R], acc_q[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
      acc_k[i] = ak[i][0].x + ak[i][0].y;
      acc_q[i] = aq[i][0].x + aq[i][0].y;
#pragma unroll
      for (int n = 1; n < NA; ++n) {
        acc_k[i] += ak[i][n].x + ak[i][n].y;
        acc_q[i] += aq[i][n].x + aq[i][n].y;
      }
    }
    /* Decay the whole tile before the cross-lane reduction is consumed (SCHED). */
    if constexpr (X2) {
      const float2 gg = make_float2(g, g);
#pragma unroll
      for (int c = 0; c < C; ++c) {
#pragma unroll
        for (int t = 0; t < 8; t += 2) {
          const float2 ns = mul2(make_float2(S[c][t], S[c][t + 1]), gg);
          S[c][t] = ns.x;
          S[c][t + 1] = ns.y;
        }
      }
    } else {
#pragma unroll
      for (int c = 0; c < C; ++c) {
#pragma unroll
        for (int t = 0; t < 8; ++t) S[c][t] = g * S[c][t];
      }
    }
#pragma unroll
    for (int off = LPR / 2; off >= 1; off >>= 1) {
#pragma unroll
      for (int i = 0; i < R; ++i) {
        acc_k[i] += __shfl_xor_sync(0xffffffffu, acc_k[i], off);
        acc_q[i] += __shfl_xor_sync(0xffffffffu, acc_q[i], off);
      }
    }
    float dk[R], raw[R], rr = 0.0f;
#pragma unroll
    for (int i = 0; i < R; ++i) {
      const float pred = g * acc_k[i] * rks;
      const float delta = (vm[i] - pred) * beta;
      raw[i] = fmaf(delta, kqd, g * acc_q[i] * rqs);
      dk[i] = delta * rks;
      rr += raw[i] * raw[i];
    }
    /* Publish, do not reduce; the butterfly runs above the update/store work.
     * The group here is the LPR-strided set of lanes that hold the same column
     * part, not the whole warp: every one of the LPR lanes of a row already holds
     * the same raw[i], so a full-warp sum would count each row LPR times. */
#pragma unroll
    for (int off = LPR; off < 32; off <<= 1)
      rr += __shfl_xor_sync(0xffffffffu, rr, off);

#pragma unroll
    for (int cc = 0; cc < CR; ++cc) {
      float kc[8];
      if constexpr (KQREG) { up8(kw[cc], kc); }
      else if constexpr (QK32) { ld8f(kstep + cc * 16 * LPR, kc); }
      else { ld8sh(kstep + cc * 16 * LPR, kc); }
#pragma unroll
      for (int i = 0; i < R; ++i) {
        if constexpr (X2) {
          const float2 dkd = make_float2(dk[i], dk[i]);
#pragma unroll
          for (int t = 0; t < 8; t += 2) {
            float* sc = S[i * CR + cc];
            const float2 ns = fma2(dkd, make_float2(kc[t], kc[t + 1]),
                                   make_float2(sc[t], sc[t + 1]));
            sc[t] = ns.x;
            sc[t + 1] = ns.y;
          }
        } else {
#pragma unroll
          for (int t = 0; t < 8; ++t)
            S[i * CR + cc][t] = fmaf(dk[i], kc[t], S[i * CR + cc][t]);
        }
      }
    }

    const int dst = dst_rec;
    if (dst > 0) {
      /* A runtime flag rather than a goto or a duplicated loop.  Everything the
       * assembly needs - including the row-run base with this lane's p*8 element
       * offset removed - is computed INSIDE the `if constexpr (ASA || ASAL)`
       * block, so a slot with neither compiles exactly the store loop it always
       * did and the promoted default's codegen is untouched. */
      bool asa_used = false;
      if constexpr (ASA || ASAL) {
        /* Row-run base in BYTES, so the whole warp's windows for (i, cc) are
         * `rb + off + 16*p` and its aligned floor is `rb - b16`. */
        char* const rb = reinterpret_cast<char*>(rowbase(0) - p * 8) +
                         2 * (int64_t)dst * ss;
        const uintptr_t rba = reinterpret_cast<uintptr_t>(rb);
        const unsigned b16 = (unsigned)(rba & 15u);
        if (!VEC && b16 != 0u) {
          asa_used = true;
          /* Aligned-block assembly.  Lane p >= 1 writes the 16-byte-ALIGNED block
           * that starts b16 bytes before its own window, built from the top b16
           * bytes of lane p-1's window (one shfl_up of a uint4) and the bottom
           * 16-b16 bytes of its own.  For p >= 1 that block lies entirely inside
           * the run - it starts at 16p-b16 >= 16-b16 >= 2 and ends at
           * 16(p+1)-b16 <= 16*LPR - b16 - so it never touches the previous row,
           * the previous head or a neighbouring slot, which is the hazard that
           * stopped the read-side trick (ALD) from being copied here directly.
           * Lanes p == 0 and p == LPR-1 keep the ST4 ladder on their own windows,
           * which covers the run's first 16-b16 and last b16 bytes that no aligned
           * block can reach without leaving the run.  The two paths overlap on
           * bytes they agree about: the ladder writes its whole 16-byte window and
           * the neighbouring aligned block writes the same bytes with the same
           * values, and both are byte-granular stores, so the overlap is
           * order-independent.
           *
           * Both halves are proved locally before this ever reaches a job.
           * map_check.check_asa proves, for every LPR and every b16, that the
           * primitive equals the byte model concat(n,u) >> (128-8*b16), that no
           * aligned block escapes the run (hazard 1), that the blocks are pairwise
           * disjoint, that the union of blocks and ladders is exactly the run, and
           * that every overlapped byte comes from the same source element so the two
           * writes carry one value.  align_check replays the whole store phase for
           * LPR in {2,4,8} x b16 in {2..14} in the device's own instruction order
           * and compares all 1024 bytes of a guarded pool against the plain ladder,
           * so a single byte written outside the run fails it.  It is negative-tested
           * both ways: including p == 0 trips the escape assert, and dropping the
           * p == LPR-1 ladder leaves the run's last b16 bytes uncovered. */
          char* const ab = reinterpret_cast<char*>(rba & ~(uintptr_t)15u);
          if constexpr (ASAL) {
            /* ASAL: the same stores, assembled in place.  Three differences from
             * the ASA arm below, all of them register budget: no kk/sb registers
             * (the switch is on b16, so the shift is a literal in each arm), no
             * second base pointer (the ladder's offset from `ab` is the small
             * scalar `lo`, so `rb` dies as soon as `rba` is taken from it), and no
             * output uint4.  The edge ladders move BEFORE the assembly because they
             * store the unshifted window and the assembly destroys it; the overlap
             * is byte-granular and carries one value (map_check.check_asa property
             * 5), so the order is free.  The shuffles inside asa_assemble stay
             * outside every predicate and every lane runs the same arm, because b16
             * is block-uniform. */
            const int lo = (int)b16 + 16 * p;
            /* CCSEAM leaves only the ROW's two edges on the ladder - chunk 0's p == 0
             * window and chunk CR-1's p == LPR-1 window - because every interior
             * chunk boundary is now written as one aligned block by asa_seam.  Which
             * ladders exist is a compile-time property of (cc, CCSEAM), so a slot
             * without the seam keeps exactly the pair it always had.
             *
             * ROWSEAM takes the two edges CCSEAM leaves, for every row whose
             * predecessor is in the same warp, and narrows the ladders to the two
             * rows a warp cannot reach: its first (its entry block belongs to the
             * previous warp's last row) and its last (its exit block belongs to the
             * next warp's first row).  A row is `rs + i*RPG` of the warp's RPW, so
             * "my predecessor is in this warp" is exactly rs != 0 for the entry
             * ladder and exactly rs != RPG-1 for the exit one.  Both are per-lane
             * predicates on a store, never on a shuffle. */
            const bool lad_lo =
                ROWSEAM ? (p == 0 && rs == 0)
                        : (CCSEAM ? (p == 0) : (p == 0 || p == LPR - 1));
            const bool lad_hi =
                ROWSEAM ? (p == LPR - 1 && rs == RPG - 1)
                        : (CCSEAM ? (p == LPR - 1) : (p == 0 || p == LPR - 1));
            const int seam_src = lane >= (LPR - 1) ? lane - (LPR - 1) : lane;
            /* ROWSEAM: the block entering the NEXT global row is written from this
             * row's LAST window and the next row's FIRST, and the next global row
             * is lane l+1's for every writer that has one: the warp's rows are
             * rs + i*RPG, so the successor of row rs+i*RPG is group rs+1 at the SAME
             * i, and lane l+1 is exactly that group's p == 0 lane.  The one exception
             * is the group RPG-1 writer, whose successor is group 0 at index i+1 -
             * lane l+1 wraps to lane 0, which is at index i, so that seam has no
             * source at this iteration and stays on the ladders.  That is all
             * `rs < RPG - 1` below and `rs == RPG - 1` in lad_hi express.
             * Both halves therefore come from registers live at THIS iteration: the
             * top b16 bytes from the writer's own chunk CR-1 tile, re-packed by
             * packw() exactly as CCSEAM re-packs its previous chunk, and the bottom
             * 16-b16 bytes by shuffling lane l+1's live chunk-0 window.  Nothing is
             * held across the loop, so the register argument is CCSEAM's verbatim:
             * peak liveness is the lane's own u (4) plus (k+1) fetched plus (4-k)
             * packed words, and map_check prints the pair per arm.
             *
             * That is why this costs no new device code: it is asa_seam() with the
             * writer's own last chunk as `prev` and lane l+1 as `src`.  The address
             * is the next row's aligned floor, `ab + oi + 2*D` bytes - 2*D = 256 is
             * the element-stride(2) == D the host entry point TORCH_CHECKs, so it is
             * a compile-time immediate and cannot change b16. */
            const int row_src = (lane + 1) & 31;
#pragma unroll
            for (int i = 0; i < R; ++i) {
              const int64_t oi = (int64_t)i * (2 * ROWOFF);
#pragma unroll
              for (int cc = 0; cc < CR; ++cc) {
                const int64_t o = oi + (int64_t)cc * (16 * LPR);
                uint4 u = pack8(S[i * CR + cc]);
                if ((cc == 0 && lad_lo) || (cc == CR - 1 && lad_hi)) {
                  if constexpr (TRIM) {
                    /* `cc` is a compile-time constant in the unrolled loop and the
                     * ladder fires at cc == 0 for the row's entry and cc == CR-1
                     * for its exit, so this specializes into two calls, each with
                     * one block-uniform cut. */
                    bf16* const ep = reinterpret_cast<bf16*>(ab + o + lo);
                    if (cc == 0) {
                      st8edge<true>(ep, u, b16);
                    } else {
                      st8edge<false>(ep, u, b16);
                    }
                  } else {
                    st8u<false, CSST, ST4>(reinterpret_cast<bf16*>(ab + o + lo), u);
                  }
                }
                if constexpr (ROWSEAM) {
                  if (cc == 0) {
                    /* Warp-collective: every lane runs the shuffles inside asa_seam,
                     * only p == LPR-1 of a group with a successor stores.  It has to
                     * come BEFORE the in-place assembly below, which destroys the
                     * chunk-0 window lane l+1 is read from. */
                    const uint4 blk =
                        asa_seam(u, S[i * CR + CR - 1], row_src, b16);
                    if (p == LPR - 1 && rs < RPG - 1) {
                      st16al<CSST>(ab + oi + (int64_t)(2 * D), blk);
                    }
                  }
                }
                if constexpr (CCSEAM && (CR >= 2)) {
                  if (cc > 0) {
                    /* Warp-collective: every lane runs the shuffles inside asa_seam,
                     * only p == LPR-1 stores.  It has to come BEFORE the in-place
                     * assembly below, which destroys the p == 0 window it reads. */
                    const uint4 blk =
                        asa_seam(u, S[i * CR + cc - 1], seam_src, b16);
                    if (p == LPR - 1) {
                      st16al<CSST>(ab + o, blk);
                    }
                  }
                }
                u = asa_assemble(u, b16);
                if (p >= 1) {
                  st16al<CSST>(ab + o + 16 * p, u);
                }
              }
            }
          } else {
            const unsigned kk = (16u - b16) >> 2;
            const unsigned sb = ((16u - b16) & 3u) * 8u;
#pragma unroll
            for (int i = 0; i < R; ++i) {
              const int64_t oi = (int64_t)i * (2 * ROWOFF);
#pragma unroll
              for (int cc = 0; cc < CR; ++cc) {
                const int64_t o = oi + (int64_t)cc * (16 * LPR);
                const uint4 u = pack8(S[i * CR + cc]);
                const uint4 n = shfl_up1(u);  /* warp-wide: outside every predicate */
                if (p >= 1) {
                  st16al<CSST>(ab + o + 16 * p, asa_block(n, u, kk, sb));
                }
                if (p == 0 || p == LPR - 1) {
                  st8<false, CSST, ST4>(
                      reinterpret_cast<bf16*>(rb + o + 16 * p), S[i * CR + cc]);
                }
              }
            }
          }
        }
      }
      if (!asa_used) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
          bf16* dp = rowbase(i) + (int64_t)dst * ss;
#pragma unroll
          for (int cc = 0; cc < CR; ++cc)
            st8<VEC, CSST, ST4>(dp + cc * 8 * LPR, S[i * CR + cc]);
        }
      }
    }

    if constexpr (PUB1) {
      /* One shared store instead of R.  After the ascending butterfly every lane
       * holds the same warp total, so a single lane can publish it; and the LPR
       * lanes of a row all hold the same raw[i], so the lanes with p < R cover the
       * warp's RPW rows exactly once each - row warp*RPW + p*RPG + rs is owned by
       * lane (rs, p) as its row i = p.  R stores with R of LPR lanes active become
       * one store with R*RPW/LPR = 32/LPR... i.e. 16 of 32 lanes active. */
      if (lane == 0) rmsp[step][hb * NW + warp] = rr;
      if (p < R) rawsh[step * D + warp * RPW + p * RPG + rs] = raw[p];
    } else if (p == 0) {
      rmsp[step][hb * NW + warp] = rr;
      if constexpr (RPERM) {
        /* One vector store instead of R scalar ones.  The element offset
         * step*D + warp*RPW + rs*R is a multiple of R, so 4*R bytes are aligned. */
        float* const rp = rawsh + step * D + hb * DR + warp * RPW + rs * R;
        if constexpr (R == 4) {
          *reinterpret_cast<float4*>(rp) = make_float4(raw[0], raw[1], raw[2], raw[3]);
        } else if constexpr (R == 2) {
          *reinterpret_cast<float2*>(rp) = make_float2(raw[0], raw[1]);
        } else {
          rp[0] = raw[0];
        }
      } else {
#pragma unroll
        for (int i = 0; i < R; ++i) rawsh[step * D + rowidx(i)] = raw[i];
      }
    }
  }

  }   /* end of the RSPLIT pass loop; one iteration when NH == 1 */

  __syncthreads();
  const int gm_mode = prm.gate_mode;
  const int cb = cup[r];
  if constexpr (EPIM >= 3) {
    /* Count-selected epilogue: below EPT run the flat (step, row) form, at or
     * above it run the steps-split form.  Both bodies below are verbatim,
     * re-indented copies of the two proven branches that follow, and those two
     * branches are textually untouched, so slots 0-18 compile to exactly what
     * they did before - in particular the promoted slot 12 stays bit-identical to
     * the tree the public engine measured at 2.911x fixed / 2.500x domain.  The
     * duplication is deliberate; collapsing it into shared helpers is only worth
     * the risk to the promoted path if this mode is promoted.
     *
     * Why: in job dv_4e08a7f3fa85 (2032 MHz, all nineteen slots swept on both
     * frozen suites, 0 mismatched elements in all 190 cells) the steps-split form
     * beat the flat form by 3.1% on fixed case 1, where every block has count=8 -
     * slots 11 and 15 measured 59.62 and 59.52 us against 61.41-62.46 for all six
     * EPIM=2 slots, each group agreeing within 0.05%, so the split tracks the
     * epilogue mode and not cell noise - and by 2-3% on the count=8 domain
     * configurations d4/d5/d8, while the flat form kept its 2.4% edge on case 0 at
     * count=2 and 1.2-5.7% on d0/d1/d2 at one to three tokens.  At eight tokens
     * the flat form pays four dependent iterations and four __shfl_sync
     * broadcasts; the split form pays one iteration whose gate read is 8-way
     * bank-conflicted instead.  Below about four tokens the split form idles most
     * of its lanes, which is why the flat form was promoted in the first place.
     *
     * `count` is block-uniform (it comes from cu_seqlens[r]), so this branch never
     * diverges a warp and the flat form's __shfl_sync still sees all 32 lanes. */
    if (count < EPT) {
      /* EPI3: a flat (step, row) epilogue.  Every value the epilogue needs lives in
       * shared or global memory, so nothing ties an output element to the lanes that
       * computed it: lane l owns row `warp*RPW + l%RPW` for the whole epilogue and
       * walks the steps l/RPW, l/RPW+SINC, ... , which covers each (step, row) of the
       * warp exactly once with all 32 lanes busy at EVERY token count.  EPIM=1 splits
       * the steps over p instead, so at LPR=8 and count=2 only 8 of 32 lanes work and
       * each of them serialises four stores - and the per-tile fixed cost is what the
       * short shapes (case 0/2/9 at two steps, d0 at one) are made of.
       *
       * The RMSNorm denominator for a step is computed once per warp in lane `step`
       * and broadcast with a shuffle instead of every lane re-reading NW partials and
       * re-summing them; the arithmetic is the same left-to-right sum of the same NW
       * FP32 partials followed by the same rsqrtf, so the value is bit-identical.
       * The shuffle sits outside the `live` predicate on purpose: all 32 lanes must
       * reach every __shfl_sync, so the iteration count is block-uniform and only the
       * store is predicated. */
      const int mw = lane % RPW;
      const int m = warp * RPW + mw;
      const float wm = b2f(prm.nwp[m]);
      float rmsl = 0.0f;
      if (lane < count) {
        float part[NW];
#pragma unroll
        for (int w = 0; w < NW; ++w) part[w] = rmsp[lane][w];
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) tot += part[w];
        rmsl = rsqrtf(tot * (1.0f / (float)D) + prm.norm_eps);
      }
      const int niter = (count + SINC - 1) / SINC;
      for (int it = 0; it < niter; ++it) {
        const int step = lane / RPW + it * SINC;
        const bool live = step < count;
        const float rms = __shfl_sync(0xffffffffu, rmsl, live ? step : 0);
        if (live) {
          const float gg = b2f(gsh[step * D + m]);
          const float gate = (gm_mode == 1) ? sigf(gg) : gg * sigf(gg);
          prm.outp[(int64_t)(cb + step) * prm.os0 + (int64_t)j * D + m] =
              __float2bfloat16(((rawsh[step * D + warp * RPW + (RPERM ? rowperm<R, RPG>(mw) : mw)] * rms) * wm) * gate);
        }
      }
    } else {
        float wm[R];
#pragma unroll
        for (int i = 0; i < R; ++i) wm[i] = b2f(prm.nwp[rowidx(i)]);
        const int sfirst = p;
        const int sstride = LPR;
        for (int step = sfirst; step < count; step += sstride) {
          float part[NW];
#pragma unroll
          for (int w = 0; w < NW; ++w) part[w] = rmsp[step][w];
          float tot = 0.0f;
#pragma unroll
          for (int w = 0; w < NW; ++w) tot += part[w];
          const float rms = rsqrtf(tot * (1.0f / (float)D) + prm.norm_eps);
#pragma unroll
          for (int i = 0; i < R; ++i) {
            const float gg = b2f(gsh[step * D + rowidx(i)]);
            const float gate = (gm_mode == 1) ? sigf(gg) : gg * sigf(gg);
            prm.outp[(int64_t)(cb + step) * prm.os0 + (int64_t)j * D + rowidx(i)] =
                __float2bfloat16(((rawsh[step * D + (RPERM ? (warp * RPW + rs * R + i) : rowidx(i))] * rms) * wm[i]) * gate);
          }
        }
    }
  } else   if constexpr (EPIM == 2) {
    /* A flat (step, row) epilogue.  Every value the epilogue needs lives in
     * shared or global memory, so nothing ties an output element to the lanes that
     * computed it: lane l owns row `GDN_EPI_ROW(warp, l)` == `warp*RPE + l%RPE` for
     * the whole epilogue and walks the steps l/RPE, l/RPE+SINCE, ... , which covers
     * each (step, row) of the warp exactly once with all 32 lanes busy at EVERY
     * token count.  EPIM=1 splits the steps over p instead, so at LPR=8 and count=2
     * only 8 of 32 lanes work and each of them serialises four stores - and the
     * per-tile fixed cost is what the short shapes (case 0/2/9 at two steps, d0 at
     * one) are made of.
     *
     * The row geometry above was CORRECTED, and the correction is the point of the
     * paragraph.  This block was copied from the `EPIM >= 3` branch and its copy
     * said `warp*RPW + l%RPW` with steps `l/RPW, l/RPW+SINC` - that branch's
     * geometry, not this one's.  RPW == RPE and SINC == SINCE hold only at NH == 1;
     * under RSPLIT (NH=2, D=128, NW=8) they are RPW=8 vs RPE=16 and SINC=4 vs
     * SINCE=2, so a reader who trusted the copy would index half the rows and
     * repeat them.  That is not hypothetical: NWE's norm_weight hoist had exactly
     * this bug in code, and the comment at its own site asserted the two forms
     * "cannot differ".  Both are now one spelling, GDN_EPI_ROW, gated by
     * experiments/launch_check.py section 10 and experiments/epi_row_check.py.
     *
     * The RMSNorm denominator for a step is computed once per warp in lane `step`
     * and broadcast with a shuffle instead of every lane re-reading NW partials and
     * re-summing them; the arithmetic is the same left-to-right sum of the same NW
     * FP32 partials followed by the same rsqrtf, so the value is bit-identical.
     * The shuffle sits outside the `live` predicate on purpose: all 32 lanes must
     * reach every __shfl_sync, so the iteration count is block-uniform and only the
     * store is predicated. */
    /* The epilogue covers all D rows, so its row geometry is RPE/SINCE and not the
     * tile pass's RPW/SINC - the two are the same numbers when NH == 1, which is
     * every historical slot, so this branch's codegen does not move.  The row-square
     * sum now runs over NH*NW partials, one per (pass, warp): that is the ONLY
     * arithmetic RSPLIT changes, and it changes the ASSOCIATION of a sum of 128
     * non-negative squares, so `out` is no longer bit-identical to the unsplit
     * kernel.  The bound is what makes that acceptable: all terms are >= 0, so the
     * sum is perfectly conditioned and a 16-leaf against an 8-leaf tree differs by
     * ~1e-7 relative, which is 5e-6 of the frozen tolerance's 0.01 + 0.05*|actual|
     * budget.  The state checkpoints are untouched - the recurrence, pred, delta and
     * the rank-1 update are all row-local and their k-reduction tree is unchanged,
     * because the split is along rows and not along columns. */
    const int m = GDN_EPI_ROW(warp, lane);
    /* NWE reads the value the block loaded with its first flight.  This comment
     * used to assert "the address is the same expression, so the two forms cannot
     * differ" - and that assumption is what let the NWE site drift onto RPW while
     * this one stayed on RPE.  The two forms differ whenever NH > 1, and the hoist
     * is evaluated OUTSIDE the pass loop, so nothing downstream forces them back
     * together.  `m` and the hoist now share one spelling, so they cannot differ by
     * construction rather than by coincidence.  (`mw` was only ever `lane % RPE`
     * for this one expression and is folded into the macro.) */
    const float wm = NWE ? nwe_wm : b2f(prm.nwp[m]);
    float rmsl = 0.0f;
    if (lane < count) {
      /* One pass's NW partials at a time, not all NH*NW: part[] is what lets the
       * eight shared loads issue in parallel, and sizing it NW*NH would hold 16
       * floats live in an instantiation whose budget is 85 registers.  Summing
       * inside the loop keeps eight live and the accumulation order identical to a
       * flat 0..NH*NW-1 walk, and at NH == 1 this is textually the code that was
       * here before. */
      float tot = 0.0f;
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        float part[NW];
#pragma unroll
        for (int w = 0; w < NW; ++w) part[w] = rmsp[lane][h * NW + w];
#pragma unroll
        for (int w = 0; w < NW; ++w) tot += part[w];
      }
      rmsl = rsqrtf(tot * (1.0f / (float)D) + prm.norm_eps);
    }
    const int niter = (count + SINCE - 1) / SINCE;
    for (int it = 0; it < niter; ++it) {
      const int step = lane / RPE + it * SINCE;
      const bool live = step < count;
      const float rms = __shfl_sync(0xffffffffu, rmsl, live ? step : 0);
      if (live) {
        const float gg = NOGATE
            ? b2f(prm.gatep[(int64_t)(cb + step) * prm.gs0 + (int64_t)j * D + m])
            : b2f(gsh[step * D + m]);
        const float gate = (gm_mode == 1) ? sigf(gg) : gg * sigf(gg);
        /* rawsh is stored permuted WITHIN each warp's RPW-row group, and with
         * RSPLIT those groups are 8 rows instead of 16, so the group base is
         * m & ~(RPW-1) rather than warp*RPW.  At NH == 1 the two are the same
         * expression: m == warp*RPW + mw with mw < RPW. */
        const int rpos = RPERM
            ? ((m & ~(RPW - 1)) + rowperm<R, RPG>(m & (RPW - 1)))
            : m;
        prm.outp[(int64_t)(cb + step) * prm.os0 + (int64_t)j * D + m] =
            __float2bfloat16(((rawsh[step * D + rpos] * rms) * wm) * gate);
      }
    }
  } else {
    /* EPIM=1 is the promoted EPI2 behaviour: the LPR lanes of a row split the
     * steps between them.  EPIM=0 leaves the whole epilogue to the p==0 lanes. */
    if (EPIM == 1 || p == 0) {
      float wm[R];
#pragma unroll
      for (int i = 0; i < R; ++i) wm[i] = b2f(prm.nwp[rowidx(i)]);
      const int sfirst = (EPIM == 1) ? p : 0;
      const int sstride = (EPIM == 1) ? LPR : 1;
      for (int step = sfirst; step < count; step += sstride) {
        float part[NW];
#pragma unroll
        for (int w = 0; w < NW; ++w) part[w] = rmsp[step][w];
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) tot += part[w];
        const float rms = rsqrtf(tot * (1.0f / (float)D) + prm.norm_eps);
#pragma unroll
        for (int i = 0; i < R; ++i) {
          const float gg = b2f(gsh[step * D + rowidx(i)]);
          const float gate = (gm_mode == 1) ? sigf(gg) : gg * sigf(gg);
          prm.outp[(int64_t)(cb + step) * prm.os0 + (int64_t)j * D + rowidx(i)] =
              __float2bfloat16(((rawsh[step * D + (RPERM ? (warp * RPW + rs * R + i) : rowidx(i))] * rms) * wm[i]) * gate);
        }
      }
    }
  }
}
#undef GDN_EPI_ROW   /* the macro expands against the kernel's own RPE; leaking it past
                      * the end of gdn_kernel_w would let a later scope spell "the
                      * epilogue row" against a different RPE and still compile. */

/* ------------------------------------------------------------------ */
/* ROW-SPLIT FAMILY: implemented, measured, REJECTED 2026-10-21.  The source is in
 * Git history (commit d240985 and the vsh-offset fix that followed); only the
 * numbers stay here, so the design is not rebuilt blind.
 *
 * The design: one block per (request, head, row slice), so the tile per thread
 * falls from 64 floats to 32 (SPLITS=2) or 16 (SPLITS=4).  That is the only way
 * this kernel can raise blocks/SM, and it did: 80 registers at 3 blocks/SM and
 * 62-64 at 4 blocks/SM, both spill-free, against the promoted kernel's hard
 * 2 blocks/SM at 126 registers.  The gated-RMSNorm denominator - the one thing
 * coupling a head's 128 rows - was resolved by a device-scratch retirement handoff
 * (publish raw rows and per-step sums of squares, fence, increment a per-head
 * counter; the slice that sees SPLITS-1 normalises all 128 rows and clears the
 * counter), so nothing spins and no schedule can deadlock.
 *
 * Measured in job dv_d7db602e8e4b (2032 MHz, minima against slot 28 in the same
 * job) on the three shapes whose handoff provably ran correctly - fixed cases 0, 1
 * and 2 each got a freshly zeroed workspace and passed the whole-storage
 * correctness screen:
 *   SPLITS=2 MINB=3 (80 regs, 3 blocks/SM, 24 warps)   0.851 / 0.803 / 0.728
 *   SPLITS=2 MINB=2 (94 regs)                          0.724 / 0.666 / 0.562
 *   SPLITS=4 MINB=4 (62 regs, 4 blocks/SM, 32 warps)   0.764 / 0.717 / 0.618
 *   SPLITS=4 MINB=2 (72 regs)                          0.691 / 0.638 / 0.520
 * Every slice on every shape was slower.  The shapes where the handoff was broken
 * were slower too, and a skipped epilogue can only make a kernel look FASTER, so
 * the verdict does not rest on the bug.
 *
 * Why the occupancy never got to pay: the grid scan (job dv_a7aaafead63e) prices a
 * wave of this kernel at ~6.3 us almost independently of the work inside it, and
 * slicing the tile multiplies the wave count by SPLITS while the per-wave cost
 * stays.  Fixed case 0 goes from 384 blocks / 1.30 waves to 768 / 2.59, i.e. from
 * 2 waves to 3, and 3 x (2.2 us of fixed cost + half the transfer) loses to
 * 2 x (2.2 us + all of it).  The scan said so before the kernel was written - at
 * 768 blocks a 64 KiB block runs at 3.43 TB/s and a 96 KiB block at 3.92 - and the
 * build was the expensive way to confirm it.  Occupancy is not this kernel's
 * binding constraint; the per-wave fixed cost and the tail are, and finer blocks
 * buy the tail only by paying the fixed cost SPLITS times over.
 *
 * Both bugs are recorded because they are the kind a screen catches only by luck.
 * (1) vsh is staged per step over all 128 rows (it has to be: the QK norms run over
 * all 128 columns) but a slice read it at its own local row instead of
 * q*ROWS + local, so slice 1 consumed slice 0's v - exactly half of `out` wrong.
 * (2) The scratch layout was derived from the CURRENT head count while the
 * workspace was only reallocated when that count GREW, so a smaller launch after a
 * larger one put its counters inside the previous launch's raw region and started
 * from garbage; that is why the failing shapes looked arbitrary (they were the ones
 * whose head count did not grow).  A production handoff needs a layout keyed on the
 * allocated capacity, not on the request.
 */
/* ------------------------------------------------------------------ */
/* Launch-configuration table: one integer knob selects a frozen tuple
* (P threads per state row, NA accumulator pairs, packed F32X2, evict-first
 * cache hint, MINB blocks per SM, MERGE store-inside-update, VEC alignment). */

struct Variant {
  int P;
  int lpr;
  /* Selects the kernel family, and therefore which launcher template the
   * table-driven dispatch below instantiates.  It cannot be inferred: narrow
   * slots set lpr == P, but so could a wide slot, and epim overlaps (narrow
   * EPI2=true and wide EPIM=1 both report 1). */
  bool wide;
  int na;
  bool f32x2;
  bool cs;
  int minb;
  bool merge;
  int epim;      /* 0 = p==0-only epilogue, 1 = steps split over p, 2 = flat */
  bool ald;      /* misaligned source reads via two aligned 16B blocks */
  bool st4;      /* misaligned stores via the widest aligned access at each step */
  bool qk32;     /* q/k pre-widened to FP32 in shared, in swizzled group order */
  bool pack;     /* one 32-byte per-step scalar record instead of three arrays */
  bool pub1;     /* per-step shared publish in one store instead of R */
  bool csb;      /* one constant-stride state base pointer instead of R of them */
  bool rperm;    /* vsh/rawsh laid out so a thread's R rows are contiguous */
  bool asa;      /* misaligned stores via cross-lane aligned-block assembly */
  bool asal;     /* the same assembly done in place: fewer live registers */
  bool csbodd;   /* CSB's constant-stride row base, VEC=false instantiation only */
  bool ccseam;   /* the chunk-boundary aligned block, replacing two interior ladders */
  bool rowseam;  /* the row-boundary aligned block, replacing the last two ladders */
  int smem_bytes;
  const char* name;
  /* Appended after `name`, with default member initializers, on purpose: an
   * aggregate may be brace-initialised with fewer initialisers than members and
   * the rest take their defaults, so every existing macro keeps emitting the same
   * 23 values in the same order and no historical slot's meaning moves.  Both are
   * the promoted 2026-10-21 startup-path mechanisms; see slot 39. */
  bool grid2d = false;
  bool hqfast = false;
  /* Cache-hint split; -1 keeps `cs` for both directions.  Appended last with a
   * default member initialiser so every existing macro keeps emitting the same
   * values in the same order. */
  int cm = -1;
  /* L2 prefetch distance in waves; 0 disables it.  Appended last with a default
   * member initialiser so every existing macro keeps emitting the same values in
   * the same order. */
  int pfw = 0;
  /* Also warm the target block's staged q/k/v/gate row slices; false in every
   * historical slot.  Appended last, same reason. */
  bool pfv = false;
  /* Narrow CCSEAM's two remaining ladders to the bytes no aligned block reaches;
   * false in every historical slot.  Appended last, same reason. */
  bool trim = false;
  /* Walk the state's rows in two passes so the register tile halves and MINB=3 fits
   * without spilling; false in every historical slot.  Appended last, same reason,
   * and it grows the shared arena by MAXS*NW*4 = 256 B for the second pass's
   * row-square partials, which is why the slot's Arena takes it too. */
  bool rsplit = false;
  /* Tail self-prefetch: when the PFW branch has no next-wave target to warm -
   * the last wave of a grid, or every block of a single-wave grid - warm this
   * block's own RSPLIT pass-2 half-tile instead; false in every historical
   * slot.  Appended last with a default member initialiser, same contract as
   * every field above it: existing macros keep emitting the same values in the
   * same order and no historical slot's meaning moves. */
  bool pft = false;
  /* Wave bound for the two-barrier longest-first request remap (LPT2W): remap only
   * when `blocks > SM_COUNT*MINB` and `blocks <= lpt2w * SM_COUNT*MINB`, i.e. between
   * one and `lpt2w` waves.  0 disables it and selects no code at all, because the
   * kernel's guard is `if constexpr (LPT2W > 0)`.
   *
   * The bound is part of the mechanism and not a tuning knob: the tail imbalance the
   * remap removes is a fraction of ONE wave's makespan while the ranking is paid by
   * EVERY wave, so above roughly two waves the ranking costs more than the tail
   * returns.  dv_8cd01dc6e45d put the crossover between 1.73 waves (fixed 4/6/8 at
   * +13.93/+4.36/+3.95%) and 3.46 (fixed 3/7 at -1.9/-2.9%).
   *
   * Appended LAST with a default member initialiser, same contract as every field
   * above it.  That contract is load-bearing rather than tidy: the aggregate is
   * brace-initialised positionally by six macros, so a field inserted anywhere but the
   * end silently shifts every later value into the wrong member.  It has already
   * happened once -- a field added between `pftb` and `pfh` put GDN_WIDERF's `pfh`
   * argument into `lpt` and left `pfh` at its default, so a slot would have shipped the
   * longest-first remap and lost the half-width warm, and only the same_tuple pins
   * caught it, at compile time, in dv_bfee84a9231c.  NOTE the arena consequence:
   * `Arena<P, QK32, PACK, LPT, (LPT2W > 0), RSPLIT>` grows by the [class][warp] table
   * and its found flag when lpt2w is non-zero, so a slot that sets it must also pass it
   * to Arena -- which is why GDN_WIDEU exists rather than reusing GDN_WIDERT. */
  int lpt2w = 0;
  /* Self pass-2 arena-reload warm (PF2), ported 2026-10-26 from arm a's
   * pre-join lineage onto this base: 0 = off, 1 = late self pass-2 warm at
   * pass-1 step 0, 2 = prologue self pass-2 warm on the has-target branch.
   * Requires RSPLIT && PFT (asserted in gdn_kernel_w).  Appended LAST with a
   * default member initialiser - the same load-bearing contract as every
   * field above it. */
  int pf2 = 0;
  /* CPA-P2 (2026-10-26): thread-private cp.async pre-staging of the pass-2
   * state half - the reload becomes an LDS from the arena-appended image
   * instead of a global ld8 flight.  Requires RSPLIT (asserted in Arena and
   * gdn_kernel_w); live in the VEC=true instantiation only, dead code in the
   * scalar twin.  Appended LAST with a default member initialiser, the same
   * append-only contract as pf2 above. */
  bool cpa = false;
  /* TSP-P1 (2026-10-26): thread-private cp.async staging of the PASS-1 tile
   * on the same contract - the prologue's serial ld8 flight becomes a
   * fire-and-forget burst whose wait lands after the qkv staging.  Requires
   * RSPLIT && !CSB && !CSBODD && !PLOV && LDB==0 (asserted); VEC=true only.
   * Appended LAST with a default member initialiser. */
  bool tsp = false;
  /* IDXPAR (2026-10-28): fetch the whole state_indices row in the prologue's
   * FIRST flight, together with cu_seqlens[r], cu_seqlens[r+1] and
   * num_accepted_tokens[r], instead of loading idxp[r*is0 + acc-1] only after
   * accp[r] has landed.  The row is at most MAXS ints inside one or two cache
   * lines and every lane wants the same values, so the speculation costs one
   * sector and removes a whole DRAM round trip from the chain that gates the
   * 32 KiB source tile.
   *
   * The MECHANISM already existed (EXP slots 202/204/205/208 carry it; every
   * promoted slot got a literal `false` from launch_slot) and was rejected on
   * the 2026-10-21 slot-28 package: +0.71%/+0.55% alone, +0.28%/+0.23% in
   * combination, over all twenty cells.  That verdict is scoped by what those
   * twenty cells are.  results/20261028T0835Z-profile-baseline-result-
   * dv_d35c04837372.json prices the frozen window's floor at 5.12 us and puts
   * fixed:0 (384 blocks, ONE wave, uniform counts) at 60% and fixed:5 (256
   * blocks, one wave) at 39% of this device's 6.231 TB/s copy ceiling, while
   * the two uniform-count MULTI-wave shapes fixed:1/fixed:2 are at 97%/95%.  A
   * round trip that only wave 2+ can hide behind another block's step loop is
   * worth re-asking on the shapes that have no wave 2, on the package that now
   * wins them.  Appended LAST with a default member initialiser - the same
   * load-bearing append-only contract as every field above it, so every
   * existing macro keeps emitting its values into the same members. */
  bool idxpar = false;
  /* PFL (2026-10-28): the three L2 prefetch sweeps walk 128-byte LINES instead of
   * striding region_bytes/threads, which at P=1 is 256 bytes and so warms every
   * other line only.  See the PFL block at the top of gdn_kernel_w for the defect
   * and for why a P=2 slot carrying it is an identical-code control.  Appended
   * LAST with a default member initialiser - the same load-bearing append-only
   * contract as every field above it. */
  bool pfl = false;
};

#define GDN_VARIANT(PV, NAV, XV, CV, MV, MG, E2)                                \
  { PV, PV, false, NAV, XV, CV, MV, MG, (E2 ? 1 : 0), false, false, false,      \
    false, false, false, false, false, false, false, false, false, Arena<PV>::BYTES,                \
    "P" #PV "/NA" #NAV "/X" #XV "/C" #CV "/M" #MV "/MG" #MG "/E2" #E2 }

/* Wide lane mapping: LPRV lanes cooperate on one state row, so the thread owns
 * LPRV/PV rows and 16/LPRV chunks of each.  See gdn_kernel_w. */
#define GDN_WIDE(LPRV, PV, NAV, XV, CV, MV, E2)                                 \
  { PV, LPRV, true, NAV, XV, CV, MV, false, (E2 ? 1 : 0), false, false, false,  \
    false, false, false, false, false, false, false, false, false, Arena<PV>::BYTES,                \
    "W" #LPRV "/P" #PV "/NA" #NAV "/X" #XV "/C" #CV "/M" #MV "/E2" #E2 }

/* Wide lane mapping with an explicit epilogue mode (EPM) and consecutive-row flag
 * (CRW).  TAG is the name suffix: "R" when the thread's rows are consecutive. */
#define GDN_WIDEX(LPRV, PV, NAV, EPM, ALDV, S4V, TAG)                          \
  { PV, LPRV, true, NAV, true, true, 2, false, EPM, ALDV, S4V, false, false,   \
    false, false, false, false, false, false, false, false, Arena<PV>::BYTES,                \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM }

/* As GDN_WIDEX plus the optional QK32 and PACK shared layouts, whose arenas
 * differ, and the two aligned-block store forms (AV = uint4 broadcast, LV = the
 * in-place assembly; at most one may be true). */
#define GDN_WIDEQ(LPRV, PV, NAV, EPM, ALDV, S4V, QKV, PKV, PBV, CBV, RPV, AV,  \
                  LV, OV, SV, RV, TAG)                                          \
  { PV, LPRV, true, NAV, true, true, 2, false, EPM, ALDV, S4V, QKV, PKV, PBV,  \
    CBV, RPV, AV, LV, OV, SV, RV, Arena<PV, QKV, PKV>::BYTES,                  \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM }

/* The promoted package with the two startup-path mechanisms made explicit: G2V
 * launches a 2-D grid and HQV removes the second runtime division.  Field order is
 * GDN_WIDEQ's plus the two appended flags, and experiments/launch_check.py checks
 * both the arity and that slot 39's tuple is what it claims. */
#define GDN_WIDEG(LPRV, PV, NAV, EPM, ALDV, S4V, RPV, LV, OV, SV, G2V, HQV, TAG) \
  { PV, LPRV, true, NAV, true, true, 2, false, EPM, ALDV, S4V, false, false,     \
    false, false, RPV, false, LV, OV, SV, false, Arena<PV>::BYTES,               \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM, G2V, HQV }

/* THREADS axis (2026-10-20 campaign restart).  The promoted package - ALD + ST4 +
 * RPERM + ASAL + CCSEAM, EPIM=2, F32X2 - with the block size, the requested blocks
 * per SM and the cache hint made explicit.  Every field GDN_WIDEQ hardcodes
 * (qk32, pack, pub1, csb, asa, rowseam = false) is false here too, so at
 * (P=2, MINB=2, CS=true) this macro reproduces promoted slot 28 exactly; the
 * name suffix records what differs.
 *
 * Why these slots exist.  The tile is register-resident, so a 256-thread block
 * needs 64 tile floats per thread and 2 blocks/SM is the ceiling: 512 threads/SM,
 * 25% occupancy, and a grid of 384 blocks (fixed case 0) runs as 2 waves on 296
 * resident slots, i.e. 65% utilisation before a single extra instruction is
 * executed.  Halving the tile per thread is the only way to raise threads/SM, and
 * the historical rejection of that axis (P=4 1.2-5.6% slower, P=8 12-18% slower)
 * was measured with the NARROW map, whose warp access covers 16 rows x 32 bytes
 * instead of the wide map's 4 rows x a full 128-byte line - the two effects were
 * confounded.  The header itself names P=4/LPR=8 as untested.  These slots hold
 * the lane map fixed at LPR=8 (the promoted one, full 128-byte lines per warp
 * access) and move only P, MINB and the cache hint. */
#define GDN_WIDEF(LPRV, PV, NAV, EPM, MINBV, CSV, ALDV, S4V, RPV, LV, OV, SV,   \
                  TAG)                                                          \
  { PV, LPRV, true, NAV, true, CSV, MINBV, false, EPM, ALDV, S4V, false, false, \
    false, false, RPV, false, LV, OV, SV, false, Arena<PV>::BYTES,              \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/M" #MINBV "/C" #CSV }

/* GDN_WIDEF plus the two promoted startup flags and the per-direction cache-hint
 * split, so a hint policy can be A/B'd against slot 39 with nothing else moving.
 * CSV is the historical single hint (what CMV < 0 falls back to) and CMV the
 * two-bit split: bit 0 streams the source-tile loads, bit 1 the checkpoint
 * stores. */
#define GDN_WIDEI(LPRV, PV, NAV, EPM, MINBV, CSV, ALDV, S4V, RPV, LV, SV,       \
                  G2V, HQV, CMV, TAG)                                           \
  { PV, LPRV, true, NAV, true, CSV, MINBV, false, EPM, ALDV, S4V, false, false, \
    false, false, RPV, false, LV, false, SV, false, Arena<PV>::BYTES,           \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/M" #MINBV "/C" #CSV           \
    "/CM" #CMV, G2V, HQV, CMV }

/* GDN_WIDEG with the L2 prefetch distance made explicit.  The cache-hint split
 * stays at -1 (follow `cs` in both directions) because the promoted package has
 * no split, so at PFWV=0 this macro reproduces slot 39 exactly. */
/* GDN_WIDEG with the L2 prefetch distance and the cache-hint split made
 * explicit.  At PFWV=0, PFVV=false and CMV=-1 this macro reproduces slot 39
 * exactly, and CMV=-1 means "follow CSV in both directions", which is what every
 * promoted slot has always done. */
#define GDN_WIDEZ(LPRV, PV, NAV, EPM, ALDV, S4V, RPV, LV, OV, SV, G2V, HQV,     \
                  PFWV, PFVV, CMV, TAG)                                         \
  { PV, LPRV, true, NAV, true, true, 2, false, EPM, ALDV, S4V, false, false,     \
    false, false, RPV, false, LV, OV, SV, false, Arena<PV>::BYTES,               \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/PF" #PFWV #PFVV "/CM" #CMV,    \
    G2V, HQV, CMV, PFWV, PFVV }

/* GDN_WIDEZ with the trimmed edge ladders made explicit.  At TV=false this
 * reproduces the corresponding GDN_WIDEZ slot exactly, including its name suffix,
 * so a TRIM arm and its control differ in one field. */
#define GDN_WIDET(LPRV, PV, NAV, EPM, ALDV, S4V, RPV, LV, OV, SV, G2V, HQV,     \
                  PFWV, PFVV, CMV, TV, TAG)                                     \
  { PV, LPRV, true, NAV, true, true, 2, false, EPM, ALDV, S4V, false, false,     \
    false, false, RPV, false, LV, OV, SV, false, Arena<PV>::BYTES,               \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/PF" #PFWV #PFVV "/CM" #CMV     \
    "/T" #TV,                                                                    \
    G2V, HQV, CMV, PFWV, PFVV, TV }

/* GDN_WIDEZ with MINB made explicit.  MINB is the one knob that moves occupancy on
 * this kernel from the table alone, and it moves TWO things at once, which is why
 * the carveout experiment earlier in the campaign found nothing: __launch_bounds__
 * (D*P, MINB) caps ptxas at 65536/(256*MINB) registers, and apply_attrs sizes the
 * shared carveout as MINB*(BYTES+1024), i.e. exactly enough for MINB blocks.  At
 * MINB=2 that is 12% of 233472 B = 28016 B, which floors to 2 blocks of the 12736-B
 * arena - so NCU's `launch__occupancy_limit_shared_mem = 2` is the carveout, not the
 * arena.  Raising MINB therefore raises both the shared allowance and the register
 * pressure at once, and the tile's own 64 floats per thread are what has to give. */
#define GDN_WIDEY(LPRV, PV, NAV, EPM, MINBV, ALDV, S4V, RPV, LV, OV, SV, G2V,   \
                  HQV, PFWV, PFVV, CMV, TAG)                                    \
  { PV, LPRV, true, NAV, true, true, MINBV, false, EPM, ALDV, S4V, false, false, \
    false, false, RPV, false, LV, OV, SV, false, Arena<PV>::BYTES,               \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/M" #MINBV "/PF" #PFWV #PFVV    \
    "/CM" #CMV,                                                                  \
    G2V, HQV, CMV, PFWV, PFVV }

/* GDN_WIDEY with the two-pass row split made explicit, and MINB with it because the
 * split only earns its keep at 3 blocks/SM.  At RV=false this reproduces the matching
 * GDN_WIDEY slot exactly.  The arena takes RV: the second pass publishes its own
 * row-square partials, so rmsp is [MAXS][NH*NW] and BYTES grows by 256 B. */
#define GDN_WIDER(LPRV, PV, NAV, EPM, MINBV, ALDV, S4V, RPV, LV, OV, SV, G2V,   \
                  HQV, PFWV, PFVV, CMV, RV, TAG)                                \
  { PV, LPRV, true, NAV, true, true, MINBV, false, EPM, ALDV, S4V, false, false, \
    false, false, RPV, false, LV, OV, SV, false,                                \
    Arena<PV, false, false, false, false, RV>::BYTES,                            \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/M" #MINBV "/PF" #PFWV #PFVV    \
    "/CM" #CMV "/H" #RV,                                                         \
    G2V, HQV, CMV, PFWV, PFVV, false, RV }

/* GDN_WIDER with the tail self-prefetch made explicit.  At PFTV=false this
 * reproduces GDN_WIDER exactly, value for value. */
#define GDN_WIDERT(LPRV, PV, NAV, EPM, MINBV, ALDV, S4V, RPV, LV, OV, SV, G2V,   \
                   HQV, PFWV, PFVV, CMV, RV, PFTV, TAG)                          \
  { PV, LPRV, true, NAV, true, true, MINBV, false, EPM, ALDV, S4V, false, false, \
    false, false, RPV, false, LV, OV, SV, false,                                 \
    Arena<PV, false, false, false, false, RV>::BYTES,                             \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/M" #MINBV "/PF" #PFWV #PFVV     \
    "/CM" #CMV "/H" #RV "/T" #PFTV,                                               \
    G2V, HQV, CMV, PFWV, PFVV, false, RV, PFTV }

/* GDN_WIDERT plus the explicit self pass-2 warm mode (PF2V), ported 2026-10-26
 * from arm a's pre-join lineage.  The aggregate gains the two values at the END
 * (explicit lpt2w=0, then pf2) per the struct's append-only field contract, and
 * the name gains "/W"#PF2V so the census can tell the rows apart.  At PF2V=0
 * this reproduces GDN_WIDERT exactly, value for value. */
#define GDN_WIDERT2(LPRV, PV, NAV, EPM, MINBV, ALDV, S4V, RPV, LV, OV, SV, G2V,  \
                    HQV, PFWV, PFVV, CMV, RV, PFTV, PF2V, TAG)                    \
  { PV, LPRV, true, NAV, true, true, MINBV, false, EPM, ALDV, S4V, false, false, \
    false, false, RPV, false, LV, OV, SV, false,                                 \
    Arena<PV, false, false, false, false, RV>::BYTES,                            \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/M" #MINBV "/PF" #PFWV #PFVV    \
    "/CM" #CMV "/H" #RV "/T" #PFTV "/W" #PF2V,                                   \
    G2V, HQV, CMV, PFWV, PFVV, false, RV, PFTV, 0, PF2V }

/* GDN_WIDERT2 plus the pass-2 cp.async pre-staging flag (CPAV), 2026-10-26.
 * The aggregate gains the value at the END (after pf2) per the struct's
 * append-only field contract, the Arena gains the 16 KiB staging image, and
 * the name gains "/C"#CPAV so the census can tell the rows apart.  At CPAV=0
 * this reproduces GDN_WIDERT2 exactly, value for value. */
#define GDN_WIDERT3(LPRV, PV, NAV, EPM, MINBV, ALDV, S4V, RPV, LV, OV, SV, G2V,  \
                    HQV, PFWV, PFVV, CMV, RV, PFTV, PF2V, CPAV, TAG)              \
  { PV, LPRV, true, NAV, true, true, MINBV, false, EPM, ALDV, S4V, false, false, \
    false, false, RPV, false, LV, OV, SV, false,                                 \
    Arena<PV, false, false, false, false, RV, CPAV>::BYTES,                      \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/M" #MINBV "/PF" #PFWV #PFVV    \
    "/CM" #CMV "/H" #RV "/T" #PFTV "/W" #PF2V "/C" #CPAV,                        \
    G2V, HQV, CMV, PFWV, PFVV, false, RV, PFTV, 0, PF2V, CPAV }

/* GDN_WIDERT3 plus the pass-1 tile cp.async staging flag (TSPV), 2026-10-26.
 * The aggregate gains the value at the END (after cpa) per the struct's
 * append-only field contract, the Arena gains the second 16 KiB staging
 * image, and the name gains "/S"#TSPV.  At TSPV=0 this reproduces
 * GDN_WIDERT3 exactly, value for value. */
#define GDN_WIDERT4(LPRV, PV, NAV, EPM, MINBV, ALDV, S4V, RPV, LV, OV, SV, G2V,  \
                    HQV, PFWV, PFVV, CMV, RV, PFTV, PF2V, CPAV, TSPV, TAG)        \
  { PV, LPRV, true, NAV, true, true, MINBV, false, EPM, ALDV, S4V, false, false, \
    false, false, RPV, false, LV, OV, SV, false,                                 \
    Arena<PV, false, false, false, false, RV, CPAV, TSPV>::BYTES,                \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/M" #MINBV "/PF" #PFWV #PFVV    \
    "/CM" #CMV "/H" #RV "/T" #PFTV "/W" #PF2V "/C" #CPAV "/S" #TSPV,             \
    G2V, HQV, CMV, PFWV, PFVV, false, RV, PFTV, 0, PF2V, CPAV, TSPV }

/* GDN_WIDERT plus the two-barrier longest-first remap's wave bound.  At L2WV=0 this
 * reproduces GDN_WIDERT exactly, value for value AND byte for byte: the Arena's fifth
 * template argument is `(L2WV > 0)`, so at 0 it selects the same CORE-only arena
 * GDN_WIDERT asks for.  A separate macro rather than a new argument on GDN_WIDERT
 * because the arena must see the bound -- `launch_slot` instantiates
 * `Arena<c.P, c.qk32, c.pack, false, false, c.rsplit>` for the wide path and
 * `launch_one_w` builds `Arena<P, QK32, PACK, LPT, (LPT2W > 0), RSPLIT>`, so a slot
 * whose smem_bytes came from the LPT2-free arena while its launcher instantiated the
 * LPT2 one would trip launch_slot's own "slot arena disagrees with the launcher's"
 * assert.  L2WV is appended last in the initialiser to match the field order. */
#define GDN_WIDEU(LPRV, PV, NAV, EPM, MINBV, ALDV, S4V, RPV, LV, OV, SV, G2V,   \
                  HQV, PFWV, PFVV, CMV, RV, PFTV, L2WV, TAG)                    \
  { PV, LPRV, true, NAV, true, true, MINBV, false, EPM, ALDV, S4V, false, false, \
    false, false, RPV, false, LV, OV, SV, false,                                 \
    Arena<PV, false, false, false, (L2WV > 0), RV>::BYTES,                       \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/M" #MINBV "/PF" #PFWV #PFVV     \
    "/CM" #CMV "/H" #RV "/T" #PFTV "/L2W" #L2WV,                                 \
    G2V, HQV, CMV, PFWV, PFVV, false, RV, PFTV, L2WV }

/* GDN_WIDERT4 plus BOTH 2026-10-28 arms as explicit arguments: IXV is the
 * promoted-table wiring of IDXPAR (the whole state_indices row in the
 * prologue's first flight) and L2WV the two-barrier longest-first remap's wave
 * bound.  At IXV=false, L2WV=0 this reproduces GDN_WIDERT4 value for value AND
 * byte for byte - Arena's fifth template argument is `(L2WV > 0)`, so at 0 it
 * selects the same CPA/TSP-only arena, and `idxpar` is the struct's last member
 * with a default of false.  A new macro rather than two more arguments on
 * GDN_WIDERT4 for the reason GDN_WIDEU gives: the arena must see the bound, and
 * launch_slot's "slot arena disagrees with the launcher's" assert compares
 * `Arena<c.P, c.qk32, c.pack, false, (c.lpt2w > 0), c.rsplit, c.cpa, c.tsp>`
 * against the table's smem_bytes, so a slot whose bytes came from an
 * LPT2-free arena while its launcher instantiated the LPT2 one fails the build. */
#define GDN_WIDEV(LPRV, PV, NAV, EPM, MINBV, ALDV, S4V, RPV, LV, OV, SV, G2V,  \
                  HQV, PFWV, PFVV, CMV, RV, PFTV, PF2V, CPAV, TSPV, IXV, L2WV, \
                  TAG)                                                         \
  { PV, LPRV, true, NAV, true, true, MINBV, false, EPM, ALDV, S4V, false, false, \
    false, false, RPV, false, LV, OV, SV, false,                                \
    Arena<PV, false, false, false, (L2WV > 0), RV, CPAV, TSPV>::BYTES,          \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/M" #MINBV "/PF" #PFWV #PFVV   \
    "/CM" #CMV "/H" #RV "/T" #PFTV "/W" #PF2V "/C" #CPAV "/S" #TSPV             \
    "/I" #IXV "/L2W" #L2WV,                                                     \
    G2V, HQV, CMV, PFWV, PFVV, false, RV, PFTV, L2WV, PF2V, CPAV, TSPV, IXV }

/* GDN_WIDEV plus PFLV, the 128-byte-line prefetch sweep.  At PFLV=false this
 * reproduces GDN_WIDEV value for value AND byte for byte (PFL is the struct's
 * last member with a default of false, and it adds no shared memory), so slot 79
 * against 77 and 78 against 71 are one-mechanism pairs.  Slot 80 is the P=2 twin
 * of slot 72: at P=2 the line loop's addresses are the old expression's, so 80 is
 * an identical-code control that says whether merely instantiating another row
 * moves anything. */
#define GDN_WIDEW(LPRV, PV, NAV, EPM, MINBV, ALDV, S4V, RPV, LV, OV, SV, G2V,  \
                  HQV, PFWV, PFVV, CMV, RV, PFTV, PF2V, CPAV, TSPV, IXV, L2WV,  \
                  PFLV, TAG)                                                    \
  { PV, LPRV, true, NAV, true, true, MINBV, false, EPM, ALDV, S4V, false, false, \
    false, false, RPV, false, LV, OV, SV, false,                                 \
    Arena<PV, false, false, false, (L2WV > 0), RV, CPAV, TSPV>::BYTES,           \
    "W" #LPRV TAG "/P" #PV "/NA" #NAV "/E" #EPM "/M" #MINBV "/PF" #PFWV #PFVV    \
    "/CM" #CMV "/H" #RV "/T" #PFTV "/W" #PF2V "/C" #CPAV "/S" #TSPV              \
    "/I" #IXV "/L2W" #L2WV "/F" #PFLV,                                           \
    G2V, HQV, CMV, PFWV, PFVV, false, RV, PFTV, L2WV, PF2V, CPAV, TSPV, IXV,     \
    PFLV }

constexpr Variant VARIANTS[] = {
  /* 0 */ GDN_VARIANT(2, 2, true, true, 2, false, true),  /* old default (2.330x) */
  /* 1 */ GDN_VARIANT(2, 2, true, true, 2, false, false), /* p==0-only epilogue */
  /* 2 */ GDN_VARIANT(4, 4, true, true, 1, false, true),  /* old sub-wave pick */
  /* 3 */ GDN_VARIANT(4, 4, true, true, 1, false, false), /* P4 control */
  /* 4 */ GDN_VARIANT(2, 4, true, true, 2, false, true),  /* NA=4 control */
  /* 5 */ GDN_VARIANT(2, 2, true, true, 2, true, true),   /* MERGE control */
  /* 6 */ GDN_VARIANT(2, 2, true, false, 2, false, true), /* evict-first A/B */
  /* 7 */ GDN_VARIANT(2, 2, true, true, 1, false, true),  /* MINB=1: no spills */
  /* 8 */ GDN_VARIANT(8, 2, true, true, 1, false, true),  /* P8: 32 warps/SM */
  /* 9 */ GDN_WIDE(4, 2, 1, true, true, 2, true),   /* WIDE4, register-neutral */
  /* 10 */ GDN_WIDE(4, 2, 2, true, true, 2, true),  /* WIDE4, NA=2 ILP */
  /* 11 */ GDN_WIDE(8, 2, 1, true, true, 2, true),  /* WIDE8: promoted default */
  /* 12 */ GDN_WIDEX(8, 2, 1, 2, true, true, "AS"), /* + widest-aligned store ladder */
  /* 13 */ GDN_WIDEX(8, 2, 1, 2, false, false, ""), /* v18 winner: flat epilogue */
  /* 14 */ GDN_WIDEX(8, 2, 1, 2, true, false, "A"), /* PROMOTED: + ALD misaligned reads */
  /* 15 */ GDN_WIDEX(8, 2, 2, 1, false, false, ""), /* WIDE8 NA=2 control */
  /* 16 */ GDN_WIDEQ(8, 2, 1, 2, true, true, true, false, false, false, false, false, false, false, false, false, "ASQ"), /* rejected: FP32 k/q */
  /* 17 */ GDN_WIDEQ(8, 2, 1, 2, true, true, false, true, false, false, false, false, false, false, false, false, "ASP"), /* rejected: packed record */
  /* 18 */ GDN_WIDEQ(8, 2, 1, 2, true, true, false, false, true, false, false, false, false, false, false, false, "ASB"), /* rejected: one-store publish */
  /* 19 */ GDN_WIDEQ(8, 2, 1, 3, true, true, false, false, false, false, false, false, false, false, false, false, "ASE6"), /* rejected: count-selected epilogue, EPT=6 */
  /* 20 */ GDN_WIDEQ(8, 2, 1, 4, true, true, false, false, false, false, false, false, false, false, false, false, "ASE8"), /* rejected: count-selected epilogue, EPT=8 */
  /* 21 */ GDN_WIDEQ(8, 2, 1, 2, true, true, false, false, false, true, false, false, false, false, false, false, "ASC"), /* one constant-stride state base */
  /* 22 */ GDN_WIDEQ(8, 2, 1, 2, true, true, false, false, false, false, true, false, false, false, false, false, "ASR"), /* PROMOTED: + row-permuted vsh/rawsh */
  /* 23 */ GDN_WIDEQ(8, 2, 1, 2, true, true, false, false, false, true, true, false, false, false, false, false, "ASCR"), /* + both */
  /* 24 */ GDN_WIDEQ(8, 2, 1, 2, true, true, false, false, false, true, true, true, false, false, false, false, "ASCA"), /* ASA: mechanism validated, package NOT promoted */
  /* 25 */ GDN_WIDEQ(8, 2, 1, 2, true, true, false, false, false, false, true, true, false, false, false, false, "ASRA"), /* ASA without CSB: does it fit slot 22's two free registers? */
  /* 26 */ GDN_WIDEQ(8, 2, 1, 2, true, true, false, false, false, false, true, false, true, false, false, false, "ASRL"), /* ASAL: the same stores, assembled in place */
  /* 27 */ GDN_WIDEQ(8, 2, 1, 2, true, true, false, false, false, false, true, false, true, true, false, false, "ASRO"), /* ASAL + the row-base fold in the VEC=false instantiation only */
  /* 28 */ GDN_WIDEQ(8, 2, 1, 2, true, true, false, false, false, false, true, false, true, false, true, false, "ASRS"), /* ASAL + the chunk-boundary seam */
  /* 29 */ GDN_WIDEQ(8, 2, 1, 2, true, true, false, false, false, false, true, false, true, false, true, true, "ASRR"), /* + the row-boundary seam: the last two ladders per warp */
  /* --- THREADS axis, 2026-10-20 restart.  Slot 33 is byte-for-byte slot 28's
   * tuple rebuilt through GDN_WIDEF and is this group's identical-code noise
   * control; every other slot changes exactly one of P, MINB or CS. --- */
  /* 30 */ GDN_WIDEF(8, 4, 1, 2, 1, true, true, true, true, true, false, true, "M1"),  /* 512 threads, 1 block/SM: 32 tile floats, 128-reg cap */
  /* 31 */ GDN_WIDEF(8, 4, 1, 2, 2, true, true, true, true, true, false, true, "M2"),  /* 512 threads, 2 blocks/SM = 1024 threads/SM iff <=64 regs */
  /* 32 */ GDN_WIDEF(8, 8, 1, 2, 1, true, true, true, true, true, false, true, "M1"),  /* 1024 threads, 1 block/SM: 16 tile floats, 64-reg cap */
  /* 33 */ GDN_WIDEF(8, 2, 1, 2, 2, true, true, true, true, true, false, true, "CTL"), /* slot 28's exact tuple via the new macro: noise floor */
  /* 34 */ GDN_WIDEF(8, 2, 1, 2, 1, true, true, true, true, true, false, true, "M1"),  /* 256 threads, 1 block/SM: half occupancy, no spills */
  /* 35 */ GDN_WIDEF(8, 2, 1, 2, 2, false, true, true, true, true, false, true, "C0"), /* slot 28 without the evict-first hint */
  /* 36 */ GDN_WIDEF(8, 4, 1, 2, 2, false, true, true, true, true, false, true, "C0"), /* slot 31 without the evict-first hint */
  /* --- ILP / epilogue-form controls on the promoted package.  Slot 15 against
   * slot 11 in job dv_c52939cc4aec says NA=2 is worth +1.6% at EPIM=1 without the
   * store package; these two re-ask both questions ON the promoted package, where
   * the register budget (126 of the 128 MINB=2 allows) is what kept NA=2 out. --- */
  /* 37 */ GDN_WIDEQ(8, 2, 2, 2, true, true, false, false, false, false, true, false, true, false, true, false, "ASRS2"), /* slot 28 + two accumulator pairs */
  /* 38 */ GDN_WIDEQ(8, 2, 1, 1, true, true, false, false, false, false, true, false, true, false, true, false, "ASRSE1"), /* slot 28 with the steps-split epilogue */
  /* 39 */ GDN_WIDEG(8, 2, 1, 2, true, true, true, true, false, true, true, true, "ASRSQ"), /* PROMOTED: slot 28 + 2-D grid + division-free q head */
  /* --- CACHE-HINT axis, 2026-10-21.  Slot 39's package with the streaming hint
   * split per direction, which is a host-selectable policy rather than new device
   * code: every one of these is a separate instantiation, so slot 39's binary is
   * untouched.  The device has 126.5 MiB of L2 and the fixed suite's short shapes
   * touch 28-80 MiB end to end, i.e. a checkpoint stream that fits in cache is
   * being pushed to DRAM by an evict-first store; the probe priced the plain hint
   * flip at +4.2% on fixed case 0 and +1.4% on domain d0 against -6.8% on fixed 1
   * and -10.8% on fixed 9, whose working sets are 288 and 173 MiB.  CMV is the
   * two-bit split (bit 0 loads, bit 1 stores), -1 means "follow CSV". */
  /* 40 */ GDN_WIDEI(8, 2, 1, 2, 2, false, true, true, true, true, true, true, true, -1, "C0G"),  /* both directions write-back */
  /* 41 */ GDN_WIDEI(8, 2, 1, 2, 2, true, true, true, true, true, true, true, true, 1, "LDST"),  /* tile loads stream, checkpoints stay */
  /* 42 */ GDN_WIDEI(8, 2, 1, 2, 2, true, true, true, true, true, true, true, true, 2, "STST"),  /* checkpoints stream, tile loads stay */
  /* --- L2 PREFETCH axis, 2026-10-21.  Slot 39's exact package plus one flag:
   * every block issues `prefetch.global.L2` for the 32 KiB source tile of the
   * block PFW waves later in the dispatch order (head fastest, so that is the
   * same head and request r + PFW*pfd, with pfd = ceil(SM_COUNT*MINB/hv) filled
   * in by the host).  256 threads x 128 B is exactly one tile and each tile is
   * warmed by exactly one block, so total DRAM traffic is unchanged - the read
   * moves one wave earlier, into headroom that NCU says is there: fixed case 2
   * runs at 40.6% of DRAM and 24% occupancy with 33.0% of warp cycles on
   * long_scoreboard and 21.1% on barrier, i.e. the block waits for a tile it
   * asked for one round trip ago and there are not enough warps to hide it.
   * This is the one startup mechanism that adds no data registers, no shared
   * buffer, no grid change and no host-visible policy, and a prefetch cannot
   * change a result, so slots 43/44 are bit-identical to 39 by construction.
   * Two distances because the lead time a wave gets is the block's own lifetime:
   * at count=2 (case 2, d2) that is short, at count=8 (case 1) long enough that
   * a line warmed one wave ahead may not survive to two. */
  /* 43 */ GDN_WIDEZ(8, 2, 1, 2, true, true, true, true, false, true, true, true, 1, false, -1, "ASRSQ"),  /* tile prefetch one wave ahead */
  /* 44 */ GDN_WIDEZ(8, 2, 1, 2, true, true, true, true, false, true, true, true, 2, false, -1, "ASRSQ"),  /* two waves ahead: rejected, +1.41%/+1.16% */
  /* 45: slot 43 plus the target's staged vectors.  Warming the tile shortens the
   * tile flight, which is what leaves the prologue barrier waiting on the four
   * 256-byte q/k/v/gate slices that the `count` staging warps pull cold and that
   * no other block touches.  8 lines of 128 B per token is exactly those slices,
   * so the extra traffic is the target's own and the q/k lines shared by a head
   * group are prefetched redundantly but fetched once. */
  /* 45 */ GDN_WIDEZ(8, 2, 1, 2, true, true, true, true, false, true, true, true, 1, true, -1, "ASRSQV"),
  /* --- the cache-hint split re-asked ON TOP of the prefetch, 2026-10-21.  Slots
   * 40/41/42 answered it before any prefetch existed and all three lost (-0.61% /
   * -1.26% / -0.87% fixed), with the explanation that the restore's own dirty
   * lines decide it.  A prefetch changes what is resident, so the question is
   * stale rather than settled: 46 keeps both directions write-back, 47 streams
   * the tile loads only, 48 streams the checkpoint stores only.  The prediction is
   * that -1 (both stream, i.e. slot 45) stays best, because an evict-first store
   * is what stops 64 KiB of checkpoint per block from displacing the next wave's
   * warm tile, and an evict-first load frees a tile that is dead the moment it is
   * widened.  Measuring it is cheaper than arguing it. */
  /* 46 */ GDN_WIDEZ(8, 2, 1, 2, true, true, true, true, false, true, true, true, 1, true, 0, "ASRSQV"),
  /* 47 */ GDN_WIDEZ(8, 2, 1, 2, true, true, true, true, false, true, true, true, 1, true, 1, "ASRSQV"),
  /* 48 */ GDN_WIDEZ(8, 2, 1, 2, true, true, true, true, false, true, true, true, 1, true, 2, "ASRSQV"),
  /* --- NA=2 re-asked on the promoted package, 2026-10-21.  Two accumulator pairs
   * split the step's two row dot products into two independent 4-FMA chains
   * instead of one 8-deep chain, which is the recurrence's longest fixed-latency
   * dependency and therefore a candidate for the `wait` stall that NCU puts at
   * 12.6% on fixed:2 and 18.3% on domain d2 after the prefetch.  It was measured
   * once, on slot 28's package (slot 37), where it was 1.08 points better than its
   * own control - and then slot 39 replaced that control with GRID2D + HQFAST and
   * the combination was never built.  The archived single-pass sweep ranks slot 37
   * at +1.61% against slot 39 and slot 28 at +2.69%, i.e. NA=2 worth about 1%, but
   * that instrument's geomean noise is of the same order (slot 27 sits 1.9 points
   * below slot 28 in it and lost in every better measurement), so this is a
   * question for the interleaved A/B and not a conclusion.
   *
   * Registers argue for trying it: the NA=2 instantiation of this package compiles
   * at 121 (VEC) / 128 (scalar) against slot 45's 123 / 128, both with zero spills,
   * so the extra accumulator pair is not what this kernel usually charges for.
   * NA=2 is NOT bit-identical - it reassociates a 16-term FP32 sum - so the
   * identity screen will flag it and the arithmetic has to be judged by the frozen
   * checker, whose tolerance is atol 0.01 / rtol 0.05 on values that are rounded to
   * BF16 either way.  49 is the promoted tuple plus NA=2, 50 is stage 1 plus NA=2,
   * so the two stages can be told apart if the fence interacts with it. */
  /* 49 */ GDN_WIDEZ(8, 2, 2, 2, true, true, true, true, false, true, true, true, 1, true, -1, "ASRSQV"),
  /* 50 */ GDN_WIDEZ(8, 2, 2, 2, true, true, true, true, false, true, true, true, 1, false, -1, "ASRSQ"),
  /* --- TRIM axis, 2026-10-22.  CCSEAM leaves two ladders per row and both write
   * their whole 16-byte window, although aligned block m=1 already covers
   * [16-b16,16) of the entry one and block m=15 covers [240,256-b16) of the exit
   * one.  TRIM keeps only the bytes no block reaches.  It is bit-identical by
   * construction rather than by measurement: the dropped bytes are written by a
   * block carrying the same source element (map_check's COVER property 5), and both
   * tools now prove the trimmed set covers every row for every LPR and b16 -
   * map_check over 15232 row bytes and align_check by replaying a whole 32 KiB
   * checkpoint into a guarded pool 147 times.  The kept pieces are a prefix/suffix
   * of the ST4 arm's own list, so no new funnel shifts or addresses appear.  Priced
   * on the host at 32-64 committed memory ops per warp saved out of the 112/80/48
   * that separate the promoted store phase from the aligned one (57-67% of what
   * CCSEAM left), for 0-24 issued instructions depending on whether ptxas hoists a
   * comparison whose only input is the block-uniform cut; the archive's exchange
   * rate (44 issued bought 112 committed at CCSEAM and won, bought 84 at ROWSEAM
   * and lost) puts that straddling the break-even, so it is measured rather than
   * deduced.  results/store-trim-20261022/. */
  /* REJECTED, measured (dv_62d9a839eb86, 6 rounds x 12 samples, identical-code
   * control at +0.05% fixed / -0.22% domain): bit-identical on all twenty
   * configurations, which is what the host proofs predicted, and +8.0/+13.3/+15.8%
   * SLOWER on d1/d2/d7 - the only three cells whose code changes - for a domain
   * geomean of -0.89% against slot 45's +2.61%.  128 registers and zero spills, so
   * it is not allocation: guarding the ST4 pieces turns straight-line stores into
   * branches and doubles the ladder family in an instantiation already at 5582
   * static instructions with 14% of its stalls in the front end.  Kept with its
   * evidence; see results/store-trim-20261022/. */
  /* 51 */ GDN_WIDET(8, 2, 1, 2, true, true, true, true, false, true, true, true, 1, true, -1, true, "ASRSQVT"),
  /* 52 */ GDN_WIDET(8, 2, 1, 2, true, true, true, true, false, true, true, true, 1, false, -1, true, "ASRSQT"),
  /* --- OCCUPANCY axis, 2026-10-22.  Slots 53/54 are slot 45's exact package at
   * MINB=3 and MINB=4, i.e. 37.5% and 50% occupancy against the promoted 24%.
   * They exist to bound the value-split axis before anyone builds it.  That axis is
   * the only structural idea left with an arithmetic path to higher occupancy: two
   * blocks of a thread-block cluster each own 64 of the state's 128 rows, which the
   * recurrence permits (with S[i,k], pred[i] = sum_k S[i,k] k[k] and the rank-1
   * update are both within one block's rows; only the gated RMSNorm's mean over i
   * crosses, one float per step, which the already-deferred epilogue can exchange
   * once).  It halves the tile to 32 floats per thread and the R-scaling arrays with
   * it, an estimated 67-77 registers, and 3 blocks/SM needs <= 85.
   * experiments/cluster_probe.py (dv_a56b76fa5958) settled the three things that
   * could have killed it: cluster launch is supported, a clustered kernel achieves
   * the SAME residency as a plain one at every footprint (3.06 blocks/SM measured at
   * 80 registers, against 3.06 plain), and one cluster.sync() plus a dsmem float
   * costs 654 cycles = 0.33 us ONCE per call.
   * What the probe cannot answer is whether the occupancy gain is worth anything on
   * this kernel, and these two slots answer it without the redesign: MINB=3 buys 3
   * blocks/SM by making ptxas fit 85 registers instead of 128, so it pays for the
   * occupancy with roughly 38 spilled registers (152 B per thread) where the value
   * split would pay with none.  If MINB=3 is faster than slot 45, the split is very
   * likely a large win; if it is slower by less than the spill traffic it added, the
   * occupancy gain is real and the split still has room; if it is slower by more,
   * occupancy is not what these cells are waiting for and the split dies here for
   * the price of one job. */
  /* REJECTED, measured (dv_62285ff2da47, 6 rounds x 12 samples, identical-code
   * control at +0.62% fixed / +0.13% domain): MINB=3 is **-38.23% fixed and
   * -37.88% domain** against slot 45's +4.46/+3.11, and MINB=4 is -58.32/-54.38.
   * ptxas met both targets - 80 registers with 312-328 B of stack for MINB=3, 64
   * with 528-536 B for MINB=4 - so the occupancy was bought and the spill traffic
   * spent more than it bought, on every one of the twenty cells.  The tile is what
   * spills, and the tile is read four times per step, so a spilled element is
   * re-fetched from L1 on every pass.  This is the strongest evidence yet that the
   * register-resident tile is not a design choice to be traded: it is the design.
   * It does NOT close the value-split axis, which reaches 3 blocks/SM by halving the
   * tile rather than by spilling it - but it prices what happens if that estimate is
   * wrong by even a few registers. */
  /* 53 */ GDN_WIDEY(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, "ASRSQV"),
  /* 54 */ GDN_WIDEY(8, 2, 1, 2, 4, true, true, true, true, false, true, true, true, 1, true, -1, "ASRSQV"),
  /* The same curve walked DOWNWARD, and the spill-free half of it: slot 45's exact
   * package at MINB=1, i.e. 12.5% occupancy instead of 24%, with ptxas free to use
   * up to 255 registers so nothing spills.  Slots 53/54 measured "occupancy gain
   * minus spill cost" and the spill cost swallowed it; this measures the occupancy
   * term alone, from the other side.  If halving occupancy costs little, then
   * raising it buys little and the value-split redesign dies here; if it costs a
   * third, the slope says the split's 24% -> 37.5% is worth building. */
  /* 55 */ GDN_WIDEY(8, 2, 1, 2, 1, true, true, true, true, false, true, true, true, 1, true, -1, "ASRSQV"),
  /* --- ROW-SPLIT axis, 2026-10-22.  One block walks the state's 128 rows in two
   * passes of 64, so the register tile halves from 64 floats per thread to 32 and
   * MINB=3 becomes reachable WITHOUT spilling it - which is the only way up the
   * occupancy curve this kernel has left, since buying 3 blocks/SM out of the full
   * tile (slot 53) costs 312-328 B of stack and 38% on every cell, and occupancy
   * itself is measured at ~21% per doubling (slot 55 against 45, spill-free).
   * The recurrence is row-local, so the only cross-pass term is the gated RMSNorm's
   * mean over rows, which the promoted epilogue already defers to a shared partial:
   * the split adds one column to rmsp and no second kernel, no global scratch and no
   * extra launch, which are the three costs that made the archived 32-row split lose
   * 5-22%.  Slot 57 is the same split at MINB=2, i.e. the same occupancy as the
   * promoted pick, so 57 against 45 isolates the split's own instruction cost and 56
   * against 57 isolates the occupancy gain.
   * NOT bit-identical: the row-square sum goes from 8 leaves to 16, which
   * reassociates a sum of 128 non-negative squares and moves `out` by ~1e-7
   * relative, 5e-6 of the frozen tolerance.  The state checkpoints ARE bit-identical
   * - the k-reduction tree is along columns and the split is along rows. */
  /* 56 */ GDN_WIDER(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, "ASRSQV"),
  /* 57 */ GDN_WIDER(8, 2, 1, 2, 2, true, true, true, true, false, true, true, true, 1, true, -1, true, "ASRSQV"),
  /* --- PFT, the tail self-prefetch, 2026-10-23 (v2 era).  Slot 56's tuple with
   * pft=true and nothing else moved.  Derived and priced from the cold-cache
   * pcsamp dv_d331ce30238e (results/20261023T1730Z-pcsamp-pricing-PFT-
   * derivation.json): the RSPLIT pass-2 tile unpack is the largest single
   * long_sb site on f2 (147/2277 stall samples, 6.5%) and 7.2% of stalls on
   * f0, and f0 is single-wave (pfd = ceil(444/48) = 10 > nreq = 8) so slot
   * 56's next-wave prefetch never fires on ANY f0 block - every pass-2 half
   * there is a cold DRAM round trip mid-block, and with the resident blocks in
   * near-lockstep there is no phase diversity for MINB=3 absorption to cover.
   * PFT turns the dead else case of the PFW branch into a self-warm of rows
   * [D/2, D) with the whole pass-1 loop as lead time; destinationless, so
   * bit-identical by construction and register-neutral by expectation (the
   * allocator gate decides).  PROMOTED 2026-10-23 = MULTI_WAVE_VARIANT: won
   * two independent replicated A/B jobs against 56 (dv_6a550a272a17 fixed-10
   * geomean 0.9687 / domain 0.9697; dv_79e85fb1485d 0.9715 / 0.9746;
   * allocator-neutral 78/0 on VEC; 32/32 spots clean incl. odd-padding d2 on
   * the VEC=false path).  Full record at the MULTI_WAVE_VARIANT pick and in
   * results/20261023T1805Z-pft-ab-dv_6a550a272a17.json. */
  /* 58 */ GDN_WIDERT(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, "ASRSQV"),
  /* Slot 58's one-field neighbourhood, added 2026-10-25 because it did not exist:
   * exactly one one-field neighbour of the promoted multi-wave pick was in the
   * table (56, pft off) and no EXP_VARIANTS slot was within two fields of it.
   * Nine of the ten fixed cells run slot 58, so any fixed-suite gain has to come
   * from making 58 faster, and that needs a local search space to search.
   *
   * Both are GDN_WIDERT with exactly one argument moved, so they are pinned
   * relationally to 58 below and cannot drift.  Neither changes the arena
   * (Arena<> does not depend on PFW or PFV), so neither moves the carveout, and
   * both keep MINB=3 - which matters because occupancy here is register-capped,
   * not shared-capped: 3 x 256 x 78 = 59,904 of 65,536 with 7 registers of
   * headroom per thread, and MINB=4 spills 620-1104 B on this tree.  An arm that
   * costs registers is therefore dead before it is timed, and these two should
   * not.
   *
   * 59 asks whether one wave is still the right prefetch lead.  PFW=2 measured
   * +1.41%/+1.16% against PFW=1 in the 2026-10-21 campaign, but that was
   * pre-`restore-l2-evict-event-v2` and before RSPLIT was promoted, and prefetch
   * distance is the parameter most likely to interact with a wave structure that
   * has since changed.  Memory-hierarchy arms are also the class this workspace
   * measured as *inverting sign* between the clean graph window and the scored
   * one, so an old verdict here is the weakest kind of evidence.
   *
   * 60 isolates PFV, the staged-vector prefetch.  It has never been A/B'd
   * against the promoted pick on any tree - only PFT has, via slot 56 - so the
   * family's archived +7.66%/+8.95% is attributed to the two mechanisms together
   * and not to either.  `!PFV || PFW > 0` is satisfied, so this compiles. */
  /* 59 */ GDN_WIDERT(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 2, true, -1, true, true, "ASRSQV"),
  /* 60 */ GDN_WIDERT(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, false, -1, true, true, "ASRSQV"),
  /* The two unmeasured points on the geometric axes, added 2026-10-25.
   *
   * 61 is LPR=16.  The lanes-per-row axis was swept over {2,4,8} and 8 won, and the
   * header's own rationale for the axis is that one warp access covers 32/LPR rows
   * x 16*LPR contiguous bytes -- 32 B per row at LPR=2, 64 at 4, and "at LPR=8 it is
   * 4 rows x a full 128-byte line".  LPR=16 is 2 rows x 256 B, i.e. the next point
   * in the direction that made 8 win, and it is legal: 16 % 16 == 0, 16 % P == 0,
   * and with P=2/NH=2/DR=64/NW=8 it gives C=4, R=C*LPR/16=4, CR=16/LPR=1,
   * RPG=32/LPR=2, RPW=DR/NW=8 == R*RPG, satisfying every static_assert including
   * !RSPLIT || (D%NH==0 && DR%RPW==0 && RPW%RPG==0).  The tile is still C*8 = 32
   * floats, so the register tile does not move.  CR=1 also removes the chunk loop
   * and every interior chunk boundary, so CCSEAM has nothing left to supply.
   * experiments/map_check.py --lprs 2,4,8,16 proves the store path at LPR=16:
   * MAP bijection, ASAL's 7 parsed arms in lockstep, ESCAPE, SEAM, SEAM-GEOM, SET
   * (blocks exactly m=1..15, disjoint) and COVER over 20,608 (byte, LPR, b16, seam,
   * trim) tuples with every row byte equal to its own element's golden value and no
   * write outside the row -- 0 failures.  What it cannot tell me is whether the
   * RMSNorm row-square sum keeps its association when RPG halves, so this arm is
   * NOT assumed bit-identical; the identity screen decides and the margin gate
   * follows if it says no.
   *
   * 62 is NA=2 at slot 58's geometry.  NA was swept as {1,2,4} at older mechanism
   * generations, never against this package (rsplit + pft + pfv + pfw=1 + minb=3),
   * and it is the one main-loop field that trades registers for independent
   * accumulators in the k-reduction.  Occupancy here is register-capped with 7
   * registers of headroom (3 x 256 x 78 = 59,904 of 65,536) and MINB=4 spills
   * 620-1104 B, so the build log answers "does it fit?" before any timing is worth
   * reading -- which is the point of putting it in the same job.  Splitting the
   * reduction into cc % NA partial sums changes the association, so this arm is
   * expected NOT to be bit-identical and is gated on the margin, not on identity. */
  /* 61 */ GDN_WIDERT(16, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, "ASRSQV"),
  /* 62 */ GDN_WIDERT(8, 2, 2, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, "ASRSQV"),
  /* --- LPR=4, the other boundary of the lane-map axis, and with MINB=4 the only
   * route to four resident blocks that is not priced by spills.  2026-10-25.
   *
   * The legal lpr set at this package is {4, 8, 16}, not the {2,4,8,16} the axis was
   * believed to have: C = DR/(8*P) and RSPLIT makes NH=2, so C=4 and lpr in {1,2}
   * gives R = C*lpr/16 = 0, which fails `R*CR == C`.  lpr=8 is the incumbent and 16
   * is slot 61.  lpr=4 is the third point and has never been measured here -- the
   * archived {2,4,8} sweep ran at an NH=1 package where C=8.
   *
   * Why these two are a pair rather than one arm.  R = C*lpr/16 sizes eight register
   * arrays (sjb[R], mr[R], vm[R], ak[R][NA], aq[R][NA], acc_k[R]/acc_q[R],
   * dk[R]/raw[R], wm[R]) and is {1,2,4} at lpr {4,8,16}.  Slot 58 measures 78
   * registers against __launch_bounds__(256,3)'s cap of floor(65536/(256*3)) = 85, so
   * it has 7 of headroom.  Dropping R from 2 to 1 removes roughly 13 registers of
   * R-sized state, which puts slot 63 near 65 -- and MINB=4's cap is
   * floor(65536/(256*4)) = 64.  So slot 64 is within about one register of the cap
   * where slot 54 (lpr=8, minb=4) was fourteen short of it and spilled 620-1104 B.
   *
   * That is the gap the occupancy closure names for itself: "a spill-free bit-identical
   * NH=2 kernel at 64 registers has never been measured".  Both measured routes to four
   * resident blocks are confounded -- slot 81 bought them with a bf16 tile and its
   * conversions, and the NH=4 route bought them with two extra passes -- and both lost,
   * so what was refuted is those routes, not the concurrency.  The ceiling is not small:
   * slot 81 delivered achieved warp occupancy 31.29 -> 35.64%, and on a kernel whose
   * dominant stall is long_scoreboard at 18-45% with schedulers idle ~80%, hiding more
   * latency scales with warps in flight.
   *
   * 63 is 64's control and is not optional: without it a win at 64 could not be
   * attributed between the lane map and the occupancy, since both move together off
   * slot 58.  58 vs 63 isolates lpr, 63 vs 64 isolates MINB.
   *
   * NEITHER is assumed bit-identical.  lpr changes RPG from 4 to 8, so which warp
   * contributes which row partial to rmsp[MAXS][NH*NW] may change the RMSNorm
   * row-square sum's association; minb changes no arithmetic at all, so 63 and 64
   * should be bit-identical to EACH OTHER and that is a check on the screen, not a
   * hope.  The identity screen decides and the margin gate follows if it says no. */
  /* 63 */ GDN_WIDERT(4, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, "ASRSQVL4"),
  /* 64 */ GDN_WIDERT(4, 2, 1, 2, 4, true, true, true, true, false, true, true, true, 1, true, -1, true, true, "ASRSQVL4M4"),
  /* --- Slot 65: slot 58's exact tuple plus LPT2W=2, the two-barrier longest-first
   * request remap bounded to grids of one to two waves.  2026-10-25.
   *
   * Why this arm exists now rather than in 2026-10-21 when LPT2 was last measured.
   * dv_8cd01dc6e45d put LPT2 at +13.93% on fixed:4, +4.36% on fixed:6, +3.95% on
   * fixed:8 and +8.79% on d1 -- every 512- and 384-block configuration in the band --
   * and it still could not be promoted, for one reason: the ten configurations OUTSIDE
   * the band, where the runtime guard skips the remap before a single instruction of it
   * runs, averaged 1.5% slower than an identical-code control on the same ten.  "One
   * extra register and two dead basic blocks at the top of the kernel are enough."
   *
   * That cost is an artefact of the bound living INSIDE one kernel.  `if constexpr
   * (LPT2W > 0)` removes the code entirely from a slot with lpt2w=0; what it cannot
   * remove is the code from a slot with lpt2w=2 whose runtime `blocks <= LPT2W*cap` test
   * then skips it.  Slot 65 exists to be selected BY DISPATCH rather than self-limiting
   * at runtime, so out-of-band grids instantiate plain slot 58 with no remap code at
   * all and pay nothing.  What is left is then the mechanism's own value.
   *
   * The excess it targets is still present on the CURRENT certified latencies, not just
   * on the v1 tree it was measured on: results/budget-20261024/'s per-cell roof puts
   * fixed:3/4/6/7/8 at 18.6-21.9% above their own achievable floor, +2.37 to +4.99 us,
   * and attributes it to intra-wave length imbalance.  Every other remedy for that
   * excess is a measured loss (barrier LPT -0.41%, warp-redundant LPTW -7.13%, HB=2
   * -13.7%, persistent stride and dynamic queue closed four ways, occupancy bracketed).
   *
   * TWO CAVEATS, both stated before any window is timed rather than after.
   *
   * 1. LPT2 was measured against slot 39, a package with NO rsplit and NO prefetch.
   *    Slot 58 has rsplit + pfw=1 + pfv + pft, worth +10.18% fixed over slot 39, so
   *    some of the same latency slack is already recovered and the gains may not be
   *    additive.
   * 2. THE REMAP MISTIMES THE PREFETCH, and this arm does not fix that.  `r` is set from
   *    blockIdx.y, the remap reassigns it, and the prefetch then computes
   *    `pfr = r + PFW * prm.pfd` -- treating a SORTED REQUEST INDEX as a DISPATCH
   *    POSITION.  Under the identity mapping those coincide; under a permutation the
   *    block PFW waves later handles a different request than pfr names.  The 70-vs-69
   *    factorial priced exactly this at 2.11 points.  So slot 65 measures the remap WITH
   *    that defect live, which makes it the conservative arm: a gain here survives the
   *    mistiming, and a loss does not distinguish "the mechanism is dead" from "the
   *    mistiming ate it".  Reading the result requires saying which. */
  /* 65 */ GDN_WIDEU(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 2, "ASRSQV"),

  /* 66: slot 58's tuple with P 2->1 and minb 3->4 and nothing else moved.  P=1
  * puts 128 threads and a 64-float tile in a block, so four blocks per SM fit
  * under a 128-register cap instead of three under an 85-register one: the only
  * point on the capacity axis that reaches four resident streams without the
  * register squeeze that killed MINB=4, SACC and NH=4.  UNDISPATCHED - pick_cfg
  * still returns 39 and 58 only.  See results/p1-four-streams-20261026/. */
  /* 66 */ GDN_WIDERT(8, 1, 1, 2, 4, true, true, true, true, false, true, true, true, 1, true, -1, true, true, "ASRSQVP1"),
  /* 67: slot 58's exact tuple with pf2=2 (the prologue self pass-2 warm),
   * ported from arm a's pre-join PF2 lineage (arm-7 dv_3d5cce982dfb).
   * PROMOTED 2026-10-26 = MULTI_WAVE_VARIANT: A/B dv_5ba7036214b6 +
   * identical replication dv_474c98f94424 - bitwise-identical outputs, zero
   * register cost, f1 -1.2..-2.0us minima and pipe cand-favoring in all four
   * passes, f2/f9 neutral, band-invariance close (results/20261026T1545Z).
   * RE-SCOPED later the same day by the CPA-P2 adoption: pick_cfg now
   * returns 39, 67 (SCALAR multi-wave grids, where cpa is dead code and the
   * pf2 warm measured +1.03us on d2), 68 (VEC multi-wave) and 66 (P=1 band);
   * slot 58 remains in the table as the pf2=0 control row for future
   * same-job A/Bs. */
  /* 67 */ GDN_WIDERT2(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 2, "ASRSQV"),
  /* The CPA-P2 research pair, 2026-10-26 (prereg results/20261026T1930Z-cpa-
   * p2-prereg.json): 68 = slot 58's tuple + cpa (the mechanism WITHOUT the PF2
   * warm - cp.async staging makes the L2 warm redundant in principle, and the
   * pair separates the two), 69 = slot 67's tuple + cpa (the composition the
   * production pin would adopt if the A/B wins).  The staging image replaces
   * the pass-2 reload's global ld8 flight with a prologue-issued per-thread
   * cp.async into a 16 KiB arena extension; outputs are bit-identical by
   * construction (same uint4 bytes, same up8).  MEASURED and PROMOTED
   * 2026-10-26: 68 = CPA_VARIANT, the VEC multi-wave pick (two draws
   * dv_dafa261558bb + dv_738ac327576b: 68-vs-58 negative on all six cells,
   * case1 -4.7..-5.2%; full-20 confirmation dv_3ca8c71f57ea; records
   * 20261026T2030Z + 20261026T2050Z) - CERTIFIED as tree 90fb8882 at
   * 22:09Z.  SUPERSEDED 2026-10-27 as the pick by slot 72 (68 + tsp, the
   * composition close dv_9794e563e288 + dv_6817d269e892); 68 stays as the
   * composition A/B's control row.  69 stays UNDISPATCHED - the pf2
   * composition measured redundant-to-harmful (case2: keeping PF2 costs
   * 0.4-0.6us at b2048/w2). */
  /* 68 */ GDN_WIDERT3(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, true, "ASRSQV"),
  /* 69 */ GDN_WIDERT3(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 2, true, "ASRSQV"),
  /* 70: slot 66's exact tuple + cpa - the P=1 band row WITH the thread-private
   * cp.async pass-2 staging (prereg results/20261026T2200Z-cpa-p1-band-prereg
   * .json).  Slot 66 is rsplit=TRUE (P=1 = one thread per row pair, 128
   * threads, two passes): the band's cells DO pay a pass-2 reload, warmed
   * today only by PFT's L2 self-prefetch - the same exposure CPA just proved
   * strictly stronger than on the multi-wave family (68 vs 67).  Arena grows
   * 12736 -> 29120; at MINB=4 that is 116480 B of the 233472 B budget.  The
   * register margin is 3 (125 used of the 128 cap): the probe's census is a
   * hard gate.  MEASURED NEGATIVE (dv_5cea0a04c463: case6 -0.109us, case8
   * +0.042 - pass-2 staging is null where the single-wave band keeps reloads
   * L2-resident; probe census 128/0 passed).  Stays UNDISPATCHED as the
   * record of that close; the band moved to 71 (pass-1 staging) instead. */
  /* 70 */ GDN_WIDERT3(8, 1, 1, 2, 4, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, true, "ASRSQVP1"),
  /* 71: slot 66's exact tuple + tsp - the P=1 band row with cp.async staging
   * of the PASS-1 tile (prereg results/20261026T2300Z-tsp-p1-prereg.json).
   * The case6 profile (dv_88d06c742419, first-ever band profile) prices the
   * exposure: long_sb 1.69 TOP stall class at 18.8% warp occupancy - the
   * single-wave band gets no PFW next-wave warm and PFT warms only pass-2,
   * so the cold pass-1 flight is serial with the qkv staging (today's
   * order).  TSP turns tile_latency + staging into max(tile_latency,
   * staging) at ZERO register cost (register-PLOV needs +32 vs the 125/128
   * cap) and +16 KiB smem (29120 x MINB=4 = 116480 <= 233472).  The
   * archived TSTG loss is pre-v2 AND P=2-multi-wave-scoped (absorption
   * holds there; the 1730Z derivation scopes it OUT for single-wave).
   * MEASURED and PROMOTED 2026-10-27 = P1_VARIANT: dv_37c6db8acaa3 -
   * case6 -0.999us (-4.86%), case8 -0.764us (-3.69%), per-pass consistent,
   * rel_l2 digit-identical to 66, probe census 125/0/0 = 66 exactly. */
  /* 71 */ GDN_WIDERT4(8, 1, 1, 2, 4, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, false, true, "ASRSQVP1"),
  /* 72: slot 68's exact tuple + tsp - the P=2 multi-wave CPA row WITH the
   * pass-1 tile staging composed on top (both 16 KiB images; arena
   * 12992+16384+16384 = 45760, x3 = 137280 <= 233472).  Motivation: the
   * TSP band A/B control showed 71 beating 66 by -6.59% on case0 (b384,
   * single-wave) - pass-1 flights are PFW-unwarmed on the FIRST wave of
   * any grid, so the P=2 family's cold-start blocks carry the same
   * exposed flight the band cells do (prereg
   * results/20261026T2340Z-tsp-p1-band-result: next_step).  MEASURED and
   * PROMOTED 2026-10-27 = CPA_VARIANT (two identical-command draws
   * dv_9794e563e288 + dv_6817d269e892: case0 -0.376/-1.397us both draws
   * per-pass consistent, case2 -0.818/-0.339, case9 -0.474/-0.127,
   * case1/case7 neutral, d9 +0.09/+0.15 reported; rel_l2 digit-identical
   * to 68; probe census 78/0/0 = 68 exactly; record
   * results/20261027T0010Z-tsp-compose-close-flip-prereg.json). */
  /* 72 */ GDN_WIDERT4(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, true, true, "ASRSQV"),
  /* --- 2026-10-28 startup-chain and schedule-order arms, all on slot 72's exact
   * promoted tuple (W8/P2/NA1/E2/M3, ALD+ST4+RPERM+ASAL+CCSEAM, GRID2D+HQFAST,
   * PFW=1+PFV, CM=-1, RSPLIT+PFT, CPA+TSP).  Slot 72 is row 73's identical-code
   * control for 73/74/75/76 and slot 71 is row 77's, so every arm below is one
   * mechanism away from a row already in the table and already on the dispatch.
   *
   * Why these two and not another occupancy or lane-map arm: the sweep in
   * results/20261028T0835Z-profile-baseline-result-dv_d35c04837372.json closes
   * both of those axes on this package (P=4/P=8 wide blocks 30/31/32/34/36 lose
   * by 25-90% everywhere; MINB=4 at LPR=8 slot 54 is catastrophic and LPR=4
   * MINB=4 slot 64 loses by 45-60%), and it separates the ten scored shapes
   * cleanly by TWO properties instead of by size.  The only two shapes at 95-97%
   * of the 6.231 TB/s copy ceiling are fixed:1 and fixed:2, and they are the
   * only two with uniform token counts AND three or more waves.  Everything
   * else is either count-imbalanced (3/4/5/6/7/8/9, at 39-82%) or a single wave
   * (0 and 5).  fixed:1 and fixed:3 are the same 1024 blocks on the same slot
   * and differ by 6.06 vs 4.49 TB/s purely in count uniformity, so the loss is
   * schedule shape and exposed startup, not DRAM and not arithmetic.
   *
   * 73 = 72 + IDXPAR.  With L2 evicted before every scored call the block's
   * first THREE memory round trips are serial - cup/accp, then the source slot
   * (its index comes from accp), then the 32 KiB tile (its address comes from
   * the slot) - and a single-wave grid has no other block's step loop to hide
   * any of them behind.  One sector of speculation removes the middle one.
   *
   * 74/76 = 72 + the two-barrier longest-first remap, bounded to two waves and
   * to four waves.  The bound is part of the mechanism: the imbalance it removes
   * is a fraction of ONE wave's makespan while the ranking is paid by every
   * wave.  dv_8cd01dc6e45d put the crossover between 1.73 waves (fixed 4/6/8
   * gain 13.93/4.36/3.95%) and 3.46 (fixed 3/7 lose 1.9/2.9%), but on the
   * slot-39 package at MINB=2, whose capacities make those 512- and 1024-block
   * grids 1.73 and 3.46 waves; at slot 72's MINB=3 the SAME grids are 1.15 and
   * 2.31 waves, i.e. both have moved down one side of the crossover.  74 and 76
   * bracket it rather than assuming where it landed.
   *
   * 75 = 73 + 74, the composition a dispatch would actually adopt if both win,
   * and the row that says whether they are additive or the same slack twice.
   *
   * 77 = slot 71 (the P=1 band's row, MINB=4) + the remap at two waves.  The
   * band is where fixed:6/8 live and where the sweep shows the imbalance is
   * still 1.49x of ceiling, but slot 71's capacity is SM_COUNT*4 = 592, so a
   * 512-block grid is ONE wave there and the historical `blocks > SM_COUNT*MINB`
   * fire guard never triggers.  The guard now reads `blocks > SM_COUNT`, which
   * is the condition the mechanism actually needs - an SM hosts two or more
   * blocks, so which blocks those are is decided by the dispatch order - and
   * 77 is the row that measures it where the old guard was blind.  Slot 65 keeps
   * its place in the table as the historical LPT2 row; its fire condition moved
   * with the guard, and it is undispatched under all five pick constants, so no
   * promoted behaviour depends on the old bound.
   *
   * NONE of 73-77 changes an arithmetic result.  IDXPAR reads the same ints the
   * dependent load would have read and selects the same entry, and the remap
   * reassigns `r` before cup/accp/idxp are indexed, so every block computes the
   * identical request - only the order the blocks are handed to the SMs moves.
   * Both are gated on the fingerprint anyway: the probe reports out and written-
   * checkpoint digests relative to the control row for every slot it times. */
  /* 73 */ GDN_WIDEV(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, true, true, true, 0, "ASRSQV"),
  /* 74 */ GDN_WIDEV(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, true, true, false, 2, "ASRSQV"),
  /* 75 */ GDN_WIDEV(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, true, true, true, 2, "ASRSQV"),
  /* 76 */ GDN_WIDEV(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, true, true, false, 4, "ASRSQV"),
  /* 77 */ GDN_WIDEV(8, 1, 1, 2, 4, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, false, true, false, 2, "ASRSQVP1"),
  /* --- 2026-10-28 PFL arms: the prefetch sweeps walk 128-byte lines instead of
   * striding region_bytes/threads.  `prefetch.global.L2` warms ONE 128-byte line,
   * and 2*D/P is 128 bytes at P=2 but 256 at P=1, so every P=1 row in this table
   * has been warming half of what its comment claims - 16 of 32 KiB for PFW's
   * next-wave tile and 8 of 16 KiB for PFT's own pass-2 rows.  That is slots 66,
   * 70, 71 and 77, i.e. the whole band the dispatch rewrite above just widened to
   * fixed:3/4/6/7/8.  P=2 slots are unaffected by construction, which is what
   * makes slot 80 an identical-code control rather than a fourth mechanism.
   *
   * 78 = 71 + PFL, the lower sub-band's row (444 < blocks <= 592).
   * 79 = 77 + PFL, the upper sub-band's row (592 < blocks <= 1184).
   * 80 = 72 + PFL at P=2: the addresses are provably the same, so any measured
   *    difference between 80 and 72 is the cost of instantiating another row and
   *    not of the mechanism, and 78-vs-71 / 79-vs-77 have to be read net of it.
   *
   * A prefetch has no destination and writes no byte, and the real loads still
   * perform their own coherency, so all three are bit-identical by construction;
   * the probe fingerprints out and the written checkpoints against the control
   * anyway, because "by construction" has been wrong here before. */
  /* 78 */ GDN_WIDEW(8, 1, 1, 2, 4, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, false, true, false, 0, true, "ASRSQVP1"),
  /* 79 */ GDN_WIDEW(8, 1, 1, 2, 4, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, false, true, false, 2, true, "ASRSQVP1"),
  /* 80 */ GDN_WIDEW(8, 2, 1, 2, 3, true, true, true, true, false, true, true, true, 1, true, -1, true, true, 0, true, true, false, 0, true, "ASRSQV"),
};
constexpr int NUM_VARIANTS = (int)(sizeof(VARIANTS) / sizeof(VARIANTS[0]));
#undef GDN_VARIANT
#undef GDN_WIDE
#undef GDN_WIDEX
#undef GDN_WIDEQ
#undef GDN_WIDEF
#undef GDN_WIDEI
#undef GDN_WIDEG
#undef GDN_WIDEZ
#undef GDN_WIDET
#undef GDN_WIDEY
#undef GDN_WIDER
#undef GDN_WIDERT
#undef GDN_WIDERT2
#undef GDN_WIDERT3
#undef GDN_WIDERT4
#undef GDN_WIDEU
#undef GDN_WIDEV
#undef GDN_WIDEW

/* PROMOTED 2026-10-19 (job dv_fa6a8d312731, 2032 MHz confirmed): slot 11 is the
 * WIDE8 lane map (256 threads, 8 lanes per state row, 4 rows and 2 chunks per
 * thread).  In that one job, against slot 0 - the configuration that scored
 * 2.330x on the official fixed suite - it is faster on all ten fixed shapes
 * (geomean 31.23 us against 39.62 us, -21.2%) and on all ten frozen domain
 * configurations (41.83 against 48.72 us, -14.1%), with 0 mismatched elements in
 * every spot check of both suites.  The two narrower wide maps are within 1% of
 * it (v9 W4/NA1 31.55/42.04, v10 W4/NA2 31.67/42.64) and every non-wide slot is
 * 39.6 us or worse, so the wide map - not a tuning detail inside it - is the
 * mechanism.  Measured causes: ptxas reports 127 registers and ZERO spills
 * against the old default's 166 wanted / 128 allowed / ~38 spilled, executed
 * instructions on case 1 fall 32.97M -> 27.84M, shared loads 1.57M -> 0.93M, and
 * every stall ratio drops (long_scoreboard 2.97 -> 2.18, short_scoreboard
 * 2.28 -> 1.16, lg_throttle 0.21 -> 0.03) while gpc cycles fall 166285 -> 116442. */
/* PROMOTED 2026-10-19, second step (job dv_307406cb1e54, 2032 MHz, all sixteen slots
 * swept on both frozen suites, 0 mismatched elements in every spot check): slot 13
 * adds the flat epilogue (EPIM=2) to the promoted WIDE8 map.  Against slot 11 in the
 * same job it is +2.03% on the fixed ten (31.63 vs 32.29 us geomean, better on all
 * ten shapes: case 3 +4.4%, case 2 +2.8%, case 9 +2.6%, case 5 +2.6%, case 8 +2.4%)
 * and +1.49% on the frozen domain ten (38.39 vs 38.97, better on eight of ten).
 * NCU on case 0: executed instructions 3.72M -> 3.49M, global loads and stores
 * 61,440 -> 52,224 each, shared loads 110,592 -> 92,160, shared bank conflicts
 * 45,572 -> 27,275, gpc cycles 31,760 -> 30,624.
 *
 * REJECTED in the same job, with the reason kept so it is not retried blind:
 *  - CONTIG + 256-bit accesses (measured as slots 12/14 of that job, code since
 *    removed; those slots now hold the CROW variants): -4.36% fixed, -2.16% domain.  The
 *    mechanism worked as designed on the global side - store requests halved
 *    (557,056 -> 294,912) with IDENTICAL sectors (8,650,752) - and still lost, because
 *    sm__inst_executed_pipe_lsu went UP (4.01M -> 4.30M) and, decisively, the
 *    contiguous column ownership makes the shared k/q reads 2-way bank-conflicted:
 *    conflicts 1,297,351 -> 2,345,034 and shared wavefronts 5.03M -> 6.78M, with
 *    mio_throttle 0.25 -> 0.51 and short_scoreboard 0.95 -> 1.22.  The interleaved
 *    map is conflict-free by construction (lane p reads 16 B at byte 16p, so one
 *    LDS.128 phase spans all 32 banks).  A perfect-shuffle swizzle of the shared
 *    k/q layout would remove the conflicts, but the global-side evidence says the
 *    store instruction count is not the limiter - sectors and DRAM bytes are - so
 *    the best case is a wash.  Closed.
 *  - NA=2 at LPR=8 (slot 15): +0.30% fixed, -0.06% domain, inside the noise, and it
 *    costs four registers.  Rejected; NA=1 stays. */
/* PROMOTED 2026-10-19, third step (job dv_094854d33115, 2032 MHz, sixteen slots swept on
 * both frozen suites, 0 mismatched elements in every spot check): slot 14 is slot 13 plus
 * ALD, the aligned-block form of the misaligned state read.  Against slot 13 in the same
 * job it is +0.98% on the frozen domain ten and neutral on the fixed ten (-0.11%, inside
 * the +-2.5% slot-to-slot scatter, which is exactly what the mechanism predicts: ALD lives
 * only in the VEC=false instantiation, so an aligned slot stride never reaches it).  The
 * gain is where it was designed to be - the three odd-padding domain configurations, and
 * only those: d1 +3.6%, d7 +3.9%, d2 +2.5%, against -0.5%..+0.6% on the seven aligned ones.
 * NCU on d2 confirms the mechanism end to end: load sectors 18,690,048 -> 9,252,864,
 * global load instructions 2,174,976 -> 798,720, LSU pipe 5.15M -> 3.77M, lg_throttle
 * 0.70 -> 0.35, long_scoreboard 11.68 -> 10.64, gpc cycles 325,863 -> 313,076, at the cost
 * of +5.3% total instructions (the four funnel shifts) and unchanged store sectors.
 * Bit-exactness against the ladder it replaces is proved locally for every alignment class
 * by experiments/align_check.py (65,408 comparisons, 0 mismatches) before it was ever run
 * on the device.
 *
 * The store side keeps the ladder on purpose: the same two-aligned-blocks trick would
 * write bytes belonging to neighbouring lanes.  Removing the remaining 18.3M store sectors
 * on d2 needs cross-lane assembly of whole aligned 32-byte blocks (one STG.256 per block
 * from half the lanes, two shuffles to fetch a neighbour's edge, and the ladder retained
 * for the first and last partial block of each row so nothing outside the row is ever
 * written), which is the next mechanism and is designed in PROGRESS.md. */
/* NOT PROMOTED 2026-10-20, sixth step attempted (dev jobs dv_c9b6cf80b9e2 and
 * dv_226131bef9f1, official dv_96321c8e4d6a/dv_9c5809f0f762 fixed and
 * dv_894f576de1b6/dv_49fb50be172c domain, all 2032 MHz, all 0 mismatched).  Slot 24
 * is slot 22 plus CSB plus ASA, the cross-lane aligned-block store for the misaligned
 * checkpoint path, and it is the most instructive negative result in this file:
 * **the mechanism worked and the package still lost.**
 * 2032 MHz, all twenty-five slots swept on both frozen suites, 750 of 750 spot cells
 * clean in each): slot 24 is slot 22 plus CSB plus ASA, the cross-lane aligned-block
 * store for the misaligned checkpoint path.  It attacks the largest measured
 * inefficiency left in the kernel - the three odd-padding domain configurations run at
 * 3.12x their own aligned memory ceiling against 1.62x for the seven aligned ones, so
 * the misaligned path costs an extra 1.93x (probe modes 3/4, dv_947dc0ae3a75).
 *
 * Minimum-based geomeans against slot 22 in the same job, twice:
 *   domain, all ten          +1.05%  then +1.31%
 *   domain, odd-padding only +6.25%  then +4.95%   (d1, d2, d7)
 *   fixed, all ten           -0.28%  then -0.34%
 * and the per-config effect reproduces where it is above the noise: d2 -4.1% then
 * -5.0%, d7 -10.7% then -10.2%, while d1 (-2.7% then +1.5%) does not reproduce and is
 * not claimed.  The fixed number is not a regression: ASA is compile-time dead when
 * VEC is true, which it is for all ten fixed shapes, so slots 23 and 24 run the same
 * code there - and they differ by -1.29% and -0.24%, i.e. the identical-code noise
 * floor of this job is wider than the -0.31% being interpreted.  The aligned domain
 * configs give the same floor from the other side (d0 0.985/1.024, d5 0.975/1.015,
 * the other five within 0.8%).
 *
 * NCU on d2 confirms the mechanism end to end (slot 22 -> slot 24): store sectors
 * 14,778,368 -> 10,584,064 (-28.4%, against the -25% the design predicted), load
 * sectors unchanged at 9,252,864, long_scoreboard 10 -> 7 (-33%), lg_throttle -90.9%,
 * mio_throttle -55.6%, gpc__cycles_elapsed 309,696 -> 297,941 (-3.8%) - for total
 * executed instructions +20.4% and global store instructions +23.4%.  That trade is
 * the law in action: seven mechanisms that cut instructions or registers bought
 * nothing, and this one cut SECTORS at a 20% instruction cost and paid.
 *
 * It sits at 128 registers, exactly the MINB=2 cap, with zero spills - which is why it
 * is built on CSB (122) rather than on slot 22 (126): the assembly needs the
 * neighbour's uint4 alongside its own, and only CSB's four freed registers make that
 * fit.  CSB's own timing is neutral, so it is carried here purely as register
 * headroom.
 *
 * **Why it is not the default.**  ASA delivered exactly what it was designed to
 * deliver, on both instruments and reproducibly.  Officially, on the three
 * odd-padding configurations it targets, slot 24 against slot 22 (median of two runs
 * each): d1 35.71 vs 38.27 (-6.7%), d2 156.76 vs 160.79 (-2.5%), d7 32.14 vs 34.68
 * (-7.3%).  In the dev sweeps d2 was -4.1% then -5.0% and d7 -10.7% then -10.2%,
 * against an identical-code noise floor of +-2.5% read off slots 23 and 24 on the
 * aligned configs where ASA is compile-time dead.  That part is settled, and it is the
 * first time anything has moved the odd-padding path since ST4.
 *
 * What kills it is the CSB dependency.  Carrying CSB costs the seven ALIGNED domain
 * configurations, where ASA does nothing: officially they average ~1.4% worse and d4
 * (pad 32, sub-wave, 128 blocks) is 46.10 against 37.88 us, +21.7%, in both slot-24
 * runs against both slot-22 runs.  The net on the official engine - the instrument of
 * record - is slot 24 domain 41.70/41.80 us against slot 22's 41.27/40.74, and fixed
 * 35.22/34.99 against 34.57/34.89: **worse on both phases in two runs each.**  The dev
 * sweeps disagree (slot 24 +1.05% then +1.31% on domain, -0.28% then -0.34% on fixed),
 * so the two instruments straddle zero at the 1-2% level, which is where both are
 * unreliable.  A promotion has to be a clear win; this is not one, so the default
 * stays slot 22 and slot 24 stays compiled as the record.
 *
 * The remaining prize is larger than what this captured, and the path is now specific:
 * 255 of the 256 aligned blocks in a warp's 4096-byte region are fully interior, so
 * the floor is 128 sectors per warp-step against the ~136 this design can reach and
 * the 512 it replaced.  Reaching it needs the cc-boundary and row-boundary seams
 * assembled too, which is another 4-8 live registers.  **The binding problem is
 * therefore not the assembly - it is that ASA cannot be had without CSB's four
 * registers, and CSB is not free.**  Either find four registers that cost nothing on
 * the aligned path, or make the assembly fit in two.
 *
 * RESOLVED the next turn, by the second of those two options - see ASAL and slot 26.
 * The assembly did not need to shrink to two registers; it needed to stop paying for
 * a neighbour uint4 and a separate output uint4, which are eight live values it never
 * needed simultaneously.  Assembling the block in place over the lane's own words,
 * descending, and switching on b16 so each arm fetches only the words it uses, fits
 * the same stores in 128 registers with 0 spills and no CSB - so the aligned configs
 * stopped paying for it.  Slot 24 stays compiled as the record of the mechanism and of
 * the package that lost; slot 26 is the promoted form.  What is still open is the seam
 * work above: 255 of 256 blocks interior is a further -68% sectors against this
 * design's -28%, and it still needs registers this slot no longer has to give. */
/* PROMOTED 2026-10-19, fifth step (jobs dv_5a225f0c35b0 and dv_d8ca1eaaf62e, both
 * 2032 MHz, all twenty-four slots swept on both frozen suites, 720 of 720 spot cells
 * clean in each): slot 22 is slot 12 plus RPERM, the row-permuted layout of the two
 * shared arrays the step loop addresses per row.  It is the first mechanism promoted
 * since ST4 and the only one that reduced shared WAVEFRONTS rather than shared
 * instruction count.
 *
 * Minimum-based geomeans against slot 12 in the same job, twice:
 *   fixed   29.56 vs 30.31 us (+2.52%)  then  30.48 vs 30.99 (+1.68%)
 *   domain  34.64 vs 35.29 us (+1.90%)  then  35.73 vs 36.35 (+1.75%)
 * and it beat slot 12 per case on 9-10 of the ten fixed shapes and 8-9 of the ten
 * domain configurations in each job.  Two jobs were required: CSB looked equally good
 * in its first (+1.87% fixed, best of seven wide slots on both statistics) and
 * measured -0.00% in its replicate, so a same-job win is now only a promotion
 * candidate once it reproduces.
 *
 * NCU confirms the designed mechanism on both a large aligned shape and the largest
 * odd-padding one (case 1 / domain d2, slot 12 -> slot 22):
 *   shared load wavefronts        2,167,029 -> 1,970,452 (-9.1%) | 979,739 -> 879,158 (-10.3%)
 *   shared load instructions        933,888 ->   737,280 (-21.1%) | 557,056 -> 458,752 (-17.6%)
 *   shared store instructions       401,408 ->   229,376 (-42.9%) | 200,704 -> 114,688 (-42.9%)
 *   total shared wavefronts       4,556,049 -> 4,416,441 (-3.1%) | 2,376,273 -> 2,301,240 (-3.2%)
 *   gpc__cycles_elapsed             117,305 ->   116,435 (-0.7%) | 310,483 -> 309,699 (-0.3%)
 *   registers                            126 ->        126 (0 spills)
 * The four scalar `vm` reads per thread-step became one vector read and the four
 * scalar `rawsh` publishes became one 16-byte store.  Total instructions were flat
 * (+0.0% / +0.6%), which is the point: five mechanisms that cut instructions without
 * cutting wavefronts (CONTIG, QK32, PACK, PUB1, CSB) bought nothing, and this one cut
 * wavefronts with the instruction count unchanged and paid.
 *
 * Cost, measured rather than assumed: shared STORE wavefronts rose 12.6% / 11.9% and
 * total bank conflicts 6.2% / 20.0% (load conflicts were flat at +0.0% and -7.7%),
 * because the prologue's v store became four scattered half-word writes instead of one
 * uint2.  That is a once-per-block cost against a per-step saving, and the net is
 * still negative wavefronts, but it is the obvious next thing to attack.
 *
 * REJECTED alongside it: slot 23 (CSB + RPERM) is consistently WORSE than slot 22
 * alone (+1.45%/+0.45% then +1.86%/+0.81% against slot 12), so CSB does not stack
 * with RPERM even though it is register-compatible.  Slot 21 (CSB alone) stays as the
 * register-headroom enabler for the misaligned-store assembly, which needs 4 more
 * registers than slot 22's 126 and CSB provides exactly that (122). */
/* PROMOTED 2026-10-19, fourth step (job dv_d0c29dfcd961, 2032 MHz, sixteen slots swept
 * on both frozen suites, 0 mismatched elements in every spot check of every slot): slot 12
 * is slot 14 plus ST4, the widest-aligned-access store ladder for misaligned windows.
 *
 * Domain ten: 36.18 us geomean against 36.96 for slot 14 (+2.11%) and 37.21 for slot 13,
 * with the gain exactly where the mechanism lives and nowhere else - d7 +10.2%, d2 +5.0%,
 * d1 +3.6% (the three odd-padding configurations) against -0.1%..+1.3% on the seven
 * aligned ones.  Fixed ten: -0.62%, which is noise by construction: all ten fixed shapes
 * have 16-byte-aligned slot strides, so the ST4 branch is dead code there and slots 12
 * and 14 run the same instructions (the per-cell scatter in that comparison, one cell at
 * -12.3%, is a useful calibration of single-cell noise inside one job).
 *
 * The controlled padding-parity pairs, run twice in the same job under GDN_V=14 and
 * GDN_V=12 with identical shapes and seeds, agree: odd pad at one step 165.92 -> 157.70 us
 * (-5.0%) and at four steps 62.91 -> 58.82 us (-6.5%), with the even-pad controls at
 * -0.1% and -2.2%.  The odd/even price of the misaligned path falls from 2.18x to 2.08x
 * at one step and from 2.31x to 2.21x at four.
 *
 * NCU on d2 confirms the chain: store sectors 18,317,312 -> 14,778,368 (-19.3%), store
 * requests 1,048,576 -> 851,968, store instructions 1,191,936 -> 978,944, LSU pipe
 * 3.77M -> 3.56M, lg_throttle 0.32 -> 0.11, long_scoreboard 10.53 -> 9.83, gpc cycles
 * 322,973 -> 305,431 (-5.4%), duration 159.97 -> 150.43 us (-6.0%), for +1.3% total
 * instructions.  Load sectors are unchanged at 9,252,864 (that was ALD's half).
 *
 * Bit-exactness was proved locally first: experiments/align_check.py compares st8 with and
 * without ST4 over 9,427,968 byte comparisons at all sixteen alignment classes and both
 * round-trip the source half-words exactly. */
/* PROMOTED 2026-10-20, sixth turn (jobs dv_7ee140bb3519 and dv_d42fdd0851f4, both
 * 2032 MHz confirmed, both sweeping every slot on both frozen suites with 810 spot
 * cells and 0 mismatched elements each): slot 26 is slot 22 plus ASAL, the
 * aligned-block store assembled IN PLACE - ASA's sectors without CSB's registers.
 *
 * This is the resolution of the ASA rejection.  ASA cut d2's store sectors 28.4% and
 * the three odd-padding domain configurations by 6.7/2.5/7.3% on the official engine,
 * and still could not be promoted, because it cost six registers and slot 22 had two
 * free, so the only way to build it was on CSB - and CSB costs the seven ALIGNED
 * domain configurations, where ASA is dead code, more than ASA wins back (slot 23,
 * which has no ASA at all, is +5.6% on d4 and +1.7..+2.4% on d3/d8/d9 against slot 22
 * in dv_c9b6cf80b9e2's minima).  ASAL removes the two things ASA spent the registers
 * on: the block is concat(n,u) shifted right, so writing block word j over the lane's
 * own word j from j=3 DOWN clobbers only already-consumed words and no output uint4
 * exists; and switching on b16 rather than on k makes the shift a literal, so each arm
 * fetches only the neighbour words it uses - peak three live values against ASA's eight.
 *
 * Registers: 128 / 0 stack / 0 spills in both jobs, against slot 22's 126 and slot
 * 24's 128.  It fits at the MINB=2 cap WITHOUT CSB, which is the whole point: the
 * aligned configs keep slot 22's codegen budget and only the misaligned path changes.
 *
 * Two independent jobs, both against slot 22 in the same job.  MINIMA, which is the
 * statistic the dev sweep has to be argued on (its medians are contaminated exactly
 * like the official engine's), with the median geomeans beside them:
 *   domain geomean  34.028 vs 34.865 (+2.40%)  then  33.753 vs 34.756 (+2.89%)
 *                   [medians 35.606 vs 35.942 (+0.93%), 35.382 vs 35.998 (+1.71%)]
 *   fixed  geomean  29.710 vs 29.700 (-0.03%)  then  29.333 vs 29.415 (+0.28%)
 *                   [medians 31.243 vs 31.321, 30.985 vs 31.012 - both within 0.3%]
 *   d2 minimum     144.35 vs 158.72 (-9.1%)   then  143.10 vs 157.54 (-9.2%)
 *   d7 minimum      26.62 vs  29.79 (-10.6%)  then   27.01 vs  29.60 (-8.7%)
 *   d1 minimum      27.81 vs  28.42 (-2.1%)   then   26.62 vs  28.54 (-6.7%)
 * and the seven aligned configurations flat in both (d4 34.85->34.78, d8 36.64->36.67,
 * d9 60.38->60.38, d3 63.42->63.36, d6 26.59->26.59) - no CSB-shaped regression.
 * The fixed suite is a wash because it has to be: every fixed shape is aligned, so it
 * runs the VEC=true instantiation, in which slots 22, 25, 26 and 27 have the same
 * static instruction count and the same 126 registers.
 * Slot 26 also beat slot 24 (ASA+CSB) on the domain minimum geomean in both jobs,
 * +1.94% and +1.67% (+1.45%/+1.10% on medians), and on d2 itself by 5.4% at IDENTICAL
 * store sectors, which is the in-place form's shorter live ranges showing up as better
 * scheduling rather than as registers.
 *
 * CORRECTION, seventh turn: the two geomean lines above were originally quoted as
 * minima and were in fact the median geomeans, because sweep_table.py computed its
 * single geomean from the med column while its docstring said otherwise.  The per-case
 * minima were always read from the min column and are unchanged.  The tool now prints
 * both, labelled; on the correct statistic this promotion is stronger than recorded
 * (+2.40%/+2.89% domain rather than +0.93%/+1.71%) and brackets the official
 * same-window A/B's +2.68% on minima, which the median figures did not.
 *
 * The controlled padding-parity pairs, all four slots in one job (dv_d42fdd0851f4,
 * 2032 MHz throughout), price the mechanism directly: odd pad at one step 157.73 ->
 * 144.35 us and at four steps 58.43 -> 51.55, i.e. -10.1% on the odd pair, with the
 * even-pad controls at 0.998 and 0.985 - neutral, as they must be for dead code.
 *
 * NCU d2, slot 22 -> 26: store sectors 14,778,368 -> 10,584,064 (-28.4%, identical to
 * ASA's to the digit), load sectors unchanged at 9,252,864, duration -4.1%; stall
 * ratios long_scoreboard 9.92 -> 6.82, barrier 2.45 -> 1.72, lg_throttle 0.10 -> 0.01,
 * mio_throttle 0.35 -> 0.04, sm__cycles_active -4.8%, with the cost exactly where the
 * design put it - branch_resolving 0.26 -> 0.66 and no_instruction 0.26 -> 0.47, the
 * seven-arm switch.
 *
 * Bit-exactness was proved locally before either job, and both proofs are
 * negative-tested: map_check.check_asal parses the seven arms out of this header and
 * runs them statement by statement across a simulated warp - so the shuffle reads the
 * neighbour's live register and the fetch-before-overwrite order is load-bearing - and
 * asserts every arm equals the byte model, equals asa_block, covers exactly the seven
 * legal b16 values and holds at most three temps; align_check replays the real
 * compiled arms and the whole store phase, ladders first, over a guarded pool (22,960
 * comparisons, 0 bad). */
/* PROMOTED 2026-10-20, eighth turn (jobs dv_1ad921fb994b and dv_99664833676e, both
 * 2032 MHz, both sweeping every slot on both frozen suites with 870 spot cells and 0
 * mismatched elements each): slot 28 is slot 26 plus CCSEAM, the chunk-boundary
 * aligned block.  It replaces two of every row's four edge ladders with one 16-byte
 * aligned store, and it is the last piece of the misaligned store path that pays.
 *
 * Sectors, which is the only currency this kernel has (see the law in CONTINUE.md):
 * d2 store sectors 10,584,064 -> 7,897,088 (-25.4%) and d1 2,033,664 -> 1,505,280
 * (-26.0%), load sectors unchanged at 9,252,864.  That is -46.6% cumulatively from
 * slot 22's 14,778,368 and takes d2 from 2.5x its logical store volume to 1.88x.
 *
 * Timing, two jobs, against slot 26 in the same job, on minima: domain geomean
 * +0.50% then +0.24% (medians +0.65% then +1.74%); d2 -1.5% then -2.1%; d7 -6.3%
 * then -2.3%; d1 +0.1% then +2.9%.  The shape-controlled padding-parity pairs, which
 * hold everything but the padding fixed and are the instrument that isolates this
 * mechanism: odd pair -2.3% then -4.7%, even-pad control 0.991 then 1.001.
 *
 * Why a +0.4% frozen-domain effect is worth promoting when larger candidates were
 * declined.  First, the downside is provable rather than measured: `vec` is true
 * exactly when the state pointer and stride(0)/stride(1) are 16-byte aligned, so all
 * ten fixed shapes and the seven aligned domain configurations run the VEC=true
 * instantiation, and there slot 28's static opcode delta against slot 26 is
 * **+0 instructions** with 126 registers for both - they cannot move.  Second, the
 * frozen domain dilutes this mechanism: three of its ten configurations have an odd
 * state_slot_padding, while roughly half the legal range does, and on the instrument
 * that holds shape fixed the gain is 2-5%, not 0.4%.  The frozen ten are a sample,
 * not the target.
 *
 * The cost is recorded so it is not rediscovered: d1 is flat-to-1.5% worse in both
 * jobs, and the front end pays for the arms - branch_resolving 0.66 -> 0.77,
 * no_instruction 0.48 -> 0.59, VEC=false static instructions 4,729 -> 5,582 (+18%),
 * store requests +10.6%.  d1 is the smallest of the three odd configurations and the
 * one whose store phase is the smallest share of its runtime, which is consistent with
 * code size costing small latency-bound shapes what sectors save large store-bound
 * ones.  Registers 128 / 0 stack / 0 spills on VEC=false, 126 / 0 / 0 on VEC=true.
 *
 * Proved locally before either job, both proofs negative-tested: map_check.
 * check_ccseam parses the seven arms out of this header and proves the geometry for
 * every LPR and b16 - all fifteen blocks inside the 256-byte row so nothing can reach
 * the previous row, the previous head or a neighbouring slot; pairwise disjoint;
 * exactly covering the row with only the two row-edge ladders left; and that the
 * seam's two sources really are the two windows adjacent to its block - then runs
 * each arm against the byte model concat(n,v) >> (128-8*b16) and asserts the per-arm
 * word counts.  align_check replays a whole row in the device's instruction order over
 * a guarded pool: 24,192 byte comparisons, 0 bad. */
/* PROMOTED 2026-10-21 (job dv_a3bf600bddb3, 2032 MHz, both frozen suites, two
 * passes, 40 cells per candidate measured against the identical-code control slot
 * 200, zero tolerance violations in any cell): slot 39 is slot 28's exact package
 * plus two startup-path mechanisms.  GRID2D launches dim3(hv, nreq) so the block's
 * (request, head) pair comes from blockIdx instead of a signed division and
 * remainder by the runtime head count.  HQFAST replaces the SECOND runtime
 * division, j / (hv/h), with a comparison against the value-head ratios and a
 * shift, keeping the general division as the fallback for any legal hv/h.
 *
 * Both sit on the block's critical path to its first global load, because r indexes
 * cu_seqlens, num_accepted_tokens and state_indices while hq indexes mixed_qkv: no
 * address can be formed until they are done.  PC sampling on fixed case 0 (job
 * dv_d7db602e8e4b, warm caches, 2032 MHz) is what pointed here - of 732 stall
 * samples the hottest sites were the startup chain (UMOV UR4 10.8%, an IMAD-based
 * BF16 widen 9.3%, PRMT 6.4%, the IABS of the division sequence 5.2%, LDC 3.1%, the
 * 2-byte dt_bias load 2.7%, ISETP 2.6%) and not one instruction of the recurrence,
 * in a kernel sitting at 19.7% compute, 22.9% DRAM, 29.7% L1 and 21.3% occupancy.
 *
 * Candidate against the identical-code control in the same job, on minima:
 *   fixed ten   pass 1  +2.32%   pass 2  +0.77%
 *   all twenty  pass 1  +1.45%   pass 2  +0.50%
 * and on the pipelined device-only timing, which contains no host path at all,
 * +1.13% over all 40 cells with 34 of them positive.  ptxas: 123 registers against
 * slot 28's 126, 0 spills, 32 B stack.  Nothing else about the launch moves - the
 * 2-D grid linearises to the same dispatch order (x + y*gridDim.x == r*hv + j with
 * the head fastest), so locality, the tail and which slot a block touches are
 * exactly what they were.
 *
 * REJECTED alongside it, same job, same control: IDXPAR (prefetch the whole
 * state_indices row together with cu_seqlens instead of after it) is +0.71%/+0.55%
 * alone but drags the combination to +0.28%/+0.23%, so eight speculative uniform
 * loads cost more than the round trip they remove once the divisions are gone; and
 * NOGATE (stop staging the output-gate vector, read it in the epilogue) is -4.5% -
 * the recurrence never reads the gate, but the epilogue's 2-byte global loads cost
 * far more than the one 16-byte staged load per warp that they save. */
constexpr int DEFAULT_VARIANT = 39;
/* Portable path used when the slot stride itself breaks 16B alignment; ld8/st8
 * still vectorise per address, so this only fixes the compile-time assumption. */
constexpr int SCALAR_VARIANT = DEFAULT_VARIANT;
/* The sub-wave launch pick is now a no-op: it exists only so that a measured
 * sub-wave-specific configuration can be reinstated without touching pick_cfg.
 *
 * History: below one wave the same SMs are busy either way, so the only question
 * used to be how many threads work on one tile, and the 512-thread P=4 block
 * (slot 2) beat the 256-thread P=2 default on d5 (48 blocks, 8 steps) across five
 * jobs.  The WIDE map supersedes that reasoning because it improves the sub-wave
 * cases too, without more threads per tile: in dv_fa6a8d312731 slot 11 beat slot
 * 2 on both sub-wave domain configurations (d0 9.76 against 10.05 us, d5 16.48
 * against 17.02) and on both sub-wave fixed shapes (case 0 15.94 against 19.94,
 * case 5 15.97 against 18.94), while slot 2 loses 20-28% above one wave.  A wide
 * 512-thread block (P=4, LPR=8) is the untested candidate for a reinstated pick. */
constexpr int SMALL_GRID_VARIANT = DEFAULT_VARIANT;

/* PROMOTED 2026-10-21: above one wave of resident blocks the pick is slot 45 -
 * slot 39's exact tuple plus a two-stage L2 prefetch of the next wave's inputs,
 * the second stage fenced by coverage in `Params.pfvg`.  Below one wave it is slot
 * 39 itself.  Both stages were ranked by interleaved same-job A/B on all twenty
 * frozen configurations against identical-code control 207, and then confirmed -
 * or, for the unfenced second stage, refuted - in the scored window.
 *
 * Stage 1, promoted (job dv_8ef9e8542f6f, 7 rounds x 15 engine samples per cell,
 * control at +0.18% fixed / +0.13% domain): slot 43 is slot 39's exact tuple plus
 * PFW=1 - every block warms the 32 KiB source tile of the block one wave later in
 * the dispatch order with `prefetch.global.L2`.  Whole-suite geomean +2.05% fixed /
 * +2.29% domain, largest cells d2 138.24 -> 123.94 us and fixed:2 52.22 -> 48.42;
 * reproduced at +2.02% / +2.35% by dv_40f64d09c04d (9 x 15, control +0.01% /
 * +0.08%).
 *
 * PFW=2 (slot 44) gave +1.41% / +1.16%: one wave is the better lead, because the
 * constraint is eviction, not arrival.  NCU on the same two shapes says the
 * mechanism did what it was built for - long_scoreboard 32.75 -> 26.12 on fixed:2
 * and 44.66 -> 32.02 on d2, kernel duration 47.78 -> 43.81 and 140.03 -> 120.10
 * us, memory throughput 40.7 -> 44.3% and 27.3 -> 32.5% - and that it exposed the
 * next one: `barrier` is now the second-largest stall on fixed:2 at 21.62% and
 * grew on d2 from 11.55 to 12.92, because the prologue's other cold reads are the
 * four 256-byte q/k/v/gate slices that the staging warps pull and that no other
 * block touches.
 *
 * STAGE 2, FENCED BY COVERAGE.  Slot 45 is slot 43 plus PFV, which also warms
 * those q/k/v/gate slices - 8 lines of 128 bytes per token is exactly the
 * target's staged input, so the extra traffic is the target's own and 3.1-25% of
 * the tile's bytes.  Unfenced it was promoted as best/v3 on three interleaved
 * sweeps (+4.31/+3.46, +4.01/+3.38, +2.90/+3.19 against slot 39, bit-identical,
 * same registers) and then rejected by the scored window, reproducibly:
 *
 *   domain geomean  slot 43: 38.67, 38.63, 38.15   slot 45: 39.17, 39.51
 *   domain:d4       slot 43: 38.02, 36.98, 37.89   slot 45: 42.64, 42.30  (-15%)
 *   domain:d8       slot 43: 35.84, 35.68, 35.84   slot 45: 41.94, 40.45  (-14%)
 *   fixed ten       slot 43: 33.24, 33.24          slot 45: 33.50, 33.18  (a tie)
 *
 * What separates them is coverage, and coverage is the mechanism's own dose: the
 * benefit is proportional to the fraction of blocks whose target exists,
 * (nreq-pfd)/nreq, while the cost is paid by every block.  Across the three
 * sweeps the eight configurations at <=42% coverage average +0.2% and the eight at
 * >=68% average +3.1%, and in the scored window the three worst cells are exactly
 * the three lowest-coverage multi-wave grids - d4 25%, d8 37.5%, d6 37.5% at
 * -10.8%, -13.0% and -7.3% - while every cell at >=68% held its gain (fixed:1
 * +1.6%, fixed:2 +2.6%, fixed:3 +1.2%, d2 +1.5%, d3 +0.3%, d9 +4.4%).  Coverage
 * >= 50% is exactly `nreq >= 2*pfd`, i.e. at least two full waves of resident
 * blocks, which is the same kind of geometric statement as the one-wave bound
 * below and not a threshold fitted to cells: it is where half the grid can
 * participate.  The host computes it into `Params.pfvg` and PFV reads it, so the
 * low-coverage grids run slot 45's binary with the vector stage switched off -
 * which costs nothing measurable, slot 45 compiling at the same 123/128 registers
 * and zero spills as slot 43 and slot 39.
 *
 * Fenced, recomputed from the same three sweeps: +3.66% fixed / +3.09% domain and
 * +3.68% / +3.31%, against +2.31/+2.18 and +2.05/+2.38 for slot 43 alone and
 * +4.13/+3.34 and +3.96/+3.36 unfenced - i.e. the fence keeps most of stage 2 and
 * gives up 0.3% of it, on the cells where the scored window says it was never
 * there.  results/official-vs-sweep-20261021/discrepancy.md holds the control run
 * and the full table.
 *
 * The pick is the geometry the mechanism is defined by.  Below one wave of
 * resident blocks no block has a later block to warm, `pfr < nreq` fails for
 * every block, and experiments/pf_model.py computes a provable 0% coverage - and
 * the measurements put exactly that at a small loss (d0 -1.24%, d7 -0.50%,
 * fixed:5 -0.45%), which is code size, not the mechanism.  So those four
 * configurations keep slot 39's own instantiation and cannot move at all.  This
 * is a block-count rule, not a shape, case or seed lookup: it is derived from
 * nreq*hv against SM_COUNT*MINB, the two quantities the launch already computes,
 * and it generalises to every grid in the supported range. */
/* PROMOTED 2026-10-22 (dv_267060157574, 6 rounds x 11 samples, identical-code
 * control at +0.09% fixed / +0.21% domain): above one wave the pick is slot 56,
 * slot 45's exact package plus RSPLIT at MINB=3.  Against slot 45 in the same job
 * it is **+9.00% fixed and +5.99% domain over slot 39, where slot 45 reads +3.46%
 * and +3.12%** - about 5.4 points of fixed geomean and 2.8 of domain - with d2 at
 * 121.86 -> 105.46 us, fixed:0 15.36 -> 13.31, fixed:2 46.53 -> 44.03 and every
 * multi-wave cell positive.  results/vsplit-20261022/.
 *
 * It is the occupancy curve cashed in.  MINB=3 at the full tile costs 312-328 B of
 * stack and 38% on every cell (slot 53); MINB=1 spill-free is 21% slower than
 * MINB=2 (slot 55), so occupancy is worth ~21% per doubling; and splitting the rows
 * halves the register tile, which reaches 3 blocks/SM at **78 registers with zero
 * spills** on VEC and 80 with 24 B of stack on the scalar instantiation - a
 * thirteenth of the spill that made slot 53 lose.
 *
 * The threshold stays at SM_COUNT*2 and its meaning changes, which the cells
 * confirm rather than assume.  Below 296 blocks the unsplit kernel already fits in
 * one wave, so the split's duplicated per-step work (the k/q staging reads, the
 * scalar record, the rr butterfly, the loop induction: slot 57 against 45 at the
 * SAME occupancy is +0.96% fixed / +1.92% domain, i.e. it costs 2.5 and 1.2 points)
 * is uncompensated - d7 (192 blocks) reads -7.17% and d0/d5 -0.7/-0.9%.  At 296 and
 * above the split either removes a wave outright (384 blocks: one wave of 444
 * against two of 296, worth +15.18% on fixed:0) or raises occupancy 24% -> 37.5%
 * (d2 +17.64%).  So the four sub-wave and middle-band configurations keep slot 39's
 * instantiation exactly, and cannot move at all.
 *
 * NOT bit-identical, and the bound is measured rather than argued: the row-square
 * sum goes from 8 leaves to 16, which reassociates a sum of 128 non-negative
 * squares and moves `out` by at most one BF16 ULP - 0 tolerance violations on all
 * twenty configurations, max abs diff 2.0e-3, worst element 8.4% of its
 * atol 0.01 + rtol 0.05 budget.  The state checkpoints ARE bit-identical: the
 * k-reduction runs along columns and the split is along rows, so pred, delta and
 * the rank-1 update keep their exact tree.
 *
 * PROMOTED 2026-10-23 (v2 era): the pick moves 56 -> 58 = 56's tuple + PFT,
 * the tail self-prefetch.  Under restore-l2-evict-event-v2 every timed window
 * starts cold, and the blocks whose next-wave prefetch target does not exist
 * (r >= nreq - pfd: the last wave, or EVERY block of a single-wave grid) pay
 * the RSPLIT pass-2 tile flight cold, with the resident blocks of a single-
 * wave grid in near-lockstep so the MINB=3 absorption has no phase diversity
 * to work with - the largest single long_sb site on the cold f2/f0 pcsamp
 * (dv_d331ce30238e; derivation results/20261023T1730Z).  Two independent
 * rotated 2-pass A/B jobs on the v2-aligned instrument, both far over the >1%
 * decision bar: dv_6a550a272a17 fixed-10 geomean 58/56 = 0.9687, domain
 * 0.9697; dv_79e85fb1485d 0.9715 / 0.9746.  Allocator-neutral (the VEC=true
 * instantiation is 78 registers / 0 spills, identical to 56; the VEC=false
 * pair is identical at 80 + 40/24 B), 32/32 spot cells clean on both slots
 * across the two jobs INCLUDING odd-padding d2, which exercises 58's VEC=false
 * path.  Wins track the self-prefetch fraction: f0 -11.8/-11.4% (every block
 * self, cross-pass exact), the 512-block band f6 -8.1/-8.2, f8 -6.5/-8.3,
 * f4 -6.4/-3.4, d4 -12.5/-10.4, d8 -11.2/-8.0; the heavy multi-wave cells
 * where most blocks keep a valid target are neutral within ~0.5% (f1/f2/f9/
 * d1/d2/d3/d6/d9 across the two jobs).  No cell carries a cross-job-consistent
 * regression: job 1's f7 +2.6% read +0.1% in job 2 (pooled +1.3%), and job
 * 2's f9 +2.4% (one pass's draw) read +0.0% in job 1.  Records:
 * results/20261023T1805Z-pft-ab-dv_6a550a272a17.json +
 * results/20261023T180544Z-265983-l20d.json. */
/* PROMOTED 2026-10-26 (this phase, arm a): 67 = slot 58's exact tuple +
 * pf2=2, the self pass-2 arena-reload warm ported from arm a's pre-join
 * lineage.  Same-job A/B dv_5ba7036214b6 + identical replication
 * dv_474c98f94424: outputs BITWISE IDENTICAL, 78 regs/0 spills (zero cost
 * over 58), f1 minima -1.2..-2.0us and pipe medians cand-favoring in all
 * four passes, f2/f9 neutral, regression scans clean; closed on the
 * band-invariance rule (both jobs weight-divergent in the f1 event-median
 * continuum with sign+class in agreement across every quantile).  Record:
 * results/20261026T1545Z-pf2-band-invariance-close-adopt.json
 * RE-SCOPED 2026-10-26 (later): since the CPA-P2 adoption (below) 67 is the
 * SCALAR multi-wave pick - its pf2=2 warm is exactly what the odd-padding
 * grids keep, because CPA's cp.async staging is dead code in the VEC=false
 * instantiation and the full-20 confirmation measured PF2's scalar benefit
 * on d2 at +1.03us mean (dv_3ca8c71f57ea). */
constexpr int MULTI_WAVE_VARIANT = 67;
/* PROMOTED 2026-10-26 (this phase, arm a): 68 = slot 58's exact tuple + cpa,
 * the thread-private cp.async pre-staging of the pass-2 state half - the VEC
 * multi-wave pick.  Two identical-command draws (dv_dafa261558bb +
 * dv_738ac327576b) + the full-20 confirmation (dv_3ca8c71f57ea): outputs
 * bitwise identical to 58 on every spot (cp.async is byte transport through
 * the same up8), census 78 regs / 0 spills / 0 stack = 58 exactly (the
 * staging costs ZERO registers), 68-vs-58 negative on all six A/B cells in
 * both draws (case1 -2.8..-3.1us = -4.7..-5.2%, d9 -2.4..-2.8, case2 -0.64
 * ..-0.70 - the first mechanism to move the event-pinned 44.0 level, case7
 * -0.48..-0.71, case9 -0.37..-0.55, case3 -0.24..-0.27); closed on the
 * band-invariance rule (both draws weight-divergent on case1 with sign+class
 * agreement across every quantile: core +3.05/+3.07, P25 +4.06/+4.05) and
 * ADOPT-GATE PASS on the 68-vs-67 pair in draw 2.  Beats the previous pin 67
 * on 5 of 6 cells in both draws; case9 uniquely prefers 67 (+0.27..+0.45us,
 * reported, still -0.9..-1.3% better than certified 58-pin).  The pick is
 * VEC-SCOPED: scalar grids keep 67 (CPA is dead code there; flipping them
 * would only give up PF2 - d2's +1.03us).  Records:
 * results/20261026T1930Z-cpa-p2-prereg.json (mechanism + budget),
 * results/20261026T2030Z-cpa-band-invariance-close-adopt-decision.json
 * (two-draw close), results/20261026T2050Z-cpa-full20-confirmation-adopt
 * .json (full-20 + the scoped-flip decision).
 * SUPERSEDED 2026-10-27: the VEC multi-wave pick moved 68 -> 72 (68's tuple
 * + tsp, the pass-1 tile staging composed on top) on the two-draw
 * composition close dv_9794e563e288 + dv_6817d269e892 (case0 -0.376/-1.397
 * both draws, case2 -0.818/-0.339, case9 -0.474/-0.127, case1/case7/d9
 * neutral-within-bounds, bitwise identical, 78/0/0 census); 68 stays in the
 * table as the composition A/B's control row.  Record:
 * results/20261027T0010Z-tsp-compose-close-flip-prereg.json. */
constexpr int CPA_VARIANT = 72;
/* One wave of resident blocks for the UNSPLIT kernel, which is what the sub-wave
 * side of this threshold still runs: slots 39, 43 and 45 all request MINB=2 and all
 * three achieve it (123 and 128 registers against the 128 that
 * __launch_bounds__(256,2) allows, zero spills).  Slot 56 requests MINB=3 and
 * achieves it (78/80 registers - and the pick's slot 58 is allocator-identical
 * on both instantiations), so its own one-wave capacity is SM_COUNT*3 - but
 * the threshold is deliberately left at SM_COUNT*2, because between the two the
 * split removes a wave rather than needing one (see fixed:0 above). */
constexpr int64_t MULTI_WAVE_BLOCKS = (int64_t)SM_COUNT * 2;

/* The P=1 band.  Slot 66 runs 128 threads and a 64-float tile, so four blocks are
 * resident per SM against slot 58's three: capacity 4*SM_COUNT against 3*SM_COUNT.  A
 * grid strictly between the two collapses from two waves to one, and that is the only
 * regime where the extra stream pays for P=1's doubled per-thread work per step.
 * Measured rather than argued: 440 blocks -4.48%, 448 +8.95%, 592 +8.26%, 600 -6.17%,
 * so BOTH edges sit on the derived constants to within eight blocks
 * (results/p1-hv-edges-20261026/).
 *
 * `width` bounds the step count through the frozen invariant max(lengths) <= width, and
 * the gain is monotone decreasing in it - +13.24 / +17.63 / +10.30 / +8.45 / -0.11 /
 * -3.34 / -4.86 at widths 1/2/3/4/5/6/8, all at 512 blocks - crossing zero between 4
 * and 5 (results/p1-width-sweep-20261026/).  It is an upper bound, so a cell whose
 * width exceeds its true max(lengths) is merely left on the old pick, never moved onto
 * a losing one.
 *
 * `vec` is a SAFETY predicate here, not a performance one.  Slot 66's VEC=0
 * instantiation spills - 128 registers, 32 B stack, 44/64 B spill stores/loads - where
 * its VEC=1 one is 125/0/0/0, and scalar_substitute() passes any non-DEFAULT slot
 * through untouched, so without this term the band would dispatch the spilling kernel at
 * odd state_slot_padding.  That path measured +8.34% once, on one cell, and is not
 * relied on.  Note this is a DIFFERENT reason from the one W4 refuted: W4 showed vec
 * does not change the P=1 gain ratio, which is true and is why this term is not claimed
 * as a performance predicate.
 *
 * `hv` is deliberately absent.  Slot 66's median is 23.55 us on all eight hv
 * factorizations of 480 blocks - hv = 4, 6, 8, 12, 16, 24, 32, 48 across all four tp
 * values - identical to two decimals in each of two jobs, and never above slot 58's
 * (results/p1-hv-replicate-20261026/).  blocks sets the wave count; hv only factors it.
 *
 * BAND ROW UPDATED 2026-10-27: the pick is slot 71 = slot 66's exact tuple +
 * tsp (the thread-private cp.async staging of the PASS-1 tile).  The band is
 * single-wave, so PFW's next-wave warm never fires there and PFT warms only
 * the pass-2 half - the pass-1 cold flight was the band's top stall class
 * (case6 NCU profile dv_88d06c742419: long_sb 1.69 at 18.8% occupancy).
 * Measured: 71 beats 66 on BOTH band cells (case6 -0.999us/-4.86%, case8
 * -0.764/-3.69%, per-pass consistent, bitwise identical, 125/0/0 census) in
 * dv_37c6db8acaa3; slot 66 stays in the table as the band A/B's control row
 * and every band predicate above (bounds, width, vec) is UNCHANGED.
 * Records: results/20261026T2300Z-tsp-p1-prereg.json (mechanism),
 * results/20261026T2340Z-tsp-p1-band-result-dv_37c6db8acaa3.json (the win). */
constexpr int P1_VARIANT = 71;
/* The band is TWO rows, not one, and the split is a measurement rather than a
 * tidy-up.  Slot 77 is slot 71's exact tuple plus the remap, and above its own
 * wave bound the remap is dead code - but dead code is not free here: on d3
 * (2400 blocks, where 77's guard is false and only the two basic blocks and the
 * nine-entry `hist` array remain) slot 77 measures 57.312 us against slot 71's
 * 55.984, +2.4%.  That is the same order as the 1.5% the archived LPT2 note
 * attributed to "one extra register and two dead basic blocks at the top of the
 * kernel", and it is larger than the margin the lower half of this band has, so a
 * single row covering both halves would spend fixed:6/fixed:8's tie and fixed:4's
 * -3.31% on code that cannot run.  Below one wave of the P=1 capacity the band
 * therefore names slot 71, which contains no remap at all; above it, where the
 * remap fires and is worth -6.1 to -6.3%, the band names 77. */
constexpr int P1B_VARIANT = 77;
constexpr int64_t P1_BLOCKS_LO = (int64_t)SM_COUNT * 3;   /* 444 = slot 58's capacity */
constexpr int64_t P1_BLOCKS_HI = (int64_t)SM_COUNT * 4;   /* 592 = one wave at minb 4 */
constexpr int64_t P1B_BLOCKS_HI = (int64_t)SM_COUNT * 8;  /* 1184 = two waves at 592 */
/* The band's token predicate, as a MEAN and not as `width`.
 *
 * `width` bounds max(lengths), so `width <= 4` bounds the mean too - but the mean
 * is the quantity the tradeoff actually turns on, and the two come apart exactly
 * where the band was leaving money.  P=1 halves the threads per block, so it
 * doubles each thread's per-step work and buys capacity 592 against slot 72's 444;
 * the per-step cost scales with the steps a block runs, the capacity gain with how
 * many blocks there are, and neither reads `width`.  dv_e2d8c1ba2b29 (30 samples x
 * 3 interleaved rounds) prices both sides at 512 blocks:
 *   fixed:4  tokens/nreq = 256/64 = 4.00, width 8, counts 8/1/2/5
 *            slot 71 26.656 against slot 72 27.568, -3.31%, 3/3 rounds
 *   d6       tokens/nreq =  72/16 = 4.50, width 8, counts mixed
 *            slot 71 26.624 against slot 72 24.576, +8.33%
 *   d4/d8    tokens/nreq =  32/4  = 8.00, width 8
 *            slot 71 38.912 against slot 72 34.848, +11.66%
 * So the sign flips between a mean of 4.00 and 4.50 at the same width, which is
 * what makes the mean the predicate and the width a proxy that happened to be
 * right for the shapes it was tuned on.  The bound is `tokens <= 4*nreq`, the
 * measured edge; 4.5 is already a loss.
 *
 * Both terms are host metadata the caller passes in - mixed_qkv.size(0) and
 * state_indices.size(0) - so this reads no tensor contents, syncs nothing, and
 * names no case, seed or benchmark phase.
 *
 * ONE CAVEAT, stated rather than hidden: tokens/nreq averages over ALL requests,
 * including the zero-length ones the contract allows, so a shape that is mostly
 * empty requests with a few 8-token ones can pass this test while its ACTIVE mean
 * is above it.  The measured downside of being wrong is bounded: at a mean of 8
 * the P=1 rows cost +9.1 to +10.7% (fixed:1, d9), so a mis-classified shape loses
 * about a tenth rather than failing, and no shape in the frozen fixed suite or the
 * predeclared domain suite has that profile.  `width` cannot substitute for the
 * missing information either - it is the same upper bound in both directions. */
constexpr int P1_MAX_MEAN = 4;
static_assert(P1_VARIANT == 71 && P1B_VARIANT == 77 &&
                  VARIANTS[P1_VARIANT].P == 1 && VARIANTS[P1B_VARIANT].P == 1 &&
                  VARIANTS[P1_VARIANT].minb == 4 && VARIANTS[P1B_VARIANT].minb == 4 &&
                  VARIANTS[P1_VARIANT].tsp && VARIANTS[P1B_VARIANT].tsp &&
                  VARIANTS[P1_VARIANT].lpt2w == 0 && VARIANTS[P1B_VARIANT].lpt2w == 2 &&
                  SM_COUNT <= MULTI_WAVE_BLOCKS &&
                  MULTI_WAVE_BLOCKS < P1_BLOCKS_LO && P1_BLOCKS_LO < P1_BLOCKS_HI &&
                  P1_BLOCKS_HI < P1B_BLOCKS_HI &&
                  P1_BLOCKS_HI == (int64_t)SM_COUNT * VARIANTS[P1_VARIANT].minb &&
                  P1B_BLOCKS_HI == 2 * P1_BLOCKS_HI &&
                  P1_MAX_MEAN >= 1 && P1_MAX_MEAN <= 8,
              "the P=1 band must be ordered after the multi-wave threshold, must name "
              "two P=1 slots at minb=4 carrying the pass-1 staging - one WITHOUT the "
              "remap below a single wave of its own capacity and one WITH it up to two "
              "waves - and its mean bound must sit inside the frozen 1..8 token range. "
              "The two sub-bands must tile (P1_BLOCKS_LO, P1B_BLOCKS_HI] with the seam "
              "exactly at the remap's own fire threshold, so that no grid ever lands on "
              "a row whose remap is dead code: an unordered or mis-seamed band would be "
              "empty, or would pay the +2.4% measured for slot 77 out of band.");
static_assert(MULTI_WAVE_VARIANT == 67 && VARIANTS[67].pfw == 1 &&
                  VARIANTS[67].pfv && VARIANTS[67].minb == 3 &&
                  VARIANTS[67].rsplit && VARIANTS[67].pft &&
                  VARIANTS[67].pf2 == 2 && VARIANTS[58].pf2 == 0 &&
                  !VARIANTS[56].pft && VARIANTS[39].minb == 2 &&
                  !VARIANTS[39].rsplit && !VARIANTS[DEFAULT_VARIANT].rsplit,
              "the SCALAR multi-wave pick is the row split at MINB=3 with the fenced "
              "two-stage prefetch, the tail self-prefetch AND the self pass-2 "
              "warm (slot 58's tuple + pf2=2, adopted 2026-10-26 on the "
              "dv_5ba7036214b6 + dv_474c98f94424 band-invariance close); "
              "slot 58 stays pinned as the pf2=0 same-job control, and the "
              "sub-wave picks are not split");
static_assert(CPA_VARIANT == 72 && VARIANTS[72].cpa && VARIANTS[72].tsp &&
                  VARIANTS[72].pf2 == 0 &&
                  VARIANTS[72].pfw == 1 && VARIANTS[72].pfv &&
                  VARIANTS[72].minb == 3 && VARIANTS[72].rsplit &&
                  VARIANTS[72].pft && VARIANTS[72].lpt2w == 0 &&
                  VARIANTS[72].smem_bytes == VARIANTS[58].smem_bytes + 32768 &&
                  VARIANTS[68].cpa && !VARIANTS[68].tsp &&
                  VARIANTS[68].smem_bytes == VARIANTS[58].smem_bytes + 16384 &&
                  VARIANTS[69].cpa && VARIANTS[69].pf2 == 2 &&
                  VARIANTS[69].smem_bytes == VARIANTS[68].smem_bytes &&
                  CPA_VARIANT != MULTI_WAVE_VARIANT &&
                  CPA_VARIANT != DEFAULT_VARIANT &&
                  CPA_VARIANT != SMALL_GRID_VARIANT &&
                  CPA_VARIANT != SCALAR_VARIANT && CPA_VARIANT != P1_VARIANT &&
                  CPA_VARIANT != 59 && CPA_VARIANT != 60 &&
                  CPA_VARIANT != 61 && CPA_VARIANT != 62 &&
                  CPA_VARIANT != 65 && CPA_VARIANT != 66 &&
                  CPA_VARIANT != 68 && CPA_VARIANT != 69 &&
                  CPA_VARIANT != 70 &&
                  !VARIANTS[DEFAULT_VARIANT].cpa &&
                  !VARIANTS[MULTI_WAVE_VARIANT].cpa &&
                  !VARIANTS[SMALL_GRID_VARIANT].cpa && !VARIANTS[66].cpa &&
                  !VARIANTS[P1_VARIANT].cpa && !VARIANTS[P1B_VARIANT].cpa,
              "the VEC multi-wave pick is 72 (slot 58's tuple + cpa + tsp: BOTH "
              "cp.async staging images, adopted 2026-10-27 on the two-draw "
              "composition close dv_9794e563e288 + dv_6817d269e892); it is the "
              "ONLY dispatched slot carrying cpa, its arena is 58's plus exactly "
              "the two 16 KiB staging images, and every other research/control "
              "row (59/60/61/62/65/66/68/69/70) stays undispatched under all "
              "five pick constants");

/* Pin the aggregate order.  Variant is filled positionally by four macros, so a
 * reordered or shortened struct would not fail to compile - it would silently
 * change what every slot means, and the table-driven launcher below feeds those
 * fields straight into template arguments.  same_tuple() names every field, and
 * the pinned slots are chosen so that each field takes both values somewhere in
 * the list: a permutation surviving all of them would have to swap two fields
 * that are equal in every pinned slot.  f32x2 is the one exception (true in all
 * nineteen slots), so it is unpinnable and swapping it is behaviour-preserving.
 * A new slot needs no update here unless it introduces a new field. */
constexpr bool same_tuple(int i, int P, int lpr, bool wide, int na, bool f32x2,
                          bool cs, int minb, bool merge, int epim, bool ald,
                          bool st4, bool qk32, bool pack, bool pub1, bool csb,
                          bool rperm, bool asa, bool asal, bool csbodd,
                          bool ccseam, bool rowseam, bool grid2d = false,
                          bool hqfast = false, int cm = -1, int pfw = 0,
                          bool pfv = false, bool trim = false,
                          bool rsplit = false, bool pft = false,
                          int lpt2w = 0, int pf2 = 0, bool cpa = false,
                          bool tsp = false, bool idxpar = false,
                          bool pfl = false) {
  const Variant& c = VARIANTS[i];
  return c.P == P && c.lpr == lpr && c.wide == wide && c.na == na &&
         c.f32x2 == f32x2 && c.cs == cs && c.minb == minb && c.merge == merge &&
         c.epim == epim && c.ald == ald && c.st4 == st4 && c.qk32 == qk32 &&
         c.pack == pack && c.pub1 == pub1 && c.csb == csb && c.rperm == rperm &&
         c.asa == asa && c.asal == asal && c.csbodd == csbodd &&
         c.ccseam == ccseam && c.rowseam == rowseam && c.grid2d == grid2d &&
         c.hqfast == hqfast && c.cm == cm && c.pfw == pfw && c.pfv == pfv &&
         c.trim == trim && c.rsplit == rsplit && c.pft == pft &&
         c.lpt2w == lpt2w && c.pf2 == pf2 && c.cpa == cpa && c.tsp == tsp &&
         c.idxpar == idxpar && c.pfl == pfl;
}
static_assert(same_tuple(0, 2, 2, false, 2, true, true, 2, false, 1, false, false, false, false, false, false, false, false, false, false, false, false),
              "slot 0 tuple moved (narrow slot 0)");
static_assert(same_tuple(1, 2, 2, false, 2, true, true, 2, false, 0, false, false, false, false, false, false, false, false, false, false, false, false),
              "slot 1 tuple moved (p==0-only epilogue)");
static_assert(same_tuple(5, 2, 2, false, 2, true, true, 2, true, 1, false, false, false, false, false, false, false, false, false, false, false, false),
              "slot 5 tuple moved (MERGE control)");
static_assert(same_tuple(6, 2, 2, false, 2, true, false, 2, false, 1, false, false, false, false, false, false, false, false, false, false, false, false),
              "slot 6 tuple moved (evict-first A/B)");
static_assert(same_tuple(7, 2, 2, false, 2, true, true, 1, false, 1, false, false, false, false, false, false, false, false, false, false, false, false),
              "slot 7 tuple moved (MINB=1)");
static_assert(same_tuple(8, 8, 8, false, 2, true, true, 1, false, 1, false, false, false, false, false, false, false, false, false, false, false, false),
              "slot 8 tuple moved (P8 narrow)");
static_assert(same_tuple(11, 2, 8, true, 1, true, true, 2, false, 1, false, false, false, false, false, false, false, false, false, false, false, false),
              "slot 11 tuple moved (WIDE8 with the EPIM=1 epilogue, the flat-epilogue control)");
static_assert(same_tuple(12, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, false, false, false, false, false, false),
              "slot 12 tuple moved (promoted default)");
static_assert(same_tuple(14, 2, 8, true, 1, true, true, 2, false, 2, true, false, false, false, false, false, false, false, false, false, false, false),
              "slot 14 tuple moved (ALD without ST4)");
static_assert(same_tuple(16, 2, 8, true, 1, true, true, 2, false, 2, true, true, true, false, false, false, false, false, false, false, false, false),
              "slot 16 tuple moved (QK32)");
static_assert(same_tuple(17, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, true, false, false, false, false, false, false, false, false),
              "slot 17 tuple moved (PACK)");
static_assert(same_tuple(18, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, true, false, false, false, false, false, false, false),
              "slot 18 tuple moved (PUB1)");
static_assert(same_tuple(19, 2, 8, true, 1, true, true, 2, false, 3, true, true, false, false, false, false, false, false, false, false, false, false),
              "slot 19 tuple moved (count-selected epilogue, EPT=6)");
static_assert(same_tuple(20, 2, 8, true, 1, true, true, 2, false, 4, true, true, false, false, false, false, false, false, false, false, false, false),
              "slot 20 tuple moved (count-selected epilogue, EPT=8)");
static_assert(same_tuple(21, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, true, false, false, false, false, false, false),
              "slot 21 tuple moved (CSB, one constant-stride state base)");
static_assert(same_tuple(22, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, false, false, false, false),
              "slot 22 tuple moved (RPERM, row-permuted vsh/rawsh)");
static_assert(same_tuple(23, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, true, true, false, false, false, false, false),
              "slot 23 tuple moved (CSB + RPERM)");
static_assert(same_tuple(24, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, true, true, true, false, false, false, false),
              "slot 24 tuple moved (ASA, cross-lane aligned-block stores)");
static_assert(same_tuple(25, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, true, false, false, false, false),
              "slot 25 tuple moved (ASA without CSB)");
static_assert(same_tuple(26, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, false, false),
              "slot 26 tuple moved (ASAL, the in-place aligned-block assembly)");
static_assert(same_tuple(27, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, true, false, false),
              "slot 27 tuple moved (ASAL + CSBODD, the fold in the VEC=false instantiation only)");
static_assert(same_tuple(28, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false),
              "slot 28 tuple moved (ASAL + CCSEAM, the chunk-boundary aligned block)");
static_assert(same_tuple(29, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, true),
              "slot 29 tuple moved (ASAL + CCSEAM + the row-boundary seam)");
static_assert(same_tuple(39, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true),
              "slot 39 tuple moved (promoted: slot 28 + GRID2D + HQFAST)");
static_assert(same_tuple(43, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1),
              "slot 43 must be slot 39's tuple with pfw=1 and nothing else moved");
static_assert(same_tuple(44, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 2),
              "slot 44 must be slot 39's tuple with pfw=2 and nothing else moved");
static_assert(VARIANTS[43].pfw == 1 && VARIANTS[44].pfw == 2 &&
                  VARIANTS[39].pfw == 0,
              "the prefetch distance is the only thing slots 43/44 add");
static_assert(same_tuple(45, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true),
              "slot 45 must be slot 43's tuple with pfv on and nothing else moved");
static_assert(!VARIANTS[43].pfv && !VARIANTS[44].pfv && VARIANTS[45].pfv &&
                  VARIANTS[43].pfw == 1,
              "the vector stage is slot 45's alone; 43 is the tile-only arm");
static_assert(same_tuple(46, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, 0, 1, true) &&
                  same_tuple(47, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, 1, 1, true) &&
                  same_tuple(48, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, 2, 1, true),
              "slots 46/47/48 must be slot 45's tuple with only the hint split moved");
static_assert(VARIANTS[45].cm == -1 && VARIANTS[43].cm == -1,
              "the promoted multi-wave picks stream both directions");
static_assert(same_tuple(49, 2, 8, true, 2, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true) &&
                  same_tuple(50, 2, 8, true, 2, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, false),
              "slots 49/50 must be slots 45/43 with na=2 and nothing else moved");
static_assert(VARIANTS[49].na == 2 && VARIANTS[50].na == 2 && VARIANTS[45].na == 1,
              "NA is the only thing slots 49/50 add");
static_assert(same_tuple(51, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, true),
              "slot 51 must be slot 45's tuple with trim=true and nothing else moved");
static_assert(same_tuple(52, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, false, true),
              "slot 52 must be slot 43's tuple with trim=true and nothing else moved");
static_assert(same_tuple(53, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true),
              "slot 53 must be slot 45's tuple with minb=3 and nothing else moved");
static_assert(same_tuple(54, 2, 8, true, 1, true, true, 4, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true),
              "slot 54 must be slot 45's tuple with minb=4 and nothing else moved");
static_assert(same_tuple(55, 2, 8, true, 1, true, true, 1, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true),
              "slot 55 must be slot 45's tuple with minb=1 and nothing else moved");
static_assert(VARIANTS[55].minb == 1 && VARIANTS[55].smem_bytes ==
                                          VARIANTS[45].smem_bytes,
              "MINB is the only thing slot 55 moves, and it does not move the arena");
static_assert(same_tuple(56, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true),
              "slot 56 must be slot 45's tuple with minb=3 and rsplit=true");
static_assert(same_tuple(57, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true),
              "slot 57 must be slot 45's tuple with rsplit=true and minb left at 2");
static_assert(VARIANTS[56].rsplit && VARIANTS[57].rsplit &&
                  !VARIANTS[45].rsplit && !VARIANTS[39].rsplit &&
                  VARIANTS[56].minb == 3 && VARIANTS[57].minb == 2 &&
                  VARIANTS[56].smem_bytes == VARIANTS[57].smem_bytes &&
                  VARIANTS[56].smem_bytes == VARIANTS[45].smem_bytes + 256,
              "RSPLIT is the only thing slots 56/57 add, and it grows the arena by "
              "exactly the second pass's MAXS*NW row-square partials");
static_assert(same_tuple(58, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true),
              "slot 58 must be slot 56's tuple with pft=true and nothing else moved");
static_assert(same_tuple(66, 1, 8, true, 1, true, true, 4, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true),
              "slot 66 must be slot 58's tuple with P=1 and minb=4 and nothing else moved");
static_assert(VARIANTS[66].P == 1 && VARIANTS[66].minb == 4 &&
                  VARIANTS[66].lpr == 8 && VARIANTS[66].lpt2w == 0 &&
                  VARIANTS[58].P == 2 && VARIANTS[58].minb == 3 &&
                  VARIANTS[66].na == VARIANTS[58].na &&
                  VARIANTS[66].epim == VARIANTS[58].epim &&
                  VARIANTS[66].rsplit == VARIANTS[58].rsplit &&
                  VARIANTS[66].pft == VARIANTS[58].pft &&
                  VARIANTS[66].pfv == VARIANTS[58].pfv &&
                  VARIANTS[66].pfw == VARIANTS[58].pfw &&
                  VARIANTS[66].cm == VARIANTS[58].cm &&
                  VARIANTS[66].smem_bytes == VARIANTS[58].smem_bytes - 256,
              "slot 66 moves only P and minb off slot 58; the arena shrinks by "
              "exactly the RAW_OFF term MAXS*NW*NH*4, which is 8*4*2*4 = 256 here "
              "against 8*8*2*4 = 512 at slot 58, because NW halves and no other "
              "arena offset depends on it");
static_assert(same_tuple(67, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 0, 2),
              "slot 67 must be slot 58's tuple with pf2=2 and nothing else moved");
static_assert(same_tuple(68, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 0, 0, true),
              "slot 68 must be slot 58's tuple with cpa=true and nothing else moved");
static_assert(same_tuple(69, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 0, 2, true),
              "slot 69 must be slot 67's tuple with cpa=true and nothing else moved");
static_assert(same_tuple(70, 1, 8, true, 1, true, true, 4, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 0, 0, true),
              "slot 70 must be slot 66's tuple with cpa=true and nothing else moved");
static_assert(VARIANTS[70].cpa && VARIANTS[70].P == 1 &&
                  VARIANTS[70].minb == 4 && !VARIANTS[66].cpa &&
                  VARIANTS[70].smem_bytes == VARIANTS[66].smem_bytes + 16384 &&
                  VARIANTS[70].rsplit && VARIANTS[70].pft &&
                  VARIANTS[70].pfv && VARIANTS[70].pfw == 1 &&
                  VARIANTS[70].pf2 == 0 && VARIANTS[70].lpt2w == 0 &&
                  P1_VARIANT != 70 && P1B_VARIANT != 70 && !VARIANTS[70].tsp &&
                  CPA_VARIANT != 70 && MULTI_WAVE_VARIANT != 70 &&
                  DEFAULT_VARIANT != 70 && SMALL_GRID_VARIANT != 70 &&
                  SCALAR_VARIANT != 70,
              "slot 70 moves only cpa off slot 66, grows the P=1 arena by "
              "exactly the 16 KiB staging image, keeps the band's whole "
              "prefetch set, and stays UNDISPATCHED: its band A/B closed "
              "NEGATIVE (dv_5cea0a04c463 - CPA pass-2 staging is null where "
              "the single-wave band keeps reloads L2-resident), and the band "
              "pick moved to 71 (pass-1 staging) instead");
static_assert(same_tuple(71, 1, 8, true, 1, true, true, 4, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 0, 0, false, true),
              "slot 71 must be slot 66's tuple with tsp=true and nothing else moved");
static_assert(same_tuple(72, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 0, 0, true, true),
              "slot 72 must be slot 68's tuple with tsp=true and nothing else moved");
static_assert(VARIANTS[72].cpa && VARIANTS[72].tsp && VARIANTS[72].pf2 == 0 &&
                  VARIANTS[72].smem_bytes == VARIANTS[68].smem_bytes + 16384 &&
                  VARIANTS[72].smem_bytes == 45760 &&
                  VARIANTS[72].minb == 3 && VARIANTS[72].rsplit &&
                  VARIANTS[72].pft && VARIANTS[72].pfw == 1 &&
                  VARIANTS[72].lpt2w == 0 &&
                  !VARIANTS[68].tsp && !VARIANTS[67].tsp && !VARIANTS[58].tsp &&
                  CPA_VARIANT == 72 && P1_VARIANT == 71 && P1B_VARIANT == 77 &&
                  MULTI_WAVE_VARIANT != 72 && MULTI_WAVE_VARIANT != 71 &&
                  MULTI_WAVE_VARIANT != 77 && DEFAULT_VARIANT != 72 &&
                  SMALL_GRID_VARIANT != 72 && SCALAR_VARIANT != 72 &&
                  DEFAULT_VARIANT != 71 && DEFAULT_VARIANT != 77 &&
                  SMALL_GRID_VARIANT != 71 && SMALL_GRID_VARIANT != 77 &&
                  SCALAR_VARIANT != 71 && SCALAR_VARIANT != 77 &&
                  MULTI_WAVE_VARIANT != 68 && P1_VARIANT != 68 &&
                  P1B_VARIANT != 68,
              "slots 71/72 are the ADOPTED staging rows (2026-10-27): 72 = 68's "
              "tuple + tsp is the VEC multi-wave pick (both staging images live, "
              "arena 45760 = 58's + 32768), 71 = 66's tuple + tsp is the P=1 band "
              "pick; their A/B predecessors (66, 68) stay in the table as control "
              "rows under no pick constant");
static_assert(VARIANTS[71].tsp && !VARIANTS[71].cpa && VARIANTS[71].P == 1 &&
                  VARIANTS[71].minb == 4 && !VARIANTS[66].tsp &&
                  !VARIANTS[70].tsp && !VARIANTS[68].tsp &&
                  !VARIANTS[69].tsp && !VARIANTS[58].tsp &&
                  VARIANTS[71].smem_bytes == VARIANTS[66].smem_bytes + 16384 &&
                  VARIANTS[71].rsplit && VARIANTS[71].pft &&
                  VARIANTS[71].pfv && VARIANTS[71].pfw == 1 &&
                  VARIANTS[71].pf2 == 0 && VARIANTS[71].lpt2w == 0 &&
                  VARIANTS[77].P == 1 && VARIANTS[77].minb == 4 &&
                  VARIANTS[77].tsp && VARIANTS[77].lpt2w == 2 &&
                  VARIANTS[77].smem_bytes == VARIANTS[71].smem_bytes + 160 &&
                  P1_VARIANT == 71 && P1B_VARIANT != 71 && CPA_VARIANT != 71 &&
                  MULTI_WAVE_VARIANT != 71 && DEFAULT_VARIANT != 71 &&
                  SMALL_GRID_VARIANT != 71 && SCALAR_VARIANT != 71,
              "slot 71 moves only tsp off slot 66, grows the P=1 arena by "
              "exactly the 16 KiB pass-1 staging image, keeps the band's whole "
              "prefetch set, and IS the P=1 band's lower sub-band pick (adopted "
              "2026-10-27 on dv_37c6db8acaa3: case6 -4.86%, case8 -3.69%, "
              "bitwise identical, 125/0/0 census).  Slot 77 is 71's exact tuple "
              "plus the two-wave remap and is the UPPER sub-band pick as of "
              "2026-10-28 (dv_e2d8c1ba2b29: fixed:3 -6.25% and fixed:7 -6.10% "
              "against slot 72 at 1024 blocks, 3/3 interleaved rounds, where "
              "slot 71 alone is -2.78% and 0.00%).  The sub-bands are seam-bound "
              "to the remap's own fire threshold so no grid lands on 77 where its "
              "remap would be dead code, which measures +2.4% on d3.");
static_assert(!VARIANTS[DEFAULT_VARIANT].tsp && !VARIANTS[MULTI_WAVE_VARIANT].tsp &&
                  !VARIANTS[SMALL_GRID_VARIANT].tsp &&
                  VARIANTS[P1_VARIANT].tsp && VARIANTS[P1B_VARIANT].tsp &&
                  VARIANTS[CPA_VARIANT].tsp,
              "exactly the two adopted staging picks (P1_VARIANT 71 and "
              "CPA_VARIANT 72) carry tsp; the sub-wave and scalar multi-wave "
              "picks do not - the field is appended last to a positional "
              "aggregate, and a shift that moved a staging image onto a "
              "non-staging pick would change production smem while every "
              "per-slot pin still read correctly");
static_assert(VARIANTS[68].cpa && VARIANTS[68].pf2 == 0 &&
                  VARIANTS[69].cpa && VARIANTS[69].pf2 == 2 &&
                  !VARIANTS[58].cpa && !VARIANTS[67].cpa && !VARIANTS[66].cpa &&
                  !VARIANTS[39].cpa &&
                  VARIANTS[68].smem_bytes == VARIANTS[58].smem_bytes + 16384 &&
                  VARIANTS[69].smem_bytes == VARIANTS[67].smem_bytes + 16384 &&
                  VARIANTS[68].minb == 3 && VARIANTS[69].minb == 3 &&
                  VARIANTS[68].rsplit && VARIANTS[69].rsplit &&
                  VARIANTS[68].pft && VARIANTS[69].pft &&
                  VARIANTS[68].pfw == 1 && VARIANTS[69].pfw == 1 &&
                  CPA_VARIANT != 68 && MULTI_WAVE_VARIANT != 68 &&
                  P1_VARIANT != 68 && P1B_VARIANT != 68 &&
                  MULTI_WAVE_VARIANT != 69 &&
                  DEFAULT_VARIANT != 69 && SMALL_GRID_VARIANT != 69 &&
                  SCALAR_VARIANT != 69 && P1_VARIANT != 69 &&
                  P1B_VARIANT != 69,
              "slots 68/69 move only cpa off slots 58/67, grow the arena by "
              "exactly the 16 KiB pass-2 staging image, keep MINB=3 and the "
              "whole prefetch set; 68 WAS the VEC multi-wave pick (adopted "
              "2026-10-26, certified as tree 90fb8882) and is superseded by "
              "72 (68 + tsp) on the 2026-10-27 composition close - 68 stays "
              "as its control row; 69 - the pf2 composition the A/B showed "
              "redundant-to-harmful - stays undispatched under every pick "
              "constant");
static_assert(MULTI_WAVE_VARIANT == 67 && DEFAULT_VARIANT != 67 &&
                  SMALL_GRID_VARIANT != 67 && SCALAR_VARIANT != 67 &&
                  P1_VARIANT != 67 && P1B_VARIANT != 67 && CPA_VARIANT != 67,
              "slot 67 is reached ONLY as the SCALAR multi-wave pick (adopted "
              "2026-10-26, re-scoped the same day by the CPA adoption: vector "
              "grids take CPA_VARIANT); the sub-wave picks, the P=1 band and "
              "the vec multi-wave pick are untouched");
static_assert(NUM_VARIANTS == 81,
              "slots 73-77 (the 2026-10-28 IDXPAR / longest-first arms) and 78-80 "
              "(the PFL prefetch-line arms) appended; update the count and the pins");
/* Pin the five 2026-10-28 arms.  73/74/75/76 are slot 72's tuple with exactly
 * one of {idxpar, lpt2w} moved, and 77 is slot 71's tuple with lpt2w moved, so
 * pinning all five against the two promoted rows they derive from makes a
 * positional slip in GDN_WIDEV - the failure mode the append-only contract
 * exists to prevent, and the one that already shipped once (a field inserted
 * between pftb and pfh put GDN_WIDERF's pfh argument into lpt) - a compile error
 * rather than a silently different kernel. */
static_assert(same_tuple(73, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 0, 0, true, true, true),
              "slot 73 must be slot 72's exact tuple plus idxpar and nothing else");
static_assert(same_tuple(74, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 2, 0, true, true),
              "slot 74 must be slot 72's exact tuple plus the two-wave remap");
static_assert(same_tuple(75, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 2, 0, true, true, true),
              "slot 75 must be slot 74 plus idxpar, i.e. the composition arm");
static_assert(same_tuple(76, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 4, 0, true, true),
              "slot 76 must be slot 74 with the remap's wave bound at four");
static_assert(same_tuple(77, 1, 8, true, 1, true, true, 4, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 2, 0, false, true),
              "slot 77 must be slot 71's exact tuple plus the two-wave remap");
static_assert(same_tuple(78, 1, 8, true, 1, true, true, 4, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 0, 0, false, true, false, true),
              "slot 78 must be slot 71's exact tuple plus pfl and nothing else");
static_assert(same_tuple(79, 1, 8, true, 1, true, true, 4, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 2, 0, false, true, false, true),
              "slot 79 must be slot 77's exact tuple plus pfl and nothing else");
static_assert(same_tuple(80, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 0, 0, true, true, false, true),
              "slot 80 must be slot 72's exact tuple plus pfl, i.e. the P=2 control");
static_assert(VARIANTS[78].smem_bytes == VARIANTS[71].smem_bytes &&
                  VARIANTS[79].smem_bytes == VARIANTS[77].smem_bytes &&
                  VARIANTS[80].smem_bytes == VARIANTS[72].smem_bytes,
              "PFL changes no shared memory: it re-walks the same regions in "
              "128-byte lines instead of striding region_bytes/threads, so the "
              "arena is untouched and a slot's carveout cannot move under it");
static_assert(!VARIANTS[71].pfl && !VARIANTS[72].pfl && !VARIANTS[77].pfl &&
                  VARIANTS[78].pfl && VARIANTS[79].pfl && VARIANTS[80].pfl &&
                  VARIANTS[78].lpt2w == 0 && VARIANTS[79].lpt2w == 2 &&
                  VARIANTS[80].P == 2 && VARIANTS[78].P == 1 &&
                  VARIANTS[79].P == 1,
              "pfl is the ONLY field 78/79/80 move off 71/77/72");
static_assert(DEFAULT_VARIANT != 78 && DEFAULT_VARIANT != 79 &&
                  DEFAULT_VARIANT != 80 && MULTI_WAVE_VARIANT != 78 &&
                  MULTI_WAVE_VARIANT != 79 && MULTI_WAVE_VARIANT != 80 &&
                  SMALL_GRID_VARIANT != 78 && SMALL_GRID_VARIANT != 79 &&
                  SMALL_GRID_VARIANT != 80 && SCALAR_VARIANT != 78 &&
                  SCALAR_VARIANT != 79 && SCALAR_VARIANT != 80 &&
                  P1_VARIANT != 78 && P1_VARIANT != 79 && P1_VARIANT != 80 &&
                  P1B_VARIANT != 78 && P1B_VARIANT != 79 && P1B_VARIANT != 80 &&
                  CPA_VARIANT != 78 && CPA_VARIANT != 79 && CPA_VARIANT != 80,
              "slots 78-80 are research rows reachable only through set_config or "
              "GDN_V until the PFL arm is measured");
static_assert(VARIANTS[73].smem_bytes == VARIANTS[72].smem_bytes &&
                  VARIANTS[75].smem_bytes == VARIANTS[74].smem_bytes &&
                  VARIANTS[76].smem_bytes == VARIANTS[74].smem_bytes &&
                  VARIANTS[74].smem_bytes == VARIANTS[72].smem_bytes + 304 &&
                  VARIANTS[77].smem_bytes == VARIANTS[71].smem_bytes + 160,
              "IDXPAR adds no shared memory; the remap adds exactly its "
              "[MAXS+1][NW] count table plus the found flag, rounded up to a "
              "16-byte arena end: (9*8*4 + 4 + 15) & ~15 = 304 B at NW=8 (P=2, "
              "45760 -> 46064) and (9*4*4 + 4 + 15) & ~15 = 160 B at NW=4 "
              "(P=1, 29120 -> 29280).  launch_slot's arena assert checks this "
              "symbolically; pinning the numbers too means a silent NW or MAXS "
              "change cannot move a carveout underneath a dispatched slot.");
static_assert(DEFAULT_VARIANT != 73 && DEFAULT_VARIANT != 74 &&
                  DEFAULT_VARIANT != 75 && DEFAULT_VARIANT != 76 &&
                  MULTI_WAVE_VARIANT != 73 && MULTI_WAVE_VARIANT != 74 &&
                  MULTI_WAVE_VARIANT != 75 && MULTI_WAVE_VARIANT != 76 &&
                  SMALL_GRID_VARIANT != 73 && SMALL_GRID_VARIANT != 74 &&
                  SMALL_GRID_VARIANT != 75 && SMALL_GRID_VARIANT != 76 &&
                  SCALAR_VARIANT != 73 && SCALAR_VARIANT != 74 &&
                  SCALAR_VARIANT != 75 && SCALAR_VARIANT != 76 &&
                  P1_VARIANT != 73 && P1_VARIANT != 74 && P1_VARIANT != 75 &&
                  P1_VARIANT != 76 && P1B_VARIANT != 73 && P1B_VARIANT != 74 &&
                  P1B_VARIANT != 75 && P1B_VARIANT != 76 &&
                  CPA_VARIANT != 73 && CPA_VARIANT != 74 &&
                  CPA_VARIANT != 75 && CPA_VARIANT != 76,
              "slots 73-76 are REJECTED RESEARCH ROWS and stay reachable only "
              "through set_config or GDN_V: IDXPAR (73) measured neutral-to-worse "
              "on 18 of 20 shapes, and the remap composed onto the P=2 package "
              "(74/75/76) does not pay at slot 72's capacity either - it is worth "
              "-6% only on the P=1 row, whose 592 capacity puts a 1024-block grid "
              "at 1.73 waves instead of 2.31.  They are kept, with their pins, so "
              "the negative result is reproducible from the same binary rather "
              "than from a description of it.  Slot 77 IS dispatched, as the P=1 "
              "band's upper sub-band row, and is pinned by the band asserts "
              "above.");
static_assert(DEFAULT_VARIANT != 66 && MULTI_WAVE_VARIANT != 66 &&
                  SMALL_GRID_VARIANT != 66 && SCALAR_VARIANT != 66 &&
                  P1_VARIANT != 66 && P1B_VARIANT != 66 && CPA_VARIANT != 66,
              "slot 66 is no longer dispatched: the P=1 band moved to slot 71 "
              "(66's tuple + tsp) on the 2026-10-27 close (dv_37c6db8acaa3: "
              "case6 -4.86%, case8 -3.69%, bitwise identical) and to slot 77 "
              "(71's tuple + the two-wave remap) on 2026-10-28.  No dispatch "
              "CONSTANT names 66, so a grid outside 444..1184 blocks, a mean "
              "token count above P1_MAX_MEAN, or a state slot stride that breaks "
              "16-byte alignment all keep the pick they had before the band "
              "existed.  This assert said 'slot 66 is undispatched' until the "
              "band was added, then 'reached ONLY through the P=1 band' while 66 "
              "was the band row; it is rewritten rather than deleted because the "
              "property it guards - that the pick constants alone bound the "
              "band's blast radius to the band - is unchanged");
static_assert(VARIANTS[58].pft && !VARIANTS[56].pft && !VARIANTS[57].pft &&
                  VARIANTS[58].smem_bytes == VARIANTS[56].smem_bytes,
              "PFT is the only thing slot 58 adds, and it does not move the arena");
/* Slots 59 and 60 are slot 58's tuple with exactly one field moved each, pinned both
 * by value and relationally.  A positional brace-initialiser can silently shift a
 * later value into the wrong member, and a value pin alone would not say *which*
 * field the arm is supposed to be testing - so the relational pin names it. */
static_assert(same_tuple(59, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 2, true, false, true, true),
              "slot 59 must be slot 58's tuple with pfw=2 and nothing else moved");
static_assert(same_tuple(60, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, false, false, true, true),
              "slot 60 must be slot 58's tuple with pfv=false and nothing else moved");
static_assert(same_tuple(61, 2, 16, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true),
              "slot 61 must be slot 58's tuple with lpr=16 and nothing else moved");
static_assert(same_tuple(62, 2, 8, true, 2, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true),
              "slot 62 must be slot 58's tuple with na=2 and nothing else moved");
/* Slots 63 and 64 move lpr to the axis's other legal boundary, and 64 additionally
 * moves minb.  Pinned by value and relationally like 59-62: a positional brace
 * initialiser can silently shift a later value into the wrong member, and a value pin
 * alone would not say which field the arm is supposed to be testing.  64 is pinned
 * against 63 and not against 58, because the pair's whole purpose is that MINB is the
 * only thing between them -- if a third field drifted, 63 vs 64 would stop isolating
 * occupancy and the arm would be uninterpretable rather than merely null. */
static_assert(same_tuple(63, 2, 4, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true),
              "slot 63 must be slot 58's tuple with lpr=4 and nothing else moved");
static_assert(same_tuple(64, 2, 4, true, 1, true, true, 4, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true),
              "slot 64 must be slot 63's tuple with minb=4 and nothing else moved");
static_assert(VARIANTS[63].lpr == 4 && VARIANTS[63].minb == 3 && VARIANTS[63].na == 1 &&
                  VARIANTS[64].lpr == 4 && VARIANTS[64].minb == 4 && VARIANTS[64].na == 1 &&
                  VARIANTS[58].lpr == 8 && VARIANTS[58].minb == 3 &&
                  VARIANTS[63].smem_bytes == VARIANTS[58].smem_bytes &&
                  VARIANTS[64].smem_bytes == VARIANTS[58].smem_bytes &&
                  VARIANTS[63].rsplit && VARIANTS[64].rsplit &&
                  VARIANTS[63].pft && VARIANTS[64].pft &&
                  VARIANTS[63].pfv && VARIANTS[64].pfv &&
                  VARIANTS[63].pfw == 1 && VARIANTS[64].pfw == 1 &&
                  MULTI_WAVE_VARIANT != 63 && MULTI_WAVE_VARIANT != 64 &&
                  DEFAULT_VARIANT != 63 && DEFAULT_VARIANT != 64 &&
                  SMALL_GRID_VARIANT != 63 && SMALL_GRID_VARIANT != 64,
              "slots 63/64 share slot 58's arena, prefetch set and rsplit, differ from it "
              "in lpr and from each other in minb only, and neither is dispatched");
/* Slot 65 is slot 58's tuple with the two-barrier longest-first remap bounded to two
 * waves, and nothing else moved.  Pinned by value and relationally like every other arm
 * here, plus one pin that the others do not need: `lpt2w` is a NEW field appended to a
 * positionally brace-initialised aggregate, which is the exact change that once put a
 * `pfh` argument into `lpt` and was caught only by these pins (dv_bfee84a9231c).  So the
 * load-bearing assertion is not slot 65's own value -- it is that NO DISPATCHED slot
 * acquired a non-zero bound.  A shift that gave slot 58 a remap would change production
 * behaviour on nine of ten fixed cells while every per-slot pin still read correctly. */
static_assert(same_tuple(65, 2, 8, true, 1, true, true, 3, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true, -1, 1, true, false, true, true, 2),
              "slot 65 must be slot 58's tuple with lpt2w=2 and nothing else moved");
static_assert(VARIANTS[DEFAULT_VARIANT].lpt2w == 0 &&
                  VARIANTS[MULTI_WAVE_VARIANT].lpt2w == 0 &&
                  VARIANTS[SMALL_GRID_VARIANT].lpt2w == 0,
              "no dispatched slot carries the remap: lpt2w is appended last to a positional "
              "aggregate, and a shift that gave a dispatched slot a non-zero bound would "
              "change production while every per-slot pin still read correctly");
static_assert(VARIANTS[65].lpt2w == 2 && VARIANTS[58].lpt2w == 0 &&
                  VARIANTS[65].lpr == 8 && VARIANTS[65].minb == 3 &&
                  VARIANTS[65].na == 1 && VARIANTS[65].epim == 2 &&
                  VARIANTS[65].rsplit && VARIANTS[65].pft && VARIANTS[65].pfv &&
                  VARIANTS[65].pfw == 1 && VARIANTS[65].cm == -1 &&
                  VARIANTS[65].smem_bytes == VARIANTS[58].smem_bytes + 304 &&
                  MULTI_WAVE_VARIANT != 65 && DEFAULT_VARIANT != 65 &&
                  SMALL_GRID_VARIANT != 65,
              "slot 65 moves only lpt2w off slot 58, keeps the whole prefetch set and the "
              "lane map, grows the arena by exactly the [class][warp] table plus its found "
              "flag, and is not dispatched");
static_assert(VARIANTS[61].lpr == 16 && VARIANTS[61].na == 1 && VARIANTS[58].lpr == 8 &&
                  VARIANTS[62].na == 2 && VARIANTS[62].lpr == 8 &&
                  VARIANTS[61].smem_bytes == VARIANTS[58].smem_bytes &&
                  VARIANTS[62].smem_bytes == VARIANTS[58].smem_bytes &&
                  VARIANTS[61].minb == 3 && VARIANTS[62].minb == 3 &&
                  VARIANTS[61].rsplit && VARIANTS[62].rsplit &&
                  VARIANTS[61].pft && VARIANTS[62].pft &&
                  VARIANTS[61].pfv && VARIANTS[62].pfv &&
                  MULTI_WAVE_VARIANT != 61 && MULTI_WAVE_VARIANT != 62 &&
                  DEFAULT_VARIANT != 61 && DEFAULT_VARIANT != 62,
              "slots 61/62 move one geometric field each off slot 58, keep the arena, MINB and "
              "the whole prefetch set, and neither is dispatched");
static_assert(VARIANTS[59].pfw == 2 && VARIANTS[58].pfw == 1 && VARIANTS[60].pfw == 1 &&
                  VARIANTS[58].pfv && VARIANTS[59].pfv && !VARIANTS[60].pfv &&
                  VARIANTS[59].smem_bytes == VARIANTS[58].smem_bytes &&
                  VARIANTS[60].smem_bytes == VARIANTS[58].smem_bytes &&
                  VARIANTS[59].minb == 3 && VARIANTS[60].minb == 3 &&
                  VARIANTS[59].rsplit && VARIANTS[60].rsplit &&
                  VARIANTS[59].pft && VARIANTS[60].pft &&
                  MULTI_WAVE_VARIANT == 67 && DEFAULT_VARIANT == 39 &&
                  MULTI_WAVE_VARIANT != 59 && MULTI_WAVE_VARIANT != 60,
              "slots 59/60 move one prefetch field each off slot 58, keep the arena, MINB "
              "and RSPLIT/PFT, and neither is dispatched; since the 2026-10-26 adoption "
              "the multi-wave pick is 67 (slot 58's tuple + pf2=2)");
static_assert(VARIANTS[53].minb == 3 && VARIANTS[54].minb == 4 &&
                  VARIANTS[45].minb == 2 && VARIANTS[53].pfw == 1 &&
                  VARIANTS[53].pfv && VARIANTS[53].smem_bytes ==
                                          VARIANTS[45].smem_bytes,
              "MINB is the only thing slots 53/54 move, and it does not move the arena");
static_assert(VARIANTS[51].trim && VARIANTS[52].trim && !VARIANTS[45].trim &&
                  !VARIANTS[43].trim && !VARIANTS[39].trim,
              "TRIM is the only thing slots 51/52 add");
static_assert(VARIANTS[51].pfv == VARIANTS[45].pfv &&
                  VARIANTS[52].pfv == VARIANTS[43].pfv,
              "slot 51 keeps slot 45's vector prefetch, slot 52 does not");
static_assert(!same_tuple(28, 2, 8, true, 1, true, true, 2, false, 2, true, true, false, false, false, false, true, false, true, false, true, false, true, true),
              "slot 28 must stay the pre-2026-10-21 package, without the startup flags");
struct Cfg {
  int variant;
  int carve;   /* -1: smallest carveout that fits minb blocks per SM */
  int cs;      /* retained for the reported configuration; the cache hint is a
                  compile-time property of the selected variant */
  bool forced; /* the slot was chosen explicitly, not by the launch rule */
};

/* Defined with the row-split table further down; declared here so that both
 * configuration entry points can range-check one slot space covering both
 * families without moving the (historically pinned) configuration block. */
inline int num_slots_total();

inline Cfg& live_cfg() {
  static Cfg cfg = [] {
    Cfg c{DEFAULT_VARIANT, -1, 1, false};
    if (const char* e = std::getenv("GDN_V")) {
      const int v = std::atoi(e);
      if (v >= 0 && v < num_slots_total()) c.variant = v;
    }
    if (const char* e = std::getenv("GDN_CARVE")) c.carve = std::atoi(e);
    if (const char* e = std::getenv("GDN_STREAM")) c.cs = std::atoi(e) != 0;
    return c;
  }();
  return cfg;
}

/* True when the slot was chosen explicitly rather than by the launch rule: either
 * GDN_V in the environment or dev.py's set_config.  The development sweep needs this,
 * because gdn_run replaces the selected slot with SCALAR_VARIANT whenever the slot
 * stride breaks 16-byte alignment - a correct production rule that also made every
 * odd-padding configuration in the sweep run the same kernel whatever slot was being
 * timed (job dv_fa6a8d312731 reported d1 within 0.13% and d2 within 0.35% across
 * twelve slots that differ by 25% elsewhere).  With the slot explicitly forced, the
 * alignment-dependent choice of *instantiation* is left to ld8/st8's own ladder, which
 * is what makes an odd-padding A/B measurable in one job. */
inline bool variant_forced() {
  static const bool env_forced = std::getenv("GDN_V") != nullptr;
  return env_forced || live_cfg().forced;
}

/* The launch rule.  `blocks` is nreq*hv, `tokens` is mixed_qkv.size(0) and `nreq`
 * is state_indices.size(0); all three are host metadata the caller already has, so
 * nothing here reads a tensor's contents, syncs, or consults a case, seed or
 * benchmark phase.
 *
 * REWRITTEN 2026-10-28 on dv_e2d8c1ba2b29 + the same job's decomposition probe.
 * Three changes, each with the measurement that forced it:
 *
 * 1. A VECTOR grid of any size takes CPA_VARIANT.  The old rule sent everything at
 *    or below MULTI_WAVE_BLOCKS to SMALL_GRID_VARIANT (39), and 39 is the one row
 *    in the promoted family with neither the row split nor the cp.async staging.
 *    At hv=4, count 2, 292 and 296 blocks - i.e. inside the old sub-wave band -
 *    slot 72 measures 12.304 us against the pick's 14.304, -14.0%, and at 256
 *    blocks the synthetic cells give -13.4% (uniform count 2), -6.3% ([4,0]) and
 *    0.0% ([3,1]).  On the frozen suite the same flip is -0.10% on fixed:5 and
 *    0.00% on d0/d5, so it is a general-domain gain with no scored regression.
 *    A SCALAR grid keeps 39 below the threshold: d7 (192 blocks, odd padding)
 *    measures 26.864 on 39 against 28.704 on 72, +6.85%, and every row-split slot
 *    is ~28.7 there, because slot 66's and 72's VEC=0 twins are the spilling
 *    instantiations.  That is the same reason `vec` gates the band below.
 *
 * 2. The P=1 band's upper bound moves from one wave of its capacity (592) to two
 *    (1184), and its pick moves from slot 71 to slot 77, which is 71 plus the
 *    remap bounded to two waves of 592.  At 1024 blocks the band's own row is
 *    1.73 waves and the remap fires: fixed:3 30.720 against slot 72's 32.768
 *    (-6.25%, 3/3 rounds) and fixed:7 30.768 against 32.768 (-6.10%, 3/3).  Slot
 *    71 without the remap is only -2.78% and 0.00% on those two, so the remap is
 *    what pays there, and it is switched off by its own wave bound everywhere the
 *    band reaches below one wave - which is why the same row is also correct at
 *    512 blocks, where the remap measured +1.97 us on fixed:4.
 *
 * 3. The band's shape predicate becomes the mean token count; see P1_MAX_MEAN.
 *
 * The band is applied last so that inside its own range it overrides the multi-wave
 * pick rather than competing with it, exactly as before. */
inline Cfg pick_cfg(int64_t blocks, int64_t tokens, int64_t nreq, bool vec) {
  Cfg cfg = live_cfg();
  if (!variant_forced()) {
    if (vec) {
      cfg.variant = CPA_VARIANT;
    } else {
      cfg.variant = (blocks > MULTI_WAVE_BLOCKS) ? MULTI_WAVE_VARIANT
                                                 : SMALL_GRID_VARIANT;
    }
    if (vec && nreq > 0 && tokens <= (int64_t)P1_MAX_MEAN * nreq) {
      if (P1_BLOCKS_LO < blocks && blocks <= P1_BLOCKS_HI)
        cfg.variant = P1_VARIANT;
      else if (P1_BLOCKS_HI < blocks && blocks <= P1B_BLOCKS_HI)
        cfg.variant = P1B_VARIANT;
    }
  }
  return cfg;
}

/* The odd-slot-padding substitution, in one place so the run path and the
 * `which()` report cannot disagree about which slot will launch.  An explicitly
 * forced slot keeps its own identity (the development sweep needs that), and so
 * does a geometry pick: MULTI_WAVE_VARIANT has the same two VEC instantiations
 * as the default, and `vec` at the launch site already chooses between them, so
 * rewriting the pick here would silently drop the mechanism on exactly the
 * misaligned multi-wave configurations it pays most on (domain d2, padding 7,
 * 4096 blocks: 141.09 -> 122.82 us in job dv_e26507e69078). */
inline int scalar_substitute(int variant, bool vec) {
  if (vec || variant_forced() || variant != DEFAULT_VARIANT) return variant;
  return SCALAR_VARIANT;
}

inline void set_cfg(int variant, int carve, int cs) {
  Cfg& c = live_cfg();
  if (variant >= 0 && variant < num_slots_total()) {
    c.variant = variant;
    c.forced = true;
  }
  if (carve >= -1 && carve <= 100) c.carve = carve;
  if (cs >= 0 && cs <= 1) c.cs = cs;
}

/* Undo set_cfg and return the process to the launch rule's own choice, including
 * its environment-derived default.  Development only; gdn_run never calls it.
 *
 * It exists because `forced` is sticky and a sweep that forces a slot cannot
 * afterwards ask pick_cfg what it WOULD have chosen - `which()` answers from the
 * same live Cfg the launch does, which is exactly the property that makes it a
 * trustworthy report and exactly what makes it useless mid-sweep.  The first
 * probe of this line (job dv_d35c04837372) hit that: it recorded `which()` per
 * shape AFTER a previous shape's sweep had forced slot 71, so nineteen of its
 * twenty "auto" labels named slot 71 and every gain was measured against the
 * wrong control.  The per-slot timings in that job were still valid, and the
 * picks were re-derived offline from pick_cfg's own rule, but a sweep that cannot
 * name its control is a sweep that has to be re-run. */
inline void clear_cfg() {
  Cfg& c = live_cfg();
  c.variant = DEFAULT_VARIANT;
  c.carve = -1;
  c.cs = 1;
  c.forced = false;
  if (const char* e = std::getenv("GDN_V")) {
    const int v = std::atoi(e);
    if (v >= 0 && v < num_slots_total()) {
      c.variant = v;
      c.forced = true;
    }
  }
  if (const char* e = std::getenv("GDN_CARVE")) c.carve = std::atoi(e);
  if (const char* e = std::getenv("GDN_STREAM")) c.cs = std::atoi(e) != 0;
}

#define GDN_PARAMS \
  dim3 grid, cudaStream_t stream, int carve, const Params& prm
#define GDN_FWD grid, stream, carve, prm

/* Shared memory per SM on this device class; only used to size the carveout. */
constexpr int SMEM_PER_SM = 233472;

template <typename KF, int MINB, int BYTES>
void apply_attrs(KF kernel, int carve) {
  static int applied_carve = -12345;
  if (applied_carve == carve) return;
  applied_carve = carve;
  if (BYTES > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         BYTES);
  }
  int want = carve;
  if (want < 0) {
    const long need = (long)MINB * (BYTES + 1024);
    want = (int)((100 * need + SMEM_PER_SM - 1) / SMEM_PER_SM);
    if (want > 100) want = 100;
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                       want);
}

template <int P, int NA, bool F32X2, bool CS, int MINB, bool MERGE, bool EPI2,
          bool VEC>
void launch_one(GDN_PARAMS) {
  using A = Arena<P>;
  auto* const kernel = gdn_kernel<P, NA, F32X2, CS, MINB, MERGE, EPI2, VEC>;
  apply_attrs<decltype(kernel), MINB, A::BYTES>(kernel, carve);
  kernel<<<grid, D * P, A::BYTES, stream>>>(prm);
}


/* Same launch path for the wide lane mapping; the block size, the shared arena
 * and the grid are unchanged, only the kernel template differs. */
template <int P, int LPR, int NA, bool F32X2, bool CS, int MINB, int EPIM,
          bool ALD, bool ST4, bool QK32, bool PACK, bool PUB1, bool VEC,
          bool CSB = false, bool RPERM = false, bool ASA = false,
          bool ASAL = false, bool CSBODD = false, bool CCSEAM = false,
          bool ROWSEAM = false, bool GRID2D = false, bool IDXPAR = false,
          bool NOGATE = false, bool HQFAST = false, int LDB = 0,
          bool NWE = false, bool KQREG = false, bool PLOV = false,
          bool HOIST = false, bool NOUNROLL = false, int CM = -1,
          bool LPT = false, int LPTW = 0, int LPT2W = 0, int PFW = 0,
          bool PFV = false, bool TRIM = false, bool RSPLIT = false,
          bool PFT = false, int PF2 = 0, bool CPA = false, bool TSP = false,
          bool PFL = false>
void launch_one_w(GDN_PARAMS) {
  using A = Arena<P, QK32, PACK, LPT, (LPT2W > 0), RSPLIT, CPA, TSP>;
  auto* const kernel = gdn_kernel_w<P, LPR, NA, F32X2, CS, MINB, EPIM, ALD, ST4,
                                    QK32, PACK, PUB1, VEC, CSB, RPERM, ASA, ASAL,
                                    CSBODD, CCSEAM, ROWSEAM, GRID2D, IDXPAR,
                                    NOGATE, HQFAST, LDB, NWE, KQREG, PLOV,
                                    HOIST, NOUNROLL, CM, LPT, LPTW, LPT2W, PFW,
                                    PFV, TRIM, RSPLIT, PFT, PF2, CPA, TSP, PFL>;
  apply_attrs<decltype(kernel), MINB, A::BYTES>(kernel, carve);
  kernel<<<grid, D * P, A::BYTES, stream>>>(prm);
}

/* Slot dispatch is derived from VARIANTS[] rather than written out again, so the
 * table and the launcher cannot disagree about which slots exist or what tuple a
 * slot means.  This is not cosmetics: a hand-written switch once gained a VARIANTS
 * entry (slot 18, PUB1) without its matching `case`, and the resulting
 * `TORCH_CHECK(false, "unknown gdn variant 18")` aborted dev job dv_56d378a9e090
 * in the middle of its sweep.  No local gate caught it: the type-checking stub's
 * TORCH_CHECK is a no-op (this header defines its own only inside #ifndef), so
 * the `default:` arm compiled cleanly and the gap was invisible until the device
 * run asked for the slot.  The recursion below makes the omission
 * unrepresentable - every index 0..NUM_VARIANTS-1 is reached by construction.
 *
 * Two further nets, both compile-time so they run in every gate and in the real
 * nvcc build: same_tuple() pins eleven slots' full tuples, which is enough that
 * no field permutation can survive (see its comment), and launch_slot asserts
 * that the arena the table advertises is the arena it instantiates.
 * experiments/launch_check.py checks the source table itself - contiguous
 * indices, macro arity against each #define, and that the highest slot is one of
 * the pinned ones.
 *
 * The chain is a compile-time recursion over the slot index; `if constexpr`
 * discards the launcher that does not belong to the slot, so narrow slots never
 * instantiate gdn_kernel_w and wide slots never instantiate gdn_kernel. */
template <int I, bool VEC>
inline void launch_slot(GDN_PARAMS) {
  static_assert(I >= 0 && I < NUM_VARIANTS, "slot index out of range");
  constexpr Variant c = VARIANTS[I];
  static_assert(c.P > 0 && c.lpr > 0 && c.na > 0 && c.minb > 0 && c.smem_bytes > 0,
                "slot tuple is not fully initialised");
  static_assert(c.epim >= 0 && c.epim <= 4, "epilogue mode out of range");
  static_assert(16 % c.lpr == 0 && c.lpr % c.P == 0,
                "the lane map needs lpr | 16 and P | lpr");
  if constexpr (c.wide) {
    /* The table's smem_bytes drives the carveout and the dynamic shared size at
     * the launch site, so it must be the arena this launcher instantiates.
     *
     * This assert has to MIRROR the instantiation below, not restate an assumption
     * about it, and it was wrong for exactly as long as the assumption held.
     * `launch_one_w` builds `Arena<P, QK32, PACK, LPT, (LPT2W > 0), RSPLIT>`, and this
     * launcher passes LPT=false and LPT2W=c.lpt2w -- so the fifth argument here is
     * `(c.lpt2w > 0)`. It used to read a hardcoded `false`, which was correct only
     * while every slot had lpt2w == 0 and launch_slot passed the literal 0. Wiring the
     * field through made the two disagree, and slot 65 (lpt2w=2, arena 13,296 B against
     * the launcher's 12,992 B) failed the build on it. That is the assert doing its job:
     * a wrong smem_bytes here would otherwise have set a carveout smaller than the
     * kernel's own [class][warp] table, i.e. an out-of-bounds shared write at runtime
     * rather than a compile error. */
    static_assert(Arena<c.P, c.qk32, c.pack, false, (c.lpt2w > 0), c.rsplit,
                      c.cpa, c.tsp>::BYTES == c.smem_bytes,
                  "slot arena disagrees with the launcher's");
    /* IDXPAR is the table's `c.idxpar` and not the literal `false` this used to
     * be: slots 73/75 are the promoted-table arms of a mechanism that until now
     * only the EXP slot space could instantiate.  Every row appended before
     * 2026-10-28 leaves the new field at its default of false, so the template
     * argument they receive is the same constant it always was and their codegen
     * cannot move.  NOGATE stays a literal false - it is measured at -4.5% and
     * is wired into the flat epilogue only, so it has no promoted arm. */
    launch_one_w<c.P, c.lpr, c.na, c.f32x2, c.cs, c.minb, c.epim, c.ald, c.st4,
                 c.qk32, c.pack, c.pub1, VEC, c.csb, c.rperm, c.asa,
                 c.asal, c.csbodd, c.ccseam, c.rowseam, c.grid2d, c.idxpar, false,
                 c.hqfast, 0, false, false, false, false, false, c.cm, false, 0,
                 c.lpt2w, c.pfw, c.pfv, c.trim, c.rsplit, c.pft, c.pf2,
                 c.cpa, c.tsp, c.pfl>(GDN_FWD);
  } else {
    static_assert(Arena<c.P>::BYTES == c.smem_bytes,
                  "slot arena disagrees with the launcher's");
    launch_one<c.P, c.na, c.f32x2, c.cs, c.minb, c.merge, c.epim != 0, VEC>(
        GDN_FWD);
  }
}

template <int I, bool VEC>
inline void launch_from(int variant, GDN_PARAMS) {
  if (variant == I) {
    launch_slot<I, VEC>(GDN_FWD);
    return;
  }
  if constexpr (I + 1 < NUM_VARIANTS) {
    launch_from<I + 1, VEC>(variant, GDN_FWD);
  } else {
    TORCH_CHECK(false, "unknown gdn variant ", variant);
  }
}

template <bool VEC>
void launch_variant(int variant, GDN_PARAMS) {
  launch_from<0, VEC>(variant, GDN_FWD);
}

/* ------------------------------------------------------------------ */
/* Experimental mechanism table (ids >= EXP_BASE).
 *
 * A second slot space so a candidate mechanism can be A/B'd against the promoted
 * package WITHOUT touching the promoted table: every entry here is slot 28's exact
 * tuple (P2/LPR8/NA1/EPIM2/MINB2, CS, ALD, ST4, RPERM, ASAL, CCSEAM) with one or
 * more flags below flipped, and slot EXP_BASE flips none of them - it instantiates
 * the SAME kernel as slot 28 and is therefore this job's identical-code control.
 * experiments/launch_check.py asserts that equivalence rather than trusting it.
 *
 * GRID2D  launch dim3(hv, nreq) instead of dim3(nreq*hv).  The linear dispatch
 *         order is unchanged (block (x,y) linearises to x + y*gridDim.x, exactly
 *         bid = r*hv + j with the head fastest), so locality and the tail are
 *         untouched; what disappears is the signed division and remainder by the
 *         runtime head count at the top of every block, which sits on the critical
 *         path to the block's first global load because r indexes cu_seqlens,
 *         num_accepted_tokens and state_indices.  PC sampling on fixed case 0 puts
 *         the IABS of that sequence alone at 5.2% of all stall samples.
 * IDXPAR  fetch the whole state_indices row with cu_seqlens/num_accepted_tokens
 *         instead of after them, collapsing three serial memory round trips at
 *         block start into two.
 * NOGATE  stop staging the output-gate vector in shared: the recurrence never
 *         reads it, only the flat epilogue does, once per output element, from an
 *         L2-warm line.
 *
 * Rejected before it reached a slot: redux.sync.add.f32, which would have replaced
 * the two per-row dot butterflies and the prologue's QK-norm reduction with one
 * instruction each (~40 of a step's ~250 instructions per warp).  ptxas 13.1
 * rejects it for sm_103a - "Incorrect type '.f32' for operation '.add' in
 * instruction 'redux'" - so this target has integer warp reductions only.  Job
 * dv_d4939d2b3d39 is that answer; the flag is gone rather than left to rot.
 */
struct ExpVariant {
  int lpr;
  int p;
  int na;
  int epim;
  int minb;
  bool cs;
  bool ald;
  bool st4;
  bool rperm;
  bool asal;
  bool ccseam;
  bool grid2d;
  bool idxpar;
  bool nogate;
  bool hqfast;
  int smem_bytes;
  const char* name;
  /* Fields an aggregate may omit, in order, and the default each one takes.  The
   * three memory-flight flags are appended AFTER `name` so GDN_EXP keeps emitting
   * 17 values and slots 200-208 keep their exact meaning. */
  int ldb = 0;
  bool nwe = false;
  bool kqreg = false;
  bool pack = false;
  bool plov = false;
  bool hoist = false;
  bool nounroll = false;
  /* Cache-hint split, -1 = follow `cs` for both directions (every historical
   * slot).  Bit 0 streams the source-tile loads, bit 1 streams the checkpoint
   * stores; see CM in gdn_kernel_w. */
  int cm = -1;
  /* Longest-first request dispatch; see the LPT block in gdn_kernel_w. */
  bool lpt = false;
  /* The warp-redundant form of the same remap, and the largest wave count it is
   * allowed to run at; 0 disables it.  See LPTW in gdn_kernel_w. */
  int lptw = 0;
  /* The two-barrier form of the remap, and the largest wave count it runs at; 0
   * disables it.  See LPT2 in gdn_kernel_w. */
  int lpt2w = 0;
};

#define GDN_EXP(G2V, IXV, NGV, HQV, TAG)                                     \
  { 8, 2, 1, 2, 2, true, true, true, true, true, true, G2V, IXV, NGV, HQV,   \
    Arena<2>::BYTES, TAG }

/* The promoted startup package (GRID2D + HQFAST, i.e. slot 39's code) with one
 * memory-flight mechanism on top.  LDB is a batch SIZE, not a flag: 0 leaves the
 * tile ladder alone, C batches the thread's whole share into one flight.  PKV
 * selects the packed per-step scalar record, which changes the arena size, so it
 * is part of the smem expression rather than a bolt-on. */
#define GDN_EXPX(G2V, IXV, NGV, HQV, LDBV, NWEV, KQV, PKV, PLOVV, HOIV, NRU, \
                 TAG)                                                        \
  { 8, 2, 1, 2, 2, true, true, true, true, true, true, G2V, IXV, NGV, HQV,   \
    Arena<2, false, PKV>::BYTES, TAG, LDBV, NWEV, KQV, PKV, PLOVV, HOIV, NRU }

/* As GDN_EXPX with the two 2026-10-21 arms explicit: CMV is the load/store cache
 * hint split (-1 keeps `cs`) and LPTV the longest-first dispatch remap, which
 * also grows the shared arena by its scratch. */
#define GDN_EXPL(G2V, IXV, NGV, HQV, LDBV, NWEV, KQV, PKV, PLOVV, HOIV, NRU, \
                 CMV, LPTV, TAG)                                              \
  { 8, 2, 1, 2, 2, true, true, true, true, true, true, G2V, IXV, NGV, HQV,   \
    Arena<2, false, PKV, LPTV>::BYTES, TAG, LDBV, NWEV, KQV, PKV, PLOVV,      \
    HOIV, NRU, CMV, LPTV }

/* As GDN_EXPL with the barrier-free remap instead of the shared-memory one:
 * LPTWV is the wave bound (0 = off), and the arena is the plain one because the
 * warp-redundant form needs no scratch. */
#define GDN_EXPW(G2V, IXV, NGV, HQV, LDBV, NWEV, KQV, PKV, PLOVV, HOIV, NRU, \
                 CMV, LPTWV, TAG)                                             \
  { 8, 2, 1, 2, 2, true, true, true, true, true, true, G2V, IXV, NGV, HQV,   \
    Arena<2, false, PKV>::BYTES, TAG, LDBV, NWEV, KQV, PKV, PLOVV,            \
    HOIV, NRU, CMV, false, LPTWV }

/* The two-barrier remap: its arena carries the small count table, and `lptw`
 * stays 0 because the wave bound lives in the LPT2 block itself. */
#define GDN_EXP2(G2V, IXV, NGV, HQV, LDBV, NWEV, KQV, PKV, PLOVV, HOIV, NRU, \
                 CMV, L2WV, TAG)                                              \
  { 8, 2, 1, 2, 2, true, true, true, true, true, true, G2V, IXV, NGV, HQV,   \
    Arena<2, false, PKV, false, (L2WV > 0)>::BYTES, TAG, LDBV, NWEV, KQV,     \
    PKV, PLOVV, HOIV, NRU, CMV, false, 0, L2WV }

constexpr ExpVariant EXP_VARIANTS[] = {
  /* 200 */ GDN_EXP(false, false, false, false, "EXP/CTL"),  /* == slot 28, same instantiation */
  /* 201 */ GDN_EXP(true, false, false, false, "EXP/G2"),    /* 2-D grid: no division by hv */
  /* 202 */ GDN_EXP(false, true, false, false, "EXP/IX"),    /* parallel index-row prefetch */
  /* 203 */ GDN_EXP(false, false, true, false, "EXP/NG"),    /* gate read in the epilogue */
  /* 204 */ GDN_EXP(true, true, false, false, "EXP/G2IX"),
  /* 205 */ GDN_EXP(true, true, true, false, "EXP/G2IXNG"),
  /* 206 */ GDN_EXP(false, false, false, true, "EXP/HQ"),    /* no division for the q head */
  /* 207 */ GDN_EXP(true, false, false, true, "EXP/G2HQ"),   /* both startup divisions gone */
  /* 208 */ GDN_EXP(true, true, false, true, "EXP/G2IXHQ"),
  /* 209 */ GDN_EXPX(true, false, false, true, 8, false, false, false, false, false, false, "EXP/LDB8"),
  /* 210 */ GDN_EXPX(true, false, false, true, 0, true, false, false, false, false, false, "EXP/NWE"),
  /* 211 */ GDN_EXPX(true, false, false, true, 8, true, false, false, false, false, false, "EXP/LDB8NWE"),
  /* 212 */ GDN_EXPX(true, false, false, true, 0, false, true, false, false, false, false, "EXP/KQREG"),
  /* 213 */ GDN_EXPX(true, false, false, true, 4, false, false, false, false, false, false, "EXP/LDB4"),
  /* 214 */ GDN_EXPX(true, false, false, true, 8, true, true, false, false, false, false, "EXP/LDB8NWEKQ"),
  /* 215 */ GDN_EXPX(true, false, false, true, 0, false, false, true, false, false, false, "EXP/PACK"),
  /* 216 */ GDN_EXPX(true, false, false, true, 8, true, false, true, false, false, false, "EXP/LDB8NWEPK"),
  /* 217 */ GDN_EXPX(true, false, false, true, 8, true, true, true, false, false, false, "EXP/LDB8NWEKQPK"),
  /* 218 */ GDN_EXPX(true, false, false, true, 0, false, false, false, true, false, false, "EXP/PLOV"),
  /* 219 */ GDN_EXPX(true, false, false, true, 0, true, false, false, true, false, false, "EXP/PLOVNWE"),
  /* 220 */ GDN_EXPX(true, false, false, true, 0, false, false, true, true, false, false, "EXP/PLOVPK"),
  /* 221 */ GDN_EXPX(true, false, false, true, 0, true, true, true, true, false, false, "EXP/PLOVALL"),
  /* 222 */ GDN_EXPX(true, false, false, true, 0, false, false, false, false, true, false, "EXP/HOIST"),
  /* 223 */ GDN_EXPX(true, true, false, true, 0, false, false, false, false, true, false, "EXP/HOISTIX"),
  /* 224 */ GDN_EXPX(true, true, false, true, 0, true, false, false, true, true, false, "EXP/HOIXPLOVNWE"),
  /* 225 */ GDN_EXPX(true, false, false, true, 0, false, false, false, false, false, true, "EXP/NOUNROLL"),
  /* 226 */ GDN_EXPX(true, true, false, true, 0, true, false, true, false, true, true, "EXP/HOIXNRPK"),
  /* --- 2026-10-21 L20D campaign: the dispatch-order and cache-hint arms.
   * LPT reorders the requests longest-first (see the LPT block in the kernel);
   * CM splits the one streaming hint into a load hint and a store hint, which
   * matters because this device has 126.5 MiB of L2 and the fixed suite's short
   * shapes touch 28-80 MiB, i.e. a checkpoint stream that fits in cache is being
   * pushed to DRAM by an evict-first store.  CM bit 0 streams the source-tile
   * loads, bit 1 streams the checkpoint stores, so cm=3 is today's CS=true and
   * cm=-1 (every slot above) keeps following `cs`. */
  /* 227 */ GDN_EXPL(true, false, false, true, 0, false, false, false, false, false, false, -1, true,  "EXP/LPT"),
  /* 228 */ GDN_EXPL(true, false, false, true, 0, false, false, false, false, false, false, 0, true,   "EXP/LPTCM0"),
  /* 229 */ GDN_EXPL(true, false, false, true, 0, false, false, false, false, false, false, 0, false,  "EXP/CM0"),
  /* 230 */ GDN_EXPL(true, false, false, true, 0, false, false, false, false, false, false, 1, false,  "EXP/CM1"),
  /* 231 */ GDN_EXPL(true, false, false, true, 0, false, false, false, false, false, false, 1, true,   "EXP/LPTCM1"),
  /* 232 */ GDN_EXPL(true, false, false, true, 0, false, false, false, false, false, false, 2, false,  "EXP/CM2"),
  /* The warp-redundant remap: no shared memory, no barrier, and the count-array
   * load replaces the cup[r] load the identity path issues anyway.  233 runs it
   * at any wave count, 234 only up to two waves, which is where the measured
   * crossover sits. */
  /* 233 */ GDN_EXPW(true, false, false, true, 0, false, false, false, false, false, false, -1, 99, "EXP/LPTW"),
  /* 234 */ GDN_EXPW(true, false, false, true, 0, false, false, false, false, false, false, -1, 2,  "EXP/LPTW2"),
  /* 235: the two-barrier remap, bounded in the kernel to grids of one to two
   * waves.  Same permutation as slot 227 at about a quarter of the per-block
   * cost, which is what the uniform-count configurations were paying 5% for. */
  /* 235 */ GDN_EXP2(true, false, false, true, 0, false, false, false, false, false, false, -1, 2, "EXP/LPT2W2"),
  /* 236 */ GDN_EXP2(true, false, false, true, 0, false, false, false, false, false, false, -1, 4, "EXP/LPT2W4"),
  /* 237 */ GDN_EXP2(true, false, false, true, 0, false, false, false, false, false, false, -1, 99, "EXP/LPT2W99"),
};
constexpr int NUM_EXP = (int)(sizeof(EXP_VARIANTS) / sizeof(EXP_VARIANTS[0]));
#undef GDN_EXP
#undef GDN_EXPX
#undef GDN_EXPL
#undef GDN_EXPW
#undef GDN_EXP2

constexpr int EXP_BASE = 200;
static_assert(EXP_BASE >= NUM_VARIANTS,
              "the experimental slot space must not overlap the promoted one");
constexpr int NUM_SLOTS_ALL = EXP_BASE + NUM_EXP;

template <int I, bool VEC>
inline void launch_exp_slot(GDN_PARAMS) {
  static_assert(I >= 0 && I < NUM_EXP, "experimental slot index out of range");
  constexpr ExpVariant c = EXP_VARIANTS[I];
  static_assert(Arena<c.p, false, c.pack, c.lpt, (c.lpt2w > 0)>::BYTES ==
                    c.smem_bytes,
                "exp slot arena mismatch");
  launch_one_w<c.p, c.lpr, c.na, true, c.cs, c.minb, c.epim, c.ald, c.st4,
               false, c.pack, false, VEC, false, c.rperm, false, c.asal, false,
               c.ccseam, false, c.grid2d, c.idxpar, c.nogate, c.hqfast, c.ldb,
               c.nwe, c.kqreg, c.plov, c.hoist, c.nounroll, c.cm, c.lpt, c.lptw,
               c.lpt2w>(GDN_FWD);
}

template <int I, bool VEC>
inline void launch_exp_from(int slot, GDN_PARAMS) {
  if (slot == EXP_BASE + I) {
    launch_exp_slot<I, VEC>(GDN_FWD);
    return;
  }
  if constexpr (I + 1 < NUM_EXP) {
    launch_exp_from<I + 1, VEC>(slot, GDN_FWD);
  } else {
    TORCH_CHECK(false, "unknown experimental variant ", slot);
  }
}

template <bool VEC>
void launch_exp(int slot, GDN_PARAMS) {
  launch_exp_from<0, VEC>(slot, GDN_FWD);
}

/* Declared above the configuration block, defined here because the slot space is
 * only complete once both tables exist. */
inline int num_slots_total() { return NUM_SLOTS_ALL; }

/* True when the selected experimental slot wants a 2-D grid; the host has to build
 * the grid the same way the kernel decomposes it. */
inline bool exp_grid2d(int slot) {
  const int index = slot - EXP_BASE;
  return (index >= 0 && index < NUM_EXP) ? EXP_VARIANTS[index].grid2d : false;
}

/* The requested blocks per SM, so the host can size the prefetch distance the
 * same way for both slot spaces. */
inline int exp_minb(int slot) {
  const int index = slot - EXP_BASE;
  return (index >= 0 && index < NUM_EXP) ? EXP_VARIANTS[index].minb : 2;
}

}  // namespace
}  // namespace gdn
