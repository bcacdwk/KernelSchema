#pragma once
// Shared device helpers for the GDN prefill candidate (pure CUDA C++/PTX, L20D sm_103).
// Included by gdn_kernel.cu and by experiments/mma_unit.py test kernels so the
// exact fragment-load patterns are validated on-device.
//
// Math (mirrors problem/reference.py fp32 block solve, block=64):
//   b_t   = inclusive cumsum of log-decay g within the chunk (fp32, left-to-right)
//   D[i,j]= exp(b_i - b_j) for i>=j else 0 ;  A_t = exp(b_t) ; scale = 128^-0.5
//   gram  = k k^T ;  system = I + tril(beta_i * D_ij * gram_ij, -1)
//   Tinv  = system^{-1} (unit lower triangular, fp32-exact SIMT hierarchical inverse)
//   rhs   = beta * (v - A * (k @ S))        S = [K,V] fp32 state at chunk start
//   u     = Tinv @ rhs
//   o     = A * scale * (q @ S) + (scale*(q k^T) * D) @ u        (stored bf16)
//   S'    = exp(b_last) * S + k^T @ (D_last * u),  D_last[t] = exp(b_last - b_t)
//   final_state[n,h,v,k] = S[k,v]  (storage layout is [V,K]; internal is [K,V])
//
// Precision: bf16 tensor-core mma.sync (fp32 accumulate). q/k/v inputs are exact
// bf16, so gram/qk products are exact-to-fp32. fp32-derived operands (S, Tinv,
// rhs, u, QKd) are split into bf16 hi+lo residual planes (~2^-17 relative), and
// products keep hi*hi + hi*lo + lo*hi terms => near-fp32 accuracy on the state
// chain, which the fp32 final_state tolerance (atol .01, zero mismatch) needs.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#define DEVINL __device__ __forceinline__

namespace gdn {

constexpr int CDIM = 64;    // chunk tokens
constexpr int KDIM = 128;   // key dim
constexpr int VDIM = 128;   // value dim
constexpr int VT = 32;      // v-tile per chain block (vsplit = 4)
constexpr int NTHR = 256;   // threads/block (8 warps)
constexpr float QSCALE = 0.08838834764831845f;  // float(128**-0.5), reference constant

// smem tile strides (elements). NOTE: any buffer consumed by ldmatrix must have
// a row stride that is a multiple of 8 bf16 elems (16B): PL_STR=72 (144B),
// KS_STR=136 (272B), SP_STR/US_STR=40 (80B). 68 (136B) faults ldmatrix on odd rows.
constexpr int KS_STR = 136;  // k/q tiles [64][136] bf16 (272B rows)
constexpr int PL_STR = 72;   // Tinv/QKd planes [64][72] bf16 (144B rows)
constexpr int VS_STR = 40;   // v tile [64][40] bf16
constexpr int SS_STR = 36;   // state fp32 [128][36]
constexpr int SP_STR = 40;   // state planes bf16 [128][40]
constexpr int US_STR = 40;   // rhs/u/du planes [64][40] bf16

// gmem scratch per (slot,hv): TinvH, QkdH planes + b[64] fp32 vec tail.
// V2f dropped QkdL; V3d made the chain single-plane (u = TinvH@rhsH) so
// TinvL is no longer consumed -> dropped from the slot (V3e).
// E4: TRIANGLE-PACKED planes — the chain's step-4/6 ldm_a sites only ever
// read 16x16 frag tiles (m, ks) with ks <= mw (unit-lower-triangular skip),
// i.e. 10 of the 16 tiles. The slot stores those 10 tiles contiguously,
// tile-major: t(m,ks) = m*(m+1)/2 + ks, each 512B.
// E4b: each 512B tile is HALF-COLUMN-SPLIT: [cols 0-7 of 16 rows @16B
// pitch][cols 8-15 same] (2 x 256B halves). A plain [16][32B] tile would
// put rows r and r+4 in the SAME smem banks (32B pitch => 2-way ldmatrix
// conflicts, measured +4-7% on the vt16 chain cases, dv_447d024dea8e);
// the 16B-pitch halves rotate banks 4r mod 32 = conflict-free in every
// x4 phase, and ldm_a_pk's lane-group addressing reproduces the PTX A-frag
// register order a0..a3 exactly. -42% plane bytes preserved.
// BITWISE: identical values, layout only; skipped tiles were exact zeros
// or never read. Containment wire untouched (all 64 rows kept per tile).
constexpr int TRI_TILES = 10;
constexpr int PLANE_PACKED_BYTES = TRI_TILES * 512;          // 5120
constexpr int SLOT_BYTES = 2 * PLANE_PACKED_BYTES + 4 * CDIM * 4;  // 11264 (E4; was 19456), 1024-multiple
DEVINL int tri_tile(int m, int ks) { return (m * (m + 1) / 2 + ks) * 256; }  // elems
// V-J2 (arm-a V5.35a port): the ONLY k/q tile form tree-wide — 4 SWIZZLE_128B
// sub-tiles (kT0 kT1 qT0 qT1, 8,192B each; two {64,64,1} TMA boxes per tile).
// P1's TMA prologue writes this into smem; s2g emits the contiguous image to
// scratchq; the chain (bulk AND KQTMA paths) consumes it via swz_addr
// ldmatrix (unit-bench verified incl .trans: arm-a results/swz-ldm-unit.txt).
constexpr int KQ_TMA_BYTES = 4 * 8192;
// V2e: P1-stored q/k tile per (slot, qk-head) for the chain's stage load.
// V-J2: SWIZZLE image [kT0 kT1 qT0 qT1] — was the padded [128][KS_STR]
// 34,816B form (−2,048B per tile of s2g/stage traffic).
constexpr int KSQB_BYTES = KQ_TMA_BYTES;

DEVINL uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Force a 16B-aligned base for dynamic shared memory (float4/ldmatrix safety).
// V5.15a: OFFSET-based round-up as a macro at the call site — the old
// uintptr_t/inttoptr round-trip destroyed NVVM addrspace(3) provenance, so
// EVERY plain C++ deref of smem-derived pointers (E1 sVs reads, plane
// read/write, final-state staging) emitted as generic LD.E/ST.E: tracked on
// the LONG scoreboard and routed through the L1TEX global path (chain vt64
// census dv_e0ef699f8762: long_sb 13.4%, epilogue STG long_sb 692 samples).
// p + int keeps the shared qualification -> LDS/STS (short scoreboard).
#define SMEM_ALIGN16(p) \
  ((p) + (int)((16u - (reinterpret_cast<uintptr_t>(p) & 15u)) & 15u))
DEVINL char* smem_align16(char* base) {
  return base + (int)((16u - (reinterpret_cast<uintptr_t>(base) & 15u)) & 15u);
}

DEVINL void cp_async16(void* dst, const void* src, bool pred) {
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(sz));
}
DEVINL void cp_async8(void* dst, const void* src, bool pred) {
  int sz = pred ? 8 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(sz));
}
DEVINL void cp_async4(void* dst, const void* src, bool pred) {
  int sz = pred ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(sz));
}
DEVINL void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
DEVINL void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

DEVINL void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A[16,16] row-major frag via ldmatrix.x4 (base tile origin offsets row0/col0).
DEVINL void ldm_a(uint32_t* a, const __nv_bfloat16* base, int stride, int row0, int col0) {
  int lane = threadIdx.x & 31;
  int m = lane >> 3, r = lane & 7;
  int row = row0 + (m & 1) * 8 + r;
  int col = col0 + (m >> 1) * 8;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem_u32(base + row * stride + col)));
}

// E4b: A-frag m16k16 from a half-split packed tile (512B, two 8-col 16B-
// pitch halves). Lane group g (lanes 8g..8g+7) addresses half (g>>1),
// rows 8*(g&1)..+8 -> dest regs = {a0:(m0-7,k0-7), a1:(m8-15,k0-7),
// a2:(m0-7,k8-15), a3:(m8-15,k8-15)} = the PTX m16n8k16 A order.
// Bank pattern per phase: 8 rows at 16B pitch = 4r mod 32 distinct
// (a plain [16][32B] tile would 2-way conflict: rows r, r+4 share banks).
DEVINL void ldm_a_pk(uint32_t* a, const __nv_bfloat16* tile) {
  int lane = threadIdx.x & 31;
  uint32_t addr = smem_u32(tile + ((lane >> 4) & 1) * 128 + (lane & 15) * 8);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}

// A frag = src^T[16,16] where src stored row-major [16 rows][16 cols] (trans pattern).
DEVINL void ldm_a_trans(uint32_t* a, const __nv_bfloat16* base, int stride, int row0, int col0) {
  int lane = threadIdx.x & 31;
  int m = lane >> 3, r = lane & 7;
  int srow = row0 + ((m >> 1) & 1) * 8 + r;
  int scol = col0 + (m & 1) * 8;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem_u32(base + srow * stride + scol)));
}

// B k16 x n16 (two n8 tiles -> regs b[0..3]) from k-major [k][n] storage: x4.trans.
DEVINL void ldm_b2_kmajor(uint32_t* b, const __nv_bfloat16* base, int stride, int k0, int n0) {
  int lane = threadIdx.x & 31;
  int m = lane >> 3, r = lane & 7;
  int krow = k0 + (m & 1) * 8 + r;
  int ncol = n0 + (m >> 1) * 8;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
               : "r"(smem_u32(base + krow * stride + ncol)));
}

// B k16 x n8 from k-major [k][n] storage: x2 variant (half the L1 bytes of
// x4 when only one n8 tile is consumed - NPER==1 / VTT=16 paths). Lanes
// 16-31 supply valid-but-ignored addresses. Bit-identical regs to x4 b[0..1].
DEVINL void ldm_b1_kmajor(uint32_t* b, const __nv_bfloat16* base, int stride, int k0, int n0) {
  int lane = threadIdx.x & 31;
  int m = lane >> 3, r = lane & 7;
  int krow = k0 + (m & 1) * 8 + r;
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(smem_u32(base + krow * stride + n0)));
}

// B k16 x n16 from n-major [n][k] storage (rows n, cols k): x4 no trans.
DEVINL void ldm_b2_nmajor(uint32_t* b, const __nv_bfloat16* base, int stride, int k0, int n0) {
  int lane = threadIdx.x & 31;
  int m = lane >> 3, r = lane & 7;
  int nrow = n0 + ((m >> 1) & 1) * 8 + r;
  int kcol = k0 + (m & 1) * 8;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
               : "r"(smem_u32(base + nrow * stride + kcol)));
}

// ---- V-J2 (arm-a V5.33a/V5.35b port): swizzle-address ldmatrix for TMA
// sub-tile q/k images ----
// Unit-bench VERIFIED (arm-a results/swz-ldm-unit.txt + swz-ldm-unit-v2.txt,
// dv_14a371482a8f): all three patterns produce frags BITWISE-EQUAL to their
// plain KS_STR-stride twins when the tile is written by TWO {64,64,1} TMA
// boxes with CU_TENSOR_MAP_SWIZZLE_128B (128B inner-extent constraint => two
// 64-key sub-tiles of 8,192B each, adjacent). Includes .trans (the chain's
// at8 K^T pattern). Address model (dst-address-derived phase, bench-confirmed
// base-invariant at every phase):
//   element (row, col) of the logical [64][128] tile:
//     s = col>>6 (sub-tile); k = col&63; line = (base>>7) + row
//     addr = base + s*8192 + row*128 + ((((k>>3) ^ (line&7))<<4) | ((k&7)<<1))
//   16-key-wide frags never cross the sub-tile boundary (16 | 64).
// LESSON 19 (arm-a bench T4): TMA SWIZZLE bakes the WRITER dst phase into the
// bytes — the s2g->bulk-in image handoff is phase-transparent ONLY when the
// writer (P1 sK) and reader (chain stage) bases match mod 1024; both are
// 1024B-aligned tree-wide (V-J2), making every base phase 0.
DEVINL uint32_t swz_addr(uint32_t base, int row, int col) {
  const int s = col >> 6, k = col & 63;
  const int line = (int)(base >> 7) + row;
  return base + (uint32_t)s * 8192u + (uint32_t)row * 128u +
         (uint32_t)((((k >> 3) ^ (line & 7)) << 4) | ((k & 7) << 1));
}
DEVINL void ldm_a_swz(uint32_t* a, uint32_t base, int row0, int col0) {
  int lane = threadIdx.x & 31;
  int m = lane >> 3, r = lane & 7;
  int row = row0 + (m & 1) * 8 + r;
  int col = col0 + (m >> 1) * 8;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(swz_addr(base, row, col)));
}
DEVINL void ldm_a_trans_swz(uint32_t* a, uint32_t base, int row0, int col0) {
  int lane = threadIdx.x & 31;
  int m = lane >> 3, r = lane & 7;
  int srow = row0 + ((m >> 1) & 1) * 8 + r;
  int scol = col0 + (m & 1) * 8;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(swz_addr(base, srow, scol)));
}
DEVINL void ldm_b2_nmajor_swz(uint32_t* b, uint32_t base, int k0, int n0) {
  int lane = threadIdx.x & 31;
  int m = lane >> 3, r = lane & 7;
  int nrow = n0 + ((m >> 1) & 1) * 8 + r;
  int kcol = k0 + (m & 1) * 8;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
               : "r"(swz_addr(base, nrow, kcol)));
}
// LESSON 20 (arm-a V5.35b): under the 1024B base invariant, at EVERY frag
// site phase == lane&7 (row0 and m*8 terms are multiples of 8) and k&7 == 0
// (cols are multiples of 8) => addr = rowbase + xg + imm with the xg =
// ((kg ^ (lane&7))<<4) group regs precomputed (lane constants, kg <= 4
// distinct per ks-loop via i = ks&3) and sub-tile/q offsets as immediates:
// ONE IADD3 per site, bit-identical to the generic form (which measured
// +25-29ns/chunk on dense-issue vt16 chains).
DEVINL void ldm_swzf(uint32_t* a, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}
DEVINL void ldm_trans_swzf(uint32_t* a, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}

DEVINL __nv_bfloat16 f_hi(float x) { return __float2bfloat16_rn(x); }
DEVINL __nv_bfloat16 f_lo(float x, __nv_bfloat16 hi) {
  return __float2bfloat16_rn(x - __bfloat162float(hi));
}
DEVINL void store_pair_hi_lo(__nv_bfloat16* hp, __nv_bfloat16* lp, float x, float y) {
  __nv_bfloat16 h0 = f_hi(x), h1 = f_hi(y);
  __nv_bfloat162 h = {h0, h1};
  __nv_bfloat162 l = {f_lo(x, h0), f_lo(y, h1)};
  *reinterpret_cast<__nv_bfloat162*>(hp) = h;
  *reinterpret_cast<__nv_bfloat162*>(lp) = l;
}

// Inclusive cumsum of 64 fp32 values in smem (in place), block-wide, left-to-right.
// Uses one warp per 32 elements + cross-warp fixup (matches torch cumsum order closely
// enough: exact same associative order as sequential for 32-wide shfl scan).
// ---- cp.async.bulk 1D + mbarrier (validated sm_103: experiments/bulk_unit.py;
// 80KB in ~320ns vs cp.async16 ~520ns from L2 and ~3.5us/88KB queue-limited
// from DRAM). Bulk writes complete in the async proxy; mbarrier wait is the
// acquire that publishes them to generic reads (ldmatrix/LDS) — no proxy fence
// needed on the consumer side.
DEVINL void bulk_expect_tx(uint32_t mbar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes));
}
// V5.13b: TMA 3D box load (v-tile) with a host-encoded per-call map passed
// as __grid_constant__. Box {VTT,1,64} over v[1,T,HV,128] -> packed [64][VTT]
// smem; OOB rows (tok0+63 >= T) zero-filled by hardware; rows >= Lc within T
// load real (finite) v — harmless: beta is zero-filled there (V5.12a), so
// rhs/DU rows >= Lc stay exact zero and dS = kT@DU cannot be poisoned.
DEVINL void tma3d_box(const CUtensorMap* map, void* dst, int c0, int c1, int c2,
                      uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3, %4}], [%5];"
               ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)),
                 "r"(c0), "r"(c1), "r"(c2), "r"(mbar) : "memory");
}
DEVINL void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
DEVINL void mbar_init(uint32_t mbar_smem, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_smem), "r"(count));
}
DEVINL void mbar_wait(uint32_t mbar_smem, uint32_t phase) {
  asm volatile("{\n\t.reg .pred P;\n\tLAB_WAIT_%=:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
               "@P bra DONE_%=;\n\tbra LAB_WAIT_%=;\n\tDONE_%=:\n\t}\n"
               ::"r"(mbar_smem), "r"(phase));
}

DEVINL void cumsum64(float* s, int tid) {
  if (tid < 64) {
    float acc = s[tid];
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      float o = __shfl_up_sync(0xffffffffu, acc, off, 32);
      if ((tid & 31) >= off) acc += o;
    }
    s[tid] = acc;
  }
  __syncthreads();
  if (tid >= 32 && tid < 64) s[tid] += s[31];
}

}  // namespace gdn
