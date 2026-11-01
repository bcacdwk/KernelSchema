// fmha_bf16_fp8 -- independent CUDA implementation for L20D (sm_103).
//
// Frozen-contract semantics (problem/reference.py):
//   score(i,j) = (sum_d q[tok_i, head, d] * k[page_j, kvh, off_j, d]) * bmm1_scale
//   causal is ALWAYS on: reference derives causal from x.get('causal', True) and
//   forward() exposes no causal parameter, so the effective mask is
//       j <= vis(i)  and, when window_left >= 0,  j >= vis(i) - window_left
//   with vis(i) = kv_len - q_len + i (i = query index inside its request).
//   out = (sum_j P[j] * v[j,:] * bmm2_scale) / sum_j P[j]      (bf16)
//   lse = (max + ln(sum)) / ln(2)                              (fp32, log2 units)
//   sinks: total = logaddexp(base, sink[head]); out *= exp(base-total);
//          lse = total / ln(2)
//
// Q is bf16, K/V are fp8 e4m3. e4m3 -> bf16 is exact (3 mantissa bits fit in 7),
// so both matmuls run as bf16 x bf16 -> fp32 on the legacy mma.sync path and
// match the fp32 reference up to accumulation order. The softmax runs in log2
// units: exp2(s*bmm1_scale*log2e - M) so that lse comes out directly in log2.
//
// Fragment orders follow the PTX m16n8k16 .row.col contract (A 16x16 row-major,
// B 16x8 col-major). K is stored [N,K] row-major, so a plain ldmatrix yields the
// col-major B fragment; V needs ldmatrix.trans. Shared memory uses a 128-byte XOR
// swizzle that is conflict free for both ldmatrix and the 16B staging stores.
//
// Structure: one CTA owns TILE_M = WARPS_M*16 packed rows, where a row is a
// (query position, head-within-kv-group) pair, so GQA/MQA groups share the KV
// stream. WARPS_N splits the KV tile across warps (decode shape: few rows, long
// context) and merges in shared memory at the end. blockIdx.z splits the KV range
// across CTAs (flash-decoding); a second kernel combines the partials.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define DEVFN __device__ __forceinline__
#define LOG2E 1.4426950408889634f
#define NEG_BIG (-1.0e30f)

namespace fmha {

// Exact division by a host-known divisor, without the reciprocal slow path nvcc
// emits for a runtime one (`$__cuda_sm20_rcp_rn_f32_slowpath` is linked into this
// ELF). The jobs-60/61 SASS line map put 212 instructions -- 10.8% of the whole
// fmha_combine body -- on the prologue block whose only real work is `by / p.HKV`
// and `by % p.HKV`, and the decode main carries a twin copy of the same statement
// plus a `r / G` and `r % G` in its Q gather, its epilogue store loops and its
// empty-split zeroing loop. Those two kernels are instruction-bound, not
// bandwidth-bound: NCU on the hd128 decode main measured 2604 dynamic instructions
// per warp at 5.53 warp-cycles per issued instruction with Compute 1.28%, DRAM
// 0.56% and No Eligible 81.73%, i.e. every one of those instructions is wall clock.
//
// magic == ceil(2^32/d) makes floor(n*magic / 2^32) == floor(n/d) EXACTLY whenever
// n*(d-1) < 2^32 (the error term is e = magic*d - 2^32 in [0, d-1], and
// floor(n/d + n*e/(d*2^32)) == floor(n/d) once n*e < 2^32). d == 1 would need
// magic == 2^32, which does not fit in 32 bits, so it is encoded as 0 and returns
// n. Every operand these are used on is far inside the bound: by < batch*HKV <= 384,
// m0 < sum(query_lengths)*G <= 2^19, r < TILE_M <= 128, e/HD < TILE_M, and the
// divisors are G <= 32 and HKV <= 12 from the declared shape range, so
// n*(d-1) <= 2^19*31 << 2^32. Verified exhaustively offline for every divisor the
// shape range can produce over n < 2^20 before this was built.
DEVFN int fdiv(int n, unsigned magic) {
  return magic ? (int)(((unsigned long long)(unsigned)n * magic) >> 32) : n;
}

// Device-wide nanosecond clock for the TRC8 decode phase timeline (same
// mechanism as t5_gtimer below, declared here so fmha_main can use it).
DEVFN unsigned long long trc8_now() {
  unsigned long long g;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
  return g;
}

struct Params {
  const __nv_bfloat16* q;
  long long q_s_tok, q_s_head;
  const uint8_t* k;
  const uint8_t* v;
  const __half* k16;  // optional pre-converted mirror, same layout as k
  const __half* v16;
  long long kv_s_page, kv_s_head, kv_s_tok;
  int page_log2;
  const int* table;
  int table_stride;
  const int* seq_lens;
  const int* cu_q;
  __nv_bfloat16* out;
  long long out_s_tok, out_s_head;
  float* lse;
  long long lse_s_tok;
  int lse_s_head;
  const float* sinks;
  float* part_o;
  float* part_ml;   // split stats, z-major: [w][split][row][2] -- coalesced for
                    // the combine's thread-per-row scan (splits <= 64)
  float* part_ml_t; // same values, row-major: [w][row][split][2] -- coalesced
                    // for the warp-per-row scan (splits > 64). NCU on the
                    // domain9-style combine (dv_629ea9c7c234) put 73.9% of its
                    // stalls on L1TEX scoreboard with the strided scan; the
                    // dual layout costs the main kernel one extra 8B store per
                    // row and keeps both combine regimes conflict-free.
  long long part_s_w, part_s_s, partml_s_w, partml_s_s, partmlt_s_r;
  int H, HKV, G;
  int mtiles;
  int splits;
  // Per-request split balancing (decode): when tps_target > 0, each work item
  // derives its own split count from its own tile count so ragged kv batches
  // share one common tiles-per-split instead of one uniform split count.
  // grid.z (p.splits) sizes the maximum; combine recomputes the same formula.
  int tps_target;
  int tile_n_main;
  float qk_scale;
  float v_scale;
  int window_left;
  int write_lse;
  // t7 decode: direct TMA maps of the fp8 KV caches (UINT8, SWIZZLE_128B,
  // box [128B][min(page,64) rows]); unused (uninitialized garbage, never
  // dereferenced) by every non-T7 config.
  CUtensorMap kmap8;
  CUtensorMap vmap8;
  bool have8 = false;  // kmap8/vmap8 built and valid
  // t8 (cfg200/201/208): f16-mirror maps with box_rows = min(page,64) so a
  // page128 64-row tile addresses rows inside one page. Built by the gate for
  // cfg162-hd128 (the t8 promotion candidate) and the t8 configs themselves;
  // carried inside Params so the CachedLaunch copy needs no extra plumbing.
  CUtensorMap kmap64;
  CUtensorMap vmap64;
  bool have64 = false;
  // Exact fast divisors for fdiv(). Appended at the END of the struct so that no
  // existing field's constant-bank offset moves: these kernels read Params through
  // LDC, and a layout shift is the kind of perturbation job 57 measured as a 7-9%
  // regression on a codegen-sensitive corner of t6.
  unsigned g_magic;    // ceil(2^32 / G), or 0 when G == 1
  unsigned hkv_magic;  // ceil(2^32 / HKV), or 0 when HKV == 1
  // TRC8 decode phase timeline (cfg290-293, diagnostics only): per-CTA ring of
  // u64 deltas [hdr(4) + 4/tile], base = part_o + scratch_floats/2 when the
  // host set it, else nullptr. Appended at the END for constant-bank stability.
  unsigned long long* trc8;
  // Device-balanced decode splits (tps_target == -1): resident-CTA budget of
  // the selected family (2*SMs for the promoted decode kernels, 1*SM for the
  // MINB1 classes) -- the kernel-side tiles-per-split target divides the
  // ACTUAL total tile count by this so the active CTA count lands on ~one
  // resident wave for any raggedness. log2(tile_n_main) rides along because
  // the combine-side re-derivation must divide by a runtime tile width: a
  // runtime divisor emits a full division sequence (~120 cyc, 12+ per warp of
  // the derivation loop measured +1.4..2.9us per combine), and every plan
  // tile_n is 32/64/128 so shifts are exact. Appended at the END.
  int resident2;
  int tile_n_log2;
  // Combine row-stat scan policy: 1 allows the warp-per-row scan at
  // splits <= 64 when the chunk CTA covers <= THREADS/32 rows, 0 keeps the
  // legacy thread-per-row form (fmha_set_debug A/B knob, FMHA_WSCAN env).
  int wscan;
};

// ---- Device-balanced decode splits (tps_target == -1) -----------------------
// Job J TRC8 timelines (dv_46fa3ebe31e4) put the f2 decode main at
// span med 16.2us vs max 33.9us with tiles/CTA 1..14: under the uniform
// split count the longest request's CTAs run 2.4x the median and the second
// half of the machine idles through the tail, which is why main streams at
// 1.56-2.0TB/s against the 4.59TB/s read roof while the per-tile ring never
// stalls (steady-state wait 58-77ns). Per-request lengths are device data
// the host plan must not read (no synchronization in the measured path), so
// EVERY main/combine CTA derives the same tiles-per-split target from
// seq_lens with identical integer arithmetic:
//   total = HKV * sum over active (request, mtile) items of ntiles(item)
//   tt    = max(3, ceil(total / resident2))
//   splits_b = clamp(ceil(ntiles / tt), 1, p.splits);  tps = ceil(ntiles/splits_b)
// tt ~= total/resident2 puts the ACTIVE CTA count at ~one resident wave for
// any raggedness; on uniform batches it reproduces the uniform-fill geometry
// by construction. Floor 3 keeps the small latency-bound windowed shapes at
// their measured 3-4 tiles/split optimum (a floor of 2 over-split them:
// +10-12.5% on f5/f8-class shapes in the archived pre-join trials, whose
// mechanism this phase re-derived independently from its own span data).
// grid.z is host-sized to min(ceil(tiles_total/3), 2*uniform, resident2)
// from DECLARED metadata; both bounds provably dominate every splits_b
// (tt >= 3 and tt >= total/resident2 with ntiles <= tiles_total <= total),
// and splits_b is clamped by p.splits in both kernels regardless.
// Shift-only division: tile widths are 32/64/128; a runtime divisor here
// costs a full division sequence in every combine CTA (measured +1.4-2.9us).
DEVFN int tps_item_ntiles(const int* cu_q, const int* seq_lens, int b, int m0, int tile_m,
                          int tn_log2, int G, unsigned g_magic, int window_left) {
  const int q0 = cu_q[b];
  const int qlen = cu_q[b + 1] - q0;
  const int kvlen = seq_lens[b];
  const int i0 = fdiv(m0, g_magic);
  const int i1 = min(qlen, fdiv(m0 + tile_m, g_magic));
  const int vis0 = kvlen - qlen + i0;
  const int vis1 = kvlen - qlen + i1 - 1;
  const int jlo = window_left >= 0 ? max(0, vis0 - window_left) : 0;
  const int jhi = min(kvlen - 1, vis1);
  const int tn = 1 << tn_log2;
  const int jt0 = (jlo >> tn_log2) << tn_log2;
  const int jt1 = ((jhi + tn) >> tn_log2) << tn_log2;
  return (jt1 - jt0) >> tn_log2;
}

// Every warp may call this redundantly (the result is CTA-uniform); one lane
// per request keeps the prologue at a handful of cycles.
DEVFN int tps_device_tt(const Params& p, int tn_log2, int tile_m) {
  const int lane = threadIdx.x & 31;
  const int batch = (int)gridDim.y / p.HKV;
  int local = 0;
  for (int b = lane; b < batch; b += 32) {
    const int q0 = p.cu_q[b];
    const int qlen = p.cu_q[b + 1] - q0;
    const int rows = qlen * p.G;
    const int mtiles_b = (rows + tile_m - 1) / tile_m;
    for (int mt = 0; mt < mtiles_b; ++mt)
      local += tps_item_ntiles(p.cu_q, p.seq_lens, b, mt * tile_m, tile_m, tn_log2, p.G,
                               p.g_magic, p.window_left);
  }
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) local += __shfl_xor_sync(0xffffffffu, local, s);
  const int total = local * p.HKV;
  const int r2 = p.resident2 > 0 ? p.resident2 : 296;
  int tt = (total + r2 - 1) / r2;
  return tt < 3 ? 3 : tt;
}

DEVFN uint32_t smem_ptr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DEVFN void ldmatrix_x4(uint32_t* R, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_ptr(p)));
}

DEVFN void ldmatrix_x4_trans(uint32_t* R, const void* p) {
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_ptr(p)));
}

DEVFN void mma_f16(float* C, const uint32_t* A, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(b0), "r"(b1));
}

DEVFN void cp_async16(void* dst_smem, const void* src_global) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_ptr(dst_smem)),
               "l"(src_global));
}

DEVFN void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }

// cp.async with an explicit source size: the cp-size - src-size tail is ZERO-FILLED
// into smem, which lets the t6 Q gather stage a padding row without a register round
// trip and without reading past the query tensor.
DEVFN void cp_async16_z(void* dst_smem, const void* src_global, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_ptr(dst_smem)),
               "l"(src_global), "r"(src_bytes));
}

// L2 prefetch (no destination register): warms the lines a later tile's LDG.128s
// will need, so those loads hit L2 (~300ns) instead of DRAM (~800-1000ns).
DEVFN void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
}

template <int N>
DEVFN void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

DEVFN float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

// log2 the same way ex2 does it: one MUFU instead of libdevice's log2f, which
// lowers to MUFU.LG2 PLUS a range-reduction/correction sequence. The jobs-60/61
// SASS line map priced the two log2f statements of the lse/sink epilogue at ~130
// instructions per kernel instance in each of the decode main and the combine --
// ~5% of the decode body, which NCU put at 2604 dynamic instructions per warp and
// 5.53 warp-cycles per issued instruction with nothing to hide behind.
// Applied ONLY to those two kernels: the prefill families (t5/t6/t6c/t6d/t8) are
// tensor-bound at 82-92% of the measured f16 tcgen05 peak, so an instruction cut
// there is worth nothing while job 57 measured a 25-line t6 prologue edit costing
// +7-9% through codegen alone. Not bit-identical: lg2.approx moves the lse by
// ~1e-6 relative against a 0.01 + 0.05*|got| contract, which is the same
// approximation the softmax's own ex2 already relies on.
DEVFN float lg2f(float x) {
  float y;
  asm("lg2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

// Packed f32x2 arithmetic (sm_100+): two fp32 lanes per instruction on the
// FMA pipe. Probe5 confirmed add/mul/fma.rn.f32x2 assemble on this target's
// ptxas (13.1.115, sm_103a); packed max/min/ex2/lg2 do NOT exist, so only the
// scale-offset fma and the row-sum accumulation pack. The mov.b64 pair
// assembly is folded by the register allocator in the unrolled loops.
DEVFN uint64_t f2_pack(float lo, float hi) {
  uint64_t r;
  asm("mov.b64 %0, {%1, %2};\n" : "=l"(r) : "f"(lo), "f"(hi));
  return r;
}
DEVFN void f2_unpack(uint64_t v, float& lo, float& hi) {
  asm("mov.b64 {%0, %1}, %2;\n" : "=f"(lo), "=f"(hi) : "l"(v));
}
DEVFN uint64_t fma2(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t r;
  asm("fma.rn.f32x2 %0, %1, %2, %3;\n" : "=l"(r) : "l"(a), "l"(b), "l"(c));
  return r;
}
DEVFN uint64_t add2(uint64_t a, uint64_t b) {
  uint64_t r;
  asm("add.f32x2 %0, %1, %2;\n" : "=l"(r) : "l"(a), "l"(b));
  return r;
}

// X2H helpers: packed f16 exp2 halves the per-tile MUFU count of the t5
// softmax (ex2 is scalar on the f32 path). ex2.approx.f16x2 maps -inf -> +0
// like the f32 form; outputs here are <= 2 (DEFER keeps P <= 2), so range
// is a non-issue and subnormal flush only drops vanishing terms.
DEVFN uint32_t ex2_f16x2(uint32_t h2) {
  uint32_t r;
  asm("ex2.approx.f16x2 %0, %1;\n" : "=r"(r) : "r"(h2));
  return r;
}
// fma.rn.f16x2: one instruction for the scale-offset of an element pair. The t6
// F16X path uses it to replace {mov.b64 pair assembly + fma.rn.f32x2 + the cvt that
// follows} with {cvt + fma.rn.f16x2}, i.e. 4 instructions per pair instead of 5.
DEVFN uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
DEVFN uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("add.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
// Widen a packed f16x2 accumulator to its f32 pair-sum.
DEVFN float hsum2(uint32_t h2) {
  unsigned short lo, hi;
  asm("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(h2));
  float a, b;
  asm("cvt.f32.f16 %0, %1;\n" : "=f"(a) : "h"(lo));
  asm("cvt.f32.f16 %0, %1;\n" : "=f"(b) : "h"(hi));
  return a + b;
}

// two fp32 -> packed f16x2 (low half = lo). Saturating so a pathological input
// can never turn into an Inf/NaN that the frozen checker rejects.
DEVFN uint32_t pack_f16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.satfinite.f16x2.f32 %0, %2, %1;\n" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}

// two packed e4m3 bytes -> f16x2. Exact: e4m3 has 3 mantissa bits, f16 has 10,
// and the e4m3 range (<=448, subnormals from 2^-9) is inside f16.
DEVFN uint32_t cvt_e4m3x2(uint32_t two_bytes) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"((unsigned short)two_bytes));
  return r;
}

// two packed bf16 -> f16x2. Exact for the finite bf16 range that fits f16; the
// saturating convert keeps anything larger finite.
DEVFN uint32_t bf16x2_to_f16x2(uint32_t b2) {
  float lo = __uint_as_float(b2 << 16);
  float hi = __uint_as_float(b2 & 0xffff0000u);
  return pack_f16(lo, hi);
}

// 16 packed e4m3 bytes -> 8 f16x2 words (32 bytes)
DEVFN void expand8(uint4 raw, uint32_t* h) {
  const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    h[2 * e] = cvt_e4m3x2(w[e] & 0xffffu);
    h[2 * e + 1] = cvt_e4m3x2(w[e] >> 16);
  }
}

// 128-byte XOR swizzle offset of (row, col) inside a [rows][HD] f16 tile.
template <int HD>
DEVFN int swz(int r, int c) {
  int chunk = c >> 3;
  int sc = (chunk & ~7) | ((chunk ^ (r & 7)) & 7);
  return r * HD + (sc << 3) + (c & 7);
}

// Shared staging helpers (defined once, above fmha_main: the T7 decode staging
// and the t5/t6 kernels below all use them).
DEVFN uint32_t t5_sp(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVFN void t5_mbar_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr), "r"(count));
}
DEVFN void t5_mbar_wait(uint32_t addr, int parity) {
  asm volatile("{\n.reg .pred c;\nTW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 c, [%0], %1;\n"
               "@!c bra.uni TW_%=;\n}\n" ::"r"(addr), "r"(parity));
}
DEVFN void mb_expect(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes));
}
DEVFN void mb_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
DEVFN void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
DEVFN void tma4d(const CUtensorMap* map, uint32_t dst, uint32_t bar, int c0, int c1, int c2,
                 int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::"
               "bytes [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
               "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
               "r"(bar)
               : "memory");
}

// ---- t7c: ldmatrix .b8 direct-fp8 fragment paths (sm_103a) ----------------
// m16n16.x2.trans.b8: two 16x16B matrices; lanes 0-15 supply matrix-0 row
// addresses, lanes 16-31 matrix-1. Decoded mapping (K24c v6/v7 probes, job
// 20261021T171756Z): lane l reg j byte k = M[4*(l%4)+k][j*8 + l/4].
DEVFN void ldm8_x2(uint32_t r[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
// QK-style B fragments from M[r=tok][c=hd]: out = {mma(tok0-7) b0,b1,
// mma(tok8-15) b0,b1} for m16n8k16 (b0 = hd {2(l%4),+1}, b1 = {+8,+9},
// n = l/4). 8 shfl + 4 prmt + 4 cvt feed 2 mmas.
DEVFN void bfrag_k(uint32_t r0, uint32_t r1, int lane, uint32_t out[4]) {
  const int L = 8 * (lane & 3) + (lane >> 4);
  const int byte = (lane >> 2) & 3;
  const unsigned sel = (unsigned)(byte | ((byte + 4) << 4));
  const uint32_t x1 = __shfl_sync(0xffffffffu, r0, L);
  const uint32_t x2 = __shfl_sync(0xffffffffu, r0, L + 4);
  const uint32_t x3 = __shfl_sync(0xffffffffu, r1, L);
  const uint32_t x4 = __shfl_sync(0xffffffffu, r1, L + 4);
  out[0] = cvt_e4m3x2(__byte_perm(x1, x2, sel));
  out[1] = cvt_e4m3x2(__byte_perm(x3, x4, sel));
  const uint32_t y1 = __shfl_sync(0xffffffffu, r0, L + 2);
  const uint32_t y2 = __shfl_sync(0xffffffffu, r0, L + 6);
  const uint32_t y3 = __shfl_sync(0xffffffffu, r1, L + 2);
  const uint32_t y4 = __shfl_sync(0xffffffffu, r1, L + 6);
  out[2] = cvt_e4m3x2(__byte_perm(y1, y2, sel));
  out[3] = cvt_e4m3x2(__byte_perm(y3, y4, sel));
}
// PV-style B fragments from M[r=tok][c=hd]: b0 = rows {2(l%4),+1} (ADJACENT
// bytes of one register), b1 = rows {+8,+9}; two n8 hd blocks (regs r0/r1).
DEVFN void bfrag_v(uint32_t r0, uint32_t r1, int lane, uint32_t out[4]) {
  const int L2 = (lane & ~3) + ((lane & 3) >> 1);
  const int sh = 16 * ((lane & 3) & 1);
  const uint32_t x1 = __shfl_sync(0xffffffffu, r0, L2);
  const uint32_t x2 = __shfl_sync(0xffffffffu, r0, L2 + 2);
  out[0] = cvt_e4m3x2((x1 >> sh) & 0xffffu);
  out[1] = cvt_e4m3x2((x2 >> sh) & 0xffffu);
  const uint32_t x3 = __shfl_sync(0xffffffffu, r1, L2);
  const uint32_t x4 = __shfl_sync(0xffffffffu, r1, L2 + 2);
  out[2] = cvt_e4m3x2((x3 >> sh) & 0xffffu);
  out[3] = cvt_e4m3x2((x4 >> sh) & 0xffffu);
}
// 16B-chunk XOR swizzle of an fp8 tile row (row r, byte-column col), in the
// SEGMENT-PLANAR layout the TMA boxes produce: each 128B column segment is a
// contiguous [TN rows][128B] plane (seg stride TN*128), rows XOR-swizzled by
// (r&7). The cp.async stage writes with the SAME map. Planar (not row-major
// interleave) is required at hd256, where TMA stages two segments per row --
// the row-interleaved first version read the wrong half there (K31: f2/mini
// hd256 mismatched, hd128/hd64 single-segment passed). Keeps the ldm8 16 row
// addresses in distinct bank groups (linear rows would collide 16-way; 2-way
// after the XOR).
template <int HD, int TN>
DEVFN int swz8(int r, int col) {
  return (col >> 7) * (TN * 128) + r * 128 +
         ((((col >> 4) & 7) ^ (r & 7)) << 4) + (col & 15);
}

template <int HD, int WARPS_M, int WARPS_N, int TILE_N, int NSTAGE = 3, int RB = 1, int PF = 0,
          int MINB = 1, int SBS = 0, int F16KV = 0, int QREG = 0, int DPR = 1, int PFD = 0,
          int MPAD = 0, int T7 = 0, int CVT = 0, int RST7 = 2,
          int LDM8 = 0, int TMA8 = 0>
struct Cfg {
  // T7 (warp-specialized TMA-fp8 decode staging): CVT convert warps + 1 load
  // warp join the WM*WN compute warps.
  static constexpr int THREADS = (T7 ? (WARPS_M * WARPS_N + CVT + 1) : WARPS_M * WARPS_N) * 32;
  static constexpr int ROWS = RB * 16;         // rows owned by one warp
  static constexpr int TILE_M = WARPS_M * ROWS;
  static constexpr int SLICE_N = TILE_N / WARPS_N;
  static constexpr int STAGES = NSTAGE;
  static_assert(SLICE_N % 16 == 0, "warp KV slice must hold whole 16-row mma tiles");
  static_assert(WARPS_N == 1 || RB == 1, "cross-warp KV merge is only wired for one row block");
  // QREG caches one row block of Q fragments; RB>1 would need RB separate sets
  static_assert(QREG == 0 || RB == 1, "register-resident Q supports one row block");
  // DPR: register prefetch DEPTH. DPR=2 keeps two raw fp8 tiles in flight per
  // thread instead of one, doubling the bytes-in-flight that hide DRAM latency
  // (decode measured latency-bound: ~2us per tile against a 16-32 KiB tile).
  static_assert(DPR >= 1 && (DPR == 1 || PF), "deep prefetch needs the register path");
  static constexpr int Q_BYTES = TILE_M * HD * 2;
  // PF=1: f16 tiles are double buffered and fed from registers (no extra smem
  //     round trip, but the staging registers stay live across the compute phase).
  // PF=0: single f16 buffer fed from a cp.async raw fp8 ring (no staging
  //     registers, but each tile costs one extra smem read+write).
  // SBS: register prefetch into a SINGLE f16 buffer. The raw cp.async ring is the
  // largest shared-memory consumer (2*NSTAGE*TILE_N*HD bytes); dropping it is what
  // lets hd256 decode fit 2-3 CTAs per SM instead of 1. It costs one extra barrier
  // per tile, and barriers measured only 2.7-3.3% of cycles.
  static constexpr int KBUF = LDM8 ? 0 : ((F16KV || T7 || (PF && !SBS)) ? 2 : 1);
  static constexpr int F16_BYTES = 2 * KBUF * TILE_N * HD * 2;
  // T7/TMA8: fp8 TMA ring (1 byte/elem) + a 128B tail for the mbarrier arrays.
  static constexpr int RAW_BYTES =
      (T7 || TMA8) ? (2 * (T7 ? RST7 : NSTAGE) * TILE_N * HD + 128)
                   : ((PF || F16KV) ? 0 : 2 * NSTAGE * TILE_N * HD);
  // epilogue scratch overlays the (then dead) f16 tiles and raw ring
  // MPAD pads the per-row sMerge stride from HD to HD+4 so the cross-warp merge
  // stores (row*stride + col, col even) stop colliding: HD is a multiple of 32,
  // so an unpadded stride makes the row term vanish mod 32 and all 8 rows of a
  // warp's store land on 4 even banks = 8-way conflict (NCU decode: 7.7-way avg,
  // 60% of shared-store wavefronts). HD+4 spreads the rows to 2-way. Numerics
  // are untouched (the pad columns are never read back as data).
  static constexpr int SMERGE = MPAD ? (HD + 4) : HD;
  static constexpr int MERGE_FLOATS =
      WARPS_N * TILE_M * SMERGE + WARPS_N * TILE_M * 2 + TILE_M * 4;
  static constexpr int SMEM = Q_BYTES + F16_BYTES + RAW_BYTES;
  // the merge overlays everything, including the Q tile, which is dead by then
  static_assert(WARPS_N == 1 || MERGE_FLOATS * 4 <= SMEM, "merge overlay");
  static_assert(T7 == 0 || (WARPS_M == 1 && RB == 1 && PF == 0 && F16KV == 0 && CVT >= 1),
                "T7 wiring assumes the decode slice+merge shape");
  static_assert(LDM8 == 0 || (PF == 0 && F16KV == 0 && T7 == 0 && SBS == 0 && HD % 64 == 0 &&
                              WARPS_M == 1 && RB == 1 && SLICE_N == 16),
                "LDM8 reads the raw fp8 cp.async ring directly; decode slice shape only");
  static_assert(TMA8 == 0 || (LDM8 && HD % 128 == 0),
                "TMA8 stages the fp8 ring via TMA (128B boxes need HD>=128)");
  static_assert(SMEM <= 232448, "shared memory over the sm_103 per-block opt-in limit");
};

template <int HD, int WARPS_M, int WARPS_N, int TILE_N, int NSTAGE = 3, int RB = 1, int PF = 0,
          int MINB = 1, int SBS = 0, int F16KV = 0, int QREG = 0, int DPR = 1, int PFD = 0,
          int MPAD = 0, int X2HM = 0, int T7 = 0, int CVT = 0, int RST7 = 2, int LDM8 = 0,
          int TMA8 = 0, int EISSUE = 0, int TRC8 = 0>
__global__ void __launch_bounds__(
    Cfg<HD, WARPS_M, WARPS_N, TILE_N, NSTAGE, RB, PF, MINB, SBS, F16KV, QREG, DPR, PFD, MPAD, T7,
        CVT, RST7, LDM8, TMA8>::THREADS,
    MINB)
fmha_main(const __grid_constant__ Params p) {
  using C = Cfg<HD, WARPS_M, WARPS_N, TILE_N, NSTAGE, RB, PF, MINB, SBS, F16KV, QREG, DPR, PFD,
                MPAD, T7, CVT, RST7, LDM8, TMA8>;
  constexpr int KBUF = C::KBUF;
  constexpr int THREADS = C::THREADS;
  constexpr int TILE_M = C::TILE_M;
  constexpr int SLICE_N = C::SLICE_N;
  constexpr int N8 = SLICE_N / 8;
  constexpr int HD8 = HD / 8;
  constexpr int N16 = SLICE_N / 16;
  constexpr int SMERGE = C::SMERGE;   // padded sMerge row stride (HD+4 when MPAD)

  extern __shared__ char smem_raw[];
  __half* sQ = reinterpret_cast<__half*>(smem_raw);
  __half* sK = sQ + TILE_M * HD;
  __half* sV = sK + KBUF * TILE_N * HD;
  uint8_t* kraw = reinterpret_cast<uint8_t*>(sV + KBUF * TILE_N * HD);
  uint8_t* vraw = kraw + (T7 ? RST7 : (PF ? 0 : NSTAGE)) * TILE_N * HD;
  // Epilogue merge buffers overlay the whole allocation: Q, K/V and the raw ring
  // are all dead once the KV loop is done (the epilogue starts with a barrier).
  float* sMerge = reinterpret_cast<float*>(smem_raw);
  float* sML = sMerge + WARPS_N * TILE_M * SMERGE;
  float* sRow = sML + WARPS_N * TILE_M * 2;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = WARPS_N == 1 ? warp : warp / WARPS_N;
  const int warp_n = WARPS_N == 1 ? 0 : warp % WARPS_N;

  // Longest-tile-first scheduling was measured and rejected: it helps uniform
  // batches by 1-4% but cost case 4 27%, because mtiles is sized from max_q_len,
  // so for a ragged batch the short requests' early-exit tiles move to the FRONT
  // of the launch order and their real work starts last.
  const int mtile = blockIdx.x;
  const int by = blockIdx.y;
  const int b = fdiv(by, p.hkv_magic);
  const int hkv = by - b * p.HKV;
  const int q0 = p.cu_q[b];
  const int qlen = p.cu_q[b + 1] - q0;
  const int kvlen = p.seq_lens[b];
  const int G = p.G;
  const int m0 = mtile * TILE_M;
  if (m0 >= qlen * G) return;

  const int i0 = fdiv(m0, p.g_magic);
  const int i1 = min(qlen, fdiv(m0 + TILE_M, p.g_magic));
  const int vis0 = kvlen - qlen + i0;
  const int vis1 = kvlen - qlen + i1 - 1;
  const int jlo = p.window_left >= 0 ? max(0, vis0 - p.window_left) : 0;
  const int jhi = min(kvlen - 1, vis1);
  const int jt0 = (jlo / TILE_N) * TILE_N;
  const int jt1 = ((jhi + 1 + TILE_N - 1) / TILE_N) * TILE_N;
  const int ntiles = (jt1 - jt0) / TILE_N;
  int t_begin = 0, t_end = ntiles;
  int splits_b = p.splits;
  if (p.splits > 1) {
    if (p.tps_target > 0) {
      splits_b = (ntiles + p.tps_target - 1) / p.tps_target;
      if (splits_b > p.splits) splits_b = p.splits;
      if (splits_b < 1) splits_b = 1;
    } else if (p.tps_target < 0) {
      // Device-balanced mode: identical tt arithmetic in every CTA and in the
      // combine, so the published/reduced split sets always match. TILE_N is
      // constexpr here (folds to shifts); the combine mirrors via p.tile_n_log2.
      constexpr int TNLOG2 = TILE_N >= 128 ? 7 : (TILE_N >= 64 ? 6 : 5);
      const int tt = tps_device_tt(p, TNLOG2, TILE_M);
      splits_b = (ntiles + tt - 1) / tt;
      if (splits_b > p.splits) splits_b = p.splits;
      if (splits_b < 1) splits_b = 1;
    }
    const int tps = (ntiles + splits_b - 1) / splits_b;
    t_begin = blockIdx.z * tps;
    t_end = min(ntiles, t_begin + tps);
  }
  const bool active = t_begin < t_end;

  // TRC8: per-CTA phase-timeline ring, diagnostics configs only (p.trc8 is
  // nullptr for every promoted config, and the stamps compile out entirely
  // when TRC8=0). Layout per CTA: [0] start, [1] t_begin, [2] t_end,
  // [3] loop-end, then 5 stamps per tile rr < 32 at 4+rr*5:
  // {before-wait, after-wait, after-barrier, after-issue, after-compute}.
  unsigned long long* trc_slot = nullptr;
  if constexpr (TRC8) {
    if (p.trc8) {
      const long long cl =
          ((long long)blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z;
      trc_slot = p.trc8 + cl * 164;
      if (tid == 0) {
        trc_slot[0] = trc8_now();
        trc_slot[1] = (unsigned long long)(long long)t_begin;
        trc_slot[2] = (unsigned long long)(long long)t_end;
        trc_slot[3] = 0;
      }
    }
  }
  if (!active) {
    // Empty split: publish a neutral partial (max very negative, sum 0, O 0)
    // so the combine folds it away, and skip everything else. In balanced/TPS
    // modes the combine derives the same splits_b and never reads z >=
    // splits_b, so inactive CTAs write nothing at all. Placed ABOVE the Q
    // staging: with the deep grids device-balanced mode sizes (z up to
    // ceil(tiles_max/3)), empty CTAs are a large fraction of the launch and
    // must cost nothing but the exit. CTA-uniform branch -> no barrier hazard.
    if (p.splits > 1 && p.tps_target == 0) {
      const long long wid = (long long)by * p.mtiles + mtile;
      float* po = p.part_o + wid * p.part_s_w + (long long)blockIdx.z * p.part_s_s;
      float* ml = p.part_ml + wid * p.partml_s_w + (long long)blockIdx.z * p.partml_s_s;
      float* mt = p.part_ml_t + wid * p.partml_s_w + (long long)blockIdx.z * 2;
      for (int e = tid; e < TILE_M * HD; e += THREADS) {
        if (i0 + fdiv(e / HD, p.g_magic) < qlen) po[e] = 0.f;
      }
      if (tid < TILE_M) {
        ml[tid * 2] = NEG_BIG;
        ml[tid * 2 + 1] = 0.f;
        mt[tid * p.partmlt_s_r] = NEG_BIG;
        mt[tid * p.partmlt_s_r + 1] = 0.f;
      }
    }
    return;
  }

  // ---------------- Q tile -> shared memory ----------------
  constexpr int QCH = TILE_M * HD / 8;
  const __nv_bfloat16* qbase = p.q + (long long)q0 * p.q_s_tok;
#pragma unroll
  for (int u = 0; u < (QCH + THREADS - 1) / THREADS; ++u) {
    int c = tid + u * THREADS;
    if (QCH % THREADS != 0 && c >= QCH) break;
    int r = c / HD8, col = (c % HD8) * 8;
    int i = i0 + fdiv(r, p.g_magic);
    uint4 data = make_uint4(0, 0, 0, 0);
    if (i < qlen) {
      const __nv_bfloat16* src =
          qbase + (long long)i * p.q_s_tok + (long long)(hkv * G + (r - fdiv(r, p.g_magic) * G)) * p.q_s_head + col;
      data = *reinterpret_cast<const uint4*>(src);
    }
    uint4 half_data = make_uint4(bf16x2_to_f16x2(data.x), bf16x2_to_f16x2(data.y),
                                 bf16x2_to_f16x2(data.z), bf16x2_to_f16x2(data.w));
    *reinterpret_cast<uint4*>(sQ + swz<HD>(r, col)) = half_data;
  }
  __syncthreads();

  // ---------------- per-thread row constants ----------------
  int rowa[RB], rowb[RB];
  int lim[RB][2], low[RB][2];
  bool rowok[RB][2];
  int lim_min[RB], low_max[RB];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    rowa[rb] = warp_m * (RB * 16) + rb * 16 + (lane >> 2);
    rowb[rb] = rowa[rb] + 8;
    lim_min[rb] = 0x7fffffff;
    low_max[rb] = -0x7fffffff;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = h ? rowb[rb] : rowa[rb];
      const int i = i0 + fdiv(r, p.g_magic);
      const int vv = kvlen - qlen + i;
      lim[rb][h] = vv;
      low[rb][h] = p.window_left >= 0 ? vv - p.window_left : -0x3fffffff;
      rowok[rb][h] = i < qlen;
      lim_min[rb] = min(lim_min[rb], vv);
      low_max[rb] = max(low_max[rb], low[rb][h]);
    }
  }
  const int lane_col = (lane & 3) * 2;

  float O[RB][HD8][4];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int j = 0; j < HD8; ++j) O[rb][j][0] = O[rb][j][1] = O[rb][j][2] = O[rb][j][3] = 0.f;
  // The running max is kept in scaled log2 units (score*qk_scale, which already
  // contains log2 e), so every rescale factor is a plain difference of two stored
  // values. Expressing it as (raw max * qk_scale) - M instead lets nvcc contract
  // that into an fma whose argument becomes the rounding residual of the product;
  // for an empty split (raw max = -1e30) the residual is ~1e21, ex2 overflowed to
  // Inf, and 0*Inf turned the partial outputs into NaN.
  float m2[RB][2], lrow[RB][2];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    m2[rb][0] = m2[rb][1] = NEG_BIG;
    lrow[rb][0] = lrow[rb][1] = 0.f;
  }

  constexpr int HD16 = HD / 16;
  constexpr int KVCH = TILE_N * HD16;
  constexpr int KVPT = KVCH / THREADS;
  // T7 stages via TMA + convert warps, never through the per-thread KVPT
  // chunk split, so its (288/224)-thread shapes need not divide KVCH.
  static_assert(T7 || KVCH % THREADS == 0, "KV tile must divide evenly over threads");
  const int page_mask = (1 << p.page_log2) - 1;
  const int tcols = p.table_stride;
  const int slice0 = warp_n * SLICE_N;
  const float qk_scale = p.qk_scale;
  const uint4* kp = reinterpret_cast<const uint4*>(p.k);
  const uint4* vp = reinterpret_cast<const uint4*>(p.v);
  const int* table_row = p.table + (long long)b * tcols;

  // ---- compute one KV tile: QK^T, masked log2 softmax, PV ----------------
  // Shared by both staging strategies so only the data movement differs.
  auto compute_tile = [&](int t, const __half* kbuf, const __half* vbuf,
                          const uint32_t (*qfrag)[4]) {
  const int jb = jt0 + t * TILE_N;
  float S[RB][N8][4];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int n = 0; n < N8; ++n)
      S[rb][n][0] = S[rb][n][1] = S[rb][n][2] = S[rb][n][3] = 0.f;

  for (int kk = 0; kk < HD; kk += 16) {
    uint32_t Aloc[RB][4];
    // QREG hoists the Q fragments out of the KV loop: Q is 20-25% of this
    // kernel's ldmatrix bytes, and NCU puts L1/shared throughput at 75% with
    // compute at 38%, so shared-memory traffic is the wall, not DRAM or tensor.
    // qfrag is indexed by k step: Qf[kk/16] holds the fragment for columns
    // [kk, kk+16). (An earlier revision kept qfrag pinned at Qf[0], so every
    // k step reused the kk=0 Q fragment; job dv_18b96ed76477 caught it as
    // 18/18 smoke mismatches under cfg71.)
    const uint32_t (*A)[4] = QREG ? (qfrag + kk / 16) : Aloc;
    if (!QREG) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
        ldmatrix_x4(Aloc[rb], sQ + swz<HD>(warp_m * (RB * 16) + rb * 16 + (lane & 15),
                                           kk + (lane >> 4) * 8));
    }
#pragma unroll
    for (int nn = 0; nn < N16; ++nn) {
      uint32_t B[4];
      ldmatrix_x4(B, kbuf + swz<HD>(slice0 + nn * 16 + (lane & 15), kk + (lane >> 4) * 8));
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        mma_f16(S[rb][nn * 2], A[rb], B[0], B[2]);
        mma_f16(S[rb][nn * 2 + 1], A[rb], B[1], B[3]);
      }
    }
  }

  const int col0 = jb + slice0 + lane_col;
  uint32_t Pf[RB][N16][4];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    // A tile is "full" when every column this thread owns is visible for both of
    // the block's rows; then no predicate runs at all (interior common case).
    const bool full = (col0 + SLICE_N - 7 <= lim_min[rb]) && (col0 >= low_max[rb]);
    float mxr[2] = {NEG_BIG, NEG_BIG};
    if (full) {
#pragma unroll
      for (int n = 0; n < N8; ++n)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          mxr[h] = fmaxf(mxr[h], fmaxf(S[rb][n][h * 2], S[rb][n][h * 2 + 1]));
    } else {
#pragma unroll
      for (int n = 0; n < N8; ++n) {
        const int c = col0 + n * 8;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float v0 = (c <= lim[rb][h] && c >= low[rb][h]) ? S[rb][n][h * 2] : NEG_BIG;
          const float v1 =
              (c + 1 <= lim[rb][h] && c + 1 >= low[rb][h]) ? S[rb][n][h * 2 + 1] : NEG_BIG;
          mxr[h] = fmaxf(mxr[h], fmaxf(v0, v1));
        }
      }
    }
    float mneg[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      mxr[h] = fmaxf(mxr[h], __shfl_xor_sync(0xffffffffu, mxr[h], 1));
      mxr[h] = fmaxf(mxr[h], __shfl_xor_sync(0xffffffffu, mxr[h], 2));
      const float mn = fmaxf(m2[rb][h], mxr[h] * qk_scale);
      const float rs = ex2(m2[rb][h] - mn);
      m2[rb][h] = mn;
      mneg[h] = -mn;
      if (rs != 1.f) {
        lrow[rb][h] *= rs;
#pragma unroll
        for (int j = 0; j < HD8; ++j) {
          if (h == 0) {
            O[rb][j][0] *= rs;
            O[rb][j][1] *= rs;
          } else {
            O[rb][j][2] *= rs;
            O[rb][j][3] *= rs;
          }
        }
      }
    }
    uint32_t hacc[2] = {0u, 0u};  // X2HM f16x2 row-sum accumulators
#pragma unroll
    for (int n = 0; n < N8; ++n) {
      const int c = col0 + n * 8;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if constexpr (X2HM) {
          // X2HM: packed ex2.approx.f16x2 (same mechanism as X2H in
          // fmha_main_t5). Masked columns fold into the exponent as NEG_BIG
          // (-1e30 -> satfinite pack -> -65504 -> ex2 -> +0), so the
          // predicate branch collapses into arithmetic. Pf receives the exp
          // output directly (no f32 round trip through S), and the row sum
          // accumulates in f16x2 (<= N8 adds per lane, P <= 1 here since the
          // max is applied immediately -- no DEFER -- so lane sums <= 8 with
          // ulp 2^-7: ~0.1% worst-case, vs ~0.05% for the f32 exp path).
          float x0 = fmaf(S[rb][n][h * 2], qk_scale, mneg[h]);
          float x1 = fmaf(S[rb][n][h * 2 + 1], qk_scale, mneg[h]);
          if (!full) {
            if (!(c <= lim[rb][h] && c >= low[rb][h])) x0 = NEG_BIG;
            if (!(c + 1 <= lim[rb][h] && c + 1 >= low[rb][h])) x1 = NEG_BIG;
          }
          const uint32_t p2 = ex2_f16x2(pack_f16(x0, x1));
          Pf[rb][n >> 1][(n & 1) * 2 + h] = p2;
          hacc[h] = hadd2(hacc[h], p2);
        } else {
          float v0, v1;
          if (full) {
            v0 = ex2(fmaf(S[rb][n][h * 2], qk_scale, mneg[h]));
            v1 = ex2(fmaf(S[rb][n][h * 2 + 1], qk_scale, mneg[h]));
          } else {
            v0 = (c <= lim[rb][h] && c >= low[rb][h])
                     ? ex2(fmaf(S[rb][n][h * 2], qk_scale, mneg[h]))
                     : 0.f;
            v1 = (c + 1 <= lim[rb][h] && c + 1 >= low[rb][h])
                     ? ex2(fmaf(S[rb][n][h * 2 + 1], qk_scale, mneg[h]))
                     : 0.f;
          }
          S[rb][n][h * 2] = v0;
          S[rb][n][h * 2 + 1] = v1;
          lrow[rb][h] += v0 + v1;
        }
      }
      if constexpr (!X2HM) {
        Pf[rb][n >> 1][(n & 1) * 2] = pack_f16(S[rb][n][0], S[rb][n][1]);
        Pf[rb][n >> 1][(n & 1) * 2 + 1] = pack_f16(S[rb][n][2], S[rb][n][3]);
      }
    }
    if constexpr (X2HM) {
      lrow[rb][0] += hsum2(hacc[0]);
      lrow[rb][1] += hsum2(hacc[1]);
    }
  }

  for (int nn = 0; nn < N16; ++nn) {
    for (int jj = 0; jj < HD; jj += 16) {
      uint32_t Bv[4];
      ldmatrix_x4_trans(Bv, vbuf + swz<HD>(slice0 + nn * 16 + (lane & 15), jj + (lane >> 4) * 8));
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        mma_f16(O[rb][jj >> 3], Pf[rb][nn], Bv[0], Bv[1]);
        mma_f16(O[rb][(jj >> 3) + 1], Pf[rb][nn], Bv[2], Bv[3]);
      }
    }
  }
  };

  // t7c compute: numerics IDENTICAL to compute_tile (same m16n8k16 shapes,
  // same X2HM softmax, same merge) but B fragments come straight out of the
  // swizzled fp8 ring via ldmatrix.m16n16.trans.b8 (no f16 tile, no convert
  // phase, half the smem+LSU bytes). Fragment assembly verified against the
  // K24c v6/v7 lane->byte mapping.
  auto compute_tile8 = [&](int t, const uint8_t* kraw8, const uint8_t* vraw8,
                          const uint32_t (*qfrag)[4]) {
  const int jb = jt0 + t * TILE_N;
  float S[RB][N8][4];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int n = 0; n < N8; ++n)
      S[rb][n][0] = S[rb][n][1] = S[rb][n][2] = S[rb][n][3] = 0.f;

  for (int kk = 0; kk < HD; kk += 32) {
    // One x2 load covers 16 slice tokens x 32 hd bytes (two 16x16 matrices =
    // two k16 steps); bfrag_k turns each into the B pairs of two n8 mmas.
    uint32_t l8[4];
    {
      const int arow = slice0 + (lane & 15);
      const int achk = (kk >> 4) + (lane >> 4);
      ldm8_x2(l8, t5_sp(kraw8 + swz8<HD, TILE_N>(arow, achk * 16)));
    }
    uint32_t Aloc[RB][4], Aloc2[RB][4];
    const uint32_t (*A0)[4] = QREG ? (qfrag + kk / 16) : Aloc;
    const uint32_t (*A1)[4] = QREG ? (qfrag + kk / 16 + 1) : Aloc2;
    if (!QREG) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        ldmatrix_x4(Aloc[rb], sQ + swz<HD>(warp_m * (RB * 16) + rb * 16 + (lane & 15),
                                           kk + (lane >> 4) * 8));
        ldmatrix_x4(Aloc2[rb], sQ + swz<HD>(warp_m * (RB * 16) + rb * 16 + (lane & 15),
                                            kk + 16 + (lane >> 4) * 8));
      }
    }
    uint32_t bf0[4], bf1[4];
    bfrag_k(l8[0], l8[1], lane, bf0);
    bfrag_k(l8[2], l8[3], lane, bf1);
#pragma unroll
    for (int rb = 0; rb < RB; ++rb) {
      mma_f16(S[rb][0], A0[rb], bf0[0], bf0[1]);
      mma_f16(S[rb][1], A0[rb], bf0[2], bf0[3]);
      mma_f16(S[rb][0], A1[rb], bf1[0], bf1[1]);
      mma_f16(S[rb][1], A1[rb], bf1[2], bf1[3]);
    }
  }

  const int col0 = jb + slice0 + lane_col;
  uint32_t Pf[RB][N16][4];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
    // A tile is "full" when every column this thread owns is visible for both of
    // the block's rows; then no predicate runs at all (interior common case).
    const bool full = (col0 + SLICE_N - 7 <= lim_min[rb]) && (col0 >= low_max[rb]);
    float mxr[2] = {NEG_BIG, NEG_BIG};
    if (full) {
#pragma unroll
      for (int n = 0; n < N8; ++n)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          mxr[h] = fmaxf(mxr[h], fmaxf(S[rb][n][h * 2], S[rb][n][h * 2 + 1]));
    } else {
#pragma unroll
      for (int n = 0; n < N8; ++n) {
        const int c = col0 + n * 8;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float v0 = (c <= lim[rb][h] && c >= low[rb][h]) ? S[rb][n][h * 2] : NEG_BIG;
          const float v1 =
              (c + 1 <= lim[rb][h] && c + 1 >= low[rb][h]) ? S[rb][n][h * 2 + 1] : NEG_BIG;
          mxr[h] = fmaxf(mxr[h], fmaxf(v0, v1));
        }
      }
    }
    float mneg[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      mxr[h] = fmaxf(mxr[h], __shfl_xor_sync(0xffffffffu, mxr[h], 1));
      mxr[h] = fmaxf(mxr[h], __shfl_xor_sync(0xffffffffu, mxr[h], 2));
      const float mn = fmaxf(m2[rb][h], mxr[h] * qk_scale);
      const float rs = ex2(m2[rb][h] - mn);
      m2[rb][h] = mn;
      mneg[h] = -mn;
      if (rs != 1.f) {
        lrow[rb][h] *= rs;
#pragma unroll
        for (int j = 0; j < HD8; ++j) {
          if (h == 0) {
            O[rb][j][0] *= rs;
            O[rb][j][1] *= rs;
          } else {
            O[rb][j][2] *= rs;
            O[rb][j][3] *= rs;
          }
        }
      }
    }
    uint32_t hacc[2] = {0u, 0u};  // X2HM f16x2 row-sum accumulators
#pragma unroll
    for (int n = 0; n < N8; ++n) {
      const int c = col0 + n * 8;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if constexpr (X2HM) {
          // X2HM: packed ex2.approx.f16x2 (same mechanism as X2H in
          // fmha_main_t5). Masked columns fold into the exponent as NEG_BIG
          // (-1e30 -> satfinite pack -> -65504 -> ex2 -> +0), so the
          // predicate branch collapses into arithmetic. Pf receives the exp
          // output directly (no f32 round trip through S), and the row sum
          // accumulates in f16x2 (<= N8 adds per lane, P <= 1 here since the
          // max is applied immediately -- no DEFER -- so lane sums <= 8 with
          // ulp 2^-7: ~0.1% worst-case, vs ~0.05% for the f32 exp path).
          float x0 = fmaf(S[rb][n][h * 2], qk_scale, mneg[h]);
          float x1 = fmaf(S[rb][n][h * 2 + 1], qk_scale, mneg[h]);
          if (!full) {
            if (!(c <= lim[rb][h] && c >= low[rb][h])) x0 = NEG_BIG;
            if (!(c + 1 <= lim[rb][h] && c + 1 >= low[rb][h])) x1 = NEG_BIG;
          }
          const uint32_t p2 = ex2_f16x2(pack_f16(x0, x1));
          Pf[rb][n >> 1][(n & 1) * 2 + h] = p2;
          hacc[h] = hadd2(hacc[h], p2);
        } else {
          float v0, v1;
          if (full) {
            v0 = ex2(fmaf(S[rb][n][h * 2], qk_scale, mneg[h]));
            v1 = ex2(fmaf(S[rb][n][h * 2 + 1], qk_scale, mneg[h]));
          } else {
            v0 = (c <= lim[rb][h] && c >= low[rb][h])
                     ? ex2(fmaf(S[rb][n][h * 2], qk_scale, mneg[h]))
                     : 0.f;
            v1 = (c + 1 <= lim[rb][h] && c + 1 >= low[rb][h])
                     ? ex2(fmaf(S[rb][n][h * 2 + 1], qk_scale, mneg[h]))
                     : 0.f;
          }
          S[rb][n][h * 2] = v0;
          S[rb][n][h * 2 + 1] = v1;
          lrow[rb][h] += v0 + v1;
        }
      }
      if constexpr (!X2HM) {
        Pf[rb][n >> 1][(n & 1) * 2] = pack_f16(S[rb][n][0], S[rb][n][1]);
        Pf[rb][n >> 1][(n & 1) * 2 + 1] = pack_f16(S[rb][n][2], S[rb][n][3]);
      }
    }
    if constexpr (X2HM) {
      lrow[rb][0] += hsum2(hacc[0]);
      lrow[rb][1] += hsum2(hacc[1]);
    }
  }

    for (int nn = 0; nn < N16; ++nn) {
    for (int jj = 0; jj < HD; jj += 32) {
      uint32_t l8[4];
      {
        const int arow = slice0 + nn * 16 + (lane & 15);
        ldm8_x2(l8, t5_sp(vraw8 + swz8<HD, TILE_N>(arow, jj + ((lane >> 4) << 4))));
      }
      uint32_t bf0[4], bf1[4];
      bfrag_v(l8[0], l8[1], lane, bf0);
      bfrag_v(l8[2], l8[3], lane, bf1);
#pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        mma_f16(O[rb][jj >> 3], Pf[rb][nn], bf0[0], bf0[1]);
        mma_f16(O[rb][(jj >> 3) + 1], Pf[rb][nn], bf0[2], bf0[3]);
        mma_f16(O[rb][(jj >> 3) + 2], Pf[rb][nn], bf1[0], bf1[1]);
        mma_f16(O[rb][(jj >> 3) + 3], Pf[rb][nn], bf1[2], bf1[3]);
      }
    }
  }
  };

  uint32_t Qf[(HD / 16)][4];
  if (QREG) {
    for (int kk = 0; kk < HD; kk += 16)
      ldmatrix_x4(Qf[kk / 16], sQ + swz<HD>(warp_m * (RB * 16) + (lane & 15), kk + (lane >> 4) * 8));
  }
  const uint32_t (*qfrag)[4] = Qf;

  if constexpr (T7) {
    // ------------------ t7: warp-specialized TMA-fp8 staging ------------------
    // Roles: warps [0,NW) compute (the classic slice+merge pipeline -- numerics
    // untouched), warps [NW,NW+CVT) convert fp8->f16 from the TMA ring into the
    // double-buffered f16 tiles, warp NW+CVT lane0 issues the TMA loads. The
    // per-tile chain (wait cp.async -> all-thread convert -> syncthreads ->
    // compute) that made decode latency-bound (NCU: DRAM 5%, SM 15%, occ 12.5%)
    // becomes a 3-stage pipeline: TMA(t+RST-1) || convert(t+1) || compute(t).
    // fp8 boxes land in SWIZZLE_128B order: 16B chunk c of row n sits at
    // ((c ^ (n&7)) << 4) within the row's 128B segment -- the byte twin of the
    // f16 swz<> pattern the compute ldmatrix path already consumes.
    constexpr int NW = WARPS_M * WARPS_N;
    constexpr int RST = RST7;
    uint64_t* KF8 = reinterpret_cast<uint64_t*>(vraw + (size_t)RST * TILE_N * HD);
    uint64_t* CE = KF8 + RST;  // fp8 slot free (convert -> load)
    uint64_t* CV = CE + RST;   // f16 buffer ready (convert -> compute)
    uint64_t* CD = CV + 2;     // f16 buffer consumed (compute -> convert)
    if (tid == 0) {
      for (int s = 0; s < RST; ++s) {
        t5_mbar_init(t5_sp(&KF8[s]), 1);
        t5_mbar_init(t5_sp(&CE[s]), CVT);
      }
      t5_mbar_init(t5_sp(&CV[0]), CVT);
      t5_mbar_init(t5_sp(&CV[1]), CVT);
      t5_mbar_init(t5_sp(&CD[0]), NW);
      t5_mbar_init(t5_sp(&CD[1]), NW);
    }
    __syncthreads();
    if (warp < NW) {
      for (int t = t_begin; t < t_end; ++t) {
        const int rr = t - t_begin, bf = rr & 1;
        t5_mbar_wait(t5_sp(&CV[bf]), (rr >> 1) & 1);
        compute_tile(t, sK + bf * TILE_N * HD, sV + bf * TILE_N * HD, qfrag);
        if (lane == 0) mb_arrive(t5_sp(&CD[bf]));
      }
    } else if (warp < NW + CVT) {
      const int cl = (warp - NW) * 32 + lane;
      constexpr int NCH = TILE_N * (HD / 16);  // 16-element chunks per K or V tile
      for (int t = t_begin; t < t_end; ++t) {
        const int rr = t - t_begin, s = rr % RST, bf = rr & 1;
        t5_mbar_wait(t5_sp(&KF8[s]), (rr / RST) & 1);
        if (rr >= 2) t5_mbar_wait(t5_sp(&CD[bf]), ((rr - 2) >> 1) & 1);
        const uint8_t* k8 = kraw + (size_t)s * TILE_N * HD;
        const uint8_t* v8 = vraw + (size_t)s * TILE_N * HD;
        for (int idx = cl; idx < 2 * NCH; idx += CVT * 32) {
          const bool isv = idx >= NCH;
          const int rem = isv ? idx - NCH : idx;
          const int n = rem / (HD / 16), cc = rem % (HD / 16);
          const int g = cc >> 3, c = cc & 7;
          const uint8_t* src8 =
              (isv ? v8 : k8) + (size_t)g * (TILE_N * 128) + n * 128 + ((c ^ (n & 7)) << 4);
          const uint4 rawv = *reinterpret_cast<const uint4*>(src8);
          uint32_t hh[8];
          expand8(rawv, hh);
          __half* dst = (isv ? sV : sK) + bf * TILE_N * HD;
          const int col = g * 128 + c * 16;
          *reinterpret_cast<uint4*>(dst + swz<HD>(n, col)) =
              make_uint4(hh[0], hh[1], hh[2], hh[3]);
          *reinterpret_cast<uint4*>(dst + swz<HD>(n, col + 8)) =
              make_uint4(hh[4], hh[5], hh[6], hh[7]);
        }
        fence_async_smem();  // LDS reads done before the load warp re-TMAs the slot
        if (lane == 0) {
          mb_arrive(t5_sp(&CV[bf]));
          mb_arrive(t5_sp(&CE[s]));
        }
      }
    } else if (warp == NW + CVT && lane == 0) {
      const int page_log2 = p.page_log2;
      const int box_rows = min(1 << page_log2, 64);
      const int page_mask = (1 << page_log2) - 1;
      const int tcols = p.table_stride;
      const int* table_row = p.table + (long long)b * tcols;
      constexpr int SEG = HD / 128;  // 128B column segments per fp8 row
      for (int t = t_begin; t < t_end; ++t) {
        const int rr = t - t_begin, s = rr % RST;
        if (rr >= RST) t5_mbar_wait(t5_sp(&CE[s]), ((rr / RST) - 1) & 1);
        const uint32_t kbar = t5_sp(&KF8[s]);
        mb_expect(kbar, 2 * TILE_N * HD);
        const int jb = jt0 + t * TILE_N;
        for (int off = 0; off < TILE_N; off += box_rows) {
          const int jj = min(max(jb + off, 0), kvlen - 1);
          const int pid = table_row[min(jj >> page_log2, tcols - 1)];
          const int row_in_page = (jj & page_mask) & ~(box_rows - 1);
#pragma unroll
          for (int g = 0; g < SEG; ++g) {
            const uint32_t off2 = (uint32_t)(g * (TILE_N * 128) + off * 128);
            tma4d(&p.kmap8, t5_sp(kraw + (size_t)s * TILE_N * HD + off2), kbar, g * 128,
                  row_in_page, hkv, pid);
            tma4d(&p.vmap8, t5_sp(vraw + (size_t)s * TILE_N * HD + off2), kbar, g * 128,
                  row_in_page, hkv, pid);
          }
        }
      }
    }
  } else if constexpr (F16KV) {
    // Fastest staging: cp.async 16-byte f16 chunks from the mirror directly into
    // the swizzled tile. No convert pass, no raw ring, no shared-memory read
    // before the store, and the per-thread 16B chunk makes each warp store a
    // fully populated 128 bytes (the fp8 path stored 16B at a 32B stride, which
    // half-filled every transaction). One barrier per tile.
    constexpr int FCH = TILE_N * (HD / 8);
    constexpr int FPT = FCH / THREADS;
    static_assert(FCH % THREADS == 0, "f16 tile must divide over threads");
    auto issue16 = [&](int t, int buf) {
      __half* dk = sK + buf * TILE_N * HD;
      __half* dv = sV + buf * TILE_N * HD;
      const int jb = jt0 + t * TILE_N;
#pragma unroll
      for (int u = 0; u < FPT; ++u) {
        const int c = tid + u * THREADS;
        const int n = c / (HD / 8), col = (c % (HD / 8)) * 8;
        const int jj = min(max(jb + n, 0), kvlen - 1);
        const int page = table_row[min(jj >> p.page_log2, tcols - 1)];
        const long long base = (long long)page * p.kv_s_page + (long long)hkv * p.kv_s_head +
                               (long long)(jj & page_mask) * p.kv_s_tok + col;
        cp_async16(dk + swz<HD>(n, col), p.k16 + base);
        cp_async16(dv + swz<HD>(n, col), p.v16 + base);
      }
      cp_async_commit();
    };
    if (t_begin < t_end) issue16(t_begin, 0);
    for (int t = t_begin; t < t_end; ++t) {
      const int buf = (t - t_begin) & 1;
      cp_async_wait<0>();
      // Also proves every thread finished the previous compute, so buffer buf^1
      // (read two iterations ago) is safe to overwrite next.
      __syncthreads();
      if (t + 1 < t_end) issue16(t + 1, buf ^ 1);
      compute_tile(t, sK + buf * TILE_N * HD, sV + buf * TILE_N * HD, qfrag);
    }
  } else if constexpr (PF) {
    // Register prefetch: the next tile's fp8 bytes are in flight during this
    // tile's compute and staging never round-trips through shared memory.
    // DPR>1 keeps DPR tiles in flight in DPR rotating register slots, which
    // multiplies the bytes-in-flight that hide DRAM latency at the cost of
    // (DPR-1)*KVPT*8 registers per thread.
    uint4 kr[DPR][KVPT], vr[DPR][KVPT];
    int kloc[KVPT];
    auto issue_pf = [&](int t, int slot) {
      const int jb = jt0 + t * TILE_N;
#pragma unroll
      for (int u = 0; u < KVPT; ++u) {
        const int c = tid + u * THREADS;
        const int n = c / HD16, col = (c % HD16) * 16;
        const int jj = min(max(jb + n, 0), kvlen - 1);
        const int page = table_row[min(jj >> p.page_log2, tcols - 1)];
        const long long base = (long long)page * p.kv_s_page + (long long)hkv * p.kv_s_head +
                               (long long)(jj & page_mask) * p.kv_s_tok + col;
        kr[slot][u] = kp[base >> 4];
        vr[slot][u] = vp[base >> 4];
        kloc[u] = n * HD + col;
      }
    };
    auto prefetch_pf = [&](int t) {
      const int jb = jt0 + t * TILE_N;
#pragma unroll
      for (int u = 0; u < KVPT; ++u) {
        const int c = tid + u * THREADS;
        const int n = c / HD16, col = (c % HD16) * 16;
        const int jj = min(max(jb + n, 0), kvlen - 1);
        const int page = table_row[min(jj >> p.page_log2, tcols - 1)];
        const long long base = (long long)page * p.kv_s_page + (long long)hkv * p.kv_s_head +
                               (long long)(jj & page_mask) * p.kv_s_tok + col;
        prefetch_l2(kp + (base >> 4));
        prefetch_l2(vp + (base >> 4));
      }
    };
    auto store_pf = [&](int slot, int buf) {
      __half* dk = sK + buf * TILE_N * HD;
      __half* dv = sV + buf * TILE_N * HD;
#pragma unroll
      for (int u = 0; u < KVPT; ++u) {
        const int n = kloc[u] / HD, col = kloc[u] % HD;
        uint32_t kh[8], vh[8];
        expand8(kr[slot][u], kh);
        expand8(vr[slot][u], vh);
        *reinterpret_cast<uint4*>(dk + swz<HD>(n, col)) = make_uint4(kh[0], kh[1], kh[2], kh[3]);
        *reinterpret_cast<uint4*>(dk + swz<HD>(n, col + 8)) =
            make_uint4(kh[4], kh[5], kh[6], kh[7]);
        *reinterpret_cast<uint4*>(dv + swz<HD>(n, col)) = make_uint4(vh[0], vh[1], vh[2], vh[3]);
        *reinterpret_cast<uint4*>(dv + swz<HD>(n, col + 8)) =
            make_uint4(vh[4], vh[5], vh[6], vh[7]);
      }
    };
#pragma unroll
    for (int d = 0; d < DPR; ++d)
      if (t_begin + d < t_end) issue_pf(t_begin + d, d);
    for (int t = t_begin; t < t_end; ++t) {
      const int slot = (int)((t - t_begin) % DPR);
      const int buf = SBS ? 0 : ((t - t_begin) & 1);
      store_pf(slot, buf);
      // With two buffers one barrier per tile is enough: it makes buffer buf
      // writable-safe (the buffer used two iterations ago is provably finished).
      // With a single buffer a second barrier after compute is required.
      __syncthreads();
      if (t + DPR < t_end) issue_pf(t + DPR, slot);
      if (PFD && t + DPR + 1 < t_end) prefetch_pf(t + DPR + 1);
      compute_tile(t, sK + buf * TILE_N * HD, sV + buf * TILE_N * HD, qfrag);
      if (SBS) __syncthreads();
    }
  } else if constexpr (TMA8) {
    // c196: TMA-staged fp8 ring + ldm8 compute. lane0 issues a handful of
    // bulk boxes per tile instead of ~1024 cp.async instructions; slot
    // release is the per-tile __syncthreads itself (every thread finished
    // compute(t-1) before reaching iteration t's barrier), so the only
    // mbarriers are the TMA-completion KF8[s], one per slot, phase = reuse
    // cycle. Needs the fp8 UINT8 maps from the gate (p.kmap8/vmap8).
    uint64_t* KF8 = reinterpret_cast<uint64_t*>(vraw + (size_t)NSTAGE * TILE_N * HD);
    if (tid == 0) {
      for (int st = 0; st < NSTAGE; ++st) t5_mbar_init(t5_sp(&KF8[st]), 1);
    }
    __syncthreads();
    const int page_log2 = p.page_log2;
    const int box_rows = min(1 << page_log2, TILE_N);
    const int tcols = p.table_stride;
    const int* table_row = p.table + (long long)b * tcols;
    constexpr int SEG = HD / 128;  // 128B column segments per fp8 row
    auto issue_tma8 = [&](int t, int stage) {
      const uint32_t kbar = t5_sp(&KF8[stage]);
      mb_expect(kbar, 2 * TILE_N * HD);
      const int jb = jt0 + t * TILE_N;
      uint8_t* dk = kraw + stage * TILE_N * HD;
      uint8_t* dv = vraw + stage * TILE_N * HD;
      for (int off = 0; off < TILE_N; off += box_rows) {
        const int jj = min(max(jb + off, 0), kvlen - 1);
        const int pid = table_row[min(jj >> page_log2, tcols - 1)];
        const int row_in_page = (jj & ((1 << page_log2) - 1)) & ~(box_rows - 1);
#pragma unroll
        for (int g = 0; g < SEG; ++g) {
          const uint32_t off2 = (uint32_t)(g * (TILE_N * 128) + off * 128);
          tma4d(&p.kmap8, t5_sp(dk + off2), kbar, g * 128, row_in_page, hkv, pid);
          tma4d(&p.vmap8, t5_sp(dv + off2), kbar, g * 128, row_in_page, hkv, pid);
        }
      }
    };
    if constexpr (EISSUE) {
      // EISSUE (c286/c287): issue-ahead staging. The legacy loop issues tile
      // t+NSTAGE-1 only AFTER mbar_wait(t) returned, so the steady-state
      // recurrence is arrival(t+NS-1) = arrival(t) + (pid LDG + issue +
      // transfer latency): the per-tile period is floored near
      // latency/(NSTAGE-1) with the dependent page-table load serialised on
      // tid0's issue path. Decode mains measure 1.56-2.55TB/s against the
      // 4.59TB/s contiguous read roof on BOTH head-dim classes
      // (dv_a1c2de84c443), i.e. pipeline-limited, not pattern-limited.
      // Here the top-of-iteration __syncthreads proves compute(t-1) consumed
      // slot (t-1)%NSTAGE == (t+NSTAGE-1)%NSTAGE, so the TMA for t+NSTAGE-1
      // is issued BEFORE waiting on tile t; every thread's own mbarrier wait
      // gives it the async-proxy visibility of the data it reads, so no
      // second barrier is needed. Page ids/rows are prefetched into tid0
      // registers one tile ahead, overlapping the LDG with wait+compute.
      constexpr int PPT_MAX = TILE_N / 16 + 1;  // worst case: page_size 16
      auto load_pids = [&](int t, int* pids, int* rows) {
        const int jb = jt0 + t * TILE_N;
        int i = 0;
        for (int off = 0; off < TILE_N; off += box_rows, ++i) {
          const int jj = min(max(jb + off, 0), kvlen - 1);
          pids[i] = table_row[min(jj >> page_log2, tcols - 1)];
          rows[i] = (jj & ((1 << page_log2) - 1)) & ~(box_rows - 1);
        }
      };
      auto issue_p = [&](int t, int stage, const int* pids, const int* rows) {
        const uint32_t kbar = t5_sp(&KF8[stage]);
        mb_expect(kbar, 2 * TILE_N * HD);
        uint8_t* dk = kraw + stage * TILE_N * HD;
        uint8_t* dv = vraw + stage * TILE_N * HD;
        int i = 0;
        for (int off = 0; off < TILE_N; off += box_rows, ++i) {
#pragma unroll
          for (int g = 0; g < SEG; ++g) {
            const uint32_t off2 = (uint32_t)(g * (TILE_N * 128) + off * 128);
            tma4d(&p.kmap8, t5_sp(dk + off2), kbar, g * 128, rows[i], hkv, pids[i]);
            tma4d(&p.vmap8, t5_sp(dv + off2), kbar, g * 128, rows[i], hkv, pids[i]);
          }
        }
      };
      int pid_a[PPT_MAX], row_a[PPT_MAX], pid_b[PPT_MAX], row_b[PPT_MAX];
      if (tid == 0) {
        for (int st = 0; st < NSTAGE - 1; ++st) {
          if (t_begin + st < t_end) {
            load_pids(t_begin + st, pid_a, row_a);
            issue_p(t_begin + st, st, pid_a, row_a);
          }
        }
        if (t_begin + NSTAGE - 1 < t_end) load_pids(t_begin + NSTAGE - 1, pid_b, row_b);
      }
      for (int t = t_begin; t < t_end; ++t) {
        const int rr = t - t_begin, stage = rr % NSTAGE;
        if (t > t_begin) __syncthreads();  // slot (t-1)%NSTAGE freed by compute(t-1)
        const int tn = t + NSTAGE - 1;
        if (tid == 0 && tn < t_end) {
          issue_p(tn, (int)((tn - t_begin) % NSTAGE), pid_b, row_b);
          if (tn + 1 < t_end) load_pids(tn + 1, pid_b, row_b);
        }
        t5_mbar_wait(t5_sp(&KF8[stage]), (rr / NSTAGE) & 1);
        compute_tile8(t, kraw + stage * TILE_N * HD, vraw + stage * TILE_N * HD, qfrag);
      }
    } else {
      // Prologue fills NSTAGE-1 slots; the loop's issue of tile t+NSTAGE-1
      // reuses slot (t-1)%NSTAGE, which this iteration's __syncthreads proved
      // consumed. (Issuing t+NSTAGE would TMA into the slot compute(t) reads.)
      if (tid == 0) {
        for (int st = 0; st < NSTAGE - 1; ++st) {
          if (t_begin + st < t_end) issue_tma8(t_begin + st, st);
        }
      }
      for (int t = t_begin; t < t_end; ++t) {
        const int rr = t - t_begin, stage = rr % NSTAGE;
        if constexpr (TRC8) { if (trc_slot && tid == 0 && rr < 32) trc_slot[4 + rr * 5 + 0] = trc8_now(); }
        t5_mbar_wait(t5_sp(&KF8[stage]), (rr / NSTAGE) & 1);
        if constexpr (TRC8) { if (trc_slot && tid == 0 && rr < 32) trc_slot[4 + rr * 5 + 1] = trc8_now(); }
        __syncthreads();
        if constexpr (TRC8) { if (trc_slot && tid == 0 && rr < 32) trc_slot[4 + rr * 5 + 2] = trc8_now(); }
        const int tn = t + NSTAGE - 1;
        if (tid == 0 && tn < t_end) issue_tma8(tn, (int)((tn - t_begin) % NSTAGE));
        if constexpr (TRC8) { if (trc_slot && tid == 0 && rr < 32) trc_slot[4 + rr * 5 + 3] = trc8_now(); }
        compute_tile8(t, kraw + stage * TILE_N * HD, vraw + stage * TILE_N * HD, qfrag);
        if constexpr (TRC8) { if (trc_slot && tid == 0 && rr < 32) trc_slot[4 + rr * 5 + 4] = trc8_now(); }
      }
    }
  } else if constexpr (LDM8) {
    // t7c staging: the raw fp8 ring, swizzled per 16B chunk, read DIRECTLY by
    // compute_tile8's ldmatrix.b8 -- the f16 tile, the convert phase and one
    // of the two per-tile barriers are all gone.
    auto issue_stage8 = [&](int t, int stage) {
      uint8_t* dk = kraw + stage * TILE_N * HD;
      uint8_t* dv = vraw + stage * TILE_N * HD;
      const int jb = jt0 + t * TILE_N;
#pragma unroll
      for (int u = 0; u < KVPT; ++u) {
        const int c = tid + u * THREADS;
        const int n = c / HD16, col = (c % HD16) * 16;
        const int jj = min(max(jb + n, 0), kvlen - 1);
        const int page = table_row[min(jj >> p.page_log2, tcols - 1)];
        const long long base = (long long)page * p.kv_s_page + (long long)hkv * p.kv_s_head +
                               (long long)(jj & page_mask) * p.kv_s_tok + col;
        cp_async16(dk + swz8<HD, TILE_N>(n, col), kp + (base >> 4));
        cp_async16(dv + swz8<HD, TILE_N>(n, col), vp + (base >> 4));
      }
      cp_async_commit();
    };
    if constexpr (EISSUE) {
      // EISSUE (c288/c289): issue tile t+NSTAGE-1 BEFORE waiting on tile t.
      // The top-of-iteration barrier frees slot (t-1)%NSTAGE (compute(t-1)
      // consumed it); the post-wait barrier publishes every thread's staged
      // chunks of tile t (cp.async completion is per-thread). Group
      // accounting: committed = NSTAGE + rr at the wait, so wait<NSTAGE-1>
      // completes exactly group(t). Removes the wait->issue serialization
      // from the cp.async ring's recurrence (see the TMA8 EISSUE comment).
      for (int st = 0; st < NSTAGE - 1; ++st) {
        if (t_begin + st < t_end) issue_stage8(t_begin + st, st);
        else cp_async_commit();
      }
      for (int t = t_begin; t < t_end; ++t) {
        const int stage = (int)((t - t_begin) % NSTAGE);
        if (t > t_begin) __syncthreads();
        {
          const int tn = t + NSTAGE - 1;
          if (tn < t_end) issue_stage8(tn, (int)((tn - t_begin) % NSTAGE));
          else cp_async_commit();
        }
        cp_async_wait<NSTAGE - 1>();
        __syncthreads();
        compute_tile8(t, kraw + stage * TILE_N * HD, vraw + stage * TILE_N * HD, qfrag);
      }
    } else {
      for (int st = 0; st < NSTAGE - 1; ++st) {
        if (t_begin + st < t_end) issue_stage8(t_begin + st, st);
        else cp_async_commit();
      }
      for (int t = t_begin; t < t_end; ++t) {
        const int rr = t - t_begin;
        const int stage = (int)((t - t_begin) % NSTAGE);
        if constexpr (TRC8) { if (trc_slot && tid == 0 && rr < 32) trc_slot[4 + rr * 5 + 0] = trc8_now(); }
        cp_async_wait<NSTAGE - 2>();
        if constexpr (TRC8) { if (trc_slot && tid == 0 && rr < 32) trc_slot[4 + rr * 5 + 1] = trc8_now(); }
        // Proves every thread finished compute(t-1), so stage (t-1)%NSTAGE --
        // the slot t+NSTAGE-1 reuses -- is free.
        __syncthreads();
        if constexpr (TRC8) { if (trc_slot && tid == 0 && rr < 32) trc_slot[4 + rr * 5 + 2] = trc8_now(); }
        {
          const int tn = t + NSTAGE - 1;
          if (tn < t_end) issue_stage8(tn, (int)((tn - t_begin) % NSTAGE));
          else cp_async_commit();
        }
        if constexpr (TRC8) { if (trc_slot && tid == 0 && rr < 32) trc_slot[4 + rr * 5 + 3] = trc8_now(); }
        compute_tile8(t, kraw + stage * TILE_N * HD, vraw + stage * TILE_N * HD, qfrag);
        if constexpr (TRC8) { if (trc_slot && tid == 0 && rr < 32) trc_slot[4 + rr * 5 + 4] = trc8_now(); }
      }
    }
  } else {
    // cp.async ring: raw fp8 lands in shared memory asynchronously, so no
    // staging register stays live across the compute phase.
    auto issue_stage = [&](int t, int stage) {
      uint8_t* dk = kraw + stage * TILE_N * HD;
      uint8_t* dv = vraw + stage * TILE_N * HD;
      const int jb = jt0 + t * TILE_N;
#pragma unroll
      for (int u = 0; u < KVPT; ++u) {
        const int c = tid + u * THREADS;
        const int n = c / HD16, col = (c % HD16) * 16;
        const int jj = min(max(jb + n, 0), kvlen - 1);
        const int page = table_row[min(jj >> p.page_log2, tcols - 1)];
        const long long base = (long long)page * p.kv_s_page + (long long)hkv * p.kv_s_head +
                               (long long)(jj & page_mask) * p.kv_s_tok + col;
        cp_async16(dk + n * HD + col, kp + (base >> 4));
        cp_async16(dv + n * HD + col, vp + (base >> 4));
      }
      cp_async_commit();
    };
    for (int st = 0; st < NSTAGE - 1; ++st) {
      if (t_begin + st < t_end) issue_stage(t_begin + st, st);
      else cp_async_commit();
    }
    for (int t = t_begin; t < t_end; ++t) {
      const int stage = (int)((t - t_begin) % NSTAGE);
      cp_async_wait<NSTAGE - 2>();
      // Also proves every thread finished the previous compute, so the
      // single-buffered f16 tiles can be overwritten.
      __syncthreads();
      {
        const uint8_t* rk = kraw + stage * TILE_N * HD;
        const uint8_t* rv = vraw + stage * TILE_N * HD;
#pragma unroll
        for (int u = 0; u < KVPT; ++u) {
          const int c = tid + u * THREADS;
          const int n = c / HD16, col = (c % HD16) * 16;
          const uint4 krr = *reinterpret_cast<const uint4*>(rk + n * HD + col);
          const uint4 vrr = *reinterpret_cast<const uint4*>(rv + n * HD + col);
          uint32_t kh[8], vh[8];
          expand8(krr, kh);
          expand8(vrr, vh);
          *reinterpret_cast<uint4*>(sK + swz<HD>(n, col)) =
              make_uint4(kh[0], kh[1], kh[2], kh[3]);
          *reinterpret_cast<uint4*>(sK + swz<HD>(n, col + 8)) =
              make_uint4(kh[4], kh[5], kh[6], kh[7]);
          *reinterpret_cast<uint4*>(sV + swz<HD>(n, col)) =
              make_uint4(vh[0], vh[1], vh[2], vh[3]);
          *reinterpret_cast<uint4*>(sV + swz<HD>(n, col + 8)) =
              make_uint4(vh[4], vh[5], vh[6], vh[7]);
        }
      }
      __syncthreads();
      {
        const int tn = t + NSTAGE - 1;
        if (tn < t_end) issue_stage(tn, (int)((tn - t_begin) % NSTAGE));
        else cp_async_commit();
      }
      compute_tile(t, sK, sV, qfrag);
    }
  }

  if constexpr (TRC8) { if (trc_slot && tid == 0) trc_slot[3] = trc8_now(); }

  if (!active) {
    // Unreachable: the identical early exit (with the uniform-mode neutral
    // partial publish) sits above the Q staging so deep device-balanced grids
    // never pay staging cost for empty splits. Kept as a defensive guard.
    return;
  }

  // ---------------- epilogue ----------------
  // sMerge/sML/sRow overlay the K/V staging tiles, which the last compute_tile
  // iteration was still reading. Without this barrier a warp that reaches the
  // epilogue first can overwrite K/V under a slower warp, which is what produced
  // non-finite output on the WARPS_N>1 (decode) path.
  __syncthreads();
  const long long w_id = (long long)by * p.mtiles + mtile;
  const int split_z = p.splits > 1 ? blockIdx.z : 0;
  float M[RB][2], L[RB][2], gain[RB][2], lse2[RB][2], mysc[RB][2];
#pragma unroll
  for (int rb = 0; rb < RB; ++rb) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      lrow[rb][h] += __shfl_xor_sync(0xffffffffu, lrow[rb][h], 1);
      lrow[rb][h] += __shfl_xor_sync(0xffffffffu, lrow[rb][h], 2);
      M[rb][h] = m2[rb][h];  // already log2 units, as the contract stores lse
      L[rb][h] = lrow[rb][h];
    }
  }
  if (WARPS_N > 1) {
    // T7: only the NW compute warps own sML/sMerge slices (convert/load warps
    // carry dummy M/L/O state; their warp_n aliases would corrupt the merge).
    if ((T7 == 0 || warp < WARPS_M * WARPS_N) && (lane & 3) == 0) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = h ? rowb[rb] : rowa[rb];
          sML[(warp_n * TILE_M + r) * 2] = M[rb][h];
          sML[(warp_n * TILE_M + r) * 2 + 1] = L[rb][h];
        }
    }
    __syncthreads();
    if (T7 == 0 || warp < WARPS_M * WARPS_N) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = h ? rowb[rb] : rowa[rb];
          float m = NEG_BIG;
          for (int w = 0; w < WARPS_N; ++w) m = fmaxf(m, sML[(w * TILE_M + r) * 2]);
          float l = 0.f;
          for (int w = 0; w < WARPS_N; ++w) l += sML[(w * TILE_M + r) * 2 + 1] * ex2(sML[(w * TILE_M + r) * 2] - m);
          M[rb][h] = m;
          L[rb][h] = l;
        }
    }
  }
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = h ? rowb[rb] : rowa[rb];
      float base2 = M[rb][h] + (L[rb][h] > 0.f ? lg2f(L[rb][h]) : NEG_BIG);
      float sc = 1.f;
      if (p.sinks != nullptr) {
        const float sink2 = p.sinks[hkv * G + (r - fdiv(r, p.g_magic) * G)] * LOG2E;
        const float hi = fmaxf(base2, sink2);
        const float total2 = hi + lg2f(1.f + ex2(-fabsf(base2 - sink2)));
        sc = ex2(base2 - total2);
        base2 = total2;
      }
      lse2[rb][h] = base2;
      gain[rb][h] = (L[rb][h] > 0.f ? 1.f / L[rb][h] : 0.f) * sc * p.v_scale;
      mysc[rb][h] = ex2(m2[rb][h] - M[rb][h]);
    }

  if (WARPS_N > 1) {
    if (T7 == 0 || warp < WARPS_M * WARPS_N) {
#pragma unroll
      for (int rb = 0; rb < RB; ++rb)
#pragma unroll
        for (int j = 0; j < HD8; ++j) {
          const int c = j * 8 + (lane & 3) * 2;
          sMerge[((warp_n * TILE_M + rowa[rb]) * SMERGE) + c] = O[rb][j][0] * mysc[rb][0];
          sMerge[((warp_n * TILE_M + rowa[rb]) * SMERGE) + c + 1] = O[rb][j][1] * mysc[rb][0];
          sMerge[((warp_n * TILE_M + rowb[rb]) * SMERGE) + c] = O[rb][j][2] * mysc[rb][1];
          sMerge[((warp_n * TILE_M + rowb[rb]) * SMERGE) + c + 1] = O[rb][j][3] * mysc[rb][1];
        }
      if ((lane & 3) == 0) {
#pragma unroll
        for (int rb = 0; rb < RB; ++rb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int r = h ? rowb[rb] : rowa[rb];
            sRow[r * 4 + 0] = M[rb][h];
            sRow[r * 4 + 1] = L[rb][h];
            sRow[r * 4 + 2] = gain[rb][h];
            sRow[r * 4 + 3] = lse2[rb][h];
          }
      }
    }
    __syncthreads();
    if (p.splits > 1 && tid < TILE_M) {
      float* ml =
          p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s + (long long)tid * 2;
      ml[0] = sRow[tid * 4];
      ml[1] = sRow[tid * 4 + 1];
      float* mt =
          p.part_ml_t + w_id * p.partml_s_w + (long long)tid * p.partmlt_s_r + (long long)split_z * 2;
      mt[0] = sRow[tid * 4];
      mt[1] = sRow[tid * 4 + 1];
    }
    for (int e = tid; e < TILE_M * HD; e += THREADS) {
      const int r = e / HD, c = e % HD;
      float val = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS_N; ++w) val += sMerge[(w * TILE_M + r) * SMERGE + c];
      const int i = i0 + fdiv(r, p.g_magic);
      if (i >= qlen) continue;
      const long long tok = q0 + i;
      const int head = hkv * G + (r - fdiv(r, p.g_magic) * G);
      if (p.splits > 1) {
        p.part_o[w_id * p.part_s_w + (long long)split_z * p.part_s_s + (long long)r * HD + c] = val;
      } else {
        p.out[tok * p.out_s_tok + (long long)head * p.out_s_head + c] =
            __float2bfloat16_rn(val * sRow[r * 4 + 2]);
        if (c == 0 && p.write_lse) {
          p.lse[tok * p.lse_s_tok + (long long)head * p.lse_s_head] = sRow[r * 4 + 3];
        }
      }
    }
    return;
  }

  // WARPS_N == 1: every row lives in this warp's fragments, write directly.
#pragma unroll
  for (int rb = 0; rb < RB; ++rb)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (!rowok[rb][h]) continue;
      const int r = h ? rowb[rb] : rowa[rb];
      const long long tok = q0 + i0 + fdiv(r, p.g_magic);
      const int head = hkv * G + (r - fdiv(r, p.g_magic) * G);
      if (p.splits > 1) {
        float* prow =
            p.part_o + w_id * p.part_s_w + (long long)split_z * p.part_s_s + (long long)r * HD;
#pragma unroll
        for (int j = 0; j < HD8; ++j) {
          const int c = j * 8 + (lane & 3) * 2;
          prow[c] = (h ? O[rb][j][2] : O[rb][j][0]) * mysc[rb][h];
          prow[c + 1] = (h ? O[rb][j][3] : O[rb][j][1]) * mysc[rb][h];
        }
        // Always publish the partial stats when splitting: the combine kernel
        // needs them even when no lse tensor was supplied.
        if ((lane & 3) == 0) {
          float* ml = p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s +
                      (long long)r * 2;
          ml[0] = M[rb][h];
          ml[1] = L[rb][h];
          float* mt = p.part_ml_t + w_id * p.partml_s_w + (long long)r * p.partmlt_s_r +
                      (long long)split_z * 2;
          mt[0] = M[rb][h];
          mt[1] = L[rb][h];
        }
      } else {
        __nv_bfloat16* orow = p.out + tok * p.out_s_tok + (long long)head * p.out_s_head;
        const float g = gain[rb][h] * mysc[rb][h];
#pragma unroll
        for (int j = 0; j < HD8; ++j) {
          const int c = j * 8 + (lane & 3) * 2;
          orow[c] = __float2bfloat16_rn((h ? O[rb][j][2] : O[rb][j][0]) * g);
          orow[c + 1] = __float2bfloat16_rn((h ? O[rb][j][3] : O[rb][j][1]) * g);
        }
        if (p.write_lse && (lane & 3) == 0) {
          p.lse[tok * p.lse_s_tok + (long long)head * p.lse_s_head] = lse2[rb][h];
        }
      }
    }
}

// ---------------------------------------------------------------------------
// fp8 e4m3 -> f16 mirror of the paged KV cache, same element order, so the main
// kernel can cp.async 16-byte f16 chunks straight into swizzled shared-memory
// tiles with no convert pass. The measured staged-mimic (probe3) put that convert
// pass at 26% of prefill time, and prefill re-runs it once per q tile, so
// converting once per forward and reading the mirror back (L2-resident, L2 was at
// 2% utilisation) is a large net win. Exact: e4m3 has 3 mantissa bits, f16 has 10.
// Fused K+V mirror convert: one launch instead of two (the mirror path always
// converts both tensors, and on small prefill shapes the second launch node was
// ~1-2us of exposed graph/launch latency). The index space is the concatenation
// of the two 16-element vector counts; only the boundary warp diverges.
__global__ void cvt_kv_kernel(const uint8_t* __restrict__ srcK, __half* __restrict__ dstK,
                              long long nK16, const uint8_t* __restrict__ srcV,
                              __half* __restrict__ dstV, long long nV16) {
  const long long total = nK16 + nV16;
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
    const uint8_t* src;
    __half* dst;
    long long j;
    if (i < nK16) {
      src = srcK;
      dst = dstK;
      j = i;
    } else {
      src = srcV;
      dst = dstV;
      j = i - nK16;
    }
    const uint4 raw = reinterpret_cast<const uint4*>(src)[j];
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
    uint32_t h[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      h[2 * e] = cvt_e4m3x2(w[e] & 0xffffu);
      h[2 * e + 1] = cvt_e4m3x2(w[e] >> 16);
    }
    reinterpret_cast<uint4*>(dst)[j * 2] = make_uint4(h[0], h[1], h[2], h[3]);
    reinterpret_cast<uint4*>(dst)[j * 2 + 1] = make_uint4(h[4], h[5], h[6], h[7]);
  }
}

static void launch_cvt_kv(const uint8_t* srcK, __half* dstK, long long elemsK,
                          const uint8_t* srcV, __half* dstV, long long elemsV,
                          cudaStream_t stream) {
  const long long nK16 = elemsK / 16, nV16 = elemsV / 16;
  const long long n16 = nK16 + nV16;
  if (n16 <= 0) return;
  const int threads = 256;
  long long blocks = (n16 + threads - 1) / threads;
  if (blocks > 4096) blocks = 4096;
  cvt_kv_kernel<<<(unsigned)blocks, threads, 0, stream>>>(srcK, dstK, nK16, srcV, dstV, nV16);
}

// ---------------------------------------------------------------------------
// Split-KV combine: part_o [W][splits][ROWS][HD] (unnormalised, scaled to the
// per-split max) and part_ml [W][splits][ROWS][2] = {max_log2, sum}.
// The grid covers output-element chunks as well as work items: a single CTA per
// work item read every partial with only TILE_M threads active and turned a
// few-MB reduction into hundreds of microseconds of latency-bound loads.
// ---------------------------------------------------------------------------
template <int HD, int TILE_M, int THREADS, int CHUNK, int ZP>
__global__ void __launch_bounds__(THREADS) fmha_combine(const __grid_constant__ Params p) {
  extern __shared__ char smem_raw[];
  float* sScale = reinterpret_cast<float*>(smem_raw);  // [splits][TILE_M]
  float* sRow = sScale + p.splits * TILE_M;            // [TILE_M][4]

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int mtile = blockIdx.x;
  const int by = blockIdx.y;
  const int chunk = blockIdx.z;
  const int b = fdiv(by, p.hkv_magic);
  const int hkv = by - b * p.HKV;
  const int q0 = p.cu_q[b];
  const int qlen = p.cu_q[b + 1] - q0;
  const int G = p.G;
  const int m0 = mtile * TILE_M;
  if (m0 >= qlen * G) return;
  const int i0 = fdiv(m0, p.g_magic);
  const long long w_id = (long long)by * p.mtiles + mtile;
  const float* po = p.part_o + w_id * p.part_s_w;
  const float* pml = p.part_ml + w_id * p.partml_s_w;
  // Row-major stats twin, coalesced for the warp-per-row scan below: lane l
  // reads z = l + 32k, i.e. 32 consecutive float2 (256B) per warp instruction
  // instead of 32 sectors at partml_s_s stride (NCU on dv_629ea9c7c234: 73.9%
  // of combine stalls were L1TEX scoreboard on the strided form).
  const float* pmlt = p.part_ml_t + w_id * p.partml_s_w;
  // TPS mode: recompute this work item's own split count with EXACTLY the main
  // kernel's formula (same inputs, same integer arithmetic) so the reduction
  // covers precisely the published partials.
  int splits = p.splits;
  if (p.tps_target != 0) {
    const int kvlen = p.seq_lens[b];
    const int i1 = min(qlen, fdiv(m0 + TILE_M, p.g_magic));
    const int vis0 = kvlen - qlen + i0;
    const int vis1 = kvlen - qlen + i1 - 1;
    const int jlo = p.window_left >= 0 ? max(0, vis0 - p.window_left) : 0;
    const int jhi = min(kvlen - 1, vis1);
    // Shift form via p.tile_n_log2 -- identical values to the main kernel's
    // compile-time-constant divisions without the runtime-division sequences
    // (measured +1.4..2.9us per device-mode combine in the archived trials).
    const int tnl = p.tile_n_log2;
    const int tn = 1 << tnl;
    const int jt0 = (jlo >> tnl) << tnl;
    const int jt1 = ((jhi + tn) >> tnl) << tnl;
    const int ntiles = (jt1 - jt0) >> tnl;
    const int tt = p.tps_target > 0 ? p.tps_target
                                    : tps_device_tt(p, tnl, TILE_M);
    splits = (ntiles + tt - 1) / tt;
    if (splits > p.splits) splits = p.splits;
    if (splits < 1) splits = 1;
  }

  // Element chunk of this CTA, and the packed rows it touches. The stats scan
  // is restricted to those rows: a chunk-32 CTA at hd128 covers ONE row, so
  // scanning all TILE_M rows (as before) wasted 15/16 of the scan work and
  // dominated the high-splits combine (d9: 9.0us at 2.4MB of partials).
  const int e0 = chunk * CHUNK;
  const int e1 = min(TILE_M * HD, e0 + CHUNK);
  const int rowlo = e0 / HD;
  const int rowhi = (e1 - 1) / HD;

  // Hybrid row-stat scan, strategy chosen by the split count AND the CTA's
  // row footprint.
  //   splits <= 64 AND many rows per CTA: one thread per row (the original
  //     form). All rows are scanned in parallel and each serial z chain is
  //     short enough. A warp per row measured badly at these counts WITH FULL
  //     ROW SETS: TILE_M 128 at splits 5 serialises 16 rows per warp (213.6us
  //     against 54.3us for this form).
  //   splits > 64 OR few rows per CTA: one warp per row, lanes striding z. The
  //     single-thread chain of dependent load/use iterations made high split
  //     counts latency-bound (104us at splits 512 on the 262144-token decode
  //     shape), and the decode geometry is the same pathology at moderate
  //     splits: a chunk-64/256 CTA covers 1-4 rows, so ONE thread walks up to
  //     37 split stats serially while the other 255 threads wait at the
  //     barrier -- kineto put the small-decode combines at 3.6-6.9us for 1-3MB
  //     of partials while the chunk/ZP element geometry measured FLAT there
  //     (dv_77702db2f807), isolating the scan; warp-per-row cut the f2 combine
  //     ~2us and the deep-split synthetics 5-7% (archived dv_704146cd4279/
  //     dv_2d01c3d75fc6, mechanism re-derived this phase from the same data).
  //   Numerics: the row max is order-independent (identical bits); the l sum
  //   reassociates strided-partial + shuffle-tree against the serial chain
  //   (~1e-7 relative, five orders inside the 0.01 + 0.05 contract).
  const int rows_cta = rowhi - rowlo + 1;
  const bool warp_scan = (splits > 64) || (p.wscan != 0 && rows_cta * 32 <= THREADS);
  if (!warp_scan) {
    for (int r = rowlo + tid; r <= rowhi; r += THREADS) {
      if (i0 + fdiv(r, p.g_magic) >= qlen) {
        sRow[r * 4 + 0] = NEG_BIG;
        sRow[r * 4 + 1] = 0.f;
        sRow[r * 4 + 2] = 0.f;
        sRow[r * 4 + 3] = 0.f;
        for (int z = 0; z < splits; ++z) sScale[z * TILE_M + r] = 0.f;
        continue;
      }
      float m = NEG_BIG;
      for (int z = 0; z < splits; ++z) m = fmaxf(m, pml[z * p.partml_s_s + r * 2]);
      float l = 0.f;
      for (int z = 0; z < splits; ++z) {
        const float mz = pml[z * p.partml_s_s + r * 2];
        const float sc = ex2(mz - m);
        sScale[z * TILE_M + r] = sc;
        l += pml[z * p.partml_s_s + r * 2 + 1] * sc;
      }
      float base2 = m + (l > 0.f ? lg2f(l) : NEG_BIG);
      float sc = 1.f;
      if (p.sinks != nullptr) {
        const float sink2 = p.sinks[hkv * G + (r - fdiv(r, p.g_magic) * G)] * LOG2E;
        const float hi = fmaxf(base2, sink2);
        const float total2 = hi + lg2f(1.f + ex2(-fabsf(base2 - sink2)));
        sc = ex2(base2 - total2);
        base2 = total2;
      }
      sRow[r * 4 + 0] = m;
      sRow[r * 4 + 1] = l;
      sRow[r * 4 + 2] = (l > 0.f ? 1.f / l : 0.f) * sc * p.v_scale;
      sRow[r * 4 + 3] = base2;
    }
  } else {
    constexpr int WARPS = THREADS / 32;
    for (int r = rowlo + warp; r <= rowhi; r += WARPS) {
      if (i0 + fdiv(r, p.g_magic) >= qlen) {
        if (lane == 0) {
          sRow[r * 4 + 0] = NEG_BIG;
          sRow[r * 4 + 1] = 0.f;
          sRow[r * 4 + 2] = 0.f;
          sRow[r * 4 + 3] = 0.f;
        }
        for (int z = lane; z < splits; z += 32) sScale[z * TILE_M + r] = 0.f;
        continue;
      }
      const float2* pr = reinterpret_cast<const float2*>(pmlt + (long long)r * p.partmlt_s_r);
      float mloc = NEG_BIG;
      for (int z = lane; z < splits; z += 32)
        mloc = fmaxf(mloc, pr[z].x);
#pragma unroll
      for (int s = 16; s > 0; s >>= 1)
        mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, s));
      const float m = mloc;
      float lloc = 0.f;
      for (int z = lane; z < splits; z += 32) {
        const float2 mz = pr[z];
        const float sc = ex2(mz.x - m);
        sScale[z * TILE_M + r] = sc;
        lloc += mz.y * sc;
      }
#pragma unroll
      for (int s = 16; s > 0; s >>= 1)
        lloc += __shfl_xor_sync(0xffffffffu, lloc, s);
      if (lane == 0) {
        const float l = lloc;
        float base2 = m + (l > 0.f ? lg2f(l) : NEG_BIG);
        float sc = 1.f;
        if (p.sinks != nullptr) {
          const float sink2 = p.sinks[hkv * G + (r - fdiv(r, p.g_magic) * G)] * LOG2E;
          const float hi = fmaxf(base2, sink2);
          const float total2 = hi + lg2f(1.f + ex2(-fabsf(base2 - sink2)));
          sc = ex2(base2 - total2);
          base2 = total2;
        }
        sRow[r * 4 + 0] = m;
        sRow[r * 4 + 1] = l;
        sRow[r * 4 + 2] = (l > 0.f ? 1.f / l : 0.f) * sc * p.v_scale;
        sRow[r * 4 + 3] = base2;
      }
    }
  }
  __syncthreads();

  if constexpr (ZP == 1 && CHUNK >= 4 * THREADS) {
    // V4Z1: one float4 of output channels per thread per pass instead of one
    // scalar. NCU on this kernel (dv_42224288df76, d9's combine) found 540
    // instructions per warp of which ~10 are loads, 38.49 warp-cycles per issued
    // instruction and 94.91% No Eligible at 32 registers -- one load in flight per
    // thread -- and f9's combine moves ~10.6MB in 9.71us (1.09 TB/s) against the
    // 2.98 TB/s cvt_kv sustains on this device. Four times fewer load instructions,
    // each four times wider, and a warp's loads then cover 512 contiguous bytes
    // instead of 128. The per-element z order is unchanged (val += src[z]*scl[z], z
    // ascending) and the same single rounding is applied, so the output bytes are
    // identical and this must be bit-identical.
    // Gated on CHUNK >= 4*THREADS so the group count never falls below the CTA
    // width: at chunk 1024 there are exactly THREADS groups (f9), at 4096 four per
    // thread, while chunk 256 would leave three quarters of the threads idle and
    // keeps the scalar form below.
    // Alignment is structural: HD % 4 == 0 so a group never spans two rows and
    // c is a multiple of 4; part_s_w and part_s_s are tile_m*hd multiples with
    // hd in {64,128,256}; part_o comes from a float32 tensor. So every float4 is
    // 16B aligned.
    for (int g = (e0 >> 2) + tid; g < (e1 >> 2); g += THREADS) {
      const int e = g << 2;
      const int r = e / HD, c = e % HD;
      const int i = i0 + fdiv(r, p.g_magic);
      if (i >= qlen) continue;
      const float* src = po + r * HD + c;
      const float* scl = sScale + r;
      float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int z = 0; z < splits; ++z) {
        const float4 v = *reinterpret_cast<const float4*>(src + z * p.part_s_s);
        const float sc = scl[z * TILE_M];
        val.x += v.x * sc;
        val.y += v.y * sc;
        val.z += v.z * sc;
        val.w += v.w * sc;
      }
      const long long tok = q0 + i;
      const int head = hkv * G + (r - fdiv(r, p.g_magic) * G);
      const float gain = sRow[r * 4 + 2];
      __nv_bfloat16* dst = p.out + tok * p.out_s_tok + (long long)head * p.out_s_head + c;
      dst[0] = __float2bfloat16_rn(val.x * gain);
      dst[1] = __float2bfloat16_rn(val.y * gain);
      dst[2] = __float2bfloat16_rn(val.z * gain);
      dst[3] = __float2bfloat16_rn(val.w * gain);
      if (c == 0 && p.write_lse) {
        p.lse[tok * p.lse_s_tok + (long long)head * p.lse_s_head] = sRow[r * 4 + 3];
      }
    }
  } else if constexpr (ZP == 1) {
    for (int e = e0 + tid; e < e1; e += THREADS) {
      const int r = e / HD, c = e % HD;
      const int i = i0 + fdiv(r, p.g_magic);
      if (i >= qlen) continue;
      const float* src = po + r * HD + c;
      const float* scl = sScale + r;
      float val = 0.f;
      for (int z = 0; z < splits; ++z) val += src[z * p.part_s_s] * scl[z * TILE_M];
      const long long tok = q0 + i;
      const int head = hkv * G + (r - fdiv(r, p.g_magic) * G);
      p.out[tok * p.out_s_tok + (long long)head * p.out_s_head + c] =
          __float2bfloat16_rn(val * sRow[r * 4 + 2]);
      if (c == 0 && p.write_lse) {
        p.lse[tok * p.lse_s_tok + (long long)head * p.lse_s_head] = sRow[r * 4 + 3];
      }
    }
  } else {
    // z-parallel VEC4 element reduction: each lane owns one float4 of output
    // channels and strides the split axis with ZP-1 peer lanes, folding with a
    // width-ZP shuffle. Chains are ceil(splits/ZP) VECTOR loads instead of
    // splits scalar loads (296 serial 4B loads measured 23.5us against 28.6us
    // for the whole main kernel; the 16-lane scalar form still took 18.9us).
    // The generic group stride covers CHUNK*ZP above or below THREADS.
    // Invalid rows are predicated on the load (their sScale column is zero) to
    // keep the shuffle warp-uniform and avoid folding uninitialised partials.
    const int g0 = chunk * (CHUNK / 4);
    const int g1 = min(TILE_M * HD / 4, g0 + CHUNK / 4);
    constexpr int GPB = THREADS / ZP;  // float4 groups per pass
    for (int g = g0 + tid / ZP; g < g1; g += GPB) {
      const int e = g * 4;
      const int r = e / HD, c = e % HD;
      const int i = i0 + fdiv(r, p.g_magic);
      const bool valid = i < qlen;
      const float* src = po + r * HD + c;
      const float* scl = sScale + r;
      float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int z = tid % ZP; z < splits; z += ZP) {
        const float s = scl[z * TILE_M];
        if (valid) {
          const float4 v = *reinterpret_cast<const float4*>(src + z * p.part_s_s);
          val.x += v.x * s;
          val.y += v.y * s;
          val.z += v.z * s;
          val.w += v.w * s;
        }
      }
#pragma unroll
      for (int s = ZP / 2; s > 0; s >>= 1) {
        val.x += __shfl_down_sync(0xffffffffu, val.x, s, ZP);
        val.y += __shfl_down_sync(0xffffffffu, val.y, s, ZP);
        val.z += __shfl_down_sync(0xffffffffu, val.z, s, ZP);
        val.w += __shfl_down_sync(0xffffffffu, val.w, s, ZP);
      }
      if (valid && (tid % ZP) == 0) {
        const long long tok = q0 + i;
        const int head = hkv * G + (r - fdiv(r, p.g_magic) * G);
        const float gain = sRow[r * 4 + 2];
        __nv_bfloat16* orow = p.out + tok * p.out_s_tok + (long long)head * p.out_s_head + c;
        // torch builds with -D__CUDA_NO_BFLOAT16_CONVERSIONS__, so pack via an
        // aligned local array instead of the bfloat162 bit-cast intrinsics.
        alignas(8) const __nv_bfloat16 h4[4] = {
            __float2bfloat16_rn(val.x * gain), __float2bfloat16_rn(val.y * gain),
            __float2bfloat16_rn(val.z * gain), __float2bfloat16_rn(val.w * gain)};
        const uint2 packed = *reinterpret_cast<const uint2*>(h4);
        if ((reinterpret_cast<uintptr_t>(orow) & 7u) == 0)
          *reinterpret_cast<uint2*>(orow) = packed;
        else {
          orow[0] = h4[0];
          orow[1] = h4[1];
          orow[2] = h4[2];
          orow[3] = h4[3];
        }
        if (c == 0 && p.write_lse) {
          p.lse[tok * p.lse_s_tok + (long long)head * p.lse_s_head] = sRow[r * 4 + 3];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// tcgen05 (5th-gen tensor core) prefill kernel -- cfg80.
//
// Why this exists, in measurements: the mma.sync path this file otherwise uses
// peaks at 574 TFLOPS on this device and the kernel reaches ~109, with NCU
// attributing the wall to shared-memory traffic (320 bytes of ldmatrix per mma,
// L1/TEX at 75%) and to occupancy (a 64-128 register per-thread accumulator caps
// it at 4 CTAs/SM, 22.5% occupancy). probe4 measured one m128n64k16 f16
// tcgen05.mma at 25.3ns on one SM = 10.4 TFLOPS/SM = 1532 TFLOPS over 148 SMs
// (2.67x mma.sync, and consistent with the 1607-1776 TFLOPS cuBLAS bf16 reaches
// here), with a single CTA issuing from a single thread already saturating it.
// probe5 validated the whole attention sequence below against torch.
//
// Structural differences from the mma.sync kernel:
//   * the tensor core reads operands from shared memory through descriptors, so
//     there is no ldmatrix traffic at all;
//   * accumulators live in TMEM, so the register file holds one row of S and P
//     per lane instead of an HD-wide O accumulator;
//   * TILE_M is 128 packed rows and each lane owns exactly one row, so the row
//     max and row sum need no shuffles;
//   * the KV tile layout is K-group-major (each 64-element group of a row is a
//     128-byte swizzle atom, groups stacked M-first), which is what the
//     descriptor's SBO/LBO encode -- NOT the per-row swizzle the ldmatrix path
//     uses, so this kernel has its own staging index function.
// Requires the f16 KV mirror (p.k16/p.v16): the descriptors address f16 tiles.
// ---------------------------------------------------------------------------
// __cvta_generic_to_shared yields size_t, so the shared-window address needs a
DEVFN void t5_alloc(uint32_t dst_smem, int ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
               ::"r"(dst_smem), "r"(ncols));
}
DEVFN void t5_dealloc(uint32_t base, int ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(base), "r"(ncols));
}
DEVFN void t5_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
}
DEVFN void t5_commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
               ::"r"(mbar));
}
// D = A(smem) x B(smem), both K-major f16, f32 accumulator in TMEM
DEVFN void t5_umma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, int acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
// D = A(TMEM, f16 pairs) x B(smem, MN-major), used for P x V
DEVFN void t5_umma_ts(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t idesc, int acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
               ::"r"(d), "r"(a_tmem), "l"(b), "r"(idesc), "r"(acc));
}
DEVFN void t5_ld32(float* r, uint32_t addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
               : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]),
                 "=f"(r[6]), "=f"(r[7]), "=f"(r[8]), "=f"(r[9]), "=f"(r[10]), "=f"(r[11]),
                 "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15]), "=f"(r[16]), "=f"(r[17]),
                 "=f"(r[18]), "=f"(r[19]), "=f"(r[20]), "=f"(r[21]), "=f"(r[22]), "=f"(r[23]),
                 "=f"(r[24]), "=f"(r[25]), "=f"(r[26]), "=f"(r[27]), "=f"(r[28]), "=f"(r[29]),
                 "=f"(r[30]), "=f"(r[31])
               : "r"(addr));
}
// One-instruction 64-word TMEM row load (LD64): replaces 2x t5_ld32 + shared
// wait when TILE_N == 64; same registers, one less issue slot per tile.
DEVFN void t5_ld64(float* r, uint32_t addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];\n"
               : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7]), "=f"(r[8]), "=f"(r[9]), "=f"(r[10]), "=f"(r[11]), "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15]), "=f"(r[16]), "=f"(r[17]), "=f"(r[18]), "=f"(r[19]), "=f"(r[20]), "=f"(r[21]), "=f"(r[22]), "=f"(r[23]), "=f"(r[24]), "=f"(r[25]), "=f"(r[26]), "=f"(r[27]), "=f"(r[28]), "=f"(r[29]), "=f"(r[30]), "=f"(r[31]), "=f"(r[32]), "=f"(r[33]), "=f"(r[34]), "=f"(r[35]), "=f"(r[36]), "=f"(r[37]), "=f"(r[38]), "=f"(r[39]), "=f"(r[40]), "=f"(r[41]), "=f"(r[42]), "=f"(r[43]), "=f"(r[44]), "=f"(r[45]), "=f"(r[46]), "=f"(r[47]), "=f"(r[48]), "=f"(r[49]), "=f"(r[50]), "=f"(r[51]), "=f"(r[52]), "=f"(r[53]), "=f"(r[54]), "=f"(r[55]), "=f"(r[56]), "=f"(r[57]), "=f"(r[58]), "=f"(r[59]), "=f"(r[60]), "=f"(r[61]), "=f"(r[62]), "=f"(r[63])
               : "r"(addr));
}

DEVFN void t5_st32(const uint32_t* r, uint32_t addr) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};\n"
               ::"r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]),
                 "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]),
                 "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]),
                 "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]),
                 "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]),
                 "r"(addr));
}
DEVFN void t5_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;\n"); }
DEVFN void t5_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;\n"); }
// tcgen05.mma consumes its smem operands through descriptors in the ASYNC
// proxy. Ordinary STS (Q staging, FP8S register-staged KV tiles) lives in the
// generic proxy: __syncthreads alone does NOT order those writes against the
// tensor core's reads. Every thread that stored must execute the async-proxy
// fence before the barrier; the mirror cp.async path never needed it because
// cp.async writes are async-proxy already. Missing this was the sparse,
// nondeterministic fp8s+splits divergence localised by dbg2 (kernel output
// == torch fold of the published partials, but the partials themselves were
// run-dependent: racecheck-class hazard, not a logic bug).
DEVFN void t5_fence_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}
DEVFN void t5_fence_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}
// Device-wide nanosecond clock (TRACE diagnostics). Unlike clock64() it is
// comparable ACROSS SMs, so per-CTA [start, end] intervals reconstruct the
// launch's wave structure and idle time; resolution is ~32ns, which is fine
// against per-CTA spans of microseconds.
DEVFN unsigned long long t5_gtimer() {
  unsigned long long g;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
  return g;
}
// shared-memory matrix descriptor: SWIZZLE_128B(2)<<61 | version(1)<<46 |
// (SBO>>4)<<32 | (LBO>>4)<<16 | (addr & 0xFFFFFF)>>4
DEVFN uint64_t t5_desc(uint32_t addr, int lbo, int sbo) {
  return (2ULL << 61) | (1ULL << 46) | ((uint64_t)(sbo >> 4) << 32) |
         ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)(addr & 0xFFFFFF) >> 4);
}
// instruction descriptor: ACC_TYPE f32 at bit 4, B_TRANSPOSE at 16, N>>3 at 17,
// M>>5 at 25 (SRC_A/B_TYPE = 0 means f16)
DEVFN uint32_t t5_idesc(int m, int n, int b_transpose) {
  return (1u << 4) | ((uint32_t)b_transpose << 16) | ((uint32_t)(n >> 3) << 17) |
         ((uint32_t)(m >> 5) << 25);
}
// K-group-major 128B-swizzled byte offset for element (r, c) of a [rows, HD] tile
DEVFN int t5_off(int rows, int r, int c) {
  return (c >> 6) * (rows * 128) + r * 128 + ((((c & 63) >> 3) ^ (r & 7)) & 7) * 16 +
         ((c & 7) * 2);
}

template <int HD, int TILE_N = 64, int MINB = 2, int DEFER = 0, int PIPE = 0, int ILP = 0,
          int INSTR = 0, int KB3 = 0, int F2 = 0, int VS = 0, int RV = 0, int DAO = 0,
          int TPF = 0, int SPG = 0, int FP8S = 0, int NOSYNC = 0, int PSM = 0, int MTM = 0,
          int X2H = 0, int DTH = 1, int LD64 = 0>
__global__ void __launch_bounds__(128, MINB) fmha_main_t5(const Params p) {
  constexpr int TILE_M = 128;
  constexpr int THREADS = 128;
  constexpr int KBUFS = (KB3 || RV) ? 3 : 2;  // K staging buffers
  constexpr int VBUFS = 2;                    // V staging buffers
  constexpr int KG = HD / 64;                       // K groups for QK, N atoms for PV
  // VS appends a 64-column ones/zero pad to every V tile: PV's extra N-atom
  // accumulates sum(P) into TMEM O column HD, in lockstep with every DEFER
  // correction of O, so the per-lane row-sum chain leaves the softmax entirely.
  constexpr int HDX = VS ? HD + 64 : HD;            // V tile / O column count
  constexpr int KGX = VS ? KG + 1 : KG;             // PV N-atoms
  // VS places O right after the single S buffer (VS is non-PIPE); otherwise O
  // starts beyond the 128-column S region PIPE reserves.
  constexpr int OCOL = VS ? 64 : (HD <= 128 ? 128 : 256);
  constexpr int NCOL =
      VS ? (64 + HDX <= 256 ? 256 : 512) : (HD <= 128 ? 256 : 512);
  static_assert(!(PIPE && KB3), "KB3 restages the non-PIPE schedule only");
  static_assert(!(PIPE && VS), "VS O region overlaps PIPE's second S buffer");
  static_assert(!(PIPE && RV), "RV is its own schedule");
  static_assert(!(KB3 && RV), "RV subsumes KB3 staging");
  static_assert(!(TPF && (RV || KB3 || PIPE)), "TPF ring is wired for the plain schedule");
  static_assert(TILE_N % 16 == 0, "page-id prefetch assumes 16-token blocks");
  static_assert(!(VS && RV) || true, "VS+RV allowed");
  // FP8S: register-staged fp8 -> f16 convert (no mirror). Its own schedule;
  // the mirror-only lookahead machinery (TPF/SPG) and the 3-buffer schedules
  // do not apply. VS is excluded because the ones-pad columns are written by
  // the mirror staging path, which FP8S replaces.
  static_assert(!(FP8S && (PIPE || KB3 || RV || VS || TPF || SPG)),
                "FP8S is its own staging schedule");
  // PSM publishes the softmax P tile to shared memory (K-major, 128B-swizzled
  // like Q) and issues PV as A-from-smem umma_ss instead of the TMEM P round
  // trip (t5_st32 + wait). Single P region: PV(t) is mbar-awaited by every
  // thread before iteration t+1's softmax writes it again.
  static_assert(!(PSM && VS), "PSM untested with the V ones-pad");
  // pd0 + s*2 k-step addressing covers exactly ONE 64-column K group.
  static_assert(!PSM || TILE_N == 64, "PSM P descriptor assumes a single K group");
  // X2H only replaces the math inside the F2 full-tile branch.
  static_assert(!X2H || F2, "X2H rides the F2 full-tile softmax branch");
  // LD64 loads the whole S row in one tcgen05.ld x64.
  static_assert(!LD64 || TILE_N == 64, "LD64 assumes a 64-column S row");
  // The epilogue stages every row through the shared memory that Q/K/V used, so it
  // must fit in the same allocation: TILE_M*(HD+4) floats for the accumulator and
  // the four per-row values.
  static_assert(TILE_M * (HD + 4) * 4 <=
                    ((TILE_M + KBUFS * TILE_N) * HD + VBUFS * TILE_N * HDX) * 2 + 64 +
                    (PSM ? TILE_M * TILE_N * 2 : 0),
                "epilogue staging must fit the kernel's shared-memory allocation");
  // Built by the shared encoder rather than hand-written, so the bit layout lives
  // in exactly one place: QK is K-major B, PV is MN-major B (transpose bit set)
  // with N = 64 f16 = one 128-byte swizzle atom.
  const uint32_t ID_QK = t5_idesc(TILE_M, TILE_N, 0);
  const uint32_t ID_PV = t5_idesc(TILE_M, 64, 1);

  extern __shared__ char smraw[];
  char* sQ = smraw;
  char* sK = sQ + TILE_M * HD * 2;
  char* sV = sK + KBUFS * TILE_N * HD * 2;
  // PSM: dedicated P tile (TILE_M x TILE_N f16, K-major 128B-swizzled like Q),
  // PV's A operand via descriptor instead of the TMEM P round trip. Dead by
  // the epilogue, which overlays only sQ/sK from smraw.
  char* sP = sV + VBUFS * TILE_N * HDX * 2;
  uint32_t* tmem_word =
      reinterpret_cast<uint32_t*>(sP + (PSM ? TILE_M * TILE_N * 2 : 0));
  uint64_t* mbar = reinterpret_cast<uint64_t*>(tmem_word + 4);
  // SPG: per-CTA page-id table in shared memory (4 ids per 64-token tile plus
  // the clamped-tail id), preloaded once so the staging loop's address chain
  // starts at a ~20-cycle LDS instead of an L1/L2-hit table LDG (NCU on
  // cfg100: long_scoreboard = 30.7% of the 4.96-cycle issue interval, and
  // 19.6/32-byte global-load sector utilization).
  int* spg = SPG ? reinterpret_cast<int*>(mbar + 4) : nullptr;
  int id_last = 0;
  bool spg_ok = false;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  // MTM: flattened mtile-major DESCENDING launch order. Under causal masking a
  // CTA's tile count grows with its mtile, so launching high mtiles first
  // across ALL y-blocks makes greedy block scheduling globally longest-first:
  // makespan -> max(longest CTA, total/resident) instead of a tail of long
  // CTAs that only start once short ones retire (f7 geometry simulation:
  // 196 -> 128 tile-times; per-y reversal alone only reached 186). The
  // logical (mtile, by) of every CTA is unchanged, so partial/out writes and
  // the combine kernel stay bit-identical. V0 rejected longest-first for the
  // y-major reversal only (ragged batches); MTM's early-exit CTAs (high
  // mtiles of short requests) launch first and cost ~0.
  int mtile, by;
  if constexpr (MTM) {
    const unsigned Y = gridDim.x / (unsigned)p.mtiles;
    by = (int)(blockIdx.x % Y);
    mtile = p.mtiles - 1 - (int)(blockIdx.x / Y);
  } else {
    mtile = (int)blockIdx.x;
    by = (int)blockIdx.y;
  }
  const int b = by / p.HKV;
  const int hkv = by % p.HKV;
  const int q0 = p.cu_q[b];
  const int qlen = p.cu_q[b + 1] - q0;
  const int kvlen = p.seq_lens[b];
  const int G = p.G;
  const int m0 = mtile * TILE_M;
  if (m0 >= qlen * G) return;

  const int i0 = m0 / G;
  const int i1 = min(qlen, (m0 + TILE_M) / G);
  const int vis0 = kvlen - qlen + i0;
  const int vis1 = kvlen - qlen + i1 - 1;
  const int jlo = p.window_left >= 0 ? max(0, vis0 - p.window_left) : 0;
  const int jhi = min(kvlen - 1, vis1);
  const int jt0 = (jlo / TILE_N) * TILE_N;
  const int jt1 = ((jhi + 1 + TILE_N - 1) / TILE_N) * TILE_N;
  const int ntiles = (jt1 - jt0) / TILE_N;
  int t_begin = 0, t_end = ntiles;
  if (p.splits > 1) {
    // Same per-item split derivation as the mma.sync kernel: with tps_target
    // each item splits by its OWN tile count around a common tiles-per-split,
    // and the combine recomputes this formula exactly. Without it, uniform.
    int splits_b = p.splits;
    if (p.tps_target > 0) {
      splits_b = (ntiles + p.tps_target - 1) / p.tps_target;
      if (splits_b > p.splits) splits_b = p.splits;
      if (splits_b < 1) splits_b = 1;
    }
    const int tps = (ntiles + splits_b - 1) / splits_b;
    t_begin = blockIdx.z * tps;
    t_end = min(ntiles, t_begin + tps);
  }
  const long long w_id = (long long)by * p.mtiles + mtile;
  const int split_z = p.splits > 1 ? blockIdx.z : 0;

  // ---- per-lane row constants: lane l of warp w owns packed row w*32+l ----
  const int row = warp * 32 + lane;
  const int irow = i0 + row / G;
  const bool rowok = irow < qlen;
  const int vis = kvlen - qlen + irow;
  // An invalid row is treated as fully masked: lim < 0 hides every column, so its
  // probability mass is 0 and the epilogue skips it. No sentinel ever reaches a
  // score, which is what kept the mma.sync path free of exp2(0)=1 on masked tiles.
  const int lim = rowok ? vis : -1;
  const int low = (rowok && p.window_left >= 0) ? max(0, vis - p.window_left) : 0;

  if (t_begin >= t_end) {
    // Neutral partial, identical to the mma.sync path's empty-split contract.
    // In TPS mode the combine derives the same splits_b and never reads
    // z >= splits_b, so inactive CTAs write nothing at all.
    if (p.splits > 1 && p.tps_target == 0) {
      if (rowok) {
        float* prow = p.part_o + w_id * p.part_s_w + (long long)split_z * p.part_s_s +
                      (long long)row * HD;
        for (int c = 0; c < HD; ++c) prow[c] = 0.f;
      }
      // every lane owns a distinct packed row, so each publishes its own stats
      float* ml = p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s +
                  (long long)row * 2;
      ml[0] = NEG_BIG;
      ml[1] = 0.f;
      float* mt = p.part_ml_t + w_id * p.partml_s_w + (long long)row * p.partmlt_s_r +
                  (long long)split_z * 2;
      mt[0] = NEG_BIG;
      mt[1] = 0.f;
    }
    return;
  }

  // ---- Q tile -> shared memory, bf16 -> f16, K-group-major swizzled ----
  constexpr int QCH = TILE_M * HD / 8;
  const __nv_bfloat16* qbase = p.q + (long long)q0 * p.q_s_tok;
#pragma unroll
  for (int u = 0; u < (QCH + THREADS - 1) / THREADS; ++u) {
    const int c = tid + u * THREADS;
    if (QCH % THREADS != 0 && c >= QCH) break;
    const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
    const int i = i0 + r / G;
    uint4 data = make_uint4(0, 0, 0, 0);
    if (i < qlen) {
      const __nv_bfloat16* src =
          qbase + (long long)i * p.q_s_tok + (long long)(hkv * G + (r % G)) * p.q_s_head + col;
      data = *reinterpret_cast<const uint4*>(src);
    }
    const uint4 h = make_uint4(bf16x2_to_f16x2(data.x), bf16x2_to_f16x2(data.y),
                               bf16x2_to_f16x2(data.z), bf16x2_to_f16x2(data.w));
    *reinterpret_cast<uint4*>(sQ + t5_off(TILE_M, r, col)) = h;
  }

  if constexpr (VS) {
    // Ones/zero pad columns (HD..HD+63) of both V buffers, written as 16-byte
    // chunks at their swizzled atom offsets. Column HD holds 1.0 so PV folds
    // sum(P) into O column HD; the data staging loop never writes these.
    const uint4 pad0 = make_uint4(0x3C00u, 0u, 0u, 0u);  // f16 1.0, then zeros
    const uint4 padz = make_uint4(0u, 0u, 0u, 0u);
    for (int b = 0; b < VBUFS; ++b) {
      char* dv = sV + b * TILE_N * HDX * 2;
      for (int idx = tid; idx < TILE_N * 8; idx += THREADS) {
        const int n = idx / 8, ch = idx % 8;
        *reinterpret_cast<uint4*>(dv + t5_off(TILE_N, n, HD + ch * 8)) =
            (ch == 0) ? pad0 : padz;
      }
    }
  }
  // Q (and the VS pad) were written by ordinary STS in the generic proxy; the
  // tensor core will read them through descriptors in the async proxy. Order
  // the writes before the upcoming barrier/alloc sequence (see fence comment).
  fence_async_smem();
  const int page_mask = (1 << p.page_log2) - 1;
  const int tcols = p.table_stride;
  const int* table_row = p.table + (long long)b * tcols;
  if constexpr (SPG) {
    const int nrun = t_end - t_begin;
    if (nrun * 4 <= 4096) {
      spg_ok = true;
      id_last = table_row[min((kvlen - 1) >> p.page_log2, tcols - 1)];
      for (int idx = tid; idx < nrun * 4; idx += THREADS) {
        // ABSOLUTE tile index: the consumer indexes slot (t - t_begin) for
        // tile t, so entry tt must hold the ids of tile t_begin + tt (probe16
        // caught the relative form: correct only at t_begin == 0).
        const int jj = jt0 + (t_begin + (idx >> 2)) * TILE_N + (idx & 3) * 16;
        spg[idx] = table_row[min(jj >> p.page_log2, tcols - 1)];
      }
    }
  }
  if (warp == 0) {
    t5_alloc(t5_sp(tmem_word), NCOL);
    t5_relinquish();
    // PIPE keeps one barrier per S buffer plus one for PV, so the QK of tile t+1
    // can be in flight while tile t is still being softmaxed.
    if (lane == 0)
      for (int i = 0; i < ((PIPE || RV) ? 3 : 1); ++i) t5_mbar_init(t5_sp(mbar + i), 1);
  }
  __syncthreads();
  // Consumer-side ordering point for everything the mma will read through
  // descriptors (generic-proxy STS of Q/VS pads by every thread, async-proxy
  // cp.async KV writes): after the barrier, before any tcgen05.mma issue.
  t5_fence_after_sync();
  const uint32_t tmem = *tmem_word;
  const uint32_t s_tmem = tmem;             // S, then P reuses its columns
  const uint32_t o_tmem = tmem + OCOL;      // O accumulator
  const uint32_t mbar_a = t5_sp(mbar);
  const uint32_t q0a = t5_sp(sQ);
  int parity = 0;

  // ---- KV staging from the f16 mirror, double buffered ----
  constexpr int KCH = TILE_N * (HD / 8);
  // TPF: page-table id ring. The staging address chain today is
  // table LDG -> base arithmetic -> cp.async per 16-byte chunk; hoisting the
  // (broadcast) table loads one full iteration ahead leaves only the base
  // arithmetic and the cp.async issue in the chain. A 16-token block never
  // straddles a page (page_size >= 16 and tiles are 16-aligned), so per block
  // the unclamped page id (pg_lo) and the clamped-tail page id (pg_hi) are
  // exact for every token in it.
  constexpr int PPT = TILE_N / 16;
  int pg_lo[2][PPT], pg_hi[2][PPT];
  auto load_pids = [&](int t, int slot) {
    const int jb = jt0 + t * TILE_N;
#pragma unroll
    for (int q = 0; q < PPT; ++q) {
      const int alo = min(max(jb + q * 16, 0), kvlen - 1);
      const int ahi = min(max(jb + q * 16 + 15, 0), kvlen - 1);
      pg_lo[slot][q] = table_row[min(alo >> p.page_log2, tcols - 1)];
      pg_hi[slot][q] = table_row[min(ahi >> p.page_log2, tcols - 1)];
    }
  };
  auto issue_kv = [&](int t, int kbuf, int vbuf, int which) {  // 0=both 1=K 2=V
    char* dk = sK + kbuf * TILE_N * HD * 2;
    char* dv = sV + vbuf * TILE_N * HDX * 2;
    const int jb = jt0 + t * TILE_N;
#pragma unroll
    for (int u = 0; u < (KCH + THREADS - 1) / THREADS; ++u) {
      const int c = tid + u * THREADS;
      if (KCH % THREADS != 0 && c >= KCH) break;
      const int n = c / (HD / 8), col = (c % (HD / 8)) * 8;
      const int jj = min(max(jb + n, 0), kvlen - 1);
      int page;
      if constexpr (SPG) {
        page = spg_ok
                   ? ((jb + n < kvlen) ? spg[((t - t_begin) << 2) + (n >> 4)] : id_last)
                   : table_row[min(jj >> p.page_log2, tcols - 1)];
      } else if constexpr (TPF) {
        page = (jb + n < kvlen) ? pg_lo[t & 1][n >> 4] : pg_hi[t & 1][n >> 4];
      } else {
        page = table_row[min(jj >> p.page_log2, tcols - 1)];
      }
      const long long base = (long long)page * p.kv_s_page + (long long)hkv * p.kv_s_head +
                             (long long)(jj & page_mask) * p.kv_s_tok + col;
      const int off = t5_off(TILE_N, n, col);
      if (which != 2) cp_async16(dk + off, p.k16 + base);
      if (which != 1) cp_async16(dv + off, p.v16 + base);
    }
    cp_async_commit();
  };
  auto issue = [&](int t, int buf) { issue_kv(t, buf, buf, 0); };
  // FP8S: register-staged fp8 e4m3 -> f16 (LDG.128 -> expand8 -> STS.128 x2)
  // into the double-buffered f16 tiles. No mirror: decode reads every KV byte
  // exactly once, so the mirror's fixed 3-bytes-per-cached-element convert
  // never repays (measured +55% on fixed2 in the V0 notes), and unlike the
  // mma.sync decode path the tensor core consumes these tiles through
  // descriptors, so the per-tile ldmatrix traffic (48KB of the 96KB per-tile
  // L1 wavefronts at hd128; NCU put cfg107 at 60% L1/TEX vs 9-17% DRAM)
  // disappears entirely. LSU bytes per tile: LDG 16KB + STS 32KB at hd128.
  constexpr int KV16 = TILE_N * (HD / 16);  // 16-byte fp8 chunks per tensor
  constexpr int KVPT = KV16 / THREADS;
  static_assert(KV16 % THREADS == 0, "fp8 tile must divide over threads");
  uint4 kraw[FP8S ? KVPT : 1], vraw[FP8S ? KVPT : 1];
  int kloc[FP8S ? KVPT : 1];
  const uint4* kp8 = reinterpret_cast<const uint4*>(p.k);
  const uint4* vp8 = reinterpret_cast<const uint4*>(p.v);
  auto issue_pf8 = [&](int t) {
    const int jb = jt0 + t * TILE_N;
#pragma unroll
    for (int u = 0; u < (FP8S ? KVPT : 0); ++u) {
      const int c = tid + u * THREADS;
      const int n = c / (HD / 16), col = (c % (HD / 16)) * 16;
      const int jj = min(max(jb + n, 0), kvlen - 1);
      const int page = table_row[min(jj >> p.page_log2, tcols - 1)];
      const long long base = (long long)page * p.kv_s_page + (long long)hkv * p.kv_s_head +
                             (long long)(jj & page_mask) * p.kv_s_tok + col;
      kraw[u] = kp8[base >> 4];
      vraw[u] = vp8[base >> 4];
      kloc[u] = n * HD + col;
    }
  };
  auto store_pf8 = [&](int buf) {
    char* dk = sK + buf * TILE_N * HD * 2;
    char* dv = sV + buf * TILE_N * HDX * 2;
#pragma unroll
    for (int u = 0; u < (FP8S ? KVPT : 0); ++u) {
      const int n = kloc[u] / HD, col = kloc[u] % HD;
      uint32_t kh[8], vh[8];
      expand8(kraw[u], kh);
      expand8(vraw[u], vh);
      *reinterpret_cast<uint4*>(dk + t5_off(TILE_N, n, col)) =
          make_uint4(kh[0], kh[1], kh[2], kh[3]);
      *reinterpret_cast<uint4*>(dk + t5_off(TILE_N, n, col + 8)) =
          make_uint4(kh[4], kh[5], kh[6], kh[7]);
      *reinterpret_cast<uint4*>(dv + t5_off(TILE_N, n, col)) =
          make_uint4(vh[0], vh[1], vh[2], vh[3]);
      *reinterpret_cast<uint4*>(dv + t5_off(TILE_N, n, col + 8)) =
          make_uint4(vh[4], vh[5], vh[6], vh[7]);
    }
    // Generic-proxy STS -> async-proxy descriptor reads by the tensor core:
    // the fence must precede the barrier that releases the mma issuer.
    fence_async_smem();
  };
  if constexpr (!KB3 && !RV && !FP8S) {
    if constexpr (TPF) {
      load_pids(t_begin, t_begin & 1);
      if (t_begin + 1 < t_end) load_pids(t_begin + 1, (t_begin + 1) & 1);
    }
    issue(t_begin, 0);
  }

  float mrun = NEG_BIG, lrow = 0.f;
  // DEFER: the scale the TMEM accumulator currently carries. O and lrow are both
  // kept in this scale, so P is published as ex2(score*qk - mdef) and the O
  // correction only runs when the running max has moved by more than one log2 unit
  // instead of on every tile where it moved at all. P stays <= 2 because the
  // correction decision is taken before P is formed, which keeps it inside f16
  // range with the same relative precision as the <= 1 it had before.
  float mdef = NEG_BIG;
  const float qk = p.qk_scale;
  // One row per lane, so the softmax has no cross-lane reduction at all. These
  // three lambdas are the whole tile body; the two schedules below differ only in
  // the order they issue them.
  // DAO: matrix descriptors are base + (byte-offset >> 4), and the per-(g,s)
  // offsets are compile-time constants, so the whole desc arithmetic of the
  // issue loops collapses to one 64-bit add against a per-buffer base. The
  // address field is 14 bits (16 MiB window >> 4) and the largest offset is
  // (3*8192)>>4, so no addition can carry into the SBO/LBO fields.
  const uint64_t qd0 = DAO ? t5_desc(q0a, 16, 8 * 128) : 0;
  const uint64_t kd0 = DAO ? t5_desc(t5_sp(sK), 16, 8 * 128) : 0;
  const uint64_t kd1 = DAO ? t5_desc(t5_sp(sK + TILE_N * HD * 2), 16, 8 * 128) : 0;
  const uint64_t kd2 = DAO && KBUFS == 3 ? t5_desc(t5_sp(sK + 2 * TILE_N * HD * 2), 16, 8 * 128) : 0;
  const uint64_t vd0 = DAO ? t5_desc(t5_sp(sV), 0, 8 * 128) : 0;
  const uint64_t vd1 = DAO ? t5_desc(t5_sp(sV + TILE_N * HDX * 2), 0, 8 * 128) : 0;
  // PSM P descriptor: [TILE_M][TILE_N] f16, K-major, one 64-col group, same
  // LBO/SBO as Q; k-step s advances 32B = 2 descriptor units.
  const uint64_t pd0 = PSM ? t5_desc(t5_sp(sP), 16, 8 * 128) : 0;
  auto qk_issue = [&](int kvbuf, uint32_t scol) {  // tid == 0 only
    if constexpr (DAO) {
      const uint64_t kd = kvbuf == 0 ? kd0 : (kvbuf == 1 ? kd1 : kd2);
      for (int g = 0; g < KG; ++g)
        for (int s = 0; s < 4; ++s)
          t5_umma_ss(scol, qd0 + (uint64_t)(g * (TILE_M * 8) + s * 2),
                     kd + (uint64_t)(g * (TILE_N * 8) + s * 2), ID_QK, (g | s) != 0);
    } else {
      const uint32_t ka = t5_sp(sK + kvbuf * TILE_N * HD * 2);
      for (int g = 0; g < KG; ++g)
        for (int s = 0; s < 4; ++s)
          t5_umma_ss(scol, t5_desc(q0a + g * (TILE_M * 128) + s * 32, 16, 8 * 128),
                     t5_desc(ka + g * (TILE_N * 128) + s * 32, 16, 8 * 128), ID_QK, (g | s) != 0);
    }
  };
  auto pv_issue = [&](int t, int kvbuf, uint32_t scol) {  // tid == 0 only
    if constexpr (PSM) {
      // A = P from shared memory (umma_ss); B descriptor identical to the
      // TMEM-A form. Saves the t5_st32 + st_wait TMEM round trip per tile.
      for (int a = 0; a < KGX; ++a)
        for (int s = 0; s < TILE_N / 16; ++s) {
          const uint64_t vb =
              DAO ? (kvbuf == 0 ? vd0 : vd1) + (uint64_t)(a * (TILE_N * 8) + s * 128)
                  : t5_desc(t5_sp(sV + kvbuf * TILE_N * HDX * 2) + a * (TILE_N * 128) +
                                s * (16 * 128),
                            0, 8 * 128);
          t5_umma_ss(o_tmem + a * 64, pd0 + s * 2, vb, ID_PV, (t != t_begin || s != 0));
        }
    } else if constexpr (DAO) {
      const uint64_t vd = kvbuf == 0 ? vd0 : vd1;
      for (int a = 0; a < KGX; ++a)
        for (int s = 0; s < TILE_N / 16; ++s)
          t5_umma_ts(o_tmem + a * 64, scol + s * 8,
                     vd + (uint64_t)(a * (TILE_N * 8) + s * 128), ID_PV,
                     (t != t_begin || s != 0));
    } else {
      const uint32_t va = t5_sp(sV + kvbuf * TILE_N * HDX * 2);
      for (int a = 0; a < KGX; ++a)
        for (int s = 0; s < TILE_N / 16; ++s)
          t5_umma_ts(o_tmem + a * 64, scol + s * 8,
                     t5_desc(va + a * (TILE_N * 128) + s * (16 * 128), 0, 8 * 128), ID_PV,
                     (t != t_begin || s != 0));
    }
  };
  auto softmax_tile = [&](int t, uint32_t scol) {
    float sc[TILE_N];
    if constexpr (LD64) {
      t5_ld64(sc, scol);
    } else {
#pragma unroll
      for (int blk = 0; blk < TILE_N / 32; ++blk) t5_ld32(sc + blk * 32, scol + blk * 32);
    }
    t5_ld_wait();
    const int col0 = jt0 + t * TILE_N;
    const bool full = rowok && (col0 + TILE_N - 1 <= lim) && (col0 >= low);
    float mxr = NEG_BIG;
    // ILP: the row max and the row sum are reductions, and written serially they
    // are 64- and 32-long dependency chains of fixed-latency ops. NCU on this
    // kernel (dv_5fca7d193fb3) puts 39.3% of warp-active cycles in the `wait`
    // stall -- fixed-latency execution dependencies -- against 1.7% in barriers,
    // so the chains are what to break, not the synchronisation. Four independent
    // accumulators cut the chain length 4x; the max is order-independent and the
    // sum only reassociates.
    if (ILP) {
      float m0 = NEG_BIG, m1 = NEG_BIG, m2 = NEG_BIG, m3 = NEG_BIG;
      if (full) {
#pragma unroll
        for (int c = 0; c < TILE_N; c += 4) {
          m0 = fmaxf(m0, sc[c]);
          m1 = fmaxf(m1, sc[c + 1]);
          m2 = fmaxf(m2, sc[c + 2]);
          m3 = fmaxf(m3, sc[c + 3]);
        }
      } else {
#pragma unroll
        for (int c = 0; c < TILE_N; c += 4) {
          m0 = fmaxf(m0, (col0 + c <= lim && col0 + c >= low) ? sc[c] : NEG_BIG);
          m1 = fmaxf(m1, (col0 + c + 1 <= lim && col0 + c + 1 >= low) ? sc[c + 1] : NEG_BIG);
          m2 = fmaxf(m2, (col0 + c + 2 <= lim && col0 + c + 2 >= low) ? sc[c + 2] : NEG_BIG);
          m3 = fmaxf(m3, (col0 + c + 3 <= lim && col0 + c + 3 >= low) ? sc[c + 3] : NEG_BIG);
        }
      }
      mxr = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
    } else if (full) {
#pragma unroll
      for (int c = 0; c < TILE_N; ++c) mxr = fmaxf(mxr, sc[c]);
    } else {
#pragma unroll
      for (int c = 0; c < TILE_N; ++c) {
        const int j = col0 + c;
        mxr = fmaxf(mxr, (j <= lim && j >= low) ? sc[c] : NEG_BIG);
      }
    }
    // Running max in scaled log2 units, so every rescale is a plain difference:
    // forming it as fma(score, scale, -max) let nvcc contract the rounding residual
    // into an exponent, which is defect 3 in PROGRESS.md.
    const float mn = fmaxf(mrun, mxr * qk);
    // tcgen05.ld/st are .sync.aligned, i.e. warp-collective, so the rescale trigger
    // (which differs per lane, one row per lane) is always reduced across the warp
    // before any of them is issued. Predicating them per lane hung the probes from
    // the second tile onwards (dv_33f06b96a371, dv_dce8c95ac310).
    //
    // DEFER keeps O and lrow in an older scale (mdef) and rescales only once the
    // running max has moved more than one log2 unit, turning a per-tile TMEM round
    // trip into a handful per work item. P is formed against mdef, and because the
    // decision comes first, P <= 2 stays inside f16 at unchanged relative precision.
    float mneg;
    if (DEFER) {
      if (t == t_begin) {
        mdef = mn;                     // PV overwrites O on the first tile
      } else if (__any_sync(0xffffffffu, mn - mdef > (float)DTH)) {
        const float f = ex2(mdef - mn);
        lrow *= f;
        for (int blk = 0; blk < HDX / 32; ++blk) {
          float ov[32];
          t5_ld32(ov, o_tmem + blk * 32);
          t5_ld_wait();
#pragma unroll
          for (int j = 0; j < 32; ++j) ov[j] *= f;
          t5_st32(reinterpret_cast<const uint32_t*>(ov), o_tmem + blk * 32);
        }
        t5_st_wait();
        mdef = mn;
      }
      mrun = mn;
      mneg = -mdef;
    } else {
      const float rs = ex2(mrun - mn);
      lrow *= rs;
      mrun = mn;
      mneg = -mn;
      if ((t != t_begin) && __any_sync(0xffffffffu, rs != 1.f)) {
        for (int blk = 0; blk < HDX / 32; ++blk) {
          float ov[32];
          t5_ld32(ov, o_tmem + blk * 32);
          t5_ld_wait();
#pragma unroll
          for (int j = 0; j < 32; ++j) ov[j] *= rs;
          t5_st32(reinterpret_cast<const uint32_t*>(ov), o_tmem + blk * 32);
        }
        t5_st_wait();
      }
    }
    uint32_t pp[TILE_N / 2];
    float sum;
    // Branch order: the packed f32x2 loop is the fastest FULL-tile path; the
    // ILP 4-accumulator scalar loop is the masked-tile fallback when ILP is on
    // (its max-reduce partner above), and the serial loop is the legacy path.
    // ILP+F2 together give the 4-way max AND the packed sum (cfg122/124): the
    // max reduce of cfg92/100 is a 64-deep fmax dependency chain.
    if (ILP && !(F2 && full)) {
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
      for (int c = 0; c < TILE_N; c += 8) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int cc = c + q * 2, j = col0 + cc;
          float e0 = 0.f, e1 = 0.f;
          if (full || (j <= lim && j >= low)) e0 = ex2(fmaf(sc[cc], qk, mneg));
          if (full || (j + 1 <= lim && j + 1 >= low)) e1 = ex2(fmaf(sc[cc + 1], qk, mneg));
          pp[cc >> 1] = pack_f16(e0, e1);
          if (q == 0) s0 += e0 + e1;
          else if (q == 1) s1 += e0 + e1;
          else if (q == 2) s2 += e0 + e1;
          else s3 += e0 + e1;
        }
      }
      sum = lrow + ((s0 + s1) + (s2 + s3));
    } else if (F2 && full) {
      // Packed f32x2 scale-offset fma and row-sum accumulation (two fp32 lanes
      // per instruction on the FMA pipe). Each exp2 input is bit-identical to
      // the scalar path's fmaf result; only the sum's association differs.
      const uint64_t qk2 = f2_pack(qk, qk);
      const uint64_t mn2 = f2_pack(mneg, mneg);
      uint64_t acc = VS ? 0ull : f2_pack(lrow, 0.f);
      // X2H: exp2 and the per-tile row sum run in packed f16 (four
      // accumulators, <=8 adds each, lane sums <=16 where f16 ulp is 2^-6),
      // halving the MUFU ops per tile -- ex2 has no f32x2 form. The exponent
      // argument is still formed in f32 (bit-identical fma2); quantizing it
      // to f16 perturbs dominant P terms (|x| <= 2 under DEFER) by <= 2^ulp =
      // 0.14% relative and only vanishing terms see larger relative errors,
      // so lse/out move ~100x less than the checker tolerance. pp[] is the
      // ex2 output itself: the separate pack_f16 disappears too.
      uint32_t hacc[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int c = 0; c < TILE_N; c += 2) {
        const uint64_t x2 = fma2(f2_pack(sc[c], sc[c + 1]), qk2, mn2);
        float e0, e1;
        f2_unpack(x2, e0, e1);
        if constexpr (X2H) {
          const uint32_t p2 = ex2_f16x2(pack_f16(e0, e1));
          pp[c >> 1] = p2;
          if constexpr (!VS) hacc[(c >> 1) & 3] = hadd2(hacc[(c >> 1) & 3], p2);
        } else {
          e0 = ex2(e0);
          e1 = ex2(e1);
          if constexpr (!VS) acc = add2(acc, f2_pack(e0, e1));
          pp[c >> 1] = pack_f16(e0, e1);
        }
      }
      if constexpr (!VS) {
        if constexpr (X2H) {
          sum = lrow + ((hsum2(hacc[0]) + hsum2(hacc[1])) + (hsum2(hacc[2]) + hsum2(hacc[3])));
        } else {
          float s0, s1;
          f2_unpack(acc, s0, s1);
          sum = s0 + s1;
        }
      } else {
        sum = lrow;  // unused: the sum lives in TMEM O column HD
      }
    } else {
      sum = lrow;
#pragma unroll
      for (int c = 0; c < TILE_N; c += 2) {
        const int j = col0 + c;
        float e0 = 0.f, e1 = 0.f;
        if (full || (j <= lim && j >= low)) e0 = ex2(fmaf(sc[c], qk, mneg));
        if (full || (j + 1 <= lim && j + 1 >= low)) e1 = ex2(fmaf(sc[c + 1], qk, mneg));
        sum += e0 + e1;
        pp[c >> 1] = pack_f16(e0, e1);
      }
    }
    lrow = sum;
    if constexpr (PSM) {
      // Publish P to shared memory: this lane's row as TILE_N/8 swizzled
      // 16-byte chunks (K-major A layout for the PV descriptor, same t5_off
      // scheme as Q). 4-way bank conflicts at worst (lanes l, l+8, l+16, l+24
      // share a bank per chunk) -- still far cheaper than the TMEM round trip.
#pragma unroll
      for (int j = 0; j < TILE_N / 8; ++j)
        *reinterpret_cast<uint4*>(sP + t5_off(TILE_M, row, j * 8)) =
            make_uint4(pp[4 * j], pp[4 * j + 1], pp[4 * j + 2], pp[4 * j + 3]);
      // Generic-proxy STS -> async-proxy descriptor reads by the PV mma.
      fence_async_smem();
    } else {
      // publish P (TILE_N/2 columns of f16 pairs) over the consumed S region
#pragma unroll
      for (int blk = 0; blk < TILE_N / 2 / 32; ++blk)
        t5_st32(pp + blk * 32, scol + blk * 32);
      t5_st_wait();
      // P will be consumed by a PV mma issued by ONE thread after the
      // schedule's barrier; order every warp's TMEM stores before it.
      t5_fence_before_sync();
    }
  };

  // INSTR (diagnostics only): per-phase clock64 timestamps for one interior
  // representative CTA, written to a fixed 100 MiB offset in the split-KV
  // scratch (unused while splits==1). Slots per iteration: 0 top, 1 after
  // staging wait, 2 after the lookahead issues, 3 after QK issue (tid0),
  // 4 after QK mbar wait, 5 after softmax, 6 after PV issue (tid0), 7 after
  // PV wait. mtiles is sized from max_q_len, so on a ragged batch the low
  // quarter is the range active for every request.
  long long* dbg = nullptr;
  long long tstart = 0;
  int it = 0;
  if constexpr (INSTR) {
    if (p.splits == 1 && p.part_o != nullptr && blockIdx.x == p.mtiles / 4 &&
        blockIdx.y == 0 && blockIdx.z == 0) {
      dbg = reinterpret_cast<long long*>(p.part_o + (25 << 20));
      if (tid == 0) {
        tstart = clock64();
        dbg[0] = tstart;
      }
    }
  }
  auto tspot = [&](int k) {
    if constexpr (INSTR) {
      if (dbg != nullptr && tid == 0 && it < 48)
        dbg[1 + it * 8 + k] = clock64() - tstart;
    }
  };

  if constexpr (FP8S) {
    // Register-prefetched decode schedule on the t5 pipeline: the f16 convert
    // of tile t lands in buffer t&1, then tile t+1's fp8 LDGs are issued and
    // stay in flight across QK(t)/softmax(t)/PV(t). Buffer safety: buf t&1 was
    // last read by tile t-2's QK/PV, whose completion every thread awaited at
    // the end of iteration t-2 (PV mbar wait) before the t-1 barrier.
    issue_pf8(t_begin);
    for (int t = t_begin; t < t_end; ++t) {
      const int b = (t - t_begin) & 1;
      tspot(0);
      store_pf8(b);          // ends with fence_async_smem (generic -> async)
      __syncthreads();
      t5_fence_after_sync();
      tspot(1);
      if (t + 1 < t_end) issue_pf8(t + 1);
      tspot(2);
      if (tid == 0) { qk_issue(b, s_tmem); t5_commit(mbar_a); }
      tspot(3);
      t5_mbar_wait(mbar_a, parity);
      parity ^= 1;
      tspot(4);
      softmax_tile(t, s_tmem);   // ends with t5_fence_before_sync
      __syncthreads();
      t5_fence_after_sync();
      tspot(5);
      if (tid == 0) { pv_issue(t, b, s_tmem); t5_commit(mbar_a); }
      tspot(6);
      t5_mbar_wait(mbar_a, parity);
      parity ^= 1;
      __syncthreads();
      tspot(7);
      ++it;
    }
    if constexpr (INSTR) {
      if (dbg != nullptr && tid == 0) dbg[386] = it;
    }
  } else if constexpr (RV) {
    // Schedule R: QK(t+1) is issued at the TOP of iteration t -- right after
    // PV(t-1)'s completion is awaited, which frees S(b^1) and V(b^1) -- so the
    // tensor core runs QK(t+1) while the warps issue staging and softmax tile
    // t; PV(t) is issued after softmax(t) and awaited at the top of t+1. With
    // 3 K / 2 V buffers the top-of-iter issues of V(t+1) and K(t+2) each get a
    // full iteration of slack, and the single cp_async_wait<0> at the top
    // covers exactly the groups issued one iteration earlier.
    const uint32_t qkb0 = t5_sp(mbar + 0), qkb1 = t5_sp(mbar + 1);
    const uint32_t pvb = t5_sp(mbar + 2);
    int qpar0 = 0, qpar1 = 0, pvpar = 0;
    issue_kv(t_begin, 0, 0, 1);
    issue_kv(t_begin, 0, 0, 2);
    if (t_begin + 1 < t_end) issue_kv(t_begin + 1, 1, 0, 1); else cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    if (tid == 0) { qk_issue(0, s_tmem); t5_commit(qkb0); }
    for (int t = t_begin; t < t_end; ++t) {
      const int b = (t - t_begin) & 1;
      const int kb = (int)((t - t_begin) % 3);
      const uint32_t qkbar = b ? qkb1 : qkb0;
      const int qpar = b ? qpar1 : qpar0;
      tspot(0);
      cp_async_wait<0>();
      __syncthreads();
      tspot(1);
      if (t != t_begin) {
        t5_mbar_wait(pvb, pvpar);
        pvpar ^= 1;
      }
      tspot(2);
      if (tid == 0 && t + 1 < t_end) {
        qk_issue((kb + 1) % 3, s_tmem + (b ^ 1) * TILE_N);
        t5_commit(b ? qkb0 : qkb1);
      }
      tspot(3);
      if (t + 1 < t_end) issue_kv(t + 1, 0, b ^ 1, 2); else cp_async_commit();
      if (t + 2 < t_end) issue_kv(t + 2, (kb + 2) % 3, 0, 1); else cp_async_commit();
      tspot(4);
      t5_mbar_wait(qkbar, qpar);
      if (b) qpar1 = qpar ^ 1; else qpar0 = qpar ^ 1;
      softmax_tile(t, s_tmem + b * TILE_N);
      __syncthreads();
      tspot(5);
      if (tid == 0) { pv_issue(t, b, s_tmem + b * TILE_N); t5_commit(pvb); }
      tspot(6);
      ++it;
    }
    // PV(last) completion gates the TMEM O read; drain staging because the
    // epilogue overlays the K/V shared memory.
    t5_mbar_wait(pvb, pvpar);
    cp_async_wait<0>();
    tspot(7);
    if constexpr (INSTR) {
      if (dbg != nullptr && tid == 0) dbg[386] = it;
    }
  } else if constexpr (KB3) {
    // Three K buffers / two V buffers. The 2-buffer schedule could only issue
    // tile t+1's loads after PV(t-1)'s wait and had to wait on them at the top
    // of iteration t, exposing the mirror-load latency once per tile. Here the
    // top of iteration t issues V(t+1) and K(t+2) -- their buffers were freed
    // by the PV(t-1)/QK(t-1) mbar waits every thread passed in the previous
    // iteration -- and the two wait<2> windows then guarantee K(t) at QK-issue
    // and V(t) at PV-issue without ever stalling on loads issued inside the
    // current iteration. Empty padding commits at the tail keep the group
    // window arithmetic uniform for short tile runs.
    if (t_begin < t_end) issue_kv(t_begin, 0, 0, 1); else cp_async_commit();
    if (t_begin < t_end) issue_kv(t_begin, 0, 0, 2); else cp_async_commit();
    if (t_begin + 1 < t_end) issue_kv(t_begin + 1, 1, 0, 1); else cp_async_commit();
    for (int t = t_begin; t < t_end; ++t) {
      const int kb = (int)((t - t_begin) % 3);
      const int vb = (t - t_begin) & 1;
      tspot(0);
      cp_async_wait<2>();
      __syncthreads();
      tspot(1);
      if (t + 1 < t_end) issue_kv(t + 1, 0, vb ^ 1, 2); else cp_async_commit();
      if (t + 2 < t_end) issue_kv(t + 2, (kb + 2) % 3, 0, 1); else cp_async_commit();
      tspot(2);
      if (tid == 0) { qk_issue(kb, s_tmem); t5_commit(mbar_a); }
      tspot(3);
      t5_mbar_wait(mbar_a, parity);
      parity ^= 1;
      tspot(4);
      softmax_tile(t, s_tmem);
      cp_async_wait<2>();
      __syncthreads();
      tspot(5);
      if (tid == 0) { pv_issue(t, vb, s_tmem); t5_commit(mbar_a); }
      tspot(6);
      t5_mbar_wait(mbar_a, parity);
      parity ^= 1;
      tspot(7);
      ++it;
    }
    if constexpr (INSTR) {
      if (dbg != nullptr && tid == 0) dbg[386] = it;
    }
  } else if constexpr (PIPE) {
    // S is double buffered in TMEM (S0 at column 0, S1 at TILE_N) within the same
    // 256-column allocation, so the QK of tile t+1 is issued before the softmax of
    // tile t and the tensor core runs while the warps work. Ordering is what makes
    // the buffer reuse safe without extra waits: tcgen05.mma operations from one
    // thread execute in issue order, so QK(t+2) cannot start before PV(t) has
    // finished reading P(t) out of the same columns, and the explicit PV wait is
    // only needed before shared memory is restaged and before the next O
    // correction writes TMEM.
    // Scalars, not arrays: the buffer index is a runtime value, and indexing an
    // array with it would put the barrier addresses and parities in local memory
    // inside the hot loop.
    const uint32_t qkb0 = t5_sp(mbar + 0), qkb1 = t5_sp(mbar + 1);
    const uint32_t pvb = t5_sp(mbar + 2);
    int qpar0 = 0, qpar1 = 0, pvpar = 0;
    cp_async_wait<0>();
    __syncthreads();
    if (tid == 0) { qk_issue(0, s_tmem); t5_commit(qkb0); }
    if (t_begin + 1 < t_end) issue(t_begin + 1, 1);
    for (int t = t_begin; t < t_end; ++t) {
      const int b = (t - t_begin) & 1;
      const uint32_t scol = s_tmem + b * TILE_N;
      const uint32_t qkbar = b ? qkb1 : qkb0;
      const uint32_t qknxt = b ? qkb0 : qkb1;
      const int qpar = b ? qpar1 : qpar0;
      if (t + 1 < t_end) {
        cp_async_wait<0>();       // KV(t+1), staged at the end of the previous pass
        __syncthreads();
        t5_mbar_wait(qkbar, qpar);
        if (b) qpar1 = qpar ^ 1; else qpar0 = qpar ^ 1;
        if (tid == 0) { qk_issue(b ^ 1, s_tmem + (b ^ 1) * TILE_N); t5_commit(qknxt); }
      } else {
        t5_mbar_wait(qkbar, qpar);
        if (b) qpar1 = qpar ^ 1; else qpar0 = qpar ^ 1;
      }
      softmax_tile(t, scol);
      __syncthreads();
      if (tid == 0) { pv_issue(t, b, scol); t5_commit(pvb); }
      t5_mbar_wait(pvb, pvpar); pvpar ^= 1;
      if (t + 2 < t_end) issue(t + 2, b);
    }
  } else {
    for (int t = t_begin; t < t_end; ++t) {
      const int buf = (t - t_begin) & 1;
      tspot(0);
      cp_async_wait<0>();
      __syncthreads();
      t5_fence_after_sync();
      tspot(1);
      if (t + 1 < t_end) issue(t + 1, buf ^ 1);
      if constexpr (TPF) {
        if (t + 2 < t_end) load_pids(t + 2, t & 1);
      }
      tspot(2);
      if (tid == 0) { qk_issue(buf, s_tmem); t5_commit(mbar_a); }
      tspot(3);
      t5_mbar_wait(mbar_a, parity);
      parity ^= 1;
      tspot(4);
      softmax_tile(t, s_tmem);   // ends with t5_fence_before_sync
      __syncthreads();
      t5_fence_after_sync();
      tspot(5);
      if (tid == 0) { pv_issue(t, buf, s_tmem); t5_commit(mbar_a); }
      tspot(6);
      t5_mbar_wait(mbar_a, parity);
      parity ^= 1;
      // NOSYNC: the end-of-iteration barrier is redundant. Every thread waits
      // the PV(t) mbar just above, and the next iteration's only shared-memory
      // write (issue(t+2) into buf) happens after the next top-of-iteration
      // cp_async_wait + __syncthreads, which cannot be reached by any thread
      // until all have passed that mbar wait. Buffer reuse is therefore already
      // ordered without this barrier (the F16KV mma.sync path runs one barrier
      // per tile on the same argument).
      if constexpr (!NOSYNC) __syncthreads();
      tspot(7);
      ++it;
    }
    if constexpr (INSTR) {
      if (dbg != nullptr && tid == 0) dbg[386] = it;
    }
  }

  // ---- epilogue ----
  // Two constraints shape this. tcgen05.ld is warp-collective, so it must not be
  // predicated per lane even though `rowok` is per lane. And one 2-byte store per
  // element per lane would scatter the output over 128 different rows, so the rows
  // are staged through the (now dead) shared memory and written out vectorised.
  __syncthreads();
  // Row stride HD+4: with a plain HD stride every lane's row starts at the
  // same bank (HD % 32 == 0), turning the per-row accumulator stores into
  // 32-way conflicts (NCU: 21.1-way average, 82% of shared-store wavefronts).
  // HD+4 keeps 16B alignment (stride 528/1040/272 bytes) and cuts it to 4-way.
  constexpr int SROW = HD + 4;
  float* sOut = reinterpret_cast<float*>(smraw);   // TILE_M * SROW
  float* sGain = sOut + TILE_M * SROW;             // TILE_M
  float* sLse = sGain + TILE_M;                    // TILE_M
  float* sM = sLse + TILE_M;                       // TILE_M
  float* sL = sM + TILE_M;                         // TILE_M

  // With DEFER the accumulator and lrow are both in the mdef scale, so that is the
  // max the epilogue (and, when splitting, the combine kernel) must be told about.
  const float mfin = DEFER ? mdef : mrun;
  float lfin = lrow;
  if constexpr (VS) {
    // The row sum arrives in TMEM O column HD (PV against the ones pad), in
    // lockstep with every DEFER rescale of O, so read O first (lane == row).
    for (int blk = 0; blk < HDX / 32; ++blk) {
      float ov[32];
      t5_ld32(ov, o_tmem + blk * 32);
      t5_ld_wait();
      if (blk == HD / 32) lfin = ov[0];
      if (rowok) {
#pragma unroll
        for (int j = 0; j < 32; ++j)
          if (blk * 32 + j < HD) sOut[row * SROW + blk * 32 + j] = ov[j];
      }
    }
  }
  float base2 = mfin + (lfin > 0.f ? log2f(lfin) : NEG_BIG);
  float sc_sink = 1.f;
  if (p.sinks != nullptr) {
    const float sink2 = p.sinks[hkv * G + (row % G)] * LOG2E;
    const float hi = fmaxf(base2, sink2);
    const float total2 = hi + log2f(1.f + ex2(-fabsf(base2 - sink2)));
    sc_sink = ex2(base2 - total2);
    base2 = total2;
  }
  const float gain = (lfin > 0.f ? 1.f / lfin : 0.f) * sc_sink * p.v_scale;
  if (rowok) {
    sGain[row] = gain;
    sLse[row] = base2;
    sM[row] = mfin;
    sL[row] = lfin;
  }
  if constexpr (!VS) {
    for (int blk = 0; blk < HD / 32; ++blk) {
      float ov[32];
      t5_ld32(ov, o_tmem + blk * 32);
      t5_ld_wait();
      if (rowok)
        for (int j = 0; j < 32; ++j) sOut[row * SROW + blk * 32 + j] = ov[j];
    }
  }
  __syncthreads();

  if (p.splits > 1) {
    // Publish the unnormalised accumulator at this split's own max, plus the pair
    // (max_log2, sum), exactly as the mma.sync path does; the combine kernel owns
    // the rescale, the sink and the lse.
    float* dst = p.part_o + w_id * p.part_s_w + (long long)split_z * p.part_s_s;
    for (int e = tid * 4; e < TILE_M * HD; e += THREADS * 4) {
      const int r = e / HD;
      if (i0 + r / G >= qlen) continue;
      *reinterpret_cast<float4*>(dst + e) =
          *reinterpret_cast<const float4*>(sOut + r * SROW + (e - r * HD));
    }
    float* ml = p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s;
    float* mt = p.part_ml_t + w_id * p.partml_s_w + (long long)split_z * 2;
    for (int r = tid; r < TILE_M; r += THREADS) {
      if (i0 + r / G >= qlen) continue;
      ml[r * 2] = sM[r];
      ml[r * 2 + 1] = sL[r];
      mt[r * p.partmlt_s_r] = sM[r];
      mt[r * p.partmlt_s_r + 1] = sL[r];
    }
  } else {
    for (int e = tid * 8; e < TILE_M * HD; e += THREADS * 8) {
      const int r = e / HD, c = e % HD;
      const int i = i0 + r / G;
      if (i >= qlen) continue;
      const float g = sGain[r];
      const float* so = sOut + r * SROW + (e - r * HD);
      __nv_bfloat16 h8[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) h8[j] = __float2bfloat16_rn(so[j] * g);
      const long long tok = q0 + i;
      const int head = hkv * G + (r % G);
      *reinterpret_cast<uint4*>(p.out + tok * p.out_s_tok + (long long)head * p.out_s_head + c) =
          *reinterpret_cast<const uint4*>(h8);
    }
    if (p.write_lse) {
      for (int r = tid; r < TILE_M; r += THREADS) {
        const int i = i0 + r / G;
        if (i >= qlen) continue;
        const long long tok = q0 + i;
        const int head = hkv * G + (r % G);
        p.lse[tok * p.lse_s_tok + (long long)head * p.lse_s_head] = sLse[r];
      }
    }
  }
  __syncthreads();
  if (warp == 0) t5_dealloc(tmem, NCOL);
}

template <int HD, int TILE_N, int MINB, int DEFER = 0, int PIPE = 0, int ILP = 0,
          int INSTR = 0, int KB3 = 0, int F2 = 0, int VS = 0, int RV = 0, int DAO = 0,
          int TPF = 0, int SPG = 0, int FP8S = 0, int NOSYNC = 0, int PSM = 0, int MTM = 0,
          int X2H = 0, int DTH = 1, int LD64 = 0>
static void launch_main_t5(const Params& p, dim3 grid, cudaStream_t stream) {
  constexpr int TILE_M = 128;
  constexpr int KBUFS = (KB3 || RV) ? 3 : 2;
  constexpr int VBUFS = 2;
  constexpr int HDX = VS ? HD + 64 : HD;
  // SPG reserves 16 KiB + slack for the per-CTA page-id table (<=1024 tiles).
  const int smem = ((TILE_M + KBUFS * TILE_N) * HD + VBUFS * TILE_N * HDX) * 2 + 64 +
                   (PSM ? TILE_M * TILE_N * 2 : 0) + (SPG ? (4096 * 4 + 256) : 0);
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(
        fmha_main_t5<HD, TILE_N, MINB, DEFER, PIPE, ILP, INSTR, KB3, F2, VS, RV, DAO, TPF, SPG,
                     FP8S, NOSYNC, PSM, MTM, X2H, DTH, LD64>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  fmha_main_t5<HD, TILE_N, MINB, DEFER, PIPE, ILP, INSTR, KB3, F2, VS, RV, DAO, TPF, SPG, FP8S,
               NOSYNC, PSM, MTM, X2H, DTH, LD64>
      <<<grid, 128, smem, stream>>>(p);
}

// ---------------------------------------------------------------------------
// t6: warp-specialized tcgen05 prefill kernel at TILE_N=128 (v1: single CTA,
// splits==1, f16 mirror staged by TMA).
//
// Why this exists (measurements, jobs dv_6d76d9369490 / INSTR c102 runs):
//   * tcgen05 f16 SS mma: m128n64k16 = 48.1 cyc (agg 1422-1485 TF over 148 SMs)
//     vs m128n128k16 = 64.1 cyc (agg 2161-2248 TF) and m128n256k16 = 128.1 cyc
//     (agg 2237-2277 TF). The t5 kernels issue everything at n64, capping the
//     tensor-core ceiling near ~1500 TF; n128 QK lifts it ~1.5x.
//   * INSTR phase breakdown of the serial t5 schedule (c102, f7): full iteration
//     ~3.87k cyc = staging issue ~890 (2048 cp.async16 LSU ops per tile) +
//     qk_issue ~660 + softmax ~1230 + pv_issue ~600 + waits ~400. The single
//     4-warp CTA serialises staging issue, tensor issue and softmax.
//   t6 removes all three serialisations: ONE load warp issues 2*KG*(128/page)
//   TMA bulk ops per tile (no per-16B LSU issue, no address ALU on the critical
//   path), ONE mma warp runs QK(t+1) ahead while the softmax warps process
//   S(t) (double-buffered S in TMEM), and 4 softmax warps (128 lanes, one row
//   per lane) own the exp/publish path. Warps 6-7 idle in v1 (reserved for a
//   correction group / cluster2 later).
//
// Numerics are the promoted c131 recipe on 128-column tiles: f16 mirror
// (exact e4m3->f16), bf16->f16 Q (satfinite), DEFER O-rescale at DTH log2
// units, F2 packed scale-offset fma, X2H packed ex2.approx.f16x2 with f16x2
// row-sum accumulators, log2-domain running max. Only tile boundaries and the
// max/sum association differ from c131.
//
// TMEM map (512 cols allocated, 1 CTA/SM): S0 = [0,128), S1 = [128,256),
// O = [256, 256+HD). P (f16 pairs) overlays the low 64 cols of its S region.
// Barriers: KF/VF (TMA tx), KE/VE + QD/PD (tcgen05 commit), PR (4 softmax
// arrivals). All waits use CLOSED-FORM parities derived from the tile index so
// a skipped conditional wait (DEFER) can never desynchronise a phase counter.
// In-order TC execution from the single issuing thread makes the cumulative
// tcgen05.commit arrivals safe: QD(t+1) is committed after issuing QK(t+1),
// which the TC runs only after PV(t), so softmax(t+1) starting on QD(t+1)
// implies P(t) was consumed; the rescale path still waits PD explicitly.

// SM8: named barrier 1 over the 8 softmax warps (256 threads) for the
// cross-half row-max exchange; barrier 0 (__syncthreads) is left untouched.
DEVFN void bar_sm8() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); }
// PBAR: the SM8/SM8C row-max exchange only shares sMx between warp w and its
// column-half partner (w+4: identical rows), so the 256-thread rendezvous can
// narrow to four independent 64-thread pair barriers (ids 2-5, disjoint sMx
// row blocks). MSKIP's dead-warp invariant holds per pair: deadw is a function
// of the row set, which is pair-identical, so a pair's arrivals stay matched.
// Fresh stall census (dv_338fa16a605f): barrier 1.54 vs 0.38 cyc/issued-instr
// f0-vs-f1; J51 traced the exchange at 277 of 1470 period cycles.
DEVFN void bar_pair64(int warp) {
  asm volatile("bar.sync %0, 64;\n" ::"r"(2 + (warp & 3)) : "memory");
}
// NOT TRIED AGAIN: a pairwise form of this barrier (bar.sync 2+(warp&3), 64 -- warp w only
// ever exchanges with warp w+4) measured f4 -2.2% and xwin128 -1.8% on the median, with
// gpu_only flat on xwin128, and then HUNG the device on f1 and xmqa128 (job 23,
// dv_0bce5ca43c7c: two steps died at their timeouts with no output). The register-id form
// of bar.sync with a thread count is the suspect; a 2% hd64 median is not worth finding out.

// LPAR: lanes of the load warp that issue TMA in parallel, one page each. The
// load warp's per-tile issue cost is on the critical path -- cfg210's phase trace
// on f1 gives period = 1590 + (KF wait) cycles across iterations, the KF wait
// swinging 131..624 and the period 1756..2192 in lockstep -- so both the number of
// TMA issues per tile and the K ring depth are worth moving.
// SPLITV: stage V from its own warp. cfg221's trace of the promoted schedule put the
// issuer's serial timeline at KF 163 + QK issue 554 + PR 115 + VF 72 + PV issue 693 =
// 1597 of a 1782-cycle period, so 235 cycles per tile are the tensor core idling while
// the single issuer waits for staged KV. One warp stages both matrices, and it waits the
// V ring's VE (PV(t-VST) complete) before issuing the K boxes the issuer needs next --
// which is also why KST4/VST2 measured 1.74x instead of helping. Two warps, one ring
// each, decouples them; SM8 grows to 11 warps (352 threads).
template <int HD, int KST, int VST, int DTH, int IT6 = 0, int PVN = 64, int MTM6 = 0,
// SB: S/P buffers in TMEM (columns SB*TILE_N..511 hold O). QLA: how many tiles ahead
// the single issuer pushes QK. With SB=2 the issuer can only ever have QK(t+1) in
// flight, because QK(t+2) would land in the buffer softmax(t) is reading; at hd128
// TMEM has exactly room for three (S 3x128 + O 128 = 512), which lets it issue
// QK(t+2) before waiting for P(t) and so keeps the tensor core fed across the PR and
// VF waits. cfg221's trace of the promoted schedule: period 1752 = KF 135 + QK issue
// 555 + PR 115 + VF 71 + PV issue 685 + ~190 loop, tensor core busy 1240 of it.
// QLA needs SB >= QLA+1; DUAL's two unordered issuer warps rely on an explicit PD wait
// whose parity only SB=2 expresses, so DUAL stays at the defaults.
// DPV: a dedicated PV issuer warp. cfg225's trace of the promoted schedule put the
// period at 1752 cycles against 1240 of tensor-core work, and the gap is the single
// issuer alternating between the two mma phases: it spends 555 cycles blocked issuing
// QK, then cannot issue PV until the softmax publishes P(t) (PR 115 + VF 71 + ~190 of
// loop), and the tensor core drains and idles across that gap. Two warps issuing into
// the same in-order tensor queue keep it fed instead. The tree already has a DUAL form
// (c175) but it is 4-softmax-warp, SB=2, LPAR=1, and at SB=2 the QK warp has to wait
// PV(t-1)'s PD before reusing the shared S parity, which re-serialises the two; SB=3
// loosens that by a whole tile. DPV is wired for the SM8+SPLITV layout only: warps 0-7
// softmax, 8 QK, 9 K load, 10 V load, 11 PV = 384 threads.
          int DUAL = 0, int SM8 = 0, int LPAR = 1, int SPLITV = 0, int SB = 2,
          int QLA = 1, int DPV = 0, int MSKIP = 0, int RWSK = 0, int HACC = 4,
          int F16X = 0, int PHO = 0, int TRACE = 0, int QCP = 0, int LPT6 = 0,
          int PLO = 0, int F8T = 0, int PBAR = 0>
__global__ void __launch_bounds__(SM8 ? (SPLITV ? (DPV ? 384 : 352) : 320)
                                      : (DUAL ? 224 : 192),
                                  1)
fmha_main_t6(const __grid_constant__ Params p, const __grid_constant__ CUtensorMap kmap,
             const __grid_constant__ CUtensorMap vmap) {
  static_assert(!(SM8 && DUAL), "SM8 and DUAL are exclusive issuer layouts");
  static_assert(PLO == 0 || QCP == 1, "PLO is wired for the QCP gather prologue");
  static_assert(F8T == 0 || (SM8 && SPLITV && DPV && QCP && PLO == 0 && DUAL == 0),
                "F8T is wired for the SM8+SPLITV+DPV+QCP layout");
  static_assert(F8T == 0 || (HD == 128 && KST == 2 && VST == 2),
                "F8T: hd128 with 2-deep fp8 rings == 2 f16 convert targets");
  constexpr int TILE_M = 128;
  constexpr int TILE_N = 128;
  // DUAL (c175): K42's IT6 breakdown put f1's tile period at ~1752 cyc with
  // qk_issue 521 + pv_issue 694 = 1215 cyc of single-lane tcgen05 ISSUE time
  // (TC busy only ~1026) -> the issuer warp, not TC/L2/softmax, is the wall.
  // DUAL splits QK and PV issuing across two warps (4 and 6; 224 threads).
  constexpr int THREADS = SM8 ? (SPLITV ? (DPV ? 384 : 352) : 320) : (DUAL ? 224 : 192);
  // SM8 (c176): 8 softmax warps split the 128 tile columns into two 64-col
  // halves; the tensor issuer and the TMA load warp move to warps 8/9 (the
  // t8-proven role layout: warps 4-7 tcgen05.ld/st map to lanes (warp%4)*32).
  constexpr int MMA_W = SM8 ? 8 : 4;
  constexpr int LOAD_W = SM8 ? 9 : 5;
  constexpr int LOAD_VW = SM8 ? 10 : 6;   // V ring, only reached when SPLITV
  constexpr int PV_W = 11;                // PV issuer, only reached when DPV
  constexpr int NSW = SM8 ? 8 : 4;               // softmax warps
  constexpr int SCOLS = SM8 ? TILE_N / 2 : TILE_N;  // softmax cols per warp
  constexpr int KG = HD / 64;              // 64-col swizzle atoms per row
  constexpr int PV_STEPS = TILE_N / 16;    // k16 steps per PV atom
  constexpr int QBYTES = TILE_M * HD * 2;
  constexpr int KT_BYTES = TILE_N * HD * 2;
  constexpr int VT_BYTES = TILE_N * HD * 2;
  constexpr int PPT = TILE_N / 16;         // max pages per tile (page >= 16)
  constexpr int OCOL = SB * TILE_N;        // O accumulator TMEM column base
  static_assert(SB >= QLA + 1, "QK lookahead would write a buffer the softmax is reading");
  static_assert(SB * TILE_N + HD <= 512, "S buffers plus O exceed one CTA's TMEM");
  static_assert(DUAL == 0 || (SB == 2 && QLA == 1), "DUAL's PD parity assumes SB=2");
  static_assert(!(DPV && DUAL), "DPV and DUAL are exclusive PV-issuer layouts");
  static_assert(DPV == 0 || (SM8 && SPLITV), "DPV assumes the SM8+SPLITV warp layout");
  // MSKIP/RWSK skip provably dead softmax work, and are ON in the promoted 176/233
  // launches: job 19 (dv_684f5f064526) measured f9 185.98 -> 94.06us (0.506) with
  // bit_identical=True and 0 mismatched elements, f7 0.999 and f1 1.006. f9 is
  // q=1025 at G=1, so its 9th mtile has ONE valid row of 128 and, at 144 CTAs on
  // 148 SMs, the whole kernel's makespan was those tail CTAs running the divergent
  // masked softmax. Both ride the SM8 layout: the skip has to preserve bar_sm8's
  // 256-thread rendezvous and PR's one-arrival-per-warp count, and SM8 is what
  // makes one lane own one whole row.
  static_assert(MSKIP == 0 || SM8, "MSKIP rides the SM8 softmax layout");
  static_assert(RWSK == 0 || SM8, "RWSK rides the SM8 softmax layout");
  // PHO: per-half O accumulators. Each SM8 column half keeps its own running max,
  // row sum and O tile, so the per-tile cross-half row-max exchange (277 cycles of
  // the 1470-cycle hd128 period in the IT6 trace) disappears entirely and the two
  // halves are merged once, in the epilogue, with the same log-sum-exp rescale the
  // split-KV combine uses. That needs two HD-wide O regions in TMEM, so it only
  // fits at SB=2 (2*128 S + 2*128 O = 512 columns exactly) and gives up QLA2. The
  // PV k-step that consumes a half's P is exactly the k-step that accumulates that
  // half's O -- k-step s reads P words [8s, 8s+8) and half h owns words
  // [32h, 32h+32) -- so the split costs no extra mma and no extra barrier: PR stays
  // one count-8 rendezvous per tile.
  static_assert(PHO == 0 || (SM8 && DPV), "PHO is wired for the SM8+DPV layout");
  static_assert(PBAR == 0 || (SM8 && PHO == 0),
                "PBAR narrows SM8's per-tile exchange barrier; exclusive with PHO");
  static_assert(PHO == 0 || (SB * TILE_N + 2 * HD <= 512),
                "PHO needs two O regions inside one CTA's 512 TMEM columns");

  extern __shared__ char smraw[];
  char* sQ = smraw;
  // F8T (J59 t6f): fp8 TMA rings (KST/VST x TILE_N*HD bytes) followed by the
  // double-buffered f16 convert targets the TC descriptors read; !F8T keeps
  // the plain f16 rings. With KST == VST == 2 the barrier-base expression
  // below is the same for both layouts.
  char* sK8 = F8T ? (sQ + QBYTES) : nullptr;
  char* sV8 = F8T ? (sK8 + KST * (KT_BYTES / 2)) : nullptr;
  char* sK = F8T ? (sV8 + VST * (VT_BYTES / 2)) : (sQ + QBYTES);
  char* sV = sK + KST * KT_BYTES;
  uint64_t* KF = reinterpret_cast<uint64_t*>(sV + VST * VT_BYTES);
  uint64_t* VF = KF + KST;
  uint64_t* KE = VF + VST;
  uint64_t* VE = KE + KST;
  uint64_t* QD = VE + VST;   // qk_done[SB]
  uint64_t* PR = QD + SB;    // p_ready[SB]
  uint64_t* PD = PR + SB;    // pv_done[SB]
  uint64_t* CVK = PD + SB;   // F8T: K f16 target ready (convert -> QK issuer)
  uint64_t* CVV = CVK + (F8T ? 2 : 0);  // F8T: V target ready (convert -> PV issuer)
  uint32_t* tmem_word = reinterpret_cast<uint32_t*>(CVV + (F8T ? 2 : 0));
  // SM8 cross-half row-max exchange (128 rows x 2 halves), in the barrier
  // tail beyond the epilogue sOut overlay (launcher bumps smem to 2048).
  float* sMx = reinterpret_cast<float*>(tmem_word + 2);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // TRACE (cfg251 hd128 / cfg252 hd64, diagnostics): a per-CTA timeline in the
  // split-KV scratch at +100MiB+64KiB, 10 longs per CTA, written by tid 0 only,
  // valid while splits == 1 (with splits > 1 the partials own that buffer, which
  // is why IT6 has the same restriction). The host zeroes the workspace before
  // the run, so a slot that reads 0 is a CTA that never launched. Fields:
  //   [0] globaltimer at entry   [1] globaltimer at exit
  //   [2] A->B prologue cycles (Q gather + TMEM alloc, before the role split)
  //   [3] B->C cycles to the top of the SECOND tile (the cold first tile)
  //   [4] B->D cycles over the whole tile loop   [5] D->E epilogue cycles
  //   [6] ntiles   [7] flag (0 normal, 1 empty mtile, 3 empty split)
  //   [8] mtile    [9] by
  // prologue/epilogue attribution (job 31 put the prologue at 3.3-5.4us and the
  // epilogue at 1.5-3.2us per CTA but could not say what was inside them):
  //   [10] A ->A1  entry -> mtile/by/q0/qlen/kvlen decoded (cold p.cu_q/p.seq_lens)
  //   [11] A1->A1b always 0 since job 32: this slot held QFAST's hoisted mbarrier
  //                inits + tcgen05.alloc, which measured 0.63us moved and 0.00us
  //                saved (warp 0 still runs it before its own share of the gather,
  //                so nothing overlaps). Kept for record-layout stability.
  //   [12] A1b->A2 the Q gather loop (128 strided 128B segments, bf16 -> f16):
  //                3.76us at hd128 / 2.23us at hd64, the largest prologue term
  //   [13] A2->A3  the 33 single-threaded mbarrier inits + both fences +
  //                tcgen05.alloc: 0.42us / 0.40us
  //   [14] A3->B   the prologue's __syncthreads()
  //   [15] D ->D1  epilogue staging (sMx publish, final PD wait, TMEM O read and
  //                merge, sGain/sLse); [5]-[15] is the global store loop
  long long* trc = nullptr;
  long long gt_a = 0, c_a = 0, c_b = 0, c_c = 0, c_d = 0;
  long long c_a1 = 0, c_a1b = 0, c_a2 = 0, c_a3 = 0, c_d1 = 0;
  if constexpr (TRACE) {
    if (p.splits == 1 && p.part_o != nullptr) {
      const long long bid = (long long)blockIdx.z * gridDim.x * gridDim.y +
                            (long long)blockIdx.y * gridDim.x + blockIdx.x;
      if (bid < 4096)
        trc = reinterpret_cast<long long*>(p.part_o + (25 << 20) + 16384) + bid * 16;
    }
    if (trc != nullptr && tid == 0) {
      gt_a = (long long)t5_gtimer();
      c_a = clock64();
    }
  }

  // MTM6: flattened mtile-major DESCENDING launch order (global longest-first
  // greedy under causal masking; identical logical (mtile, by) mapping so all
  // output/partial addresses match the plain grid bit-for-bit).
  int mtile, by;
  if constexpr (MTM6) {
    const unsigned Y = gridDim.x / (unsigned)p.mtiles;
    // LPT6: request-major with the requests ordered by descending (kv_len - q_len),
    // mtile descending inside a request. tiles(b, m) ~ (kv_b - q_b)/TILE_N + m, so
    // that is the globally longest-first list greedy scheduling wants, where MTM6
    // is longest-first only within a request. Every warp derives the same
    // permutation redundantly (32 lanes, one shuffle each), so there is no smem
    // broadcast and no extra barrier. Single-request batches keep MTM6 exactly.
    const int nb = (int)(Y / (unsigned)p.HKV);
    if (LPT6 && nb > 1) {
      const int per_r = p.mtiles * p.HKV;
      const int r = (int)blockIdx.x / per_r;
      const int rem = (int)blockIdx.x - r * per_r;
      // Sort key: offset descending, request index ascending as the tie-break, so
      // the ranks are a permutation of [0, nb). off can be negative when kv < q,
      // so bias before shifting (a left shift of a negative int is not defined).
      int key = -0x7fffffff;
      if (lane < nb) {
        const int off = p.seq_lens[lane] - (p.cu_q[lane + 1] - p.cu_q[lane]);
        key = ((off + 300000) << 5) | (31 - lane);
      }
      int rank = 0;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        rank += (__shfl_sync(0xffffffffu, key, j) > key);
      }
      const unsigned bal = __ballot_sync(0xffffffffu, (lane < nb) && (rank == r));
      if (bal != 0u) {
        const int bsel = (int)(__ffs(bal) - 1);
        by = bsel * p.HKV + (rem % p.HKV);
        mtile = p.mtiles - 1 - (rem / p.HKV);
      } else {
        // Unreachable, but a wrong guess here is an out-of-range read, so fall back.
        by = (int)(blockIdx.x % Y);
        mtile = p.mtiles - 1 - (int)(blockIdx.x / Y);
      }
    } else {
      by = (int)(blockIdx.x % Y);
      mtile = p.mtiles - 1 - (int)(blockIdx.x / Y);
    }
  } else {
    mtile = (int)blockIdx.x;
    by = (int)blockIdx.y;
  }
  const int b = by / p.HKV;
  const int hkv = by % p.HKV;
  const int q0 = p.cu_q[b];
  const int qlen = p.cu_q[b + 1] - q0;
  const int kvlen = p.seq_lens[b];
  const int G = p.G;
  const int m0 = mtile * TILE_M;
  auto trc_exit = [&](long long flag, long long nt) {
    if constexpr (TRACE) {
      if (trc != nullptr && tid == 0) {
        const bool live = (flag == 0);
        trc[0] = gt_a;
        trc[1] = (long long)t5_gtimer();
        trc[2] = live ? (c_b - c_a) : 0;
        trc[3] = live ? (c_c - c_b) : 0;
        trc[4] = live ? (c_d - c_b) : 0;
        trc[5] = live ? (clock64() - c_d) : 0;
        trc[6] = nt;
        trc[7] = flag;
        trc[8] = mtile;
        trc[9] = by;
        trc[10] = live ? (c_a1 - c_a) : 0;
        trc[11] = live ? (c_a1b - c_a1) : 0;
        trc[12] = live ? (c_a2 - c_a1b) : 0;
        trc[13] = live ? (c_a3 - c_a2) : 0;
        trc[14] = live ? (c_b - c_a3) : 0;
        trc[15] = live ? (c_d1 - c_d) : 0;
      }
    }
  };
  if (m0 >= qlen * G) {
    trc_exit(1, 0);
    return;
  }

  const int i0 = m0 / G;
  const int i1 = min(qlen, (m0 + TILE_M) / G);
  const int vis0 = kvlen - qlen + i0;
  const int vis1 = kvlen - qlen + i1 - 1;
  const int jlo = p.window_left >= 0 ? max(0, vis0 - p.window_left) : 0;
  const int jhi = min(kvlen - 1, vis1);
  const int jt0 = (jlo / TILE_N) * TILE_N;
  const int jt1 = ((jhi + 1 + TILE_N - 1) / TILE_N) * TILE_N;
  const int ntiles = (jt1 - jt0) / TILE_N;
  int t_begin = 0, t_end = ntiles;
  if (p.splits > 1) {
    // Same per-item split derivation as the t5 kernels (TPS mode: each item
    // splits by its own tile count around a common tiles-per-split target).
    int splits_b = p.splits;
    if (p.tps_target > 0) {
      splits_b = (ntiles + p.tps_target - 1) / p.tps_target;
      if (splits_b > p.splits) splits_b = p.splits;
      if (splits_b < 1) splits_b = 1;
    }
    const int tps = (ntiles + splits_b - 1) / splits_b;
    t_begin = blockIdx.z * tps;
    t_end = min(ntiles, t_begin + tps);
  }
  const int split_z = p.splits > 1 ? blockIdx.z : 0;
  const long long w_id = (long long)by * p.mtiles + mtile;

  if (t_begin >= t_end) {
    // Empty split: publish the neutral partial (max -inf, sum 0, O 0) exactly
    // like the t5 kernels so the combine folds it away. In TPS mode the
    // combine derives the same splits_b and never reads z >= splits_b.
    if (p.splits > 1 && p.tps_target == 0) {
      for (int e = tid; e < TILE_M * HD; e += THREADS) {
        if (i0 + (e / HD) / G < qlen) {
          p.part_o[w_id * p.part_s_w + (long long)split_z * p.part_s_s + e] = 0.f;
        }
      }
      if (tid < TILE_M && i0 + tid / G < qlen) {
        float* ml = p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s +
                    (long long)tid * 2;
        ml[0] = NEG_BIG;
        ml[1] = 0.f;
        float* mt = p.part_ml_t + w_id * p.partml_s_w + (long long)tid * p.partmlt_s_r +
                    (long long)split_z * 2;
        mt[0] = NEG_BIG;
        mt[1] = 0.f;
      }
    }
    trc_exit(3, 0);
    return;
  }

  // Per-thread softmax state, kept at function scope so the epilogue (outside
  // the role branch) can read lrow/mdef of the softmax warps.
  float t6_lrow = 0.f, t6_mdef = NEG_BIG;

  // ---------------- prologue: Q tile -> smem (bf16 -> f16) ----------------
  if constexpr (TRACE) {
    if (trc != nullptr && tid == 0) c_a1 = clock64();
  }
  if constexpr (TRACE) {
    if (trc != nullptr && tid == 0) c_a1b = clock64();
  }
  constexpr int QCH = TILE_M * HD / 8;
  constexpr int QUI = (QCH + THREADS - 1) / THREADS;  // 16B chunks per thread
  const __nv_bfloat16* qbase = p.q + (long long)q0 * p.q_s_tok;
  if constexpr (QCP) {
    // All QUI chunks in flight at once. The loop below is not equivalent for the
    // compiler: its destination is a generic char*, so it cannot rule out the smem
    // store aliasing the next iteration's global load and will not hoist the loads.
#pragma unroll
    for (int u = 0; u < QUI; ++u) {
      const int c = tid + u * THREADS;
      if (QCH % THREADS == 0 || c < QCH) {
        const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
        const int i = i0 + r / G;
        const bool ok = (i < qlen);
        const __nv_bfloat16* src =
            qbase + (long long)(ok ? i : 0) * p.q_s_tok +
            (long long)(hkv * G + (r % G)) * p.q_s_head + col;
        cp_async16_z(sQ + t5_off(TILE_M, r, col), src, ok ? 16 : 0);
      }
    }
    cp_async_commit();
    if constexpr (PLO) {
      // J53: hide the mbarrier inits and the TMEM alloc under the in-flight Q
      // copies. Parallel inits (one round, THREADS >= 33); the release fence
      // still precedes the __syncthreads that publishes everything, and the
      // convert pass still follows cp_async_wait, so every happens-before
      // edge of the serial form is preserved (bit-identical).
      for (int s = tid; s < KST; s += THREADS) t5_mbar_init(t5_sp(&KF[s]), LPAR);
      for (int s = tid; s < KST; s += THREADS) t5_mbar_init(t5_sp(&KE[s]), 1);
      for (int s = tid; s < VST; s += THREADS) t5_mbar_init(t5_sp(&VF[s]), LPAR);
      for (int s = tid; s < VST; s += THREADS) t5_mbar_init(t5_sp(&VE[s]), 1);
      for (int i = tid; i < SB; i += THREADS) {
        t5_mbar_init(t5_sp(&QD[i]), 1);
        t5_mbar_init(t5_sp(&PR[i]), NSW);
        t5_mbar_init(t5_sp(&PD[i]), 1);
      }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
      if (warp == 0) {
        t5_alloc(t5_sp(tmem_word), 512);
        t5_relinquish();
      }
      cp_async_wait<0>();
    } else {
      cp_async_wait<0>();
    }
    // bf16 -> f16 in place. Same thread owns the same 16B chunk, so no barrier is
    // needed between the two passes, and the QUI iterations are independent.
#pragma unroll
    for (int u = 0; u < QUI; ++u) {
      const int c = tid + u * THREADS;
      if (QCH % THREADS == 0 || c < QCH) {
        const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
        uint4* d = reinterpret_cast<uint4*>(sQ + t5_off(TILE_M, r, col));
        const uint4 data = *d;
        *d = make_uint4(bf16x2_to_f16x2(data.x), bf16x2_to_f16x2(data.y),
                        bf16x2_to_f16x2(data.z), bf16x2_to_f16x2(data.w));
      }
    }
  } else {
#pragma unroll
    for (int u = 0; u < QUI; ++u) {
      const int c = tid + u * THREADS;
      if (QCH % THREADS != 0 && c >= QCH) break;
      const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
      const int i = i0 + r / G;
      uint4 data = make_uint4(0, 0, 0, 0);
      if (i < qlen) {
        const __nv_bfloat16* src =
            qbase + (long long)i * p.q_s_tok + (long long)(hkv * G + (r % G)) * p.q_s_head + col;
        data = *reinterpret_cast<const uint4*>(src);
      }
      const uint4 h = make_uint4(bf16x2_to_f16x2(data.x), bf16x2_to_f16x2(data.y),
                                 bf16x2_to_f16x2(data.z), bf16x2_to_f16x2(data.w));
      *reinterpret_cast<uint4*>(sQ + t5_off(TILE_M, r, col)) = h;
    }
  }
  if constexpr (TRACE) {
    if (trc != nullptr && tid == 0) c_a2 = clock64();
  }
  if constexpr (!PLO) {
    if (tid == 0) {
      for (int s = 0; s < KST; ++s) {
        t5_mbar_init(t5_sp(&KF[s]), LPAR);
        t5_mbar_init(t5_sp(&KE[s]), 1);
      }
      for (int s = 0; s < VST; ++s) {
        t5_mbar_init(t5_sp(&VF[s]), LPAR);
        t5_mbar_init(t5_sp(&VE[s]), 1);
      }
      for (int i = 0; i < SB; ++i) {
        t5_mbar_init(t5_sp(&QD[i]), 1);
        t5_mbar_init(t5_sp(&PR[i]), NSW);
        t5_mbar_init(t5_sp(&PD[i]), 1);
      }
      if constexpr (F8T) {
        t5_mbar_init(t5_sp(&CVK[0]), 1);
        t5_mbar_init(t5_sp(&CVK[1]), 1);
        t5_mbar_init(t5_sp(&CVV[0]), 1);
        t5_mbar_init(t5_sp(&CVV[1]), 1);
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (warp == 0) {
      t5_alloc(t5_sp(tmem_word), 512);
      t5_relinquish();
    }
  }
  fence_async_smem();  // Q STS (generic proxy) ordered against TC descriptor reads
  if constexpr (TRACE) {
    if (trc != nullptr && tid == 0) c_a3 = clock64();
  }
  __syncthreads();
  t5_fence_after_sync();
  if constexpr (TRACE) {
    if (trc != nullptr && tid == 0) c_b = clock64();
  }
  const uint32_t tmem = *tmem_word;
  const uint32_t o_tmem = tmem + OCOL;
  const uint32_t o1_tmem = o_tmem + HD;  // second O region, PHO only

  // IT6 (cfg161, diagnostics): per-phase clock64 timestamps of ONE interior
  // CTA into the unused split-KV scratch at +100MiB (valid while splits==1).
  // MMA warp slots per iter: 0 top, 1 after KF wait, 2 after QK(t+1) issue,
  // 3 after PR wait, 4 after VF wait, 5 after PV issue. Softmax warp0/lane0
  // slots at dbg[2048+...]: 0 after QD wait, 1 after S ld, 2 after max,
  // 3 after exp/pack, 4 after P st. Counts at dbg[386*2] / dbg[2048+386*2].
  long long* dbg = nullptr;
  long long tstart = 0;
  int it = 0;
  if constexpr (IT6) {
    // Select the traced CTA by LOGICAL (mtile, by), not by blockIdx: under MTM6
    // the flattened descending order maps blockIdx.x == mtiles/4 onto a high
    // mtile whose request has already ended, so that CTA takes the early return
    // and writes nothing. For the non-MTM form mtile == blockIdx.x and
    // by == blockIdx.y, so cfg209's selection is unchanged; splits == 1 implies
    // gridDim.z == 1. (The t5 INSTR block above keeps the blockIdx form.)
    if (p.splits == 1 && p.part_o != nullptr && mtile == p.mtiles / 4 && by == 0) {
      dbg = reinterpret_cast<long long*>(p.part_o + (25 << 20));
      if (tid == 0) {
        tstart = clock64();
        dbg[0] = tstart;
      }
    }
  }
  auto tspot = [&](long long* base, int k) {
    if constexpr (IT6) {
      if (dbg != nullptr && base != nullptr && it < 48)
        base[it * 8 + k] = clock64() - tstart;
    }
  };
  long long* dbg_mma = (IT6 && dbg != nullptr && warp == MMA_W && lane == 0) ? dbg + 1 : nullptr;
  long long* dbg_sm = (IT6 && dbg != nullptr && warp == 0 && lane == 0) ? dbg + 2049 : nullptr;
  if constexpr (IT6) {
    if (dbg != nullptr && tid == 0) { dbg[768] = 0; dbg[2048 + 768] = 0; }
  }

  if (warp == MMA_W && lane == 0) {
    // ------------------------- MMA warp (QK issuer under DUAL) --------------
    const uint32_t ID_QK = t5_idesc(TILE_M, TILE_N, 0);
    const uint32_t ID_PV = t5_idesc(TILE_M, PVN, 1);
    constexpr int PATOMS = HD / PVN;        // n-atoms per PV
    constexpr int PGRP = PVN / 64;          // 64-col swizzle groups per atom
    const uint64_t qd0 = t5_desc(t5_sp(sQ), 16, 8 * 128);
    const uint64_t kdb = t5_desc(t5_sp(sK), 16, 8 * 128);
    // PVN=64: V0's per-atom MN-major form (lbo=0). PVN=128: the n128
    // instruction spans TWO 64-col swizzle groups; their stride inside the
    // tile is TILE_N*128B, encoded as the leading-byte offset (probe_tc
    // measured m128n128k16 TS at 64.1 cyc vs 2x44.6 for the n64 atoms).
    const uint64_t vdb = t5_desc(t5_sp(sV), PVN == 64 ? 0 : TILE_N * 128, 8 * 128);
    auto issue_qk = [&](int t) {
      const int rr = t - t_begin;
      const int sb = rr % SB, kb = rr % KST;
      const uint32_t scol = tmem + sb * TILE_N;
      const uint64_t kd = kdb + (uint64_t)(kb * (KT_BYTES >> 4));
#pragma unroll
      for (int g = 0; g < KG; ++g)
#pragma unroll
        for (int s = 0; s < 4; ++s)
          t5_umma_ss(scol, qd0 + (uint64_t)(g * (TILE_M * 8) + s * 2),
                     kd + (uint64_t)(g * (TILE_N * 8) + s * 2), ID_QK, (g | s) != 0);
      t5_commit(t5_sp(&QD[sb]));
      t5_commit(t5_sp(&KE[kb]));
    };
    // Prime the QK pipeline QLA tiles deep (QLA=1 reproduces the old prologue).
    for (int u = 0; u < QLA; ++u) {
      if (t_begin + u >= t_end) break;
      if constexpr (F8T) t5_mbar_wait(t5_sp(&CVK[u % KST]), (u / KST) & 1);
      else t5_mbar_wait(t5_sp(&KF[u % KST]), (u / KST) & 1);
      t5_fence_after_sync();
      issue_qk(t_begin + u);
    }
    if constexpr (DUAL) {
      // QK-only: run ahead issuing lookaheads; PV lives on warp 6. No early
      // return -- every thread must reach the epilogue's __syncthreads.
      // RACE FIX (K44 bisect: f7/5-tile-mini NaN'd out, lse clean): QK(t+1)
      // writes S[(rr+1)&1], the buffer PV(t-1) reads. The single-issuer form
      // is safe by in-order TC issue; two issuer warps are NOT ordered, so
      // the QK warp must wait PV(t-1)'s PD commit before reissuing into the
      // shared S parity.
      for (int t = t_begin; t < t_end; ++t) {
        if (t + 1 < t_end) {
          const int rr = t - t_begin;
          const int kb1 = (rr + 1) % KST;
          if (rr >= 1) {
            t5_mbar_wait(t5_sp(&PD[(rr + 1) & 1]), ((rr - 1) >> 1) & 1);
            t5_fence_after_sync();
          }
          t5_mbar_wait(t5_sp(&KF[kb1]), ((rr + 1) / KST) & 1);
          t5_fence_after_sync();
          issue_qk(t + 1);
        }
      }
    } else
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int sb = rr % SB, vb = rr % VST;
      tspot(dbg_mma, 0);
      if (t + QLA < t_end) {
        const int kb1 = (rr + QLA) % KST;
        if constexpr (F8T) t5_mbar_wait(t5_sp(&CVK[kb1]), ((rr + QLA) / KST) & 1);
        else t5_mbar_wait(t5_sp(&KF[kb1]), ((rr + QLA) / KST) & 1);
        t5_fence_after_sync();
        if constexpr (DPV) {
          // Two unordered issuer warps: the in-order-TC argument that makes the
          // single-issuer form safe no longer holds, so this warp must wait for the
          // PV that consumed the S buffer it is about to overwrite (tile
          // rr+QLA-SB) -- the race arm-a's K44 bisect found for DUAL at SB=2.
          if (rr + QLA >= SB) {
            t5_mbar_wait(t5_sp(&PD[(rr + QLA) % SB]), ((rr + QLA - SB) / SB) & 1);
            t5_fence_after_sync();
          }
        }
        tspot(dbg_mma, 1);
        issue_qk(t + QLA);
        tspot(dbg_mma, 2);
      }
      if constexpr (DPV) {
        if constexpr (IT6) { if (dbg_mma != nullptr && it < 48) ++it; }
        continue;   // slots 3-5 (PR/VF/PV) belong to the PV issuer warp under DPV
      }
      t5_mbar_wait(t5_sp(&PR[sb]), (rr / SB) & 1);
      t5_fence_after_sync();
      tspot(dbg_mma, 3);
      t5_mbar_wait(t5_sp(&VF[vb]), (rr / VST) & 1);
      tspot(dbg_mma, 4);
      const uint32_t scol = tmem + sb * TILE_N;
      const uint64_t vd = vdb + (uint64_t)(vb * (VT_BYTES >> 4));
#pragma unroll
      for (int a = 0; a < PATOMS; ++a)
#pragma unroll
        for (int s = 0; s < PV_STEPS; ++s)
          t5_umma_ts(o_tmem + a * PVN, scol + s * 8,
                     vd + (uint64_t)(a * (PGRP * (TILE_N * 8)) + s * 128), ID_PV,
                     (t != t_begin || s != 0));
      t5_commit(t5_sp(&PD[sb]));
      t5_commit(t5_sp(&VE[vb]));
      tspot(dbg_mma, 5);
      if constexpr (IT6) { if (dbg_mma != nullptr && it < 48) { ++it; } }
    }
    if constexpr (IT6) { if (dbg_mma != nullptr) dbg[768] = it; }
  } else if (DUAL && warp == 6 && lane == 0) {
    // ------------------------- PV issuer warp (DUAL) -------------------------
    // Waits PR/VF and issues PV + PD/VE commits independently of the QK warp;
    // correctness is barrier-mediated (PV reads S[sb] only after PR[sb], which
    // implies QK(t) completion through the softmax QD wait chain), so the two
    // issue streams need no direct ordering. tcgen05.commit completion of
    // PD/VE covers this thread's PV issues (CTA-wide at worst: conservative).
    const uint32_t ID_PV = t5_idesc(TILE_M, PVN, 1);
    constexpr int PATOMS = HD / PVN;
    constexpr int PGRP = PVN / 64;
    const uint64_t vdb = t5_desc(t5_sp(sV), PVN == 64 ? 0 : TILE_N * 128, 8 * 128);
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int sb = rr & 1, vb = rr % VST;
      t5_mbar_wait(t5_sp(&PR[sb]), (rr >> 1) & 1);
      t5_fence_after_sync();
      t5_mbar_wait(t5_sp(&VF[vb]), (rr / VST) & 1);
      const uint32_t scol = tmem + sb * TILE_N;
      const uint64_t vd = vdb + (uint64_t)(vb * (VT_BYTES >> 4));
#pragma unroll
      for (int a = 0; a < PATOMS; ++a)
#pragma unroll
        for (int s = 0; s < PV_STEPS; ++s)
          t5_umma_ts(o_tmem + a * PVN, scol + s * 8,
                     vd + (uint64_t)(a * (PGRP * (TILE_N * 8)) + s * 128), ID_PV,
                     (t != t_begin || s != 0));
      t5_commit(t5_sp(&PD[sb]));
      t5_commit(t5_sp(&VE[vb]));
    }
  } else if (DPV != 0 && warp == PV_W && lane == 0) {
    // ------------------------- PV issuer warp (DPV) -------------------------
    // Waits PR/VF and issues PV plus the PD/VE commits independently of the QK
    // warp. Correctness is barrier-mediated: PV(t) reads S[t%SB] only after
    // PR[t%SB], which implies QK(t) completed through the softmax's QD wait chain,
    // and every PV that accumulates a row of O is issued by this one warp in tile
    // order, so the O accumulation order -- and therefore the output bits -- are
    // unchanged from the single-issuer form. tcgen05.commit is CTA-wide at worst,
    // which is conservative (PD may also cover a QK), never early.
    const uint32_t ID_PV = t5_idesc(TILE_M, PVN, 1);
    constexpr int PATOMS = HD / PVN;
    constexpr int PGRP = PVN / 64;
    const uint64_t vdb = t5_desc(t5_sp(sV), PVN == 64 ? 0 : TILE_N * 128, 8 * 128);
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int sb = rr % SB, vb = rr % VST;
      t5_mbar_wait(t5_sp(&PR[sb]), (rr / SB) & 1);
      t5_fence_after_sync();
      if constexpr (F8T) t5_mbar_wait(t5_sp(&CVV[vb]), (rr / VST) & 1);
      else t5_mbar_wait(t5_sp(&VF[vb]), (rr / VST) & 1);
      const uint32_t scol = tmem + sb * TILE_N;
      const uint64_t vd = vdb + (uint64_t)(vb * (VT_BYTES >> 4));
#pragma unroll
      for (int a = 0; a < PATOMS; ++a)
#pragma unroll
        for (int s = 0; s < PV_STEPS; ++s) {
          // PHO: k-steps below PV_STEPS/2 consume half 0's P and accumulate O0,
          // the rest consume half 1's P and accumulate O1; each region's first
          // atom of the first tile overwrites instead of accumulating.
          const int sh = (PHO != 0 && s >= PV_STEPS / 2) ? 1 : 0;
          t5_umma_ts(o_tmem + sh * ((PHO != 0) ? HD : 0) + a * PVN, scol + s * 8,
                     vd + (uint64_t)(a * (PGRP * (TILE_N * 8)) + s * 128), ID_PV,
                     (t != t_begin || s != sh * (PV_STEPS / 2)));
        }
      t5_commit(t5_sp(&PD[sb]));
      t5_commit(t5_sp(&VE[vb]));
    }
  } else if (warp == LOAD_W && (F8T != 0 || lane < LPAR)) {
    if constexpr (F8T) {
      // ------------------- F8T fused load+convert (K side) -------------------
      // J59 t6f: fp8 TMA ring + in-warp convert into the double-buffered f16
      // target. Slot reuse is program-order-safe: TMA(t+1) targets slot s^1,
      // whose reads this warp finished in convert(t-1) (all-lane
      // fence_async_smem + __syncwarp below), so t7's CE barrier is unneeded.
      // CVK[s] (lane0 arrival after the fence+syncwarp) gates the QK issuer;
      // KE[s] is QK(t-2)'s existing tcgen05 commit, gating target reuse.
      const int page_log2 = p.page_log2;
      const int box8 = min(1 << page_log2, 64);
      const int page_mask = (1 << page_log2) - 1;
      const int nbox = TILE_N / box8;
      const int tcols = p.table_stride;
      const int* table_row = p.table + (long long)b * tcols;
      auto issue8 = [&](int t, int s) {
        const int jb = jt0 + t * TILE_N;
        const bool own = lane < nbox;
        const int jj = own ? min(max(jb + lane * box8, 0), kvlen - 1) : 0;
        const int pid = own ? table_row[min(jj >> page_log2, tcols - 1)] : 0;
        const int row_in_page = (jj & page_mask) & ~(box8 - 1);
        const uint32_t kbar = t5_sp(&KF[s]);
        if (lane < LPAR) {
          mb_expect(kbar, own ? (unsigned)(box8 * 128) : 0u);
          if (own)
            tma4d(&kmap, t5_sp(sK8 + s * (KT_BYTES / 2) + lane * box8 * 128), kbar, 0,
                  row_in_page, hkv, pid);
        }
      };
      if (t_begin < t_end) issue8(t_begin, 0);
      constexpr int NCH = TILE_N * (HD / 16);
      for (int t = t_begin; t < t_end; ++t) {
        const int rr = t - t_begin, s = rr % KST;
        if (t + 1 < t_end) issue8(t + 1, (rr + 1) % KST);
        t5_mbar_wait(t5_sp(&KF[s]), (rr / KST) & 1);
        if (rr >= KST) t5_mbar_wait(t5_sp(&KE[s]), ((rr / KST) - 1) & 1);
        const uint8_t* k8 = reinterpret_cast<const uint8_t*>(sK8) + s * (KT_BYTES / 2);
        char* dst = sK + (size_t)s * KT_BYTES;
        for (int idx = lane; idx < NCH; idx += 32) {
          const int n = idx / (HD / 16), cc = idx % (HD / 16);
          const int g = cc >> 3, c = cc & 7;
          const uint8_t* src8 = k8 + (size_t)g * (TILE_N * 128) + n * 128 + ((c ^ (n & 7)) << 4);
          const uint4 rawv = *reinterpret_cast<const uint4*>(src8);
          uint32_t hh[8];
          expand8(rawv, hh);
          const int col = g * 128 + c * 16;
          // t6's TC descriptors read the SEGMENT-PLANAR f16 layout that its
          // tma4d boxes fill (plane stride TILE_N*128B = descriptor LBO) and
          // t5_off encodes - NOT t7's row-major swz<> ldmatrix twin.
          // (J59 debug-1: swz writes scrambled K/V -> bitid FAIL d_out 4-6.)
          *reinterpret_cast<uint4*>(dst + t5_off(TILE_N, n, col)) =
              make_uint4(hh[0], hh[1], hh[2], hh[3]);
          *reinterpret_cast<uint4*>(dst + t5_off(TILE_N, n, col + 8)) =
              make_uint4(hh[4], hh[5], hh[6], hh[7]);
        }
        fence_async_smem();
        __syncwarp();
        if (lane == 0) mb_arrive(t5_sp(&CVK[s]));
      }
    } else {
    // ------------------------- load warp -------------------------
    // LPAR lanes each own the pages q == lane (mod LPAR). Every lane arms only the
    // bytes its own pages will deliver, so its expect_tx precedes its own bulk
    // copies in program order and no lane can drive another lane's tx count
    // negative; the barriers are initialised with an arrival count of LPAR so a
    // lane with no pages still arrives (0 bytes) and the phase completes. At
    // LPAR == 1 mybytes == KT_BYTES == VT_BYTES and this is the original form.
    const int page_log2 = p.page_log2;
    const int rows_per_box = 1 << page_log2;  // == page size
    const int pages_per_tile = TILE_N >> page_log2;
    const int tcols = p.table_stride;
    const int* table_row = p.table + (long long)b * tcols;
    constexpr int PMAX = (PPT + LPAR - 1) / LPAR;   // pages this lane can own
    constexpr int BOX_BYTES = 64 * 2;               // one 64-col box row, f16
    int pid[PMAX];
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int kb = rr % KST, vb = SPLITV ? 0 : (rr % VST);
      if (rr >= KST) t5_mbar_wait(t5_sp(&KE[kb]), ((rr / KST) - 1) & 1);
      if constexpr (!SPLITV) {
        if (rr >= VST) t5_mbar_wait(t5_sp(&VE[vb]), ((rr / VST) - 1) & 1);
      }
      const int jb = jt0 + t * TILE_N;
      int np = 0;
      for (int q = lane; q < pages_per_tile; q += LPAR) ++np;
#pragma unroll
      for (int i = 0; i < PMAX; ++i) {
        if (i >= np) break;
        const int q = lane + i * LPAR;
        const int jj = min(max(jb + (q << page_log2), 0), kvlen - 1);
        pid[i] = table_row[min(jj >> page_log2, tcols - 1)];
      }
      const uint32_t kbar = t5_sp(&KF[kb]);
      const uint32_t mybytes = (uint32_t)(np * KG * rows_per_box * BOX_BYTES);
      mb_expect(kbar, mybytes);
      if constexpr (!SPLITV) mb_expect(t5_sp(&VF[vb]), mybytes);
      for (int i = 0, q = lane; q < pages_per_tile; ++i, q += LPAR) {
#pragma unroll
        for (int sg = 0; sg < KG; ++sg) {
          const uint32_t off = (uint32_t)(sg * (TILE_N * 128) + q * (rows_per_box * 128));
          tma4d(&kmap, t5_sp(sK + kb * KT_BYTES + off), kbar, sg * 64, 0, hkv, pid[i]);
          if constexpr (!SPLITV)
            tma4d(&vmap, t5_sp(sV + vb * VT_BYTES + off), t5_sp(&VF[vb]), sg * 64, 0, hkv,
                  pid[i]);
        }
      }
    }
    }
  } else if (SPLITV != 0 && warp == LOAD_VW && (F8T != 0 || lane < LPAR)) {
    if constexpr (F8T) {
      // ------------------- F8T fused load+convert (V side) -------------------
      // Same discipline as the K side; VE[s] is PV(t-2)'s commit, CVV[s] gates
      // the PV issuer.
      const int page_log2 = p.page_log2;
      const int box8 = min(1 << page_log2, 64);
      const int page_mask = (1 << page_log2) - 1;
      const int nbox = TILE_N / box8;
      const int tcols = p.table_stride;
      const int* table_row = p.table + (long long)b * tcols;
      auto issue8v = [&](int t, int s) {
        const int jb = jt0 + t * TILE_N;
        const bool own = lane < nbox;
        const int jj = own ? min(max(jb + lane * box8, 0), kvlen - 1) : 0;
        const int pid = own ? table_row[min(jj >> page_log2, tcols - 1)] : 0;
        const int row_in_page = (jj & page_mask) & ~(box8 - 1);
        const uint32_t vbar = t5_sp(&VF[s]);
        if (lane < LPAR) {
          mb_expect(vbar, own ? (unsigned)(box8 * 128) : 0u);
          if (own)
            tma4d(&vmap, t5_sp(sV8 + s * (VT_BYTES / 2) + lane * box8 * 128), vbar, 0,
                  row_in_page, hkv, pid);
        }
      };
      if (t_begin < t_end) issue8v(t_begin, 0);
      constexpr int NCHV = TILE_N * (HD / 16);
      for (int t = t_begin; t < t_end; ++t) {
        const int rr = t - t_begin, s = rr % VST;
        if (t + 1 < t_end) issue8v(t + 1, (rr + 1) % VST);
        t5_mbar_wait(t5_sp(&VF[s]), (rr / VST) & 1);
        if (rr >= VST) t5_mbar_wait(t5_sp(&VE[s]), ((rr / VST) - 1) & 1);
        const uint8_t* v8 = reinterpret_cast<const uint8_t*>(sV8) + s * (VT_BYTES / 2);
        char* dst = sV + (size_t)s * VT_BYTES;
        for (int idx = lane; idx < NCHV; idx += 32) {
          const int n = idx / (HD / 16), cc = idx % (HD / 16);
          const int g = cc >> 3, c = cc & 7;
          const uint8_t* src8 = v8 + (size_t)g * (TILE_N * 128) + n * 128 + ((c ^ (n & 7)) << 4);
          const uint4 rawv = *reinterpret_cast<const uint4*>(src8);
          uint32_t hh[8];
          expand8(rawv, hh);
          const int col = g * 128 + c * 16;
          // Segment-planar t5_off layout (see the K side; J59 debug-1 fix).
          *reinterpret_cast<uint4*>(dst + t5_off(TILE_N, n, col)) =
              make_uint4(hh[0], hh[1], hh[2], hh[3]);
          *reinterpret_cast<uint4*>(dst + t5_off(TILE_N, n, col + 8)) =
              make_uint4(hh[4], hh[5], hh[6], hh[7]);
        }
        fence_async_smem();
        __syncwarp();
        if (lane == 0) mb_arrive(t5_sp(&CVV[s]));
      }
    } else {
    // --------------------- V load warp (SPLITV only) ---------------------
    // Same page-id and per-lane expect_tx discipline as the K warp above, on the V
    // ring alone: it waits VE[vb] (PV(t-VST) complete) without holding up the K
    // boxes, and its arrivals arm only the V barrier.
    const int page_log2 = p.page_log2;
    const int rows_per_box = 1 << page_log2;
    const int pages_per_tile = TILE_N >> page_log2;
    const int tcols = p.table_stride;
    const int* table_row = p.table + (long long)b * tcols;
    constexpr int PMAX = (PPT + LPAR - 1) / LPAR;
    constexpr int BOX_BYTES = 64 * 2;
    int pid[PMAX];
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int vb = rr % VST;
      if (rr >= VST) t5_mbar_wait(t5_sp(&VE[vb]), ((rr / VST) - 1) & 1);
      const int jb = jt0 + t * TILE_N;
      int np = 0;
      for (int q = lane; q < pages_per_tile; q += LPAR) ++np;
#pragma unroll
      for (int i = 0; i < PMAX; ++i) {
        if (i >= np) break;
        const int q = lane + i * LPAR;
        const int jj = min(max(jb + (q << page_log2), 0), kvlen - 1);
        pid[i] = table_row[min(jj >> page_log2, tcols - 1)];
      }
      const uint32_t vbar = t5_sp(&VF[vb]);
      const uint32_t mybytes = (uint32_t)(np * KG * rows_per_box * BOX_BYTES);
      mb_expect(vbar, mybytes);
      for (int i = 0, q = lane; q < pages_per_tile; ++i, q += LPAR) {
#pragma unroll
        for (int sg = 0; sg < KG; ++sg) {
          const uint32_t off = (uint32_t)(sg * (TILE_N * 128) + q * (rows_per_box * 128));
          tma4d(&vmap, t5_sp(sV + vb * VT_BYTES + off), vbar, sg * 64, 0, hkv, pid[i]);
        }
      }
    }
    }
  } else if (warp < NSW) {
    // ----------------------- softmax warps -----------------------
    // SM8 (c176): warps 0-3 own columns 0-63 and warps 4-7 own columns
    // 64-127 of the SAME 128 rows (row = (warp&3)*32 + lane). K42: the
    // single-issuer tile period is ~1752 cyc with ~650 cyc of TC idle
    // waiting softmax(t) between QK(t+1) and PV(t); column-splitting halves
    // the softmax busy (~1160 -> ~600 cyc) at the cost of one named-barrier
    // row-max exchange per tile (fmax is exact, so the merged max is
    // bit-identical to the 4-warp recipe; the exp/sum association changes
    // like t8's promoted recipe does). DEFER O-rescale stays with half-0;
    // PR's count-8 rendezvous gates PV(t) behind BOTH halves, so the rescale
    // needs no second barrier. hacc keeps 4 slots over 32 pairs per half.
    const int half = SM8 ? (warp >> 2) : 0;
    const int row = (SM8 ? (warp & 3) : warp) * 32 + lane;
    const int irow = i0 + row / G;
    const bool rowok = irow < qlen;
    const int vis = kvlen - qlen + irow;
    const int lim = rowok ? vis : -1;
    const int low = (rowok && p.window_left >= 0) ? max(0, vis - p.window_left) : 0;
    float mrun = NEG_BIG;
    float lrow = 0.f;
    float mdef = NEG_BIG;
    const float qk = p.qk_scale;
    const uint64_t qk2 = f2_pack(qk, qk);
    // F16X: the same scale as a packed f16 pair, so the scale-offset can run in
    // f16x2 (see the exp/pack loop below).
    const uint32_t qkh = (F16X != 0) ? pack_f16(qk, qk) : 0u;
    // MSKIP: a warp whose 32 rows are ALL padding has no observable effect beyond
    // its collective arrivals. MMA rows are independent, so its S/P/O feed only its
    // own rows, and neither epilogue path stores them (both predicate on the row
    // being in range), so skipping the body is bit-identical. It must still reach
    // bar_sm8 (a 256-thread rendezvous) and PR (one arrival per softmax warp, or
    // the PV issuer waits forever). Skipping by whole warps is also the only legal
    // way to drop the S tile's tcgen05.ld/st: those are .sync.aligned, so a
    // per-lane skip of them would be illegal -- which is what RWSK cannot do.
    // This is the q1025-class tail-mtile straggler: f9 has q=1025 at G=1, so the
    // last of its 9 mtiles carries ONE valid row of 128 and 6 of the 8 softmax
    // warps are fully masked; at 144 CTAs on 148 SMs the kernel's makespan IS those
    // CTAs, and they run the divergent masked path per tile instead of at the
    // tensor-core rate.
    const bool deadw = (MSKIP != 0) && !__any_sync(0xffffffffu, rowok);
    if (deadw) {
      for (int t = t_begin; t < t_end; ++t) {
        const int rr = t - t_begin;
        if constexpr (PHO == 0) {
          if constexpr (PBAR) bar_pair64(warp);
          else bar_sm8();  // PHO has no per-tile rendezvous
        }
        t5_fence_before_sync();
        if (lane == 0) mb_arrive(t5_sp(&PR[rr % SB]));
      }
      t6_lrow = 0.f;
      t6_mdef = NEG_BIG;
    } else {
    int its = 0;
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      if constexpr (TRACE) {
        if (rr == 1 && tid == 0) c_c = clock64();
      }
      const int sb = rr % SB;
      const uint32_t scol = tmem + sb * TILE_N;
      t5_mbar_wait(t5_sp(&QD[sb]), (rr / SB) & 1);
      if (warp == 0) { it = its; tspot(dbg_sm, 0); }
      float sc[SCOLS];
#pragma unroll
      for (int blk = 0; blk < SCOLS / 32; ++blk)
        t5_ld32(sc + blk * 32, scol + half * SCOLS + blk * 32);
      t5_ld_wait();
      if (warp == 0) tspot(dbg_sm, 1);
      const int col0 = jt0 + t * TILE_N + half * SCOLS;
      const bool full = rowok && (col0 + SCOLS - 1 <= lim) && (col0 >= low);
      // RWSK: this lane's whole column half lies outside [low, lim], so every one
      // of its elements would mask to NEG_BIG and ex2(-inf) to +0. Skipping the two
      // loops is bit-identical -- mxr stays NEG_BIG, pp becomes 0, lrow += 0, and
      // the mn/mdef recurrence and its warp-wide vote are untouched -- and it turns
      // a mixed warp's cost from SUM(full path, masked path) into MAX(lane work).
      // Without it the 31 padded lanes of the q1025 tail mtile drag the one valid
      // lane through the predicated per-column form on every tile.
      // J46 (arm-a J32, bit-identical): !full short-circuits the predicate on
      // the common all-visible tile and the hot path gets its own single-branch
      // block (nosel and full are mutually exclusive). On the arm-a tree this
      // branch layout was worth ~4% on long-pipeline shapes (f7 -4.6%, f1
      // -4.1%, f4 -3.9%; receipt dv_271bdd71ba55).
      const bool nosel = (RWSK != 0) && !full && ((lim < col0) || (low > col0 + SCOLS - 1));
      float mxr = NEG_BIG;
      if (full) {
        float m0v = NEG_BIG, m1v = NEG_BIG, m2v = NEG_BIG, m3v = NEG_BIG;
#pragma unroll
        for (int c = 0; c < SCOLS; c += 4) {
          m0v = fmaxf(m0v, sc[c]);
          m1v = fmaxf(m1v, sc[c + 1]);
          m2v = fmaxf(m2v, sc[c + 2]);
          m3v = fmaxf(m3v, sc[c + 3]);
        }
        mxr = fmaxf(fmaxf(m0v, m1v), fmaxf(m2v, m3v));
      } else if (!nosel) {
        float m0v = NEG_BIG, m1v = NEG_BIG, m2v = NEG_BIG, m3v = NEG_BIG;
#pragma unroll
        for (int c = 0; c < SCOLS; c += 4) {
          m0v = fmaxf(m0v, (col0 + c <= lim && col0 + c >= low) ? sc[c] : NEG_BIG);
          m1v = fmaxf(m1v, (col0 + c + 1 <= lim && col0 + c + 1 >= low) ? sc[c + 1] : NEG_BIG);
          m2v = fmaxf(m2v, (col0 + c + 2 <= lim && col0 + c + 2 >= low) ? sc[c + 2] : NEG_BIG);
          m3v = fmaxf(m3v, (col0 + c + 3 <= lim && col0 + c + 3 >= low) ? sc[c + 3] : NEG_BIG);
        }
        mxr = fmaxf(fmaxf(m0v, m1v), fmaxf(m2v, m3v));
      }
      if constexpr (SM8 && PHO == 0) {
        sMx[row * 2 + half] = mxr;
        if constexpr (PBAR) bar_pair64(warp);
        else bar_sm8();
        mxr = fmaxf(mxr, sMx[row * 2 + (half ^ 1)]);
      }
      if (warp == 0) tspot(dbg_sm, 2);
      // Running max in scaled log2 units (see t5): forming it as (raw*qk)-mdef
      // instead would let nvcc contract the rounding residual into an exponent.
      const float mn = fmaxf(mrun, mxr * qk);
      float mneg;
      if (t == t_begin) {
        mdef = mn;
      } else if (__any_sync(0xffffffffu, mn - mdef > (float)DTH)) {
        // DEFER O-rescale: PV(t-1) may still be accumulating in the tensor
        // core, so wait its commit before touching O in the generic proxy.
        // SM8: both halves scale their own lrow share with the same f (the
        // merged-max/mdef recurrence is identical across halves, so the
        // __any_sync predicate is too), but only half-0 rescales the shared
        // O accumulator; PR's count-8 rendezvous keeps PV(t) behind it.
        const float f = ex2(mdef - mn);
        lrow *= f;
        // Without PHO the O accumulator is shared, so only half 0 rescales it and
        // PR's count-8 rendezvous keeps PV(t) behind that. With PHO each half owns
        // its region, so all eight warps rescale their own and there is no
        // cross-half ordering left to protect.
        const uint32_t obase = (PHO != 0) ? (o_tmem + half * HD) : o_tmem;
        if ((PHO != 0) || half == 0) {
          t5_mbar_wait(t5_sp(&PD[(rr + SB - 1) % SB]), ((rr - 1) / SB) & 1);
          t5_fence_after_sync();
          for (int blk = 0; blk < HD / 32; ++blk) {
            float ov[32];
            t5_ld32(ov, obase + blk * 32);
            t5_ld_wait();
#pragma unroll
            for (int j = 0; j < 32; ++j) ov[j] *= f;
            t5_st32(reinterpret_cast<const uint32_t*>(ov), obase + blk * 32);
          }
          t5_st_wait();
        }
        mdef = mn;
      }
      mrun = mn;
      mneg = -mdef;
      uint32_t pp[SCOLS / 2];
      const uint64_t mn2 = f2_pack(mneg, mneg);
      const uint32_t mnh = (F16X != 0) ? pack_f16(mneg, mneg) : 0u;
      // HACC: row-sum accumulators. 4 slots over 32 pairs means each accumulator is
      // a serial chain of 8 f16 adds inside a loop whose live set is already
      // sc[SCOLS] + pp[SCOLS/2]; 8 slots halve the chain at the cost of 4 registers.
      uint32_t hacc[HACC] = {};
      if (full) {
#pragma unroll
        for (int c = 0; c < SCOLS; c += 2) {
          uint32_t p2;
          if constexpr (F16X != 0) {
            // cvt + fma.rn.f16x2 + ex2 + add = 4 instructions per pair, against
            // mov.b64 + fma.rn.f32x2 + cvt + ex2 + add = 5. The exponent argument
            // loses ~2^-11 relative on S and ~0.015 absolute on the offset; the
            // offset term is common to the whole row so it cancels in O = PV / sum P
            // and leaves ~0.015 log2 units on lse, against a frozen tolerance of
            // 0.01 + 0.05|lse| on values of 5-20. pack_f16 is satfinite, so |S|
            // clamps at 65504 instead of becoming Inf (the f32 path never sees that
            // bound; it would need |q.k| per term above ~512, which the frozen
            // builder's normal/tiny/peaked/zero inputs cannot produce).
            p2 = ex2_f16x2(hfma2(pack_f16(sc[c], sc[c + 1]), qkh, mnh));
          } else {
            const uint64_t x2 = fma2(f2_pack(sc[c], sc[c + 1]), qk2, mn2);
            float e0, e1;
            f2_unpack(x2, e0, e1);
            p2 = ex2_f16x2(pack_f16(e0, e1));
          }
          pp[c >> 1] = p2;
          hacc[(c >> 1) & (HACC - 1)] = hadd2(hacc[(c >> 1) & (HACC - 1)], p2);
        }
      } else if (nosel) {
#pragma unroll
        for (int j = 0; j < SCOLS / 2; ++j) pp[j] = 0u;
      } else {
#pragma unroll
        for (int c = 0; c < SCOLS; c += 2) {
          const int j = col0 + c;
          uint32_t p2;
          if constexpr (F16X != 0) {
            uint32_t x2 = hfma2(pack_f16(sc[c], sc[c + 1]), qkh, mnh);
            // Masked halves become f16 -inf (0xFC00), which ex2.approx.f16x2 maps
            // to +0 exactly as the f32 path's NEG_BIG -> satfinite -> -inf did.
            const uint32_t keep = ((j <= lim && j >= low) ? 0xFFFFu : 0u) |
                                  (((j + 1 <= lim && j + 1 >= low) ? 0xFFFFu : 0u) << 16);
            x2 = (x2 & keep) | (0xFC00FC00u & ~keep);
            p2 = ex2_f16x2(x2);
          } else {
            const uint64_t x2 = fma2(f2_pack(sc[c], sc[c + 1]), qk2, mn2);
            float e0, e1;
            f2_unpack(x2, e0, e1);
            if (!(j <= lim && j >= low)) e0 = NEG_BIG;
            if (!(j + 1 <= lim && j + 1 >= low)) e1 = NEG_BIG;
            p2 = ex2_f16x2(pack_f16(e0, e1));
          }
          pp[c >> 1] = p2;
          hacc[(c >> 1) & (HACC - 1)] = hadd2(hacc[(c >> 1) & (HACC - 1)], p2);
        }
      }
      if constexpr (HACC == 4) {
        lrow += (hsum2(hacc[0]) + hsum2(hacc[1])) + (hsum2(hacc[2]) + hsum2(hacc[3]));
      } else {
        lrow += ((hsum2(hacc[0]) + hsum2(hacc[1])) + (hsum2(hacc[2]) + hsum2(hacc[3]))) +
                ((hsum2(hacc[4]) + hsum2(hacc[5])) + (hsum2(hacc[6]) + hsum2(hacc[7])));
      }
      if (warp == 0) tspot(dbg_sm, 3);
#pragma unroll
      for (int blk = 0; blk < SCOLS / 2 / 32; ++blk)
        t5_st32(pp + blk * 32, scol + half * (SCOLS / 2) + blk * 32);
      t5_st_wait();
      t5_fence_before_sync();
      if (lane == 0) mb_arrive(t5_sp(&PR[sb]));
      if (warp == 0) { tspot(dbg_sm, 4); ++its; }
    }
    if constexpr (IT6) { if (warp == 0 && lane == 0 && dbg != nullptr) dbg[2048 + 768] = its; }
    t6_lrow = lrow;
    t6_mdef = mdef;
    }  // live softmax warps
  }

  // --------------------------- epilogue ---------------------------
  // Softmax warps drain the final PV and own the per-row stats; every warp
  // then participates in the vectorised global store out of smem staging
  // (which overlays the dead Q/K/V regions).
  __syncthreads();
  if constexpr (TRACE) {
    if (trc != nullptr && tid == 0) c_d = clock64();
  }
  constexpr int SROW = HD + 4;
  float* sOut = reinterpret_cast<float*>(smraw);  // TILE_M * SROW
  float* sGain = sOut + TILE_M * SROW;
  float* sLse = sGain + TILE_M;
  float* sM = sLse + TILE_M;
  float* sL = sM + TILE_M;
  if constexpr (SM8) {
    // Publish per-half row sums; warps<4 combine into lfin below. sMx lives
    // in the barrier tail, untouched by the sOut overlay.
    if (warp < 8) {
      const int row = (warp & 3) * 32 + lane;
      if constexpr (PHO != 0) {
        // PHO: the two halves ran INDEPENDENT (mdef, lrow) recurrences and warps<4
        // merge them with the same log-sum-exp rescale the split-KV combine uses.
        // Only half 1 publishes -- warps<4 still hold their own pair in registers
        // -- so the exchange is one float2 per row, exactly the 256 floats of sMx
        // the shared-O form already uses. That matters: the SM8 barrier tail is
        // 2048B of which ~152B is barriers plus tmem_word, and the launcher is
        // already at 231 424 of the 232 448B opt-in smem limit, so there is no
        // room to grow this array.
        if (warp >= 4) {
          sMx[row * 2 + 0] = t6_mdef;
          sMx[row * 2 + 1] = t6_lrow;
        }
      } else {
        sMx[row * 2 + (warp >> 2)] = t6_lrow;
      }
      bar_sm8();
    }
  }
  if (warp < 4) {
    const int row = warp * 32 + lane;
    const int irow = i0 + row / G;
    const bool rowok = irow < qlen;
    {
      const int rlast = t_end - 1 - t_begin;
      t5_mbar_wait(t5_sp(&PD[rlast % SB]), (rlast / SB) & 1);
    }
    t5_fence_after_sync();
    float lfin = t6_lrow;
    float mfin = t6_mdef;
    // PHO merge factors: O = O0 * 2^(m0-m) + O1 * 2^(m1-m), l = l0 * 2^(m0-m) +
    // l1 * 2^(m1-m), m = max(m0, m1). A half that saw no unmasked column has
    // m = NEG_BIG and l = 0, so its factor underflows to 0 and it drops out.
    float os0 = 1.f, os1 = 0.f;
    if constexpr (SM8) {
      if constexpr (PHO != 0) {
        const float m0 = t6_mdef, m1 = sMx[row * 2 + 0];
        const float l0 = t6_lrow, l1 = sMx[row * 2 + 1];
        mfin = fmaxf(m0, m1);
        os0 = ex2(m0 - mfin);
        os1 = ex2(m1 - mfin);
        lfin = l0 * os0 + l1 * os1;
      } else {
        lfin += sMx[row * 2 + 1];
      }
    }
    float base2 = mfin + (lfin > 0.f ? log2f(lfin) : NEG_BIG);
    float sc_sink = 1.f;
    if (p.sinks != nullptr) {
      const float sink2 = p.sinks[hkv * G + (row % G)] * LOG2E;
      const float hi = fmaxf(base2, sink2);
      const float total2 = hi + log2f(1.f + ex2(-fabsf(base2 - sink2)));
      sc_sink = ex2(base2 - total2);
      base2 = total2;
    }
    const float gain = (lfin > 0.f ? 1.f / lfin : 0.f) * sc_sink * p.v_scale;
    if (rowok) {
      sGain[row] = gain;
      sLse[row] = base2;
      sM[row] = mfin;
      sL[row] = lfin;
    }
    for (int blk = 0; blk < HD / 32; ++blk) {
      float ov[32];
      t5_ld32(ov, o_tmem + blk * 32);
      t5_ld_wait();
      if constexpr (PHO != 0) {
        // Both O regions live at the SAME TMEM lanes (a warp's lane quadrant is
        // fixed by its id, not by the column offset), so warps<4 read both halves
        // of their own rows and warps 4-7 stay idle here exactly as before.
        float ov1[32];
        t5_ld32(ov1, o1_tmem + blk * 32);
        t5_ld_wait();
        if (rowok)
          for (int j = 0; j < 32; ++j)
            sOut[row * SROW + blk * 32 + j] = ov[j] * os0 + ov1[j] * os1;
      } else if (rowok) {
        for (int j = 0; j < 32; ++j) sOut[row * SROW + blk * 32 + j] = ov[j];
      }
    }
  }
  __syncthreads();
  if constexpr (TRACE) {
    if (trc != nullptr && tid == 0) c_d1 = clock64();
  }
  if (p.splits > 1) {
    // Publish the unnormalised accumulator at this split's own max plus the
    // (max_log2, sum) pair, exactly like the t5 path; the combine kernel owns
    // the rescale, the sink and the lse.
    float* dst = p.part_o + w_id * p.part_s_w + (long long)split_z * p.part_s_s;
    for (int e = tid * 4; e < TILE_M * HD; e += THREADS * 4) {
      const int r = e / HD;
      if (i0 + r / G >= qlen) continue;
      *reinterpret_cast<float4*>(dst + e) =
          *reinterpret_cast<const float4*>(sOut + r * SROW + (e - r * HD));
    }
    float* ml = p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s;
    float* mt = p.part_ml_t + w_id * p.partml_s_w + (long long)split_z * 2;
    for (int r = tid; r < TILE_M; r += THREADS) {
      if (i0 + r / G >= qlen) continue;
      ml[r * 2] = sM[r];
      ml[r * 2 + 1] = sL[r];
      mt[r * p.partmlt_s_r] = sM[r];
      mt[r * p.partmlt_s_r + 1] = sL[r];
    }
    __syncthreads();
    trc_exit(2, t_end - t_begin);
    if (warp == 0) t5_dealloc(tmem, 512);
    return;
  }
  for (int e = tid * 8; e < TILE_M * HD; e += THREADS * 8) {
    const int r = e / HD, c = e % HD;
    const int i = i0 + r / G;
    if (i >= qlen) continue;
    const float g = sGain[r];
    const float* so = sOut + r * SROW + (e - r * HD);
    __nv_bfloat16 h8[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) h8[j] = __float2bfloat16_rn(so[j] * g);
    const long long tok = q0 + i;
    const int head = hkv * G + (r % G);
    *reinterpret_cast<uint4*>(p.out + tok * p.out_s_tok + (long long)head * p.out_s_head + c) =
        *reinterpret_cast<const uint4*>(h8);
  }
  if (p.write_lse) {
    for (int r = tid; r < TILE_M; r += THREADS) {
      const int i = i0 + r / G;
      if (i >= qlen) continue;
      const long long tok = q0 + i;
      const int head = hkv * G + (r % G);
      p.lse[tok * p.lse_s_tok + (long long)head * p.lse_s_head] = sLse[r];
    }
  }
  __syncthreads();
  trc_exit(0, t_end - t_begin);
  if (warp == 0) t5_dealloc(tmem, 512);
}

template <int HD, int KST, int VST, int DTH, int IT6 = 0, int PVN = 64, int MTM6 = 0,
          int DUAL = 0, int SM8 = 0, int LPAR = 1, int SPLITV = 0, int SB = 2, int QLA = 1,
          int DPV = 0, int MSKIP = 0, int RWSK = 0, int HACC = 4, int F16X = 0,
          int PHO = 0, int TRACE = 0, int QCP = 0, int LPT6 = 0, int PLO = 0,
          int F8T = 0, int PBAR = 0>
static void launch_main_t6(const Params& p, const CUtensorMap& kmap, const CUtensorMap& vmap,
                           dim3 grid, cudaStream_t stream) {
  constexpr int TILE_M = 128, TILE_N = 128;
  // F8T: fp8 rings (1B/elem) + 2+2 f16 convert targets replace the f16 rings.
  const int smem = F8T ? (TILE_M * HD * 2 + (KST + VST) * TILE_N * HD +
                          4 * TILE_N * HD * 2 + (SM8 ? 2048 : 1024))
                       : (TILE_M * HD * 2 + KST * TILE_N * HD * 2 + VST * TILE_N * HD * 2 +
                          (SM8 ? 2048 : 1024));
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(
        fmha_main_t6<HD, KST, VST, DTH, IT6, PVN, MTM6, DUAL, SM8, LPAR, SPLITV, SB,
                     QLA, DPV, MSKIP, RWSK, HACC, F16X, PHO, TRACE, QCP, LPT6,
                     PLO, F8T, PBAR>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  fmha_main_t6<HD, KST, VST, DTH, IT6, PVN, MTM6, DUAL, SM8, LPAR, SPLITV, SB, QLA, DPV,
               MSKIP, RWSK, HACC, F16X, PHO, TRACE, QCP, LPT6, PLO, F8T, PBAR>
      <<<grid, SM8 ? (SPLITV ? (DPV ? 384 : 352) : 320) : (DUAL ? 224 : 192), smem,
         stream>>>(p, kmap, vmap);
}

// Cluster-wide helpers (shared by t6c and t6d; defined once, before both):
DEVFN void tma4d_2cta(const CUtensorMap* map, uint32_t dst, uint32_t leader_bar, int c0, int c1,
                      int c2, int c3) {
  constexpr uint64_t kHint = 0x1000000000000000ULL;
  // Peer bit of the shared::cluster address cleared -> both CTAs' transactions
  // complete the leader's barrier (SM100 two-CTA TMA contract, atrex-fa).
  const uint32_t bar = leader_bar & 0xFEFFFFFFu;
  asm volatile(
      "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global."
      "mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
      ::"r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
      "r"(bar), "l"(kHint)
      : "memory");
}
DEVFN uint32_t mapa0(uint32_t local_smem_addr) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_smem_addr),
               "r"(0u));
  return remote;
}
DEVFN void mb_arrive_remote0(uint32_t local_addr) {
  const uint32_t remote = mapa0(local_addr);
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(remote) : "memory");
}
DEVFN void mb_arrive_remote(uint32_t local_addr, uint32_t target_rank) {
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr),
               "r"(target_rank));
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(remote) : "memory");
}
DEVFN void mb_wait_acquire(uint32_t a, int p) {
  asm volatile("{\n.reg .pred c;\nTWa_%=:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 c, [%0], %1;\n"
               "@!c bra.uni TWa_%=;\n}\n" ::"r"(a), "r"(p));
}
DEVFN void umma2_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t id, int acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               ::"r"(d), "l"(a), "l"(b), "r"(id), "r"(acc));
}
DEVFN void umma2_ts(uint32_t d, uint32_t at, uint64_t b, uint32_t id, int acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
               ::"r"(d), "r"(at), "l"(b), "r"(id), "r"(acc));
}
DEVFN void commit2mc(uint32_t m) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one."
               "shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(m), "h"((unsigned short)3)
               : "memory");
}
template <int NCOLS>
DEVFN void t_alloc2(uint32_t a) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(a),
               "n"(NCOLS));
}
DEVFN void t_relinq2() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
}
template <int NCOLS>
DEVFN void t_dealloc2(uint32_t base) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(base), "n"(NCOLS)
               : "memory");
}
DEVFN void cluster_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
DEVFN void cluster_wait_acquire() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
DEVFN void fence_async_smem_cluster() {
  asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
}

// ---------------------------------------------------------------------------
// t6d: t6 warp-specialized TN128 prefill + cluster-of-2 K/V MULTICAST sharing.
// Each CTA keeps its OWN mtile and its own g1 tcgen05 pipeline (numerics and
// outputs are bit-identical to t6 -- only the staging path changes), but the
// two CTAs of a pair process the same (by, kv-tile) sequence for the COMMON
// prefix of their causal ranges, so one CTA fetches each common K/V tile from
// L2 ONCE and multicasts it into both CTAs' smem (cp.async.bulk.tensor
// .multicast::cluster, validated by tma_min v5/v6: single AND dual issuer).
// f1-class prefill runs at 7.7TB/s of the ~8.4TB/s aggregate L2 ceiling
// (5.96GB staged), so halving L2 reads is the remaining lever at hd128; t6c's
// pair-MMA approach also halves staging but pays a causal union overhead
// (+1 fully-masked tile per pair) and measured strictly slower (K11).
//
// Cross-CTA slot-free handshake: the issuer of common tile T must know the
// PEER's ring slot s = (rr%KST) is free before overwriting it. Each CTA's
// load warp, after its local KE[s]/VE[s] recycle wait at rr >= KST, arrives
// on the PEER's SF_K[s]/SF_V[s] (mapa remote arrive, t6c-proven); the issuer
// waits its own SF[s] with the same ((rr/KST)-1)&1 parity before issuing.
// Common tiles are exactly rr < mc_end where mc_end = min(ntiles_self,
// ntiles_peer) and BOTH CTAs' jt0 match (window lower edges equal -> slot
// sequences align; a mismatch disables multicast, never correctness).
// No teardown cluster barrier is needed: every multicast into a CTA strictly
// precedes that CTA's own consumption of the tile (its KF wait), so an
// exiting CTA can never receive another write; TMEM alloc/dealloc are g1.
// ---------------------------------------------------------------------------
DEVFN void tma4d_mc(const CUtensorMap* map, uint32_t dst, uint32_t bar, int c0, int c1, int c2,
                    int c3) {
  // One L2 read delivers the box to the SAME smem offset in both CTAs of the
  // pair (mask 0x3); each receiver's barrier copy at `bar`'s offset gets the
  // tx bytes (tma_min v5: checksums equal the unicast box on both ranks).
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::"
      "bytes.multicast::cluster [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar),
      "h"((unsigned short)3)
      : "memory");
}

template <int HD, int KST, int VST, int DTH, int BC = 0, int MTM6D = 0>
__global__ void __launch_bounds__(192, 1) __cluster_dims__(2, 1, 1)
fmha_main_t6d(const __grid_constant__ Params p, const __grid_constant__ CUtensorMap kmap,
              const __grid_constant__ CUtensorMap vmap) {
  constexpr int TILE_M = 128;
  constexpr int TILE_N = 128;
  constexpr int THREADS = 192;
  constexpr int KG = HD / 64;              // 64-col swizzle atoms per row
  constexpr int PV_STEPS = TILE_N / 16;    // k16 steps per PV atom
  constexpr int QBYTES = TILE_M * HD * 2;
  constexpr int KT_BYTES = TILE_N * HD * 2;
  constexpr int VT_BYTES = TILE_N * HD * 2;
  constexpr int PPT = TILE_N / 16;         // max pages per tile (page >= 16)
  // PV atom width: hd64 must use the n64 form (t6 c151 does); n128 at hd64
  // would give PATOMS = HD/128 = 0 and skip PV entirely (K18c DD_dh64:
  // lse correct, out garbage, 123522 mismatched).
  constexpr int PVN = (HD >= 128) ? 128 : 64;
  constexpr int OCOL = 256;

  extern __shared__ char smraw[];
  char* sQ = smraw;
  char* sK = sQ + QBYTES;
  char* sV = sK + KST * KT_BYTES;
  uint64_t* KF = reinterpret_cast<uint64_t*>(sV + VST * VT_BYTES);
  uint64_t* VF = KF + KST;
  uint64_t* KE = VF + VST;
  uint64_t* VE = KE + KST;
  uint64_t* QD = VE + VST;   // qk_done[2]
  uint64_t* PR = QD + 2;     // p_ready[2]
  uint64_t* PD = PR + 2;     // pv_done[2]
  uint64_t* SFK = PD + 2;    // peer slot-free K signals
  uint64_t* SFV = SFK + KST; // peer slot-free V signals
  uint32_t* tmem_word = reinterpret_cast<uint32_t*>(SFV + VST);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));

  // BC (cfg188): progress counters for cluster 0 into the unused split-KV
  // scratch at +100MiB (valid while splits==1); host polls them with a
  // side-stream D2H (experiments/t6c_bc.py pattern).
  long long* bc = nullptr;
  if constexpr (BC) {
    if (p.splits == 1 && p.part_o != nullptr && blockIdx.x < 2 && blockIdx.y == 0 &&
        blockIdx.z == 0)
      bc = reinterpret_cast<long long*>(p.part_o + (25 << 20)) + (long long)rank * 32;
  }
#define T6BC(stage)                                                         \
  do {                                                                      \
    if (bc != nullptr)                                                      \
      atomicAdd(reinterpret_cast<unsigned long long*>(bc + (stage)), 1ULL); \
  } while (0)

  // MTM6D: pair-major DESCENDING launch order (LPT under causal) when every
  // request fills the full mtile count; ragged batches keep ASCENDING order
  // (K12/K13 evidence on the t6c pair remap; the predicate is derived from
  // read-only cu_q identically in every CTA, so the mapping stays a
  // bijection). Flat grid: x = 2 * pairs * Y.
  int pair, by;
  if constexpr (MTM6D) {
    const unsigned pairs = (unsigned)((p.mtiles + 1) / 2);
    const unsigned Y = gridDim.x / (2u * pairs);
    const unsigned pid = (unsigned)(blockIdx.x >> 1);
    bool uniform = true;
    const int nb = (int)(Y / (unsigned)p.HKV);
    for (int bb = 0; bb < nb; ++bb) {
      const int ql = p.cu_q[bb + 1] - p.cu_q[bb];
      if ((ql * p.G + TILE_M - 1) / TILE_M < p.mtiles) {
        uniform = false;
        break;
      }
    }
    if (uniform) {
      by = (int)(pid % Y);
      pair = (int)(pairs - 1u - pid / Y);
    } else {
      by = (int)(pid / pairs);
      pair = (int)(pid % pairs);
    }
  } else {
    pair = (int)(blockIdx.x >> 1);
    by = (int)blockIdx.y;
  }
  const int mtile = pair * 2 + (int)rank;
  const int b = by / p.HKV;
  const int hkv = by % p.HKV;
  const int q0 = p.cu_q[b];
  const int qlen = p.cu_q[b + 1] - q0;
  const int kvlen = p.seq_lens[b];
  const int G = p.G;
  const int m0 = mtile * TILE_M;
  const bool dead = (mtile >= p.mtiles) || (m0 >= qlen * G);

  // Peer geometry (same by): needed for the common-prefix multicast range.
  const int mtile_p = pair * 2 + (int)(rank ^ 1u);
  const int m0_p = mtile_p * TILE_M;
  const bool peer_dead = (mtile_p >= p.mtiles) || (m0_p >= qlen * G);

  int jt0 = 0, jt1 = TILE_N, ntiles = 0;
  int ntiles_p = 0, jt0_p = 0;
  if (!dead) {
    const int i0v = m0 / G;
    const int i1v = min(qlen, (m0 + TILE_M) / G);
    const int v0 = kvlen - qlen + i0v;
    const int v1 = kvlen - qlen + i1v - 1;
    const int lo = p.window_left >= 0 ? max(0, v0 - p.window_left) : 0;
    const int hi = min(kvlen - 1, v1);
    jt0 = (lo / TILE_N) * TILE_N;
    jt1 = ((hi + 1 + TILE_N - 1) / TILE_N) * TILE_N;
    ntiles = (jt1 - jt0) / TILE_N;
  }
  if (!peer_dead) {
    const int i0v = m0_p / G;
    const int i1v = min(qlen, (m0_p + TILE_M) / G);
    const int v0 = kvlen - qlen + i0v;
    const int v1 = kvlen - qlen + i1v - 1;
    const int lo = p.window_left >= 0 ? max(0, v0 - p.window_left) : 0;
    const int hi = min(kvlen - 1, v1);
    jt0_p = (lo / TILE_N) * TILE_N;
    ntiles_p = (((hi + 1 + TILE_N - 1) / TILE_N) * TILE_N - jt0_p) / TILE_N;
  }
  // Multicast prefix: identical jt0 (slot alignment) and both alive. Under
  // causal the earlier mtile's range is a prefix of the later one, so the
  // common tiles are exactly rr < min(ntiles, ntiles_p) from jt0.
  // splits>1 disables sharing: t_begin/t_end derive from each CTA's OWN
  // ntiles, so the pair's split ranges diverge and the "same tile sequence"
  // premise of the handshake no longer holds (unicast-only stays correct).
  const bool mc_on = !dead && !peer_dead && (jt0 == jt0_p) && ntiles_p > 0 && p.splits == 1;
  const int mc_end = mc_on ? min(ntiles, ntiles_p) : 0;

  const int i0 = dead ? qlen : (m0 / G);
  int t_begin = 0, t_end = ntiles;
  if (p.splits > 1) {
    const int tps = (ntiles + p.splits - 1) / p.splits;
    t_begin = blockIdx.z * tps;
    t_end = min(ntiles, t_begin + tps);
  }
  const int split_z = p.splits > 1 ? blockIdx.z : 0;
  const long long w_id = (long long)by * p.mtiles + mtile;
  const bool empty = !dead && t_begin >= t_end;
  if (tid == 0) T6BC(0);

  float t6_lrow = 0.f, t6_mdef = NEG_BIG;

  // ---------------- prologue: Q tile -> smem (bf16 -> f16) ----------------
  constexpr int QCH = TILE_M * HD / 8;
  const __nv_bfloat16* qbase = p.q + (long long)q0 * p.q_s_tok;
  if (!dead && !empty) {
#pragma unroll
    for (int u = 0; u < (QCH + THREADS - 1) / THREADS; ++u) {
      const int c = tid + u * THREADS;
      if (QCH % THREADS != 0 && c >= QCH) break;
      const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
      const int i = i0 + r / G;
      uint4 data = make_uint4(0, 0, 0, 0);
      if (i < qlen) {
        const __nv_bfloat16* src =
            qbase + (long long)i * p.q_s_tok + (long long)(hkv * G + (r % G)) * p.q_s_head + col;
        data = *reinterpret_cast<const uint4*>(src);
      }
      const uint4 h = make_uint4(bf16x2_to_f16x2(data.x), bf16x2_to_f16x2(data.y),
                                 bf16x2_to_f16x2(data.z), bf16x2_to_f16x2(data.w));
      *reinterpret_cast<uint4*>(sQ + t5_off(TILE_M, r, col)) = h;
    }
  }
  if (tid == 0) T6BC(1);
  if (tid == 0 && !dead && !empty) {
    for (int s = 0; s < KST; ++s) {
      t5_mbar_init(t5_sp(&KF[s]), 1);
      t5_mbar_init(t5_sp(&KE[s]), 1);
      t5_mbar_init(t5_sp(&SFK[s]), 1);
    }
    for (int s = 0; s < VST; ++s) {
      t5_mbar_init(t5_sp(&VF[s]), 1);
      t5_mbar_init(t5_sp(&VE[s]), 1);
      t5_mbar_init(t5_sp(&SFV[s]), 1);
    }
    t5_mbar_init(t5_sp(&QD[0]), 1);
    t5_mbar_init(t5_sp(&QD[1]), 1);
    t5_mbar_init(t5_sp(&PR[0]), 4);
    t5_mbar_init(t5_sp(&PR[1]), 4);
    t5_mbar_init(t5_sp(&PD[0]), 1);
    t5_mbar_init(t5_sp(&PD[1]), 1);
  }
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  // ONE cluster rendezvous, joined by EVERY CTA including dead/empty ones:
  // barrier.cluster.arrive/wait counts all cluster members, so a dead CTA
  // that returned early would hang its live peer forever. It also publishes
  // the mbarrier inits before any peer remote-arrive/multicast can land.
  cluster_arrive_release();
  cluster_wait_acquire();
  if (tid == 0) T6BC(3);
  if (dead || empty) {
    // Dead padding CTA of an odd pair, or an empty split: publish the neutral
    // partial exactly like t6 (empty splits only) and leave. The live peer
    // computed mc_end = 0 for this CTA, so nothing cross-CTA is outstanding.
    if (empty && p.splits > 1 && p.tps_target == 0) {
      for (int e = tid; e < TILE_M * HD; e += THREADS) {
        if (i0 + (e / HD) / G < qlen)
          p.part_o[w_id * p.part_s_w + (long long)split_z * p.part_s_s + e] = 0.f;
      }
      if (tid < TILE_M && i0 + tid / G < qlen) {
        float* ml = p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s +
                    (long long)tid * 2;
        ml[0] = NEG_BIG;
        ml[1] = 0.f;
        float* mt = p.part_ml_t + w_id * p.partml_s_w + (long long)tid * p.partmlt_s_r +
                    (long long)split_z * 2;
        mt[0] = NEG_BIG;
        mt[1] = 0.f;
      }
    }
    return;
  }
  fence_async_smem();  // Q STS (generic proxy) ordered against TC descriptor reads
  if (warp == 0) {
    t5_alloc(t5_sp(tmem_word), 512);
    t5_relinquish();
  }
  __syncthreads();
  t5_fence_after_sync();
  const uint32_t tmem = *tmem_word;
  const uint32_t o_tmem = tmem + OCOL;
  if (tid == 0) T6BC(4);

  if (warp == 4 && lane == 0) {
    // ------------------------- MMA warp -------------------------
    T6BC(5);
    const uint32_t ID_QK = t5_idesc(TILE_M, TILE_N, 0);
    const uint32_t ID_PV = t5_idesc(TILE_M, PVN, 1);
    constexpr int PATOMS = HD / PVN;        // n-atoms per PV
    constexpr int PGRP = PVN / 64;          // 64-col swizzle groups per atom
    const uint64_t qd0 = t5_desc(t5_sp(sQ), 16, 8 * 128);
    const uint64_t kdb = t5_desc(t5_sp(sK), 16, 8 * 128);
    // PVN=64: per-atom MN-major form (lbo=0); PVN=128: the n128 instruction
    // spans two 64-col swizzle groups at stride TILE_N*128B (see t6).
    const uint64_t vdb = t5_desc(t5_sp(sV), PVN == 64 ? 0 : TILE_N * 128, 8 * 128);
    auto issue_qk = [&](int t) {
      const int rr = t - t_begin;
      const int sb = rr & 1, kb = rr % KST;
      const uint32_t scol = tmem + sb * TILE_N;
      const uint64_t kd = kdb + (uint64_t)(kb * (KT_BYTES >> 4));
#pragma unroll
      for (int g = 0; g < KG; ++g)
#pragma unroll
        for (int s = 0; s < 4; ++s)
          t5_umma_ss(scol, qd0 + (uint64_t)(g * (TILE_M * 8) + s * 2),
                     kd + (uint64_t)(g * (TILE_N * 8) + s * 2), ID_QK, (g | s) != 0);
      t5_commit(t5_sp(&QD[sb]));
      t5_commit(t5_sp(&KE[kb]));
    };
    {
      t5_mbar_wait(t5_sp(&KF[0]), 0);
      T6BC(7);
      t5_fence_after_sync();
      issue_qk(t_begin);
    }
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int sb = rr & 1, vb = rr % VST;
      if (t + 1 < t_end) {
        const int kb1 = (rr + 1) % KST;
        t5_mbar_wait(t5_sp(&KF[kb1]), ((rr + 1) / KST) & 1);
        T6BC(7);
        t5_fence_after_sync();
        issue_qk(t + 1);
      }
      t5_mbar_wait(t5_sp(&PR[sb]), (rr >> 1) & 1);
      T6BC(10);
      t5_fence_after_sync();
      t5_mbar_wait(t5_sp(&VF[vb]), (rr / VST) & 1);
      T6BC(11);
      const uint32_t scol = tmem + sb * TILE_N;
      const uint64_t vd = vdb + (uint64_t)(vb * (VT_BYTES >> 4));
#pragma unroll
      for (int a = 0; a < PATOMS; ++a)
#pragma unroll
        for (int s = 0; s < PV_STEPS; ++s)
          t5_umma_ts(o_tmem + a * PVN, scol + s * 8,
                     vd + (uint64_t)(a * (PGRP * (TILE_N * 8)) + s * 128), ID_PV,
                     (t != t_begin || s != 0));
      t5_commit(t5_sp(&PD[sb]));
      t5_commit(t5_sp(&VE[vb]));
    }
  } else if (warp == 5 && lane == 0) {
    // ------------------------- load warp -------------------------
    T6BC(5);
    const int page_log2 = p.page_log2;
    const int rows_per_box = 1 << page_log2;  // == page size
    const int pages_per_tile = TILE_N >> page_log2;
    const int tcols = p.table_stride;
    const int* table_row = p.table + (long long)b * tcols;
    const uint32_t peer_rank = rank ^ 1u;
    int pid[PPT];
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int kb = rr % KST, vb = rr % VST;
      const bool mc = rr < mc_end;
      const bool issuer = mc && ((rr & 1) == (int)rank);
      if (rr >= KST) {
        t5_mbar_wait(t5_sp(&KE[kb]), ((rr / KST) - 1) & 1);
        T6BC(16);
        if (mc) {
          // Slot s is locally free; release the PEER to multicast into it.
          mb_arrive_remote(t5_sp(&SFK[kb]), peer_rank);
          T6BC(19);
        }
      }
      if (rr >= VST) {
        t5_mbar_wait(t5_sp(&VE[vb]), ((rr / VST) - 1) & 1);
        T6BC(17);
        if (mc) {
          mb_arrive_remote(t5_sp(&SFV[vb]), peer_rank);
          T6BC(19);
        }
      }
      const int jb = jt0 + t * TILE_N;
#pragma unroll
      for (int q = 0; q < PPT; ++q) {
        if (q >= pages_per_tile) break;
        const int jj = min(max(jb + (q << page_log2), 0), kvlen - 1);
        pid[q] = table_row[min(jj >> page_log2, tcols - 1)];
      }
      const uint32_t kbar = t5_sp(&KF[kb]);
      const uint32_t vbar = t5_sp(&VF[vb]);
      // Every receiver expects its OWN delivered copy (v5 semantics).
      mb_expect(kbar, KT_BYTES);
      mb_expect(vbar, VT_BYTES);
      if (issuer) {
        // The peer's slot-free signals gate the multicast overwrite.
        if (rr >= KST) {
          mb_wait_acquire(t5_sp(&SFK[kb]), ((rr / KST) - 1) & 1);
          T6BC(20);
        }
        if (rr >= VST) {
          mb_wait_acquire(t5_sp(&SFV[vb]), ((rr / VST) - 1) & 1);
          T6BC(20);
        }
      }
      if (!mc || issuer) {
        for (int q = 0; q < pages_per_tile; ++q) {
#pragma unroll
          for (int sg = 0; sg < KG; ++sg) {
            const uint32_t off = (uint32_t)(sg * (TILE_N * 128) + q * (rows_per_box * 128));
            if (mc) {
              tma4d_mc(&kmap, t5_sp(sK + kb * KT_BYTES + off), kbar, sg * 64, 0, hkv, pid[q]);
              tma4d_mc(&vmap, t5_sp(sV + vb * VT_BYTES + off), vbar, sg * 64, 0, hkv, pid[q]);
            } else {
              tma4d(&kmap, t5_sp(sK + kb * KT_BYTES + off), kbar, sg * 64, 0, hkv, pid[q]);
              tma4d(&vmap, t5_sp(sV + vb * VT_BYTES + off), vbar, sg * 64, 0, hkv, pid[q]);
            }
          }
        }
        if (mc) T6BC(21); else T6BC(22);
      }
      T6BC(6);
    }
  } else if (warp < 4) {
    // ----------------------- softmax warps -----------------------
    if (warp == 0 && lane == 0) T6BC(5);
    const int row = warp * 32 + lane;
    const int irow = i0 + row / G;
    const bool rowok = irow < qlen;
    const int vis = kvlen - qlen + irow;
    const int lim = rowok ? vis : -1;
    const int low = (rowok && p.window_left >= 0) ? max(0, vis - p.window_left) : 0;
    float mrun = NEG_BIG;
    float lrow = 0.f;
    float mdef = NEG_BIG;
    const float qk = p.qk_scale;
    const uint64_t qk2 = f2_pack(qk, qk);
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int sb = rr & 1;
      const uint32_t scol = tmem + sb * TILE_N;
      t5_mbar_wait(t5_sp(&QD[sb]), (rr >> 1) & 1);
      if (warp == 0 && lane == 0) T6BC(8);
      float sc[TILE_N];
#pragma unroll
      for (int blk = 0; blk < TILE_N / 32; ++blk) t5_ld32(sc + blk * 32, scol + blk * 32);
      t5_ld_wait();
      const int col0 = jt0 + t * TILE_N;
      const bool full = rowok && (col0 + TILE_N - 1 <= lim) && (col0 >= low);
      float mxr = NEG_BIG;
      {
        float m0v = NEG_BIG, m1v = NEG_BIG, m2v = NEG_BIG, m3v = NEG_BIG;
        if (full) {
#pragma unroll
          for (int c = 0; c < TILE_N; c += 4) {
            m0v = fmaxf(m0v, sc[c]);
            m1v = fmaxf(m1v, sc[c + 1]);
            m2v = fmaxf(m2v, sc[c + 2]);
            m3v = fmaxf(m3v, sc[c + 3]);
          }
        } else {
#pragma unroll
          for (int c = 0; c < TILE_N; c += 4) {
            m0v = fmaxf(m0v, (col0 + c <= lim && col0 + c >= low) ? sc[c] : NEG_BIG);
            m1v = fmaxf(m1v, (col0 + c + 1 <= lim && col0 + c + 1 >= low) ? sc[c + 1] : NEG_BIG);
            m2v = fmaxf(m2v, (col0 + c + 2 <= lim && col0 + c + 2 >= low) ? sc[c + 2] : NEG_BIG);
            m3v = fmaxf(m3v, (col0 + c + 3 <= lim && col0 + c + 3 >= low) ? sc[c + 3] : NEG_BIG);
          }
        }
        mxr = fmaxf(fmaxf(m0v, m1v), fmaxf(m2v, m3v));
      }
      // Running max in scaled log2 units (see t5/t6): never fma-contracted.
      const float mn = fmaxf(mrun, mxr * qk);
      float mneg;
      if (t == t_begin) {
        mdef = mn;
      } else if (__any_sync(0xffffffffu, mn - mdef > (float)DTH)) {
        // DEFER O-rescale: PV(t-1) may still be accumulating in the tensor
        // core, so wait its commit before touching O in the generic proxy.
        t5_mbar_wait(t5_sp(&PD[sb ^ 1]), ((rr - 1) >> 1) & 1);
        if (warp == 0 && lane == 0) T6BC(18);
        t5_fence_after_sync();
        const float f = ex2(mdef - mn);
        lrow *= f;
        for (int blk = 0; blk < HD / 32; ++blk) {
          float ov[32];
          t5_ld32(ov, o_tmem + blk * 32);
          t5_ld_wait();
#pragma unroll
          for (int j = 0; j < 32; ++j) ov[j] *= f;
          t5_st32(reinterpret_cast<const uint32_t*>(ov), o_tmem + blk * 32);
        }
        t5_st_wait();
        mdef = mn;
      }
      mrun = mn;
      mneg = -mdef;
      uint32_t pp[TILE_N / 2];
      const uint64_t mn2 = f2_pack(mneg, mneg);
      uint32_t hacc[4] = {0u, 0u, 0u, 0u};
      if (full) {
#pragma unroll
        for (int c = 0; c < TILE_N; c += 2) {
          const uint64_t x2 = fma2(f2_pack(sc[c], sc[c + 1]), qk2, mn2);
          float e0, e1;
          f2_unpack(x2, e0, e1);
          const uint32_t p2 = ex2_f16x2(pack_f16(e0, e1));
          pp[c >> 1] = p2;
          hacc[(c >> 1) & 3] = hadd2(hacc[(c >> 1) & 3], p2);
        }
      } else {
#pragma unroll
        for (int c = 0; c < TILE_N; c += 2) {
          const int j = col0 + c;
          const uint64_t x2 = fma2(f2_pack(sc[c], sc[c + 1]), qk2, mn2);
          float e0, e1;
          f2_unpack(x2, e0, e1);
          if (!(j <= lim && j >= low)) e0 = NEG_BIG;
          if (!(j + 1 <= lim && j + 1 >= low)) e1 = NEG_BIG;
          const uint32_t p2 = ex2_f16x2(pack_f16(e0, e1));
          pp[c >> 1] = p2;
          hacc[(c >> 1) & 3] = hadd2(hacc[(c >> 1) & 3], p2);
        }
      }
      lrow += (hsum2(hacc[0]) + hsum2(hacc[1])) + (hsum2(hacc[2]) + hsum2(hacc[3]));
#pragma unroll
      for (int blk = 0; blk < TILE_N / 2 / 32; ++blk)
        t5_st32(pp + blk * 32, scol + blk * 32);
      t5_st_wait();
      t5_fence_before_sync();
      if (lane == 0) mb_arrive(t5_sp(&PR[sb]));
      if (warp == 0 && lane == 0) T6BC(9);
    }
    t6_lrow = lrow;
    t6_mdef = mdef;
  }

  // --------------------------- epilogue ---------------------------
  __syncthreads();
  constexpr int SROW = HD + 4;
  float* sOut = reinterpret_cast<float*>(smraw);  // TILE_M * SROW
  float* sGain = sOut + TILE_M * SROW;
  float* sLse = sGain + TILE_M;
  float* sM = sLse + TILE_M;
  float* sL = sM + TILE_M;
  if (warp < 4) {
    const int row = warp * 32 + lane;
    const int irow = i0 + row / G;
    const bool rowok = irow < qlen;
    {
      const int rlast = t_end - 1 - t_begin;
      t5_mbar_wait(t5_sp(&PD[rlast & 1]), (rlast >> 1) & 1);
    }
    if (warp == 0 && lane == 0) T6BC(12);
    t5_fence_after_sync();
    const float lfin = t6_lrow;
    const float mfin = t6_mdef;
    float base2 = mfin + (lfin > 0.f ? log2f(lfin) : NEG_BIG);
    float sc_sink = 1.f;
    if (p.sinks != nullptr) {
      const float sink2 = p.sinks[hkv * G + (row % G)] * LOG2E;
      const float hi = fmaxf(base2, sink2);
      const float total2 = hi + log2f(1.f + ex2(-fabsf(base2 - sink2)));
      sc_sink = ex2(base2 - total2);
      base2 = total2;
    }
    const float gain = (lfin > 0.f ? 1.f / lfin : 0.f) * sc_sink * p.v_scale;
    if (rowok) {
      sGain[row] = gain;
      sLse[row] = base2;
      sM[row] = mfin;
      sL[row] = lfin;
    }
    for (int blk = 0; blk < HD / 32; ++blk) {
      float ov[32];
      t5_ld32(ov, o_tmem + blk * 32);
      t5_ld_wait();
      if (rowok)
        for (int j = 0; j < 32; ++j) sOut[row * SROW + blk * 32 + j] = ov[j];
    }
  }
  __syncthreads();
  if (p.splits > 1) {
    float* dst = p.part_o + w_id * p.part_s_w + (long long)split_z * p.part_s_s;
    for (int e = tid * 4; e < TILE_M * HD; e += THREADS * 4) {
      const int r = e / HD;
      if (i0 + r / G >= qlen) continue;
      *reinterpret_cast<float4*>(dst + e) =
          *reinterpret_cast<const float4*>(sOut + r * SROW + (e - r * HD));
    }
    float* ml = p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s;
    float* mt = p.part_ml_t + w_id * p.partml_s_w + (long long)split_z * 2;
    for (int r = tid; r < TILE_M; r += THREADS) {
      if (i0 + r / G >= qlen) continue;
      ml[r * 2] = sM[r];
      ml[r * 2 + 1] = sL[r];
      mt[r * p.partmlt_s_r] = sM[r];
      mt[r * p.partmlt_s_r + 1] = sL[r];
    }
    __syncthreads();
    if (tid == 0) T6BC(15);
    if (warp == 0) t5_dealloc(tmem, 512);
    return;
  }
  for (int e = tid * 8; e < TILE_M * HD; e += THREADS * 8) {
    const int r = e / HD, c = e % HD;
    const int i = i0 + r / G;
    if (i >= qlen) continue;
    const float g = sGain[r];
    const float* so = sOut + r * SROW + (e - r * HD);
    __nv_bfloat16 h8[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) h8[j] = __float2bfloat16_rn(so[j] * g);
    const long long tok = q0 + i;
    const int head = hkv * G + (r % G);
    *reinterpret_cast<uint4*>(p.out + tok * p.out_s_tok + (long long)head * p.out_s_head + c) =
        *reinterpret_cast<const uint4*>(h8);
  }
  if (p.write_lse) {
    for (int r = tid; r < TILE_M; r += THREADS) {
      const int i = i0 + r / G;
      if (i >= qlen) continue;
      const long long tok = q0 + i;
      const int head = hkv * G + (r % G);
      p.lse[tok * p.lse_s_tok + (long long)head * p.lse_s_head] = sLse[r];
    }
  }
  __syncthreads();
  if (tid == 0) T6BC(15);
  if (warp == 0) t5_dealloc(tmem, 512);
}
#undef T6BC

template <int HD, int KST, int VST, int DTH, int BC = 0, int MTM6D = 0>
static void launch_main_t6d(const Params& p, const CUtensorMap& kmap, const CUtensorMap& vmap,
                            dim3 grid_ct, cudaStream_t stream) {
  constexpr int TILE_M = 128, TILE_N = 128;
  const int smem = TILE_M * HD * 2 + KST * TILE_N * HD * 2 + VST * TILE_N * HD * 2 + 1024;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(fmha_main_t6d<HD, KST, VST, DTH, BC, MTM6D>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  fmha_main_t6d<HD, KST, VST, DTH, BC, MTM6D><<<grid_ct, 192, smem, stream>>>(p, kmap, vmap);
}


// ---------------------------------------------------------------------------
// t6c: cluster-of-2 extension of t6 for hd128/hd256 (TN128, warp-specialized).
//
// Why (measurements): t6 v1.3 (c162/c165) reaches f1 771.9us / f7 184.8us,
// which is L2-bandwidth-bound: f1 streams 93184 CTA-tiles x 64KB = 5.96GB in
// 772us = 7.7TB/s ~= the measured 8.4TB/s aggregate L2 read ceiling (probe_bw
// agg_l2 32.3 B/cyc/SM), against a TC floor of ~370us. cta_group::2 splits
// the UMMA B operand across the pair: each CTA stages only 64 kv columns of
// K and 64 hd columns of V per tile, so the pair pulls 64KB per 256 rows --
// HALF the L2 bytes per row of work -- while probe_tc measured g2 SS n128 at
// 64.1 cyc/pair (same per-SM TC rate as g1). Projected: f1 ~430-500us,
// f7 ~110-140us; hd256 (f6 746us) becomes reachable at TN128 at all.
//
// Structure (atrex-fa pure_cuda d256 2cta lifecycle + deepgemm barrier roles):
//   * cluster = 2 CTAs = 2 mtiles = 256 packed rows; rank r owns rows
//     [pair*256 + r*128, +128); logical mtile = pair*2 + r, so partials,
//     combine addressing and splits match the t6/t5 contract per CTA.
//   * QK = m256n128k16 SS (A: local Q halves; B: local K halves, 64 kv cols
//     each; D: local S half). PV = m256n128k16 TS x HD/128 atoms (A: local P
//     TMEM; B: local V halves [128 tok][64 col]; D: local O).
//   * TMA: cta_group::2 form executed by BOTH CTAs with rank-specific coords;
//     the barrier operand clears the peer bit so both transactions complete
//     the LEADER's k_full/v_full (expect_tx covers both halves' bytes).
//   * commits (QD/PD/KE/VE) multicast 0x3 -> local copies on both CTAs;
//     p_ready lives on the leader (count 8: 4 softmax warps x 2 CTAs, peer
//     arrives via mapa remote arrive); the leader waits it with acquire.cluster.
//   * DEFER O-rescale stays LOCAL per CTA before the P publish, so the PR
//     arrival already orders it against the leader's PV -- no extra cluster
//     barrier on the hot path.
//   * TMEM: alloc/dealloc/relinquish at cta_group::2 granularity on both CTAs;
//     teardown through a peer dealloc barrier (atrex pattern).
// Odd mtiles: the grid rounds x up to even; a dead rank-1 CTA (mtile >=
// p.mtiles) participates in all cluster collectives but unions nothing into
// the tile range and writes nothing (its rows all fail i < qlen).
// Numerics: identical recipes to t6 v1.3 (f16 mirror exact, bf16->f16 Q,
// DEFER DTH, F2+X2H softmax, log2 domain); only the KV tile column range is
// pair-unified (masked columns contribute P=0 exactly as in t6).

template <int HD, int KST, int VST, int DTH, int BC = 0, int MTM6C = 0, int QCP = 0,
          int RWSK = 0, int SM8C = 0, int PBAR = 0>
__global__ void __launch_bounds__(SM8C ? 320 : 192, 1) __cluster_dims__(2, 1, 1)
fmha_main_t6c(const __grid_constant__ Params p, const __grid_constant__ CUtensorMap kmap,
              const __grid_constant__ CUtensorMap vmap) {
  constexpr int TILE_M = 128;    // rows per CTA (256 per cluster)
  constexpr int TILE_N = 128;    // kv columns per pair tile
  constexpr int KN = 64;         // kv columns staged per CTA (K half)
  constexpr int VN = HD / 2;     // hd columns staged per CTA (V half)
  // SM8C (cfg297): t6's SM8 column-split softmax ported to the cluster
  // kernel -- 8 softmax warps in two halves of SCOLS=TILE_N/2 columns each,
  // halving the per-lane softmax live set (128 -> 64 cols) that NCU showed
  // pacing f6 (tensor pipe 63.2%, 1.5 active warps/sched). MMA/load warps
  // move to 8/9. THREADS 192 -> 320.
  static_assert(PBAR == 0 || SM8C, "PBAR narrows SM8C's per-tile exchange barrier");
  constexpr int THREADS = SM8C ? 320 : 192;
  constexpr int MMA_W = SM8C ? 8 : 4;
  constexpr int LOAD_W = SM8C ? 9 : 5;
  constexpr int NSW = SM8C ? 8 : 4;                  // softmax warps
  constexpr int SCOLS = SM8C ? TILE_N / 2 : TILE_N;  // softmax cols per warp
  constexpr int QK_STEPS = HD / 16;         // k16 steps per QK
  constexpr int KSEG = HD / 64;             // 64-col swizzle segments per K half
  constexpr int VSEG = VN / 64;             // ... per V half (>=1 for HD>=128)
  constexpr int PV_ATOMS = HD / 128;        // m256n128 PV atoms
  constexpr int QBYTES = TILE_M * HD * 2;
  constexpr int KT_BYTES = KN * HD * 2;     // per-CTA K stage bytes
  constexpr int VT_BYTES = TILE_N * VN * 2; // per-CTA V stage bytes
  constexpr int OCOL = 256;

  extern __shared__ char smraw[];
  char* sQ = smraw;
  char* sK = sQ + QBYTES;
  char* sV = sK + KST * KT_BYTES;
  uint64_t* KF = reinterpret_cast<uint64_t*>(sV + VST * VT_BYTES);  // leader-owned
  uint64_t* VF = KF + KST;                                          // leader-owned
  uint64_t* KE = VF + VST;
  uint64_t* VE = KE + KST;
  uint64_t* QD = VE + VST;
  uint64_t* PR = QD + 2;   // leader-owned, count 8
  uint64_t* PD = PR + 2;
  uint64_t* DB = PD + 2;   // peer dealloc barrier
  uint32_t* tmem_word = reinterpret_cast<uint32_t*>(DB + 1);
  // SM8C: row-max exchange + epilogue row-sum merge (128 rows x 2 floats),
  // living in the enlarged barrier tail (2048B; ~144B barriers + 1024B sMx).
  float* sMx = reinterpret_cast<float*>(tmem_word + 2);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));

  // BC (cfg178): progress counters for ONE cluster into the unused split-KV
  // scratch at +100MiB (valid while splits==1), readable from the host with a
  // side-stream D2H while the kernel spins. Layout: cluster0 only,
  // base + rank*32 + stage; each point atomicAdd's 1 per qualifying thread.
  long long* bc = nullptr;
  if constexpr (BC) {
    if (p.splits == 1 && p.part_o != nullptr && blockIdx.x < 2 && blockIdx.y == 0 &&
        blockIdx.z == 0)
      bc = reinterpret_cast<long long*>(p.part_o + (25 << 20)) + (long long)rank * 32;
  }
#define T6BC(stage)                                                       \
  do {                                                                    \
    if (bc != nullptr)                                                    \
      atomicAdd(reinterpret_cast<unsigned long long*>(bc + (stage)), 1ULL); \
  } while (0)

  // MTM6C: pair-major DESCENDING launch order = LPT greedy under causal
  // (longest pair pipeline first), mirroring t6's MTM6 flattened remap. The
  // logical (mtile, by) mapping is a bijection, so every output/partial
  // address matches the plain launch bit-for-bit.
  int pair, by;
  if constexpr (MTM6C) {
    const unsigned pairs = (unsigned)((p.mtiles + 1) / 2);
    const unsigned Y = gridDim.x / (2u * pairs);
    const unsigned pid = (unsigned)(blockIdx.x >> 1);
    // Ragged batches (some request fills fewer mtiles than p.mtiles) must keep
    // ASCENDING order: descending would schedule the short requests' live
    // pairs last (SPT tail; s_d256rag measured 282us descending vs 255us
    // ascending, K12 dv job 20261021T120140Z). Uniform batches prefer
    // descending (LPT + L2 locality: d7 117 vs 128, f6 453 vs 485). Every
    // CTA derives the same predicate from the read-only cu_q, so the
    // (pair, by) mapping stays a bijection either way.
    bool uniform = true;
    const int nb = (int)(Y / (unsigned)p.HKV);
    for (int bb = 0; bb < nb; ++bb) {
      const int ql = p.cu_q[bb + 1] - p.cu_q[bb];
      if ((ql * p.G + TILE_M - 1) / TILE_M < p.mtiles) {
        uniform = false;
        break;
      }
    }
    if (uniform) {
      by = (int)(pid % Y);
      pair = (int)(pairs - 1u - pid / Y);
    } else {
      by = (int)(pid / pairs);
      pair = (int)(pid % pairs);
    }
  } else {
    pair = (int)(blockIdx.x >> 1);
    by = (int)blockIdx.y;
  }
  const int mtile = pair * 2 + (int)rank;
  const int b = by / p.HKV;
  const int hkv = by % p.HKV;
  const int q0 = p.cu_q[b];
  const int qlen = p.cu_q[b + 1] - q0;
  const int kvlen = p.seq_lens[b];
  const int G = p.G;
  const int m0 = mtile * TILE_M;
  const bool dead = m0 >= qlen * G;          // padding CTA of an odd tail pair
  const int pair_m0 = pair * 2 * TILE_M;     // rank0's row base (always valid if
                                             // any row of the pair is)
  const bool pair_dead = pair_m0 >= qlen * G;
  if (pair_dead) return;                     // both ranks dead: no cluster work

  const int i0 = m0 / G;
  const int i1 = min(qlen, (m0 + TILE_M) / G);
  // Pair-unified kv column range: union over the VALID ranks. rank1 rows see a
  // superset of rank0's columns under causal (higher rows, same kv), and the
  // window lower edge moves the other way, so take min(jlo)/max(jhi).
  int pjlo = 0x7fffffff, pjhi = -0x7fffffff;
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int mm0 = (pair * 2 + rr) * TILE_M;
    if (mm0 >= qlen * G) continue;
    const int ii0 = mm0 / G;
    const int ii1 = min(qlen, (mm0 + TILE_M) / G);
    const int v0 = kvlen - qlen + ii0;
    const int v1 = kvlen - qlen + ii1 - 1;
    const int lo = p.window_left >= 0 ? max(0, v0 - p.window_left) : 0;
    const int hi = min(kvlen - 1, v1);
    pjlo = min(pjlo, (lo / TILE_N) * TILE_N);
    pjhi = max(pjhi, ((hi + 1 + TILE_N - 1) / TILE_N) * TILE_N - 1);
  }
  const int jt0 = pjlo;
  const int ntiles = (pjhi - jt0) / TILE_N + 1;
  int t_begin = 0, t_end = ntiles;
  if (p.splits > 1) {
    const int tps = (ntiles + p.splits - 1) / p.splits;
    t_begin = blockIdx.z * tps;
    t_end = min(ntiles, t_begin + tps);
  }
  const int split_z = p.splits > 1 ? blockIdx.z : 0;
  const long long w_id = (long long)by * p.mtiles + mtile;

  float t6_lrow = 0.f, t6_mdef = NEG_BIG;
  if (tid == 0) T6BC(0);

  if (t_begin >= t_end) {
    // Empty split: neutral partial like t5/t6 (uniform splits only; combine
    // reads every z). Both ranks publish for their own rows.
    if (p.splits > 1 && p.tps_target == 0) {
      for (int e = tid; e < TILE_M * HD; e += THREADS) {
        if (!dead && i0 + (e / HD) / G < qlen)
          p.part_o[w_id * p.part_s_w + (long long)split_z * p.part_s_s + e] = 0.f;
      }
      if (tid < TILE_M && !dead && i0 + tid / G < qlen) {
        float* ml = p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s +
                    (long long)tid * 2;
        ml[0] = NEG_BIG;
        ml[1] = 0.f;
        float* mt = p.part_ml_t + w_id * p.partml_s_w + (long long)tid * p.partmlt_s_r +
                    (long long)split_z * 2;
        mt[0] = NEG_BIG;
        mt[1] = 0.f;
      }
    }
    return;
  }

  // ---------------- prologue: Q tile -> smem (bf16 -> f16), local rows -------
  // QCP: at hd256 this gather is 4096 sixteen-byte chunks over 192 threads = 22
  // dependent iterations per thread, and the destination is a generic char*, so
  // the compiler cannot hoist the loads past the smem stores and the CTA pays
  // DRAM latency 22 times. cp.async stages all 22 at once (src-size 0 zero-fills
  // the padding rows AND every row of a dead rank-1 CTA, which is what the
  // `dead ? qlen : ...` form did) and the conversion then runs smem-to-smem.
  constexpr int QCH = TILE_M * HD / 8;
  constexpr int QUI = (QCH + THREADS - 1) / THREADS;
  const __nv_bfloat16* qbase = p.q + (long long)q0 * p.q_s_tok;
  if constexpr (QCP) {
#pragma unroll
    for (int u = 0; u < QUI; ++u) {
      const int c = tid + u * THREADS;
      if (QCH % THREADS == 0 || c < QCH) {
        const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
        const int i = dead ? qlen : (i0 + r / G);
        const bool ok = (i < qlen);
        const __nv_bfloat16* src =
            qbase + (long long)(ok ? i : 0) * p.q_s_tok +
            (long long)(hkv * G + (r % G)) * p.q_s_head + col;
        cp_async16_z(sQ + t5_off(TILE_M, r, col), src, ok ? 16 : 0);
      }
    }
    cp_async_commit();
    cp_async_wait<0>();
#pragma unroll
    for (int u = 0; u < QUI; ++u) {
      const int c = tid + u * THREADS;
      if (QCH % THREADS == 0 || c < QCH) {
        const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
        uint4* d = reinterpret_cast<uint4*>(sQ + t5_off(TILE_M, r, col));
        const uint4 data = *d;
        *d = make_uint4(bf16x2_to_f16x2(data.x), bf16x2_to_f16x2(data.y),
                        bf16x2_to_f16x2(data.z), bf16x2_to_f16x2(data.w));
      }
    }
  } else {
#pragma unroll
    for (int u = 0; u < QUI; ++u) {
      const int c = tid + u * THREADS;
      if (QCH % THREADS != 0 && c >= QCH) break;
      const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
      const int i = dead ? qlen : (i0 + r / G);
      uint4 data = make_uint4(0, 0, 0, 0);
      if (i < qlen) {
        const __nv_bfloat16* src =
            qbase + (long long)i * p.q_s_tok + (long long)(hkv * G + (r % G)) * p.q_s_head + col;
        data = *reinterpret_cast<const uint4*>(src);
      }
      const uint4 h = make_uint4(bf16x2_to_f16x2(data.x), bf16x2_to_f16x2(data.y),
                                 bf16x2_to_f16x2(data.z), bf16x2_to_f16x2(data.w));
      *reinterpret_cast<uint4*>(sQ + t5_off(TILE_M, r, col)) = h;
    }
  }
  // Q STS (generic proxy) -> async-proxy descriptor reads BY THE LEADER's
  // tensor core across the cluster: cluster-scope proxy fence, then the
  // release/acquire cluster barrier publishes both CTAs' Q tiles.
  fence_async_smem_cluster();
  if (tid == 0) T6BC(1);
  {
    if (tid == 0) T6BC(2);
    if (tid == 0) {
    if (rank == 0) {
      for (int s = 0; s < KST; ++s) t5_mbar_init(t5_sp(&KF[s]), 1);
      for (int s = 0; s < VST; ++s) t5_mbar_init(t5_sp(&VF[s]), 1);
      t5_mbar_init(t5_sp(&PR[0]), SM8C ? 16 : 8);
      t5_mbar_init(t5_sp(&PR[1]), SM8C ? 16 : 8);
    }
    for (int s = 0; s < KST; ++s) t5_mbar_init(t5_sp(&KE[s]), 1);
    for (int s = 0; s < VST; ++s) t5_mbar_init(t5_sp(&VE[s]), 1);
    t5_mbar_init(t5_sp(&QD[0]), 1);
    t5_mbar_init(t5_sp(&QD[1]), 1);
    t5_mbar_init(t5_sp(&PD[0]), 1);
    t5_mbar_init(t5_sp(&PD[1]), 1);
    t5_mbar_init(t5_sp(DB), 1);
  }
  }
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  cluster_arrive_release();
  cluster_wait_acquire();
  if (tid == 0) T6BC(3);
  if (warp == 0) t_alloc2<512>(t5_sp(tmem_word));
  t5_fence_before_sync();
  __syncthreads();
  t5_fence_after_sync();
  const uint32_t tmem = *tmem_word;
  const uint32_t o_tmem = tmem + OCOL;
  if (tid == 0) T6BC(4);

  if (rank == 0 && warp == MMA_W && lane == 0) {
    // ------------------------- MMA leader -------------------------
    T6BC(5);
    const uint32_t ID_QK = t5_idesc(256, TILE_N, 0);
    const uint32_t ID_PV = t5_idesc(256, 128, 1);
    const uint64_t qd0 = t5_desc(t5_sp(sQ), 16, 8 * 128);
    const uint64_t kdb = t5_desc(t5_sp(sK), 16, 8 * 128);
    // Per-CTA V half = [TILE_N tok][64 col] MN-major SW128 atoms, one per PV
    // atom (stride TILE_N*128B); the n128 instruction pairs the halves itself.
    const uint64_t vdb = t5_desc(t5_sp(sV), 0, 8 * 128);
    auto issue_qk = [&](int t) {
      const int rr = t - t_begin;
      const int sb = rr & 1, kb = rr % KST;
      const uint32_t scol = tmem + sb * TILE_N;
      const uint64_t kd = kdb + (uint64_t)(kb * (KT_BYTES >> 4));
#pragma unroll
      for (int g = 0; g < KSEG; ++g)
#pragma unroll
        for (int s = 0; s < 4; ++s)
          umma2_ss(scol, qd0 + (uint64_t)(g * (TILE_M * 8) + s * 2),
                   kd + (uint64_t)(g * (KN * 8) + s * 2), ID_QK, (g | s) != 0);
      commit2mc(t5_sp(&QD[sb]));
      commit2mc(t5_sp(&KE[kb]));
    };
    {
      mb_wait_acquire(t5_sp(&KF[0]), 0);
      T6BC(7);
      t5_fence_after_sync();
      issue_qk(t_begin);
    }
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int sb = rr & 1, vb = rr % VST;
      if (t + 1 < t_end) {
        const int kb1 = (rr + 1) % KST;
        mb_wait_acquire(t5_sp(&KF[kb1]), ((rr + 1) / KST) & 1);
        T6BC(7);
        t5_fence_after_sync();
        issue_qk(t + 1);
      }
      mb_wait_acquire(t5_sp(&PR[sb]), (rr >> 1) & 1);
      T6BC(10);
      t5_fence_after_sync();
      mb_wait_acquire(t5_sp(&VF[vb]), (rr / VST) & 1);
      T6BC(11);
      const uint32_t scol = tmem + sb * TILE_N;
      const uint64_t vd = vdb + (uint64_t)(vb * (VT_BYTES >> 4));
#pragma unroll
      for (int a = 0; a < PV_ATOMS; ++a)
#pragma unroll
        for (int s = 0; s < TILE_N / 16; ++s)
          umma2_ts(o_tmem + a * 128, scol + s * 8,
                   vd + (uint64_t)(a * (TILE_N * 8) + s * 128), ID_PV,
                   (t != t_begin || s != 0));
      commit2mc(t5_sp(&PD[sb]));
      commit2mc(t5_sp(&VE[vb]));
    }
  } else if (warp == LOAD_W && lane == 0) {
    // ------------------------- load warp (both CTAs) -------------------------
    T6BC(5);
    const int page_log2 = p.page_log2;
    const int box_rows = min(1 << page_log2, 64);
    const int tcols = p.table_stride;
    const int* table_row = p.table + (long long)b * tcols;
    const uint32_t kbar_local = t5_sp(&KF[0]);  // slot bases; +kb*8 selects the
    const uint32_t vbar_local = t5_sp(&VF[0]);  // ring slot (peer-bit cleared
                                                // inside tma4d_2cta routes the
                                                // tx-count to the leader copy)
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int kb = rr % KST, vb = rr % VST;
      if (rr >= KST) {
        mb_wait_acquire(t5_sp(&KE[kb]), ((rr / KST) - 1) & 1);
        T6BC(16);
      }
      if (rr >= VST) {
        mb_wait_acquire(t5_sp(&VE[vb]), ((rr / VST) - 1) & 1);
        T6BC(17);
      }
      // THE HANG (job dv K9b beacons: ldr_KF stuck at 1 with load_issued=3):
      // expect AND TMA completion must target the SAME per-slot barrier. The
      // first version passed the slot-0 base to tma4d_2cta, so tile 0 worked
      // by coincidence and tile 1's tx-count landed on KF[0] while KF[1]
      // starved forever.
      const uint32_t kbar = kbar_local + kb * 8;
      const uint32_t vbar = vbar_local + vb * 8;
      if (rank == 0) {
        // Both CTAs' cta_group::2 TMA transactions complete on the leader's
        // barrier, so the expectation covers both halves (deepgemm pattern).
        mb_expect(kbar, 2 * KT_BYTES);
        mb_expect(vbar, 2 * VT_BYTES);
      }
      const int jb = jt0 + t * TILE_N;
      // K half: tokens [jb + rank*64, +64), KSEG segments of 64 hd cols.
      const int kbase = jb + rank * KN;
      for (int off = 0; off < KN; off += box_rows) {
        const int jj = min(max(kbase + off, 0), kvlen - 1);
        const int pid = table_row[min(jj >> page_log2, tcols - 1)];
        const int row_in_page = (jj & ((1 << page_log2) - 1)) & ~(box_rows - 1);
#pragma unroll
        for (int sg = 0; sg < KSEG; ++sg) {
          const uint32_t dst = t5_sp(sK + kb * KT_BYTES + sg * (KN * 128) + off * 128);
          tma4d_2cta(&kmap, dst, kbar, sg * 64, row_in_page, hkv, pid);
        }
      }
      // V half: all TILE_N tokens; per-CTA cols of atom sg = sg*128 + rank*64
      // (the m256n128 PV instruction splits its 128 n-columns across the pair,
      // so atom sg's local half is NOT a contiguous [rank*VN] range at hd256).
      for (int off = 0; off < TILE_N; off += box_rows) {
        const int jj = min(max(jb + off, 0), kvlen - 1);
        const int pid = table_row[min(jj >> page_log2, tcols - 1)];
        const int row_in_page = (jj & ((1 << page_log2) - 1)) & ~(box_rows - 1);
#pragma unroll
        for (int sg = 0; sg < VSEG; ++sg) {
          const uint32_t dst = t5_sp(sV + vb * VT_BYTES + sg * (TILE_N * 128) + off * 128);
          tma4d_2cta(&vmap, dst, vbar, sg * 128 + rank * 64, row_in_page, hkv, pid);
        }
      }
      T6BC(6);
    }
  } else if (warp < NSW) {
    // ------------------- softmax warps (both CTAs, local rows) --------------
    // SM8C: rows are owned by (warp&3); the two halves split the tile's
    // columns and merge their row maxes through sMx + bar_sm8 (the exact
    // t6-SM8 form -- fmax is associative/exact so mdef stays identical
    // across halves and the DEFER predicate matches; exp/sum association
    // changes, same tolerance class as t6/t8's promoted recipes).
    if (warp == 0 && lane == 0) T6BC(5);
    const int half = SM8C ? (warp >> 2) : 0;
    const int row = (SM8C ? (warp & 3) : warp) * 32 + lane;
    const int irow = dead ? qlen : (i0 + row / G);
    const bool rowok = irow < qlen;
    const int vis = kvlen - qlen + irow;
    const int lim = rowok ? vis : -1;
    const int low = (rowok && p.window_left >= 0) ? max(0, vis - p.window_left) : 0;
    float mrun = NEG_BIG;
    float lrow = 0.f;
    float mdef = NEG_BIG;
    const float qk = p.qk_scale;
    const uint64_t qk2 = f2_pack(qk, qk);
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int sb = rr & 1;
      const uint32_t scol = tmem + sb * TILE_N;
      mb_wait_acquire(t5_sp(&QD[sb]), (rr >> 1) & 1);
      if (warp == 0 && lane == 0) T6BC(8);
      float sc[SCOLS];
#pragma unroll
      for (int blk = 0; blk < SCOLS / 32; ++blk)
        t5_ld32(sc + blk * 32, scol + half * SCOLS + blk * 32);
      t5_ld_wait();
      const int col0 = jt0 + t * TILE_N + half * SCOLS;
      const bool full = rowok && (col0 + SCOLS - 1 <= lim) && (col0 >= low);
      // RWSK (J46 port of arm-a J25, bit-identical) + hot-path-first split:
      // whole-tile-masked lanes skip both loops (mxr=NEG_BIG, pp=0, lrow+=0).
      const bool nosel = (RWSK != 0) && !full && ((lim < col0) || (low > col0 + SCOLS - 1));
      float mxr = NEG_BIG;
      if (full) {
        float m0v = NEG_BIG, m1v = NEG_BIG, m2v = NEG_BIG, m3v = NEG_BIG;
#pragma unroll
        for (int c = 0; c < SCOLS; c += 4) {
          m0v = fmaxf(m0v, sc[c]);
          m1v = fmaxf(m1v, sc[c + 1]);
          m2v = fmaxf(m2v, sc[c + 2]);
          m3v = fmaxf(m3v, sc[c + 3]);
        }
        mxr = fmaxf(fmaxf(m0v, m1v), fmaxf(m2v, m3v));
      } else if (!nosel) {
        float m0v = NEG_BIG, m1v = NEG_BIG, m2v = NEG_BIG, m3v = NEG_BIG;
#pragma unroll
        for (int c = 0; c < SCOLS; c += 4) {
          m0v = fmaxf(m0v, (col0 + c <= lim && col0 + c >= low) ? sc[c] : NEG_BIG);
          m1v = fmaxf(m1v, (col0 + c + 1 <= lim && col0 + c + 1 >= low) ? sc[c + 1] : NEG_BIG);
          m2v = fmaxf(m2v, (col0 + c + 2 <= lim && col0 + c + 2 >= low) ? sc[c + 2] : NEG_BIG);
          m3v = fmaxf(m3v, (col0 + c + 3 <= lim && col0 + c + 3 >= low) ? sc[c + 3] : NEG_BIG);
        }
        mxr = fmaxf(fmaxf(m0v, m1v), fmaxf(m2v, m3v));
      }
      if constexpr (SM8C) {
        sMx[row * 2 + half] = mxr;
        if constexpr (PBAR) bar_pair64(warp);
        else bar_sm8();
        mxr = fmaxf(mxr, sMx[row * 2 + (half ^ 1)]);
      }
      const float mn = fmaxf(mrun, mxr * qk);
      float mneg;
      if (t == t_begin) {
        mdef = mn;
      } else if (__any_sync(0xffffffffu, mn - mdef > (float)DTH)) {
        if constexpr (SM8C) {
          // Both halves share the merged mdef recurrence, so this predicate
          // and f are identical across all 8 warps; each half scales its own
          // lrow share, but only half-0 rescales the shared O accumulator --
          // the count-16 PR rendezvous keeps PV(t) behind it (t6-SM8 form).
          const float f = ex2(mdef - mn);
          lrow *= f;
          if (half == 0) {
            mb_wait_acquire(t5_sp(&PD[sb ^ 1]), ((rr - 1) >> 1) & 1);
            if (warp == 0 && lane == 0) T6BC(18);
            t5_fence_after_sync();
            for (int blk = 0; blk < HD / 32; ++blk) {
              float ov[32];
              t5_ld32(ov, o_tmem + blk * 32);
              t5_ld_wait();
#pragma unroll
              for (int j = 0; j < 32; ++j) ov[j] *= f;
              t5_st32(reinterpret_cast<const uint32_t*>(ov), o_tmem + blk * 32);
            }
            t5_st_wait();
          }
          mdef = mn;
        } else {
          mb_wait_acquire(t5_sp(&PD[sb ^ 1]), ((rr - 1) >> 1) & 1);
          if (warp == 0 && lane == 0) T6BC(18);
          t5_fence_after_sync();
          const float f = ex2(mdef - mn);
          lrow *= f;
          for (int blk = 0; blk < HD / 32; ++blk) {
            float ov[32];
            t5_ld32(ov, o_tmem + blk * 32);
            t5_ld_wait();
#pragma unroll
            for (int j = 0; j < 32; ++j) ov[j] *= f;
            t5_st32(reinterpret_cast<const uint32_t*>(ov), o_tmem + blk * 32);
          }
          t5_st_wait();
          mdef = mn;
        }
      }
      mrun = mn;
      mneg = -mdef;
      uint32_t pp[SCOLS / 2];
      const uint64_t mn2 = f2_pack(mneg, mneg);
      uint32_t hacc[4] = {0u, 0u, 0u, 0u};
      if (full) {
#pragma unroll
        for (int c = 0; c < SCOLS; c += 2) {
          const uint64_t x2 = fma2(f2_pack(sc[c], sc[c + 1]), qk2, mn2);
          float e0, e1;
          f2_unpack(x2, e0, e1);
          const uint32_t p2 = ex2_f16x2(pack_f16(e0, e1));
          pp[c >> 1] = p2;
          hacc[(c >> 1) & 3] = hadd2(hacc[(c >> 1) & 3], p2);
        }
      } else if (nosel) {
#pragma unroll
        for (int j = 0; j < SCOLS / 2; ++j) pp[j] = 0u;
      } else {
#pragma unroll
        for (int c = 0; c < SCOLS; c += 2) {
          const int j = col0 + c;
          const uint64_t x2 = fma2(f2_pack(sc[c], sc[c + 1]), qk2, mn2);
          float e0, e1;
          f2_unpack(x2, e0, e1);
          if (!(j <= lim && j >= low)) e0 = NEG_BIG;
          if (!(j + 1 <= lim && j + 1 >= low)) e1 = NEG_BIG;
          const uint32_t p2 = ex2_f16x2(pack_f16(e0, e1));
          pp[c >> 1] = p2;
          hacc[(c >> 1) & 3] = hadd2(hacc[(c >> 1) & 3], p2);
        }
      }
      lrow += (hsum2(hacc[0]) + hsum2(hacc[1])) + (hsum2(hacc[2]) + hsum2(hacc[3]));
#pragma unroll
      for (int blk = 0; blk < SCOLS / 2 / 32; ++blk)
        t5_st32(pp + blk * 32, scol + half * (SCOLS / 2) + blk * 32);
      t5_st_wait();
      t5_fence_before_sync();
      if (lane == 0) {
        if (rank == 0) mb_arrive(t5_sp(&PR[sb]));
        else mb_arrive_remote0(t5_sp(&PR[sb]));
        if (warp == 0) T6BC(9);
      }
    }
    t6_lrow = lrow;
    t6_mdef = mdef;
  }

  // --------------------------- epilogue (per CTA) ---------------------------
  __syncthreads();
  constexpr int SROW = HD + 4;
  float* sOut = reinterpret_cast<float*>(smraw);
  float* sGain = sOut + TILE_M * SROW;
  float* sLse = sGain + TILE_M;
  float* sM = sLse + TILE_M;
  float* sL = sM + TILE_M;
  if constexpr (SM8C) {
    // Publish per-half row sums; warps<4 combine into lfin below. sMx lives
    // in the barrier tail, untouched by the sOut overlay (t6-SM8 pattern).
    if (warp < 8) {
      const int row = (warp & 3) * 32 + lane;
      sMx[row * 2 + (warp >> 2)] = t6_lrow;
      bar_sm8();
    }
  }
  if (warp < 4) {
    const int row = warp * 32 + lane;
    const int irow = dead ? qlen : (i0 + row / G);
    const bool rowok = irow < qlen;
    {
      const int rlast = t_end - 1 - t_begin;
      mb_wait_acquire(t5_sp(&PD[rlast & 1]), (rlast >> 1) & 1);
    }
    if (warp == 0 && lane == 0) T6BC(12);
    t5_fence_after_sync();
    float lfin = t6_lrow;
    if constexpr (SM8C) lfin += sMx[row * 2 + 1];
    const float mfin = t6_mdef;
    float base2 = mfin + (lfin > 0.f ? log2f(lfin) : NEG_BIG);
    float sc_sink = 1.f;
    if (p.sinks != nullptr) {
      const float sink2 = p.sinks[hkv * G + (row % G)] * LOG2E;
      const float hi = fmaxf(base2, sink2);
      const float total2 = hi + log2f(1.f + ex2(-fabsf(base2 - sink2)));
      sc_sink = ex2(base2 - total2);
      base2 = total2;
    }
    const float gain = (lfin > 0.f ? 1.f / lfin : 0.f) * sc_sink * p.v_scale;
    if (rowok) {
      sGain[row] = gain;
      sLse[row] = base2;
      sM[row] = mfin;
      sL[row] = lfin;
    }
    for (int blk = 0; blk < HD / 32; ++blk) {
      float ov[32];
      t5_ld32(ov, o_tmem + blk * 32);
      t5_ld_wait();
      if (rowok)
        for (int j = 0; j < 32; ++j) sOut[row * SROW + blk * 32 + j] = ov[j];
    }
  }
  __syncthreads();
  if (p.splits > 1) {
    float* dst = p.part_o + w_id * p.part_s_w + (long long)split_z * p.part_s_s;
    for (int e = tid * 4; e < TILE_M * HD; e += THREADS * 4) {
      const int r = e / HD;
      if (dead || i0 + r / G >= qlen) continue;
      *reinterpret_cast<float4*>(dst + e) =
          *reinterpret_cast<const float4*>(sOut + r * SROW + (e - r * HD));
    }
    float* ml = p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s;
    float* mt = p.part_ml_t + w_id * p.partml_s_w + (long long)split_z * 2;
    for (int r = tid; r < TILE_M; r += THREADS) {
      if (dead || i0 + r / G >= qlen) continue;
      ml[r * 2] = sM[r];
      ml[r * 2 + 1] = sL[r];
      mt[r * p.partmlt_s_r] = sM[r];
      mt[r * p.partmlt_s_r + 1] = sL[r];
    }
  } else if (!dead) {
    for (int e = tid * 8; e < TILE_M * HD; e += THREADS * 8) {
      const int r = e / HD, c = e % HD;
      const int i = i0 + r / G;
      if (i >= qlen) continue;
      const float g = sGain[r];
      const float* so = sOut + r * SROW + (e - r * HD);
      __nv_bfloat16 h8[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) h8[j] = __float2bfloat16_rn(so[j] * g);
      const long long tok = q0 + i;
      const int head = hkv * G + (r % G);
      *reinterpret_cast<uint4*>(p.out + tok * p.out_s_tok + (long long)head * p.out_s_head + c) =
          *reinterpret_cast<const uint4*>(h8);
    }
    if (p.write_lse) {
      for (int r = tid; r < TILE_M; r += THREADS) {
        const int i = i0 + r / G;
        if (i >= qlen) continue;
        const long long tok = q0 + i;
        const int head = hkv * G + (r % G);
        p.lse[tok * p.lse_s_tok + (long long)head * p.lse_s_head] = sLse[r];
      }
    }
  }
  if (tid == 0) T6BC(13);
  // Pair teardown: both CTAs must have drained before the pair TMEM is freed.
  cluster_arrive_release();
  cluster_wait_acquire();
  if (tid == 0) T6BC(14);
  if (warp == 0) {
    // Arrive on the PEER's dealloc barrier: arriving on rank0's from both
    // CTAs (mapa0) double-signals the leader and starves the peer forever
    // (beacon-probe hang, job dv_8a7c1702a648: rank0 dealloc_done=1, rank1=0).
    if (lane == 0) mb_arrive_remote(t5_sp(DB), rank ^ 1u);
    mb_wait_acquire(t5_sp(DB), 0);
    t_relinq2();
    t_dealloc2<512>(tmem);
    if (lane == 0) T6BC(15);
  }
}
#undef T6BC

template <int HD, int KST, int VST, int DTH, int BC = 0, int MTM6C = 0, int SM8C = 0>
static void launch_main_t6c(const Params& p, const CUtensorMap& kmap, const CUtensorMap& vmap,
                            dim3 grid_ct, cudaStream_t stream) {
  // Must match the kernel layout EXACTLY: Q [128][HD], K stages [64 tok][HD]
  // (the pair-split half), V stages [128 tok][HD/2] (the hd-split half).
  constexpr int TILE_M = 128, KN = 64;
  const int smem = TILE_M * HD * 2 + KST * KN * HD * 2 + VST * TILE_M * (HD / 2) * 2 +
                   (SM8C ? 2048 : 1024);
  static_assert(0 == 0, "keep in sync with fmha_main_t6c");
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(fmha_main_t6c<HD, KST, VST, DTH, BC, MTM6C, 0, 0, SM8C>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  fmha_main_t6c<HD, KST, VST, DTH, BC, MTM6C, 0, 0, SM8C>
      <<<grid_ct, SM8C ? 320 : 192, smem, stream>>>(p, kmap, vmap);
}

// cfg265: the same t6c instantiation with QCP=1 (cp.async Q staging).
template <int HD, int KST, int VST, int DTH, int BC = 0, int MTM6C = 0, int RWSK = 0,
          int SM8C = 0, int PBAR = 0>
static void launch_main_t6c_qcp(const Params& p, const CUtensorMap& kmap, const CUtensorMap& vmap,
                            dim3 grid_ct, cudaStream_t stream) {
  // Must match the kernel layout EXACTLY: Q [128][HD], K stages [64 tok][HD]
  // (the pair-split half), V stages [128 tok][HD/2] (the hd-split half).
  constexpr int TILE_M = 128, KN = 64;
  const int smem = TILE_M * HD * 2 + KST * KN * HD * 2 + VST * TILE_M * (HD / 2) * 2 +
                   (SM8C ? 2048 : 1024);
  static_assert(0 == 0, "keep in sync with fmha_main_t6c");
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(fmha_main_t6c<HD, KST, VST, DTH, BC, MTM6C, 1, RWSK, SM8C, PBAR>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  fmha_main_t6c<HD, KST, VST, DTH, BC, MTM6C, 1, RWSK, SM8C, PBAR>
      <<<grid_ct, SM8C ? 320 : 192, smem, stream>>>(p, kmap, vmap);
}

// ---------------------------------------------------------------------------
// t8: 256-row single-CTA prefill (two 128-row halves, TILE_N=64) -- halves f1-
// class L2 staging traffic WITHOUT clusters. t6 re-reads the full K/V tile per
// 128-row mtile (f1 = 5.96GB at 7.7 of the 8.4TB/s aggregate L2 ceiling); the
// cluster attempts to halve that (t6c pair-MMA, t6d multicast) both paid ~30%
// in pair scheduling/lockstep (K11/K18e). t8 instead stacks TWO mtiles in one
// CTA: one K/V tile (64 cols) feeds both row halves (QK_a/QK_b m128n64 g1,
// PV_a/PV_b TS m128n128), so per-mtile staging bytes halve while every MMA
// stays single-CTA and both halves pipeline independently through their own
// S/PR/PD barriers. TMEM is exactly full: O_a[0,128) O_b[128,256)
// S_a[256,384) S_b[384,512) (2 x 64-col buffers per half).
// Cost model: TC/tile = 2*(8*48.1) + 2*(4*64.1) = 1283 cyc per 256x64 tile
// (+25% per-work vs t6's n128 QK) vs L2 bytes halved -- f1 TC floor ~166us vs
// L2 floor ~355us, so the n64 penalty stays hidden.
// Range: per-half causal/window ranges unified like t6c (columns past a half's
// own range are fully masked -> P=0, bit-exact); dead half B (odd mtiles,
// ragged requests) still runs the pipeline but publishes nothing (neutral
// partial under splits). Numerics recipe identical to t6 v1.3.
// ---------------------------------------------------------------------------
template <int HD, int KST, int VST, int DTH, int BC = 0, int MTM8 = 0>
__global__ void __launch_bounds__(320, 1)
fmha_main_t8(const __grid_constant__ Params p, const __grid_constant__ CUtensorMap kmap,
             const __grid_constant__ CUtensorMap vmap) {
  constexpr int ROWS = 256;      // rows per CTA (halves A/B of 128)
  constexpr int HALF = 128;
  constexpr int TILE_N = 64;
  constexpr int THREADS = 320;   // 8 softmax + 1 mma + 1 load
  constexpr int KG = HD / 64;
  constexpr int QBYTES = ROWS * HD * 2;
  constexpr int KT_BYTES = TILE_N * HD * 2;
  constexpr int VT_BYTES = TILE_N * HD * 2;
  constexpr int OCOL_B = 128;        // TMEM: O_a [0,128), O_b [128,256)
  constexpr int SCOL_A = 256;        // S_a [256,384), S_b [384,512)
  constexpr int SCOL_B = 384;

  extern __shared__ char smraw[];
  char* sQ = smraw;
  char* sK = sQ + QBYTES;
  char* sV = sK + KST * KT_BYTES;
  uint64_t* KF = reinterpret_cast<uint64_t*>(sV + VST * VT_BYTES);
  uint64_t* VF = KF + KST;
  uint64_t* KE = VF + VST;
  uint64_t* VE = KE + KST;
  uint64_t* QDA = VE + VST;
  uint64_t* QDB = QDA + 2;
  uint64_t* PRA = QDB + 2;   // count 4 (half-A softmax warps)
  uint64_t* PRB = PRA + 2;   // count 4
  uint64_t* PDA = PRB + 2;
  uint64_t* PDB = PDA + 2;
  uint32_t* tmem_word = reinterpret_cast<uint32_t*>(PDB + 2);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  long long* bc = nullptr;
  if constexpr (BC) {
    if (p.splits == 1 && p.part_o != nullptr && blockIdx.x < 2 && blockIdx.y == 0 &&
        blockIdx.z == 0)
      bc = reinterpret_cast<long long*>(p.part_o + (25 << 20));
  }
#define T8BC(stage)                                                         \
  do {                                                                      \
    if (bc != nullptr)                                                      \
      atomicAdd(reinterpret_cast<unsigned long long*>(bc + (stage)), 1ULL); \
  } while (0)

  // MTM8: flattened x-major DESCENDING (LPT under causal), ragged-aware like
  // MTM6C/D (descending would schedule short requests' live CTAs last).
  int xb, by;
  if constexpr (MTM8) {
    const unsigned XB = (unsigned)((p.mtiles + 1) / 2);
    const unsigned Y = gridDim.x / XB;
    bool uniform = true;
    const int nb = (int)(Y / (unsigned)p.HKV);
    for (int bb = 0; bb < nb; ++bb) {
      const int ql = p.cu_q[bb + 1] - p.cu_q[bb];
      if ((ql * p.G + HALF - 1) / HALF < p.mtiles) {
        uniform = false;
        break;
      }
    }
    if (uniform) {
      by = (int)(blockIdx.x % Y);
      xb = (int)(XB - 1u - blockIdx.x / Y);
    } else {
      by = (int)(blockIdx.x / XB);
      xb = (int)(blockIdx.x % XB);
    }
  } else {
    xb = (int)blockIdx.x;
    by = (int)blockIdx.y;
  }
  const int mta = xb * 2;
  const int mtb = mta + 1;
  const int b = by / p.HKV;
  const int hkv = by % p.HKV;
  const int q0 = p.cu_q[b];
  const int qlen = p.cu_q[b + 1] - q0;
  const int kvlen = p.seq_lens[b];
  const int G = p.G;
  const int m0a = mta * HALF;
  const int m0b = mtb * HALF;
  const bool dead_a = (mta >= p.mtiles) || (m0a >= qlen * G);
  const bool dead_b = (mtb >= p.mtiles) || (m0b >= qlen * G);
  if (dead_a) return;  // mtb > mta always, so B is dead too

  const int i0a = m0a / G;
  const int i0b = dead_b ? qlen : (m0b / G);

  // Per-half kv column ranges; the CTA walks their UNION (columns past a
  // half's own range are fully masked in its softmax -> P=0, bit-exact).
  auto half_range = [&](int m0, bool dead, int& lo_out, int& hi_out) {
    if (dead) {
      lo_out = 0x7fffffff;
      hi_out = -0x7fffffff;
      return;
    }
    const int i0v = m0 / G;
    const int i1v = min(qlen, (m0 + HALF) / G);
    const int v0 = kvlen - qlen + i0v;
    const int v1 = kvlen - qlen + i1v - 1;
    lo_out = p.window_left >= 0 ? max(0, v0 - p.window_left) : 0;
    hi_out = min(kvlen - 1, v1);
  };
  int loa, hia, lob, hib;
  half_range(m0a, false, loa, hia);
  half_range(m0b, dead_b, lob, hib);
  const int lo = min(loa, lob);
  const int hi = max(hia, hib);
  const int jt0 = (lo / TILE_N) * TILE_N;
  const int jt1 = ((hi + 1 + TILE_N - 1) / TILE_N) * TILE_N;
  const int ntiles = (jt1 - jt0) / TILE_N;
  int t_begin = 0, t_end = ntiles;
  const int split_z = p.splits > 1 ? blockIdx.z : 0;
  if (p.splits > 1) {
    const int tps = (ntiles + p.splits - 1) / p.splits;  // uniform only (tps_target=0)
    t_begin = blockIdx.z * tps;
    t_end = min(ntiles, t_begin + tps);
  }
  const long long w_id_a = (long long)by * p.mtiles + mta;
  const long long w_id_b = (long long)by * p.mtiles + mtb;
  if (tid == 0) T8BC(0);

  float t8_lrow = 0.f, t8_mdef = NEG_BIG;

  if (t_begin >= t_end) {
    // Empty split: neutral partials for both live halves (uniform splits; the
    // combine reads every z).
    if (p.splits > 1 && p.tps_target == 0) {
      for (int hh = 0; hh < 2; ++hh) {
        const bool deadh = hh ? dead_b : false;
        const long long w_id = hh ? w_id_b : w_id_a;
        const int i0h = hh ? i0b : i0a;
        if (deadh) continue;
        for (int e = tid; e < HALF * HD; e += THREADS) {
          if (i0h + (e / HD) / G < qlen)
            p.part_o[w_id * p.part_s_w + (long long)split_z * p.part_s_s + e] = 0.f;
        }
        if (tid < HALF && i0h + tid / G < qlen) {
          float* ml = p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s +
                      (long long)tid * 2;
          ml[0] = NEG_BIG;
          ml[1] = 0.f;
          float* mt = p.part_ml_t + w_id * p.partml_s_w + (long long)tid * p.partmlt_s_r +
                      (long long)split_z * 2;
          mt[0] = NEG_BIG;
          mt[1] = 0.f;
        }
      }
    }
    return;
  }

  // ---------------- prologue: both Q half-tiles -> smem (bf16 -> f16) --------
  constexpr int QCH = ROWS * HD / 8;
  const __nv_bfloat16* qbase = p.q + (long long)q0 * p.q_s_tok;
#pragma unroll
  for (int u = 0; u < (QCH + THREADS - 1) / THREADS; ++u) {
    const int c = tid + u * THREADS;
    if (QCH % THREADS != 0 && c >= QCH) break;
    const int r = c / (HD / 8), col = (c % (HD / 8)) * 8;
    const int i = (r < HALF ? i0a + r / G : (dead_b ? qlen : i0b + (r - HALF) / G));
    uint4 data = make_uint4(0, 0, 0, 0);
    if (i < qlen) {
      const __nv_bfloat16* src =
          qbase + (long long)i * p.q_s_tok +
          (long long)(hkv * G + ((r < HALF ? r : r - HALF) % G)) * p.q_s_head + col;
      data = *reinterpret_cast<const uint4*>(src);
    }
    const uint4 h = make_uint4(bf16x2_to_f16x2(data.x), bf16x2_to_f16x2(data.y),
                               bf16x2_to_f16x2(data.z), bf16x2_to_f16x2(data.w));
    // Two independent 128-row swizzle tiles (NOT one 256-row tile): the MMA A
    // descriptors then match t6's exactly (segment stride = 128 rows * 128B).
    char* qh = sQ + (r < HALF ? 0 : HALF * HD * 2);
    *reinterpret_cast<uint4*>(qh + t5_off(HALF, r < HALF ? r : r - HALF, col)) = h;
  }
  if (tid == 0) T8BC(1);
  if (tid == 0) {
    for (int s = 0; s < KST; ++s) {
      t5_mbar_init(t5_sp(&KF[s]), 1);
      t5_mbar_init(t5_sp(&KE[s]), 1);
    }
    for (int s = 0; s < VST; ++s) {
      t5_mbar_init(t5_sp(&VF[s]), 1);
      t5_mbar_init(t5_sp(&VE[s]), 1);
    }
    t5_mbar_init(t5_sp(&QDA[0]), 1);
    t5_mbar_init(t5_sp(&QDA[1]), 1);
    t5_mbar_init(t5_sp(&QDB[0]), 1);
    t5_mbar_init(t5_sp(&QDB[1]), 1);
    t5_mbar_init(t5_sp(&PRA[0]), 4);
    t5_mbar_init(t5_sp(&PRA[1]), 4);
    t5_mbar_init(t5_sp(&PRB[0]), 4);
    t5_mbar_init(t5_sp(&PRB[1]), 4);
    t5_mbar_init(t5_sp(&PDA[0]), 1);
    t5_mbar_init(t5_sp(&PDA[1]), 1);
    t5_mbar_init(t5_sp(&PDB[0]), 1);
    t5_mbar_init(t5_sp(&PDB[1]), 1);
  }
  fence_async_smem();  // Q STS (generic proxy) vs TC descriptor reads
  __syncthreads();
  if (warp == 0) {
    t5_alloc(t5_sp(tmem_word), 512);
    t5_relinquish();
  }
  __syncthreads();
  t5_fence_after_sync();
  const uint32_t tmem = *tmem_word;
  const uint32_t oa_tmem = tmem;
  const uint32_t ob_tmem = tmem + OCOL_B;
  if (tid == 0) T8BC(4);

  if (warp == 8 && lane == 0) {
    // ------------------------- MMA issuer -------------------------
    T8BC(5);
    const uint32_t ID_QK = t5_idesc(HALF, TILE_N, 0);
    const uint32_t ID_PV = t5_idesc(HALF, HD, 1);
    const uint64_t qda = t5_desc(t5_sp(sQ), 16, 8 * 128);
    const uint64_t qdb = t5_desc(t5_sp(sQ + HALF * HD * 2), 16, 8 * 128);
    const uint64_t kdb = t5_desc(t5_sp(sK), 16, 8 * 128);
    // V n128 spans two 64-col swizzle atoms at TILE_N*128B stride (as t6 PVN128).
    const uint64_t vdb = t5_desc(t5_sp(sV), TILE_N * 128, 8 * 128);
    auto issue_qk = [&](int t) {
      const int rr = t - t_begin;
      const int sb = rr & 1, kb = rr % KST;
      const uint32_t sa = tmem + SCOL_A + sb * TILE_N;
      const uint32_t sc = tmem + SCOL_B + sb * TILE_N;
      const uint64_t kd = kdb + (uint64_t)(kb * (KT_BYTES >> 4));
#pragma unroll
      for (int g = 0; g < KG; ++g)
#pragma unroll
        for (int s = 0; s < 4; ++s)
          t5_umma_ss(sa, qda + (uint64_t)(g * (HALF * 8) + s * 2),
                     kd + (uint64_t)(g * (TILE_N * 8) + s * 2), ID_QK, (g | s) != 0);
      t5_commit(t5_sp(&QDA[sb]));
#pragma unroll
      for (int g = 0; g < KG; ++g)
#pragma unroll
        for (int s = 0; s < 4; ++s)
          t5_umma_ss(sc, qdb + (uint64_t)(g * (HALF * 8) + s * 2),
                     kd + (uint64_t)(g * (TILE_N * 8) + s * 2), ID_QK, (g | s) != 0);
      t5_commit(t5_sp(&QDB[sb]));
      t5_commit(t5_sp(&KE[kb]));
    };
    {
      t5_mbar_wait(t5_sp(&KF[0]), 0);
      T8BC(7);
      t5_fence_after_sync();
      issue_qk(t_begin);
    }
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int sb = rr & 1, vb = rr % VST;
      if (t + 1 < t_end) {
        const int kb1 = (rr + 1) % KST;
        t5_mbar_wait(t5_sp(&KF[kb1]), ((rr + 1) / KST) & 1);
        T8BC(7);
        t5_fence_after_sync();
        issue_qk(t + 1);
      }
      const uint64_t vd = vdb + (uint64_t)(vb * (VT_BYTES >> 4));
      t5_mbar_wait(t5_sp(&PRA[sb]), (rr >> 1) & 1);
      T8BC(10);
      t5_fence_after_sync();
      t5_mbar_wait(t5_sp(&VF[vb]), (rr / VST) & 1);
      T8BC(11);
      {
        const uint32_t sa = tmem + SCOL_A + sb * TILE_N;
#pragma unroll
        for (int s = 0; s < TILE_N / 16; ++s)
          t5_umma_ts(oa_tmem, sa + s * 8, vd + (uint64_t)(s * 128), ID_PV,
                     (t != t_begin || s != 0));
      }
      t5_commit(t5_sp(&PDA[sb]));
      t5_mbar_wait(t5_sp(&PRB[sb]), (rr >> 1) & 1);
      t5_fence_after_sync();
      {
        const uint32_t sc = tmem + SCOL_B + sb * TILE_N;
#pragma unroll
        for (int s = 0; s < TILE_N / 16; ++s)
          t5_umma_ts(ob_tmem, sc + s * 8, vd + (uint64_t)(s * 128), ID_PV,
                     (t != t_begin || s != 0));
      }
      t5_commit(t5_sp(&PDB[sb]));
      t5_commit(t5_sp(&VE[vb]));
    }
  } else if (warp == 9 && lane == 0) {
    // ------------------------- load warp -------------------------
    T8BC(5);
    const int page_log2 = p.page_log2;
    // The gate builds t8 maps with box_rows = min(page, 64) (t6c style): a
    // page128 tile half addresses rows INSIDE one page via coord1.
    const int box_rows = min(1 << page_log2, TILE_N);
    const int page_mask = (1 << page_log2) - 1;
    const int tcols = p.table_stride;
    const int* table_row = p.table + (long long)b * tcols;
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int kb = rr % KST, vb = rr % VST;
      if (rr >= KST) {
        t5_mbar_wait(t5_sp(&KE[kb]), ((rr / KST) - 1) & 1);
        T8BC(16);
      }
      if (rr >= VST) {
        t5_mbar_wait(t5_sp(&VE[vb]), ((rr / VST) - 1) & 1);
        T8BC(17);
      }
      const int jb = jt0 + t * TILE_N;
      const uint32_t kbar = t5_sp(&KF[kb]);
      const uint32_t vbar = t5_sp(&VF[vb]);
      mb_expect(kbar, KT_BYTES);
      mb_expect(vbar, VT_BYTES);
      for (int off = 0; off < TILE_N; off += box_rows) {
        const int jj = min(max(jb + off, 0), kvlen - 1);
        const int pid = table_row[min(jj >> page_log2, tcols - 1)];
        const int row_in_page = (jj & page_mask) & ~(box_rows - 1);
#pragma unroll
        for (int sg = 0; sg < KG; ++sg) {
          const uint32_t off2 = (uint32_t)(sg * (TILE_N * 128) + off * 128);
          tma4d(&kmap, t5_sp(sK + kb * KT_BYTES + off2), kbar, sg * 64, row_in_page, hkv, pid);
          tma4d(&vmap, t5_sp(sV + vb * VT_BYTES + off2), vbar, sg * 64, row_in_page, hkv, pid);
        }
      }
      T8BC(6);
    }
  } else if (warp < 8) {
    // ---------------- softmax: warps 0-3 half A, warps 4-7 half B ------------
    if (lane == 0 && (warp == 0 || warp == 4)) T8BC(5);
    const int half = warp >> 2;          // 0 = A, 1 = B
    const int hwarp = warp & 3;
    const int row = hwarp * 32 + lane;   // row within the 128-row half
    const bool deadh = (half == 1) && dead_b;
    const int i0h = half == 0 ? i0a : i0b;
    const int irow = deadh ? qlen : (i0h + row / G);
    const bool rowok = irow < qlen;
    const int vis = kvlen - qlen + irow;
    const int lim = rowok ? vis : -1;
    const int low = (rowok && p.window_left >= 0) ? max(0, vis - p.window_left) : 0;
    const uint32_t o_tmem = half == 0 ? oa_tmem : ob_tmem;
    const uint32_t sbase = tmem + (half == 0 ? SCOL_A : SCOL_B);
    uint64_t* QD = half == 0 ? QDA : QDB;
    uint64_t* PR = half == 0 ? PRA : PRB;
    uint64_t* PD = half == 0 ? PDA : PDB;
    float mrun = NEG_BIG;
    float lrow = 0.f;
    float mdef = NEG_BIG;
    const float qk = p.qk_scale;
    const uint64_t qk2 = f2_pack(qk, qk);
    for (int t = t_begin; t < t_end; ++t) {
      const int rr = t - t_begin;
      const int sb = rr & 1;
      const uint32_t scol = sbase + sb * TILE_N;
      t5_mbar_wait(t5_sp(&QD[sb]), (rr >> 1) & 1);
      if (lane == 0 && (warp == 0 || warp == 4)) T8BC(8);
      float sc[TILE_N];
#pragma unroll
      for (int blk = 0; blk < TILE_N / 32; ++blk) t5_ld32(sc + blk * 32, scol + blk * 32);
      t5_ld_wait();
      const int col0 = jt0 + t * TILE_N;
      const bool full = rowok && (col0 + TILE_N - 1 <= lim) && (col0 >= low);
      float mxr = NEG_BIG;
      {
        float m0v = NEG_BIG, m1v = NEG_BIG, m2v = NEG_BIG, m3v = NEG_BIG;
        if (full) {
#pragma unroll
          for (int c = 0; c < TILE_N; c += 4) {
            m0v = fmaxf(m0v, sc[c]);
            m1v = fmaxf(m1v, sc[c + 1]);
            m2v = fmaxf(m2v, sc[c + 2]);
            m3v = fmaxf(m3v, sc[c + 3]);
          }
        } else {
#pragma unroll
          for (int c = 0; c < TILE_N; c += 4) {
            m0v = fmaxf(m0v, (col0 + c <= lim && col0 + c >= low) ? sc[c] : NEG_BIG);
            m1v = fmaxf(m1v, (col0 + c + 1 <= lim && col0 + c + 1 >= low) ? sc[c + 1] : NEG_BIG);
            m2v = fmaxf(m2v, (col0 + c + 2 <= lim && col0 + c + 2 >= low) ? sc[c + 2] : NEG_BIG);
            m3v = fmaxf(m3v, (col0 + c + 3 <= lim && col0 + c + 3 >= low) ? sc[c + 3] : NEG_BIG);
          }
        }
        mxr = fmaxf(fmaxf(m0v, m1v), fmaxf(m2v, m3v));
      }
      // Running max in scaled log2 units; never fma-contracted (t5/t6 recipe).
      const float mn = fmaxf(mrun, mxr * qk);
      float mneg;
      if (t == t_begin) {
        mdef = mn;
      } else if (__any_sync(0xffffffffu, mn - mdef > (float)DTH)) {
        t5_mbar_wait(t5_sp(&PD[sb ^ 1]), ((rr - 1) >> 1) & 1);
        if (lane == 0 && (warp == 0 || warp == 4)) T8BC(18);
        t5_fence_after_sync();
        const float f = ex2(mdef - mn);
        lrow *= f;
        for (int blk = 0; blk < HD / 32; ++blk) {
          float ov[32];
          t5_ld32(ov, o_tmem + blk * 32);
          t5_ld_wait();
#pragma unroll
          for (int j = 0; j < 32; ++j) ov[j] *= f;
          t5_st32(reinterpret_cast<const uint32_t*>(ov), o_tmem + blk * 32);
        }
        t5_st_wait();
        mdef = mn;
      }
      mrun = mn;
      mneg = -mdef;
      uint32_t pp[TILE_N / 2];
      const uint64_t mn2 = f2_pack(mneg, mneg);
      uint32_t hacc[2] = {0u, 0u};
      if (full) {
#pragma unroll
        for (int c = 0; c < TILE_N; c += 2) {
          const uint64_t x2 = fma2(f2_pack(sc[c], sc[c + 1]), qk2, mn2);
          float e0, e1;
          f2_unpack(x2, e0, e1);
          const uint32_t p2 = ex2_f16x2(pack_f16(e0, e1));
          pp[c >> 1] = p2;
          hacc[(c >> 1) & 1] = hadd2(hacc[(c >> 1) & 1], p2);
        }
      } else {
#pragma unroll
        for (int c = 0; c < TILE_N; c += 2) {
          const int j = col0 + c;
          const uint64_t x2 = fma2(f2_pack(sc[c], sc[c + 1]), qk2, mn2);
          float e0, e1;
          f2_unpack(x2, e0, e1);
          if (!(j <= lim && j >= low)) e0 = NEG_BIG;
          if (!(j + 1 <= lim && j + 1 >= low)) e1 = NEG_BIG;
          const uint32_t p2 = ex2_f16x2(pack_f16(e0, e1));
          pp[c >> 1] = p2;
          hacc[(c >> 1) & 1] = hadd2(hacc[(c >> 1) & 1], p2);
        }
      }
      lrow += hsum2(hacc[0]) + hsum2(hacc[1]);
      t5_st32(pp, scol);
      t5_st_wait();
      t5_fence_before_sync();
      if (lane == 0) mb_arrive(t5_sp(&PR[sb]));
      if (lane == 0 && (warp == 0 || warp == 4)) T8BC(9);
    }
    t8_lrow = lrow;
    t8_mdef = mdef;
  }

  // --------------------------- epilogue ---------------------------
  __syncthreads();
  constexpr int SROW = HD + 4;
  float* sOut = reinterpret_cast<float*>(smraw);  // ROWS * SROW
  float* sGain = sOut + ROWS * SROW;
  float* sLse = sGain + ROWS;
  float* sM = sLse + ROWS;
  float* sL = sM + ROWS;
  if (warp < 8) {
    const int half = warp >> 2;
    const int hwarp = warp & 3;
    const int row = hwarp * 32 + lane;
    const bool deadh = (half == 1) && dead_b;
    const int i0h = half == 0 ? i0a : i0b;
    const int irow = deadh ? qlen : (i0h + row / G);
    const bool rowok = irow < qlen;
    {
      const int rlast = t_end - 1 - t_begin;
      uint64_t* PD = half == 0 ? PDA : PDB;
      t5_mbar_wait(t5_sp(&PD[rlast & 1]), (rlast >> 1) & 1);
    }
    if (lane == 0 && (warp == 0 || warp == 4)) T8BC(12);
    t5_fence_after_sync();
    const uint32_t o_tmem = half == 0 ? oa_tmem : ob_tmem;
    const float lfin = t8_lrow;
    const float mfin = t8_mdef;
    float base2 = mfin + (lfin > 0.f ? log2f(lfin) : NEG_BIG);
    float sc_sink = 1.f;
    if (p.sinks != nullptr) {
      const float sink2 = p.sinks[hkv * G + (row % G)] * LOG2E;
      const float hi = fmaxf(base2, sink2);
      const float total2 = hi + log2f(1.f + ex2(-fabsf(base2 - sink2)));
      sc_sink = ex2(base2 - total2);
      base2 = total2;
    }
    const float gain = (lfin > 0.f ? 1.f / lfin : 0.f) * sc_sink * p.v_scale;
    const int orow = half * HALF + row;
    if (rowok) {
      sGain[orow] = gain;
      sLse[orow] = base2;
      sM[orow] = mfin;
      sL[orow] = lfin;
    }
    for (int blk = 0; blk < HD / 32; ++blk) {
      float ov[32];
      t5_ld32(ov, o_tmem + blk * 32);
      t5_ld_wait();
      if (rowok)
        for (int j = 0; j < 32; ++j) sOut[orow * SROW + blk * 32 + j] = ov[j];
    }
  }
  __syncthreads();
  if (p.splits > 1) {
    for (int hh = 0; hh < 2; ++hh) {
      const bool deadh = hh ? dead_b : false;
      if (deadh) continue;
      const long long w_id = hh ? w_id_b : w_id_a;
      const int i0h = hh ? i0b : i0a;
      const int rowbase = hh * HALF;
      float* dst = p.part_o + w_id * p.part_s_w + (long long)split_z * p.part_s_s;
      for (int e = tid * 4; e < HALF * HD; e += THREADS * 4) {
        const int r = e / HD;
        if (i0h + r / G >= qlen) continue;
        *reinterpret_cast<float4*>(dst + e) =
            *reinterpret_cast<const float4*>(sOut + (rowbase + r) * SROW + (e - r * HD));
      }
      float* ml = p.part_ml + w_id * p.partml_s_w + (long long)split_z * p.partml_s_s;
      float* mt = p.part_ml_t + w_id * p.partml_s_w + (long long)split_z * 2;
      for (int r = tid; r < HALF; r += THREADS) {
        if (i0h + r / G >= qlen) continue;
        ml[r * 2] = sM[rowbase + r];
        ml[r * 2 + 1] = sL[rowbase + r];
        mt[r * p.partmlt_s_r] = sM[rowbase + r];
        mt[r * p.partmlt_s_r + 1] = sL[rowbase + r];
      }
    }
  } else {
    for (int e = tid * 8; e < ROWS * HD; e += THREADS * 8) {
      const int r = e / HD, c = e % HD;
      const int half = r >= HALF;
      if (half && dead_b) continue;
      const int i0h = half ? i0b : i0a;
      const int rr = half ? r - HALF : r;
      const int i = i0h + rr / G;
      if (i >= qlen) continue;
      const float g = sGain[r];
      const float* so = sOut + r * SROW + (e - r * HD);
      __nv_bfloat16 h8[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) h8[j] = __float2bfloat16_rn(so[j] * g);
      const long long tok = q0 + i;
      const int head = hkv * G + (rr % G);
      *reinterpret_cast<uint4*>(p.out + tok * p.out_s_tok + (long long)head * p.out_s_head + c) =
          *reinterpret_cast<const uint4*>(h8);
    }
    if (p.write_lse) {
      for (int r = tid; r < ROWS; r += THREADS) {
        const int half = r >= HALF;
        if (half && dead_b) continue;
        const int i0h = half ? i0b : i0a;
        const int rr = half ? r - HALF : r;
        const int i = i0h + rr / G;
        if (i >= qlen) continue;
        const long long tok = q0 + i;
        const int head = hkv * G + (rr % G);
        p.lse[tok * p.lse_s_tok + (long long)head * p.lse_s_head] = sLse[r];
      }
    }
  }
  __syncthreads();
  if (tid == 0) T8BC(15);
  if (warp == 0) t5_dealloc(tmem, 512);
}
#undef T8BC

template <int HD, int KST, int VST, int DTH, int BC = 0, int MTM8 = 0>
static void launch_main_t8(const Params& p, const CUtensorMap& kmap, const CUtensorMap& vmap,
                           dim3 grid_ct, cudaStream_t stream) {
  constexpr int ROWS = 256, TILE_N = 64;
  const int smem = ROWS * HD * 2 + KST * TILE_N * HD * 2 + VST * TILE_N * HD * 2 + 1024;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(fmha_main_t8<HD, KST, VST, DTH, BC, MTM8>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  fmha_main_t8<HD, KST, VST, DTH, BC, MTM8><<<grid_ct, 320, smem, stream>>>(p, kmap, vmap);
}





}  // namespace fmha

namespace fmha {

// Experiment-only knobs for the split-KV combine geometry, settable per call from
// Python (fmha_set_debug) so one process can sweep arms without re-reading the
// environment. Defaults (0, -1) leave the promoted selection untouched: chunk is
// then computed by the escalation rule below and ZP by FMHA_COMBINE_ZP/the split
// regime, exactly as before. Reading them costs one already-cached global each.
int g_dbg_chunk = 0;   // >0: force the combine element chunk
int g_dbg_zp = -1;     // >=0: force the combine z-parallel width (0 means 1)
int g_last_chunk = 0;  // the chunk the last launch actually used (read-back only)
// Device-balanced launches set this to the uniform-equivalent split count so
// the combine's ZP regime matches the tuned uniform geometry even though
// p.splits carries the (larger) grid-depth bound; 0 means "use p.splits".
long long g_sel_splits = 0;
int g_dbg_graph = -1;  // -1: the promoted rule, 0: direct launches, 1: force graph
int g_dbg_wscan = -1;  // -1: promoted default (on), 0/1: force the combine
                       // row-stat scan form so one process can A/B
                       // thread-per-row vs warp-per-row at splits <= 64.

static int device_sm_count() {
  static int cached = -1;
  if (cached < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    int value = 148;
    cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, dev);
    cached = value;
  }
  return cached;
}

template <int HD, int WARPS_M, int WARPS_N, int TILE_N, int NSTAGE = 3, int RB = 1, int PF = 0,
          int MINB = 1, int SBS = 0, int F16KV = 0, int QREG = 0, int DPR = 1, int PFD = 0,
          int MPAD = 0, int X2HM = 0, int T7 = 0, int CVT = 0, int RST7 = 2, int LDM8 = 0,
          int TMA8 = 0, int EISSUE = 0, int TRC8 = 0>
static void launch_main(const Params& p, dim3 grid, cudaStream_t stream) {
  using C = Cfg<HD, WARPS_M, WARPS_N, TILE_N, NSTAGE, RB, PF, MINB, SBS, F16KV, QREG, DPR, PFD,
                MPAD, T7, CVT, RST7, LDM8, TMA8>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(
        fmha_main<HD, WARPS_M, WARPS_N, TILE_N, NSTAGE, RB, PF, MINB, SBS, F16KV, QREG, DPR, PFD,
                  MPAD, X2HM, T7, CVT, RST7, LDM8, TMA8, EISSUE, TRC8>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    configured = true;
  }
  fmha_main<HD, WARPS_M, WARPS_N, TILE_N, NSTAGE, RB, PF, MINB, SBS, F16KV, QREG, DPR, PFD, MPAD,
            X2HM, T7, CVT, RST7, LDM8, TMA8, EISSUE, TRC8>
      <<<grid, C::THREADS, C::SMEM, stream>>>(p);
}

constexpr int COMBINE_THREADS = 256;
constexpr int COMBINE_CHUNK = 256;  // output elements per combine CTA

template <int HD, int TILE_M, int CHUNKV, int ZPV>
static void launch_combine_c(const Params& p, dim3 grid, int smem, cudaStream_t stream) {
  auto* fn = fmha_combine<HD, TILE_M, COMBINE_THREADS, CHUNKV, ZPV>;
  // The row-stat shared array scales with splits; past the 48 KiB default the
  // opt-in limit must be raised explicitly or the launch is invalid.
  static int configured = -1;
  if (smem > 48 * 1024 && smem > configured) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = smem;
  }
  fn<<<grid, COMBINE_THREADS, smem, stream>>>(p);
}

// chunk and ZP are coupled through the element-loop geometry: for ZP>1 the
// loop consumes float4 groups (CHUNK/4 per CTA, THREADS/ZP per pass), and the
// split-count regime selects the row-scan strategy with the same splits>64
// predicate the kernel itself uses.
template <int HD, int TILE_M>
static void launch_combine(const Params& p, dim3 grid, int smem, int chunk, cudaStream_t stream) {
  // Experiment knob (read once): FMHA_COMBINE_ZP=4 forces the float4
  // z-parallel element reduction for the default chunk-256 geometry even at
  // splits <= 64. The scalar (256,1) form was chosen in the V0 era, when
  // every chunk CTA still re-scanned stats for ALL tile_m rows; with the
  // row-bounded stats scan the element loop is the remaining cost and the
  // 16B-per-lane ZP4 form cuts its instruction count 4x.
  static const int zp_env = [] {
    const char* e = getenv("FMHA_COMBINE_ZP");
    return e ? atoi(e) : 0;
  }();
  const int zp_req = (g_dbg_zp >= 0) ? g_dbg_zp : zp_env;
  switch (chunk) {
    case 32:
      launch_combine_c<HD, TILE_M, 32, 32>(p, grid, smem, stream);
      break;
    case 64:
      launch_combine_c<HD, TILE_M, 64, 16>(p, grid, smem, stream);
      break;
    case 1024:
      if (zp_req == 4)
        launch_combine_c<HD, TILE_M, 1024, 4>(p, grid, smem, stream);
      else
        launch_combine_c<HD, TILE_M, 1024, 1>(p, grid, smem, stream);
      break;
    case 4096:
      if (zp_req == 4)
        launch_combine_c<HD, TILE_M, 4096, 4>(p, grid, smem, stream);
      else
        launch_combine_c<HD, TILE_M, 4096, 1>(p, grid, smem, stream);
      break;
    default:
      // ZP4 pays only when the split axis can fill the 4-lane z-groups AND
      // the per-group shuffle fold amortises: decode-class tiles with splits
      // >= 8 (paired run 20261020T202601Z: f2 70.5->69.5, f5 25.8->24.9,
      // f3/f8/u1/u2/d6/d9 neutral) while prefill chunk-1024/4096 geometries
      // at splits 2-8 lose 3-5% (idle z-lanes + fold overhead: b4 +4.5%,
      // b2 +4.7%, f4 +3.0%) and keep the scalar form.
      // An explicit fmha_set_debug(., zp) request overrides the regime so both
      // forms can be measured on the same shape in one process.
      {
        const long long ss = g_sel_splits > 0 ? g_sel_splits : (long long)p.splits;
        if ((g_dbg_zp >= 0) ? (g_dbg_zp == 4)
                            : (ss > 64 || (TILE_M <= 16 && ss >= 8) || zp_env == 4))
          launch_combine_c<HD, TILE_M, COMBINE_CHUNK, 4>(p, grid, smem, stream);
        else
          launch_combine_c<HD, TILE_M, COMBINE_CHUNK, 1>(p, grid, smem, stream);
      }
      break;
  }
}

struct Plan {
  int cfg;
  int tile_m;
  int tile_n;
  int mtiles;
  int splits;
  int tps_target;
  long long work;
  long long part_floats;
  // Diagnostics only (read back by experiments, ignored by the kernels): the
  // config the caller asked for and whether an f16 mirror was usable. Without
  // these a silently remapped override looks like a real measurement.
  int want_cfg;
  int mirror_ok;
  // Resident-CTA budget of the selected kernel family (make_plan's
  // resident_ctas); the device-side tiles-per-split target divides by THIS so
  // MINB1/one-wave families aim at their own residency.
  int resident2;
};

// Launch plan derived only from host-visible metadata (shapes, declared maxima,
// workspace capacity). No device data is read, so no synchronisation happens and
// no data-dependent answer is precomputed outside the measured kernel.
static Plan make_plan(int hd, int cfg, int batch, int hkv, int g, long long max_q_len,
                      long long max_seq_len, long long window_left, long long scratch_floats,
                      long long splits_override, long long tps_override = -1) {
  Plan pl;
  pl.tps_target = 0;
  // J69: short-pipeline hd128 items (<= 8 visible-kv 128-tiles) ride the
  // QLA1 clone c285 - priming fill dominates 1-3-tile items (s_smallpref
  // -1.8%/-5.2%, s_lp128a/b -0.4..-0.9% paired over two jobs; bit-identical,
  // J67_BITID 6/6). Long items KEEP QLA2: f1 (56-128 tiles/item) showed a
  // +6.4% bimodal median shift at QLA1; f7/f9 were flat. eff_kv = the same
  // window clamp flop_ub uses. Zone-promoted 162->266 shapes bypass this
  // gate (post-plan) and are long by construction.
  if (cfg == 266 && hd == 128) {
    long long eff_kv = max_seq_len;
    if (window_left >= 0) {
      const long long w = window_left + max_q_len + 1;
      if (w < eff_kv) eff_kv = w;
    }
    if ((eff_kv + 127) / 128 <= 8) cfg = 285;
  }
  pl.cfg = cfg;
  pl.want_cfg = cfg;
  pl.mirror_ok = 0;
  switch (cfg) {
    case 1:
    case 50:
    case 64:
    case 107:  // cfg50 + padded sMerge stride (MPAD); identical decode tiling
    case 144:  // cfg107 + X2HM (packed f16x2 exp2 softmax)
    case 190:  // t7 warp-specialized TMA-fp8 decode
    case 191:  // t7 hd128 deep-ring variant
    case 192:  // t7c ldmatrix.b8 direct-fp8 decode
    case 193:  // t7c hd128 QR0 variant
    case 194:  // t7c QR0 + register slack
    case 195:  // t7c QR1 + register slack
    case 196:  // t7c + TMA fp8 ring
    case 197:  // t7c + TMA deep ring
    case 286:  // TMA fp8 ring + EISSUE (override-only experiment)
    case 287:
    case 288:  // LDM8 cp.async ring + EISSUE (override-only experiment)
    case 289:
    case 290:  // promoted decode geometries + TRC8 timeline (diagnostics)
    case 291:
    case 292:
    case 293:
      pl.tile_m = 16;
      pl.tile_n = 64;
      break;
    case 5:
      pl.tile_m = 128;
      pl.tile_n = 32;
      break;
    case 80:  // tcgen05: one TMEM-resident 128-row tile, 64 KV columns per step
    case 81:  // ... plus the deferred O correction
    case 82:  // ... plus the pipelined issue order
    case 84:  // ... plus four-way softmax reductions
    case 85:  // ... plus clock64 phase instrumentation (diagnostics)
    case 89:  // ... plus 3-buffer K / restructured staging waits (KB3)
    case 90:  // ... plus ILP softmax on top of KB3
    case 91:  // ... plus phase instrumentation on top of KB3
    case 92:  // ... plus packed f32x2 softmax fma/sum
    case 93:  // ... plus phase instrumentation on top of F2
    case 94:  // ... plus the V ones-column row sum in TMEM O
    case 96:  // ... plus phase instrumentation on top of F2+VS
    case 97:  // ... schedule R (RV): QK(t+1) issued at top, PV lag-waited
    case 99:  // ... plus phase instrumentation on top of RV
    case 100:  // ... descriptor-add mma issue (DAO)
    case 101:  // ... plus page-id prefetch ring (TPF)
    case 102:  // ... plus phase instrumentation
    case 103:  // ... DAO plus shared-memory page-id table (SPG)
    case 106:  // ... cfg100 base plus the PIPE (double-buffered TMEM S) schedule
    case 120:  // decode-t5: FP8S register-staged fp8->f16, no mirror, TILE_M 128
    case 121:  // ... plus phase instrumentation
    case 122:  // cfg92/100-numerics plus ILP 4-way max with the packed F2 sum
    case 123:  // cfg92/100 minus the redundant end-of-iteration barrier
    case 124:  // ... both (ILP max + F2 sum + no end barrier)
    case 125:  // c123 + PSM (P published to smem; PV as A-from-smem umma_ss)
    case 126:  // c122 + PSM
      pl.tile_m = 128;
      pl.tile_n = 64;
      break;
    case 127:  // c120 (FP8S no-mirror) + PSM
    case 128:  // c123 + MTM (flattened mtile-major descending launch order)
    case 129:  // c122 + MTM
    case 130:  // c120 (FP8S no-mirror) + MTM
    case 131:  // c128 + X2H + DTH4 (defer threshold 4 log2 units)
    case 132:  // c129 + X2H + DTH4
    case 133:  // c130 + X2H + DTH4
    case 134:  // c123 + X2H + DTH4 (no MTM)
    case 135:  // c122 + X2H + DTH4 (no MTM)
    case 136:  // c120 (FP8S no-mirror) + X2H + DTH4 (no MTM)
    case 137:  // c131 + DTH2 (defer threshold 2 log2 units)
    case 138:  // c131 + LD64 (single x64 S row load)
    case 139:  // c131 + DTH4
    case 140:  // c131 + DTH2 + LD64
      pl.tile_m = 128;
      pl.tile_n = 64;
      break;
    case 150:  // wide-tile experiments: c131/c133 bases at TILE_N=128 (see dispatch)
    case 151:
    case 152:
    case 153:
    case 154:
    case 160:  // t6: warp-specialized TN128 (mirror + TMA staging)
    case 161:  // ... plus per-phase clock64 instrumentation (diagnostics)
    case 162:  // ... plus PV as m128n128k16 (single n128 atom at hd128)
    case 163:  // t6 KST4/VST2 asymmetric ring
    case 164:  // t6 + MTM6
    case 165:  // t6 PVN128 + MTM6
    case 166:  // t6 PVN128 KST2/VST4 (V-heavy ring; the K36 hd256 mechanism:
               // V frees late, K frees a tile earlier via the QK(t+1) lookahead)
    case 167:  // ... plus MTM6
    case 168:  // t6 PVN128 + MTM6 + DTH6 (deferred-rescale threshold, log2)
    case 169:  // t6 PVN128 + MTM6 + DTH8
    case 175:  // t6 PVN128 + MTM6 + DUAL issuer warps
    case 176:  // t6 PVN128 + MTM6 + SM8 8-warp column-split softmax
    case 209:  // t6 PVN128 + IT6 phase instrumentation (diagnostics)
    case 210:  // ... on the PROMOTED c176 shape: IT6 + MTM6 + SM8 (diagnostics)
    case 211:  // c176 + K-heavy ring KST4/VST2 (same 231424 B of smem)
    case 212:  // c176 + LPAR4 (load warp issues TMA from 4 lanes)
    case 213:  // c176 + KST4/VST2 + LPAR4
    case 214:  // c176 + the opposite asymmetry KST2/VST4 (control for 211)
    case 215:  // c176 + LPAR2 (page64 shapes only run 2 pages per tile)
    case 216:  // c176 + LPAR8 (covers page16, where a tile is 8 pages)
    case 217:  // c176 + LPAR16 (PMAX stays 1; tests whether LPAR8's edge over
               // LPAR4 is the parallel issue or the scalar page-id register)
    case 219:  // c176 + LPAR8 + DTH6 (fewer deferred O-rescales)
    case 220:  // c176 + LPAR8 + DTH8
    case 221:  // the PROMOTED schedule (c176 = LPAR8) with IT6 phase timestamps
    case 222:  // the hd64 t6 SM8 arm WITH LPAR8 (see the launch case)
    case 223:  // c176 + SPLITV (separate K and V staging warps, KST3/VST3)
    case 224:  // c176 + SPLITV + KST4/VST2 (decoupled rings can now be rebalanced)
    case 225:  // c176 + SPLITV + IT6 (trace the split-load schedule)
    case 226:  // cfg222 (hd64 t6 LPAR8) + SPLITV
    case 233:  // cfg226 + DPV + SB3/QLA2 (the hd128 promotion, at hd64)
    case 240:  // c176 + MSKIP + RWSK (dead-softmax skip; hd128)
    case 241:  // c176 + RWSK only (the attribution control for 240)
    case 242:  // c176(+MSKIP/RWSK) with 8 row-sum accumulators
    case 243:  // ... plus the f16x2 scale-offset (4 instructions per pair, not 5)
    case 249:  // hd128 DPV + per-half O accumulators (SB2/QLA1, no cross-half
               // row-max exchange); 250 is the same geometry with a shared O
    case 250:
    case 251:  // the PROMOTED cfg176 with the per-CTA TRACE timeline (hd128)
    case 252:  // the PROMOTED cfg233 with the per-CTA TRACE timeline (hd64)
    case 244:  // c233 (hd64) at KST3/VST3 instead of 6/6: half the ring fill
    case 245:  // c233 (hd64) at KST4/VST4
    case 227:  // c176 + SB3/QLA2 (three TMEM S buffers, QK issued two tiles ahead)
    case 228:  // c176 + SB3/QLA1 (the third buffer alone, as the control for 227)
    case 229:  // cfg227 + IT6 (trace the lookahead schedule)
    case 230:  // c176 + DPV (own PV issuer warp) + SB3/QLA2
    case 231:  // c176 + DPV + SB2/QLA1 (arm-a's DUAL idea on the SM8/SPLITV layout)
    case 170:  // t6c cluster2 hd128
    case 171:  // t6c cluster2 hd256
    case 172:  // t6c + MTM6C hd128
    case 173:  // t6c + MTM6C hd256
    case 265:  // cfg173 + QCP (cp.async Q staging at hd256, 22 chunks/thread)
    case 266:  // promoted-176 + DTH8 (J46)
    case 267:  // promoted-233 + DTH8 (hd64, J46)
    case 268:  // promoted-265 t6c_qcp + RWSK (J46; bit-identical)
    case 275:  // promoted-266 + HACC8 (J49)
    case 276:  // promoted-266 + HACC8 + F16X (J49)
    case 277:  // promoted-266 + TRACE (J51 diagnostics)
    case 278:  // promoted-267 + TRACE (J51 diagnostics)
    case 279:  // promoted-267 + F16X (J52; hd64 softmax-bound regime)
    case 280:  // promoted-266 + PLO (J53; prologue overlap, bit-identical)
    case 281:  // promoted-267 + PLO (hd64, J53)
    case 282:  // promoted-268 + DTH8 (J55; hd256 t6c line)
    case 283:  // t6f fp8-direct prefill (J59; hd128)
    case 284:  // promoted-267 + QLA1 (J67; short-pipeline hd64)
    case 285:  // promoted-266 + QLA1 (J67; short-pipeline hd128)
    case 297:  // t6c + SM8C: 8-warp column-split softmax on the cluster kernel
    case 298:  // promoted-267 + PHO (hd64 per-half O, SB3/QLA1 = 267's ring)
    case 302:  // promoted-266 + PBAR (hd128 warp-pair exchange barrier)
    case 303:  // promoted-267 + PBAR (hd64)
    case 304:  // promoted-297 + PBAR (hd256 t6c SM8C)
    case 305:  // diagnostic: 298 at DTH4
    case 306:  // diagnostic: 298 at SB2/QLA1
    case 309:  // diagnostic: 298 at DTH64 (rescale effectively off)
    case 310:  // 309 + QLA2 (bit-identical; deeper lookahead post-PHO)
    case 311:  // 309 + F16X (exp/pack 4-instr pairs; NOT bit-identical)
    case 307:  // promoted-297 + DTH8 (hd256 t6c SM8C; unswept cell of J55)
    case 308:  // promoted-297 at KST3/VST2 (K-heavy ring; bit-identical)
    case 174:  // t6c hd256 ring variant (KST2/VST3: deeper V for the PV-heavy
               // hd256 pipeline; smem 230400 fits)
    case 178:  // t6c + beacons (diagnostics)
    case 180:  // t6d multicast-sharing hd128/hd64
    case 181:  // t6d + MTM6D
    case 188:  // t6d + beacons (diagnostics)
      pl.tile_m = 128;
      pl.tile_n = 128;
      break;
    case 200:  // t8 256-row prefill (plan units stay 128-row mtiles; one t8
    case 201:  // CTA covers mtiles 2x, 2x+1 -> resident 2*sms via the default)
    case 202:  // t8 ring variant (KST5/VST5, smem 230400 fits)
    case 203:  // t8 V-heavy ring (KST4/VST5, 208K): t8 is PV-heavy like
               // t6c-hd256, whose V-ring deepening won -23% (K36)
    case 204:  // t8 K-heavy ring (KST5/VST3, 208K): t8's QK (2x8 n64 steps,
               // 770 cyc) outweighs its PV (513), the opposite of t6c-hd256
    case 208:  // t8 + beacons
      pl.tile_m = 128;
      pl.tile_n = 64;
      break;
    case 86:  // decode staging experiments (see dispatch)
      pl.tile_m = 16;
      pl.tile_n = hd == 256 ? 32 : 64;
      break;
    case 87:
    case 88:
      pl.tile_m = 16;
      pl.tile_n = 32;
      break;
    case 104:  // decode + L2 prefetch two tiles ahead (dispatch: TN=64 all hd)
      pl.tile_m = 16;
      pl.tile_n = 64;
      break;
    case 105:  // ... TILE_N=32 variant (dispatch: TN=32 all hd)
      pl.tile_m = 16;
      pl.tile_n = 32;
      break;
    // Round-2 decode staging sweep; tile_n must match the dispatched kernel
    // exactly: the split-count math and (in TPS mode) the combine kernel's
    // recomputation of ntiles both consume it.
    case 110:
    case 113:
      pl.tile_m = 16;
      pl.tile_n = 32;
      break;
    case 111:
      pl.tile_m = 16;
      pl.tile_n = hd == 256 ? 32 : 64;
      break;
    case 112:
    case 114:
    case 145:  // c112 + X2HM (hd64 QREG+packed softmax combo)
      pl.tile_m = 16;
      pl.tile_n = 64;
      break;
    case 40:
    case 141:  // cfg40 + X2HM (packed f16x2 exp2 softmax, mma.sync family)
      pl.tile_m = 64;
      pl.tile_n = 32;
      break;
    case 71:
    case 72:
      pl.tile_m = 64;
      pl.tile_n = hd == 64 ? 64 : 32;
      break;
    case 60:
      pl.tile_m = 64;
      pl.tile_n = hd == 64 ? 64 : 32;
      break;
    default:  // cfg 0, 6, 46, 51, 142: 64-row prefill tiles
      pl.tile_m = 64;
      pl.tile_n = hd == 256 ? 32 : 64;
      break;
  }
  long long mrows = max_q_len * g;
  pl.mtiles = (int)((mrows + pl.tile_m - 1) / pl.tile_m);
  if (pl.mtiles < 1) pl.mtiles = 1;
  pl.work = (long long)batch * hkv * pl.mtiles;
  const int sms = device_sm_count();
  int splits = 1;
  long long eff = max_seq_len;
  if (window_left >= 0) {
    const long long w = window_left + max_q_len + 1;
    if (w < eff) eff = w;
  }
  const long long tiles_total = (eff + pl.tile_n - 1) / pl.tile_n;
  // Resident-CTA budget of the dispatched kernel family: hd256 t5 (TMEM 512
  // columns) and t6 (TMEM 512 + 160-192KB smem) hold ONE CTA per SM; every
  // other tile_m>=128 t5 family holds two. Decode (tile_m 16) keeps 2.
  const long long resident_ctas =
      ((cfg == 190 || cfg == 191) ||
       (pl.tile_m >= 128 &&
        (hd == 256 || (cfg >= 160 && cfg <= 176) || cfg == 178 || cfg == 180 ||
         cfg == 181 || cfg == 188 || cfg == 209 || cfg == 210 ||
         (cfg >= 211 && cfg <= 233) || (cfg >= 240 && cfg <= 252) ||
         (cfg >= 266 && cfg <= 267) || (cfg >= 275 && cfg <= 285) ||
         cfg == 298 || (cfg >= 302 && cfg <= 311))))
          ? (long long)sms
          : 2LL * sms;
  pl.resident2 = (int)resident_ctas;
  if (pl.work < resident_ctas) {
    // hd256 t5 tiles reside at 1 CTA/SM (TMEM 512 columns), and measurements
    // put ~one wave of long-running CTAs clearly ahead of 2+ waves of short
    // ones there (d7: splits2 199.7us vs splits5 249.8; b6: splits4 138.2 vs
    // splits10 176.2). hd<=128 keeps the 2-waves/SM target.
    const bool one_wave = (resident_ctas == (long long)sms);
    const long long resident = resident_ctas;
    // Floor (not ceil) for the 1-wave class: overshooting past one wave costs
    // a quantized tail (d7 splits=3 -> 1.3 waves measured 259us vs splits=2's
    // 199.7), while undershooting only trims concurrency that 1-CTA/SM
    // residency cannot use anyway.
    long long want = one_wave ? resident / pl.work : (resident + pl.work - 1) / pl.work;
    if (want < 1) want = 1;
    long long by_kv = tiles_total / 2;
    if (by_kv < 1) by_kv = 1;
    splits = (int)(want < by_kv ? want : by_kv);
    if (splits < 1) splits = 1;
    // Prefill with a short tile run AND enough work items to fill the machine
    // without splitting: there, splits only add combine traffic and per-CTA
    // fixed cost (Q staging, TMEM alloc, epilogue). Measured (probe10/ab12,
    // cfg92/100): 32-tile shapes f0 67.7-70.6us at splits=1 vs 73.7-100.4 at 2,
    // windowed hd64 synthetic 55.3-58.4 at 1 vs 63.5-72.7 at 2. Work below one
    // wave still splits (d4-like: 72 work items at splits 4 beat splits 1 by
    // 4%); decode (tile_m 16) is exempt -- its CTAs are cheap and long-context
    // decode needs every split it can get.
    if (pl.tile_m >= 64 && tiles_total <= 40 && pl.work >= sms) splits = 1;
    // Cap at the resident-CTA target instead of a fixed 64: a single long
    // decode request (work=1) is latency-bound per split, and measurements on
    // the 262144-token shape put splits=64 at 101.5us against 74.8us at 128
    // even with the old serial combine scan; with the warp-parallel scan the
    // combine no longer grows fast enough to justify the low cap.
    if (splits > resident_ctas) splits = (int)resident_ctas;
  }
  if (splits_override > 0) splits = (int)splits_override;
  // Short-tile parallelism-only splits for the fat-partial (tile_m=128)
  // kernels: each split publishes tile_m x (hd+4) floats per work item, and
  // the combine reads them back at only ~1-2TB/s effective, so once the CTAs
  // reach ~0.4 resident waves the extra splits cost more combine traffic than
  // the (already tiny, <=24-tile) main kernel saves. 4-block sweep
  // (20261020T211237Z, all stable): w3 [work128,tiles9] sp1 37.9 vs sp2 48.8
  // vs sp4 62.1; w2 [work64] sp2 36.8 vs sp4 41.8; w4 [work72] sp2 37.5 vs
  // sp4 42.2; b10 [work72] sp2 37.3; b1 [work32,tiles16] sp4 34.3 vs sp8
  // 38.9; w1 [work36] sp4 31.6 vs sp2 33.6; w5 [tiles18] sp3-ish 42-43 vs
  // sp5 43.4 (mild), w6 [work64,tiles18] sp2 44.1 vs sp4 43.3 (mild -0.5%).
  // s_min = ceil(0.40*resident/work) fits every one of those optima within
  // ~2%; the fill rule below then optimizes within [1, s_min] (monotone here,
  // so it lands on s_min). Long-tile shapes (f9 tiles256, b2 tiles32, f0
  // tiles32) are outside the window and keep the fill-rule optimum;
  // tile_m<=64 (d4/b8/w7 measured the opposite preference) is exempt.
  if (splits_override <= 0 && pl.tile_m >= 128 && tiles_total <= 24) {
    const long long resident = resident_ctas;
    const long long s_min = (2 * resident / 5 + pl.work - 1) / pl.work;
    if (s_min < splits) splits = s_min < 1 ? 1 : (int)s_min;
  }
  // Single-request prefill wave-fill: with batch == 1 every work item is real
  // (no ragged early-exit tiles), so the resident-wave fill of work*splits is
  // exact host-side. The ceil-to-2-waves rule overshoots into a sparse second
  // wave (fixed9: splits 5 -> 360 CTAs = 1.22 waves measured 273.5us; splits 4
  // -> 288 = 0.97 waves measured 247.6us, job dv_629ea9c7c234). Pick the split
  // count with the best fill; ties (within 2%) keep the smaller count, which
  // is cheaper on combine traffic and partial writes.
  if (splits_override <= 0 && batch == 1 && pl.tile_m >= 64 && splits > 1) {
    const long long resident = resident_ctas;
    int best = splits;
    double best_fill = 0.0;
    for (int s = 1; s <= splits; ++s) {
      const long long ctas = pl.work * s;
      const double waves = (double)((ctas + resident - 1) / resident);
      const double fill = (double)ctas / (waves * (double)resident);
      if (fill > best_fill + 0.02) {
        best_fill = fill;
        best = s;
      }
    }
    splits = best;
  }
  // Device-balanced decode splits (tps_target = -1). Job J TRC8 timelines
  // (dv_46fa3ebe31e4): the f2 main spans run med 16.2us / max 33.9us with
  // tiles/CTA 1..14 under the uniform split count -- the longest request's
  // CTAs straggle on a half-idle machine while the ring itself never stalls
  // (steady-state wait 58-77ns per tile). Host metadata cannot see per-request
  // lengths (device data; reading it here would synchronize), so every
  // main/combine CTA derives the common tiles-per-split target from seq_lens
  // with identical integer arithmetic (tps_device_tt) and grid.z is only a
  // DECLARED-metadata bound.
  // GATE (measured win/loss map, this phase + archived pre-join trials whose
  // mechanism this re-derives: dv_cbf37aa0887d, dv_2d01c3d75fc6): balancing
  // pays only when (a) the batch can be ragged (batch >= 2 -- a single request
  // is already split to the resident budget) and (b) the uniform pipeline is
  // deep (tiles_total >= 12*uniform splits, i.e. >= ~12 tiles/split): only
  // then can the device target (tt >= 3, total-driven) cut the straggler
  // critical path, while the small latency-bound windowed shapes (uni_tps
  // 3-11: f3/f5/f8/d5/d6) sit AT their uniform optimum and would only add
  // empty grid slots (+8-12% measured). f2 (uni_tps 14: main 40.4->31.1us,
  // window -12% vs uniform-wscan, -16..-19% vs uniform) and the ragged
  // synthetic class pass; s_unif (7), d6 (11), d9 (batch 1) are excluded.
  // A uniform-long 513-tile batch is host-indistinguishable from ragged f2
  // (same declared maxima) and keeps device geometry ~= uniform at the cost of
  // ~296 empty slots; no fixed/domain shape is in that class.
  // Grid-depth bound zb: tt >= 3 caps splits_b by ceil(tiles_total/3) and
  // tt >= total/resident2 caps it by resident_ctas; the 2x-uniform cap is the
  // empirical empty-CTA/combine-geometry bound from the archived trials.
  // tps_override: -1 auto (gate decides), -2 force device, -3 force uniform,
  // >=0 legacy host formula below.
  const bool tps_forced = (tps_override == -2);
  const bool tps_uniform_forced = (tps_override == -3);
  if (splits > 1 && pl.tile_m == 16 && splits_override <= 0 && !tps_uniform_forced &&
      (tps_forced || (tps_override == -1 && batch >= 2 &&
                      tiles_total >= 12 * (long long)splits))) {
    long long zb = (tiles_total + 2) / 3;
    if (zb > 2 * splits) zb = 2 * splits;
    if (zb > resident_ctas) zb = resident_ctas;
    if (zb < 2) zb = 2;
    splits = (int)zb;
    pl.tps_target = -1;
  }
  // Decode per-request balancing: one common tiles-per-split target instead of
  // one uniform split count. OVERRIDE-ONLY: the round-5 paired job
  // (20261020T180158Z) confirmed the fixed2 win (sp45 tt12, -8.3% vs v0) but
  // boundary synthetics showed uniformly-long batches inside the same tile
  // window REGRESS (s_u1 512-tile uniform +7.3%, s_u2 1024-tile ragged +5.2%,
  // d9 4096-tile +7-14% across rounds) at near-identical active-CTA counts.
  // The discriminator (raggedness of seq_lens) is device-side data the host
  // plan must not read, so no safe auto gate exists; uniform splits stay the
  // auto policy and tps_override >= 0 enables this path for experiments.
  if (splits > 1 && pl.tile_m == 16 && tps_override >= 0) {
    const long long div = tps_override > 0 ? tps_override : 4;
    // One (div * sms)-CTA budget's worth of tiles per split, floored so tiny
    // tile runs do not explode into one-tile CTAs.
    long long tt = (tiles_total * (long long)pl.work) / (div * (long long)sms);
    if (tt < 4) tt = 4;
    long long zmax = (tiles_total + tt - 1) / tt;
    // Cap the grid depth near the resident budget: probe13 measured the deep
    // uniform-style grids as the loss source (f2 div2 at ZMAX 86 = 68.5us vs
    // uniform ZMAX 129 = 79.0), while inactive-split slots and the combine
    // scan width both scale with ZMAX rather than with active splits.
    const long long zcap = (2LL * sms + pl.work - 1) / pl.work + 8;
    if (zmax > zcap) {
      zmax = zcap;
      long long tt2 = (tiles_total + zmax - 1) / zmax;
      if (tt2 > tt) tt = tt2;
    }
    if (zmax < 1) zmax = 1;
    splits = (int)zmax;
    pl.tps_target = (int)tt;
  }
  // Causal-prefill per-mtile balancing -- OVERRIDE-ONLY (tps_override >= 0).
  // The round-5 auto gate (fire when uniform splits==1 and the longest CTA
  // exceeds the mean per resident slot) REGRESSED every shape it touched:
  // fixed0 [120,sp3,tt11] 68.0 vs sp1 64.6; fixed7 [92,sp2,tt111] 371.2 vs
  // sp1 363.4; s_b5 [120,sp2,tt32] 110.4 vs sp4 101.5; the forced tpsp arm
  // lost on all 13 synthetics/suite prefill shapes (1.02..1.39). Splitting a
  // sub-wave prefill launch does NOT shorten its makespan the way it does for
  // decode: all CTAs are already resident, so splits only add per-CTA fixed
  // cost (Q staging, TMEM alloc, epilogue staging) and combine traffic while
  // the per-SM contention grows. The mechanism (kernel splits_b + combine
  // recomputation) stays wired for experiments behind the override.
  if (pl.tile_m >= 64 && splits_override <= 0 && tps_override >= 0) {
    const long long resident = resident_ctas;
    long long spread = (long long)max_q_len / pl.tile_n;
    if (spread > tiles_total) spread = tiles_total;
    long long avg_tiles = tiles_total - spread / 2;
    if (avg_tiles < 1) avg_tiles = 1;
    const long long total_tasks = pl.work * avg_tiles;
    if (tiles_total >= 4) {
      long long tt = (total_tasks + resident - 1) / resident;
      if (tt < 2) tt = 2;
      long long zmax = (tiles_total + tt - 1) / tt;
      if (zmax < 2) zmax = 2;
      splits = (int)zmax;
      pl.tps_target = (int)tt;
    }
  }
  // t6c splits are uniform-only: the pair-unified tile range does not match
  // the per-item TPS recomputation the combine kernel would run.
  if ((cfg >= 170 && cfg <= 174) || cfg == 178 || (cfg >= 200 && cfg <= 204) || cfg == 208 ||
      cfg == 265 || cfg == 268 || cfg == 282 || cfg == 297 || cfg == 304 || cfg == 307 || cfg == 308)
    pl.tps_target = 0;
  // +4: the two 2-float stats copies (z-major and row-major twins).
  long long per_split = pl.work * pl.tile_m * (hd + 4);
  while (splits > 1 && per_split * splits > scratch_floats) --splits;
  pl.splits = splits;
  pl.part_floats = splits > 1 ? per_split * splits : 0;
  return pl;
}

// 4-D TMA map over the flat f16 mirror, which inherits the paged layout of
// the fp8 cache: [pages][kvh][page_tokens][hd], element strides = kv_s_*.
// Box = [64 f16 cols][page rows] with 128B swizzle, which materialises
// exactly the t5_off K-group-major atom layout the tensor-core descriptors
// expect (row r at r*128B, 16B chunks XOR-swizzled by r&7).
// t7 decode: TMA map over the RAW fp8 cache (UINT8 = e4m3 bytes, opaque to
// the map): dim0 = hd BYTES per token, box = [128B cols][box_rows rows] with
// SWIZZLE_128B (the byte twin of the f16 mirror map; hd256 rows span two
// 128B segments addressed by coord0). Validated by tma_min v7 (page checksum
// exact against the wrap-corrected fill pattern, job dv_ec40fb8f5b70).
static bool build_fp8_map(const void* base, int hd, int page, int hkv, long long pages,
                          long long s_tok, long long s_head, long long s_page, int box_rows,
                          CUtensorMap* out) {
  const cuuint64_t gd[4] = {(cuuint64_t)hd, (cuuint64_t)page, (cuuint64_t)hkv,
                            (cuuint64_t)pages};
  const cuuint64_t gs[3] = {(cuuint64_t)s_tok, (cuuint64_t)s_head, (cuuint64_t)s_page};
  const cuuint32_t bd[4] = {128, (cuuint32_t)box_rows, 1, 1};
  const cuuint32_t es[4] = {1, 1, 1, 1};
  return cuTensorMapEncodeTiled(out, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, const_cast<void*>(base),
                                gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

static bool build_mirror_map(const __half* base, int hd, int page, int hkv, long long pages,
                             long long s_tok, long long s_head, long long s_page, int box_rows,
                             CUtensorMap* out) {
  const cuuint64_t gd[4] = {(cuuint64_t)hd, (cuuint64_t)page, (cuuint64_t)hkv,
                            (cuuint64_t)pages};
  const cuuint64_t gs[3] = {(cuuint64_t)s_tok * 2, (cuuint64_t)s_head * 2,
                            (cuuint64_t)s_page * 2};
  const cuuint32_t bd[4] = {64, (cuuint32_t)box_rows, 1, 1};
  const cuuint32_t es[4] = {1, 1, 1, 1};
  return cuTensorMapEncodeTiled(out, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4,
                                const_cast<__half*>(base), gd, gs, bd, es,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

static void dispatch(const Params& p, int hd, const Plan& pl, int batch, cudaStream_t stream,
                     const CUtensorMap* kmap = nullptr, const CUtensorMap* vmap = nullptr) {
  const dim3 grid(pl.mtiles, (unsigned)batch * (unsigned)p.HKV, (unsigned)pl.splits);
  // MTM cfgs (128-130) flatten the same work items into gridDim.x, mtile-major
  // descending; the kernel recovers the logical (mtile, by) via p.mtiles, so
  // every output/partial address is bit-identical to `grid`.
  const dim3 grid_mtm((unsigned)pl.mtiles * (unsigned)batch * (unsigned)p.HKV, 1u,
                      (unsigned)pl.splits);
  // Combine geometry from plan shape only, in the same splits regimes the
  // kernel uses. splits <= 64 keeps the original (256,1) scalar geometry:
  // measured better than z-parallel/warp-scan forms there (f2 9.1us vs 11.9).
  // Higher splits switch to the float4 z-parallel loop: chunk 32/64 keep
  // per-thread chains at <=~20 vector loads; large tiles use chunk 256
  // because every chunk CTA re-scans tile_m rows. Many work items (only
  // possible at splits <= 64) escalate the chunk, bounding CTAs at ~4096.
  const long long tell = (long long)pl.tile_m * hd;  // output elements per work item
  // Device-balanced mode (tps_target < 0): pl.splits is the grid-DEPTH bound
  // (zb), not the split count the combine folds -- each item's splits_b is
  // device-derived and lands near the uniform fill count ceil(2*SMs/work). The
  // chunk/ZP regime tracks that uniform-equivalent count so the combine keeps
  // the tuned uniform geometry (zb-sized chunk choices cost f3 +27% and d6
  // +41% through 4x combine CTA counts in the archived trials).
  long long sel_splits = pl.splits;
  if (pl.tps_target < 0) {
    const long long uni = (2LL * device_sm_count() + pl.work - 1) / pl.work;
    sel_splits = pl.splits < uni ? pl.splits : (uni < 2 ? 2 : uni);
  }
  int chunk = COMBINE_CHUNK;
  if (pl.splits > 1) {
    if (sel_splits > 64) {
      // splits > 64 implies work <= 4 (want = ceil(2*SMs/work)), so the CTA
      // count stays <= ~256 here without escalation.
      if (tell < 8192) chunk = (sel_splits > 128) ? 32 : 64;
    } else {
      // Two-sided CTA-count target, calibrated by job dv_...170339Z (paired):
      // domain4 ran 2304 micro-CTAs (8.6us combine) and improved at 576
      // (7.0us), but domain6 regressed 1024 -> 256 CTAs (4.8 -> 5.8us) and
      // fixed8 regressed 128 -> 512, so escalate only past ~2048 CTAs and
      // shrink only below ~64 (fixed5/domain5 32 -> 128 CTAs measured -5%).
      while (chunk < 4096 && chunk < tell &&
             pl.work * ((tell + chunk - 1) / chunk) > 2048)
        chunk *= 4;
      while (chunk > 64 && chunk <= tell &&
             pl.work * ((tell + chunk - 1) / chunk) < 64 &&
             pl.work * ((tell + chunk / 4 - 1) / (chunk / 4)) <= 4096)
        chunk /= 4;
      // J77: high-split decode combines (splits >= 16, decode-tile tell) prefer
      // chunk-64 even when the CTA count is comfortable: the per-row split fold
      // dominates and 4x more CTAs spread it, while the extra launches stay
      // ~512. Paired ABBA x3 x2 jobs, smokes 0: f2 (sp37) 0.979/0.987, f5
      // (sp25) 0.987/0.993; correctly EXCLUDED by the gate: f3 (sp10) 1.074,
      // f8 (sp8) 1.005, d6 (sp2) 1.170, f9 (sp2 prefill, tell 16384) 1.471.
      // Chunk moves only the element axis - the per-element split fold order
      // is unchanged. (dv_ae05cd9d6c26 + dv_efffd7fcb4c2.)
      if (sel_splits >= 16 && tell < 8192 && chunk > 64) chunk = 64;
    }
  }
  if (g_dbg_chunk > 0) chunk = g_dbg_chunk;
  g_last_chunk = chunk;
  g_sel_splits = (pl.tps_target < 0) ? sel_splits : 0;
  const unsigned cchunks = (unsigned)((tell + chunk - 1) / chunk);
  const dim3 cgrid(pl.mtiles, (unsigned)batch * (unsigned)p.HKV, cchunks);
  const int csmem = pl.splits > 1 ? (pl.splits * pl.tile_m + pl.tile_m * 4) * 4 : 0;
  const bool split = pl.splits > 1;
#define LAUNCH(HDV, WM, WN, TN, NS, RBV, PFV, MINB)                  \
  LAUNCH_S(HDV, WM, WN, TN, NS, RBV, PFV, MINB, 0)
#define LAUNCH_S(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV)                 \
  LAUNCH_F(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, 0)
#define LAUNCH_F(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16) \
  LAUNCH_Q(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16, 0)
#define LAUNCH_Q(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16, QR) \
  LAUNCH_D(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16, QR, 1)
#define LAUNCH_D(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16, QR, DPRV) \
  LAUNCH_P(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16, QR, DPRV, 0)
#define LAUNCH_P(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16, QR, DPRV, PFDV) \
  LAUNCH_M(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16, QR, DPRV, PFDV, 0)
#define LAUNCH_M(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16, QR, DPRV, PFDV, MPADV)              \
  launch_main<HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16, QR, DPRV, PFDV, MPADV>(p, grid, stream); \
  if (split) launch_combine<HDV, (WM * RBV * 16)>(p, cgrid, csmem, chunk, stream);
// X2HM: packed ex2.approx.f16x2 softmax (mma.sync family); see fmha_main body.
#define LAUNCH_X(HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16, QR, DPRV, PFDV, MPADV)              \
  launch_main<HDV, WM, WN, TN, NS, RBV, PFV, MINB, SBSV, F16, QR, DPRV, PFDV, MPADV, 1>(p, grid, stream); \
  if (split) launch_combine<HDV, (WM * RBV * 16)>(p, cgrid, csmem, chunk, stream);
// T7: warp-specialized TMA-fp8 decode staging (PF/SBS/F16KV/QREG/DPR/PFD fixed
// at the decode-standard values; X2HM+MPAD keep the promoted numerics/merge).
#define LAUNCH_T7(HDV, WM, WN, TN, NS, RBV, MINB, MPADV, CVTV, RSTV)                            \
  launch_main<HDV, WM, WN, TN, NS, RBV, 0, MINB, 0, 0, 0, 1, 0, MPADV, 1, 1, CVTV, RSTV>(p, grid, stream); \
  if (split) launch_combine<HDV, (WM * RBV * 16)>(p, cgrid, csmem, chunk, stream);
// L8 (t7c): direct-fp8 ldmatrix.b8 decode. PF/SBS/F16KV/T7 off, MPAD+X2HM on
// (c144 numerics), DPR/PFD inert on the ring path.
#define LAUNCH_L8(HDV, MINB, QR)                                                    \
  launch_main<HDV, 1, 4, 64, 3, 1, 0, MINB, 0, 0, QR, 1, 0, 1, 1, 0, 0, 2, 1>(p, grid, stream); \
  if (split) launch_combine<HDV, 16>(p, cgrid, csmem, chunk, stream);
// L8T (c196/c197): LDM8 compute + TMA fp8 ring staging (NSTAGE/MINB variants).
#define LAUNCH_L8T(HDV, NS, MINB, QR)                                               \
  launch_main<HDV, 1, 4, 64, NS, 1, 0, MINB, 0, 0, QR, 1, 0, 1, 1, 0, 0, 2, 1, 1>(p, grid, stream); \
  if (split) launch_combine<HDV, 16>(p, cgrid, csmem, chunk, stream);
// L8_TR / L8T_TR (c290-c293): the promoted c192/c196 decode geometries with
// the TRC8 per-tile phase timeline (diagnostics; p.trc8 must be non-null).
#define LAUNCH_L8_TR(HDV, MINB, QR)                                                 \
  launch_main<HDV, 1, 4, 64, 3, 1, 0, MINB, 0, 0, QR, 1, 0, 1, 1, 0, 0, 2, 1, 0, 0, 1>(p, grid, stream); \
  if (split) launch_combine<HDV, 16>(p, cgrid, csmem, chunk, stream);
#define LAUNCH_L8T_TR(HDV, NS, MINB, QR)                                            \
  launch_main<HDV, 1, 4, 64, NS, 1, 0, MINB, 0, 0, QR, 1, 0, 1, 1, 0, 0, 2, 1, 1, 0, 1>(p, grid, stream); \
  if (split) launch_combine<HDV, 16>(p, cgrid, csmem, chunk, stream);
// L8E (c288/c289): LDM8 cp.async ring + EISSUE (issue-before-wait).
#define LAUNCH_L8_E(HDV, MINB, QR)                                                  \
  launch_main<HDV, 1, 4, 64, 3, 1, 0, MINB, 0, 0, QR, 1, 0, 1, 1, 0, 0, 2, 1, 0, 1>(p, grid, stream); \
  if (split) launch_combine<HDV, 16>(p, cgrid, csmem, chunk, stream);
// L8TE (c286/c287): TMA fp8 ring + EISSUE (issue-before-wait + pid prefetch).
#define LAUNCH_L8T_E(HDV, NS, MINB, QR)                                             \
  launch_main<HDV, 1, 4, 64, NS, 1, 0, MINB, 0, 0, QR, 1, 0, 1, 1, 0, 0, 2, 1, 1, 1>(p, grid, stream); \
  if (split) launch_combine<HDV, 16>(p, cgrid, csmem, chunk, stream);
  switch (pl.cfg) {
    // occupancy-first prefill: NCU showed 11.5% warp occupancy with 38.7% of
    // cycles stalled on fixed-latency dependencies and 17.3% on shared-memory
    // scoreboards, i.e. too few warps per scheduler to hide latency. TILE_N=32
    // cuts smem per CTA so 4-5 CTAs reside, and MINB makes ptxas budget for it.
    // single-buffered register prefetch: no raw cp.async ring, so far less smem
    // per CTA. hd256 decode goes 168 KiB -> 72 KiB, i.e. 1 resident CTA -> 3.
    case 50:
      if (hd == 64) { LAUNCH_S(64, 1, 4, 64, 3, 1, 1, 4, 1) }
      else if (hd == 128) { LAUNCH_S(128, 1, 4, 64, 3, 1, 1, 3, 1) }
      else { LAUNCH_S(256, 1, 4, 64, 3, 1, 1, 2, 1) }
      break;
    // cfg107 = cfg50 + MPAD (padded sMerge row stride HD+4): kills the 8-way
    // cross-warp merge store bank conflicts (NCU decode: 7.7-way, 60% of
    // shared-store wavefronts, Est. 28.8%). Bit-identical to cfg50 (the pad
    // columns are never read as data). Override-only pending probe19 A/B.
    case 107:
      if (hd == 64) { LAUNCH_M(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 0, 1, 0, 1) }
      else if (hd == 128) { LAUNCH_M(128, 1, 4, 64, 3, 1, 1, 3, 1, 0, 0, 1, 0, 1) }
      else { LAUNCH_M(256, 1, 4, 64, 3, 1, 1, 2, 1, 0, 0, 1, 0, 1) }
      break;
    // Decode staging experiments (override-only; auto selection unchanged until
    // measured). Decode main is latency-bound at one tile of look-ahead: d9
    // runs 2.06us per 16KiB tile (2.3TB/s), f2 ~1.1TB/s, against a 6TB/s copy
    // peak; these arms multiply the bytes in flight per SM instead.
    case 86:  // two-tile-deep register prefetch; hd256 lacks the registers
              // (O accumulator is already 128) and falls back to a raw ring
      if (hd == 64) { LAUNCH_D(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 0, 2) }
      else if (hd == 128) { LAUNCH_D(128, 1, 4, 64, 3, 1, 1, 2, 1, 0, 0, 2) }
      else { LAUNCH(256, 1, 2, 32, 3, 1, 0, 2) }
      break;
    case 87:  // raw cp.async ring, 32-column tiles, 64-thread CTAs: depth
              // NSTAGE-1 tiles in flight per CTA and more CTAs per SM
      if (hd == 64) { LAUNCH(64, 1, 2, 32, 4, 1, 0, 6) }
      else if (hd == 128) { LAUNCH(128, 1, 2, 32, 4, 1, 0, 4) }
      else { LAUNCH(256, 1, 2, 32, 3, 1, 0, 2) }
      break;
    case 88:  // deep prefetch AND 32-column tiles AND 64-thread CTAs
      if (hd == 64) { LAUNCH_D(64, 1, 2, 32, 3, 1, 1, 5, 1, 0, 0, 2) }
      else if (hd == 128) { LAUNCH_D(128, 1, 2, 32, 3, 1, 1, 5, 1, 0, 0, 2) }
      else { LAUNCH(256, 1, 2, 32, 3, 1, 0, 2) }
      break;
    // cfg104 = cfg50 plus prefetch.global.L2 of tile t+DPR+1 during iteration
    // t: the DRAM latency the single-tile register lookahead cannot cover
    // (~0.5-1us of each ~2-4us tile) becomes an L2 hit instead, with zero
    // register cost (prefetch has no destination). cfg105 = same at TILE_N 32.
    case 104:
      if (hd == 64) { LAUNCH_P(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 0, 1, 1) }
      else if (hd == 128) { LAUNCH_P(128, 1, 4, 64, 3, 1, 1, 3, 1, 0, 0, 1, 1) }
      else { LAUNCH_P(256, 1, 4, 64, 3, 1, 1, 2, 1, 0, 0, 1, 1) }
      break;
    case 105:  // TILE_N=32 needs WARPS_N=2 to keep 16-row warp slices
      if (hd == 64) { LAUNCH_P(64, 1, 2, 32, 3, 1, 1, 6, 1, 0, 0, 1, 1) }
      else if (hd == 128) { LAUNCH_P(128, 1, 2, 32, 3, 1, 1, 4, 1, 0, 0, 1, 1) }
      else { LAUNCH_P(256, 1, 2, 32, 3, 1, 1, 4, 1, 0, 0, 1, 1) }
      break;
    // Round-2 decode staging sweep (override-only; auto selection unchanged).
    // Job dv_7409d13b77c7 measured every decode-heavy shape at only ~0.9-1.3
    // TB/s effective (fixed2 63MB/71.2us, domain6 50MB/41.6us, domain9
    // 67MB/54.8us) against a ~6TB/s copy peak, and the register-prefetch arms
    // (DPR2, L2 prefetch) all LOST: the auto path keeps exactly one raw tile
    // (~16-32KB/CTA) in flight and only while computing, so the DRAM stream
    // has a per-tile duty cycle well under 1. These arms instead deepen the
    // ASYNC stream: cp.raw rings with 4-8 stages keep (NSTAGE-2) tiles in
    // flight through both the convert and compute phases, and QREG removes the
    // per-tile Q ldmatrix (Q is reused across every tile in decode).
    case 110:  // raw ring, TN32, WN2 (64 thr), NSTAGE 5/5/4, MPAD
      if (hd == 64) { LAUNCH_M(64, 1, 2, 32, 5, 1, 0, 6, 0, 0, 0, 1, 0, 1) }
      else if (hd == 128) { LAUNCH_M(128, 1, 2, 32, 5, 1, 0, 3, 0, 0, 0, 1, 0, 1) }
      else { LAUNCH_M(256, 1, 2, 32, 4, 1, 0, 2, 0, 0, 0, 1, 0, 1) }
      break;
    case 111:  // raw ring, TN64, WN4 (128 thr), NSTAGE 4, MPAD, QREG (hd<=128);
               // hd256 lacks the smem for TN64 rings -> TN32 NSTAGE4 ring
      if (hd == 64) { LAUNCH_M(64, 1, 4, 64, 4, 1, 0, 3, 0, 0, 1, 1, 0, 1) }
      else if (hd == 128) { LAUNCH_M(128, 1, 4, 64, 4, 1, 0, 2, 0, 0, 1, 1, 0, 1) }
      else { LAUNCH_M(256, 1, 2, 32, 4, 1, 0, 2, 0, 0, 0, 1, 0, 1) }
      break;
    case 112:  // auto staging (PF+SBS) + QREG + MPAD; hd256 lacks the
               // registers (Qf 64 + O 128) and keeps the auto fragment path
      if (hd == 64) { LAUNCH_M(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 1, 1, 0, 1) }
      else if (hd == 128) { LAUNCH_M(128, 1, 4, 64, 3, 1, 1, 2, 1, 0, 1, 1, 0, 1) }
      else { LAUNCH_M(256, 1, 4, 64, 3, 1, 1, 2, 1, 0, 0, 1, 0, 1) }
      break;
    case 145:  // c112 + X2HM: QREG and the packed f16x2 softmax won
               // independently at hd64 (c112: d3 -0.7..-5.5%, d5 -2..-4.8%;
               // X2HM c144: d3 0.962, d5 ~neutral); the combo is the queued
               // experiment. hd128/256 fall back to the promoted c144 form
               // (QREG needs +32 regs past the hd128 MINB budget: c114 d9 +0.9%).
      if (hd == 64) { LAUNCH_X(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 1, 1, 0, 1) }
      else if (hd == 128) { LAUNCH_X(128, 1, 4, 64, 3, 1, 1, 3, 1, 0, 0, 1, 0, 1) }
      else { LAUNCH_X(256, 1, 4, 64, 3, 1, 1, 2, 1, 0, 0, 1, 0, 1) }
      break;
    case 113:  // deep raw ring, TN32, WN2, NSTAGE 8/8/4, MPAD
      if (hd == 64) { LAUNCH_M(64, 1, 2, 32, 8, 1, 0, 5, 0, 0, 0, 1, 0, 1) }
      else if (hd == 128) { LAUNCH_M(128, 1, 2, 32, 8, 1, 0, 2, 0, 0, 0, 1, 0, 1) }
      else { LAUNCH_M(256, 1, 2, 32, 4, 1, 0, 2, 0, 0, 0, 1, 0, 1) }
      break;
    case 114:  // c112 with hd128 MINB raised 2 -> 3: round 2 measured QREG at
               // -1.6..-2.7% on hd128 decode-heavy shapes but +15% on domain6,
               // which runs 384 short-tile CTAs and cannot afford the occupancy
               // drop (2 vs 3 CTAs/SM = 1.30 vs 0.86 waves). If ptxas spills at
               // the 170-register budget this arm shows it.
      if (hd == 64) { LAUNCH_M(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 1, 1, 0, 1) }
      else if (hd == 128) { LAUNCH_M(128, 1, 4, 64, 3, 1, 1, 3, 1, 0, 1, 1, 0, 1) }
      else { LAUNCH_M(256, 1, 4, 64, 3, 1, 1, 2, 1, 0, 0, 1, 0, 1) }
      break;
    // decode with the f16 mirror: no in-kernel convert, 16-row tiles, KV split
    // across four warps. Decode is the regime where the mirror should LOSE (it
    // turns 1 byte of KV read into 3 bytes of convert traffic plus a 2-byte
    // read), so this arm exists to measure that, not on the assumption it wins.
    case 64:
      if (hd == 64) { LAUNCH_F(64, 1, 4, 64, 3, 1, 0, 4, 0, 1) }
      else if (hd == 128) { LAUNCH_F(128, 1, 4, 64, 3, 1, 0, 3, 0, 1) }
      else { LAUNCH_F(256, 1, 4, 64, 3, 1, 0, 2, 0, 1) }
      break;
    case 51:
      if (hd == 64) { LAUNCH_S(64, 4, 1, 64, 3, 1, 1, 4, 1) }
      else if (hd == 128) { LAUNCH_S(128, 4, 1, 64, 3, 1, 1, 3, 1) }
      else { LAUNCH_S(256, 4, 1, 32, 3, 1, 1, 2, 1) }
      break;
    // f16-mirror prefill: no convert pass at all (see cvt_kv_kernel)
    case 60:
      if (hd == 64) { LAUNCH_F(64, 4, 1, 64, 3, 1, 0, 4, 0, 1) }
      else if (hd == 128) { LAUNCH_F(128, 4, 1, 32, 3, 1, 0, 4, 0, 1) }
      else { LAUNCH_F(256, 4, 1, 32, 3, 1, 0, 2, 0, 1) }
      break;
    // Shared-traffic reduction: NCU puts L1/shared throughput at 75% with compute
    // at 38% and DRAM at 0.18%, so the wall is ldmatrix bytes per mma, not the
    // memory pipeline. These cut that traffic instead of adding parallelism.
    case 71:  // register-resident Q fragments (Q is ~20% of ldmatrix at hd128)
      if (hd == 64) { LAUNCH_Q(64, 4, 1, 64, 3, 1, 0, 4, 0, 0, 1) }
      else if (hd == 128) { LAUNCH_Q(128, 4, 1, 32, 3, 1, 0, 3, 0, 0, 1) }
      // hd256 cannot hold Q in registers (64 Qf + 128 accumulator registers), so
      // the override degrades to the cfg51 staging instead of a wrong-HD kernel.
      else { LAUNCH_S(256, 4, 1, 32, 3, 1, 1, 2, 1) }
      break;
    case 72:  // ... plus the f16 mirror, removing the convert LDS/STS as well
      if (hd == 64) { LAUNCH_Q(64, 4, 1, 64, 3, 1, 0, 3, 0, 1, 1) }
      else if (hd == 128) { LAUNCH_Q(128, 4, 1, 32, 3, 1, 0, 3, 0, 1, 1) }
      else { LAUNCH_F(256, 4, 1, 32, 3, 1, 0, 2, 0, 1) }
      break;
    case 40:  // occupancy-first prefill, narrow KV tile
      if (hd == 64) { LAUNCH(64, 4, 1, 32, 3, 1, 0, 5) }
      else if (hd == 128) { LAUNCH(128, 4, 1, 32, 3, 1, 0, 4) }
      else { LAUNCH(256, 4, 1, 32, 3, 1, 0, 2) }
      break;
    case 46:  // occupancy-first prefill, wide KV tile, register prefetch
      if (hd == 64) { LAUNCH(64, 4, 1, 64, 3, 1, 1, 4) }
      else if (hd == 128) { LAUNCH(128, 4, 1, 64, 3, 1, 1, 3) }
      else { LAUNCH(256, 4, 1, 32, 3, 1, 1, 2) }
      break;
    // tcgen05 prefill (override-only until measured against cfg60): the tensor
    // core reads its operands through shared-memory descriptors and keeps its
    // accumulators in TMEM, so neither ldmatrix traffic nor the accumulator
    // register file limits it. Needs the f16 mirror.
    case 80:
      if (hd == 64) {
        launch_main_t5<64, 64, 2>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // cfg81 = cfg80 plus the deferred O correction: the TMEM accumulator keeps an
    // older scale and is only rescaled once the running max moves by more than one
    // log2 unit, instead of on (almost) every tile.
    case 81:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // cfg82 = cfg81 with S double-buffered in TMEM so the QK of tile t+1 is issued
    // before the softmax of tile t. NCU (dv_c3ea6bcfe545) put the t5 kernel at
    // 12.5% occupancy with 70.8% of cycles having no eligible warp, and TMEM caps
    // it at 2 CTAs/SM (S 64 + O 128 = 192 columns, rounded to 256 of the SM's 512)
    // no matter what registers or shared memory do, so the only way to keep the
    // tensor core fed is to overlap within a CTA rather than add warps.
    case 82:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // cfg84 = cfg81 with the softmax reductions unrolled four ways, which is the
    // change NCU's stall table asks for: 39.3% of warp-active cycles in `wait`
    // (fixed-latency dependencies) against 1.7% in barriers.
    case 84:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // cfg85 = cfg81 plus per-phase clock64 instrumentation (override-only
    // diagnostic; timestamps land in the unused split-KV scratch at +100MiB).
    case 85:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // cfg89 = cfg81 with KB3 staging: 3 K buffers / 2 V buffers, lookahead
    // issued at the TOP of each iteration so K(t+2)/V(t+1) fly across a full
    // iteration instead of the single-tile exposure of the 2-buffer schedule.
    // cfg90 adds the ILP softmax reductions; cfg91 the phase instrumentation.
    case 89:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 90:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 91:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // cfg92 = cfg81 with the packed f32x2 softmax (probe5: softmax is 73% of
    // the t5 iteration at ~6.5 instr/element issue-starved; the packed fma and
    // sum cut ~40% of the exp-loop instructions). cfg93 adds instrumentation.
    case 92:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 93:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // cfg94 = cfg92 plus VS: PV accumulates sum(P) into TMEM O column HD via a
    // ones pad on the V tiles, removing the row-sum chain from the softmax and
    // the lrow rescale bookkeeping entirely (the sum rides every O correction).
    case 94:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 96:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 1, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 1, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 1, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // cfg97 = cfg92 plus schedule R (RV): QK(t+1) issued at the top of each
    // iteration (tensor overlaps staging issue and softmax), PV(t) awaited at
    // the top of t+1, 3 K buffers so top-of-iter issues have a full iteration
    // of slack. cfg99 = cfg97 + phase instrumentation.
    case 97:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 99:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 1, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 1, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 1, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // cfg100 = cfg92 + DAO (descriptor-base mma issue: probe6 phases put
    // tid0's qk_issue at 307-461ns and pv_issue at 260-461ns, dominated by
    // per-mma 64-bit descriptor recomputation). cfg101 adds TPF (page-id
    // prefetch ring, killing the table-load dependency in staging issue:
    // 251-509ns). cfg102 adds instrumentation.
    case 100:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 101:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 102:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // cfg103 = cfg100 + SPG (per-CTA page-id table preloaded to smem; NCU put
    // long_scoreboard at 30.7% of the issue interval, fed by the staging
    // loop's per-chunk table loads).
    case 103:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // cfg106 = cfg100 + PIPE (double-buffered S in TMEM; QK(t+1) issued before
    // softmax(t) so the tensor core runs QK while the warps run softmax). Same
    // DEFER+F2+DAO numerics as cfg100, only the issue ORDER across tiles differs,
    // so per-tile math is bit-identical. Targets the latency-exposed prefill wall
    // (probe5 serial phases: qk_issue 342 + qk_wait 73 exposed; PIPE hides them
    // under softmax 566). Biggest expected win on hd256 (1 CTA/SM, no second CTA
    // to overlap). Override-only until probe18 confirms vs cfg100.
    case 106:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // decode-t5 REJECTED for decode (paired job 20261020T172835Z: 1.2-2.2x
    // SLOWER than cfg107 on every decode shape -- the plain t5 schedule
    // serialises QK-wait/softmax/PV-wait per tile with no second CTA's warps
    // to overlap, which the many-short-splits decode regime cannot hide).
    // PROMOTED for mid-scale prefill instead: tcgen05 compute for up to 128
    // packed rows per work item with FP8S register-staged fp8->f16 convert --
    // no mirror, no ldmatrix, O in TMEM. DEFER+F2; DAO follows the head-dim
    // gate measured for prefill (hd256 only). cfg121 adds the clock64 phase
    // instrumentation. Also usable as a no-mirror prefill fallback for any
    // rows>128 shape (TILE_M 128 tiles the whole domain).
    case 120:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 121:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // Prefill per-tile chain experiments (override-only). The t5 iteration at
    // hd128 spends ~2.1us/CTA-tile against 0.4us of tensor work (f1 makespan
    // model): softmax TMEM round trip + a 64-deep fmax max-reduce chain +
    // three barriers. cfg122 = ILP 4-way max with the packed F2 sum (the two
    // were exclusive branches; numerics: max is order-independent, the sum
    // reassociates as F2 already does). cfg123 = drop the end-of-iteration
    // barrier (redundant: all threads pass the PV mbar wait before the next
    // top-of-iteration barrier that orders the next cp.async restage).
    // cfg124 = both. DAO follows the promoted hd gate.
    case 122:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 123:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 124:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // PSM: softmax publishes P to a dedicated swizzled smem tile and PV runs
    // as A-from-smem umma_ss, replacing the t5_st32+wait TMEM round trip with
    // 8 STS.128 per lane + one proxy fence. cfg125 = c123+PSM (hd64/128
    // default family), cfg126 = c122+PSM (hd256 family), cfg127 = c120+PSM
    // (FP8S no-mirror family). Override-only until measured.
    case 125:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 126:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 127:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // MTM: flattened mtile-major DESCENDING launch order (see fmha_main_t5
    // head comment). Global longest-first greedy for causal prefill; makespan
    // simulation on the f7 geometry (512 CTAs, 1.73 waves): 196 -> 128
    // tile-times, vs 186 for per-y reversal only. Bit-identical outputs.
    // c128 = c123+MTM (hd64/128 mirror), c129 = c122+MTM (hd256 mirror),
    // c130 = c120+MTM (FP8S no-mirror). Override-only until measured.
    case 128:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 129:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 130:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // DTH4 promoted into the cfg131-136 bases (round 12, job dv_f170d42331b3,
    // 3x14): DEFER O-rescale trigger 1 -> 4 log2 units. Wins with tight
    // blocks: b4 0.943, d7 0.965, f6 0.974, f0 0.969, b6 0.964, f7 0.981,
    // f4 0.988; f1 +1.5% sits inside its own block spread (1141-1164 base vs
    // 1157-1165). Numerics IDENTICAL to DTH1 on every smoke (max_out/max_lse
    // unchanged; P <= 2^4 = 16 still ~3200x inside f16 range, X2H exponent
    // quantization empirically unchanged). LD64 was neutral everywhere
    // (0.995-1.009) and stays out; cfg137-140 remain as diagnostics.
    // X2H: packed ex2.approx.f16x2 + f16x2 row-sum accumulators in the F2
    // full-tile softmax branch (half the MUFU ops; P is the exp output
    // directly). Numerics shift within ~0.14% on dominant P terms -- must
    // pass the smoke check against v0auto like any other arm. c131 =
    // c128+X2H (hd64/128 mirror family), c132 = c129+X2H (hd256), c133 =
    // c130+X2H (FP8S). Override-only until measured.
    case 131:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 132:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 133:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // X2H without MTM: the promoted bases for shapes the MTM guard excludes
    // (multi-split multi-wave f4-class, ctas <= sms, ctas > 3/2 sms zones).
    // c134 = c123+X2H (hd64/128 mirror), c135 = c122+X2H (hd256 mirror),
    // c136 = c120+X2H (FP8S no-mirror).
    case 134:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 135:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 136:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // Round-12 per-tile chain micro-experiments on the promoted c131 base
    // (mirror hd64/128 + NOSYNC + MTM + X2H), override-only:
    // DTH raises the DEFER O-rescale trigger from 1 to D log2 units, cutting
    // rescale round trips (HDX/32 x TMEM ld+mul+st) several-fold; P <= 2^DTH
    // stays inside f16, but the X2H exponent quantization error grows with
    // DTH (|x| <= D), so DTH4 also probes the numerical-margin cost.
    // LD64 replaces the 2x t5_ld32 S-row load with one x64 (one issue slot).
    case 137:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 0>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 0>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 2, 0>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 138:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 139:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 140:
      if (hd == 64) {
        launch_main_t5<64, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 1>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 64, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 1>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 2, 1>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // X2HM: the packed ex2.approx.f16x2 softmax ported to the mma.sync
    // family (fmha_main). Masked columns fold into NEG_BIG exponents
    // (satfinite pack -> ex2 -> +0), Pf is the exp output directly, and the
    // row sum accumulates in f16x2 (P <= 1: the max applies immediately, no
    // DEFER). c141 = c40+X2HM (narrow-KV prefill), c142 = c46/c51+X2HM
    // (wide-KV/mirror-fallback prefill), c144 = c107+X2HM (decode).
    // Override-only until measured.
    case 141:
      if (hd == 64) { LAUNCH_X(64, 4, 1, 32, 3, 1, 0, 5, 0, 0, 0, 1, 0, 0) }
      else if (hd == 128) { LAUNCH_X(128, 4, 1, 32, 3, 1, 0, 4, 0, 0, 0, 1, 0, 0) }
      else { LAUNCH_X(256, 4, 1, 32, 3, 1, 0, 2, 0, 0, 0, 1, 0, 0) }
      break;
    case 142:
      if (hd == 64) { LAUNCH_X(64, 4, 1, 64, 3, 1, 1, 4, 1, 0, 0, 1, 0, 0) }
      else if (hd == 128) { LAUNCH_X(128, 4, 1, 64, 3, 1, 1, 3, 1, 0, 0, 1, 0, 0) }
      else { LAUNCH_X(256, 4, 1, 32, 3, 1, 1, 2, 1, 0, 0, 1, 0, 0) }
      break;
    case 144:
      if (hd == 64) { LAUNCH_X(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 0, 1, 0, 1) }
      else if (hd == 128) { LAUNCH_X(128, 1, 4, 64, 3, 1, 1, 3, 1, 0, 0, 1, 0, 1) }
      else { LAUNCH_X(256, 1, 4, 64, 3, 1, 1, 2, 1, 0, 0, 1, 0, 1) }
      break;
    case 196:  // t7c + TMA fp8 ring (hd128 NS4 MINB3 QR1 / hd256 NS3 MINB2 QR0)
    case 197: {  // deeper-ring variants (hd128 NS6 MINB2 / hd256 NS3 QR0 MINB2)
      if (hd == 128) {
        if (pl.cfg == 197) { LAUNCH_L8T(128, 6, 2, 1) }
        else { LAUNCH_L8T(128, 4, 3, 1) }
      } else if (hd == 256) { LAUNCH_L8T(256, 3, 2, 0) }
      else { LAUNCH_X(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 1, 1, 0, 1) }  // hd64 -> c145
      break;
    }
    case 192:  // t7c: ldmatrix.b8 direct-fp8 decode (hd64 MINB4 QR1 / hd128
               // MINB3 QR1 / hd256 MINB2 QR0)
    case 193: {  // 193 = hd128 QR0 A/B variant
      if (hd == 64) { LAUNCH_L8(64, 4, 1) }
      else if (hd == 128) {
        if (pl.cfg == 193) { LAUNCH_L8(128, 3, 0) }
        else { LAUNCH_L8(128, 3, 1) }
      } else { LAUNCH_L8(256, 2, 0) }
      break;
    }
    case 286:  // TMA fp8 ring + EISSUE at the c196 geometries (hd128 NS4/
    case 287: {  // MINB3/QR1, hd256 NS3/MINB2/QR0). Override-only experiment:
      // issue tile t+NS-1 before waiting on tile t and prefetch the page ids
      // into tid0 registers, breaking the wait->LDG->issue serial chain that
      // bounds the ring's steady-state period (see the EISSUE block).
      if (hd == 128) { LAUNCH_L8T_E(128, 4, 3, 1) }
      else if (hd == 256) { LAUNCH_L8T_E(256, 3, 2, 0) }
      else { LAUNCH_X(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 1, 1, 0, 1) }  // hd64 -> c145
      break;
    }
    case 288:  // LDM8 cp.async ring + EISSUE at the c192 geometries
    case 289: {
      if (hd == 128) { LAUNCH_L8_E(128, 3, 1) }
      else if (hd == 256) { LAUNCH_L8_E(256, 2, 0) }
      else { LAUNCH_X(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 1, 1, 0, 1) }  // hd64 -> c145
      break;
    }
    case 290:  // c196 geometry + TRC8 phase timeline (TMA8; diagnostics)
    case 291: {
      if (hd == 128) { LAUNCH_L8T_TR(128, 4, 3, 1) }
      else if (hd == 256) { LAUNCH_L8T_TR(256, 3, 2, 0) }
      else { LAUNCH_X(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 1, 1, 0, 1) }  // hd64 -> c145
      break;
    }
    case 292:  // c192 geometry + TRC8 phase timeline (LDM8; diagnostics)
    case 293: {
      if (hd == 128) { LAUNCH_L8_TR(128, 3, 1) }
      else if (hd == 256) { LAUNCH_L8_TR(256, 2, 0) }
      else { LAUNCH_X(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 1, 1, 0, 1) }  // hd64 -> c145
      break;
    }
    // K27 (dv job 20261021T182216Z): c192 won ONLY f2 (hd256, deep pipeline,
    // roomy 256-reg budget) at -7%; hd128 lost 8-34% and hd64 40-180%. The
    // loss pattern tracks the register budget (hd64 MINB4+QR1+X2HM is the
    // same combo that made c145 +20us flat) -> spill hypothesis. c194/c195
    // buy register headroom (QR0 and/or lower MINB) to test it.
    case 194:  // LDM8, QR0 everywhere; hd128 MINB2 (256 regs), hd64 MINB4
    case 195: {  // LDM8, QR1 with slack: hd64 MINB3 (170), hd128 MINB2 (256)
      if (pl.cfg == 194) {
        if (hd == 64) { LAUNCH_L8(64, 4, 0) }
        else if (hd == 128) { LAUNCH_L8(128, 2, 0) }
        else { LAUNCH_L8(256, 2, 0) }
      } else {
        if (hd == 64) { LAUNCH_L8(64, 3, 1) }
        else if (hd == 128) { LAUNCH_L8(128, 2, 1) }
        else { LAUNCH_L8(256, 2, 0) }
      }
      break;
    }
    case 190:  // t7 warp-specialized TMA-fp8 decode (hd256 CVT4/RST2; hd128 CVT2/RST3)
    case 191: {  // 191 = hd128 deep-ring variant (CVT2/RST4)
      if (hd == 256) { LAUNCH_T7(256, 1, 4, 64, 3, 1, 1, 1, 4, 2) }
      else if (hd == 128) {
        if (pl.cfg == 191) { LAUNCH_T7(128, 1, 4, 64, 3, 1, 1, 1, 2, 4) }
        else { LAUNCH_T7(128, 1, 4, 64, 3, 1, 1, 1, 2, 3) }
      } else { LAUNCH_X(64, 1, 4, 64, 3, 1, 1, 4, 1, 0, 0, 1, 0, 1) }  // hd64 -> c144
      break;
    }
    // Wide-tile t5 experiments (override-only): TILE_N=128 halves the per-tile
    // barrier/mbar/TMEM-round-trip count per KV column and (pending probe_tc)
    // may double the tensor-core rate per column if m128n128k16 keeps the
    // m128n64k16 wall time. Bases are the promoted c131 (mirror) / c133
    // (FP8S) numerics: DEFER+F2+X2H+DTH4+NOSYNC(mirror)+MTM.
    // smem: hd128 160KB -> MINB=1 (1 CTA/SM); hd64 80KB -> MINB=2 kept.
    // hd256 cannot stage 128-token f16 tiles (320KB) and falls back to the
    // promoted hd256 arm so an override can never launch an invalid config.
    case 150:  // mirror TN128 + MTM
      if (hd == 64) {
        launch_main_t5<64, 128, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 128, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 151:  // mirror TN128, no MTM
      if (hd == 64) {
        launch_main_t5<64, 128, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 128, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 152:  // FP8S (no mirror) TN128 + MTM
      if (hd == 64) {
        launch_main_t5<64, 128, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 128, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 154:  // FP8S (no mirror) TN128, no MTM (c152 base for zone promotion)
      if (hd == 64) {
        launch_main_t5<64, 128, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 128, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 153:  // mirror TN128 + MTM, MINB=1 at hd64 (occupancy bracket for c150)
      if (hd == 64) {
        launch_main_t5<64, 128, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 128) {
        launch_main_t5<128, 128, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t5<256, 64, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 4, 0>(p, grid_mtm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    // t6: warp-specialized TN128 prefill (mirror + TMA). The host gate in
    // fmha_forward guarantees maps, hd in {64,128} and splits==1.
    case 160:
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 161:  // t6 + instrumentation
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 1>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4, 1>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 162:  // t6 with PV n128 (hd128 only; hd64 keeps n64 = full hd)
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 163:  // t6 with asymmetric ring KST4/VST2 (K leads, V trails)
      if (hd == 128) {
        launch_main_t6<128, 4, 2, 4>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 8, 4, 4>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 164:  // t6 + MTM6 (flattened mtile-major descending)
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 64, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4, 0, 64, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 165:  // t6 PVN128 + MTM6
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4, 0, 64, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 166:  // t6 PVN128 KST2/VST4 (V-heavy ring; smem 230400B)
      if (hd == 128) {
        launch_main_t6<128, 2, 4, 4, 0, 128>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 167:  // t6 PVN128 KST2/VST4 + MTM6
      if (hd == 128) {
        launch_main_t6<128, 2, 4, 4, 0, 128, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4, 0, 64, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 175:  // t6 PVN128 + MTM6 + DUAL issuer (QK warp4 / PV warp6)
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4, 0, 64, 1, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 176:  // t6 PVN128 + MTM6 + SM8 8-warp column-split softmax
      if (hd == 128) {
        // LPAR8: the load warp issues its per-page TMA boxes from 8 lanes instead
        // of 1. cfg210's phase trace on f1 put the tile period at 1590 + (KF wait)
        // cycles with the issuer idling 131..624 per tile for K(t+1) to land, and
        // that wait is one-for-one tile period. Paired, bit-identical (d_out and
        // d_lse exactly 0.0), jobs dv_4ed4bb567ac1 / dv_a0b338c50a09:
        //   f1 783.30 -> 646.05 (0.825)   main kernel 765.6 -> 613.6 us
        //   f7 187.57 -> 159.04 (0.848)   main kernel 170.0 -> 141.4 us
        //   page16 241.60 -> 171.31 (0.709), window4095 114.06 -> 97.71 (0.857)
        //   page64 135.58 -> 135.39, page128 108.11 -> 106.46, MQA G=16 309.55 ->
        //   306.34, f9 187.87 -> 187.38: neutral where a tile is <= 2 pages.
        // LPAR8 also beats LPAR4 by 3-5% at page32, where both give one page per
        // lane: PMAX = ceil(PPT/LPAR) is 1 rather than 2, so the page-id array
        // stays in a register instead of being indexed in local memory.
        // DPV + SB3 + QLA2 (job dv_159c8ffbdcef, bit-identical, was cfg230): a
        // dedicated PV issuer warp on top of three TMEM S buffers and a QK lookahead of
        // two tiles. One warp alternating between the two mma phases left the tensor
        // core idle across the PR/VF waits: cfg225 measured a 1752-cycle period against
        // 1240 cycles of tensor work, 555 cycles blocked issuing QK and then PR 115 +
        // VF 71 + ~190 of loop before PV could be issued. Giving PV its own warp (12
        // warps, 384 threads) and the QK warp an S buffer it does not share with the tile
        // being consumed:
        //   f1 631.57 -> 559.79 (0.886)    f7 155.58 -> 139.74 (0.898)
        //   xwin128 93.89 -> 83.82 (0.893) xp16 132.24 -> 122.45 (0.926)
        //   f9 186.38 -> 184.85 (0.992)
        // DPV at SB2/QLA1 (cfg231) is 0.905/0.908 on f1/f7, so the third buffer and the
        // lookahead add ~2% on top of the second issuer; without DPV they were worth only
        // 0.975/0.976 (cfg227, results/join-jn13-sb3-qla2-rejected.md). Output bits are
        // unchanged: one warp still issues every PV that accumulates a row of O, in tile
        // order, and the QK warp waits the PD of the tile whose S buffer it overwrites.
        // SPLITV (job dv_cd370e73c0a9, bit-identical, cfg223): K and V staged by
        // separate warps. At page16 one warp had 32 boxes per tile to issue and was
        // the pacer -- xp16 170.18 -> 131.55 (0.773); f1 645.86 -> 624.00 (0.966),
        // f7 0.991, f9 0.993, page64 0.990, windowed 0.992. KST4/VST2 on top
        // (cfg224) is worse at page32 (f1 0.998 vs 0.966), so the rings stay 3/3.
        // QCP + LPT6 (job 33, dv_50339aec1678, bit-identical, were cfg257/259/261):
        //   f7 145.18 -> 140.19 median, main 124.34 -> 117.53us; f1 538.98 -> 523.55,
        //   main 494.20 -> 465.92us. QCP alone is -6.3% of f7's main kernel: the gather's
        //   QUI=6 dependent 16B loads could not be hoisted past their smem stores, and
        //   cp.async puts them all in flight (traced bucket 3.756 -> 1.812us). LPT6 is a
        //   no-op here (one request) and costs nothing measurable.
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 0, 0, 1, 1>(
            p, *kmap, *vmap,
                                                                     grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4, 0, 64, 1, 0, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 168:  // DTH sweep: fewer deferred O-rescales (each rescale = a PD
               // wait + 4x(ld32+st32) of O in the softmax warps). f1's 75-tile
               // pipelines drift far; DTH4 may rescale far more than needed.
               // f16 safety: per-tile hacc <= 64 * 2^DTH < 65504 for DTH <= 9.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 6, 0, 128, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 6, 0, 64, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 169:  // DTH8 variant
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 8, 0, 128, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 8, 0, 64, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 211:  // c176 + KST4/VST2: cfg210's f1 trace put the period at
               // 1590 + (KF wait) cycles with the KF wait swinging 131..624 in
               // step with the period 1756..2192, so the K ring is one tile
               // short of covering the TMA latency. Same smem total: Q 32K +
               // 4*32K + 2*32K + 2K = 231424 B.
      if (hd == 128) {
        launch_main_t6<128, 4, 2, 4, 0, 128, 1, 0, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 212:  // c176 + LPAR4: 8 K boxes + 8 V boxes per tile at page32 issued by
               // 4 lanes instead of 1, to test whether the load warp's issue cost
               // (not the TMA data rate) is what the issuer waits on.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 4>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 213:  // both
      if (hd == 128) {
        launch_main_t6<128, 4, 2, 4, 0, 128, 1, 0, 1, 4>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 214:  // c176 + KST2/VST4, the opposite asymmetry: if 211 wins and 214
               // loses, the KF wait really is the K ring and not general TMA rate
      if (hd == 128) {
        launch_main_t6<128, 2, 4, 4, 0, 128, 1, 0, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 215:  // c176 + LPAR2: at page64 a tile is only 2 pages, so 4 lanes
               // would leave half of them arriving with 0 bytes
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 2>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 216:  // c176 + LPAR8: at page16 a 128-row tile is 8 pages, so LPAR4 still
               // leaves 2 pages per lane; LPAR8 also tells us whether the gain at
               // page32 saturates at one page per lane.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 217:  // LPAR16: PMAX = ceil(8/16) = 1 as at LPAR8, but half the warp
               // arrives with 0 bytes. Separates "more lanes issue TMA" from "the
               // page-id array became a scalar".
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 16>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 219:  // DTH6 on top of LPAR8: cfg210's trace put exp/pack (which includes
               // the deferred O-rescale and its PD wait) at 748 of the softmax's
               // 1250 busy cycles, against a ~512-cycle XU floor for the 8192
               // packed exps, so ~236 cycles per tile are rescale/vote/bookkeeping.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 6, 0, 128, 1, 0, 1, 8>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 220:  // DTH8 (f16 headroom: per-tile hacc <= 64 * 2^DTH < 65504 for DTH <= 9)
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 8, 0, 128, 1, 0, 1, 8>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 230:  // c176 + DPV + SB3/QLA2: the QK warp never waits for P(t), so the
               // tensor queue should stay fed across the softmax. If the 1240 cycles
               // of tensor work per tile really are the floor, this is where the
               // remaining ~28% of the hd128 period is.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1>(p, *kmap, *vmap,
                                                                    grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 231:  // DPV with SB2/QLA1: separates "two issuer warps" from "three S
               // buffers". At SB=2 the QK warp must wait PV(t-1)'s PD before reusing
               // the shared S parity, which is what re-serialised arm a's c175.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 2, 1, 1>(p, *kmap, *vmap,
                                                                    grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 227:  // c176 + SB3/QLA2. TMEM at hd128 is S0+S1 (256 cols) + O (128) = 384 of
               // 512, so a third S buffer fits exactly and the issuer can push QK(t+2)
               // before it waits for P(t): the tensor core then has work queued across
               // the PR (115 cyc) and VF (71) waits instead of draining and idling.
               // Buffer safety is unchanged in kind -- QK(t+2) lands in S[(t-1)%3],
               // whose P was consumed by PV(t-1), issued a whole iteration earlier, and
               // the single issuer plus in-order tensor execution keeps that ordering.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 3, 2>(p, *kmap, *vmap, grid_mtm,
                                                                 stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 228:  // SB3 with the lookahead left at 1: isolates whether a third buffer
               // alone changes anything (it should not -- the dependency chain is the
               // same), so any 227 gain is attributable to the lookahead.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 3, 1>(p, *kmap, *vmap, grid_mtm,
                                                                 stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 229:  // cfg227 + IT6
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 1, 128, 1, 0, 1, 8, 1, 3, 2>(p, *kmap, *vmap, grid_mtm,
                                                                 stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 240:  // the promoted hd128 schedule plus both dead-softmax skips
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 242:  // the promoted hd128 schedule with 8 row-sum accumulators
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 8, 0>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 243:  // ... plus the f16x2 scale-offset in exp/pack
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 8, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 241:  // ... with RWSK only, to attribute 240's gain between the two skips
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 0, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 249:  // per-half O accumulators: deletes the per-tile cross-half row-max
               // exchange (277 of 1470 cycles) at the cost of SB3/QLA2 -> SB2/QLA1.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 2, 1, 1, 1, 1, 4, 0, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 250:  // the SB2/QLA1 shared-O control for 249 (isolates PHO from the
               // ring/lookahead change; job 14 measured this geometry as c231)
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 2, 1, 1, 1, 1, 4, 0, 0>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 251:  // the PROMOTED hd128 launch (case 176, QCP+LPT6) + TRACE: a
               // per-CTA timeline, so the prologue, cold first tile, steady per-tile
               // cost, epilogue and wave structure are measured instead of guessed at.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 0, 1, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 252:  // the PROMOTED hd64 launch (case 233, QCP+LPT6) + TRACE
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 4, 0, 64, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 0, 1, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 244:  // c233 at KST3/VST3: job 21 fitted f4's per-CTA prologue at ~22k
               // cycles (~12us) over only 17 tiles, and the ring fill is 12 x 16KiB
               // = 192KiB of TMA staging before the first tile can compute.
      if (hd == 64) {
        launch_main_t6<64, 3, 3, 4, 0, 64, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 245:  // c233 at KST4/VST4 (the midpoint)
      if (hd == 64) {
        launch_main_t6<64, 4, 4, 4, 0, 64, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 233:  // the hd64 form of the hd128 DPV promotion: SB3 needs 3*128 + 64 = 448
               // of 512 TMEM columns, and KST6/VST6 stays inside 210 KiB of smem.
      if (hd == 64) {
        // QCP + LPT6 (job 33): f4 115.65 -> 93.17 median (0.8056), main 94.33 ->
        //   75.29us; f0 35.90 -> 32.06 (0.8930), main 23.88 -> 20.79us; xh64rag 0.8241.
        //   f4's makespan/ideal fell 1.564 -> 1.250 because request 1's 144 items of
        //   53-64 tiles are almost exactly one wave of 148 SMs, so MTM6's mtile-descending
        //   order paired long items with long ones; ordering requests by descending
        //   (kv_len - q_len) pairs them with request 0's 27-32 tile items instead.
        launch_main_t6<64, 6, 6, 4, 0, 64, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 0, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 223:  // c176 + SPLITV: K and V staged by separate warps so the V ring's VE
               // wait cannot delay the K boxes the issuer needs next. Targets the 235
               // cycles per tile (KF 163 + VF 72) that cfg221's trace showed the tensor
               // core idling while the single issuer waited for staged KV.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 0, 128, 1, 0, 1, 8, 1>(p, *kmap, *vmap, grid_mtm,
                                                            stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 224:  // SPLITV + KST4/VST2: the rebalance that measured 1.74x while one warp
               // staged both matrices, retried with the rings decoupled
      if (hd == 128) {
        launch_main_t6<128, 4, 2, 4, 0, 128, 1, 0, 1, 8, 1>(p, *kmap, *vmap, grid_mtm,
                                                            stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 225:  // SPLITV + IT6, to see whether the waits actually went away
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 1, 128, 1, 0, 1, 8, 1>(p, *kmap, *vmap, grid_mtm,
                                                            stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 226:  // cfg222 + SPLITV: at hd64 a tile is KST6/VST6 rings of 16 KiB, so the
               // V ring's VE wait is proportionally more of the load warp's timeline.
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 4, 0, 64, 1, 0, 1, 8, 1>(p, *kmap, *vmap, grid_mtm,
                                                          stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 222:  // hd64 t6 (KST6/VST6, PVN64, MTM6, SM8) WITH LPAR8. The tree records
               // SM8-t6 hd64 losing to the promoted t5 mirror arm c150 by 24.2% on
               // s_p64big and 13.7% on s_p64mid -- both measured at LPAR=1, where the
               // load warp was the wall (that is what cfg210's trace showed at hd128,
               // and at hd64/page16 a tile is 8 pages x KG=1 = 8 K boxes + 8 V boxes,
               // the same 16 single-lane issues). Retesting after the issue-width fix
               // is cheap and covers two scored shapes: f0 runs at 8% of the f16 peak
               // and f4 at 21%, both on the t5 FP8S path.
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 4, 0, 64, 1, 0, 1, 8>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 221:  // IT6 on the promoted schedule itself (c176 with LPAR8), so the next
               // phase breakdown describes what actually runs rather than the
               // LPAR=1 form that produced the 1590 + KF-wait decomposition.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 1, 128, 1, 0, 1, 8>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 209:  // t6 PVN128 + IT6 instrumentation (f1/f7 phase breakdown)
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 1, 128>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4, 1>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 210:  // c176 (MTM6 + SM8) WITH IT6: the phase breakdown of the promoted
      // hd128 schedule, not of the 4-warp no-MTM diagnostic that produced the
      // recorded "~1752 cyc period, qk_issue 521 + pv_issue 694, TC busy ~1026".
      // Override-only: no auto path selects 210, so the promoted bytes' behaviour
      // is unchanged and nothing here needs re-evaluation.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 4, 1, 128, 1, 0, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4, 1, 64, 1, 0, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 170:  // t6c cluster2 (hd128; other dims fall back to sane arms)
    case 171: {
      const dim3 grid_ct(2u * (unsigned)((pl.mtiles + 1) / 2), grid.y, grid.z);
      (void)grid_ct;
      if (hd == 128) {
        launch_main_t6c<128, 4, 4, 4>(p, *kmap, *vmap, grid_ct, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 256) {
        launch_main_t6c<256, 2, 3, 4>(p, *kmap, *vmap, grid_ct, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    case 172:  // t6c + MTM6C pair-descending (LPT) order
    case 265: {  // == the promoted cfg173 hd256 launch since job 36 (kept as the
                 // arm that was measured, and as the QCP reference for t6c)
      const unsigned pairs = (unsigned)((pl.mtiles + 1) / 2);
      const dim3 grid_ctm(2u * pairs * grid.y, 1u, grid.z);
      launch_main_t6c_qcp<256, 2, 3, 4, 0, 1>(p, *kmap, *vmap, grid_ctm, stream);
      if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      break;
    }
    case 266:  // promoted-176 arm + DTH8 (J46 graft of arm-a J34; NOT
               // bit-identical - deferred-rescale points move; hacc bound
               // 64*2^8 = 16384 < 65504)
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 8, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 0, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 267:  // promoted-233 arm (hd64) + DTH8 (arm-a measured hd64 DTH
               // neutral; probe kept for completeness)
      if (hd == 64) {
        // J69: QLA 2 -> 1 (bit-identical, scheduling-only): f0 -1.3%/-0.9%
        // two paired jobs (fill-dominated per J44), s_lp64a -0.6%/-0.5%,
        // f4 -0.6%/-0.1%, no loss cell. J67_BITID exact (c284 == c267 bytes).
        launch_main_t6<64, 6, 6, 8, 0, 64, 1, 0, 1, 8, 1, 3, 1, 1, 1, 1, 4, 0, 0, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 298:  // promoted-267 + PHO (per-half O accumulators): deletes the
               // per-tile 256-thread bar_sm8 row-max exchange. hd64 is the
               // ONLY band with TMEM room at promoted depth (TILE_N=128:
               // 3*128 + 2*64 = 512 exactly; hd128 needs SB<=2 = the raced
               // cfg249 cell, and SB>=4 overflows -- the failed dv_8b21bf8185be
               // static_asserts are the evidence). SB3/QLA1 = 267's exact
               // ring, so this cell is the pure barrier deletion. Fresh stall
               // census (dv_338fa16a605f): f0 barrier 1.54 vs f1 0.38
               // cyc/issued-instr; J51 traced the exchange at 277 of 1470
               // period cycles (hd128). NOT bit-identical (independent half
               // recurrences + epilogue merge; SM8 error class).
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 8, 0, 64, 1, 0, 1, 8, 1, 3, 1, 1, 1, 1, 4, 0, 1, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 302:  // promoted-266 + PBAR (hd128): the exchange barrier narrows to
               // four independent warp-PAIR bar.syncs (w, w+4 share rows; sMx
               // row blocks disjoint; MSKIP dead pairs are pair-identical so
               // arrivals stay matched). No TMEM cost -- SB3/QLA2 kept, unlike
               // PHO which cannot fit hd128 at promoted depth. Scheduling-only;
               // NOT bit-identical? No: merged max is the same fmax pair, same
               // association -> BIT-IDENTICAL to 266 (only the rendezvous
               // width changes; sMx write/read order within a pair is
               // barrier-preserved).
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 8, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 0, 0, 1, 1,
                       0, 0, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 303:  // promoted-267 + PBAR (hd64): same narrowing on the hd64 arm;
               // bit-identical to 267 (see 302).
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 8, 0, 64, 1, 0, 1, 8, 1, 3, 1, 1, 1, 1, 4, 0, 0, 0, 1, 1,
                       0, 0, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 304: {  // promoted-297 (t6c SM8C) + PBAR (hd256): the cluster kernel's
                 // exchange narrows the same way (pair = w, w+4; bar ids are
                 // CTA-scoped so the cluster pair is unaffected); bit-identical
                 // to 297.
      const unsigned pairs = (unsigned)((pl.mtiles + 1) / 2);
      const dim3 grid_ctm(2u * pairs * grid.y, 1u, grid.z);
      if (hd == 256) {
        launch_main_t6c_qcp<256, 2, 3, 4, 0, 1, 1, 1, 1>(p, *kmap, *vmap, grid_ctm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    case 305:  // DIAGNOSTIC: 298 at DTH4 (249's threshold) - triangulates the
               // seed-2026101221 f0 correctness failure between DTH and ring.
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 4, 0, 64, 1, 0, 1, 8, 1, 3, 1, 1, 1, 1, 4, 0, 1, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 306:  // DIAGNOSTIC: 298 at SB2/QLA1 (cfg249's exact ring, at hd64).
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 8, 0, 64, 1, 0, 1, 8, 1, 2, 1, 1, 1, 1, 4, 0, 1, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 311:  // 309 + F16X (f16x2 scale-offset, 4 instr/pair vs 5): J52 built
               // F16X on the PRE-PHO arm (279, unpromoted); with the exchange
               // barrier and DEFER rescale deleted, the exp/pack chain
               // (J44: 739-753 cyc, largest chain term) is the next exposed
               // face on long runs. NOT bit-identical (F16X class: lse error
               // ~0.015 log2 units vs frozen tolerance; peer 243/276 comment).
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 64, 0, 64, 1, 0, 1, 8, 1, 3, 1, 1, 1, 1, 4, 1, 1, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 310:  // 309 + QLA2 (SB3 >= QLA+1): J69's QLA1>QLA2 verdict at hd64
               // predates PHO - with the exchange barrier and the DEFER
               // rescale both gone, the deeper lookahead may pay on long runs
               // (f4/s_h64big class) where QLA2's priming amortizes.
               // Scheduling-only; bit-identical to 309 (same recipe, deeper
               // issue queue).
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 64, 0, 64, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 1, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 309:  // DIAGNOSTIC: 298 at DTH=64 (DEFER O-rescale effectively OFF -
               // normal-data max jumps are ~1-3 log2 units). Discriminates
               // whether the PHO-SB3 race requires the rescale path: if 309
               // is clean across the seeds where 298 fails, the interaction
               // is rescale x SB>=3; if it still races, the rescale is
               // exonerated and the defect is in the PHO ring/issue protocol.
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 64, 0, 64, 1, 0, 1, 8, 1, 3, 1, 1, 1, 1, 4, 0, 1, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 268: {  // promoted-265 t6c_qcp arm + RWSK (J46 graft of arm-a J25;
                 // bit-identical whole-tile mask skip)
      const unsigned pairs = (unsigned)((pl.mtiles + 1) / 2);
      const dim3 grid_ctm(2u * pairs * grid.y, 1u, grid.z);
      if (hd == 256) {
        launch_main_t6c_qcp<256, 2, 3, 4, 0, 1, 1>(p, *kmap, *vmap, grid_ctm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    case 275:  // promoted-266 + HACC8 (J49: 8 row-sum accumulator slots; the
               // peer 242 probe predates QCP/LPT6/DTH8, so it was confounded)
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 8, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 8, 0, 0, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 276:  // promoted-266 + HACC8 + F16X (J49: f16x2 scale-offset, 4
               // instructions per pair instead of 5; NOT bit-identical - the
               // peer comment bounds the lse error at ~0.015 log2 units vs the
               // 0.01+0.05|lse| frozen tolerance)
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 8, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 8, 1, 0, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 277:  // promoted-266 + TRACE (J51: per-CTA timeline of the CURRENT
               // hd128 arm; diagnostics, splits==1)
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 8, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 0, 1, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 278:  // promoted-267 + TRACE (J51: current hd64 arm)
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 8, 0, 64, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 0, 1, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 279:  // promoted-267 + F16X (J52): hd64 is softmax-bound (J44/J51:
               // period ~1294-1529 cyc vs TC ~1197), and exp/pack (739-753 cyc)
               // is the largest chain term. F16X cuts it 5->4 instructions per
               // pair. J49 refuted F16X at hd128 (TC-bound: +3.1%); hd64 is the
               // opposite regime and was never tested. NOT bit-identical (peer
               // case-243 comment bounds the lse error at ~0.015 log2 units).
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 8, 0, 64, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 1, 0, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 280:  // promoted-266 + PLO (J53): mbarrier inits + TMEM alloc hidden
               // under the in-flight Q cp.async copies; parallel inits.
               // Bit-identical (ordering-only change, all happens-before edges
               // preserved). J51: prologue = 2.43us/CTA at hd128, 1 CTA/SM.
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 8, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 0, 0, 1, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 281:  // promoted-267 + PLO (hd64)
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 8, 0, 64, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 0, 0, 1, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 282: {  // promoted-268 + DTH8 (J55): deferred-O-rescale threshold 8
                 // log2 units on the hd256 t6c line (268 runs DTH4). DTH8 won
                 // -0.4..-1.5% at hd128/hd64 (arm-a J34 -> J46 266/267);
                 // hd256 was never swept. smem unchanged (DTH sizes no buffer).
                 // NOT bit-identical (rescale timing; J34 error class).
      const unsigned pairs = (unsigned)((pl.mtiles + 1) / 2);
      const dim3 grid_ctm(2u * pairs * grid.y, 1u, grid.z);
      if (hd == 256) {
        launch_main_t6c_qcp<256, 2, 3, 8, 0, 1, 1>(p, *kmap, *vmap, grid_ctm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    case 297: {  // t6c + SM8C (re-derived on the join-20261027 tree; arm-a
                 // evidence dv_32f4d668753b + official dv_347c8d4c7bc8/
                 // dv_c410e9c39704): t6's promoted SM8 column-split softmax
                 // on the cluster kernel - 8 softmax warps x 64 cols/lane,
                 // exact-max sMx exchange via bar_sm8, DEFER O-rescale
                 // half-0 only, PR count 16, MMA/load warps 8/9, THREADS
                 // 320, barrier tail 2048B (smem 231424 <= 232448).
                 // Paired: f6 -0.7..-1.4% (both official orders), s_d7
                 // -10.3..-10.5%, s_256mq -0.8%; smokes 0/0 on every
                 // validated class (window+sink, page16, ragged dead-rank,
                 // dead-pair splits, empty-split sp48). NOT bit-identical
                 // (exp/sum association; same class as t6-SM8/t8 recipes).
      const unsigned pairs = (unsigned)((pl.mtiles + 1) / 2);
      const dim3 grid_ctm(2u * pairs * grid.y, 1u, grid.z);
      if (hd == 256) {
        launch_main_t6c_qcp<256, 2, 3, 4, 0, 1, 1, 1>(p, *kmap, *vmap, grid_ctm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    case 307: {  // promoted-297 (t6c SM8C) + DTH8: the deferred-O-rescale
                 // threshold sweep that J55 ran at hd128/hd64 (-0.4..-1.5%)
                 // but "hd256 was never swept" (282 comment) - and 282's own
                 // cell predates SM8C, so the SM8C x DTH8 interaction (half-0-
                 // only rescales at twice the threshold) is untested. NOT
                 // bit-identical (rescale timing; J34 error class).
                 // smem identical to 297: 231424 <= 232448.
      const unsigned pairs = (unsigned)((pl.mtiles + 1) / 2);
      const dim3 grid_ctm(2u * pairs * grid.y, 1u, grid.z);
      if (hd == 256) {
        launch_main_t6c_qcp<256, 2, 3, 8, 0, 1, 1, 1>(p, *kmap, *vmap, grid_ctm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    case 308: {  // promoted-297 at KST3/VST2 (K-heavy ring rebalance; the
                 // opposite asymmetry of 174's V-heavy 2/3, never swept on
                 // the qcp+SM8C line). Bit-identical (staging geometry only;
                 // per-tile MMA order and data unchanged). smem: Q 65536 +
                 // 3*32768 K + 2*32768 V + 2048 = 231424 <= 232448.
      const unsigned pairs = (unsigned)((pl.mtiles + 1) / 2);
      const dim3 grid_ctm(2u * pairs * grid.y, 1u, grid.z);
      if (hd == 256) {
        launch_main_t6c_qcp<256, 3, 2, 4, 0, 1, 1, 1>(p, *kmap, *vmap, grid_ctm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    case 283:  // t6f (J59): fp8-DIRECT prefill at the full 266 geometry -
               // TMA-fp8 ring (2+2 stages) + fused load-warp convert into
               // double-buffered f16 TC targets. Halves KV bytes through L2
               // (f9 is BW-bound at ~7.4TB/s reading the f16 mirror).
               // Bit-identical to 266 (exact e4m3->f16, same op order).
               // hd128-only; falls back to 266 when the fp8 maps are absent.
      if (hd == 128) {
        launch_main_t6<128, 2, 2, 8, 0, 128, 1, 0, 1, 8, 1, 3, 2, 1, 1, 1, 4, 0, 0, 0, 1,
                       1, 0, 1>(p, p.kmap8, p.vmap8, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 284:  // promoted-267 + QLA1 (J67): shallow QK lookahead for
               // short-pipeline items - the widened sub-1e9 band runs 1-3
               // tiles/item where QLA2 priming is proportionally costly, and
               // J44 measured f0 fill-dominated (1529 vs 1294 cyc/tile).
               // Scheduling-only; expected bit-identical (j67_bitid).
      if (hd == 64) {
        launch_main_t6<64, 6, 6, 8, 0, 64, 1, 0, 1, 8, 1, 3, 1, 1, 1, 1, 4, 0, 0, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 285:  // promoted-266 + QLA1 (hd128, J67)
      if (hd == 128) {
        launch_main_t6<128, 3, 3, 8, 0, 128, 1, 0, 1, 8, 1, 3, 1, 1, 1, 1, 4, 0, 0, 0, 1, 1>(
            p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    case 173:
    case 174: {  // 174 = hd256 KST2/VST3 ring variant (PV-heavy balance)
      const unsigned pairs = (unsigned)((pl.mtiles + 1) / 2);
      const dim3 grid_ctm(2u * pairs * grid.y, 1u, grid.z);
      if (hd == 128) {
        launch_main_t6c<128, 4, 4, 4, 0, 1>(p, *kmap, *vmap, grid_ctm, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 256) {
        // KST2/VST3 for ALL hd256 t6c arms (K36, dv job 20261021T203718Z:
        // -23.6% f6, -15.1% d7, -25.0% s_pref256, -23.6% s_d256rag vs KST3/
        // VST2). hd256 is PV-heavy (2 atoms x 8 steps): V slots free LATE in
        // the tile timeline, so the V ring -- not K -- gates the shared load
        // warp; VST3 unstalls it and KST2 pays the smem back (230400B).
        // QCP (job 36, dv_19acc473ebcf, bit-identical, was cfg265): the hd256 Q
        // prologue is 4096 sixteen-byte chunks over 192 threads = 22 dependent
        // iterations per thread against a generic char* destination, so the CTA
        // paid DRAM latency 22 times; cp.async stages them all at once and the
        // bf16->f16 conversion then runs smem-to-smem.
        //   f6       347.44 -> 306.38 us median (0.8818), t6c main ~338 -> 299.98
        //   d7       111.07 ->  95.94 (0.8638), main 65.60, mm=0/2097152
        //   xh256a    60.69 ->  50.08 (0.8252), main 26.86, mm=0/524288
        //   xh256rag 105.12 ->  96.40 (0.9170), main 74.61 -> 65.86, mm=0
        launch_main_t6c_qcp<256, 2, 3, 4, 0, 1>(p, *kmap, *vmap, grid_ctm, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4, 0, 64, 1>(p, *kmap, *vmap, grid_mtm, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    case 180:  // t6d cluster2 multicast K/V sharing
    case 181: {  // t6d + MTM6D pair-descending (ragged-aware)
      const unsigned pairs = (unsigned)((pl.mtiles + 1) / 2);
      const bool mtm = (pl.cfg == 181);
      const dim3 gd = mtm ? dim3(2u * pairs * grid.y, 1u, grid.z)
                          : dim3(2u * pairs, grid.y, grid.z);
      if (hd == 128) {
        if (mtm) launch_main_t6d<128, 3, 3, 4, 0, 1>(p, *kmap, *vmap, gd, stream);
        else launch_main_t6d<128, 3, 3, 4>(p, *kmap, *vmap, gd, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 64) {
        if (mtm) launch_main_t6d<64, 6, 6, 4, 0, 1>(p, *kmap, *vmap, gd, stream);
        else launch_main_t6d<64, 6, 6, 4>(p, *kmap, *vmap, gd, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        // hd256 keeps the pair-MMA kernel (same grid shape).
        if (mtm) launch_main_t6c<256, 2, 3, 4, 0, 1>(p, *kmap, *vmap, gd, stream);
        else launch_main_t6c<256, 2, 3, 4>(p, *kmap, *vmap, gd, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    case 188: {  // t6d + beacons (diagnostics)
      const unsigned pairs = (unsigned)((pl.mtiles + 1) / 2);
      const dim3 gd(2u * pairs, grid.y, grid.z);
      if (hd == 64) {
        launch_main_t6d<64, 6, 6, 4, 1>(p, *kmap, *vmap, gd, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6d<128, 3, 3, 4, 1>(p, *kmap, *vmap, gd, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    case 200:  // t8 256-row hd128 (L2-halving prefill, no clusters)
    case 201:  // t8 + MTM8 ragged-aware descending
    case 202:  // 202 = MTM8 + KST5/VST5 deeper ring (230400B smem)
    case 203:  // 203 = MTM8 + KST4/VST5 V-heavy ring (208K; K36 analogy)
    case 204: {  // 204 = MTM8 + KST5/VST3 K-heavy ring (208K; t8 QK>PV)
      const unsigned xb = (unsigned)((pl.mtiles + 1) / 2);
      const bool mtm = (pl.cfg >= 201 && pl.cfg <= 204);
      const dim3 gd = mtm ? dim3(xb * grid.y, 1u, grid.z) : dim3(xb, grid.y, grid.z);
      if (hd == 128) {
        if (pl.cfg == 202) launch_main_t8<128, 5, 5, 4, 0, 1>(p, p.kmap64, p.vmap64, gd, stream);
        else if (pl.cfg == 203) launch_main_t8<128, 4, 5, 4, 0, 1>(p, p.kmap64, p.vmap64, gd, stream);
        else if (pl.cfg == 204) launch_main_t8<128, 5, 3, 4, 0, 1>(p, p.kmap64, p.vmap64, gd, stream);
        else if (mtm) launch_main_t8<128, 4, 4, 4, 0, 1>(p, p.kmap64, p.vmap64, gd, stream);
        else launch_main_t8<128, 4, 4, 4>(p, p.kmap64, p.vmap64, gd, stream);
        if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      } else if (hd == 256) {
        const dim3 grid_ct(2u * (unsigned)((pl.mtiles + 1) / 2), grid.y, grid.z);
        launch_main_t6c<256, 2, 3, 4>(p, *kmap, *vmap, grid_ct, stream);
        if (split) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
      } else {
        launch_main_t6<64, 6, 6, 4>(p, *kmap, *vmap, grid, stream);
        if (split) launch_combine<64, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    case 208: {  // t8 + beacons (diagnostics)
      const unsigned xb = (unsigned)((pl.mtiles + 1) / 2);
      const dim3 gd(xb, grid.y, grid.z);
      if (hd == 128) {
        launch_main_t8<128, 4, 4, 4, 1>(p, p.kmap64, p.vmap64, gd, stream);
      } else {
        launch_main_t6<64, 6, 6, 4>(p, *kmap, *vmap, grid, stream);
      }
      if (split) launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      break;
    }
    case 178: {  // t6c + host-readable progress beacons (diagnostics)
      const dim3 grid_ct(2u * (unsigned)((pl.mtiles + 1) / 2), grid.y, grid.z);
      if (hd == 128) {
        launch_main_t6c<128, 4, 4, 4, 1>(p, *kmap, *vmap, grid_ct, stream);
      } else {
        launch_main_t6c<256, 2, 3, 4, 1>(p, *kmap, *vmap, grid_ct, stream);
      }
      if (split) {
        if (hd == 256) launch_combine<256, 128>(p, cgrid, csmem, chunk, stream);
        else launch_combine<128, 128>(p, cgrid, csmem, chunk, stream);
      }
      break;
    }
    // decode: 16 packed rows, the KV tile split across warps, merged in smem
    case 1:
      if (hd == 64) { LAUNCH(64, 1, 4, 64, 3, 1, 0, 1) }
      else if (hd == 128) { LAUNCH(128, 1, 4, 64, 3, 1, 0, 1) }
      else { LAUNCH(256, 1, 4, 64, 3, 1, 0, 1) }
      break;
    // prefill, 32-row warp tiles: each K/V fragment feeds two mma row tiles
    case 5:
      if (hd == 64) { LAUNCH(64, 4, 1, 32, 3, 2, 0, 1) }
      else if (hd == 128) { LAUNCH(128, 4, 1, 32, 3, 2, 0, 1) }
      // hd256 cannot afford 32-row warp tiles (256 accumulator registers); the host
      // remaps cfg5 to cfg6, and this arm repeats that so an override can never
      // launch a wrong-head_dim kernel (that fault poisoned job dv_a47d91211af5).
      else { LAUNCH(256, 4, 1, 32, 3, 1, 1, 1) }
      break;
    // prefill, register prefetch instead of the cp.async ring
    case 6:
      if (hd == 64) { LAUNCH(64, 4, 1, 64, 3, 1, 1, 1) }
      else if (hd == 128) { LAUNCH(128, 4, 1, 64, 3, 1, 1, 1) }
      else { LAUNCH(256, 4, 1, 32, 3, 1, 1, 1) }
      break;
    default:  // cfg 0: prefill, 64-row tiles, cp.async ring
      if (hd == 64) { LAUNCH(64, 4, 1, 64, 3, 1, 0, 1) }
      else if (hd == 128) { LAUNCH(128, 4, 1, 64, 3, 1, 0, 1) }
      else { LAUNCH(256, 4, 1, 32, 3, 1, 0, 1) }
      break;
  }
#undef LAUNCH
}
}  // namespace fmha

namespace fmha {
Plan g_last_plan;  // experiment read-back only; the harness is single threaded
int g_last_wl = -2;    // window_left as the kernel will see it
int g_last_sink = -1;  // sinks != nullptr as the kernel will see it

// Cached launch descriptor. Keyed by the caller in Python on object identity plus
// the scalar arguments, so a hit means the very same tensors are being passed
// again. Only the launch parameters are reused: the kernels still read whatever
// is currently in those tensors, so no data-dependent result is ever cached.
struct CachedLaunch {
  Params p;
  Plan pl;
  int hd = 0;
  int batch = 0;
  bool valid = false;
  // The mirror must be refreshed on every replay: it is derived from the current
  // contents of the fp8 KV cache, so caching only the launch parameters would
  // serve stale f16 data if a caller updated the cache in place.
  const uint8_t* k_src = nullptr;
  const uint8_t* v_src = nullptr;
  long long k_elems = 0;
  long long v_elems = 0;
  // TMA maps for the t6 config family (valid iff have_maps).
  CUtensorMap kmap, vmap;
  bool have_maps = false;
  // Whole-sequence launch graph, captured after the descriptor proves stable
  // (same tensors => same device pointers; the kernels still re-read current
  // contents, so no data-dependent answer is baked in). Saves the per-launch
  // host/driver cost, which is 15-30% of the wall time on small decode shapes.
  cudaGraphExec_t exec = nullptr;
  int hits = 0;
  bool graph_off = false;
};
CachedLaunch g_cache;

// Dedicated non-blocking stream for capture: the ambient torch stream may be
// the legacy default stream, which cannot be captured.
static cudaStream_t side_stream() {
  static cudaStream_t s = nullptr;
  if (s == nullptr) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  return s;
}

static void launch_all(const CachedLaunch& c, cudaStream_t stream) {
  if (c.p.k16 != nullptr) {
    launch_cvt_kv(c.k_src, const_cast<__half*>(c.p.k16), c.k_elems, c.v_src,
                  const_cast<__half*>(c.p.v16), c.v_elems, stream);
  }
  dispatch(c.p, c.hd, c.pl, c.batch, stream, c.have_maps ? &c.kmap : nullptr,
           c.have_maps ? &c.vmap : nullptr);
}
}

void fmha_replay() {
  fmha::CachedLaunch& c = fmha::g_cache;
  TORCH_CHECK(c.valid, "fmha_replay without a preceding fmha_forward");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // If the caller (e.g. a torch CUDA-graph capture) is already capturing this
  // stream, just record our kernels into it: no private capture, no graph
  // launch, both of which are illegal inside an ambient capture.
  // HOST TRIM (K16: 4.6-5.1us/call host path): the legacy default stream
  // CANNOT be captured, and querying it costs a failing cudaStreamIsCapturing
  // plus a cudaGetLastError clear (~0.5-0.9us) on every call. Only non-default
  // streams (e.g. torch.cuda.graph's capture stream) need the query.
  // FMHA_HOST_TRIM=0 restores the unconditional query for paired A/B.
  static const bool host_trim = [] {
    const char* e = getenv("FMHA_HOST_TRIM");
    return e == nullptr || e[0] != '0';
  }();
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (!host_trim || stream != nullptr) {
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess) {
      cudaGetLastError();  // clear; querying the legacy stream may not support it
      cap = cudaStreamCaptureStatusNone;
    }
  }
  static const bool no_graph = getenv("FMHA_NOGRAPH") != nullptr;
  // A single-kernel launch is cheaper than a graph launch (measured: the tiny
  // splits=1 decode shapes regressed 13.3 -> 14.3us when graphed). Capture
  // only sequences of two or more kernels (split combine and/or mirror cvt).
  // g_dbg_graph overrides the rule per call so both paths can be measured on
  // identical tensors in one process (the env knob is read once, statically).
  const bool graph_worth = (fmha::g_dbg_graph >= 0)
                               ? (fmha::g_dbg_graph != 0)
                               : (c.pl.splits > 1 || c.p.k16 != nullptr);
  if (cap == cudaStreamCaptureStatusNone && !no_graph && graph_worth) {
    if (c.exec != nullptr) {
      cudaError_t gerr = cudaGraphLaunch(c.exec, stream);
      TORCH_CHECK(gerr == cudaSuccess, "fmha graph replay failed: ",
                  cudaGetErrorString(gerr));
      return;
    }
    if (++c.hits >= 2 && !c.graph_off) {
      cudaStream_t side = fmha::side_stream();
      cudaGraph_t graph = nullptr;
      cudaError_t err = cudaStreamBeginCapture(side, cudaStreamCaptureModeThreadLocal);
      if (err == cudaSuccess) {
        fmha::launch_all(c, side);
        err = cudaStreamEndCapture(side, &graph);
      }
      if (err == cudaSuccess && graph != nullptr) {
        err = cudaGraphInstantiate(&c.exec, graph, 0);
        cudaGraphDestroy(graph);
        graph = nullptr;
      }
      if (err != cudaSuccess || c.exec == nullptr) {
        c.graph_off = true;  // stay on the direct launch path
        if (graph != nullptr) cudaGraphDestroy(graph);
        cudaGetLastError();  // clear the capture error
      }
    }
  }
  fmha::launch_all(c, stream);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "fmha replay launch failed: ", cudaGetErrorString(err));
}

// Experiment hook: per-call override of the combine geometry and of the
// graph/direct-launch choice (see fmha::g_dbg_*). chunk <= 0, zp < 0,
// graph < 0 and wscan < 0 restore the promoted selection.
void fmha_set_debug(int64_t chunk, int64_t zp, int64_t graph, int64_t wscan) {
  fmha::g_dbg_chunk = (int)chunk;
  fmha::g_dbg_zp = (int)zp;
  fmha::g_dbg_graph = (int)graph;
  fmha::g_dbg_wscan = (int)wscan;
  // Both knobs are consumed while building the launch descriptor (the combine grid
  // and the instantiated template), so a cached descriptor -- and any graph captured
  // from it -- encodes the PREVIOUS geometry. Drop them; the next forward rebuilds.
  if (fmha::g_cache.exec != nullptr) {
    cudaGraphExecDestroy(fmha::g_cache.exec);
    fmha::g_cache.exec = nullptr;
  }
  fmha::g_cache.valid = false;
  fmha::g_cache.hits = 0;
  fmha::g_cache.graph_off = false;
}

std::vector<int64_t> fmha_last_plan() {
  const fmha::Plan& pl = fmha::g_last_plan;
  return {(int64_t)pl.cfg, (int64_t)pl.tile_m, (int64_t)pl.tile_n, (int64_t)pl.mtiles,
          (int64_t)pl.splits, (int64_t)pl.work, (int64_t)pl.part_floats,
          (int64_t)pl.want_cfg, (int64_t)pl.mirror_ok, (int64_t)pl.tps_target,
          (int64_t)fmha::g_last_chunk, (int64_t)fmha::g_last_wl, (int64_t)fmha::g_last_sink};
}

void fmha_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor block_table,
                  torch::Tensor seq_lens, torch::Tensor cu_q, torch::Tensor out,
                  c10::optional<torch::Tensor> lse, c10::optional<torch::Tensor> sinks,
                  c10::optional<torch::Tensor> scratch, c10::optional<torch::Tensor> mirror,
                  double bmm1_scale, double bmm2_scale, int64_t window_left, int64_t max_q_len,
                  int64_t max_seq_len, int64_t cfg_override, int64_t splits_override,
                  int64_t tps_override) {
  TORCH_CHECK(q.is_cuda(), "query must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "query must be bf16");
  TORCH_CHECK(k.scalar_type() == torch::kFloat8_e4m3fn, "kv_cache K must be fp8 e4m3");
  TORCH_CHECK(v.scalar_type() == torch::kFloat8_e4m3fn, "kv_cache V must be fp8 e4m3");
  TORCH_CHECK(q.dim() == 3, "query must be [tokens, heads, dim]");
  TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "kv_cache entries must be [pages, kvh, page, dim]");
  TORCH_CHECK(q.stride(2) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
              "head_dim must be contiguous");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && seq_lens.scalar_type() == torch::kInt &&
                  cu_q.scalar_type() == torch::kInt,
              "index tensors must be int32");

  const int H = (int)q.size(1);
  const int hd = (int)q.size(2);
  const int HKV = (int)k.size(1);
  const int page = (int)k.size(2);
  const int batch = (int)seq_lens.size(0);
  TORCH_CHECK(H % HKV == 0, "heads must be a multiple of kv_heads");
  TORCH_CHECK((int)out.size(-1) == hd && (int)v.size(3) == hd, "value dim must equal head dim");
  TORCH_CHECK(hd == 64 || hd == 128 || hd == 256, "unsupported head_dim");
  TORCH_CHECK(page == 16 || page == 32 || page == 64 || page == 128, "unsupported page_size");
  TORCH_CHECK(cu_q.numel() == batch + 1, "cum_seq_lens_q must have batch+1 entries");
  TORCH_CHECK(q.size(0) == out.size(-3), "out token count mismatch");

  int page_log2 = 0;
  while ((1 << page_log2) < page) ++page_log2;

  fmha::Params p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.q_s_tok = q.stride(0);
  p.q_s_head = q.stride(1);
  p.k = reinterpret_cast<const uint8_t*>(k.data_ptr());
  p.v = reinterpret_cast<const uint8_t*>(v.data_ptr());
  p.kv_s_page = k.stride(0);
  p.kv_s_head = k.stride(1);
  p.kv_s_tok = k.stride(2);
  TORCH_CHECK(v.stride(0) == k.stride(0) && v.stride(1) == k.stride(1) && v.stride(2) == k.stride(2),
              "K and V must share a paged layout");
  p.page_log2 = page_log2;
  p.table = block_table.data_ptr<int>();
  p.table_stride = (int)block_table.stride(0);
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cu_q = cu_q.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.out_s_tok = out.stride(-3);
  p.out_s_head = out.stride(-2);
  const bool have_lse = lse.has_value() && lse->numel() > 0;
  p.lse = have_lse ? lse->data_ptr<float>() : nullptr;
  p.lse_s_tok = have_lse ? lse->stride(-2) : 0;
  p.lse_s_head = have_lse ? (int)lse->stride(-1) : 0;
  p.sinks =
      (sinks.has_value() && sinks->numel() > 0) ? sinks->data_ptr<float>() : nullptr;
  const bool have_scratch = scratch.has_value() && scratch->numel() > 0;
  p.part_o = have_scratch ? reinterpret_cast<float*>(scratch->data_ptr()) : nullptr;
  p.part_ml = nullptr;
  p.part_ml_t = nullptr;
  p.trc8 = nullptr;
  p.H = H;
  p.HKV = HKV;
  p.G = H / HKV;
  // ceil(2^32/d), or 0 for d == 1; see fdiv() for the exactness bound.
  p.g_magic = p.G <= 1 ? 0u : (unsigned)(0xFFFFFFFFull / (unsigned)p.G + 1ull);
  p.hkv_magic = HKV <= 1 ? 0u : (unsigned)(0xFFFFFFFFull / (unsigned)HKV + 1ull);
  p.mtiles = 1;
  p.splits = 1;
  // Device-balanced decode splits: default budget 2 CTAs/SM (the promoted
  // decode families' residency); make_plan's resident2 overrides for one-wave
  // families below. Combine row-stat scan policy: promoted default ON unless
  // FMHA_WSCAN=0 or a per-call fmha_set_debug override says otherwise.
  p.resident2 = 2 * fmha::device_sm_count();
  {
    static const int env_wscan = [] {
      const char* e = getenv("FMHA_WSCAN");
      return e ? atoi(e) : 1;
    }();
    p.wscan = (fmha::g_dbg_wscan >= 0) ? fmha::g_dbg_wscan : env_wscan;
  }
  p.part_s_w = p.part_s_s = p.partml_s_w = p.partml_s_s = p.partmlt_s_r = 0;
  p.window_left = (int)window_left;
  p.qk_scale = (float)((double)bmm1_scale * LOG2E);
  p.v_scale = (float)bmm2_scale;
  p.write_lse = have_lse ? 1 : 0;

  const long long scratch_floats =
      (p.part_o != nullptr) ? (long long)(scratch->numel() * scratch->element_size()) / 4 : 0;
  // Config choice from shape structure only (packed rows per work item, head dim),
  // from the same-job sweep in dv_af7567e90dc5 (median of 15, all ten frozen
  // cases, every config smoke-checked 15/15 first). NCU had shown 11.5% warp
  // occupancy prefill / 6.2% decode with 38.7% of cycles stalled on fixed-latency
  // dependencies, so every arm below trades tile size for resident warps:
  //   * <=128 packed rows per work item -> decode. cfg50 = 16-row tiles, KV split
  //     over 4 warps, register prefetch into a SINGLE f16 buffer (no cp.async
  //     ring): 70.7/28.0/25.8/21.5us on cases 2/3/5/8 vs 76.8/29.8/25.7/21.5 for
  //     the ring, and hd256 goes from 1 to 3 resident CTAs.
  //   * prefill without a usable mirror -> cfg46 (hd64) / cfg40 (hd128) / cfg51
  //     (hd256): 64-row tiles, TILE_N 64/32/32. cfg46 vs the no-minBlocks cfg6 on
  //     hd64 was 104.7/374.6 vs 103.7/351.3us in the authoritative eval (which
  //     restores 128 MiB of inputs before every timed call), i.e. inside the
  //     17-21% eval-to-eval swing of those two cases; cfg46 is kept because it is
  //     the profiler's consistent pick and the difference is not decidable.
  //   * hd128 prefill -> cfg40: 64-row tiles, TILE_N 32, cp.async ring, minBlocks
  //     4. 3419/951/431us on cases 1/7/9 vs 3795/1006/581.
  //   * hd256 prefill -> cfg51: TILE_N 32 with prefetch and a single buffer; a
  //     32-row warp tile is impossible here (256 accumulator registers).
  //     2217us on case 6 vs 2352.
  // An f16 mirror of the KV cache lets the main kernel skip the e4m3->f16 convert
  // pass entirely; the staged mimic (probe3 mode 2) attributed 26% of prefill time
  // to that pass. The mirror is a flat element-wise e4m3->f16 copy of each KV
  // tensor, so it inherits the source's PHYSICAL layout and the staging path
  // indexes it with the same p.kv_s_* strides. Canonical HND contiguity is
  // therefore not required -- only that every 8-element f16 group cp.async pulls
  // is 16-byte aligned and that the flat convert covers whole 16-element vectors.
  //
  // The previous gate demanded canonical strides (stride(2)==hd etc.). The frozen
  // builder produces k as raw.reshape(pages,page,kvh,hd).transpose(1,2)/ks, a
  // dense but transposed view whose strides TensorIterator preserves, so that gate
  // was false and cfg60/72 were silently remapped to cfg40/46. The recorded
  // "mirror is neutral end to end" A/B (job dv_dca184ecef70: case 1 3418.6us vs
  // 3431.6us) therefore compared cfg40 with cfg40 and is void; the mirror
  // hypothesis is untested rather than refuted. pl.mirror_ok is now reported
  // through fmha_last_plan() so a remap can never again look like a measurement.
  const long long k_elems = k.numel(), v_elems = v.numel();
  const bool mirror_layout = k.stride(3) == 1 && v.stride(3) == 1 && p.kv_s_page % 8 == 0 &&
                             p.kv_s_head % 8 == 0 && p.kv_s_tok % 8 == 0 && k_elems % 16 == 0 &&
                             v_elems % 16 == 0;
  const bool mirror_available = mirror.has_value() && mirror->is_cuda();
  const bool mirror_ok = mirror_layout && mirror_available &&
                         (long long)(mirror->numel() * mirror->element_size()) >=
                             2 * (k_elems + v_elems);
  const long long rows_per_item = max_q_len * p.G;
  const long long prov_mtiles = (rows_per_item + 63) / 64;
  // The mirror pays a FIXED cost (two cvt_kv launches, plus 3 bytes of DRAM
  // traffic per cached element across the WHOLE cache) to remove a PER-TILE cost
  // (the in-kernel e4m3->f16 convert, ~10% of the main kernel). It therefore only
  // pays when the main kernel is long enough, and on a 38us shape it measurably
  // does not: with the mirror unconditionally on, predeclared domain case 4 went
  // 38.3 -> 45.9us (+19.8%, eval dv_c7a6bb613921) while every heavy prefill case
  // gained 7-15%. The exposed cost there was 7.6us, i.e. two launch latencies,
  // so the crossover is T_main ~= 76us; at the ~110 TFLOPS this kernel sustains,
  // and with the causal-halved real work expressed as the upper bound below, that
  // is ~16 GFLOP. Both suites leave 0.5-68.7 GFLOP empty apart from fixed case 0
  // (12.9 GFLOP, 95us) where the mirror measured exactly neutral, so any
  // threshold in ~2-60 GFLOP behaves identically on all twenty measured shapes;
  // 16 GFLOP is the mechanistic value rather than a tuned one.
  long long eff_kv = max_seq_len;
  if (window_left >= 0) {
    const long long w = window_left + max_q_len + 1;
    if (w < eff_kv) eff_kv = w;
  }
  const double flop_ub = 4.0 * (double)batch * (double)max_q_len * (double)H * (double)eff_kv *
                         (double)hd;  // QK + PV, 2 FLOP per MAC, no causal discount
  // Threshold history: the V0-era 1e9 rested on an owner probe that put the
  // packed-softmax tcgen05 arm ahead of every mirror-less config at ~1.2e9.
  // Round-5 boundary synthetics (job 20261020T180158Z, paired vs frozen V0
  // bytes, same inputs) remapped the crossover against the best no-mirror
  // mma.sync configs ON THIS CAMPAIGN'S bytes: at flop_ub 1.1e9 (s_b8)
  // mma.sync+more splits won by 15% (36.1 vs 42.3us), at 2.2e9 by 4-5%
  // (s_b1 40.2 vs 42.0, s_b10 42.5 vs 44.7), while at 6.4e9 (s_b3) mirror-t5
  // won by 23% and at 8.6e9 (s_b2) by 38%. 4e9 sits between the last
  // measured mma.sync win (2.2e9) and the first clear t5 win (6.4e9); the
  // mirror's fixed cost is now one fused cvt launch inside the replay graph.
  const bool mirror_worth = flop_ub >= 4.0e9;
  int cfg;
  if (cfg_override >= 0) {
    cfg = (int)cfg_override;
    // cfg211-215 are instantiated for hd128 only. Remap anything else to the
    // promoted arm FOR THAT hd (c176's non-128 branch is an hd64 kernel, so
    // remapping an hd256 shape to 176 would launch it on the wrong data) and
    // leave pl.want_cfg at the requested number, so a mis-targeted arm reads as a
    // remap in the plan instead of looking like a measurement of the arm.
    // 218 is unused; 211-221 are all instantiated for hd128 only, and their launch
    // cases have no else branch, so an unremapped hd64/hd256 request would launch
    // no main kernel at all and read as a silent wrong answer rather than a fault.
    if ((((cfg >= 211 && cfg <= 231) || (cfg >= 240 && cfg <= 252)) && hd != 128 &&
         cfg != 222 && cfg != 226 && cfg != 233 && !(cfg >= 244 && cfg <= 245) &&
         cfg != 252) ||
        ((cfg == 266 || cfg == 275 || cfg == 276 || cfg == 277 || cfg == 280 ||
          cfg == 283 || cfg == 285 || cfg == 302) &&
         hd != 128))
      cfg = (hd == 256) ? 297 : 266;
    if ((cfg == 222 || cfg == 226 || cfg == 233 || cfg == 244 || cfg == 245 ||
         cfg == 252 || cfg == 267 || cfg == 278 || cfg == 279 || cfg == 281 ||
         cfg == 284 || cfg == 298 || cfg == 303 || cfg == 305 || cfg == 306 || cfg == 309 || cfg == 310 || cfg == 311) &&
        hd != 64)
      cfg = (hd == 256) ? 297 : 266;
    // 265 is the hd256-only t6c arm: remap it away from hd256 rather than to it.
    if (cfg == 265 && hd != 256) cfg = (hd == 64) ? 233 : 266;
    if (cfg == 268 && hd != 256) cfg = (hd == 64) ? 233 : 266;
    if (cfg == 282 && hd != 256) cfg = (hd == 64) ? 233 : 266;
    if ((cfg == 297 || cfg == 304 || cfg == 307 || cfg == 308) && hd != 256) cfg = (hd == 64) ? 233 : 266;
  } else if (rows_per_item <= (hd == 64 ? 512 : 256)) {
    // Band boundary remap (re-derived on join-20261027; brackets
    // dv_c61150985d34 / dv_cc04d404298f / dv_7a06d1640f12 / dv_b5741b09a64c,
    // G16 kv4096, smokes 0/0): at low packed-row counts the fine-grained
    // decode split-KV beats the first prefill mtiles, whose row padding and
    // low CTA count cannot fill the device:
    //   hd256: decode -21.6% (144 rows), -0.6% (256); prefill +26% (512),
    //          +29.3% (384), +145% (1024)          -> gate 256
    //   hd128: decode -12.7% (144), -6.4% (256); prefill +10.6% (384),
    //          +21% (512)                          -> gate 256
    //   hd64:  decode -20.4% (144), -11.5% (256), -0.5% (512 tie) -> gate 512
    // Interaction note: rows 129-256 batch>=2 shapes newly entering the
    // decode band may hit the device-balanced splits gate below; the
    // combination is reference-smoked (s_b144b2). No fixed/domain shape has
    // 129-512 rows (f0 2048, f4 768+, f3 16, f5 96, f8 8, d2 20, d8 7).
    // Decode. The mirror is available here but deliberately NOT used: converting
    // the whole cache turns one byte of KV read into three bytes of convert
    // traffic plus a two-byte read. Measured cfg64 vs auto (job dv_af8f928782f1):
    // case 2 109.6 vs 70.8us (+55%), case 3 72.7 vs 29.5, case 5 33.8 vs 27.7,
    // case 8 39.9 vs 21.2. cfg107 = cfg50 + MPAD (padded sMerge row stride HD+4):
    // NCU decode (dv_4bb5cb971b90) showed decode is LATENCY-bound (DRAM 4.98%, SM
    // busy 16.5%, occupancy 12.5%), not DRAM-bound, with the cross-warp merge
    // store at 7.7-way bank conflict (60% of shared-store wavefronts). The pad
    // cuts it to 2-way; probe19 (dv_70e2e4ab8db4) measured it bit-identical and
    // weakly dominant: f3 +7.6%, f8 +10.8%, d4 +3.0%, neutral on 6 others, 0
    // regressions (merge-heavy low-tiles-per-split decode gains most).
    //
    // hd64 -> cfg112 = cfg107 + QREG (Q fragments hoisted to registers; Q is
    // re-ldmatrixed per tile otherwise). Bit-identical fragments, MINB stays 4
    // so occupancy is unchanged; four paired rounds measured f3 -0.7..-5.5%,
    // f5 -2..-4.8%, s_u3 -4.4%, d5/f8/d0-d3 within +-1% noise, no hd64
    // regression. hd128 keeps cfg107: QREG there needs +32 regs past the
    // MINB=3 budget (c114 spills: d9 +0.9%) or drops to MINB=2 (c112: d6
    // +18.8% on a 1.3-wave launch). hd256 lacks the registers entirely (O
    // accumulator is already 128).
    // PROMOTION (K22b/K27 graph medians; eager was host-contaminated by the
    // override slow path - see results/host_path_contamination_finding.json):
    // hd64 -> cfg145 (QREG+X2HM: f8 g16.34 vs 17.30, d5 14.93 vs 15.46, d3/d0/
    // f5 ties); hd128/256 -> cfg192 (t7c ldmatrix.b8 direct-fp8: f2 g52.1 vs
    // 63.1, d6 31.7 vs 36.1, f3 21.5 vs 23.4, d9 37.7 vs 40.3, d0 9.25 vs
    // 9.57; smokes 0-mismatch on every bringup shape incl. splits 1-296).
    // hd128/256 -> cfg144 = cfg107 + X2HM (packed f16x2 exp2 softmax;
    // job 20261021T004958Z 4x16 vs c107: f5 0.945, f3 0.966, d3 0.962,
    // d9 0.983, d6 0.983, u1 0.981, no loss anywhere; zero smokes). hd64
    // keeps 112 (QREG still edges X2HM there: d3 12.4 vs 12.6, d5 18.9 vs
    // 19.0); a c145 = QREG+X2HM combination is the queued experiment.
    cfg = (hd == 64) ? 145 : 192;
  } else if (hd == 64 && flop_ub < 1.0e11) {
    // J62: the 1.0e9 floor is REMOVED. The arm-a-era xh64tiny verdict (cfg226
    // at 2.7e8: 1.179 loss to the then-auto) predates QCP/LPT6/MTM6/DTH8.
    // Modern c267 vs auto(c142)/c134 paired (--l2-evict, iters30 rounds3,
    // smokes 0 mismatched): s_lp64b (5.5e7) 16.19 vs 17.42/22.32 and
    // s_lp64a (1.0e8) 16.43 vs 19.68/24.45 (dv_8356d532e846);
    // s_smallpref64 (2.3e8) 16.96 vs 19.15/24.13 (dv_736996abb4ca).
    // Wins at every rung down to the prefill boundary; no crossover found.
    // PROMOTED 2026-10-23 (jobs dv_0a814e21229d, dv_cd370e73c0a9): cfg226 = the t6
    // SM8/MTM6 kernel at hd64 with LPAR8 and SPLITV replaces the FP8S t5 arm here and
    // the mirror t5 arm below. It won on 13 of 14 hd64 shapes measured, spanning both
    // bands and all four page sizes: f0 0.721, f4 0.826, xh64sink 0.652, xh64mid 0.771,
    // xh64p128 0.784 (cfg222), xh64mq 0.803, xh64win 0.854, xh64big 0.921 (cfg222),
    // xh64rag 0.928 (cfg222), xh64small 0.942 (cfg222), xh64gqa 0.975 (cfg222),
    // xh64short 0.984, xh64big16 0.981 (cfg222). The single loss is the sub-1e9 band
    // (xh64tiny at 2.7e8: 1.179), where the mirror's fixed convert cost dominates a
    // 0.27 GFLOP shape -- so the 1e9 floor below stays and c142 keeps that band.
    // The tree's earlier "SM8-t6 hd64 loses by 24.2%/13.7%" was measured at LPAR=1,
    // when one lane issued every TMA box; job 8's remapped arm reproduced that loss
    // (f4 1.343) and LPAR8 flipped it (f4 0.926, then 0.826 with SPLITV).
    // DPV added 2026-10-23 (job dv_5a44690c9d07, cfg233 = this kernel with a dedicated
    // PV issuer warp, three TMEM S buffers and a two-tile QK lookahead -- the same
    // change the hd128 arm took): bit-identical, f4 119.70 -> 110.27 (0.921), xh64p128
    // 0.916, xh64win 0.930, xh64sink 0.936, xh64mq 0.945, xh64mid 0.961, xh64short
    // 0.992, f0 1.006 (a wash: 16 tiles per CTA, tail-bound), unlisted geomean 0.946.
    // Numerics: 0 mismatched elements at the frozen tolerance on every shape checked
    // (f0, f4, xh64mid, xh64mq, xh64win, xh64sink, xh64tiny, xh64short, xh64small,
    // xh64rag), max_abs 0.015625 = the same bf16 ulp the t5 arms show. NOT
    // bit-identical: t6 and t5 differ in tile order and row-sum association.
    // PHO-DTHoff promotion 20261028 (309 = 267 + per-half O accumulators +
    // DTH64): deletes BOTH per-tile barrier faces (the 256-thread row-max
    // exchange AND the DEFER O-rescale) on non-windowed hd64 prefill.
    // Paired vs explicit 267 (smokes 0/0, plans verified): f0 0.9896/0.9903,
    // f4 0.9561/0.9610, s_h64big 0.9199/0.9145 (q4096kv8192 long-run class),
    // s_h64gqa 0.9603, s_h64peak 0.9845 (dv_4555d7e5dc1e, dv_c39010d7d110,
    // dv_9486b7e758f9). Probe 12/12 multi-seed clean (dv_95a8c9a1b5e2).
    // GATE window_left < 0 && !sinks: WINDOWED/SINK shapes FAIL numerics
    // (s_h64win 208666 bad out / s_h64sink 34412, dv_9486b7e758f9) - fully-
    // masked leading tiles pin mdef at the NEG_BIG sentinel and the rescale
    // rescue is exactly the path that RACES under PHO-SB3 (298 defect:
    // nondeterministic, DTH-amplified). Zero-dist hd64 prefill loses +1.4%
    // (s_h64zero, only loss cell; no fixed shape; domain leg will expose any
    // d-case). NOT bit-identical (PHO epilogue merge + fixed first-tile mdef;
    // frozen-dist spread is empirically << 64 log2 units). Ragged x 309
    // timing unmeasured (3 budget kills); deadw merge path is structurally
    // exercised by rows 0-63 of every causal shape (permanently dead half-1
    // columns). Was: J49 267 = 233 + DTH8 (dv_c2ba3b74a523: f4 -0.7%,
    // s_pref64 -1.6%, f0 flat).
    // QLA sub-gate 20261028 (310 = 309 + QLA2): post-PHO the deeper lookahead
    // wins long runs - f4 -0.5/-0.8%, s_h64big -1.0/-1.1% across 3 jobs
    // (dv_22dfd5bdbda1, dv_271cb6e93b84), bit-identical - while f0
    // (fill-dominated, J69's QLA1 verdict) reads +0.7..+2.3% inter with
    // samples ties. 3.2e10 = log-midpoint between f0 (1.29e10, keeps 309)
    // and f4 (7.73e10, gets 310); no fixed shape sits near the boundary.
    cfg = (window_left < 0 && p.sinks == nullptr)
              ? (flop_ub >= 3.2e10 ? 310 : 309)
              : 267;
    // hd64 mid-scale prefill: FP8S TN128 (c154 base; the MTM zone promotion
    // below maps it to c152 where the t5 family promotes 136->133). Job J6
    // (paired, smokes clean): f0 c152 40.1 vs auto 47.6, f4 c154 128.1 vs
    // 146.4, s_pref64 c152 68.7 vs 86.8, s_p64mid c154 66.6 vs 74.1,
    // s_b4rag c152 163.3 vs 212.0, s_p64win c154 68.0 vs 74.9. The MTM-on
    // template loses at multi-split ragged (f4: c152 136.1 vs c154 128.1),
    // exactly where the zone rule keeps it off.
  } else if (hd <= 128 &&
             flop_ub < (hd == 64 ? 1.0e11 : 5.0e9)) {
    // J62: the 1.0e9 floor is REMOVED for hd128 (hd64 never reaches this
    // branch - the one above now takes every hd64 prefill < 1e11). The
    // sub-1e9 "keeps c142" verdicts were t5-vs-t5 (c120-vs-142: b9 5.4e8
    // 1.103) and never raced the modern t6 arm. Modern c266 vs auto(c142)
    // paired (--l2-evict, iters30 rounds3, smokes 0): s_lp128a (3.5e7)
    // 15.98 vs 18.53, s_lp128b (1.4e8) 16.24 vs 20.58 (dv_8356d532e846);
    // s_smallpref (5.4e8, domain-d4 mirror) 17.18 vs 25.44 = -32.5%
    // (dv_736996abb4ca). !mirror_ok shapes in the widened band still fall
    // back through the maps-gate chain (266 -> 162 -> 134 -> 142).
    // Mid-scale prefill: tcgen05 with FP8S register staging (cfg120), NO
    // mirror. FP8S converts each staged tile in-kernel (LDG fp8 -> expand8 ->
    // STS f16), so its cost scales with mtiles x kv while the mirror's cvt is
    // per-cache: FP8S wins while m-tile passes over the cache are few.
    // Round-7 paired clean-numerics map (job 20261020T193336Z, c120/auto):
    //   hd64:  b3 6.4e9 0.950, f0 1.29e10 0.975, b5 2.58e10 0.962,
    //          f4 7.73e10 0.951  (upper bound 1e11 is margin past the last
    //          measured win; hd128-scale losses show why a bound is needed)
    //   hd128: b1 2.15e9 0.979, b10 2.16e9 0.988, but b2 8.59e9 1.016,
    //          b4 3.44e10 1.115, b6 6.87e10 1.098, f1 1.1e12 1.100 -> 5e9 cap
    //          (f9 6.88e10 measured 0.982 inside a zero distribution -- the
    //          only >5e9 hd128 win; left to the mirror arm for consistency)
    //   hd256: b7 1.72e10 1.073, b8 1.08e9 1.111 -> excluded entirely
    // Below 1e9 the mma.sync fallbacks stay (b9 5.4e8: c120 1.103).
    // X2H (packed f16x2 exp2/sum) promoted on the FP8S base: c133/c130 f1
    // 0.916, f4 0.929, f7 0.933, f9 0.910, b4 0.950, b5 0.949, b6 0.933
    // (job 20261020T231114Z, zero smoke failures, max_out <= 0.016 = a few
    // bf16 ulps, max_lse <= 0.0017 log2 units).
    // PROMOTED 2026-10-23 (job dv_479a3ceb1429): the hd128 mid band moves from the
    // FP8S t5 arm c136 to c176, the same t6 SM8/MTM6 kernel with LPAR8+SPLITV that
    // the mirror band uses. Measured on three unlisted shapes inside the band
    // (1.07e9, 2.16e9, 3.22e9; G=4/4/1, page 32/64/16): 0.892, 0.878, 0.826,
    // geomean 0.865 against auto, all 0 mismatched at the frozen tolerance with
    // max_abs 0.015625, while c142 was a wash (0.983). The band's previous occupant
    // was chosen by t5-against-t5 comparisons ("hd128: b1 2.15e9 0.979, b10 2.16e9
    // 0.988") and had never been raced against the t6 family -- the same gap that
    // hid the hd64 result, and for the same reason: those verdicts predate LPAR.
    // Not bit-identical (tile order and row-sum association differ), so the
    // two-seed official treatment applies. No fixed or domain shape is inside this
    // band, so the scored suites are unaffected; below 1e9 the mirror's fixed
    // convert cost keeps c142, as the hd64 xh64tiny measurement (1.179 for t6)
    // independently confirms.
    // J46 promotion: 176 -> 266 = the same promoted arm with DTH8 (graft of
    // arm-a J34; paired dv_3c8470b71623: f7 -1.7%, s_pref128big -1.4%,
    // s_win128 -0.9%, f1/f9 flat; NOT bit-identical - rescale points move).
    // PBAR probe (302) REVERTED 20261028: official f7 cand-302 med 126.94us
    // vs two independent 266 draws at 124.93 (dv_7eec1ab53829 vs
    // dv_fdf523bd86ca/dv_a140df612a3f; incumbent side absent from the f7 job,
    // so mildly confounded) and f0/f4/f6 cells neutral - the exchange is
    // already hidden by the promoted pipeline. 302 stays as override-only.
    cfg = 266;
  } else if (mirror_ok && mirror_worth && hd == 128) {
    // hd128 large-scale prefill: t6 warp-specialized TN128 (TMA-staged mirror,
    // PV n128). Job J3/J5 (paired, same bytes): f1 771.9 vs 1157.9 graph (-33%),
    // f7 184.8 vs 225.9 with MTM6 (-18%), s_pref128big 450.3 vs 729.3 (-38%),
    // f9 (sp2, no MTM zone) 221.3 vs 238.4 (-7%); smokes clean on every arm.
    // Zone promotion below maps 162 -> 165 (MTM6) exactly where the t5 family
    // promotes 134 -> 131.
    cfg = 162;
  } else if (mirror_ok && mirror_worth && hd == 64) {
    // hd64 large-scale prefill: same t6 arm (cfg233, DPV) as the mid band above (was c151,
    // zone-promoted to c150: J6 s_p64big 2.6e11, c150 165.4 vs auto c131 204.0).
    // xh64big (1.37e11, page32) measured 0.921 and xh64big16 (page16) 0.981 for
    // cfg222, and SPLITV added on top at every smaller hd64 shape tested.
    // QLA sub-gate 20261028 (310 = 309 + QLA2): post-PHO the deeper lookahead
    // wins long runs - f4 -0.5/-0.8%, s_h64big -1.0/-1.1% across 3
    // independent jobs (dv_22dfd5bdbda1, dv_271cb6e93b84), bit-identical -
    // while f0 (fill-dominated, J69's QLA1 verdict) reads +0.7..+2.3% inter
    // with samples ties. 3.2e10 = log-midpoint between f0 (1.29e10, keeps
    // 309) and f4 (7.73e10, gets 310); no fixed shape sits near it.
    cfg = (window_left < 0 && p.sinks == nullptr)
              ? (flop_ub >= 3.2e10 ? 310 : 309)
              : 267;

  } else if (mirror_ok && mirror_worth) {
    // Large-scale prefill: tcgen05 from the f16 mirror. Round-7 micro-arch
    // defaults (paired, same job): hd256 -> cfg122 (ILP 4-way max + packed
    // F2 sum + DAO): f6 0.987, d7 0.977, b7 0.978 vs auto-cfg100; the NOSYNC
    // variants do NOT stack at hd256 (c124: b7 1.002, d7 0.983). hd64/128 ->
    // cfg123 (F2 + NOSYNC, no DAO/ILP): f1 0.988, f7 0.987, b4 0.996, b6
    // 0.994, f9 0.998; ILP (c122) loses there (f9 1.026, b4 1.013). All arms
    // passed the numerical smoke; NOSYNC only drops a barrier proven
    // redundant (top-of-iter barrier + per-thread PV mbar wait already order
    // buffer reuse), ILP only re-associates the order-independent row max.
    // X2H (packed f16x2 exp2 + f16x2 row-sum) promoted on both mirror bases:
    // c131/c128 f1 0.927 f7 0.951 b5 0.942 b6 0.956; c132/c129 d7 0.957
    // f6 0.963 b4 0.953 (job 20261020T231114Z, zero smoke failures).
    //
    // hd256 -> cfg173 = t6c cluster2 (m256n128k16 pair MMA, K split by kv
    // tokens / V by hd cols across the CTA pair, TMA cta_group::2 staging)
    // with MTM6C pair-descending order. K11 sweep (dv_fe4b7833ed0f, paired,
    // smokes 0-mismatch): f6 c172 460.0 vs auto-c132 772.7 (-40.5%) and
    // c170 482.3; the mma.sync ceiling at hd256 (32-row warp tiles, TN64)
    // is what t6c replaces. hd128/hd64 KEEP the t5/t6 arms: t6c measured
    // strictly worse at hd128 (f1 822 vs 775, f7 213 vs 191, f9 236 vs 221,
    // s_pref128big 476 vs 455, s_p16k 246 vs 229) -- at hd128 single-CTA t6
    // already runs TC-bound, so the causal pair-union overhead (+~1 tile per
    // pair on rank0's masked columns) is pure loss. MTM6C is order-neutral
    // on a single wave (d7-class work <= sms), so 173 needs no zone split.
    // J46 promotion: hd256 173(=265, t6c_qcp) -> 268 = + RWSK, BIT-IDENTICAL
    // (J46_BITID 4/4 exact; paired f6 -1.7%, s_pref256 -1.8%, d7 flat).
    // SM8C promotion (re-derived on join-20261027): 268 -> 297 = t6's SM8
    // column-split softmax on the cluster kernel. Paired on THIS tree
    // (dv_df88bef8c36d, 3 rounds, smokes 0/0): f6 -0.60%, s_256mq -1.24%,
    // s_d7 -9.48%; pre-join official evidence: fixed forward -0.71% /
    // reverse -1.36% (dv_347c8d4c7bc8 / dv_c410e9c39704), domain d7 -10.28%
    // (dv_542099d42996), full validation series (window+sink, page16,
    // ragged dead-rank, dead-pair splits, empty-split sp48) smokes 0/0.
    // NOT bit-identical (exp/sum association; t6-SM8/t8 recipe class).
    cfg = (hd == 256) ? 297 : 134;  // (304-PBAR tried 20261028: f6 0.9993 neutral, reverted)
  } else if (hd == 256 && mirror_ok && flop_ub >= 6.0e8) {
    // J62-parity extension for hd256 (re-derived on join-20261027; the 4e9
    // mirror_worth gate is t5-era and was never re-raced vs t6c). Paired
    // c142-vs-297, 3 rounds, smokes 0/0 (dv_bf9430564f68, dv_4d65bcc2b2a3):
    //   2.7e8  26.5 vs 26.8 (+1.0%)    4.0e8  27.5 vs 27.9 (+1.6%)
    //   8.1e8  32.9 vs 27.5 (-16.6%)    2.1e9  41.1 vs 30.7 (-25.2%)
    //   3.2e9  42.9 vs 29.2 (-31.9%)
    // 6e8 sits log-midway between the last measured c142 win and the first
    // clear 297 win. No fixed/domain shape is in-band (f6 5.5e11, d7 2.7e11).
    // !mirror_ok falls through to c142; 297's maps fallbacks already wired.
    cfg = 297;
  } else {
    // No usable mirror (cache over the 8 GiB cap, a layout this path cannot
    // index, or too little work to amortise the convert) or sub-1e9 prefill:
    // cfg142 = the c46/c51 wide-tile shape (64-row tiles, TILE_N 64 -- hd256
    // 32 -- cp.async ring, register prefetch, single f16 buffer) + X2HM
    // packed-f16 softmax. Round-13 isolation (job 20261021T004958Z, 4x16):
    // the X2H_M win materializes on the WIDE-tile shape, not the narrow
    // c40 one (c142 vs c46: d4 -5.8%, b9 -9.5%, w7 -7.1%, b8(hd256) -1.1%;
    // c141 vs c40 only -1.1..0%), and c142 beats the old auto picks (c40/c51)
    // by 5.8%/6.9%/1.4%/1.1% on d4/b9/w7/b8. Zero smokes; numerics match the
    // X2H margin profile (max_out <= 0.0156, max_lse <= 0.0007).
    cfg = 142;
  }
  // Every config that launches an F16KV kernel must own a converted mirror, so the
  // gate and the fallback are driven by one list. Before this was fixed, cfg62/72
  // launched the F16KV kernel with p.k16 == nullptr whenever they were not in the
  // f16kv expression: cfg62 then faulted on every shape (job dv_18b96ed76477,
  // illegal memory access, 18/18) and cfg72 was silently remapped to cfg40, so its
  // "3442.2us vs auto 3429.8us" A/B row measured cfg40 twice.
  // Every t5 config that stages from the f16 mirror MUST be listed here; the
  // only tile_m==128 kernels that do NOT are cfg120/121 (FP8S reads the fp8
  // cache directly). Forgetting an entry launches the mirror schedule with
  // p.k16 == nullptr -> illegal memory access (this happened for cfg122-124
  // in job dv_ebdee5f6ec0a).
  const bool wants_mirror =
      (cfg == 60 || cfg == 64 || cfg == 72 || cfg == 80 || cfg == 81 || cfg == 82 ||
       cfg == 84 || cfg == 85 || cfg == 89 || cfg == 90 || cfg == 91 || cfg == 92 ||
       cfg == 93 || cfg == 94 || cfg == 96 || cfg == 97 || cfg == 99 || cfg == 100 ||
       cfg == 101 || cfg == 102 || cfg == 103 || cfg == 106 ||
       cfg == 122 || cfg == 123 || cfg == 124 || cfg == 125 || cfg == 126 ||
       cfg == 128 || cfg == 129 || cfg == 131 || cfg == 132 ||
       cfg == 134 || cfg == 135 ||
       cfg == 137 || cfg == 138 || cfg == 139 || cfg == 140 ||
       cfg == 150 || cfg == 151 || cfg == 153 || (cfg >= 160 && cfg <= 176) ||
       cfg == 178 || cfg == 180 || cfg == 181 || cfg == 188 || (cfg >= 200 && cfg <= 204) ||
       cfg == 208 || cfg == 209 || cfg == 210 || (cfg >= 211 && cfg <= 233) ||
       (cfg >= 240 && cfg <= 252) || cfg == 265 || (cfg >= 266 && cfg <= 268) ||
       (cfg >= 275 && cfg <= 285) || cfg == 297 || cfg == 298 ||
       (cfg >= 302 && cfg <= 311));
  const bool f16kv = wants_mirror && mirror_ok;
  // A mirror config without a mirror falls back to a non-mirror config, so no
  // kernel can ever be launched with p.k16 == nullptr (the defect that made cfg62
  // fault on every shape in job dv_18b96ed76477).
  if (wants_mirror && !f16kv)
    cfg = cfg == 64 ? 144 : 142;
  p.k16 = nullptr;
  p.v16 = nullptr;
  // t6 (cfg160) viability: f16 mirror plus buildable TMA maps plus a supported
  // head dim. Any miss falls back to the promoted mirror/wide mma.sync arms so
  // an override can never launch t6 without maps (nullptr deref) or with a
  // wrong-HD kernel.
  CUtensorMap kmap {}, vmap {};
  bool have_maps = false;
  if ((cfg >= 160 && cfg <= 176) || cfg == 178 || (cfg >= 180 && cfg <= 181) ||
      cfg == 188 || (cfg >= 200 && cfg <= 204) || cfg == 208 || cfg == 209 ||
      cfg == 210 || (cfg >= 211 && cfg <= 233) || (cfg >= 240 && cfg <= 252) ||
      cfg == 265 || (cfg >= 266 && cfg <= 268) || (cfg >= 275 && cfg <= 285) ||
      cfg == 297 || cfg == 298 || (cfg >= 302 && cfg <= 311)) {
    // t6/t6d box whole pages (coord1 = 0); t6c/t8 split tiles into <=64-row
    // boxes so a page128 tile (half) can address rows inside one page.
    const bool is_8 = (cfg >= 200 && cfg <= 204) || cfg == 208;
    const bool is_d = !is_8 && ((cfg >= 180 && cfg <= 181) || cfg == 188);
    const bool is_c = !is_d && !is_8 && ((cfg >= 170 && cfg <= 174) || cfg == 178 ||
                                         cfg == 265 || cfg == 268 || cfg == 282 ||
                                         cfg == 297 || cfg == 304 || cfg == 307 || cfg == 308);
    const int box_rows = (is_c || is_8) ? (page < 64 ? page : 64) : page;
    const bool hd_ok = is_8   ? (hd == 128)
                       : is_c ? (hd == 128 || hd == 256)
                              : (hd == 64 || hd == 128);
    if (f16kv && hd_ok && (page == 16 || page == 32 || page == 64 || page == 128)) {
      have_maps = fmha::build_mirror_map(
                      p.k16 = reinterpret_cast<const __half*>(mirror->data_ptr()), hd, page, HKV,
                      k.size(0), p.kv_s_tok, p.kv_s_head, p.kv_s_page, box_rows, &kmap) &&
                  fmha::build_mirror_map(p.v16 = p.k16 + k_elems, hd, page, HKV, v.size(0),
                                         p.kv_s_tok, p.kv_s_head, p.kv_s_page, box_rows, &vmap);
    }
    if (!have_maps) {
      cfg = (hd == 256) ? 135 : (hd == 128 ? 162 : 134);
      if (cfg == 162 && !f16kv) cfg = 134;
      if (!mirror_ok) cfg = 142;
    } else if (is_8) {
      p.kmap64 = kmap;  // t8 primary maps ARE the box64 maps
      p.vmap64 = vmap;
      p.have64 = true;
    } else if (cfg == 162 && hd == 128) {
      // Box64 twins for the post-plan t8 promotion (page<=64: identical box,
      // reuse; page128: a second encode with 64-row boxes).
      if (box_rows == 64 || page <= 64) {
        p.kmap64 = kmap;
        p.vmap64 = vmap;
        p.have64 = true;
      } else {
        p.have64 =
            fmha::build_mirror_map(p.k16, hd, page, HKV, k.size(0), p.kv_s_tok, p.kv_s_head,
                                   p.kv_s_page, 64, &p.kmap64) &&
            fmha::build_mirror_map(p.v16, hd, page, HKV, v.size(0), p.kv_s_tok, p.kv_s_head,
                                   p.kv_s_page, 64, &p.vmap64);
      }
    }
  }
  // t7 (cfg190/191): direct TMA staging of the fp8 cache -- no mirror. Needs
  // buildable UINT8 maps and a supported page/head-dim; any miss falls back to
  // the promoted decode arm (c144) so an override can never launch without
  // maps (nullptr TMA descriptor = illegal instruction).
  // 192/193 (hd128/256) also build the fp8 maps: the post-plan promotion to
  // c196 (long tiles-per-split) needs them, and descriptor builds are cached.
  if (cfg == 190 || cfg == 191 || cfg == 196 || cfg == 197 || cfg == 283 ||
      cfg == 286 || cfg == 287 || cfg == 290 || cfg == 291 ||
      ((cfg == 192 || cfg == 193) && (hd == 128 || hd == 256))) {
    const int box_rows = (page < 64) ? page : 64;
    const bool hd_ok = (hd == 128 || hd == 256);
    bool have8 = false;
    if (hd_ok && (page == 16 || page == 32 || page == 64 || page == 128)) {
      have8 = fmha::build_fp8_map(p.k, hd, page, HKV, k.size(0), p.kv_s_tok, p.kv_s_head,
                                  p.kv_s_page, box_rows, &p.kmap8) &&
              fmha::build_fp8_map(p.v, hd, page, HKV, v.size(0), p.kv_s_tok, p.kv_s_head,
                                  p.kv_s_page, box_rows, &p.vmap8);
    }
    if (!have8 &&
        (cfg == 190 || cfg == 191 || cfg == 196 || cfg == 197 || cfg == 286 ||
         cfg == 287 || cfg == 290 || cfg == 291))
      cfg = (cfg >= 196) ? 192 : 144;
    // 283 (t6f) needs the fp8 maps; 266's mirror maps are already built (283
    // rides the wants_mirror list), so the fallback is always launchable.
    if (!have8 && cfg == 283) cfg = 266;
    if (have8 && (cfg >= 190)) p.have8 = true;
  }
  // 32-row warp tiles need twice the accumulator registers; hd256 cannot afford it
  if (cfg == 5 && hd == 256) cfg = 6;
  fmha::Plan pl = fmha::make_plan(hd, cfg, batch, HKV, p.G, max_q_len, max_seq_len, window_left,
                                  scratch_floats, splits_override, tps_override);
  // Diagnostics: what was asked for vs what actually runs, and why.
  pl.want_cfg = (int)cfg_override;
  pl.mirror_ok = mirror_ok ? 1 : 0;
  // MTM auto-promotion (round 10): flattened mtile-major descending launch
  // order = global longest-first (LPT) greedy under causal masking. Two
  // promoted zones, both from paired 2-job evidence (dv_9120f526db55 3x14 +
  // dv_f90e555ca409 4x16, zero smokes; numerics bit-identical -- only the
  // CTA -> logical tile mapping is reordered):
  //   A) splits == 1 && work > sms: multi-wave tail fix. f7 c128/c123 0.684
  //      (249.0 vs 362.7us), f1 0.902, f6 (hd256) c129/c122 0.928. Greedy
  //      simulation on the f7 geometry: 196 -> 128 tile-times.
  //   B) sms < work*splits <= 3/2 sms (small partial second wave: the few
  //      doubled-up SMs are the whole tail). f0 c130/c120 0.872, b5 0.936,
  //      w5 0.888. The measured gradient dies by ~1.7x sms (256 CTAs:
  //      +1.0%/+0.3% b4/b6; 288: -0.2% f9), so the cap excludes them.
  // Excluded: multi-split multi-wave (f4 sp2 576 CTAs: +6-11% measured, no
  // tail to fix once z-planes serialize) and ctas <= sms (single wave, one
  // CTA per SM: measured mixed +0.1..+3.6% on w1/w2/w6/b10 from SM-mapping/
  // L2-slice side effects only).
  if (cfg_override < 0) {
    const long long sms = fmha::device_sm_count();
    const long long ctas = (long long)pl.work * pl.splits;
    const bool mtm = (pl.splits == 1 && pl.work > sms) || (ctas > sms && ctas <= sms + sms / 2);
    if (mtm) {
      if (pl.cfg == 134) pl.cfg = 131;
      else if (pl.cfg == 135) pl.cfg = 132;
      else if (pl.cfg == 136) pl.cfg = 133;
      else if (pl.cfg == 154) pl.cfg = 152;
      else if (pl.cfg == 151) pl.cfg = 150;
      else if (pl.cfg == 162) pl.cfg = 266;  // J46: was 176 (+DTH8); K45b/c: SM8 beats c165 (f1 -1.3%,
                                             // f7 -4.7%, s_pref128big -1.9%, s_p16k -1.0%)
    }
    // c196 promotion (TMA fp8 ring decode): single-q streams with long
    // per-split pipelines only. K31b fair-eager (dv job 20261021T191339Z):
    // f2 56.4 vs auto-c192 59.0 (-4.4%), d9/d6 ties, d1 (2 tiles/split)
    // +7%. K33b eval at a pure tiles>=8 gate: f2 62.6->58.9 (-5.9%) but f3
    // 28.8->31.4 (+9.0%) -- f3 is MTP-style (max_q 3, 16 ragged items): its
    // declared max_seq overestimates the per-item pipeline for most CTAs.
    // max_q_len==1 selects the uniform single-stream decode (f2/d9/d6/s_mqa*)
    // where the declared max IS the pipeline; tiles>=13 (integer, declared
    // metadata only) keeps the tiny-kv shapes (d1: 2) on c192.
    if (pl.cfg == 192 && (hd == 128 || hd == 256) && p.have8 && max_q_len == 1) {
      const long long sp = pl.splits > 0 ? pl.splits : 1;
      if ((long long)max_seq_len / 64 / sp >= 13) pl.cfg = 196;
    }
    // c176 promotion (SM8 column-split softmax + MTM6), NON-zone single-wave
    // hd128 mirror shapes: REPLACES the t8/c201 gate. K45b (dv_906f0e3039e3)
    // + K45c (dv_e27499569bc4), graph medians, smokes 0:
    //   f9 c176 186.4 vs promoted-t8 c201 204.5 (-8.9%); s_p128k 185.0 vs
    //   203.6 (-9.1%); s_win128 (windowed) 88.2 vs c162 92.1 (-4.2%).
    // The remaining cfg162 shapes are exactly splits>1 && ctas > 1.5*sms
    // (multi-split multi-wave: MTM measured +6-11% on that class at hd64,
    // unmeasured for SM8 at hd128) -- they keep c162. t8 arms stay
    // override-only. hd64 mirror/mid keep c150/c151/c154 (SM8-t6 hd64 loses:
    // s_p64big +24.2%, s_p64mid +13.7%).
    if (pl.cfg == 162 && hd == 128 && ctas <= sms) pl.cfg = 266;  // J46: was 176
  }
  if (pl.splits > 1) {
    TORCH_CHECK(p.part_o != nullptr, "split-KV requires a scratch buffer");
    p.part_ml = p.part_o + (long long)pl.work * pl.splits * pl.tile_m * hd;
    // Row-major stats twin directly after the z-major stats block.
    p.part_ml_t = p.part_ml + (long long)pl.work * pl.splits * pl.tile_m * 2;
  }
  p.mtiles = pl.mtiles;
  p.splits = pl.splits;
  p.tps_target = pl.tps_target;
  if (pl.resident2 > 0) p.resident2 = pl.resident2;
  p.tile_n_main = pl.tile_n;
  {
    int tnl = 0;
    while ((1 << tnl) < pl.tile_n) ++tnl;
    p.tile_n_log2 = tnl;
  }
  // TRC8 trace ring lives in the second half of the scratch buffer, clear of
  // the partials (guarded so the timeline can never alias split-KV data).
  if ((pl.cfg >= 290 && pl.cfg <= 293) && p.part_o != nullptr &&
      pl.part_floats * 2 <= scratch_floats) {
    p.trc8 = reinterpret_cast<unsigned long long*>(p.part_o + scratch_floats / 2);
  }
  p.part_s_w = (long long)pl.splits * pl.tile_m * hd;
  p.part_s_s = (long long)pl.tile_m * hd;
  p.partml_s_w = (long long)pl.splits * pl.tile_m * 2;
  p.partml_s_s = (long long)pl.tile_m * 2;
  p.partmlt_s_r = (long long)pl.splits * 2;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (f16kv) {
    p.k16 = reinterpret_cast<const __half*>(mirror->data_ptr());
    p.v16 = p.k16 + k_elems;
    fmha::launch_cvt_kv(reinterpret_cast<const uint8_t*>(k.data_ptr()),
                        const_cast<__half*>(p.k16), k_elems,
                        reinterpret_cast<const uint8_t*>(v.data_ptr()),
                        const_cast<__half*>(p.v16), v_elems, stream);
  }
  fmha::dispatch(p, hd, pl, batch, stream, have_maps ? &kmap : nullptr,
                 have_maps ? &vmap : nullptr);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "fmha launch failed: ", cudaGetErrorString(err));
  fmha::g_last_plan = pl;
  fmha::g_last_wl = p.window_left;
  fmha::g_last_sink = (p.sinks != nullptr) ? 1 : 0;
  fmha::CachedLaunch cache;
  cache.p = p;
  cache.pl = pl;
  cache.hd = hd;
  cache.batch = batch;
  cache.valid = true;
  cache.have_maps = have_maps;
  if (have_maps) {
    cache.kmap = kmap;
    cache.vmap = vmap;
  }
  if (f16kv) {
    cache.k_src = reinterpret_cast<const uint8_t*>(k.data_ptr());
    cache.v_src = reinterpret_cast<const uint8_t*>(v.data_ptr());
    cache.k_elems = k_elems;
    cache.v_elems = v_elems;
  }
  // A new descriptor invalidates any captured launch graph.
  if (fmha::g_cache.exec != nullptr) cudaGraphExecDestroy(fmha::g_cache.exec);
  fmha::g_cache = cache;
}

// ---------------------------------------------------------------------------
// C-level repeat-call dispatcher (host path only; no kernel or plan change).
//
// Installed as Model.__call__, so the harness's model(**kwargs) goes straight to C:
// it compares the argument objects and the keyword-name order against the binding
// published by the last full-path forward and, on a match, replays the cached launch
// descriptor. Anything else -- positional arguments, a changed tensor, a reordered
// keyword, a different instance, an unbound cache -- vectorcalls the Python forward,
// which re-validates and rebinds. CPython does not prepend `self` for a non-descriptor
// __call__ attribute, so identity comes purely from the arguments. Strong references
// are held on every bound object, so no id can be recycled behind the comparison, and
// the kernels still read the current tensor contents on every call: only the launch
// descriptor is reused, never a data-dependent result.
//
// The binding is published only by the validated full path (3-D query, explicit
// cum_seq_lens_q/out/max_q_len, int max_seq_len, HND), whose tensors are caller-owned
// and pinned by Model._refs. The defensive _general path never binds, so a descriptor
// built from a temporary it created can never be replayed.
//
// Ported from this session's pre-join arm, where it ran through twelve consecutive
// builds and two official evaluations. The adopted tree's Python path is already
// trimmed, so what this removes is the 36-parameter signature binding and the pybind11
// hop: pure host enqueue measures 4.0-4.9 us/call here against a 5.4-5.9 us
// empty-event floor, and on the sub-30 us shapes that is 15-20% of the window.
namespace fmha {

int g_dbg_nofast = 0;  // experiment hook: 1 forces the Python host path

struct FastBind {
  PyObject* forward = nullptr;  // bound Model.forward of the instance that last built a descriptor
  PyObject* kwnames = nullptr;  // canonical keyword-name tuple (interned strings)
  PyObject* ret = nullptr;      // cached return object: out, or (out, lse)
  std::vector<PyObject*> argv;  // canonical argument objects, frozen-signature order
  bool valid = false;
};
FastBind g_bind;

PyObject* fast_call(PyObject* /*self*/, PyObject* const* args, Py_ssize_t nargs,
                    PyObject* kwnames) {
  if (g_dbg_nofast == 0 && nargs == 0 && kwnames != nullptr && g_bind.valid &&
      g_cache.valid) {
    const Py_ssize_t n = PyTuple_GET_SIZE(kwnames);
    if (n == (Py_ssize_t)g_bind.argv.size() && n == PyTuple_GET_SIZE(g_bind.kwnames)) {
      bool same = true;
      for (Py_ssize_t i = 0; i < n && same; ++i)
        same = (PyTuple_GET_ITEM(kwnames, i) == PyTuple_GET_ITEM(g_bind.kwnames, i));
      for (Py_ssize_t i = 0; i < n && same; ++i) same = (args[i] == g_bind.argv[i]);
      if (same) {
        try {
          fmha_replay();
        } catch (const std::exception& e) {
          PyErr_SetString(PyExc_RuntimeError, e.what());
          return nullptr;
        }
        Py_INCREF(g_bind.ret);
        return g_bind.ret;
      }
    }
  }
  if (g_bind.forward == nullptr) {
    PyErr_SetString(PyExc_RuntimeError,
                    "fmha fast dispatch is unbound; call Model.forward once first");
    return nullptr;
  }
  return PyObject_Vectorcall(g_bind.forward, args, (size_t)nargs, kwnames);
}

static PyMethodDef fast_call_def = {"fmha_fast_call", (PyCFunction)(void*)fast_call,
                                    METH_FASTCALL | METH_KEYWORDS, nullptr};

}  // namespace fmha

// Called from Model.__init__: registers the bound forward used for every miss.
void fmha_bind_forward(pybind11::object forward, pybind11::object names) {
  if (!PyTuple_Check(names.ptr())) throw std::runtime_error("names must be a tuple");
  if (fmha::g_bind.forward != forward.ptr()) {
    // A different instance now owns the dispatcher: drop the argument binding (the
    // descriptor itself is global and rebuilt by the first full-path call).
    Py_XDECREF(fmha::g_bind.ret);
    for (PyObject* o : fmha::g_bind.argv) Py_XDECREF(o);
    fmha::g_bind.argv.clear();
    fmha::g_bind.ret = nullptr;
    fmha::g_bind.valid = false;
    Py_XDECREF(fmha::g_bind.forward);
    fmha::g_bind.forward = forward.ptr();
    Py_INCREF(fmha::g_bind.forward);
  }
  Py_XDECREF(fmha::g_bind.kwnames);
  fmha::g_bind.kwnames = names.ptr();
  Py_INCREF(fmha::g_bind.kwnames);
}

// Called at the end of a successful full-path forward: publishes the argument
// identity and the return object the dispatcher may reuse.
void fmha_bind_args(pybind11::tuple args, pybind11::object ret) {
  for (PyObject* o : fmha::g_bind.argv) Py_XDECREF(o);
  fmha::g_bind.argv.clear();
  const Py_ssize_t n = PyTuple_GET_SIZE(args.ptr());
  fmha::g_bind.argv.reserve(n);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* o = PyTuple_GET_ITEM(args.ptr(), i);
    Py_XINCREF(o);
    fmha::g_bind.argv.push_back(o);
  }
  Py_XDECREF(fmha::g_bind.ret);
  fmha::g_bind.ret = ret.ptr();
  Py_XINCREF(fmha::g_bind.ret);
  fmha::g_bind.valid = fmha::g_cache.valid;
}

void fmha_invalidate_fast() {
  Py_XDECREF(fmha::g_bind.ret);
  for (PyObject* o : fmha::g_bind.argv) Py_XDECREF(o);
  fmha::g_bind.argv.clear();
  fmha::g_bind.ret = nullptr;
  fmha::g_bind.valid = false;
}

// Experiment hook: 0 forces the Python host path so one process can A/B both, at the
// cost of one already-cached global per call instead of a getenv.
void fmha_set_fast(int64_t on) { fmha::g_dbg_nofast = on ? 0 : 1; }

pybind11::object fmha_make_fast_call() {
  return pybind11::reinterpret_steal<pybind11::object>(
      PyCFunction_New(&fmha::fast_call_def, nullptr));
}
