// mla_expanded_bf16 — tcgen05 (SM103 Blackwell tensor core) kernel. Phase 2.
//
// FA4-style warp-specialized asynchronous pipeline, single CTA per SM
// (v3.7: 320 threads / 10 warps; TMA loaders over per-batch-8-padded
// canonical workspaces written by mla_tc_prep each call - probe-verified
// end-to-end by probe5, job dv_4fe1367b334e):
//   warps 0-7: softmax in COLUMN-SPLIT PAIRS (warp p + warp p+4 share rows
//              [32p,32p+32); cols 0-63 / 64-127; partial max/sum exchanged
//              via smem slots + named barriers) + O correction (half 0)
//   warp  9: elected-lane TMA loader (whole-tile bulk DMA + expect_tx)
//   warp  8: elected-lane tcgen05.mma issuer (FA4 deferred-PV order:
//            QK(0); i>=1: QK(i),PV(i-1); PV(T-1)).
//   TMEM (512 cols): S/P stages at 0-127 / 128-255 / 256-383 (P packed bf16
//   in-place over consumed S columns), O accumulator at 384-511.
//   SMEM: Q 48K (K-major SW128 canonical) + 2 K stages (48K) + 2 V stages
//   (32K, MN-major SW128) + ctrl 1152B = 214144B, 1 CTA/SM.
//
// Verified encodings (probe jobs dv_0fedf02d9e34, dv_59a8bba012b1 on L20D):
//   kind::f16 idesc M128 N128 bf16 f32acc = 0x08200490; PV B-transpose bit16.
//   smem desc: hi32 = 2<<29 | 1<<14 | SBO>>4; lo32 = (addr>>4)&0x3FFF |
//   (LBO>>4)<<16. Q/K K-major SW128: SBO=3072, per-K16 addr step
//   (kk/4)*1024+(kk%4)*32, LBO=0. V MN-major SW128: LBO=1024, SBO=2048,
//   per-K16 addr step 4096. tmem P A-operand: col kk*8 per K16 phase.
//   tmem addr = lane<<16 | col; warp lane base = (warp%4)*32<<16.
//   tcgen05.ld/st are .sync.aligned: NEVER execute them under lane-divergent
//   guards (hang evidence: job dv_59a8bba012b1).
//
// Mechanisms studied from Apache/BSD-licensed reference kernels (flashinfer
// cake_mamba bf16 sm_100a; atrex-fa cudnn_fe_oss d128 fprop; FA4 translation
// doc). All code independently written for this campaign. No libraries.
//
// Build: nvcc -gencode=arch=compute_103a,code=sm_103a.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define DIN __device__ __forceinline__

namespace mla_tc {

constexpr int DQK = 192;
constexpr int DV = 128;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int KV_STAGES = 2;
constexpr int S_STAGES = 3;
constexpr int THREADS = 320;   // v3.7: 8 softmax warps (col-split pairs) +
                               // issuer + TMA loader; cap 65536/320 = 204 regs
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -1.0e30f;   // "masked" score; finite so exp2 -> 0
constexpr float LAG_MAX_LOG2 = 8.0f;  // lagged-max threshold (p <= 2^8)

constexpr int Q_BYTES = BM * DQK * 2;              // 49152
constexpr int K_BYTES = BN * DQK * 2;              // 49152
constexpr int V_BYTES = BN * DV * 2;               // 32768
constexpr int CTRL_OFF = Q_BYTES + KV_STAGES * (K_BYTES + V_BYTES);
// ctrl: 11 mbarriers (88B) + tmem holder (8B) + pair-exchange slots
// (4 pairs x 2 halves x 32 lanes x 4B = 1024B) + slack
constexpr int SMEM_BYTES = CTRL_OFF + 1152;

// TMEM column map: 3 S/P stages + O
constexpr int TCOL_S0 = 0, TCOL_S1 = 128, TCOL_S2 = 256, TCOL_O = 384;

// ---------------------------------------------------------------------------
// PTX primitives
// ---------------------------------------------------------------------------
DIN uint32_t sm32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
DIN void cp_async16(uint32_t s, const void* g, int src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(src_size));
}
DIN void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
DIN void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

// TMA 5-D tile load, verified reference form (flashinfer cake_mamba sm_100a):
// .shared::cta + .mbarrier::complete_tx::bytes; producer pairs it with one
// mbarrier.arrive.expect_tx carrying the total tile bytes.
DIN void tma_load_5d(const CUtensorMap* map, uint32_t dst, uint32_t mbar,
                     int c0, int c1, int c2, int c3, int c4) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
      ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
        "r"(mbar)
      : "memory");
}
DIN void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               ::"r"(a), "r"(bytes) : "memory");
}
// TMA 4-D box load for the DIRECT path over raw K/V:
//   K viewed as [tokens][heads][3 atoms][64 elts] (V: 2 atoms), map dims
//   {64, tokens, atoms, heads}, box {64, 8, atoms, 1}: one 8-token group.
//   Smem write order dim0(128B swizzled rows) -> dim1(8 rows) -> dim2(atoms
//   at 1024B) == the canonical km_addr/mn_addr group layout proven for the
//   5-D workspace map, so UMMA descriptors are unchanged. OOB tokens
//   (>= total_kv) zero-fill; cross-batch tokens inside the tile are masked
//   by softmax exactly like the padded-workspace path.
DIN void tma_load_4d(const CUtensorMap* map, uint32_t dst, uint32_t mbar,
                     int c0, int c1, int c2, int c3) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];"
      ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar)
      : "memory");
}
// Fire-and-forget L2 prefetch of a future KV tile (no smem write, no
// completion mechanism): hides DRAM latency behind the 2-stage smem pipeline
// (KV_STAGES=2 puts TMA latency on the per-tile critical cycle; 3 stages at
// BN=128 do not fit: 288K > 227K smem cap).
DIN void tma_prefetch_5d(const CUtensorMap* map, int c0, int c1, int c2,
                         int c3, int c4) {
  asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global [%0, {%1, %2, %3, %4, %5}];"
               ::"l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4)
               : "memory");
}

DIN void mbar_init(uint32_t a, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
}
DIN void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
// Named CTA barrier over exactly `count` threads (warp-pair sync for the
// column-split softmax; bar.sync orders shared memory for participants).
DIN void named_bar(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
DIN void mbar_wait(uint32_t a, int phase) {
  // suspendTimeHint 0x989680: spin-vs-hint measured NEUTRAL in v3.7a (job
  // 20261019T205255Z, geomean 0.99, c4 0.91) -> keep the reference kernels'
  // hint (lower power, no issue-slot pressure). try_wait returns early when
  // the phase completes.
  asm volatile(
      "{\n\t.reg .pred P;\n\tLABW%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, %2;\n\t"
      "@P bra.uni DONEW%=;\n\tbra.uni LABW%=;\n\tDONEW%=:\n\t}\n"
      ::"r"(a), "r"(phase), "r"(0x989680)
      : "memory");
}
DIN int elect_one() {
  int pred = 0;
  asm volatile("{\n\t.reg .pred p;\n\telect.sync _|p, 0xFFFFFFFF;\n\t"
               "selp.s32 %0, 1, 0, p;\n\t}\n" : "+r"(pred));
  return pred;
}

DIN uint64_t make_desc(uint32_t smem_addr, uint32_t lbo_bytes,
                       uint32_t sbo_bytes) {
  uint32_t lo = ((smem_addr >> 4) & 0x3FFF) | ((lbo_bytes >> 4) << 16);
  uint32_t hi = (2u << 29) | (1u << 14) | ((sbo_bytes >> 4) & 0x3FFF);
  return (static_cast<uint64_t>(hi) << 32) | lo;
}
constexpr uint32_t IDESC_QK = (uint32_t(128 >> 4) << 24) | (uint32_t(128 >> 3) << 17) |
                              (1u << 10) | (1u << 7) | (1u << 4);   // 0x08200490
constexpr uint32_t IDESC_PV = IDESC_QK | (1u << 16);               // B transpose

DIN void mma_ss(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc, int accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::
          "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum));
}
DIN void mma_ts(uint32_t tmem_d, uint32_t tmem_a, uint64_t db, uint32_t idesc, int accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" ::
          "r"(tmem_d), "r"(tmem_a), "l"(db), "r"(idesc), "r"(accum));
}
DIN void mma_commit(uint32_t mbar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::
      "r"(mbar));
}
// batched tmem ld/st: issue all, then ONE wait (latency amortization)
DIN void tmem_ld32_nw(float* d, uint32_t ta) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32"
      " {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]),
        "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]),
        "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]),
        "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]),
        "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]),
        "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
      : "r"(ta));
}
DIN void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
DIN void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
DIN void tmem_st32(uint32_t ta, const float* d) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::
      "r"(ta), "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "f"(d[4]), "f"(d[5]),
      "f"(d[6]), "f"(d[7]), "f"(d[8]), "f"(d[9]), "f"(d[10]), "f"(d[11]),
      "f"(d[12]), "f"(d[13]), "f"(d[14]), "f"(d[15]), "f"(d[16]), "f"(d[17]),
      "f"(d[18]), "f"(d[19]), "f"(d[20]), "f"(d[21]), "f"(d[22]), "f"(d[23]),
      "f"(d[24]), "f"(d[25]), "f"(d[26]), "f"(d[27]), "f"(d[28]), "f"(d[29]),
      "f"(d[30]), "f"(d[31]) : "memory");
}
DIN void tmem_st16bf(uint32_t ta, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::
      "r"(ta), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]),
      "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]),
      "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]) : "memory");
}
DIN void fence_after_sync() { asm volatile("tcgen05.fence::after_thread_sync;"); }
DIN void fence_before_sync() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }

DIN uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// 2^x via the MUFU pipe (ex2.approx.f32): rel err ~2^-22 for x in [-126,128],
// 0 flush for x < -126; softmax args are <= 0. No __exp2f intrinsic exists in
// CUDA 13 (job dv_f956d37daac2), so the PTX instruction is used directly.
DIN float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// canonical UMMA smem addresses (byte offsets within tile region)
DIN uint32_t km_addr(int row, int kchunk16) {   // K-major SW128, [rows][192]
  int atom = kchunk16 >> 3;
  int c = kchunk16 & 7;
  return (row >> 3) * 3072 + atom * 1024 + (row & 7) * 128 + ((c ^ (row & 7)) << 4);
}
DIN uint32_t mn_addr(int krow, int nchunk16) {  // MN-major SW128, V [128][128]
  int atom = nchunk16 >> 3;
  int c = nchunk16 & 7;
  return (krow >> 3) * 2048 + atom * 1024 + (krow & 7) * 128 + ((c ^ (krow & 7)) << 4);
}

// barriers (8B each): BF0 BF1 BE0 BE1 SF0 SF1 SF2 PF0 PF1 PF2 OC
enum { BF0 = 0, BF1, BE0, BE1, SF0, SF1, SF2, PF0, PF1, PF2, OC, QF, NBARS };
DIN uint32_t bar(char* ctrl, int idx) { return sm32(ctrl) + idx * 8; }
DIN uint32_t s_col(int st) { return st == 0 ? TCOL_S0 : (st == 1 ? TCOL_S1 : TCOL_S2); }

// progress markers (dbg == nullptr in production -> elided)
DIN void mark(unsigned long long* dbg, int slot, unsigned long long val) {
  if (dbg) atomicMax(dbg + slot, val);
}
DIN void mark_wait(unsigned long long* dbg, int site, unsigned long long idx) {
  if (dbg) atomicMax(dbg + 10 + site, idx + 1);
}
// dbg_mode==7: per-stage cycle accounting (clock64 deltas summed in registers,
// flushed to dbg slots at kernel end). Slots:
//  16 softmax tiles, 17 S-ld, 18 max, 19 exp+Pst, 20 BE-corr wait, 21 corr,
//  22 softmax period (SF-wake to SF-wake)
//  24 issuer tiles, 25 BF wait, 26 BE(S-free) wait, 27 QK issue, 28 PF wait,
//  29 OC wait, 30 PV issue
//  32 loader tiles, 33 BE wait, 34 cp issue+wait0+arrive
DIN void flush_timing(unsigned long long* dbg, int base, const long long* sums, int n) {
  for (int i = 0; i < n; ++i) atomicAdd(dbg + base + i, (unsigned long long)sums[i]);
}

// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(THREADS, 1)
mla_tc_kernel(const __grid_constant__ CUtensorMap kmap,
              const __grid_constant__ CUtensorMap vmap,
              const __grid_constant__ CUtensorMap krmap,
              const __grid_constant__ CUtensorMap vrmap,
              const __nv_bfloat16* __restrict__ qraw,
              __nv_bfloat16* __restrict__ out,
              float* __restrict__ lse,
              const float* __restrict__ sinks,
              const int32_t* __restrict__ cum_q,
              const int32_t* __restrict__ cum_kv,
              int32_t total_q, int32_t batch, int32_t heads, int32_t kv_heads,
              int32_t window_left, int32_t causal, int32_t has_lse,
              int32_t has_sink, float bmm1_scale, float bmm2_scale,
              int32_t pf_mode, int32_t lag_max, int32_t direct,
              int32_t box_big,
              unsigned long long* __restrict__ dbg, int32_t dbg_mode) {
  extern __shared__ char smem[];
  char* sq = smem;
  char* sk = smem + Q_BYTES;
  char* sv = sk + KV_STAGES * K_BYTES;
  char* ctrl = smem + CTRL_OFF;
  uint32_t* tholder = reinterpret_cast<uint32_t*>(ctrl + NBARS * 8);
  // pair-exchange slots for the column-split softmax: [pair][half][32 lanes]
  float* exch = reinterpret_cast<float*>(ctrl + NBARS * 8 + 8);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  // Roles (v3.7, 320 threads / 10 warps): softmax warps 0-7 COLUMN-SPLIT:
  // pair (w, w+4) owns S rows [32*(w&3), +32); warp w takes cols [0,64),
  // warp w+4 takes cols [64,128). Partial max/sum exchange via 1024B smem
  // slots + named barriers (ids 1-4 max, 5-8 sum). Warp 8 = elected-lane
  // tcgen05 issuer; warp 9 = elected-lane TMA loader. ptxas cap 204 regs:
  // sv[2][32] single-pass softmax (~150 live) fits WITHOUT setmaxnreg
  // (v3.1 spill lesson: 384 threads @ 168-reg cap spilled 4200B/thread).
  const bool is_softmax = warp < 8;
  const bool is_loader = warp == 9;
  const bool is_issuer = warp == 8;
  const int tile_id = blockIdx.x;
  const int h = blockIdx.y;

  // ---- CTA mapping ----
  int b = -1, m0 = 0, q_base = 0, kv_base = 0, qlen = 0, kvlen = 0;
  int kv_pad = 0;   // 8-aligned padded KV workspace base of this batch
  {
    int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
    int kvpad = 0;
    for (int i = 0; i < batch; ++i) {
      int nq = cum_q[i + 1], nkv = cum_kv[i + 1];
      int len_q = nq - prev_q;
      int len_kv = nkv - prev_kv;
      int tiles = (len_q + BM - 1) / BM;
      if (b < 0 && tile_id < acc + tiles) {
        b = i;
        // LPT order (as tc2 opts bit2): heaviest q-tile of the batch first
        // so long causal KV sweeps fill the first wave, not the tail.
        m0 = (tiles - 1 - (tile_id - acc)) * BM;
        q_base = prev_q; kv_base = prev_kv;
        qlen = len_q; kvlen = len_kv;
        kv_pad = kvpad;
        break;
      }
      acc += tiles; prev_q = nq; prev_kv = nkv;
      kvpad += (len_kv + 7) & ~7;
    }
  }
  if (b < 0) return;

  const int kv_off = kvlen - qlen;
  const int n_max0 = causal ? min(kvlen, kv_off + m0 + BM) : kvlen;
  const int n_max = min(n_max0, kvlen);
  const bool windowed = window_left >= 0;
  int n_safe = causal ? min(kvlen, kv_off + m0 + 1) : kvlen;
  if (windowed) n_safe = 0;
  const int T = (n_max + BN - 1) / BN;
  const int T_full = n_safe / BN;
  if (T <= 0) return;

  const int group_div = heads / kv_heads;
  const int h_kv = h / group_div;
  __nv_bfloat16* outp = out + (static_cast<int64_t>(q_base) * heads + h) * DV;
  float* lsep = has_lse ? lse + static_cast<int64_t>(q_base) * heads + h : nullptr;
  const int64_t o_stride = static_cast<int64_t>(heads) * DV;
  const float sc2 = bmm1_scale * LOG2E;
  const float sink2 = has_sink ? sinks[h] * LOG2E : 0.0f;

  mark(dbg, 7, 1ull);
  if (dbg && tid == 0) atomicAdd(dbg + 8, 1ull);

  // ---- control init ----
  if (tid == 0) {
    mbar_init(bar(ctrl, BF0), 1);   // TMA: elected arrive.expect_tx + bytes
    mbar_init(bar(ctrl, BF1), 1);
    mbar_init(bar(ctrl, BE0), 1);
    mbar_init(bar(ctrl, BE1), 1);
    mbar_init(bar(ctrl, SF0), 1);
    mbar_init(bar(ctrl, SF1), 1);
    mbar_init(bar(ctrl, SF2), 1);
    mbar_init(bar(ctrl, PF0), 256);   // all 8 softmax warps arrive
    mbar_init(bar(ctrl, PF1), 256);
    mbar_init(bar(ctrl, PF2), 256);
    mbar_init(bar(ctrl, OC), 256);    // correction: all 8 softmax warps
    mbar_init(bar(ctrl, QF), THREADS); // Q cp.async done (all warps)
  }
  asm volatile("fence.mbarrier_init.release.cluster;");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (is_issuer) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 ::"r"(sm32(tholder)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  fence_after_sync();
  const uint32_t tmem = *(volatile uint32_t*)tholder;  // HW-written by alloc

  // ---- K/V TMA prologue (warp-9 elected lane; issued first so the DRAM
  // fetch overlaps the Q cp.async below). TMA box writes the canonical SW128
  // layout: 8-token groups at 3072B (SBO), atom planes at 1024B, rows 128B
  // hw-swizzled (= km_addr/mn_addr); OOB tokens zero-filled by the map.
  const int kblk0 = kv_pad >> 3;
  // DIRECT mode: tile j covers raw tokens [kv_base + j*BN, +BN) as 16
  // 8-token TMA boxes over the raw K/V maps (no prep workspace). kv_base is
  // this batch's offset in the flat [total_kv, heads, dim] tensors.
  const int tok0 = kv_base;
  if (is_loader && elect_one()) {
    if (direct) {
      // box_big: ONE {64,128,atoms,1} request per operand (48KB K + 32KB V);
      // rows land contiguously (atom planes at 16384B) and the issuer descs
      // switch SBO/plane strides to match (campaign job5b: n128 QK MMAs run
      // at 8192 FLOP/cyc vs n64's 5394 - tc1's BN=128 tile has a 23% lower
      // MMA floor than tc2's dual tile, and the loader measured ~3x feed
      // slack, so the big-box feed lifts tc1's historical box8 wall).
      mbar_expect_tx(bar(ctrl, BF0), K_BYTES + V_BYTES);
      if (box_big) {
        tma_load_4d(&krmap, sm32(sk), bar(ctrl, BF0), 0, tok0, 0, h_kv);
        tma_load_4d(&vrmap, sm32(sv), bar(ctrl, BF0), 0, tok0, 0, h_kv);
      } else {
      #pragma unroll
      for (int g = 0; g < 16; ++g) {
        tma_load_4d(&krmap, sm32(sk) + g * 3072, bar(ctrl, BF0),
                    0, tok0 + g * 8, 0, h_kv);
        tma_load_4d(&vrmap, sm32(sv) + g * 2048, bar(ctrl, BF0),
                    0, tok0 + g * 8, 0, h_kv);
      }
      }
      mark(dbg, 0, 1ull);
      if (T > 1) {
        mbar_expect_tx(bar(ctrl, BF1), K_BYTES + V_BYTES);
        if (box_big) {
          tma_load_4d(&krmap, sm32(sk + K_BYTES), bar(ctrl, BF1),
                      0, tok0 + 128, 0, h_kv);
          tma_load_4d(&vrmap, sm32(sv + V_BYTES), bar(ctrl, BF1),
                      0, tok0 + 128, 0, h_kv);
        } else {
        #pragma unroll
        for (int g = 0; g < 16; ++g) {
          tma_load_4d(&krmap, sm32(sk + K_BYTES) + g * 3072, bar(ctrl, BF1),
                      0, tok0 + 128 + g * 8, 0, h_kv);
          tma_load_4d(&vrmap, sm32(sv + V_BYTES) + g * 2048, bar(ctrl, BF1),
                      0, tok0 + 128 + g * 8, 0, h_kv);
        }
        }
        mark(dbg, 0, 2ull);
      }
    } else {
      mbar_expect_tx(bar(ctrl, BF0), K_BYTES + V_BYTES);
      tma_load_5d(&kmap, sm32(sk), bar(ctrl, BF0), 0, 0, 0, h_kv, kblk0);
      tma_load_5d(&vmap, sm32(sv), bar(ctrl, BF0), 0, 0, 0, h_kv, kblk0);
      mark(dbg, 0, 1ull);
      if (T > 1) {
        mbar_expect_tx(bar(ctrl, BF1), K_BYTES + V_BYTES);
        tma_load_5d(&kmap, sm32(sk + K_BYTES), bar(ctrl, BF1), 0, 0, 0, h_kv, kblk0 + 16);
        tma_load_5d(&vmap, sm32(sv + V_BYTES), bar(ctrl, BF1), 0, 0, 0, h_kv, kblk0 + 16);
        mark(dbg, 0, 2ull);
      }
      if (T > 2 && pf_mode == 1) {   // warm L2 for the first stage-limited load
        tma_prefetch_5d(&kmap, 0, 0, 0, h_kv, kblk0 + 32);
        tma_prefetch_5d(&vmap, 0, 0, 0, h_kv, kblk0 + 32);
      }
    }
  }

  // ---- Q direct load (ALL 320 threads): raw -> canonical smem ----
  // v4.1/v4.3: Q bypasses the prep workspace (prep traffic -33%). Softmax
  // warps help load Q: they are idle until SF0, which is gated on QK(0),
  // which needs QF anyway -> free parallelism (64 threads cost ~2-4us/CTA
  // serial Q latency = +26us on c5 main; 320 threads ~1us). OOB rows are
  // zero-filled via src-size predication (v3.3-proven, jobs dv_a2d8693fddcb
  // / dv_59a8bba012b1); QF + proxy fence publish smem to the UMMA async
  // proxy before the first QK.
  {
    const __nv_bfloat16* qp =
        qraw + (static_cast<int64_t>(q_base) * heads + h) * DQK;
    const int64_t q_stride = static_cast<int64_t>(heads) * DQK;
    #pragma unroll 4
    for (int idx = tid; idx < BM * 24; idx += THREADS) {
      int r = idx / 24, c = idx % 24;
      const __nv_bfloat16* src =
          qp + static_cast<int64_t>(m0 + r) * q_stride + c * 8;
      cp_async16(sm32(sq) + km_addr(r, c), src, (m0 + r < qlen) ? 16 : 0);
    }
    cp_commit();
    cp_wait0();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    mbar_arrive(bar(ctrl, QF));
  }

  if (is_loader) {
    // ---- steady-state K/V TMA loop (warp-9 elected lane) ----
    const bool lead = elect_one();
    if (lead) {
      const bool timed_l = (dbg != nullptr) && dbg_mode == 7;
      long long tsum[3] = {0, 0, 0};
      const uint32_t bf0 = bar(ctrl, BF0), bf1 = bar(ctrl, BF1);
      for (int j = 2; j < T; ++j) {
        int s = j & 1;
        mark_wait(dbg, 4, j);
        long long u0 = timed_l ? clock64() : 0;
        mbar_wait(bar(ctrl, s ? BE1 : BE0), ((j - 2) >> 1) & 1);
        long long u1 = timed_l ? clock64() : 0;
        const uint32_t bf = s ? bf1 : bf0;
        mbar_expect_tx(bf, K_BYTES + V_BYTES);
        if (direct) {
          const int tok = tok0 + j * 128;
          if (box_big) {
            tma_load_4d(&krmap, sm32(sk + s * K_BYTES), bf, 0, tok, 0, h_kv);
            tma_load_4d(&vrmap, sm32(sv + s * V_BYTES), bf, 0, tok, 0, h_kv);
          } else {
          #pragma unroll
          for (int g = 0; g < 16; ++g) {
            tma_load_4d(&krmap, sm32(sk + s * K_BYTES) + g * 3072, bf,
                        0, tok + g * 8, 0, h_kv);
            tma_load_4d(&vrmap, sm32(sv + s * V_BYTES) + g * 2048, bf,
                        0, tok + g * 8, 0, h_kv);
          }
          }
        } else {
          tma_load_5d(&kmap, sm32(sk + s * K_BYTES), bf, 0, 0, 0, h_kv, kblk0 + j * 16);
          tma_load_5d(&vmap, sm32(sv + s * V_BYTES), bf, 0, 0, 0, h_kv, kblk0 + j * 16);
          if (pf_mode == 1 && j + 2 < T) {   // L2 prefetch two tiles ahead
            tma_prefetch_5d(&kmap, 0, 0, 0, h_kv, kblk0 + (j + 2) * 16);
            tma_prefetch_5d(&vmap, 0, 0, 0, h_kv, kblk0 + (j + 2) * 16);
          }
        }
        mark(dbg, 0, (unsigned long long)(j + 1));
        if (timed_l) {
          long long u2 = clock64();
          tsum[0] += 1; tsum[1] += u1 - u0; tsum[2] += u2 - u1;
        }
      }
      if (timed_l) flush_timing(dbg, 32, tsum, 3);
    }
  } else if (is_issuer) {
    // ---------------- MMA ISSUER (warp 8 elected lane) ----------------
    if (elect_one()) {
      mbar_wait(bar(ctrl, QF), 0);   // Q smem published by warps 8-9
      fence_after_sync();
      const uint32_t qbase = sm32(sq);
      const bool timed = (dbg != nullptr) && dbg_mode == 7;
      long long tsum[7] = {0, 0, 0, 0, 0, 0, 0};
      long long t0, t1, t2, t3, t4, t5, t6, t7;
      auto issue_pv = [&](int i) {
        const int s = i & 1;
        const int st = i % S_STAGES;
        const uint32_t tmem_s = tmem + s_col(st);
        if (timed) t4 = clock64();
        mark_wait(dbg, 2, i);
        mbar_wait(bar(ctrl, PF0 + st), (i / S_STAGES) & 1);
        if (timed) t5 = clock64();
        mark_wait(dbg, 3, i);
        mbar_wait(bar(ctrl, OC), i & 1);
        if (timed) t6 = clock64();
        fence_after_sync();
        const uint32_t vbase = sm32(sv + s * V_BYTES);
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          // box8 canonical V (MN-major SW128, = mn_addr): N-atom stride 1024B
          // (LBO), 8-token core-matrix stride 2048B (SBO), K16 step 4096B.
          // box_big: rows contiguous -> atom plane (dv half) at 16384B (LBO),
          // core-matrix stride 1024B (SBO), K16 step 2048B (16 rows x 128B).
          uint64_t db = box_big ? make_desc(vbase + kk * 2048, 16384, 1024)
                                : make_desc(vbase + kk * 4096, 1024, 2048);
          // First k-phase of tile 0 overwrites O; everything else accumulates.
          mma_ts(tmem + TCOL_O, tmem_s + kk * 8, db, IDESC_PV, (i > 0) | (kk > 0));
        }
        mma_commit(bar(ctrl, s ? BE1 : BE0));
        mark(dbg, 2, (unsigned long long)(i + 1));
        if (timed) {
          t7 = clock64();
          tsum[4] += t5 - t4;   // PF wait
          tsum[5] += t6 - t5;   // OC wait
          tsum[6] += t7 - t6;   // PV issue+commit
        }
      };
      for (int i = 0; i < T; ++i) {
        const int s = i & 1;
        const int st = i % S_STAGES;
        if (timed) t0 = clock64();
        mark_wait(dbg, 0, i);
        mbar_wait(bar(ctrl, s ? BF1 : BF0), (i >> 1) & 1);
        if (timed) t1 = clock64();
        if (i >= S_STAGES) {
          mark_wait(dbg, 1, i);
          mbar_wait(bar(ctrl, ((i - S_STAGES) & 1) ? BE1 : BE0),
                    ((i - S_STAGES) >> 1) & 1);   // S stage free (PV_{i-3} done)
        }
        if (timed) t2 = clock64();
        fence_after_sync();
        const uint32_t kbase = sm32(sk + s * K_BYTES);
        const uint32_t tmem_s = tmem + s_col(st);
        #pragma unroll
        for (int kk = 0; kk < 12; ++kk) {
          // Q always canonical cp.async km_addr layout: atom plane 1024B,
          // K16 phase 32B, SBO = 8-row group 3072B.
          uint32_t off = (kk >> 2) * 1024 + (kk & 3) * 32;
          uint64_t da = make_desc(qbase + off, 0, 3072);
          // K: box8 canonical shares Q's geometry; box_big writes rows
          // contiguously -> atom plane 16384B, SBO 1024B.
          uint64_t db = box_big
              ? make_desc(kbase + (kk >> 2) * 16384 + (kk & 3) * 32, 0, 1024)
              : make_desc(kbase + off, 0, 3072);
          mma_ss(tmem_s, da, db, IDESC_QK, kk > 0);
        }
        mma_commit(bar(ctrl, SF0 + st));
        mark(dbg, 1, (unsigned long long)(i + 1));
        if (timed) {
          t3 = clock64();
          tsum[1] += t1 - t0;   // BF wait
          tsum[2] += t2 - t1;   // BE(S-free) wait
          tsum[3] += t3 - t2;   // QK issue+commit
          tsum[0] += 1;
        }
        if (i > 0) issue_pv(i - 1);
      }
      issue_pv(T - 1);
      if (timed) flush_timing(dbg, 24, tsum, 7);
    }
  } else if (is_softmax) {
    // -------- SOFTMAX + CORRECTION (warps 0-7, 256 threads, v3.7) --------
    // Column-split pairs: warp p in 0..3 (half 0, S cols 0-63) partners with
    // warp p+4 (half 1, cols 64-127) on rows [32p, 32p+32). Halves exchange
    // partial row max/sum through per-pair smem slots under named barriers
    // (ids 1-4 max, 5-8 sum; 64 threads each). mr/d/alpha/lse2 replicate
    // bitwise across a pair; O correction and the LSE store run on half 0
    // only (OC count stays 128). Per-warp serial softmax work halves
    // (exp_pst 2045 -> ~1.1K cyc/tile target; MUFU floor 1024/SM unchanged).
    const int lane = tid & 31;
    const int pair = warp & 3;
    const int half = warp >> 2;              // 0: cols 0-63, 1: cols 64-127
    const int row = pair * 32 + lane;        // S row owned by this thread
    const uint32_t lb = static_cast<uint32_t>(pair * 32) << 16;
    const int vis = kv_off + m0 + row;
    float mr = NEG_BIG, d = 0.0f;
    float sv[2][32];
    float* exch_self = exch + (pair * 2 + half) * 32 + lane;
    float* exch_part = exch + (pair * 2 + (half ^ 1)) * 32 + lane;
    const bool timed = (dbg != nullptr) && dbg_mode == 7;
    long long tsum[7] = {0, 0, 0, 0, 0, 0, 0};
    long long c0, c1, c2, c3, c3b, c4, cprev = 0;

    for (int i = 0; i < T; ++i) {
      const int st = i % S_STAGES;
      const uint32_t ts = tmem + s_col(st) + lb;
      mark_wait(dbg, 5, i);
      mbar_wait(bar(ctrl, SF0 + st), (i / S_STAGES) & 1);
      fence_after_sync();
      if (timed && tid == 0) {
        c0 = clock64();
        if (i > 0) tsum[6] += c0 - cprev;
        cprev = c0;
      }
      const int n0 = i * BN + half * 64;     // global col base of this half
      const bool masked = i >= T_full;

      // load this half's 64 S columns (global 32-col chunks half*2, half*2+1)
      #pragma unroll
      for (int cc = 0; cc < 2; ++cc)
        tmem_ld32_nw(sv[cc], ts + (half * 2 + cc) * 32);
      tmem_wait_ld();
      if (timed && tid == 0) { c1 = clock64(); tsum[1] += c1 - c0; }

      // partial row max over this half's 64 cols (unmasked fast path skips
      // all per-element mask ops; masked path uses precomputed cutoffs)
      float tm = NEG_BIG;
      if (!masked) {
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          float pm = sv[cc][0];
          #pragma unroll
          for (int x = 1; x < 32; ++x) pm = fmaxf(pm, sv[cc][x]);
          tm = fmaxf(tm, pm);
        }
      } else {
        const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
        const int jlo = windowed ? max(0, vis - window_left) : 0;
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          float pm = NEG_BIG;
          #pragma unroll
          for (int x = 0; x < 32; ++x) {
            int j = n0 + cc * 32 + x;
            pm = fmaxf(pm, (j >= jlo && j < jhi) ? sv[cc][x] : NEG_BIG);
          }
          tm = fmaxf(tm, pm);
        }
      }
      // pair exchange: partial maxima (barriers order the shared slots)
      *exch_self = tm;
      named_bar(1 + pair, 64);
      tm = fmaxf(tm, *exch_part);
      named_bar(5 + pair, 64);   // reads done before slots reused for sums
      if (timed && tid == 0) { c2 = clock64(); tsum[2] += c2 - c1; }
      // Lagged running max (lag_max): adopt the new tile max only when it
      // exceeds the running max by more than LAG_MAX_LOG2 in the scaled log2
      // domain. Then p = exp2(s2 - m2) <= 2^LAG_MAX_LOG2 (bf16-safe) while
      // alpha == 1.0f exactly for almost every tile, so the O correction
      // (2x64B TMEM ld/st per row) is skipped. Softmax and LSE are shift
      // invariant: out = O/d and lse = m2 + log2(d) are mathematically
      // unchanged; only fp rounding differs (mechanism as in the studied
      // cudnn_fe_oss reference: OK_FA_RESCALE_THRESHOLD_LOG2).
      float nr = fmaxf(mr, tm);
      if (lag_max != 0 && (nr - mr) * sc2 <= LAG_MAX_LOG2) nr = mr;
      const float nm2 = nr * sc2;
      const float alpha = (nr == mr) ? 1.0f : ex2_approx(fmaf(mr, sc2, -nm2));
      mr = nr;

      // P = exp2(s*sc2 - m2) over this half: exp, partial sum, pack, store
      // (P col base (half*2+cc)*16 matches the PV A-operand cols kk*8)
      float pc = 0.0f;
      if (!masked) {
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          uint32_t pr[16];
          #pragma unroll
          for (int x2 = 0; x2 < 16; ++x2) {
            float p0 = ex2_approx(fmaf(sv[cc][x2 * 2], sc2, -nm2));
            float p1 = ex2_approx(fmaf(sv[cc][x2 * 2 + 1], sc2, -nm2));
            pc += p0 + p1;
            pr[x2] = pack_bf16(p0, p1);
          }
          tmem_st16bf(ts + (half * 2 + cc) * 16, pr);
        }
      } else {
        const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
        const int jlo = windowed ? max(0, vis - window_left) : 0;
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          uint32_t pr[16];
          #pragma unroll
          for (int x2 = 0; x2 < 16; ++x2) {
            float p0, p1;
            {
              int j = n0 + cc * 32 + x2 * 2;
              p0 = (j >= jlo && j < jhi)
                       ? ex2_approx(fmaf(sv[cc][x2 * 2], sc2, -nm2)) : 0.0f;
            }
            {
              int j = n0 + cc * 32 + x2 * 2 + 1;
              p1 = (j >= jlo && j < jhi)
                       ? ex2_approx(fmaf(sv[cc][x2 * 2 + 1], sc2, -nm2)) : 0.0f;
            }
            pc += p0 + p1;
            pr[x2] = pack_bf16(p0, p1);
          }
          tmem_st16bf(ts + (half * 2 + cc) * 16, pr);
        }
      }
      tmem_wait_st();
      // v3.7b: PF arrives as soon as P is stored (PV only needs P; the
      // row-sum exchange now overlaps the issuer's PV wait/issue path).
      fence_before_sync();
      mbar_arrive(bar(ctrl, PF0 + st));      // count 256 = 8 softmax warps
      // pair exchange: partial row sums
      *exch_self = pc;
      named_bar(1 + pair, 64);
      const float rs = pc + *exch_part;
      named_bar(5 + pair, 64);   // reads done before next tile's max write
      d = d * alpha + rs;
      if (tid == 0) mark(dbg, 3, (unsigned long long)(i + 1));
      if (timed && tid == 0) { c3 = clock64(); tsum[3] += c3 - c2; }

      // O correction (skip tile 0; skip when alpha==1 warp-wide). v3.7b:
      // SPLIT across the pair - each half scales its own 64 O cols in one
      // round (halves per-warp correction latency; disjoint tmem cols).
      // All 8 warps pass the BE gate so parities stay aligned.
      if (i > 0) {
        mark_wait(dbg, 6, i);
        mbar_wait(bar(ctrl, ((i - 1) & 1) ? BE1 : BE0), ((i - 1) >> 1) & 1);
        fence_after_sync();
        if (timed && tid == 0) { c3b = clock64(); tsum[4] += c3b - c3; }
        if (__any_sync(0xffffffffu, alpha != 1.0f)) {
          const uint32_t to = tmem + TCOL_O + lb;
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc)
            tmem_ld32_nw(sv[cc], to + (half * 2 + cc) * 32);
          tmem_wait_ld();
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            #pragma unroll
            for (int x = 0; x < 32; ++x) sv[cc][x] *= alpha;
            tmem_st32(to + (half * 2 + cc) * 32, sv[cc]);
          }
          tmem_wait_st();
        }
      }
      fence_before_sync();
      mbar_arrive(bar(ctrl, OC));            // count 256 = 8 softmax warps
      if (tid == 0) mark(dbg, 4, (unsigned long long)(i + 1));
      if (timed && tid == 0) {
        c4 = clock64();
        if (i > 0) tsum[5] += c4 - c3b; else tsum[5] += c4 - c3;
        tsum[0] += 1;
      }
    }
    if (timed && tid == 0) flush_timing(dbg, 16, tsum, 7);

    // ---- epilogue: each half stores its own 64 O cols; LSE half 0 only ----
    mark_wait(dbg, 7, T - 1);
    mbar_wait(bar(ctrl, ((T - 1) & 1) ? BE1 : BE0), ((T - 1) >> 1) & 1);
    fence_after_sync();
    float inv = d > 0.0f ? 1.0f / d : 0.0f;
    float lse2 = mr * sc2 + log2f(d);
    float extra = 1.0f;
    if (has_sink) {
      float t = fmaxf(lse2, sink2) + log2f(1.0f + exp2f(-fabsf(lse2 - sink2)));
      extra = exp2f(lse2 - t);
      lse2 = t;
    }
    const float g = bmm2_scale * inv * extra;
    const bool active = (m0 + row < qlen);
    // tcgen05.ld is .sync.aligned: ALL lanes must execute; stores predicated.
    {
      const uint32_t to = tmem + TCOL_O + lb;
      #pragma unroll
      for (int oc = 0; oc < 2; ++oc)
        tmem_ld32_nw(sv[oc], to + (half * 2 + oc) * 32);
      tmem_wait_ld();
      __nv_bfloat16* orow = outp + static_cast<int64_t>(m0 + row) * o_stride;
      for (int oc = 0; oc < 2; ++oc) {
        if (active) {
          __align__(16) uint32_t pk[16];
          #pragma unroll
          for (int x2 = 0; x2 < 16; ++x2) {
            pk[x2] = pack_bf16(sv[oc][x2 * 2] * g, sv[oc][x2 * 2 + 1] * g);
          }
          uint4* dst =
              reinterpret_cast<uint4*>(orow + (half * 2 + oc) * 32);
          const uint4* src4 = reinterpret_cast<const uint4*>(pk);
          dst[0] = src4[0]; dst[1] = src4[1]; dst[2] = src4[2]; dst[3] = src4[3];
        }
      }
      if (active && has_lse && half == 0) {
        lsep[static_cast<int64_t>(m0 + row) * heads] = lse2;
      }
    }
  }

  if (tid == 0) mark(dbg, 5, 1ull);
  if (dbg && tid == 0) atomicAdd(dbg + 5, 1ull);
  __syncthreads();
  if (is_issuer) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 ::"r"(tmem), "r"(512) : "memory");
  }
  if (dbg && tid == 0) atomicAdd(dbg + 6, 1ull);
}

}  // namespace mla_tc

// ------------------------------------------------------------------------// Prep kernel: reformat K/V into per-batch-8-padded canonical workspaces
// (UNSWIZZLED canonical: [block][head][atom][8 rows][128B]; TMA applies the
// SW128 swizzle when writing smem - verified by probe5, job dv_4fe1367b334e).
// v4.1: Q is NOT prepped (CTAs cp.async their own Q tile raw->canonical).
// Item = (padded block, head, {K,V}). Warp per item. Trailing declared
// blocks and per-batch pad rows are zeroed (V must stay finite for P=0).
// ---------------------------------------------------------------------------
namespace mla_tc {
__global__ void __launch_bounds__(256, 1)
mla_tc_prep(const __nv_bfloat16* __restrict__ k,
            const __nv_bfloat16* __restrict__ v,
            char* __restrict__ kws, char* __restrict__ vws,
            const int32_t* __restrict__ cum_kv,
            int batch, int kv_heads, int kvblks) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const long long n_items = (long long)kvblks * kv_heads * 2;
  const long long gstride = (long long)gridDim.x * (blockDim.x >> 5);
  for (long long item = (long long)blockIdx.x * (blockDim.x >> 5) + warp;
       item < n_items; item += gstride) {
    const bool is_v = (item & 1) != 0;
    long long t2 = item >> 1;
    int h = static_cast<int>(t2 % kv_heads);
    int blk = static_cast<int>(t2 / kv_heads);
    const int rows = is_v ? DV : DQK;
    const int cpr = rows / 8;             // 16B chunks per row: 24 (192) / 16 (128)
    const int atoms = cpr / 8;            // 3 (K) or 2 (V)
    const long long bhb = (long long)atoms * 1024;   // block-head bytes
    char* ws = (is_v ? vws : kws) +
               (static_cast<long long>(blk) * kv_heads + h) * bhb;
    // locate batch for this padded block; trailing blocks -> zero
    int tok_base = 0, valid = 0, blk_local = 0, acc = 0, prev = cum_kv[0];
    bool found = false;
    for (int i = 0; i < batch; ++i) {
      int nxt = cum_kv[i + 1];
      int l = nxt - prev;
      int bl = (l + 7) >> 3;
      if (blk < acc + bl) {
        tok_base = prev; blk_local = blk - acc; valid = l - blk_local * 8;
        found = true;
        break;
      }
      acc += bl; prev = nxt;
    }
    if (!found || valid < 0) valid = 0;
    const __nv_bfloat16* src = is_v ? v : k;
    const int64_t tok_stride = static_cast<int64_t>(kv_heads) * rows;
    const __nv_bfloat16* src0 =
        src + static_cast<int64_t>(tok_base + blk_local * 8) * tok_stride +
        static_cast<int64_t>(h) * rows;
    for (int ci = lane; ci < 8 * cpr; ci += 32) {
      int tok = ci / cpr, c = ci % cpr;
      uint4 val = make_uint4(0u, 0u, 0u, 0u);
      if (tok < valid) {
        val = *reinterpret_cast<const uint4*>(
            src0 + static_cast<int64_t>(tok) * tok_stride + c * 8);
      }
      uint32_t dst_off = static_cast<uint32_t>((c >> 3) * 1024 + tok * 128 +
                                               (c & 7) * 16);
      *reinterpret_cast<uint4*>(ws + dst_off) = val;
    }
  }
}
}  // namespace mla_tc

// ---------------------------------------------------------------------------
// Host: rank-5 tensor maps over the padded workspaces + prep + launch.
// ---------------------------------------------------------------------------
namespace {
struct MapKey5 {
  const void* ptr; uint64_t atoms, heads, blocks;
  bool operator==(const MapKey5& o) const {
    return ptr == o.ptr && atoms == o.atoms && heads == o.heads && blocks == o.blocks;
  }
};
struct MapEntry5 { MapKey5 key; CUtensorMap map; bool valid = false; };
MapEntry5 g_cache5[8];
int g_next5 = 0;

// Direct raw-K/V maps: 4-D {64 elts, tokens, atoms, heads}; strides for
// dims1..3 (bytes): token = heads*atoms*128, atom = 128, head = atoms*128.
// Box {64, 8, atoms, 1} = one canonical 8-token SW128 group (identical smem
// image to the workspace 5-D map, so km_addr/mn_addr and the UMMA
// descriptors are unchanged). Strides are 16B-multiples; non-monotonic
// strides (token > atom) are legal for cuTensorMapEncodeTiled.
struct MapKey4 {
  const void* ptr; uint64_t tokens, atoms, heads; int box_rows;
  bool operator==(const MapKey4& o) const {
    return ptr == o.ptr && tokens == o.tokens && atoms == o.atoms &&
           heads == o.heads && box_rows == o.box_rows;
  }
};
struct MapEntry4 { MapKey4 key; CUtensorMap map; bool valid = false; };
MapEntry4 g_cache4[8];
int g_next4 = 0;

int encode4(CUtensorMap* m, const void* ptr, uint64_t tokens, uint64_t heads,
            uint64_t atoms, int box_rows) {
  cuuint64_t gdim[4] = {64, tokens, atoms, heads};
  cuuint64_t gstr[3] = {heads * atoms * 128, 128, atoms * 128};
  cuuint32_t box[4] = {64, static_cast<cuuint32_t>(box_rows),
                       static_cast<cuuint32_t>(atoms), 1};
  cuuint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(ptr),
      gdim, gstr, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return static_cast<int>(r);
}
int cached4(CUtensorMap* m, const void* ptr, uint64_t tokens, uint64_t heads,
            uint64_t atoms, int box_rows) {
  MapKey4 key{ptr, tokens, atoms, heads, box_rows};
  for (auto& e : g_cache4) {
    if (e.valid && e.key == key) { *m = e.map; return 0; }
  }
  int rc = encode4(m, ptr, tokens, heads, atoms, box_rows);
  if (rc == 0) {
    MapEntry4& e = g_cache4[g_next4];
    g_next4 = (g_next4 + 1) & 7;
    e.key = key; e.map = *m; e.valid = true;
  }
  return rc;
}

int encode5(CUtensorMap* m, const void* ptr, uint64_t atoms, uint64_t heads_dim,
            uint64_t blocks) {
  // dims {64 elts, 8 rows, atoms, heads, blocks}; strides (dims1..4, bytes)
  // {128, 1024, atoms*1024, heads*atoms*1024}; box {64, 8, atoms, 1, 16}.
  cuuint64_t gdim[5] = {64, 8, atoms, heads_dim, blocks};
  cuuint64_t gstr[4] = {128, 1024, atoms * 1024, heads_dim * atoms * 1024};
  cuuint32_t box[5] = {64, 8, static_cast<cuuint32_t>(atoms), 1, 16};
  cuuint32_t estr[5] = {1, 1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, const_cast<void*>(ptr),
      gdim, gstr, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return static_cast<int>(r);
}
int cached5(CUtensorMap* m, const void* ptr, uint64_t atoms, uint64_t heads_dim,
            uint64_t blocks) {
  MapKey5 key{ptr, atoms, heads_dim, blocks};
  for (auto& e : g_cache5) {
    if (e.valid && e.key == key) { *m = e.map; return 0; }
  }
  int rc = encode5(m, ptr, atoms, heads_dim, blocks);
  if (rc == 0) {
    MapEntry5& e = g_cache5[g_next5];
    g_next5 = (g_next5 + 1) & 7;
    e.key = key; e.map = *m; e.valid = true;
  }
  return rc;
}
}  // namespace

// workspace sizing helper for the python binding (single source of truth;
// v4.1: K/V only - Q loads raw from the CTA)
extern "C" void mla_tc_ws(int total_kv, int batch, int kv_heads,
                          long long* kvblks, long long* kbytes,
                          long long* vbytes) {
  long long kb = total_kv / 8 + 1 + batch;
  *kvblks = kb;
  *kbytes = kb * kv_heads * 3072;
  *vbytes = kb * kv_heads * 2048;
}

extern "C" int mla_tc_forward(const void* q, const void* k, const void* v,
                              void* out, void* lse, const void* sinks,
                              const void* cum_q, const void* cum_kv,
                              void* kws, void* vws,
                              int total_q, int total_kv, int batch, int heads,
                              int kv_heads, int kvblks,
                              int window_left, int causal, int has_lse,
                              int has_sink, float bmm1_scale, float bmm2_scale,
                              int max_kv_len, void* stream,
                              void* dbg, int dbg_mode,
                              int pf_mode, int lag_max, int direct,
                              int box_tok, int opts) {
  (void)max_kv_len; (void)opts;  // shared ABI with kernel_tc2
  // box_tok 128 -> one big-box TMA request per K/V operand per tile
  const int box_big = (direct && box_tok == 128) ? 1 : 0;
  using namespace mla_tc;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(mla_tc_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return static_cast<int>(e);
    attr_set = true;
  }
  int tiles_ub = total_q / BM + batch;
  if (tiles_ub <= 0) return static_cast<int>(cudaGetLastError());

  CUtensorMap kmap, vmap, krmap, vrmap;
  int rc;
  if (direct) {
    // Raw-K/V maps; workspace maps unused (zeroed; kernel ignores them).
    memset(&kmap, 0, sizeof(kmap));
    memset(&vmap, 0, sizeof(vmap));
    if ((rc = cached4(&krmap, k, (uint64_t)total_kv, (uint64_t)kv_heads, 3,
                      box_big ? 128 : 8)))
      return 5000 + rc;
    if ((rc = cached4(&vrmap, v, (uint64_t)total_kv, (uint64_t)kv_heads, 2,
                      box_big ? 128 : 8)))
      return 6000 + rc;
    if (dbg_mode == 8) return 0;   // no prep kernel in direct mode
  } else {
    memset(&krmap, 0, sizeof(krmap));
    memset(&vrmap, 0, sizeof(vrmap));
    if ((rc = cached5(&kmap, kws, 3, kv_heads, kvblks))) return 2000 + rc;
    if ((rc = cached5(&vmap, vws, 2, kv_heads, kvblks))) return 3000 + rc;

    long long items = (long long)kvblks * kv_heads * 2;
    int pgrid = static_cast<int>((items + 7) / 8);
    if (pgrid > 4096) pgrid = 4096;
    if (pgrid < 1) pgrid = 1;
    mla_tc_prep<<<pgrid, 256, 0, s>>>(
        static_cast<const __nv_bfloat16*>(k), static_cast<const __nv_bfloat16*>(v),
        static_cast<char*>(kws), static_cast<char*>(vws),
        static_cast<const int32_t*>(cum_kv),
        batch, kv_heads, kvblks);
    cudaError_t pe = cudaGetLastError();
    if (pe != cudaSuccess) return 4000 + static_cast<int>(pe);
    if (dbg_mode == 8) return 0;   // measurement-only: time prep in isolation
  }

  dim3 grid(tiles_ub, heads, 1);
  mla_tc_kernel<<<grid, THREADS, SMEM_BYTES, s>>>(
      kmap, vmap, krmap, vrmap, static_cast<const __nv_bfloat16*>(q),
      static_cast<__nv_bfloat16*>(out), static_cast<float*>(lse),
      static_cast<const float*>(sinks),
      static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
      total_q, batch, heads, kv_heads, window_left, causal, has_lse, has_sink,
      bmm1_scale, bmm2_scale, pf_mode, lag_max, direct, box_big,
      reinterpret_cast<unsigned long long*>(dbg), dbg_mode);
  return static_cast<int>(cudaGetLastError());
}
