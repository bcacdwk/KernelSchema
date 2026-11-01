// mla_expanded_bf16 — tcgen05 2-CTA kernel (tc3). Campaign phase 4.
//
// Same frozen contract as kernel_tc2.cu. Structure (design validated against
// the Apache/BSD-licensed atrex-fa pure_cuda d256 2-CTA reference mechanisms;
// all code independently written for this campaign, bf16 d192/d128):
//   * cluster (2,1,1); a pair covers BM=256 q-rows (128 per CTA) and BN=128
//     kv tokens per tile; tcgen05.mma cta_group::2 m256n128k16 issued by the
//     leader CTA only; D rows land per-CTA in own TMEM.
//   * B-operand split: QK B=K split by TOKENS (CTA r holds tokens 64r..64r+63
//     of the 128-token tile, 24KB); PV B=V split by DV (CTA r holds dv cols
//     64r.., all 128 tokens, 16KB). Per-SM KV feed = 40KB per 1280-cycle
//     tile = HALF of tc2's per-flop feed.
//   * Q via ONE TMA box64-style request per CTA ({64,128,3,1}, 48KB), OOB
//     rows zero-filled by the map.
//   * 4 KV stages (160KB) + Q (48KB) = 208KB smem; TMEM: S0/S1/S2 at cols
//     0/128/256 (3 stages), O at 384.
//   * 192 threads: warps 0-3 softmax (lane = own row, 128 cols, no exchange,
//     lagged max 2^8 => correction ~never), warp 4 MMA issuer (rank 0),
//     warp 5 TMA loader (both CTAs).
//   * Cross-CTA protocol (reference-verified forms): cta_group::2 TMA with
//     completion to the leader barrier (peer bit 24 cleared); leader arms
//     expect_tx for BOTH CTAs' bytes and publishes an ARMED token to both
//     CTAs (publish_both) before either issues; tcgen05.commit multicast
//     (mask 0x3) for SF/BEPV/BE; remote mbarrier.arrive (mapa) for
//     PF/OC/QF-class arrivals; cluster_sync + peer barrier around
//     cta_group::2 alloc/dealloc.
//
// Build: nvcc -gencode=arch=compute_103a,code=sm_103a -lcuda.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define DIN __device__ __forceinline__

namespace mla_tc3 {

constexpr int DQK = 192;
constexpr int DV = 128;
constexpr int BM_PAIR = 256;        // q rows per cluster tile
constexpr int ROWS = 128;           // q rows per CTA
constexpr int BN = 128;             // kv tokens per tile
constexpr int KV_STAGES = 4;
constexpr int S_STAGES = 3;
constexpr int THREADS = 192;        // 4 softmax + mma + loader
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -1.0e30f;
constexpr float LAG_MAX_LOG2 = 8.0f;

constexpr int Q_BYTES = ROWS * DQK * 2;         // 49152
constexpr int K_BYTES = 64 * DQK * 2;           // 24576 (own 64-token half)
constexpr int V_BYTES = BN * 64 * 2;            // 16384 (own 64-dv half)
constexpr int STAGE_BYTES = K_BYTES + V_BYTES;  // 40960
constexpr int CTRL_OFF = Q_BYTES + KV_STAGES * STAGE_BYTES;   // 212992
// ctrl: 25 barriers (200B) + tmem holder + slack
constexpr int SMEM_BYTES = CTRL_OFF + 256;

constexpr uint32_t PEER_MASK = 0xFEFFFFFFu;   // clear cluster rank bit 24

// barrier indices
enum { QF = 0, QARM,
       BF0, BF1, BF2, BF3,
       BA0, BA1, BA2, BA3,     // BARM (per-CTA local)
       BE0, BE1, BE2, BE3,     // per-CTA local (multicast commit)
       SF0, SF1, SF2,          // per-CTA local (multicast commit)
       PV0, PV1, PV2,          // BEPV per-CTA local (multicast commit)
       PF0, PF1, PF2,          // leader (softmax lane0 arrives, cnt 8)
       OC,                     // leader (cnt 8)
       PD,                     // peer dealloc (cnt 32)
       NBARS };

DIN uint32_t sm32_(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
DIN uint32_t bar(char* ctrl, int idx) { return sm32_(ctrl) + idx * 8; }

// ---------------------------------------------------------------------------
// PTX primitives (V0-proven forms + 2-CTA forms from the studied reference)
// ---------------------------------------------------------------------------
DIN void mbar_init(uint32_t a, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
}
DIN void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
               ::"r"(a) : "memory");
}
DIN void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               ::"r"(a), "r"(bytes) : "memory");
}
DIN void mbar_wait(uint32_t a, int phase) {
  // acquire.cluster: PF/OC/BARM-class barriers receive REMOTE arrivals from
  // the peer CTA (reference mbarrier_wait_acquire_cluster form); cluster
  // scope orders the peer's preceding writes for this waiter.
  asm volatile(
      "{\n\t.reg .pred P;\n\tLABW%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], %1, %2;\n\t"
      "@P bra.uni DONEW%=;\n\tbra.uni LABW%=;\n\tDONEW%=:\n\t}\n"
      ::"r"(a), "r"(phase), "r"(0x989680)
      : "memory");
}
DIN uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DIN uint32_t mapa0(uint32_t local_addr) {   // address of same offset in rank0
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(remote) : "r"(local_addr), "r"(0));
  return remote;
}
DIN void mbar_arrive_leader(uint32_t local_a, uint32_t rank) {
  if (rank == 0) {
    mbar_arrive(local_a);
  } else {
    uint32_t remote = mapa0(local_a);
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(remote)
                 : "memory");
  }
}
// leader arms expect_tx and releases both CTAs' loaders (reference
// publish_both): local arrive + remote arrive into rank1's copy.
DIN void publish_both(uint32_t local_a) {
  mbar_arrive(local_a);
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(remote) : "r"(local_a), "r"(1));
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(remote)
               : "memory");
}
DIN void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
DIN int elect_one() {
  int pred = 0;
  asm volatile("{\n\t.reg .pred p;\n\telect.sync _|p, 0xFFFFFFFF;\n\t"
               "selp.s32 %0, 1, 0, p;\n\t}\n" : "+r"(pred));
  return pred;
}
// cta_group::2 TMA: executed by BOTH CTAs; completion lands on CTA0's
// barrier (peer bit cleared), matching the reference ISA contract.
DIN void tma_load_4d_2cta(const CUtensorMap* map, uint32_t dst, uint32_t mbar,
                          int c0, int c1, int c2, int c3) {
  asm volatile(
      "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];"
      ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
        "r"(mbar & PEER_MASK)
      : "memory");
}
DIN uint64_t make_desc(uint32_t smem_addr, uint32_t lbo_bytes,
                       uint32_t sbo_bytes) {
  uint32_t lo = ((smem_addr >> 4) & 0x3FFF) | ((lbo_bytes >> 4) << 16);
  uint32_t hi = (2u << 29) | (1u << 14) | ((sbo_bytes >> 4) & 0x3FFF);
  return (static_cast<uint64_t>(hi) << 32) | lo;
}
// idesc f16 M256 N128 (M field (M>>5)<<25): QK 0x10200490; PV + transpose.
constexpr uint32_t IDESC_QK2 = (uint32_t(256 >> 5) << 25) |
                               (uint32_t(128 >> 3) << 17) |
                               (1u << 10) | (1u << 7) | (1u << 4);
constexpr uint32_t IDESC_PV2 = IDESC_QK2 | (1u << 16);
DIN void mma_ss_2cta(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc,
                     int accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::
          "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum));
}
DIN void mma_ts_2cta(uint32_t tmem_d, uint32_t tmem_a, uint64_t db,
                     uint32_t idesc, int accum) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" ::
          "r"(tmem_d), "r"(tmem_a), "l"(db), "r"(idesc), "r"(accum));
}
DIN void mma_commit_mc(uint32_t mbar) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one."
      "shared::cluster.multicast::cluster.b64 [%0], %1;" ::
          "r"(mbar), "h"(uint16_t(0x3))
      : "memory");
}
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
DIN float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
DIN void mark(unsigned long long* dbg, int slot, unsigned long long val) {
  if (dbg) atomicMax(dbg + slot, val);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS, 1)
mla_tc3_kernel(const __grid_constant__ CUtensorMap qmap,
               const __grid_constant__ CUtensorMap kmap,
               const __grid_constant__ CUtensorMap vmap,
               __nv_bfloat16* __restrict__ out,
               float* __restrict__ lse,
               const float* __restrict__ sinks,
               const int32_t* __restrict__ cum_q,
               const int32_t* __restrict__ cum_kv,
               int32_t total_q, int32_t batch, int32_t heads, int32_t kv_heads,
               int32_t window_left, int32_t causal, int32_t has_lse,
               int32_t has_sink, float bmm1_scale, float bmm2_scale,
               int32_t t3opt, unsigned long long* __restrict__ dbg) {
  extern __shared__ char smem[];
  char* sq = smem;                        // 49152B (box: rows contig, atoms 16384)
  char* skv = smem + Q_BYTES;             // 4 x (K 24576 + V 16384)
  char* ctrl = smem + CTRL_OFF;
  uint32_t* tholder = reinterpret_cast<uint32_t*>(ctrl + NBARS * 8);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const uint32_t rank = cluster_rank();
  const int tile_id = blockIdx.x >> 1;    // cluster pair id
  const int h = blockIdx.y;

  // ---- CTA mapping (pair = 256 q-rows; LPT: heaviest tile of batch first) --
  int b = -1, m0 = 0, q_base = 0, kv_base = 0, qlen = 0, kvlen = 0;
  {
    int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
    for (int i = 0; i < batch; ++i) {
      int nq = cum_q[i + 1], nkv = cum_kv[i + 1];
      int len_q = nq - prev_q;
      int tiles = (len_q + BM_PAIR - 1) / BM_PAIR;
      if (b < 0 && tile_id < acc + tiles) {
        b = i;
        m0 = (tiles - 1 - (tile_id - acc)) * BM_PAIR;
        q_base = prev_q; kv_base = prev_kv;
        qlen = len_q; kvlen = nkv - prev_kv;
        break;
      }
      acc += tiles; prev_q = nq; prev_kv = nkv;
    }
  }
  if (b < 0) return;   // surplus pair (uniform across the cluster)

  const int kv_off = kvlen - qlen;
  const bool windowed = window_left >= 0;
  const int n_max = min(causal ? min(kvlen, kv_off + m0 + BM_PAIR) : kvlen,
                        kvlen);
  const int T = (n_max + BN - 1) / BN;
  // per-rank unmasked-tile count (rows of this CTA)
  const int row0 = m0 + static_cast<int>(rank) * ROWS;    // first global row
  int n_safe = causal ? min(kvlen, kv_off + row0 + 1) : kvlen;
  if (windowed) n_safe = 0;
  const int Tfull = n_safe / BN;
  if (T <= 0) return;

  const int group_div = heads / kv_heads;
  const int h_kv = h / group_div;
  __nv_bfloat16* outp = out + (static_cast<int64_t>(q_base) * heads + h) * DV;
  float* lsep = has_lse ? lse + static_cast<int64_t>(q_base) * heads + h : nullptr;
  const int64_t o_stride = static_cast<int64_t>(heads) * DV;
  const float sc2 = bmm1_scale * LOG2E;
  const float sink2 = has_sink ? sinks[h] * LOG2E : 0.0f;
  const int tok0 = kv_base;
  const int qrow0 = q_base + row0;

  if (dbg && tid == 0) atomicAdd(dbg + 8, 1ull);

  // ---- control init ----
  if (tid == 0) {
    for (int i = 0; i < NBARS; ++i) {
      int cnt = 1;
      if (i >= PF0 && i <= PF2) cnt = 8;         // 4 warps x 2 CTAs (lane0)
      else if (i == OC) cnt = 8;
      else if (i == PD) cnt = 32;
      mbar_init(bar(ctrl, i), cnt);
    }
  }
  asm volatile("fence.mbarrier_init.release.cluster;");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  cluster_sync();
  if (warp == 4) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 ::"r"(sm32_(tholder)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  fence_after_sync();
  const uint32_t tmem = *(volatile uint32_t*)tholder;

  if (warp == 5) {
    // ---------------- TMA LOADER (both CTAs, elected lane) ----------------
    if (elect_one()) {
      // Q: leader arms for BOTH CTAs (96KB total), then both issue.
      if (rank == 0) {
        mbar_expect_tx(bar(ctrl, QF), 2 * Q_BYTES);
        publish_both(bar(ctrl, QARM));
      }
      mbar_wait(bar(ctrl, QARM), 0);
      tma_load_4d_2cta(&qmap, sm32_(sq), bar(ctrl, QF), 0, qrow0, 0, h);
      // KV tiles
      for (int j = 0; j < T; ++j) {
        const int st = j & (KV_STAGES - 1);
        if (j >= KV_STAGES) {
          mbar_wait(bar(ctrl, BE0 + st), ((j >> 2) - 1) & 1);
        }
        if (rank == 0) {
          mbar_expect_tx(bar(ctrl, BF0 + st), 2 * STAGE_BYTES);
          publish_both(bar(ctrl, BA0 + st));
        }
        mbar_wait(bar(ctrl, BA0 + st), (j >> 2) & 1);
        const int tok = tok0 + j * BN;
        tma_load_4d_2cta(&kmap, sm32_(skv) + st * STAGE_BYTES,
                         bar(ctrl, BF0 + st), 0,
                         tok + static_cast<int>(rank) * 64, 0, h_kv);
        tma_load_4d_2cta(&vmap, sm32_(skv) + st * STAGE_BYTES + K_BYTES,
                         bar(ctrl, BF0 + st), 0, tok, static_cast<int>(rank),
                         h_kv);
        mark(dbg, 0, (unsigned long long)(j + 1));
      }
      // drain: consume every BE slot release so late multicast commits land
      // on live barriers before teardown
      for (int st = 0; st < KV_STAGES; ++st) {
        int n = (T > st) ? ((T - st + KV_STAGES - 1) / KV_STAGES) : 0;
        if (n > 0) mbar_wait(bar(ctrl, BE0 + st), (n - 1) & 1);
      }
    }
  } else if (warp == 4 && rank == 0) {
    // ---------------- MMA ISSUER (leader only) ----------------
    if (elect_one()) {
      mbar_wait(bar(ctrl, QF), 0);
      fence_after_sync();
      const uint32_t qbase = sm32_(sq);
      for (int i = 0; i < T; ++i) {
        const int st3 = i % S_STAGES;
        const int st4 = i & (KV_STAGES - 1);
        mbar_wait(bar(ctrl, BF0 + st4), (i >> 2) & 1);
        fence_after_sync();
        if (i >= S_STAGES) {
          mbar_wait(bar(ctrl, PV0 + st3), ((i / S_STAGES) - 1) & 1);
          fence_after_sync();
        }
        const uint32_t kbase = sm32_(skv) + st4 * STAGE_BYTES;
        const uint32_t tmem_s = tmem + st3 * 128;
        #pragma unroll
        for (int kk = 0; kk < 12; ++kk) {
          uint32_t qoff = (kk >> 2) * 16384 + (kk & 3) * 32;
          uint32_t koff = (kk >> 2) * 8192 + (kk & 3) * 32;
          uint64_t da = make_desc(qbase + qoff, 0, 1024);
          uint64_t db = make_desc(kbase + koff, 0, 1024);
          mma_ss_2cta(tmem_s, da, db, IDESC_QK2, kk > 0);
        }
        mma_commit_mc(bar(ctrl, SF0 + st3));
        mark(dbg, 1, (unsigned long long)(i + 1));
        if (i > 0) {
          const int ip = i - 1;
          const int st3p = ip % S_STAGES;
          const int st4p = ip & (KV_STAGES - 1);
          mbar_wait(bar(ctrl, PF0 + st3p), (ip / S_STAGES) & 1);
          mbar_wait(bar(ctrl, OC), ip & 1);
          fence_after_sync();
          const uint32_t vbase = sm32_(skv) + st4p * STAGE_BYTES + K_BYTES;
          const uint32_t tmem_p = tmem + st3p * 128;
          #pragma unroll
          for (int kk = 0; kk < 8; ++kk) {
            uint64_t db = make_desc(vbase + kk * 2048, 0, 1024);
            mma_ts_2cta(tmem + 384, tmem_p + kk * 8, db, IDESC_PV2,
                        (ip > 0) | (kk > 0));
          }
          mma_commit_mc(bar(ctrl, PV0 + st3p));
          mma_commit_mc(bar(ctrl, BE0 + st4p));
        }
      }
      // tail PV(T-1)
      {
        const int il = T - 1;
        const int st3 = il % S_STAGES;
        const int st4 = il & (KV_STAGES - 1);
        mbar_wait(bar(ctrl, PF0 + st3), (il / S_STAGES) & 1);
        mbar_wait(bar(ctrl, OC), il & 1);
        fence_after_sync();
        const uint32_t vbase = sm32_(skv) + st4 * STAGE_BYTES + K_BYTES;
        const uint32_t tmem_p = tmem + st3 * 128;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          uint64_t db = make_desc(vbase + kk * 2048, 0, 1024);
          mma_ts_2cta(tmem + 384, tmem_p + kk * 8, db, IDESC_PV2,
                      (il > 0) | (kk > 0));
        }
        mma_commit_mc(bar(ctrl, PV0 + st3));
        mma_commit_mc(bar(ctrl, BE0 + st4));
      }
      mark(dbg, 2, 1ull);
    }
  } else if (warp < 4) {
    // ---------------- SOFTMAX (warps 0-3, both CTAs; lane owns a row) -----
    const int lane = tid & 31;
    const int lrow = warp * 32 + lane;          // local row [0,128)
    const int grow = row0 + lrow;               // global q row
    const uint32_t lb = static_cast<uint32_t>(warp * 32) << 16;
    const int vis = kv_off + grow;
    float mr = NEG_BIG, d = 0.0f;
    float sv4[4][32];

    for (int i = 0; i < T; ++i) {
      const int st3 = i % S_STAGES;
      const uint32_t ts = tmem + st3 * 128 + lb;
      mbar_wait(bar(ctrl, SF0 + st3), (i / S_STAGES) & 1);
      fence_after_sync();
      const int n0 = i * BN;
      const bool masked = i >= Tfull;

      #pragma unroll
      for (int cc = 0; cc < 4; ++cc)
        tmem_ld32_nw(sv4[cc], ts + cc * 32);
      tmem_wait_ld();
      // S columns released for QK(i+S_STAGES) reuse tracking via PV commit
      // only (P overwrites in place below); score data is fully in registers.

      float tm = NEG_BIG;
      if (!masked) {
        #pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          float pm = sv4[cc][0];
          #pragma unroll
          for (int x = 1; x < 32; ++x) pm = fmaxf(pm, sv4[cc][x]);
          tm = fmaxf(tm, pm);
        }
      } else {
        const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
        const int jlo = windowed ? max(0, vis - window_left) : 0;
        #pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          float pm = NEG_BIG;
          #pragma unroll
          for (int x = 0; x < 32; ++x) {
            int j = n0 + cc * 32 + x;
            pm = fmaxf(pm, (j >= jlo && j < jhi) ? sv4[cc][x] : NEG_BIG);
          }
          tm = fmaxf(tm, pm);
        }
      }
      float nr = fmaxf(mr, tm);
      if (!(t3opt & 4) && (nr - mr) * sc2 <= LAG_MAX_LOG2) nr = mr;
      const float nm2 = nr * sc2;
      const float alpha = (nr == mr) ? 1.0f : ex2_approx(fmaf(mr, sc2, -nm2));
      mr = nr;

      float pc = 0.0f;
      if (!masked) {
        #pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          uint32_t pr[16];
          #pragma unroll
          for (int x2 = 0; x2 < 16; ++x2) {
            float p0 = ex2_approx(fmaf(sv4[cc][x2 * 2], sc2, -nm2));
            float p1 = ex2_approx(fmaf(sv4[cc][x2 * 2 + 1], sc2, -nm2));
            pc += p0 + p1;
            pr[x2] = pack_bf16(p0, p1);
          }
          tmem_st16bf(ts + cc * 16, pr);
        }
      } else {
        const int jhi = min(causal ? vis + 1 : INT32_MAX, kvlen);
        const int jlo = windowed ? max(0, vis - window_left) : 0;
        #pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          uint32_t pr[16];
          #pragma unroll
          for (int x2 = 0; x2 < 16; ++x2) {
            float p0, p1;
            {
              int j = n0 + cc * 32 + x2 * 2;
              p0 = (j >= jlo && j < jhi)
                       ? ex2_approx(fmaf(sv4[cc][x2 * 2], sc2, -nm2)) : 0.0f;
            }
            {
              int j = n0 + cc * 32 + x2 * 2 + 1;
              p1 = (j >= jlo && j < jhi)
                       ? ex2_approx(fmaf(sv4[cc][x2 * 2 + 1], sc2, -nm2)) : 0.0f;
            }
            pc += p0 + p1;
            pr[x2] = pack_bf16(p0, p1);
          }
          tmem_st16bf(ts + cc * 16, pr);
        }
      }
      tmem_wait_st();
      fence_before_sync();
      if (lane == 0) mbar_arrive_leader(bar(ctrl, PF0 + st3), rank);
      d = d * alpha + pc;

      // O correction: only when some row's max jumped > 2^8 (rare; lagged
      // max). Fast path arrives OC without touching O or waiting PV.
      const bool any_ne = __any_sync(0xffffffffu, alpha != 1.0f);
      const bool need_corr = (i > 0) && any_ne && !(t3opt & 2);
      if (need_corr || ((i > 0) && (t3opt & 1))) {
        mbar_wait(bar(ctrl, PV0 + ((i - 1) % S_STAGES)),
                  ((i - 1) / S_STAGES) & 1);
        fence_after_sync();
      }
      if (need_corr) {
        const uint32_t to = tmem + 384 + lb;
        #pragma unroll
        for (int cc = 0; cc < 4; ++cc)
          tmem_ld32_nw(sv4[cc], to + cc * 32);
        tmem_wait_ld();
        #pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          #pragma unroll
          for (int x = 0; x < 32; ++x) sv4[cc][x] *= alpha;
          tmem_st32(to + cc * 32, sv4[cc]);
        }
        tmem_wait_st();
      }
      fence_before_sync();
      if (lane == 0) mbar_arrive_leader(bar(ctrl, OC), rank);
      mark(dbg, 3, (unsigned long long)(i + 1));
    }

    // ---- epilogue (per-CTA own rows) ----
    {
      const int il = T - 1;
      mbar_wait(bar(ctrl, PV0 + (il % S_STAGES)), (il / S_STAGES) & 1);
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
      const bool active = (grow < qlen);
      const uint32_t to = tmem + 384 + lb;
      __nv_bfloat16* orow = outp + static_cast<int64_t>(row0 + lrow) * o_stride;
      #pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        tmem_ld32_nw(sv4[cc], to + cc * 32);
        tmem_wait_ld();
        if (active) {
          __align__(16) uint32_t pk[16];
          #pragma unroll
          for (int x2 = 0; x2 < 16; ++x2)
            pk[x2] = pack_bf16(sv4[cc][x2 * 2] * g, sv4[cc][x2 * 2 + 1] * g);
          uint4* dst = reinterpret_cast<uint4*>(orow + cc * 32);
          const uint4* src4 = reinterpret_cast<const uint4*>(pk);
          dst[0] = src4[0]; dst[1] = src4[1];
          dst[2] = src4[2]; dst[3] = src4[3];
        }
      }
      if (active && has_lse) {
        lsep[static_cast<int64_t>(row0 + lrow) * heads] = lse2;
      }
    }
  }

  if (dbg && tid == 0) atomicAdd(dbg + 5, 1ull);
  __syncthreads();
  cluster_sync();
  if (warp == 4) {
    // peer-synchronized teardown (reference pattern): both allocator warps
    // arrive on the peer's barrier, wait for the peer, then dealloc pair-wise.
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote) : "r"(bar(ctrl, PD)), "r"(rank ^ 1u));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(remote)
                 : "memory");
    mbar_wait(bar(ctrl, PD), 0);
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 ::"r"(tmem), "n"(512) : "memory");
  }
  if (dbg && tid == 0) atomicAdd(dbg + 6, 1ull);
}

}  // namespace mla_tc3

// ---------------------------------------------------------------------------
// Host: three rank-4 direct maps (Q/K/V) + cluster launch. No prep, no ws.
// ---------------------------------------------------------------------------
namespace {
struct MapKey4 {
  const void* ptr; uint64_t rows, atoms, heads; int box_rows;
  bool operator==(const MapKey4& o) const {
    return ptr == o.ptr && rows == o.rows && atoms == o.atoms &&
           heads == o.heads && box_rows == o.box_rows;
  }
};
struct MapEntry4 { MapKey4 key; CUtensorMap map; bool valid = false; };
MapEntry4 g_cache4[16];
int g_next4 = 0;

int encode4(CUtensorMap* m, const void* ptr, uint64_t rows, uint64_t heads,
            uint64_t atoms, int box_rows, int box_atoms) {
  cuuint64_t gdim[4] = {64, rows, atoms, heads};
  cuuint64_t gstr[3] = {heads * atoms * 128, 128, atoms * 128};
  cuuint32_t box[4] = {64, static_cast<cuuint32_t>(box_rows),
                       static_cast<cuuint32_t>(box_atoms), 1};
  cuuint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(ptr),
      gdim, gstr, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return static_cast<int>(r);
}
int cached4(CUtensorMap* m, const void* ptr, uint64_t rows, uint64_t heads,
            uint64_t atoms, int box_rows, int box_atoms) {
  MapKey4 key{ptr, rows, atoms, heads, box_rows};
  for (auto& e : g_cache4) {
    if (e.valid && e.key == key) { *m = e.map; return 0; }
  }
  int rc = encode4(m, ptr, rows, heads, atoms, box_rows, box_atoms);
  if (rc == 0) {
    MapEntry4& e = g_cache4[g_next4];
    g_next4 = (g_next4 + 1) & 15;
    e.key = key; e.map = *m; e.valid = true;
  }
  return rc;
}
}  // namespace

// Same exported ABI as kernel_tc.cu / kernel_tc2.cu (31 args); ws/direct/
// box/opts/pf/lm inputs are ignored (tc3 is always direct, box64-style).
extern "C" void mla_tc_ws(int total_kv, int batch, int kv_heads,
                          long long* kvblks, long long* kbytes,
                          long long* vbytes) {
  (void)total_kv; (void)batch; (void)kv_heads;
  *kvblks = 0; *kbytes = 0; *vbytes = 0;
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
  (void)kws; (void)vws; (void)kvblks; (void)max_kv_len; (void)dbg_mode;
  (void)pf_mode; (void)lag_max; (void)direct; (void)box_tok;
  using namespace mla_tc3;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(mla_tc3_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return static_cast<int>(e);
    attr_set = true;
  }
  int tiles_ub = total_q / BM_PAIR + batch;
  if (tiles_ub <= 0) return 5000;

  CUtensorMap qm, km, vm;
  int rc;
  if ((rc = cached4(&qm, q, (uint64_t)total_q, (uint64_t)heads, 3, 128, 3)))
    return 5000 + rc;
  if ((rc = cached4(&km, k, (uint64_t)total_kv, (uint64_t)kv_heads, 3, 64, 3)))
    return 6000 + rc;
  if ((rc = cached4(&vm, v, (uint64_t)total_kv, (uint64_t)kv_heads, 2, 128, 1)))
    return 7000 + rc;

  cudaLaunchConfig_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  cfg.gridDim = dim3(static_cast<unsigned>(tiles_ub) * 2u,
                     static_cast<unsigned>(heads), 1);
  cfg.blockDim = dim3(THREADS, 1, 1);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = s;
  cudaError_t le = cudaLaunchKernelEx(
      &cfg, mla_tc3_kernel,
      qm, km, vm,
      static_cast<__nv_bfloat16*>(out), static_cast<float*>(lse),
      static_cast<const float*>(sinks),
      static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
      total_q, batch, heads, kv_heads, window_left, causal, has_lse, has_sink,
      bmm1_scale, bmm2_scale, opts,
      reinterpret_cast<unsigned long long*>(dbg));
  if (le != cudaSuccess) return static_cast<int>(le);
  return static_cast<int>(cudaGetLastError());
}
