// mla_expanded_bf16 — tcgen05 split-KV dual-subtile kernel (tc2s).
// Derived from kernel_tc2p.cu (persistent, official-proven in v11) for the
// waves==1 regime where persistence cannot help: T-imbalanced causal shapes
// whose makespan equals the HEAVIEST (q-tile, head) item (probe17:
// f9 span_max 80.8k cyc = T=32 item vs span_avg 49.6k; d1/d7/d8 similar).
// tc2s splits heavy items' KV ranges across two CTAs and merges the two
// online-softmax partials in-kernel (flash-decoding style).
//
// Architecture (differences from tc2p in CAPS):
//   * PREP KERNEL (1 warp, ~3us): device-side scan of cum_q/cum_kv builds
//     the per-tile T list, picks the smallest cut >= 4 with
//     units = (real + #{T >= cut}) * heads <= grid_cap, writes the UNIT
//     TABLE {tile_id, h, k0, k1, is_split, part} (tile-fastest, LPT order =
//     heavy-first, a split item's two units ADJACENT so both partials land
//     close in time), zeroes the fetch counter and merge tickets. No memset
//     node needed; both kernels capture into CUDA graphs.
//   * Main kernel = tc2p with units instead of items: loader fetches
//     u from the counter, reads table[u], decodes geometry and applies the
//     split window: per subtile Ts = min(k1,Tsf) - min(k0,Tsf),
//     Tf = clamp(Tff - k0, 0, Ts); KV stages load absolute tiles k0..k0+T-1;
//     softmax masking uses absolute n0 = (k0+i)*BN. All ring/barrier phase
//     algebra (global g, per-subtile C, NEXT/CONS/QF/ODR handshakes) is
//     inherited unchanged - a unit is an item.
//   * SPLIT EPILOGUE (one-way protocol; the verdict is LOCAL geometry, no
//     cross-CTA negotiation): decode computes each part's sibling window
//     size sib_s. part0 with data and a non-empty sibling exports its O
//     rows (fp32, 512B/thread) + (mr, d) to a static ws slot
//     (slot = tile_id*heads + h < 148 by the dispatch gate), fences, and
//     its 4th warp sets flag[(slot,s)]; part0 with an EMPTY sibling runs
//     the standard sequential epilogue (solo - bit-identical). part1 (the
//     designated winner whenever both hold data) keeps O in TMEM, each
//     warp's lane0 spins on the flag (bounded: part0 never waits on
//     part1), then merges:
//     M = max(mr0*sc2, mr1*sc2); w_i = 2^(mr_i*sc2 - M);
//     d_m = d0*w0 + d1*w1; O_m = O0*w0 + O1*w1 (own O chunk-re-read from
//     TMEM; ODR arrive deferred until after the merge reads), then the
//     standard sink/lse/scale/store tail. Non-split units take the
//     byte-identical tc2p epilogue. ws ~19.7MB (part0 partials only).
//
// Numerics: per-tile math identical to tc2p/tc2 v1; split merge is exact
// up to fp32 recombination rounding (<< bf16 output quantization).
// Host: one-time static cudaMalloc (~39.5MB: O partials 148 slots x 2 parts
// x 256 rows x 128 f32 + md + tickets + table) on the first plain launch.
// Build: nvcc -gencode=arch=compute_103a,code=sm_103a -lcuda.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define DIN __device__ __forceinline__

namespace mla_tc2s {

constexpr int DQK = 192;
constexpr int DV = 128;
constexpr int BM = 256;
constexpr int SUB = 128;
constexpr int BN = 64;
constexpr int KV_STAGES = 3;
constexpr int THREADS = 352;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float NEG_BIG = -1.0e30f;
constexpr float LAG_MAX_LOG2 = 8.0f;

constexpr int Q_BYTES = SUB * DQK * 2;          // 49152
constexpr int Q_TOTAL = BM * DQK * 2;           // 98304
constexpr int K_BYTES = BN * DQK * 2;           // 24576
constexpr int V_BYTES = BN * DV * 2;            // 16384
constexpr int KV_TOTAL = KV_STAGES * (K_BYTES + V_BYTES);   // 122880
constexpr int CTRL_OFF = Q_TOTAL + KV_TOTAL;    // 221184
// ctrl: 26 mbarriers (208B) + tmem holder (8B at 208) + items ring
// (2 x 72B at 216..360). Region 512B.
constexpr int ITEMS_OFF = 216;     // 2 x 72B ItemGeom = 144B (216..360)
constexpr int SMEM_BYTES = CTRL_OFF + 512;

constexpr int MAX_SLOTS = 148;      // dispatch gate: tiles_ub*heads <= 148
constexpr int MAX_UNITS = 2 * MAX_SLOTS;

DIN int TCOL_S(int s, int st) { return s * 128 + st * 64; }
DIN int TCOL_O(int s) { return 256 + s * 128; }

enum { BF0 = 0, BF1, BF2, BE0, BE1, BE2,
       SF00, SF01, SF10, SF11,
       PF00, PF01, PF10, PF11,
       PV00, PV01, PV10, PV11,
       OC0, OC1, QF0, QF1, NEXT, CONS, ODR0, ODR1, NBARS };
DIN uint32_t sm32_(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
DIN uint32_t bar(char* ctrl, int idx) { return sm32_(ctrl) + idx * 8; }

// ---------------------------------------------------------------------------
// PTX primitives (identical to kernel_tc2p.cu / kernel_tc2.cu)
// ---------------------------------------------------------------------------
DIN void tma_load_4d(const CUtensorMap* map, uint32_t dst, uint32_t mbar,
                     int c0, int c1, int c2, int c3) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];"
      ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar)
      : "memory");
}
DIN void load_kv_tile(const CUtensorMap& krmap, const CUtensorMap& vrmap,
                      uint32_t kd, uint32_t vd, uint32_t mbar, int tok,
                      int h_kv) {
  #pragma unroll
  for (int g = 0; g < 8; ++g) {
    tma_load_4d(&krmap, kd + g * 3072, mbar, 0, tok + g * 8, 0, h_kv);
    tma_load_4d(&vrmap, vd + g * 2048, mbar, 0, tok + g * 8, 0, h_kv);
  }
}
DIN void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               ::"r"(a), "r"(bytes) : "memory");
}
DIN void mbar_init(uint32_t a, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
}
DIN void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
DIN void mbar_wait_h(uint32_t a, int phase, uint32_t hint) {
  asm volatile(
      "{\n\t.reg .pred P;\n\tLABW%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, %2;\n\t"
      "@P bra.uni DONEW%=;\n\tbra.uni LABW%=;\n\tDONEW%=:\n\t}\n"
      ::"r"(a), "r"(phase), "r"(hint)
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
constexpr uint32_t IDESC_QK64 = (uint32_t(128 >> 4) << 24) |
                                (uint32_t(64 >> 3) << 17) |
                                (1u << 10) | (1u << 7) | (1u << 4);
constexpr uint32_t IDESC_PV = (uint32_t(128 >> 4) << 24) |
                              (uint32_t(128 >> 3) << 17) |
                              (1u << 16) | (1u << 10) | (1u << 7) | (1u << 4);
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
          "r"(ta), "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "f"(d[4]),
          "f"(d[5]), "f"(d[6]), "f"(d[7]), "f"(d[8]), "f"(d[9]), "f"(d[10]),
          "f"(d[11]), "f"(d[12]), "f"(d[13]), "f"(d[14]), "f"(d[15]),
          "f"(d[16]), "f"(d[17]), "f"(d[18]), "f"(d[19]), "f"(d[20]),
          "f"(d[21]), "f"(d[22]), "f"(d[23]), "f"(d[24]), "f"(d[25]),
          "f"(d[26]), "f"(d[27]), "f"(d[28]), "f"(d[29]), "f"(d[30]),
          "f"(d[31]) : "memory");
}
DIN void tmem_st16bf(uint32_t ta, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::
          "r"(ta), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]),
          "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]),
          "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]) : "memory");
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

// ---------------------------------------------------------------------------
// Unit table (written by the prep kernel) and item geometry
// ---------------------------------------------------------------------------
struct UnitEntry {
  int tile_id;
  int h;
  int k0k1;      // k0 | (k1 << 16), absolute KV tile window [k0, k1)
  int flags;     // bit0 is_split, bit1 part
};

struct ItemGeom {
  int valid;
  int q_base;
  int tok0;
  int m0;
  int qlen;
  int kvlen;
  int Ts0, Tf0;   // split-windowed subtile-0 counts
  int Ts1, Tf1;
  int T;          // max(Ts0, Ts1) for this unit
  int h;
  int k0;         // absolute KV tile offset (masking n0)
  int is_split;
  int part;
  int slot;       // tile_id*heads + h (< 148 by the dispatch gate)
  int sib0, sib1; // sibling part's tile count per subtile (merge verdict)
};

DIN void decode_unit(const int32_t* __restrict__ cum_q,
                     const int32_t* __restrict__ cum_kv, int batch,
                     int causal, int window_left, int opts, int heads,
                     const UnitEntry& e, ItemGeom& it) {
  const int k0 = e.k0k1 & 0xFFFF;
  const int k1 = e.k0k1 >> 16;
  it.h = e.h;
  it.k0 = k0;
  it.is_split = e.flags & 1;
  it.part = (e.flags >> 1) & 1;
  it.slot = e.tile_id * heads + e.h;
  it.valid = 0;
  it.Ts0 = 0; it.Ts1 = 0; it.T = 0; it.Tf0 = 0; it.Tf1 = 0;
  it.q_base = 0; it.tok0 = 0; it.m0 = 0; it.qlen = 0; it.kvlen = 0;
  // batch mapping for tile_id (same scan as tc2/tc2p; the prep kernel put
  // only real tiles in the table, so b >= 0 is guaranteed, but the scan
  // stays defensive)
  int b = -1, m0 = 0, q_base = 0, tok0 = 0, qlen = 0, kvlen = 0;
  {
    int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
    for (int i = 0; i < batch; ++i) {
      int nq = cum_q[i + 1], nkv = cum_kv[i + 1];
      int len_q = nq - prev_q;
      int tiles = (len_q + BM - 1) / BM;
      if (b < 0 && e.tile_id < acc + tiles) {
        const int local = e.tile_id - acc;
        b = i;
        m0 = ((opts & 2) ? (tiles - 1 - local) : local) * BM;
        q_base = prev_q; tok0 = prev_kv;
        qlen = len_q; kvlen = nkv - prev_kv;
        break;
      }
      acc += tiles; prev_q = nq; prev_kv = nkv;
    }
  }
  if (b < 0) return;
  it.valid = 1;
  it.m0 = m0; it.q_base = q_base; it.tok0 = tok0;
  it.qlen = qlen; it.kvlen = kvlen;
  const int kv_off = kvlen - qlen;
  const bool windowed = window_left >= 0;
  const bool has_sub0 = (m0 < qlen);
  const bool has_sub1 = (m0 + SUB < qlen);
  int Ts0f, Ts1f, Tf0f, Tf1f;
  {
    int nmax = causal ? min(kvlen, kv_off + m0 + SUB) : kvlen;
    int nsafe = causal ? min(kvlen, kv_off + m0 + 1) : kvlen;
    if (windowed) nsafe = 0;
    if (!has_sub0) { nmax = 0; nsafe = 0; }
    nmax = min(nmax, kvlen);
    Ts0f = (nmax + BN - 1) / BN;
    Tf0f = nsafe / BN;
  }
  {
    int nmax = causal ? min(kvlen, kv_off + m0 + 2 * SUB) : kvlen;
    int nsafe = causal ? min(kvlen, kv_off + m0 + SUB + 1) : kvlen;
    if (windowed) nsafe = 0;
    if (!has_sub1) { nmax = 0; nsafe = 0; }
    nmax = min(nmax, kvlen);
    Ts1f = (nmax + BN - 1) / BN;
    Tf1f = nsafe / BN;
  }
  // apply the split window [k0, k1)
  const int ks0 = min(k0, Ts0f), ke0 = min(k1, Ts0f);
  const int ks1 = min(k0, Ts1f), ke1 = min(k1, Ts1f);
  (void)ks0; (void)ks1;
  it.Ts0 = ke0 - ks0;
  it.Ts1 = ke1 - ks1;
  it.Tf0 = min(max(Tf0f - k0, 0), it.Ts0);
  it.Tf1 = min(max(Tf1f - k0, 0), it.Ts1);
  it.T = max(it.Ts0, it.Ts1);
  // sibling part's per-subtile window size -> the merge verdict is LOCAL:
  // part0 exports iff its sibling holds data; part1 merges iff it does.
  if (!it.is_split) {
    it.sib0 = 0;
    it.sib1 = 0;
  } else if (it.part == 0) {
    it.sib0 = max(Ts0f - k1, 0);
    it.sib1 = max(Ts1f - k1, 0);
  } else {
    it.sib0 = min(k0, Ts0f);
    it.sib1 = min(k0, Ts1f);
  }
  if (it.T <= 0) it.valid = 0;
}

// ---------------------------------------------------------------------------
// Prep kernel: split plan + unit table + counter/ticket zeroing (1 warp)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(32, 1)
tc2s_prep(const int32_t* __restrict__ cum_q,
          const int32_t* __restrict__ cum_kv, int batch, int tiles_ub,
          int heads, int grid_cap, int causal, int window_left, int opts,
          UnitEntry* __restrict__ table, int* __restrict__ Uout,
          unsigned int* __restrict__ counter,
          unsigned int* __restrict__ tickets) {
  __shared__ int Tl[128];        // tiles_ub <= 16384/256 + 32 = 96
  __shared__ int hist[257];      // T <= ceil(16384/64) = 256
  __shared__ int sh_cut, sh_uph;
  const int lane = threadIdx.x;
  if (lane == 0) {
    for (int i = 0; i < tiles_ub && i < 128; ++i) Tl[i] = -1;
    int acc = 0, prev_q = cum_q[0], prev_kv = cum_kv[0];
    int tmax = 0;
    for (int b = 0; b < batch; ++b) {
      const int nq = cum_q[b + 1], nkv = cum_kv[b + 1];
      const int len_q = nq - prev_q, len_kv = nkv - prev_kv;
      const int tiles = (len_q + BM - 1) / BM;
      const int kv_off = len_kv - len_q;
      for (int local = 0; local < tiles && acc + local < 128; ++local) {
        const int m0 = ((opts & 2) ? (tiles - 1 - local) : local) * BM;
        const bool has1 = (m0 + SUB < len_q);
        int nmax = causal ? min(len_kv, kv_off + m0 + (has1 ? 2 * SUB : SUB))
                          : len_kv;
        nmax = min(nmax, len_kv);
        if (m0 >= len_q) nmax = 0;
        const int T = (nmax + BN - 1) / BN;
        Tl[acc + local] = T;
        if (T > tmax) tmax = T;
      }
      acc += tiles; prev_q = nq; prev_kv = nkv;
    }
    for (int i = 0; i <= tmax && i < 257; ++i) hist[i] = 0;
    int real = 0;
    for (int i = 0; i < tiles_ub && i < 128; ++i) {
      if (Tl[i] > 0) { real++; hist[Tl[i]]++; }
    }
    (void)window_left;
    // smallest cut >= 4 with units <= grid_cap (none -> cut = tmax+1 = no
    // splits; real*heads <= grid_cap is guaranteed by the dispatch gate).
    // Suffix scan: feasibility is monotone in c, so break at the first
    // (top-down) infeasible c.
    int cut = tmax + 1;
    if (!(opts & 16)) {          // bit16: diagnostic no-split mode
      int suff = 0;
      for (int c = tmax; c >= 4; --c) {
        suff += hist[c];
        if ((real + suff) * heads > grid_cap) break;
        cut = c;
      }
    }
    int u = 0;
    for (int t = 0; t < tiles_ub && t < 128; ++t) {
      if (Tl[t] > 0) u += (Tl[t] >= cut) ? 2 : 1;
    }
    sh_uph = u;                  // units per head (uniform across heads)
    sh_cut = cut;
    *Uout = u * heads;
    *counter = 0;
  }
  __syncthreads();
  {
    // lane-parallel table fill: lane L builds head blocks L, L+32, ...
    const int cut = sh_cut, uph = sh_uph;
    for (int h = lane; h < heads; h += 32) {
      int u = h * uph;
      for (int t = 0; t < tiles_ub && t < 128; ++t) {
        const int T = Tl[t];
        if (T <= 0) continue;
        if (T >= cut) {
          const int mid = (T + 1) / 2;
          table[u].tile_id = t; table[u].h = h;
          table[u].k0k1 = mid << 16; table[u].flags = 1;
          table[u + 1].tile_id = t; table[u + 1].h = h;
          table[u + 1].k0k1 = mid | (T << 16); table[u + 1].flags = 3;
          u += 2;
        } else {
          table[u].tile_id = t; table[u].h = h;
          table[u].k0k1 = T << 16; table[u].flags = 0;
          u += 1;
        }
      }
    }
    // zero merge flags/wcnt: 2 unsigned per (slot, subtile) = 4 per slot
    const int ntick = 4 * tiles_ub * heads;
    for (int i = lane; i < ntick; i += 32) tickets[i] = 0u;
  }
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(THREADS, 1)
mla_tc2s_kernel(const __grid_constant__ CUtensorMap krmap,
                const __grid_constant__ CUtensorMap vrmap,
                const __grid_constant__ CUtensorMap qrmap,
                const __nv_bfloat16* __restrict__ qraw,
                __nv_bfloat16* __restrict__ out,
                float* __restrict__ lse,
                const float* __restrict__ sinks,
                const int32_t* __restrict__ cum_q,
                const int32_t* __restrict__ cum_kv,
                int32_t total_q, int32_t batch, int32_t heads, int32_t kv_heads,
                int32_t window_left, int32_t causal, int32_t has_lse,
                int32_t has_sink, float bmm1_scale, float bmm2_scale,
                const UnitEntry* __restrict__ table,
                const int* __restrict__ Uout,
                unsigned int* __restrict__ counter,
                unsigned int* __restrict__ tickets,
                float* __restrict__ wsO, float* __restrict__ wsMD,
                int32_t opts, unsigned long long* __restrict__ dbg) {
  (void)qraw; (void)total_q;
  extern __shared__ char smem[];
  char* sq = smem;
  char* sk = smem + Q_TOTAL;
  char* sv = sk + KV_STAGES * K_BYTES;
  char* ctrl = smem + CTRL_OFF;
  uint32_t* tholder = reinterpret_cast<uint32_t*>(ctrl + NBARS * 8);
  ItemGeom* items = reinterpret_cast<ItemGeom*>(ctrl + ITEMS_OFF);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const uint32_t whint = (opts & 8) ? 0u : 0x989680u;
  const int group_div = heads / kv_heads;
  const float sc2 = bmm1_scale * LOG2E;

#ifdef TC2_TIMING
  const unsigned long long t_ent = (tid == 0) ? clock64() : 0ull;
#endif
  // ---- phase 0: pre-init fetch of the first unit (loader warp).
  // No-work CTAs (grid is sized by an upper bound) exit HERE, before
  // barrier init and tcgen05.alloc: a sentinel CTA then costs ~1us of SM
  // time instead of the full init/alloc/dealloc churn (probe18: 148 CTAs
  // launched for 96-144 units; no-op spans ~10k cyc inflated span_avg and
  // the TMEM alloc contention inflated every prologue).
  if (warp == 9 && elect_one()) {
    const int U = *(volatile const int*)Uout;
    while (true) {
      const int u = (int)atomicAdd(counter, 1u);
      if (u >= U) { items[0].valid = 0; break; }
      ItemGeom it;
      const UnitEntry e = table[u];
      decode_unit(cum_q, cum_kv, batch, causal, window_left, opts, heads,
                  e, it);
      if (it.valid) { items[0] = it; break; }
      // T==0 unit (empty split half): skip silently, fetch another
    }
  }
  __syncthreads();
  if (!items[0].valid) {
#ifdef TC2_TIMING
    if (dbg && tid == 0) atomicAdd(dbg + 66, 1ull);
#endif
    return;
  }
  if (dbg && tid == 0) atomicAdd(dbg + 8, 1ull);

  if (tid < NBARS) {
    const int i = tid;
    int cnt = 1;
    if (i >= PF00 && i <= PF11) cnt = 128;
    else if (i == OC0 || i == OC1) cnt = 128;
    else if (i == ODR0 || i == ODR1) cnt = 128;
    else if (i == CONS) cnt = 10;
    else if (i >= BE0 && i <= BE2) cnt = 2;
    mbar_init(bar(ctrl, i), cnt);
  }
  asm volatile("fence.mbarrier_init.release.cluster;");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (warp == 8) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 ::"r"(sm32_(tholder)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  fence_after_sync();
  const uint32_t tmem = *(volatile uint32_t*)tholder;

  if (warp == 9) {
    // ============ LOADER: publish units, Q(0) fill, flat KV ring ==========
    if (elect_one()) {
      const int U = *(volatile const int*)Uout;
      int f = 0;
      long long g = 0;
#ifdef TC2_TIMING
      const unsigned long long t_ld = clock64();
#endif
      while (true) {
        ItemGeom it;
        if (f == 0) {
          it = items[0];                     // staged by the pre-init fetch
        } else {
#ifdef TC2_TIMING
          const unsigned long long tf0 = clock64();
#endif
          while (true) {
            const int u = (int)atomicAdd(counter, 1u);
            if (u >= U) { it.valid = 0; it.T = 0; it.is_split = 0; break; }
            const UnitEntry e = table[u];
            decode_unit(cum_q, cum_kv, batch, causal, window_left, opts, heads,
                        e, it);
            if (it.valid) break;
          }
          mbar_wait_h(bar(ctrl, CONS), (f - 1) & 1, whint);
          items[f & 1] = it;
#ifdef TC2_TIMING
          if (dbg) atomicAdd(dbg + 56, (unsigned long long)(clock64() - tf0));
#endif
        }
        mbar_arrive(bar(ctrl, NEXT));
        if (!it.valid) break;
#ifdef TC2_TIMING
        if (dbg) {
          atomicAdd(dbg + 63, 1ull);
          if (it.is_split) atomicAdd(dbg + 65, 1ull);
        }
#endif
        if (f == 0) {
          mbar_expect_tx(bar(ctrl, QF0), 2 * Q_BYTES);
          tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF0), 0,
                      it.q_base + it.m0, 0, it.h);
          tma_load_4d(&qrmap, sm32_(sq) + Q_BYTES, bar(ctrl, QF0), 0,
                      it.q_base + it.m0 + SUB, 0, it.h);
        }
        const int h_kv = it.h / group_div;
        for (int i = 0; i < it.T; ++i, ++g) {
          const int st3 = (int)(g % KV_STAGES);
          if (g >= KV_STAGES)
            mbar_wait_h(bar(ctrl, BE0 + st3), (int)((g / KV_STAGES - 1) & 1),
                        whint);
          mbar_expect_tx(bar(ctrl, BF0 + st3), K_BYTES + V_BYTES);
          load_kv_tile(krmap, vrmap, sm32_(sk) + st3 * K_BYTES,
                       sm32_(sv) + st3 * V_BYTES, bar(ctrl, BF0 + st3),
                       it.tok0 + (it.k0 + i) * BN, h_kv);
        }
        ++f;
      }
#ifdef TC2_TIMING
      if (dbg) {
        atomicAdd(dbg + 57, (unsigned long long)(clock64() - t_ld));
        atomicAdd(dbg + 64, (unsigned long long)U);
      }
#endif
    }
  } else if (warp == 8 || warp == 10) {
    // ============ ISSUERS (identical protocol to tc2p) ====================
    if (elect_one()) {
      const bool s1warp = (warp == 10);
      const uint32_t qb = sm32_(sq) + (s1warp ? Q_BYTES : 0);
      const int kplane = 1024;
      const uint32_t ksbo = 3072u;
      const uint32_t vlbo = 1024u;
      const uint32_t vsbo = 2048u;
      const int vstep = 4096;
      const uint32_t qsbo = 1024u;
      const int qa16 = 1024;
      const uint32_t sfB = s1warp ? SF10 : SF00;
      const uint32_t pfB = s1warp ? PF10 : PF00;
      const uint32_t pvB = s1warp ? PV10 : PV00;
      const uint32_t ocB = s1warp ? OC1 : OC0;
      const uint32_t odrB = s1warp ? ODR1 : ODR0;
      const int tcol_s = s1warp ? 128 : 0;
      const int tcol_o = s1warp ? 384 : 256;
      int f = 0;
      long long Gbase = 0;
      int C = 0;
      int otherC = 0;
      int prevTs_own = 0, prevTs_other = 0;
#ifdef TC2_TIMING
      const bool timed = (dbg != nullptr);
      long long is_tiles = 0, is_bf = 0, is_bew = 0, is_qk = 0,
                is_pfw = 0, is_ocw = 0, is_pv = 0;
      long long u0 = 0, u1 = 0, u2 = 0, p0 = 0, p1 = 0, bw = 0;
#endif
      while (true) {
        mbar_wait_h(bar(ctrl, NEXT), f & 1, whint);
        fence_after_sync();
        const ItemGeom it = items[f & 1];
        mbar_arrive(bar(ctrl, CONS));
        if (!it.valid) break;
        const int Ts = s1warp ? it.Ts1 : it.Ts0;
        const int T = it.T;
        if (!s1warp) {
          if (f > 0) {
#ifdef TC2_TIMING
            const unsigned long long tr0 = timed ? clock64() : 0ull;
#endif
            if (prevTs_own > 0)
              mbar_wait_h(bar(ctrl, SF00 + ((C - 1) & 1)),
                          ((C - 1) / 2) & 1, whint);
            if (prevTs_other > 0)
              mbar_wait_h(bar(ctrl, SF10 + ((otherC - 1) & 1)),
                          ((otherC - 1) / 2) & 1, whint);
            fence_after_sync();
            mbar_expect_tx(bar(ctrl, QF0 + (f & 1)), 2 * Q_BYTES);
            tma_load_4d(&qrmap, sm32_(sq), bar(ctrl, QF0 + (f & 1)), 0,
                        it.q_base + it.m0, 0, it.h);
            tma_load_4d(&qrmap, sm32_(sq) + Q_BYTES,
                        bar(ctrl, QF0 + (f & 1)), 0,
                        it.q_base + it.m0 + SUB, 0, it.h);
#ifdef TC2_TIMING
            if (timed)
              atomicAdd(dbg + 58, (unsigned long long)(clock64() - tr0));
#endif
          }
          prevTs_own = it.Ts0;
          prevTs_other = it.Ts1;
        }
        mbar_wait_h(bar(ctrl, QF0 + (f & 1)), (f >> 1) & 1, whint);
        fence_after_sync();
        for (int i = 0; i < T; ++i) {
          const long long g = Gbase + i;
          const int st3 = (int)(g % KV_STAGES);
#ifdef TC2_TIMING
          if (timed && !s1warp) { bw = 0; u0 = clock64(); }
#endif
          mbar_wait_h(bar(ctrl, BF0 + st3), (int)((g / KV_STAGES) & 1), whint);
          fence_after_sync();
#ifdef TC2_TIMING
          if (timed && !s1warp) { u1 = clock64(); is_bf += u1 - u0; }
#endif
          const int Cg = C + i;
          if (i < Ts) {
            if (Cg >= 2) {
#ifdef TC2_TIMING
              if (timed && !s1warp) u2 = clock64();
#endif
              mbar_wait_h(bar(ctrl, pvB + (Cg & 1)),
                          (int)(((Cg - 2) / 2) & 1), whint);
              fence_after_sync();
#ifdef TC2_TIMING
              if (timed && !s1warp) {
                long long t = clock64(); is_bew += t - u2; bw += t - u2;
              }
#endif
            }
            const uint32_t tmem_s = tmem + tcol_s + (Cg & 1) * 64;
            const uint64_t da0 = make_desc(qb, 0, qsbo);
            const uint64_t db0 =
                make_desc(sm32_(sk) + st3 * K_BYTES, 0, ksbo);
            #pragma unroll
            for (int kk = 0; kk < 12; ++kk) {
              mma_ss(tmem_s, da0 + ((kk >> 2) * qa16 + (kk & 3) * 2),
                     db0 + ((kk >> 2) * (kplane >> 4) + (kk & 3) * 2),
                     IDESC_QK64, kk > 0);
            }
            mma_commit(bar(ctrl, sfB + (Cg & 1)));
          }
#ifdef TC2_TIMING
          if (timed && !s1warp) {
            u2 = clock64(); is_qk += u2 - u1 - bw; is_tiles += 1;
          }
#endif
          if (i > 0 && (i - 1) < Ts) {
            const int ip = Cg - 1;
            const int stp = ip & 1;
            const uint32_t vbase_p =
                sm32_(sv) + (int)((g - 1) % KV_STAGES) * V_BYTES;
#ifdef TC2_TIMING
            if (timed && !s1warp) p0 = clock64();
#endif
            mbar_wait_h(bar(ctrl, pfB + stp), (int)((ip / 2) & 1), whint);
#ifdef TC2_TIMING
            if (timed && !s1warp) { p1 = clock64(); is_pfw += p1 - p0; p0 = p1; }
#endif
            mbar_wait_h(bar(ctrl, ocB), ip & 1, whint);
#ifdef TC2_TIMING
            if (timed && !s1warp) { p1 = clock64(); is_ocw += p1 - p0; }
#endif
            if (i == 1 && f > 0) {
              mbar_wait_h(bar(ctrl, odrB), (f - 1) & 1, whint);
            }
            fence_after_sync();
            const uint32_t tmem_p = tmem + tcol_s + stp * 64;
            const uint64_t db0 = make_desc(vbase_p, vlbo, vsbo);
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              mma_ts(tmem + tcol_o, tmem_p + kk * 8,
                     db0 + kk * (vstep >> 4), IDESC_PV, (i > 1) | (kk > 0));
            mma_commit(bar(ctrl, pvB + stp));
#ifdef TC2_TIMING
            if (timed && !s1warp) is_pv += clock64() - p1;
#endif
          }
          if (g >= 1) {
            mma_commit(bar(ctrl, BE0 + (int)((g - 1) % KV_STAGES)));
          }
        }
        {
          const int il = Ts - 1;
          if (il >= 0 && il == T - 1) {
            const int ip = C + il;
            const int stl = ip & 1;
            const long long gl = Gbase + il;
            mbar_wait_h(bar(ctrl, pfB + stl), (int)((ip / 2) & 1), whint);
            mbar_wait_h(bar(ctrl, ocB), ip & 1, whint);
            if (f > 0 && T == 1) {
              mbar_wait_h(bar(ctrl, odrB), (f - 1) & 1, whint);
            }
            fence_after_sync();
            const uint32_t tmem_p = tmem + tcol_s + stl * 64;
            const uint64_t db0 =
                make_desc(sm32_(sv) + (int)(gl % KV_STAGES) * V_BYTES, vlbo,
                          vsbo);
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              mma_ts(tmem + tcol_o, tmem_p + kk * 8,
                     db0 + kk * (vstep >> 4), IDESC_PV, (il > 0) | (kk > 0));
            mma_commit(bar(ctrl, pvB + stl));
          }
        }
        C += Ts;
        if (!s1warp) otherC += it.Ts1;
        Gbase += T;
        ++f;
      }
#ifdef TC2_TIMING
      if (timed && !s1warp) {
        atomicAdd(dbg + 24, (unsigned long long)is_tiles);
        atomicAdd(dbg + 25, (unsigned long long)is_bf);
        atomicAdd(dbg + 26, (unsigned long long)is_bew);
        atomicAdd(dbg + 27, (unsigned long long)is_qk);
        atomicAdd(dbg + 28, (unsigned long long)is_pfw);
        atomicAdd(dbg + 29, (unsigned long long)is_ocw);
        atomicAdd(dbg + 30, (unsigned long long)is_pv);
      }
#endif
    }
  } else if (warp < 8) {
    // ============ SOFTMAX + CORRECTION + EPILOGUE (+ split merge) =========
    const int s = warp >> 2;
    const int w4 = warp & 3;
    const int lane = tid & 31;
    const int row = w4 * 32 + lane;
    const uint32_t lb = static_cast<uint32_t>(w4 * 32) << 16;
    const uint32_t sfB = s ? SF10 : SF00;
    const uint32_t pfB = s ? PF10 : PF00;
    const uint32_t pvB = s ? PV10 : PV00;
    const uint32_t ocB = s ? OC1 : OC0;
    const uint32_t odrB = s ? ODR1 : ODR0;
    const int tcol_s = s * 128;
    const int tcol_o = 256 + s * 128;
    float sv2[2][32];
    int f = 0;
    int C = 0;
#ifdef TC2_TIMING
    const bool timed_s = (dbg != nullptr) && (tid == 0);
    long long sm_tiles = 0, sm_sld = 0, sm_max = 0, sm_exp = 0,
              sm_bew = 0, sm_corr = 0, sm_per = 0;
    long long c0 = 0, cprev = 0;
#endif
    while (true) {
      mbar_wait_h(bar(ctrl, NEXT), f & 1, whint);
      fence_after_sync();
      const ItemGeom it = items[f & 1];
      if (lane == 0) mbar_arrive(bar(ctrl, CONS));
      if (!it.valid) break;
      const int grow = it.m0 + s * SUB + row;
      const int vis = (it.kvlen - it.qlen) + grow;
      const int Ts = s ? it.Ts1 : it.Ts0;
      const int Tfull = s ? it.Tf1 : it.Tf0;
      const int kvlen = it.kvlen;
      const int qlen = it.qlen;
      const int k0 = it.k0;
      __nv_bfloat16* outp =
          out + (static_cast<int64_t>(it.q_base) * heads + it.h) * DV;
      float* lsep = has_lse
          ? lse + static_cast<int64_t>(it.q_base) * heads + it.h : nullptr;
      const int64_t o_stride = static_cast<int64_t>(heads) * DV;
      const float sink2 = has_sink ? sinks[it.h] * LOG2E : 0.0f;
      const bool windowed = window_left >= 0;
      float mr = NEG_BIG, d = 0.0f;

      for (int i = 0; i < Ts; ++i) {
        const int Cg = C + i;
        const int st = Cg & 1;
        const uint32_t ts = tmem + tcol_s + st * 64 + lb;
        mbar_wait_h(bar(ctrl, sfB + st), (int)((Cg / 2) & 1), whint);
        fence_after_sync();
#ifdef TC2_TIMING
        if (timed_s) {
          c0 = clock64();
          if (Cg > 0) sm_per += c0 - cprev;
          else atomicAdd(dbg + 51, (unsigned long long)(c0 - t_ent));
          cprev = c0;
        }
#endif
        const int n0 = (k0 + i) * BN;
        const bool masked = i >= Tfull;

        #pragma unroll
        for (int cc = 0; cc < 2; ++cc)
          tmem_ld32_nw(sv2[cc], ts + cc * 32);
        tmem_wait_ld();
#ifdef TC2_TIMING
        if (timed_s) { long long t = clock64(); sm_sld += t - c0; c0 = t; }
#endif
        float tm = NEG_BIG;
        if (!masked) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            float pm = sv2[cc][0];
            #pragma unroll
            for (int x = 1; x < 32; ++x) pm = fmaxf(pm, sv2[cc][x]);
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
              pm = fmaxf(pm, (j >= jlo && j < jhi) ? sv2[cc][x] : NEG_BIG);
            }
            tm = fmaxf(tm, pm);
          }
        }
        float nr = fmaxf(mr, tm);
        if ((nr - mr) * sc2 <= LAG_MAX_LOG2) nr = mr;
        const float nm2 = nr * sc2;
        const float alpha = (nr == mr) ? 1.0f : ex2_approx(fmaf(mr, sc2, -nm2));
        mr = nr;
#ifdef TC2_TIMING
        if (timed_s) { long long t = clock64(); sm_max += t - c0; c0 = t; }
#endif
        float pc = 0.0f;
        if (!masked) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            uint32_t pr[16];
            #pragma unroll
            for (int x2 = 0; x2 < 16; ++x2) {
              float p0 = ex2_approx(fmaf(sv2[cc][x2 * 2], sc2, -nm2));
              float p1 = ex2_approx(fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2));
              pc += p0 + p1;
              pr[x2] = pack_bf16(p0, p1);
            }
            tmem_st16bf(ts + cc * 16, pr);
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
                         ? ex2_approx(fmaf(sv2[cc][x2 * 2], sc2, -nm2)) : 0.0f;
              }
              {
                int j = n0 + cc * 32 + x2 * 2 + 1;
                p1 = (j >= jlo && j < jhi)
                         ? ex2_approx(fmaf(sv2[cc][x2 * 2 + 1], sc2, -nm2)) : 0.0f;
              }
              pc += p0 + p1;
              pr[x2] = pack_bf16(p0, p1);
            }
            tmem_st16bf(ts + cc * 16, pr);
          }
        }
        tmem_wait_st();
        fence_before_sync();
        mbar_arrive(bar(ctrl, pfB + st));
        d = d * alpha + pc;
#ifdef TC2_TIMING
        if (timed_s) { long long t = clock64(); sm_exp += t - c0; c0 = t; }
#endif
        const bool need_corr =
            (i > 0) && __any_sync(0xffffffffu, alpha != 1.0f);
#ifdef TC2_TIMING
        long long cw = c0;
#endif
        if (need_corr || (i > 0 && !(opts & 1))) {
          const int ip = Cg - 1;
          mbar_wait_h(bar(ctrl, pvB + (ip & 1)), (int)((ip / 2) & 1), whint);
          fence_after_sync();
#ifdef TC2_TIMING
          if (timed_s) { long long t = clock64(); sm_bew += t - cw; cw = t; }
#endif
        }
        if (need_corr) {
          const uint32_t to = tmem + tcol_o + lb;
          #pragma unroll
          for (int half = 0; half < 2; ++half) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc)
              tmem_ld32_nw(sv2[cc], to + (half * 2 + cc) * 32);
            tmem_wait_ld();
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              #pragma unroll
              for (int x = 0; x < 32; ++x) sv2[cc][x] *= alpha;
              tmem_st32(to + (half * 2 + cc) * 32, sv2[cc]);
            }
            tmem_wait_st();
          }
        }
        fence_before_sync();
        mbar_arrive(bar(ctrl, ocB));
#ifdef TC2_TIMING
        if (timed_s) {
          long long t = clock64(); sm_corr += t - cw; sm_tiles += 1;
        }
#endif
      }
      C += Ts;

      // ---- epilogue (per unit) ----
      const bool active = (grow < qlen);
      const uint32_t to = tmem + tcol_o + lb;
      __nv_bfloat16* orow =
          outp + static_cast<int64_t>(it.m0 + s * SUB + row) * o_stride;
      if (it.is_split && it.part == 0 && Ts > 0 &&
          (s ? it.sib1 : it.sib0) > 0) {
        // ---- part0 EXPORT: hand the partial to part1 (one-way) ----
#ifdef TC2_TIMING
        const unsigned long long tx0 = timed_s ? clock64() : 0ull;
        const unsigned long long te0 = tx0;
#endif
        const int cl = C - 1;
        mbar_wait_h(bar(ctrl, pvB + (cl & 1)), (int)((cl / 2) & 1), whint);
        fence_after_sync();
        // ws layout per (slot, subtile): 4 chunk-blocks of 128 rows x 32
        // floats: address ((c*SUB + row)*32) -> consecutive lanes write
        // consecutive 128B = fully coalesced 4KB per warp-chunk (the
        // row-major variant measured 11.8k cyc export / 31.5k merge on f9:
        // 16B requests at 512B stride).
        const size_t sbase = static_cast<size_t>(it.slot) * 2 + s;
        float* wO = wsO + sbase * (SUB * DV);
        float* wMD = wsMD + sbase * SUB * 2 + row * 2;
        wMD[0] = mr;
        wMD[1] = d;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc)
            tmem_ld32_nw(sv2[cc], to + (half * 2 + cc) * 32);
          tmem_wait_ld();
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            float4* dst4 = reinterpret_cast<float4*>(
                wO + ((half * 2 + cc) * SUB + row) * 32);
            #pragma unroll
            for (int q4 = 0; q4 < 8; ++q4)
              dst4[q4] = make_float4(sv2[cc][q4 * 4], sv2[cc][q4 * 4 + 1],
                                     sv2[cc][q4 * 4 + 2], sv2[cc][q4 * 4 + 3]);
          }
        }
        __threadfence();
        unsigned* fl = &tickets[(it.slot * 2 + s) * 2];
        if (lane == 0) {
          const unsigned old = atomicAdd(&fl[1], 1u);
          if (old == 3) {
            __threadfence();
            atomicExch(&fl[0], 1u);
          }
        }
#ifdef TC2_TIMING
        if (timed_s)
          atomicAdd(dbg + 67, (unsigned long long)(clock64() - tx0));
#endif
        fence_before_sync();
        mbar_arrive(bar(ctrl, odrB));
#ifdef TC2_TIMING
        if (timed_s) {
          atomicAdd(dbg + 68, (unsigned long long)(clock64() - te0));
          atomicAdd(dbg + 70, 1ull);
        }
#endif
      } else if (it.is_split && it.part == 1 && Ts > 0) {
        // ---- part1 MERGE (designated winner whenever both hold data;
        // part0 provably exported or its sibling window is this one) ----
#ifdef TC2_TIMING
        const unsigned long long tm0 = timed_s ? clock64() : 0ull;
        const unsigned long long te1 = tm0;
#endif
        const int cl = C - 1;
        mbar_wait_h(bar(ctrl, pvB + (cl & 1)), (int)((cl / 2) & 1), whint);
        fence_after_sync();
        // bounded one-way wait: part0 never waits on part1
        if (lane == 0) {
          volatile unsigned* fl = &tickets[(it.slot * 2 + s) * 2];
          while (*fl == 0u) { }
        }
        __syncwarp();
        __threadfence();
#ifdef TC2_TIMING
        if (timed_s) atomicAdd(dbg + 59, (unsigned long long)(clock64() - tm0));
#endif
        const size_t sbase = static_cast<size_t>(it.slot) * 2 + s;
        const float* oO = wsO + sbase * (SUB * DV);
        const float* oMD = wsMD + sbase * SUB * 2 + row * 2;
        const float mr_o = oMD[0], d_o = oMD[1];
        const float a = mr * sc2, b = mr_o * sc2;
        const float M = fmaxf(a, b);
        const float w1 = ex2_approx(a - M);
        const float w2 = ex2_approx(b - M);
        const float d_m = fmaf(d, w1, d_o * w2);
        float lse2 = M + log2f(d_m);
        float extra = 1.0f;
        if (has_sink) {
          float t = fmaxf(lse2, sink2) + log2f(1.0f + exp2f(-fabsf(lse2 - sink2)));
          extra = exp2f(lse2 - t);
          lse2 = t;
        }
        const float gsc = bmm2_scale * (d_m > 0.0f ? 1.0f / d_m : 0.0f) * extra;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc)
            tmem_ld32_nw(sv2[cc], to + (half * 2 + cc) * 32);
          tmem_wait_ld();
          if (active) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              const float4* o4 = reinterpret_cast<const float4*>(
                  oO + ((half * 2 + cc) * SUB + row) * 32);
              __align__(16) uint32_t pk[16];
              #pragma unroll
              for (int q4 = 0; q4 < 8; ++q4) {
                const float4 vo = o4[q4];
                const float m0f = fmaf(sv2[cc][q4 * 4], w1, vo.x * w2);
                const float m1f = fmaf(sv2[cc][q4 * 4 + 1], w1, vo.y * w2);
                const float m2f = fmaf(sv2[cc][q4 * 4 + 2], w1, vo.z * w2);
                const float m3f = fmaf(sv2[cc][q4 * 4 + 3], w1, vo.w * w2);
                pk[q4 * 2] = pack_bf16(m0f * gsc, m1f * gsc);
                pk[q4 * 2 + 1] = pack_bf16(m2f * gsc, m3f * gsc);
              }
              uint4* dst =
                  reinterpret_cast<uint4*>(orow + (half * 2 + cc) * 32);
              const uint4* src4 = reinterpret_cast<const uint4*>(pk);
              dst[0] = src4[0]; dst[1] = src4[1];
              dst[2] = src4[2]; dst[3] = src4[3];
            }
          }
        }
        if (active && has_lse) {
          lsep[static_cast<int64_t>(it.m0 + s * SUB + row) * heads] = lse2;
        }
#ifdef TC2_TIMING
        if (timed_s) atomicAdd(dbg + 62, (unsigned long long)(clock64() - tm0));
#endif
        fence_before_sync();
        mbar_arrive(bar(ctrl, odrB));
#ifdef TC2_TIMING
        if (timed_s) {
          atomicAdd(dbg + 69, (unsigned long long)(clock64() - te1));
          atomicAdd(dbg + 71, 1ull);
        }
#endif
      } else if (Ts > 0) {
#ifdef TC2_TIMING
        const unsigned long long t_ep = timed_s ? clock64() : 0ull;
#endif
        const int cl = C - 1;
        mbar_wait_h(bar(ctrl, pvB + (cl & 1)), (int)((cl / 2) & 1), whint);
        fence_after_sync();
        float inv = d > 0.0f ? 1.0f / d : 0.0f;
        float lse2 = mr * sc2 + log2f(d);
        float extra = 1.0f;
        if (has_sink) {
          float t = fmaxf(lse2, sink2) + log2f(1.0f + exp2f(-fabsf(lse2 - sink2)));
          extra = exp2f(lse2 - t);
          lse2 = t;
        }
        const float gsc = bmm2_scale * inv * extra;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          #pragma unroll
          for (int cc = 0; cc < 2; ++cc)
            tmem_ld32_nw(sv2[cc], to + (half * 2 + cc) * 32);
          tmem_wait_ld();
          if (half == 1) {
            fence_before_sync();
            mbar_arrive(bar(ctrl, odrB));
          }
          if (active) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
              __align__(16) uint32_t pk[16];
              #pragma unroll
              for (int x2 = 0; x2 < 16; ++x2)
                pk[x2] = pack_bf16(sv2[cc][x2 * 2] * gsc,
                                   sv2[cc][x2 * 2 + 1] * gsc);
              uint4* dst =
                  reinterpret_cast<uint4*>(orow + (half * 2 + cc) * 32);
              const uint4* src4 = reinterpret_cast<const uint4*>(pk);
              dst[0] = src4[0]; dst[1] = src4[1];
              dst[2] = src4[2]; dst[3] = src4[3];
            }
          }
        }
        if (active && has_lse) {
          lsep[static_cast<int64_t>(it.m0 + s * SUB + row) * heads] = lse2;
        }
#ifdef TC2_TIMING
        if (timed_s)
          atomicAdd(dbg + 52, (unsigned long long)(clock64() - t_ep));
#endif
      } else {
        fence_before_sync();
        mbar_arrive(bar(ctrl, odrB));
      }
      ++f;
    }
#ifdef TC2_TIMING
    if (timed_s) {
      atomicAdd(dbg + 16, (unsigned long long)sm_tiles);
      atomicAdd(dbg + 17, (unsigned long long)sm_sld);
      atomicAdd(dbg + 18, (unsigned long long)sm_max);
      atomicAdd(dbg + 19, (unsigned long long)sm_exp);
      atomicAdd(dbg + 20, (unsigned long long)sm_bew);
      atomicAdd(dbg + 21, (unsigned long long)sm_corr);
      atomicAdd(dbg + 22, (unsigned long long)sm_per);
    }
#endif
  }

  if (dbg && tid == 0) atomicAdd(dbg + 5, 1ull);
  __syncthreads();
  if (warp == 8) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 ::"r"(tmem), "r"(512) : "memory");
  }
  if (dbg && tid == 0) atomicAdd(dbg + 6, 1ull);
#ifdef TC2_TIMING
  if (dbg && tid == 0) {
    const unsigned long long sp = clock64() - t_ent;
    atomicAdd(dbg + 48, sp);
    atomicAdd(dbg + 49, 1ull);
    atomicMax(dbg + 50, sp);
  }
#endif
}

}  // namespace mla_tc2s

// ---------------------------------------------------------------------------
// Host: rank-4 direct maps + prep/main launch + static split workspace.
// ---------------------------------------------------------------------------
namespace {
struct MapKey4 {
  const void* ptr; uint64_t tokens, atoms, heads; int box_tok;
  bool operator==(const MapKey4& o) const {
    return ptr == o.ptr && tokens == o.tokens && atoms == o.atoms &&
           heads == o.heads && box_tok == o.box_tok;
  }
};
struct MapEntry4 { MapKey4 key; CUtensorMap map; bool valid = false; };
MapEntry4 g_cache4[16];
int g_next4 = 0;

int encode4(CUtensorMap* m, const void* ptr, uint64_t tokens, uint64_t heads,
            uint64_t atoms, int box_tok) {
  cuuint64_t gdim[4] = {64, tokens, atoms, heads};
  cuuint64_t gstr[3] = {heads * atoms * 128, 128, atoms * 128};
  cuuint32_t box[4] = {64, static_cast<cuuint32_t>(box_tok),
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
            uint64_t atoms, int box_tok) {
  MapKey4 key{ptr, tokens, atoms, heads, box_tok};
  for (auto& e : g_cache4) {
    if (e.valid && e.key == key) { *m = e.map; return 0; }
  }
  int rc = encode4(m, ptr, tokens, heads, atoms, box_tok);
  if (rc == 0) {
    MapEntry4& e = g_cache4[g_next4];
    g_next4 = (g_next4 + 1) & 15;
    e.key = key; e.map = *m; e.valid = true;
  }
  return rc;
}

// static device workspace (one-time cudaMalloc on the first plain launch):
//   [0]     counter (unsigned, 16B pad)
//   [16]    U (int, 16B pad)
//   [32]    unit table (296 x 16B = 4736B, pad to 4864)
//   [4864]  flags/wcnt (2 words per (slot,subtile): 4*148 = 2368B)
//   [7424]  wsMD (148*2*128*2 floats = 303104B; part0 partials only)
//   [310528] wsO (148*2*128*128 floats = 19398656B)
constexpr size_t OFF_COUNTER = 0;
constexpr size_t OFF_U = 16;
constexpr size_t OFF_TABLE = 32;
constexpr size_t OFF_TICKETS = 4864;
constexpr size_t OFF_WSMD = 7424;
constexpr size_t WSMD_BYTES =
    (size_t)148 * 2 * 128 * 2 * sizeof(float);            // 303104
constexpr size_t OFF_WSO = OFF_WSMD + WSMD_BYTES;         // 310528, 256-aligned
constexpr size_t WSO_BYTES =
    (size_t)148 * 2 * 128 * 128 * sizeof(float);          // 19398656
constexpr size_t WS_TOTAL = OFF_WSO + WSO_BYTES;          // 19709184
char* g_ws = nullptr;
int g_sms = 0;
}  // namespace

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
  (void)kws; (void)vws; (void)kvblks; (void)max_kv_len;
  (void)pf_mode; (void)lag_max; (void)direct; (void)box_tok;
  using namespace mla_tc2s;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(mla_tc2s_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return static_cast<int>(e);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 9002;
    if (cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess)
      return 9003;
    if (g_sms <= 0) g_sms = 148;
    if (cudaMalloc(&g_ws, WS_TOTAL) != cudaSuccess) return 9001;
    attr_set = true;
  }
  const int tiles_ub = total_q / BM + batch;
  if (tiles_ub <= 0) return static_cast<int>(cudaGetLastError());
  const int ub_items = tiles_ub * heads;
  // dispatch gate guarantees ub_items <= MAX_SLOTS for tc2s shapes; be
  // defensive anyway (fall through safely with no splits beyond capacity)
  if (ub_items > MAX_SLOTS) return 9100;
  const int grid_cap = min(g_sms, 2 * ub_items);

  CUtensorMap krmap, vrmap, qrmap;
  int rc;
  if ((rc = cached4(&krmap, k, (uint64_t)total_kv, (uint64_t)kv_heads, 3, 8)))
    return 5000 + rc;
  if ((rc = cached4(&vrmap, v, (uint64_t)total_kv, (uint64_t)kv_heads, 2, 8)))
    return 6000 + rc;
  if ((rc = cached4(&qrmap, q, (uint64_t)total_q, (uint64_t)heads, 3, 128)))
    return 7000 + rc;

  unsigned int* counter = reinterpret_cast<unsigned int*>(g_ws + OFF_COUNTER);
  int* Uout = reinterpret_cast<int*>(g_ws + OFF_U);
  UnitEntry* table = reinterpret_cast<UnitEntry*>(g_ws + OFF_TABLE);
  unsigned int* tickets = reinterpret_cast<unsigned int*>(g_ws + OFF_TICKETS);
  float* wsMD = reinterpret_cast<float*>(g_ws + OFF_WSMD);
  float* wsO = reinterpret_cast<float*>(g_ws + OFF_WSO);

  if (dbg_mode == 2 || dbg_mode == 3) {
    // Diagnostic hooks (probes only). 2: run the prep kernel alone (timing
    // isolation; idempotent). 3: prep + copy U and the unit table into the
    // dbg buffer: dbg[0] low 32 bits = U; bytes 8.. hold U-independent
    // MAX_UNITS entries as {tile|h<<32, k0k1|(flags<<16)<<32} word pairs.
    tc2s_prep<<<dim3(1), 32, 0, s>>>(
        static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
        batch, tiles_ub, heads, grid_cap, causal, window_left, opts,
        table, Uout, counter, tickets);
    cudaError_t pe = cudaGetLastError();
    if (pe != cudaSuccess) return static_cast<int>(pe);
    if (dbg_mode == 3 && dbg) {
      if (cudaMemcpy(dbg, Uout, 4, cudaMemcpyDeviceToDevice) != cudaSuccess)
        return static_cast<int>(cudaGetLastError());
      if (cudaMemcpy(reinterpret_cast<char*>(dbg) + 8, table,
                     MAX_UNITS * sizeof(UnitEntry), cudaMemcpyDeviceToDevice)
              != cudaSuccess)
        return static_cast<int>(cudaGetLastError());
    }
    return static_cast<int>(cudaGetLastError());
  }

  tc2s_prep<<<dim3(1), 32, 0, s>>>(
      static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
      batch, tiles_ub, heads, grid_cap, causal, window_left, opts,
      table, Uout, counter, tickets);
  cudaError_t pe = cudaGetLastError();
  if (pe != cudaSuccess) return static_cast<int>(pe);

  mla_tc2s_kernel<<<dim3(grid_cap), THREADS, SMEM_BYTES, s>>>(
      krmap, vrmap, qrmap, static_cast<const __nv_bfloat16*>(q),
      static_cast<__nv_bfloat16*>(out), static_cast<float*>(lse),
      static_cast<const float*>(sinks),
      static_cast<const int32_t*>(cum_q), static_cast<const int32_t*>(cum_kv),
      total_q, batch, heads, kv_heads, window_left, causal, has_lse, has_sink,
      bmm1_scale, bmm2_scale, table, Uout, counter, tickets, wsO, wsMD, opts,
      reinterpret_cast<unsigned long long*>(dbg));
  return static_cast<int>(cudaGetLastError());
}
