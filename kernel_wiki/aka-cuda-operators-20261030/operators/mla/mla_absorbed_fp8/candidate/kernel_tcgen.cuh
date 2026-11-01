#pragma once
#include "cuda_primitives.cuh"
namespace mla_v1 {
struct Params {
  const uint8_t* q;
  const uint8_t* kv;
  const int* block_tables;
  const int* seq_lens;
  const int* cum_q;
  uint8_t* out;
  float* lse;
  const float* sinks;
  int heads;
  int uniform_q;
  int page_shift;
  int tstride;
  int batch;
  float qk_scale;   // bmm1_scale * log2(e)
  float bmm2;
  // KV-split (flash-decoding) state: partials live in the caller's declared
  // scratch workspace_buffer and are reduced by mla_v1_combine_kernel.
  float* partial_o;    // [item][split][rows_max][512] fp32, unnormalized
  float* partial_ms;   // [item][split][rows_max] log2-domain shift
  float* partial_ds;   // [item][split][rows_max] denominator at that shift
  int n_splits;
  int rows_max;
  int tiles_per_cta;   // target KV tiles per split CTA; 0 = one CTA per item
  // Total number of (item, split) work indices this launch covers, i.e. what
  // grid.x * grid.y would have been in the one-CTA-per-index layout. Only read
  // under MLA_PERSIST, but always set so both layouts share one Params.
  int n_work;
  // One workspace word: the tiles_per_cta this launch actually settled on, which
  // balanced_tpc() derives on the device from seq_lens. The main kernel writes it
  // and the combine reads it, so the combine's thousands of short-lived CTAs do
  // not each re-walk the batch. Null when the caller's scratch has no room.
  int* tpc_shared;
  int* ns_shared;
  int pv_mode;
  int skip_qk;
  int skip_softmax;
  // diagnostic: >0 issues only one cp.async per thread per KV tile so the load
  // path can be separated from the rest of the pipeline.
  int stub_loads;
  uint8_t* dbg_p;      // when non-null, dump the P tile of tile `dbg_p_tile`
  float* dbg_o;        // when non-null, dump the raw TMEM O columns [0,256)
  int dbg_p_tile;
  int* dbg;         // optional progress markers (host-mapped), null in production
};

__device__ __forceinline__ void mark(const Params& p, int slot, int value) {
  (void)p;
  (void)slot;
  (void)value;
}

// Q rows are contiguous (576 B each). Two threads share a row and each walks 18 of
// the 36 sixteen-byte chunks. The atom-blocked SMEM layout swizzles the chunk
// index per row, so each chunk address is computed explicitly (the previous
// version assumed chunks were contiguous inside a row, which broke the swizzle).
__device__ __forceinline__ void load_q_tile(const uint8_t* qbase, int valid_rows, uint8_t* dst) {
  constexpr int CHUNKS = KDIM / 16;        // 36
  constexpr int TPR = NTHREADS / M_TILE;   // threads per row (2 or 4)
  constexpr int PER_THREAD = CHUNKS / TPR; // 18 or 9
  const int r = threadIdx.x / TPR;
  const int sub = threadIdx.x - r * TPR;
  if (r >= valid_rows) return;
  const int rb = r >> 3, ri = r & 7;
  const int swz = (ri >> 1) & 3;
  const size_t rowbase = (size_t)rb * BLOCK_BYTES + (size_t)ri * SW;
  const uint8_t* src = qbase + (size_t)r * KDIM;
#pragma unroll
  for (int i = 0; i < PER_THREAD; ++i) {
    const int c = sub + i * TPR;
    const size_t off =
        (size_t)(c >> 2) * (QRB * BLOCK_BYTES) + rowbase + (size_t)((c & 3) ^ swz) * 16;
    cp_async16(smem_u32(dst + off), src + (size_t)c * 16);
  }
}

__device__ __noinline__ void zero_q_tail(uint8_t* dst, int valid_rows) {
  constexpr int TPR = NTHREADS / M_TILE;
  constexpr int PER_THREAD = (KDIM / 16) / TPR;
  const int r = threadIdx.x / TPR;
  if (r < valid_rows) return;
  const int sub = threadIdx.x - r * TPR;
  const int rb = r >> 3, ri = r & 7;
  const int swz = (ri >> 1) & 3;
  const size_t rowbase = (size_t)rb * BLOCK_BYTES + (size_t)ri * SW;
#pragma unroll
  for (int i = 0; i < PER_THREAD; ++i) {
    const int c = sub + i * TPR;
    const size_t off =
        (size_t)(c >> 2) * (QRB * BLOCK_BYTES) + rowbase + (size_t)((c & 3) ^ swz) * 16;
    asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};" ::"r"(smem_u32(dst + off)), "r"(0));
  }
}

// Four threads share one key. The paged slot is resolved separately from the
// cp.async burst so the block-table LDG can be issued while the tensor core is
// already busy; otherwise its latency sits directly in front of the load issue.
__device__ __forceinline__ int table_ldg_volatile(const int* ptr) {
  int v;
  asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}
template <bool VOLATILE>
__device__ __forceinline__ int kv_slot_t(const Params& p, const int* table, int kvlen, int key0) {
  constexpr int TPK = NTHREADS / NK;   // threads per key (4 or 8)
  const int jj = threadIdx.x / TPK;
  int key = key0 + jj;
  if (key >= kvlen) key = kvlen - 1;
  const int page = 1 << p.page_shift;
#if MLA_PROBE & 2048
  // Identity page table: same slot arithmetic, no dependent global load. See probe bit 2048.
  (void)table;
  const int entry = key >> p.page_shift;
#else
  const int entry =
      VOLATILE ? table_ldg_volatile(table + (key >> p.page_shift)) : table[key >> p.page_shift];
#endif
  return entry * page + (key & (page - 1));
}
__device__ __forceinline__ int kv_slot(const Params& p, const int* table, int kvlen, int key0) {
  return kv_slot_t<false>(p, table, kvlen, key0);
}
__device__ __forceinline__ void load_kv_slot(const uint8_t* kvbase, int slot, uint8_t* dst,
                                             int stub) {
  constexpr int CHUNKS = KDIM / 16;                       // 36
  constexpr int TPK = NTHREADS / NK;                      // 4 or 8 threads per key
  constexpr int PER_THREAD = (CHUNKS + TPK - 1) / TPK;    // 9 or 5
  const int jj = threadIdx.x / TPK;
  const int sub = threadIdx.x - jj * TPK;
  const uint8_t* src = kvbase + (size_t)slot * KDIM;
  const int rb = jj >> 3, ri = jj & 7;
  const int swz = (ri >> 1) & 3;
  const size_t rowbase = (size_t)rb * BLOCK_BYTES + (size_t)ri * SW;
#if MLA_KV_LIN
  static_assert(TPK == 4 && CHUNKS % TPK == 0,
                "the folded strides below need sub < 4 and an exact number of copies");
  // c = sub + 4i, so (c >> 2) is i and (c & 3) is sub: both addresses are a base plus a
  // compile-time stride, and the base is the i = 0 case of the expression they replace.
  const size_t off0 = rowbase + (size_t)((sub ^ swz) * 16);
  const uint8_t* src0 = src + (size_t)sub * 16;
  constexpr size_t OFF_STEP = (size_t)(TPK / 4) * (KRB * BLOCK_BYTES);
  constexpr size_t SRC_STEP = (size_t)TPK * 16;
#pragma unroll
  for (int i = 0; i < PER_THREAD; ++i) {
    if (stub && i != 0) break;
    cp_async16(smem_u32(dst + off0 + (size_t)i * OFF_STEP), src0 + (size_t)i * SRC_STEP);
  }
#else
#pragma unroll
  for (int i = 0; i < PER_THREAD; ++i) {
    if (stub && i != 0) break;
    const int c = sub + i * TPK;
    if (CHUNKS % TPK != 0 && c >= CHUNKS) break;
    const size_t off =
        (size_t)(c >> 2) * (KRB * BLOCK_BYTES) + rowbase + (size_t)((c & 3) ^ swz) * 16;
    cp_async16(smem_u32(dst + off), src + (size_t)c * 16);
  }
#endif
}
__device__ __forceinline__ void load_kv_tile(const Params& p, const int* table, int kvlen, int key0,
                                             uint8_t* dst) {
  load_kv_slot(p.kv, kv_slot(p, table, kvlen, key0), dst, 0);
}

// Drain several 32-column TMEM groups, issuing every load before one wait:
// a serialized wait per load costs roughly 200 cycles of pure latency.
template <int NB>
__device__ __forceinline__ void tmem_ld_batch(uint32_t taddr, uint32_t v[NB][32]) {
#pragma unroll
  for (int b = 0; b < NB; ++b) tmem_ld32(taddr + b * 32, v[b]);
  if (!(kProbe & 128)) tmem_ld_wait();
}
template <int NB>
__device__ __forceinline__ void tmem_st_batch(uint32_t taddr, const uint32_t v[NB][32]) {
#pragma unroll
  for (int b = 0; b < NB; ++b) tmem_st32(taddr + b * 32, v[b]);
  tmem_st_wait();
}

// Map a flat work-item id to (batch, tile, q_len, q_start). Returns false when
// the id is past the last item. Uniform Q avoids the device-side scan.
struct ItemInfo {
  int batch;
  int tile;
  int qlen;
  int q_start;
};
__device__ __forceinline__ bool resolve_item(const Params& p, int item, ItemInfo& out_info) {
  out_info.batch = -1;
  out_info.tile = 0;
  out_info.qlen = 0;
  out_info.q_start = 0;
  if (p.cum_q == nullptr) {
    const int tpb = (p.uniform_q * p.heads + M_TILE - 1) / M_TILE;
    int b = item / tpb;
    if (b >= p.batch) return false;
    out_info.batch = b;
    out_info.tile = item - b * tpb;
    out_info.qlen = p.uniform_q;
    out_info.q_start = b * p.uniform_q;
    return true;
  }
  int acc = 0;
  for (int b = 0; b < p.batch; ++b) {
    int lo = p.cum_q[b], hi = p.cum_q[b + 1];
    int nt = ((hi - lo) * p.heads + M_TILE - 1) / M_TILE;
    if (item < acc + nt) {
      out_info.batch = b;
      out_info.tile = item - acc;
      out_info.qlen = hi - lo;
      out_info.q_start = lo;
      return true;
    }
    acc += nt;
  }
  return false;
}

// Per-row geometry inside a work item. `row` is the TMEM lane this thread owns;
// both softmax column groups use the same 128 lanes, so the row is derived from
// the warp's quadrant inside its group rather than from threadIdx.x.
struct RowInfo {
  int row;
  int row0;
  int valid_rows;
  int q_idx;
  int head;
  bool row_valid;
  int visible;
  int ntiles;
};
__device__ __forceinline__ RowInfo resolve_row(const Params& p, const ItemInfo& info, int kvlen,
                                               int row) {
  RowInfo r;
  r.row0 = info.tile * M_TILE;
  r.valid_rows = min(M_TILE, info.qlen * p.heads - r.row0);
  r.row = row;
  int rglobal = r.row0 + r.row;
  r.q_idx = rglobal / p.heads;
  r.head = rglobal - r.q_idx * p.heads;
  r.row_valid = r.row < r.valid_rows;
  r.visible = r.row_valid ? (kvlen - info.qlen + r.q_idx) : -1;
  int max_q_idx = (r.row0 + r.valid_rows - 1) / p.heads;
  int last_key = min(kvlen, kvlen - info.qlen + max_q_idx + 1);
  r.ntiles = last_key > 0 ? (last_key + NK - 1) / NK : 0;
  return r;
}

// CTAs one wave holds, matching the launcher's kSplitCap.
constexpr int kSplitWaveTarget = 148;

#ifndef MLA_PERSIST
#define MLA_PERSIST 0
#endif
#define MLA_TPC(host_tpc) \
  balanced_tpc(p.seq_lens, p.cum_q, p.batch, p.uniform_q, p.heads, host_tpc)
#if MLA_PERSIST
// A dead work index moves on to the next one; in the shipped layout it retires the CTA.
#define MLA_NEXT continue
#else
#define MLA_NEXT return
#endif

#ifndef MLA_SM_OWNFIRST
#define MLA_SM_OWNFIRST 1
#endif

#ifndef MLA_MAX_TREE
#define MLA_MAX_TREE 0
#endif

#ifndef MLA_MAX_EXCHANGE
#define MLA_MAX_EXCHANGE 0
#endif
#if MLA_MAX_EXCHANGE
// Only the macros can be checked here: NGROUP is a constexpr, which the preprocessor would
// read as 0. It gets a static_assert next to the exchange itself instead.
#if (MLA_SM_OWNFIRST == 0) || (MLA_SKIP_PAD != 0)
#error "MLA_MAX_EXCHANGE needs MLA_SM_OWNFIRST=1 and MLA_SKIP_PAD=0"
#endif
// The tree form reduces both halves of sraw_all, which the exchange does not load, and it
// measured flat on its own (ag25tree), so there is nothing to combine here.
#if MLA_MAX_TREE
#error "MLA_MAX_EXCHANGE and MLA_MAX_TREE cannot both be 1"
#endif
#endif

// A/B control for balanced_tpc: 0 keeps the host's granularity everywhere, which
// is exactly the v1.24 behaviour.
#ifndef MLA_BALANCED_SPLITS
#define MLA_BALANCED_SPLITS 1
#endif
constexpr bool kBalancedSplits = (MLA_BALANCED_SPLITS != 0);

__device__ __forceinline__ void tpc_request(const int* seq_lens, const int* cum_q, int b,
                                            int uniform_q, int heads, int& nt, int& rt) {
  nt = (seq_lens[b] + NK - 1) / NK;
  const int qlen = (cum_q == nullptr) ? uniform_q : (cum_q[b + 1] - cum_q[b]);
  rt = (qlen * heads + M_TILE - 1) / M_TILE;
}

#ifndef MLA_TPC_WALK
#define MLA_TPC_WALK 3
#endif
template <int FORM, bool SECOND>
__device__ __forceinline__ void tpc_walk(const int* seq_lens, const int* cum_q, int batch,
                                         int uniform_q, int heads, int cand,
                                         long long& sum0, long long& sum1) {
#define MLA_TPC_BODY(b)                                                    \
  {                                                                        \
    int nt, rt;                                                            \
    tpc_request(seq_lens, cum_q, (b), uniform_q, heads, nt, rt);           \
    if (SECOND) sum0 += (long long)rt * ((nt + cand - 1) / cand);          \
    else { sum0 += (long long)rt * nt; sum1 += rt; }                       \
  }
  sum0 = 0;
  sum1 = 0;
  if constexpr (FORM == 3) {
#pragma unroll 8
    for (int b = 0; b < batch; ++b) MLA_TPC_BODY(b)
  } else if constexpr (FORM == 1) {
    for (int b = 0; b < batch; ++b) MLA_TPC_BODY(b)
  } else if constexpr (FORM == 2) {
#pragma unroll
    for (int b = 0; b < 32; ++b)
      if (b < batch) MLA_TPC_BODY(b)
    for (int b = 32; b < batch; ++b) MLA_TPC_BODY(b)
  } else {
    for (int b0 = 0; b0 < batch; b0 += 8) {
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int b = b0 + u;
        if (b < batch) MLA_TPC_BODY(b)
      }
    }
  }
#undef MLA_TPC_BODY
}

__device__ __noinline__ int balanced_tpc(const int* seq_lens, const int* cum_q, int batch,
                                         int uniform_q, int heads, int tpc_host) {
  if (!kBalancedSplits) return tpc_host;
  const long long target = kSplitWaveTarget;
  long long work = 0, items_total = 0;
  tpc_walk<MLA_TPC_WALK, false>(seq_lens, cum_q, batch, uniform_q, heads, 0, work,
                                items_total);
  const long long ctas_host = work / tpc_host + items_total / 2;
  if (ctas_host <= target) return tpc_host;
  const long long waves = (ctas_host + target - 1) / target;
  if (waves * target * 4 < ctas_host * 5) return tpc_host;
  const long long f_host = waves * tpc_host;
  const long long ideal = (work + target - 1) / target;
  if (f_host * 8 <= ideal * 9) return tpc_host;
  // ceil(nt/tpc) averages half a CTA of slack per work item.
  long long denom = target - items_total / 2;
  if (denom < 1) denom = 1;
  int cand = (int)((work + denom - 1) / denom);
  if (cand < tpc_host) cand = tpc_host;
  for (int iter = 0; iter < 4; ++iter) {
    long long c = 0, unused = 0;
    tpc_walk<MLA_TPC_WALK, true>(seq_lens, cum_q, batch, uniform_q, heads, cand, c,
                                 unused);
    if (c <= target) return (cand < f_host) ? cand : tpc_host;
    cand += 1 + (int)(((long long)cand * (c - target)) / (2 * target));
  }
  return tpc_host;
}

// Number of split CTAs one work item actually uses. Both the main kernel and the
// combine kernel derive it from the same inputs, so they always agree.
__device__ __forceinline__ int split_count_at(int ntiles, int tpc, int n_splits) {
  int s = (ntiles + tpc - 1) / tpc;
  if (s > n_splits) s = n_splits;
  return s < 1 ? 1 : s;
}
// Combine side: the main kernel already settled on a granularity and published it,
// and a kernel boundary on the same stream makes that write visible here.
__device__ __forceinline__ int split_count(const Params& p, int ntiles) {
  if (p.n_splits <= 1) return 1;
  const int tpc_host = p.tiles_per_cta > 0 ? p.tiles_per_cta : ntiles;
  int tpc = tpc_host;
  if (p.tpc_shared != nullptr && *p.tpc_shared >= tpc_host) tpc = *p.tpc_shared;
  else tpc = balanced_tpc(p.seq_lens, p.cum_q, p.batch, p.uniform_q, p.heads, tpc_host);
  return split_count_at(ntiles, tpc, p.n_splits);
}

#if MLA_MAX_TREE
// Pairwise tree maximum over N values, reduced in place: N-1 fmaxf either way, but the
// dependency depth is log2(N) instead of N-1. Recursion with a constant bound so every level
// is a compile-time-constant loop and nothing is left for the compiler to guess at.
template <int N>
__device__ __forceinline__ float max_tree(float* v) {
  if constexpr (N <= 1) {
    return v[0];
  } else {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) v[i] = fmaxf(v[2 * i], v[2 * i + 1]);
    return max_tree<N / 2>(v);
  }
}
#endif

// ---------------------------------------------------------------- split softmax
// Everything the per-tile softmax reads that outlives the tile, plus the four values it
// carries into the next one. One struct so the body below can be instantiated twice -
// once with the mask-free selectors forced true and once with them forced false - without
// a second copy of it in the source.
struct SoftmaxCtx {
  const Params& p;
  uint32_t tmem;
  uint32_t lane_off;
  int col0;
  int grp;
  int row;
  int lane;
  int key0;
  int visible;
  int warp_visible;
  bool row_valid;
  bool warp_rows_live;
  bool has_dbg;
  uint8_t* phi;
  uint8_t* plo;
  float& m_shift;
  float& dsum;
  float (&racc)[DRG];
  bool& first_pv;
#if MLA_MAX_EXCHANGE
  // Aliases the epilogue's sm_dsum words, whose lifetime starts after the tile loop's last
  // barrier: the same 2 x M_TILE floats, two non-overlapping uses.
  float* sm_mrow;
#endif
};

template <bool HILO, bool ALL>
__device__ __forceinline__ void softmax_tile(const SoftmaxCtx& C) {
  const Params& p = C.p;
  const uint32_t tmem = C.tmem;
  const uint32_t lane_off = C.lane_off;
  const int col0 = C.col0;
  const int grp = C.grp;
  const int row = C.row;
  const int lane = C.lane;
  const int key0 = C.key0;
  const int visible = C.visible;
  const int warp_visible = C.warp_visible;
  const bool row_valid = C.row_valid;
  const bool warp_rows_live = C.warp_rows_live;
  const bool has_dbg = C.has_dbg;
  uint8_t* phi = C.phi;
  uint8_t* plo = C.plo;
  float& m_shift = C.m_shift;
  float& dsum = C.dsum;
  float (&racc)[DRG] = C.racc;
  bool& first_pv = C.first_pv;
#if MLA_MAX_EXCHANGE
  float* const sm_mrow = C.sm_mrow;
  static_assert(NGROUP == 2, "the exchange below flips grp with ^ 1");
#endif

#if MLA_MAX_EXCHANGE
    // This group's own NKG columns and nothing else: half the score TMEM traffic, NKG fewer
    // live registers, and the other group's half of the row maximum arrives through SMEM below.
    uint32_t sraw_all[1][32];
    tmem_ld32(tmem + lane_off + TCOL_S + col0, sraw_all[0]);
    if (!(kProbe & 128)) tmem_ld_wait();
#else
    uint32_t sraw_all[NK / 32][32];
#if MLA_SM_OWNFIRST
    // col0 is grp * NKG, so col0 ^ NKG is the other group's base: NGROUP is 2.
    tmem_ld32(tmem + lane_off + TCOL_S + col0, sraw_all[0]);
    tmem_ld32(tmem + lane_off + TCOL_S + (col0 ^ NKG), sraw_all[1]);
    if (!(kProbe & 128)) tmem_ld_wait();
#else
    tmem_ld_batch<NK / 32>(tmem + lane_off + TCOL_S, sraw_all);
#endif
#endif

    // Keys this row may attend to inside this tile, and the subset of them in
    // this group's column range. Only a row's last tile can be partial, so the
    // mask-free path is the one that runs for all but one tile per work item.
    int nval = row_valid ? (visible - key0 + 1) : (kPadFull ? NK : 0);
    nval = nval > NK ? NK : (nval < 0 ? 0 : nval);
    int nloc = nval - col0;
    nloc = nloc > NKG ? NKG : (nloc < 0 ? 0 : nloc);
    const bool full = (nloc == NKG);
    int warp_nval = warp_visible - key0 + 1;
    warp_nval = warp_nval > NK ? NK : (warp_nval < 0 ? 0 : warp_nval);

#if MLA_FORCE_UNMASKED
    const bool warp_all_keys = true;
    const int warp_nloc = NKG;
    const bool warp_all_cols = true;
#else
    constexpr bool warp_all_keys = ALL;
    constexpr bool warp_all_cols = ALL;
#endif
    // Row maximum on the raw scores: qk_scale > 0 so it commutes with max,
    // which saves one multiply per element.
    float mrow = NEG_INF;
#if MLA_MAX_EXCHANGE
    // Own NKG columns only, absolute base col0 in both forms, then one exchange.
    if (warp_all_keys) {
#pragma unroll
      for (int j = 0; j < NKG; ++j) mrow = fmaxf(mrow, __uint_as_float(sraw_all[0][j]));
    } else {
#pragma unroll
      for (int j = 0; j < NKG; ++j)
        mrow = (col0 + j < nval) ? fmaxf(mrow, __uint_as_float(sraw_all[0][j])) : mrow;
    }
    // One word per (group, row). Single buffered and race free: this tile's post-P-store
    // barrier sits between the read below and the next tile's write, and between the read and
    // the epilogue's reuse of the same words.
    sm_mrow[grp * M_TILE + row] = mrow;
    __syncthreads();
    // Both groups evaluate fmaxf(own, other), so the row maximum - and the rescale decision
    // and the shift that follow from it - still agree by construction, exactly as they did
    // when both groups reduced all 64 columns themselves.
    mrow = fmaxf(mrow, sm_mrow[(grp ^ 1) * M_TILE + row]);
#else
    if (warp_all_keys) {
#if MLA_MAX_TREE && MLA_SM_OWNFIRST && (NK == 64)
      // Two independent 16-wide trees, one per half of the score tile, then one merge: 63
      // fmaxf either way, depth 6 instead of 63. The own half has to survive for the P loop
      // below, so both trees take their first level straight out of sraw_all into a scratch;
      // the other half is dead after this and its 32 registers are free for them. The two
      // trees have no data dependence on each other, so their levels interleave.
      float tm[16], om[16];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        tm[i] = fmaxf(__uint_as_float(sraw_all[0][2 * i]), __uint_as_float(sraw_all[0][2 * i + 1]));
#pragma unroll
      for (int i = 0; i < 16; ++i)
        om[i] = fmaxf(__uint_as_float(sraw_all[1][2 * i]), __uint_as_float(sraw_all[1][2 * i + 1]));
      mrow = fmaxf(max_tree<16>(tm), max_tree<16>(om));
#else
#pragma unroll
      for (int c = 0; c < NK / 32; ++c)
#pragma unroll
        for (int j = 0; j < 32; ++j) mrow = fmaxf(mrow, __uint_as_float(sraw_all[c][j]));
#endif
    } else {
#pragma unroll
      for (int c = 0; c < NK / 32; ++c)
#pragma unroll
        for (int j = 0; j < 32; ++j)
#if MLA_SM_OWNFIRST
          // sraw_all[0] holds this group's own columns, so its absolute base is col0
          // and the other half's is col0 ^ NKG. c is a compile-time constant here, and
          // the maximum is exact in any order, so this predicate is the only thing the
          // reordered load changes.
          mrow = ((((c == 0) ? col0 : (col0 ^ NKG)) + j) < nval)
                     ? fmaxf(mrow, __uint_as_float(sraw_all[c][j]))
                     : mrow;
#else
          mrow = (c * 32 + j < nval) ? fmaxf(mrow, __uint_as_float(sraw_all[c][j])) : mrow;
#endif
    }
#endif
    const float mx = (mrow == NEG_INF) ? NEG_INF : mrow * p.qk_scale;

    // This group's own 32 columns. Indexing sraw_all[grp][...] directly would make
    // the array dynamically indexed and push it into a 288-byte local-memory frame,
    // so they are copied out - under a uniform predicate, which costs one select per
    // element, unless MLA_SM_OWNFIRST put this group's half at index 0 and made the
    // copy a rename.
    uint32_t sraw[NKG];
#pragma unroll
#if MLA_SM_OWNFIRST
    for (int j = 0; j < NKG; ++j) sraw[j] = sraw_all[0][j];
#else
    for (int j = 0; j < NKG; ++j) sraw[j] = (grp == 0) ? sraw_all[0][j] : sraw_all[1][j];
#endif
    const bool need = (mx > m_shift + RESCALE_THRESH);
    // tcgen05.ld/.st are warp-synchronous, so the rescale decision is reduced
    // across the warp and every lane takes the same path (factor 1.0 when the
    // lane itself did not need it). Both groups derive the decision from the
    // same row maximum, so they always agree per row.
    if (__any_sync(0xffffffffu, need)) {
      const float new_shift = need ? (mx + SHIFT_HEAD) : m_shift;
      const float factor = need ? ((m_shift == NEG_INF) ? 0.f : ex2(m_shift - new_shift)) : 1.f;
      m_shift = new_shift;
      dsum *= factor;
#pragma unroll
      for (int j = 0; j < DRG; ++j) racc[j] *= factor;
      if (!first_pv) {
        if (has_dbg && lane == 0) atomicAdd(p.dbg + 15, 1);   // rescale events
        // 32-column chunks are handed out round-robin so every group base
        // stays 32-aligned whatever NGROUP is.
        const uint32_t obase = tmem + lane_off + TCOL_O;
        for (int c = grp; (!kTdrSkipWarp || warp_rows_live) && c < DP / 32;
             c += kRscBatch * NGROUP) {
          uint32_t vv[kRscBatch][32];
#pragma unroll
          for (int b = 0; b < kRscBatch; ++b)
            if (c + b * NGROUP < DP / 32) tmem_ld32(obase + (c + b * NGROUP) * 32, vv[b]);
          tmem_ld_wait();
#pragma unroll
          for (int b = 0; b < kRscBatch; ++b) {
            const int cc = c + b * NGROUP;
            if (cc >= DP / 32) continue;
#pragma unroll
            for (int j = 0; j < 32; ++j)
              vv[b][j] = __float_as_uint(__uint_as_float(vv[b][j]) * factor);
            tmem_st32(obase + cc * 32, vv[b]);
          }
          tmem_st_wait();
        }
      }
    }

    // P = P_SCALE * exp2(s2 - m_shift). The denominator stays group-local
    // (the rescale factor is identical in both groups) and the two halves are
    // added once in the epilogue.
    const float bias = LOG2_P_SCALE - m_shift;
    float ds[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int h = 0; h < NKG / 16; ++h) {
      float ps[16];
      if (warp_all_cols) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          // arg already carries LOG2_P_SCALE, so e is both the E4M3 operand
          // and P_SCALE*exp2(s2-m_shift); the group denominator accumulates
          // the same scaled sum the epilogue divides out exactly once.
          const float e = ex2(fmaf(__uint_as_float(sraw[h * 16 + j]), p.qk_scale, bias));
          ps[j] = e;
          ds[j & 3] += e;
        }
      } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int jj = h * 16 + j;
          const float e =
              (jj < nloc) ? ex2(fmaf(__uint_as_float(sraw[jj]), p.qk_scale, bias)) : 0.f;
          ps[j] = e;
          ds[j & 3] += e;
        }
      }
      uint32_t hiw[4], loww[4];
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        const int j0 = q4 * 4;
        const float p0 = ps[j0], p1 = ps[j0 + 1], p2 = ps[j0 + 2], p3 = ps[j0 + 3];
        const uint32_t h01 = cvt_e4m3x2(p1, p0);
        const uint32_t h23 = cvt_e4m3x2(p3, p2);
        hiw[q4] = h01 | (h23 << 16);
        if (HILO && kLoF16) {
          loww[q4] = cvt_e4m3x2_f16x2(sub_f16x2(cvt_f16x2_f32(p1, p0), cvt_f16x2_e4m3x2(h01))) |
                     (cvt_e4m3x2_f16x2(sub_f16x2(cvt_f16x2_f32(p3, p2), cvt_f16x2_e4m3x2(h23)))
                      << 16);
        } else if (HILO) {
          const float r0 = p0 - rn3(p0), r1 = p1 - rn3(p1);
          const float r2 = p2 - rn3(p2), r3 = p3 - rn3(p3);
          loww[q4] = cvt_e4m3x2(r1, r0) | (cvt_e4m3x2(r3, r2) << 16);
        } else {
          loww[q4] = 0;
        }
      }
      const int chunk16 = grp * (NKG / 16) + h;
      const uint32_t dh = smem_u32(phi + tile_offset(row, chunk16, QRB));
      asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(dh), "r"(hiw[0]), "r"(hiw[1]),
                   "r"(hiw[2]), "r"(hiw[3]));
      if (HILO) {
        const uint32_t dl = smem_u32(plo + tile_offset(row, chunk16, QRB));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(dl), "r"(loww[0]), "r"(loww[1]),
                     "r"(loww[2]), "r"(loww[3]));
      }
    }
    dsum += ds[0] + ds[1] + ds[2] + ds[3];
}

template <bool HILO>
__device__ __forceinline__ void mla_v1_attention_body(Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  // NGROUP column groups of four warps. Every group covers all M_TILE rows (one
  // TMEM lane per thread); group `grp` owns key columns [col0, col0+NKG).
  const int grp = warp >> 2;
  const int qw = warp & 3;
  const int col0 = grp * NKG;
  const uint32_t lane_off = (uint32_t)(qw * 32) << 16;
  __shared__ __align__(8) uint32_t tmem_slot;
  uint64_t* bar_s = reinterpret_cast<uint64_t*>(smem);
  uint64_t* bar_pv = reinterpret_cast<uint64_t*>(smem + 8);
#if MLA_PV_EARLY
  // The barrier head is 64 B and the softmax exchange starts at smem + 64, so byte 16 is free.
  uint64_t* bar_pv2 = reinterpret_cast<uint64_t*>(smem + 16);
#endif
  // cross-warp softmax exchange: two column groups x M_TILE rows
  float* sm_dsum = reinterpret_cast<float*>(smem + 64);
  uint8_t* smq = smem + OFF_Q;
  uint8_t* smkv[2] = {smem + OFF_KV, smem + OFF_KV + KV_BYTES};
  uint8_t* smp[2] = {smem + OFF_P, smem + OFF_P + 2 * P_BYTES};
  // Exactly what mla_api.cu passes, as compile-time constants. See MLA_DIAG_PARAMS.
  constexpr int pv_mode = 3;
  constexpr int skip_qk = 0;
  constexpr int skip_softmax = 0;
  constexpr int stub_loads = 0;
  constexpr bool has_dbg = false;
  constexpr bool has_dbg_p = false;
  constexpr bool has_dbg_o = false;

#if MLA_PERSIST
  // ---------------------------------------------------------- work index walk
  // One CTA walks several work indices instead of owning one. The mbarriers stay
  // inside the body, so phase_s/phase_pv restart from zero on every index exactly as
  // they do in the shipped layout. Only MLA_PERSIST=2 also hoists the TMEM
  // allocation, the one resource that can outlive a work index.
  const int w_first = (int)blockIdx.x;
  const int w_limit = p.n_work;
  const int w_step = (int)gridDim.x;
  uint32_t tmem_cta = 0;
#if MLA_PERSIST >= 2
  if (tid < 32) {
    tmem_alloc(&tmem_slot, 512);
    tmem_relinquish();
  }
  __syncthreads();
  tmem_cta = tmem_slot;
#endif
  // balanced_tpc() derives from the whole batch and from the host granularity alone,
  // never from the item, so the walk computes it once per CTA instead of once per work
  // index. tpc_cta stays 0 when there is nothing to balance - no splits, or no host
  // granularity to coarsen - and the body then falls back to the per-index call, so
  // the two layouts agree by construction.
  int tpc_cta = 0;
  if (p.n_splits > 1 && p.tiles_per_cta > 0) tpc_cta = MLA_TPC(p.tiles_per_cta);
  for (int w = w_first; w < w_limit; w += w_step) {
  // ==== begin work index body: indentation kept from the shipped layout
#endif
  // ------------------------------------------------- work item decomposition
#if MLA_PERSIST
  const int item = w / p.n_splits;
#else
  const int item = blockIdx.x;
#endif
  ItemInfo info;
  if (!resolve_item(p, item, info)) MLA_NEXT;
  const int batch = info.batch;
  const int q_start = info.q_start;
  const int kvlen = p.seq_lens[batch];
  const int row = qw * 32 + lane;
  const RowInfo ri = resolve_row(p, info, kvlen, row);
  const int row0 = ri.row0;
  const int valid_rows = ri.valid_rows;
  const int* table = p.block_tables + (size_t)batch * p.tstride;

  const int q_idx = ri.q_idx;
  const int head = ri.head;
  const bool row_valid = ri.row_valid;
  // Warp `qw` owns TMEM lanes [qw*32, qw*32+32) and therefore this tile's rows
  // [qw*32, qw*32+32). valid_rows is CTA-uniform (resolve_row derives it from
  // info.tile, not from row), so this is a warp-uniform comparison rather than a
  // vote. See MLA_EPI_SKIPWARP for why the drains care.
  const bool warp_rows_live = (valid_rows > qw * 32);
  const int visible = ri.visible;
  const int ntiles = ri.ntiles;
  const uint8_t* qbase = p.q + ((size_t)q_start * p.heads + row0) * KDIM;

  // KV split (flash decoding): this CTA owns tiles [t_begin, t_end). The split
  // count is per item, not global: one call can span a 64x KV range (shape 5),
  // and a single count either leaves the short requests idle or chops the long
  // ones into chunks that dominate the makespan. grid.y stays the worst-case
  // count; items that need fewer splits retire their extra CTAs here.
#if MLA_PERSIST
  const int split = w % p.n_splits;
#else
  const int split = (p.n_splits > 1) ? (int)blockIdx.y : 0;
#endif
  const int tpc_host = (p.n_splits > 1 && p.tiles_per_cta > 0) ? p.tiles_per_cta : ntiles;
  // balanced_tpc() only ever coarsens, so the host granularity bounds the split
  // count from above and CTAs past that bound can leave without walking the batch.
  if (split >= split_count_at(ntiles, tpc_host, p.n_splits)) MLA_NEXT;
#if MLA_PERSIST
  const int tpc = (p.n_splits > 1) ? (tpc_cta > 0 ? tpc_cta : MLA_TPC(tpc_host)) : ntiles;
#else
  const int tpc = (p.n_splits > 1) ? MLA_TPC(tpc_host) : ntiles;
#endif
  // Publish for the combine. Every split-0 CTA computes the same value, so the
  // redundant writes are identical.
  if (split == 0 && p.tpc_shared != nullptr) *p.tpc_shared = tpc;
  const int s_item = split_count_at(ntiles, tpc, p.n_splits);
  // Same reasoning as tpc: this CTA has already resolved this item's split count,
  // and a kernel boundary on the same stream makes the write visible to the
  // combine, which would otherwise pay a seq_lens round trip to recompute it.
  if (split == 0 && p.ns_shared != nullptr) p.ns_shared[item] = s_item;
  if (split >= s_item) MLA_NEXT;   // combine only reads the first s_item partials
  int t_begin = 0, t_end = ntiles;
  if (s_item > 1) {
    const int chunk = (ntiles + s_item - 1) / s_item;
    t_begin = split * chunk;
    t_end = min(ntiles, t_begin + chunk);
  }

  // A split with no keys publishes an empty partial and moves on to its next work
  // index. In the OFF arm this is before the TMEM allocation, so nothing has to be
  // deallocated on this path; in the ON arm the allocation is the CTA's, not the
  // item's, and is released once after the walk.
  if (t_begin >= t_end) {
    if (s_item > 1 && grp == 0 && row_valid && row < p.rows_max) {
      size_t base = ((size_t)item * p.n_splits + split) * p.rows_max + row;
      p.partial_ms[base] = NEG_INF;
      p.partial_ds[base] = 0.f;
    }
    MLA_NEXT;
  }

  // ------------------------------------------------------------- prologue
  mark(p, 0, 1);
#if MLA_PROLOGUE_ORDER
  load_q_tile(qbase, valid_rows, smq);
  if (kPadFull && valid_rows < M_TILE) zero_q_tail(smq, valid_rows);
  cp_async_commit();
  if (t_end > t_begin) load_kv_tile(p, table, kvlen, t_begin * NK, smkv[0]);
  cp_async_commit();
#if MLA_PERSIST >= 2
  // allocated once before the walk
#else
  if (tid < 32) {
    tmem_alloc(&tmem_slot, 512);
#if MLA_PERSIST
    // tcgen05.relinquish_alloc_permit makes any later tcgen05.alloc by this CTA
    // undefined, so MLA_PERSIST=1 - which allocates once per work index - cannot
    // relinquish here. Its single relinquish comes after the walk instead.
#else
    tmem_relinquish();
#endif
  }
#endif
  if (tid == 0) {
    mbar_init(smem_u32(bar_s), 1);
    mbar_init(smem_u32(bar_pv), 1);
#if MLA_PV_EARLY
    mbar_init(smem_u32(bar_pv2), 1);
#endif
    fence_barrier_init();
    fence_before();
  }
#else
#if MLA_PERSIST >= 2
  // allocated once before the walk
#else
  if (tid < 32) {
    tmem_alloc(&tmem_slot, 512);
#if !MLA_PERSIST
    tmem_relinquish();
#endif
  }
#endif
  if (tid == 0) {
    mbar_init(smem_u32(bar_s), 1);
    mbar_init(smem_u32(bar_pv), 1);
#if MLA_PV_EARLY
    mbar_init(smem_u32(bar_pv2), 1);
#endif
    fence_barrier_init();
    fence_before();
  }
  load_q_tile(qbase, valid_rows, smq);
  if (kPadFull && valid_rows < M_TILE) zero_q_tail(smq, valid_rows);
  cp_async_commit();
  if (t_end > t_begin) load_kv_tile(p, table, kvlen, t_begin * NK, smkv[0]);
  cp_async_commit();
#endif
  cp_async_wait_all();
  __syncthreads();
  fence_after();
  mark(p, 1, ntiles);

#if MLA_PERSIST >= 2
  const uint32_t tmem = tmem_cta;
#else
  const uint32_t tmem = tmem_slot;
#endif
  const uint64_t q_desc = make_desc(smem_u32(smq), BLOCK_BYTES, 0);
  const uint32_t idesc_qk = make_idesc(M_TILE, NK, 0);
  const uint32_t idesc_pv0 = make_idesc(M_TILE, 256, 1);       // dims [0,256)
  const uint32_t idesc_pv1 = make_idesc(M_TILE, DP - 256, 1);  // dims [256,DP)
  const uint32_t idesc_pv2 = make_idesc(M_TILE, DR, 1);        // dims [DP,512)

  float racc[DRG];
#pragma unroll
  for (int j = 0; j < DRG; ++j) racc[j] = 0.f;

  float m_shift = NEG_INF;
  float dsum = 0.f;
  uint32_t phase_s = 0, phase_pv = 0;
#if MLA_PV_EARLY
  uint32_t phase_pv2 = 0;
#endif
  bool first_pv = true;
  // The warp's first row decides the uniform branch selectors inside the loop.
  // Rows are (query, head) pairs laid out query-major, so q_idx rises with the
  // row and the visible range rises with it: the first row of a warp is the most
  // masked one. Padded rows count as unmasked (see MLA_PAD_FULL), which is why
  // this is the warp's first row and not its last.
  const int warp_row = row & ~31;
  const int warp_q_idx = (row0 + warp_row) / p.heads;
  const int warp_visible = kvlen - info.qlen + warp_q_idx;
  const bool warp_has_rows = !kSkipPad || (warp_row < valid_rows);
  int t_fast = t_begin;
  if (kUniformBranch) {
    const long long lim = (long long)warp_visible + 1 - NK;
    t_fast = (lim < 0) ? 0 : (int)(lim / NK) + 1;
    if (t_fast < t_begin) t_fast = t_begin;
    if (t_fast > t_end) t_fast = t_end;
  }

#if MLA_SLOT_AHEAD
  // Slot for tile t_begin+1; from here on the block-table LDG for tile t+2 is
  // issued at the top of tile t and consumed by tile t+1's prefetch.
  int slot_next =
      (t_begin + 1 < t_end) ? kv_slot(p, table, kvlen, (t_begin + 1) * NK) : 0;
#endif

  for (int t = t_begin; t < t_end; ++t) {
    const int buf = (t - t_begin) & 1;
    const int key0 = t * NK;
#if MLA_PV_EARLY == 1
    if (t > t_begin && pv_mode >= 1) {
      mbar_wait(smem_u32(bar_pv), phase_pv);
      phase_pv ^= 1;
    }
#endif
    if (!(kProbe & 16)) cp_async_wait_all();
    loop_sync();
    fence_after();
#if MLA_SLOT_AHEAD
    // First statement of the tile on purpose: everything below contains
    // memory-clobber asms, so the load cannot be sunk towards its use.
    const int slot_ahead = (t + 2 < t_end)
                               ? kv_slot_t<MLA_SLOT_AHEAD == 2>(p, table, kvlen, key0 + 2 * NK)
                               : 0;
#endif

    // ---- QK^T -> S in TMEM. Issued ahead of the prefetch below on purpose:
    // the prefetch depends on a block-table LDG whose latency used to sit
    // directly in front of the first MMA of the tile.
    if (tid == 0) {
      const uint64_t bd = make_desc(smem_u32(smkv[buf]), BLOCK_BYTES, 0);
#pragma unroll
      for (int ks = 0; ks < KSTEPS; ++ks) {
        if (skip_qk) break;
        uint32_t aoff = (uint32_t)((ks >> 1) * QRB * BLOCK_BYTES + (ks & 1) * 32);
        uint32_t boff = (uint32_t)((ks >> 1) * KRB * BLOCK_BYTES + (ks & 1) * 32);
        umma_f8(tmem + TCOL_S, q_desc + (aoff >> 4), bd + (boff >> 4), idesc_qk, (uint32_t)(ks != 0));
      }
      fence_before();
      if (!(kProbe & 64)) umma_commit(smem_u32(bar_s));
    }
    mark(p, 2, t + 1);
#if MLA_PV_EARLY == 2
    // Same ordering requirement as position 1 - ahead of the prefetch that reuses the B operand
    // buffer and ahead of the rescale - but after the QK issue, so this tile's first MMA is
    // already queued in the tensor pipe while chunks 0 and 1 of the previous tile finish and the
    // mbarrier round trip is not on the pipe's feed path.
    if (t > t_begin && pv_mode >= 1) {
      mbar_wait(smem_u32(bar_pv), phase_pv);
      phase_pv ^= 1;
    }
#endif
#if MLA_SLOT_ORDER
    // Issue the block-table load here and the cp.async burst after the MMA completion wait, so
    // the dependent global round trip overlaps the tensor pipe instead of sitting in front of
    // it. `slot_next` dies at the burst two statements later: unlike MLA_SLOT_AHEAD, nothing is
    // held live across the tile loop.
    const bool do_prefetch = (t + 1 < t_end) && !(kProbe & 32);
    const int slot_next = do_prefetch ? kv_slot(p, table, kvlen, key0 + NK) : 0;
    mbar_wait(smem_u32(bar_s), phase_s);
    phase_s ^= 1;
    fence_after();
    if (do_prefetch) {
      load_kv_slot(p.kv, slot_next, smkv[buf ^ 1], stub_loads);
      cp_async_commit();
    }
#else
    if (t + 1 < t_end && !(kProbe & 32)) {
#if MLA_SLOT_AHEAD
      load_kv_slot(p.kv, slot_next, smkv[buf ^ 1], stub_loads);
      slot_next = slot_ahead;
#else
      load_kv_slot(p.kv, kv_slot(p, table, kvlen, key0 + NK), smkv[buf ^ 1], stub_loads);
#endif
      cp_async_commit();
    }
    mbar_wait(smem_u32(bar_s), phase_s);
    phase_s ^= 1;
    fence_after();
#endif
    mark(p, 7, t + 1);

    // ---- softmax: two column groups of four warps over all M_TILE rows
    uint8_t* phi = smp[buf];
    uint8_t* plo = smp[buf] + P_BYTES;
    if (skip_softmax) {
      // ablation only: constant P, no TMEM read and no row maximum
#pragma unroll
      for (int h = 0; h < NKG / 16; ++h) {
        const int chunk16 = grp * (NKG / 16) + h;
        const uint32_t dh = smem_u32(phi + tile_offset(row, chunk16, QRB));
        asm volatile("st.shared.v4.b32 [%0], {%1,%1,%2,%1};" ::"r"(dh), "r"(0x38383838u), "r"(0));
        if (HILO) {
          const uint32_t dl = smem_u32(plo + tile_offset(row, chunk16, QRB));
          asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};" ::"r"(dl), "r"(0));
        }
      }
      dsum = (grp == 0) ? P_SCALE : 0.f;
      m_shift = 0.f;
    } else if (warp_has_rows) {
      // One branch per tile between two specializations of the same body, instead of
      // two branches inside it with the masked code interleaved between the fast maximum
      // and the fast P loop. t_fast is this warp's first masked tile, computed once per
      // work item above, so this is one compare against a register rather than the
      // nval/nloc/full/warp_nval chain that used to feed two branches.
#if MLA_MAX_EXCHANGE
      const SoftmaxCtx sctx{p,        tmem,       lane_off, col0,   grp,     row,
                            lane,     key0,       visible,  warp_visible,   row_valid,
                            warp_rows_live, has_dbg, phi,  plo,    m_shift, dsum,
                            racc,     first_pv,   sm_dsum};
#else
      const SoftmaxCtx sctx{p,        tmem,       lane_off, col0,   grp,     row,
                            lane,     key0,       visible,  warp_visible,   row_valid,
                            warp_rows_live, has_dbg, phi,  plo,    m_shift, dsum,
                            racc,     first_pv};
#endif
      if (t < t_fast)
        softmax_tile<HILO, true>(sctx);
      else
        softmax_tile<HILO, false>(sctx);
    }
    fence_before();            // order this tile's TMEM reads before the barrier
    fence_view_async_shared(); // publish the P stores to the async (UMMA) proxy
    mark(p, 3, t + 1);
    loop_sync();
    if (has_dbg_p && t == p.dbg_p_tile && blockIdx.x == 0) {
      const uint4* src = reinterpret_cast<const uint4*>(smp[buf]);
      uint4* dst = reinterpret_cast<uint4*>(p.dbg_p);
      for (int i = tid; i < (2 * P_BYTES) / 16; i += NTHREADS) dst[i] = src[i];
    }
    fence_after();             // acquire before issuing more tcgen05 work

    // ---- PV
    if (tid == 0) {
      const uint32_t kvb = smem_u32(smkv[buf]);
      const uint64_t pd = make_desc(smem_u32(smp[buf]), BLOCK_BYTES, 0);
      const uint64_t pld = HILO ? make_desc(smem_u32(smp[buf] + P_BYTES), BLOCK_BYTES, 0) : pd;
      // MN-major B: SBO = 8-key block stride, LBO = 64-dim atom stride
      const uint64_t v0 = make_desc(kvb, BLOCK_BYTES, KRB * BLOCK_BYTES);
      const uint64_t v1 = make_desc(kvb + 4 * KRB * BLOCK_BYTES, BLOCK_BYTES, KRB * BLOCK_BYTES);
      const uint64_t v2 = make_desc(kvb + (DP / SW) * KRB * BLOCK_BYTES, BLOCK_BYTES, KRB * BLOCK_BYTES);
      if (has_dbg && t == 0 && pv_mode >= 1) {
        mark(p, 16, (int)(pd & 0xffffffffu));
        mark(p, 17, (int)(pd >> 32));
        mark(p, 18, (int)(v0 & 0xffffffffu));
        mark(p, 19, (int)(v0 >> 32));
        mark(p, 20, (int)idesc_pv0);
        mark(p, 21, (int)tmem);
        mark(p, 22, (int)(pld & 0xffffffffu));
        mark(p, 23, (int)(v1 & 0xffffffffu));
        mark(p, 24, (int)(v2 & 0xffffffffu));
        mark(p, 25, (int)idesc_pv1);
        mark(p, 26, smem_u32(smkv[buf]));
        mark(p, 27, smem_u32(smp[buf]));
      }
      const uint32_t acc_init = first_pv ? 0u : 1u;
      uint32_t acc0 = acc_init, acc1 = acc_init;
#if MLA_PV_EARLY
      // Chunk 2 first, on its own barrier, so the drain below waits for these four MMAs only and
      // chunks 0 and 1 are still queued in the tensor pipe while the drain and the loop-end
      // barrier run. Same twelve MMAs, same destinations, same accumulate flags.
      if (pv_mode >= 3) {
#pragma unroll
        for (int ks = 0; ks < PV_KSTEPS; ++ks) {
          uint32_t poff = (uint32_t)(ks * 32) >> 4;
          uint32_t voff = (uint32_t)(ks * 4 * BLOCK_BYTES) >> 4;
          umma_f8(tmem + TCOL_S, pd + poff, v2 + voff, idesc_pv2, (uint32_t)(ks != 0));
          if (HILO) umma_f8(tmem + TCOL_S, pld + poff, v2 + voff, idesc_pv2, 1u);
        }
      }
      if (pv_mode >= 3) {
        fence_before();
        if (!(kProbe & 64)) umma_commit(smem_u32(bar_pv2));
      }
      if (pv_mode >= 1) {
#pragma unroll
        for (int ks = 0; ks < PV_KSTEPS; ++ks) {
          uint32_t poff = (uint32_t)(ks * 32) >> 4;
          uint32_t voff = (uint32_t)(ks * 4 * BLOCK_BYTES) >> 4;
          umma_f8(tmem + TCOL_O, pd + poff, v0 + voff, idesc_pv0, acc0);
          acc0 = 1u;
          if (HILO) umma_f8(tmem + TCOL_O, pld + poff, v0 + voff, idesc_pv0, 1u);
        }
      }
      if (pv_mode >= 2) {
#pragma unroll
        for (int ks = 0; ks < PV_KSTEPS; ++ks) {
          uint32_t poff = (uint32_t)(ks * 32) >> 4;
          uint32_t voff = (uint32_t)(ks * 4 * BLOCK_BYTES) >> 4;
          umma_f8(tmem + TCOL_O + 256, pd + poff, v1 + voff, idesc_pv1, acc1);
          acc1 = 1u;
          if (HILO) umma_f8(tmem + TCOL_O + 256, pld + poff, v1 + voff, idesc_pv1, 1u);
        }
      }
      if (pv_mode >= 1) {
        fence_before();
        if (!(kProbe & 64)) umma_commit(smem_u32(bar_pv));
      }
#else
      if (pv_mode >= 1) {
#pragma unroll
        for (int ks = 0; ks < PV_KSTEPS; ++ks) {
          uint32_t poff = (uint32_t)(ks * 32) >> 4;
          uint32_t voff = (uint32_t)(ks * 4 * BLOCK_BYTES) >> 4;
          umma_f8(tmem + TCOL_O, pd + poff, v0 + voff, idesc_pv0, acc0);
          acc0 = 1u;
          if (HILO) umma_f8(tmem + TCOL_O, pld + poff, v0 + voff, idesc_pv0, 1u);
        }
      }
      if (pv_mode >= 2) {
#pragma unroll
        for (int ks = 0; ks < PV_KSTEPS; ++ks) {
          uint32_t poff = (uint32_t)(ks * 32) >> 4;
          uint32_t voff = (uint32_t)(ks * 4 * BLOCK_BYTES) >> 4;
          umma_f8(tmem + TCOL_O + 256, pd + poff, v1 + voff, idesc_pv1, acc1);
          acc1 = 1u;
          if (HILO) umma_f8(tmem + TCOL_O + 256, pld + poff, v1 + voff, idesc_pv1, 1u);
        }
      }
      if (pv_mode >= 3) {
#pragma unroll
        for (int ks = 0; ks < PV_KSTEPS; ++ks) {
          uint32_t poff = (uint32_t)(ks * 32) >> 4;
          uint32_t voff = (uint32_t)(ks * 4 * BLOCK_BYTES) >> 4;
          umma_f8(tmem + TCOL_S, pd + poff, v2 + voff, idesc_pv2, (uint32_t)(ks != 0));
          if (HILO) umma_f8(tmem + TCOL_S, pld + poff, v2 + voff, idesc_pv2, 1u);
        }
      }
      if (pv_mode >= 1) {
        fence_before();
        if (!(kProbe & 64)) umma_commit(smem_u32(bar_pv));
      }
#endif
    }
    mark(p, 4, t + 1);
#if MLA_PV_EARLY
    if (pv_mode >= 3) {
      mbar_wait(smem_u32(bar_pv2), phase_pv2);
      phase_pv2 ^= 1;
    }
#else
    if (pv_mode >= 1) {
      mbar_wait(smem_u32(bar_pv), phase_pv);
      phase_pv ^= 1;
    }
#endif
    fence_after();
    mark(p, 10, t + 1);
    if (has_dbg_o && t == p.dbg_p_tile && blockIdx.x == 0) {
      const uint32_t daddr = tmem + lane_off + TCOL_O;
      for (int c = grp; c < 8; c += NGROUP) {
        uint32_t v[32];
        tmem_ld32(daddr + c * 32, v);
        tmem_ld_wait();
        for (int j = 0; j < 32; ++j)
          p.dbg_o[(size_t)row * 256 + c * 32 + j] = __uint_as_float(v[j]);
      }
    }
    first_pv = false;

    // Drain this tile's register-accumulated output dims: group `grp` owns the
    // DRG scratch columns holding dims [DP + grp*DRG, DP + (grp+1)*DRG).
    if (!(kProbe & 2) && (!kTdrSkipWarp || warp_rows_live)) {
      uint32_t v[DRG];
      tmem_ldg<DRG>(tmem + lane_off + TCOL_S + grp * DRG, v);
      if (!(kProbe & 128)) tmem_ld_wait();
#pragma unroll
      for (int j = 0; j < DRG; ++j) racc[j] += __uint_as_float(v[j]);
    }
    fence_before();
    // No barrier here: the next iteration's post-cp.async __syncthreads already
    // orders this drain ahead of the next QK, which rewrites the scratch cols.
  }
#if MLA_PV_EARLY
  // The last tile's PV chunks 0 and 1, which the loop deferred to the top of the next tile and
  // which the epilogue's TCOL_O drain now needs. The loop always ran at least once here: an item
  // with no tiles returned at the `t_begin >= t_end` check before the prologue.
  if (pv_mode >= 1) {
    mbar_wait(smem_u32(bar_pv), phase_pv);
    phase_pv ^= 1;
  }
  fence_after();
#endif

  mark(p, 5, 1);
  mark(p, 13, (int)tmem);
  // ------------------------------------------------------------- epilogue
  // The two groups hold complementary halves of every row, so the group-local
  // denominators are combined once here; each group then drains and stores its
  // own output dims.
  sm_dsum[grp * M_TILE + row] = dsum;
  fence_before();
  __syncthreads();
  fence_after();
#pragma unroll
  for (int g = 0; g < NGROUP; ++g)
    if (g != grp) dsum += sm_dsum[g * M_TILE + row];
  {
    const bool split_out = (p.n_splits > 1);
    const size_t part_base = split_out ? ((size_t)item * p.n_splits + split) * p.rows_max : 0;
    float* po = split_out ? (p.partial_o + part_base * kPartWords) : nullptr;
    // fp16 partials are stored normalised by this split's own denominator, which
    // keeps them inside the V convex hull; the combine re-weights by dsum.
    const float pnrm = (kPartFp16 && split_out && dsum > 0.f) ? (1.f / dsum) : 1.f;
    const float inv = (!split_out && dsum > 0.f) ? (p.bmm2 / dsum) : 0.f;
    float base_lse = LN2 * (m_shift + log2f(dsum) - LOG2_P_SCALE);
    float oscale = 1.f;
    if (!split_out && p.sinks != nullptr) {
      float sink = row_valid ? p.sinks[head] : 0.f;
      float hi = fmaxf(base_lse, sink);
      float total = hi + logf(ex2((base_lse - hi) * LOG2E) + ex2((sink - hi) * LOG2E));
      oscale = ex2((base_lse - total) * LOG2E);
      base_lse = total;
    }
    const float iscale = inv * oscale;
    __nv_bfloat16* orow =
        reinterpret_cast<__nv_bfloat16*>(p.out) + ((size_t)(q_start + q_idx) * p.heads + head) * DDIM;
    mark(p, 11, 1);
    // the TMEM drain is warp-synchronous; only the stores depend on row validity
    const uint32_t taddr = tmem + lane_off + TCOL_O;
    const bool epi_store = ((kProbe & 512) == 0) || has_dbg;
#if MLA_EPI_STAGE
    // Declared here rather than at the top of the kernel so that nothing extra stays
    // live across the tile loop: this kernel has twice paid several percent for exactly
    // that (MLA_PERSIST, MLA_SLOT_AHEAD).
    uint8_t* epi_stage = smem + OFF_Q;
    uint64_t* epi_rowdst =
        reinterpret_cast<uint64_t*>(smem + OFF_Q + M_TILE * kEpiStageStride);
    const bool stage_out = (valid_rows > kEpiStageMinRows);
#endif
    for (int c = grp; (kProbe & 256) == 0 && (!kEpiSkipWarp || warp_rows_live) && c < DP / 32;
         c += kEpiBatch * NGROUP) {
      uint32_t vv[kEpiBatch][32];
#pragma unroll
      for (int b = 0; b < kEpiBatch; ++b)
        if (c + b * NGROUP < DP / 32) tmem_ld32(taddr + (c + b * NGROUP) * 32, vv[b]);
      tmem_ld_wait();
#pragma unroll
      for (int b = 0; b < kEpiBatch; ++b) {
        const int cc = c + b * NGROUP;
        if (cc >= DP / 32) continue;
        const uint32_t* v = vv[b];
        if (has_dbg_o && blockIdx.x == 0 && cc < 8) {
          for (int j = 0; j < 32; ++j)
            p.dbg_o[128 * 256 + (size_t)row * 256 + cc * 32 + j] = __uint_as_float(v[j]);
          if (cc == 0 && row < 4 && has_dbg && grp == 0) {
            mark(p, 28 + row, __float_as_uint(dsum));
            mark(p, 32 + row, __float_as_uint(inv));
          }
        }
        if (row_valid) {
          if (split_out) {
#if MLA_EPI_STAGE
            part_store_either<32>(stage_out, epi_stage + (size_t)row * kEpiStageStride,
                                  po + (size_t)row * kPartWords, cc * 32, v, pnrm);
#else
            part_store<32>(po + (size_t)row * kPartWords, cc * 32, v, pnrm, p.dbg);
#endif
          } else if (kProbe & 1024) {
            __nv_bfloat16* dst = orow + cc * 32;
#pragma unroll
            for (int s = 0; s < 4; ++s)
              asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(dst + s * 8),
                           "r"(v[s * 4]), "r"(v[s * 4 + 1]), "r"(v[s * 4 + 2]), "r"(v[s * 4 + 3]));
          } else {
            __nv_bfloat162 packed[16];
#pragma unroll
            for (int j = 0; j < 16; ++j)
              packed[j] = __floats2bfloat162_rn(__uint_as_float(v[2 * j]) * iscale,
                                                __uint_as_float(v[2 * j + 1]) * iscale);
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
            __nv_bfloat16* dst = orow + cc * 32;
            if (epi_store) {
#pragma unroll
              for (int s = 0; s < 4; ++s)
#if MLA_EPI_STAGE
                epi_st16(stage_out, epi_stage + (size_t)row * kEpiStageStride +
                                        (size_t)(cc * 32 + s * 8) * 2,
                         reinterpret_cast<uint8_t*>(dst + s * 8), pw + s * 4);
#else
                asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(dst + s * 8),
                             "r"(pw[s * 4]), "r"(pw[s * 4 + 1]), "r"(pw[s * 4 + 2]),
                             "r"(pw[s * 4 + 3]));
#endif
            }
          }
        }
      }
    }
    if (row_valid && !(kProbe & 256)) {
      const int rdim0 = DP + grp * DRG;
      if (split_out) {
        {
#if MLA_EPI_STAGE
          part_store_either<DRG>(stage_out, epi_stage + (size_t)row * kEpiStageStride,
                                 po + (size_t)row * kPartWords, rdim0, racc, pnrm);
#else
          part_store<DRG>(po + (size_t)row * kPartWords, rdim0, racc, pnrm, p.dbg);
#endif
        }
        if (grp == 0) {
          p.partial_ms[part_base + row] = m_shift;
          p.partial_ds[part_base + row] = dsum;
        }
      } else {
        {
          __nv_bfloat162 packed[DRG / 2];
#pragma unroll
          for (int j = 0; j < DRG / 2; ++j)
            packed[j] = __floats2bfloat162_rn(racc[2 * j] * iscale, racc[2 * j + 1] * iscale);
          const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
          __nv_bfloat16* dst = orow + rdim0;
          if (epi_store) {
#pragma unroll
            for (int s = 0; s < DRG / 8; ++s)
#if MLA_EPI_STAGE
              epi_st16(stage_out,
                       epi_stage + (size_t)row * kEpiStageStride + (size_t)(rdim0 + s * 8) * 2,
                       reinterpret_cast<uint8_t*>(dst + s * 8), pw + s * 4);
#else
              asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(dst + s * 8),
                           "r"(pw[s * 4]), "r"(pw[s * 4 + 1]), "r"(pw[s * 4 + 2]),
                           "r"(pw[s * 4 + 3]));
#endif
          }
        }
        if (grp == 0 && p.lse != nullptr)
          p.lse[(size_t)(q_start + q_idx) * p.heads + head] = base_lse;
      }
    }
#if MLA_EPI_STAGE
    // Publish each row's destination once, from the group whose lanes own that row
    // index, so the copy below needs no per-row division. grp 0's four warps cover
    // rows 0..127 exactly once.
    if (stage_out && epi_store && row_valid && grp == 0)
      epi_rowdst[row] =
          (uint64_t)(split_out ? reinterpret_cast<uint8_t*>(po + (size_t)row * kPartWords)
                               : reinterpret_cast<uint8_t*>(orow));
    // Orders every thread's staging writes before every other thread's copy reads.
    // Plain ld.shared consumes them, so no async-proxy fence is needed here. Only the
    // staging path needs it: the direct path is already ordered by the barrier that
    // follows the whole block.
    if (stage_out) __syncthreads();
    if (stage_out && epi_store && !(kProbe & 256)) {
      // One warp per row per pass: 32 lanes x 16 B = 512 contiguous bytes, i.e. four
      // full 128-B lines per instruction, and DDIM*2/16/32 = 2 passes per row.
      for (int r = warp; r < valid_rows; r += NTHREADS / 32) {
        const uint8_t* src = epi_stage + (size_t)r * kEpiStageStride;
        uint8_t* dstg = reinterpret_cast<uint8_t*>(epi_rowdst[r]);
#pragma unroll
        for (int half = 0; half < DDIM * 2 / 16 / 32; ++half) {
          const int chunk = half * 32 + lane;
          uint32_t w[4];
          asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
                       : "r"(smem_u32(src + (size_t)chunk * 16)));
          asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(dstg + (size_t)chunk * 16),
                       "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]));
        }
      }
    }
#endif
    mark(p, 12, 1);
  }
  fence_before();
  __syncthreads();
  mark(p, 14, 1);
#if MLA_PERSIST >= 2
  // released once after the walk
#else
  if (tid < 32) tmem_dealloc(tmem, 512);
#endif
  mark(p, 6, 1);
#if MLA_PERSIST
  // ==== end work index body ==================================================
  }
#if MLA_PERSIST >= 2
  // The walk's own deallocation, once per CTA. The __syncthreads() above is the last
  // index's boundary and every TMEM read of every index completed before it (each
  // drain ends in tmem_ld_wait()), so the columns are idle here.
  if (tid < 32) tmem_dealloc(tmem_cta, 512);
#else
  // The walk's single relinquish: it has to follow the last tcgen05.alloc, which is
  // why it is not next to the allocation. Warp 0 only, .sync.aligned, exactly as the
  // shipped layout issues it.
  if (tid < 32) tmem_relinquish();
#endif
#endif
}

// Explicit AKA entry points: the shared device body preserves the arithmetic.
__global__ void __launch_bounds__(NTHREADS, 1) mla_attention_standard(Params p) {
  mla_v1_attention_body<false>(p);
}
__global__ void __launch_bounds__(NTHREADS, 1) mla_attention_hilo(Params p) {
  mla_v1_attention_body<true>(p);
}



} // namespace mla_v1
