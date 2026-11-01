#pragma once
#include "combine.cuh"


#ifndef MLA_CARVEOUT
#define MLA_CARVEOUT (-1)
#endif
constexpr int kCarveout = MLA_CARVEOUT;

// 1 applies the two split-policy corrections in mla_v1_launch (the two-wave grid
// cap and the 64-tile critical-path floor). 0 is the v1.33 policy and is the A/B
// control; job v44split measured the optima both were fitted to.
#ifndef MLA_SPLIT_WAVECAP
#define MLA_SPLIT_WAVECAP 1
#endif
constexpr bool kSplitWaveCap = (MLA_SPLIT_WAVECAP != 0);
#ifndef MLA_SPLIT_TRAFFIC_FP16
#define MLA_SPLIT_TRAFFIC_FP16 1
#endif
constexpr bool kSplitTrafficFp16 = (MLA_SPLIT_TRAFFIC_FP16 != 0);
#ifndef MLA_SPLIT_CAP
#define MLA_SPLIT_CAP 148
#endif

namespace {
// Raising MaxDynamicSharedMemorySize is a one-time, per-instantiation property
// write. It lives in a helper so the graph path in mla_api.cu can prewarm both
// instantiations before it records a launch: a cudaFuncSetAttribute call inside a
// stream capture is not something the replay path should depend on.
void set_smem_attr(int slot) {
  static bool attr_done[2] = {false, false};
  if (attr_done[slot]) return;
  auto kernel = slot ? mla_v1::mla_attention_hilo : mla_v1::mla_attention_standard;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mla_v1::SMEM_TOTAL);
  if (kCarveout >= 0)
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, kCarveout);
  attr_done[slot] = true;
}

void set_combine_attrs() {
  static bool done = false;
  if (done || kCarveout < 0) return;
  done = true;
  const int c = kCarveout;
  cudaFuncSetAttribute(mla_v1::mla_combine_d512,
                       cudaFuncAttributePreferredSharedMemoryCarveout, c);
  cudaFuncSetAttribute(mla_v1::mla_combine_d256,
                       cudaFuncAttributePreferredSharedMemoryCarveout, c);
  cudaFuncSetAttribute(mla_v1::mla_combine_d128,
                       cudaFuncAttributePreferredSharedMemoryCarveout, c);
  cudaFuncSetAttribute(mla_v1::mla_combine_d64,
                       cudaFuncAttributePreferredSharedMemoryCarveout, c);
  cudaFuncSetAttribute(mla_v1::mla_combine_d32,
                       cudaFuncAttributePreferredSharedMemoryCarveout, c);
}
}  // namespace

int mla_v1_smem_bytes() { return mla_v1::SMEM_TOTAL; }

void mla_v1_prewarm_attrs() {
  set_smem_attr(0);
  set_smem_attr(1);
}

void mla_v1_launch(const uint8_t* q,
                   const uint8_t* kv,
                   const int* block_tables,
                   const int* seq_lens,
                   const int* cum_q,
                   uint8_t* out,
                   float* lse,
                   const float* sinks,
                   int heads,
                   int uniform_q,
                   long long rows,
                   int batch,
                   int page_shift,
                   int tstride,
                   float qk_scale,
                   float bmm2,
                   int hilo,
                   int pv_mode,
                   int skip_qk,
                   int skip_softmax,
                   int stub_loads,
                   uint8_t* dbg_p,
                   float* dbg_o,
                   int dbg_p_tile,
                   int* dbg,
                   long long items,
                   float* workspace,
                   long long workspace_floats,
                   int split_request,
                   int rows_max,
                   int max_seq_len,
                   cudaStream_t stream) {
  mla_v1::Params p;
  p.q = q;
  p.kv = kv;
  p.block_tables = block_tables;
  p.seq_lens = seq_lens;
  p.cum_q = cum_q;
  p.out = out;
  p.lse = lse;
  p.sinks = sinks;
  p.heads = heads;
  p.uniform_q = uniform_q;
  p.page_shift = page_shift;
  p.tstride = tstride;
  p.batch = batch;
  p.qk_scale = qk_scale;
  p.bmm2 = bmm2;
  p.pv_mode = pv_mode;
  p.skip_qk = skip_qk;
  p.skip_softmax = skip_softmax;
  p.stub_loads = stub_loads;
  p.dbg_p = dbg_p;
  p.dbg_o = dbg_o;
  p.dbg_p_tile = dbg_p_tile;
  p.dbg = dbg;

  constexpr int kSplitCap = MLA_SPLIT_CAP;   // one wave; more splits only add waves
  constexpr int kGridCap = 2048;      // bounds items*splits, i.e. the empty CTAs
  constexpr int kDefaultCtaTarget = 148;
  // No CTA should serialise more than this many 64-key tiles: 4096 keys, ~143 us at
  // the measured 2.24 us per tile, which no saving in wave count pays for. See the
  // v44split comment below for the sweep this came from; 28..85 all pick the same
  // split count on the six decode geometries.
  constexpr int kMaxTilesPerCta = 64;
  const long long rows_cap = rows_max > 0 ? rows_max : mla_v1::M_TILE;
  const long long items_safe = items > 0 ? items : 1;
  int tiles_cap = max_seq_len > 0 ? (int)((max_seq_len + mla_v1::NK - 1) / mla_v1::NK) : 1;
  if (tiles_cap < 1) tiles_cap = 1;
  int splits;
  // Splitting exists to fill the machine, so it is pointless once the work items
  // alone cover every SM. Measured on this card: shape 2 (384 items) went
  // 0.4609 -> 1.0668 ms and shape 8 (386 items) 2.9379 -> 3.4893 ms the moment a
  // wider partial buffer let by_ws admit a second split, purely from publishing
  // and re-reading ~100 MB of partials plus one more combine launch.
  if (workspace == nullptr || workspace_floats <= 0 ||
      items_safe >= (long long)kDefaultCtaTarget) {
    splits = 1;
  } else {
    const int cta_target = split_request < 0 ? (int)(-split_request)
                                             : (split_request > 0 ? 0 : kDefaultCtaTarget);
    // -1 for the word the device publishes its chosen tiles_per_cta in, plus one
    // more word per item when it also publishes the resolved split counts. Both
    // live past the last partial_ds word, so they have to come out of the scratch
    // the split policy is allowed to hand out or an explicit --splits request
    // could push the publication past the end of the caller's workspace_buffer.
    const long long tail_words = 1 + (mla_v1::kCombPubNs ? items_safe : 0);
    long long by_ws =
        (workspace_floats - tail_words) / (rows_cap * (mla_v1::kPartWords + 2) * items_safe);
    // grid.y is shared by every item, so items*splits also bounds how many CTAs
    // launch, retire immediately and still pay a full 188 KiB SMEM allocation.
    long long by_grid = (kGridCap + items_safe - 1) / items_safe;
    long long by_ctas = cta_target > 0 ? cta_target : kSplitCap;
    const double traffic_num = 0.5 * (double)mla_v1::KDIM;
    const double traffic_den =
        kSplitTrafficFp16 ? (double)(mla_v1::kPartWords * 4) : (double)(mla_v1::DDIM * 4);
    long long by_traffic =
        (long long)(traffic_num / traffic_den * (double)max_seq_len / (double)rows_cap);
    long long cap = by_ws;
    if (by_grid < cap) cap = by_grid;
    if (by_ctas < cap) cap = by_ctas;
    if (by_traffic < cap) cap = by_traffic;
    if (cap > kSplitCap) cap = kSplitCap;
    long long chosen = cap;
    if (kSplitWaveCap) {
      const long long by_wave = (2 * mla_v1::kSplitWaveTarget) / items_safe;
      if (by_wave >= 1 && by_wave < chosen) chosen = by_wave;
      const long long path_min = ((long long)tiles_cap + kMaxTilesPerCta - 1) / kMaxTilesPerCta;
      if (path_min > chosen) chosen = path_min;
      // by_grid bounds how many CTAs can be published at all and by_ws bounds what
      // fits in the caller's scratch, so both stay hard; by_traffic and by_ctas are
      // the heuristics the floor is allowed to overrule.
      if (chosen > by_grid) chosen = by_grid;
      if (chosen > by_ws) chosen = by_ws;
      if (chosen > kSplitCap) chosen = kSplitCap;
      if (chosen < 1) chosen = 1;
    }
    splits = split_request > 0 ? (int)split_request : (int)chosen;
    // An explicit request still has to fit the caller's scratch: the partial
    // buffers are strided by grid.y, so exceeding by_ws would write past the
    // workspace_buffer view the checker byte-compares afterwards.
    if (splits > by_ws) splits = (int)by_ws;
    if (splits > mla_v1::kMaxSplits) splits = mla_v1::kMaxSplits;
    if (splits < 1) splits = 1;
  }
  int tiles_per_cta = (tiles_cap + splits - 1) / splits;
  if (tiles_per_cta < 1) tiles_per_cta = 1;
  p.n_splits = splits;
  p.tiles_per_cta = splits > 1 ? tiles_per_cta : 0;
  p.rows_max = (int)rows_cap;
  // Every (item, split) pair this launch covers. In the historical layout that is
  // exactly grid.x * grid.y; in the persistent layout it is the length of the work
  // list the wave of CTAs walks round robin. items <= sum(Q)*heads/M_TILE <= 16384
  // and splits <= kMaxSplits, so this cannot overflow an int.
  p.n_work = (int)((long long)items * splits);
  if (splits > 1) {
    const long long scalars = (long long)items * splits * rows_cap;
    p.partial_o = workspace;
    p.partial_ms = workspace + scalars * mla_v1::kPartWords;
    p.partial_ds = p.partial_ms + scalars;
    p.tpc_shared = reinterpret_cast<int*>(p.partial_ds + scalars);
    // One word per item, immediately after the single tiles_per_cta word.
    p.ns_shared = mla_v1::kCombPubNs ? p.tpc_shared + 1 : nullptr;
  } else {
    p.partial_o = nullptr;
    p.partial_ms = nullptr;
    p.partial_ds = nullptr;
    p.tpc_shared = nullptr;
    p.ns_shared = nullptr;
  }

  set_smem_attr(hilo ? 1 : 0);
  set_combine_attrs();
  auto kernel = hilo ? mla_v1::mla_attention_hilo : mla_v1::mla_attention_standard;
  dim3 grid((unsigned)items, (unsigned)splits);
#if MLA_PERSIST
  {
    // One wave of CTAs, because 188416 B of dynamic SMEM leaves room for exactly one
    // per SM: launching more would just recreate the dead CTAs this removes. A work
    // list shorter than a wave launches only as many CTAs as there is work for.
    long long g = p.n_work;
    if (g > mla_v1::kSplitWaveTarget) g = mla_v1::kSplitWaveTarget;
    if (g < 1) g = 1;
    grid = dim3((unsigned)g, 1);
  }
#endif
  kernel<<<grid, mla_v1::NTHREADS, mla_v1::SMEM_TOTAL, stream>>>(p);
  if (splits > 1) {
    const long long base = items_safe * rows_cap;
    int slabs = 1;
    while (slabs < mla_v1::DDIM / mla_v1::kCombMinDims &&
           base * slabs < mla_v1::kCombTargetCtas)
      slabs <<= 1;
    dim3 cgrid((unsigned)items, (unsigned)rows_cap, (unsigned)slabs);
    switch (mla_v1::DDIM / slabs) {
      case 512:
        mla_v1::mla_combine_d512<<<cgrid, 128, 0, stream>>>(p);
        break;
      case 256:
        mla_v1::mla_combine_d256<<<cgrid, 128, 0, stream>>>(p);
        break;
      case 128:
        mla_v1::mla_combine_d128<<<cgrid, 128, 0, stream>>>(p);
        break;
      case 64:
        mla_v1::mla_combine_d64<<<cgrid, 128, 0, stream>>>(p);
        break;
      case 32:
        mla_v1::mla_combine_d32<<<cgrid, 128, 0, stream>>>(p);
        break;
      case 16:
        mla_v1::mla_combine_d16<<<cgrid, 128, 0, stream>>>(p);
        break;
      default:
        mla_v1::mla_combine_d8<<<cgrid, 128, 0, stream>>>(p);
        break;
    }
  }
}
