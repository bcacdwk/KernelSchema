// Shared ABI struct between fmha_kernel.cu (mma.sync path), fmha_tc.cu
// (tcgen05 path) and solution.py (ctypes packer). Keep in sync exactly.
#pragma once
#include <cstdint>

struct FmhaParams {
  const void* q;
  const void* k;
  const void* v;
  void* out;
  float* lse;
  const int32_t* table;
  const int32_t* seq_lens;
  const int32_t* cu_q;
  const float* sinks;
  float* part_o;
  float* part_ml;
  int32_t* counters;
  int64_t q_tok_stride, q_head_stride;
  int64_t out_tok_stride, out_head_stride;
  int64_t lse_tok_stride;
  int64_t k_page_stride, k_head_stride, k_slot_stride;
  int64_t v_page_stride, v_head_stride, v_slot_stride;
  int64_t sink_stride;
  int32_t batch, heads, kvh, group, total_q;
  int32_t dq, dv, page_log, page_mask, tstride;
  int32_t window_left, causal, has_lse, has_sinks;
  int32_t m_tile, n_tile, splits_ub, chunk_round, tiles_ub;
  int32_t max_kv_len;
  int32_t combine_z;
  int32_t _pad2;
  // Minimum split-KV chunk in KEYS (0 = disabled). split_range derives the
  // chunk from each tile's own length, so with heterogeneous kv_lengths the
  // blocks of one launch differ by up to max_kv/min_kv in work; the floor makes
  // the per-block work uniform, which is what the wave scheduler needs.
  int32_t chunk_floor;
  // Balanced per-tile splits (join-20261022 arm-b mechanism; adopted only
  // after own same-job A/B): when > 0, each tile uses
  // min(splits_ub, ceil(L / chunk_target)) splits of ~chunk_target tokens
  // (chunk derived per tile, chunk_floor bypassed) so CTAs of unequal-length
  // tiles finish together. 0 = classic uniform chunk_floor path.
  int32_t chunk_target;   // keeps sizeof a multiple of 8 (24 x int32)
  float qk_scale2;   // bmm1_scale * log2(e)
  float v_scale;     // bmm2_scale
  float log2e;
  float lazy_thresh; // base-2 max growth below which O rescale is deferred
};
