// paged_latent_mxfp8_nvfp4 - independently authored CUDA kernel for L20D (SM103).
//
// Frozen contract (problem/reference.py + problem/metadata.json):
//   q bf16 [B,S,H,512]; MXFP8 SWA cache uint8 [P,page,1,528] = 512 x e4m3 plus
//   16 ue8m0 group scales (2^(s-127), one per 32 values); NVFP4 compressed cache
//   uint8 [P,page,1,288] = 256 bytes holding 512 e2m1 nibbles (low nibble is the
//   even element) plus 32 e4m3 scales (one per 16 values); int32 slot selectors
//   with -1 markers; fp32 per-head attn_sink; caller-owned bf16 out returned by
//   identity; fresh fp32 lse returned as the [B,H,S] transpose of a [B,S,H]
//   buffer. out = exp(logits - logaddexp(lse, sink)) @ K, lse = logsumexp(logits).
//
// Everything numerical happens in this kernel: ordered selector compaction,
// in-kernel dequantisation of both cache formats, the QK^T GEMM, base-2 online
// softmax carrying the running row maximum, the PV GEMM, the sink rescale and the
// bf16 / lse stores. No library GEMM or attention call, no host-side numerics.
//
// All physical heads of one request share one KV set (D512/V512 MQA), so a
// request is a real GEMM pair with M = heads. One CTA owns HB heads of one (b,s)
// row and streams the selected token list in NC-token chunks. Both GEMMs use
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32. Decoded operands are EXACT
// in fp16 (e4m3 x ue8m0 needs 4 significand bits, e2m1 x e4m3 needs 7), which is
// the same observation recorded by the SM100 NVFP4 decode route in
// references/native-sources (FlashInfer, Apache-2.0); q is bf16 and also exact in
// fp16 in this domain, so the QK^T products match the reference's fp32 GEMM
// bit-for-bit and only summation order differs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <c10/cuda/CUDAGuard.h>

#include "common.cuh"

// MMA fallback: shared source instantiated for 32- and 64-head CTAs.
// Host dispatch below preserves the accepted geometry selection.
#define PLMN_CFG cfg32
#define PLMN_HB 32
#define PLMN_NT 256
#define PLMN_KERNEL plmn_kernel_hb32
#define PLMN_COMB_WIDE plmn_combine_wide_hb32
#define PLMN_COMB_NARROW plmn_combine_narrow_hb32
#include "kernel_mma.cuh"
#undef PLMN_CFG
#undef PLMN_HB
#undef PLMN_NT
#undef PLMN_KERNEL
#undef PLMN_COMB_WIDE
#undef PLMN_COMB_NARROW

#define PLMN_CFG cfg64
#define PLMN_HB 64
#define PLMN_NT 512
#define PLMN_KERNEL plmn_kernel_hb64
#define PLMN_COMB_WIDE plmn_combine_wide_hb64
#define PLMN_COMB_NARROW plmn_combine_narrow_hb64
#include "kernel_mma.cuh"
#undef PLMN_CFG
#undef PLMN_HB
#undef PLMN_NT
#undef PLMN_KERNEL
#undef PLMN_COMB_WIDE
#undef PLMN_COMB_NARROW

#include "kernel_tcgen.cuh"

// The split partials are sized with the frozen head dim from common.cuh; both
// geometries must agree with it and with each other.
static_assert(cfg32::CFG_D == plmn::FwdCtx_D && cfg64::CFG_D == plmn::FwdCtx_D,
              "kernel and host disagree on the frozen head dim");
static_assert(cfg32::CFG_HB == 32 && cfg32::CFG_NT == 256, "cfg32 geometry changed");
static_assert(cfg64::CFG_HB == 64 && cfg64::CFG_NT == 512, "cfg64 geometry changed");

namespace {

// Queried once per host thread/device. smem_per_sm only feeds the resident-CTA estimate;
// the fallback keeps the pre-query behaviour of assuming the shipped geometry.
void dev_props(int listn, int& nsm, int& smem_per_sm) {
  static thread_local int cached[2] = {0, 0};
  static thread_local int cached_device = -1;
  int dev=at::cuda::getCurrentCUDAStream().device_index();
  if (cached[0] == 0 || cached_device != dev) {
    cached_device=dev;
    C10_CUDA_CHECK(
        cudaDeviceGetAttribute(&cached[0], cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(
        cudaDeviceGetAttribute(&cached[1], cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    if (cached[0] <= 0) cached[0] = 1;
    if (cached[1] <= 0) cached[1] = 2 * cfg32::cfg_smem_bytes(listn);
  }
  nsm = cached[0];
  smem_per_sm = cached[1];
}

bool use_hb64_shape(const plmn::FwdCtx& c, const cfg32::Schedule& s32, int& splits64) {
  splits64 = 0;
  if (c.H % cfg64::CFG_HB != 0) return false;

  if (c.nest < c.listn) return false;
  const cfg64::Schedule s64 = cfg64::best_one_round(c);
  if (s64.splits == 0) return false;
  if (s64.depth > s32.depth) return false;
  if (5 * s32.sms >= 6 * s64.sms) return false;
  splits64 = s64.splits;
  return true;
}

}  // namespace

int plmn_smem_bytes(int listn) { return cfg32::cfg_smem_bytes(listn); }
int plmn_smem_bytes_hb64(int listn) { return cfg64::cfg_smem_bytes(listn); }
// Diagnostics binding: reports the geometry choice for a bare (B*S, H, listn)
// triple. plmn_forward does NOT call this; it calls use_hb64_shape with the
// context it has already built.
bool plmn_use_hb64(int64_t BS, int64_t H, int64_t listn, int64_t nest) {
  int nsm = 0, smem_per_sm = 0;
  dev_props((int)listn, nsm, smem_per_sm);
  plmn::FwdCtx c{};
  c.H = (int)H;
  c.BS = (int)BS;
  c.listn = (int)listn;
  c.nest = (int)std::max<int64_t>(1, std::min<int64_t>(listn, nest));
  c.nsm = nsm;
  c.smem_per_sm = smem_per_sm;
  int splits64 = 0;
  return use_hb64_shape(c, cfg32::schedule(c), splits64);
}

// Returns only the fresh lse view. The Python wrapper returns the caller's own
// ``out`` object so the harness's returned-to-input identity alias check sees the
// exact same PyObject the reference returns, and the (out, lse) tuple shape is
// built in Python where the strict type check runs.
torch::Tensor plmn_forward(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor indices,
    c10::optional<torch::Tensor> extra_k_cache, c10::optional<torch::Tensor> extra_indices,
    double sm_scale, torch::Tensor attn_sink, torch::Tensor out) {
  c10::cuda::CUDAGuard device_guard(q.device());
  TORCH_CHECK(q.dim() == 4, "q must be [B,S,H,D]");
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(), "q/out must be contiguous");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(q.sizes() == out.sizes(), "out must match q");
  TORCH_CHECK(q.is_cuda(), "cuda tensors required");
  const int B = q.size(0), S = q.size(1), H = q.size(2), Dh = q.size(3);
  TORCH_CHECK(Dh == plmn::FwdCtx_D, "only D=512 is frozen for this operator");
  TORCH_CHECK(H % cfg32::CFG_HB == 0, "physical heads must be a multiple of 32");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(2) == 1, "k_cache must be [P,page,1,rec]");
  TORCH_CHECK(k_cache.is_contiguous() && k_cache.scalar_type() == torch::kUInt8,
              "k_cache must be contiguous uint8");
  TORCH_CHECK(indices.dim() == 3 && indices.scalar_type() == torch::kInt && indices.is_contiguous(),
              "indices must be contiguous int32 [B,S,budget]");
  TORCH_CHECK(indices.size(0) == B && indices.size(1) == S, "indices must match q's B,S");
  TORCH_CHECK(attn_sink.dim() == 1 && attn_sink.size(0) == H &&
                  attn_sink.scalar_type() == torch::kFloat && attn_sink.is_contiguous(),
              "attn_sink must be contiguous fp32 [H]");
  const int page1 = k_cache.size(1), rec1 = k_cache.size(3);
  const int64_t npages1 = k_cache.size(0);
  TORCH_CHECK(rec1 == 528 || rec1 == 288, "unsupported SWA record size");
  TORCH_CHECK(k_cache.numel() < (int64_t(1) << 31), "SWA cache exceeds the packed offset width");
  const int budget1 = indices.size(2);

  const bool has2 = extra_k_cache.has_value() || extra_indices.has_value();
  TORCH_CHECK(extra_k_cache.has_value() == extra_indices.has_value(),
              "extra cache and extra indices must be provided together");
  const unsigned char* k2p = nullptr;
  const int* idx2p = nullptr;
  int page2 = 1, rec2 = 528, budget2 = 0;
  int64_t npages2 = 2;
  if (has2) {
    torch::Tensor ek = extra_k_cache.value(), ei = extra_indices.value();
    TORCH_CHECK(ek.dim() == 4 && ek.size(2) == 1 && ek.is_contiguous() &&
                    ek.scalar_type() == torch::kUInt8,
                "extra_k_cache must be [P,page,1,rec] contiguous uint8");
    TORCH_CHECK(ei.dim() == 3 && ei.scalar_type() == torch::kInt && ei.is_contiguous() &&
                    ei.size(0) == B && ei.size(1) == S,
                "extra_indices must be contiguous int32 [B,S,budget]");
    TORCH_CHECK(ek.size(3) == 288 || ek.size(3) == 528, "unsupported compressed record size");
    TORCH_CHECK(ek.numel() < (int64_t(1) << 31), "compressed cache exceeds the packed offset width");
    page2 = ek.size(1);
    rec2 = ek.size(3);
    npages2 = ek.size(0);
    budget2 = ei.size(2);
    k2p = reinterpret_cast<const unsigned char*>(ek.data_ptr());
    idx2p = reinterpret_cast<const int*>(ei.data_ptr());
  }
  const int listn = budget1 + budget2;

  // ---- how many tokens a request can actually select ----
  // The split rule and the geometry choice both need the number of NC-token
  // chunks a request will stream, but the selector WIDTH (budget1 + budget2) is
  // only an upper bound on it: problem/input.py fills ids with -1 beyond the
  // tokens visible to that request, so a short context selects a handful of
  // tokens out of a 640-wide list. Planning on the width therefore hands out
  // splits whose CTAs find c_first >= c_last and do nothing but publish an
  // identity partial, which the combine then reads back: domain shape d0
  // (lengths=[1], one real chunk) was being launched with splits=10, i.e. nine
  // empty CTAs and a ten-way combine per request row.
  //
  // The page count is a tensor SHAPE, so reading it costs nothing, and it bounds
  // the token count: the two caches together hold (P - 2) pages of `page` tokens
  // summed over all B requests (input.py: pages = sum(counts) + 2), so no single
  // request can see more than (P - 2) * page of them, and never more than its
  // own budget. Both bounds are applied per cache and summed, then clamped to
  // the selector width. This is an UPPER bound on the real count, so it can only
  // ever over-split - never under-split - and on every long-context shape it
  // saturates at the budget, where the rule then behaves exactly as before.
  // Verified against the frozen ten and the predeclared ten (N_est >= N_true on
  // all twenty, and equal on cases 2/3/8/9 and d7/d8): it tightens maxchunk only
  // where the width was wrong, on case 8 (10 -> 3) and on d0, d1, d4, d5, d6, d7.
  auto token_bound = [](int64_t pages, int page, int budget) -> int {
    if (pages <= 2 || page <= 0 || budget <= 0) return 0;
    const int64_t toks = (pages - 2) * (int64_t)page;
    return (int)std::min<int64_t>((int64_t)budget, toks);
  };
  const int nest = std::max(1, std::min(listn, token_bound(npages1, page1, budget1) +
                                                token_bound(npages2, page2, budget2)));

  const int BS = B * S;
  int nsm = 0, smem_per_sm = 0;
  dev_props(listn, nsm, smem_per_sm);

  auto lse_base = torch::empty_strided({B, H, S}, {(int64_t)S * H, 1, H},
                                       q.options().dtype(torch::kFloat32));

  plmn::FwdCtx c;
  c.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  c.k1 = reinterpret_cast<const unsigned char*>(k_cache.data_ptr());
  c.idx1 = reinterpret_cast<const int*>(indices.data_ptr());
  c.k2 = k2p;
  c.idx2 = idx2p;
  c.sink = reinterpret_cast<const float*>(attn_sink.data_ptr());
  c.sm_scale = (float)sm_scale;
  c.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  c.lse = reinterpret_cast<float*>(lse_base.data_ptr());
  c.H = H;
  c.BS = BS;
  c.budget1 = budget1;
  c.budget2 = budget2;
  c.page1 = page1;
  c.rec1 = rec1;
  c.page2 = page2;
  c.rec2 = rec2;
  c.listn = listn;
  c.nest = nest;
  c.nsm = nsm;
  c.smem_per_sm = smem_per_sm;

  const cfg32::Schedule s32 = cfg32::schedule(c);
  int splits64 = 0;
  const bool use64 = use_hb64_shape(c, s32, splits64);

  const int nt = use64 ? cfg64::CFG_NT : cfg32::CFG_NT;
  const int maxent = use64 ? cfg64::CFG_MAXENT : cfg32::CFG_MAXENT;
  TORCH_CHECK((budget1 + nt - 1) / nt <= maxent && (budget2 + nt - 1) / nt <= maxent,
              "selector budget exceeds the compiled compaction width");

  const int chunks = std::max(1, (listn + 63) / 64);
  // An H=128 request row can be served by one 128-head CTA or by a pair of 64-head
  // CTAs (gridDim.z). The pair stages half the Q (64 KB instead of 128 KB),
  // accumulates half the output tile and therefore costs F 7.4 us instead of
  // 10.2 us and 3.12 us per chunk instead of 4.22 - and because a row now takes
  // two CTAs, the same one-wave budget buys only half the splits, which halves the
  // split partials and their merge. On the small-batch shapes that trade is worth
  // much more than the duplicated gather: measured on this card (L20D/SM103,
  // 2032 MHz, NCU gpu__time_duration.sum, median of 10 captures;
  // evidence/online-20261024/raw/{cbase-1,cnhb2-1,sweep2})
  //   BS=3  20.976 -> 15.968   BS=8  20.752 -> 18.912
  //   BS=12 20.848 -> 19.328   BS=32 28.656 -> 28.464
  // but BS=48 leaves only one split per pair and costs 34.624 -> 44.672, so the
  // pair is used exactly while it still leaves room for two splits of the row.
  const int tc_nhb = (H == 128 && (long)BS * 4 <= nsm) ? 2 : 1;
  // A split TCGEN wave must fill at least one third of the SMs. Small batches can
  // still meet this condition when the sparse list offers enough independent tiles,
  // and the paired geometry doubles the tile count of an H=128 row.
  const bool use_tc = H <= 128 && listn >= 128 &&
                      BS * tc_nhb * std::min(16, chunks) >= (nsm + 2) / 3;
  // Split fan-out. The deepest any CTA goes is ceil(chunks/splits), so the widest
  // fan-out that still fits one wave wins on the main kernel; but every extra split
  // is one more partial per row to write and to merge, and a split count that does
  // not divide the chunk count leaves straggler CTAs that the merged tail spins on.
  // So among the counts that reach the minimum depth, take the largest one that
  // divides the chunk count evenly. Measured on this card with the batched merge
  // (20 NCU captures, medians; evidence/online-20261024/raw/rp-*, raw/rp2-*):
  //   shape4 (BS=8,  nhb=2, 10 chunks) 18.400 at 5 splits vs 18.688 / 18.976 /
  //     19.024 / 18.784 at 6 / 7 / 8 / 9
  //   shape5 (BS=12, nhb=2)            18.560 at 5 vs 18.912 at 6, 22.032 at 4
  //   shape7 (BS=24, nhb=1)            18.720 at 5 vs 18.896 at 6
  //   shape1 (BS=48, smax 3, no even divisor in range) stays at 3: 25.472 vs
  //     27.696 for 2 splits.
  const int max_splits = std::min(16, std::min(chunks, std::max(1, nsm / (BS * tc_nhb))));
  int splits = use_tc ? max_splits : (use64 ? splits64 : s32.splits);
  if (use_tc) {
    const int depth = (chunks + max_splits - 1) / max_splits;
    for (int s = max_splits; s >= 1; --s) {
      if ((chunks + s - 1) / s != depth) continue;   // a deeper CTA costs more than the traffic
      if (chunks % s == 0) {
        splits = s;
        break;
      }
    }
  }

  // Grow-only internal scratch for the split partials. It is fully overwritten
  // by every call that uses it, holds no answer across calls, and keeping it
  // alive removes two allocations per forward from the timed path (the public
  // runner times the whole forward, so host-side cost is measured too). It is
  // shared by both geometries because only one of them runs per call.
  // Private scratch per host thread, device and CUDA stream. Concurrent streams
  // must not race on the split partials; no previous input/output is cached.
  struct Scratch {
    torch::Tensor part_o, part_ml, counters;
  };
  static thread_local std::map<std::pair<int, uintptr_t>, Scratch> scratch_by_stream;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto& scratch = scratch_by_stream[{q.get_device(), reinterpret_cast<uintptr_t>((cudaStream_t)stream)}];
  auto& part_o = scratch.part_o;
  auto& part_ml = scratch.part_ml;
  __half* pop = nullptr;
  float* pml = nullptr;
  if (splits > 1 || use_tc) {
    const long need_o = (long)splits * BS * H * plmn::FwdCtx_D;
    const long need_ml = (long)splits * BS * H * 2;
    if (!part_o.defined() || part_o.numel() < need_o)
      part_o = torch::empty({need_o}, q.options().dtype(torch::kHalf));
    if (!part_ml.defined() || part_ml.numel() < need_ml)
      part_ml = torch::empty({need_ml}, q.options().dtype(torch::kFloat32));
    pop = reinterpret_cast<__half*>(part_o.data_ptr());
    pml = reinterpret_cast<float*>(part_ml.data_ptr());
  }

  // Fold the split-KV merge into the TCGEN main kernel when that is measured to be
  // the cheaper of the two ways to pay for the split partials.
  //
  // Safety: the fused tail spins on its row peers, so every split of a row must be
  // resident at once -- one CTA per SM (the block's dynamic shared memory is more
  // than half an SM's) and a grid no larger than the SM count. The split rule
  // already caps BS*splits at nsm; both conditions are re-checked here because the
  // guarantee, not the heuristic, is what makes the spin safe.
  //
  // Benefit: fusing removes the second kernel's ~2 us measured floor outright, and
  // it removes one split's worth of partial traffic, because a CTA keeps its own
  // contribution in shared memory instead of writing it out and reading it back.
  // What it can cost is merge parallelism: the fused merge runs on BS*splits
  // blocks of 512 threads where the standalone combine runs on BS*H/4 blocks of
  // 128, and it merges ceil(H/(16*splits)) serial rounds per CTA.
  //
  // Which side wins depends on how big a row's partial footprint is, and the two
  // head counts of this operator's ABI sit on opposite sides of it. Measured on
  // L20D/SM103, same card, ten alternating blocks, geometric mean of both orders
  // (raw/accp17 unfused, raw/accp18 fused, same main kernel either way):
  //   * the 128-head template wins fused at every split count, because a row's
  //     partials are 128 KB per split: splits=3 shape0 1.6068 -> 1.5533,
  //     splits=4 shape9 1.3262 -> 1.3040, splits=10 shape4 0.9444 -> 0.9509 and
  //     shape5 0.9661 -> 0.9688 (the last two already pass either way).
  //   * H=64 wins fused only while the merging blocks keep at least as many
  //     threads in flight as the combine, BS*splits*512 >= BS*H*32, i.e.
  //     splits*16 >= H: splits=10 shape6 0.7304 -> 0.6599 and splits=6 shape7
  //     0.8214 -> 0.8049 win, while splits=3 shape1 1.0358 -> 1.0632 and
  //     splits=2 shape3 1.3524 -> 1.4142 lose -- a 64-head row's partials are
  //     half the bytes, so there is less traffic to save and the same number of
  //     serial merge rounds to pay for.
  // Unconditional fusing (the second clause dropped) is raw/accp18: it costs
  // shape1 and shape3 more than it returns on shape0 and shape9.
  int* counters = nullptr;
  // The per-CTA head count, not c.H, decides the merge's parallelism: a fused CTA
  // group covers tc_heads heads with `splits` CTAs of 512 threads, where the
  // standalone combine covers the same row with c.H*32 threads.
  const int tc_heads = use_tc ? H / tc_nhb : H;
  const bool fuse_tc = use_tc && (long)BS * tc_nhb * splits <= nsm &&
                       (tc_heads != 64 || (long)splits * 512 >= tc_heads * 32) &&
                       (tc_heads == 64 ? plmn_tc::one_cta_per_sm<64>(smem_per_sm)
                                       : plmn_tc::one_cta_per_sm<128>(smem_per_sm));
  if (fuse_tc) {
    auto& cnt = scratch.counters;
    // The kernel's last CTA per group stores zero back, so only growth re-zeroes.
    const long need = 2 * (long)BS * tc_nhb;
    if (!cnt.defined() || cnt.numel() < need)
      cnt = torch::zeros({need}, q.options().dtype(torch::kInt32));
    counters = reinterpret_cast<int*>(cnt.data_ptr());
  }

  if (use_tc) {
    plmn_tc::launch(c, splits, pop, pml, counters, tc_nhb);
  } else if (use64) {
    cfg64::launch(c, splits, pop, pml);
  } else {
    cfg32::launch(c, splits, pop, pml);
  }
  return lse_base;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &plmn_forward, "paged_latent_mxfp8_nvfp4 sparse decode (independent CUDA)");
  m.def("smem_bytes", &plmn_smem_bytes, "dynamic shared memory bytes for a selector width");
  m.def("smem_bytes_hb64", &plmn_smem_bytes_hb64, "same for the HB=64/NT=512 geometry");
  m.def("use_hb64", &plmn_use_hb64, "true when the HB=64/NT=512 geometry is the faster one");
}
