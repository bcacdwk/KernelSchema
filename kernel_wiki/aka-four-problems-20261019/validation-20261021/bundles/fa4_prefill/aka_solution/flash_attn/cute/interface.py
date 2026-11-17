# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.

import os
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Callable

import torch



import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from quack.compile_utils import make_fake_tensor as fake_tensor
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.testing import is_fake_mode


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from flash_attn.cute import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


from flash_attn.cute import utils
from flash_attn.cute import fa_logging

# hd256 paged64 KV load mechanism: per-page TMA copies (default) or the cp.async
# page-assembly fallback ("0") for debug/AB comparison.
_HD256_PAGED64_TMA = os.environ.get("FA_HD256_PAGED64_TMA", "1") != "0"
# hd256 dev A/B knobs (compile-time): exp2 emulation override ("auto" keeps the
# tuned default, "0" forces hardware ex2, "1" forces emulation) and KV stage
# count override (0 keeps the kernel default).
_HD256_EX2EMU = os.environ.get("FA_HD256_EX2EMU", "auto")
_HD256_KVSTAGE = int(os.environ.get("FA_HD256_KVSTAGE", "0"))
# Use SM103 TMEM load-and-reduce for unmasked score row maxima.
_HD256_LDRED = os.environ.get("FA_HD256_LDRED", "1") != "0"
# Use the bounded FP8 max-offset/rescale-threshold policy from the general
# SM100 kernel to amortize exact output-accumulator rescaling.
_HD256_FP8_MAX_OFFSET = os.environ.get("FA_HD256_FP8_MAX_OFFSET", "1") != "0"
# Separate the causal softmax loop into an unmasked prefix and masked suffix.
_HD256_STATIC_MASK_LOOPS = os.environ.get("FA_HD256_STATIC_MASK_LOOPS", "1") != "0"
# hd256 conditional correction-rescale skip: the correction warps skip the O
# TMEM load/mul/store round trip on tiles whose row max did not move (exact
# no-op; warp-uniform ballot). "1" (default) enables it, "0" forces the
# unconditional rescale (debug/AB).
_HD256_SKIPCORR = os.environ.get("FA_HD256_SKIPCORR", "1") != "0"
# hd256 coalesced O epilogue: stage the finalized O tile in smem and store it
# with 128-bit coalesced row-wise global stores instead of the per-element
# uncoalesced autovec stores. "1" (default) enables it, "0" keeps the legacy
# store path (debug/AB).
_HD256_COLEPI = os.environ.get("FA_HD256_COLEPI", "1") != "0"
_HD256_EARLY_O_RELEASE = os.environ.get("FA_HD256_EARLY_O_RELEASE", "0") != "0"
# hd256 static M-raster permutation (dev A/B): 0 identity, 1 reversed cluster
# order, 2 stride-2 halves interleave (default). The interleave balances the
# causal diagonal's light/heavy cluster work across the launch order for the
# non-persistent hd256 kernel.
_HD256_MRASTER = int(os.environ.get("FA_HD256_MRASTER", "2"))
# Compact request-local scheduling for multi-request long-prefill regimes.
_HD256_FLATVARLEN = os.environ.get("FA_HD256_FLATVARLEN", "1") != "0"
_HD256_FLATLPT = os.environ.get("FA_HD256_FLATLPT", "1") != "0"
_HD256_SHARE_RESCALE = os.environ.get("FA_HD256_SHARE_RESCALE", "1") != "0"
# Issue native page64 K and V TMA streams from independent producer warps.
_HD256_SPLIT_KV_LOADS = os.environ.get("FA_HD256_SPLIT_KV_LOADS", "1") != "0"
# pack_gqa in the hd256 2-CTA kernel: "auto" honors the caller's pack_gqa flag
# (when the tile divides the GQA group size), "1" forces it on, "0" forces the
# unpacked path (debug/AB).
_HD256_PACKGQA = os.environ.get("FA_HD256_PACKGQA", "auto")
# Internal split-KV dispatch of the hd256 paged kernel for low-parallelism
# shapes: "auto" applies the shape heuristic, "0" forces the single-pass path
# (debug/AB). Runtime toggle for A/B: interface._HD256_SPLITKV.
_HD256_SPLITKV = os.environ.get("FA_HD256_SPLITKV", "auto")
# Split-KV engagement thresholds (host-visible, width-bounded KV blocks of 128
# tokens). Splitting pays a combine kernel + partial O/LSE traffic overhead, so
# it is only engaged when the KV range bound is long enough to give every split
# >= _HD256_SPLIT_BLOCKS_PER_SPLIT blocks. The combine kernel re-reads
# num_splits x total_q x num_heads x 256 fp32 partial elements, so the split
# count is additionally capped by a partial-traffic budget. Dev A/B may
# monkeypatch these.
_HD256_SPLIT_MIN_KV_BLOCKS = int(os.environ.get("FA_HD256_SPLIT_MIN_BLOCKS", "16"))
_HD256_SPLIT_BLOCKS_PER_SPLIT = int(os.environ.get("FA_HD256_SPLIT_BLOCKS_PER_SPLIT", "2"))
_HD256_SPLIT_MAX = int(os.environ.get("FA_HD256_SPLIT_MAX", "64"))
_HD256_SPLIT_COMBINE_MAX_MB = int(os.environ.get("FA_HD256_SPLIT_COMBINE_MAX_MB", "24"))
# Fast host dispatch for the fp8 hd256 paged64 varlen causal contract. The
# eager evaluation timing window includes the Python submission path, and for
# short-q shapes the ~60us autograd+validation+compile-key dispatch is the
# binding cost (the device kernel finishes before the next submission lands).
# The fast path replays the exact compiled kernel the full path selected for a
# signature, skipping everything between the public entry point and the launch.
# "0" disables it (debug/AB), runtime toggle: interface._HD256_FASTPATH_ENABLED.
_HD256_FASTPATH_ENABLED = os.environ.get("FA_HD256_FASTPATH", "1") != "0"
# Persistent multi-tile grid for the flattened varlen non-split hd256 path. The
# kernel's four warp-section work loops are already re-entrant (per-CTA
# pipelines/TMEM/barriers allocated once outside the loop), so a cluster can
# process many work tiles and amortize CTA launch + TMEM alloc + cluster barrier
# chain + exit (~2us of the ~4.7us per-cluster fixed cost) instead of paying it
# once per tile. A no-op when the grid already fits one wave (num_m_ctas<=148).
# 0=off (incumbent), 1=plain stride, 2=serpentine-balanced, 3=serpentine+raster.
# Default 1: gated to request_count>4 in _flash_attn_fwd (see hd256_persist_eff).
_HD256_PERSIST = os.environ.get("FA_HD256_PERSIST", "1")
# Batch-1 (single-request) persistence. batch_size==1 never took the flattened
# varlen grid before (the flatten gate required 1 < batch_size), so the v32
# persistent path structurally excluded every single-request shape. For b1 the
# flattened mapping degenerates to the identity (one request, no LPT), so the
# same proven persistent machinery applies: each 2-CTA cluster amortizes CTA
# launch + TMEM alloc + cluster barrier chain + exit across the many waves of
# large single-request prefill (e.g. q4096=4 waves, q8192=7, q16384=14) instead
# of paying them per wave. "1" (default) routes batch_size==1 through the
# flattened grid and the persistent gate; "0" keeps the incumbent rectangular
# non-persistent batch-1 path (byte-identical compile).
_HD256_PERSIST_B1 = os.environ.get("FA_HD256_PERSIST_B1", "1")
# CLC dynamic persistent scheduling for small multi-request batches
# (1 < batch_size <= 4) on the flattened varlen non-split hd256 path. Static
# stride persistence regresses heterogeneous small batches (v32: b4_ragged
# 0.79x) because a coarse stride overloads the CTAs that receive heavy
# clusters, while the incumbent non-persistent launch relies on hardware
# dynamic dispatch + the device-side LPT sort. Blackwell cluster launch
# control (try_cancel) restores greedy hardware load balancing AND keeps the
# persistent fixed-cost amortization: the full logical grid is launched,
# resident clusters cancel pending clusters and inherit their coordinates.
# "4" selects persist_mode 4 (CLC) for the gated regime; "0" keeps the
# incumbent non-persistent small-batch path.
# DEFAULT "0" (OFF) - measured negative result (episode 112): CLC wins big on
# uniform small batches (b4_q2048 1.24x, b4_q1024 1.19x) and on large ragged
# batch>4 (b16_ragged 1.31x vs static stride), but structurally REGRESSES
# shapes whose resident set is dominated by heavy tiles (b4_ragged 0.855x,
# b8_outlier 0.67x): with num_clc_stage=1 every resident cluster reserves its
# next tile at the START of the current one, so a heavy cluster strands its
# reserved light tile for the whole heavy-tile duration. Uniformity is a
# device-side value property (seq_lens); host-side gating on it is forbidden,
# so CLC cannot be enabled safely for any regime. Scaffolding is repaired and
# bit-exact (31 runs) for a future late-try_cancel or weight-partitioned
# variant; flipping this knob reproduces the measurement.
_HD256_CLC4 = os.environ.get("FA_HD256_CLC4", "0")
# Static persistence for small multi-request batches (1 < batch_size <= 4) on
# the flattened varlen non-split hd256 path. batch<=4 was excluded from the v32
# static stride because ONE forced measurement (b4_ragged, one kv8192 request
# among short-q ones) regressed to 0.79x: the non-persistent small-batch path
# runs a device-side LPT sort and the hardware dispatch is near-greedy, while a
# static assignment makespan is ~W/G + T_max (a cluster holding the heaviest
# tile also carries its full light-tile share). The regression needs a tile
# weight T_max comparable to the per-cluster fair share W/G; when every tile is
# light relative to W/G the static deal is balanced and the persistent
# fixed-cost amortization (the ~2us/cluster launch+TMEM+barrier+exit chain) is
# a pure win, exactly as measured for batch>4 (1.09-1.36x).
#
# The ratio D = T_max / (W/G) is bounded from HOST-KNOWN shape metadata alone:
#   T_max <= ceil(max_seqlen_k / 128)             (heaviest tile trip count)
#   W_r   >= q_r^2 / 4096 per request             (causal ramp trip sum with
#                kv_r >= q_r; cluster = 16 q tokens; trip(l) >= (l+1)/8)
#   W     >= total_q^2 / (4096 * batch)           (Cauchy-Schwarz on sum q_r^2)
#   D     <= ceil(max_seqlen_k/128) * G * 4096 * batch / total_q^2 =: D_bound
# with G = num_SMs//2 resident clusters. The bound holds for EVERY device-side
# (seq_lens, query lengths) configuration consistent with the host metadata, so
# gating on it is shape-metadata schedule selection (permitted), not
# value-dependent path selection. Small D_bound certifies that no admissible
# value configuration can make a single tile dominate the fair share, i.e. the
# static deal cannot structurally regress. Near-pinned metadata (slack =
# batch*max_seqlen_k - total_q with slack*8 <= max_seqlen_k) forces every
# request within 14% of square-uniform, so heavy tiles cannot concentrate and
# the static deal is balanced regardless of D_bound. "1" (default) enables the
# gate, "0" keeps the incumbent non-persistent small-batch path
# byte-identically, "force" bypasses the D bound / near-pinned / wave floor
# (dev A/B only).
#
# Episode 113 measurement (ABBA do_bench-exact, L20D/sm_103 gateway, median of
# 4 paired blocks x 100 iters): accepted class wins 1.041x (worst-case
# adversary AT the D bound: q=[16,4096,4096,4080] kv=[4736,...]) to 1.244x
# (b4_q2048 uniform), typical 1.09-1.24x; near-pinned boundary probes at
# slack=max_kv/8 win 1.17-1.21x. Every measured static-stride loss regime is
# rejected by the gate: few-heavy-cluster shapes 0.805-0.867x (D_bound
# 1.05-1.89), kv-outlier shapes 0.931-0.937x (D_bound 6.2-37), K=2-3 small
# pinned 0.976-0.999x (wave floor). Gated-off shapes keep the incumbent
# compile_key byte-identically; accepted shapes are bit-identical to the
# incumbent (bijection over work tiles; 22 case x seed runs ndiff=0, rel_l2 <=
# 0.0266 vs the immutable reference).
_HD256_PERSIST_LE4 = os.environ.get("FA_HD256_PERSIST_LE4", "1")
# D_bound acceptance threshold for FA_HD256_PERSIST_LE4 (dev-tunable).
_HD256_LE4_DMAX = float(os.environ.get("FA_HD256_LE4_DMAX", "0.3"))
# Minimum persistent waves K = ceil(num_logical_clusters / num_resident_clusters)
# for FA_HD256_PERSIST_LE4. Measured (episode 113 ABBA, do_bench-exact): K=2
# neutral (1.00x), K=3 slight loss (0.976-0.989x: per-tile remap latency on
# tiny 1-3-trip tiles exceeds the amortized fixed cost), K=4 wins (1.09-1.21x),
# K>=7 wins 1.16-1.24x. Episode 114 re-check with the hoisted remap table
# (FA_HD256_REMAP_HOIST=1): forced static persistence on K=1-3 shapes measured
# 1.000-1.015 (still neutral-to-negative even with the remap latency removed;
# the ~40us fixed-cost amortization is the binding term) - floor stays 4.
_HD256_LE4_KMIN = int(os.environ.get("FA_HD256_LE4_KMIN", "4"))
# Tightened dominance bound for FA_HD256_PERSIST_LE4 (dev-tunable; 0 disables).
#
# _HD256_LE4_DMAX gates on D_bound = ceil(max_seqlen_k/128)*G*4096*batch/
# total_q^2, whose work lower bound W >= total_q^2/(4096*batch) comes from
# Cauchy-Schwarz on sum q_r^2 under the WEAKEST admissible assumption kv_r ==
# q_r for every request. That bound is 15-30x below the true W exactly for the
# long-KV prefill regime (kv_r >> q_r), because it drops the causal-prefix term
# sum_r n_r*(kv_r - q_r)/128 entirely. The gate therefore rejects almost every
# long-KV small-batch shape even though static persistence wins 7-21% there.
#
# D_tight keeps the same "no single tile can dominate a cluster's fair share"
# meaning but bounds W from below over the admissible set much more tightly,
# using two facts the host metadata pins exactly:
#   (a) max_seqlen_q is the TRUE max query length, so every request satisfies
#       q_r >= q_lo = total_q - (batch-1)*max_seqlen_q  (clamped to [1, max_q]);
#   (b) max_seqlen_k is the true max KV length rounded UP to a 64-token page,
#       so at least one request has kv_r >= max_seqlen_k - 63.
# With n_r = ceil(q_r/16) tiles and trips(r,j) = ceil((kv_r - q_r + 16(j+1))/128):
#   W = sum_r sum_j trips(r,j)
#     >= sum_r [ n_r*(kv_r - q_r)/128 + n_r*(n_r+1)/16 ]        (ceil(x) >= x)
#     >= B_lb + A_lb, where
#   A_lb = sum_r q_r^2/4096 + sum_r q_r/256
#        >= total_q^2/(4096*batch) + total_q/256                (Cauchy-Schwarz)
#   B_lb = min over the two endpoints of x*(max_seqlen_k-63-x) on [q_lo, max_q],
#          divided by 2048, floored at 0. Only the max-KV request is forced to
#          carry a long causal prefix, and x*(C-x) is concave so its minimum
#          over the interval is at an endpoint.
# Both bounds hold for EVERY (q_r, kv_r) configuration consistent with the host
# metadata, so D_tight = ceil((max_seqlen_k+15)/128)*G/(A_lb+B_lb) is a valid
# worst-case dominance ratio. This is shape-metadata schedule selection (both
# outcomes are the same compiled math and stay in the compile cache), never a
# value-dependent path: no tensor content is read.
#
# Episode 116 measurement (L20D/sm_103 gateway). Protocol matters here: the
# authoritative timer is atrex-bench eager triton do_bench, whose 256MB
# cache.zero_() only hides ~31us of the ~66us host submission, so any dev case
# whose DEVICE time is under ~55us is measured with host time mixed in (this
# episode's exposure curve: db-lf = +36.4us at D=13.3, +25.7 at D=27.6, +9.7 at
# D=45.7, 0.0 at D>=58). Every number below therefore comes from an
# ABBA-interleaved DEVICE-TIME-ONLY protocol (1GiB flush per iteration, so all
# host submission is hidden) cross-checked against do_bench on the same cases;
# 4 paired blocks, median of block medians, ratio < 1 == candidate faster.
# On the 24 batch<=4 configurations this bound newly admits - the uniform case
# AND, for every metadata class, the worst-case admissible value configuration
# (one or two requests carrying kv = max_seqlen_k against otherwise square
# requests, plus a 1-token request) - static persistence measures GEOMEAN 0.899
# (1.113x) with a WORST CASE of 1.010: 0.740-0.805 near-square (mk<=2*mq),
# 0.849-0.946 at ratio 2-4, 0.916-0.996 at ratio 4-8, 0.918-0.930 on q-skewed
# batches. Every measured static-stride LOSS regime is excluded, and the two
# excluded families are separated by FA_HD256_LE4_KVSKEW below rather than by
# D_tight (b2_outl_kv32768 D_tight 0.581 loses 1.105x while b2_q2048_kv8192
# D_tight 0.590 wins 0.937x - the bounds genuinely cannot separate them, which
# is why the anti-skew condition is needed). The new branch is OR-ed with
# D_bound/near-pinned, so no shape the episode-113/114 gate accepted can change
# decision; the 11 rejected-class controls measured 0.997-1.001 (identical
# compile) and the 6 already-accepted controls 0.998-1.006.
_HD256_LE4_DTIGHT = float(os.environ.get("FA_HD256_LE4_DTIGHT", "0.60"))
# Anti-skew companion bound for FA_HD256_LE4_DTIGHT. D_tight rules out a single
# tile dominating a cluster's fair share, but static stride ALSO loses when tile
# weights are bimodal ACROSS requests: requests are contiguous segments of the
# flattened index, so one request carrying kv = max_seqlen_k with many clusters
# produces a contiguous run of heavy tiles, and a stride-G sample then hands some
# clusters two heavy tiles and others none. The per-request m_raster fold only
# balances the causal ramp WITHIN a request; it cannot balance across requests.
# Since kv_r <= max_seqlen_k and q_r <= max_seqlen_q bound both ends, capping
# max_seqlen_k <= _HD256_LE4_KVSKEW * max_seqlen_q caps the heavy/light
# tile-weight ratio for EVERY admissible value configuration.
# 8.0 is measured, not guessed. Device-time ratio of forced static persistence
# by skew ratio max_seqlen_k / max_seqlen_q:
#   <= 2 : 0.740-0.924 (14 configurations, incl. all worst-case adversaries)
#   2 -> 4 : 0.876-0.986        4 -> 8 : 0.916-0.998
#   16   : 0.952-0.992 on UNIFORM kv, but 1.009-1.016 / 1.043 / 1.105 on the
#          one-heavy-request adversaries (b4_outl_kv16384, b4_outl_2heavy,
#          b2_outl_kv32768) and 1.285 on the short-q heavy-kv adversary
#          (b4_outl_shortq, D_tight 4.18 so D_tight alone already rejects it).
# 8.0 therefore admits every class whose measured worst-case adversary is
# <= 1.010 and excludes both families with a measured double-digit loss.
_HD256_LE4_KVSKEW = float(os.environ.get("FA_HD256_LE4_KVSKEW", "8.0"))
# Hoisted request-mapping table for the persistent flattened grid. The
# per-tile remap in the four persistent warp-section loops re-derived the
# request mapping (LPT sort: 4 seqused_k loads; segment scan: up to
# 2*batch cum_seqlen_q loads + ceil_divs) from GLOBAL memory on EVERY tile.
# With this knob an elected thread per CTA builds the identical segment table
# (sorted request ids + cluster prefix boundaries, 36 Int32 smem slots) once
# before the work loops (one barrier(4)), and the per-tile remap becomes a
# pure smem segment scan. Same mapping values -> bit-identical outputs (a
# bijection-preserving latency hoist), lower exposed per-tile remap latency
# for every persistent class (b1, batch 2..4 LE4 static, batch>4 static,
# CLC). Episode 114 measurement (L20D/sm_103 gateway, ABBA do_bench-exact,
# median of 4 paired blocks x 100 iters, ratio hoist/incumbent): batch>4
# static class 0.891-0.973 (b16_q1024 0.891, b16_ragged 0.920, b8_q1024
# 0.935, b8_outlier 0.973), b1 multi-wave 0.971-0.987 (q8192 0.980,
# longkv_q4096 0.976, odd_q4095 0.971), LE4 accepted class stacks on the
# gate win (both/le4 0.892-0.999; combined 0.727-0.879 vs incumbent =
# 1.14-1.37x), non-persistent and gated-off shapes neutral (40us-class
# shapes ride the +-2.5% harness noise floor with mixed signs). Bit-identity
# vs the per-tile remap: 91 case x seed x arm runs ndiff=0 including CLC
# mode 4 (4a), serpentine modes 2/3, flatlpt/share_rescale, odd-q tails and
# split-KV controls. "1" (default) hoists, "0" keeps the per-tile remap.
_HD256_REMAP_HOIST = os.environ.get("FA_HD256_REMAP_HOIST", "1")
# Bounded binary segment search inside the hoisted remap table. The hoisted
# per-tile remap still walks the request prefix table LINEARLY
# (request_count-1 iterations; up to 15 at the batch-16 flatten cap) on every
# tile in all four persistent warp-section loops, and that walk sits on the
# load warp's per-tile critical path. With this knob the search becomes a fixed
# 4-step binary search over the same non-decreasing prefix table (16 segments
# -> 4 halvings), returning the identical "largest j with p[j] <= cl" segment
# index. Value-identical => bit-identical outputs; only exposed remap latency
# changes. Requires FA_HD256_REMAP_HOIST (it searches the hoisted table).
#
# Selection is host-metadata-only and const_expr (compile-time), so each
# compile carries exactly one search variant and pays no dynamic branch:
#   "auto" (default) -> enabled iff batch_size > 4 AND the shape is persistent
#                       (batch is request_count here, so the linear scan is
#                       <=3 iterations for batch<=4 - cheaper than 4 dependent
#                       smem steps - and a byte-identical episode-114 compile);
#   "1" / "0"        -> force on/off (dev A/B).
# Episode 115 measurement (L20D/sm_103 gateway, ABBA do_bench-exact, median of
# 4 paired blocks x 100 iters, ratio bsearch/linear-scan on the hoisted remap):
# b16_q1024 0.972 (2.8% faster, blocks 0.971-0.988), b8_outlier_kv16384 0.988,
# b8_q1024 / b6_q1408 / b5_mixed / b8_odd_tails / b16_ragged 1.000 (neutral);
# batch<=4 and b1 shapes keep the linear-scan compile byte-identically.
_HD256_REMAP_BSEARCH = os.environ.get("FA_HD256_REMAP_BSEARCH", "auto")
from flash_attn.cute.cute_dsl_utils import (
    get_aux_tensor_metadata,
    get_broadcast_dims,
    to_cute_aux_tensor,
    to_cute_tensor,
)
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80
from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100, DescaleTensors
from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.flash_bwd_sm120 import FlashAttentionBackwardSm120
from flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine
from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100
from flash_attn.cute.flash_bwd_mla_sm100 import FlashAttentionSparseMLABackwardSm100
from flash_attn.cute.flash_bwd_mla_dq_dqv_sm100 import dQdQvGemmKernel
from flash_attn.cute.flash_bwd_mla_dk_sm100 import dKGemmKernel

# SM100 head_dim=256 2CTA kernel imports
from flash_attn.cute.sm100_hd256_2cta_fmha_forward import BlackwellFusedMultiHeadAttentionForward
from flash_attn.cute.sm100_hd256_2cta_fmha_backward import BlackwellFusedMultiHeadAttentionBackward

from flash_attn.cute.utils import AuxData
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    block_sparse_bwd_supports_2cta,
    get_kv_subtile_factor,
    get_sparse_q_block_size,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_config,
    normalize_block_sparse_config_bwd,
)

def _parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int (e.g. 80, 90, 100)."""
    import re
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    """Cached device arch check.

    Override with FLASH_ATTENTION_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).

    For CPU-only compilation (no GPU), set both:
      FLASH_ATTENTION_ARCH=sm_80  (kernel selection)
      CUTE_DSL_ARCH=sm_80         (compilation target)
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def _validate_head_dims(head_dim: int, head_dim_v: int, compute_capability: int, alignment: int) -> None:
    """Validate head dimension constraints based on compute capability."""
    is_deepseek_shape = head_dim == 192 and head_dim_v == 128
    is_deepseek_mla_absorbed_shape = (head_dim == 64 or head_dim == head_dim_v) and head_dim_v == 512
    is_dedicate_kernel_shape = head_dim == 256 and head_dim_v == 256
    is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128

    is_sm90_range = 8 <= head_dim <= 256 and 8 <= head_dim_v <= 256
    if compute_capability == 9:
        assert is_sm90_range and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM90. "
            f"head_dim and head_dim_v must be between 8 and 256 and divisible by {alignment}."
        )
    elif compute_capability in [10, 11]:
        assert (is_standard_range or is_deepseek_shape or is_deepseek_mla_absorbed_shape or is_dedicate_kernel_shape) and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM100/SM110. "
            f"head_dim and head_dim_v must be between 8 and 128 and divisible by {alignment}, or (192, 128) for DeepSeek, or (256, 256) for hd256."
        )


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def _tile_size_fwd_sm90(head_dim, head_dim_v, is_causal, is_local, sparse_block_size_q=None):
    """Return FwdConfig for SM90 forward.

    Tile sizes and flags based on tile_size_fwd_sm90 in hopper/tile_size.h, adjusted
    for the Python kernel's different register/smem tradeoffs (benchmarked on H100 SXM).

    When sparse_block_size_q is set, tile_m must divide it. For head_dim <= 96 the
    optimal tile_m=192 is used when compatible, otherwise we fall back to 128.
    """
    if head_dim <= 64:
        # C++: 192×192 non-causal, 192×128 causal/local.
        # Python: 192×128 RS+OL is consistently best across seqlens.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, True, True)
        return FwdConfig(192, 128, True, True)
    elif head_dim <= 96:
        # C++: 192×144 noRS+OL for all cases.
        # Python: RS is catastrophic with 192× tiles (~300 vs ~600 TFLOPS).
        # noRS+OL is always required. Causal: 192×128 slightly better short seqlen.
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, False, True)
        if is_causal or is_local:
            return FwdConfig(192, 128, False, True)
        else:
            return FwdConfig(192, 144, False, True)
    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    elif head_dim <= 192:
        tile_n = 96 if is_local else (128 if head_dim_v <= 128 else 112)
        return FwdConfig(128, tile_n, True, True)
    else:  # hdim 256
        tile_n = 64 if is_local else 80
        return FwdConfig(128, tile_n, True, True)

@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int
    n_block_size: int
    num_stages_Q: int
    num_stages_dO: int
    num_stages_PdS: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_wg: int = 2  # MMA warp groups (total threads = (num_wg + 1) * 128)
    dQ_single_wg: bool = False


def _tile_size_bwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=None):
    """Return BwdConfig for SM90.

    Configs based on C++ FA3 hopper/flash_bwd_launch_template.h,
    benchmarked on H100 SXM.
    """
    if head_dim <= 64:
        # C++ FA3: 128, 128, 64, ..., 2, 2, true, false, false, 2, 1, 2, 2
        return BwdConfig(
            m_block_size=128, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2,
        )
    elif head_dim <= 96:
        # C++ FA3: 64, 128, 96, dQ_swapAB=False
        return BwdConfig(
            m_block_size=64, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
            dQ_single_wg=True,
        )
    elif head_dim <= 128:
        # C++ FA3: causal/local: 64, 128; non-causal: 80, 128 with dQ_swapAB
        is_causal_or_local = causal or local
        m_block_size = 64 if is_causal_or_local else 80
        if sparse_block_size_q is not None and sparse_block_size_q % m_block_size != 0:
            m_block_size = 64
        return BwdConfig(
            m_block_size=m_block_size,
            n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False,
            dQ_swapAB=m_block_size % 64 != 0,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
        )
    elif head_dim <= 192:
        hdimv128 = head_dim_v <= 128
        if hdimv128:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=2, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
        else:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=1, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
    else:
        # hdim 256
        return BwdConfig(
            m_block_size=64, n_block_size=64,
            num_stages_Q=1, num_stages_dO=1, num_stages_PdS=1,
            SdP_swapAB=False, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=1, AtomLayoutMdQ=1,
        )



def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    if not is_fake_mode():
        assert t.is_cuda, f"{name} must be on CUDA"

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1
    # Avoid ZeroDivisionError when batch_size or seqlen_q is 0. The empty-Q
    # early-exit in _flash_attn_fwd handles correctness for those shapes; this
    # guard just keeps the heuristic safe if called in other contexts.
    if total_mblocks == 0:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _hd256_split_heuristic(
    total_q, qhead_per_kvhead, num_head_kv, batch_size, max_num_pages_per_seq, num_SMs
):
    """KV split factor for the dedicated hd256 2-CTA paged kernel.

    The kernel launches one 2-CTA cluster per (packed-row tile, kv head,
    request). When that count underfills the device (short packed-q decode-like
    shapes), split each request's KV block range across extra clusters; the
    partial O/LSE reductions run in the stock combine kernel afterwards. All
    decision inputs are host-visible shape metadata; KV lengths themselves are
    only bounded by the block-table width, never read from device tensors.

    The split path pays a combine-kernel overhead and re-reads
    num_splits x total_q x num_heads partial rows, so it is only engaged when
    the width-bounded KV range is long enough that every split still owns
    several 128-token blocks, and the split count is capped by a partial-O
    traffic budget. num_splits is a runtime kernel argument, so the exact
    fill factor needs no extra compiled variants.
    """
    packed_rows = total_q * qhead_per_kvhead
    cluster_tile_rows = 256  # one 2-CTA cluster covers 2 x 128 packed rows
    num_m_clusters = (packed_rows + cluster_tile_rows - 1) // cluster_tile_rows
    num_clusters = max(1, num_m_clusters * num_head_kv * batch_size)
    cluster_slots = max(1, num_SMs // 2)
    if num_clusters >= cluster_slots:
        return 1
    # Upper bound on 128-token KV blocks from the block-table width (page 64).
    kv_blocks = (max_num_pages_per_seq * 64 + 127) // 128
    if kv_blocks < _HD256_SPLIT_MIN_KV_BLOCKS:
        return 1
    # cluster_slots // num_clusters <= cluster_slots already; combine supports
    # up to 256 splits and empty split sub-ranges are handled in-kernel.
    splits = cluster_slots // num_clusters
    # Require several width-bounded KV blocks per split so short-KV shapes do
    # not pay the combine overhead for one-trip sub-ranges.
    splits = min(splits, kv_blocks // _HD256_SPLIT_BLOCKS_PER_SPLIT)
    # Combine partial-O traffic grows with splits x rows; bound it.
    row_bytes = packed_rows * num_head_kv * 256 * 4
    if row_bytes > 0:
        combine_cap = (_HD256_SPLIT_COMBINE_MAX_MB * 1000000) // row_bytes
        splits = min(splits, max(1, combine_cap))
    return max(1, min(splits, _HD256_SPLIT_MAX))


def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
    """Resolve causal/local/window settings into canonical form.

    Returns (causal, local, window_size_left, window_size_right).
    """
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if causal:
        window_size_right = 0
    if window_size_left is not None and window_size_right is not None and window_size_left + window_size_right < 0:
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right

def _flash_attn_fwd(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    mma_pv_is_rs: Optional[bool] = None,
    intra_wg_overlap: Optional[bool] = None,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _arch: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    aux_scalars: Optional[tuple] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
            The returned LSE supports taking gradient.
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
        aux_scalars: Runtime scalar captures used by score_mod or mask_mod.
    """
    aux_scalars = tuple(aux_scalars) if aux_scalars else None
    q, k, v, qv = [maybe_contiguous(t) for t in (q, k, v, qv)]
    assert q is not None or qv is not None
    assert v is not None
    q_descale, k_descale, v_descale = [maybe_contiguous(t) for t in (q_descale, k_descale, v_descale)]
    q_shape = q.shape if q is not None else qv.shape
    num_head, head_dim = q_shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q_shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q_shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = v.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = v.shape[-3]
    num_head_kv = v.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k is None or k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k is None or k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k is None or k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    assert v.dtype in [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2], (
        "inputs must be float16, bfloat16, fp8 e4m3fn, or fp8 e5m2"
    )
    
    input_tensors = {"q": q, "k": k, "v": v, "qv": qv}
    present = {name: t for name, t in input_tensors.items() if t is not None}
    names = list(present.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            assert present[a].dtype == present[b].dtype, f"{a}.dtype {present[a].dtype} != {b}.dtype {present[b].dtype}"

    q_dtype = q.dtype if q is not None else qv.dtype

    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    if not is_fake_mode():
        assert all(
            t is None or t.is_cuda
            for t in (
                q,
                k,
                v,
                qv,
                q_descale,
                k_descale,
                v_descale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                learnable_sink,
            )
        ), "inputs must be on CUDA device"
    arch = _get_device_arch() if _arch is None else _arch
    assert arch // 10 in [8, 9, 10, 11, 12], "Unsupported compute capability. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // v.element_size()
    if arch // 10 not in [8, 12]:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = (
            1.0 / math.sqrt(head_dim) if qv is None or q is None
            else 1.0 / math.sqrt(head_dim + head_dim_v)
        )
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    is_fp8 = v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    requires_grad = any(t is not None and t.requires_grad for t in [q, k, v, qv])
    if is_fp8 and requires_grad:
        raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")
    out_torch_dtype = torch.bfloat16 if is_fp8 else q_dtype
    device = v.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)

    if qv is None:
        lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    else:
        # num_head contiguous better for MQA in MLA absorbed
        lse_shape = (batch_size, seqlen_q, num_head) if cu_seqlens_q is None else (total_q, num_head)

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        _validate_tensor(out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), out_torch_dtype, device)

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    if seqlen_k == 0 or total_q == 0:
        out.zero_()
        if lse is not None:
            lse.fill_(float("-inf"))
        return out, lse, None, None

    if is_fp8:
        for t, name in ((q_descale, "q_descale"), (k_descale, "k_descale"), (v_descale, "v_descale")):
            if t is not None:
                _validate_tensor(t, name, (batch_size, num_head_kv), torch.float32, device)
    else:
        assert q_descale is None and k_descale is None and v_descale is None, (
            "q_descale/k_descale/v_descale are only supported for FP8 inputs"
        )

    dtype = torch2cute_dtype_map[q_dtype]
    if is_fp8:
        assert arch // 10 == 10, "FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."
    use_block_sparsity = block_sparse_tensors is not None

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right, mask_mod
    )

    requested_use_clc_scheduler = utils._get_use_clc_scheduler_default()
    requested_disable_2cta = utils._get_disable_2cta_default(is_fwd=True)

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # SM80/SM120: uses SM80 MMA, 128 threads (4 warps)
    if arch // 10 in [8, 12]:
        num_threads = 128

    fwd_cfg = FwdConfig(128, 128, True, True)  # default
    if tile_mn is None:
        if arch // 10 == 12:
            # SM120 tile sizes tuned for 99 KB SMEM capacity:
            # D<=64:  128x128 → 48 KB (good occupancy)
            # D>64:   128x64  → 64 KB (128x128 would use 96 KB, hurting occupancy)
            if head_dim <= 64:
                fwd_cfg = FwdConfig(128, 128, True, True)
            else:
                fwd_cfg = FwdConfig(128, 64, True, True)
        elif arch // 10 == 8:
            fwd_cfg = FwdConfig(128, 64, True, True)  # SM80, should tune
        elif arch // 10 == 9:
            sparse_q = get_sparse_q_block_size(block_sparse_tensors, seqlen_q)
            fwd_cfg = _tile_size_fwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=sparse_q)
    else:
        fwd_cfg = FwdConfig(tile_mn[0], tile_mn[1], fwd_cfg.mma_pv_is_rs, fwd_cfg.intra_wg_overlap)
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    if mma_pv_is_rs is None:
        mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
    if intra_wg_overlap is None:
        intra_wg_overlap = fwd_cfg.intra_wg_overlap

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    if cu_seqlens_k is None and seqused_k is None:
        min_seqlen_k = seqlen_k 
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if arch // 10 in [10, 11]:
        q_stage = 2 if seqlen_q_packgqa > tile_m else 1
    else:
        q_stage = 1

    m_block_size_effective = q_stage * tile_m
    seqlen_k_loaded = max_seqlen_k if not local else max(0, min(max_seqlen_k, (window_size_right or max_seqlen_k) + (window_size_left or max_seqlen_k) + 1 + tile_m))
    num_m_blocks = (seqlen_q_packgqa + m_block_size_effective - 1) // m_block_size_effective
    total_mblocks = batch_size * num_head_kv * num_m_blocks
    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
    num_SMs = 132 if is_fake_mode() else torch.cuda.get_device_properties(device).multi_processor_count
    if arch // 10 == 12:
        assert num_splits == 1, "SM120 forward only supports num_splits=1"
    elif num_splits < 1:
        num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)

    # SplitKV uses float32 partial output, which doubles the O buffer size
    # in shared memory, causing OOM for diff-headdim (192, 128)
    if arch // 10 in [10, 11] and head_dim != head_dim_v and num_splits > 1:
        if num_n_blocks >= 64 and head_dim_v != 512:
            tile_n = 64
            num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
            num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)
        else:
            num_splits = 1

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = torch.empty(num_splits, *q_batch_seqlen_shape, num_head, head_dim_v, dtype=torch.float32, device=device)
        lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)

    use_2cta_instrs = (
        arch // 10 in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and not is_split_kv
        and cu_seqlens_q is None
        and seqused_q is None
        and not use_block_sparsity
        and page_size in [None, 128]
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
    )

    # hd=256 2CTA forward uses dedicated kernel (Blackwell family)
    use_dedicated_hd256_kernel = arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel
    # Batch-1 persistence extension (FA_HD256_PERSIST_B1): route single-request
    # shapes through the same flattened grid + persistent gate as batch>4. For
    # b1 the flatten mapping degenerates to the identity (grid_m_total = h_r*s_q
    # with no multi-request term), so the kernel-side request scan/LPT sees
    # request_count==1 and must take the guarded trivial branch.
    # Multi-wave gate: the persistent grid is min(num_SMs, num_m_ctas), so a
    # barely-multi-wave b1 shape pays the per-tile remap (two cum_seqlen_q
    # global loads on the critical path) with almost nothing to amortize.
    # Measured crossover (ABBA, do_bench-exact): W=1.03-1.08 regress 8-9%,
    # W=1.30 neutral, W>=1.51 gain 5.7-12.7%. Require W > 1.5, i.e.
    # num_m_ctas > num_SMs + num_SMs//2 (222 on the 148-SM B300): q1792
    # (224 CTAs, 0.93x) is in, q1536 (192, 1.0016x) and below stay on the
    # incumbent rectangular compile byte-identically. num_m_ctas and num_SMs
    # are host-known shape/hardware metadata (permitted schedule selection).
    hd256_persist_b1 = (
        _HD256_PERSIST_B1 != "0"
        and batch_size == 1
        and -(-max_seqlen_q * qhead_per_kvhead // 128) > num_SMs + num_SMs // 2
    )
    hd256_flatten_varlen = (
        _HD256_FLATVARLEN
        and use_dedicated_hd256_kernel
        and cu_seqlens_q is not None
        and seqused_k is not None
        and page_table is not None
        and (1 < batch_size <= 16 or hd256_persist_b1)
        and num_head == 16
        and num_head_kv == 1
    )
    hd256_flatten_lpt = (
        _HD256_FLATLPT
        and hd256_flatten_varlen
        and batch_size == 4
        and total_q == 16384
    )
    hd256_share_rescale = _HD256_SHARE_RESCALE and hd256_flatten_lpt
    hd256_split_kv_loads = (
        _HD256_SPLIT_KV_LOADS
        and use_dedicated_hd256_kernel
        and page_table is not None
        and page_size == 64
        and _HD256_PAGED64_TMA
        and (batch_size > 1 or max_seqlen_k >= 96 * 1024)
    )

    if (
        _HD256_SPLITKV != "0"
        and _HD256_PACKGQA != "0"
        and use_dedicated_hd256_kernel
        and not is_split_kv
        and num_splits == 1
        and cu_seqlens_q is not None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is not None
        and page_table is not None
        and page_size == 64
        and pack_gqa
        and not use_block_sparsity
        and score_mod is None
        and mask_mod is None
        and softcap is None
        and learnable_sink is None
        and not return_lse
        and qv is None
    ):
        # Decode-like shapes underfill the machine; the dedicated kernel splits
        # the KV range internally and reduces partials with the combine kernel.
        num_splits = _hd256_split_heuristic(
            total_q,
            qhead_per_kvhead,
            num_head_kv,
            batch_size,
            max_num_pages_per_seq,
            num_SMs,
        )
        if num_splits > 1:
            is_split_kv = True
            out_partial = torch.empty(
                num_splits, *q_batch_seqlen_shape, num_head, head_dim_v,
                dtype=torch.float32, device=device,
            )
            lse_partial = torch.empty(
                num_splits, *lse_shape, dtype=torch.float32, device=device
            )

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)
    elif score_mod is not None:
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")
        
    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    # CLC regressed for varlen MHA and dense noncausal. Imbalanced varlen shapes
    # keep more K/V blocks in flight and hurt L2; dense noncausal mostly just
    # pays work-stealing overhead.
    is_varlen_mha = is_varlen and qhead_per_kvhead == 1
    is_dense_noncausal = not is_varlen and not causal and not local
    use_clc_scheduler = requested_use_clc_scheduler and not is_varlen_mha and not is_dense_noncausal

    if use_block_sparsity:
        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)
        head_dim_idx = 0 if block_sparse_tensors.mask_block_cnt.ndim == 2 else 1
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[head_dim_idx] != 1:
            pack_gqa = False
        if cu_seqlens_q is not None:
            assert block_sparse_tensors.cu_total_m_blocks is not None, (
                "Varlen block sparsity requires block_sparse_tensors.cu_total_m_blocks."
            )
            if (
                block_sparse_tensors.cu_block_idx_offsets is None
                and (cu_seqlens_k is not None or seqused_k is not None)
            ):
                raise ValueError(
                    "Varlen block sparsity with cu_seqlens_k or seqused_k requires "
                    "block_sparse_tensors.cu_block_idx_offsets."
                )

    # See get_broadcast_dims for why this is needed in compile key
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = 1
    kv_subtile_factor = 1
    if block_sparse_tensors is not None:
        block_sparse_config = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(tile_m, tile_n),
            q_stage=q_stage,
            allow_kv_subtile=arch // 10 in [10, 11],
        )
        normalized_block_sparse_tensors = block_sparse_config.tensors
        block_sparse_broadcast_pattern = block_sparse_config.broadcast_pattern
        q_subtile_factor = block_sparse_config.q_subtile_factor
        kv_subtile_factor = block_sparse_config.kv_subtile_factor
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None
    aux_scalar_metadata = tuple(type(s) for s in aux_scalars) if aux_scalars is not None else None

    if qv is not None:
        assert arch // 10 in [10, 11], "only support Blackwell arch with qv"
        assert q is None or qv.shape[:-1] == q.shape[:-1]
        assert qv.shape[-1] == head_dim_v
        assert head_dim_v == 512
        assert q is None or head_dim == 64
        assert not local, "local not yet supported with qv"
        assert q_descale is None and k_descale is None and v_descale is None, (
            "q_descale/k_descale/v_descale are not yet supported with qv"
        )
        assert tile_n == 128

        assert not is_split_kv, "split kv not supported with qv"
        assert learnable_sink is None
        assert softcap is None
        assert score_mod is None
        assert mask_mod is None

        if page_table is not None:
            assert gather_kv_indices is None, "paged KV + topk sparsity not yet supported together"
        
        qv = maybe_contiguous(qv)

        gather_kv_length = 2048  # dummy value
        sparse_kv = gather_kv_indices is not None
        # always use kv bitmask by default (handles -1 sentinel)
        disable_sparse_kv_bitmask = False
        if sparse_kv:
            assert gather_kv_indices.shape[:-1] == qv.shape[:-2]
            gather_kv_length = gather_kv_indices.shape[-1]
            assert gather_kv_length % 128 == 0
            # if min_seqlen_k is None or causal:
            #     disable_sparse_kv_bitmask = False
            # else:
            #     # seqlen_k_boundary = min_seqlen_k - max_seqlen_q + 1 if causal else min_seqlen_k
            #     seqlen_k_boundary = min_seqlen_k
            #     disable_sparse_kv_bitmask = seqlen_k_boundary >= gather_kv_length
        
        if requires_grad and sparse_kv:
            if cu_seqlens_q is None:
                p = torch.empty(batch_size, seqlen_q, num_head, gather_kv_length, dtype=q_dtype, device=device)
                row_max = torch.empty(batch_size, seqlen_q, gather_kv_length//128, num_head, dtype=torch.float32, device=device)
            else:
                p = torch.empty(total_q, num_head, gather_kv_length, dtype=q_dtype, device=device)
                row_max = torch.empty(total_q, gather_kv_length//128, num_head, dtype=torch.float32, device=device)
        else:
            p = row_max = None
    else:
        assert gather_kv_indices is None, "gather_kv_indices is only supported with qv"
        gather_kv_length = None
        sparse_kv = None
        disable_sparse_kv_bitmask = None
        p = row_max = None

    # Persistent multi-tile grid gate (FA_HD256_PERSIST). Static persistent
    # stride trades away the hardware's dynamic load balancing. For
    # request_count<=4 the non-persistent path's LPT sort front-loads heavy
    # requests so they overlap (near-ideal), and static persistence regresses
    # heterogeneous small-batch shapes (measured b4 ragged 0.79x). For
    # request_count>4 the mapping uses a plain serial scan with no LPT, the
    # non-persistent launch leaves exploitable tails, and persistent stride won
    # on every measured case (uniform and heterogeneous: b8/b16 1.03-1.36x).
    # request_count==1 (FA_HD256_PERSIST_B1) has no cross-request imbalance at
    # all: the within-request m_raster fold spreads the monotone causal ramp
    # across the strided assignment, so static stride is safe there and
    # amortizes the per-cluster fixed cost over the many waves of large
    # single-request prefill (b1 q4096=4 waves, q8192=7, q16384=14).
    # batch_size is host-known shape metadata (permitted schedule selection, not
    # a value-dependent fast path); both variants stay in the compile cache.
    # CLC dynamic persistence for small multi-request batches (FA_HD256_CLC4).
    # Mirrors the kernel's flattened grid extent (qk_mma_tiler[0] == 256 is an
    # hd256 kernel constant) to apply the same W>1.5 multi-wave gate as b1:
    # barely-multi-wave grids pay per-tile remap + CLC response latency with
    # nothing to amortize. batch_size/total_q are host-known shape metadata.
    # "4" = batches 2..4 only (production candidate); "4a" = every flattened
    # batch (dev A/B vs the batch>4 static stride default).
    hd256_clc_le4 = (
        _HD256_CLC4 in ("4", "4a")
        and (
            batch_size > 1
            if _HD256_CLC4 == "4a"
            else 1 < batch_size <= 4
        )
        and hd256_flatten_varlen
        and not is_split_kv
        and -(-(
            qhead_per_kvhead * total_q
            + qhead_per_kvhead * (batch_size - 1) * (256 // qhead_per_kvhead - 1)
        ) // 128) > num_SMs + num_SMs // 2
    )
    # Static persistence for small multi-request batches (FA_HD256_PERSIST_LE4).
    # Three host-metadata conditions, all shape/hardware-only (permitted
    # schedule selection; both outcomes stay in the compile cache and the
    # persistent bijection keeps outputs bit-identical):
    #
    # (1) K >= _HD256_LE4_KMIN persistent waves, K = ceil(n_clusters /
    #     g_clusters) over the flattened logical grid. Measured floor: K=2
    #     neutral, K=3 slight loss (per-tile remap latency on tiny 1-3-trip
    #     tiles exceeds the amortized ~2us/cluster fixed cost), K=4 wins
    #     1.09-1.21x, K>=7 wins 1.16-1.24x (episode 113 ABBA).
    # (2) dominance bound D_bound <= _HD256_LE4_DMAX with
    #       D_bound = ceil(max_seqlen_k/128) * G * 4096 * batch / total_q^2,
    #     derived at the knob definition. Bounds T_max/(W/G) over EVERY device
    #     value configuration consistent with (total_q, batch, max_seqlen_k),
    #     so no single tile can dominate the per-cluster fair share - the
    #     structural precondition whose violation caused the v32 static-stride
    #     b4_ragged 0.79x regression. Worst-case adversary probe AT the bound
    #     (q=[16,4096,4096,4080], kv=[4736,...], D_bound=0.297, K=11) still
    #     measured 1.04x. Note D_bound <= 0.3 with slack implies total_q >
    #     ~7900, hence K >= 7 automatically: branch (2) always carries strong
    #     amortization.
    # (3) OR near-pinned metadata: slack = batch*max_seqlen_k - total_q with
    #     slack*8 <= max_seqlen_k. Every request then satisfies
    #     q_r >= max_kv - slack >= (7/8)*max_kv and kv_r <= max_kv, so every
    #     request is within 14% of square-uniform: heavy tiles cannot
    #     concentrate (each request's causal ramp contributes >= max_kv/32
    #     near-max tiles >= G for max_kv >= 2368) and the static deal is
    #     balanced regardless of D_bound. Validated at the boundary: slack =
    #     max_kv/8 exactly, max-skew q vectors (b4 [1856,2112,2112,2112] and
    #     b2 [3584,4096]) measured 1.21x / 1.17x.
    hd256_persist_le4 = False
    if (
        _HD256_PERSIST != "0"
        and _HD256_PERSIST_LE4 != "0"
        and 1 < batch_size <= 4
        and hd256_flatten_varlen
        and not is_split_kv
    ):
        if _HD256_PERSIST_LE4 == "force":
            hd256_persist_le4 = True
        else:
            num_m_ctas_flat = -(-(
                qhead_per_kvhead * total_q
                + qhead_per_kvhead * (batch_size - 1) * (256 // qhead_per_kvhead - 1)
            ) // 128)
            n_clusters = -(-num_m_ctas_flat // 2)
            g_clusters = max(1, min(num_SMs // 2, n_clusters))
            k_waves = -(-n_clusters // g_clusters)
            if k_waves >= _HD256_LE4_KMIN:
                t_max_blocks = -(-max_seqlen_k // 128)
                w_lower = (total_q * total_q) / (4096.0 * batch_size)
                d_bound = t_max_blocks * g_clusters / w_lower
                slack = batch_size * max_seqlen_k - total_q
                near_pinned = slack * 8 <= max_seqlen_k
                # Tightened worst-case dominance bound (see FA_HD256_LE4_DTIGHT).
                # Purely additive: OR-ed with the incumbent conditions so every
                # shape the episode-113/114 gate accepted keeps its decision.
                tight_ok = False
                if (
                    _HD256_LE4_DTIGHT > 0.0
                    and isinstance(max_seqlen_q, int)
                    and max_seqlen_q > 0
                    and max_seqlen_k <= _HD256_LE4_KVSKEW * max_seqlen_q
                ):
                    # q_lo: every request's query length floor, from
                    # sum q_r == total_q and q_r <= max_seqlen_q (the caller's
                    # true max). Clamped into [1, max_seqlen_q].
                    q_lo = total_q - (batch_size - 1) * max_seqlen_q
                    q_lo = min(max(1, q_lo), max_seqlen_q)
                    # A_lb: causal-ramp work, sum_r n_r(n_r+1)/16 bounded below.
                    a_lb = w_lower + total_q / 256.0
                    # B_lb: the max-KV request's forced causal prefix. x*(C-x)
                    # is concave, so the interval minimum is at an endpoint.
                    c_kv = max_seqlen_k - 63
                    b_lb = max(
                        0.0,
                        min(
                            q_lo * (c_kv - q_lo),
                            max_seqlen_q * (c_kv - max_seqlen_q),
                        ),
                    ) / 2048.0
                    # A tile covers 16 query tokens, so its last token can sit
                    # 15 tokens past q_r: bound the trip count from above too.
                    t_max_tight = -(-(max_seqlen_k + 15) // 128)
                    d_tight = t_max_tight * g_clusters / (a_lb + b_lb)
                    tight_ok = d_tight <= _HD256_LE4_DTIGHT
                hd256_persist_le4 = (
                    d_bound <= _HD256_LE4_DMAX or near_pinned or tight_ok
                )
    if hd256_clc_le4:
        hd256_persist_eff = "4"
    elif hd256_persist_le4:
        hd256_persist_eff = _HD256_PERSIST
    else:
        hd256_persist_eff = (
            _HD256_PERSIST
            if (
                _HD256_PERSIST != "0"
                and (batch_size > 4 or hd256_persist_b1)
                and hd256_flatten_varlen
                and not is_split_kv
            )
            else "0"
        )

    # Compile-time selection of the hoisted-remap segment search (see the
    # FA_HD256_REMAP_BSEARCH knob). On the flattened path request_count ==
    # batch_size, so batch>4 is exactly the class whose linear scan runs 5..15
    # iterations; batch<=4 and b1 keep the episode-114 linear-scan compile
    # byte-identically. Host-metadata-only selection, both variants stay in the
    # compile cache, and the two searches return identical segment indices, so
    # outputs are bit-identical either way.
    hd256_remap_bsearch = (
        _HD256_REMAP_HOIST != "0"
        and hd256_persist_eff != "0"
        and (
            _HD256_REMAP_BSEARCH == "1"
            or (_HD256_REMAP_BSEARCH == "auto" and batch_size > 4)
        )
    )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        aux_scalar_metadata,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        q_descale is not None,
        k_descale is not None,
        v_descale is not None,
        block_sparse_tensors is None or block_sparse_tensors.cu_total_m_blocks is None,
        block_sparse_tensors is None or block_sparse_tensors.cu_block_idx_offsets is None,
        tile_m,
        tile_n,
        q_stage,
        num_threads,
        is_split_kv,
        pack_gqa,
        arch,
        page_size not in [None, tile_n],  # paged KV non-TMA
        page_size,  # exact page size (hd256 per-page TMA boxes are page-sized)
        _HD256_PAGED64_TMA,  # hd256 paged64 load mechanism
        _HD256_EX2EMU,  # hd256 exp2 emulation override (dev A/B)
        _HD256_KVSTAGE,  # hd256 KV stage override (dev A/B)
        _HD256_LDRED,  # hd256 SM103 hardware row-max reduction (dev A/B)
        _HD256_FP8_MAX_OFFSET,  # hd256 bounded FP8 rescale policy (dev A/B)
        _HD256_STATIC_MASK_LOOPS,  # hd256 statically partitioned mask loops (dev A/B)
        _HD256_SKIPCORR,  # hd256 conditional correction-rescale skip (dev A/B)
        _HD256_COLEPI,  # hd256 coalesced O epilogue (dev A/B)
        _HD256_EARLY_O_RELEASE,
        _HD256_MRASTER,  # hd256 static M-raster permutation (dev A/B)
        hd256_flatten_varlen,  # compact request-local grid
        hd256_flatten_lpt,  # full-query compact-grid schedule
        hd256_share_rescale,  # full-query shared softmax/correction rescale
        hd256_split_kv_loads,  # native page64 independent K/V TMA producers
        _HD256_PACKGQA,  # hd256 pack_gqa override (dev A/B)
        hd256_persist_eff,  # hd256 persistent grid mode (batch>4 gated)
        hd256_persist_b1,  # hd256 batch-1 flatten+persistence extension
        _HD256_REMAP_HOIST,  # hd256 persistent remap table hoist (dev A/B)
        hd256_remap_bsearch,  # hd256 hoisted remap binary segment search (batch>4)
        use_2cta_instrs,
        q_subtile_factor,
        kv_subtile_factor,
        mma_pv_is_rs,
        intra_wg_overlap,
        use_clc_scheduler,
        q is not None,
        qv is not None,
        p is not None,
        row_max is not None,
        gather_kv_length,
        sparse_kv,
        disable_sparse_kv_bitmask,
        fa_logging.get_fa_log_level(),
    )

    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            if t is not None
            else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
        ]
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        else:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)

        q_descale_tensor, k_descale_tensor, v_descale_tensor = (
            to_cute_tensor(t, assumed_align=4, leading_dim=1)
            for t in (q_descale, k_descale, v_descale)
        )
        descale_tensors_tensor = (
            DescaleTensors(
                q_descale=q_descale_tensor,
                k_descale=k_descale_tensor,
                v_descale=v_descale_tensor,
            )
            if q_descale_tensor is not None
            or k_descale_tensor is not None
            or v_descale_tensor is not None
            else None
        )

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        qv_tensor = to_cute_tensor(qv)
        gather_kv_indices_tensor = to_cute_tensor(gather_kv_indices)
        p_tensor = to_cute_tensor(p)
        row_max_tensor = to_cute_tensor(row_max)

        if arch // 10 == 8:
            assert page_table is None, "paged KV not supported on SM 8.0"
            assert not is_split_kv, "SplitKV not supported on SM 8.0"
            fa_fwd = FlashAttentionForwardSm80(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        elif arch // 10 == 9:
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=intra_wg_overlap,
                mma_pv_is_rs=mma_pv_is_rs,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
                paged_kv_non_tma=page_size not in [None, tile_n],
            )
        elif arch // 10 in [10, 11]:
            if qv is not None:
                paged_kv_cpasync = page_table is not None and page_size != tile_n
                has_qk = q is not None
                fa_fwd = FlashAttentionMLAForwardSm100(
                    is_causal=causal,
                    use_cpasync_load_KV=sparse_kv or paged_kv_cpasync,
                    topk_length=gather_kv_length,
                    is_topk_gather=sparse_kv,
                    pack_gqa=pack_gqa,
                    qhead_per_kvhead=qhead_per_kvhead,
                    nheads_kv=num_head_kv,
                    has_seqused_q=seqused_q is not None,
                    has_cu_seqlens_q=cu_seqlens_q is not None,
                    disable_bitmask=disable_sparse_kv_bitmask,
                    has_qk=has_qk,
                )
            else:
                if use_dedicated_hd256_kernel:
                    # hd=256 2CTA forward: check for currently unsupported features
                    assert softcap is None, "SM100 forward with head_dim=256 does not support softcap"
                    assert not use_block_sparsity, \
                        "SM100 forward with head_dim=256 does not support block sparsity"
                    assert learnable_sink is None, \
                        "SM100 forward with head_dim=256 does not support learnable_sink"
                    assert seqused_q is None, \
                        "SM100 forward with head_dim=256 does not support seqused_q"
                    assert seqused_k is None or page_table is not None, \
                        "SM100 forward with head_dim=256 supports seqused_k only with paged KV"
                    if page_table is not None:
                        assert max_seqlen_k % page_size == 0, (
                            f"SM100 hd256 2CTA paged KV requires max_seqlen_k divisible by "
                            f"page_size ({page_size}), got max_seqlen_k={max_seqlen_k}"
                        )
                        assert page_table.shape[1] == max_seqlen_k // page_size, (
                            f"SM100 hd256 2CTA paged KV requires page_table.shape[1] == "
                            f"max_seqlen_k // page_size ({max_seqlen_k} // {page_size} = "
                            f"{max_seqlen_k // page_size}), got {page_table.shape[1]}; "
                            f"pass page_table[:, :{max_seqlen_k // page_size}] to slice to "
                            f"the actual sequence length"
                        )
                    # pack_gqa packs the query heads of each GQA group into the
                    # tile's M dimension so all heads in a group share KV loads.
                    # The hd256 2-CTA kernel supports it when the M tile divides
                    # evenly by the group size; otherwise keep the unpacked path.
                    if _HD256_PACKGQA == "0":
                        pack_gqa = False
                    elif _HD256_PACKGQA == "1":
                        pack_gqa = tile_m % qhead_per_kvhead == 0
                    elif pack_gqa and tile_m % qhead_per_kvhead != 0:
                        pack_gqa = False

                hd256_paged_kwargs = (
                    {
                        "paged64_use_tma": _HD256_PAGED64_TMA,
                        "paged_page_size": page_size,
                        "ex2_emu_override": _HD256_EX2EMU,
                        "kv_stage_override": _HD256_KVSTAGE,
                        "use_ldred_rowmax": _HD256_LDRED,
                        "fp8_max_offset": _HD256_FP8_MAX_OFFSET,
                        "static_mask_loops": _HD256_STATIC_MASK_LOOPS,
                        "skip_rescale": _HD256_SKIPCORR,
                        "m_raster": _HD256_MRASTER,
                        "coalesced_epilog": _HD256_COLEPI,
                        "early_o_release": _HD256_EARLY_O_RELEASE,
                        "flatten_varlen_grid": hd256_flatten_varlen,
                        "flatten_varlen_lpt": hd256_flatten_lpt,
                        "share_rescale": hd256_share_rescale,
                        "split_kv_loads": hd256_split_kv_loads,
                        "persist_mode": int(hd256_persist_eff),
                        "remap_hoist": _HD256_REMAP_HOIST != "0",
                        "remap_bsearch": hd256_remap_bsearch,
                    }
                    if use_dedicated_hd256_kernel
                    else {}
                )
                flash_fwd_obj_cls = (
                    BlackwellFusedMultiHeadAttentionForward
                    if use_dedicated_hd256_kernel
                    else FlashAttentionForwardSm100
                )

                fa_fwd = flash_fwd_obj_cls(
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    is_split_kv=is_split_kv,
                    pack_gqa=pack_gqa,
                    m_block_size=tile_m,
                    n_block_size=tile_n,
                    q_stage=q_stage,
                    is_persistent=not causal
                        and not local
                        and cu_seqlens_q is None
                        and seqused_q is None
                        and not is_split_kv,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=page_size not in [None, tile_n],
                    is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                    q_subtile_factor=q_subtile_factor,
                    kv_subtile_factor=kv_subtile_factor,
                    use_2cta_instrs=use_2cta_instrs,
                    use_clc_scheduler=use_clc_scheduler,
                    **hd256_paged_kwargs,
                )
        elif arch // 10 == 12:
            # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity
            assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
            assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
            assert not is_split_kv, "SplitKV not supported on SM 12.0 in this PR"
            fa_fwd = FlashAttentionForwardSm120(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {arch}. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
            )
        # TODO: check @can_implement
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                fa_fwd,
                q_tensor,
                qv_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                p_tensor,
                row_max_tensor,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                gather_kv_indices_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                current_stream,
                options="--enable-tvm-ffi",
            )
        else:
            compile_args = [
                fa_fwd,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
            ]
            if use_dedicated_hd256_kernel:
                # Runtime split factor (grid.z decode + per-split trip ranges).
                compile_args.append(num_splits)
            compile_args.append(learnable_sink_tensor)
            if arch // 10 in [10, 11]:
                compile_args.append(descale_tensors_tensor)
            compile_args.extend([
                sparse_tensors,
                AuxData(cute_aux_tensors, aux_scalars),
            ])
            compile_args.append(current_stream)
            _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
                *compile_args, options="--enable-tvm-ffi"
            )

    fast_record = False
    if not is_fake_mode():
        q_call, k_call, v_call, qv_call = [
            t.detach() if t is not None else None
            for t in (q, k, v, qv)
        ]
        if is_fp8:
            # need uint8 workaround until we pin torch >= 2.11.0 where fp8 export is supported
            q_call, k_call, v_call, qv_call = [
                t.view(torch.uint8) if t is not None else None
                for t in (q_call, k_call, v_call, qv_call)
            ]
        descale_tensors = (
            DescaleTensors(q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
            if q_descale is not None or k_descale is not None or v_descale is not None
            else None
        )
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key](
                q_call,
                qv_call,
                k_call,
                v_call,
                out.detach(),
                lse,
                softmax_scale,
                p,
                row_max,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                gather_kv_indices,
                page_table,
                window_size_left,
                window_size_right,
            )
        else:
            call_args = [
                q_call,
                k_call,
                v_call,
                out.detach() if not is_split_kv else out_partial,
                lse_partial if is_split_kv else lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                window_size_left,
                window_size_right,
            ]
            if use_dedicated_hd256_kernel:
                call_args.append(num_splits)
            call_args.append(learnable_sink)
            if arch // 10 in [10, 11]:
                call_args.append(descale_tensors)
            call_args.extend([
                (
                    normalized_block_sparse_tensors.mask_block_cnt,
                    normalized_block_sparse_tensors.mask_block_idx,
                    normalized_block_sparse_tensors.full_block_cnt,
                    normalized_block_sparse_tensors.full_block_idx,
                    normalized_block_sparse_tensors.cu_total_m_blocks,
                    normalized_block_sparse_tensors.cu_block_idx_offsets,
                    normalized_block_sparse_tensors.dq_write_order,
                    normalized_block_sparse_tensors.dq_write_order_full,
                )
                if normalized_block_sparse_tensors is not None
                else None,
                AuxData(aux_tensors, aux_scalars),
            ])
            fn = _flash_attn_fwd.compile_cache[compile_key]
            fast_record = (
                _HD256_FASTPATH_ENABLED
                and use_dedicated_hd256_kernel
                and not is_fake_mode()
                and q is not None
                and k is not None
                and qv is None
                and len(call_args) == 18  # dedicated hd256 arg layout incl. num_splits
            )
            if fast_record and not is_split_kv and call_args[4] is None:
                # Record the exact dispatch for the fast path (see
                # _hd256_fastpath): same compiled kernel, same static tail
                # (num_splits and the trailing None/static slots), fresh
                # tensors plugged into the fixed positions per call.
                sig = _hd256_fast_sig(q, k, v, cu_seqlens_q, seqused_k, page_table)
                if sig is not None:
                    _HD256_FAST_CACHE[sig] = ("main", fn, tuple(call_args[11:]))
            fn(*call_args)
    if is_split_kv:
        if fast_record:
            _HD256_COMBINE_HOOK["fn"] = None
            _HD256_COMBINE_HOOK["armed"] = True
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
        )
        if fast_record:
            # Record the two-kernel split sequence for the fast path: the
            # compiled main kernel (with the num_splits runtime value in the
            # tail) plus the compiled combine kernel selected for it.
            comb_fn = _HD256_COMBINE_HOOK["fn"]
            _HD256_COMBINE_HOOK["fn"] = None
            _HD256_COMBINE_HOOK["armed"] = False
            sig = _hd256_fast_sig(q, k, v, cu_seqlens_q, seqused_k, page_table)
            if sig is not None and comb_fn is not None:
                _HD256_FAST_CACHE[sig] = (
                    "split",
                    fn,
                    tuple(call_args[11:]),
                    comb_fn,
                    tuple(out_partial.shape),
                    tuple(lse_partial.shape),
                )
    return out, lse, p, row_max


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")


# ---------------------------------------------------------------------------
# hd256 fast host dispatch
#
# Eager do_bench-style timing serializes the host submission with the device
# work whenever the kernel is shorter than the Python path, so the per-call
# host cost is part of the measured end-to-end operator latency. For the fp8
# hd256 paged64 varlen causal contract the full dispatch (autograd wrapper,
# validation, heuristics, compile-key construction, argument marshalling)
# costs ~60us of Python; the device kernel for decode-like shapes is shorter
# than that, so the submission gap is fully exposed.
#
# _hd256_fastpath() is consulted at the public entry point and replays the
# exact compiled kernel the full path selected for a shape signature, which
# the full path records here on first dispatch. Split-KV signatures record the
# whole two-kernel sequence (main kernel with its runtime num_splits tail plus
# the matching combine kernel) and replay it with freshly allocated partial
# buffers, so the split shapes pay no more host dispatch than the single-pass
# ones. Anything outside the canonical contract falls back to the full path
# unchanged.
# ---------------------------------------------------------------------------
_HD256_FAST_CACHE: dict = {}
# Hook used by _flash_attn_fwd to capture the compiled combine kernel of a
# split-KV dispatch so the fast path can replay the two-kernel sequence.
_HD256_COMBINE_HOOK: dict = {"fn": None, "armed": False}
_FP8_E4M3_DTYPE = torch.float8_e4m3fn
_I32_DTYPE = torch.int32
_BF16_DTYPE = torch.bfloat16
_U8_DTYPE = torch.uint8
_F32_DTYPE = torch.float32


def _hd256_fast_sig(q, k, v, cu_seqlens_q, seqused_k, page_table):
    """Shape/device signature identifying one compiled hd256 dispatch."""
    if (
        q is None
        or k is None
        or v is None
        or cu_seqlens_q is None
        or seqused_k is None
        or page_table is None
    ):
        return None
    return (
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        tuple(cu_seqlens_q.shape),
        tuple(seqused_k.shape),
        tuple(page_table.shape),
        q.device.index,
        _HD256_PERSIST,  # keep persistent/non-persistent dispatches distinct
        _HD256_PERSIST_B1,  # keep batch-1 flatten/persist dispatches distinct
        _HD256_CLC4,  # keep CLC small-batch dispatches distinct
        _HD256_PERSIST_LE4,  # keep batch<=4 static-persist dispatches distinct
        _HD256_LE4_DMAX,  # D_bound threshold is part of dispatch selection
        _HD256_LE4_KMIN,  # wave floor is part of dispatch selection
        _HD256_LE4_DTIGHT,  # tightened dominance threshold is dispatch selection
        _HD256_LE4_KVSKEW,  # anti-skew bound is part of dispatch selection
        _HD256_REMAP_HOIST,  # keep hoisted/per-tile remap dispatches distinct
        _HD256_REMAP_BSEARCH,  # keep linear/binary segment search distinct
    )


def _hd256_fastpath(q, k, v, cu_seqlens_q, seqused_k, page_table, softmax_scale):
    """Fast dispatch for the fp8 hd256 paged64 varlen causal GQA contract.

    Returns ``(out, None)`` like FlashAttnVarlenFunc.apply, or None to fall
    back to the full path. Only replays recorded dispatches; every structural
    guard below failing means the full path runs exactly as before.
    """
    # dtype / rank / head-dim guards (cheap attribute reads only)
    if q.dtype is not _FP8_E4M3_DTYPE or k.dtype is not _FP8_E4M3_DTYPE or v.dtype is not _FP8_E4M3_DTYPE:
        return None
    if q.dim() != 3 or k.dim() != 4 or v.dim() != 4:
        return None
    q_shape = q.shape
    if q_shape[2] != 256 or v.shape[3] != 256:
        return None
    if q.requires_grad or k.requires_grad or v.requires_grad:
        return None
    # page_size 64 only (the native per-page TMA path; contract page size)
    if k.shape[1] != 64:
        return None
    if (
        cu_seqlens_q.dtype is not _I32_DTYPE
        or seqused_k.dtype is not _I32_DTYPE
        or page_table.dtype is not _I32_DTYPE
    ):
        return None
    h_q = q_shape[1]
    h_k = k.shape[2]
    if h_q % h_k != 0:
        return None
    if page_table.shape[0] != cu_seqlens_q.shape[0] - 1:
        return None
    if cu_seqlens_q.stride(0) != 1 or seqused_k.stride(0) != 1 or page_table.stride(1) != 1:
        return None
    if q_shape[0] == 0:
        return None
    # Canonical contiguous NHD layouts. The full path would silently
    # .contiguous()-ify anything else; the fast path declines it instead.
    if q.stride() != (h_q * 256, 256, 1):
        return None
    k_stride = (64 * h_k * 256, h_k * 256, 256, 1)
    if k.stride() != k_stride or v.stride() != k_stride:
        return None
    sig = (
        q_shape,
        k.shape,
        v.shape,
        cu_seqlens_q.shape,
        seqused_k.shape,
        page_table.shape,
        q.device.index,
    )
    entry = _HD256_FAST_CACHE.get(sig)
    if entry is None:
        return None
    if entry[0] == "split":
        # Split-KV replay is only valid while the split dispatch is enabled;
        # otherwise the full path must re-decide (single-pass) for this shape.
        if _HD256_SPLITKV == "0":
            return None
        _, fn_main, tail_main, fn_comb, op_shape, lse_p_shape = entry
        out = torch.empty(q_shape[0], h_q, 256, dtype=_BF16_DTYPE, device=q.device)
        # One flat fp32 allocation carved into the two partial buffers.
        numel_op = math.prod(op_shape)
        numel_lse = math.prod(lse_p_shape)
        flat = torch.empty(numel_op + numel_lse, dtype=_F32_DTYPE, device=q.device)
        out_partial = flat[:numel_op].view(op_shape)
        lse_partial = flat[numel_op:].view(lse_p_shape)
        fn_main(
            q.view(_U8_DTYPE),
            k.view(_U8_DTYPE),
            v.view(_U8_DTYPE),
            out_partial,
            lse_partial,
            softmax_scale,
            cu_seqlens_q,
            None,
            None,
            seqused_k,
            page_table,
            *tail_main,
        )
        fn_comb(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            None,
            cu_seqlens_q,
            None,
            None,
            None,
            None,
        )
        return out, None
    _, fn, tail = entry
    out = torch.empty(q_shape[0], h_q, 256, dtype=_BF16_DTYPE, device=q.device)
    fn(
        q.view(_U8_DTYPE),
        k.view(_U8_DTYPE),
        v.view(_U8_DTYPE),
        out,
        None,
        softmax_scale,
        cu_seqlens_q,
        None,
        None,
        seqused_k,
        page_table,
        *tail,
    )
    return out, None



def make_fake_bwd_tensors(dtype, has_gqa, varlen_q, varlen_k, nheads_major=False):
    sym = cute.sym_int
    # divisibility in elements: assumed_align_bytes = divisibility * dtype.width // 8
    # For 16-byte align: fp16/bf16 → divisibility=8, float32 → divisibility=4
    div = 128 // dtype.width  # 8 for fp16/bf16
    # Shared sym_ints for dimensions that must match across tensors
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    topk = sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    total_q, total_k, total_q_rounded, total_k_rounded = sym(), sym(), sym(), sym()
    total_q_d_rounded, total_k_d_rounded, total_k_dv_rounded = sym(), sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    mQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mdK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mdV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)

    sq    = seqlen_q         if not varlen_q else total_q
    sq_r  = seqlen_q_rounded if not varlen_q else total_q_rounded
    sq_dr = seqlen_q_d_rounded if not varlen_q else total_q_d_rounded

    def shape(*dims):
        batch = (b,) if not varlen_q else ()
        return (*batch, h_q, *dims) if not nheads_major else (*batch, *dims, h_q)

    mLSE     = fake_tensor(Float32, shape(sq),       divisibility=1)
    mLSElog2 = fake_tensor(Float32, shape(sq_r),     divisibility=4)
    mPdPsum  = fake_tensor(Float32, shape(sq_r),     divisibility=4)
    dQaccum  = fake_tensor(Float32, shape(sq_dr),    divisibility=4)
    mScaleP  = fake_tensor(Float32, shape(sq, topk), divisibility=4)

    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        if not varlen_k:
            mdKaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
        else:
            mdKaccum = fake_tensor(Float32, (h_kv, total_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (h_kv, total_k_dv_rounded), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, dQaccum, mdKaccum, mdVaccum, mScaleP


def _compile_bwd_preprocess(
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    has_cuseqlens_q,
    has_seqused_q,
    has_dlse,
    has_dq_accum,
    has_scaleP,
    use_padded_offsets,
    nheads_major,
    pack_gqa,
    qhead_per_kvhead,
    nheads_kv,
):
    """Compile bwd preprocess kernel using cute fake tensors (no real GPU tensors needed)."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum, mScaleP = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False, nheads_major=nheads_major,
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSequsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
    mLSElog2 = None if has_scaleP else mLSElog2
    mdQaccum = mdQaccum if has_dq_accum else None
    mRowMax = fake_tensor(Float32, mScaleP.shape, divisibility=1) if has_scaleP else None
    mScaleP = fake_tensor(Float32, mScaleP.shape, divisibility=1) if has_scaleP else None
    softmax_scale = Float32(1.0)
    fa_bwd_pre = FlashAttentionBackwardPreprocess(
        dtype, head_dim, head_dim_v, m_block_size,
        use_padded_offsets=use_padded_offsets,
        nheads_major=nheads_major,
        pack_gqa=pack_gqa,
        qhead_per_kvhead=qhead_per_kvhead,
        nheads_kv=nheads_kv,
    )
    return cute.compile(
        fa_bwd_pre, mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, mCuSeqlensQ, mSequsedQ, mdLSE,
        mRowMax, mScaleP, softmax_scale,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(
    out, dout, dpsum, lse, lse_log2, dq_accum,
    cu_seqlens_q, seqused_q, dlse,
    dtype, head_dim, head_dim_v, m_block_size,
    row_max=None,
    scale_p=None,
    use_padded_offsets=True,
    nheads_major=False,
    pack_gqa=False,
    qhead_per_kvhead=1,  # only used with pack_gqa
    nheads_kv=1,         # only used with pack_gqa
    softmax_scale=1.0,   # only used with scale_p
):
    """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum."""
    if row_max is not None:
        assert scale_p is not None
    compile_key = (
        dtype, head_dim, head_dim_v, m_block_size,
        cu_seqlens_q is not None,
        seqused_q is not None,
        dlse is not None,
        dq_accum is not None,
        row_max is not None,
        use_padded_offsets,
        nheads_major,
        pack_gqa,
        qhead_per_kvhead,
        nheads_kv,
    )
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
    if not is_fake_mode():
        _bwd_preprocess.compile_cache[compile_key](
            out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q, seqused_q, dlse,
            row_max, scale_p,
            softmax_scale,
        )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre")


def _compile_bwd_postprocess(
    dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
    has_cuseqlens_q, has_seqused_q,
    use_2cta_instrs, cluster_size, arch,
):
    """Compile bwd postprocess kernel using cute fake tensors."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum, mScaleP = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSeqUsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    fa_bwd_post = FlashAttentionBackwardPostprocess(
        dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab,
        use_2cta_instrs=use_2cta_instrs,
        cluster_size=cluster_size,
    )
    return cute.compile(
        fa_bwd_post, mdQaccum, mdQ, Float32(0.0), mCuSeqlensQ, mSeqUsedQ,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(
    accum, output, scale,
    cu_seqlens, seqused,
    arch, dtype, hdim, block_size, num_threads,
    atom_layout, swap_ab,
    use_2cta_instrs=False, cluster_size=1,
):
    """Backward postprocess: convert float32 accumulator to bf16/fp16 output."""
    compile_key = (
        dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
        cu_seqlens is not None, seqused is not None,
        use_2cta_instrs, cluster_size, arch,
    )
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(*compile_key)
    if not is_fake_mode():
        _bwd_postprocess_convert.compile_cache[compile_key](
            accum, output, scale, cu_seqlens, seqused,
        )


_bwd_postprocess_convert.compile_cache = get_jit_cache("bwd_post")


def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aux_scalars = tuple(aux_scalars) if aux_scalars else None
    arch = _get_device_arch()
    assert arch // 10 in [9, 10, 11, 12], "Unsupported compute capability. Supported: 9.x, 10.x, 11.x, 12.x"
    if block_sparse_tensors is not None:
        assert (
            cu_seqlens_q is None
            and cu_seqlens_k is None
            and seqused_q is None
            and seqused_k is None
        ), "Varlen backward with block sparsity is not yet supported"
    sparse_q = None
    kv_subtile_factor = 1
    if block_sparse_tensors is not None:
        if block_sparse_tensors.block_size is not None:
            sparse_q = block_sparse_tensors.block_size[0]
        elif arch // 10 == 9:
            sparse_q = 128

    num_head, head_dim = q.shape[-2:]
    head_dim_v = v.shape[-1]

    window_size = [window_size_left, window_size_right]
    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right
    )

    if arch // 10 == 12:
        # SM120: uses SM80 MMA with 99 KB SMEM, 128 threads (4 warps).
        m_block_size = 64
        n_block_size = 64
        if head_dim <= 64:
            num_stages_Q = 2
            num_stages_dO = 2
        else:
            num_stages_Q = 1
            num_stages_dO = 1
        SdP_swapAB = False
        dKV_swapAB = False
        dQ_swapAB = False
        AtomLayoutMSdP = 4
        AtomLayoutNdKV = 4
        AtomLayoutMdQ = 4
        V_in_regs = False
        dQ_single_wg = False
        cluster_size = 1
        use_2cta_instrs = False
        num_threads = 128
        assert not (block_sparse_tensors is not None), "Block sparsity backward not supported on SM 12.0"
        assert score_mod is None and score_mod_bwd is None, "score_mod backward not supported on SM 12.0"
        assert mask_mod is None, "mask_mod backward not supported on SM 12.0"
        assert deterministic is False, "deterministic backward not supported on SM 12.0"
    elif arch // 10 == 9:
        cfg = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            causal,
            local,
            sparse_block_size_q=sparse_q,
        )
        m_block_size = cfg.m_block_size
        n_block_size = cfg.n_block_size
        num_stages_Q = cfg.num_stages_Q
        num_stages_dO = cfg.num_stages_dO
        num_stages_PdS = cfg.num_stages_PdS
        SdP_swapAB = cfg.SdP_swapAB
        dKV_swapAB = cfg.dKV_swapAB
        dQ_swapAB = cfg.dQ_swapAB
        AtomLayoutMSdP = cfg.AtomLayoutMSdP
        AtomLayoutNdKV = cfg.AtomLayoutNdKV
        AtomLayoutMdQ = cfg.AtomLayoutMdQ
        num_threads = (cfg.num_wg + 1) * 128
        dQ_single_wg = cfg.dQ_single_wg
        cluster_size = 1
        use_2cta_instrs = False
        is_varlen = (
            cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or seqused_q is not None
            or seqused_k is not None
        )
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1
        requested_disable_2cta = utils._get_disable_2cta_default()
        kv_subtile_factor = get_kv_subtile_factor(block_sparse_tensors, n_block_size)
        use_2cta_instrs = (
            head_dim >= 128
            and not requested_disable_2cta
            and block_sparse_bwd_supports_2cta(block_sparse_tensors, n_block_size)
        )
        if block_sparse_tensors is not None and head_dim == 192 and not use_2cta_instrs:
            reason = (
                "2CTA was disabled by request"
                if requested_disable_2cta
                else (
                    f"sparse_block_size[1] must cover an even number of tile_n={n_block_size} "
                    f"tiles; got factor {kv_subtile_factor}"
                )
            )
            raise ValueError(
                f"SM100 block-sparse backward with head_dim=192 requires 2CTA; {reason}."
            )
        cluster_size = 2 if use_2cta_instrs else 1

    use_dedicated_hd256_kernel = arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel

    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = [
        maybe_contiguous(t)
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q.shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        total_k = k.shape[0]
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k

    num_head_kv = k.shape[-2]

    use_block_sparsity = block_sparse_tensors is not None
    if sparse_q is not None and (sparse_q <= 0 or sparse_q % m_block_size != 0):
        raise ValueError(
            "Block sparsity requires sparse_block_size[0] to be a multiple of "
            f"tile_m={m_block_size}; got {sparse_q}."
        )
    q_subtile_factor = sparse_q // m_block_size if sparse_q is not None else 2
    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
    num_n_blocks = seqlen_k_rounded // n_block_size
    if cluster_size == 2 and num_n_blocks % cluster_size != 0:
        seqlen_k_rounded = seqlen_k_rounded + n_block_size

    # The single-block specialization below only guards against TVM stride poisoning,
    # which is a host-side branch predicate that selects a kernel variant. When
    # max_seqlen is passed as a tensor (e.g. HF/TE varlen), seqlen_*_rounded are tensors,
    # so `seqlen_*_rounded // block == 1` would leak a tensor into the compile key. Its
    # pickle hash differs every call, forcing a recompile per step. Only specialize when
    # the seqlen is already a host scalar; tensor callers fall back to the multi-block
    # default, keeping the key stable with no device sync.
    single_q_block = (not torch.is_tensor(seqlen_q_rounded)) and (seqlen_q_rounded // m_block_size == 1)
    single_k_block = (not torch.is_tensor(seqlen_k_rounded)) and (seqlen_k_rounded // n_block_size == 1)

    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

        assert out.shape == (total_q, num_head, head_dim_v)
        assert dout.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (num_head, total_q), "lse must have shape (num_head, total_q)"
    else:
        assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert lse.shape == (batch_size, num_head, seqlen_q), (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    if dlse is not None:
        dlse = maybe_contiguous(dlse)
    if not is_fake_mode():
        assert all(
            t is None or t.is_cuda for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
        ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    if arch // 10 != 12:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    # pack_gqa backward not yet supported in bwd
    pack_gqa = False
    
    if softcap != 0.0:
        assert score_mod is None and score_mod_bwd is None, (
            "softcap and score_mod/score_mod_bwd cannot be used together"
        )
        score_mod = utils.create_softcap_scoremod(softcap)
        score_mod_bwd = utils.create_softcap_scoremod_bwd(softcap)
    if score_mod is not None:
        assert score_mod_bwd is not None, "score_mod_bwd is required when score_mod is provided"
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")

    device = q.device
    out_torch_dtype = q.dtype

    if dq is None:
        dq = torch.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = torch.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = torch.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if cu_seqlens_q is None:
        dq_accum = (
            None
            if use_dedicated_hd256_kernel
            else torch.empty(
                batch_size,
                num_head,
                seqlen_q_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
        )
        dpsum = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        )
        dq_accum = (
            None
            if use_dedicated_hd256_kernel
            else torch.empty(
                num_head, total_q_rounded_padded * head_dim_rounded, dtype=torch.float32, device=device
            )
        )
        dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

    # GQA (qhead_per_kvhead > 1) needs dK/dV accum+postprocess since multiple Q heads
    # accumulate into the same dK/dV. SM90 varlen_k with qhead_per_kvhead==1 now uses
    # ragged TMA tensors for direct store, so no longer needs accum+postprocess.
    # hd=256 2CTA backward has its own internal postprocess for dK/dV.
    dKV_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel
    if dKV_postprocess:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        if cu_seqlens_k is None:
            dk_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )
        else:
            cluster_tile_n = cluster_size * n_block_size
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * cluster_tile_n - 1) // cluster_tile_n * cluster_tile_n
            )
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if deterministic:
        dQ_semaphore = torch.zeros(batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size, dtype=torch.int32, device=device)
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
        dV_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
    else:
        dK_semaphore = None
        dV_semaphore = None

    # Preprocess kernel: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum.
    # For hd=256 dedicated path, dq_accum is None so preprocess only fills dpsum/lse_log2.
    _bwd_preprocess(
        out, dout, dpsum, lse, lse_log2, dq_accum,
        cu_seqlens_q, seqused_q, dlse,
        dtype, head_dim, head_dim_v, m_block_size,
    )
    # num_threads: SM90 derives from BwdConfig.num_wg, SM120 is set to 128 above,
    # SM100/SM110 uses default from function signature (384).
    if arch // 10 not in [9, 12]:
        num_threads = 384

    # Backward kernel: compute dk, dv, dq_accum.
    score_mod_hash = utils.hash_callable(score_mod) if score_mod else False
    score_mod_bwd_hash = utils.hash_callable(score_mod_bwd) if score_mod_bwd else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod else False
    num_aux_tensors = len(aux_tensors) if aux_tensors else 0
    aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors) if aux_tensors is not None else None
    aux_scalar_metadata = tuple(type(s) for s in aux_scalars) if aux_scalars is not None else None
    cute_aux_tensors = None
    if aux_tensors is not None:
        cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    if use_block_sparsity:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
        ) = normalize_block_sparse_config_bwd(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            q_subtile_factor=q_subtile_factor,
            kv_subtile_factor=kv_subtile_factor,
        )
        if deterministic:
            if normalized_block_sparse_tensors.dq_write_order is None:
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order in block_sparse_tensors"
                )
            if (
                normalized_block_sparse_tensors.full_block_cnt is not None
                and normalized_block_sparse_tensors.dq_write_order_full is None
            ):
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order_full when full blocks are present"
                )
            if normalized_block_sparse_tensors.spt is None:
                raise ValueError(
                    "deterministic block-sparse backward requires block_sparse_tensors.spt "
                    "to match dq_write_order direction"
                )
    if (
        normalized_block_sparse_tensors is not None
        and normalized_block_sparse_tensors.spt is not None
    ):
        spt = normalized_block_sparse_tensors.spt and deterministic
    else:
        spt = (causal or local) and deterministic

    if arch // 10 in [8, 9, 12]:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
            dQ_single_wg,
            deterministic,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            aux_tensor_metadata,
            aux_scalar_metadata,
            use_block_sparsity,
            q_subtile_factor,
            block_sparse_broadcast_pattern,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # Prevent TVM stride poisoning when only one block is present.
            single_q_block,
            single_k_block,
        )
    else:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
            use_2cta_instrs,
            q_subtile_factor,
            kv_subtile_factor,
            deterministic,
            spt,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            aux_tensor_metadata,
            aux_scalar_metadata,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # Prevent TVM stride poisoning when only one block is present.
            single_q_block,
            single_k_block,
        )

    if compile_key not in _flash_attn_bwd.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        lse_log2_tensor, dpsum_tensor = [to_cute_tensor(t) for t in (lse_log2, dpsum)]
        dq_accum_tensor = to_cute_tensor(dq_accum) if dq_accum is not None else None
        if dKV_postprocess:
            dk_accum_tensor, dv_accum_tensor = [
                to_cute_tensor(t) for t in (dk_accum, dv_accum)
            ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ]
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order())
            if t is not None else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        if arch // 10 in [8, 12]:
            flash_bwd_obj_cls = FlashAttentionBackwardSm120 if arch // 10 == 12 else FlashAttentionBackwardSm80
            fa_bwd_obj = flash_bwd_obj_cls(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                m_block_size,
                n_block_size,
                num_stages_Q,
                num_stages_dO,
                num_threads,
                pack_gqa,
                causal,
                local,
                SdP_swapAB,
                dKV_swapAB,
                dQ_swapAB,
                AtomLayoutMSdP,
                AtomLayoutNdKV,
                AtomLayoutMdQ,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
            )
        elif arch // 10 == 9:
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                is_local=local,
                deterministic=deterministic,
                tile_m=m_block_size,
                tile_n=n_block_size,
                Q_stage=num_stages_Q,
                dO_stage=num_stages_dO,
                PdS_stage=num_stages_PdS,
                SdP_swapAB=SdP_swapAB,
                dKV_swapAB=dKV_swapAB,
                dQ_swapAB=dQ_swapAB,
                AtomLayoutMSdP=AtomLayoutMSdP,
                AtomLayoutNdKV=AtomLayoutNdKV,
                AtomLayoutMdQ=AtomLayoutMdQ,
                num_threads=num_threads,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
                dQ_single_wg=dQ_single_wg,
            )
        else:
            if use_dedicated_hd256_kernel:
                assert softcap == 0.0, "SM100 backward with head_dim=256 does not support softcap"
                assert block_sparse_tensors is None, \
                    "SM100 backward with head_dim=256 does not support block sparsity"
                assert dlse is None, \
                    "SM100 backward with head_dim=256 does not support dlse"
                assert seqused_q is None and seqused_k is None, \
                    "SM100 backward with head_dim=256 does not support seqused_q/seqused_k"

                dq_tile_mn = (128, 128)
                dkdv_tile_mn = (128, 64)
                fa_bwd_obj = BlackwellFusedMultiHeadAttentionBackward(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    is_local=local,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_persistent=False,
                    deterministic=deterministic,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    q_subtile_factor=q_subtile_factor,
                    tile_m_dq=dq_tile_mn[0],
                    tile_n_dq=dq_tile_mn[1],
                    tile_m_dkdv=dkdv_tile_mn[0],
                    tile_n_dkdv=dkdv_tile_mn[1],
                )
            else:
                fa_bwd_obj = FlashAttentionBackwardSm100(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    is_local=local,
                    qhead_per_kvhead=qhead_per_kvhead,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    deterministic=deterministic,
                    spt=spt,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    q_subtile_factor=q_subtile_factor,
                    kv_subtile_factor=kv_subtile_factor,
                )

        # Block sparse tensors for backward use Q-direction indexing (transposed from forward).
        sparse_tensors_compile = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors_compile = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)
        dq_accum_tensor = dq_tensor if use_dedicated_hd256_kernel else dq_accum_tensor

        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not dKV_postprocess else dk_accum_tensor,
            dv_tensor if not dKV_postprocess else dv_accum_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            window_size_left,
            window_size_right,
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            AuxData(cute_aux_tensors, aux_scalars),
            sparse_tensors_compile,
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        dq_accum = dq if use_dedicated_hd256_kernel else dq_accum
        _flash_attn_bwd.compile_cache[compile_key](
            q.detach(),
            k.detach(),
            v.detach(),
            dout,
            lse_log2,
            dpsum,
            dq_accum,
            dk if not dKV_postprocess else dk_accum,
            dv if not dKV_postprocess else dv_accum,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            window_size_left,
            window_size_right,
            dQ_semaphore,
            dK_semaphore,
            dV_semaphore,
            AuxData(aux_tensors, aux_scalars),
            (
                normalized_block_sparse_tensors.mask_block_cnt,
                normalized_block_sparse_tensors.mask_block_idx,
                normalized_block_sparse_tensors.full_block_cnt,
                normalized_block_sparse_tensors.full_block_idx,
                normalized_block_sparse_tensors.cu_total_m_blocks,
                normalized_block_sparse_tensors.cu_block_idx_offsets,
                normalized_block_sparse_tensors.dq_write_order,
                normalized_block_sparse_tensors.dq_write_order_full,
            )
            if normalized_block_sparse_tensors is not None
            else None,
        )
    # Postprocess: convert dq_accum from float32 to dq in bf16/fp16
    # hd=256 2CTA backward has its own internal postprocess, skip here.
    if not use_dedicated_hd256_kernel:
        if arch // 10 == 9:
            # dQ postprocess: match main kernel's MMA WG count, unless dQ_single_wg
            num_threads_post_dQ = 128 if dQ_single_wg else cfg.num_wg * 128
            num_threads_post_dKV = cfg.num_wg * 128
        else:
            num_threads_post_dQ = 128
            num_threads_post_dKV = 128

        _bwd_postprocess_convert(
            dq_accum, dq, softmax_scale,
            cu_seqlens_q, seqused_q,
            arch, dtype, head_dim, m_block_size, num_threads_post_dQ,
            AtomLayoutMdQ, dQ_swapAB,
            use_2cta_instrs=use_2cta_instrs, cluster_size=1,
        )

        if dKV_postprocess:
            # Postprocess: convert dk_accum from float32 to dk in bf16/fp16
            _bwd_postprocess_convert(
                dk_accum, dk, softmax_scale,
                cu_seqlens_k, seqused_k,
                arch, dtype, head_dim, n_block_size, num_threads_post_dKV,
                AtomLayoutNdKV, dKV_swapAB,
                cluster_size=cluster_size,
            )
            # Postprocess: convert dv_accum from float32 to dv in bf16/fp16
            _bwd_postprocess_convert(
                dv_accum, dv, 1.0,
                cu_seqlens_k, seqused_k,
                arch, dtype, head_dim_v, n_block_size, num_threads_post_dKV,
                AtomLayoutNdKV, dKV_swapAB,
                cluster_size=cluster_size,
            )

    return dq, dk, dv


_flash_attn_bwd.compile_cache = get_jit_cache("bwd")


def _flash_attn_bwd_sparse_mla(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: torch.Tensor,
    qv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    p: torch.Tensor,
    row_max: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    learnable_sink: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    m_block_size: int = 128,
    n_block_size: int = 64,
    num_threads: int = 256,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    dqv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    arch = _get_device_arch()
    assert arch // 10 in [10, 11], "Unsupported compute capability. Supported: 10.x, 11.x"
    assert gather_kv_indices is not None, "require gather kv indices for backward"

    q_shape = q.shape if q is not None else qv.shape
    nheads, head_dim = q_shape[-2:]
    nheads_kv, head_dim_v = v.shape[-2:]
    qhead_per_kvhead = nheads // nheads_kv
    gather_kv_length = gather_kv_indices.shape[-1]
    assert nheads_kv == 1 and qhead_per_kvhead == 128, f"sparse MLA bwd: only MQA 128 supported for now"
    assert gather_kv_length % 128 == 0, f"sparse MLA bwd: {gather_kv_length=} must be divisible by 128"
    assert deterministic is False, "sparse MLA bwd: deterministic mode not yet supported"
    assert learnable_sink is None, "sparse MLA bwd: learnable sink not yet supported"
    assert seqused_q is None and seqused_k is None, "sparse MLA bwd: seqused_q,k not yet supported"

    if softmax_scale is None:
        softmax_scale = (
            1.0 / math.sqrt(head_dim) if qv is None or q is None
            else 1.0 / math.sqrt(head_dim + head_dim_v)
        )

    q, k, v, qv, out, dout, lse, p, row_max = [
        maybe_contiguous(t)
        for t in (q, k, v, qv, out, dout, lse, p, row_max)
    ]
    gather_kv_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink = [
        maybe_contiguous(t)
        for t in (gather_kv_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
    ]
    device = v.device

    varlen_q = cu_seqlens_q is not None or seqused_q is not None
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q_shape[:2]
        total_q = batch_size * seqlen_q
        p_shape = (batch_size, seqlen_q, nheads, gather_kv_length)
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q_shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q
        p_shape = (total_q, nheads, gather_kv_length)

    varlen_k = cu_seqlens_k is not None or seqused_k is not None
    if cu_seqlens_k is None:
        batch_size, seqlen_k = v.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        total_k = v.shape[0]
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k
    if not varlen_k:
        min_seqlen_k = seqlen_k 

    assert varlen_q == varlen_k, "sparse MLA bwd: either q and k are both varlen or not"

    # always use kv bitmask by default (handles -1 sentinel)
    disable_sparse_kv_bitmask = False
    # if min_seqlen_k is None or causal:
    #     disable_sparse_kv_bitmask = False
    # else:
    #     disable_sparse_kv_bitmask = min_seqlen_k >= gather_kv_length

    prealloc_dq = dq is not None
    prealloc_dk = dk is not None
    prealloc_dqv = dqv is not None
    prealloc_dv = dv is not None
    dq = dk = None
    if not prealloc_dq and q is not None:
        dq = torch.empty_like(q)
    if not prealloc_dk and k is not None:
        dk = torch.zeros_like(k, dtype=torch.float32)
    if not prealloc_dv:
        dv = torch.zeros_like(v, dtype=torch.float32)
    if not prealloc_dqv:
        dqv = torch.empty_like(qv)
    ds = torch.empty_like(p)

    device = v.device
    dtype = v.dtype
    if q is not None:
        _validate_tensor(dq, "dq", q.shape, dtype, device)
    if k is not None:
        _validate_tensor(dk, "dk", k.shape, torch.float32, device)
    _validate_tensor(dv, "dv", v.shape, torch.float32, device)
    _validate_tensor(dqv, "dqv", qv.shape, dtype, device)
    _validate_tensor(p, "p", p_shape, dtype, device)

    if cu_seqlens_q is None:
        dpsum = torch.empty(batch_size, seqlen_q, nheads, dtype=torch.float32, device=device)
    else:
        dpsum = torch.empty(total_q, nheads, dtype=torch.float32, device=device)
    scale_p = torch.empty_like(row_max)

    dtype = torch2cute_dtype_map[dout.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Preprocess kernel: compute (o * dout).sum(dim=-1), scale_p.
    _bwd_preprocess(
        out, dout, dpsum, lse, None, None,
        cu_seqlens_q, seqused_q, None,
        dtype, head_dim, head_dim_v, m_block_size,
        row_max=row_max,
        scale_p=scale_p,
        use_padded_offsets=False,
        nheads_major=True,
        pack_gqa=True,
        qhead_per_kvhead=qhead_per_kvhead,
        nheads_kv=nheads_kv,
        softmax_scale=softmax_scale,
    )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        q is not None,
        gather_kv_length,
        learnable_sink is not None,
        disable_sparse_kv_bitmask,
    )

    if compile_key not in _flash_attn_bwd_sparse_mla.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        (
            v_tensor,
            qv_tensor,
            do_tensor,
            p_tensor,
            scale_p_tensor,
            dpsum_tensor,
            ds_tensor,
            dv_tensor,
            gather_kv_indices_tensor,
         ) = [
            to_cute_tensor(t) for t in (v, qv, dout, p, scale_p, dpsum, ds, dv, gather_kv_indices)
        ]

        fa_bwd_obj = FlashAttentionSparseMLABackwardSm100(
            is_causal=causal,
            topk_length=gather_kv_length,
            qhead_per_kvhead=qhead_per_kvhead,
            nheads_kv=nheads_kv,
            has_seqused_q=seqused_q is not None,
            disable_bitmask=disable_sparse_kv_bitmask,
        )
        fa_bwd_kernel = cute.compile(
            fa_bwd_obj,
            do_tensor,
            v_tensor,
            qv_tensor,
            p_tensor,
            dv_tensor,
            ds_tensor,
            gather_kv_indices_tensor,
            softmax_scale,
            scale_p_tensor,
            dpsum_tensor,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
        _flash_attn_bwd_sparse_mla.compile_cache[compile_key] = fa_bwd_kernel

    if not is_fake_mode():
        _flash_attn_bwd_sparse_mla.compile_cache[compile_key](
            dout,
            v,
            qv,
            p,
            dv,
            ds,
            gather_kv_indices,
            softmax_scale,
            scale_p,
            dpsum,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
        )

    v = v.squeeze(-2)
    if k is not None:
        k = k.squeeze(-2)
    
    _sparse_mla_dq_dqv(
        ds, k, v, dq, dqv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k,
    )

    if k is not None:
        dk = dk.squeeze(-2)
        _sparse_mla_dk(ds, gather_kv_indices, q, dk, cu_seqlens_q, cu_seqlens_k)
        dk = dk.unsqueeze(-2)
    
    # return dk, dv in float32: all-reduce across sequence-parallel ranks must happen
    # before downcasting to avoid rounding error during inter-rank grad accumulation
    return dq, dk, dv, dqv

_flash_attn_bwd_sparse_mla.compile_cache = get_jit_cache("bwd_dsa")


def _compile_sparse_mla_dq_dqv(
    dtype, nheads, head_dim, head_dim_v, top_k, varlen_q, varlen_k, compute_dq,
):
    sym = cute.sym_int 
    b, b_plus_1, seqlen_q, seqlen_k = sym(), sym(), sym(), sym()
    total_q, total_k = sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    
    div = 128 // dtype.width  # 8 for fp16/bf16
    
    mdS = fake_tensor(dtype, (*b_seqlenq, nheads, top_k), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, head_dim), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, head_dim_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, nheads, head_dim), divisibility=div)
    mdQv = fake_tensor(dtype, (*b_seqlenq, nheads, head_dim_v), divisibility=div)
    mIdxTopK = fake_tensor(Int32, (*b_seqlenq, top_k), divisibility=div)
    
    mCuSeqlensQ = fake_tensor(Int32, (b_plus_1,), divisibility=1) if varlen_q else None 
    mCuSeqlensK = fake_tensor(Int32, (b_plus_1,), divisibility=1) if varlen_k else None 
    
    dq_dqv_gemm = dQdQvGemmKernel(
        acc_dtype=Float32,
        nheads=nheads,
        head_dim_k=head_dim,
        head_dim_v=head_dim_v,
        top_k=top_k,
    )
    
    return cute.compile(
        dq_dqv_gemm,
        mdS,
        mK if compute_dq else None,
        mV,
        mdQ if compute_dq else None,
        mdQv,
        mIdxTopK,
        mCuSeqlensQ,
        mCuSeqlensK,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _sparse_mla_dq_dqv(
    ds, k, v, dq, dqv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k,
):
    """Compute dQ = dS @ K and dQv = dS @ V"""
    *_, nheads, gather_kv_length = ds.shape
    
    head_dim_v = v.shape[-1]
    head_dim = k.shape[-1] if k is not None else 0
    
    dtype = ds.dtype
    dtype_cute = torch2cute_dtype_map[dtype]
    
    varlen_q = cu_seqlens_q is not None
    varlen_k = cu_seqlens_k is not None
    
    compile_key = (
        dtype_cute, nheads, head_dim, head_dim_v, gather_kv_length, varlen_q, varlen_k, k is not None,
    )
    if compile_key not in _sparse_mla_dq_dqv.compile_cache:
        _sparse_mla_dq_dqv.compile_cache[compile_key] = _compile_sparse_mla_dq_dqv(
            *compile_key
        )
    if not is_fake_mode():
        _sparse_mla_dq_dqv.compile_cache[compile_key](
            ds, k, v, dq, dqv, gather_kv_indices, cu_seqlens_q, cu_seqlens_k
        )

_sparse_mla_dq_dqv.compile_cache = get_jit_cache("dq_dqv_gemm")


def _compile_sparse_mla_dk(
    dtype,
    dtype_acc,
    nheads: int,
    head_dim: int,
    topk: int,
    varlen: bool,
):
    kernel = dKGemmKernel(
        topk,
        nheads,
        head_dim,
        varlen,
    )
    # Check if configuration can be implemented
    kernel.check_can_implement()

    div = 128 // dtype.width

    sym = cute.sym_int
    batch_fake = sym()
    batchp1_fake = sym()
    seqlen_q_fake = sym()
    seqlen_k_fake = sym()
    total_q_fake = (batch_fake, seqlen_q_fake) if not varlen else (sym(),)
    total_k_fake = (batch_fake, seqlen_k_fake) if not varlen else (sym(),)

    mdS = fake_tensor(dtype, (*total_q_fake, nheads, topk), divisibility=div)
    mI = fake_tensor(Int32, (*total_q_fake, topk), divisibility=div)
    mQ = fake_tensor(dtype, (*total_q_fake, nheads, head_dim), divisibility=div)
    mdK = fake_tensor(dtype_acc, (*total_k_fake, head_dim), divisibility=div)
    mCuSeqlensQ = fake_tensor(Int32, (batchp1_fake,), divisibility=1) if varlen else None
    mCuSeqlensK = fake_tensor(Int32, (batchp1_fake,), divisibility=1) if varlen else None
    
    return cute.compile(
        kernel,
        mdS,
        mI,
        mQ,
        mdK,
        mCuSeqlensQ,
        mCuSeqlensK,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _sparse_mla_dk(
    dS: torch.Tensor,
    index_topk: torch.Tensor,
    q: torch.Tensor,
    dk: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
):
    """Compute dKaccum = scatter(dS'^T @ Q, I).

    Args:
      dS:          (*total_q, heads, topk), bf16
      index_topk:  (*total_q, topk), int32
      Q:           (*total_q, heads, dim), bf16
      dK:          (*total_q, dim), fp32
      cuSeqlensQ:  (batch + 1,), int32, omit for non-varlen
      cuSeqlensK:  (batch + 1,), int32, omit for non-varlen

    Accumulates in place on top of dK.

    For varlen, total_q and total_k are 1-dimensional, and the seqlen indices per batch are
    determined using the cuSeqlensQ and cuSeqlensK tensors.
    For non-varlen, total_q and total_k are (batch, seqlen_q) and (batch, seqlen_k).
    """
    dtype = dS.dtype
    dtype_cute = torch2cute_dtype_map[dtype]
    dtype_acc = dk.dtype
    dtype_acc_cute = torch2cute_dtype_map[dtype_acc]

    varlen = cu_seqlens_q is not None
    nheads, topk = dS.shape[-2], dS.shape[-1]
    head_dim = q.shape[-1] if q is not None else 0

    compile_key = (
        dtype_cute, dtype_acc_cute, nheads, head_dim, topk, varlen,
    )

    if compile_key not in _sparse_mla_dk.compile_cache:
        _sparse_mla_dk.compile_cache[compile_key] = _compile_sparse_mla_dk(*compile_key)

    if not is_fake_mode():
        _sparse_mla_dk.compile_cache[compile_key](dS, index_topk, q, dk, cu_seqlens_q, cu_seqlens_k)
    
_sparse_mla_dk.compile_cache = get_jit_cache("dk_gemm")


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        score_mod_bwd: Optional[Callable] = None,
        mask_mod: Optional[Callable] = None,
        aux_tensors: Optional[list] = None,
        aux_scalars: Optional[tuple] = None,
        block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
        block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
        return_lse: bool = False,
    ):
        aux_scalars = tuple(aux_scalars) if aux_scalars else None
        shared_kv = k is v
        if shared_kv and v.shape[-1] == 512:
            # specialize MLA attention formula
            # O = softmax(Q @ K.T + Qv @ V.T) @ V
            # by setting q, k to None
            qv = q if qv is None else qv
            q = k = None
        out, lse, p, row_max = _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            aux_scalars=aux_scalars,
            block_sparse_tensors=block_sparse_tensors,
            return_lse=return_lse,
            gather_kv_indices=gather_kv_indices,
        )
        ctx.save_for_backward(q, k, v, qv, out, lse, p, row_max, gather_kv_indices, *(aux_tensors or ()))
        ctx.shared_kv = shared_kv
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.score_mod = score_mod 
        ctx.score_mod_bwd = score_mod_bwd 
        ctx.mask_mod = mask_mod
        ctx.aux_scalars = aux_scalars
        ctx.block_sparse_tensors_bwd = block_sparse_tensors_bwd
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, qv, out, lse, p, row_max, gather_kv_indices, *aux = ctx.saved_tensors
        aux_tensors = aux if aux else None
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        if qv is not None:
            dq, dk, dv, dqv = _flash_attn_bwd_sparse_mla(
                q,
                k,
                v,
                qv,
                out,
                dout,
                lse,
                p,
                row_max,
                gather_kv_indices,
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
            )
            if ctx.shared_kv:
                return dqv, dv, None, None, *((None,) * 30)
            else:
                return dq, dk, dv, dqv, *((None,) * 30)
        else:
            dq, dk, dv = _flash_attn_bwd(
                q,
                k,
                v,
                out,
                dout,
                lse,
                ctx.softmax_scale,
                ctx.causal,
                ctx.softcap,
                window_size_left=ctx.window_size[0],
                window_size_right=ctx.window_size[1],
                deterministic=ctx.deterministic,
                score_mod=ctx.score_mod,
                score_mod_bwd=ctx.score_mod_bwd,
                mask_mod=ctx.mask_mod,
                aux_tensors=aux_tensors,
                aux_scalars=ctx.aux_scalars,
                block_sparse_tensors=ctx.block_sparse_tensors_bwd,
                dlse=dlse,
            )
            return dq, dk, dv, *((None,) * 30)  # Extra Nones is fine


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: torch.Tensor,
        qv: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        min_seqlen_k: Optional[int] = None,
        gather_kv_indices: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        score_mod_bwd: Optional[Callable] = None,
        mask_mod: Optional[Callable] = None,
        block_sparse_tensors: Optional[list] = None,
        aux_tensors: Optional[list] = None,
        aux_scalars: Optional[tuple] = None,
        return_lse: bool = False,
    ):
        aux_scalars = tuple(aux_scalars) if aux_scalars else None
        shared_kv = k is v
        if shared_kv and v.shape[-1] == 512:
            # specialize MLA attention formula
            # O = softmax(Q @ K.T + Qv @ V.T) @ V
            # by setting q, k to None
            qv = q if qv is None else qv
            q = k = None
        out, lse, p, row_max = _flash_attn_fwd(
            q,
            k,
            v,
            qv=qv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_k=min_seqlen_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors,
            aux_tensors=aux_tensors,
            aux_scalars=aux_scalars,
            return_lse=return_lse,
            gather_kv_indices=gather_kv_indices,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            qv,
            out,
            lse,
            p,
            row_max,
            gather_kv_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            *(aux_tensors or ()),
        )
        ctx.shared_kv = shared_kv
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.min_seqlen_k = min_seqlen_k
        ctx.return_lse = return_lse
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.mask_mod = mask_mod
        ctx.aux_scalars = aux_scalars
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, qv, out, lse, p, row_max, gather_kv_indices, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, *aux = ctx.saved_tensors
        aux_tensors = aux if aux else None
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        if qv is not None:
            dq, dk, dv, dqv = _flash_attn_bwd_sparse_mla(
                q,
                k,
                v,
                qv,
                out,
                dout,
                lse,
                p,
                row_max,
                gather_kv_indices,
                softmax_scale=ctx.softmax_scale,
                causal=ctx.causal,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                min_seqlen_k=ctx.min_seqlen_k,
            )
            if ctx.shared_kv:
                return dqv, dv, None, None, *((None,) * 31)
            else:
                return dq, dk, dv, dqv, *((None,) * 31)
        else:
            dq, dk, dv = _flash_attn_bwd(
                q,
                k,
                v,
                out,
                dout,
                lse,
                ctx.softmax_scale,
                ctx.causal,
                ctx.softcap,
                window_size_left=ctx.window_size[0],
                window_size_right=ctx.window_size[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                deterministic=ctx.deterministic,
                score_mod=ctx.score_mod,
                score_mod_bwd=ctx.score_mod_bwd,
                aux_tensors=aux_tensors,
                aux_scalars=ctx.aux_scalars,
                mask_mod=ctx.mask_mod,
                dlse=dlse,
            )
            return dq, dk, dv, *((None,) * 31)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        qv,
        gather_kv_indices,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        aux_tensors,
        aux_scalars,
        block_sparse_tensors,
        block_sparse_tensors_bwd,
        return_lse,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    return_lse: bool = False,
):
    """
    Tensor arguments:
        q:  (total_q, nheads,   hdim)   or (batch, seqlen_q, nheads,   hdim)
        k:  (total_k, nheads_k, hdim)   or (batch, seqlen_k, nheads_k, hdim)
        v:  (total_k, nheads_k, hdim_v) or (batch, seqlen_k, nheads_k, hdim_v)
        qv: (total_q, nheads,   hdim_v) or (batch, seqlen_q, nheads,   hdim_v)
        cu_seqlens_q: (batch + 1)       or seqused_q: (batch)
        cu_seqlens_k: (batch + 1)       or seqused_k: (batch)
        gather_kv_indices: (total_q, gather_kv_length) or
                           (batch, seqlen_q, gather_kv_length)
        page_table: (batch, max_num_pages_per_seq)
    
    Return:
       out: (total_q, nheads, hdim) or (batch, seqlen_q, nheads, hdim)
       lse: (nheads, total_q)       or (batch, nheads, seqlen_q) if not has_qv (standard)
            (total_q, nheads)       or (batch, seqlen_q, nheads) if has_qv

    Explanation of some optional arguments & decisions:

    qv: we write the MLA weight absorbed formula as
        O = softmax(scale * (Q @ K.T + Qv @ V.T)) @ V
        where Q = q_pe, Qv = q_nope, K = pe_cache, V = kv_cache.

    lse return shape: with Qv, MQA with nheads at least divisible by 4 is typical,
        so we arrange for nheads as the contiguous mode for better vectorization.

    gather_kv_indices: used for topk sparsity with MLA absorption kernel.
    """
    if (
        _HD256_FASTPATH_ENABLED
        and causal
        and pack_gqa
        and num_splits == 1
        and not return_lse
        and qv is None
        and cu_seqlens_q is not None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is not None
        and gather_kv_indices is None
        and page_table is not None
        and softmax_scale is not None
        and window_size == (None, None)
        and learnable_sink is None
        and softcap == 0.0
        and not deterministic
        and score_mod is None
        and mask_mod is None
        and block_sparse_tensors is None
        and aux_tensors is None
        and aux_scalars is None
    ):
        res = _hd256_fastpath(q, k, v, cu_seqlens_q, seqused_k, page_table, softmax_scale)
        if res is not None:
            return res
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_k,
        gather_kv_indices,
        page_table,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        score_mod,
        score_mod_bwd,
        mask_mod,
        block_sparse_tensors,
        aux_tensors,
        aux_scalars,
        return_lse,
    )


def _compile_fwd_combine(
    dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
    has_cu_seqlens, has_seqused, has_lse, has_varlen_batch_idx,
):
    """Compile fwd combine kernel using cute fake tensors (no real GPU tensors needed)."""
    sym = cute.sym_int
    div = 128 // dtype_partial.width  # 16-byte alignment in elements

    fa_combine = FlashAttentionForwardCombine(
        dtype=dtype,
        dtype_partial=dtype_partial,
        head_dim=head_dim,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
    )
    if not fa_combine.can_implement(
        dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
        num_threads=256,
    ):
        raise RuntimeError(
            "FlashAttention combine kernel cannot be implemented with given parameters"
        )

    if has_cu_seqlens:
        # Varlen: (num_splits, total_q, nheads, headdim)
        num_splits, total_q, nheads = sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, total_q, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=1)
        mO = fake_tensor(dtype, (total_q, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=0) if has_lse else None
    else:
        # Batched: (num_splits, batch, seqlen, nheads, headdim)
        num_splits, batch, seqlen, nheads = sym(), sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, batch, seqlen, nheads), divisibility=1, leading_dim=2)
        mO = fake_tensor(dtype, (batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (batch, seqlen, nheads), divisibility=1, leading_dim=1) if has_lse else None
        batch = mO_partial.shape[1]

    batch_for_1d = batch if not has_cu_seqlens else sym()
    batchp1 = sym()
    mCuSeqlens = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cu_seqlens else None
    mSeqused = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_seqused else None
    mNumSplitsDynamic = None  # Not parametrized in compile_key
    mVarlenBatchIdx = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_varlen_batch_idx else None
    mSemaphore = None  # Not parametrized in compile_key

    return cute.compile(
        fa_combine,
        mO_partial, mLSE_partial, mO, mLSE,
        mCuSeqlens, mSeqused, mNumSplitsDynamic, mVarlenBatchIdx, mSemaphore,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                       (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    if not is_fake_mode():
        assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4
    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            if not is_fake_mode():
                assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"
    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    # Create combine kernel configuration
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]
    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
        varlen_batch_idx is not None,
    )
    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        _flash_attn_fwd_combine.compile_cache[compile_key] = _compile_fwd_combine(
            *compile_key
        )
    if _HD256_COMBINE_HOOK.get("armed"):
        # hd256 split fast path: expose the selected compiled combine kernel
        # to the caller (no-op for every other consumer).
        _HD256_COMBINE_HOOK["fn"] = _flash_attn_fwd_combine.compile_cache[compile_key]
    if not is_fake_mode():
        _flash_attn_fwd_combine.compile_cache[compile_key](
            out_partial, lse_partial, out, lse,
            cu_seqlens, seqused, num_splits_dynamic_ptr, varlen_batch_idx,
            semaphore_to_reset,
        )


_flash_attn_fwd_combine.compile_cache = get_jit_cache("fwd_combine")


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flash Attention combine function for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs. This is the main user-facing
    interface for the combine kernel.

    Args:
        out_partial: Partial outputs tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads, head_size) for regular batched input
            - (num_splits, total_q, num_heads, head_size) for variable length input
        lse_partial: Partial LSE tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads) for regular batched input
            - (num_splits, total_q, num_heads) for variable length input
        out: Optional output tensor. If None, will be created automatically.
        out_dtype: Optional output dtype. If None, will use fp16/bf16 based on input.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        varlen_batch_idx: Optional mapping from virtual batch index to real batch index
            (int32 tensor of shape (batch_size,)). Used by persistent tile schedulers
            that reorder batch processing for load balancing.
        return_lse: Whether to return the combined LSE tensor. Default is True.

    Returns:
        Tuple of (out, lse) where:
        - out: Combined output tensor with shape (batch_size, seqlen, num_heads, head_size)
              or (total_q, num_heads, head_size) for varlen
        - lse: Combined log-sum-exp tensor with shape (batch_size, seqlen, num_heads)
              or (total_q, num_heads) for varlen. None if return_lse=False

    Note:
        This function expects the input tensors to be in the format produced by
        split attention computation, where the first dimension is num_splits.
        The permuting from user format to kernel format is now done inside the kernel.
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4
    if is_varlen:
        # Variable length: (num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        batch_size = 1  # Treat as single batch for varlen
        seqlen = total_q
    else:
        # Regular batched: (num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
    # Determine output dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype
    # Create output if not provided
    device = out_partial.device
    if out is None:
        if is_varlen:
            out = torch.empty(total_q, num_heads, head_size, dtype=out_dtype, device=device)
        else:
            out = torch.empty(
                batch_size, seqlen, num_heads, head_size, dtype=out_dtype, device=device
            )
    # Create lse output only if requested
    if return_lse:
        if is_varlen:
            lse = torch.empty(num_heads, total_q, dtype=torch.float32, device=device)
        else:
            lse = torch.empty(batch_size, num_heads, seqlen, dtype=torch.float32, device=device)
        lse = lse.transpose(-1, -2)
    else:
        lse = None
    _flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        varlen_batch_idx=varlen_batch_idx,
    )
    return out, lse
