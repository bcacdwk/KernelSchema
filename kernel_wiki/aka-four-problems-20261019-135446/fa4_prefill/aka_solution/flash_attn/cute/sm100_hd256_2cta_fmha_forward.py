# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

import math
from typing import Tuple, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL
from cutlass.cute.typing import Int32, Int64, Float32

from cutlass.utils import ClcDynamicPersistentTileScheduler
from flash_attn.cute.tile_scheduler import (
    ClcState,
    compute_sm100_fmha_grid as compute_grid,
    compute_sm100_fmha_grid_clc as compute_grid_clc,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
    Sm100FmhaStaticTileScheduler as FmhaStaticTileScheduler,
    Sm100FmhaStaticTileSchedulerParams as FmhaStaticTileSchedulerParams,
    Sm100FmhaClcDynamicTileScheduler as FmhaClcDynamicTileScheduler,
    Sm100FmhaClcDynamicTileSchedulerParams as FmhaClcDynamicTileSchedulerParams,
)
from flash_attn.cute.mask import (
    Sm100FusedMask as FusedMask,
)
from flash_attn.cute.pack_gqa import PackGQA
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.tile_scheduler import SM100_TMEM_CAPACITY_COLUMNS
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors, _TUNING_CONFIG
from flash_attn.cute.utils import ex2_emulation_2, as_bshkrd_tensor, AuxData, elem_pointer
from quack import copy_utils as quack_copy_utils


class BlackwellFusedMultiHeadAttentionForward:
    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int = 1,
        kv_subtile_factor: int = 1,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
        is_persistent: bool = True,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors: bool = False,
        paged_kv_non_tma: bool = False,
        paged64_use_tma: bool = True,
        paged_page_size: Optional[int] = None,
        ex2_emu_override: str = "auto",
        kv_stage_override: int = 0,
        use_ldred_rowmax: bool = True,
        fp8_max_offset: bool = True,
        static_mask_loops: bool = False,
        skip_rescale: bool = True,
        m_raster: int = 0,
        coalesced_epilog: bool = True,
        early_o_release: bool = False,
        flatten_varlen_grid: bool = False,
        flatten_varlen_lpt: bool = False,
        share_rescale: bool = False,
        split_kv_loads: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        use_clc_scheduler: bool = False,
        persist_mode: int = 0,
        remap_hoist: bool = False,
        remap_bsearch: bool = False,
    ):
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert head_dim == 256 and head_dim_v == 256, (
            "SM100 dedicated kernel only supports (head_dim, head_dim_v) = (256, 256)"
        )
        assert score_mod is None, "SM100 forward with head_dim=256 does not support score_mod"
        assert mask_mod is None, "SM100 forward with head_dim=256 does not support mask_mod"
        assert not has_aux_tensors, "SM100 forward with head_dim=256 does not support aux tensors"
        # paged_kv_non_tma=True: paged KV with page_size != tile_n (e.g. page_size 64,
        # tile_n 128). With paged64_use_tma (default) each CTA's stage box is loaded
        # with per-page TMA copies (one 64-token page == one CTA K-stage box; two
        # pages per V stage). With paged64_use_tma=False the stages are assembled
        # with per-row cp.async loads instead (debug fallback).
        self.paged_kv_non_tma = paged_kv_non_tma
        self.use_tma_page_loads = paged_kv_non_tma and paged64_use_tma
        self.paged_page_size = paged_page_size
        if self.use_tma_page_loads:
            assert paged_page_size is not None and n_block_size == 2 * paged_page_size, (
                "hd256 per-page TMA loads require page_size == tile_n / 2, got "
                f"page_size={paged_page_size}, tile_n={n_block_size}"
            )
        # pack_gqa folds the qhead_per_kvhead query heads of each GQA group into
        # the M (row) dimension of the 2-CTA tile so that all heads in a group
        # share the same KV tile loads (KV traffic drops by qhead_per_kvhead).
        # Row r of a packed tile maps to (head = r % h_r, token = r // h_r).
        # The cluster M tile (256) and each CTA's half (128) must be divisible by
        # h_r so TMA boxes align on group boundaries.
        self.pack_gqa = pack_gqa
        if pack_gqa:
            assert m_block_size % qhead_per_kvhead == 0, (
                f"SM100 hd256 pack_gqa requires tile_m ({m_block_size}) divisible by "
                f"qhead_per_kvhead ({qhead_per_kvhead})"
            )
        # Split-KV: the KV tile range of each work tile is partitioned across
        # num_splits clusters (grid.z = batch * num_splits); every split writes
        # an fp32 O partial (unnormalized by splits, normalized by its own
        # row_sum) plus an LSE partial, which the stock flash_fwd_combine kernel
        # reduces. Used for low-parallelism (short packed-q) shapes where the
        # M x kv-head cluster count underfills the machine.
        self.is_split_kv = is_split_kv
        if is_split_kv:
            assert pack_gqa, (
                "SM100 forward with head_dim=256 split-KV requires pack_gqa"
            )
        assert q_subtile_factor == 1, (
            "SM100 forward with head_dim=256 does not support q_subtile_factor"
        )
        assert kv_subtile_factor == 1, (
            "SM100 forward with head_dim=256 does not support kv_subtile_factor"
        )
        assert m_block_size == 128 and n_block_size == 128, (
            "SM100 dedicated kernel only supports tile_m=128 and tile_n=128"
        )
        # q_stage / persistence / scheduler knobs are accepted for interface parity,
        # but this dedicated kernel uses fixed internal settings.

        qk_acc_dtype = cutlass.Float32
        pv_acc_dtype = cutlass.Float32
        mma_tiler = (128, 128, head_dim)
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.qhead_per_kvhead = qhead_per_kvhead
        self.mma_tiler = mma_tiler
        assert mma_tiler[0] == 128 and mma_tiler[1] == 128, "Only 128x128 tile impl is supported"
        assert mma_tiler[2] == 256, "Only 256 is supported for 128x128 tile impl"
        self.cta_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.qk_mma_tiler = (
            2 * mma_tiler[0],
            mma_tiler[1],
            min(self.cta_tiler[2], 128),
        )
        self.pv_mma_tiler = self.qk_mma_tiler
        self.pv_block_tiler = (
            self.pv_mma_tiler[0] // 2,
            self.pv_mma_tiler[1],
            self.pv_mma_tiler[2],
        )
        self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
        self.iterations_pv = self.cta_tiler[2] // self.pv_mma_tiler[1]
        self.cluster_shape_mn = (2, 1)
        self.tmem_warp_shape_mn = (4, 1)
        # Dedicated hd256 kernel uses fixed scheduling policy, except for the
        # opt-in persistent multi-tile grid (FA_HD256_PERSIST -> persist_mode) on
        # the flattened varlen non-split path. There the four warp-section work
        # loops are already re-entrant (pipelines/TMEM/barriers are per-CTA and
        # allocated once outside the loop; producer.tail() runs after it), so a
        # cluster can amortize CTA launch + TMEM alloc + cluster barrier chain +
        # exit across many work tiles instead of paying them once per tile.
        # persist_mode: 0=off (incumbent), 1=plain stride, 2=serpentine-balanced
        # (no within-request m_raster), 3=serpentine + m_raster, 4=CLC dynamic
        # persistent (Blackwell cluster-launch-control work stealing: the full
        # logical grid is launched, resident clusters cancel pending clusters
        # and inherit their coords, so tile handout is dynamically load
        # balanced - the safe persistence mode for heterogeneous small batches
        # where static stride overloads the CTAs that receive heavy clusters).
        self.persist_mode = int(persist_mode)
        # Hoisted request-mapping table for the persistent flattened grid
        # (FA_HD256_REMAP_HOIST): build the LPT-sorted segment request ids and
        # cluster prefix boundaries ONCE per CTA in smem instead of re-deriving
        # them from cum_seqlen_q/seqused_k global loads on every tile in all
        # four warp-section work loops. Pure latency hoist: the table entries
        # are exactly the values the per-tile remap computes, so the tile
        # mapping (and every output tile's math) is bit-identical.
        self.remap_hoist = bool(remap_hoist)
        # Bounded binary segment search inside the hoisted remap table
        # (FA_HD256_REMAP_BSEARCH): the hoisted per-tile remap still walks the
        # request prefix table LINEARLY (request_count-1 iterations, up to 15
        # for the batch-16 flatten cap) on every tile in all four warp-section
        # loops. For request_count>4 the search becomes a fixed 4-step binary
        # search over the same ascending prefix table (16 segments -> 4
        # halvings), which returns the identical segment index. request_count<=4
        # keeps the linear scan (at most 3 cheap iterations, fewer dependent
        # smem loads than 4 binary steps). Value-identical => bit-identical
        # outputs; only the exposed per-tile remap latency changes.
        self.remap_bsearch = bool(remap_bsearch)
        self.is_persistent = (
            self.persist_mode > 0
            and flatten_varlen_grid
            and pack_gqa
            and not is_split_kv
        )
        self.is_causal = is_causal
        self.is_local = is_local
        self.use_semantic_trip_range = is_causal or is_local
        # CLC reuses the flattened-grid per-tile remap: the try_cancel response
        # carries the canceled cluster's raw linear M-CTA index in the padded
        # flattened grid, and _persist_remap_coord maps it to
        # (mapped_m_cta, mapped_batch) exactly like the static stride does.
        self.use_clc_scheduler = self.is_persistent and self.persist_mode == 4

        self.softmax_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_id = (10, 11)
        self.v_load_warp_id = self.empty_warp_id[0]
        # The CLC scheduler warp must not collide with the split-KV V-load warp
        # (v_load_warp_id == empty_warp_id[0] enters the LOAD consumer loop when
        # split_kv_loads is on); warp 11 is unconditionally free.
        self.sched_warp_id = self.empty_warp_id[1] if self.use_clc_scheduler else None
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax_warp_ids,  # this is to get a round num threads
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.empty_warp_id,
            )
        )

        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )

        self.tmem_s_offset = 0
        self.tmem_o_offset = 256
        self.tmem_p_offset = self.tmem_s_offset

        # sm_103 (B300) has a fast hardware ex2 unit; like the general SM100
        # kernel, exp2 emulation should be disabled there.  Detect the real arch
        # instead of assuming sm100 so the hd256 kernel picks the right tuning.
        _arch = BaseDSL._get_dsl().get_arch_enum()
        self.is_sm103 = _arch.is_family_of(Arch.sm_103f)
        _tune_key = (True, is_causal, 256, self.is_sm103)
        _tune = _TUNING_CONFIG.get(_tune_key, {})
        # Fall back to the sm100 entry when an arch-specific one is absent.
        if not _tune:
            _tune = _TUNING_CONFIG.get((True, is_causal, 256, False), {})
        self.num_regs_softmax = _tune.get("num_regs_softmax", 256)
        self.num_regs_correction = _tune.get("num_regs_correction", 160)
        self.num_regs_other = 32  # fixed for hd256; not derived from 512 budget like other kernels
        self.ex2_emu_freq = _tune.get("ex2_emu_freq", 4)
        self.ex2_emu_res = _tune.get("ex2_emu_res", 3)
        self.ex2_emu_start_frg = _tune.get("ex2_emu_start_frg", 0)
        # Dev A/B: exp2 emulation override. "0" forces hardware ex2 for every
        # element (freq=0), "1" forces the tuned emulation setting, an integer
        # sets ex2_emu_freq directly (freq sweep), "auto" keeps the
        # arch-selected default.
        self.ex2_emu_override = ex2_emu_override
        if ex2_emu_override == "0":
            self.ex2_emu_freq = 0
        elif ex2_emu_override == "1":
            if self.ex2_emu_freq == 0:
                self.ex2_emu_freq = 14
        elif ex2_emu_override.isdigit():
            self.ex2_emu_freq = int(ex2_emu_override)
        self.kv_stage_override = kv_stage_override
        self.use_ldred_rowmax = use_ldred_rowmax and self.is_sm103
        self.fp8_max_offset = fp8_max_offset
        self.static_mask_loops = static_mask_loops and self.is_causal and not self.is_local
        # Warp-uniform conditional skip of the per-tile O rescale in the
        # correction warps (exact no-op when the row max is unchanged). Dev A/B
        # via FA_HD256_SKIPCORR in interface.py; the default keeps it enabled.
        self.skip_rescale = skip_rescale
        # Static M-raster permutation for the non-persistent scheduler (dev A/B):
        # 0 = identity grid order; 1 = reversed cluster order; 2 = stride-2
        # "halves" interleave that spreads light/heavy causal work evenly over
        # the cluster launch order (balances the causal diagonal's monotone
        # trip counts and ragged multi-request work across the streaming
        # launch window). The permutation is applied at cluster granularity
        # (cluster M index = blockIdx.x // 2) and preserves the CTA parity
        # inside each 2-CTA cluster, so it stays a bijection over the work
        # tiles.
        self.m_raster = m_raster
        # Coalesced O epilogue: stage the finalized O tile in smem and store it
        # with 128-bit coalesced global stores (row-wise, PackGQA pointer
        # predication) instead of the per-element TMEM-fragment autovec stores,
        # which are 2-byte-per-sector uncoalesced (NCU: 93% excessive sectors,
        # L1TEX the busiest pipe). Dev A/B via FA_HD256_COLEPI in interface.py;
        # the default keeps it enabled.
        self.coalesced_epilog = coalesced_epilog
        self.early_o_release = early_o_release and coalesced_epilog
        self.flatten_varlen_grid = flatten_varlen_grid
        self.flatten_varlen_lpt = flatten_varlen_lpt
        self.share_rescale = share_rescale
        self.split_kv_loads = split_kv_loads

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.q_stage = self.iterations_qk
        self.kv_stage = self.kv_stage_override if self.kv_stage_override > 0 else 4
        self.qk_acc_stage = 2
        self.mma_corr_stage = 1
        if cutlass.const_expr(self.use_clc_scheduler):
            self.num_clc_stage = 1
            self.num_clc_response_bytes = 16

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        num_splits: Int32 | int = 1,
        learnable_sink: Optional[cute.Tensor] = None,
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        stream: cuda.CUstream = None,
    ):
        # Keep parity with FlashAttentionForwardSm100.__call__ interface.
        # (TODO@wangsiyu) Implement these features.
        assert mSeqUsedQ is None, (
            "SM100 forward with head_dim=256 does not support seqused_q"
        )
        assert mSeqUsedK is None or mPageTable is not None, (
            "SM100 forward with head_dim=256 supports seqused_k only with paged KV"
        )
        assert learnable_sink is None, (
            "SM100 forward with head_dim=256 does not support learnable_sink"
        )
        assert blocksparse_tensors is None, (
            "SM100 forward with head_dim=256 does not support block sparsity"
        )
        assert aux_data.tensors is None, (
            "SM100 forward with head_dim=256 does not support aux_tensors"
        )
        assert aux_data.scalars is None, (
            "SM100 forward with head_dim=256 does not support aux_scalars"
        )
        assert not self.is_local, (
            "SM100 forward with head_dim=256 does not support local attention yet"
        )
        assert window_size_left is None and window_size_right is None, (
            "SM100 forward with head_dim=256 does not support runtime window_size overrides"
        )
        assert descale_tensors is None, (
            "SM100 forward with head_dim=256 does not support descale_tensors"
        )

        q_tensor, k_tensor, v_tensor, o_tensor = mQ, mK, mV, mO
        lse_tensor = mLSE
        cum_seqlen_q = mCuSeqlensQ
        cum_seqlen_k = mCuSeqlensK
        seqused_k = mSeqUsedK

        q_rank = len(mQ.shape)
        k_rank = len(mK.shape)
        if cutlass.const_expr(cum_seqlen_q is not None):
            # Varlen path accepts either legacy 5D tensors or standard 3D tensors.
            if cutlass.const_expr(q_rank == 5):
                s_q = mQ.shape[1]
                h_q = mQ.shape[2] * mQ.shape[3]
                d = mQ.shape[4]
            elif cutlass.const_expr(q_rank == 3):
                s_q = mQ.shape[0]
                h_q = mQ.shape[1]
                d = mQ.shape[2]
            else:
                raise RuntimeError(f"hd256 forward varlen expects q rank 3 or 5, got rank {q_rank}")
        else:
            # Non-varlen path accepts either legacy 5D tensors or standard 4D tensors.
            if cutlass.const_expr(q_rank == 5):
                s_q = mQ.shape[1]
                h_q = mQ.shape[2] * mQ.shape[3]
                d = mQ.shape[4]
            elif cutlass.const_expr(q_rank == 4):
                s_q = mQ.shape[1]
                h_q = mQ.shape[2]
                d = mQ.shape[3]
            else:
                raise RuntimeError(
                    f"hd256 forward non-varlen expects q rank 4 or 5, got rank {q_rank}"
                )

        if cutlass.const_expr(cum_seqlen_k is not None):
            if cutlass.const_expr(k_rank == 5):
                s_k = mK.shape[1]
                h_k = mK.shape[2]
            elif cutlass.const_expr(k_rank == 3):
                s_k = mK.shape[0]
                h_k = mK.shape[1]
            else:
                raise RuntimeError(f"hd256 forward varlen expects k rank 3 or 5, got rank {k_rank}")
        else:
            if cutlass.const_expr(k_rank == 5):
                s_k = mK.shape[1]
                h_k = mK.shape[2]
            elif cutlass.const_expr(k_rank == 4):
                s_k = mK.shape[1]
                h_k = mK.shape[2]
            else:
                raise RuntimeError(
                    f"hd256 forward non-varlen expects k rank 4 or 5, got rank {k_rank}"
                )
        if cutlass.const_expr(cum_seqlen_q is not None):
            b = mCuSeqlensQ.shape[0] - 1
        elif cutlass.const_expr(cum_seqlen_k is not None):
            b = mCuSeqlensK.shape[0] - 1
        else:
            b = mQ.shape[0]

        scale_softmax = softmax_scale
        scale_softmax_log2 = softmax_scale * math.log2(math.exp(1.0))
        scale_output = 1.0
        s_lse = s_q
        h_r = h_q // h_k
        s_q64 = Int64(s_q)
        s_k64 = Int64(s_k)
        s_lse64 = Int64(s_lse)
        h_r64 = Int64(h_r)
        h_k64 = Int64(h_k)
        b64 = Int64(b)
        s_q_total = (
            q_tensor.shape[1]
            if cum_seqlen_q is not None and q_rank == 5
            else (q_tensor.shape[0] if cum_seqlen_q is not None else s_q64)
        )
        s_k_total = (
            k_tensor.shape[1]
            if cum_seqlen_k is not None and k_rank == 5
            else (k_tensor.shape[0] if cum_seqlen_k is not None else s_k64)
        )
        b_lse = b64 if cum_seqlen_q is None else 1
        stride_b_lse = h_r64 * h_k64 * s_lse64 if cum_seqlen_q is None else 0

        varlen_q = cum_seqlen_q is not None
        varlen_k = cum_seqlen_k is not None
        q_norm = as_bshkrd_tensor(q_tensor, h_k, h_r, varlen_q)
        if cutlass.const_expr(self.is_split_kv):
            # Split-KV builds the partial-O layout directly from the rank-4
            # out_partial tensor below; the rank-3/5 normalizer does not apply.
            o_norm = None
        else:
            o_norm = as_bshkrd_tensor(o_tensor, h_k, h_r, varlen_q)

        if cutlass.const_expr(self.pack_gqa):
            assert lse_tensor is None or self.is_split_kv, (
                "SM100 forward with head_dim=256 pack_gqa does not support LSE output "
                "(split-KV partial LSE is internal only)"
            )
        if cutlass.const_expr(self.is_split_kv):
            assert mPageTable is not None, (
                "SM100 forward with head_dim=256 split-KV requires paged KV"
            )
        if cutlass.const_expr(self.pack_gqa):
            # Packed forward layout: ((h_r, s), d, (h_k, b)). Row index
            # r = token * h_r + head_in_group; the h_r heads of a GQA group are
            # the fastest submode so each group's rows land on aligned segments.
            q = cute.make_tensor(
                q_norm.iterator,
                cute.make_layout(
                    ((h_r, s_q_total), d, (h_k, b)),
                    stride=(
                        (q_norm.stride[3], q_norm.stride[1]),
                        q_norm.stride[4],
                        (q_norm.stride[2], q_norm.stride[0]),
                    ),
                ),
            )
        else:
            # Forward layout: (s, d, ((h_r, h_k), b)). Stride picks from canonical
            # positions 1=S, 4=D, 3=H_r, 2=H_k, 0=B.
            q = cute.make_tensor(
                q_norm.iterator,
                cute.make_layout(
                    (s_q_total, d, ((h_r, h_k), b)),
                    stride=(
                        q_norm.stride[1],
                        q_norm.stride[4],
                        ((q_norm.stride[3], q_norm.stride[2]), q_norm.stride[0]),
                    ),
                ),
            )
        if cutlass.const_expr(mPageTable is not None):
            # Paged: input k/v are rank-4 (num_pages, page_size, h_k, d); the kernel
            # consumes K as (page_size, d, h_k, num_pages) and V as
            # (d, page_size, h_k, num_pages).
            # cute.select reorders modes while preserving input strides
            page_size = k_tensor.shape[1]
            max_seqlen_k_paged = Int32(mPageTable.shape[1] * page_size)
            k = cute.make_tensor(k_tensor.iterator, cute.select(k_tensor.layout, mode=[1, 3, 2, 0]))
            v = cute.make_tensor(v_tensor.iterator, cute.select(v_tensor.layout, mode=[3, 1, 2, 0]))
            page_table = cute.make_tensor(
                mPageTable.iterator,
                cute.make_layout(
                    (b, mPageTable.shape[1]),
                    stride=(mPageTable.stride[0], mPageTable.stride[1]),
                ),
            )
        else:
            # K/V have no h_r dim; pass h_r=1 to the normalizer and override the
            # h_r stride to 0 below to broadcast across the query-grouped heads.
            k_norm = as_bshkrd_tensor(k_tensor, h_k, 1, varlen_k)
            v_norm = as_bshkrd_tensor(v_tensor, h_k, 1, varlen_k)
            if cutlass.const_expr(self.pack_gqa):
                # Packed Q rows already carry the group head index; K/V use a
                # plain (h_k, b) head mode (no h_r broadcast dimension).
                k = cute.make_tensor(
                    k_norm.iterator,
                    cute.make_layout(
                        (s_k_total, d, (h_k, b)),
                        stride=(
                            k_norm.stride[1],
                            k_norm.stride[4],
                            (k_norm.stride[2], k_norm.stride[0]),
                        ),
                    ),
                )
                v = cute.make_tensor(
                    v_norm.iterator,
                    cute.make_layout(
                        (d, s_k_total, (h_k, b)),
                        stride=(
                            v_norm.stride[4],
                            v_norm.stride[1],
                            (v_norm.stride[2], v_norm.stride[0]),
                        ),
                    ),
                )
            else:
                # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
                k = cute.make_tensor(
                    k_norm.iterator,
                    cute.make_layout(
                        (s_k_total, d, ((h_r, h_k), b)),
                        stride=(
                            k_norm.stride[1],
                            k_norm.stride[4],
                            ((0, k_norm.stride[2]), k_norm.stride[0]),
                        ),
                    ),
                )
                # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
                v = cute.make_tensor(
                    v_norm.iterator,
                    cute.make_layout(
                        (d, s_k_total, ((h_r, h_k), b)),
                        stride=(
                            v_norm.stride[4],
                            v_norm.stride[1],
                            ((0, v_norm.stride[2]), v_norm.stride[0]),
                        ),
                    ),
                )
            page_table = None
            max_seqlen_k_paged = None
        # (s, d, ((h_r, h_k), b)) or packed ((h_r, s), d, (h_k, b))
        if cutlass.const_expr(self.is_split_kv):
            # Split-KV partial O: out_partial (num_splits, total_q, num_heads, d)
            # fp32. Rows are stored UNPACKED (token, head) so the stock combine
            # kernel can reduce over splits with the varlen cu_seqlens layout.
            # Reuse the packed nested row mode so the correction epilog's tile
            # partitioning/predicates stay unchanged; only the strides unpack.
            o = cute.make_tensor(
                o_tensor.iterator,
                cute.make_layout(
                    ((h_r, s_q_total), d, (h_k, b), o_tensor.shape[0]),
                    stride=(
                        (o_tensor.stride[2], o_tensor.stride[1]),
                        o_tensor.stride[3],
                        (o_tensor.stride[2] * h_r, 0),
                        o_tensor.stride[0],
                    ),
                ),
            )
        elif cutlass.const_expr(self.pack_gqa):
            o = cute.make_tensor(
                o_norm.iterator,
                cute.make_layout(
                    ((h_r, s_q_total), d, (h_k, b)),
                    stride=(
                        (o_norm.stride[3], o_norm.stride[1]),
                        o_norm.stride[4],
                        (o_norm.stride[2], o_norm.stride[0]),
                    ),
                ),
            )
        else:
            o = cute.make_tensor(
                o_norm.iterator,
                cute.make_layout(
                    (s_q_total, d, ((h_r, h_k), b)),
                    stride=(
                        o_norm.stride[1],
                        o_norm.stride[4],
                        ((o_norm.stride[3], o_norm.stride[2]), o_norm.stride[0]),
                    ),
                ),
            )
        if cutlass.const_expr(lse_tensor is not None):
            if cutlass.const_expr(self.is_split_kv):
                # Split-KV partial LSE: lse_partial (num_splits, num_heads, total_q);
                # addressed as (token, ((h_r, h_k), 1), split).
                lse_layout = cute.make_layout(
                    (s_lse64, ((h_r, h_k), Int64(1)), lse_tensor.shape[0]),
                    stride=(1, ((s_lse64, h_r64 * s_lse64), 0), s_lse64 * h_r64 * h_k64),
                )
            else:
                # (s, ((h_r, h_k), b))
                lse_layout = cute.make_layout(
                    (s_lse64, ((h_r, h_k), b_lse)),
                    stride=(1, ((s_lse64, h_r64 * s_lse64), stride_b_lse)),
                )
            lse = cute.make_tensor(lse_tensor.iterator, lse_layout)
        else:
            lse = None

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.softmax_max_offset = (
            4
            if self.fp8_max_offset and self.q_dtype is cutlass.Float8E4M3FN
            else 8 if self.fp8_max_offset and self.q_dtype.width == 8 else 0
        )
        self.softmax_rescale_threshold = (
            4.0 if self.fp8_max_offset and self.q_dtype.width == 8 else 0.0
        )
        self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.q_dtype.width

        if cutlass.const_expr(self.pack_gqa and cum_seqlen_q is not None):
            grid_m_total = h_r * s_q
            if cutlass.const_expr(self.flatten_varlen_grid and not self.is_split_kv):
                tokens_per_cluster = self.qk_mma_tiler[0] // h_r
                grid_m_total += h_r * (b - 1) * (tokens_per_cluster - 1)
                if cutlass.const_expr(self.is_persistent):
                    # Persistent validity is `linear_idx < num_blocks` where
                    # num_blocks = ceil(grid_m_total / 128) is the M-CTA count.
                    # A 2-CTA cluster needs BOTH its CTAs valid or both invalid,
                    # else the pair diverges and the 2-CTA MMA deadlocks. That
                    # holds iff num_blocks is even, so round the flattened M
                    # extent up to a multiple of 256 (= 2 CTAs). The extra
                    # padding cluster maps out of range and is rejected by
                    # check_valid_work_for_seqlen_q, exactly like the incumbent's
                    # ragged tail; the non-persistent grid is left untouched.
                    grid_m_total = ((grid_m_total + 255) // 256) * 256
                grid_o_shape = (grid_m_total, o.shape[1], (o.shape[2][0], Int64(1)))
            else:
                # The rectangular incumbent launches this total-Q upper bound
                # independently for every request and rejects invalid tiles.
                grid_o_shape = (grid_m_total, o.shape[1], o.shape[2])
        elif cutlass.const_expr(cum_seqlen_q is not None):
            grid_o_shape = (s_q, o.shape[1], o.shape[2])
        else:
            grid_o_shape = o.shape
        if cutlass.const_expr(self.use_clc_scheduler):
            self.tile_sched_params, grid = compute_grid_clc(
                grid_o_shape,
                self.cta_tiler,
                (*self.cluster_shape_mn, 1),
            )
        else:
            self.tile_sched_params, grid = compute_grid(
                grid_o_shape,
                self.cta_tiler,
                self.is_persistent,
            )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        # check type consistency
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        # the intermediate tensor p is from tmem & k-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.pv_block_tiler[:2]

        if cutlass.const_expr(self.coalesced_epilog):
            # Swizzled smem staging for one (128 x 128) O epilogue sub-tile
            # (one PV d-slice). Same builder the general SM100 kernel uses for
            # its O epilogue staging.
            self.o_epi_smem_layout = sm100_utils.make_smem_layout_epi(
                self.o_dtype, self.o_layout, self.epi_tile, 1
            )
            # 128-bit coalesced smem->gmem copy over the 128-thread correction
            # group: threads_per_row covers one staged row (128 o elements).
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            num_epi_threads = self.threads_per_warp * len(self.correction_warp_ids)
            tO_shape_dim_1 = self.epi_tile[1] // async_copy_elems
            assert num_epi_threads % tO_shape_dim_1 == 0
            tO_layout = cute.make_ordered_layout(
                (num_epi_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            assert self.epi_tile[0] % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
                atom_universal_copy, tO_layout, vO_layout
            )
        else:
            self.o_epi_smem_layout = None
            self.gmem_tiled_copy_O = None

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.qk_acc_stage,
        )
        p_tmem_layout = cute.select(p_tmem_layout_staged, mode=[0, 1, 2])
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.kv_stage,
        )
        # TMA load for Q
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        q_page_smem_layout = None
        if cutlass.const_expr(self.pack_gqa):
            # pack_gqa folds the GQA group's heads into the M dimension, so the Q
            # M-mode is the nested dynamic layout ((h_r, s)). The MMA-projection
            # path (make_tiled_tma_atom_A) cannot statically tile a nested dynamic
            # M-mode, so build the per-CTA Q box with the plain TMA builder -- the
            # same mechanism the paged64 K/V per-page loads use. Each CTA's stage
            # box is rows_per_cta packed rows x d_slice_q, regrouped as
            # ((h_r, rows_per_cta/h_r), d_slice_q) so packed row r' = head +
            # h_r*tok maps to the K-major smem offset r' * d_slice_q.
            rows_per_cta = self.qk_mma_tiler[0] // self.cluster_shape_mn[0]
            d_slice_q = self.qk_mma_tiler[2]
            h_r = self.qhead_per_kvhead
            tok_per_cta = rows_per_cta // h_r
            q_outer = q_smem_layout_staged.outer
            assert cute.size(q_outer.shape[0]) * cute.size(q_outer.shape[1]) * cute.size(
                q_outer.shape[2]
            ) == rows_per_cta * d_slice_q
            q_page_outer = cute.make_layout(
                ((h_r, tok_per_cta), d_slice_q, q_outer.shape[3]),
                stride=((d_slice_q, h_r * d_slice_q), 1, q_outer.stride[3]),
            )
            q_page_smem_layout = q_page_outer
            q_page_smem_staged = cute.make_composed_layout(
                q_smem_layout_staged.inner, 0, q_page_outer
            )
            tma_atom_q, tma_tensor_q = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op, q, q_page_smem_staged, ((h_r, tok_per_cta), d_slice_q)
            )
        else:
            tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                q,
                q_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )

        # TMA load for K / V. Sub-tile paged KV (page_size != tile_n) either uses
        # per-page TMA copies (use_tma_page_loads) or cp.async page assembly.
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        k_page_smem_layout = None
        v_page_smem_layout = None
        if cutlass.const_expr(self.paged_kv_non_tma):
            if cutlass.const_expr(self.use_tma_page_loads):
                # Per-page TMA loads. Each 64-token page fills exactly one CTA's
                # K-stage box (64 tokens x 128 d-slice); each V stage (128 tokens
                # x 64 d per CTA) is assembled from two page halves (64 tokens x
                # 64 d each). Regroup the MMA-B smem layouts into
                # (page-box, rest, stage) forms so make_tiled_tma_atom can derive
                # the swizzled per-page boxes.
                page_size = self.paged_page_size
                d_slice_k = self.qk_mma_tiler[2]
                d_half_v = self.qk_mma_tiler[1] // self.cluster_shape_mn[0]
                k_outer = k_smem_layout_staged.outer
                # K stage address function is 128*tok + d (K-major rows); flatten
                # the atom/rest grouping into a plain (page_size, d_slice) box so
                # the top-level shape matches the TMA CTA V-map.
                assert cute.size(k_outer.shape[0]) * cute.size(k_outer.shape[1]) * cute.size(
                    k_outer.shape[2]
                ) == page_size * d_slice_k
                k_page_outer = cute.make_layout(
                    (page_size, d_slice_k, k_outer.shape[3]),
                    stride=(d_slice_k, 1, k_outer.stride[3]),
                )
                k_page_smem_layout = k_page_outer
                k_page_smem_staged = cute.make_composed_layout(
                    k_smem_layout_staged.inner, 0, k_page_outer
                )
                v_outer = v_smem_layout_staged.outer
                # V stage address function is 64*tok + d (token-major rows of the
                # CTA's d half). Each page fills a 64-token half of the stage.
                assert cute.size(v_outer.shape[0]) * cute.size(v_outer.shape[1]) * cute.size(
                    v_outer.shape[2]
                ) == 2 * page_size * d_half_v
                assert v_outer.shape[2] % 2 == 0
                half_stride = (v_outer.shape[2] // 2) * v_outer.stride[2]
                v_page_outer = cute.make_layout(
                    (d_half_v, page_size, (2, v_outer.shape[3])),
                    stride=(1, d_half_v, (half_stride, v_outer.stride[3])),
                )
                v_page_smem_layout = v_page_outer
                v_page_smem_staged = cute.make_composed_layout(
                    v_smem_layout_staged.inner, 0, v_page_outer
                )
                tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
                    tma_load_op, k, k_page_smem_staged, (page_size, d_slice_k)
                )
                tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
                    tma_load_op, v, v_page_smem_staged, (d_half_v, page_size)
                )
            else:
                tma_atom_k = None
                tma_tensor_k = k
                tma_atom_v = None
                tma_tensor_v = v
        else:
            tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                k,
                k_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            # TMA load for V
            tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                v,
                v_smem_layout,
                self.pv_mma_tiler,
                pv_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_q_bytes = q_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_kv_bytes = k_copy_size * cute.size(qk_tiled_mma.thr_id.shape)

        @cute.struct
        class SharedStorage:
            # TMA G2S load barriers: LOAD warp (producer) -> MMA warp (consumer)
            load_q_mbar_ptr: cute.struct.MemRange[
                Int64, self.q_stage * 2
            ]  # load_q_{producer,consumer}
            load_kv_mbar_ptr: cute.struct.MemRange[
                Int64, self.kv_stage * 2
            ]  # load_kv_{producer,consumer}
            load_v_mbar_ptr: cute.struct.MemRange[
                Int64, self.kv_stage * 2
            ]  # load_v_{producer,consumer}
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            # Softmax -> Correction signaling barriers (row_max/row_sum vec ready)
            s_corr_mbar_ptr: cute.struct.MemRange[
                Int64, self.qk_acc_stage * 2
            ]  # s_corr_{producer,consumer}
            sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
            # MMA -> Correction ownership barriers for O_partial tokens (online rescale/finalize)
            mma_corr_mbar_ptr: cute.struct.MemRange[
                Int64, self.mma_corr_stage * 2
            ]  # mma_corr_{producer,consumer}
            # A CTA-wide "TMEM lifetime" barrier used to safely deallocate TMEM after all users finish.
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # CLC pipeline barriers and response buffer
            clc_mbar_ptr: cute.struct.MemRange[Int64, 2]
            clc_response: cute.struct.MemRange[Int32, 4]
            # Persistent flattened-grid request-mapping table (remap_hoist):
            # [0..15] segment request ids (LPT-sorted when request_count<=4),
            # [16..32] cluster prefix boundaries p[0..request_count] (p[0]=0),
            # [33] request_count. Built once per CTA by the kernel prologue;
            # read by the per-tile _persist_remap_coord_hoist lookups (linear
            # scan, or a bounded 4-step binary search when remap_bsearch).
            # Kept unconditional because the cute.struct builder only sees
            # top-level fields (144B; the flatten gate caps batch at 16).
            req_map: cute.struct.MemRange[Int32, 36]
            # O epilogue staging (one 128x128 o_dtype sub-tile), 1KB-aligned.
            # Only touched when coalesced_epilog is on; kept unconditional
            # because the cute.struct builder only sees top-level fields.
            sO_epi: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, self.epi_tile[0] * self.epi_tile[1]],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        grid = cute.round_up(grid, self.cluster_shape_mnk)
        if cutlass.const_expr(self.is_split_kv):
            # grid.z enumerates (batch, split): batch is the fastest submode.
            grid = (grid[0], grid[1], grid[2] * num_splits)
        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            o,
            cum_seqlen_q,
            cum_seqlen_k,
            seqused_k,
            lse,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            page_table,
            max_seqlen_k_paged,
            window_size_left,
            window_size_right,
            num_splits,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            q_page_smem_layout,
            k_smem_layout_staged,
            p_tmem_layout,
            v_smem_layout_staged,
            k_page_smem_layout,
            v_page_smem_layout,
            self.tile_sched_params,
            self.o_epi_smem_layout,
            self.gmem_tiled_copy_O,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _persist_remap_coord(
        self,
        linear_idx_in,
        cum_seqlen_q: cute.Tensor,
        seqused_k: Optional[cute.Tensor],
        num_m_ctas: Int32,
    ):
        """Persistent flattened-varlen work-tile remap.

        The persistent static scheduler hands each warp loop a *linear* M-CTA
        index (blockIdx.x + step*grid.x). This maps that index to the same
        (mapped_m_cta, mapped_batch) the non-persistent thread-0 prologue would
        produce, so every output tile is written exactly once with identical
        values (any bijection over work tiles is output-equivalent). Computed
        per warp (warp-uniform), with no thread-0 election and no barrier(4).

        persist_mode 2/3 additionally fold the persistent cluster assignment with
        a serpentine permutation so each cluster samples a balanced mix of light
        and heavy causal tiles (a plain stride is ~15% imbalanced on a monotone
        causal profile). The fold is a bijection on [0, num_clusters).
        """
        linear_idx = Int32(linear_idx_in)
        cta_parity = linear_idx % 2
        cl = linear_idx // 2
        if cutlass.const_expr(self.persist_mode in (2, 3)):
            g_cl = cute.arch.grid_dim()[0] // 2
            n_cl = (num_m_ctas + 1) // 2
            band = cl // g_cl
            pos = cl - band * g_cl
            band_start = band * g_cl
            band_count = g_cl
            rem = n_cl - band_start
            if rem < band_count:
                band_count = rem
            if band % 2 == 1:
                pos = band_count - 1 - pos
            cl = band_start + pos
        # ---- request mapping: faithful copy of the non-persistent prologue ----
        local_cl = cl
        mapped_batch = Int32(0)
        request_prefix = Int32(0)
        for request_idx in cutlass.range(cum_seqlen_q.shape[0] - 2, unroll=1):
            request_prefix += cute.ceil_div(
                (cum_seqlen_q[request_idx + 1] - cum_seqlen_q[request_idx])
                * self.qhead_per_kvhead,
                self.qk_mma_tiler[0],
            )
            if cl >= request_prefix:
                mapped_batch = request_idx + 1
                local_cl = cl - request_prefix
        request_count = cum_seqlen_q.shape[0] - 1
        request_clusters = Int32(0)
        if request_count <= 4:
            request_0 = Int32(0)
            request_1 = Int32(1)
            request_2 = Int32(2)
            request_3 = Int32(3)
            if request_count == 1:
                # Single request (batch-1 flatten): no LPT sort. The sort would
                # read seqused_k[1..3] out of bounds; request_0 == 0 already.
                pass
            elif request_count == 2:
                if seqused_k[request_1] > seqused_k[request_0]:
                    request_0, request_1 = request_1, request_0
            elif request_count == 3:
                if seqused_k[request_1] > seqused_k[request_0]:
                    request_0, request_1 = request_1, request_0
                if seqused_k[request_2] > seqused_k[request_1]:
                    request_1, request_2 = request_2, request_1
                if seqused_k[request_1] > seqused_k[request_0]:
                    request_0, request_1 = request_1, request_0
            else:
                if seqused_k[request_1] > seqused_k[request_0]:
                    request_0, request_1 = request_1, request_0
                if seqused_k[request_3] > seqused_k[request_2]:
                    request_2, request_3 = request_3, request_2
                if seqused_k[request_2] > seqused_k[request_0]:
                    request_0, request_2 = request_2, request_0
                if seqused_k[request_3] > seqused_k[request_1]:
                    request_1, request_3 = request_3, request_1
                if seqused_k[request_2] > seqused_k[request_1]:
                    request_1, request_2 = request_2, request_1
            mapped_batch = request_0
            local_cl = cl
            request_clusters = cute.ceil_div(
                (cum_seqlen_q[request_0 + 1] - cum_seqlen_q[request_0])
                * self.qhead_per_kvhead,
                self.qk_mma_tiler[0],
            )
            request_prefix = request_clusters
            if request_count > 1 and cl >= request_prefix:
                mapped_batch = request_1
                local_cl = cl - request_prefix
                request_clusters = cute.ceil_div(
                    (cum_seqlen_q[request_1 + 1] - cum_seqlen_q[request_1])
                    * self.qhead_per_kvhead,
                    self.qk_mma_tiler[0],
                )
            if request_count > 2:
                request_prefix += cute.ceil_div(
                    (cum_seqlen_q[request_1 + 1] - cum_seqlen_q[request_1])
                    * self.qhead_per_kvhead,
                    self.qk_mma_tiler[0],
                )
                if cl >= request_prefix:
                    mapped_batch = request_2
                    local_cl = cl - request_prefix
                    request_clusters = cute.ceil_div(
                        (cum_seqlen_q[request_2 + 1] - cum_seqlen_q[request_2])
                        * self.qhead_per_kvhead,
                        self.qk_mma_tiler[0],
                    )
            if request_count > 3:
                request_prefix += cute.ceil_div(
                    (cum_seqlen_q[request_2 + 1] - cum_seqlen_q[request_2])
                    * self.qhead_per_kvhead,
                    self.qk_mma_tiler[0],
                )
                if cl >= request_prefix:
                    mapped_batch = request_3
                    local_cl = cl - request_prefix
                    request_clusters = cute.ceil_div(
                        (cum_seqlen_q[request_3 + 1] - cum_seqlen_q[request_3])
                        * self.qhead_per_kvhead,
                        self.qk_mma_tiler[0],
                    )
        else:
            request_clusters = cute.ceil_div(
                (cum_seqlen_q[mapped_batch + 1] - cum_seqlen_q[mapped_batch])
                * self.qhead_per_kvhead,
                self.qk_mma_tiler[0],
            )
        # Within-request m_raster interleave: kept for persist_mode 1/3 (the
        # incumbent locality permutation) and mode 4 (CLC hands tiles out in
        # ascending logical order, so the fold's light->heavy->light profile
        # keeps the greedy tail light); skipped for mode 2 so the serpentine
        # fold alone controls balance on the monotone causal profile.
        if cutlass.const_expr(self.persist_mode in (1, 3, 4)):
            if local_cl < request_clusters:
                if cutlass.const_expr(self.m_raster == 1):
                    local_cl = request_clusters - 1 - local_cl
                elif cutlass.const_expr(self.m_raster == 2):
                    request_half = (request_clusters + 1) // 2
                    if local_cl < request_half:
                        local_cl = 2 * local_cl
                    else:
                        local_cl = 2 * (request_clusters - 1 - local_cl) + 1
        mapped_m = 2 * local_cl + cta_parity
        return (mapped_m, mapped_batch)

    @cute.jit
    def _persist_remap_coord_hoist(
        self,
        linear_idx_in,
        req_map: cute.Tensor,
        num_m_ctas: Int32,
    ):
        """Table-based twin of _persist_remap_coord (remap_hoist=True).

        Reads the request-mapping table the CTA prologue built once into smem
        (segment request ids + cluster prefix boundaries). The scan below is
        the same segment search the per-tile version derives from global
        loads: for request_count<=4 the table holds the LPT-sorted order and
        its prefix sums (identical to the sorted-network branch), for
        request_count>4 the plain ascending prefix sums (identical to the
        serial-scan branch). The serpentine fold (modes 2/3) and the
        within-request m_raster fold (modes 1/3/4) are copied verbatim, so
        the returned (mapped_m_cta, mapped_batch) is identical for every
        linear index - a pure latency hoist, bit-identical outputs.

        With remap_bsearch the segment search for request_count>4 is a fixed
        4-step binary search over the same non-decreasing prefix table instead
        of a request_count-1 iteration linear walk (up to 15 at the batch-16
        flatten cap); both searches return the largest j with p[j] <= cl, so
        the mapping is unchanged (bit-identical), only the exposed per-tile
        smem-search latency differs.
        """
        linear_idx = Int32(linear_idx_in)
        cta_parity = linear_idx % 2
        cl = linear_idx // 2
        if cutlass.const_expr(self.persist_mode in (2, 3)):
            g_cl = cute.arch.grid_dim()[0] // 2
            n_cl = (num_m_ctas + 1) // 2
            band = cl // g_cl
            pos = cl - band * g_cl
            band_start = band * g_cl
            band_count = g_cl
            rem = n_cl - band_start
            if rem < band_count:
                band_count = rem
            if band % 2 == 1:
                pos = band_count - 1 - pos
            cl = band_start + pos
        # ---- segment search over the hoisted prefix-boundary table ----
        request_count = req_map[33]
        seg = Int32(0)
        if cutlass.const_expr(self.remap_bsearch):
            # Bounded 4-step binary search (FA_HD256_REMAP_BSEARCH). The
            # flatten gate caps request_count at 16 and the prefix table
            # p[0..request_count] (slots 16..32) is non-decreasing with
            # p[0] == 0 <= cl, so the invariant "p[lo] <= cl < p[hi]" (hi
            # exclusive, lo < hi) converges within 4 halvings for ANY
            # request_count in [1, 16]. A step with hi == lo + 1 is a no-op
            # (mid == lo and p[lo] <= cl keeps lo), so the fixed trip count
            # needs no early-exit predicate and no dynamic request_count
            # branch. The result is the same "largest j with p[j] <= cl" the
            # linear scan yields, including duplicate boundaries (a request
            # that contributes zero clusters). Selected host-side for
            # request_count>4 shapes only (see FA_HD256_REMAP_BSEARCH "auto"),
            # where 4 dependent steps beat up to 15 linear iterations; the
            # <=4 compile keeps the linear scan verbatim (byte-identical to
            # the episode-114 hoist).
            lo = Int32(0)
            hi = request_count
            for _bst in cutlass.range(4, unroll=1):
                mid = (lo + hi) // 2
                if cl >= req_map[16 + mid]:
                    lo = mid
                else:
                    hi = mid
            seg = lo
        else:
            for seg_idx in cutlass.range(request_count - 1, unroll=1):
                seg_end = req_map[17 + seg_idx]
                if cl >= seg_end:
                    seg = seg_idx + 1
        # local_cl == cl - p[seg]: the linear scan's last seg_end update and the
        # p[0] == 0 initial value are the same quantity, so hoisting the
        # subtraction out of the search is value-identical for both searches.
        local_cl = cl - req_map[16 + seg]
        mapped_batch = req_map[seg]
        request_clusters = req_map[17 + seg] - req_map[16 + seg]
        if cutlass.const_expr(self.persist_mode in (1, 3, 4)):
            if local_cl < request_clusters:
                if cutlass.const_expr(self.m_raster == 1):
                    local_cl = request_clusters - 1 - local_cl
                elif cutlass.const_expr(self.m_raster == 2):
                    request_half = (request_clusters + 1) // 2
                    if local_cl < request_half:
                        local_cl = 2 * local_cl
                    else:
                        local_cl = 2 * (request_clusters - 1 - local_cl) + 1
        mapped_m = 2 * local_cl + cta_parity
        return (mapped_m, mapped_batch)

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        mO_qdl: cute.Tensor,
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        seqused_k: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        mPageTable: Optional[cute.Tensor],
        max_seqlen_k: Optional[Int32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        num_splits: Int32,
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        q_page_smem_layout: Optional[cute.Layout],
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        k_page_smem_layout: Optional[cute.Layout],
        v_page_smem_layout: Optional[cute.Layout],
        tile_sched_params: FmhaStaticTileSchedulerParams | FmhaClcDynamicTileSchedulerParams,
        o_epi_smem_layout: Optional[cute.ComposedLayout],
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        #
        # Prefetch tma desc
        #
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            if cutlass.const_expr(not self.paged_kv_non_tma or self.use_tma_page_loads):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
                if cutlass.const_expr(not self.split_kv_loads):
                    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
        if cutlass.const_expr(self.split_kv_loads):
            if warp_idx == self.v_load_warp_id:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Split-KV decode: grid.z enumerates (batch, split) with batch fastest.
        # The tile scheduler below only sees the rebased batch coordinate; the
        # split index selects this cluster's KV sub-range.
        if cutlass.const_expr(self.is_split_kv):
            num_batch_rt = mPageTable.shape[0]
            blk_z = cute.arch.block_idx()[2]
            split_idx = blk_z // num_batch_rt
            batch_idx_grid = blk_z % num_batch_rt
        else:
            split_idx = Int32(0)
            batch_idx_grid = cute.arch.block_idx()[2]

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        if cutlass.const_expr(self.paged_kv_non_tma and not self.use_tma_page_loads):
            # cp.async page-assembly. PipelineAsyncUmma for a 2-CTA kernel expects
            # every producer thread of the pair to arrive on the leading CTA's full
            # barrier (cp.async mbarrier arrives are CTA-local, so completion is
            # waited explicitly before a plain remote arrive).
            load_kv_pipeline = pipeline.PipelineAsyncUmma.create(
                num_stages=self.kv_stage,
                producer_group=make_thread_cooperative_group(
                    self.threads_per_warp * self.cluster_shape_mn[0]
                ),
                consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
                barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            # Dense TMA path and per-page TMA paged64 path: both CTAs issue their
            # own TMA copies, each contributing its bytes to the leading CTA's
            # transaction barrier (tx_count covers the CTA pair).
            load_kv_pipeline = pipeline.PipelineTmaUmma.create(
                num_stages=self.kv_stage,
                producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
                consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
                tx_count=self.tma_copy_kv_bytes,
                barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        load_kv_producer, load_kv_consumer = load_kv_pipeline.make_participants()
        if cutlass.const_expr(self.split_kv_loads):
            load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=self.kv_stage,
                producer_group=make_thread_cooperative_group(len([self.v_load_warp_id])),
                consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
                tx_count=self.tma_copy_kv_bytes,
                barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ).make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.sum_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.correction_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        tmem.allocate(self.tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
        # Initialize CLC state if using dynamic scheduler
        if cutlass.const_expr(self.use_clc_scheduler):
            clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            cluster_size = cute.size(self.cluster_shape_mnk)
            num_clc_consumer_threads = self.threads_per_warp * (
                1  # sched_warp (CTA 0 only)
                + cluster_size
                * (
                    len(self.softmax_warp_ids)
                    + len(self.correction_warp_ids)
                    + 1  # mma_warp
                    + 1  # load_warp
                    # split_kv_loads puts the independent V producer warp in the
                    # LOAD consumer loop as well; it consumes CLC responses too,
                    # so the empty-barrier arrival count must include it.
                    + (1 if self.split_kv_loads else 0)  # v_load_warp
                )
            )
            clc_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_clc_consumer_threads
            )
            clc_response_ptr = storage.clc_response.data_ptr()
            clc = ClcState.create(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    # Use the region-isolated kernel argument, NOT
                    # self.tile_sched_params: the host-traced attribute holds
                    # SSA values from outside the kernel region and fails MLIR
                    # region isolation ('cute.make_int_tuple' op using value
                    # defined outside the region).
                    tile_sched_params.clc_hw_params(),
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                    num_stages=self.num_clc_stage,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=self.num_clc_response_bytes,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                ),
                consumer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_clc_stage
                ),
                producer_state=pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_clc_stage
                ),
            )
        else:
            clc = None
            clc_response_ptr = None

        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        # K and V now use separate memory since we removed the transform stage
        sV = smem.allocate_tensor(
            element_type=self.v_dtype,
            layout=v_smem_layout_staged.outer,
            swizzle=v_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sSum = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
            byte_alignment=128,
        )
        if cutlass.const_expr(self.coalesced_epilog):
            # make_smem_layout_epi is staged (m, n, stage); take the single stage.
            sO_epi = storage.sO_epi.get_tensor(
                o_epi_smem_layout.outer, swizzle=o_epi_smem_layout.inner
            )[None, None, 0]
        else:
            sO_epi = None
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = qk_thr_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        pv_acc_shape = pv_thr_mma.partition_shape_C((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                self.iterations_pv,
                stride=self.pv_mma_tiler[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO_staged = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        for _i in cutlass.range_constexpr(len(self.empty_warp_id)):
            if warp_idx == self.empty_warp_id[_i]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        # Persistent flattened-grid request-mapping table (remap_hoist): one
        # elected thread per CTA derives the segment layout (LPT-sorted ids +
        # cluster prefix boundaries for request_count<=4, ascending prefix sums
        # above) that _persist_remap_coord would otherwise re-derive from
        # cum_seqlen_q/seqused_k GLOBAL loads on every tile in every warp
        # section. Faithful hoist: same values, same <=4 sorting network and
        # per-count seqused_k guards (count==1 must not read seqused_k[1..3]),
        # so the per-tile mapping is unchanged. One-time cost: ~2*batch L2
        # loads + one CTA barrier(4) (id 4 is free in persistent mode - the
        # non-persistent flatten prologue that uses it is const_expr-excluded),
        # amortized over K waves x 4 warp sections. Runs at CTA scope before
        # the warp branches and before any reg alloc/dealloc-dependent section,
        # mirroring the non-persistent prologue's placement.
        req_map = None
        if cutlass.const_expr(
            self.is_persistent and self.remap_hoist and cum_seqlen_q is not None
        ):
            req_map = storage.req_map.get_tensor(cute.make_layout(36))
            tidx, _, _ = cute.arch.thread_idx()
            if tidx == 0:
                request_count = cum_seqlen_q.shape[0] - 1
                req_map[33] = request_count
                req_map[16] = Int32(0)
                if request_count <= 4:
                    request_0 = Int32(0)
                    request_1 = Int32(1)
                    request_2 = Int32(2)
                    request_3 = Int32(3)
                    if request_count == 1:
                        # Single request (batch-1 flatten): no LPT sort. The
                        # sort would read seqused_k[1..3] out of bounds;
                        # request_0 == 0 already.
                        pass
                    elif request_count == 2:
                        if seqused_k[request_1] > seqused_k[request_0]:
                            request_0, request_1 = request_1, request_0
                    elif request_count == 3:
                        if seqused_k[request_1] > seqused_k[request_0]:
                            request_0, request_1 = request_1, request_0
                        if seqused_k[request_2] > seqused_k[request_1]:
                            request_1, request_2 = request_2, request_1
                        if seqused_k[request_1] > seqused_k[request_0]:
                            request_0, request_1 = request_1, request_0
                    else:
                        if seqused_k[request_1] > seqused_k[request_0]:
                            request_0, request_1 = request_1, request_0
                        if seqused_k[request_3] > seqused_k[request_2]:
                            request_2, request_3 = request_3, request_2
                        if seqused_k[request_2] > seqused_k[request_0]:
                            request_0, request_2 = request_2, request_0
                        if seqused_k[request_3] > seqused_k[request_1]:
                            request_1, request_3 = request_3, request_1
                        if seqused_k[request_2] > seqused_k[request_1]:
                            request_1, request_2 = request_2, request_1
                    req_map[0] = request_0
                    seg_prefix = cute.ceil_div(
                        (cum_seqlen_q[request_0 + 1] - cum_seqlen_q[request_0])
                        * self.qhead_per_kvhead,
                        self.qk_mma_tiler[0],
                    )
                    req_map[17] = seg_prefix
                    if request_count > 1:
                        req_map[1] = request_1
                        seg_prefix += cute.ceil_div(
                            (cum_seqlen_q[request_1 + 1] - cum_seqlen_q[request_1])
                            * self.qhead_per_kvhead,
                            self.qk_mma_tiler[0],
                        )
                        req_map[18] = seg_prefix
                    if request_count > 2:
                        req_map[2] = request_2
                        seg_prefix += cute.ceil_div(
                            (cum_seqlen_q[request_2 + 1] - cum_seqlen_q[request_2])
                            * self.qhead_per_kvhead,
                            self.qk_mma_tiler[0],
                        )
                        req_map[19] = seg_prefix
                    if request_count > 3:
                        req_map[3] = request_3
                        seg_prefix += cute.ceil_div(
                            (cum_seqlen_q[request_3 + 1] - cum_seqlen_q[request_3])
                            * self.qhead_per_kvhead,
                            self.qk_mma_tiler[0],
                        )
                        req_map[20] = seg_prefix
                else:
                    seg_prefix = Int32(0)
                    for request_idx in cutlass.range(request_count, unroll=1):
                        req_map[request_idx] = request_idx
                        seg_prefix += cute.ceil_div(
                            (cum_seqlen_q[request_idx + 1] - cum_seqlen_q[request_idx])
                            * self.qhead_per_kvhead,
                            self.qk_mma_tiler[0],
                        )
                        req_map[17 + request_idx] = seg_prefix
            cute.arch.barrier(barrier_id=4, number_of_threads=self.threads_per_cta)

        if cutlass.const_expr(self.use_clc_scheduler):
            tile_sched = FmhaClcDynamicTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
                clc,
            )
        else:
            blk_idx = cute.arch.block_idx()
            m_blk_x = blk_idx[0]
            if cutlass.const_expr(
                not self.is_persistent
                and self.flatten_varlen_grid
                and self.pack_gqa
                and cum_seqlen_q is not None
                and not self.is_split_kv
            ):
                scheduler_coord = storage.clc_response.get_tensor(cute.make_layout(2))
                tidx, _, _ = cute.arch.thread_idx()
                if tidx == 0:
                    cta_parity = m_blk_x % 2
                    launch_cl = m_blk_x // 2
                    cl = launch_cl
                    local_cl = cl
                    mapped_batch = Int32(0)
                    request_prefix = Int32(0)
                    for request_idx in cutlass.range(cum_seqlen_q.shape[0] - 2, unroll=1):
                        request_prefix += cute.ceil_div(
                            (cum_seqlen_q[request_idx + 1] - cum_seqlen_q[request_idx])
                            * self.qhead_per_kvhead,
                            self.qk_mma_tiler[0],
                        )
                        if cl >= request_prefix:
                            mapped_batch = request_idx + 1
                            local_cl = cl - request_prefix
                    request_count = cum_seqlen_q.shape[0] - 1
                    request_clusters = Int32(0)
                    if request_count <= 4:
                        request_0 = Int32(0)
                        request_1 = Int32(1)
                        request_2 = Int32(2)
                        request_3 = Int32(3)
                        if request_count == 1:
                            # Single request (batch-1 flatten): no LPT sort. The
                            # sort would read seqused_k[1..3] out of bounds;
                            # request_0 == 0 already.
                            pass
                        elif request_count == 2:
                            if seqused_k[request_1] > seqused_k[request_0]:
                                request_0, request_1 = request_1, request_0
                        elif request_count == 3:
                            if seqused_k[request_1] > seqused_k[request_0]:
                                request_0, request_1 = request_1, request_0
                            if seqused_k[request_2] > seqused_k[request_1]:
                                request_1, request_2 = request_2, request_1
                            if seqused_k[request_1] > seqused_k[request_0]:
                                request_0, request_1 = request_1, request_0
                        else:
                            if seqused_k[request_1] > seqused_k[request_0]:
                                request_0, request_1 = request_1, request_0
                            if seqused_k[request_3] > seqused_k[request_2]:
                                request_2, request_3 = request_3, request_2
                            if seqused_k[request_2] > seqused_k[request_0]:
                                request_0, request_2 = request_2, request_0
                            if seqused_k[request_3] > seqused_k[request_1]:
                                request_1, request_3 = request_3, request_1
                            if seqused_k[request_2] > seqused_k[request_1]:
                                request_1, request_2 = request_2, request_1
                        mapped_batch = request_0
                        local_cl = cl
                        request_clusters = cute.ceil_div(
                            (cum_seqlen_q[request_0 + 1] - cum_seqlen_q[request_0])
                            * self.qhead_per_kvhead,
                            self.qk_mma_tiler[0],
                        )
                        request_prefix = request_clusters
                        if request_count > 1 and cl >= request_prefix:
                            mapped_batch = request_1
                            local_cl = cl - request_prefix
                            request_clusters = cute.ceil_div(
                                (cum_seqlen_q[request_1 + 1] - cum_seqlen_q[request_1])
                                * self.qhead_per_kvhead,
                                self.qk_mma_tiler[0],
                            )
                        if request_count > 2:
                            request_prefix += cute.ceil_div(
                                (cum_seqlen_q[request_1 + 1] - cum_seqlen_q[request_1])
                                * self.qhead_per_kvhead,
                                self.qk_mma_tiler[0],
                            )
                            if cl >= request_prefix:
                                mapped_batch = request_2
                                local_cl = cl - request_prefix
                                request_clusters = cute.ceil_div(
                                    (cum_seqlen_q[request_2 + 1] - cum_seqlen_q[request_2])
                                    * self.qhead_per_kvhead,
                                    self.qk_mma_tiler[0],
                                )
                        if request_count > 3:
                            request_prefix += cute.ceil_div(
                                (cum_seqlen_q[request_2 + 1] - cum_seqlen_q[request_2])
                                * self.qhead_per_kvhead,
                                self.qk_mma_tiler[0],
                            )
                            if cl >= request_prefix:
                                mapped_batch = request_3
                                local_cl = cl - request_prefix
                                request_clusters = cute.ceil_div(
                                    (cum_seqlen_q[request_3 + 1] - cum_seqlen_q[request_3])
                                    * self.qhead_per_kvhead,
                                    self.qk_mma_tiler[0],
                                )
                    else:
                        request_clusters = cute.ceil_div(
                            (cum_seqlen_q[mapped_batch + 1] - cum_seqlen_q[mapped_batch])
                            * self.qhead_per_kvhead,
                            self.qk_mma_tiler[0],
                        )
                    if local_cl < request_clusters:
                        if cutlass.const_expr(self.m_raster == 1):
                            local_cl = request_clusters - 1 - local_cl
                        elif cutlass.const_expr(self.m_raster == 2):
                            request_half = (request_clusters + 1) // 2
                            if local_cl < request_half:
                                local_cl = 2 * local_cl
                            else:
                                local_cl = 2 * (request_clusters - 1 - local_cl) + 1
                    scheduler_coord[0] = 2 * local_cl + cta_parity
                    scheduler_coord[1] = mapped_batch
                cute.arch.barrier(barrier_id=4, number_of_threads=self.threads_per_cta)
                m_blk_x = scheduler_coord[0]
                batch_idx_grid = scheduler_coord[1]
            elif cutlass.const_expr(not self.is_persistent and self.m_raster != 0):
                # Permute the cluster-level M index (blockIdx.x // 2) while
                # keeping the intra-cluster CTA parity, then rebuild blockIdx.x.
                cta_parity = m_blk_x % 2
                cl = m_blk_x // 2
                num_cl = cute.arch.grid_dim()[0] // 2
                if cutlass.const_expr(self.m_raster == 1):
                    cl = num_cl - 1 - cl
                elif cutlass.const_expr(self.m_raster == 2):
                    half = (num_cl + 1) // 2
                    if cl < half:
                        cl = 2 * cl
                    else:
                        cl = 2 * (num_cl - 1 - cl) + 1
                m_blk_x = 2 * cl + cta_parity
            sched_blk_coord = (m_blk_x, blk_idx[1], batch_idx_grid)
            tile_sched = FmhaStaticTileScheduler(
                tile_sched_params, m_blk_x, sched_blk_coord, cute.arch.grid_dim()
            )
        # Persistent flattened grid: total logical M-CTA extent. The per-tile
        # remap in the work loops derives the cluster count from it for the
        # serpentine load-balance fold. Both scheduler param types carry
        # problem_shape_mbh; unused (DCE'd) when not persistent.
        persist_num_m_ctas = Int32(tile_sched_params.problem_shape_mbh[0])
        work_tile = tile_sched.initial_work_tile_info()

        # Cluster wait
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id or (
            cutlass.const_expr(self.split_kv_loads) and warp_idx == self.v_load_warp_id
        ):
            if warp_idx == self.load_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx  # (q_tile_idx, 0, (head_idx, batch_idx))
                if cutlass.const_expr(self.is_persistent and cum_seqlen_q is not None):
                    if cutlass.const_expr(self.remap_hoist):
                        _pm, _pb = self._persist_remap_coord_hoist(
                            curr_block_coord[0], req_map, persist_num_m_ctas
                        )
                    else:
                        _pm, _pb = self._persist_remap_coord(
                            curr_block_coord[0], cum_seqlen_q, seqused_k, persist_num_m_ctas
                        )
                    curr_block_coord = (
                        _pm, curr_block_coord[1], (curr_block_coord[2][0], _pb)
                    )
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                batch_coord = curr_block_coord[2][1]
                seqlen_q = (
                    mQ_qdl.shape[0][1]
                    if cutlass.const_expr(self.pack_gqa)
                    else mQ_qdl.shape[0]
                )
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                if cutlass.const_expr(seqused_k is not None):
                    seqlen_k = seqused_k[batch_coord]
                cuseqlen_q = Int32(0)
                cuseqlen_k = Int32(0)
                if cutlass.const_expr(self.pack_gqa):
                    block_offset = (
                        (Int32(0), Int32(0)),
                        Int32(0),
                        Int32(0),
                        (Int32(0), Int32(0)),
                    )
                else:
                    block_offset = (
                        Int32(0),
                        Int32(0),
                        Int32(0),
                        ((Int32(0), Int32(0)), Int32(0)),
                    )
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    if cutlass.const_expr(self.pack_gqa):
                        block_offset = (
                            (Int32(0), cuseqlen_q),
                            cuseqlen_k,
                            Int32(0),
                            (Int32(0), Int32(0)),
                        )
                    else:
                        block_offset = (
                            cuseqlen_q,
                            cuseqlen_k,
                            Int32(0),
                            ((Int32(0), Int32(0)), Int32(0)),
                        )
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q * self.qhead_per_kvhead
                        if cutlass.const_expr(self.pack_gqa)
                        else seqlen_q,
                    )
                seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                    FusedMask.get_trip_start_count_via_block_info(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                        self.qhead_per_kvhead if self.pack_gqa else 1,
                    )
                )
                seqlen_kv_loop_start, seqlen_kv_loop_steps = self._split_trip_range(
                    seqlen_kv_loop_start,
                    seqlen_kv_loop_steps,
                    seqlen_k,
                    split_idx,
                    num_splits,
                )
                if cutlass.const_expr(self.is_split_kv):
                    # Empty split sub-range: no Q/K/V traffic for this cluster.
                    if seqlen_kv_loop_steps == 0:
                        continue_cond = True
                if not continue_cond:
                    mQ_qdl_ = cute.domain_offset(cute.select(block_offset, mode=[0, 2, 3]), mQ_qdl)
                    # Local tile partition global tensors
                    if cutlass.const_expr(self.pack_gqa):
                        # Packed Q: each CTA loads its own ((h_r, tok_per_cta), d_slice)
                        # box via the plain TMA atom (see __call__). Select this tile's
                        # (kv-head, batch) slice, tile the packed rows into per-CTA
                        # groups, and partition against the regrouped smem box. No
                        # cluster multicast: each CTA loads its own rows.
                        rows_per_cta_q = self.qk_mma_tiler[0] // self.cluster_shape_mn[0]
                        d_slice_q = self.qk_mma_tiler[2]
                        h_r = self.qhead_per_kvhead
                        tok_per_cta = rows_per_cta_q // h_r
                        mQ_hb = mQ_qdl_[None, None, mma_block_coord[2]]
                        gQp = cute.flat_divide(mQ_hb, ((h_r, tok_per_cta), d_slice_q))
                        sQp = cute.make_tensor(sQ.iterator, q_page_smem_layout)
                        # flat_divide keeps each mode's tile/rest nested: modes are
                        # (box_rows, box_d, (1, groups), d_rest); group the two box
                        # modes only.
                        tQsQ, tQgQ_p = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_q,
                            0,
                            cute.make_layout((1,)),
                            cute.group_modes(sQp, 0, 2),
                            cute.group_modes(gQp, 0, 2),
                        )
                        q_cta_group_idx = (
                            mma_block_coord[0] * self.cluster_shape_mn[0] + cta_rank_in_cluster
                        )
                        tQgQ = None
                    else:
                        q_cta_layout = cute.make_layout(
                            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
                        )
                        # (bM, bK, loopM, loopK, loopL)
                        gQ_qdl = cute.flat_divide(
                            mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2])
                        )
                        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_q,
                            block_in_cluster_coord_vmnk[2],
                            q_cta_layout,
                            cute.group_modes(sQ, 0, 3),
                            cute.group_modes(tSgQ_qdl, 0, 3),
                        )
                        tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]
                        tQgQ_p = None
                        q_cta_group_idx = Int32(0)
                    if cutlass.const_expr(self.paged_kv_non_tma):
                        # Sub-tile pages (page_size != tile_n): each CTA's 64-token
                        # half of every 128-token tile is exactly one 64-token page.
                        head_kv_coord = (
                            curr_block_coord[2][0]
                            if self.pack_gqa
                            else curr_block_coord[2][0] // self.qhead_per_kvhead
                        )
                        load_tidx, _, _ = cute.arch.thread_idx()
                        load_tid = load_tidx % self.threads_per_warp
                        mK_h = mK_kdl[None, None, head_kv_coord, None]
                        mV_h = mV_dkl[None, None, head_kv_coord, None]
                        rows_per_cta = self.qk_mma_tiler[1] // self.cluster_shape_mn[0]
                        half_base_off = cta_rank_in_cluster * rows_per_cta
                        if cutlass.const_expr(self.use_tma_page_loads):
                            # Per-page TMA: partition the regrouped (page-box, rest,
                            # stage) smem layouts against the page-tiled gmem. No
                            # cluster multicast: each CTA loads its own pages.
                            page_size = self.paged_page_size
                            d_slice_k = self.qk_mma_tiler[2]
                            d_half_v = self.qk_mma_tiler[1] // self.cluster_shape_mn[0]
                            sKp = cute.make_tensor(
                                sK.iterator, k_page_smem_layout
                            )
                            gKp = cute.flat_divide(mK_h, (page_size, d_slice_k))
                            tKsKp, tKgKp = cute.nvgpu.cpasync.tma_partition(
                                tma_atom_k,
                                0,
                                cute.make_layout((1,)),
                                cute.group_modes(sKp, 0, 2),
                                cute.group_modes(gKp, 0, 2),
                            )
                            sVp = cute.make_tensor(
                                sV.iterator, v_page_smem_layout
                            )
                            gVp = cute.flat_divide(mV_h, (d_half_v, page_size))
                            tVsVp, tVgVp = cute.nvgpu.cpasync.tma_partition(
                                tma_atom_v,
                                0,
                                cute.make_layout((1,)),
                                cute.group_modes(sVp, 0, 2),
                                cute.group_modes(gVp, 0, 2),
                            )
                        tKgK = None
                        tVgV = None
                    else:
                        kv_cta_layout = cute.make_layout(
                            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                        )
                        if cutlass.const_expr(mPageTable is None):
                            # Dense path: domain_offset K/V by batch block, select batch via mma_block_coord[2].
                            mK_kdl_ = cute.domain_offset(
                                cute.select(block_offset, mode=[1, 2, 3]), mK_kdl
                            )
                            mV_dkl_ = cute.domain_offset(
                                cute.select(block_offset, mode=[2, 1, 3]), mV_dkl
                            )
                            gK_kdl = cute.flat_divide(
                                mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                            )
                            tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                            tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                                tma_atom_k,
                                block_in_cluster_coord_vmnk[1],
                                kv_cta_layout,
                                cute.group_modes(sK, 0, 3),
                                cute.group_modes(tSgK_kdl, 0, 3),
                            )
                            gV_dkl = cute.flat_divide(
                                mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
                            )
                            tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                            tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                                tma_atom_v,
                                block_in_cluster_coord_vmnk[1],
                                kv_cta_layout,
                                cute.group_modes(sV, 0, 3),
                                cute.group_modes(tSgV_dkl, 0, 3),
                            )
                            # ((atom_v, rest_v), RestN, RestK)
                            tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
                            tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
                        else:
                            # Paged path: slice K/V by KV head, keep num_pages dim for page_idx-based TMA.
                            head_kv_coord = (
                                curr_block_coord[2][0]
                                if self.pack_gqa
                                else curr_block_coord[2][0] // self.qhead_per_kvhead
                            )
                            mK_kdl_ = mK_kdl[None, None, head_kv_coord, None]
                            mV_dkl_ = mV_dkl[None, None, head_kv_coord, None]
                            gK_kdl = cute.flat_divide(
                                mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                            )
                            tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                            tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                                tma_atom_k,
                                block_in_cluster_coord_vmnk[1],
                                kv_cta_layout,
                                cute.group_modes(sK, 0, 3),
                                cute.group_modes(tSgK_kdl, 0, 3),
                            )
                            gV_dkl = cute.flat_divide(
                                mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
                            )
                            tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                            tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                                tma_atom_v,
                                block_in_cluster_coord_vmnk[1],
                                kv_cta_layout,
                                cute.group_modes(sV, 0, 3),
                                cute.group_modes(tSgV_dkl, 0, 3),
                            )
                            tKgK = tKgK_kdl
                            tVgV = tVgV_dkl

                    seqlen_kv_loop_end = seqlen_kv_loop_start + seqlen_kv_loop_steps
                    # Q
                    if warp_idx == self.load_warp_id:
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            q_handle = load_q_producer.acquire_and_advance()
                            if cutlass.const_expr(self.pack_gqa):
                                cute.copy(
                                    tma_atom_q,
                                    tQgQ_p[None, q_cta_group_idx, iter],
                                    tQsQ[None, q_handle.index],
                                    tma_bar_ptr=q_handle.barrier,
                                )
                            else:
                                cute.copy(
                                    tma_atom_q,
                                    tQgQ[None, iter],
                                    tQsQ[None, q_handle.index],
                                    tma_bar_ptr=q_handle.barrier,
                                )

                    # K0
                    kv_coord = seqlen_kv_loop_start
                    if cutlass.const_expr(self.paged_kv_non_tma and self.use_tma_page_loads):
                        # Per-page TMA loads. K stage = one TMA copy of this CTA's
                        # page (64 tokens x 128 d-slice); V stage = two TMA copies
                        # (page0 -> stage tokens 0-63, page1 -> tokens 64-127), each
                        # 64 tokens x 64 d. Both CTAs issue their own copies; the
                        # 2-CTA TMA op arrives on the leading CTA's transaction
                        # barrier, whose tx_count covers the pair.
                        tile_n = self.qk_mma_tiler[1]
                        if cutlass.const_expr(self.split_kv_loads):
                            if warp_idx == self.load_warp_id:
                                for i in cutlass.range(seqlen_kv_loop_steps, unroll=1):
                                    tma_k_tile = seqlen_kv_loop_start + i
                                    tma_k_half_base = tma_k_tile * tile_n + half_base_off
                                    tma_k_page_idx = 2 * tma_k_tile + (
                                        cta_rank_in_cluster
                                        if tma_k_half_base < seqlen_k
                                        else Int32(0)
                                    )
                                    tma_k_page = mPageTable[batch_coord, tma_k_page_idx]
                                    for iter in cutlass.range_constexpr(self.iterations_qk):
                                        k_handle = load_kv_producer.acquire_and_advance()
                                        cute.copy(
                                            tma_atom_k,
                                            tKgKp[None, 0, iter, tma_k_page],
                                            tKsKp[None, k_handle.index],
                                            tma_bar_ptr=k_handle.barrier,
                                        )
                            if warp_idx == self.v_load_warp_id:
                                for i in cutlass.range(seqlen_kv_loop_steps, unroll=1):
                                    tma_v_tile = seqlen_kv_loop_start + i
                                    tma_vp0_idx = 2 * tma_v_tile
                                    tma_vp1_idx = tma_vp0_idx + (
                                        Int32(1)
                                        if tma_v_tile * tile_n + tile_n // 2 < seqlen_k
                                        else Int32(0)
                                    )
                                    tma_v_page0 = mPageTable[batch_coord, tma_vp0_idx]
                                    tma_v_page1 = mPageTable[batch_coord, tma_vp1_idx]
                                    for iter in cutlass.range_constexpr(self.iterations_pv):
                                        v_handle = load_v_producer.acquire_and_advance()
                                        v_d_rest = (
                                            cta_rank_in_cluster
                                            + iter * self.cluster_shape_mn[0]
                                        )
                                        cute.copy(
                                            tma_atom_v,
                                            tVgVp[None, v_d_rest, 0, tma_v_page0],
                                            tVsVp[None, (0, v_handle.index)],
                                            tma_bar_ptr=v_handle.barrier,
                                        )
                                        cute.copy(
                                            tma_atom_v,
                                            tVgVp[None, v_d_rest, 0, tma_v_page1],
                                            tVsVp[None, (1, v_handle.index)],
                                            tma_bar_ptr=v_handle.barrier,
                                        )
                        else:
                            # Stage order must match MMA consumption: Ki stages,
                            # then V(i-1) stages.
                            tma_k_half_base = kv_coord * tile_n + half_base_off
                            tma_k_page_idx = 2 * kv_coord + (
                                cta_rank_in_cluster
                                if tma_k_half_base < seqlen_k
                                else Int32(0)
                            )
                            tma_k_page = mPageTable[batch_coord, tma_k_page_idx]
                            for iter in cutlass.range_constexpr(self.iterations_qk):
                                k_handle = load_kv_producer.acquire_and_advance()
                                cute.copy(
                                    tma_atom_k,
                                    tKgKp[None, 0, iter, tma_k_page],
                                    tKsKp[None, k_handle.index],
                                    tma_bar_ptr=k_handle.barrier,
                                )
                            kv_coord += 1
                            for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                                # Ki
                                tma_k_half_base = kv_coord * tile_n + half_base_off
                                tma_k_page_idx = 2 * kv_coord + (
                                    cta_rank_in_cluster
                                    if tma_k_half_base < seqlen_k
                                    else Int32(0)
                                )
                                tma_k_page = mPageTable[batch_coord, tma_k_page_idx]
                                for iter in cutlass.range_constexpr(self.iterations_qk):
                                    k_handle = load_kv_producer.acquire_and_advance()
                                    cute.copy(
                                        tma_atom_k,
                                        tKgKp[None, 0, iter, tma_k_page],
                                        tKsKp[None, k_handle.index],
                                        tma_bar_ptr=k_handle.barrier,
                                    )
                                # V(i-1)
                                tma_v_tile = kv_coord - 1
                                tma_vp0_idx = 2 * tma_v_tile
                                tma_vp1_idx = tma_vp0_idx + (
                                    Int32(1)
                                    if tma_v_tile * tile_n + tile_n // 2 < seqlen_k
                                    else Int32(0)
                                )
                                tma_v_page0 = mPageTable[batch_coord, tma_vp0_idx]
                                tma_v_page1 = mPageTable[batch_coord, tma_vp1_idx]
                                for iter in cutlass.range_constexpr(self.iterations_pv):
                                    v_handle = load_kv_producer.acquire_and_advance()
                                    v_d_rest = (
                                        cta_rank_in_cluster
                                        + iter * self.cluster_shape_mn[0]
                                    )
                                    cute.copy(
                                        tma_atom_v,
                                        tVgVp[None, v_d_rest, 0, tma_v_page0],
                                        tVsVp[None, (0, v_handle.index)],
                                        tma_bar_ptr=v_handle.barrier,
                                    )
                                    cute.copy(
                                        tma_atom_v,
                                        tVgVp[None, v_d_rest, 0, tma_v_page1],
                                        tVsVp[None, (1, v_handle.index)],
                                        tma_bar_ptr=v_handle.barrier,
                                    )
                                kv_coord += 1
                            # Vend: V of the last tile.
                            tma_v_tile = kv_coord - 1
                            tma_vp0_idx = 2 * tma_v_tile
                            tma_vp1_idx = tma_vp0_idx + (
                                Int32(1)
                                if tma_v_tile * tile_n + tile_n // 2 < seqlen_k
                                else Int32(0)
                            )
                            tma_v_page0 = mPageTable[batch_coord, tma_vp0_idx]
                            tma_v_page1 = mPageTable[batch_coord, tma_vp1_idx]
                            for iter in cutlass.range_constexpr(self.iterations_pv):
                                v_handle = load_kv_producer.acquire_and_advance()
                                v_d_rest = (
                                    cta_rank_in_cluster
                                    + iter * self.cluster_shape_mn[0]
                                )
                                cute.copy(
                                    tma_atom_v,
                                    tVgVp[None, v_d_rest, 0, tma_v_page0],
                                    tVsVp[None, (0, v_handle.index)],
                                    tma_bar_ptr=v_handle.barrier,
                                )
                                cute.copy(
                                    tma_atom_v,
                                    tVgVp[None, v_d_rest, 0, tma_v_page1],
                                    tVsVp[None, (1, v_handle.index)],
                                    tma_bar_ptr=v_handle.barrier,
                                )
                    if cutlass.const_expr(self.paged_kv_non_tma and not self.use_tma_page_loads):
                        # Sub-tile pages: K stages hold this CTA's 64-token half of
                        # each 128-token tile (one page); V stages hold this CTA's
                        # 64-d half across all 128 tile tokens (both pages). Stage
                        # order must match MMA consumption: Ki stages, then V(i-1)
                        # stages, round-robin over kv_stage buffers.
                        #
                        # Overlap: cp.async groups are issued back to back; each stage
                        # is arrived only after the next group is committed
                        # (wait_group(1) drains everything but the newest group).
                        tile_n = self.qk_mma_tiler[1]
                        d_slice_k = self.qk_mma_tiler[2]
                        d_slice_v = self.pv_mma_tiler[2]

                        # K0 (both d-slice stages).
                        k_half_base = kv_coord * tile_n + half_base_off
                        k_page_idx = 2 * kv_coord + (
                            cta_rank_in_cluster if k_half_base < seqlen_k else Int32(0)
                        )
                        k_page = mPageTable[batch_coord, k_page_idx]
                        k0_handles = [
                            load_kv_producer.acquire_and_advance()
                            for _ in range(self.iterations_qk)
                        ]
                        for iter in cutlass.range_constexpr(self.iterations_qk):
                            sK_cur = sK[None, None, None, k0_handles[iter].index]
                            self.load_paged64_k_stage(
                                mK_h,
                                k_page,
                                iter * d_slice_k,
                                sK_cur,
                                seqlen_k,
                                k_half_base,
                                load_tid,
                            )
                            cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(1)
                        cute.arch.fence_view_async_shared()
                        load_kv_pipeline.sync_object_full.arrive(k0_handles[0].index, 0)
                        prev_handle = k0_handles[1]
                        kv_coord += 1
                        for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                            # Ki
                            k_half_base = kv_coord * tile_n + half_base_off
                            k_page_idx = 2 * kv_coord + (
                                cta_rank_in_cluster if k_half_base < seqlen_k else Int32(0)
                            )
                            k_page = mPageTable[batch_coord, k_page_idx]
                            k_handles = [
                                load_kv_producer.acquire_and_advance()
                                for _ in range(self.iterations_qk)
                            ]
                            sK_cur = sK[None, None, None, k_handles[0].index]
                            self.load_paged64_k_stage(
                                mK_h,
                                k_page,
                                Int32(0),
                                sK_cur,
                                seqlen_k,
                                k_half_base,
                                load_tid,
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(1)
                            cute.arch.fence_view_async_shared()
                            load_kv_pipeline.sync_object_full.arrive(prev_handle.index, 0)
                            prev_handle = k_handles[0]
                            sK_cur = sK[None, None, None, k_handles[1].index]
                            self.load_paged64_k_stage(
                                mK_h,
                                k_page,
                                d_slice_k,
                                sK_cur,
                                seqlen_k,
                                k_half_base,
                                load_tid,
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(1)
                            cute.arch.fence_view_async_shared()
                            load_kv_pipeline.sync_object_full.arrive(prev_handle.index, 0)
                            prev_handle = k_handles[1]
                            # V(i-1)
                            v_tile = kv_coord - 1
                            v_page0_idx = 2 * v_tile
                            v_page1_idx = v_page0_idx + (
                                Int32(1)
                                if v_tile * tile_n + tile_n // 2 < seqlen_k
                                else Int32(0)
                            )
                            v_page0 = mPageTable[batch_coord, v_page0_idx]
                            v_page1 = mPageTable[batch_coord, v_page1_idx]
                            v_base = v_tile * tile_n
                            v_handles = [
                                load_kv_producer.acquire_and_advance()
                                for _ in range(self.iterations_pv)
                            ]
                            sV_cur = sV[None, None, None, v_handles[0].index]
                            self.load_paged64_v_stage(
                                mV_h,
                                v_page0,
                                v_page1,
                                half_base_off,
                                sV_cur,
                                seqlen_k,
                                v_base,
                                load_tid,
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(1)
                            cute.arch.fence_view_async_shared()
                            load_kv_pipeline.sync_object_full.arrive(prev_handle.index, 0)
                            prev_handle = v_handles[0]
                            sV_cur = sV[None, None, None, v_handles[1].index]
                            self.load_paged64_v_stage(
                                mV_h,
                                v_page0,
                                v_page1,
                                d_slice_v + half_base_off,
                                sV_cur,
                                seqlen_k,
                                v_base,
                                load_tid,
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(1)
                            cute.arch.fence_view_async_shared()
                            load_kv_pipeline.sync_object_full.arrive(prev_handle.index, 0)
                            prev_handle = v_handles[1]
                            kv_coord += 1
                        # Vend: V of the last tile.
                        v_tile = kv_coord - 1
                        v_page0_idx = 2 * v_tile
                        v_page1_idx = v_page0_idx + (
                            Int32(1)
                            if v_tile * tile_n + tile_n // 2 < seqlen_k
                            else Int32(0)
                        )
                        v_page0 = mPageTable[batch_coord, v_page0_idx]
                        v_page1 = mPageTable[batch_coord, v_page1_idx]
                        v_base = v_tile * tile_n
                        vend_handles = [
                            load_kv_producer.acquire_and_advance()
                            for _ in range(self.iterations_pv)
                        ]
                        sV_cur = sV[None, None, None, vend_handles[0].index]
                        self.load_paged64_v_stage(
                            mV_h,
                            v_page0,
                            v_page1,
                            half_base_off,
                            sV_cur,
                            seqlen_k,
                            v_base,
                            load_tid,
                        )
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(1)
                        cute.arch.fence_view_async_shared()
                        load_kv_pipeline.sync_object_full.arrive(prev_handle.index, 0)
                        sV_cur = sV[None, None, None, vend_handles[1].index]
                        self.load_paged64_v_stage(
                            mV_h,
                            v_page0,
                            v_page1,
                            d_slice_v + half_base_off,
                            sV_cur,
                            seqlen_k,
                            v_base,
                            load_tid,
                        )
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        load_kv_pipeline.sync_object_full.arrive(vend_handles[0].index, 0)
                        load_kv_pipeline.sync_object_full.arrive(vend_handles[1].index, 0)
                    if cutlass.const_expr(not self.paged_kv_non_tma):
                        k_page_idx = (
                            mPageTable[batch_coord, kv_coord]
                            if cutlass.const_expr(mPageTable is not None)
                            else None
                        )
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            k_handle = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                tKgK[None, kv_coord, iter]
                                if cutlass.const_expr(mPageTable is None)
                                else tKgK[None, 0, iter, k_page_idx],
                                tKsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                        kv_coord += 1
                        # v_page_idx_prev carries K[i-1]'s page index for use as V[i-1]'s page
                        # (K and V for the same KV block share the same physical page).
                        # Also serves as the Vend page index when seqlen_kv_loop_steps == 1.
                        v_page_idx_prev = (
                            k_page_idx if cutlass.const_expr(mPageTable is not None) else None
                        )
                        # Prefetch K1 page after K0 TMA dispatch to hide L2 latency.
                        if cutlass.const_expr(mPageTable is not None):
                            if seqlen_kv_loop_steps > 1:
                                k_page_idx = mPageTable[batch_coord, kv_coord]

                        for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                            # Ki: k_page_idx was prefetched at end of previous iteration
                            # (or in the prologue for i==1); L2 latency already hidden.
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                k_handle = load_kv_producer.acquire_and_advance()
                                cute.copy(
                                    tma_atom_k,
                                    tKgK[None, kv_coord, iter]
                                    if cutlass.const_expr(mPageTable is None)
                                    else tKgK[None, 0, iter, k_page_idx],
                                    tKsK[None, k_handle.index],
                                    tma_bar_ptr=k_handle.barrier,
                                )
                            # Vi-1: reuse v_page_idx_prev (= K[i-1]'s page), no extra GMEM read.
                            for iter in cutlass.range(self.iterations_pv, unroll=1):
                                v_handle = load_kv_producer.acquire_and_advance()
                                cute.copy(
                                    tma_atom_v,
                                    tVgV[None, iter, kv_coord - 1]
                                    if cutlass.const_expr(mPageTable is None)
                                    else tVgV[None, iter, 0, v_page_idx_prev],
                                    tVsV[None, v_handle.index],
                                    tma_bar_ptr=v_handle.barrier,
                                )
                            v_page_idx_prev = (
                                k_page_idx if cutlass.const_expr(mPageTable is not None) else None
                            )
                            kv_coord += 1
                            # Prefetch next K page while V TMA is in flight.
                            if cutlass.const_expr(mPageTable is not None):
                                if kv_coord < seqlen_kv_loop_end:
                                    k_page_idx = mPageTable[batch_coord, kv_coord]
                        # Vend: reuse v_page_idx_prev (= K[end-1]'s page), no extra GMEM read.
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            v_handle = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                tVgV[None, iter, seqlen_kv_loop_end - 1]
                                if cutlass.const_expr(mPageTable is None)
                                else tVgV[None, iter, 0, v_page_idx_prev],
                                tVsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )

                work_tile = tile_sched.advance_to_next_work()
                # End of persistent scheduler loop
            if warp_idx == self.load_warp_id:
                load_kv_producer.tail()
                load_q_producer.tail()
            if cutlass.const_expr(self.split_kv_loads):
                if warp_idx == self.v_load_warp_id:
                    load_v_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_leader_cta = cta_rank_in_cluster % 2 == 0

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                if cutlass.const_expr(self.is_persistent and cum_seqlen_q is not None):
                    if cutlass.const_expr(self.remap_hoist):
                        _pm, _pb = self._persist_remap_coord_hoist(
                            curr_block_coord[0], req_map, persist_num_m_ctas
                        )
                    else:
                        _pm, _pb = self._persist_remap_coord(
                            curr_block_coord[0], cum_seqlen_q, seqused_k, persist_num_m_ctas
                        )
                    curr_block_coord = (
                        _pm, curr_block_coord[1], (curr_block_coord[2][0], _pb)
                    )
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                seqlen_q = (
                    mQ_qdl.shape[0][1]
                    if cutlass.const_expr(self.pack_gqa)
                    else mQ_qdl.shape[0]
                )
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                batch_coord = curr_block_coord[2][1]
                if cutlass.const_expr(seqused_k is not None):
                    seqlen_k = seqused_k[batch_coord]
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q * self.qhead_per_kvhead
                        if cutlass.const_expr(self.pack_gqa)
                        else seqlen_q,
                    )

                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                seqlen_kv_loop_start, seqlen_kv_loop_steps = (
                    FusedMask.get_trip_start_count_via_block_info(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                        self.qhead_per_kvhead if self.pack_gqa else 1,
                    )
                )
                seqlen_kv_loop_start, seqlen_kv_loop_steps = self._split_trip_range(
                    seqlen_kv_loop_start,
                    seqlen_kv_loop_steps,
                    seqlen_k,
                    split_idx,
                    num_splits,
                )
                if cutlass.const_expr(self.is_split_kv):
                    # Empty split sub-range: no MMAs for this cluster.
                    if seqlen_kv_loop_steps == 0:
                        continue_cond = True

                if not continue_cond:
                    seqlen_kv_loop_end = seqlen_kv_loop_start + seqlen_kv_loop_steps

                    load_q_releaser = load_q_consumer.clone()
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if seqlen_kv_loop_steps > 1:
                        # QK0
                        if is_leader_cta:
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                            s_handle.commit()
                        for i in cutlass.range(1, seqlen_kv_loop_steps - 1, 1, unroll=1):
                            # QKi
                            if is_leader_cta:
                                s_handle = mma_s_producer.acquire_and_advance()
                                tStS_slice = tStS[None, None, None, s_handle.index]
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                for iter in cutlass.range(self.iterations_qk, unroll=1):
                                    tSrQ_slice = tSrQ[None, None, None, iter]
                                    k_handle = load_kv_consumer.wait_and_advance()
                                    tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                    num_kphases = cute.size(tSrQ_slice, mode=[2])
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    k_handle.release()
                                s_handle.commit()

                                # PVi-1
                                p_handle = p_mma_consumer.wait_and_advance()
                                o_handle = mma_corr_producer.acquire_and_advance()
                                pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                                for iter in cutlass.range(self.iterations_pv, unroll=1):
                                    if cutlass.const_expr(self.split_kv_loads):
                                        v_handle = load_v_consumer.wait_and_advance()
                                    else:
                                        v_handle = load_kv_consumer.wait_and_advance()
                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                                    tOtO_slice = tOtO_staged[None, None, None, iter]
                                    tStS_slice = tStS[None, None, None, p_handle.index]
                                    tP = cute.make_tensor(
                                        tStS_slice.iterator, p_tmem_layout_staged.outer
                                    )
                                    tOrP = pv_thr_mma.make_fragment_A(tP)
                                    tOrP_slice = cute.make_tensor(
                                        cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                        tOrP.layout,
                                    )
                                    tOrV_slice = tOrV[None, None, None, v_handle.index]
                                    num_kphases = cute.size(tOrV_slice, mode=[2])
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            pv_tiled_mma,
                                            tOtO_slice,
                                            tOrP_slice[kphase_coord],
                                            tOrV_slice[kphase_coord],
                                            tOtO_slice,
                                        )
                                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    v_handle.release()
                                o_handle.commit()
                                p_handle.release()
                        if is_leader_cta:
                            # QKend
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                            # PVend-1
                            p_handle = p_mma_consumer.wait_and_advance()
                            o_handle = mma_corr_producer.acquire_and_advance()
                            pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            for iter in cutlass.range(self.iterations_pv, unroll=1):
                                if cutlass.const_expr(self.split_kv_loads):
                                    v_handle = load_v_consumer.wait_and_advance()
                                else:
                                    v_handle = load_kv_consumer.wait_and_advance()
                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                                tOtO_slice = tOtO_staged[None, None, None, iter]
                                tStS_slice = tStS[None, None, None, p_handle.index]
                                tP = cute.make_tensor(
                                    tStS_slice.iterator, p_tmem_layout_staged.outer
                                )
                                tOrP = pv_thr_mma.make_fragment_A(tP)
                                tOrP_slice = cute.make_tensor(
                                    cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                    tOrP.layout,
                                )
                                tOrV_slice = tOrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tOrV_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        pv_tiled_mma,
                                        tOtO_slice,
                                        tOrP_slice[kphase_coord],
                                        tOrV_slice[kphase_coord],
                                        tOtO_slice,
                                    )
                                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                            o_handle.commit()
                            p_handle.release()
                    else:
                        if is_leader_cta:
                            # QK0
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for iter in cutlass.range(self.iterations_qk, unroll=1):
                                load_q_consumer.wait_and_advance()
                                tSrQ_slice = tSrQ[None, None, None, iter]
                                k_handle = load_kv_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                load_q_releaser.release()
                                load_q_releaser.advance()
                            s_handle.commit()

                    if is_leader_cta:
                        # PVend
                        p_handle = p_mma_consumer.wait_and_advance()
                        o_handle = mma_corr_producer.acquire_and_advance()
                        pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            if cutlass.const_expr(self.split_kv_loads):
                                v_handle = load_v_consumer.wait_and_advance()
                            else:
                                v_handle = load_kv_consumer.wait_and_advance()
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                            tOtO_slice = tOtO_staged[None, None, None, iter]
                            tStS_slice = tStS[None, None, None, p_handle.index]
                            tP = cute.make_tensor(tStS_slice.iterator, p_tmem_layout_staged.outer)
                            tOrP = pv_thr_mma.make_fragment_A(tP)
                            tOrP_slice = cute.make_tensor(
                                cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                tOrP.layout,
                            )
                            tOrV_slice = tOrV[None, None, None, v_handle.index]
                            num_kphases = cute.size(tOrV_slice, mode=[2])
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                kphase_coord = (None, None, kphase_idx)
                                cute.gemm(
                                    pv_tiled_mma,
                                    tOtO_slice,
                                    tOrP_slice[kphase_coord],
                                    tOrV_slice[kphase_coord],
                                    tOtO_slice,
                                )
                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            v_handle.release()
                        o_handle.commit()
                        p_handle.release()
                work_tile = tile_sched.advance_to_next_work()
            # End of persistent scheduler loop
            mma_s_producer.tail()
            mma_corr_producer.tail()

        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                if cutlass.const_expr(self.is_persistent and cum_seqlen_q is not None):
                    if cutlass.const_expr(self.remap_hoist):
                        _pm, _pb = self._persist_remap_coord_hoist(
                            curr_block_coord[0], req_map, persist_num_m_ctas
                        )
                    else:
                        _pm, _pb = self._persist_remap_coord(
                            curr_block_coord[0], cum_seqlen_q, seqused_k, persist_num_m_ctas
                        )
                    curr_block_coord = (
                        _pm, curr_block_coord[1], (curr_block_coord[2][0], _pb)
                    )
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                seqlen_q = (
                    mQ_qdl.shape[0][1]
                    if cutlass.const_expr(self.pack_gqa)
                    else mQ_qdl.shape[0]
                )
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                if cutlass.const_expr(seqused_k is not None):
                    seqlen_k = seqused_k[batch_coord]
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q * self.qhead_per_kvhead
                        if cutlass.const_expr(self.pack_gqa)
                        else seqlen_q,
                    )
                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                start_count, trip_count = FusedMask.get_trip_start_count_via_block_info(
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    self.is_causal,
                    self.is_local,
                    window_size_left,
                    window_size_right,
                    self.qhead_per_kvhead if self.pack_gqa else 1,
                )
                start_count, trip_count = self._split_trip_range(
                    start_count,
                    trip_count,
                    seqlen_k,
                    split_idx,
                    num_splits,
                )
                if cutlass.const_expr(self.is_split_kv):
                    # Empty split sub-range: no softmax steps and no LSE store
                    # here; the correction warp writes the -inf partial LSE.
                    if trip_count == 0:
                        continue_cond = True
                if not continue_cond:
                    row_max = -Float32.inf
                    row_max_prev = -Float32.inf
                    row_sum = 0.0

                    end_count = start_count + trip_count
                    # require at least one softmax iteration for zero trip_count case;
                    # rely on masking this iteration for correctness
                    if end_count <= start_count:
                        start_count = 0
                        end_count = 1
                    if cutlass.const_expr(self.use_semantic_trip_range):
                        n_block_min_causal_local_mask, n_block_min_before_local_mask = (
                            FusedMask.get_trip_mask_bounds_via_block_info(
                                mma_block_coord,
                                self.qk_mma_tiler,
                                seqlen_q,
                                seqlen_k,
                                self.is_causal,
                                self.is_local,
                                window_size_left,
                                window_size_right,
                                self.qhead_per_kvhead if self.pack_gqa else 1,
                            )
                        )
                    cS_base = cute.make_identity_tensor(
                        (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
                    )
                    cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0), cS_base)
                    tScS = qk_thr_mma.partition_C(cS)

                    if cutlass.const_expr(self.static_mask_loops):
                        masked_start = cutlass.min(
                            cutlass.max(start_count, n_block_min_causal_local_mask),
                            end_count,
                        )
                        for step in cutlass.range(start_count, masked_start, 1, unroll=1):
                            cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
                            tScS_iter = qk_thr_mma.partition_C(cS_iter)
                            (
                                row_max,
                                row_sum,
                                mma_s_consumer,
                                p_mma_producer,
                                s_corr_producer,
                            ) = self.softmax_step(
                                (False, window_size_left, window_size_right),
                                (
                                    row_max_prev,
                                    row_sum,
                                    seqlen_q,
                                    seqlen_k,
                                    scale_softmax_log2,
                                ),
                                (tStS, tScS_iter),
                                (mma_s_consumer, p_mma_producer, s_corr_producer),
                                1,
                            )
                            row_max_prev = row_max
                        for step in cutlass.range(masked_start, end_count, 1, unroll=1):
                            cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
                            tScS_iter = qk_thr_mma.partition_C(cS_iter)
                            (
                                row_max,
                                row_sum,
                                mma_s_consumer,
                                p_mma_producer,
                                s_corr_producer,
                            ) = self.softmax_step(
                                (True, window_size_left, window_size_right),
                                (
                                    row_max_prev,
                                    row_sum,
                                    seqlen_q,
                                    seqlen_k,
                                    scale_softmax_log2,
                                ),
                                (tStS, tScS_iter),
                                (mma_s_consumer, p_mma_producer, s_corr_producer),
                                2,
                            )
                            row_max_prev = row_max
                    else:
                        for step in cutlass.range(start_count, end_count, 1, unroll=1):
                            cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
                            tScS_iter = qk_thr_mma.partition_C(cS_iter)
                            if cutlass.const_expr(self.use_semantic_trip_range):
                                need_apply_mask = (
                                    step >= n_block_min_causal_local_mask
                                    or step < n_block_min_before_local_mask
                                    or step == end_count - 1
                                )
                            else:
                                need_apply_mask = step == end_count - 1
                            (
                                row_max,
                                row_sum,
                                mma_s_consumer,
                                p_mma_producer,
                                s_corr_producer,
                            ) = self.softmax_step(
                                (need_apply_mask, window_size_left, window_size_right),
                                (
                                    row_max_prev,
                                    row_sum,
                                    seqlen_q,
                                    seqlen_k,
                                    scale_softmax_log2,
                                ),
                                (tStS, tScS_iter),
                                (mma_s_consumer, p_mma_producer, s_corr_producer),
                            )
                            row_max_prev = row_max
                    sum_producer = self.store_sum_max(
                        row_max,
                        mLSE,
                        row_sum,
                        sSum,
                        sum_producer,
                        curr_block_coord,
                        seqlen_q,
                        cum_seqlen_q,
                        cuseqlen_q,
                        scale_softmax,
                        split_idx,
                    )
                work_tile = tile_sched.advance_to_next_work()
            p_mma_producer.tail()
            s_corr_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)

            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                if cutlass.const_expr(self.is_persistent and cum_seqlen_q is not None):
                    if cutlass.const_expr(self.remap_hoist):
                        _pm, _pb = self._persist_remap_coord_hoist(
                            curr_block_coord[0], req_map, persist_num_m_ctas
                        )
                    else:
                        _pm, _pb = self._persist_remap_coord(
                            curr_block_coord[0], cum_seqlen_q, seqused_k, persist_num_m_ctas
                        )
                    curr_block_coord = (
                        _pm, curr_block_coord[1], (curr_block_coord[2][0], _pb)
                    )
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                seqlen_q = (
                    mQ_qdl.shape[0][1]
                    if cutlass.const_expr(self.pack_gqa)
                    else mQ_qdl.shape[0]
                )
                seqlen_k = (
                    mK_kdl.shape[0] if cutlass.const_expr(mPageTable is None) else max_seqlen_k
                )
                if cutlass.const_expr(seqused_k is not None):
                    seqlen_k = seqused_k[batch_coord]
                continue_cond = False
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_block_coord[0],
                        seqlen_q * self.qhead_per_kvhead
                        if cutlass.const_expr(self.pack_gqa)
                        else seqlen_q,
                    )

                if not continue_cond:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    mO_qdl_eff = mO_qdl
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        if cutlass.const_expr(self.pack_gqa):
                            block_offset_o = (
                                (Int32(0), cuseqlen_q),
                                Int32(0),
                                Int32(0),
                                (Int32(0), Int32(0)),
                            )
                        else:
                            block_offset_o = (
                                cuseqlen_q,
                                Int32(0),
                                Int32(0),
                                ((Int32(0), Int32(0)), Int32(0)),
                            )
                        if cutlass.const_expr(self.is_split_kv):
                            # Packed rows + trailing split mode (offset 0).
                            block_offset_o = block_offset_o + (Int32(0),)
                            o_offset_modes = [0, 2, 3, 4]
                        else:
                            o_offset_modes = [0, 2, 3]
                        mO_qdl_eff = cute.domain_offset(
                            cute.select(block_offset_o, mode=o_offset_modes), mO_qdl
                        )

                    # (bM, bN, loopM, loopN, loopL)
                    if cutlass.const_expr(self.pack_gqa):
                        # Packed rows: tile the nested (h_r, s) row mode with the
                        # matching nested box ((h_r, tok_per_cta)) so the tile
                        # shape stays static (a flat 128 tiler over the nested
                        # dynamic mode yields dynamic tile coordinates).
                        rows_per_cta_o = self.pv_block_tiler[0]
                        h_r_o = self.qhead_per_kvhead
                        tok_per_cta_o = rows_per_cta_o // h_r_o
                        o_tile = ((h_r_o, tok_per_cta_o), self.pv_block_tiler[1])
                    else:
                        o_tile = cute.select(self.pv_block_tiler, mode=[0, 1])
                    gO_qdl = cute.flat_divide(mO_qdl_eff, o_tile)
                    cO_qdl = cute.flat_divide(
                        cute.make_identity_tensor(mO_qdl_eff.shape),
                        o_tile,
                    )

                    corr_trip_start, seqlen_kv_loop_steps = FusedMask.get_trip_start_count_via_block_info(
                        mma_block_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        self.is_causal,
                        self.is_local,
                        window_size_left,
                        window_size_right,
                        self.qhead_per_kvhead if self.pack_gqa else 1,
                    )
                    _, seqlen_kv_loop_steps = self._split_trip_range(
                        corr_trip_start,
                        seqlen_kv_loop_steps,
                        seqlen_k,
                        split_idx,
                        num_splits,
                    )
                    corr_has_work = (
                        seqlen_kv_loop_steps > 0
                        if cutlass.const_expr(self.is_split_kv)
                        else True
                    )
                    if cutlass.const_expr(self.is_split_kv):
                        if not corr_has_work:
                            # Empty split sub-range: nothing was produced on the
                            # pipelines for this tile, so skip every wait and just
                            # record -inf partial LSE for the combine reduction.
                            self.store_split_empty_lse(
                                mLSE, curr_block_coord, cuseqlen_q, seqlen_q, split_idx
                            )
                    if corr_has_work:
                        if cutlass.const_expr(self.is_split_kv):
                            gO_staged = gO_qdl[
                                None, None, curr_block_coord[0], None, curr_block_coord[2], split_idx
                            ]
                            cO_staged = cO_qdl[
                                None, None, curr_block_coord[0], None, curr_block_coord[2], split_idx
                            ]
                        else:
                            gO_staged = gO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                            cO_staged = cO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                        cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                        tScS = qk_thr_mma.partition_C(cS)

                        # Empty step as the first step is no need for correction
                        stats_handle = s_corr_consumer.wait_and_advance()
                        stats_handle.release()
                        for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                            # Oi-1 -> Oi
                            mma_corr_consumer, s_corr_consumer = self.correction_rescale(
                                scale_softmax_log2,
                                (s_corr_consumer, tStS, tScS),
                                (mma_corr_consumer, tOtO_staged, cO_staged),
                                self.epi_tile,
                            )
                        # O_partial -> O_final. With pack_gqa the O rows live in the
                        # packed (h_r, token) space, so the row bound is seqlen_q*h_r.
                        o_row_bound = (
                            seqlen_q * self.qhead_per_kvhead
                            if cutlass.const_expr(self.pack_gqa)
                            else seqlen_q
                        )
                        if cutlass.const_expr(self.coalesced_epilog and self.pack_gqa):
                            # Request-relative ((h_r, s), d) O view for the row-wise
                            # coalesced store (d-slice offsets are applied inside).
                            if cutlass.const_expr(self.is_split_kv):
                                mO_req = mO_qdl_eff[
                                    None, None, curr_block_coord[2], split_idx
                                ]
                            else:
                                mO_req = mO_qdl_eff[None, None, curr_block_coord[2]]
                            mma_corr_consumer, sum_consumer = self.correction_epilog_coalesced(
                                (seqlen_q, scale_output),
                                (sum_consumer, sSum),
                                (mma_corr_consumer, tOtO_staged),
                                (
                                    mO_req,
                                    curr_block_coord[0],
                                    sO_epi,
                                    gmem_tiled_copy_O,
                                    pv_thr_mma,
                                ),
                            )
                        else:
                            mma_corr_consumer, sum_consumer = self.correction_epilog(
                                (o_row_bound, scale_output),
                                (sum_consumer, sSum),
                                (mma_corr_consumer, gO_staged, cO_staged, tOtO_staged),
                                self.epi_tile,
                            )
                work_tile = tile_sched.advance_to_next_work()
            # NOTE: tmem.free() moved to kernel end to enable cluster-wide sync

        # ///////////////////////////////////////////////////////////////////////////////
        #  Scheduler Warp (only for CLC dynamic scheduler)
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.use_clc_scheduler):
            is_first_cta_in_cluster = cta_rank_in_cluster == 0

            if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
                # Private work_tile (mirrors the general SM100 kernel's
                # clc_scheduler_warp): the shared work_tile variable belongs to
                # the consumer sections; the sched warp drives the producer loop
                # from its own initial (always-valid launch coord) tile.
                sched_work_tile = tile_sched.initial_work_tile_info()
                while sched_work_tile.is_valid_tile:
                    tile_sched.prefetch_next_work()
                    sched_work_tile = tile_sched.advance_to_next_work()
                tile_sched.producer_tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Empty warps reg dealloc
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.use_clc_scheduler):
            if warp_idx > self.load_warp_id:
                if not (warp_idx == self.sched_warp_id and is_first_cta_in_cluster):
                    cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
        else:
            if warp_idx > self.load_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cooperative TMEM Deallocation (2CTA)
        # ///////////////////////////////////////////////////////////////////////////////
        # All warps (including scheduler) have finished by this point.
        # Cluster-wide sync ensures both CTAs reach here before dealloc.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

        return

    @cute.jit
    def _split_trip_range(self, start, steps, seqlen_k, split_idx, num_splits):
        """Intersect a work tile's causal trip range with this split's KV block
        range. Split s covers 128-token KV blocks [s*per_split, (s+1)*per_split)
        capped at the request's block count; blocks are absolute within the
        request so the returned range stays in the mask's coordinate space."""
        if cutlass.const_expr(self.is_split_kv):
            tile_n = self.qk_mma_tiler[1]
            total_blocks = (seqlen_k + tile_n - 1) // tile_n
            per_split = (total_blocks + num_splits - 1) // num_splits
            lo = split_idx * per_split
            hi = lo + per_split if lo + per_split < total_blocks else total_blocks
            eff_start = start if start > lo else lo
            eff_end = start + steps
            eff_end = eff_end if eff_end < hi else hi
            eff_steps = eff_end - eff_start if eff_end > eff_start else Int32(0)
            return eff_start, eff_steps
        else:
            return start, steps

    @cute.jit
    def store_split_empty_lse(
        self,
        mLSE,
        current_block_coord,
        cuseqlen_q,
        seqlen_q,
        split_idx,
    ):
        """Write -inf partial LSE for every in-bounds row of a split whose KV
        sub-range does not intersect the tile's trip (combine weights it 0)."""
        if cutlass.const_expr(mLSE is not None and self.pack_gqa and self.is_split_kv):
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
            h_r = self.qhead_per_kvhead
            packed_row = current_block_coord[0] * self.cta_tiler[0] + thread_idx
            if cute.elem_less(packed_row, seqlen_q * h_r):
                h_r_idx = packed_row % h_r
                tok = packed_row // h_r + cuseqlen_q
                kv_head = current_block_coord[2][0]
                mLSE[tok, ((h_r_idx, kv_head), Int32(0)), split_idx] = -Float32.inf

    @cute.jit
    def load_paged64_k_stage(
        self,
        mK_h: cute.Tensor,
        page: Int32,
        d_offset: Int32,
        sK_stage: cute.Tensor,
        seqlen_k: Int32,
        tile_half_base: Int32,
        load_tid: Int32,
    ):
        """cp.async one K stage: 64 tokens x 128 d-slice from a single page.

        The 2-CTA pair splits the QK B operand along tokens, so each CTA stages
        its own 64-token half of the tile (exactly one page). The stage offset
        function is 128*tok + d, i.e. contiguous 128-d rows, matching the gmem
        page rows -> direct 16B cp.async chunks, no transpose.
        """
        elems_per_chunk = 128 // self.k_dtype.width  # 16 e4m3 elements = 16B
        chunks_per_row = self.qk_mma_tiler[2] // elems_per_chunk
        rows_per_pass = self.threads_per_warp // chunks_per_row
        rows_per_cta = self.qk_mma_tiler[1] // self.cluster_shape_mn[0]
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            self.k_dtype,
            num_bits_per_copy=128,
        )
        thr_layout = cute.make_ordered_layout(
            (rows_per_pass, chunks_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, elems_per_chunk))
        tiled_copy = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(load_tid)
        sK_flat = cute.make_tensor(
            sK_stage.iterator,
            cute.make_layout(
                (rows_per_cta, self.qk_mma_tiler[2]),
                stride=(self.qk_mma_tiler[2], 1),
            ),
        )
        cK = cute.make_identity_tensor((rows_per_cta, self.qk_mma_tiler[2]))
        tKsK = thr_copy.partition_D(sK_flat)
        tKcK = thr_copy.partition_S(cK)
        row_group = load_tid // chunks_per_row
        for m in cutlass.range(cute.size(tKsK, mode=[1]), unroll=4):
            row = m * rows_per_pass + row_group
            # Rows at/past seqlen_k keep stale smem (masked out in softmax).
            row_valid = tile_half_base + row < seqlen_k
            should_load = cute.make_fragment_like(tKsK[(0, None), m, 0], cute.Boolean)
            should_load.fill(row_valid)
            row_ptr = elem_pointer(mK_h, (row, d_offset, page)).toint()
            mX_row = cute.make_tensor(
                cute.make_ptr(
                    self.k_dtype, row_ptr, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout((self.qk_mma_tiler[2],)),
            )
            mX_row_copy = cute.tiled_divide(mX_row, (elems_per_chunk,))
            for k in cutlass.range_constexpr(cute.size(tKsK, mode=[2])):
                ki = tKcK[0, 0, k][1] // elems_per_chunk
                dst_k = tKsK[None, m, k]
                src_k = mX_row_copy[None, ki]
                src_k = cute.make_tensor(src_k.iterator, dst_k.layout)
                cute.copy(tiled_copy, src_k, dst_k, pred=should_load)

    @cute.jit
    def load_paged64_v_stage(
        self,
        mV_h: cute.Tensor,
        page0: Int32,
        page1: Int32,
        d_offset: Int32,
        sV_stage: cute.Tensor,
        seqlen_k: Int32,
        tile_token_base: Int32,
        load_tid: Int32,
    ):
        """cp.async one V stage: 128 tile tokens x 64 d (this CTA's d half).

        The V stage offset function is 64*tok + d_local (token-major rows of
        64 contiguous d values), so gmem d-contiguous rows copy straight in.
        Tokens 0-63 come from page0, 64-127 from page1 (caller clamps page1 to
        page0 when the KV length does not cover the second half of the tile).
        """
        elems_per_chunk = 128 // self.k_dtype.width  # 16
        tile_n = self.qk_mma_tiler[1]                # 128 tokens
        d_per_cta = tile_n // self.cluster_shape_mn[0]  # 64
        chunks_per_row = d_per_cta // elems_per_chunk   # 4
        rows_per_pass = self.threads_per_warp // chunks_per_row  # 8
        half_tokens = tile_n // 2
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            self.k_dtype,
            num_bits_per_copy=128,
        )
        thr_layout = cute.make_ordered_layout(
            (rows_per_pass, chunks_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, elems_per_chunk))
        tiled_copy = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(load_tid)
        sV_flat = cute.make_tensor(
            sV_stage.iterator,
            cute.make_layout((tile_n, d_per_cta), stride=(d_per_cta, 1)),
        )
        tVsV = thr_copy.partition_D(sV_flat)
        cVt = cute.make_identity_tensor((tile_n, d_per_cta))
        tVcV = thr_copy.partition_S(cVt)
        for m in cutlass.range(cute.size(tVsV, mode=[1]), unroll=4):
            tok = m * rows_per_pass + (load_tid // chunks_per_row)
            in_second_half = tok >= half_tokens
            page = page1 if in_second_half else page0
            tok_in_page = tok - half_tokens if in_second_half else tok
            # Token rows at/past seqlen_k keep stale smem (masked out in softmax);
            # clamp the page index so masked rows never dereference a missing page.
            row_valid = tile_token_base + tok < seqlen_k
            if not row_valid:
                page = page0
            row_ptr = elem_pointer(mV_h, (d_offset, tok_in_page, page)).toint()
            mX_row = cute.make_tensor(
                cute.make_ptr(
                    self.k_dtype, row_ptr, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout((d_per_cta,)),
            )
            mX_row_copy = cute.tiled_divide(mX_row, (elems_per_chunk,))
            should_load = cute.make_fragment_like(tVsV[(0, None), m, 0], cute.Boolean)
            should_load.fill(row_valid)
            for k in cutlass.range_constexpr(cute.size(tVsV, mode=[2])):
                ki = tVcV[0, 0, k][1] // elems_per_chunk
                dst_k = tVsV[None, m, k]
                src_k = mX_row_copy[None, ki]
                src_k = cute.make_tensor(src_k.iterator, dst_k.layout)
                cute.copy(tiled_copy, src_k, dst_k, pred=should_load)

    @cute.jit
    def softmax_step(
        self,
        mask_args: Tuple,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        mask_mode: cutlass.Constexpr[int] = 0,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        need_apply_mask, window_size_left, window_size_right = mask_args
        row_max, row_sum, seqlen_q, seqlen_k, scale_softmax_log2 = value_args
        tStS, tScS = tensor_args
        mma_s_consumer, p_mma_producer, s_corr_producer = pipeline_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        tmem_load_op = (
            tcgen05.copy.LdRed32x32bOp(tcgen05.copy.Repetition(32))
            if cutlass.const_expr(self.use_ldred_rowmax)
            else tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
        )
        tmem_load_atom = cute.make_copy_atom(tmem_load_op, self.qk_acc_dtype)
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        old_row_max = row_max
        hw_row_max = Float32(-Float32.inf)
        if cutlass.const_expr(self.use_ldred_rowmax):
            tTMEM_LOADrS_red = cute.make_rmem_tensor(
                ((1, 1), *tTMEM_LOADrS.shape[1:]), self.qk_acc_dtype
            )
            cute.copy(tmem_tiled_load, tTMEM_LOADtS, (tTMEM_LOADrS, tTMEM_LOADrS_red))
            for i in cutlass.range_constexpr(cute.size(tTMEM_LOADrS_red.shape)):
                hw_row_max = cute.arch.fmax(hw_row_max, tTMEM_LOADrS_red[i])
        else:
            cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
        cute.arch.fence_view_async_tmem_load()
        s_handle.release()
        if cutlass.const_expr(mask_mode == 2):
            FusedMask.apply_mask_via_causal_local(
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                seqlen_q,
                seqlen_k,
                self.use_semantic_trip_range,
                self.is_causal,
                self.is_local,
                window_size_left,
                window_size_right,
                qhead_per_kvhead_packgqa=self.qhead_per_kvhead if self.pack_gqa else 1,
            )
        elif cutlass.const_expr(mask_mode == 0):
            if need_apply_mask:
                FusedMask.apply_mask_via_causal_local(
                    tTMEM_LOADrS,
                    tTMEM_LOADcS,
                    seqlen_q,
                    seqlen_k,
                    self.use_semantic_trip_range,
                    self.is_causal,
                    self.is_local,
                    window_size_left,
                    window_size_right,
                    qhead_per_kvhead_packgqa=self.qhead_per_kvhead if self.pack_gqa else 1,
                )
        if cutlass.const_expr(self.use_ldred_rowmax):
            if cutlass.const_expr(mask_mode == 1):
                row_max = cute.arch.fmax(row_max, hw_row_max)
            elif cutlass.const_expr(mask_mode == 2):
                row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
            else:
                row_max = (
                    tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
                    if need_apply_mask
                    else cute.arch.fmax(row_max, hw_row_max)
                )
        else:
            row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0
        row_max_delta = old_row_max - row_max_safe
        if cutlass.const_expr(self.softmax_rescale_threshold > 0.0):
            acc_scale_log2 = scale_softmax_log2 * row_max_delta
            if acc_scale_log2 >= -self.softmax_rescale_threshold:
                row_max = old_row_max
                row_max_safe = old_row_max
                row_max_delta = Float32(0.0)

        stats_handle = s_corr_producer.acquire_and_advance()
        shared_rescale = Float32(0.0)
        if cutlass.const_expr(self.share_rescale):
            shared_rescale = cute.math.exp2(
                scale_softmax_log2 * row_max_delta, fastmath=True
            )
            stats_layout = cute.composition(
                tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 1))
            )
            stats_c_layout = cute.composition(
                tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 1))
            )
            tOtStats = cute.make_tensor(
                tStS_slice.iterator + self.tilePlikeFP32, stats_layout
            )
            tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
            tmem_store_stats_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)),
                self.qk_acc_dtype,
            )
            tiled_tmem_store_stats = tcgen05.make_tmem_copy(tmem_store_stats_atom, tOtStats)
            thr_tmem_store_stats = tiled_tmem_store_stats.get_slice(thread_idx)
            tTMEM_STOREcStats = thr_tmem_store_stats.partition_S(tOcStats)
            tTMEM_STORErStats = cute.make_rmem_tensor(
                tTMEM_STOREcStats.shape, self.qk_acc_dtype
            )
            tTMEM_STORErStats[0] = shared_rescale
            tTMEM_STOREtStats = thr_tmem_store_stats.partition_D(tOtStats)
            cute.copy(tiled_tmem_store_stats, tTMEM_STORErStats, tTMEM_STOREtStats)
        else:
            stats_layout = cute.composition(
                tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2))
            )
            stats_c_layout = cute.composition(
                tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2))
            )
            tOtStats = cute.make_tensor(
                tStS_slice.iterator + self.tilePlikeFP32, stats_layout
            )
            tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
            tmem_store_stats_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
                self.qk_acc_dtype,
            )
            tiled_tmem_store_stats = tcgen05.make_tmem_copy(tmem_store_stats_atom, tOtStats)
            thr_tmem_store_stats = tiled_tmem_store_stats.get_slice(thread_idx)
            tTMEM_STOREcStats = thr_tmem_store_stats.partition_S(tOcStats)
            tTMEM_STORErStats = cute.make_rmem_tensor(
                tTMEM_STOREcStats.shape, self.qk_acc_dtype
            )
            tTMEM_STORErStats[0] = old_row_max
            tTMEM_STORErStats[1] = row_max_safe
            tTMEM_STOREtStats = thr_tmem_store_stats.partition_D(tOtStats)
            cute.copy(tiled_tmem_store_stats, tTMEM_STORErStats, tTMEM_STOREtStats)
        cute.arch.fence_view_async_tmem_store()
        stats_handle.commit()

        scale = scale_softmax_log2
        minus_row_max_scale = Float32(self.softmax_max_offset) - row_max_safe * scale
        # Acquire P write slot early — overlaps any pipeline stall with exp2 compute
        p_handle = p_mma_producer.acquire_and_advance()
        # Fragment-based FMA + exp2 + bf16 conversion
        # Trades SFU for FMA via polynomial emulation on a fraction of elements
        ex2_frg_tile = 32
        ex2_frg_cnt = cute.size(tTMEM_LOADrS) // ex2_frg_tile
        tTMEM_LOADrS_ex2 = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(ex2_frg_tile))
        tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.q_dtype)
        tTMEM_STORErP_ex2 = cute.logical_divide(tTMEM_STORErP, cute.make_layout(ex2_frg_tile))
        for j in cutlass.range_constexpr(ex2_frg_cnt):
            for k in cutlass.range_constexpr(0, ex2_frg_tile, 2):
                tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j] = cute.arch.fma_packed_f32x2(
                    (tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                )
                if cutlass.const_expr(self.ex2_emu_freq == 0):
                    tTMEM_LOADrS_ex2[k, j] = cute.math.exp2(tTMEM_LOADrS_ex2[k, j], fastmath=True)
                    tTMEM_LOADrS_ex2[k + 1, j] = cute.math.exp2(
                        tTMEM_LOADrS_ex2[k + 1, j], fastmath=True
                    )
                else:
                    if cutlass.const_expr(
                        k % self.ex2_emu_freq < self.ex2_emu_freq - self.ex2_emu_res
                        or j >= ex2_frg_cnt - 1
                        or j < self.ex2_emu_start_frg
                    ):
                        tTMEM_LOADrS_ex2[k, j] = cute.math.exp2(
                            tTMEM_LOADrS_ex2[k, j], fastmath=True
                        )
                        tTMEM_LOADrS_ex2[k + 1, j] = cute.math.exp2(
                            tTMEM_LOADrS_ex2[k + 1, j], fastmath=True
                        )
                    else:
                        tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j] = ex2_emulation_2(
                            tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]
                        )
            tTMEM_STORErP_ex2[None, j].store(tTMEM_LOADrS_ex2[None, j].load().to(self.q_dtype))
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tilePlikeFP32 = tStS_slice.shape[1] // Float32.width * self.q_dtype.width
        tStS_P_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], tilePlikeFP32))
        )
        tStS_P = cute.make_tensor(tStS_slice.iterator, tStS_P_layout)
        tScS_P_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], tilePlikeFP32))
        )
        tScS_P = cute.make_tensor(tScS_slice.iterator, tScS_P_layout)
        tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_store = tmem_tiled_store.get_slice(thread_idx)
        tTMEM_STOREtP = thr_store.partition_D(tStS_P)
        tTMEM_STOREcS = thr_store.partition_S(tScS_P)
        tTMEM_STORErP_ = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErP.iterator, dtype=self.qk_acc_dtype),
            tTMEM_STOREcS.shape,
        )
        cute.copy(tmem_tiled_store, tTMEM_STORErP_, tTMEM_STOREtP)
        cute.arch.fence_view_async_tmem_store()

        p_handle.commit()
        if cutlass.const_expr(self.share_rescale):
            acc_scale = shared_rescale * 0.5
        else:
            acc_scale_ = scale * row_max_delta
            acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        row_sum *= acc_scale
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)
        reduction_unroll = 4
        frg_tile = cute.size(tTMEM_LOADrS) // reduction_unroll
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2):
            local_row_sum_0 = cute.arch.add_packed_f32x2(
                local_row_sum_0, (tTMEM_LOADrS_frg[j, 0], tTMEM_LOADrS_frg[j + 1, 0])
            )
            local_row_sum_1 = cute.arch.add_packed_f32x2(
                local_row_sum_1, (tTMEM_LOADrS_frg[j, 1], tTMEM_LOADrS_frg[j + 1, 1])
            )
            local_row_sum_2 = cute.arch.add_packed_f32x2(
                local_row_sum_2, (tTMEM_LOADrS_frg[j, 2], tTMEM_LOADrS_frg[j + 1, 2])
            )
            local_row_sum_3 = cute.arch.add_packed_f32x2(
                local_row_sum_3, (tTMEM_LOADrS_frg[j, 3], tTMEM_LOADrS_frg[j + 1, 3])
            )
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]
        return row_max, row_sum, mma_s_consumer, p_mma_producer, s_corr_producer

    @cute.jit
    def correction_rescale(
        self,
        scale_softmax_log2: Float32,
        stats_args: tuple,
        o_args: tuple,
        epi_tile: cute.Tile,
    ) -> pipeline.PipelineConsumer:
        (s_corr_consumer, tStS, tScS) = stats_args
        (mma_o_consumer, tOtO_staged, cO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))

        stats_handle = s_corr_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, stats_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        if cutlass.const_expr(self.share_rescale):
            stats_layout = cute.composition(
                tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 1))
            )
            stats_c_layout = cute.composition(
                tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 1))
            )
            tOtStats = cute.make_tensor(
                tStS_slice.iterator + self.tilePlikeFP32, stats_layout
            )
            tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
            tmem_load_stats_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)),
                self.qk_acc_dtype,
            )
            tiled_tmem_load_stats = tcgen05.make_tmem_copy(
                tmem_load_stats_atom, tOtStats
            )
            thr_tmem_load_stats = tiled_tmem_load_stats.get_slice(thread_idx)
            tTMEM_LOADtStats = thr_tmem_load_stats.partition_S(tOtStats)
            tTMEM_LOADcStats = thr_tmem_load_stats.partition_D(tOcStats)
            tTMEM_LOADrStats = cute.make_rmem_tensor(
                tTMEM_LOADcStats.shape, self.qk_acc_dtype
            )
            cute.copy(tiled_tmem_load_stats, tTMEM_LOADtStats, tTMEM_LOADrStats)
            scale = tTMEM_LOADrStats[0]
            if cutlass.const_expr(self.skip_rescale):
                should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
            else:
                should_rescale = True
        else:
            stats_layout = cute.composition(
                tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2))
            )
            stats_c_layout = cute.composition(
                tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2))
            )
            tOtStats = cute.make_tensor(
                tStS_slice.iterator + self.tilePlikeFP32, stats_layout
            )
            tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
            tmem_load_stats_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
                self.qk_acc_dtype,
            )
            tiled_tmem_load_stats = tcgen05.make_tmem_copy(
                tmem_load_stats_atom, tOtStats
            )
            thr_tmem_load_stats = tiled_tmem_load_stats.get_slice(thread_idx)
            tTMEM_LOADtStats = thr_tmem_load_stats.partition_S(tOtStats)
            tTMEM_LOADcStats = thr_tmem_load_stats.partition_D(tOcStats)
            tTMEM_LOADrStats = cute.make_rmem_tensor(
                tTMEM_LOADcStats.shape, self.qk_acc_dtype
            )
            cute.copy(tiled_tmem_load_stats, tTMEM_LOADtStats, tTMEM_LOADrStats)
            scale = scale_softmax_log2 * (
                tTMEM_LOADrStats[0] - tTMEM_LOADrStats[1]
            )
            scale = cute.math.exp2(scale, fastmath=True)
            if cutlass.const_expr(self.skip_rescale):
                should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
            else:
                should_rescale = True
        stats_handle.release()
        o_handle = mma_o_consumer.wait_and_advance()
        if should_rescale:
            for iter in cutlass.range(self.iterations_pv, unroll_full=True):
                tOtO = tOtO_staged[(None, None), 0, 0, iter]
                cO = cO_staged[None, None, iter]
                tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
                cO_epi = cute.zipped_divide(cO, epi_tile)
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(16)),
                    self.pv_acc_dtype,
                )
                tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi)
                thr_load = tmem_tiled_load.get_slice(thread_idx)
                tmem_store_atom = cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.Repetition(16)),
                    self.pv_acc_dtype,
                )
                tmem_store_atom = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_epi)
                thr_store = tmem_store_atom.get_slice(thread_idx)
                tTMEM_LOADtO = thr_load.partition_S(tOtO_epi)
                tTMEM_LOADcO = thr_load.partition_D(cO_epi)
                tTMEM_STOREtO = thr_store.partition_D(tOtO_epi)
                tTMrO = cute.make_rmem_tensor_like(
                    cute.append(
                        cute.make_layout(tTMEM_LOADcO[None, 0, 0].shape),
                        cute.make_layout(2, stride=cute.size(tTMEM_LOADcO[None, 0, 0].shape)),
                    ),
                    self.pv_acc_dtype,
                )
                tTMEM_LOADtO_0 = tTMEM_LOADtO[None, 0, 0]
                cute.copy(tmem_tiled_load, tTMEM_LOADtO_0, tTMrO[None, 0])
                iter_num = cute.size(tTMEM_LOADtO, mode=[1])
                for i in cutlass.range(1, iter_num, unroll_full=True):
                    tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                    cute.copy(tmem_tiled_load, tTMEM_LOADtO_i, tTMrO[None, i % 2])
                    for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                        tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2] = cute.arch.mul_packed_f32x2(
                            (tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2]),
                            (scale, scale),
                        )
                    tTMEM_STOREtO_prev_i = tTMEM_STOREtO[None, i - 1, 0]
                    cute.copy(tmem_store_atom, tTMrO[None, (i - 1) % 2], tTMEM_STOREtO_prev_i)

                for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                    tTMrO[j, (iter_num - 1) % 2], tTMrO[j + 1, (iter_num - 1) % 2] = (
                        cute.arch.mul_packed_f32x2(
                            (
                                tTMrO[j, (iter_num - 1) % 2],
                                tTMrO[j + 1, (iter_num - 1) % 2],
                            ),
                            (scale, scale),
                        )
                    )
                cute.copy(
                    tmem_store_atom,
                    tTMrO[None, (iter_num - 1) % 2],
                    tTMEM_STOREtO[None, iter_num - 1, 0],
                )
            cute.arch.fence_view_async_tmem_store()
        o_handle.release()
        return mma_o_consumer, s_corr_consumer

    @cute.jit
    def correction_epilog(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: Tuple,
        epi_tile: cute.Tile,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        (seqlen_q, scale_output) = value_args
        (sum_consumer, sSum) = sum_args
        (mma_o_consumer, gO_staged, cO_staged, tOtO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
        scale = scale_output / row_sum if not row_sum_is_zero_or_nan else 0.0
        o_handle = mma_o_consumer.wait_and_advance()
        for iter in cutlass.range(self.iterations_pv):
            gO = gO_staged[None, None, iter]
            cO = cO_staged[None, None, iter]
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            gO_epi = cute.zipped_divide(gO, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADgO = thr_tmem_load.partition_D(gO_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            for i in cutlass.range(cute.size(tTMEM_LOADtO, mode=[1]), unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                tTMEM_LOADgO_i = tTMEM_LOADgO[None, i, 0]
                tTMEM_LOADcO_i = tTMEM_LOADcO[None, i, 0]
                tTMrO = cute.make_rmem_tensor(tTMEM_LOADcO[None, 0, i].shape, self.pv_acc_dtype)
                cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                for j in cutlass.range(0, cute.size(tTMrO), 2, unroll_full=True):
                    tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j], tTMrO[j + 1]),
                        (scale, scale),
                    )
                tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                o_vec = tTMrO.load()
                tSMrO.store(o_vec.to(self.o_dtype))
                if cutlass.const_expr(self.pack_gqa):
                    # Packed rows: the element row coordinate is the nested
                    # (head, token) pair of the request-local packed row.  The
                    # row is in-bounds iff its packed index head + h_r * token
                    # is below seqlen_q * h_r (passed as seqlen_q here).
                    # Comparing only the head submode would let the last tile's
                    # overshoot rows (token >= seqlen_q) be stored into the
                    # first packed rows of the next request, since the packed
                    # row space is contiguous across requests.
                    o_row_coord = tTMEM_LOADcO_i[0][0]
                    o_packed_row = (
                        o_row_coord[0] + self.qhead_per_kvhead * o_row_coord[1]
                    )
                    o_row_ok = cute.elem_less(o_packed_row, seqlen_q)
                else:
                    o_row_ok = cute.elem_less(tTMEM_LOADcO_i[0][0], seqlen_q)
                if o_row_ok:
                    cute.autovec_copy(tSMrO, tTMEM_LOADgO_i)
        o_handle.release()
        return mma_o_consumer, sum_consumer

    @cute.jit
    def correction_epilog_coalesced(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: Tuple,
        store_args: Tuple,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        """O epilogue with smem-staged coalesced global stores.

        Same math as correction_epilog (final 1/row_sum rescale, o_dtype
        conversion), but the tile is staged through swizzled smem and written
        to gmem with 128-bit row-wise coalesced stores (PackGQA row-pointer
        predication) instead of per-element TMEM-fragment autovec stores,
        which were 2-byte-per-sector uncoalesced.
        """
        (seqlen_q, scale_output) = value_args
        (sum_consumer, sSum) = sum_args
        (mma_o_consumer, tOtO_staged) = o_args
        (mO_req, m_blk_idx, sO_epi, gmem_tiled_copy_O, pv_thr_mma) = store_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
        scale = scale_output / row_sum if not row_sum_is_zero_or_nan else 0.0
        o_handle = mma_o_consumer.wait_and_advance()

        num_epi_threads = self.threads_per_warp * len(self.correction_warp_ids)
        corr_tile_size = 8 * 32 // self.o_dtype.width
        mma_tiler_mn = (self.epi_tile[0], self.epi_tile[1])
        tOcO = pv_thr_mma.partition_C(cute.make_identity_tensor(mma_tiler_mn))
        packgqa_op = PackGQA(
            self.epi_tile[0],
            self.epi_tile[1],
            False,
            self.qhead_per_kvhead,
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(thread_idx)
        for iter in cutlass.range(self.iterations_pv):
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.epi_tile[0], corr_tile_size)))
            tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.epi_tile[0], corr_tile_size)))
            tOsO = pv_thr_mma.get_slice(0).partition_C(sO_epi)
            tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.epi_tile[0], corr_tile_size)))
            epi_subtile = (self.epi_tile[0], corr_tile_size)
            tmem_copy_atom = sm100_utils.get_tmem_load_op(
                self.pv_mma_tiler,
                self.o_layout,
                self.o_dtype,
                self.pv_acc_dtype,
                epi_subtile,
                use_2cta_instrs=True,
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            smem_copy_atom = sm100_utils.get_smem_store_op(
                self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
            )
            tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)
            tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
            tOsO_s2r = quack_copy_utils.partition_D_position_independent(
                thr_tmem_load, tOsO_i[(None, None), None]
            )
            tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
            for i in cutlass.range_constexpr(self.epi_tile[1] // corr_tile_size):
                tOrO_frg = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
                cute.copy(tiled_tmem_load, tOtO_t2r[None, 0, 0, i], tOrO_frg)
                for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                    tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                        (tOrO_frg[j], tOrO_frg[j + 1]),
                        (scale, scale),
                    )
                quack_copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_s2r[None, 0, 0, i])
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=2, number_of_threads=num_epi_threads)
            if cutlass.const_expr(self.early_o_release):
                if iter == self.iterations_pv - 1:
                    cute.arch.fence_view_async_tmem_load()
                    o_handle.release()
            tOsO_r = gmem_thr_copy_O.partition_S(sO_epi)
            tOrO = cute.make_fragment_like(tOsO_r, self.o_dtype)
            cute.autovec_copy(tOsO_r, tOrO)
            mO_d = cute.make_tensor(
                mO_req.iterator + iter * self.epi_tile[1], mO_req.layout
            )
            packgqa_op.store_O(
                mO_d, tOrO, gmem_tiled_copy_O, thread_idx, m_blk_idx, seqlen_q
            )
            cute.arch.barrier(barrier_id=3, number_of_threads=num_epi_threads)
        if cutlass.const_expr(not self.early_o_release):
            o_handle.release()
        return mma_o_consumer, sum_consumer

    @cute.jit
    def store_sum_max(
        self,
        row_max,
        mLSE,
        row_sum,
        sSum,
        sum_producer,
        current_block_coord,
        seqlen_q,
        cum_seqlen_q,
        cuseqlen_q,
        scale_softmax,
        split_idx,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_producer.acquire_and_advance()
        sSum[thread_idx] = row_sum
        cute.arch.fence_view_async_shared()
        sum_handle.commit()
        row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum

        if cutlass.const_expr(mLSE is not None):
            lse_value = (
                scale_softmax * row_max
                + cute.math.log(row_sum, fastmath=True)
                - Float32(self.softmax_max_offset) * math.log(2.0)
                if not row_sum_is_zero_or_nan
                else -Float32.inf
            )
            if cutlass.const_expr(self.pack_gqa and self.is_split_kv):
                # Packed rows: row r of this CTA's tile maps to
                # (head = r % h_r, token = r // h_r); the partial LSE is stored
                # unpacked per (token, head) for the combine kernel.
                h_r = self.qhead_per_kvhead
                packed_row = current_block_coord[0] * self.cta_tiler[0] + thread_idx
                if cute.elem_less(packed_row, seqlen_q * h_r):
                    h_r_idx = packed_row % h_r
                    tok = packed_row // h_r + cuseqlen_q
                    kv_head = current_block_coord[2][0]
                    mLSE[tok, ((h_r_idx, kv_head), Int32(0)), split_idx] = lse_value
            else:
                q_idx = current_block_coord[0] * self.cta_tiler[0] + tidx
                hb_idx = (
                    (current_block_coord[2][0], Int32(0))
                    if cutlass.const_expr(cum_seqlen_q is not None)
                    else current_block_coord[2]
                )
                if cute.elem_less(q_idx, seqlen_q):
                    global_q_idx = (
                        q_idx + cuseqlen_q if cutlass.const_expr(cum_seqlen_q is not None) else q_idx
                    )
                    mLSE[global_q_idx, hb_idx] = lse_value
        return sum_producer
