# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Literal, Type, cast
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    NamedBarrier as nbar,
)

import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import (
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int32,
    Int64,
    Optional,
    Union,
)

from .gqa_decode import (
    # Kernel invariants
    mma_modes,
    mma_dice,
    warp_threads,
    warpgroup_warps,
    warpgroup_threads,
    max_reduction_iters,
    # Math helpers
    min_f32,
    log2_e,
    exp2,
    warp_fmax,
    smem_fmax,
    # Kernel code
    GroupedQueryAttentionDecode as GqaDecode,
)
from .mask import AttentionMask

# Math helpers
warp_or = partial(cute.arch.warp_redux_sync, kind="or")  # skip predicate reduction

# Debug helpers
debug_blasst = False
gmem_add = partial(cute.arch.atomic_add, sem="relaxed", scope="gpu")  # count skips


class GroupedQueryAttentionDecodePaged:
    def __init__(
        self,
        page_size,
        headdim,
        grouped_head_tile,
        prediction_tile=1,
        sequence_tile=256,
        reduction_mode: Literal["kernel", "atomic", "none"] = "kernel",
        softmax_warpgroups=1,
        compact_q4_splits=False,
        table_stride=None,
    ):
        """
        Parameters
        ----------
        page_size
            Tokens per page.
        headdim
            Head dimension.
        grouped_head_tile
            Grouped heads per threadblock (GQA packing factor).
        prediction_tile
            Predicted tokens per threadblock.
        sequence_tile
            KV tokens per threadblock per loop iteration.
        reduction_mode
            Split-K reduction algorithm:
              - ``"kernel"``: deterministic kernel reduction with partial result workspace.
              - ``"atomic"``: cluster reduction with atomic adds, no workspace.
              - ``"none"``: no split-K, flash decoding disabled.
        softmax_warpgroups
            Number of softmax warpgroups (1 or 2).
        table_stride
            Compile-time page-table row stride (``table_offsets[b] ==
            b * table_stride``). When set, the kernel derives each batch's
            page-table start without a gmem offset fetch, which lets the
            first-tile page-table prefetch issue before the sequence-length
            load returns. ``None`` keeps the runtime ``table_offsets`` path.
        """
        self.headdim = headdim
        self.grouped_head_tile = grouped_head_tile
        self.page_size = page_size
        self.prediction_tile = prediction_tile
        self.sequence_tile = sequence_tile
        self.do_kernel_red = reduction_mode == "kernel"
        self.do_atomic_red = reduction_mode == "atomic"
        self.do_none_red = reduction_mode == "none" or reduction_mode is None
        self.softmax_warpgroups = softmax_warpgroups
        self.compact_q4_splits = compact_q4_splits
        self.table_stride = table_stride
        self.threads_per_cta = (2 + softmax_warpgroups) * warpgroup_threads

        assert headdim > 0 and headdim % 64 == 0
        assert grouped_head_tile * prediction_tile in (1, 2, 4, 8, 16, 32)
        assert sequence_tile > 0 and sequence_tile % 128 == 0
        assert page_size in (8, 16, 32, 64)
        assert self.softmax_warpgroups in (1, 2)
        assert self.do_kernel_red ^ self.do_atomic_red ^ self.do_none_red

    def can_implement(
        self,
        kv_splits,
        qo_shape,
        kv_shape,
        qkv_dtype,
        o_dtype,
        mask_config,
        threshold_scale_factor,
    ):
        GqaDecode.can_implement(
            self, kv_splits, qo_shape, kv_shape, qkv_dtype, o_dtype, mask_config
        )

        if threshold_scale_factor is not None and not threshold_scale_factor > 0:
            raise ValueError(
                f"threshold_scale_factor must be None or > 0, "
                f"got {threshold_scale_factor}"
            )

    ##############################
    # Decode Kernel launch
    ##############################
    @cute.jit
    def __call__(
        self,
        kv_splits: Int32,
        seqlens: Union[cute.Tensor, Int32],
        table_offsets: Optional[cute.Tensor],
        page_table: cute.Tensor,
        k_bshd: cute.Tensor,
        v_bshd: cute.Tensor,
        q_bshd: cute.Tensor,
        o_bshd: cute.Tensor,
        l_bsh: Optional[cute.Tensor],
        m_bsh: Optional[cute.Tensor],
        o_partial_bshd: Optional[cute.Tensor],
        l_partial_bsh: Optional[cute.Tensor],
        m_partial_bsh: Optional[cute.Tensor],
        sink_h: Optional[cute.Tensor],
        mask_config,  # duck-typed AttentionMask, TVM FFI conversion breaks if we annotate with base class for now
        scale_s: Float32,
        scale_o: Float32,
        threshold_scale_factor: Optional[Float32],
        stream: cuda.CUstream,
        enable_pdl: bool = True,
    ):
        """
        Parameters
        ----------
        kv_splits
            Threadblocks per sequence (flash decoding).
        seqlens
            Per-batch sequence lengths.
        table_offsets
            Starting offset into ``page_table`` for each batch.
        page_table
            Logical → virtual page index mapping.
        k_bshd, v_bshd
            Paged K/V tensors of shape ``(page_count, page_size, h_k, d)``.
        q_bshd
            Q tensor in BSHD logical view (strides can be BHSD)
        o_bshd
            Output tensor. For atomic reduction the kernel zero-initializes
            each cluster's output tile in-kernel (split 0) before the
            reduce-add epilogue, so no host-side zero init is required.
        l_bsh
            Log-sum-exp output (Float32, log2 base). May be None.
        m_bsh
            ``colmax_s`` running accumulator. Must be -inf initialized
            (kernel-red workspace).
        o_partial_bshd
            Partial O per kv split (kernel-red workspace).
        l_partial_bsh
            Partial ``colsum_p`` per kv split (kernel-red workspace).
        m_partial_bsh
            Partial ``colmax_s`` per kv split (kernel-red workspace).
        sink_h
            Pre-scaled attention sink logits per head
        scale_s
            Softmax scale.
        scale_o
            Output scale, applied in the reduction epilogue.
        mask_config
            Attention logit masking configuration.
        threshold_scale_factor
            BLASST per-batch skip-softmax threshold scale factor. The kernel
            divides this by each batch's KV seqlen to obtain the effective
            per-request threshold. ``None`` disables BLASST.
        stream
            CUDA stream to launch on.
        enable_pdl
            Programmatic Dependent Launch. Runtime-dynamic — no recompile on
            toggle.
        """
        ##############################
        # TiledMma creation
        ##############################
        mma_dtype = q_bshd.dtype
        acc_dtype = Float32
        assert k_bshd.dtype == v_bshd.dtype == mma_dtype

        # Block tile sets the granularity at which threadblocks consume work
        blk_tile_s = self.sequence_tile
        blk_tile_h = self.grouped_head_tile
        blk_tile_p = self.prediction_tile
        blk_tile_d = self.headdim
        blk_tile_shpd = (blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d)

        # MMA tile sets the granularity at which TMAs + MMAs are staged
        mma_tile_m = 128
        mma_tile_k = 128 * 8 // mma_dtype.width
        # N-major 8b B in smem requires N multiple of 16
        min_mma_tile_n = 16 if mma_dtype.width == 8 else 8
        blk_tile_n = blk_tile_h * blk_tile_p  # linearized tiler
        mma_tile_n = max(min_mma_tile_n, blk_tile_n)
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)

        # MMA tiles per block tile
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_dm = math.ceil(blk_tile_d / mma_tile_m)
        tiles_dk = math.ceil(blk_tile_d / mma_tile_k)
        pages_s = blk_tile_s // self.page_size
        assert blk_tile_s % mma_tile_m == 0
        assert mma_tile_n % blk_tile_n == 0

        # GEMM1: (S_K, H_R, D, (H_K, B))
        tiled_mma_kq = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,  # K
            OperandMajorMode.K,  # Q
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
        )

        # GEMM2: (D, H_R, S_K, (H_K, B))
        tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.MN,  # V
            OperandMajorMode.MN,  # P
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
        )

        ##############################
        # Calculate stage counts
        ##############################
        # Fixed stage counts
        self.pt_stages = pt_stages = 4  # smem page table buffer
        self.sp_stages = sp_stages = 4  # smem skip predicates
        self.p_stages = p_stages = 4  # smem P (BMM2 B)
        self.o_stages = o_stages = 2  # tmem O (BMM2 C)

        # Calculate tmem alloc
        tmem_capacity_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        tmem_s_stage_cols = tiles_sm * mma_tile_n
        tmem_alloc_cols = mma_tile_n * o_stages  # per-thread colsum
        tmem_alloc_cols += tiles_dm * mma_tile_n * o_stages  # O
        max_s_stages = (tmem_capacity_cols - tmem_alloc_cols) // tmem_s_stage_cols
        self.s_stages = s_stages = min(max_s_stages, p_stages)

        tmem_alloc_cols += tmem_s_stage_cols * s_stages  # S
        tmem_alloc_cols = 2 ** math.ceil(math.log2(tmem_alloc_cols))  # po2
        self.tmem_alloc_cols = tmem_alloc_cols
        assert tmem_alloc_cols <= tmem_capacity_cols

        # Calculate smem alloc
        smem_alloc_bits = 0
        mbarrier_bits = Int64.width
        pipe_stage_bits = mbarrier_bits * 2  # producer + consumer
        mk_stage_bits = mma_tile_m * mma_tile_k * mma_dtype.width
        nk_stage_bits = mma_tile_n * mma_tile_k * mma_dtype.width
        mn_stage_bits = mma_tile_m * mma_tile_n * mma_dtype.width
        # tmem ptr
        smem_alloc_bits += Int32.width
        # seqlen, table offset
        is_varlen = isinstance(seqlens, cute.Tensor)
        smem_alloc_bits += (Int32.width * 2) if is_varlen else 0
        # page table
        smem_alloc_bits += pt_stages * pages_s * Int32.width
        # skip predicates
        if cutlass.const_expr(threshold_scale_factor is not None):
            smem_alloc_bits += sp_stages * (Int32.width + pipe_stage_bits)
        # colmax + colsum
        smem_alloc_bits += blk_tile_n * acc_dtype.width
        smem_alloc_bits += blk_tile_n * warpgroup_warps * acc_dtype.width
        if cutlass.const_expr(self.do_atomic_red):
            smem_alloc_bits += max_reduction_iters * blk_tile_n * acc_dtype.width * 2
            smem_alloc_bits += max_reduction_iters * mbarrier_bits * 2
            # zero_done mbarrier + its 4 x f32 dsmem signal slot
            smem_alloc_bits += mbarrier_bits
            smem_alloc_bits += 4 * acc_dtype.width
        # Q, S, P, O
        smem_alloc_bits += tiles_dk * nk_stage_bits + mbarrier_bits  # 1 mbar for Q
        smem_alloc_bits += s_stages * pipe_stage_bits  # s in tmem
        smem_alloc_bits += p_stages * (2 * tiles_sm * mn_stage_bits + pipe_stage_bits)
        smem_alloc_bits += o_stages * pipe_stage_bits  # o in tmem
        alignment_bits = 1024 - (smem_alloc_bits % 1024)
        # K, V
        smem_capacity_bits = utils.get_smem_capacity_in_bytes("sm_100") * 8
        remaining_bits = smem_capacity_bits - smem_alloc_bits - alignment_bits
        kv_stages = remaining_bits // mk_stage_bits
        kv_stages -= 1 if kv_stages * pipe_stage_bits > alignment_bits else 0

        ##############################
        # TMA creation
        ##############################
        h_k = k_bshd.shape[2]
        o_bshd_ = o_partial_bshd if self.do_kernel_red else o_bshd

        # Reorder and group modes for GEMM
        # ((h_g, s_q), d, (h_k, b))
        mQ_nkl = GqaDecode.gemm_view(GqaDecode.gqa_pack(q_bshd, h_k), True)
        mK_mkl = GqaDecode.gemm_view(k_bshd, True)  # (page_size, d, (h_k, page_count))
        mV_mkl = GqaDecode.gemm_view(v_bshd, False)  # (d, page_size, (h_k, page_count))
        # (d, (h_g, s_q), (h_k, b_partial))
        mO_mnl = GqaDecode.gemm_view(GqaDecode.gqa_pack(o_bshd_, h_k), False)

        # ((MMA_N, MMA_K), #MMA_N, #MMA_K, q_stages)
        smem_layout_q = sm100_utils.make_smem_layout_b(
            tiled_mma_kq, mma_tile_mnk, mma_dtype, tiles_dk
        )

        # ((MMA_M, MMA_K), #MMA_M, #MMA_K, kv_stages)
        smem_layout_k_mma = sm100_utils.make_smem_layout_a(
            tiled_mma_kq, mma_tile_mnk, mma_dtype, kv_stages
        )
        smem_layout_v_mma = sm100_utils.make_smem_layout_a(
            tiled_mma_vp, mma_tile_mnk, mma_dtype, kv_stages
        )

        # (MMA_TILE_M, MMA_TILE_K, kv_stages)
        smem_layout_k_mk = cute.composition(
            smem_layout_k_mma,
            cute.make_layout((mma_tile_m, mma_tile_k, kv_stages)),
        )
        smem_layout_v_mk = cute.composition(
            smem_layout_v_mma,
            cute.make_layout((mma_tile_m, mma_tile_k, kv_stages)),
        )

        # ((PAGE, MMA_TILE_K), #PAGE_M, 1, kv_stages)
        smem_layout_k_tma = cute.tiled_divide(
            smem_layout_k_mk, (self.page_size, mma_tile_k)
        )
        # (TMA, #PAGE_M, kv_stages)
        smem_layout_k_tma = cute.select(smem_layout_k_tma, [0, 1, 3])
        # ((MMA_TILE_M, PAGE), 1, #PAGE_K, kv_stages)
        smem_layout_v_tma = cute.tiled_divide(
            smem_layout_v_mk, (mma_tile_m, self.page_size)
        )
        # (TMA, #PAGE_K, kv_stages)
        smem_layout_v_tma = cute.select(smem_layout_v_tma, [0, 2, 3])

        o_smem_dtype = mO_mnl.dtype
        smem_layout_atom_o = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, o_smem_dtype
        )
        smem_layout_o = cute.tile_to_shape(
            smem_layout_atom_o, (max(blk_tile_d, mma_tile_m), mma_tile_n), order=(1, 0)
        )
        smem_layout_o = cute.flat_divide(smem_layout_o, (mma_tile_m, mma_tile_n))

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = (
            cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp()
            if self.do_atomic_red
            else cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        )

        # Construct multimode gmem tiler
        tma_tile_n = (blk_tile_h, mma_tile_n // blk_tile_h)
        tma_tile_mnk = (mma_tile_m, tma_tile_n, mma_tile_k)
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ_nkl,
            cute.select(smem_layout_q, mma_modes),
            tma_tile_mnk,
            tiled_mma_kq,
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op, mK_mkl, smem_layout_k_tma[0], (self.page_size, mma_tile_k)
        )
        tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op, mV_mkl, smem_layout_v_tma[0], (mma_tile_m, self.page_size)
        )
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            mO_mnl,
            cute.select(smem_layout_o, mode=[0, 1]),
            tma_tile_mnk[:2],
        )

        # GEMM view for LSE output
        # ((h_g, s_q), (h_k, b))
        mL_nl = None if l_bsh is None else GqaDecode.gemm_view_bsh(l_bsh, h_k)
        assert l_bsh is None or l_bsh.dtype == acc_dtype

        # GEMM views for workspace tensors
        mM_partial_nl = mL_partial_nl = None
        if cutlass.const_expr(self.do_kernel_red):
            assert (
                m_bsh.dtype
                == m_partial_bsh.dtype
                == l_partial_bsh.dtype
                == o_partial_bshd.dtype
                == acc_dtype
            )

            # ((h_g, s_q), (h_k, b), kv_splits)
            # m_bsh is unused by the kernels: the reduction kernel derives the
            # final colmax from the per-split m_partial values directly, so no
            # -inf pre-fill / gmem atomic-max is needed.
            mM_partial_nl = GqaDecode.gemm_view_bsh(m_partial_bsh, h_k)
            mL_partial_nl = GqaDecode.gemm_view_bsh(l_partial_bsh, h_k)

        if cutlass.const_expr(sink_h is not None):
            assert sink_h.dtype == acc_dtype

            h_g = sink_h.shape[0] // h_k
            mSink = cute.make_tensor(
                sink_h.iterator,
                cute.make_layout((h_g, h_k), stride=(1, h_g)),
            )
        else:
            mSink = None

        ##############################
        # Launch kernel(s)
        ##############################
        scale_s_log2_e = scale_s * log2_e

        enable_blasst = threshold_scale_factor is not None
        log2_threshold_scale_factor = (
            Float32(cute.math.log2(threshold_scale_factor)) if enable_blasst else None
        )

        n_tiles = cute.ceil_div(mQ_nkl.shape[0], (blk_tile_h, blk_tile_p))
        grid_y = cute.size(n_tiles)
        grid_z = cute.size(mQ_nkl.shape[2])  # l tiles
        grid_x = 1 if self.do_none_red else kv_splits
        grid = (grid_x, grid_y, grid_z)
        cluster_x = kv_splits if self.do_atomic_red else 1

        self.decode(
            # MMA
            blk_tile_shpd,
            mma_tile_mnk,
            tiled_mma_kq,
            tiled_mma_vp,
            mma_dtype,
            o_smem_dtype,
            # Page Table
            seqlens.iterator if is_varlen else seqlens,
            table_offsets.iterator if is_varlen else None,
            page_table.iterator,
            # K
            smem_layout_k_mma,
            smem_layout_k_tma,
            tma_atom_k,
            tma_tensor_k,
            # V
            smem_layout_v_mma,
            smem_layout_v_tma,
            tma_atom_v,
            tma_tensor_v,
            # Q
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            # O
            smem_layout_o,
            tma_atom_o,
            tma_tensor_o,
            # Plain (non-TMA) O view for the atomic-mode in-kernel zeroing
            mO_mnl if self.do_atomic_red else None,
            # Rest
            mL_nl,
            mL_partial_nl,
            mM_partial_nl,
            mSink,
            mask_config,
            scale_s_log2_e,
            scale_o,
            log2_threshold_scale_factor,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[cluster_x, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=enable_pdl,
        )

        if cutlass.const_expr(self.do_kernel_red):
            GqaDecode.launch_reduction(
                self.headdim,
                o_bshd,
                l_bsh,
                o_partial_bshd,
                l_partial_bsh,
                m_partial_bsh,
                sink_h,
                scale_o,
                self.prediction_tile == 1 or not self.compact_q4_splits,
                stream,
                enable_pdl,
            )

    @cute.kernel
    def decode(
        self,
        # MMA
        blk_tile_shpd: cute.Tile,
        mma_tile_mnk: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        mma_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        # Page Table
        seqlens_iter: Union[cute.Pointer, Int32],
        table_offsets_iter: Optional[cute.Pointer],
        page_table_iter: cute.Pointer,
        # K
        smem_layout_k_mma: cute.ComposedLayout,
        smem_layout_k_tma: cute.ComposedLayout,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        # V
        smem_layout_v_mma: cute.ComposedLayout,
        smem_layout_v_tma: cute.ComposedLayout,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        # Q
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        # O
        smem_layout_o: cute.ComposedLayout,
        tma_atom_o: cute.CopyAtom,
        mO: cute.Tensor,
        # Plain-address (non-TMA) O view, atomic mode only: used for the
        # in-kernel split-0 output zeroing that replaces the host-side
        # out.zero_() launch. None in kernel/none reduction modes.
        mO_plain: Optional[cute.Tensor],
        # Rest
        mL: Optional[cute.Tensor],  # LSE output (Float32, log2 base)
        mL_partial: Optional[cute.Tensor],
        mM_partial: Optional[cute.Tensor],
        mSink: Optional[cute.Tensor],  # (h_g, h_k)
        mask_config: AttentionMask,
        scale_s_log2_e: Float32,
        scale_o: Float32,
        log2_threshold_scale_factor: Optional[Float32],
    ):
        ##############################
        # Static variables
        ##############################
        # Smem alloc helper
        svector_align = 16
        stensor_align = 128
        smem = utils.SmemAllocator()

        # No multicast
        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))  # vmnk

        # Alias types
        q_dtype = k_dtype = mma_dtype
        o_dtype = out_dtype
        acc_dtype = Float32

        # Shapes for MMA tile indexing
        blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d = blk_tile_shpd
        blk_tile_hp = (blk_tile_h, blk_tile_p)  # multimode tiler
        blk_tile_n = blk_tile_h * blk_tile_p  # linearized tiler
        mma_tile_m, mma_tile_n, mma_tile_k = mma_tile_mnk
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_sk = blk_tile_s // mma_tile_k
        tiles_dm = cute.ceil_div(blk_tile_d, mma_tile_m)
        tiles_dk = cute.ceil_div(blk_tile_d, mma_tile_k)
        page_size = self.page_size
        pages_s = blk_tile_s // page_size
        pages_m = mma_tile_m // page_size
        pages_k = mma_tile_k // page_size

        # Static control flow
        do_kernel_red = self.do_kernel_red
        do_atomic_red = self.do_atomic_red
        do_none_red = self.do_none_red
        compact_q4_splits = self.compact_q4_splits
        store_lse = mL is not None
        is_varlen = isinstance(seqlens_iter, cute.Pointer)
        enable_blasst = log2_threshold_scale_factor is not None
        use_sink = mSink is not None

        ##############################
        # Warp specialization
        ##############################
        # Warp assignments
        warpgroup_id = 0
        mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        tma_qk_warp_id = warpgroup_id * warpgroup_warps + 2
        tma_vo_warp_id = warpgroup_id * warpgroup_warps + 3
        reduction_warp_id = mma_kq_warp_id
        warpgroup_id += 1

        softmax_warpgroups = self.softmax_warpgroups
        softmax_warpgroup_ids = tuple(
            range(warpgroup_id, warpgroup_id + softmax_warpgroups)
        )
        warpgroup_id += softmax_warpgroups
        assert softmax_warpgroups in (1, 2)

        correction_warpgroup_id = warpgroup_id
        warpgroup_id += 1
        assert self.threads_per_cta == warpgroup_id * warpgroup_threads

        # Register allocations
        #
        # Warp-group register reconfiguration for wide-N tiles (blk_tile_n
        # > 16, i.e. prediction_tile > 1 at grouped_head_tile 16; same gate
        # as the canonical non-paged decoder). The single softmax warpgroup
        # of those kernels keeps the uniform launch allocation while the
        # mma/tma warpgroup decreases to 64, funding the correction
        # warpgroup's increase to its full 256-register budget. Without
        # reconfiguration every warpgroup is pinned to the uniform launch
        # allocation (168 at 384 threads) and the correction warpgroup
        # spills its O/colsum fragments to local memory inside the
        # per-iteration s-loop. Measured on sm_103 (b11 q4 ~1.31M-token
        # aggregate, NCU --set full): local traffic = 99.7% of all L1TEX
        # sectors (2.9M ld + 2.9M st sectors, 690k spill requests per
        # launch), L1TEX 92% vs DRAM 51% while the y-twin KV double-read
        # is already fully L2-deduped (dram read == single-read floor).
        # With reconfiguration: 0 spills, bitwise-identical outputs,
        # +1.2..+5.4% hot device time across q4 kernel-mode cells
        # (mandated (a)/(b)/(c) classes + mid-band + guards), q1/atomic
        # cells unchanged.
        # Pool arithmetic (launch allocation L=168, 384 threads, CTA pool
        # L*384 = 64512 of the 64K register file): decrease(64) frees
        # (L-64)*128 = 13312; increase(256) claims (256-L)*128 = 11264,
        # so it always fits. A softmax INCREASE would need the full
        # (64+192+256)*128 = 65536 > pool and deadlocks setmaxnreg.inc
        # (measured: kernel hangs), so softmax_regs must never exceed the
        # launch allocation under this gate.
        use_reg_reconfig = blk_tile_n > 16
        max_sw_regs_per_wg_thread = 256  # CUDA limitation
        max_hw_regs_per_wg_thread = 64 * 1024 // warpgroup_threads  # 64K regs per SM
        mma_tma_regs = 64
        # Reconfig (single-softmax-warpgroup) kernels keep softmax at the
        # launch budget: decreasing it to 120 spills the 64-wide q4 S/P
        # fragment chains (measured -28% at b1 q4 k32), and increasing it
        # exceeds the CTA pool (above). The 120/208 split below only
        # parameterizes the (disabled) narrow-N path.
        softmax_regs = 168 if use_reg_reconfig else 120
        correction_regs = min(
            max_sw_regs_per_wg_thread,
            max_hw_regs_per_wg_thread
            - mma_tma_regs
            - softmax_regs * softmax_warpgroups,
        )
        assert (
            mma_tma_regs + softmax_regs * softmax_warpgroups + correction_regs
        ) <= max_hw_regs_per_wg_thread

        # Read thread indices
        kv_splits, tiles_hp, tiles_hb = cute.arch.grid_dim()
        kv_split_idx, coord_hp, coord_hb = cute.arch.block_idx()
        if cutlass.const_expr(do_none_red):
            kv_splits, kv_split_idx = (1, 0)
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 1  # warp 0 does all pipeline inits for now

        # Unpack multimodes
        grouped_heads, prediction = mQ.shape[0]
        heads_k, batches = tiles_hb = mQ.shape[2]
        tiles_hp = cute.ceil_div(mQ.shape[0], blk_tile_hp)
        coord_hb = cute.idx2crd(coord_hb, tiles_hb)
        coord_hp = cute.idx2crd(coord_hp, tiles_hp)
        coord_hg, coord_p = coord_hp
        coord_hk, coord_b = coord_hb

        ##############################
        # Prefetch Seqlen
        ##############################
        cpasync_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), Int32, num_bits_per_copy=32
        )
        if cutlass.const_expr(is_varlen):
            scalar_layout = cute.make_layout(1)
            seqlen_smem = smem.allocate_tensor(Int32, scalar_layout)
            table_offset_smem = smem.allocate_tensor(Int32, scalar_layout)
            if warp_idx == init_warp:
                seqlen_gmem = cute.make_tensor(seqlens_iter + coord_b, scalar_layout)
                cute.arch.griddepcontrol_wait()
                with cute.arch.elect_one():
                    cute.copy(cpasync_atom, seqlen_gmem, seqlen_smem)
                    if cutlass.const_expr(self.table_stride is None):
                        table_offset_gmem = cute.make_tensor(
                            table_offsets_iter + coord_b, scalar_layout
                        )
                        cute.copy(
                            cpasync_atom, table_offset_gmem, table_offset_smem
                        )
                cute.arch.cp_async_commit_group()
                # The group wait is deferred to just before the CTA-wide
                # sync so the seqlen gmem round trip overlaps descriptor
                # prefetch, tmem alloc, pipeline init and (in atomic mode)
                # the output zeroing handoff instead of stalling here.
            init_warp += 1

        ##############################
        # Prefetch TMA descriptor
        ##############################
        if warp_idx == init_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)
        init_warp += 1

        ##############################
        # Tmem Allocation
        ##############################
        tmem_alloc_cols = self.tmem_alloc_cols
        tmem_ptr_smem_ptr = smem.allocate_array(Int32)
        if warp_idx == init_warp:
            cute.arch.alloc_tmem(tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1

        ##############################
        # Pipeline Allocation + Init
        ##############################
        softmax_threads = warpgroup_threads
        correction_threads = warpgroup_threads
        reduction_threads = warp_threads
        mma_threads = warp_threads
        tma_threads = warp_threads
        tma_order_k_nbar = nbar(1, tma_threads + tma_threads)
        tma_order_v_nbar = nbar(2, tma_threads + tma_threads)
        mma_order_kq_nbar = nbar(3, mma_threads + mma_threads)
        mma_order_vp_nbar = nbar(4, mma_threads + mma_threads)
        sM_producer_nbar = nbar(5, softmax_threads + correction_threads)
        sM_consumer_nbar = nbar(6, softmax_threads + correction_threads)
        tL_producer_nbar = nbar(7, softmax_threads + correction_threads)
        tL_consumer_nbar = nbar(9, softmax_threads + correction_threads)
        sM_final_nbar = nbar(11, correction_threads + reduction_threads)
        sL_final_nbar = nbar(12, correction_threads + reduction_threads)
        sO_final_nbar = nbar(13, correction_threads + tma_threads)
        # dual softmax and blasst are mutually exclusive
        sSP_producer_nbar = nbar(14, softmax_threads)
        sM_mutex_nbar = nbar(14, softmax_threads * softmax_warpgroups)

        # named barrier stage helper
        def with_phase(nbar_, phase):
            return nbar(nbar_.barrier_id + phase, nbar_.num_threads)

        # Alias thread cooperatives
        thr_cg = lambda t: CooperativeGroup(Agent.Thread, t)
        elect_one_cooperative = thr_cg(1)
        warpgroup_cooperative = thr_cg(warpgroup_threads)
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        softmax_group = warpgroup_cooperative
        correction_group = warpgroup_cooperative

        # Initialize cluster colmax + colsum mbar (even if this split exits early)
        # plus one extra "zero_done" mbar (index max_reduction_iters * 2) that
        # releases the reduce-add epilogue once split 0's overlapped output
        # zeroing is device-visible.
        if cutlass.const_expr(do_atomic_red):
            reduction_mbars_ptr = smem.allocate_array(
                Int64, max_reduction_iters * 2 + 1
            )
            zero_done_mbar_ptr = reduction_mbars_ptr + max_reduction_iters * 2
            zero_done_slot = smem.allocate_tensor(
                acc_dtype, cute.make_layout(4), svector_align
            )
            if warp_idx == init_warp:
                if lane_idx < max_reduction_iters * 2:
                    mbar_ptr = reduction_mbars_ptr + lane_idx
                    arrive_count = 1
                    expect_tx_bytes = blk_tile_n * acc_dtype.width // 8
                    cute.arch.mbarrier_init(mbar_ptr, arrive_count)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expect_tx_bytes)
                elif lane_idx == max_reduction_iters * 2:
                    cute.arch.mbarrier_init(zero_done_mbar_ptr, 1)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        zero_done_mbar_ptr, 4 * acc_dtype.width // 8
                    )
            init_warp += 1
            # Output zeroing moved off this critical path: split 0's
            # correction warpgroup zeroes the tile during mainloop slack and
            # releases every split's reduce-add through zero_done_mbar. The
            # cluster rendezvous here only publishes the mbarrier inits.
            cute.arch.cluster_arrive_relaxed()

        # Initialize Q load mbarrier
        q_load_mbar = smem.allocate_array(Int64, 1)
        if warp_idx == init_warp:
            expect_tx_bytes = cute.size_in_bytes(q_dtype, smem_layout_q)
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_load_mbar, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.mbarrier_arrive_and_expect_tx(q_load_mbar, expect_tx_bytes)
        init_warp += 1

        # Initialize pipelines
        kv_stages = smem_layout_k_tma.shape[-1]
        kv_stage_bytes = mma_tile_m * mma_tile_k * k_dtype.width // 8
        kv_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
        kv_producer, kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=kv_stages,
            producer_group=tma_group,
            consumer_group=mma_group,
            tx_count=kv_stage_bytes,
            barrier_storage=kv_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()

        s_stages = self.s_stages
        s_pipeline_ptr = smem.allocate_array(Int64, s_stages * 2)
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.s_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=s_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        p_stages = self.p_stages
        p_pipeline_ptr = smem.allocate_array(Int64, p_stages * 2)
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_stages,
            producer_group=softmax_group,
            consumer_group=mma_group,
            barrier_storage=p_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        o_stages = self.o_stages
        o_pipeline_ptr = smem.allocate_array(Int64, o_stages * 2)
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=o_stages,
            producer_group=mma_group,
            consumer_group=correction_group,
            barrier_storage=o_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        if cutlass.const_expr(enable_blasst):
            skip_group = thr_cg(correction_threads + (mma_threads + tma_threads) * 2)

            sp_stages = self.sp_stages
            sp_pipeline_ptr = smem.allocate_array(Int64, sp_stages * 2)
            sp_producer, sp_consumer = pipeline.PipelineAsync.create(
                num_stages=sp_stages,
                producer_group=softmax_group,
                consumer_group=skip_group,
                barrier_storage=sp_pipeline_ptr,
                defer_sync=True,
            ).make_participants()

            # SP - skip predicates
            sSP_i32 = smem.allocate_tensor(Int32, cute.make_layout(sp_stages))
            sSP_i8 = cute.make_tensor(
                cute.recast_ptr(sSP_i32.iterator, dtype=Int8),
                cute.make_layout((warpgroup_warps, sp_stages)),
            )

        ##############################
        # Smem Tensor Allocation
        ##############################
        # Threadblock slice
        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)

        # Q, K, V
        tAsK = smem.allocate_tensor(
            k_dtype, smem_layout_k_mma.outer, stensor_align, smem_layout_k_mma.inner
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        tAsV = cute.make_tensor(
            tAsK.iterator, smem_layout_v_mma.outer
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner
        )  # (MMA, #MMA_N, #MMA_K, q_stages)

        # S
        # (MMA_MN, #MMA_M=1, #MMA_N=1, #TILE_SM, s_stages)
        tCtS_shape = tiled_mma_kq.partition_shape_C(
            (mma_tile_m, mma_tile_n, tiles_sm, s_stages)
        )
        tCtS = thrblk_mma_kq.make_fragment_C(tCtS_shape)

        # P - Treat MN C tile of BMM0 as NM B tile of BMM1
        # (MMA_NK, #MMA_N=1, #MMA_K=TILE_S/MMA_K, p_stages)
        blk_tile_nm = (None, mma_tile_n, mma_tile_m * tiles_sm)
        tBsP_nm_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_vp, blk_tile_nm, mma_dtype, p_stages
        )
        tBsP_nm = smem.allocate_tensor(
            mma_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner
        )
        tBsP_residual_nm = smem.allocate_tensor(
            mma_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner
        )

        # Tile for NK B tile iteration
        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B(
            (mma_tile_n, mma_tile_k)
        )  # (MMA_NK, #MMA_N=1, #MMA_K=MMA_TILE_K/MMA_K, #TILE_SK=TILE_S/MMA_TILE_K, p_stages)
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))
        tBsP_residual_nk = cute.local_tile(
            tBsP_residual_nm, tBsP_nk_tile, (0, 0, None, None)
        )

        # Reshape NM B tile of BMM1 to become MN C tile of BMM0
        # (MMA_NK, #MMA_N, #MMA_K=TILE_S/MMA_K, p_stages) ->
        # (MMA_MN, #MMA_M, #MMA_N, #TILE_SM, p_stages)
        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4, 5))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)
        tCsP_residual = cute.composition(tBsP_residual_nm, tCsP_tile)

        # O
        # Reuse KV smem for O TMA store
        sO_iterator = cute.recast_ptr(tAsK.iterator, smem_layout_o.inner, dtype=o_dtype)
        # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN)
        sO_mma = cute.make_tensor(sO_iterator, smem_layout_o.outer)
        # (MMA, #MMA_M, #MMA_N, #TILE_DM, #TILE_HN=1)
        tCsO = thrblk_mma_vp.partition_C(sO_mma)
        tCsO = tCsO[mma_dice + (None, 0)]
        # (MMA, #MMA_M, #MMA_N, #TILE_DM, o_stages)
        tCtO = thrblk_mma_vp.make_fragment_C((*tCsO.shape, o_stages))

        # PT - Page Table lookup buffer
        pt_stages = self.pt_stages
        sPT_layout = cute.make_layout((pages_s, pt_stages))
        sPT = smem.allocate_tensor(Int32, sPT_layout, svector_align)

        # M - colmax
        sM_layout = cute.make_layout(blk_tile_n)
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n
        if warp_idx == init_warp:
            if lane_store_max:
                sM[lane_idx] = -Float32.inf
        init_warp += 1

        # L - colsum
        sL_layout = cute.make_layout((blk_tile_n, warpgroup_warps))
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        if warp_idx == init_warp:
            for i in cutlass.range_constexpr(0, cute.size(sL), warp_threads):
                if i + lane_idx < cute.size(sL):
                    sL[i + lane_idx] = Float32(0)
        init_warp += 1

        # Sink
        if cutlass.const_expr(use_sink and not do_kernel_red):
            sSink_layout = cute.make_layout((blk_tile_hp,), stride=((1, 0),))
            sSink = smem.allocate_tensor(acc_dtype, sSink_layout, svector_align)
            gSink = cute.local_tile(mSink, (blk_tile_h,), (coord_hg, coord_hk))
            sSink_lane = cute.local_tile(sSink[((None, 0),)], (1,), (lane_idx,))
            gSink_lane = cute.local_tile(gSink, (1,), (lane_idx,))
            if warp_idx == reduction_warp_id and lane_idx < blk_tile_h:
                cute.copy(cpasync_atom, gSink_lane, sSink_lane)
            init_warp += 1
        else:
            sSink = None

        # per-thread colsum
        # (MMA_MN, #MMA_M=1, #MMA_N=1, o_stages)
        tCtL_shape = tiled_mma_kq.partition_shape_C((mma_tile_m, mma_tile_n, o_stages))
        tCtL = thrblk_mma_kq.make_fragment_C(tCtL_shape)

        # R - cluster reduction buffers for colmax + colsum
        if cutlass.const_expr(do_atomic_red):
            sR_layout = cute.make_layout((blk_tile_n, max_reduction_iters, 2))
            sR = smem.allocate_tensor(acc_dtype, sR_layout, svector_align)

        ##############################
        # Sync
        ##############################
        # Speculative first-tile page-table prefetch: with a compile-time
        # table stride the address does not depend on the seqlen load, so
        # the fetch overlaps that round trip and pipeline init. Bounds:
        # kv_split_idx <= kv_splits-1 <= 63 (host policy cap) and
        # pages_s <= 4 keep the reads inside this batch's block-table row.
        # Entries beyond this request's page_count are only consumed for
        # tiles >= tiles_s (never loaded) or for tail pages whose KV rows
        # the softmax masks to -inf exactly as in the guarded path (the
        # TMA zero-fills OOB page coordinates).
        spec_pt_prefetch = is_varlen and self.table_stride is not None
        if cutlass.const_expr(spec_pt_prefetch):
            if warp_idx == tma_qk_warp_id:
                cute.arch.griddepcontrol_wait()
                spec_pt_gmem = cute.make_tensor(
                    page_table_iter
                    + coord_b * self.table_stride
                    + kv_split_idx * pages_s,
                    cute.make_layout((pages_s,)),
                )
                spec_pt_load = cute.make_tiled_copy(
                    cpasync_atom,
                    cute.make_ordered_layout((pages_s, 1), order=(1, 0)),
                    (pages_s,),
                )
                thr_spec_pt = spec_pt_load.get_slice(lane_idx)
                if lane_idx < pages_s:
                    cute.copy(
                        cpasync_atom,
                        thr_spec_pt.partition_S(spec_pt_gmem)[None, 0],
                        thr_spec_pt.partition_D(sPT)[None, 0, 0],
                    )
                cute.arch.sync_warp()
                cute.arch.cp_async_commit_group()

        # Ensure visibility of cluster mbarriers
        if cutlass.const_expr(do_atomic_red):
            cute.arch.cluster_wait()

        # Complete the deferred seqlen/table-offset fetch and the speculative
        # page-table fetch (the wait is a no-op for warps without pending
        # cp.async groups), then publish everything CTA-wide together with
        # the local mbarrier inits and the tmem allocation.
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        assert init_warp <= (self.threads_per_cta // warp_threads), (
            f"used {init_warp} init warps, {self.threads_per_cta // warp_threads} warps available"
        )

        # Runtime control flow
        if cutlass.const_expr(is_varlen):
            seqlen = seqlen_smem[0]
            page_count = cute.ceil_div(seqlen, page_size)
            if cutlass.const_expr(self.table_stride is not None):
                table_offset = coord_b * self.table_stride
            else:
                table_offset = table_offset_smem[0]
        else:
            seqlen = seqlens_iter
            page_count = cute.ceil_div(seqlen, page_size)
            table_offset = coord_b * page_count
        work_splits = kv_splits
        if cutlass.const_expr(compact_q4_splits):
            lane_min = Int32(131072)
            lane_max = Int32(0)
            if lane_idx < batches:
                lane_len = cute.make_tensor(
                    seqlens_iter + lane_idx, scalar_layout
                ).load()[0]
                lane_min = lane_len
                lane_max = lane_len
            min_len = cute.arch.warp_redux_sync(lane_min, kind="min")
            max_len = cute.arch.warp_redux_sync(lane_max, kind="max")
            if (batches >= 10) & (batches <= 17):
                if kv_splits == 32:
                    work_splits = cute.ceil_div(seqlen * kv_splits, max_len)
                    if work_splits < 4:
                        work_splits = Int32(4)
                elif (kv_splits == 8) & (batches <= 12):
                    if min_len * 4 >= max_len * 3:
                        if batches < 12:
                            work_splits = Int32(6)
                            if batches == 10:
                                work_splits = Int32(7)
                        else:
                            work_splits = cute.ceil_div(seqlen * kv_splits, max_len)
                    elif batches == 10:
                        work_splits = cute.ceil_div(seqlen * kv_splits, max_len)
        tiles_s = cute.ceil_div(seqlen, blk_tile_s)
        iters_s = cute.ceil_div(tiles_s - kv_split_idx, work_splits)
        exit_early = kv_split_idx >= tiles_s or kv_split_idx >= work_splits
        prefetch_iters = min(2, s_stages - 1)  # MMA KQ iters to hide first softmax
        assert pt_stages > prefetch_iters + 1

        ##############################
        # Tmem tensor allocation
        ##############################
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_ptr_smem_ptr)
        tmem_offset = 0

        tCtS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)

        tCtL = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtL.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtL)

        tCtO = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)

        assert tmem_offset <= tmem_alloc_cols, (
            f"\t{tmem_offset} tmem cols used, {tmem_alloc_cols} tmem cols allocated"
        )

        ##############################
        # Exit early
        ##############################
        if exit_early:
            if warp_idx == mma_vp_warp_id:
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)

            elif warpgroup_idx == correction_warpgroup_id:
                sM_final_nbar.arrive()
                sL_final_nbar.arrive()

        ##############################
        # TMA QK Dispatch
        ##############################
        elif warp_idx == tma_qk_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            # Slice and partition Q
            # (TILE_H, TILE_D)
            gQ = cute.local_tile(
                mQ,
                tiler=(blk_tile_hp, blk_tile_d),
                coord=(coord_hp, 0, coord_hb),
            )
            # (MMA_TILE_N, MMA_TILE_K, #TILE_DK)
            gQ_mma = cute.local_tile(gQ, (mma_tile_n, mma_tile_k), coord=(0, None))
            # (MMA, #MMA_N, #MMA_K, #TILE_DK)
            tBgQ = thrblk_mma_kq.partition_B(gQ_mma)
            # (TMA, #TILE_DK)
            tBsQ_tma, tBgQ_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgQ, 0, 3),
            )

            # Slice and partition K
            sK = cute.make_tensor(
                tAsK.iterator, smem_layout_k_tma.outer
            )  # ((PAGE, MMA_TILE_K), #PAGE_M, k_stages)
            gK = cute.local_tile(
                mK, (page_size, mma_tile_k), coord=(0, None, (coord_hk, None))
            )  # (PAGE, MMA_TILE_K, #TILE_DK, #PAGE_S)
            sK_tma, gK_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_k,
                mcast_coord,
                mcast_layout,
                smem_tensor=sK,
                gmem_tensor=cute.group_modes(gK, 0, 2),
            )  # (TMA, Rest...)

            # Construct page table for this batch
            gPT = cute.make_tensor(page_table_iter + table_offset, (page_count,))
            cPT = cute.make_identity_tensor(page_count)

            # Partition page table
            pt_load = cute.make_tiled_copy(
                cpasync_atom,
                # 1 thread per page
                cute.make_ordered_layout((pages_s, 1), order=(1, 0)),
                (pages_s,),
            )
            lane_load_page = lane_idx < pages_s
            thr_pt_load = pt_load.get_slice(lane_idx)
            tPTgPT = thr_pt_load.partition_S(gPT)  # (CPY=1, #CPY=#TILE_S)
            tPTcPT = thr_pt_load.partition_S(cPT)  # (CPY=1, #CPY=#TILE_S)
            tPTsPT = thr_pt_load.partition_D(sPT)  # (CPY=1, #CPY=1, pt_stages)

            cute.arch.griddepcontrol_wait()

            # Load Q first: it does not depend on the page table, and an
            # early issue clears the MMA warp's q_load_mbar sooner.
            cute.copy(tma_atom_q, tBgQ_tma, tBsQ_tma, tma_bar_ptr=q_load_mbar)

            if cutlass.const_expr(not spec_pt_prefetch):
                # Prefetch page indices for first tile
                if lane_load_page:
                    logical_page_idx = tPTcPT[0, kv_split_idx]
                    if logical_page_idx < page_count:
                        cute.copy(
                            cpasync_atom, tPTgPT[None, kv_split_idx], tPTsPT[None, 0, 0]
                        )
                    else:
                        tPTsPT[0] = -1  # load OOB zeros
                cute.arch.sync_warp()
                cute.arch.cp_async_commit_group()

            # Sequence loop
            pt_index = 0
            for s in cutlass.range(iters_s):
                # Prefetch page indices for next tile
                pt_index_next = 0 if pt_index == (pt_stages - 1) else pt_index + 1
                if s < iters_s - 1 and lane_load_page:
                    tile_s_next = (s + 1) * work_splits + kv_split_idx
                    logical_page_idx = tPTcPT[0, tile_s_next]
                    virt_page_idx_gmem = tPTgPT[None, tile_s_next]
                    virt_page_idx_smem = tPTsPT[None, 0, pt_index_next]
                    if logical_page_idx < page_count:
                        cute.copy(cpasync_atom, virt_page_idx_gmem, virt_page_idx_smem)
                    else:
                        virt_page_idx_smem[0] = -1  # load OOB zeros
                cute.arch.sync_warp()
                cute.arch.cp_async_commit_group()

                # Load page indices
                cute.arch.cp_async_wait_group(1)
                rPT = sPT[None, pt_index].load().reshape((pages_m, tiles_sm))
                pt_index = pt_index_next

                # Load K
                tma_order_v_nbar.arrive_and_wait()
                kv_token = kv_producer.try_acquire()
                for sm in cutlass.range_constexpr(tiles_sm):
                    for dk in cutlass.range_constexpr(tiles_dk):
                        kv_handle = kv_producer.acquire_and_advance(kv_token)
                        is_last_iter = sm == tiles_sm - 1 and dk == tiles_dk - 1
                        if is_last_iter:
                            tma_order_k_nbar.arrive()
                        else:
                            kv_token = kv_producer.try_acquire()

                        for pm in cutlass.range_constexpr(pages_m):
                            virtual_page_idx = rPT[pm, sm]
                            cute.copy(
                                tma_atom_k,
                                gK_tma[None, dk, virtual_page_idx],
                                sK_tma[None, pm, kv_handle.index],
                                tma_bar_ptr=kv_handle.barrier,
                            )

                # Advance for TMA V
                if s >= prefetch_iters:
                    # Load skip predicate
                    keep_tile = not enable_blasst
                    if cutlass.const_expr(enable_blasst):
                        sp_handle = sp_consumer.wait_and_advance()
                        keep_tile = sSP_i32[sp_handle.index] != 0
                        sp_handle.release()

                    if keep_tile:
                        for _ in cutlass.range_constexpr(tiles_dm * tiles_sk):
                            kv_producer.advance()

            # Tail V loop
            for s in cutlass.range_constexpr(prefetch_iters):
                tma_order_v_nbar.arrive_and_wait()
                tma_order_k_nbar.arrive()
                if cutlass.const_expr(enable_blasst):
                    if s < min(prefetch_iters, iters_s):
                        sp_handle = sp_consumer.wait_and_advance()
                        sp_handle.release()

        ##############################
        # TMA VO Dispatch
        ##############################
        elif warp_idx == tma_vo_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            # Slice and partition V
            sV = cute.make_tensor(
                tAsV.iterator, smem_layout_v_tma.outer
            )  # ((MMA_TILE_M, PAGE), #PAGE_K, v_stages)
            gV = cute.local_tile(
                mV, (mma_tile_m, page_size), coord=(None, 0, (coord_hk, None))
            )  # (MMA_TILE_M, PAGE, #TILE_DM, #PAGE_S)
            sV_tma, gV_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v,
                mcast_coord,
                mcast_layout,
                smem_tensor=sV,
                gmem_tensor=cute.group_modes(gV, 0, 2),
            )  # (TMA, Rest...)

            # Prefetch K loop
            tma_order_v_nbar.arrive()
            for s in cutlass.range_constexpr(prefetch_iters):
                if s < iters_s:
                    for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                        kv_producer.advance()
                tma_order_k_nbar.arrive_and_wait()
                tma_order_v_nbar.arrive()

            # Sequence loop
            pt_index = 0
            for s in cutlass.range(iters_s):
                # Advance for TMA K
                if s < iters_s - prefetch_iters:
                    for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                        kv_producer.advance()

                # Load skip predicate
                keep_tile = not enable_blasst
                if cutlass.const_expr(enable_blasst):
                    sp_handle = sp_consumer.wait_and_advance()
                    keep_tile = sSP_i32[sp_handle.index] != 0
                    sp_handle.release()
                    if not keep_tile:
                        tma_order_k_nbar.arrive_and_wait()
                        tma_order_v_nbar.arrive()

                if keep_tile:
                    # Load page indices
                    tma_order_k_nbar.arrive_and_wait()
                    rPT = sPT[None, pt_index].load().reshape((pages_k, tiles_sk))

                    # Load V
                    kv_token = kv_producer.try_acquire()
                    for sk in cutlass.range_constexpr(tiles_sk):
                        for dm in cutlass.range_constexpr(tiles_dm):
                            kv_handle = kv_producer.acquire_and_advance(kv_token)
                            is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                            if is_last_iter:
                                tma_order_v_nbar.arrive()
                            else:
                                kv_token = kv_producer.try_acquire()

                            for pk in cutlass.range_constexpr(pages_k):
                                virtual_page_idx = rPT[pk, sk]
                                cute.copy(
                                    tma_atom_v,
                                    gV_tma[None, dm, virtual_page_idx],
                                    sV_tma[None, pk, kv_handle.index],
                                    tma_bar_ptr=kv_handle.barrier,
                                )
                pt_index = 0 if pt_index == (pt_stages - 1) else pt_index + 1

            # Slice and partition O
            # (TILE_D, TILE_H)
            coord_b_partial = (
                kv_split_idx * batches + coord_b if do_kernel_red else coord_b
            )
            gO = cute.local_tile(
                mO,
                tiler=(blk_tile_d, blk_tile_hp),
                coord=(0, coord_hp, (coord_hk, coord_b_partial)),
            )
            # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN=1)
            gO_mma = cute.flat_divide(gO, (mma_tile_m, mma_tile_n))
            # (TMA, #TILE_DM, #TILE_HN)
            sO_tma, gO_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_o,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(sO_mma, 0, 2),
                gmem_tensor=cute.group_modes(gO_mma, 0, 2),
            )

            # Store O to gmem
            sO_final_nbar.arrive_and_wait()
            if cutlass.const_expr(do_atomic_red):
                # Acquire split 0's overlapped output zeroing (released via
                # the zero_done mbarrier after a gpu-scope fence), then
                # re-enter the async proxy so the reduce-add TMA observes
                # the zero stores.
                cute.arch.mbarrier_wait(zero_done_mbar_ptr, phase=0)
                cute.arch.fence_proxy("async")
            cute.copy(tma_atom_o, sO_tma, gO_tma)

        ##############################
        # MMA KQ (BMM1) Dispatch
        ##############################
        elif warp_idx == mma_kq_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            # Setup mma descriptors
            tAsK_desc = thrblk_mma_kq.make_fragment_A(tAsK)
            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)

            # Wait for Q
            cute.arch.mbarrier_wait(q_load_mbar, phase=0)

            # Sequence loop
            for s in cutlass.range(iters_s):
                s_token = s_producer.try_acquire()

                mma_order_vp_nbar.arrive_and_wait()
                k_token = kv_consumer.try_wait()

                s_handle = s_producer.acquire_and_advance(s_token)
                for sm in cutlass.range_constexpr(tiles_sm):
                    tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                    for dk in cutlass.range_constexpr(tiles_dk):
                        k_handle = kv_consumer.wait_and_advance(k_token)
                        is_last_iter = sm == tiles_sm - 1 and dk == tiles_dk - 1
                        if is_last_iter:
                            mma_order_kq_nbar.arrive()
                        else:
                            k_token = kv_consumer.try_wait()

                        mmas_k = cute.size(tAsK.shape[2])
                        for mma_k in cutlass.range_constexpr(mmas_k):
                            cute.gemm(
                                tiled_mma_kq,
                                tCtS[mma_dice + (sm, s_handle.index)],
                                tAsK_desc[None, None, mma_k, k_handle.index],
                                tBsQ_desc[None, None, mma_k, dk],
                                tCtS[mma_dice + (sm, s_handle.index)],
                            )
                            if dk == 0 and mma_k == 0:
                                tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                        k_handle.release()
                s_handle.commit()

                # Advance for MMA VP
                if s >= prefetch_iters:
                    keep_tile = not enable_blasst
                    if cutlass.const_expr(enable_blasst):
                        sp_handle = sp_consumer.wait_and_advance()
                        keep_tile = sSP_i32[sp_handle.index] != 0
                        sp_handle.release()

                    if keep_tile:
                        for _ in cutlass.range_constexpr(tiles_dm * tiles_sk):
                            kv_consumer.advance()

            # Tail loop
            for s in cutlass.range_constexpr(prefetch_iters):
                mma_order_vp_nbar.arrive_and_wait()
                mma_order_kq_nbar.arrive()
                if cutlass.const_expr(enable_blasst):
                    if s < min(prefetch_iters, iters_s):
                        sp_handle = sp_consumer.wait_and_advance()
                        sp_handle.release()

        ##############################
        # MMA VP (BMM2) Dispatch
        ##############################
        elif warp_idx == mma_vp_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            # Setup mma descriptors
            tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
            tAsV_desc = thrblk_mma_vp.make_fragment_A(tAsV)
            tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)
            tBsP_residual_desc = thrblk_mma_vp.make_fragment_B(tBsP_residual_nk)

            # Prefetch loop
            mma_order_vp_nbar.arrive()
            for s in cutlass.range_constexpr(prefetch_iters):
                if s < iters_s:
                    for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                        kv_consumer.advance()
                mma_order_kq_nbar.arrive_and_wait()
                mma_order_vp_nbar.arrive()

            # Sequence loop
            for s in cutlass.range(iters_s):
                # Advance for MMA KQ
                if s < iters_s - prefetch_iters:
                    for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                        kv_consumer.advance()

                keep_tile = not enable_blasst
                if cutlass.const_expr(enable_blasst):
                    sp_handle = sp_consumer.wait_and_advance()
                    keep_tile = sSP_i32[sp_handle.index] != 0
                    sp_handle.release()
                    if not keep_tile:
                        mma_order_kq_nbar.arrive_and_wait()
                        mma_order_vp_nbar.arrive()

                if keep_tile:
                    p_token = p_consumer.try_wait()
                    o_token = o_producer.try_acquire()

                    mma_order_kq_nbar.arrive_and_wait()
                    v_token = kv_consumer.try_wait()

                    p_handle = p_consumer.wait_and_advance(p_token)
                    o_handle = o_producer.acquire_and_advance(o_token)
                    for sk in cutlass.range_constexpr(tiles_sk):
                        for dm in cutlass.range_constexpr(tiles_dm):
                            v_handle = kv_consumer.wait_and_advance(v_token)
                            is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                            if is_last_iter:
                                mma_order_vp_nbar.arrive()
                            else:
                                v_token = kv_consumer.try_wait()

                            mmas_k = cute.size(tAsV.shape[2])
                            for mma_k in cutlass.range_constexpr(mmas_k):
                                cute.gemm(
                                    tiled_mma_vp,
                                    tCtO[mma_dice + (dm, o_handle.index)],
                                    tAsV_desc[None, None, mma_k, v_handle.index],
                                    tBsP_desc[None, None, mma_k, sk, p_handle.index],
                                    tCtO[mma_dice + (dm, o_handle.index)],
                                )
                                cute.gemm(
                                    tiled_mma_vp,
                                    tCtO[mma_dice + (dm, o_handle.index)],
                                    tAsV_desc[None, None, mma_k, v_handle.index],
                                    tBsP_residual_desc[
                                        None, None, mma_k, sk, p_handle.index
                                    ],
                                    tCtO[mma_dice + (dm, o_handle.index)],
                                )
                            v_handle.release()
                    p_handle.release()
                    o_handle.commit()

            # Wait for signal to dealloc tmem, then dealloc
            if iters_s == 1:
                # Epilogue still reads the empty buffer
                o_producer.commit()
                o_producer.advance()
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        ##############################
        # Softmax Dispatch
        ##############################
        elif warpgroup_idx in softmax_warpgroup_ids:
            # Free registers
            # Under the wide-N reconfiguration the single softmax warpgroup
            # keeps the launch allocation (softmax_regs == launch budget;
            # decreasing spills the 64-wide q4 chains, increasing exceeds
            # the CTA register pool). Only a hypothetical narrow-N kernel
            # with reconfiguration enabled would decrease here.
            if cutlass.const_expr(use_reg_reconfig and softmax_warpgroups == 2):
                cute.arch.setmaxregister_decrease(softmax_regs)

            # Initialize for dual warpgroups
            softmax_phase = 0
            if cutlass.const_expr(softmax_warpgroups == 2):
                softmax_phase = (warpgroup_idx - 1) % softmax_warpgroups
                sM_acquire_nbar = with_phase(sM_mutex_nbar, softmax_phase)
                sM_release_nbar = with_phase(sM_mutex_nbar, softmax_phase ^ 1)
                if softmax_phase == 1:
                    s_consumer.advance()
                    p_producer.advance()
                    sM_release_nbar.arrive()
            if iters_s == 1 and softmax_phase == softmax_warpgroups - 1:
                with_phase(tL_producer_nbar, 1).arrive()
            assert not (enable_blasst and softmax_warpgroups != 1), (
                "blasst only supports 1 softmax wg"
            )

            # Construct copy atom for S
            tmem_repeat_op_s = blk_tile_n
            if cutlass.const_expr(mma_tile_n == blk_tile_n and tiles_sm in (2, 4)):
                tmem_repeat_op_s *= tiles_sm
            tmem_repeat_op_s = tcgen05.Repetition(tmem_repeat_op_s)
            tmem_load_op_s = tcgen05.Ld32x32bOp(tmem_repeat_op_s)
            tmem_load_atom_s = cute.make_copy_atom(tmem_load_op_s, acc_dtype)
            # Tile atom and slice
            tCtS_stage = tCtS[mma_dice + (None, 0)]
            tmem_load_s = tcgen05.make_tmem_copy(tmem_load_atom_s, tCtS_stage)
            thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)
            # Partition S and P
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #CPY_SM, stages)
            tStS = thr_load_s.partition_S(tCtS)
            tSsP = thr_load_s.partition_D(tCsP)
            tSsP_residual = thr_load_s.partition_D(tCsP_residual)
            # Slice unused modes
            tStS = tStS[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, s_stages)
            tSsP = tSsP[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, p_stages)
            tSsP_residual = tSsP_residual[
                None, 0, 0, 0, None, None
            ]  # (CPY, #CPY_SM, p_stages)

            # Construct copy atom for L
            tmem_repeat_op_l = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_l = tcgen05.Ld32x32bOp(tmem_repeat_op_l)
            tmem_store_op_l = tcgen05.St32x32bOp(tmem_repeat_op_l)
            tmem_load_atom_l = cute.make_copy_atom(tmem_load_op_l, acc_dtype)
            tmem_store_atom_l = cute.make_copy_atom(tmem_store_op_l, acc_dtype)
            # Tile atom and slice
            tCtL_phase = tCtL[mma_dice + (softmax_phase,)]
            tmem_load_l = tcgen05.make_tmem_copy(tmem_load_atom_l, tCtL_phase)
            thr_load_l = tmem_load_l.get_slice(warpgroup_tidx)
            # Partition L
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages)
            tStL = thr_load_l.partition_S(tCtL)
            # Slice unused modes
            # (CPY, stages)
            tStL = tStL[None, 0, 0, 0, None]
            tSrL_shape = thr_load_l.partition_D(tCtL_phase).shape[:1]

            # Mask configuration loop args
            range_args = mask_config.get_range_args(
                prediction,
                seqlen,
                blk_tile_p,
                blk_tile_s,
                tiles_s,
                iters_s,
                work_splits,
                kv_split_idx,
                softmax_warpgroups,
                softmax_phase,
            )
            num_mask_phases = len(range_args)

            if cutlass.const_expr(enable_blasst):
                # Tile skip tracking
                sM_lane_prev = -Float32.inf
                # Per-batch BLASST threshold, matching trtllm:
                # effective threshold_p = scale_factor / seqlen
                log2_threshold_p = log2_threshold_scale_factor - cute.math.log2(
                    Float32(seqlen)
                )

            # Masking phase loop over all sequence tiles
            loop_idx = 0
            for mask_phase in cutlass.range_constexpr(num_mask_phases):
                # Sequence tile loop per masking phase
                start, stop, step, is_masked = range_args[mask_phase]
                for coord_s in cutlass.range(start, stop, step):
                    # Load S from tmem and notify BMM1
                    s_token = s_consumer.try_wait()
                    s_handle = s_consumer.wait_and_advance(s_token)
                    tStS_s = tStS[None, None, s_handle.index]
                    tSrS_s = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                    cute.copy(tmem_load_atom_s, tStS_s, tSrS_s)
                    cute.arch.fence_view_async_tmem_load()
                    s_handle.release()

                    # Apply mask
                    if cutlass.const_expr(is_masked):
                        masked = cute.make_rmem_tensor(
                            (blk_tile_h, blk_tile_p, tiles_sm), acc_dtype
                        )
                        masked.store(tSrS_s.load().reshape(masked.shape))
                        offset_p = coord_p * blk_tile_p
                        offset_s = coord_s * blk_tile_s + warpgroup_tidx
                        for sm in cutlass.range_constexpr(tiles_sm):
                            for p in cutlass.range_constexpr(blk_tile_p):
                                idx_q = offset_p + p
                                idx_kv = offset_s + sm * mma_tile_m
                                is_oob_kv = mask_config.is_oob_kv(
                                    idx_q, idx_kv, prediction, seqlen
                                )
                                mask = -Float32.inf if is_oob_kv else Float32(0)
                                masked_p = masked[None, p, sm]
                                masked_p.store(masked_p.load() + mask)
                        scores = masked.load().reshape((blk_tile_n, tiles_sm))
                    else:
                        scores = tSrS_s.load().reshape((blk_tile_n, tiles_sm))

                    # Reduce colmax in thread RF
                    rM = cute.make_rmem_tensor_like(sM)
                    rM.store(
                        scores.reduce(
                            cute.ReductionOp.MAX,
                            # prevent nan accumulations with masking, see explanation in gqa_decode.py
                            init_val=min_f32,
                            reduction_profile=(None, 0),
                        )
                    )

                    # Reduce colmax in warp RF
                    rM_lane = Float32(0)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        rM[n] = warp_fmax(rM[n])  # warp reduction
                        # Avoid dynamic register indexing (creates spills)
                        if n == lane_idx:
                            rM_lane = rM[n]
                    rM_lane *= scale_s_log2_e  # apply scale

                    # Compute skip predicate
                    keep_tile = not enable_blasst
                    if cutlass.const_expr(enable_blasst):
                        # warp reduction
                        lane_keep_tile = rM_lane - sM_lane_prev >= log2_threshold_p
                        lane_keep_tile &= lane_store_max  # oob lanes skip
                        lane_keep_tile |= loop_idx < o_stages  # correction loop is s-2
                        warp_keep_tile = warp_or(Int32(lane_keep_tile))
                        loop_idx += 1

                        # warpgroup reduction
                        sp_handle = sp_producer.acquire_and_advance()
                        with cute.arch.elect_one():
                            sSP_i8[warpgroup_widx, sp_handle.index] = Int8(
                                warp_keep_tile
                            )
                        sp_handle.commit()
                        sSP_producer_nbar.sync()
                        keep_tile = sSP_i32[sp_handle.index] != 0

                    if keep_tile:
                        p_token = p_producer.try_acquire()

                        # Reduce colmax in smem
                        if cutlass.const_expr(softmax_warpgroups == 2):
                            sM_acquire_nbar.arrive_and_wait()
                        sM_consumer_nbar.arrive_and_wait()
                        if lane_store_max:
                            smem_fmax(sM.iterator + sM.layout(lane_idx), rM_lane)

                        # Wait for empty P buffer
                        # Here so we can interleave ex2 with convert ops
                        p_handle = p_producer.acquire_and_advance(p_token)
                        tSsP_s = tSsP[None, None, p_handle.index]

                        # Load colmax
                        sM_producer_nbar.arrive_and_wait()
                        colmax = sM.load()
                        if cutlass.const_expr(enable_blasst):
                            if lane_store_max:
                                sM_lane_prev = sM[lane_idx]
                        if cutlass.const_expr(softmax_warpgroups == 2):
                            sM_release_nbar.arrive()

                        # Compute online softmax
                        probs = exp2(scale_s_log2_e * scores - colmax)

                        # Store P to smem and notify BMM2
                        scaled_probs = probs * Float32(448.0)
                        probs_leading = scaled_probs.to(mma_dtype)
                        tSsP_s.store(probs_leading.reshape(tSsP_s.shape))
                        tSsP_residual_s = tSsP_residual[
                            None, None, p_handle.index
                        ]
                        probs_residual = scaled_probs - probs_leading.to(Float32)
                        tSsP_residual_s.store(
                            probs_residual.to(mma_dtype).reshape(tSsP_residual_s.shape)
                        )
                        cute.arch.fence_view_async_shared()
                        p_handle.commit()

                        # Accumulate per-thread colsum
                        colsum = probs[None, 0]
                        for sm in cutlass.range_constexpr(1, tiles_sm, 1):
                            colsum += probs[None, sm]
                        tSrL = cute.make_rmem_tensor(tSrL_shape, acc_dtype)
                        tSrL.store(colsum.reshape(tSrL.shape))

                        # Store per-thread colsum to tmem
                        with_phase(tL_consumer_nbar, softmax_phase).arrive_and_wait()
                        cute.copy(tmem_store_atom_l, tSrL, tStL[None, softmax_phase])
                        cute.arch.fence_view_async_tmem_store()
                        with_phase(tL_producer_nbar, softmax_phase).arrive()

                        # Advance state
                        if cutlass.const_expr(softmax_warpgroups == 2):
                            s_consumer.advance()
                            p_producer.advance()
                        else:
                            softmax_phase ^= 1

        ##############################
        # Correction Dispatch
        ##############################
        elif warpgroup_idx == correction_warpgroup_id:
            # Alloc registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_increase(correction_regs)

            # Select copy atoms for O and L
            tmem_repeat_op_o = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_o = tcgen05.Ld32x32bOp(tmem_repeat_op_o)
            tmem_store_op_o = tcgen05.St32x32bOp(tmem_repeat_op_o)
            tmem_load_atom_o = cute.make_copy_atom(tmem_load_op_o, acc_dtype)
            tmem_store_atom_o = cute.make_copy_atom(tmem_store_op_o, acc_dtype)
            # Tile atoms and slice
            tCtO_dm = tCtO[mma_dice + (0, 0)]
            tmem_load_o = tcgen05.make_tmem_copy(tmem_load_atom_o, tCtO_dm)
            thr_load_o = tmem_load_o.get_slice(warpgroup_tidx)
            # Partition O and L
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, o_stages)
            tOtO = thr_load_o.partition_S(tCtO)
            tOsO = thr_load_o.partition_D(tCsO)
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, o_stages)
            tOtL = thr_load_o.partition_S(tCtL)
            # Slice unused modes
            tOtO = tOtO[None, 0, 0, 0, None, None]  # (CPY, #TILE_DM, o_stages)
            tOsO = tOsO[None, 0, 0, 0, None]  # (CPY, #TILE_DM)
            tOtL = tOtL[None, 0, 0, 0, None]  # (CPY, o_stages)

            # colsum load helper
            def colsum_load(
                phase,
                blk_tile_n=blk_tile_n,
                tOtL=tOtL,
                tOrO_shape=tOsO.shape[:1],
                tmem_load_atom_o=tmem_load_atom_o,
                tL_producer_nbar=tL_producer_nbar,
                tL_consumer_nbar=tL_consumer_nbar,
            ):
                with_phase(tL_producer_nbar, phase).arrive_and_wait()
                tOtL_s = tOtL[None, phase]
                tOrL_s = cute.make_rmem_tensor(tOrO_shape, Float32)
                cute.copy(tmem_load_atom_o, tOtL_s, tOrL_s)
                cute.arch.fence_view_async_tmem_load()
                with_phase(tL_consumer_nbar, phase).arrive()
                return tOrL_s.load().reshape(blk_tile_n)

            # Initialize O and colsum in tmem
            tOrO = cute.make_rmem_tensor(tOsO.shape, acc_dtype)
            tOrO.fill(Float32(0))
            for phase in cutlass.range_constexpr(o_stages):
                cute.copy(tmem_store_atom_o, tOrO, tOtO[None, None, phase])
            cute.copy(tmem_store_atom_o, tOrO[None, 0], tOtL[None, 1])
            cute.arch.fence_view_async_tmem_store()

            # Initialize consumer barriers
            sM_consumer_nbar.arrive()
            for phase in cutlass.range_constexpr(o_stages):
                with_phase(tL_consumer_nbar, phase).arrive()

            # Split 0 zeroes its cluster's output tile here, overlapped with
            # the mainloop ramp, instead of serializing zeroing + gpu-scope
            # fence ahead of every CTA's first KV load (old start-of-kernel
            # path). One warp stores the tile, releases with a gpu-scope
            # fence, then publishes completion into every split's zero_done
            # mbarrier via dsmem signal stores; each split's reduce-add TMA
            # acquires that mbarrier (plus an async-proxy fence) before
            # touching out. Split 0 never exits early (seqlen > 0 implies
            # tiles_s >= 1), so the signal always fires.
            if cutlass.const_expr(do_atomic_red):
                if kv_split_idx == 0:
                    if warpgroup_widx == 0:
                        zero_vec = 128 // o_dtype.width
                        zero_frag = cute.make_rmem_tensor(
                            cute.make_layout(zero_vec), o_dtype
                        )
                        for zi in cutlass.range_constexpr(zero_vec):
                            zero_frag[zi] = o_dtype(0)
                        _, o_prediction = mO_plain.shape[1]
                        o_heads = mO_plain.shape[1][0]
                        zero_chunks_d = blk_tile_d // zero_vec
                        zero_total = zero_chunks_d * blk_tile_n
                        zero_lanes = warp_threads
                        for zit in cutlass.range_constexpr(
                            (zero_total + zero_lanes - 1) // zero_lanes
                        ):
                            zc = lane_idx + zit * zero_lanes
                            if zc < zero_total:
                                zr = zc // zero_chunks_d
                                zv = zc - zr * zero_chunks_d
                                zh = zr % blk_tile_h
                                zp = zr // blk_tile_h
                                zh_g = coord_hg * blk_tile_h + zh
                                zp_g = coord_p * blk_tile_p + zp
                                if (zh_g < o_heads) & (zp_g < o_prediction):
                                    zchunk = cute.local_tile(
                                        mO_plain,
                                        tiler=(zero_vec, (1, 1)),
                                        coord=(zv, (zh_g, zp_g), (coord_hk, coord_b)),
                                    )
                                    cute.autovec_copy(zero_frag, zchunk[(None, (0, 0))])
                        cute.arch.sync_warp()
                        if lane_idx == 0:
                            cute.arch.fence_acq_rel_gpu()
                            zd_atom = cute.make_copy_atom(
                                cute.nvgpu.cpasync.CopyDsmemStoreOp(),
                                acc_dtype,
                                num_bits_per_copy=128,
                            )
                            zd_frag = cute.make_rmem_tensor(
                                cute.make_layout(4), acc_dtype
                            )
                            for zi in cutlass.range_constexpr(4):
                                zd_frag[zi] = acc_dtype(0)
                            for peer in cutlass.range(kv_splits):
                                zd_slot_peer = cute.make_tensor(
                                    cute.arch.map_dsmem_ptr(
                                        zero_done_slot.iterator, peer
                                    ),
                                    zero_done_slot.layout,
                                )
                                zd_mbar_peer = cute.arch.map_dsmem_ptr(
                                    zero_done_mbar_ptr, peer
                                )
                                cute.copy(
                                    zd_atom, zd_frag, zd_slot_peer, mbar_ptr=zd_mbar_peer
                                )

            # Initialize colsum in RF
            colsum_p = cute.make_rmem_tensor((blk_tile_n, o_stages), Float32)
            colsum_0, colsum_1 = colsum_p[None, 0], colsum_p[None, 1]
            colsum_p.fill(Float32(0))

            # Load colmax of s-2, s-1
            sM_lane_prev_prev = sM_lane_prev = Float32(0)
            for s in cutlass.range_constexpr(o_stages):
                sM_lane_prev_prev = sM_lane_prev
                if not (s == 1 and iters_s == 1):
                    if cutlass.const_expr(enable_blasst):
                        sp_handle = sp_consumer.wait_and_advance()
                        sp_handle.release()
                    sM_producer_nbar.arrive_and_wait()
                    if lane_store_max:
                        sM_lane_prev = sM[lane_idx]
                    sM_consumer_nbar.arrive()

            # Sequence loop
            softmax_phase = 0
            unroll = o_stages if not enable_blasst else 1
            keep_tile = not enable_blasst
            for s in cutlass.range(iters_s - o_stages, unroll=unroll):
                # Load skip predicate
                if cutlass.const_expr(enable_blasst):
                    sp_handle = sp_consumer.wait_and_advance()
                    keep_tile = sSP_i32[sp_handle.index] != 0
                    sp_handle.release()

                if keep_tile:
                    # Load colsum of s-2
                    colsum_s = colsum_load(softmax_phase)

                    # Load colmax of s
                    sM_producer_nbar.arrive_and_wait()
                    if s == iters_s - o_stages - 1:
                        sM_final_nbar.arrive()
                    sM_lane = Float32(0)
                    if lane_store_max:
                        sM_lane = sM[lane_idx]
                    sM_consumer_nbar.arrive()

                    # Wait for O of s-2
                    # Here so we can interleave shuffle_sync with correction muls
                    o_token = o_consumer.try_wait()
                    o_handle = o_consumer.wait_and_advance(o_token)

                    # Compute correction of s-2
                    correction_lane = exp2(sM_lane_prev_prev - sM_lane)
                    correction = cute.make_rmem_tensor_like(sM)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        correction[n] = cute.arch.shuffle_sync(correction_lane, n)
                    correction = correction.load()
                    sM_lane_prev_prev, sM_lane_prev = sM_lane_prev, sM_lane

                    # Correct O of s-2 and notify MMA VP
                    correction_o = correction.reshape(tOsO.shape[:1])
                    for dm in cutlass.range_constexpr(tiles_dm):
                        tOtO_dm = tOtO[None, dm, softmax_phase]
                        tOrO_dm = cute.make_rmem_tensor(tOsO.shape[:1], acc_dtype)
                        cute.copy(tmem_load_atom_o, tOtO_dm, tOrO_dm)
                        tOrO_dm.store(correction_o * tOrO_dm.load())
                        cute.copy(tmem_store_atom_o, tOrO_dm, tOtO_dm)
                    cute.arch.fence_view_async_tmem_store()
                    o_handle.release()

                    # Correct and accumulate colsum of s-2
                    colsum_s *= correction
                    if softmax_phase == 0:
                        colsum_0.store(correction * colsum_0.load() + colsum_s)
                    elif softmax_phase == 1:
                        colsum_1.store(correction * colsum_1.load() + colsum_s)

                    # Next softmax producer phase
                    softmax_phase ^= 1

            # Notify for final colmax if we didn't already
            if not keep_tile or iters_s <= o_stages:
                sM_final_nbar.arrive()

            # Compute correction of s-1
            correction_lane = exp2(sM_lane_prev_prev - sM_lane_prev)
            correction = cute.make_rmem_tensor_like(sM)
            for n in cutlass.range_constexpr(blk_tile_n):
                correction[n] = cute.arch.shuffle_sync(correction_lane, n)
            correction = correction.load()

            # Correct and accumulate final colsum
            tail_phase = softmax_phase if enable_blasst else iters_s % o_stages
            for phase in cutlass.range_constexpr(o_stages):
                if tail_phase == phase:
                    # Accumulate in thread RF
                    colsum_prev = colsum_load(phase)
                    colsum_final = colsum_load(phase ^ 1)
                    colsum_prev += colsum_p[None, phase].load()
                    colsum_final += colsum_p[None, phase ^ 1].load()
                    colsum_final += correction * colsum_prev
                    # Reduce colsum in warp RF
                    rL_lane = Float32(0.0)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        rL_n = cute.arch.warp_reduction_sum(colsum_final[n])
                        if n == lane_idx:
                            rL_lane = rL_n
                    # Store partial colsum in smem and notify
                    if lane_store_max:
                        sL[lane_idx, warpgroup_widx] = rL_lane
                    # Wait to ensure reduction warp has reset sM_final_nbar
                    sL_final_nbar.arrive_and_wait()

            # Load O of s-1, s
            tOrO_tail = cute.make_rmem_tensor((*tOsO.shape, o_stages), acc_dtype)
            for s in cutlass.range_constexpr(o_stages):
                o_handle = o_consumer.wait_and_advance()
                tOtO_s = tOtO[None, None, tail_phase ^ s]
                tOrO_s = tOrO_tail[None, None, s]
                cute.copy(tmem_load_atom_o, tOtO_s, tOrO_s)
                cute.arch.fence_view_async_tmem_load()
                o_handle.release()  # Final release signals tmem dealloc
            tOrO_prev = tOrO_tail[None, None, 0].load()
            tOrO_final = tOrO_tail[None, None, 1].load()

            # Correct and accumulate output
            output_prev = tOrO_prev.reshape((blk_tile_n, tiles_dm))
            output_final = tOrO_final.reshape((blk_tile_n, tiles_dm))
            output_final += correction * output_prev

            # Apply final normalization
            if cutlass.const_expr(do_atomic_red or do_none_red):
                # final normalization stored in sM
                sM_final_nbar.arrive_and_wait()
                normalization = sM.load()
                output_final *= normalization

            # Store O to smem and notify
            tOsO.store(output_final.to(o_dtype).reshape(tOsO.shape))
            cute.arch.fence_view_async_shared()
            sO_final_nbar.arrive()

            # Debug
            if cutlass.const_expr(enable_blasst and debug_blasst):
                batches = mQ.shape[-1][1]
                # Append skip count to page counts
                tiles_skipped_ptr = seqlens_iter + batches
                tiles_kept = o_consumer.current_handle().count if iters_s > 1 else 1
                tiles_skipped = iters_s - tiles_kept
                if warpgroup_tidx == 0:
                    gmem_add(tiles_skipped_ptr, Int32(tiles_skipped))

        ##############################
        # Reduction Dispatch
        ##############################
        if warp_idx == reduction_warp_id:
            if cutlass.const_expr(sSink is not None):
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_warp()

            if cutlass.const_expr(do_kernel_red):
                GqaDecode.reduction_epilogue(
                    blk_tile_hp,
                    coord_hp,
                    coord_hb,
                    kv_split_idx,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    sM,
                    sL,
                    mM_partial,
                    mL_partial,
                )
            elif cutlass.const_expr(do_atomic_red):
                gL = None
                if cutlass.const_expr(store_lse):
                    gL = cute.local_tile(mL, (blk_tile_hp,), (coord_hp, coord_hb))
                GqaDecode.reduction_cluster(
                    blk_tile_n,
                    kv_splits,
                    kv_split_idx,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    reduction_mbars_ptr,
                    sM,
                    sL,
                    sR,
                    gL,
                    sSink,
                    scale_o,
                )
            elif cutlass.const_expr(do_none_red):
                gL = None
                if cutlass.const_expr(store_lse):
                    gL = cute.local_tile(mL, (blk_tile_hp,), (coord_hp, coord_hb))
                GqaDecode.reduction_none(
                    lane_store_max,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    sM,
                    sL,
                    gL,
                    sSink,
                    scale_o,
                )
            cute.arch.griddepcontrol_launch_dependents()

        return


