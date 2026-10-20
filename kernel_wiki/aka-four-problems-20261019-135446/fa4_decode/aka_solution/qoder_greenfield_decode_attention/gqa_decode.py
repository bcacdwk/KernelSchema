# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Literal, Type, Tuple, cast
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
    BFloat16,
    Float16,
    Int32,
    Int64,
    Optional,
)

from .mask import AttentionMask

mma_modes = (0, 1, 2)
mma_dice = (None, None, None)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128
max_reduction_iters = 4

min_f32 = Float32(-3.4028234663852886e38)
log2_e = math.log2(math.e)
exp2 = partial(cute.math.exp2, fastmath=True)
warp_fmax = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")
gmem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="gpu")


class GroupedQueryAttentionDecode:
    def can_implement(
        self,
        kv_splits,
        qo_shape_bshd,
        kv_shape_bshd,
        qkv_dtype,
        o_dtype,
        mask_config,
    ):
        b_k, s_q, h_q, d_q = qo_shape_bshd
        b_q, s_k, h_k, d_k = kv_shape_bshd

        if qkv_dtype is cutlass.Float8E4M3:
            raise TypeError("use Float8E4M3FN instead of Float8E4M3")

        if not (d_q == d_k == self.headdim):
            raise ValueError(
                f"headdim_q({d_q}), headdim_k({d_k}) must be {self.headdim}"
            )

        if h_q % h_k != 0:
            raise ValueError(f"heads_q({h_q}) must be a multiple of heads_k({h_k})")

        if 0 < s_k < s_q:
            raise ValueError(
                f"non-zero seqlen({s_k}) must be at least prediction({s_q})"
            )

        if b_k != b_q:
            raise ValueError(f"batches_k({b_k}) and batches_q({b_q}) mismatch")

        if self.do_atomic_red:
            if kv_splits not in (1, 2, 4, 8, 16):
                raise ValueError(
                    f"atomic reduction requires kv_splits po2 <= 16, got {kv_splits}"
                )

            if o_dtype not in (Float32, BFloat16, Float16):
                raise TypeError(
                    f"atomic reduction requires (Float32, BFloat16, Float16) o_dtype, got {o_dtype}"
                )

        if self.do_none_red and kv_splits != 1:
            raise ValueError("KV splits must be 1 if flash decoding is disabled")

        mask_config.can_implement(s_q, s_k, self.prediction_tile, self.sequence_tile)

    @staticmethod
    def gqa_pack(t_bshd: cute.Tensor, h_k: int):
        d, h_q, s_q, b = tuple(reversed(t_bshd.shape))[:4]
        stride_d, stride_h, stride_s, stride_b = tuple(reversed(t_bshd.stride))[:4]
        has_partial = cute.rank(t_bshd) == 5
        b_partial = b * t_bshd.shape[0] if has_partial else b
        h_g = h_q // h_k
        gqa_shape = (b_partial, (h_g, s_q), h_k, d)
        gqa_stride = (stride_b, (stride_h, stride_s), stride_h * h_g, stride_d)
        gqa_layout = cute.make_layout(gqa_shape, stride=gqa_stride)
        return cute.make_tensor(t_bshd.iterator, gqa_layout)

    @staticmethod
    def gemm_view(t_bshd: cute.Tensor, s_first: bool):
        sdhb = (1, 3, 2, 0)
        dshb = (3, 1, 2, 0)
        reorder = sdhb if s_first else dshb
        mT_layout = cute.select(t_bshd.layout, reorder)
        mT_layout = cute.group_modes(mT_layout, 2, 4)
        return cute.make_tensor(t_bshd.iterator, mT_layout)

    @staticmethod
    def gemm_view_bsh(t_bsh: cute.Tensor, h_k: int):
        h_q, s_q, b = tuple(reversed(t_bsh.shape))[:3]
        stride_h, stride_s, stride_b = tuple(reversed(t_bsh.stride))[:3]
        h_g = h_q // h_k
        mT_shape = ((h_g, s_q), (h_k, b))
        mT_stride = ((stride_h, stride_s), (stride_h * h_g, stride_b))
        has_partial = cute.rank(t_bsh) == 4
        mT_shape += (t_bsh.shape[0],) if has_partial else ()
        mT_stride += (t_bsh.stride[0],) if has_partial else ()
        mT_layout = cute.make_layout(mT_shape, stride=mT_stride)
        return cute.make_tensor(t_bsh.iterator, mT_layout)
    @staticmethod
    @cute.jit
    def reduction_none(
        lane_store_max: bool,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        sM: cute.Tensor,
        sL: cute.Tensor,
        gL: Optional[cute.Tensor],
        sSink: Optional[cute.Tensor],
        scale_o: Float32,
    ):
        store_lse = gL is not None
        colmax = Float32(0)
        colsum = Float32(0)
        sM_final_nbar.arrive_and_wait()
        if cutlass.const_expr(store_lse):
            if lane_store_max:
                colmax = sM[lane_idx]
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane = sL[lane_idx, None]
            colsum = sL_lane[0] + sL_lane[1] + sL_lane[2] + sL_lane[3]
            if cutlass.const_expr(sSink is not None):
                colsum += exp2(log2_e * sSink[lane_idx] - colmax)
            normalization = cute.arch.rcp_approx(colsum) * scale_o
            sM[lane_idx] = normalization
        sM_final_nbar.arrive()
        if cutlass.const_expr(store_lse):
            if lane_store_max:
                gL[lane_idx] = colmax + cute.math.log2(colsum)

    @staticmethod
    @cute.jit
    def reduction_epilogue(
        blk_tile_hp: Tuple[int, int],
        coord_hp: Tuple[Int32, Int32],
        coord_hb: Tuple[Int32, Int32],
        kv_split_idx: Int32,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        sM: cute.Tensor,
        sL: cute.Tensor,
        mM_partial: cute.Tensor,
        mL_partial: cute.Tensor,
    ):
        coord_h = (coord_hp, coord_hb, kv_split_idx)
        gM_partial = cute.local_tile(mM_partial, (blk_tile_hp,), coord_h)
        gL_partial = cute.local_tile(mL_partial, (blk_tile_hp,), coord_h)

        blk_tile_h, blk_tile_p = blk_tile_hp
        blk_tile_n = blk_tile_h * blk_tile_p
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n

        grouped_heads, prediction = mM_partial.shape[0]
        cM = cute.make_identity_tensor(mM_partial.shape[0])
        cM = cute.local_tile(cM, blk_tile_hp, coord_hp)
        idx_hg, idx_p = cM[lane_idx]
        lane_store_max &= idx_hg < grouped_heads
        lane_store_max &= idx_p < prediction

        cute.arch.fence_acq_rel_cta()
        sM_final_nbar.arrive_and_wait()
        if lane_store_max:
            sM_lane = sM[lane_idx]
            gM_partial[lane_idx] = sM_lane

        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane_wg = sL[lane_idx, None]
            sL_lane = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]
            gL_partial[lane_idx] = sL_lane

    @staticmethod
    @cute.jit
    def reduction_cluster(
        blk_tile_n: int,
        kv_splits: Int32,
        kv_split_idx: Int32,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        reduction_mbars_ptr: cute.Pointer,
        sM: cute.Tensor,
        sL: cute.Tensor,
        sR: cute.Tensor,
        gL: Optional[cute.Tensor],
        sSink: Optional[cute.Tensor],
        scale_o: Float32,
    ):
        acc_dtype = sM.dtype
        colmax_bits = blk_tile_n * acc_dtype.width
        copy_vec_bits = min(colmax_bits, 128)
        dsmem_store_threads = colmax_bits // copy_vec_bits
        dsmem_store_values = copy_vec_bits // acc_dtype.width
        dsmem_store_atom_r = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyDsmemStoreOp(),
            acc_dtype,
            num_bits_per_copy=copy_vec_bits,
        )
        dsmem_store_r = cute.make_tiled_copy(
            dsmem_store_atom_r,
            cute.make_ordered_layout(
                (dsmem_store_threads, dsmem_store_values), order=(1, 0)
            ),
            (blk_tile_n,),
        )
        thr_store_r = dsmem_store_r.get_slice(lane_idx)
        tRsM = thr_store_r.partition_S(sM)  # (CPY, #CPY)
        tRsL = thr_store_r.partition_S(sL)  # (CPY, #CPY, warpgroup_warps)
        tRsR = thr_store_r.partition_S(sR)  # (CPY, #CPY, max_red_iters, 2)

        tRrM_shape = thr_store_r.partition_D(sM).shape
        tRrM_final = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        tRrM_prev = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        tRrL_final = cute.make_rmem_tensor(tRrM_shape, acc_dtype)

        cute.arch.fence_acq_rel_cta()
        sM_final_nbar.arrive_and_wait()
        is_reduction_lane = lane_idx < dsmem_store_threads
        if is_reduction_lane:
            tRrM_prev.store(tRsM.load())
            tRrM_final.store(tRrM_prev.load())

            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 0x01 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRsR_local = tRsR[None, None, i, 0]
                    tRsR_peer = cute.make_tensor(
                        cute.arch.map_dsmem_ptr(tRsR_local.iterator, peer_idx),
                        tRsR_local.layout,
                    )
                    local_mbar = reduction_mbars_ptr + i
                    peer_mbar = cute.arch.map_dsmem_ptr(local_mbar, peer_idx)
                    cute.copy(
                        dsmem_store_atom_r, tRrM_final, tRsR_peer, mbar_ptr=peer_mbar
                    )
                    cute.arch.fence_acq_rel_cta()  # dont reorder dsmem store after wait
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    tRrR = tRsR_local.load()
                    for j in cutlass.range_constexpr(cute.size(tRrM_final)):
                        tRrM_final[j] = cute.arch.fmax(tRrM_final[j], tRrR[j])

        sL_final_nbar.arrive_and_wait()
        if is_reduction_lane:
            colsum = tRsL[None, None, 0].load()
            for i in cutlass.range_constexpr(1, warpgroup_warps, 1):
                colsum += tRsL[None, None, i].load()

            correction = exp2(tRrM_prev.load() - tRrM_final.load())
            correction = correction.reshape(colsum.shape)
            colsum *= correction

            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 0x01 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRrL_local = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
                    tRrL_local.store(colsum)
                    tRsR_local = tRsR[None, None, i, 1]
                    tRsR_peer = cute.make_tensor(
                        cute.arch.map_dsmem_ptr(tRsR_local.iterator, peer_idx),
                        tRsR_local.layout,
                    )
                    local_mbar = reduction_mbars_ptr + max_reduction_iters + i
                    peer_mbar = cute.arch.map_dsmem_ptr(local_mbar, peer_idx)
                    cute.copy(
                        dsmem_store_atom_r, tRrL_local, tRsR_peer, mbar_ptr=peer_mbar
                    )
                    cute.arch.fence_acq_rel_cta()  # dont reorder dsmem store after wait
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    colsum += tRsR_local.load()

            if cutlass.const_expr(sSink is not None):
                tRsSink = thr_store_r.partition_S(sSink)  # (CPY, #CPY)
                tRrSink = tRsSink.load().reshape(tRrM_final.shape)
                sink_prob = exp2(log2_e * tRrSink - tRrM_final.load())
                colsum += sink_prob.reshape(colsum.shape)

            rcp_colsum = cute.make_rmem_tensor(colsum.shape, acc_dtype)
            for i in cutlass.range(cute.size(colsum.shape)):
                rcp_colsum[i] = cute.arch.rcp_approx(colsum[i])
            tRsM.store(correction * rcp_colsum.load() * scale_o)

            if cutlass.const_expr(gL is not None):
                tRrL_final.store(colsum)

        sM_final_nbar.arrive()

        if cutlass.const_expr(gL is not None):
            tRgL = thr_store_r.partition_D(gL)  # (CPY, #CPY=1)
            if kv_split_idx == 0 and is_reduction_lane:
                lse = tRrM_final.load() + cute.math.log2(tRrL_final.load())
                tRgL.store(lse)

    @staticmethod
    @cute.jit
    def launch_reduction(
        d_per_blk: int,
        o_bshd: cute.Tensor,
        l_bsh: Optional[cute.Tensor],
        o_partial_bshd: cute.Tensor,
        l_partial_bsh: cute.Tensor,
        m_partial_bsh: cute.Tensor,
        sink_h: Optional[cute.Tensor],
        scale_o: Float32,
        warp_parallel: cutlass.Constexpr,
        stream: cuda.CUstream,
        enable_pdl: bool = True,
    ):
        splits, b, s_q, h_q, d = o_partial_bshd.shape

        def reverse(t: cute.Tensor):
            modes = tuple(reversed(range(cute.rank(t))))
            layout = cute.select(t.layout, modes)
            return cute.make_tensor(t.iterator, layout)

        o_dhsb = reverse(o_bshd)
        l_hsb = reverse(l_bsh) if l_bsh is not None else None
        o_partial_dhsb = reverse(o_partial_bshd)
        l_partial_hsb = reverse(l_partial_bsh)
        m_partial_hsb = reverse(m_partial_bsh)

        d_per_thr = 32 // o_bshd.dtype.width
        thr_per_blk = d_per_blk // d_per_thr
        d_blks = cute.ceil_div(d, d_per_blk)
        smem_bytes = (splits * 2 + 2) * Float32.width // 8

        GroupedQueryAttentionDecode.reduction_kernel(
            (thr_per_blk, d_per_thr, d_per_blk),
            o_dhsb,
            l_hsb,
            o_partial_dhsb,
            l_partial_hsb,
            m_partial_hsb,
            sink_h,
            scale_o,
            warp_parallel,
        ).launch(
            grid=[d_blks, h_q * s_q, b],
            block=[thr_per_blk, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            smem=smem_bytes,
            min_blocks_per_mp=1,
            use_pdl=enable_pdl,
        )

    @staticmethod
    @cute.kernel
    def reduction_kernel(
        tile_d: cute.Tile,
        o_dhsb: cute.Tensor,
        l_hsb: Optional[cute.Tensor],
        o_partial_dhsb: cute.Tensor,
        l_partial_hsb: cute.Tensor,
        m_partial_hsb: cute.Tensor,
        sink_h: Optional[cute.Tensor],
        scale_o: Float32,
        warp_parallel: cutlass.Constexpr,
    ):
        thr_per_blk, d_per_thr, d_per_blk = tile_d
        d, h_q, s_q, b, splits = o_partial_dhsb.shape
        d_blk_idx, coord_hs, coord_b = cute.arch.block_idx()
        coord_h, coord_s = cute.idx2crd(coord_hs, (h_q, s_q))
        tidx, _, _ = cute.arch.thread_idx()

        not_oob_d = True
        if d % d_per_blk != 0:
            not_oob_d = d_blk_idx * d_per_blk + tidx * d_per_thr < d

        coord_o = (d_blk_idx, coord_h, coord_s, coord_b, None)
        gO = cute.local_tile(o_dhsb, (d_per_blk,), coord_o[:-1])
        gO_partial = cute.local_tile(o_partial_dhsb, (d_per_blk,), coord_o)

        gM_partial = cute.local_tile(m_partial_hsb, (1,), coord_o[1:])
        gL_partial = cute.local_tile(l_partial_hsb, (1,), coord_o[1:])
        gM_partial_0 = gM_partial[None, tidx]
        gL_partial_0 = gL_partial[None, tidx]

        smem_ptr = cute.arch.get_dyn_smem(Float32)
        partial_layout = cute.make_layout((1, splits))
        sM_partial = cute.make_tensor(smem_ptr, partial_layout)
        sL_partial = cute.make_tensor(smem_ptr + splits, partial_layout)
        sL_partial_0 = sL_partial[None, tidx]
        sM_partial_0 = sM_partial[None, tidx]

        scalar_layout = cute.make_layout(1)

        use_sink = sink_h is not None
        if cutlass.const_expr(use_sink):
            gSink = cute.local_tile(sink_h, (1,), (coord_h,))
            sSink = cute.make_tensor(smem_ptr + splits * 2, scalar_layout)

        cpasync_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32
        )

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        tv_layout = cute.make_ordered_layout((thr_per_blk, d_per_thr), order=(1, 0))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, (d_per_blk,))
        thr_copy = tiled_copy.get_slice(tidx)

        tCgO_partial = thr_copy.partition_S(gO_partial)
        tCgO_partial = tCgO_partial[None, 0, None]
        tCgO = thr_copy.partition_D(gO)
        tCgO = tCgO[None, 0]
        tCrO_final = cute.zeros_like(tCgO, Float32)

        cute.arch.fence_acq_rel_cta()
        cute.arch.griddepcontrol_wait()

        if tidx == 0:
            if cutlass.const_expr(use_sink):
                cute.copy(cpasync_atom, gSink, sSink)

        if tidx < splits:
            cute.copy(cpasync_atom, gL_partial_0, sL_partial_0)
            cute.copy(cpasync_atom, gM_partial_0, sM_partial_0)

        for split_idx in cutlass.range(thr_per_blk + tidx, splits, thr_per_blk):
            gL_partial_n = gL_partial[None, split_idx]
            sL_partial_n = sL_partial[None, split_idx]
            cute.copy(cpasync_atom, gL_partial_n, sL_partial_n)

            gM_partial_n = gM_partial[None, split_idx]
            sM_partial_n = sM_partial[None, split_idx]
            cute.copy(cpasync_atom, gM_partial_n, sM_partial_n)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        max_final = -Float32.inf
        sum_final = Float32(0)
        if cutlass.const_expr(warp_parallel):
            lane_idx = tidx % warp_threads
            split_idx_hi = lane_idx + warp_threads
            max_lane = -Float32.inf
            if lane_idx < splits:
                max_lane = sM_partial[0, lane_idx]
            if split_idx_hi < splits:
                max_lane = cute.arch.fmax(max_lane, sM_partial[0, split_idx_hi])
            max_final = warp_fmax(max_lane)
            sum_lane = Float32(0)
            correction_lo = Float32(0)
            correction_hi = Float32(0)
            if max_final > -Float32.inf:
                if cutlass.const_expr(use_sink):
                    if lane_idx == 0:
                        sum_lane = exp2(log2_e * sSink[0] - max_final)
                if lane_idx < splits:
                    max_partial = sM_partial[0, lane_idx]
                    if max_partial > -Float32.inf:
                        correction_lo = exp2(max_partial - max_final)
                        sum_lane += correction_lo * sL_partial[0, lane_idx]
                if split_idx_hi < splits:
                    max_partial = sM_partial[0, split_idx_hi]
                    if max_partial > -Float32.inf:
                        correction_hi = exp2(max_partial - max_final)
                        sum_lane += correction_hi * sL_partial[0, split_idx_hi]
            sum_final = cute.arch.warp_reduction_sum(sum_lane)

            if max_final > -Float32.inf and not_oob_d:
                # A flat unroll hint whose depth does not divide the trip
                # count leaves a poorly scheduled remainder loop (measured
                # +40..90% reduction time at splits 9/12/24 on sm_103), so
                # the deeper 16-split pipelining is used only when it tiles
                # the split count exactly; 8 divides every remainder-free
                # trip count the policy emits below 16.
                if splits % 16 == 0:
                    for split_idx in cutlass.range(splits, unroll=16):
                        correction = Float32(0)
                        if split_idx < warp_threads:
                            correction = cute.arch.shuffle_sync(
                                correction_lo, split_idx
                            )
                        else:
                            correction = cute.arch.shuffle_sync(
                                correction_hi, split_idx - warp_threads
                            )
                        tCrO_final += (
                            correction * tCgO_partial[None, split_idx].load()
                        )
                else:
                    for split_idx in cutlass.range(splits, unroll=8):
                        correction = Float32(0)
                        if split_idx < warp_threads:
                            correction = cute.arch.shuffle_sync(
                                correction_lo, split_idx
                            )
                        else:
                            correction = cute.arch.shuffle_sync(
                                correction_hi, split_idx - warp_threads
                            )
                        tCrO_final += (
                            correction * tCgO_partial[None, split_idx].load()
                        )
                tCrO_final *= cute.arch.rcp_approx(sum_final) * scale_o
        else:
            for split_idx in cutlass.range(splits, unroll=8):
                max_final = cute.arch.fmax(max_final, sM_partial[0, split_idx])
            if cutlass.const_expr(use_sink):
                sum_final = exp2(log2_e * sSink[0] - max_final)
            if max_final > -Float32.inf and not_oob_d:
                if splits % 16 == 0:
                    for split_idx in cutlass.range(splits, unroll=16):
                        max_partial = sM_partial[0, split_idx]
                        if max_partial > -Float32.inf:
                            correction = exp2(max_partial - max_final)
                            sum_final += correction * sL_partial[0, split_idx]
                            tCrO_final += (
                                correction * tCgO_partial[None, split_idx].load()
                            )
                else:
                    for split_idx in cutlass.range(splits, unroll=8):
                        max_partial = sM_partial[0, split_idx]
                        if max_partial > -Float32.inf:
                            correction = exp2(max_partial - max_final)
                            sum_final += correction * sL_partial[0, split_idx]
                            tCrO_final += (
                                correction * tCgO_partial[None, split_idx].load()
                            )
                tCrO_final *= cute.arch.rcp_approx(sum_final) * scale_o

        cute.arch.griddepcontrol_launch_dependents()

        if not_oob_d:
            tCgO.store(tCrO_final.to(o_dhsb.dtype))

        if cutlass.const_expr(l_hsb is not None):
            if d_blk_idx == 0 and tidx == 0:
                l_hsb[coord_h, coord_s, coord_b] = max_final + cute.math.log2(sum_final)

        return


