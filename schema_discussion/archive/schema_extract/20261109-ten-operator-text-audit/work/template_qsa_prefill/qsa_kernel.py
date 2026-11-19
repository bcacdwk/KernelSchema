"""Native CuTeDSL core-QSA sparse paged GQA forward for B300 / sm_103.

V3: persistent warp-specialized row-block-union kernel.

Mapping (unchanged from V1/V2): the work unit is one (row block of ``br``
consecutive query rows, kv head) pair.  The MMA M side packs the block's
``br * group`` real (row, head) lanes (lane = r*group + h, padded to 128) and
the N side walks the *union* of the KV tokens selected by the block's rows,
precomputed by ``qsa_preprocess.QSAPreprocessUnion`` as (req, group,
nibble-mask) entries of 4 consecutive logical tokens each.  An N tile holds
``ent_tile`` entries, i.e. ``n_tile = 4 * ent_tile`` tokens.

Per-column softmax masking uses the entry nibble pack: bit (4*r + l) of the
entry's 64-bit mask is set iff row r of the block selected token 4*g + l, so
each (row, head) lane attends exactly to its own packed valid selection, in
union order, replicating reference.py semantics (invalid selections never set
bits; rows whose selection is empty accumulate row_sum=0 -> inv=0 -> zeros).

Why V3 is persistent.  V2 launched one CTA per unit (820 CTAs of ~196KB SMEM
+ 512 TMEM cols => exactly 1 CTA/SM, 5.54 sequential waves).  NCU cap-probes
measured ~76k locked cycles of per-CTA fixed cost (exposed Q-gather latency,
the serialized SMEM-staging epilogue, role joins and TMEM re-alloc) against
~1018 cycles of steady-state per-tile tensor-core time: fixed cost was ~40%
of the main kernel.  V3 launches ``min(#SMs, #units)`` CTAs; each CTA loops
over a strided, heaviest-first unit assignment and pipelines ACROSS units:

* the K/V gather warps keep filling the ring with the next unit's tiles as
  soon as stages free (no unit-boundary drain),
* the softmax warpgroup refills sQ with the next unit's Q rows right after
  the current unit's last QK retired, overlapping its own epilogue,
* the MMA warp gates only the next unit's first PV on ``odrained`` (the O
  accumulator fully read into registers by the epilogue); QKs run ahead,
* TMEM is allocated once per CTA; the epilogue writes gated rows directly to
  ``out`` with vectorized per-thread row stores (no SMEM staging, no scalar
  U16 copy-out pass).

Roles: warps 0-7 are TWO column-split softmax warpgroups (each covers the
same 128 TMEM lanes/rows but half of each tile's N columns, halving the
quarter-rate MUFU exp2 load per scheduler, the t2r/t2st width and the
register footprint; the per-row tile max is exchanged through SMEM under a
256-thread named barrier, and only WG-A rescales O / refills sQ), warp 8 is
the MMA warp (QK two tiles ahead of PV), warps 9/10 are the K and V gather
warps (cp.async paged gather into a ``kv_stages`` ring released by deferred
tcgen05 commits, with a 3-stage mWS->mBT->address resolution pipeline), and
warp 11 is the unit-ticket scheduler (one global atomicAdd per unit; the
host passes a wrap-safe per-launch ticket base).

All ring barrier phases derive from a CTA-cumulative tile index, so the
pipeline flows seamlessly across unit boundaries; qready/odrained use the
CTA-local unit ordinal parity.  Unit u maps to block ``nblocks-1-(u>>1)`` and
kv head ``u&1``: heaviest-first (the fitted law's regime-change trailing rows
have the largest unions) and both heads of one block on neighboring CTAs for
L2 locality of the paged gather.

Numerics match reference.py: fp32 QK/PV accumulate, softmax over each row's
valid set, BF16 rounding of the attention result before the fp32 sigmoid
gate, BF16 store into ``out``.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import BFloat16, Float32, Int32, Int64

LOG2_E = 1.4426950408889634

# Softmax is exactly invariant to the reference max: scaling every probability
# of a row by one common power of two leaves O/row_sum unchanged.  The
# reference therefore only moves when a tile max runs more than
# RESCALE_THRESH log2 units ahead, which bounds probabilities by
# 2**RESCALE_THRESH (no bf16 P or fp32 O overflow) and turns the 128x256 fp32
# TMEM accumulator rescale from a per-tile cost into a rare correction path.
RESCALE_THRESH = 8.0


@cute.jit
def _elem_pointer(x: cute.Tensor, coord) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout)


@cute.jit
def _shfl(value: Int64, src_lane: Int32, width: cutlass.Constexpr) -> Int64:
    mask_and_clamp = ((32 - width) << 8) | 31
    return Int64(cute.arch.shuffle_sync(value, src_lane, mask_and_clamp=mask_and_clamp))


class QSAForward:
    SOFTMAX_WARPS = 8      # warps 0..7: two column-split softmax warpgroups
    MMA_WARP = 8           # warp 8: issues every tcgen05 MMA
    KLOAD_WARP = 9         # warp 9: K ring producer
    VLOAD_WARP = 10        # warp 10: V ring producer
    SCHED_WARP = 11        # warp 11: unit-ticket scheduler
    NWARPS = 12

    def __init__(self, group: int, head_dim: int, page_size: int, sel_width: int,
                 bt_width: int, br: int, maxe: int,
                 m_tile: int = 128, n_tile: int = 64, ent_tile: int = 16,
                 kv_stages: int = 2, s_stages: int = 3):
        assert n_tile == 4 * ent_tile, "one union entry expands to 4 tokens"
        assert ent_tile <= 32, "one gather-warp lane resolves one union entry"
        assert n_tile * s_stages + head_dim <= 512, "TMEM column overflow"
        self.group = group
        self.head_dim = head_dim
        self.page_size = page_size
        self.sel_width = sel_width
        self.bt_width = bt_width
        self.br = br            # rows per CTA block
        self.maxe = maxe        # union entry bound per block
        self.m_tile = m_tile
        self.n_tile = n_tile    # tokens per tile
        self.ent_tile = ent_tile  # union entries per tile
        self.kv_stages = kv_stages
        self.s_stages = s_stages
        self.threads = 32 * self.NWARPS
        self.dtype = BFloat16
        self.acc = Float32
        self.arch_str = "sm_100"
        self.copy_elems = 8                 # 128 bits / 16 bits
        self.gptr = 8                       # threads cooperating per token row

    @cute.jit
    def __call__(self, mQ, mK, mV, mWS, mCount, mTicket, mBT, mGate, mOut,
                 softmax_scale: Float32, nblocks: Int32, ncta: Int32,
                 tbase: Int32, stream):
        self.scale_log2 = softmax_scale * Float32(LOG2_E)
        M, N, D = self.m_tile, self.n_tile, self.head_dim
        qk_tiler = (M, N, D)
        pv_tiler = (M, D, N)
        qk_mma = sm100_utils.make_trivial_tiled_mma(
            self.dtype, self.dtype, tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K, self.acc, tcgen05.CtaGroup.ONE, qk_tiler[:2])
        # The PV A operand (P) is read from TMEM, aliasing the drained S stage.
        pv_mma = sm100_utils.make_trivial_tiled_mma(
            self.dtype, self.dtype, tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN, self.acc, tcgen05.CtaGroup.ONE,
            pv_tiler[:2], tcgen05.OperandSource.TMEM)
        q_lay = sm100_utils.make_smem_layout_a(qk_mma, qk_tiler, self.dtype, 1)
        k_lay = sm100_utils.make_smem_layout_b(qk_mma, qk_tiler, self.dtype,
                                               self.kv_stages)
        v_lay = sm100_utils.make_smem_layout_b(pv_mma, pv_tiler, self.dtype,
                                               self.kv_stages)
        # Single-stage P layout: the stage stride is applied by hand in the
        # kernel, because make_smem_layout_a packs stages back-to-back (N/2
        # columns apart) while the S/P ping-pong stages are N columns apart.
        p_tmem_lay = sm100_utils.make_smem_layout_a(pv_mma, pv_tiler, self.dtype, 1)
        self.kernel(qk_mma, pv_mma, q_lay, k_lay, v_lay, p_tmem_lay,
                    mQ, mK, mV, mWS, mCount, mTicket, mBT, mGate, mOut,
                    self.scale_log2, nblocks, ncta, tbase).launch(
            grid=(ncta, 1, 1), block=[self.threads, 1, 1],
            cluster=(1, 1, 1), stream=stream, min_blocks_per_mp=1)

    # ------------------------------------------------------------------ device
    @cute.kernel
    def kernel(self, qk_mma, pv_mma, q_lay, k_lay, v_lay, p_tmem_lay,
               mQ, mK, mV, mWS, mCount, mTicket, mBT, mGate, mOut,
               scale_log2: Float32, nblocks: Int32, ncta: Int32,
               tbase: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // 32)
        cta, _, _ = cute.arch.block_idx()
        M, N, D = self.m_tile, self.n_tile, self.head_dim
        group, page = self.group, self.page_size
        br, maxe, etile = self.br, self.maxe, self.ent_tile
        kvs, ssts = self.kv_stages, self.s_stages
        sm_threads = 32 * self.SOFTMAX_WARPS

        Hkv = Int32(mK.shape[2])
        total_units = nblocks * Hkv

        @cute.struct
        class SharedStorage:
            kfull: cute.struct.MemRange[Int64, kvs]
            kfree: cute.struct.MemRange[Int64, kvs]
            vfull: cute.struct.MemRange[Int64, kvs]
            vfree: cute.struct.MemRange[Int64, kvs]
            sfull: cute.struct.MemRange[Int64, ssts]
            pfull: cute.struct.MemRange[Int64, ssts]
            pvdone: cute.struct.MemRange[Int64, 4]
            qready: Int64
            odrained: Int64
            unit_posted: Int64
            unit_taken: Int64
            sUnit: cute.struct.MemRange[Int32, 1]
            sMax: cute.struct.MemRange[Float32, 256]   # per-tile half-max swap
            sSum: cute.struct.MemRange[Float32, 256]   # epilogue row_sum swap
            tmem_holding_buf: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = smem.allocate_tensor(self.dtype, q_lay.outer, 128, q_lay.inner)
        sK = smem.allocate_tensor(self.dtype, k_lay.outer, 128, k_lay.inner)
        sV = smem.allocate_tensor(self.dtype, v_lay.outer, 128, v_lay.inner)

        kfull = storage.kfull.data_ptr()
        kfree = storage.kfree.data_ptr()
        vfull = storage.vfull.data_ptr()
        vfree = storage.vfree.data_ptr()
        sfull = storage.sfull.data_ptr()
        pfull = storage.pfull.data_ptr()
        pvdone = storage.pvdone.data_ptr()
        qready = storage.qready.ptr
        odrained = storage.odrained.ptr
        unit_posted = storage.unit_posted.ptr
        unit_taken = storage.unit_taken.ptr
        sUnit = storage.sUnit.data_ptr()
        sMax = storage.sMax.data_ptr()
        sSum = storage.sSum.data_ptr()
        if tidx == 0:
            for i in cutlass.range_constexpr(kvs):
                cute.arch.mbarrier_init(kfull + i, 32)
                cute.arch.mbarrier_init(kfree + i, 1)
                cute.arch.mbarrier_init(vfull + i, 32)
                cute.arch.mbarrier_init(vfree + i, 1)
            for i in cutlass.range_constexpr(ssts):
                cute.arch.mbarrier_init(sfull + i, 1)
                cute.arch.mbarrier_init(pfull + i, sm_threads)
            for i in cutlass.range_constexpr(4):
                cute.arch.mbarrier_init(pvdone + i, 1)
            cute.arch.mbarrier_init(qready, 128)   # WG-A only refills sQ
            cute.arch.mbarrier_init(odrained, sm_threads)
            cute.arch.mbarrier_init(unit_posted, 1)
            cute.arch.mbarrier_init(unit_taken, 32 * (self.NWARPS - 1))
            cute.arch.mbarrier_init_fence()

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr,
                                   barrier_for_retrieve=tmem_bar)
        tmem.allocate(cute.arch.get_max_tmem_alloc_cols(self.arch_str))
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc)

        qk_thr = qk_mma.get_slice(0)
        pv_thr = pv_mma.get_slice(0)
        # TMEM map: S/P stages own columns [0, N*ssts), O owns the next D.
        # make_fragment_C derives the TMEM lane/column encoding; the stage mode
        # is appended with an N-column stride so stage s sits at column s*N.
        s_frag = qk_thr.make_fragment_C(cute.make_layout(qk_thr.partition_shape_C((M, N))))
        s_staged = cute.append(s_frag.layout, cute.make_layout((ssts,), stride=(N,)))
        tStS = cute.make_tensor(tmem_ptr, s_staged)
        o_frag = pv_thr.make_fragment_C(cute.make_layout(pv_thr.partition_shape_C((M, D))))
        tOtO = cute.make_tensor(tmem_ptr + N * ssts, o_frag.layout)

        tSrQ = qk_thr.make_fragment_A(sQ)
        tSrK = qk_thr.make_fragment_B(sK)
        tOrV = pv_thr.make_fragment_B(sV)
        # PV A operand: bf16 P packed two-per-word inside the S TMEM columns.
        # The fragment layout is built for one stage, then a stage mode with the
        # real S/P ping-pong stride is appended: P stage s lives at TMEM column
        # s*N, which is s*N*2 bf16 units from the base.
        tP_tm = cute.make_tensor(tStS.iterator, p_tmem_lay.outer)
        tOrP_1 = pv_thr.make_fragment_A(tP_tm)[(None, None, None, 0)]
        tOrP = cute.make_tensor(
            cute.recast_ptr(tStS.iterator, dtype=self.dtype),
            cute.append(tOrP_1.layout, cute.make_layout((ssts,), stride=(N * 2,))))

        # ---- 128-thread gather copy: Q refills (softmax warpgroup) ----
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype, num_bits_per_copy=128)
        thr_layout = cute.make_ordered_layout((128 // self.gptr, self.gptr),
                                              order=(1, 0))
        val_layout = cute.make_layout((1, self.copy_elems))
        gcopy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)
        gthr = gcopy.get_slice(tidx % 128)
        sQf = self._flatten(sQ, False)                 # [M, D] (lane, dim)
        tQsQ = gthr.partition_D(sQf)
        cMD = cute.make_identity_tensor((M, D))
        tQcQ = gthr.partition_S(cMD)
        tQc0 = gcopy.get_slice(0).partition_S(cMD)

        # ---- 32-thread gather copy: one warp per K/V ring ----
        kv_thr_layout = cute.make_ordered_layout((32 // self.gptr, self.gptr),
                                                 order=(1, 0))
        kvgcopy = cute.make_tiled_copy_tv(atom, kv_thr_layout, val_layout)
        lane = tidx % 32
        kvgthr = kvgcopy.get_slice(lane)

        voff = _elem_pointer(mV, (0, 0, 0, 0)).toint() - \
            _elem_pointer(mK, (0, 0, 0, 0)).toint()
        st1b = Int64(mK.stride[1]) * Int64(2)  # bytes per token step in a page

        cute.arch.sync_threads()   # mbarrier init + TMEM ptr visible to all

        # ==================== role split =====================================
        if warp_idx < self.SOFTMAX_WARPS:
            self._softmax_role(
                tidx, warp_idx // 4, unit_posted, unit_taken, sUnit,
                total_units, nblocks,
                mQ, mWS, mCount, mGate, mOut,
                tStS, tOtO, sfull, pfull, pvdone, qready, odrained,
                sMax, sSum,
                sQf, gcopy, gthr, tQsQ, tQcQ, tQc0,
                scale_log2, M, N, D, group, br, maxe, etile, ssts)
        else:
            if warp_idx == self.MMA_WARP:
                self._mma_role(unit_posted, unit_taken, sUnit, total_units,
                               nblocks, mCount,
                               qk_mma, pv_mma, tStS, tOtO, tSrQ, tSrK,
                               tOrP, tOrV, kfull, kfree, vfull, vfree,
                               sfull, pfull, pvdone, qready, odrained,
                               etile, kvs, ssts)
            elif warp_idx == self.KLOAD_WARP:
                self._gather_role(mK, sK, kvgcopy, kvgthr, mWS, mBT, mCount,
                                  unit_posted, unit_taken, sUnit, total_units,
                                  nblocks, page, D, Int64(0), st1b, kfull,
                                  kfree, kvs, lane, maxe, etile, False)
            elif warp_idx == self.VLOAD_WARP:
                self._gather_role(mK, sV, kvgcopy, kvgthr, mWS, mBT, mCount,
                                  unit_posted, unit_taken, sUnit, total_units,
                                  nblocks, page, D, voff, st1b, vfull, vfree,
                                  kvs, lane, maxe, etile, True)
            elif warp_idx == self.SCHED_WARP:
                # ---- scheduler warp: one thread hands out unit tickets ----
                if tidx == 32 * self.SCHED_WARP:
                    term = False
                    ph = Int32(0)
                    while not term:
                        # Tickets grow monotonically across launches; the
                        # host passes this launch's base (int32 wrap-safe).
                        u_t = cute.arch.atomic_add(
                            mTicket.iterator, Int32(1), sem="relaxed",
                            scope="gpu") - tbase
                        (sUnit).store(u_t)
                        cute.arch.mbarrier_arrive(unit_posted)
                        cute.arch.mbarrier_wait(unit_taken, ph & 1)
                        term = u_t >= total_units
                        ph += 1

        cute.arch.sync_threads()   # all roles done; TMEM can be released
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    # ------------------------------------------------------------ unit utils
    @cute.jit
    def _unit_blk(self, u, nblocks):
        """Heaviest-first: the fitted law's regime-change trailing blocks have
        the largest unions, so unit u walks block nblocks-1-(u>>1)."""
        return nblocks - 1 - (u >> 1)

    # ------------------------------------------------------------ softmax WGs
    @cute.jit
    def _softmax_role(self, tidx, wg, unit_posted, unit_taken, sUnit,
                      total_units, nblocks,
                      mQ, mWS, mCount, mGate, mOut,
                      tStS, tOtO, sfull, pfull, pvdone, qready, odrained,
                      sMax, sSum,
                      sQf, gcopy, gthr, tQsQ, tQcQ, tQc0,
                      scale_log2, M, N, D, group, br, maxe, etile, ssts):
        """Two warpgroups split every N-tile's columns 50/50.

        Both WGs cover the same 128 TMEM lanes (rows): WG w reads/writes the
        (M, N/2) column subtile ``w`` of each S stage, so the per-scheduler
        MUFU exp2 load, t2r/t2st width and register footprint all halve while
        the tile stays N=64 (SMEM ring budget).  The per-row tile max is
        exchanged through sMax under a 256-thread named barrier; the k-th
        barrier completion pairs the two WGs' k-th stores, and a WG can never
        run more than one tile ahead (sfull/S-stage aliasing bounds it), so
        the exchange can never mis-pair.  P store: each WG packs its N/2
        probabilities into its half of the stage's 32 packed TMEM word
        columns; pfull needs all 256 arrivals before the PV MMA reads P.
        Only WG-A rescales O (it owns the warp-collective TMEM RMW over all
        128 lanes), refills sQ and publishes qready; per-warp vote bits are
        identical across the two WGs because both compute the same per-row
        tile_max and mref.
        """
        rows = Int32(mQ.shape[0])
        sg = tidx % 128                     # warpgroup-local thread / M lane
        isA = wg == 0
        ncolh = N // 2                      # columns per WG (32)
        ent_h = ncolh // 4                  # entries per WG per tile (8)

        # ---- fetch the first unit ticket ----
        ph = Int32(0)
        cute.arch.mbarrier_wait(unit_posted, ph & 1)
        u = Int32(sUnit.load())
        cute.arch.mbarrier_arrive(unit_taken)

        # ---- WG-A: prologue Q gather, then publish sQ ----
        u0 = cutlass.min(u, total_units - 1)
        blk0 = self._unit_blk(u0, nblocks)
        kvh0 = u0 & 1
        row0_0 = blk0 * br
        r_act0 = cutlass.min(rows - row0_0, Int32(br))
        lane_rows0 = group * r_act0
        if isA:
            self._load_q(mQ, sQf, tQsQ, tQcQ, tQc0, gcopy,
                         row0_0, kvh0, group, lane_rows0)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(qready)

        # ---- S t2r over this WG's (M, N/2) column subtile ----
        s_slice = tStS[(None, None), 0, 0, 0]
        s_half = cute.flat_divide(s_slice, (M, ncolh))   # (M,32, 1, 2)
        ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)),
                                      self.acc)
        tiled_ld = tcgen05.make_tmem_copy(ld_atom, s_half[(None, None, 0, 0)])
        thr_ld = tiled_ld.get_slice(sg)
        cS = cute.make_identity_tensor((M, N))
        cS_h = cute.flat_divide(cS, (M, ncolh))
        tTR_cS = thr_ld.partition_D(cS_h)
        rS = cute.make_rmem_tensor(tTR_cS[(None, None, None, 0, 0)].shape,
                                   self.acc)
        # subtile-local (row, col) coords are identical for both WGs
        tTR_cS = tTR_cS[(None, None, None, 0, 0)]

        # ---- P store path: this WG's half of the packed word columns ----
        p_words = N // 32 * 16                  # 32 packed words per stage
        pw_h = p_words // 2                     # 16 words per WG
        sP_lay = cute.composition(s_slice.layout, cute.make_layout((M, pw_h)))
        cS_P = cute.make_tensor(
            cS.iterator, cute.composition(cS.layout, cute.make_layout((M, pw_h))))
        st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(16)),
                                      self.acc)
        tiled_st = tcgen05.make_tmem_copy(
            st_atom, cute.make_tensor(s_slice.iterator, sP_lay))
        thr_st = tiled_st.get_slice(sg)
        tST_cP = thr_st.partition_S(cS_P)
        rP = cute.make_rmem_tensor(rS.shape, self.dtype)
        rP_w = cute.make_tensor(cute.recast_ptr(rP.iterator, dtype=self.acc),
                                tST_cP.shape)
        nchunk = cute.size(rS) // 32
        rS_c = cute.logical_divide(rS, cute.make_layout(32))
        rP_c = cute.logical_divide(rP, cute.make_layout(32))

        row_r = sg // group
        sh4 = Int64(row_r * 4)
        low_half = sh4 < Int64(32)                 # rows 0..7 -> lo, 8..9 -> hi
        sh_lo = cutlass.select_(low_half, sh4, sh4 - Int64(32))
        wsel = Int32(2) + cutlass.select_(low_half, Int32(0), Int32(1))
        sh_lo32 = Int32(sh_lo)
        # this WG's entries inside each tile: WG-A entries 0..7, WG-B 8..15
        wg_ent_off = Int32(wg * ent_h)

        L = sg                                     # this thread's M lane
        # ---- O epilogue: each WG drains half of the D columns ----
        o_slice = tOtO[(None, None), 0, 0]
        epi_tile = (M, 32)
        o_epi = cute.flat_divide(o_slice, epi_tile)     # (M,32, 1, D/32)
        o_ld = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)),
                                   self.acc)
        o_tiled = tcgen05.make_tmem_copy(o_ld, o_epi[(None, None, 0, 0)])
        o_thr = o_tiled.get_slice(sg)
        cO = cute.make_identity_tensor((M, D))
        cO_epi = cute.flat_divide(cO, epi_tile)
        tO = o_thr.partition_S(o_epi)
        tcO = o_thr.partition_D(cO_epi)
        nsub_all = cute.size(tO, mode=[4])         # D/32 subtiles total
        nsub = nsub_all // 2                       # per WG
        nsub_c = cute.size(tcO[(None, None, None, 0, 0)])

        gt = Int32(0)        # cumulative tiles completed before this unit
        while u < total_units:
            blk = self._unit_blk(u, nblocks)
            kvh = u & 1
            row0 = blk * br
            r_act = cutlass.min(rows - row0, Int32(br))
            lane_rows = group * r_act
            count = Int32(mCount[blk])
            num_tiles = (count + etile - 1) // etile
            ws_base = (blk * maxe) * 4
            wbase = ws_base + wsel

            # ---- per-unit softmax state ----
            mref = Float32(-cutlass.Float32.inf)
            row_sum = Float32(0.0)
            last_base = (num_tiles - 1) * etile
            # raw nibble words are LDG'd one tile ahead and packed on use
            cbw = self._colbits2(mWS, wbase, count, sh_lo32,
                                 wg_ent_off, ent_h, last_base + wg_ent_off, 1)

            for i in cutlass.range(num_tiles):
                g = gt + i
                s = g % ssts
                ebase = i * etile
                pbase = cutlass.min(ebase + etile, last_base)
                # issue tile i+1's raw nibble LDGs early; they land under this
                # tile's softmax and are packed at the end of the iteration
                nrw = self._colbits_raw(mWS, wbase,
                                        pbase + wg_ent_off, 1)
                colbits = cbw

                cute.arch.mbarrier_wait(sfull + s, (g // ssts) & 1)
                # partition the LIVE stage slice every tile (the stage
                # index lives in the tensor, not just the pointer)
                s_h = cute.flat_divide(tStS[(None, None), 0, 0, s],
                                       (M, ncolh))
                cute.copy(tiled_ld,
                          thr_ld.partition_S(s_h)[(None, None, None, 0, wg)],
                          rS)
                cute.arch.fence_view_async_tmem_load()

                # ---- half-tile max, exchanged with the sibling warpgroup.
                # Masked columns still hold genuine q.k scores (dead entries
                # duplicate entry 0), so the reference is a finite in-range
                # value and shifting it only multiplies every probability of
                # the row by one common power of two. ----
                part_max = rS.load().reduce(cute.ReductionOp.MAX,
                                            Float32(-cutlass.Float32.inf),
                                            0) * scale_log2
                (sMax + wg * 128 + sg).store(part_max)
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
                other = (sMax + (1 - wg) * 128 + sg).load()
                tile_max = cutlass.max(part_max, other)
                first = mref == Float32(-cutlass.Float32.inf)
                drift = cutlass.select_(first, Float32(0.0), tile_max - mref)
                # warp-uniform and identical in both WGs (same rows, same
                # tile_max): the rescale stays a warp-collective RMW.
                vote = cute.arch.vote_any_sync(drift > Float32(RESCALE_THRESH))
                new_ref = cutlass.select_(vote, tile_max, mref)
                new_ref = cutlass.select_(first, tile_max, new_ref)
                neg = Float32(0.0) - new_ref

                # ---- probabilities against the (rarely moved) reference ----
                for j in cutlass.range_constexpr(nchunk):
                    for kk in cutlass.range_constexpr(32):
                        idx = j * 32 + kk
                        col = tTR_cS[idx][1]
                        try:
                            col_py = int(col)
                        except Exception:
                            col_py = None
                        if cutlass.const_expr(col_py is not None):
                            valid = ((colbits[col_py >> 5] >> (col_py & 31)) & Int32(1)) != 0
                        else:
                            valid = Int32(1) != 0
                        p = cute.math.exp2(rS[idx] * scale_log2 + neg, fastmath=True)
                        rS[idx] = cutlass.select_(valid, p, Float32(0.0))
                tsum = rS.load().reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                for j in cutlass.range_constexpr(nchunk):
                    rP_c[None, j].store(rS_c[None, j].load().to(self.dtype))

                # ---- O quiescence is only needed for the rare rescale, and
                # only WG-A performs it (both WGs' pfull arrivals gate the PV
                # MMA, and WG-A arrives after its rescale in program order).
                # Ring-slot parity is unambiguous: the next commit to slot
                # (g-1)&3 is PV(g+3), gated on P(g+3) from this WG. ----
                if i > 0:
                    if vote:
                        if isA:
                            cute.arch.mbarrier_wait(pvdone + ((g - 1) & 3),
                                                    ((g - 1) // 4) & 1)
                if vote:
                    alpha = cute.math.exp2(mref - new_ref, fastmath=True)
                    if isA:
                        self._rescale_O(tOtO, alpha, sg, M, D)
                        cute.arch.fence_view_async_tmem_store()
                    row_sum = row_sum * alpha
                mref = new_ref
                row_sum = row_sum + tsum

                sP_st = cute.make_tensor(
                    s_slice.iterator + Int32(s) * Int32(N)
                    + Int32(wg) * Int32(pw_h), sP_lay)
                cute.copy(tiled_st, rP_w, thr_st.partition_D(sP_st))
                cute.arch.fence_view_async_tmem_store()
                cute.arch.mbarrier_arrive(pfull + s)
                cbw = self._colbits_pack(nrw, count, sh_lo32,
                                         pbase + wg_ent_off, ent_h, 1)

            gt_next = gt + num_tiles

            # ---- unit epilogue: fetch the next ticket; WG-A refills sQ
            # (its last QK retired and the next QK waits on qready). ----
            ph += 1
            cute.arch.mbarrier_wait(unit_posted, ph & 1)
            u2 = Int32(sUnit.load())
            cute.arch.mbarrier_arrive(unit_taken)
            has_next = u2 < total_units
            u_n = cutlass.min(u2, total_units - 1)
            blk_n = self._unit_blk(u_n, nblocks)
            kvh_n = u_n & 1
            row0_n = blk_n * br
            r_act_n = cutlass.min(rows - row0_n, Int32(br))
            lane_rows_n = group * r_act_n
            if has_next:
                if isA:
                    self._load_q(mQ, sQf, tQsQ, tQcQ, tQc0, gcopy,
                                 row0_n, kvh_n, group, lane_rows_n)
                    cute.arch.cp_async_commit_group()

            # ---- combine the two half row_sums ----
            (sSum + wg * 128 + sg).store(row_sum)
            cute.arch.barrier(barrier_id=3, number_of_threads=256)
            row_sum_t = row_sum + (sSum + (1 - wg) * 128 + sg).load()
            inv = cute.arch.rcp_approx(row_sum_t) if row_sum_t != Float32(0.0) else Float32(0.0)
            empty_row = row_sum_t == Float32(0.0)
            if gt_next > 0:
                # last PV of this unit retired (ring-phase argument as above)
                cute.arch.mbarrier_wait(pvdone + ((gt_next - 1) & 3),
                                        ((gt_next - 1) // 4) & 1)

            # ---- drain this WG's half of O through the sigmoid gate with
            # vectorized per-thread row stores (thread L owns output row L,
            # subtiles wg*4..wg*4+3 cover columns wg*128..wg*128+127). ----
            valid_lane = L < lane_rows
            rr = cutlass.min(row_r, Int32(br) - 1)
            hh = cutlass.min(L - row_r * group, group - 1)
            gate_addr = _elem_pointer(mGate, (row0 + rr, kvh * group + hh, 0)).toint()
            out_addr = _elem_pointer(mOut, (row0 + rr, kvh * group + hh, 0)).toint()
            rO = cute.make_rmem_tensor(tcO[(None, None, None, 0, 0)].shape, self.acc)
            rG = cute.make_rmem_tensor((nsub_c,), self.dtype)
            rW = cute.make_rmem_tensor((nsub_c,), self.dtype)
            for jl in cutlass.range_constexpr(nsub):
                j = wg * nsub + jl
                # warp-collective TMEM load: executed uniformly by all threads
                cute.copy(o_tiled, tO[(None, None, None, 0, j)], rO)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(jl == nsub - 1):
                    # this WG's O half is in registers
                    cute.arch.mbarrier_arrive(odrained)
                    if isA:
                        if has_next:
                            cute.arch.cp_async_wait_group(0)
                            cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_arrive(qready)
                if valid_lane:
                    gj = j * nsub_c
                    gG = cute.make_tensor(
                        cute.make_ptr(self.dtype, gate_addr + Int64(gj * 2),
                                      cute.AddressSpace.gmem, assumed_align=16),
                        cute.make_layout((nsub_c,)))
                    cute.autovec_copy(gG, rG)
                    for e in cutlass.range_constexpr(nsub_c):
                        o = Float32(cutlass.select_(empty_row, Float32(0.0),
                                                    rO[e] * inv))
                        attn = o.to(self.dtype)      # BF16 round before gate
                        gf = Float32(rG[e])
                        sig = Float32(1.0) / (Float32(1.0) + cute.math.exp2(
                            -gf * Float32(LOG2_E), fastmath=True))
                        rW[e] = (attn.to(Float32) * sig).to(self.dtype)
                    gO = cute.make_tensor(
                        cute.make_ptr(self.dtype, out_addr + Int64(gj * 2),
                                      cute.AddressSpace.gmem, assumed_align=16),
                        cute.make_layout((nsub_c,)))
                    cute.autovec_copy(rW, gO)
            gt = gt_next
            u = u2

    # ---------------------------------------------------------------- MMA role
    @cute.jit
    def _mma_role(self, unit_posted, unit_taken, sUnit, total_units, nblocks,
                  mCount, qk_mma, pv_mma, tStS, tOtO, tSrQ, tSrK,
                  tOrP, tOrV, kfull, kfree, vfull, vfree,
                  sfull, pfull, pvdone, qready, odrained,
                  etile, kvs, ssts):
        """One warp issues every tcgen05 MMA, QK two tiles ahead of PV.

        Iteration ``i`` of a unit issues QK(i) then PV(i-2), so the in-order
        tensor-core queue runs ``QK(0), QK(1), PV(0), QK(2), PV(1), ...``:
        the softmax warpgroup gets a two-QK window per tile and the TC queue
        stays fed while it works.  Tile indices are CTA-cumulative across
        units (``gt``), so ring/stage phases never reset.  With ssts >= 3 the
        aliased S/P stage overwritten by QK(i) was last read by PV(i-ssts),
        which in-order execution retired before QK(i-1) <= PV(i-2) < QK(i).
        """
        ph = Int32(0)
        cute.arch.mbarrier_wait(unit_posted, 0)
        u = Int32(sUnit.load())
        cute.arch.mbarrier_arrive(unit_taken)
        cute.arch.mbarrier_wait(qready, 0)      # sQ resident for unit 0
        nkq = cute.size(tSrQ, mode=[2])
        nkp = cute.size(tOrV, mode=[2])
        gt = Int32(0)
        while u < total_units:
            if ph > 0:
                cute.arch.mbarrier_wait(qready, ph & 1)
            blk = self._unit_blk(u, nblocks)
            count = Int32(mCount[blk])
            num_tiles = (count + etile - 1) // etile
            # QK runs two tiles ahead of the PV slot when the S/P ring has
            # >= 3 stages; with ssts == 2 the stage QK(i+1) would overwrite is
            # still owned by PV(i-1), so the order falls back to 1-ahead.
            ahead = 2 if ssts >= 3 else 1
            for i in cutlass.range(num_tiles + ahead):
                if i < num_tiles:
                    g = gt + i
                    s = g % ssts
                    ks = g % kvs
                    cute.arch.mbarrier_wait(kfull + ks, (g // kvs) & 1)
                    tStS_s = tStS[(None, None, None, s)]
                    qk_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kp in cutlass.range_constexpr(nkq):
                        cute.gemm(qk_mma, tStS_s, tSrQ[(None, None, kp, 0)],
                                  tSrK[(None, None, kp, ks)], tStS_s)
                        qk_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(sfull + s, None, tcgen05.CtaGroup.ONE)
                        tcgen05.commit(kfree + ks, None, tcgen05.CtaGroup.ONE)
                if i >= ahead:
                    j = i - ahead
                    g = gt + j
                    ps = g % ssts
                    vs = g % kvs
                    if j == 0:
                        if ph > 0:
                            # O still holds the previous unit's output until
                            # its epilogue drained it into registers.
                            cute.arch.mbarrier_wait(odrained, (ph - 1) & 1)
                    cute.arch.mbarrier_wait(pfull + ps, (g // ssts) & 1)
                    cute.arch.mbarrier_wait(vfull + vs, (g // kvs) & 1)
                    pv_mma.set(tcgen05.Field.ACCUMULATE, j != 0)
                    for kp in cutlass.range_constexpr(nkp):
                        cute.gemm(pv_mma, tOtO, tOrP[(None, None, kp, ps)],
                                  tOrV[(None, None, kp, vs)], tOtO)
                        pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # Deferred releases: both fire only once the PV MMA has
                    # really read the V stage and the P columns, so producers
                    # may reuse them; ofree tells the softmax warpgroup that O
                    # is quiescent (epilogue drain / rare rescale).
                    with cute.arch.elect_one():
                        tcgen05.commit(vfree + vs, None, tcgen05.CtaGroup.ONE)
                        # PV-retirement ring: barrier g&3 sees every 4th PV,
                        # so a waiter can never observe it aliased two phases
                        # (the next commit to the same barrier is PV(g+3),
                        # gated on the waiter's own progress).
                        tcgen05.commit(pvdone + (g & 3), None,
                                       tcgen05.CtaGroup.ONE)
            gt += num_tiles
            ph += 1
            cute.arch.mbarrier_wait(unit_posted, ph & 1)
            u = Int32(sUnit.load())
            cute.arch.mbarrier_arrive(unit_taken)

    # ------------------------------------------------------------ gather role
    @cute.jit
    def _gather_role(self, mK, sX, kvgcopy, kvgthr, mWS, mBT, mCount,
                     unit_posted, unit_taken, sUnit, total_units, nblocks,
                     page, D, voff, st1b, full_bar, free_bar, kvs, lane,
                     maxe, etile, transpose):
        """One warp fills a K (or V) ring stage per tile with cp.async
        gathers, flowing seamlessly across unit boundaries.

        Lane ``l`` resolves union entry ``l`` of the tile (etile <= 32) into a
        token byte address; the token rows each lane copies are then reached
        by warp shuffles of those addresses, so the block-table lookup happens
        once per entry instead of once per copy.  Entries past ``count`` clamp
        to entry 0 so every SMEM slot is refreshed each tile: masked columns
        then contribute P=0 x finite V = 0, never 0 x stale-NaN = NaN.
        """
        N = self.n_tile
        ce = self.copy_elems
        gptr = self.gptr
        cND = cute.make_identity_tensor((N, D))
        tXcX = kvgthr.partition_S(cND)
        npass = cute.size(tXcX, mode=[1])
        nchunk = cute.size(tXcX, mode=[2])
        bt_w = self.bt_width
        ph = Int32(0)
        cute.arch.mbarrier_wait(unit_posted, 0)
        u = Int32(sUnit.load())
        cute.arch.mbarrier_arrive(unit_taken)
        gt = Int32(0)
        while u < total_units:
            blk = self._unit_blk(u, nblocks)
            kvh = u & 1
            count = Int32(mCount[blk])
            num_tiles = (count + etile - 1) // etile
            ws_base = (blk * maxe) * 4
            last_eb = (num_tiles - 1) * etile

            # ---- 3-stage entry-resolution pipeline: at iteration i the mWS
            # words for tile i+2 and the mBT lookup for tile i+1 are ISSUED,
            # while tile i fills from the address resolved at i-1.  Both
            # dependent global-load latencies therefore hide under a full
            # tile of cp.async issue work instead of pacing the ring. ----
            # prime stage W (tile 0 + tile 1 words) and stage M (tile 0 bt).
            eid0 = lane
            src0 = cutlass.select_(eid0 < count, eid0, Int32(0))
            wr0 = Int32(mWS[ws_base + src0 * 4])
            wg0 = Int32(mWS[ws_base + src0 * 4 + 1])
            eid1 = cutlass.min(etile, last_eb) + lane
            src1 = cutlass.select_(eid1 < count, eid1, Int32(0))
            wr1 = Int32(mWS[ws_base + src1 * 4])
            wg1 = Int32(mWS[ws_base + src1 * 4 + 1])
            # clamp the prime lookups so a degenerate unit (count == 0, only
            # possible for an empty union) cannot index mBT out of bounds
            req0 = cutlass.min(cutlass.max(wr0, Int32(0)), Int32(0))
            lidx0 = wg0 << 2
            lpage0 = cutlass.min(cutlass.max(lidx0 // page, Int32(0)), bt_w - 1)
            loff0 = lidx0 - (lidx0 // page) * page
            mp0 = Int32(mBT[req0, lpage0])

            for i in cutlass.range(num_tiles):
                g = gt + i
                st = g % kvs
                e2 = cutlass.min((i + 2) * etile, last_eb)
                eid2 = e2 + lane
                src2 = cutlass.select_(eid2 < count, eid2, Int32(0))
                wr2 = Int32(mWS[ws_base + src2 * 4])
                wg2 = Int32(mWS[ws_base + src2 * 4 + 1])
                # stage M for tile i+1 (wr1/wg1 landed a full iteration ago)
                lidx1 = wg1 << 2
                lpage1 = lidx1 // page
                loff1 = lidx1 - lpage1 * page
                mp1 = Int32(mBT[wr1, lpage1])
                # stage P for tile i (mp0 landed a full iteration ago)
                myaddr = _elem_pointer(mK, (mp0, loff0, kvh, 0)).toint() + voff
                if g >= kvs:
                    cute.arch.mbarrier_wait(free_bar + st, ((g // kvs) - 1) & 1)
                tXsX = kvgthr.partition_D(
                    self._flatten(sX[(None, None, None, st)], transpose))
                for mm in cutlass.range_constexpr(npass):
                    # pass mm copies token 4*mm + lane//gptr, i.e. union entry mm
                    addr = _shfl(myaddr, mm, 32) + Int64(lane // gptr) * st1b
                    gptr_t = cute.make_ptr(self.dtype, addr, cute.AddressSpace.gmem,
                                           assumed_align=16)
                    mg = cute.make_tensor(gptr_t, cute.make_layout((D,)))
                    mgc = cute.tiled_divide(mg, (ce,))
                    for k in cutlass.range_constexpr(nchunk):
                        ki = tXcX[0, 0, k][1] // ce
                        srck = cute.make_tensor(mgc[None, ki].iterator,
                                                tXsX[(None, mm, k)].layout)
                        cute.copy(kvgcopy, srck, tXsX[(None, mm, k)])
                cute.arch.cp_async_commit_group()
                wr1 = wr2
                wg1 = wg2
                mp0 = mp1
                loff0 = loff1
                if g > 0:
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(full_bar + (g - 1) % kvs)
            gt += num_tiles
            ph += 1
            cute.arch.mbarrier_wait(unit_posted, ph & 1)
            u = Int32(sUnit.load())
            cute.arch.mbarrier_arrive(unit_taken)
        if gt > 0:
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(full_bar + (gt - 1) % kvs)

    # ---------------------------------------------------------------- helpers
    @cute.jit
    def _flatten(self, sX: cute.Tensor, transpose: cutlass.Constexpr) -> cute.Tensor:
        lay = cute.make_layout(
            (sX.shape[0][0], (sX.shape[0][1], sX.shape[2])),
            stride=(sX.stride[0][0], (sX.stride[0][1], sX.stride[2])))
        t = cute.make_tensor(sX.iterator, lay)
        if cutlass.const_expr(transpose):
            t = cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 0]))
        return t

    @cute.jit
    def _load_q(self, mSrc, sXf, tXsX, tXcX, tXc0, gcopy,
                row0, kvh, group, lane_rows):
        """cp.async gather of Q: M lane m -> q[row0 + m//group, kvh*group + m%group]."""
        gptr = self.gptr
        ce = self.copy_elems
        D = self.head_dim
        ti = cute.arch.thread_idx()[0]
        token_row = (ti % gptr) * (128 // gptr) + ti // gptr
        valid = token_row < lane_rows
        r = token_row // group
        h = token_row % group
        rr = cutlass.min(r, Int32(self.br) - 1)
        hh = cutlass.min(h, group - 1)
        ptr = _elem_pointer(mSrc, (row0 + rr, kvh * group + hh, 0)).toint()
        ptr = cutlass.select_(valid, ptr, Int64(0))
        rowlim = lane_rows
        for mm in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            src_lane = mm % gptr
            ptr_m = _shfl(ptr, src_lane, gptr)
            row_valid = tXc0[0, mm, 0][0] < (rowlim - tXcX[0][0])
            if row_valid:
                gptr_t = cute.make_ptr(self.dtype, ptr_m, cute.AddressSpace.gmem,
                                       assumed_align=16)
                mg = cute.make_tensor(gptr_t, cute.make_layout((D,)))
                mgc = cute.tiled_divide(mg, (ce,))
                for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                    ki = tXcX[0, 0, k][1] // ce
                    srck = cute.make_tensor(mgc[None, ki].iterator, tXsX[(None, mm, k)].layout)
                    cute.copy(gcopy, srck, tXsX[(None, mm, k)])

    @cute.jit
    def _colbits_raw(self, mWS, wbase, ebase, nw):
        """Issue the ``nw`` raw nibble-word LDGs for the tile at ``ebase``.

        Returned as a register list; the caller defers the shift/OR packing
        to a later iteration so the global-load latency overlaps real work.
        ``ebase`` must already be clamped into this unit's entry range (dead
        tails re-read the last live tile's words: harmless, they are masked).
        """
        raw = []
        for ee in cutlass.range_constexpr(nw * 8):
            raw.append(Int32(mWS[wbase + (ebase + ee) * 4]))
        return raw

    @cute.jit
    def _colbits_pack(self, raw, count, sh_lo32, ebase, etile, nw):
        """Pack pre-loaded raw nibble words into this thread's validity bits.

        Entry ``ebase + ee`` contributes its 4 lane nibbles at bits ``4*ee``
        of the returned word list; entries at/after ``count`` contribute 0.
        """
        nlive = cutlass.min(Int32(etile), count - ebase)
        ws = []
        for w in cutlass.range_constexpr(nw):
            acc = Int32(0)
            for ee in cutlass.range_constexpr(8):
                k = w * 8 + ee
                nib = (raw[k] >> sh_lo32) & Int32(0xF)
                nib = cutlass.select_(k < nlive, nib, Int32(0))
                acc = acc | (nib << (ee * 4))
            ws.append(acc)
        return ws

    @cute.jit
    def _colbits2(self, mWS, wbase, count, sh_lo32, ebase, etile, last_base, nw):
        raw = self._colbits_raw(mWS, wbase, cutlass.min(ebase, last_base), nw)
        return self._colbits_pack(raw, count, sh_lo32, ebase, etile, nw)

    @cute.jit
    def _rescale_O(self, tOtO, acc_scale, tidx, M, D):
        """Rare correction path: re-reference the O accumulator inside TMEM."""
        o_slice = tOtO[(None, None), 0, 0]
        epi_tile = (M, 64)
        o_epi = cute.flat_divide(o_slice, epi_tile)      # (M,64, 1, D/64)
        cO = cute.make_identity_tensor((M, D))
        cO_epi = cute.flat_divide(cO, epi_tile)
        ld = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.acc)
        tiled_ld = tcgen05.make_tmem_copy(ld, o_epi[(None, None, 0, 0)])
        thr_ld = tiled_ld.get_slice(tidx)
        tO = thr_ld.partition_S(o_epi)
        tcO = thr_ld.partition_D(cO_epi)
        st = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.acc)
        tiled_st = tcgen05.make_tmem_copy(st, o_epi[(None, None, 0, 0)])
        thr_st = tiled_st.get_slice(tidx)
        tO_st = thr_st.partition_D(o_epi)
        for j in cutlass.range_constexpr(cute.size(tO, mode=[4])):
            rO = cute.make_rmem_tensor(tcO[(None, None, None, 0, j)].shape, self.acc)
            cute.copy(tiled_ld, tO[(None, None, None, 0, j)], rO)
            cute.arch.fence_view_async_tmem_load()
            for i in cutlass.range_constexpr(cute.size(rO)):
                rO[i] = rO[i] * acc_scale
            cute.copy(tiled_st, rO, tO_st[(None, None, None, 0, j)])
            cute.arch.fence_view_async_tmem_store()
