"""Host entry point for the native CuTeDSL core-QSA operator (V4 form).

Two native CuTeDSL kernels per call:
  1. ``QSAPreprocessUnion``: one CTA per row block; vectorized pipelined
     quad loads + SMEM hash dedup of the block rows' packed selections into
     (req, group, nibble-mask) union entries in a scratch workspace, with a
     deterministic warp-prefix compaction.
  2. ``QSAForward``: persistent warp-specialized kernel over min(#SMs,
     #units) CTAs pulling (row block, kv head) unit tickets from a global
     counter; two column-split softmax warpgroups, P-in-TMEM tcgen05 MMAs,
     cross-unit pipelined K/V gather ring, fused sigmoid gate.

``out`` is mutated in place and returned, matching ``reference.py``.
"""
from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .qsa_kernel import QSAForward
from .qsa_preprocess import QSAPreprocessUnion

_COMPILE_CACHE: dict = {}
_WS_CACHE: dict = {}

# 128 M lanes / heads-per-group rows per CTA block
_BR_FOR_GROUP = {12: 10}


def _cute_stream() -> "cute.runtime.Stream":
    import cuda.bindings.driver as cuda_drv
    return cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)


def _c(t: torch.Tensor, align: int, leading_dim: int) -> cute.Tensor:
    return from_dlpack(t, assumed_align=align).mark_layout_dynamic(leading_dim=leading_dim)


def run(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    output_gate: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    del use_prefill_config  # semantics are identical for the scored domain
    if out is None:
        out = torch.empty_like(q)

    R, H, D = q.shape
    P, S, Hkv, Dk = k_cache.shape
    # The captured ABI exposes K and V as two views of one interleaved page
    # allocation: identical shapes/strides, differing only by the element
    # offset. The kernel exploits this (V address = K address + base delta).
    if v_cache.stride() != k_cache.stride() or v_cache.shape != k_cache.shape:
        raise NotImplementedError(
            "qsa_cutedsl requires k_cache/v_cache to share shape and strides")
    # The coalesced epilogue moves gate/out rows in 16B-granular chunks: the
    # head-dim must be contiguous and the row strides 16B-compatible.
    for t in (output_gate, out):
        if t.stride(2) != 1 or t.stride(1) % 8 or t.stride(0) % 8:
            raise NotImplementedError(
                "qsa_cutedsl requires 16B-compatible gate/out strides")
    sel_width = logical_indices.shape[1] - 1
    bt_width = block_table.shape[1]
    # The preprocess kernel reads packed index quads as 16B vectors: rows are
    # 2052 words (513 x 16B), so quad addresses are 16B-aligned iff the base
    # is. Torch contiguous int32 allocations always are; guard anyway.
    if logical_indices.data_ptr() % 16 or logical_indices.stride(1) != 1:
        raise NotImplementedError(
            "qsa_cutedsl requires contiguous 16B-aligned logical_indices")
    group = H // Hkv
    softmax_scale = float(D) ** -0.5

    br = _BR_FOR_GROUP.get(group)
    if br is None:
        br = max(1, 128 // group)
    maxe = br * (sel_width // 4 + 1)
    nblocks = (R + br - 1) // br

    key = (
        tuple(q.shape), tuple(q.stride()),
        tuple(k_cache.shape), tuple(k_cache.stride()),
        tuple(v_cache.stride()),
        tuple(logical_indices.shape), tuple(logical_indices.stride()),
        tuple(block_table.shape), tuple(block_table.stride()),
        tuple(token_to_req.shape), tuple(token_to_req.stride()),
        tuple(output_gate.stride()), tuple(out.stride()),
        q.dtype, k_cache.dtype, logical_indices.dtype,
    )

    # Scratch workspace: union entries (req, g, nib_lo, nib_hi) + per-block
    # counts, reused across same-shape calls (every consumed word is rewritten
    # by the preprocess kernel before the main kernel reads it). +128-word
    # slack keeps the tile-granular entry prefetch inside the allocation for
    # the final block.
    ws_key = (q.device, nblocks, maxe)
    ws_entry = _WS_CACHE.get(ws_key)
    if ws_entry is None:
        ws = torch.empty(nblocks * maxe * 4 + 128, dtype=torch.int32,
                         device=q.device)
        cnt = torch.empty(nblocks, dtype=torch.int32, device=q.device)
        # Unit-ticket counter for the persistent main kernel.  Tickets grow
        # monotonically; the host tracks this launch's base offset (int32
        # wrap-safe) and passes it to the kernel.
        ticket = torch.zeros(1, dtype=torch.int32, device=q.device)
        ws_entry = [ws, cnt, _c(ws, 4, 0), _c(cnt, 4, 0), _c(ticket, 4, 0), 0]
        _WS_CACHE[ws_key] = ws_entry
    _, _, mWS, mCount, mTicket, _tb = ws_entry

    mQ = _c(q, 16, 2)
    mK = _c(k_cache, 16, 3)
    mV = _c(v_cache, 16, 3)
    mLI = _c(logical_indices, 16, 1)
    mBT = _c(block_table, 4, 1)
    mT2R = _c(token_to_req, 4, 0)
    mGate = _c(output_gate, 16, 2)
    mOut = _c(out, 16, 2)
    scale = cutlass.Float32(softmax_scale)
    nb = cutlass.Int32(nblocks)
    rows = cutlass.Int32(R)
    bt_rows = cutlass.Int32(block_table.shape[0])
    num_pages = cutlass.Int32(P)
    # Persistent main kernel: one CTA per SM (the ~196KB SMEM + full TMEM
    # footprint allows exactly one resident), strided over the work units.
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    nct = cutlass.Int32(max(1, min(sms, nblocks * Hkv)))
    stream = _cute_stream()

    entry = _COMPILE_CACHE.get(key)
    if entry is None:
        pre = QSAPreprocessUnion(br=br, sel_width=sel_width, page_size=S,
                                 bt_width=bt_width)
        pre_compiled = cute.compile(
            pre.__call__, mLI, mBT, mT2R, mWS, mCount, rows, bt_rows,
            num_pages, stream)
        op = QSAForward(group=group, head_dim=D, page_size=S,
                        sel_width=sel_width, bt_width=bt_width, br=br, maxe=maxe)
        main_compiled = cute.compile(
            op.__call__, mQ, mK, mV, mWS, mCount, mTicket, mBT, mGate, mOut,
            scale, nb, nct, cutlass.Int32(_tb), stream)
        _COMPILE_CACHE[key] = (pre_compiled, main_compiled)
        entry = (pre_compiled, main_compiled)

    pre_compiled, main_compiled = entry
    pre_compiled(mLI, mBT, mT2R, mWS, mCount, rows, bt_rows,
                 num_pages, stream)
    # int32 two's-complement-normalized launch base; every launch consumes
    # exactly total_units + ncta tickets (each CTA ends on one terminal grab).
    tbase = cutlass.Int32((ws_entry[5] + 2**31) % 2**32 - 2**31)
    ws_entry[5] += nblocks * Hkv + max(1, min(sms, nblocks * Hkv))
    main_compiled(mQ, mK, mV, mWS, mCount, mTicket, mBT, mGate, mOut, scale,
                  nb, nct, tbase, stream)
    return out
