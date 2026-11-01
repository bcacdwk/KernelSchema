"""Runtime geometry, workspace and local CUDA dispatch."""

import torch
from .build import (
    _load_context,
    _load_smallm,
    _load_masked_context,
    _load_hd256_context,
    _load_split_context,
)


def _context_tile_mma_work(q_len: int, kv_len: int, tokens_per_tile: int) -> int:
    """Mirror the standalone route's bottom-right causal tile-work model."""
    total = 0
    full_n_blocks = (kv_len + 127) // 128
    shift = kv_len - q_len
    for m_block in range((q_len + tokens_per_tile - 1) // tokens_per_tile):
        max_n = (m_block + 1) * tokens_per_tile + shift
        total += (max_n + 127) // 128 if max_n < kv_len else full_n_blocks
    return total


def _context_pack_g(
    max_q_len: int, max_kv_len: int, num_q_heads: int, num_kv_heads: int
) -> int:
    """Choose the canonical packed-GQA axis from host-visible maxima."""
    group = num_q_heads // num_kv_heads
    if group <= 1 or group > 128:
        return 1
    unpacked = num_q_heads * _context_tile_mma_work(max_q_len, max_kv_len, 256)
    packed = num_kv_heads * _context_tile_mma_work(
        max_q_len, max_kv_len, 2 * (128 // group)
    )
    return group if packed < unpacked else 1


def _run_context(
    query,
    key,
    value,
    workspace,
    tables,
    lengths,
    maxq,
    maxkv,
    scale,
    batch,
    cuq,
    cukv,
    out,
    lse,
    shared,
    counter,
    causal,
    sinks,
):
    import tvm_ffi

    heads, kv_heads = (query.shape[1], key.shape[1])
    pack = _context_pack_g(maxq, maxkv, heads, kv_heads)
    tokens = 128 // pack
    blocks = (maxq + 2 * tokens - 1) // (2 * tokens)
    swizzle = 8 if batch * (heads // pack) % 8 == 0 else 1
    major, minor = torch.cuda.get_device_capability(query.device)
    module = _load_context(
        major,
        minor,
        blocks,
        heads,
        kv_heads,
        pack,
        key.shape[2],
        swizzle,
        lse is not None,
        causal,
        sinks is not None,
        query.shape[-1],
    )
    sm_count = torch.cuda.get_device_properties(query.device).multi_processor_count
    counter_bytes = (max(batch * heads, sm_count) + 7) // 8 * 8 * 4
    if counter is None:
        counter = (
            workspace
            if query.shape[-1] == 64 and lse is not None and (sinks is None)
            else torch.empty(counter_bytes, dtype=torch.uint8, device=query.device)
        )
    elif (
        counter.device != query.device
        or not counter.is_contiguous()
        or counter.data_ptr() % 16
        or (counter.numel() * counter.element_size() < counter_bytes)
    ):
        raise ValueError("Invalid multi-CTA counter buffer for context attention")
    with tvm_ffi.use_torch_stream():
        module.cake_paged_attention_context(
            out,
            None,
            query,
            key,
            value,
            workspace,
            counter,
            tables,
            lengths,
            maxq,
            maxkv,
            float(scale),
            1.0,
            -1.0,
            -1,
            0,
            batch,
            -1,
            cuq,
            cukv,
            sm_count,
            False,
            workspace.numel() * workspace.element_size(),
            sinks,
            None,
            None,
            None,
            shared,
            False,
            False,
            bool(causal),
            lse,
            lse.stride(0) if lse is not None else 0,
            lse.stride(1) if lse is not None else 0,
        )


def _run_smallm(
    query, key, value, workspace, tables, lengths, maxkv, scale, batch, out, lse
):
    import tvm_ffi

    sms = torch.cuda.get_device_properties(query.device).multi_processor_count
    tiles = batch * key.shape[1]
    if tiles > sms:
        raise ValueError("small-M Split-KV requires a resident wave")
    split = max(1, min(sms // tiles, max((maxkv + 127) // 128, 32), 256))
    module = _load_smallm(key.shape[2], query.shape[1] // key.shape[1], split)
    with tvm_ffi.use_torch_stream():
        module.cake_paged_attention_decode(
            out,
            None,
            query,
            key,
            value,
            workspace,
            workspace,
            tables,
            lengths,
            1,
            maxkv,
            float(scale),
            1.0,
            -1.0,
            -1,
            0,
            batch,
            -1,
            0,
            sms,
            False,
            workspace.numel() * workspace.element_size(),
            None,
            None,
            None,
            None,
            None,
            True,
            lse,
            lse.stride(0) if lse is not None else 0,
            lse.stride(1) if lse is not None else 0,
            False,
            None,
        )


def _run_masked_context(
    query,
    key,
    value,
    workspace,
    tables,
    lengths,
    maxq,
    maxkv,
    scale,
    batch,
    cuq,
    cukv,
    out,
    lse,
    shared,
    counter,
    causal,
    sinks,
    window,
):
    import tvm_ffi

    heads, kv_heads = (query.shape[1], key.shape[1])
    pack = _context_pack_g(maxq, maxkv, heads, kv_heads)
    tokens = 128 // pack
    blocks = (maxq + 2 * tokens - 1) // (2 * tokens)
    swizzle = 8 if batch * (heads // pack) % 8 == 0 else 1
    major, minor = torch.cuda.get_device_capability(query.device)
    module = _load_masked_context(
        major,
        minor,
        blocks,
        heads,
        kv_heads,
        pack,
        key.shape[2],
        swizzle,
        lse is not None,
        causal,
        sinks is not None,
        query.shape[-1],
    )
    sm_count = torch.cuda.get_device_properties(query.device).multi_processor_count
    counter_bytes = (max(batch * heads, sm_count) + 7) // 8 * 8 * 4
    if counter is None:
        counter = torch.empty(counter_bytes, dtype=torch.uint8, device=query.device)
    elif (
        counter.device != query.device
        or not counter.is_contiguous()
        or counter.data_ptr() % 16
        or (counter.numel() * counter.element_size() < counter_bytes)
    ):
        raise ValueError("Invalid multi-CTA counter buffer for context attention")
    with tvm_ffi.use_torch_stream():
        module.cake_paged_attention_context(
            out,
            None,
            query,
            key,
            value,
            workspace,
            counter,
            tables,
            lengths,
            maxq,
            maxkv,
            float(scale),
            1.0,
            -1.0,
            -1,
            0,
            batch,
            int(window),
            cuq,
            cukv,
            sm_count,
            False,
            workspace.numel() * workspace.element_size(),
            sinks,
            None,
            None,
            None,
            shared,
            False,
            False,
            bool(causal),
            lse,
            lse.stride(0) if lse is not None else 0,
            lse.stride(1) if lse is not None else 0,
        )


def _run_hd256_context(
    query,
    key,
    value,
    workspace,
    tables,
    lengths,
    maxq,
    maxkv,
    scale,
    batch,
    cuq,
    cukv,
    out,
    lse,
):
    import tvm_ffi

    heads = query.shape[1]
    kvheads = key.shape[1]
    blocks = (maxq + 127) // 128
    micro_pages = (maxkv + 127) // 128
    required = (
        batch * kvheads * micro_pages * 128 * 1024
        + batch * heads * 12
        + batch * kvheads * micro_pages * 4
        + 128
    )
    scratch = (
        workspace
        if workspace.numel() * workspace.element_size() >= required
        else torch.empty(required, dtype=torch.uint8, device=query.device)
    )
    counter = torch.empty(4, dtype=torch.uint8, device=query.device)
    module = _load_hd256_context(blocks, heads, kvheads, key.shape[2])
    with tvm_ffi.use_torch_stream():
        module.cake_paged_attention_context(
            out,
            None,
            query,
            key,
            value,
            scratch,
            counter,
            tables,
            lengths,
            maxq,
            maxkv,
            float(scale),
            1.0,
            -1.0,
            -1,
            0,
            batch,
            -1,
            cuq,
            cukv,
            torch.cuda.get_device_properties(query.device).multi_processor_count,
            False,
            scratch.numel() * scratch.element_size(),
            None,
            None,
            None,
            None,
            True,
            False,
            False,
            True,
            lse,
            lse.stride(0) if lse is not None else 0,
            lse.stride(1) if lse is not None else 0,
        )


def _run_split_context(
    query,
    key,
    value,
    workspace,
    tables,
    lengths,
    maxq,
    maxkv,
    scale,
    batch,
    cuq,
    cukv,
    out,
    lse,
    shared,
    counter,
    causal,
    sinks,
    window,
):
    import tvm_ffi

    heads, kv_heads = (query.shape[1], key.shape[1])
    pack = _context_pack_g(maxq, maxkv, heads, kv_heads)
    tokens = 128 // pack
    blocks = (maxq + 2 * tokens - 1) // (2 * tokens)
    swizzle = 8 if batch * (heads // pack) % 8 == 0 else 1
    major, minor = torch.cuda.get_device_capability(query.device)
    sms = torch.cuda.get_device_properties(query.device).multi_processor_count
    split = max(1, min(32, sms // (batch * (heads // pack))))
    required = (
        16
        + 12 * batch * (heads // pack)
        + query.shape[0] * heads * (query.shape[-1] + 1) * 4 * split
    )
    if workspace.numel() * workspace.element_size() < required:
        workspace = torch.empty(required, dtype=torch.uint8, device=query.device)
    module = _load_split_context(
        major,
        minor,
        blocks,
        heads,
        kv_heads,
        pack,
        key.shape[2],
        swizzle,
        lse is not None,
        causal,
        sinks is not None,
        query.shape[-1],
        split,
        query.shape[-1] == 64 and maxq <= tokens,
    )
    sm_count = torch.cuda.get_device_properties(query.device).multi_processor_count
    counter_bytes = (max(batch * heads, sm_count) + 7) // 8 * 8 * 4
    if counter is None:
        counter = workspace
    elif (
        counter.device != query.device
        or not counter.is_contiguous()
        or counter.data_ptr() % 16
        or (counter.numel() * counter.element_size() < counter_bytes)
    ):
        raise ValueError("Invalid multi-CTA counter buffer for context attention")
    with tvm_ffi.use_torch_stream():
        module.cake_paged_attention_context(
            out,
            None,
            query,
            key,
            value,
            workspace,
            counter,
            tables,
            lengths,
            maxq,
            maxkv,
            float(scale),
            1.0,
            -1.0,
            -1,
            0,
            batch,
            int(window),
            cuq,
            cukv,
            sm_count,
            False,
            workspace.numel() * workspace.element_size(),
            sinks,
            None,
            None,
            None,
            shared,
            False,
            False,
            bool(causal),
            lse,
            lse.stride(0) if lse is not None else 0,
            lse.stride(1) if lse is not None else 0,
        )
