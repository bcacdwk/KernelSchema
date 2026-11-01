"""Runtime geometry, workspace and local CUDA dispatch."""

import torch
from functools import lru_cache
from .build import _load_d06tc, _load_d06tc_short, _load_d06tc_window_split


def _d06_tile_work(q_len, kv_len, tokens):
    full = (kv_len + 127) // 128
    shift = kv_len - q_len
    return sum(
        (
            min(full, ((m + 1) * tokens + shift + 127) // 128)
            for m in range((q_len + tokens - 1) // tokens)
        )
    )


@lru_cache(maxsize=256)
def _d06_pack_g(maxq, maxkv, heads, kvheads):
    group = heads // kvheads
    if group <= 1 or group > 128:
        return 1
    return (
        group
        if kvheads * _d06_tile_work(maxq, maxkv, 2 * (128 // group))
        < heads * _d06_tile_work(maxq, maxkv, 256)
        else 1
    )


@lru_cache(maxsize=32)
def _d06_hardware(device):
    p = torch.cuda.get_device_properties(device)
    return (p.major, p.minor, p.multi_processor_count)


def _run_d06tc(
    query,
    key,
    value,
    ks,
    vs,
    workspace,
    tables,
    lengths,
    maxq,
    maxkv,
    scale1,
    scale2,
    batch,
    cuq,
    cukv,
    out,
    lse,
    sinks,
    causal,
    shared,
    window,
):
    heads = query.shape[1]
    kvheads = key.shape[1]
    dim = query.shape[2]
    pack = _d06_pack_g(maxq, maxkv, heads, kvheads)
    tokens = 128 // pack
    blocks = (maxq + 2 * tokens - 1) // (2 * tokens)
    swizzle = 8 if batch * (heads // pack) % 8 == 0 else 1
    major, minor, sm_count = _d06_hardware(query.device)
    module = _load_d06tc(
        major,
        minor,
        blocks,
        heads,
        kvheads,
        pack,
        key.shape[2],
        swizzle,
        lse is not None,
        causal,
        sinks is not None,
        dim,
        window,
    )
    page = key.shape[2]
    capacity = (maxq + window + 128 + page - 1) // page + 1 if window >= 0 else 0
    compact = window >= 0 and batch * capacity < key.shape[0]
    pages = batch * capacity if compact else key.shape[0]
    rows = query.shape[0] * heads
    required = 3 * batch * (heads // pack) * 4 + rows * 0 * (dim + 2) * 4
    required += 2 * pages * kvheads * page * dim + (rows * dim if dim == 64 else 0)
    required += (tables.numel() * tables.element_size() if compact else 0) + 1024
    scratch = (
        workspace
        if workspace.numel() * workspace.element_size() >= required
        else torch.empty(required, dtype=torch.uint8, device=query.device)
    )
    stream = int(torch.cuda.current_stream(query.device).cuda_stream)
    module.d06_fused_context(
        out,
        lse,
        query,
        key,
        value,
        ks,
        vs,
        scratch,
        tables,
        lengths,
        cuq,
        cukv,
        maxq,
        maxkv,
        float(scale1),
        float(scale2),
        batch,
        sm_count,
        sinks,
        causal,
        shared,
        window,
        capacity,
        compact,
        stream,
    )


def _run_d06tc256(
    query,
    key,
    value,
    ks,
    vs,
    workspace,
    tables,
    lengths,
    maxq,
    maxkv,
    scale1,
    scale2,
    batch,
    cuq,
    cukv,
    out,
    lse,
    shared,
    causal,
):
    heads = query.shape[1]
    kvheads = key.shape[1]
    blocks = (maxq + 127) // 128
    major, minor, sm_count = _d06_hardware(query.device)
    module = _load_d06tc(
        major,
        minor,
        blocks,
        heads,
        kvheads,
        1,
        key.shape[2],
        8,
        lse is not None,
        causal,
        False,
        256,
    )
    decoded_key = torch.empty(
        (*key.shape[:3], 256), dtype=torch.float8_e4m3fn, device=query.device
    )
    decoded_value = torch.empty_like(decoded_key)
    qrows = batch * heads * blocks * 128
    micro = batch * kvheads * ((maxkv + 15) // 16)
    workspace_bytes = (
        qrows * (256 + 512)
        + 2 * micro * 16 * 256
        + 3 * batch * heads * 4
        + micro * 4
        + 64
    )
    scratch = (
        workspace
        if workspace.numel() * workspace.element_size() >= workspace_bytes
        else torch.empty(workspace_bytes, dtype=torch.uint8, device=query.device)
    )
    counter = torch.empty(8, dtype=torch.uint8, device=query.device)
    two_tables = tables.unsqueeze(1).expand(-1, 2, -1) if shared else tables
    stream = int(torch.cuda.current_stream(query.device).cuda_stream)
    module.d06_prepare(
        query, query, key, value, ks, vs, decoded_key, decoded_value, counter, stream
    )
    module.cake_paged_attention_context(
        out,
        None,
        query,
        decoded_key,
        decoded_value,
        scratch,
        counter,
        two_tables,
        lengths,
        maxq,
        maxkv,
        float(scale1),
        float(scale2),
        -1.0,
        -1,
        0,
        batch,
        -1,
        cuq,
        cukv,
        sm_count,
        False,
        scratch.numel() * scratch.element_size(),
        None,
        None,
        None,
        None,
        False,
        False,
        False,
        causal,
        lse,
        lse.stride(0) if lse is not None else 0,
        lse.stride(1) if lse is not None else 0,
        stream,
    )


def _run_d06tc256_short(
    query,
    key,
    value,
    ks,
    vs,
    workspace,
    tables,
    lengths,
    maxq,
    maxkv,
    scale1,
    scale2,
    batch,
    cuq,
    cukv,
    out,
    lse,
    shared,
    causal,
):
    heads = query.shape[1]
    kvheads = key.shape[1]
    pack = 4 if heads // kvheads % 4 == 0 else 1
    blocks = (maxq * pack + 127) // 128
    major, minor, sm_count = _d06_hardware(query.device)
    module = _load_d06tc_short(
        major,
        minor,
        blocks,
        heads,
        kvheads,
        pack,
        key.shape[2],
        8 if batch * (heads // pack) % 8 == 0 else 1,
        lse is not None,
        causal,
        False,
        256,
    )
    decoded_key = torch.empty(
        (*key.shape[:3], 256), dtype=torch.float8_e4m3fn, device=query.device
    )
    decoded_value = torch.empty_like(decoded_key)
    qrows = batch * (heads // pack) * blocks * 128
    micro = batch * kvheads * ((maxkv + 15) // 16)
    workspace_bytes = (
        qrows * (256 + 512)
        + 2 * micro * 16 * 256
        + 3 * batch * heads * 4
        + micro * 4
        + 64
    )
    workspace_bytes += qrows * (256 + 2) * 4 * 4
    scratch = (
        workspace
        if workspace.numel() * workspace.element_size() >= workspace_bytes
        else torch.empty(workspace_bytes, dtype=torch.uint8, device=query.device)
    )
    counter = torch.empty(8, dtype=torch.uint8, device=query.device)
    two_tables = tables.unsqueeze(1).expand(-1, 2, -1) if shared else tables
    stream = int(torch.cuda.current_stream(query.device).cuda_stream)
    module.d06_prepare(
        query, query, key, value, ks, vs, decoded_key, decoded_value, counter, stream
    )
    module.cake_paged_attention_context(
        out,
        None,
        query,
        decoded_key,
        decoded_value,
        scratch,
        counter,
        two_tables,
        lengths,
        maxq,
        maxkv,
        float(scale1),
        float(scale2),
        -1.0,
        -1,
        0,
        batch,
        -1,
        cuq,
        cukv,
        sm_count,
        False,
        scratch.numel() * scratch.element_size(),
        None,
        None,
        None,
        None,
        False,
        False,
        False,
        causal,
        lse,
        lse.stride(0) if lse is not None else 0,
        lse.stride(1) if lse is not None else 0,
        stream,
    )


def _run_d06tc_window_split(
    query,
    key,
    value,
    ks,
    vs,
    workspace,
    tables,
    lengths,
    maxq,
    maxkv,
    scale1,
    scale2,
    batch,
    cuq,
    cukv,
    out,
    lse,
    sinks,
    causal,
    shared,
    window,
):
    heads = query.shape[1]
    kvheads = key.shape[1]
    dim = query.shape[2]
    pack = _d06_pack_g(maxq, maxkv, heads, kvheads)
    tokens = 128 // pack
    blocks = (maxq + 2 * tokens - 1) // (2 * tokens)
    swizzle = 8 if batch * (heads // pack) % 8 == 0 else 1
    major, minor, sm_count = _d06_hardware(query.device)
    splits = max(1, min(32, sm_count // (batch * (heads // pack))))
    module = _load_d06tc_window_split(
        major,
        minor,
        blocks,
        heads,
        kvheads,
        pack,
        key.shape[2],
        swizzle,
        lse is not None,
        causal,
        sinks is not None,
        dim,
        window,
        splits,
    )
    page = key.shape[2]
    capacity = (maxq + window + 128 + page - 1) // page + 1 if window >= 0 else 0
    compact = window >= 0 and batch * capacity < key.shape[0]
    pages = batch * capacity if compact else key.shape[0]
    rows = query.shape[0] * heads
    required = 3 * batch * (heads // pack) * 4 + rows * splits * (dim + 2) * 4
    required += 2 * pages * kvheads * page * dim + (rows * dim if dim == 64 else 0)
    required += (tables.numel() * tables.element_size() if compact else 0) + 1024
    scratch = (
        workspace
        if workspace.numel() * workspace.element_size() >= required
        else torch.empty(required, dtype=torch.uint8, device=query.device)
    )
    stream = int(torch.cuda.current_stream(query.device).cuda_stream)
    module.d06_fused_context(
        out,
        lse,
        query,
        key,
        value,
        ks,
        vs,
        scratch,
        tables,
        lengths,
        cuq,
        cukv,
        maxq,
        maxkv,
        float(scale1),
        float(scale2),
        batch,
        sm_count,
        sinks,
        causal,
        shared,
        window,
        capacity,
        compact,
        stream,
    )
