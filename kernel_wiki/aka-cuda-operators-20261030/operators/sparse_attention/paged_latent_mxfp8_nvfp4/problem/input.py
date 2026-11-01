import torch

def _workload_profiles_mapping(count, layout, device='cpu'):
    if layout == 'contiguous':
        return torch.arange(count, device=device)
    if layout == 'scattered':
        return torch.randperm(count, device=device)
    raise ValueError(layout)

def _workload_profiles_select(visible, budget, pattern, scores=None, device='cpu'):
    count = min(visible, budget)
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if pattern == 'score':
        assert scores is not None and scores.numel() >= visible
        return scores[:visible].topk(count).indices.to(device)
    if pattern == 'local':
        return torch.arange(visible - count, visible, device=device)
    if pattern == 'shared_prefix':
        return torch.arange(count, device=device)
    if pattern == 'strided':
        return torch.linspace(0, visible - 1, count, device=device).round().long()
    if pattern == 'random':
        return torch.randperm(visible, device=device)[:count]
    raise ValueError(pattern)

def _packing_guarded(x):
    b = torch.full((x.numel() + 256,), 73, dtype=x.dtype, device=x.device)
    y = b[128:-128].view(x.shape)
    y.copy_(x)
    return y

def _packing_pack(x, mixed):
    pages, page, _ = x.shape
    tile = 16 if mixed else 32
    g = x.float().reshape(pages, page, 512 // tile, tile)
    a = g.abs().amax(-1)
    if mixed:
        s = (a / 6).clamp(2.0 ** (-9), 448).to(torch.float8_e4m3fn)
        z = g / s.float()[..., None]
        mag = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
        mids = (mag[1:] + mag[:-1]) * 0.5
        c = torch.zeros_like(z, dtype=torch.uint8)
        for i in range(7):
            c += ((z.abs() > mids[i]) | (z.abs() == mids[i]) & (i % 2 == 1)).to(torch.uint8)
        c |= torch.signbit(z).to(torch.uint8) << 3
        c = c.reshape(pages, page, 512)
        d = c[..., ::2] | c[..., 1::2] << 4
        s = s.view(torch.uint8)
    else:
        e = torch.ceil(torch.log2(a.clamp_min(0.0001) / 448))
        s = (e + 127).to(torch.uint8)
        d = (g / torch.exp2(e)[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    return torch.cat((d.reshape(pages, -1), s.reshape(pages, -1)), -1).reshape(pages, page, 1, 288 if mixed else 528)
mapping = _workload_profiles_mapping
select = _workload_profiles_select
pack = _packing_pack
guarded = _packing_guarded

def _make_cache(page, compression, window, budget, fp4, *, lengths, query_len, physical_layout, device, centers, amp, q, index_pattern, b):
    starts = [max(0, (n - query_len + 1) // compression - window) // page if window else 0 for n in lengths]
    ends = [(n // compression + page - 1) // page for n in lengths]
    counts = [end - start for start, end in zip(starts, ends)]
    pages = sum(counts) + 2
    physical = mapping(pages, physical_layout)
    kv = torch.empty(pages, page, 1, 288 if fp4 else 528, dtype=torch.uint8, device=device)
    features = torch.empty((pages, page, 8), device=device)
    for s in range(0, pages, 64):
        values = (0.8 * torch.randn(min(64, pages - s), page, 512, device=device) * amp + 0.6 * centers[torch.randint(4, (min(64, pages - s), page), device=device)] * (amp / 0.35)).to(torch.bfloat16)
        features[s:s + len(values)] = values[:, :, :8].float()
        kv[s:s + len(values)] = pack(values, fp4)
    ids = torch.full((b, query_len, budget), -1, dtype=torch.int32, device=device)
    offset = 0
    for r, (n, start, count) in enumerate(zip(lengths, starts, counts)):
        phys = physical[offset:offset + count]
        offset += count
        for j in range(query_len):
            visible = (n - query_len + j + 1) // compression
            scores = None if window else features[phys.to(device)].reshape(-1, 8) @ q[r, j, :, :8].float().mean(0)
            chosen = torch.arange(max(0, visible - window), visible) if window else select(visible, budget, index_pattern, scores)
            if len(chosen):
                slots = phys[chosen // page - start] * page + chosen % page
                ids[r, j, :len(slots)] = slots.to(device=device, dtype=torch.int32)
    return (kv, ids)

def _make_inputs(lengths, tp, ratio, query_len, domain='normal', mixed=True, physical_heads=64, device='cuda', physical_layout='scattered', index_pattern='score'):
    assert mixed is True
    assert query_len in (1, 3) and physical_heads in (64, 128) and (tp in (1, 2, 4, 8))
    assert ratio in (0, 1, 2) and all((query_len <= n <= 1048576 for n in lengths))
    b = len(lengths)
    h = physical_heads
    amp = 0.0 if domain == 'zero' else 0.35 * 2 ** (-12) if domain == 'tiny' else 0.35
    centers = torch.randn((4, 512), device=device) * 0.35
    q = 0.8 * torch.randn(b, query_len, h, 512, device=device) * 0.35 + 0.6 * centers[torch.randint(4, (b, query_len, h), device=device)]
    if domain == 'peaked':
        q *= 16
    q = q.to(torch.bfloat16)
    if domain == 'tiny':
        q *= 2 ** (-8)
    kv, ids = _make_cache(32, 1, 128, 128, False, lengths=lengths, query_len=query_len, physical_layout=physical_layout, device=device, centers=centers, amp=amp, q=q, index_pattern=index_pattern, b=b)
    extra, eids = _make_cache(128 // ratio, ratio, 0, 512, mixed, lengths=lengths, query_len=query_len, physical_layout=physical_layout, device=device, centers=centers, amp=amp, q=q, index_pattern=index_pattern, b=b) if ratio else (None, None)
    sink = torch.randn(h, device=device)
    if domain == 'sink':
        sink[:] = 10.0
        sink[0] = torch.inf
        sink[1] = -torch.inf
    x = dict(q=q, k_cache=kv, indices=ids, extra_k_cache=extra, extra_indices=eids, sm_scale=512 ** (-0.5), attn_sink=sink, out=torch.empty_like(q))
    p = {k: guarded(v) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
    return {'q':p['q'], 'k_cache':p['k_cache'], 'indices':p['indices'], 'extra_k_cache':p['extra_k_cache'], 'extra_indices':p['extra_indices'], 'sm_scale':p['sm_scale'], 'attn_sink':p['attn_sink'], 'out':p['out']}

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['out']}
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['out']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
