import math
import torch
_workload_profiles_LAST_STATS = {}
_workload_profiles_INPUT_RECORDS = []

def _workload_profiles_matrix_values(shape, device, std=1.0, role='projection'):
    x = torch.randn(shape, device=device)
    channel = torch.linspace(0.65, 1.35, shape[-1], device=device)
    x.mul_(channel)
    if role == 'fc2':
        gate = torch.randn_like(x)
        up = 0.25 * gate + math.sqrt(1 - 0.25 ** 2) * x
        x = torch.nn.functional.silu(gate) * up
    elif role == 'weight':
        x.mul_(torch.linspace(0.7, 1.3, shape[-2], device=device)[:, None])
    elif role != 'projection':
        raise ValueError(role)
    x.mul_(std)
    return x

def _workload_profiles_routes(shape, tokens, experts, topk, device, rank=0, world=1):
    """IDs and normalized weights derive from the SAME router logits.

    Adjacent groups of eight tokens share a feature, with token-specific noise.
    Bias profiles control expert imbalance without changing the forward ABI.
    """
    assert 0 < topk <= experts and experts % world == 0
    mode = shape.get('routing', 'balanced')
    features = torch.randn(((tokens + 7) // 8, 8), device=device)
    features = features.repeat_interleave(8, 0)[:tokens]
    features = 0.6 * features + 0.8 * torch.randn((tokens, 8), device=device)
    projection = torch.randn((8, experts), device=device) / math.sqrt(8)
    logits = features @ projection + 0.35 * torch.randn((tokens, experts), device=device)
    expert = torch.arange(experts, device=device)
    if mode == 'skewed':
        logits += 1.5 * torch.cos(expert.float() * 2 * math.pi / experts)
    elif mode == 'hotspot':
        logits[:, :topk] += 32.0
    elif mode in ('remote', 'local'):
        assert world > 1
        destination = (rank + 1) % world if mode == 'remote' else rank
        logits += (expert // (experts // world) == destination).float() * 4.0
    elif mode not in ('balanced', 'uniform', 'empty'):
        raise ValueError('unknown routing profile: ' + mode)
    selected, ids = logits.topk(topk, dim=-1)
    weights = selected.softmax(-1)
    if mode == 'empty':
        ids.fill_(-1)
    if shape.get('masked_routes'):
        ids[::3, -1] = -1
    valid_tokens = shape.get('rank_lengths', [tokens] * world)[rank]
    ids[valid_tokens:] = -1
    weights.masked_fill_(ids < 0, 0.0)
    valid = ids[ids >= 0]
    loads = torch.bincount(valid, minlength=experts)
    destinations = torch.bincount(valid // (experts // world), minlength=world)
    _workload_profiles_LAST_STATS['router'] = dict(seed=torch.initial_seed(), profile=mode, rank=rank, world_size=world, valid_tokens=valid_tokens, routed_rows=int(valid.numel()), active_experts=int((loads > 0).sum()), expert_rows=loads.cpu().tolist(), destination_rows=destinations.cpu().tolist(), max_to_mean_load=float(loads.max()) / max(float(loads.float().mean()), 1e-30), mean_top1_weight=float(weights.max(-1).values.mean()) if tokens else 0.0, ids_and_weights_from_same_logits=True)
    _workload_profiles_INPUT_RECORDS.append(dict(_workload_profiles_LAST_STATS['router']))
    del _workload_profiles_INPUT_RECORDS[:-2]
    return (ids, weights)
'Canonical stored-precision inputs; no dependency on a serving framework.'
import math
import torch
matrix_values = _workload_profiles_matrix_values
routes = _workload_profiles_routes

def guarded(shape, dtype, device):
    size = math.prod(shape)
    backing = torch.full((size + 128,), -17, dtype=dtype, device=device)
    return backing[64:-64].view(shape)

def quantize(value):
    return (value.to(torch.bfloat16), None)

def matrix(shape, device, std=1.0, zero=False):
    if len(shape) == 3:
        chunks = [matrix(shape[1:], device, std, zero) for _ in range(shape[0])]
        return (torch.stack([x[0] for x in chunks]), None if chunks[0][1] is None else torch.stack([x[1] for x in chunks]))
    value = matrix_values(shape, device, std, role='weight' if std != 1.0 else 'projection')
    if zero:
        value.zero_()
    return quantize(value)

def interleave(t):
    e, n, k = t.shape
    return torch.stack((t[:, :n // 2].reshape(e, n // 16, 8, k), t[:, n // 2:].reshape(e, n // 16, 8, k)), 2).reshape(e, n, k).contiguous()

def weights(q, gated):
    if gated:
        q = interleave(q)
    return q.contiguous()

def make(shape, seed=0, device='cuda'):
    torch.manual_seed(seed)
    t, h, i, e, k = map(shape.__getitem__, ('tokens', 'hidden', 'intermediate', 'local_experts', 'topk'))
    act = shape.get('activation', 'swiglu')
    aq, sa = matrix((t, h), device, zero=shape.get('distribution') == 'zero')
    w1, s1 = matrix((e, 2 * i, h), device, std=h ** (-0.5))
    w2, s2 = matrix((e, h, i), device, std=(i * shape.get('expert_tp', 1)) ** (-0.5))
    align = 1920
    cap = (t + align - 1) // align * align
    mode = 'bf16xbf16'
    size, slicer = heap_layout(1, e, cap, k, h, i, mode != 'bf16xbf16', device)
    buf = torch.zeros(size, dtype=torch.int8, device=device)
    v = slicer(buf)
    v[0][:t].copy_(aq)
    ids, scores = routes(shape, t, e, k, device)
    v[2].fill_(-1)
    v[2][:t].copy_(ids)
    v[3].zero_()
    v[3][:t].copy_(scores)
    p = dict(y=guarded((t, h), torch.bfloat16, device), l1_weights=weights(w1, True), l2_weights=weights(w2, False), shared_l1_weights=None, shared_l2_weights=None, cumulative_local_expert_recv_stats=None, sym_buffer=buf, sym_buffer_ptrs=[buf.data_ptr()], rank_idx=0, num_max_tokens_per_rank=cap, num_experts=e, num_topk=k, activation=act, activation_clamp_opt=shape.get('activation_limit'), fast_math=False, activation_alpha=4.0 if act == 'situ' else shape.get('activation_alpha', 1.0), activation_beta=25.0 if act == 'situ' else shape.get('activation_beta', 0.0))
    return p

def _make_inputs(hidden, intermediate, local_experts, tokens, topk, activation_alpha=None, activation_beta=None, activation_limit=None, distribution=None, masked_routes=None, routing=None, model_geometry=None, expert_tp=None, device='cuda'):
    shape = {'device': device, **{k: v for k, v in {'activation_alpha': activation_alpha, 'activation_beta': activation_beta, 'activation_limit': activation_limit, 'distribution': distribution, 'hidden': hidden, 'intermediate': intermediate, 'local_experts': local_experts, 'masked_routes': masked_routes, 'routing': routing, 'tokens': tokens, 'topk': topk, 'model_geometry': model_geometry, 'expert_tp': expert_tp}.items() if v is not None}}
    p = make({k: v for k, v in shape.items() if k != 'device'}, seed=torch.initial_seed(), device=device)
    return {'y': p['y'], 'l1_weights': p['l1_weights'], 'l2_weights': p['l2_weights'], 'shared_l1_weights': p['shared_l1_weights'], 'shared_l2_weights': p['shared_l2_weights'], 'cumulative_local_expert_recv_stats': p['cumulative_local_expert_recv_stats'], 'sym_buffer': p['sym_buffer'], 'sym_buffer_ptrs': p['sym_buffer_ptrs'], 'rank_idx': p['rank_idx'], 'num_max_tokens_per_rank': p['num_max_tokens_per_rank'], 'num_experts': p['num_experts'], 'num_topk': p['num_topk'], 'activation': p['activation'], 'activation_clamp_opt': p['activation_clamp_opt'], 'fast_math': p['fast_math'], 'activation_alpha': p['activation_alpha'], 'activation_beta': p['activation_beta']}
"""Only native byte-layout metadata; no numerical implementation."""
def heap_layout(ranks, experts, cap, topk, hidden, intermediate, with_sf, device):
    """Fixed production heap ABI: signals, source metadata, operands, ring, combine.

    The 1920-token alignment and ring capacity describe allocation only;
    candidates remain free to choose their own execution schedule.
    """
    ceil = lambda n, d: (n + d - 1) // d
    align = lambda n, a: ceil(n, a) * a
    local = experts // ranks
    routed = ranks * cap * min(topk, local)
    clusters = torch.cuda.get_device_properties(device).multi_processor_count // 2
    b1, b2 = intermediate // 128, hidden // 256
    ring = 0
    blocks = (8, 16, 32, 64, 96, 128, 192, 240)
    for bm in blocks:
        pool_blocks = ceil(routed, bm) + local
        warmup = max(ceil(ceil(clusters, b2) * b1, clusters), ceil(b1 + (pool_blocks - 1) * max(b1 - b2, 0), clusters) + 1)
        warmup = min(warmup, ceil(pool_blocks * b1, clusters))
        live = ceil(min(warmup * clusters, pool_blocks * b1), b1)
        live += ceil(pool_blocks * max(b2 - b1, 0), b2) + ceil(clusters, min(b1, b2))
        ring = max(ring, min(pool_blocks, live) * bm)
    ring = align(ring, 1920)
    sf_ring = max(ring // bm * align(bm, 128) for bm in blocks) if with_sf else 0
    signals = align(align(align(44, 128) + 72 * 8, 128) + 72 * 8, 128)
    signals = align(signals + 2 * 2048 * 8 + 512 * 8 + (1 << 20) * 20 + (1 << 15) * 4, 128)
    pool = align(routed + local * 239, 1920)
    offset = align(signals + experts * cap * 4 + pool * 12, 16)
    dtype = torch.float8_e4m3fn if with_sf else torch.bfloat16
    operands = []
    for shape, dt in [((cap, hidden), dtype), ((cap, hidden // 128), torch.int32) if with_sf else (None, None), ((cap, topk), torch.int64), ((cap, topk), torch.float32)]:
        if shape is None:
            operands.append(None)
            continue
        size = math.prod(shape) * torch.empty((), dtype=dt).element_size()
        operands.append((offset, size, shape, dt))
        offset += size
    offset += ring * (hidden + intermediate) * (1 if with_sf else 2)
    offset += sf_ring * (hidden + intermediate) // 32 + ring * 4
    offset += topk * cap * hidden * 2
    def slice_inputs(buffer):
        raw = buffer.view(torch.uint8).flatten()
        return tuple(None if part is None else raw[part[0]:part[0] + part[1]].view(part[3]).view(part[2]) for part in operands)
    return offset, slice_inputs

def _heap_views(p):
    w = p['l1_weights']
    q = w[0] if isinstance(w, tuple) else w
    _, slicer = heap_layout(len(p['sym_buffer_ptrs']), p['num_experts'], p['num_max_tokens_per_rank'], p['num_topk'], p['y'].shape[-1], q.shape[1] // 2, isinstance(w, tuple), p['y'].device)
    return slicer(p['sym_buffer'])


def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['y', 'cumulative_local_expert_recv_stats', 'sym_buffer']}
    heap = x['sym_buffer']
    raw = heap.view(torch.uint8).flatten()
    operands = sorted((part.data_ptr() - heap.data_ptr(), part.data_ptr() - heap.data_ptr() + part.numel() * part.element_size()) for part in _heap_views(x)[:4] if part is not None)
    cursor = 0
    regions = []
    for lo, hi in operands + [(raw.numel(), raw.numel())]:
        if lo > cursor:
            regions.append(raw[cursor:lo])
        cursor = max(cursor, hi)
    views['sym_buffer'] = regions
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['y', 'cumulative_local_expert_recv_stats']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
