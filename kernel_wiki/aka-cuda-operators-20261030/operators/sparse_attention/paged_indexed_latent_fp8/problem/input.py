import torch
_workload_profiles_INPUT_RECORDS = []

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

def _workload_profiles_observe(indices, page_size, *, name='main', limit=None):
    rows = indices.reshape(-1, indices.shape[-1])
    measurements = []
    previous = None
    for row in rows[:16]:
        good = row[row >= 0]
        if limit is not None:
            good = good[good < limit]
        values = set(good.cpu().tolist())
        pages = {x // page_size for x in values}
        overlap = len(values & previous) / max(1, len(values | previous)) if previous is not None else None
        measurements.append(dict(selected=len(values), unique_pages=len(pages), selected_per_page=len(values) / max(1, len(pages)), adjacent_query_jaccard=overlap))
        previous = values
    _workload_profiles_INPUT_RECORDS.append(dict(seed=torch.initial_seed(), cache=name, page_size=page_size, total_query_rows=len(rows), diagnostic_rows=measurements, scope='first at most 16 query rows; physical token indices'))
    del _workload_profiles_INPUT_RECORDS[:-4]
import math
import torch
_attention_profiles_INPUT_RECORDS = []

def _attention_profiles_qkv(nq, nk, heads, kv_heads, dim, value_dim, device, profile='normal', scale=None):
    """THD operands. Common latent topics induce QK affinity and KV structure.

    Four latent topics and eight fixed sinusoidal projection components are
    synthetic assumptions. They are not a reproduction of learned weights.
    """
    assert heads % kv_heads == 0
    position = torch.arange(max(dim, value_dim), device=device).float()
    basis = torch.sin((torch.arange(1, 9, device=device).float()[:, None] * position[None, :] + 0.3) / 17.0)
    basis = basis / basis.square().mean(-1, keepdim=True).sqrt() / math.sqrt(8)
    latent = torch.randn((kv_heads, 4, 8), device=device)
    centers = latent @ basis[:, :dim]
    vcenters = latent @ torch.cos((torch.arange(1, 9, device=device).float()[:, None] * position[None, :value_dim] + 0.7) / 19.0) / 2.0
    k = torch.empty((nk, kv_heads, dim), device=device)
    v = torch.empty((nk, kv_heads, value_dim), device=device)
    kgain = torch.linspace(0.7, 1.3, kv_heads, device=device)[None, :, None]
    cgain = torch.linspace(0.7, 1.3, dim, device=device)
    hidx = torch.arange(kv_heads, device=device)[None, :]
    for start in range(0, nk, 1024):
        count = min(1024, nk - start)
        topic = torch.randint(4, (count, kv_heads), device=device)
        k[start:start + count] = (0.8 * torch.randn((count, kv_heads, dim), device=device) + 0.6 * centers[hidx, topic]) * kgain * cgain
        v[start:start + count] = (0.8 * torch.randn((count, kv_heads, value_dim), device=device) + 0.6 * vcenters[hidx, topic]) * kgain
    q = torch.empty((nq, heads, dim), device=device)
    qcenters = centers.repeat_interleave(heads // kv_heads, 0)
    qhead = torch.arange(heads, device=device)[None, :]
    qgain = torch.linspace(0.65, 1.35, heads, device=device)[None, :, None]
    for start in range(0, nq, 1024):
        count = min(1024, nq - start)
        topic = torch.randint(4, (count, heads), device=device)
        q[start:start + count] = (0.8 * torch.randn((count, heads, dim), device=device) + 0.6 * qcenters[qhead, topic]) * qgain * cgain
    if dim == 576:
        q[..., 512:] *= 0.5
        k[..., 512:] *= 0.5
    qs = q[:min(nq, 8), ::max(1, heads // 4)].float()
    hs = torch.arange(heads, device=device)[::max(1, heads // 4)] // (heads // kv_heads)
    ks = k[:min(nk, 256), hs].float()
    if nq and nk:
        logits = torch.einsum('qhd,khd->qhk', qs, ks) * (scale or dim ** (-0.5))
        amplitude = 0.03125 if profile == 'tiny' else 3.0 if profile == 'peaked' else 1.0
        prob = (logits * amplitude).softmax(-1)
        _attention_profiles_INPUT_RECORDS.append(dict(seed=torch.initial_seed(), profile=profile, nq=nq, nk=nk, heads=heads, kv_heads=kv_heads, dim=dim, shared_latent_topics=4, diagnostics_scope='pre-quantization subsample: at most 8 queries, 256 keys and 4 heads; not full-context entropy', sampled_logit_std=float((logits * amplitude).std()), sampled_entropy=float(-(prob * prob.clamp_min(1e-30).log()).sum(-1).mean()), sampled_max_probability=float(prob.max(-1).values.mean())))
        del _attention_profiles_INPUT_RECORDS[:-2]
    return (q, k, v)
'Prepare native HND cache, query/table axes, folded scales and scratch.'
import torch
mapping = _workload_profiles_mapping
select = _workload_profiles_select
observe = _workload_profiles_observe
joint_qkv = _attention_profiles_qkv

def guarded(x):
    raw = torch.full((x.numel() + 256,), 73, dtype=x.dtype, device=x.device)
    y = raw[128:-128].view(x.shape)
    y.copy_(x)
    return y

def _make_inputs(query_lengths, lengths, heads, page_size, topk, dim=576, q_scale=0.5, k_scale=0.25, v_scale=0.25, domain='normal', device='cuda', physical_layout='scattered', index_pattern='score'):
    assert heads in (8, 16, 32, 64, 128) and dim in (512, 576) and (page_size in (1, 32, 64, 128))
    assert topk in (128, 512, 1024, 2048, 2052) and all((0 < q <= n <= 1048576 for q, n in zip(query_lengths, lengths)))
    assert min(q_scale, k_scale, v_scale) > 0
    t = sum(query_lengths)
    amp = 0.0 if domain == 'zero' else 0.35 * 2 ** (-10) if domain == 'tiny' else 0.35
    counts = [(n + page_size - 1) // page_size for n in lengths]
    pages = sum(counts) + 2
    physical = mapping(pages, physical_layout)
    raw_q, raw_k, _ = joint_qkv(t, pages * page_size, heads, 1, dim, 0, device)
    q = (raw_q * (0.35 * 16 if domain == 'peaked' else 0.35) / q_scale).to(torch.float8_e4m3fn)
    kv = torch.empty(pages, 1, page_size, dim, dtype=torch.float8_e4m3fn, device=device)
    for s in range(0, pages, 64):
        kv[s:s + 64] = (raw_k[s * page_size:(s + 64) * page_size].reshape(-1, 1, page_size, dim) * amp / k_scale).to(kv.dtype)
    features = raw_k[:, :, :8].reshape(pages, page_size, 8)
    del raw_k
    ids = torch.full((t, topk), -1, dtype=torch.int32, device=device)
    lens = torch.empty(t, dtype=torch.int32, device=device)
    row = offset = 0
    for nq, n, count in zip(query_lengths, lengths, counts):
        phys = physical[offset:offset + count]
        offset += count
        for j in range(nq):
            visible = n - nq + j + 1
            scores = features[phys.to(device)].reshape(-1, 8) @ raw_q[row, :, :8].mean(0)
            chosen = select(visible, topk, index_pattern, scores)
            chosen = chosen[torch.randperm(len(chosen))]
            slots = phys[chosen // page_size] * page_size + chosen % page_size
            ids[row, :len(slots)] = slots.to(device=device, dtype=torch.int32)
            lens[row] = len(slots)
            row += 1
    observe(ids, page_size)
    scale = (256 if dim == 512 else 192) ** (-0.5)
    sms = torch.cuda.get_device_properties(device).multi_processor_count if str(device) != 'cpu' else 148
    x = dict(query=q.unsqueeze(1), kv_cache=kv, block_tables=ids.unsqueeze(1), seq_lens=lens, sparse_mla_top_k_lens=lens.clone() if dim == 512 else None, bmm1_scale=scale * q_scale * k_scale, bmm2_scale=v_scale, workspace_buffer=torch.zeros(128 * 1024 ** 2, dtype=torch.uint8, device=device), multi_ctas_kv_counter_buffer=torch.zeros((max(t * heads, sms) + 7) // 8 * 8 * 4, dtype=torch.uint8, device=device), out=torch.empty(t, 1, heads, 512, dtype=torch.bfloat16, device=device))
    p = {k: guarded(v) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
    return {'query': p['query'], 'kv_cache': p['kv_cache'], 'block_tables': p['block_tables'], 'seq_lens': p['seq_lens'], 'sparse_mla_top_k_lens': p['sparse_mla_top_k_lens'], 'bmm1_scale': p['bmm1_scale'], 'bmm2_scale': p['bmm2_scale'], 'workspace_buffer': p['workspace_buffer'], 'multi_ctas_kv_counter_buffer': p['multi_ctas_kv_counter_buffer'], 'out': p['out']}

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['out', 'multi_ctas_kv_counter_buffer', 'workspace_buffer']}
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['out']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
