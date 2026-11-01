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
'V4 Flash/Pro E4M3 sparse MLA: C4 top-k, C128 all compressed, SWA128.\n\nModel-derived synthetic cache allocation and legal selection, not traffic or\nindexer validation. Cache writes/selection/RoPE are outside the timed callable.\n'
import torch
mapping = _workload_profiles_mapping
select = _workload_profiles_select
observe = _workload_profiles_observe
joint_qkv = _attention_profiles_qkv

def _make_inputs(query_lens: list[int], seq_lens_values: list[int], num_heads: int, compress_ratio: int, index_topk: int, page_size: int, distribution: str='normal', device: str='cuda', q_scale: float=1.0, kv_scale: float=1.0, index_pattern: str='random', physical_layout: str='scattered'):
    assert compress_ratio in (0, 1, 2, 4, 128) and num_heads in (8, 16, 32, 64, 128)
    assert index_topk in (512, 1024)
    assert len(query_lens) == len(seq_lens_values) and query_lens
    assert all((0 < q <= s <= 1048576 for q, s in zip(query_lens, seq_lens_values)))
    assert distribution in ('normal', 'tiny', 'peaked', 'strong_sink', 'zero', 'quant_ties', 'nonunit_scale')
    assert q_scale > 0 and kv_scale > 0
    assert page_size in (128, 256) and page_size % max(1, compress_ratio) == 0
    assert index_pattern in ('random', 'local', 'strided', 'shared_prefix', 'score')
    dim, page = (512, page_size)
    compressed_page = page // max(1, compress_ratio)
    cu = [0]
    for length in query_lens:
        cu.append(cu[-1] + length)
    swa_pages = [list(range(max(0, s - q - 127) // page, (s + page - 1) // page)) for q, s in zip(query_lens, seq_lens_values)]
    perm = mapping(sum(map(len, swa_pages)), physical_layout).tolist()
    swa_tables, used = ([], 0)
    for pages in swa_pages:
        swa_tables.append(dict(zip(pages, perm[used:used + len(pages)])))
        used += len(pages)
    comp_counts = [(s // compress_ratio + compressed_page - 1) // compressed_page if compress_ratio else 0 for s in seq_lens_values]
    perm = mapping(sum(comp_counts), physical_layout).tolist()
    comp_tables, used = ([], 0)
    for count in comp_counts:
        comp_tables.append(perm[used:used + count])
        used += count
    nswa = sum(map(len, swa_pages)) * page
    nglob = max(1, sum(comp_counts)) * compressed_page
    q, latent, _ = joint_qkv(cu[-1], nswa + nglob, num_heads, 1, dim, 0, device)
    q = q * 0.5
    swa = latent[:nswa].reshape(-1, page, dim) * 0.25 - 0.0625
    glob = latent[nswa:].reshape(-1, compressed_page, dim) * 0.25 + 0.0625
    selection_features = glob[:, :, :8].float().clone()
    del latent
    if distribution == 'tiny':
        swa *= 2.0 ** (-6)
        glob *= 2.0 ** (-6)
    elif distribution == 'peaked':
        q *= 6
        swa *= 4
        glob *= 4
    elif distribution == 'zero':
        q.zero_()
        swa.zero_()
        glob.zero_()
    elif distribution == 'quant_ties':
        mids = torch.tensor([1.0625, 1.1875, 1.3125, 1.4375, 1.5625, 1.6875, 1.8125, 1.9375], device=device)
        q[..., :8] = mids
        swa[..., :8] = mids
        glob[..., :8] = -mids
    q = (q / q_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    swa = (swa / kv_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    glob = (glob / kv_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    if compress_ratio == 0:
        glob = swa
        width = 0
    elif compress_ratio in (1, 2, 4):
        width = index_topk
    else:
        count = max(1, max(seq_lens_values) // 128)
        width = max(128, 1 << (count - 1).bit_length())
    indices = torch.full((cu[-1], 128 + width), -1, dtype=torch.int32)
    lens = torch.full((cu[-1],), 128, dtype=torch.int32)
    for request, (nq, final_len) in enumerate(zip(query_lens, seq_lens_values)):
        for j in range(nq):
            row, visible = (cu[request] + j, final_len - nq + j + 1)
            window = [swa_tables[request][t // page] * page + t % page for t in range(max(0, visible - 128), visible)]
            indices[row, :len(window)] = torch.tensor(window, dtype=torch.int32)
            if compress_ratio:
                candidates = visible // compress_ratio
                if compress_ratio == 128 or candidates <= index_topk:
                    selected = torch.arange(candidates)
                elif index_pattern == 'score':
                    feature = selection_features[torch.tensor(comp_tables[request], device=device)].reshape(-1, 8)
                    selected = select(candidates, index_topk, index_pattern, feature @ q[row, :, :8].float().mean(0))
                elif index_pattern == 'local':
                    selected = torch.arange(candidates - index_topk, candidates)
                elif index_pattern == 'strided':
                    selected = torch.linspace(0, candidates - 1, index_topk).long()
                elif index_pattern == 'shared_prefix':
                    selected = torch.arange(index_topk)
                else:
                    selected = torch.randperm(candidates)[:index_topk]
                chosen = [comp_tables[request][t // compressed_page] * compressed_page + t % compressed_page for t in selected.tolist()]
                indices[row, 128:128 + len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
                lens[row] += len(chosen)
    observe(indices[:, :128], page, name='swa')
    if width:
        observe(indices[:, 128:], compressed_page, name='compressed')
    sinks = torch.randn((num_heads,), dtype=torch.float32, device=device)
    if distribution == 'strong_sink':
        sinks[::2] += 8
    p = dict(query=q, swa_kv_cache=swa, workspace_buffer=torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device), sparse_indices=indices.to(device), compressed_kv_cache=glob, sparse_topk_lens=lens.to(device), seq_lens=torch.tensor(seq_lens_values, dtype=torch.int32, device=device), out=torch.full(q.shape, float('nan'), dtype=torch.bfloat16, device=device), bmm1_scale=dim ** (-0.5) * q_scale * kv_scale, bmm2_scale=kv_scale, sinks=sinks, kv_layout='HND', cum_seq_lens_q=torch.tensor(cu, dtype=torch.int32, device=device), max_q_len=max(query_lens), enable_pdl=None, swa_topk_lens=None, extra_sparse_indices=None, extra_sparse_topk_lens=None)
    return {'query': p['query'], 'swa_kv_cache': p['swa_kv_cache'], 'workspace_buffer': p['workspace_buffer'], 'sparse_indices': p['sparse_indices'], 'compressed_kv_cache': p.get('compressed_kv_cache', None), 'sparse_topk_lens': p.get('sparse_topk_lens', None), 'seq_lens': p.get('seq_lens', None), 'out': p.get('out', None), 'bmm1_scale': p.get('bmm1_scale', 1.0), 'bmm2_scale': p.get('bmm2_scale', 1.0), 'sinks': p.get('sinks', None), 'kv_layout': p.get('kv_layout', 'HND'), 'cum_seq_lens_q': p.get('cum_seq_lens_q', None), 'max_q_len': p.get('max_q_len', None), 'enable_pdl': p.get('enable_pdl', None), 'swa_topk_lens': p.get('swa_topk_lens', None), 'extra_sparse_indices': p.get('extra_sparse_indices', None), 'extra_sparse_topk_lens': p.get('extra_sparse_topk_lens', None)}

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['out', 'workspace_buffer']}
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['out']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
