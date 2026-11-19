import torch

def _workload_profiles_mapping(count, layout, device='cpu'):
    if layout == 'contiguous':
        return torch.arange(count, device=device)
    if layout == 'scattered':
        return torch.randperm(count, device=device)
    raise ValueError(layout)

import math

def _attention_profiles_qk(nq, nk, heads, device):
    """Generate correlated Q/K from four synthetic latent topics."""
    dim, kv_heads = (512, 1)
    position = torch.arange(dim, device=device).float()
    basis = torch.sin((torch.arange(1, 9, device=device).float()[:, None] * position[None, :] + 0.3) / 17.0)
    basis = basis / basis.square().mean(-1, keepdim=True).sqrt() / math.sqrt(8)
    latent = torch.randn((kv_heads, 4, 8), device=device)
    centers = latent @ basis[:, :dim]
    k = torch.empty((nk, kv_heads, dim), device=device)
    kgain = torch.linspace(0.7, 1.3, kv_heads, device=device)[None, :, None]
    cgain = torch.linspace(0.7, 1.3, dim, device=device)
    hidx = torch.arange(kv_heads, device=device)[None, :]
    for start in range(0, nk, 1024):
        count = min(1024, nk - start)
        topic = torch.randint(4, (count, kv_heads), device=device)
        k[start:start + count] = (0.8 * torch.randn((count, kv_heads, dim), device=device) + 0.6 * centers[hidx, topic]) * kgain * cgain
    q = torch.empty((nq, heads, dim), device=device)
    qcenters = centers.repeat_interleave(heads // kv_heads, 0)
    qhead = torch.arange(heads, device=device)[None, :]
    qgain = torch.linspace(0.65, 1.35, heads, device=device)[None, :, None]
    for start in range(0, nq, 1024):
        count = min(1024, nq - start)
        topic = torch.randint(4, (count, heads), device=device)
        q[start:start + count] = (0.8 * torch.randn((count, heads, dim), device=device) + 0.6 * qcenters[qhead, topic]) * qgain * cgain
    return (q, k)

mapping = _workload_profiles_mapping
joint_qk = _attention_profiles_qk

def _make_inputs(query_lens: list[int], seq_lens_values: list[int], num_heads: int, compress_ratio: int, index_topk: int, page_size: int, distribution: str='normal', device: str='cuda', index_pattern: str='random', physical_layout: str='scattered'):
    assert compress_ratio in (0, 1, 2, 4, 128) and num_heads in (8, 16, 32, 64, 128)
    assert index_topk in (512, 1024)
    assert len(query_lens) == len(seq_lens_values) and query_lens
    assert all((0 < q <= s <= 1048576 for q, s in zip(query_lens, seq_lens_values)))
    assert distribution in ('normal', 'tiny', 'peaked', 'strong_sink')
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
    q, latent = joint_qk(cu[-1], nswa + nglob, num_heads, device)
    q = q * 0.5
    swa = latent[:nswa].reshape(-1, page, dim) * 0.25 - 0.0625
    glob = latent[nswa:].reshape(-1, compressed_page, dim) * 0.25 + 0.0625
    selection_features = glob[:, :, :8].float().clone()
    del latent
    if distribution == 'tiny':
        swa *= 2.0 ** (-10)
        glob *= 2.0 ** (-10)
    elif distribution == 'peaked':
        q *= 6
        swa *= 4
        glob *= 4
    q, swa, glob = [x.to(torch.bfloat16) for x in (q, swa, glob)]
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
                    scores = feature @ q[row, :, :8].float().mean(0)
                    selected = scores[:candidates].topk(index_topk).indices.cpu()
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
    sinks = torch.randn((num_heads,), dtype=torch.float32, device=device)
    if distribution == 'strong_sink':
        sinks[::2] += 8
    p = dict(query=q, swa_kv_cache=swa, workspace_buffer=torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device), sparse_indices=indices.to(device), compressed_kv_cache=glob, sparse_topk_lens=lens.to(device), seq_lens=torch.tensor(seq_lens_values, dtype=torch.int32, device=device), out=torch.full_like(q, float('nan')), bmm1_scale=dim ** (-0.5), bmm2_scale=1.0, sinks=sinks, kv_layout='HND', cum_seq_lens_q=torch.tensor(cu, dtype=torch.int32, device=device), max_q_len=max(query_lens), enable_pdl=None, swa_topk_lens=None, extra_sparse_indices=None, extra_sparse_topk_lens=None)
    return {'query':p['query'], 'swa_kv_cache':p['swa_kv_cache'], 'workspace_buffer':p['workspace_buffer'], 'sparse_indices':p['sparse_indices'], 'compressed_kv_cache':p.get('compressed_kv_cache'), 'sparse_topk_lens':p.get('sparse_topk_lens'), 'seq_lens':p.get('seq_lens'), 'out':p.get('out'), 'bmm1_scale':p.get('bmm1_scale', 1.0), 'bmm2_scale':p.get('bmm2_scale', 1.0), 'sinks':p.get('sinks'), 'kv_layout':p.get('kv_layout', 'HND'), 'cum_seq_lens_q':p.get('cum_seq_lens_q'), 'max_q_len':p.get('max_q_len'), 'enable_pdl':p.get('enable_pdl'), 'swa_topk_lens':p.get('swa_topk_lens'), 'extra_sparse_indices':p.get('extra_sparse_indices'), 'extra_sparse_topk_lens':p.get('extra_sparse_topk_lens')}

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
