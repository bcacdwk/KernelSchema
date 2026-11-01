import math
import torch
_workload_profiles_INPUT_RECORDS = []

def _workload_profiles_qkv(nq, nk, heads, kv_heads, dim, value_dim, device, profile='normal', scale=None):
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
        amplitude = 1.0
        prob = (logits * amplitude).softmax(-1)
        _workload_profiles_INPUT_RECORDS.append(dict(seed=torch.initial_seed(), profile=profile, nq=nq, nk=nk, heads=heads, kv_heads=kv_heads, dim=dim, shared_latent_topics=4, diagnostics_scope='base latent before caller-specific stress, quantization and masks: at most 8 queries, 256 keys and 4 heads; not final-input or full-context entropy', sampled_logit_std=float((logits * amplitude).std()), sampled_entropy=float(-(prob * prob.clamp_min(1e-30).log()).sum(-1).mean()), sampled_max_probability=float(prob.max(-1).values.mean())))
        del _workload_profiles_INPUT_RECORDS[:-2]
    return (q, k, v)
'Construct the exact production call; all data preparation is outside forward.'
import torch
joint_qkv = _workload_profiles_qkv

def guarded(shape, dtype, device):
    n = 1
    for d in shape:
        n *= d
    raw = torch.full((n + 256,), -17, dtype=dtype, device=device)
    return raw[128:-128].view(shape)

def _make_paged_storage(query_lengths, kv_lengths, heads, kv_heads, head_dim, page_size, precision='bf16', causal=True, window_left=-1, sink=False, return_lse=True, distribution='normal', value_dim=None, device='cuda', q_scale_value=None, k_scale_value=None, v_scale_value=None):
    assert len(query_lengths) == len(kv_lengths) and heads % kv_heads == 0
    assert all((0 < q <= k for q, k in zip(query_lengths, kv_lengths)))
    dv = value_dim or head_dim
    dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
    qdtype = torch.float8_e4m3fn if precision == 'fp8' else dtype
    kdtype = torch.float8_e4m3fn if precision in ('fp8', 'mixed') else dtype
    qs, ks, vs = (0.5 if precision == 'fp8' else 1.0, 0.25 if kdtype == torch.float8_e4m3fn else 1.0, 0.75 if kdtype == torch.float8_e4m3fn else 1.0)
    qs = qs if q_scale_value is None else q_scale_value
    ks = ks if k_scale_value is None else k_scale_value
    vs = vs if v_scale_value is None else v_scale_value
    counts = [(n + page_size - 1) // page_size for n in kv_lengths]
    pages = sum(counts) + 1
    raw_q, raw_k, raw_v = joint_qkv(sum(query_lengths), pages * page_size, heads, kv_heads, head_dim, dv, device, distribution, (192 if head_dim == 576 else head_dim) ** (-0.5))
    q = (raw_q / qs).to(qdtype)
    k = (raw_k.reshape(pages, page_size, kv_heads, head_dim).transpose(1, 2) / ks).to(kdtype)
    if head_dim == 576 and dv == 512:
        v = k[..., :512]
        vs = ks
    else:
        v = (raw_v.reshape(pages, page_size, kv_heads, dv).transpose(1, 2) / vs).to(kdtype)
    del raw_q, raw_k, raw_v
    if distribution == 'tiny':
        q.copy_((q.float() * 0.03125).to(qdtype))
    if distribution == 'peaked':
        q.copy_((q.float() * 3).clamp(-400, 400).to(qdtype))
    if distribution == 'zero':
        v.zero_()
    if distribution == 'balanced':
        q.zero_()
        signs = (torch.arange(page_size, device=device) % 2 * 2 - 1).float()
        v.copy_((signs[None, None, :, None] / vs).expand_as(v).to(kdtype))
    order = torch.randperm(pages - 1, device=device) + 1
    table = torch.zeros(len(counts), max(counts) + 1, dtype=torch.int32, device=device)
    cursor = 0
    for b, count in enumerate(counts):
        table[b, :count] = order[cursor:cursor + count]
        cursor += count
    ptr = [0]
    for length in query_lengths:
        ptr.append(ptr[-1] + length)
    return dict(q=q, k=k, v=v, out=guarded((q.shape[0], heads, dv), dtype, device), cu_seqlens_q=torch.tensor(ptr, dtype=torch.int32, device=device), seq_lens=torch.tensor(kv_lengths, dtype=torch.int32, device=device), block_table=table, softmax_scale=head_dim ** (-0.5), q_scale=qs, k_scale=ks, v_scale=vs, causal=causal, window_left=window_left, sinks=torch.linspace(-1, 4, heads, dtype=torch.float32, device=device) if sink else None, lse=guarded((q.shape[0], heads), torch.float32, device) if return_lse else None)

def make(**kw):
    x = _make_paged_storage(**kw)
    q, k, v = (x['q'], x['k'], x['v'])
    ptr = x['cu_seqlens_q']
    lens = x['seq_lens']
    device = q.device
    ws = torch.zeros(128 * 1024 ** 2, dtype=torch.uint8, device=device)
    z = dict(query=q, workspace_buffer=ws, seq_lens=lens, bmm1_scale=x['softmax_scale'] * x['q_scale'] * x['k_scale'], bmm2_scale=x['v_scale'], out=x['out'], lse=x['lse'], return_lse=x['lse'] is not None, enable_pdl=False, backend='trtllm-gen')
    maxq = max(kw['query_lengths'])
    maxkv = max(kw['kv_lengths'])
    z.update(kv_cache=k, block_tables=x['block_table'])
    z.update(kv_cache=(k, v), window_left=x['window_left'], sinks=x['sinks'], kv_layout='HND', out_dtype=x['out'].dtype)
    z.update(max_seq_len=maxkv, q_len_per_req=None, cum_seq_lens_q=ptr, max_q_len=maxq)
    if 'k_block_scales' in x:
        z['kv_cache_sf'] = (x['k_block_scales'], x['v_block_scales'])
    return z

def _make_inputs(head_dim, heads, kv_heads, kv_lengths, page_size, query_lengths, causal=None, distribution=None, k_scale_value=None, precision=None, return_lse=None, sink=None, v_scale_value=None, window_left=None, device='cuda'):
    shape = {'device': device, **{k: v for k, v in {'causal': causal, 'distribution': distribution, 'head_dim': head_dim, 'heads': heads, 'k_scale_value': k_scale_value, 'kv_heads': kv_heads, 'kv_lengths': kv_lengths, 'page_size': page_size, 'precision': precision, 'query_lengths': query_lengths, 'return_lse': return_lse, 'sink': sink, 'v_scale_value': v_scale_value, 'window_left': window_left}.items() if v is not None}}
    p = make(**shape)
    return {'query': p['query'], 'kv_cache': p['kv_cache'], 'workspace_buffer': p['workspace_buffer'], 'block_tables': p['block_tables'], 'seq_lens': p['seq_lens'], 'max_seq_len': p['max_seq_len'], 'bmm1_scale': p.get('bmm1_scale', 1.0), 'bmm2_scale': p.get('bmm2_scale', 1.0), 'window_left': p.get('window_left', -1), 'out': p.get('out', None), 'out_dtype': p.get('out_dtype', None), 'o_sf_scale': p.get('o_sf_scale', None), 'o_sf_vec_size': p.get('o_sf_vec_size', None), 'sinks': p.get('sinks', None), 'kv_layout': p.get('kv_layout', 'HND'), 'enable_pdl': p.get('enable_pdl', None), 'backend': p.get('backend', 'auto'), 'q_len_per_req': p.get('q_len_per_req', 1), 'o_scale': p.get('o_scale', 1.0), 'mask': p.get('mask', None), 'max_q_len': p.get('max_q_len', None), 'cum_seq_lens_q': p.get('cum_seq_lens_q', None), 'skip_softmax_threshold_scale_factor': p.get('skip_softmax_threshold_scale_factor', None), 'kv_cache_sf': p.get('kv_cache_sf', None), 'uses_shared_paged_kv_idx': p.get('uses_shared_paged_kv_idx', True), 'lse': p.get('lse', None), 'return_lse': p.get('return_lse', False), 'bmm1_scale_log2': p.get('bmm1_scale_log2', None), 'multi_ctas_kv_counter_buffer': p.get('multi_ctas_kv_counter_buffer', None), 'enable_block_sparse_attention': p.get('enable_block_sparse_attention', False), 'cp_world': p.get('cp_world', 1), 'cp_rank': p.get('cp_rank', 0), 'causal_seqlens_kv_global': p.get('causal_seqlens_kv_global', None), 'bf16q_fp8kv_transform_mode': p.get('bf16q_fp8kv_transform_mode', None), 'request_order': p.get('request_order', None), 'request_order_plan': p.get('request_order_plan', None)}

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['out', 'lse', 'workspace_buffer']}
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['out', 'lse']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
