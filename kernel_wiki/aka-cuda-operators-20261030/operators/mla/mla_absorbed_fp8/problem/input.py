import math
import torch

def _workload_profiles_qkv(nq, nk, heads, kv_heads, dim, value_dim, device):
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
    return (q, k, v)

joint_qkv = _workload_profiles_qkv

def guarded(shape, dtype, device):
    n = 1
    for d in shape:
        n *= d
    raw = torch.full((n + 256,), -17, dtype=dtype, device=device)
    return raw[128:-128].view(shape)

def _make_paged_storage(query_lengths, kv_lengths, heads, kv_heads, head_dim, page_size, precision, causal, window_left, sink, return_lse, value_dim, device='cuda'):
    assert len(query_lengths) == len(kv_lengths) and heads % kv_heads == 0
    assert all((0 < q <= k for q, k in zip(query_lengths, kv_lengths)))
    assert (kv_heads, head_dim, value_dim, precision) == (1, 576, 512, 'fp8')
    dtype = torch.bfloat16
    qs, ks = (0.5, 0.25)
    counts = [(n + page_size - 1) // page_size for n in kv_lengths]
    pages = sum(counts) + 1
    raw_q, raw_k, raw_v = joint_qkv(sum(query_lengths), pages * page_size, heads, kv_heads, head_dim, value_dim, device)
    q = (raw_q / qs).to(torch.float8_e4m3fn)
    k = (raw_k.reshape(pages, page_size, kv_heads, head_dim).transpose(1, 2) / ks).to(torch.float8_e4m3fn)
    v = k[..., :value_dim]
    vs = ks
    del raw_q, raw_k, raw_v
    order = torch.randperm(pages - 1, device=device) + 1
    table = torch.zeros(len(counts), max(counts) + 1, dtype=torch.int32, device=device)
    cursor = 0
    for b, count in enumerate(counts):
        table[b, :count] = order[cursor:cursor + count]
        cursor += count
    ptr = [0]
    for length in query_lengths:
        ptr.append(ptr[-1] + length)
    return dict(q=q, k=k, v=v, out=guarded((q.shape[0], heads, value_dim), dtype, device), cu_seqlens_q=torch.tensor(ptr, dtype=torch.int32, device=device), seq_lens=torch.tensor(kv_lengths, dtype=torch.int32, device=device), block_table=table, softmax_scale=192 ** (-0.5), q_scale=qs, k_scale=ks, v_scale=vs, causal=causal, window_left=window_left, sinks=torch.linspace(-1, 4, heads, dtype=torch.float32, device=device) if sink else None, lse=guarded((q.shape[0], heads), torch.float32, device) if return_lse else None)

def make(**kw):
    x = _make_paged_storage(**kw)
    q, k = (x['q'], x['k'])
    ptr = x['cu_seqlens_q']
    lens = x['seq_lens']
    device = q.device
    ws = torch.zeros(128 * 1024 ** 2, dtype=torch.uint8, device=device)
    z = dict(query=q, workspace_buffer=ws, seq_lens=lens, bmm1_scale=x['softmax_scale'] * x['q_scale'] * x['k_scale'], bmm2_scale=x['v_scale'], out=x['out'], lse=x['lse'], return_lse=x['lse'] is not None, enable_pdl=False, backend='trtllm-gen')
    maxq = max(kw['query_lengths'])
    maxkv = max(kw['kv_lengths'])
    z.update(kv_cache=k, block_tables=x['block_table'])
    z.update(qk_nope_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64, max_seq_len=maxkv, sinks=x['sinks'], return_lse_base='basee')
    if len(set(kw['query_lengths'])) == 1:
        z['query'] = q.view(len(lens), maxq, *q.shape[1:])
        z['out'] = x['out'].view(len(lens), maxq, *x['out'].shape[1:])
    else:
        z.update(cum_seq_lens_q=ptr, max_q_len=maxq)
    return z

def _make_inputs(head_dim, heads, kv_heads, kv_lengths, page_size, query_lengths, causal=None, precision=None, return_lse=None, sink=None, value_dim=None, window_left=None, device='cuda'):
    if causal is not True:
        raise ValueError('mla_absorbed_fp8 requires causal=True (bottom-right causal attention)')
    shape = {'device': device, **{k: v for k, v in {'causal': causal, 'head_dim': head_dim, 'heads': heads, 'kv_heads': kv_heads, 'kv_lengths': kv_lengths, 'page_size': page_size, 'precision': precision, 'query_lengths': query_lengths, 'return_lse': return_lse, 'sink': sink, 'value_dim': value_dim, 'window_left': window_left}.items() if v is not None}}
    p = make(**shape)
    return {'query':p['query'], 'kv_cache':p['kv_cache'], 'workspace_buffer':p['workspace_buffer'], 'qk_nope_head_dim':p['qk_nope_head_dim'], 'kv_lora_rank':p['kv_lora_rank'], 'qk_rope_head_dim':p['qk_rope_head_dim'], 'block_tables':p['block_tables'], 'seq_lens':p['seq_lens'], 'max_seq_len':p['max_seq_len'], 'sparse_mla_top_k':p.get('sparse_mla_top_k', 0), 'out':p.get('out'), 'bmm1_scale':p.get('bmm1_scale', 1.0), 'bmm2_scale':p.get('bmm2_scale', 1.0), 'sinks':p.get('sinks'), 'skip_softmax_threshold_scale_factor':p.get('skip_softmax_threshold_scale_factor'), 'enable_pdl':p.get('enable_pdl'), 'backend':p.get('backend', 'auto'), 'is_var_seq':p.get('is_var_seq', True), 'uses_shared_paged_kv_idx':p.get('uses_shared_paged_kv_idx', True), 'lse':p.get('lse'), 'return_lse':p.get('return_lse', False), 'cute_dsl_impl':p.get('cute_dsl_impl', 'auto'), 'kv_scale_format':p.get('kv_scale_format', 'auto'), 'cum_seq_lens_q':p.get('cum_seq_lens_q'), 'max_q_len':p.get('max_q_len'), 'multi_ctas_kv_counter_buffer':p.get('multi_ctas_kv_counter_buffer'), 'sparse_mla_top_k_lens':p.get('sparse_mla_top_k_lens'), 'enable_dcp':p.get('enable_dcp', False), 'cp_world':p.get('cp_world', 1), 'cp_rank':p.get('cp_rank', 0), 'causal_seqlens_kv_global':p.get('causal_seqlens_kv_global'), 'use_fp16_softmax':p.get('use_fp16_softmax'), 'return_lse_base':p.get('return_lse_base')}

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
