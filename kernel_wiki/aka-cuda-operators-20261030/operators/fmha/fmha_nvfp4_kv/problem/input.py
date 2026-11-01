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

def _make_unquantized_pages(query_lengths, kv_lengths, heads, kv_heads, head_dim, page_size, precision='bf16', causal=True, window_left=-1, sink=False, return_lse=True, distribution='normal', device='cuda'):
    assert len(query_lengths) == len(kv_lengths) and heads % kv_heads == 0
    assert all((0 < q <= k for q, k in zip(query_lengths, kv_lengths)))
    assert precision == 'bf16' and head_dim in (64, 128, 256)
    counts = [(n + page_size - 1) // page_size for n in kv_lengths]
    pages = sum(counts) + 1
    raw_q, raw_k, raw_v = joint_qkv(sum(query_lengths), pages * page_size, heads, kv_heads, head_dim, head_dim, device)
    q = raw_q.to(torch.bfloat16)
    k = raw_k.reshape(pages, page_size, kv_heads, head_dim).transpose(1, 2).to(torch.bfloat16)
    v = raw_v.reshape(pages, page_size, kv_heads, head_dim).transpose(1, 2).to(torch.bfloat16)
    del raw_q, raw_k, raw_v
    if distribution == 'tiny':
        q.copy_((q.float() * 0.03125).to(torch.bfloat16))
    if distribution == 'peaked':
        q.copy_((q.float() * 3).clamp(-400, 400).to(torch.bfloat16))
    if distribution == 'zero':
        v.zero_()
    order = torch.randperm(pages - 1, device=device) + 1
    table = torch.zeros(len(counts), max(counts) + 1, dtype=torch.int32, device=device)
    cursor = 0
    for b, count in enumerate(counts):
        table[b, :count] = order[cursor:cursor + count]
        cursor += count
    ptr = [0]
    for length in query_lengths:
        ptr.append(ptr[-1] + length)
    return dict(q=q, k=k, v=v, out=guarded((q.shape[0], heads, head_dim), torch.bfloat16, device), cu_seqlens_q=torch.tensor(ptr, dtype=torch.int32, device=device), seq_lens=torch.tensor(kv_lengths, dtype=torch.int32, device=device), block_table=table, softmax_scale=head_dim ** (-0.5), q_scale=1.0, k_scale=1.0, v_scale=1.0, causal=causal, window_left=window_left, sinks=torch.linspace(-1, 4, heads, dtype=torch.float32, device=device) if sink else None, lse=guarded((q.shape[0], heads), torch.float32, device) if return_lse else None)

def quantize_pages(cache, swizzle_v=False):
    """NVFP4: global FP32 gain, E4M3 block scales, nearest-even E2M1 pairs.

    Quantization reads the stored BF16 cache. K scales stay linear; V scales
    interleave groups of four tokens. No hidden unquantized truth is retained.
    """
    shape = cache.shape
    gain = torch.tensor([448.0 / max(cache.float().abs().amax().item(), 1e-12)], dtype=torch.float32, device=cache.device)
    blocks = cache.float().reshape(*shape[:-1], shape[-1] // 16, 16)
    peak = blocks.abs().amax(-1)
    scales = (gain * (peak * (1.0 / 6.0))).to(torch.float8_e4m3fn)
    decode_scale = scales.float() * gain.reciprocal()
    encode_scale = torch.where(peak != 0, decode_scale.reciprocal(), 0.0)
    values = (blocks * encode_scale.unsqueeze(-1)).reshape(shape)
    levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    magnitude = values.abs()
    code = torch.zeros_like(values, dtype=torch.uint8)
    for j in range(7):
        midpoint = (levels[j] + levels[j + 1]) / 2
        code += (magnitude > midpoint if j % 2 == 0 else magnitude >= midpoint).to(torch.uint8)
    code |= torch.signbit(values).to(torch.uint8) << 3
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()
    if swizzle_v:
        pages, heads, tokens, groups = scales.shape
        scales = scales.view(torch.uint8).reshape(pages, heads, tokens // 4, 4, 4, groups // 4).permute(0, 1, 2, 4, 5, 3).reshape(pages, heads, tokens, groups).contiguous().view(torch.float8_e4m3fn)
    return packed, scales, gain.reciprocal().item()

def _make_paged_storage(**kw):
    kw = dict(kw, precision='bf16')
    x = _make_unquantized_pages(**kw)
    x['q_scale'] = 0.5
    x['q'] = (x['q'].float() / x['q_scale']).to(torch.float8_e4m3fn)
    x['k'], x['k_block_scales'], x['k_scale'] = quantize_pages(x['k'])
    x['v'], x['v_block_scales'], x['v_scale'] = quantize_pages(x['v'], swizzle_v=True)
    return x

def pointer(lengths, device):
    return torch.tensor([0] + list(__import__('itertools').accumulate(lengths)), dtype=torch.int32, device=device)

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
    z.update(max_q_len=maxq, max_kv_len=maxkv, batch_size=len(lens), cum_seq_lens_q=ptr, cum_seq_lens_kv=pointer(kw['kv_lengths'], device), causal=x['causal'])
    if 'k_block_scales' in x:
        z['kv_cache_sf'] = (x['k_block_scales'], x['v_block_scales'])
    return z

def _make_inputs(head_dim, heads, kv_heads, kv_lengths, page_size, query_lengths, causal=None, distribution=None, precision=None, return_lse=None, sink=None, window_left=None, device='cuda'):
    shape = {'device': device, **{k: v for k, v in {'causal': causal, 'distribution': distribution, 'head_dim': head_dim, 'heads': heads, 'kv_heads': kv_heads, 'kv_lengths': kv_lengths, 'page_size': page_size, 'precision': precision, 'query_lengths': query_lengths, 'return_lse': return_lse, 'sink': sink, 'window_left': window_left}.items() if v is not None}}
    p = make(**shape)
    return {'query':p['query'], 'kv_cache':p['kv_cache'], 'workspace_buffer':p['workspace_buffer'], 'block_tables':p['block_tables'], 'seq_lens':p['seq_lens'], 'max_q_len':p['max_q_len'], 'max_kv_len':p['max_kv_len'], 'bmm1_scale':p['bmm1_scale'], 'bmm2_scale':p['bmm2_scale'], 'batch_size':p['batch_size'], 'cum_seq_lens_q':p['cum_seq_lens_q'], 'cum_seq_lens_kv':p['cum_seq_lens_kv'], 'window_left':p.get('window_left', -1), 'out':p.get('out'), 'out_dtype':p.get('out_dtype'), 'o_sf_scale':p.get('o_sf_scale'), 'o_sf_vec_size':p.get('o_sf_vec_size'), 'kv_layout':p.get('kv_layout', 'HND'), 'enable_pdl':p.get('enable_pdl'), 'sinks':p.get('sinks'), 'kv_cache_sf':p.get('kv_cache_sf'), 'skip_softmax_threshold_scale_factor':p.get('skip_softmax_threshold_scale_factor'), 'uses_shared_paged_kv_idx':p.get('uses_shared_paged_kv_idx', True), 'causal':p.get('causal', True), 'lse':p.get('lse'), 'return_lse':p.get('return_lse', False), 'multi_ctas_kv_counter_buffer':p.get('multi_ctas_kv_counter_buffer'), 'use_fp16_softmax':p.get('use_fp16_softmax'), 'uses_spcompress':p.get('uses_spcompress'), 'backend':p.get('backend', 'trtllm-gen')}

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
