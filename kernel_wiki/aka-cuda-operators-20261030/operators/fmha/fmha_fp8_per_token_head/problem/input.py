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
import math
import torch
joint_qkv = _workload_profiles_qkv

def guarded(shape, dtype, device, value=0):
    backing = torch.full((math.prod(shape) + 256,), value, dtype=dtype, device=device)
    return backing[128:-128].view(shape)

def _make_paged_storage(query_lengths, kv_lengths, heads, kv_heads, head_dim, page_size, distribution='normal', physical_layout='HND', device='cuda'):
    assert len(query_lengths) == len(kv_lengths) and all((0 < q <= k for q, k in zip(query_lengths, kv_lengths))) and (heads % kv_heads == 0)
    assert physical_layout in ('HND', 'NHD') and page_size in (16, 32, 64, 128) and (head_dim in (64, 128, 256))
    batch = len(query_lengths)
    t = sum(query_lengths)
    sizes = [math.ceil(k / page_size) for k in kv_lengths]
    blocks = sum(sizes) + 3
    d = head_dim
    content = 2 * (d + 4)
    jq, jk, jv = joint_qkv(t, blocks * page_size, heads, kv_heads, d, d, device, distribution)
    q = guarded((t + 3, heads, d), torch.bfloat16, device, 0.375)
    q[:t].copy_(jq)
    shape = (blocks, kv_heads, page_size, content) if physical_layout == 'HND' else (blocks, page_size, kv_heads, content)
    cache = guarded(shape, torch.uint8, device, 91)
    if physical_layout == 'NHD':
        cache = cache.transpose(1, 2)
    base = torch.empty(0, dtype=torch.float32, device=device).set_(cache.untyped_storage())
    stride = (cache.stride(0) // 4, cache.stride(2) // 4, cache.stride(1) // 4)
    offset = cache.storage_offset() // 4
    ks = torch.as_strided(base, (blocks, page_size, kv_heads), stride, offset + d // 4)
    vs = torch.as_strided(base, (blocks, page_size, kv_heads), stride, offset + (2 * d + 4) // 4)
    if distribution == 'peaked':
        q[:t].mul_(4)
    if distribution == 'balanced':
        q[:t].zero_()
    assert distribution in ('normal', 'tiny', 'peaked', 'zero', 'balanced')
    for first in range(0, blocks, 32):
        n = min(32, blocks - first)
        slots = torch.arange(first * page_size, (first + n) * page_size, device=device).reshape(n, 1, page_size, 1)
        hs = torch.arange(kv_heads, device=device).reshape(1, kv_heads, 1, 1)
        k = jk[first * page_size:(first + n) * page_size].reshape(n, page_size, kv_heads, d).transpose(1, 2) * torch.pow(2.0, ((slots + hs * 3) % 7 - 3).float())
        v = jv[first * page_size:(first + n) * page_size].reshape(n, page_size, kv_heads, d).transpose(1, 2) * torch.pow(2.0, ((slots * 3 + hs) % 7 - 3).float())
        if distribution == 'peaked':
            k.mul_(4)
        if distribution == 'tiny':
            v.mul_(2 ** (-20))
        if distribution == 'zero':
            v.zero_()
        if distribution == 'balanced':
            v.copy_((slots % 2 * 2 - 1).expand_as(v))
        for values, scales, start in ((k, ks, 0), (v, vs, d + 4)):
            stored_bf16 = values.bfloat16().float()
            scale = (stored_bf16.abs().amax(-1) / 448).clamp_min(1e-06)
            scales[first:first + n].copy_(scale.transpose(1, 2))
            quant = (stored_bf16 / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
            cache[first:first + n, :, :, start:start + d].copy_(quant)
    table = guarded((batch, max(sizes)), torch.int32, device, -1)
    perm = torch.randperm(blocks - 1, device=device) + 1
    offset = 0
    for i, n in enumerate(sizes):
        table[i, :n] = perm[offset:offset + n].int()
        offset += n
    if distribution == 'balanced':
        for i, length in enumerate(kv_lengths):
            p = int(table[i, (length - 1) // page_size])
            slot = (length - 1) % page_size
            vs[p, slot].fill_(1.125 / 448)
            cache[p, :, slot, d + 4:2 * d + 4].fill_(int(torch.tensor(448.0, dtype=torch.float8_e4m3fn).view(torch.uint8)))
    cu = guarded((batch + 1,), torch.int32, device, -3)
    cu[0] = 0
    cu[1:] = torch.tensor(query_lengths, device=device).cumsum(0).int()
    lens = guarded((batch,), torch.int32, device, -5)
    lens.copy_(torch.tensor(kv_lengths, device=device))
    threshold = 128 // kv_heads
    segments = 16
    return dict(q=q, kv_cache=cache, out=guarded((t + 5, heads, d), torch.bfloat16, device, -17), cu_seqlens_q=cu, seqused_k=lens, block_table=table, k_scale_cache=ks, v_scale_cache=vs, max_seqlen_q=max(query_lengths), max_seqlen_k=max(kv_lengths), softmax_scale=d ** (-0.5), seq_threshold_3D=threshold, num_par_softmax_segments=segments, softmax_segm_output=guarded((threshold, heads, segments, d), torch.float32, device, -37), softmax_segm_max=guarded((threshold, heads, segments), torch.float32, device, -37), softmax_segm_expsum=guarded((threshold, heads, segments), torch.float32, device, -37))

def make(**kw):
    x = _make_paged_storage(**kw)
    d = x['q'].shape[-1]
    t = int(x['cu_seqlens_q'][-1])
    cache = x.pop('kv_cache').transpose(1, 2)
    x['q'] = x['q'][:t]
    x['out'] = x['out'][:t]
    # Public integer ABI: 3 denotes FP8 per-token/head KV quantization.
    x.update(k=cache[..., :d].view(torch.float8_e4m3fn), v=cache[..., d + 4:2 * d + 4].view(torch.float8_e4m3fn), causal=True, window_size=(-1, -1), softcap=0.0, q_descale=None, k_descale=None, v_descale=None, kv_quant_mode=3)
    return x

def _make_inputs(head_dim, heads, kv_heads, kv_lengths, page_size, query_lengths, distribution=None, physical_layout=None, device='cuda'):
    shape = {'device': device, **{k: v for k, v in {'distribution': distribution, 'head_dim': head_dim, 'heads': heads, 'kv_heads': kv_heads, 'kv_lengths': kv_lengths, 'page_size': page_size, 'physical_layout': physical_layout, 'query_lengths': query_lengths}.items() if v is not None}}
    p = make(**shape)
    return {'q': p['q'], 'k': p['k'], 'v': p['v'], 'out': p['out'], 'cu_seqlens_q': p['cu_seqlens_q'], 'max_seqlen_q': p['max_seqlen_q'], 'seqused_k': p['seqused_k'], 'max_seqlen_k': p['max_seqlen_k'], 'softmax_scale': p['softmax_scale'], 'causal': p['causal'], 'window_size': p['window_size'], 'block_table': p['block_table'], 'softcap': p['softcap'], 'q_descale': p['q_descale'], 'k_descale': p['k_descale'], 'v_descale': p['v_descale'], 'seq_threshold_3D': p.get('seq_threshold_3D', None), 'num_par_softmax_segments': p.get('num_par_softmax_segments', None), 'softmax_segm_output': p.get('softmax_segm_output', None), 'softmax_segm_max': p.get('softmax_segm_max', None), 'softmax_segm_expsum': p.get('softmax_segm_expsum', None), 'alibi_slopes': p.get('alibi_slopes', None), 'output_scale': p.get('output_scale', None), 'qq_bias': p.get('qq_bias', None), 'sinks': p.get('sinks', None), 'mm_prefix_range': p.get('mm_prefix_range', None), 'rswa_prefix_lens': p.get('rswa_prefix_lens', None), 'rswa_window': p.get('rswa_window', None), 'use_alibi_sqrt': p.get('use_alibi_sqrt', False), 'kv_quant_mode': p.get('kv_quant_mode', 0), 'k_scale_cache': p.get('k_scale_cache', None), 'v_scale_cache': p.get('v_scale_cache', None), 'chunk_lookback': p.get('chunk_lookback', -1), 'use_td': p.get('use_td', False), 'mm_prefix_clamp_sliding_window': p.get('mm_prefix_clamp_sliding_window', False)}

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['out', 'softmax_segm_expsum', 'softmax_segm_max', 'softmax_segm_output']}
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['out']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
