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
import torch
joint_qkv = _workload_profiles_qkv

def make_payload(lengths, heads, distribution='normal', host_lengths=True):
    t = sum(lengths)
    shape = (t, heads, 128)
    tensors = [x.to(torch.bfloat16) for x in joint_qkv(t, t, heads, heads, 128, 128, 'cuda', distribution)]
    for v in tensors:
        if distribution == 'tiny':
            v *= 0.01
        if distribution == 'peaked':
            v *= 3
        if distribution == 'zero':
            v.zero_()
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    n = t * heads * 128
    out = torch.full((n + 256,), -17, device='cuda', dtype=torch.bfloat16)[128:-128].view(shape)
    return dict(setup={}, reference_options={}, call=dict(query=tensors[0], key=tensors[1], value=tensors[2], cu_seqlens=torch.tensor(bounds, device='cuda', dtype=torch.int32), cu_seqlens_host=bounds if host_lengths else None, pv_mode='fp4', softmax_scale=128 ** (-0.5), out=out, backend='cake'))

def _make_inputs(heads, lengths, distribution='normal', host_lengths=True, device='cuda'):
    shape = {'device': device, 'distribution': distribution, 'host_lengths': host_lengths, **{k: v for k, v in {'heads': heads, 'lengths': lengths}.items() if v is not None}}
    if device != 'cuda':
        raise ValueError('This native generator requires CUDA')
    shape.pop('device')
    p = make_payload(**shape)['call']
    return {'query': p['query'], 'key': p['key'], 'value': p['value'], 'cu_seqlens': p['cu_seqlens'], 'pv_mode': p.get('pv_mode', 'fp8'), 'softmax_scale': p.get('softmax_scale', None), 'out': p.get('out', None), 'cu_seqlens_host': p.get('cu_seqlens_host', None), 'backend': p.get('backend', 'cake')}

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
