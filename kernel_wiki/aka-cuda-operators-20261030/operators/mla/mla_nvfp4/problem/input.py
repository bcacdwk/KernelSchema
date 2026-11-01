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

def guarded(shape, dtype, device):
    n = 1
    for d in shape:
        n *= d
    raw = torch.full((n + 256,), -17, dtype=dtype, device=device)
    return raw[128:-128].view(shape)

def swizzle(scales):
    b, h, s, g = scales.shape
    sp, gp = (math.ceil(s / 128) * 128, math.ceil(g / 4) * 4)
    padded = torch.zeros((b, h, sp, gp), dtype=scales.dtype, device=scales.device)
    padded[:, :, :s, :g].copy_(scales)
    return padded.reshape(b * h, sp // 128, 4, 32, gp // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().flatten()

def quantized(x, mode, swizzled=True):
    group = 16
    grouped = x.float().reshape(*x.shape[:-1], -1, group)
    sf = (grouped.abs().amax(-1) / 6).clamp_min(2 ** (-9)).to(torch.float8_e4m3fn).float()
    values = grouped / sf[..., None]
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    codes = (values.abs()[..., None] - lut).abs().argmin(-1).to(torch.uint8)
    codes |= (values < 0).to(torch.uint8) * 8
    codes = codes.flatten(-2)
    payload = codes[..., ::2] | codes[..., 1::2] << 4
    storage = sf.to(torch.float8_e4m3fn).view(torch.uint8)
    if swizzled:
        storage = swizzle(storage.transpose(1, 2))
        storage = storage.view(torch.float8_e4m3fn)
    return (payload, storage)

def make(**kw):
    lens = kw['kv_lengths']
    b = len(lens)
    sq = kw['query_len']
    h = kw['heads']
    device = kw.get('device', 'cuda')
    counts = [math.ceil(n / 64) for n in lens]
    p = sum(counts) + 1
    query, kv, _ = joint_qkv(b * sq, p * 64, h, 1, 512, 512, device, kw.get('distribution', 'normal'))
    kv = kv.reshape(p, 64, 512)
    if kw.get('distribution') == 'tiny':
        query *= 0.03125
    if kw.get('distribution') == 'peaked':
        query *= 3
    if kw.get('distribution') == 'zero':
        kv.zero_()
    query, qs = quantized(query, 'nvfp4', False)
    kv, ks = quantized(kv, 'nvfp4', False)
    table = torch.zeros(b, max(counts), dtype=torch.int32, device=device)
    order = torch.randperm(p - 1, device=device) + 1
    cursor = 0
    for i, n in enumerate(counts):
        table[i, :n] = order[cursor:cursor + n]
        cursor += n
    return dict(query=query, query_scale=qs, kv_cache=kv, kv_scale=ks, block_tables=table, seq_lens=torch.tensor(lens, dtype=torch.int32, device=device), workspace_buffer=torch.zeros(128 * 1024 ** 2, dtype=torch.uint8, device=device), sm_scale=512 ** (-0.5), sinks=torch.linspace(-1, 3, h, device=device) if kw.get('sink') else None, out=guarded((b * sq, h, 512), torch.bfloat16, device), lse=guarded((b * sq, h), torch.float32, device) if kw.get('return_lse', True) else None, return_lse=kw.get('return_lse', True), seq_lens_cpu=torch.tensor(lens, dtype=torch.int32), backend='cake')

def _make_inputs(heads, kv_lengths, query_len, distribution=None, return_lse=None, sink=None, device='cuda'):
    shape = {'device': device, **{k: v for k, v in {'distribution': distribution, 'heads': heads, 'kv_lengths': kv_lengths, 'query_len': query_len, 'return_lse': return_lse, 'sink': sink}.items() if v is not None}}
    p = make(**shape)
    return {'query': p['query'], 'query_scale': p['query_scale'], 'kv_cache': p['kv_cache'], 'kv_scale': p['kv_scale'], 'block_tables': p['block_tables'], 'seq_lens': p['seq_lens'], 'workspace_buffer': p['workspace_buffer'], 'sm_scale': p['sm_scale'], 'sinks': p.get('sinks', None), 'out': p.get('out', None), 'lse': p.get('lse', None), 'return_lse': p.get('return_lse', False), 'seq_lens_cpu': p.get('seq_lens_cpu', None), 'backend': p.get('backend', 'cake')}

def prepare_model(model_class, inputs, init_kwargs, input_kwargs):
    """Bind native tensor-dependent preparation before forward timing."""
    return model_class(**inputs), {}

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
