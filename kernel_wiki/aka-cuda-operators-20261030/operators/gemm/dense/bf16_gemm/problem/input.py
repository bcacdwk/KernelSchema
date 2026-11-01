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
    sample = x.reshape(-1)[:1 << 18].float()
    if sample.numel():
        mean = sample.mean()
        centered = sample - mean
        rms = sample.square().mean().sqrt()
        _workload_profiles_LAST_STATS[role] = dict(role=role, seed=torch.initial_seed(), sampled_elements=sample.numel(), mean=float(mean), rms=float(rms), absmax=float(sample.abs().max()), standardized_third_moment=float(centered.pow(3).mean() / centered.square().mean().clamp_min(1e-20).pow(1.5)))
        if role != 'weight':
            _workload_profiles_INPUT_RECORDS.append(dict(_workload_profiles_LAST_STATS[role]))
            del _workload_profiles_INPUT_RECORDS[:-2]
    return x
'Canonical stored-precision inputs; no dependency on a serving framework.'
import math
import torch
matrix_values = _workload_profiles_matrix_values

def guarded(shape, dtype, device):
    size = math.prod(shape)
    backing = torch.full((size + 128,), -17, dtype=dtype, device=device)
    return backing[64:-64].view(shape)

def quantize(value):
    return (value.to(torch.bfloat16), None)

def matrix(shape, device, std=1.0, zero=False, profile=None, input_role='projection'):
    value = matrix_values(shape, device, std, role='weight' if std != 1.0 else input_role)
    if profile == 'channel_outliers':
        value[..., ::64] *= 8.0
    if zero:
        value.zero_()
    return quantize(value)

def make(shape, seed=0, device='cuda'):
    input_role = 'fc2' if (shape['n'], shape['k']) in ((4096, 12288), (7168, 2048), (4096, 1536)) else 'projection'
    torch.manual_seed(seed)
    m, n, k = (shape['m'], shape['n'], shape['k'])
    aq, sa = matrix((m, k), device, zero=shape.get('distribution') == 'zero', profile=shape.get('distribution'), input_role=input_role)
    bq, sb = matrix((n, k), device, std=k ** (-0.5), input_role=input_role)
    dtype = getattr(torch, shape.get('output_dtype', 'bfloat16'))
    d = guarded((m, n), dtype, device)
    c = torch.randn((m, n), device=device).to(dtype) if shape.get('accumulate') else None
    if shape.get('c_alias_d'):
        d.copy_(torch.randn((m, n), device=device))
        c = d
    if shape.get('strided'):
        aa = torch.empty_strided((m, k), (1, (m + 7) // 8 * 8), dtype=aq.dtype, device=device)
        bb = torch.empty_strided((n, k), (1, (n + 7) // 8 * 8), dtype=bq.dtype, device=device)
        aa.copy_(aq)
        bb.copy_(bq)
        aq, bq = (aa, bb)
    return dict(a=aq, b=bq, d=d, c=c, compiled_dims='nk', alpha=shape.get('alpha'))

def _make_inputs(m, n, k, strided=False, accumulate=False, alpha=None, output_dtype='bfloat16', distribution='normal', c_alias_d=False, device='cuda'):
    """Construct the same stored operands with an explicit Collection signature."""
    p = make(dict(m=m, n=n, k=k, strided=strided, accumulate=accumulate, alpha=alpha, output_dtype=output_dtype, distribution=distribution, c_alias_d=c_alias_d), seed=torch.initial_seed(), device=device)
    return {'a': p['a'], 'b': p['b'], 'd': p['d'], 'c': p['c'], 'compiled_dims': p['compiled_dims'], 'alpha': p['alpha']}

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['d']}
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['d']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
