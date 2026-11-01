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
FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

def guarded(shape, dtype, device):
    size = math.prod(shape)
    backing = torch.full((size + 128,), -17, dtype=dtype, device=device)
    return backing[64:-64].view(shape)

def quantize(value, block, global_scale=1.0, scale_rows=1):
    k = value.shape[-1]
    assert k % block == 0
    groups = value.float().reshape(*value.shape[:-1], k // block, block)
    peak = groups.abs().amax(-1).clamp_min(2.0 ** (-20))
    if scale_rows != 1:
        assert value.shape[-2] % scale_rows == 0
        peak = peak.reshape(*peak.shape[:-2], value.shape[-2] // scale_rows, scale_rows, peak.shape[-1]).amax(-2)
    bound = 6.0
    scale = peak / bound / global_scale
    exponents = torch.ceil(torch.log2(scale)).clamp(-126, 127)
    stored_scale = (exponents + 127).to(torch.uint8)
    scale = torch.exp2(exponents)
    expanded = scale.repeat_interleave(scale_rows, -2)
    normalized = (groups / (expanded.unsqueeze(-1) * global_scale)).reshape_as(value)
    magnitude = normalized.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for j in range(7):
        midpoint = (FP4[j] + FP4[j + 1]) / 2
        crosses = magnitude > midpoint if j % 2 == 0 else magnitude >= midpoint
        code += crosses.to(torch.uint8)
    code |= torch.signbit(normalized).to(torch.uint8) << 3
    payload = code[..., 0::2] | code[..., 1::2] << 4
    return (payload, stored_scale)

def matrix(shape, block, device, std=1.0, zero=False, global_scale=1.0, scale_rows=1, profile=None, input_role='projection'):
    value = matrix_values(shape, device, std, role='weight' if std != 1.0 else input_role)
    if profile == 'channel_outliers':
        value[..., ::64] *= 8.0
    if zero:
        value.zero_()
    return quantize(value, block, global_scale, scale_rows)

def make(shape, seed=0, device='cuda'):
    input_role = 'fc2' if (shape['n'], shape['k']) in ((4096, 12288), (7168, 2048), (4096, 1536)) else 'projection'
    torch.manual_seed(seed)
    m, n, k = (shape['m'], shape['n'], shape['k'])
    aq, sa = matrix((m, k), 32, device, zero=shape.get('distribution') == 'zero', profile=shape.get('distribution'), input_role=input_role)
    bq, sb = matrix((n, k), 32, device, std=k ** (-0.5), input_role=input_role)
    dtype = getattr(torch, shape.get('output_dtype', 'bfloat16'))
    d = guarded((m, n), dtype, device)
    c = torch.randn((m, n), device=device).to(dtype) if shape.get('accumulate') else None
    if shape.get('c_alias_d'):
        d.copy_(torch.randn((m, n), device=device))
        c = d
    sa = torch.exp2(sa.float() - 127)
    sb = torch.exp2(sb.float() - 127)
    aq = aq.view(torch.int8)
    bq = bq.view(torch.int8)
    return dict(a=(aq, sa), b=(bq, sb), d=d, c=c, recipe=(1, 1, 32), recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, alpha=shape.get('alpha'))

def _make_inputs(m, n, k, accumulate=None, alpha=None, c_alias_d=None, distribution=None, device='cuda'):
    """Explicit generator signature; original make and stored payload unchanged."""
    shape = {'m': m, 'n': n, 'k': k, **{k: v for k, v in {'accumulate': accumulate, 'alpha': alpha, 'c_alias_d': c_alias_d, 'distribution': distribution}.items() if v is not None}}
    p = make(shape, seed=torch.initial_seed(), device=device)
    return {'a': p['a'], 'b': p['b'], 'd': p['d'], 'c': p['c'], 'recipe': p['recipe'], 'recipe_a': p['recipe_a'], 'recipe_b': p['recipe_b'], 'compiled_dims': p['compiled_dims'], 'disable_ue8m0_cast': p['disable_ue8m0_cast'], 'alpha': p['alpha']}

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
