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

def quantize(value, block, global_scale=1.0, scale_rows=1):
    k = value.shape[-1]
    assert k % block == 0
    groups = value.float().reshape(*value.shape[:-1], k // block, block)
    peak = groups.abs().amax(-1).clamp_min(2.0 ** (-20))
    if scale_rows != 1:
        assert value.shape[-2] % scale_rows == 0
        peak = peak.reshape(*peak.shape[:-2], value.shape[-2] // scale_rows, scale_rows, peak.shape[-1]).amax(-2)
    bound = 448.0
    scale = peak / bound / global_scale
    stored_scale = scale
    expanded = scale.repeat_interleave(scale_rows, -2)
    normalized = (groups / (expanded.unsqueeze(-1) * global_scale)).reshape_as(value)
    payload = normalized.clamp(-448, 448).to(torch.float8_e4m3fn)
    return (payload, stored_scale)

def matrix(shape, block, device, std=1.0, zero=False, global_scale=1.0, scale_rows=1, profile=None, input_role='projection'):
    value = matrix_values(shape, device, std, role='weight' if std != 1.0 else input_role)
    if profile == 'channel_outliers':
        value[..., ::64] *= 8.0
    if zero:
        value.zero_()
    return quantize(value, block, global_scale, scale_rows)

def make(s, seed=0, device='cuda'):
    input_role = 'fc2' if (s['n'], s['k']) in ((4096, 12288), (7168, 2048), (4096, 1536)) else 'projection'
    torch.manual_seed(seed)
    m, n, k = (s['m'], s['n'], s['k'])
    a, sa = matrix((m, k), 128, device, zero=s.get('distribution') == 'zero', profile=s.get('distribution'), input_role=input_role)
    b, sb = matrix((n, k), 128, device, std=k ** (-0.5), scale_rows=128, input_role=input_role)
    sa = sa.contiguous()
    return dict(a=a, b=b, a_scale=sa, b_scale=sb, scale_major_mode='K', mma_sm=1, scale_granularity_mnk=(1, 128, 128), out=guarded((m, n), torch.bfloat16, device), out_dtype=torch.bfloat16, backend='cutlass')

def _make_inputs(m, n, k, distribution=None, logical_m=None, output_dtype=None, device='cuda'):
    """Explicit generator signature; original make and stored payload unchanged."""
    shape = {'m': m, 'n': n, 'k': k, **{k: v for k, v in {'distribution': distribution, 'logical_m': logical_m, 'output_dtype': output_dtype}.items() if v is not None}}
    p = make(shape, seed=torch.initial_seed(), device=device)
    return {'a': p['a'], 'b': p['b'], 'a_scale': p['a_scale'], 'b_scale': p['b_scale'], 'scale_major_mode': p['scale_major_mode'], 'mma_sm': p['mma_sm'], 'scale_granularity_mnk': p['scale_granularity_mnk'], 'out': p['out'], 'out_dtype': p['out_dtype'], 'backend': p['backend']}

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
