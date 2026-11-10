import math
import torch

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
    return x

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

def matrix(shape, block, device, std=1.0, zero=False, global_scale=1.0, scale_rows=1, input_role='projection'):
    if len(shape) == 3:
        chunks = [matrix(shape[1:], block, device, std, zero, global_scale, scale_rows, input_role=input_role) for _ in range(shape[0])]
        payload = torch.stack([x[0] for x in chunks])
        scales = None if chunks[0][1] is None else torch.stack([x[1] for x in chunks])
        return (payload, scales)
    value = matrix_values(shape, device, std, role='weight' if std != 1.0 else input_role)
    if zero:
        value.zero_()
    return quantize(value, block, global_scale, scale_rows)

def make(s, seed=0, device='cuda'):
    input_role = 'fc2' if (s['n'], s['k']) in ((4096, 12288), (7168, 2048), (4096, 1536)) else 'projection'
    torch.manual_seed(seed)
    counts = s['group_sizes']
    m = sum(counts)
    e = len(counts)
    n, k = (s['n'], s['k'])
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    indptr = torch.tensor(offsets, dtype=torch.int32, device=device)
    block = 128
    a, sa = matrix((m, k), block, device, zero=s.get('distribution') == 'zero', input_role=input_role)
    b, sb = matrix((e, n, k), block, device, std=k ** (-0.5), scale_rows=128, input_role=input_role)
    out = guarded((m, n), torch.bfloat16, device)
    return dict(a=a, b=b, a_scale=sa.T.contiguous(), b_scale=sb.transpose(-1, -2).contiguous(), m_indptr=indptr, scale_granularity_mnk=(1, 128, 128), scale_major_mode='MN', mma_sm=1, out=out, out_dtype=torch.bfloat16, backend='trtllm', segment_alignment=1)

def _make_inputs(m, n, k, groups, b_scale_rows=None, group_sizes=None, logical_group_sizes=None, mode=None, device='cuda'):
    """Explicit generator signature; original make and stored payload unchanged."""
    shape = {'m': m, 'n': n, 'k': k, **{k: v for k, v in {'b_scale_rows': b_scale_rows, 'group_sizes': group_sizes, 'groups': groups, 'logical_group_sizes': logical_group_sizes, 'mode': mode}.items() if v is not None}}
    p = make(shape, seed=torch.initial_seed(), device=device)
    return {'a':p['a'], 'b':p['b'], 'a_scale':p['a_scale'], 'b_scale':p['b_scale'], 'm_indptr':p['m_indptr'], 'scale_granularity_mnk':p['scale_granularity_mnk'], 'scale_major_mode':p['scale_major_mode'], 'mma_sm':p['mma_sm'], 'out':p['out'], 'out_dtype':p['out_dtype'], 'backend':p['backend'], 'segment_alignment':p['segment_alignment']}

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
