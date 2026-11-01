import math
import torch
import torch.nn.functional as F
_workload_profiles_INPUT_RECORDS = []

def _workload_profiles_correlated_gates(raw, beta):
    channel = torch.linspace(-0.4, 0.4, raw.shape[-1], device=raw.device)
    shared = torch.randn((*raw.shape[:-1], 1), device=raw.device)
    raw.copy_(0.65 * shared + math.sqrt(1 - 0.65 ** 2) * torch.randn_like(raw.float()) + channel)
    beta.copy_(0.45 * shared.squeeze(-1) + math.sqrt(1 - 0.45 ** 2) * torch.randn_like(beta.float()))

def _workload_profiles_prepare(p, shape, kind):
    """Populate normal carried states from 32 previous valid tokens.

    Checkpoint j is the result after 32+j tokens. Conv slot zero of each
    request holds the matching rolling raw history; accepted index chooses
    the same history endpoint for the matrix and convolution states.
    """
    profile = shape.get('domain', shape.get('distribution', 'normal'))
    if profile not in ('normal', 'core_buffer', 'output_only'):
        return p
    kda = kind.startswith('kda')
    prefill = 'prefill' in kind
    state = p.get('state', p.get('initial_state'))
    device = state.device
    hv = state.shape[1]
    d = state.shape[-1]
    conv = p.get('conv_state')
    isconv = conv is not None
    if 'metadata' in p:
        n = shape['batch']
        bounds = list(range(n + 1))
        table = p['metadata']['non_spec_state_indices_tensor'].long()[:, None]
        accepts = [1] * n
        raw_current = p['qkvz']
        h = (raw_current.shape[-1] - 2 * hv * d) // (2 * d)
    elif prefill:
        lengths = shape['lengths']
        bounds = [0]
        for size in lengths:
            bounds.append(bounds[-1] + size)
        n = len(lengths)
        table = torch.arange(n, device=device)[:, None]
        accepts = [1] * n
        h = p['q'].shape[-2]
    else:
        bounds = p.get('cu_seqlens', p.get('query_start_loc')).tolist()
        n = len(bounds) - 1
        table = p.get('state_indices', p.get('ssm_state_indices')).long()
        if table.ndim == 1:
            table = table[:, None]
        accepted = p.get('num_accepted_tokens')
        accepts = accepted.tolist() if accepted is not None else [1] * n
        h = hv if kda else (p['mixed_qkv'].shape[1] - hv * d) // (2 * d)
        raw_current = p.get('x', p.get('mixed_qkv'))
    width = table.shape[1]
    channels = (2 * h + hv) * d
    history_tokens = 32
    if kda:
        alog = p['A_log'].float().reshape(hv)
        bias = p.get('dt_bias', p.get('g_bias')).float().reshape(hv, d)
        taps = p['weight'].float().permute(0, 2, 1).reshape(channels, 4) if isconv else None
        raw_gate = p.get('raw_gate', p.get('raw_g', p.get('g')))
        raw_beta = p.get('raw_beta', p.get('beta'))
        if raw_gate is not None and raw_beta is not None and shape.get('beta_is_logit', True):
            if 'g_bias' not in p or 'raw_beta' in p:
                _workload_profiles_correlated_gates(raw_gate, raw_beta)
    else:
        alog = p['A_log'].float() if 'A_log' in p else torch.linspace(-3, 1, hv, device=device)
        bias = p['dt_bias'].float() if 'dt_bias' in p else torch.linspace(-3, 3, hv, device=device).to(torch.bfloat16).float()
        taps = p['conv_weight'].float() if isconv else None
    gain = torch.cat([torch.linspace(0.6, 1.4, hh, device=device)[:, None] * torch.linspace(0.7, 1.3, d, device=device)[None, :] for hh in (h, h, hv)]).flatten()
    for first in range(0, n, 8):
        owners = list(range(first, min(first + 8, n)))
        count = len(owners)
        current_len = max((bounds[r + 1] - bounds[r] for r in owners))
        future = min(current_len, 8) if prefill else current_len
        steps = history_tokens + width - 1
        raw = torch.randn((count, steps + future + 3, channels), device=device) * 0.4 * gain
        for t in range(1, raw.shape[1]):
            raw[:, t] = 0.5 * raw[:, t - 1] + math.sqrt(0.75) * raw[:, t]
        raw = raw.to(torch.bfloat16).float()
        if isconv:
            post = F.silu((raw.unfold(1, 4, 1) * taps[None, None, :, :]).sum(-1))
        else:
            post = F.silu(raw[:, 3:])
        carried = torch.zeros((count, hv, d, d), device=device)
        for t in range(steps):
            kv = post[:, t, h * d:2 * h * d].reshape(count, h, d)
            kv = kv * torch.rsqrt(kv.square().sum(-1, keepdim=True) + 1e-06)
            kv = kv.repeat_interleave(hv // h, 1)
            value = post[:, t, 2 * h * d:].reshape(count, hv, d)
            raw_beta = torch.randn((count, hv), device=device).to(torch.bfloat16).float()
            if kda:
                gate = (0.45 * raw_beta[:, :, None] + 0.8 * torch.randn((count, hv, d), device=device)).to(torch.bfloat16).float()
                z = alog.exp()[None, :, None] * (gate + bias)
                log_decay = -alog.exp()[None, :, None] * F.softplus(gate + bias) if p.get('lower_bound', -5.0) is None else p.get('lower_bound', -5.0) * z.sigmoid()
                carried *= log_decay.exp()[:, :, None, :]
            else:
                gate = (0.45 * raw_beta + 0.8 * torch.randn_like(raw_beta)).to(torch.bfloat16).float()
                carried *= torch.exp(-alog.exp() * F.softplus(gate + bias))[:, :, None, None]
            delta = (value - (carried * kv[:, :, None, :]).sum(-1)) * raw_beta.sigmoid()[:, :, None]
            carried += delta[:, :, :, None] * kv[:, :, None, :]
            if t >= history_tokens - 1:
                j = t - history_tokens + 1
                for local, r in enumerate(owners):
                    dst = int(table[r, j])
                    if dst > 0 or prefill:
                        state[dst].copy_(carried[local].to(state.dtype))
        if isconv:
            for local, r in enumerate(owners):
                slot = int(table[r, 0])
                start = history_tokens + accepts[r] - 1
                if slot <= 0:
                    continue
                conv[slot, :, :width + 2].copy_(raw[local, history_tokens:history_tokens + width + 2].T.to(conv.dtype))
                begin, end = bounds[r:r + 2]
                raw_current[begin:end, :channels].copy_(raw[local, start + 3:start + 3 + end - begin].to(raw_current.dtype))
    mode = shape.get('state_mode', shape.get('state_regime', 'continuation'))
    if mode == 'zero':
        state.zero_()
    elif mode == 'mixed':
        for r in range(0, n, 4):
            state[table[r]] = 0
            if isconv and int(table[r, 0]) > 0:
                conv[int(table[r, 0])].zero_()
    _workload_profiles_INPUT_RECORDS.append(dict(seed=torch.initial_seed(), family=kind, profile=profile, history_tokens=history_tokens, state_shape=list(state.shape), state_stride=list(state.stride()), state_storage_bytes=state.untyped_storage().nbytes(), reserved_slots=int(table.numel()), active_requests=sum((bounds[r + 1] > bounds[r] for r in range(n))), independent_past_recurrence=True, continuous_conv_history=isconv))
    del _workload_profiles_INPUT_RECORDS[:-2]
    return p
import math
import torch
from torch.nn import functional as F

def _gdn_workload_gains(heads, device):
    head = torch.linspace(0.6, 1.4, heads, device=device)
    channel = torch.linspace(0.7, 1.3, 128, device=device)
    return head[:, None] * channel[None, :]

def _gdn_workload_gate_logits(a, b):
    """Per-head Gaussian-copula logits; production computes decay and sigmoid."""
    rho = torch.linspace(-0.65, 0.8, a.shape[-1], device=a.device)
    z = torch.randn(a.shape, device=a.device)
    b.copy_(rho * z + (1 - rho.square()).sqrt() * torch.randn_like(z))
    a.copy_(z)

def _gdn_workload_structured_state(state, regime='continuation'):
    """Decaying rank-16 modes plus full-rank residual; slot/head diversity.

    Independent random left/right factors produce shared row/column structure.
    This is a synthetic state family, not an exact replay of a past recurrence.
    """
    if regime not in ('continuation', 'mixed', 'zero'):
        raise ValueError('unknown state regime')
    if regime == 'zero':
        state.zero_()
        return
    hv = state.shape[1]
    spectrum = torch.exp(-torch.arange(16, device=state.device).float() / 4)
    spectrum /= spectrum.square().sum().sqrt()
    amplitude = torch.linspace(0.1, 0.3, hv, device=state.device)[None, :, None, None]
    coordinate = torch.linspace(0.7, 1.3, 128, device=state.device)
    for begin in range(0, state.shape[0], 8):
        block = state[begin:begin + 8]
        left = torch.randn((*block.shape[:2], 128, 16), device=state.device)
        right = torch.randn_like(left)
        value = left * spectrum @ right.transpose(-1, -2)
        value *= coordinate[None, None, :, None] * coordinate[None, None, None, :]
        value *= amplitude
        value += torch.randn_like(value) * 0.02
        block.copy_(value)
    if regime == 'mixed':
        state[::4].zero_()
'Real Qwen post-preparation operands: normalized BF16 Q/K, FP32 log gates.'
import math
import torch
prepare = _workload_profiles_prepare
gains = _gdn_workload_gains
gate_logits = _gdn_workload_gate_logits
structured_state = _gdn_workload_structured_state
from torch.nn import functional as F

def guarded(shape, dtype, device, value=0):
    n = math.prod(shape)
    base = torch.full((n + 32,), value, dtype=dtype, device=device)
    return base[16:16 + n].view(shape)

def _make_inputs(tp, lengths, state_mode='continuation', domain='normal', state_dtype='fp32', device='cuda', head_ratio=3):
    h, hv = (16 // tp, 16 * head_ratio // tp)
    t = sum(lengths)
    n = len(lengths)
    q = guarded((1, t, h, 128), torch.bfloat16, device, 7)
    k = guarded(q.shape, q.dtype, device, 8)
    for x in (q, k):
        raw = F.silu(torch.randn(x.shape, device=device) * 0.4 * gains(h, device))
        if domain == 'nearzero_qk':
            raw.mul_(1e-05)
        x.copy_(raw / (raw.square().sum(-1, keepdim=True) + 1e-06).sqrt())
    v = guarded((1, t, hv, 128), torch.bfloat16, device, 9)
    v.normal_(0, 0.4)
    v.mul_(gains(hv, device))
    v.copy_(F.silu(v.float()))
    g = guarded((1, t, hv), torch.float32, device, 11)
    beta = guarded(g.shape, g.dtype, device, 12)
    a = torch.randn((1, t, hv), device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    gate_logits(a, b)
    dt = torch.linspace(-3, 3, hv, device=device, dtype=torch.bfloat16)
    alog = torch.linspace(-3, 1, hv, device=device)
    g.copy_(-alog.exp() * F.softplus(a.float() + dt.float()))
    beta.copy_(torch.sigmoid(b.float()))
    state = guarded((n, hv, 128, 128), {'fp32': torch.float32, 'bf16': torch.bfloat16}[state_dtype], device, 13)
    structured_state(state, 'continuation')
    if state_mode == 'zero':
        state.zero_()
    elif state_mode == 'mixed':
        state[::2].zero_()
    elif state_mode != 'continuation':
        raise ValueError(state_mode)
    if domain == 'tiny':
        v.mul_(1e-07)
        state.mul_(1e-07)
    elif domain == 'zero':
        v.zero_()
        state.zero_()
    elif domain == 'slow_decay':
        g.copy_(-alog.exp() * F.softplus(torch.full_like(a, -20).float() + dt.float()))
    elif domain == 'cancellation':
        v[..., ::2] *= -8
        state[..., ::2] *= -8
    elif domain == 'beta_zero':
        beta.zero_()
    cu = guarded((n + 1,), torch.int32, device, 15)
    cu[0] = 0
    cu[1:] = torch.tensor(lengths, device=device).cumsum(0)
    counts = [(s + 63) // 64 for s in lengths]
    pairs = [(i, j) for i, c in enumerate(counts) for j in range(c)]
    ci = guarded((len(pairs), 2), torch.int32, device, 17)
    ci.copy_(torch.tensor(pairs, device=device))
    co = guarded((n + 1,), torch.int32, device, 19)
    co[0] = 0
    co[1:] = torch.tensor(counts, device=device).cumsum(0)
    core = guarded((t + 3, hv, 128), torch.bfloat16, device, 21) if domain == 'core_buffer' else None
    p = dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=state, output_final_state=domain != 'output_only', cu_seqlens=cu, chunk_indices=ci, chunk_offsets=co, use_qk_l2norm_in_kernel=False, core_attn_out=core)
    shape = {'tp': tp, 'lengths': lengths, 'state_mode': state_mode, 'domain': domain, 'state_dtype': state_dtype, 'device': device, 'head_ratio': head_ratio}
    prepare(p, shape, 'gdn_prefill')
    return {'q': p['q'], 'k': p['k'], 'v': p['v'], 'g': p['g'], 'beta': p['beta'], 'initial_state': p['initial_state'], 'output_final_state': p['output_final_state'], 'cu_seqlens': p.get('cu_seqlens', None), 'chunk_indices': p.get('chunk_indices', None), 'chunk_offsets': p.get('chunk_offsets', None), 'use_qk_l2norm_in_kernel': p.get('use_qk_l2norm_in_kernel', True), 'core_attn_out': p.get('core_attn_out', None)}

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['core_attn_out']}
    if x.get('core_attn_out') is not None:
        views['core_attn_out'] = x['core_attn_out'].view(-1)[:x['v'].numel()]
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['core_attn_out']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
