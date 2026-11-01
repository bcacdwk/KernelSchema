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

def _gdn_workload_projected(x, h, hv, postconv=False):
    """Q/K/V and optional output-gate views keep their original backing."""
    start = 0
    for heads in (h, h, hv, hv):
        end = start + heads * 128
        if end > x.shape[-1]:
            break
        view = x[..., start:end].unflatten(-1, (heads, 128))
        view.mul_(_gdn_workload_gains(heads, x.device))
        if postconv and start < (2 * h + hv) * 128:
            view.copy_(F.silu(view.float()))
        start = end

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

def _gdn_workload_convolution_history(conv, h, hv):
    """Stationary AR(1), rho=.5, with the same channel scales as projection."""
    scale = torch.cat([_gdn_workload_gains(n, conv.device).flatten() for n in (h, h, hv)])
    conv.mul_(scale[None, :, None])
    for t in range(1, conv.shape[-1]):
        conv[:, :, t].copy_(0.5 * conv[:, :, t - 1].float() + math.sqrt(0.75) * conv[:, :, t].float())

def _gdn_workload_slots(count, pool_size, pattern, device):
    if pool_size <= count:
        raise ValueError('pool must include distinct requested slots plus null slot 0')
    if pattern == 'scattered':
        return torch.randperm(pool_size - 1, device=device)[:count] + 1
    if pattern == 'high':
        return torch.arange(pool_size - count, pool_size, device=device)
    if pattern == 'contiguous':
        return torch.arange(1, count + 1, device=device)
    raise ValueError('unknown slot pattern')
import math
import torch
_postconv_input_gains = _gdn_workload_gains
_postconv_input_projected = _gdn_workload_projected
_postconv_input_gate_logits = _gdn_workload_gate_logits
_postconv_input_structured_state = _gdn_workload_structured_state
_postconv_input_convolution_history = _gdn_workload_convolution_history
_postconv_input_slots = _gdn_workload_slots

def _postconv_input_guarded(shape, dtype, device, value=0):
    n = math.prod(shape)
    base = torch.full((n + 32,), value, dtype=dtype, device=device)
    return base[16:16 + n].view(shape)

def _postconv_input_make_inputs(tp=1, width=2, lengths=(2,), accepted_mode='cycle', domain='normal', state_dtype='bf16', device='cuda', head_ratio=3, state_regime='continuation', pool_size=2048, state_slot_padding=32, slot_pattern='scattered', preconv=False):
    h, hv = (16 // tp, 16 * head_ratio // tp)
    n = len(lengths)
    t = sum(lengths)
    qkv = (2 * h + hv) * 128
    packed = _postconv_input_guarded((t, qkv + hv * 128), torch.bfloat16, device, 7)
    packed.normal_(0, 0.4)
    mixed = packed[:, :qkv]
    gate = packed[:, qkv:].view(t, hv, 128)
    _postconv_input_projected(packed, h, hv, postconv=not preconv)
    ba = _postconv_input_guarded((t, 2 * hv), torch.bfloat16, device, 9)
    ba.normal_(0, 1)
    b, a = (ba[:, :hv], ba[:, hv:])
    _postconv_input_gate_logits(a, b)
    alog = _postconv_input_guarded((hv,), torch.float32, device, 3)
    alog.copy_(torch.linspace(-3, 1, hv, device=device))
    dt = _postconv_input_guarded((hv,), torch.bfloat16, device, 4)
    dt.copy_(torch.linspace(-3, 3, hv, device=device))
    weight = _postconv_input_guarded((128,), torch.bfloat16, device, 5)
    weight.uniform_(0.7, 1.3)
    dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[state_dtype]
    pool = pool_size
    slot_size = hv * 128 * 128
    stride = slot_size + state_slot_padding
    backing = _postconv_input_guarded((pool * stride,), dtype, device, 11)
    state = backing.as_strided((pool, hv, 128, 128), (stride, 128 * 128, 128, 1))
    _postconv_input_structured_state(state, state_regime)
    indices = _postconv_input_guarded((n, width), torch.int32, device, 17)
    indices.copy_(_postconv_input_slots(n * width, pool, slot_pattern, device).view(n, width))
    accepted = _postconv_input_guarded((n,), torch.int32, device, 19)
    accepted.copy_(torch.arange(n, device=device) % width + 1 if accepted_mode == 'cycle' else torch.full((n,), width, device=device))
    cu = _postconv_input_guarded((n + 1,), torch.int32, device, 23)
    cu[0] = 0
    cu[1:] = torch.tensor(lengths, device=device).cumsum(0)
    out = _postconv_input_guarded((t, hv, 128), torch.bfloat16, device, 13)
    out.zero_()
    if domain == 'tiny':
        mixed[:, 2 * h * 128:] *= 1e-07
        state.mul_(1e-07)
    elif domain == 'zero':
        mixed.zero_()
        state.zero_()
    elif domain == 'qk_nearzero':
        mixed[:, :2 * h * 128] *= 1e-05
    elif domain == 'weak_decay':
        a.fill_(-20)
        b.fill_(-5)
    elif domain == 'strong_decay':
        a.fill_(20)
        b.fill_(8)
    elif domain == 'cancellation':
        mixed[:, :2 * h * 128:2] *= -8
        state[..., ::2] *= -8
    elif domain == 'gate_negative':
        gate.fill_(-20)
    elif domain == 'beta_negative':
        b.fill_(-20)
    elif domain == 'large':
        mixed.mul_(20)
        state.mul_(20)
    elif domain == 'api_null':
        indices[2].zero_()
        indices[3, 0] = 0
    elif domain != 'normal':
        raise ValueError(domain)
    return dict(mixed_qkv=mixed, a=a, b=b, A_log=alog, dt_bias=dt, state_indices=indices, cu_seqlens=cu, num_accepted_tokens=accepted, state=state, output_gate=gate, norm_weight=weight, out=out, scale=128 ** (-0.5), norm_eps=1e-06, output_gate_activation='sigmoid')
'Pre-conv QKVZ with default SD conv-cache layout, BF16 GDN state.'
import torch
prepare = _workload_profiles_prepare
convolution_history = _gdn_workload_convolution_history
post_inputs = _postconv_input_make_inputs
guarded = _postconv_input_guarded

def _make_inputs(tp, width, lengths, accepted_mode='cycle', domain='normal', state_dtype='bf16', device='cuda', head_ratio=3, state_regime='continuation', pool_size=2048, state_slot_padding=32, slot_pattern='scattered', conv_slot_padding=32):
    x = post_inputs(tp, width, lengths, accepted_mode, domain, state_dtype, device, head_ratio=head_ratio, pool_size=pool_size, state_slot_padding=state_slot_padding, slot_pattern=slot_pattern, state_regime=state_regime, preconv=True)
    channels = x['mixed_qkv'].shape[1]
    pool = x['state'].shape[0]
    history = width + 2
    stride = channels * history + conv_slot_padding
    backing = guarded((pool * stride,), torch.bfloat16, device, 15)
    conv = backing.as_strided((pool, channels, history), (stride, 1, channels))
    conv.normal_(0, 0.4)
    convolution_history(conv, 16 // tp, 16 * head_ratio // tp)
    weight = guarded((channels, 4), torch.bfloat16, device, 17)
    weight.normal_(0, 0.25)
    qk = 2 * (16 // tp) * 128
    if domain == 'zero':
        conv.zero_()
    elif domain == 'tiny':
        conv[:, qk:] *= 1e-07
    elif domain == 'qk_nearzero':
        conv[:, :qk] *= 1e-05
    elif domain == 'large':
        conv.mul_(20)
    elif domain == 'cancellation':
        conv[:, ::2] *= -8
    p = dict(x, conv_state=conv, conv_weight=weight)
    shape = {'tp': tp, 'width': width, 'lengths': lengths, 'accepted_mode': accepted_mode, 'domain': domain, 'state_dtype': state_dtype, 'device': device, 'head_ratio': head_ratio, 'state_regime': state_regime, 'pool_size': pool_size, 'state_slot_padding': state_slot_padding, 'slot_pattern': slot_pattern, 'conv_slot_padding': conv_slot_padding}
    prepare(p, shape, 'gdn_conv')
    return {'mixed_qkv': p['mixed_qkv'], 'conv_state': p['conv_state'], 'conv_weight': p['conv_weight'], 'a': p['a'], 'b': p['b'], 'A_log': p['A_log'], 'dt_bias': p['dt_bias'], 'state_indices': p['state_indices'], 'cu_seqlens': p['cu_seqlens'], 'num_accepted_tokens': p['num_accepted_tokens'], 'state': p['state'], 'output_gate': p['output_gate'], 'norm_weight': p['norm_weight'], 'out': p.get('out', None), 'scale': p.get('scale', 128 ** (-0.5)), 'norm_eps': p.get('norm_eps', 1e-05), 'output_gate_activation': p.get('output_gate_activation', 'silu')}

def _state_write_slots(x):
    cu = x['cu_seqlens'].tolist()
    indices = x['state_indices'].tolist()
    accepted = x['num_accepted_tokens'].tolist()
    slots = []
    null = []
    for i, (ss, acc) in enumerate(zip(indices, accepted)):
        count = cu[i + 1] - cu[i]
        if count <= 0:
            continue
        if count > 8 or not 0 < acc <= len(ss) or ss[acc - 1] <= 0:
            null.extend(range(cu[i], cu[i + 1]))
            continue
        slots.extend((s for s in ss[:count] if s > 0))
    return (sorted(set(slots)), null)

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['mixed_qkv', 'conv_state', 'state', 'out']}
    slots, _ = _state_write_slots(x)
    views['state'] = [x['state'][s] for s in slots]
    bounds = x['cu_seqlens'].tolist()
    slots = x['state_indices'][:, 0].tolist()
    views['conv_state'] = [x['conv_state'][s, :, :end-begin+2] for s, begin, end in zip(slots, bounds, bounds[1:]) if s > 0 and end > begin]
    views['mixed_qkv'] = [x['mixed_qkv'][begin:end] for s, begin, end in zip(slots, bounds, bounds[1:]) if s > 0 and end > begin]
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['mixed_qkv', 'conv_state', 'state', 'out']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
