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
'Real ordinary decode: one token and one state slot per request.'
import math
import torch
prepare = _workload_profiles_prepare
projected = _gdn_workload_projected
gate_logits = _gdn_workload_gate_logits
structured_state = _gdn_workload_structured_state
convolution_history = _gdn_workload_convolution_history
slots = _gdn_workload_slots

def guarded(shape, dtype, device, value=7):
    n = math.prod(shape)
    base = torch.full((n + 32,), value, dtype=dtype, device=device)
    return base[16:16 + n].view(shape)

def _make_inputs(tp, batch, domain='normal', spare=0, null_mode='none', state_dtype='fp32', device='cuda', head_ratio=3, state_regime='continuation', pool_size=2048, state_slot_padding=32, slot_pattern='scattered', conv_slot_padding=32):
    h, hv = (16 // tp, 16 * head_ratio // tp)
    t = batch
    c = (2 * h + hv) * 128
    pool = pool_size
    qkvz = guarded((t + 3, c + hv * 128), torch.bfloat16, device)
    qkvz.normal_(0, 0.4)
    projected(qkvz, h, hv)
    ba = guarded((t + 3, 2 * hv), torch.bfloat16, device, 8)
    ba.normal_()
    gate_logits(ba[:, hv:], ba[:, :hv])
    weight = guarded((c, 4), torch.bfloat16, device, 9)
    weight.normal_(0, 0.25)
    width = 3 + spare
    stride = c * width + conv_slot_padding
    conv = guarded((pool * stride,), torch.bfloat16, device, 10).as_strided((pool, c, width), (stride, 1, c))
    conv.normal_(0, 0.4)
    convolution_history(conv, h, hv)
    num = hv * 128 * 128
    ss = num + state_slot_padding
    state = guarded((pool * ss,), {'bf16': torch.bfloat16, 'fp32': torch.float32}[state_dtype], device, 11).as_strided((pool, hv, 128, 128), (ss, 128 * 128, 128, 1))
    structured_state(state, state_regime)
    alog = guarded((hv,), torch.float32, device, 12)
    alog.copy_(torch.linspace(-3, 1, hv, device=device))
    dt = guarded((hv,), torch.bfloat16, device, 13)
    dt.copy_(torch.linspace(-3, 3, hv, device=device))
    norm = guarded((128,), torch.bfloat16, device, 14)
    norm.uniform_(0.75, 1.25)
    out = guarded((t + 3, hv, 128), torch.bfloat16, device, 15)
    si = 1 + spare
    slot_ids = guarded((t * si,), torch.int32, device, 16)[::si]
    slot_ids.copy_(slots(t, pool, slot_pattern, device))
    if null_mode == 'mixed':
        slot_ids[::3] = 0
    elif null_mode == 'all':
        slot_ids.zero_()
    qk = 2 * h * 128
    if domain == 'tiny':
        qkvz[:t, qk:c] *= 1e-07
        conv[:, qk:] *= 1e-07
        state *= 1e-07
    elif domain == 'zero':
        qkvz[:t, qk:c] = 0
        conv[:, qk:] = 0
        state.zero_()
    elif domain == 'nearzero_qk':
        qkvz[:t, :qk] *= 1e-05
        conv[:, :qk] *= 1e-05
    elif domain == 'slow_decay':
        ba[:t, hv:] = -20
    elif domain == 'cancellation':
        qkvz[:t, qk:c:2] *= -8
        conv[:, qk::2] *= -8
        state[..., ::2] *= -8
    elif domain == 'gate_negative':
        qkvz[:t, c:] = -20
    metadata = dict(num_prefills=0, num_prefill_tokens=0, num_decodes=t, num_decode_tokens=t, num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=t, non_spec_state_indices_tensor=slot_ids)
    p = dict(qkvz=qkvz, ba=ba, conv_weight=weight, conv_state=conv, A_log=alog, dt_bias=dt, state=state, norm_weight=norm, out=out, metadata=metadata)
    shape = {'tp': tp, 'batch': batch, 'domain': domain, 'spare': spare, 'null_mode': null_mode, 'state_dtype': state_dtype, 'device': device, 'head_ratio': head_ratio, 'state_regime': state_regime, 'pool_size': pool_size, 'state_slot_padding': state_slot_padding, 'slot_pattern': slot_pattern, 'conv_slot_padding': conv_slot_padding}
    prepare(p, shape, 'gdn_postconv')
    return {'qkvz': p['qkvz'], 'ba': p['ba'], 'conv_weight': p['conv_weight'], 'conv_state': p['conv_state'], 'A_log': p['A_log'], 'dt_bias': p['dt_bias'], 'state': p['state'], 'norm_weight': p['norm_weight'], 'out': p['out'], 'metadata': p['metadata']}

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['qkvz', 'conv_state', 'state', 'out']}
    slots = x['metadata']['non_spec_state_indices_tensor'].tolist()
    active = [(row, s) for row, s in enumerate(slots) if s > 0]
    views['state'] = [x['state'][s] for _, s in active]
    views['conv_state'] = [x['conv_state'][s, :, :3] for _, s in active]
    channels = x['conv_weight'].shape[0]
    views['qkvz'] = [x['qkvz'][row, :channels] for row, _ in active]
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['qkvz', 'conv_state', 'state', 'out']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
