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
import torch
_raw_input_STATE_DTYPE = torch.float32

def _raw_input_make_inputs(lengths, tp, state_mode='mixed', distribution='normal', device='cuda', heads=None):
    assert tp in (1, 2, 4, 8) and lengths and all((n > 0 for n in lengths))
    h, d, total = (heads or 96 // tp, 128, sum(lengths))
    assert h in (8, 12, 16, 24, 32, 48, 64, 96)
    assert max(lengths) <= 16384
    shape = (1, total, h, d)
    q, k, v = [torch.nn.functional.silu(torch.randn(shape, dtype=torch.bfloat16, device=device)) for _ in range(3)]
    g = torch.randn(shape, dtype=torch.bfloat16, device=device)
    gates = torch.randn((1, total, h * 2), dtype=torch.bfloat16, device=device)
    beta = gates[..., h:]
    A_log = torch.linspace(-2.0, 1.0, h, device=device, dtype=torch.float32)
    bias = torch.randn(h * d, device=device, dtype=torch.float32) * 0.25 - 1.0
    state = torch.randn(len(lengths), h, d, d, device=device, dtype=torch.float32) * 0.02
    if state_mode == 'zero':
        state.zero_()
    elif state_mode == 'mixed':
        state[::2].zero_()
    elif state_mode != 'continuation':
        raise ValueError(state_mode)
    if distribution == 'tiny':
        q.mul_(0.001)
        k.mul_(0.001)
        v.mul_(0.001)
        state.mul_(0.001)
    elif distribution == 'persistent':
        g.fill_(-6.0)
        A_log.zero_()
        bias.zero_()
    elif distribution == 'rapid_decay':
        g.fill_(6.0)
        A_log.zero_()
        bias.zero_()
    elif distribution == 'beta_extremes':
        beta[..., ::2] = -24.0
        beta[..., 1::2] = 24.0
    elif distribution == 'near_zero_qk':
        q.mul_(1e-06)
        k.mul_(1e-06)
    elif distribution != 'normal':
        raise ValueError(distribution)
    pointer = [0]
    for length in lengths:
        pointer.append(pointer[-1] + length)
    return dict(q=q, k=k, v=v, raw_g=g, raw_beta=beta, A_log=A_log, g_bias=bias, scale=d ** (-0.5), initial_state=state.to(_raw_input_STATE_DTYPE), output_final_state=True, lower_bound=-5.0, use_qk_l2norm_in_kernel=True, cu_seqlens=torch.tensor(pointer, device=device, dtype=torch.int32), out=None)
'Packed raw-beta prefill; storage guards are part of ABI validation.'
import torch
prepare = _workload_profiles_prepare
STATE_DTYPE = torch.bfloat16

def guarded(value, tail=0):
    guard = 128 // value.element_size()
    owner = torch.full((guard + value.numel() + tail + guard,), 19, dtype=value.dtype, device=value.device)
    result = owner[guard:guard + value.numel()].view(value.shape)
    result.copy_(value)
    return result

def _make_inputs(lengths, tp, state_mode=None, distribution='normal', device='cuda', heads=None, checkpoint_every=0, output_owner='caller', beta_layout='contiguous'):
    requested_state_mode = state_mode
    if state_mode is None:
        state_mode = 'mixed'
    a = _raw_input_make_inputs(lengths, tp, state_mode, distribution, device, heads)
    total, h = a['q'].shape[1:3]
    assert total > 1
    if beta_layout == 'contiguous':
        beta = guarded(a['raw_beta'])
    elif beta_layout == 'row_strided':
        beta_owner = guarded(torch.zeros((1, total, 2 * h), dtype=torch.bfloat16, device=device))
        beta = beta_owner[..., h:]
        beta.copy_(a['raw_beta'])
    else:
        raise ValueError('beta_layout must be contiguous or row_strided')
    x = dict(q=guarded(a['q']), k=guarded(a['k']), v=guarded(a['v']), g=guarded(a['raw_g']), beta=beta, A_log=guarded(a['A_log']), dt_bias=guarded(a['g_bias']), scale=128 ** (-0.5), initial_state=guarded(a['initial_state'].to(STATE_DTYPE)), output_final_state=True, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, lower_bound=-5.0, cu_seqlens=guarded(a['cu_seqlens'].to(torch.int64)), beta_is_logit=True, output=guarded(torch.full_like(a['v'], -17), 128 * h * 128) if output_owner == 'caller' else None, state_checkpoints=None, checkpoint_cu_starts=None, checkpoint_every_n_tokens=checkpoint_every)
    if checkpoint_every:
        assert checkpoint_every in (32, 64, 128)
        counts = [0]
        for length in lengths:
            counts.append(counts[-1] + (length + checkpoint_every - 1) // checkpoint_every)
        x['checkpoint_cu_starts'] = guarded(torch.tensor(counts, dtype=torch.int64, device=device))
        x['state_checkpoints'] = guarded(torch.full((counts[-1], h, 128, 128), -17, dtype=STATE_DTYPE, device=device))
    p = x
    shape = {'lengths': lengths, 'tp': tp, 'state_mode': state_mode, 'distribution': distribution, 'device': device, 'heads': heads, 'checkpoint_every': checkpoint_every, 'output_owner': output_owner, 'beta_layout': beta_layout}
    if requested_state_mode is None:
        shape.pop('state_mode')
    prepare(p, shape, 'kda_prefill')
    return {'q': p['q'], 'k': p['k'], 'v': p['v'], 'g': p['g'], 'beta': p['beta'], 'A_log': p.get('A_log', None), 'dt_bias': p.get('dt_bias', None), 'scale': p.get('scale', None), 'initial_state': p.get('initial_state', None), 'output_final_state': p.get('output_final_state', False), 'use_qk_l2norm_in_kernel': p.get('use_qk_l2norm_in_kernel', True), 'use_gate_in_kernel': p.get('use_gate_in_kernel', False), 'lower_bound': p.get('lower_bound', None), 'cu_seqlens': p.get('cu_seqlens', None), 'beta_is_logit': p.get('beta_is_logit', False), 'output': p.get('output', None), 'state_checkpoints': p.get('state_checkpoints', None), 'checkpoint_cu_starts': p.get('checkpoint_cu_starts', None), 'checkpoint_every_n_tokens': p.get('checkpoint_every_n_tokens', 0)}

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['output', 'initial_state', 'state_checkpoints']}
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['output', 'initial_state', 'state_checkpoints']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
