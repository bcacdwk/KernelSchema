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
    return x

def _workload_profiles_routes(shape, tokens, experts, topk, device, rank=0, world=1):
    """IDs and normalized weights derive from the SAME router logits.

    Adjacent groups of eight tokens share a feature, with token-specific noise.
    Bias profiles control expert imbalance without changing the forward ABI.
    """
    assert 0 < topk <= experts and experts % world == 0
    mode = shape.get('routing', 'balanced')
    features = torch.randn(((tokens + 7) // 8, 8), device=device)
    features = features.repeat_interleave(8, 0)[:tokens]
    features = 0.6 * features + 0.8 * torch.randn((tokens, 8), device=device)
    projection = torch.randn((8, experts), device=device) / math.sqrt(8)
    logits = features @ projection + 0.35 * torch.randn((tokens, experts), device=device)
    expert = torch.arange(experts, device=device)
    if mode == 'skewed':
        logits += 1.5 * torch.cos(expert.float() * 2 * math.pi / experts)
    elif mode == 'hotspot':
        logits[:, :topk] += 32.0
    elif mode in ('remote', 'local'):
        assert world > 1
        destination = (rank + 1) % world if mode == 'remote' else rank
        logits += (expert // (experts // world) == destination).float() * 4.0
    elif mode not in ('balanced', 'uniform', 'empty'):
        raise ValueError('unknown routing profile: ' + mode)
    selected, ids = logits.topk(topk, dim=-1)
    weights = selected.softmax(-1)
    if mode == 'empty':
        ids.fill_(-1)
    if shape.get('masked_routes'):
        ids[::3, -1] = -1
    valid_tokens = shape.get('rank_lengths', [tokens] * world)[rank]
    ids[valid_tokens:] = -1
    weights.masked_fill_(ids < 0, 0.0)
    valid = ids[ids >= 0]
    loads = torch.bincount(valid, minlength=experts)
    destinations = torch.bincount(valid // (experts // world), minlength=world)
    _workload_profiles_LAST_STATS['router'] = dict(seed=torch.initial_seed(), profile=mode, rank=rank, world_size=world, valid_tokens=valid_tokens, routed_rows=int(valid.numel()), active_experts=int((loads > 0).sum()), expert_rows=loads.cpu().tolist(), destination_rows=destinations.cpu().tolist(), max_to_mean_load=float(loads.max()) / max(float(loads.float().mean()), 1e-30), mean_top1_weight=float(weights.max(-1).values.mean()) if tokens else 0.0, ids_and_weights_from_same_logits=True)
    _workload_profiles_INPUT_RECORDS.append(dict(_workload_profiles_LAST_STATS['router']))
    del _workload_profiles_INPUT_RECORDS[:-2]
    return (ids, weights)
'Canonical stored-precision inputs; no dependency on a serving framework.'
import math
import torch
matrix_values = _workload_profiles_matrix_values
routes = _workload_profiles_routes

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

def matrix(shape, block, device, std=1.0, zero=False, global_scale=1.0, scale_rows=1):
    if len(shape) == 3:
        chunks = [matrix(shape[1:], block, device, std, zero, global_scale, scale_rows) for _ in range(shape[0])]
        return (torch.stack([x[0] for x in chunks]), None if chunks[0][1] is None else torch.stack([x[1] for x in chunks]))
    value = matrix_values(shape, device, std, role='weight' if std != 1.0 else 'projection')
    if zero:
        value.zero_()
    return quantize(value, block, global_scale, scale_rows)

def make(s, seed=0, device='cuda'):
    torch.manual_seed(seed)
    t, h, i, e, k = map(s.__getitem__, ('tokens', 'hidden', 'intermediate', 'local_experts', 'topk'))
    a, sa = matrix((t, h), 128, device, zero=s.get('distribution') == 'zero')
    w1, s1 = matrix((e, 2 * i, h), 128, device, std=h ** (-0.5), scale_rows=128)
    w2, s2 = matrix((e, h, i), 128, device, std=(i * s.get('expert_tp', 1)) ** (-0.5), scale_rows=128)
    ids, score = routes(s, t, e, k, device)
    ids = ids.to(torch.int32)
    return dict(topk_ids=(ids, score), routing_bias=None, hidden_states=a, hidden_states_scale=sa.T.contiguous(), gemm1_weights=w1, gemm1_weights_scale=s1, gemm2_weights=w2, gemm2_weights_scale=s2, num_experts=e, top_k=k, n_group=None, topk_group=None, intermediate_size=i, local_expert_offset=0, local_num_experts=e, routed_scaling_factor=1.0, routing_method_type=0, use_shuffled_weight=False, weight_layout=0, do_finalize=True, enable_pdl=False, gemm1_lora_delta=None, output=guarded((t, h), torch.bfloat16, device), tune_max_num_tokens=8192, fp8_quantization_type=1, activation_type=3, gemm1_alpha=torch.full((e,), s.get('activation_alpha', 1.0), dtype=torch.float32, device=device), gemm1_beta=torch.full((e,), s.get('activation_beta', 0.0), dtype=torch.float32, device=device), gemm1_clamp_limit=None if s.get('activation_limit') is None else torch.full((e,), s['activation_limit'], dtype=torch.float32, device=device), valid_hidden_size=None, valid_intermediate_size=None)

def _make_inputs(hidden, intermediate, local_experts, tokens, topk, activation_alpha=None, activation_beta=None, activation_limit=None, b_scale_rows=None, distribution=None, masked_routes=None, routing=None, model_geometry=None, expert_tp=None, device='cuda'):
    shape = {'device': device, **{k: v for k, v in {'activation_alpha': activation_alpha, 'activation_beta': activation_beta, 'activation_limit': activation_limit, 'b_scale_rows': b_scale_rows, 'distribution': distribution, 'hidden': hidden, 'intermediate': intermediate, 'local_experts': local_experts, 'masked_routes': masked_routes, 'routing': routing, 'tokens': tokens, 'topk': topk, 'model_geometry': model_geometry, 'expert_tp': expert_tp}.items() if v is not None}}
    p = make({k: v for k, v in shape.items() if k != 'device'}, seed=torch.initial_seed(), device=device)
    return {'topk_ids': p['topk_ids'], 'routing_bias': p['routing_bias'], 'hidden_states': p['hidden_states'], 'hidden_states_scale': p['hidden_states_scale'], 'gemm1_weights': p['gemm1_weights'], 'gemm1_weights_scale': p['gemm1_weights_scale'], 'gemm2_weights': p['gemm2_weights'], 'gemm2_weights_scale': p['gemm2_weights_scale'], 'num_experts': p['num_experts'], 'top_k': p['top_k'], 'n_group': p['n_group'], 'topk_group': p['topk_group'], 'intermediate_size': p['intermediate_size'], 'local_expert_offset': p['local_expert_offset'], 'local_num_experts': p['local_num_experts'], 'routed_scaling_factor': p['routed_scaling_factor'], 'routing_method_type': p.get('routing_method_type', 0), 'use_shuffled_weight': p.get('use_shuffled_weight', False), 'weight_layout': p.get('weight_layout', 0), 'do_finalize': p.get('do_finalize', True), 'enable_pdl': p.get('enable_pdl', None), 'gemm1_lora_delta': p.get('gemm1_lora_delta', None), 'output': p.get('output', None), 'tune_max_num_tokens': p.get('tune_max_num_tokens', 8192), 'fp8_quantization_type': p.get('fp8_quantization_type', 1), 'activation_type': p.get('activation_type', 3), 'gemm1_alpha': p.get('gemm1_alpha', None), 'gemm1_beta': p.get('gemm1_beta', None), 'gemm1_clamp_limit': p.get('gemm1_clamp_limit', None), 'valid_hidden_size': p.get('valid_hidden_size', None), 'valid_intermediate_size': p.get('valid_intermediate_size', None)}

def writable_views(x):
    """Exact caller-visible write regions, including permitted scratch."""
    views = {k: x[k] for k in ['output']}
    return views

def observable_inputs(x):
    """Logical mutated outputs; inactive storage is protected separately."""
    views = writable_views(x)
    return {k: views[k] for k in ['output']}

def replay_tensors(x):
    """Rewind request values while retaining compiled resources and allocations."""
    return writable_views(x)
