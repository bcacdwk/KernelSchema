import math
import torch

def _workload_profiles_matrix_values(shape, device, std=1.0, role='projection'):
    x = torch.randn(shape, device=device)
    channel = torch.linspace(0.65, 1.35, shape[-1], device=device)
    x.mul_(channel)
    if role == 'weight':
        x.mul_(torch.linspace(0.7, 1.3, shape[-2], device=device)[:, None])
    elif role != 'projection':
        raise ValueError(role)
    x.mul_(std)
    return x

def _workload_profiles_routes(shape, tokens, experts, topk, device):
    """IDs and normalized weights derive from the SAME router logits.

    Adjacent groups of eight tokens share a feature, with token-specific noise.
    Bias profiles control expert imbalance without changing the forward ABI.
    """
    assert 0 < topk <= experts
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
    elif mode not in ('balanced', 'empty'):
        raise ValueError('unknown routing profile: ' + mode)
    selected, ids = logits.topk(topk, dim=-1)
    weights = selected.softmax(-1)
    if mode == 'empty':
        ids.fill_(-1)
    if shape.get('masked_routes'):
        ids[::3, -1] = -1
    weights.masked_fill_(ids < 0, 0.0)
    return (ids, weights)

matrix_values = _workload_profiles_matrix_values
routes = _workload_profiles_routes
FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

def guarded(shape, dtype, device):
    size = math.prod(shape)
    backing = torch.full((size + 128,), -17, dtype=dtype, device=device)
    return backing[64:-64].view(shape)

def quantize(value, block):
    k = value.shape[-1]
    assert k % block == 0
    groups = value.float().reshape(*value.shape[:-1], k // block, block)
    peak = groups.abs().amax(-1).clamp_min(2.0 ** (-20))
    bound = 6.0
    scale = peak / bound
    stored_scale = scale.clamp(2.0 ** (-9), 448).to(torch.float8_e4m3fn)
    scale = stored_scale.float()
    normalized = (groups / scale.unsqueeze(-1)).reshape_as(value)
    magnitude = normalized.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for j in range(7):
        midpoint = (FP4[j] + FP4[j + 1]) / 2
        crosses = magnitude > midpoint if j % 2 == 0 else magnitude >= midpoint
        code += crosses.to(torch.uint8)
    code |= torch.signbit(normalized).to(torch.uint8) << 3
    payload = code[..., 0::2] | code[..., 1::2] << 4
    return (payload, stored_scale)

def matrix(shape, block, device, std=1.0, zero=False):
    if len(shape) == 3:
        chunks = [matrix(shape[1:], block, device, std, zero) for _ in range(shape[0])]
        return (torch.stack([x[0] for x in chunks]), None if chunks[0][1] is None else torch.stack([x[1] for x in chunks]))
    value = matrix_values(shape, device, std, role='weight' if std != 1.0 else 'projection')
    if zero:
        value.zero_()
    return quantize(value, block)

def block_scales(scale):
    """128-row x 4-column scale tiles, stored as [tile, column, row32, row4, col4]."""
    e, n, k = scale.shape
    pn, pk = (n + 127) // 128 * 128, (k + 3) // 4 * 4
    raw = torch.zeros((e, pn, pk), dtype=torch.uint8, device=scale.device)
    raw[:, :n, :k] = scale.view(torch.uint8)
    return raw.reshape(e, pn // 128, 4, 32, pk // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().reshape(e, pn // 128, pk // 4, 32, 16).view(scale.dtype)

def prepare_weights(q, sf, gated):
    # Inverse of the 32-row permutation: old row r maps to (r % 4)*8 + r//4.
    n = q.shape[1]
    row = torch.arange(n, device=q.device)
    row = (row // 32) * 32 + (row % 8) * 4 + (row % 32) // 8
    if gated:
        row = row // 2 + (row % 2) * (n // 2)
    payload = q[:, row].contiguous()
    scales = block_scales(sf.view(torch.uint8)[:, row]).reshape_as(sf).view(sf.dtype)
    return payload, scales

def _make_inputs(hidden, intermediate, local_experts, tokens, topk, activation=None, activation_alpha=None, activation_beta=None, activation_limit=None, distribution=None, masked_routes=None, routing=None, model_geometry=None, expert_tp=None, device='cuda'):
    torch.manual_seed(torch.initial_seed())
    e, h, i, k, t = (local_experts, hidden, intermediate, topk, tokens)
    a, sa = matrix((t, h), 16, device, zero=distribution == 'zero')
    w1, s1 = matrix((e, 2 * i, h), 16, device, std=h ** (-0.5))
    w2, s2 = matrix((e, h, i), 16, device, std=(i * (expert_tp or 1)) ** (-0.5))
    w1, s1 = prepare_weights(w1, s1, True)
    w2, s2 = prepare_weights(w2, s2, False)
    ids, scores = routes({'routing':routing or 'balanced', 'masked_routes':masked_routes}, t, e, k, device)
    ids = ids.to(torch.int32)
    situ = activation == 'situ'
    return {'topk_ids':(ids, scores), 'routing_bias':None, 'hidden_states':a, 'hidden_states_scale':sa, 'gemm1_weights':w1, 'gemm1_weights_scale':s1, 'gemm1_bias':None, 'gemm1_alpha':torch.full((e,), 4.0 if situ else (activation_alpha or 1.0), dtype=torch.float32, device=device), 'gemm1_beta':torch.full((e,), 25.0 if situ else (activation_beta or 0.0), dtype=torch.float32, device=device), 'gemm1_clamp_limit':None if activation_limit is None else torch.full((e,), activation_limit, dtype=torch.float32, device=device), 'gemm2_weights':w2, 'gemm2_weights_scale':s2, 'gemm2_bias':None, 'output1_scale_scalar':torch.ones(e, device=device), 'output1_scale_gate_scalar':torch.ones(e, device=device), 'output2_scale_scalar':torch.ones(e, device=device), 'num_experts':e, 'top_k':k, 'n_group':None, 'topk_group':None, 'intermediate_size':i, 'local_expert_offset':0, 'local_num_experts':e, 'routed_scaling_factor':1.0, 'routing_method_type':0, 'do_finalize':True, 'enable_pdl':False, 'activation_type':10 if situ else 3, 'per_token_scale':None, 'output':guarded((t, h), torch.bfloat16, device), 'tune_max_num_tokens':8192, 'gemm1_lora_delta':None, 'valid_hidden_size':None, 'valid_intermediate_size':None, 'num_fused_shared_experts':None, 'hidden_states_scale_layout':2}

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
