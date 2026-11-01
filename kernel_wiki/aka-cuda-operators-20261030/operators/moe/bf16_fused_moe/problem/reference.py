"""Independent FP32 oracle from stored operands. No candidate imports."""
import torch

def activation(value, x):
    gate, up = value.chunk(2, -1)
    if x['activation'] == 'situ':
        gate = x['situ_gate'] * torch.tanh(gate / x['situ_gate']) * torch.sigmoid(gate)
        up = x['situ_up'] * torch.tanh(up / x['situ_up'])
        return gate * up
    if x['activation_limit'] is not None:
        gate = gate.clamp(max=x['activation_limit'])
        up = up.clamp(-x['activation_limit'], x['activation_limit'])
    return gate * torch.sigmoid(x['activation_alpha'] * gate) * (up + x['activation_beta'])

def views(p):
    """Read only production input regions of the fixed DeepGEMM heap.

    ABI: e1f418c2 layout/mega_moe.cuh and sym_buffer.cuh. Signals have
    72 peer slots and 128-byte struct alignment; input offsets depend on
    full-pool metadata, not ring scheduling or the number of GPU SMs.
    """
    ranks = len(p['sym_buffer_ptrs'])
    experts = p['num_experts']
    cap = p['num_max_tokens_per_rank']
    topk = p['num_topk']
    hidden = p['y'].shape[-1]
    align128 = lambda n: (n + 127) // 128 * 128
    signals = align128(align128(align128(44) + 72 * 8) + 72 * 8)
    signals = align128(signals + 2 * 2048 * 8 + 512 * 8 + (1 << 20) * 20 + (1 << 15) * 4)
    local = experts // ranks
    pool = (ranks * cap * min(topk, local) + local * 239 + 1919) // 1920 * 1920
    offset = (signals + experts * cap * 4 + pool * 12 + 15) // 16 * 16
    raw = p['sym_buffer'].view(torch.uint8).reshape(-1)

    def take(shape, dtype):
        nonlocal offset
        count = 1
        for size in shape:
            count *= size
        count *= torch.empty((), dtype=dtype).element_size()
        result = raw[offset:offset + count].view(dtype).view(shape)
        offset += count
        return result
    x = take((cap, hidden), torch.bfloat16)
    sf = None
    indices = take((cap, topk), torch.int64)
    weights = take((cap, topk), torch.float32)
    return (x, sf, indices, weights)

def unpack_weights(w, gated):
    value = w.float()
    if gated:
        e, n, k = value.shape
        value = value.reshape(e, n // 16, 2, 8, k).permute(0, 2, 1, 3, 4).reshape(e, n, k)
    return value

def expected(p):
    v = views(p)
    t, h = p['y'].shape
    a = v[0][:t].float()
    w1 = unpack_weights(p['l1_weights'], True)
    w2 = unpack_weights(p['l2_weights'], False)
    ids, scores = (v[2][:t], v[3][:t])
    y = torch.zeros((t, h), dtype=torch.float32, device=a.device)
    cfg = dict(activation=p['activation'], situ_gate=p['activation_alpha'], situ_up=p['activation_beta'], activation_alpha=p['activation_alpha'], activation_beta=p['activation_beta'], activation_limit=p['activation_clamp_opt'])
    for e in range(w1.shape[0]):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        for off in range(0, len(tok), 64):
            rows = tok[off:off + 64]
            weights = scores[rows, slot[off:off + 64]]
            # FC1 and the routing-weighted activation are stored BF16 tensors.
            fc1 = torch.einsum('th,ih->ti', a[rows], w1[e]).to(torch.bfloat16).float()
            hidden = (activation(fc1, cfg) * weights[:, None]).to(torch.bfloat16).float()
            value = torch.einsum('ti,hi->th', hidden, w2[e]).to(torch.bfloat16).float()
            y.index_add_(0, rows, value)
    return y.to(p['y'].dtype)

class Model(torch.nn.Module):

    def forward(self, y, l1_weights, l2_weights, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats, sym_buffer, sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_experts, num_topk, activation, activation_clamp_opt, fast_math, activation_alpha, activation_beta):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = {'y': y, 'l1_weights': l1_weights, 'l2_weights': l2_weights, 'shared_l1_weights': shared_l1_weights, 'shared_l2_weights': shared_l2_weights, 'cumulative_local_expert_recv_stats': cumulative_local_expert_recv_stats, 'sym_buffer': sym_buffer, 'sym_buffer_ptrs': sym_buffer_ptrs, 'rank_idx': rank_idx, 'num_max_tokens_per_rank': num_max_tokens_per_rank, 'num_experts': num_experts, 'num_topk': num_topk, 'activation': activation, 'activation_clamp_opt': activation_clamp_opt, 'fast_math': fast_math, 'activation_alpha': activation_alpha, 'activation_beta': activation_beta}
        result = expected(p)
        y.copy_(result)
        return None
