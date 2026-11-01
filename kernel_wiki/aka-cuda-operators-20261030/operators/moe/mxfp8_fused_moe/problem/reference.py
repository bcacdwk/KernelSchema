"""Independent FP32 reference over the actual stored production inputs.

1. Decode input E4M3 payloads with UE8M0 scales per contiguous 32 hidden
   features. Decode expert weights from their declared MXFP8 storage,
   scales and gate/up interleave. Scale bytes mean 2**(byte-127); byte 255 is
   excluded. No pre-quantization hidden values.
2. FC1 is independent FP32 matmul with TF32 disabled. Round its gate/up outputs
   to BF16, round-to-nearest ties-to-even, then decode to FP32 for activation.
3. SwiGLU clamps gate at +limit and up at ±limit when specified, then computes
   gate * sigmoid(alpha*gate) * (up+beta). SiTU computes
   alpha*tanh(gate/alpha)*sigmoid(gate), multiplied by beta*tanh(up/beta) when
   beta>0, otherwise by up. Alpha is positive. Clamp applies only to SwiGLU.
4. Multiply each selected expert's activation by its routing weight BEFORE
   quantization/FC2. For each token and contiguous 32 intermediate features,
   set scale=2**max(-22,ceil(log2(amax/448))). For all-zero groups use 2**-22.
   Quantize activation/scale to finite E4M3, nearest ties-to-even, saturation
   ±448, then decode and multiply by scale. Scale is UE8M0 (dequantization
   direction), one scale per group; its packed MN-major layout is internal.
5. FC2 uses independent FP32 matmul of decoded activations/weights. Round
   each expert contribution to BF16 before FP32 route reduction; store final
   y as BF16. Masked/absent routes contribute zero. Multi-rank reduction has
   the same mathematical sum; no particular communication schedule required.

Source: fixed wheel atrex_grouped_mxfp8_native, source revision
e1f418c2a4f20818221f6b0e578b4c2f634d4c3f, wheel SHA256
5ecc99f111a71680f26f40e1d21a22ac8ce468f951925eda2de6421049329b3e.
include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1054-1166,
1268-1282,1438-1466; corresponding _situ.cuh:1076-1165;
include/deep_gemm/common/math.cuh:112-145. L1 output block 64 has two
32-feature scaling groups, independent of implementation scheduling.

Numerical comparison uses the supplied standard runner config. No native
kernel or hidden prequantization tensor supplies the scoring truth.
"""
import torch

def decode(payload, sf, fmt, block, global_scale):
    value = payload.float()
    assert bool((sf < 255).all()), 'NaN E8M0 scale outside input domain'
    scale = torch.ldexp(torch.ones_like(sf, dtype=torch.float32), sf.to(torch.int32) - 127)
    scale = scale.repeat_interleave(value.shape[-2] // scale.shape[-2], -2)
    return value * scale.repeat_interleave(block, -1) * global_scale

def activation(value, x):
    gate, up = value.chunk(2, -1)
    if x['activation'] == 'situ':
        gate = x['situ_gate'] * torch.tanh(gate / x['situ_gate']) * torch.sigmoid(gate)
        if x['situ_up'] > 0:
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
    x = take((cap, hidden), torch.float8_e4m3fn)
    sf = take((cap, hidden // 128), torch.int32)
    indices = take((cap, topk), torch.int64)
    weights = take((cap, topk), torch.float32)
    return (x, sf, indices, weights)

def unpack_weights(w, gated):
    if isinstance(w, tuple):
        q, sf = w
        e, n, c = sf.shape
        sf = sf.contiguous().reshape(e, n // 128, 32, 4, c).transpose(2, 3).reshape(e, n, c).contiguous().view(torch.uint8)
        fmt = 'mxfp8'
        value = decode(q, sf, 'mxfp8', 32, 1.0)
    else:
        value = w.float()
    if gated:
        e, n, k = value.shape
        value = value.reshape(e, n // 16, 2, 8, k).permute(0, 2, 1, 3, 4).reshape(e, n, k)
    return value


def activation_qdq(value):
    """Per-token contiguous 32-feature E4M3/UE8M0 QDQ, independently in FP32."""
    groups = value.reshape(*value.shape[:-1], -1, 32)
    amax = groups.abs().amax(-1, keepdim=True)
    # 448 = 0.875 * 2**9. frexp avoids logarithm rounding at power boundaries.
    mantissa, exponent = torch.frexp(amax)
    scale_exp = (exponent - 9 + (mantissa > 0.875).to(exponent.dtype)).clamp(min=-22)
    scale = torch.ldexp(torch.ones_like(amax), scale_exp)
    payload = (groups / scale).clamp(-448., 448.).to(torch.float8_e4m3fn)
    return (payload.float() * scale).reshape_as(value)

def expected(p):
    v = views(p)
    t, h = p['y'].shape
    if isinstance(p['l1_weights'], tuple):
        sf = v[1][:t].contiguous().view(torch.uint8)
        a = decode(v[0][:t], sf, 'mxfp8', 32, 1.0)
    else:
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
            fc1 = torch.einsum('th,ih->ti', a[rows], w1[e]).to(torch.bfloat16).float()
            hidden = activation_qdq(activation(fc1, cfg) * weights[:, None])
            value = torch.einsum('ti,hi->th', hidden, w2[e]).to(torch.bfloat16).float()
            y.index_add_(0, rows, value)
    return y.to(p['y'].dtype)

class Model(torch.nn.Module):

    def forward(self, y, l1_weights, l2_weights, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats, sym_buffer, sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_experts, num_topk, recipe, activation, activation_clamp_opt, fast_math, activation_alpha, activation_beta):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = {'y': y, 'l1_weights': l1_weights, 'l2_weights': l2_weights, 'shared_l1_weights': shared_l1_weights, 'shared_l2_weights': shared_l2_weights, 'cumulative_local_expert_recv_stats': cumulative_local_expert_recv_stats, 'sym_buffer': sym_buffer, 'sym_buffer_ptrs': sym_buffer_ptrs, 'rank_idx': rank_idx, 'num_max_tokens_per_rank': num_max_tokens_per_rank, 'num_experts': num_experts, 'num_topk': num_topk, 'recipe': recipe, 'activation': activation, 'activation_clamp_opt': activation_clamp_opt, 'fast_math': fast_math, 'activation_alpha': activation_alpha, 'activation_beta': activation_beta}
        result = expected(p)
        y.copy_(result)
        return None
