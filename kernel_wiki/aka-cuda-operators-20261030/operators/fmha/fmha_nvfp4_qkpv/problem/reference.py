"""Independent FP32 reference over the actual stored production inputs.

**version**: explicit-nvfp4-qkpv-v1

**graph**: External BF16 Q/K/V → NVFP4 Q/K/V QDQ → FP32 QK → FP4 probability QDQ → FP32 PV accumulation with unrounded softmax denominator → BF16 output.

**qkv**: Q/K groups16 consecutive head channels. V groups16 consecutive tokens per segment/head/channel; tail zero padded only within the quantization group. sf=RN_E4M3FN_sat448(max(amax/6,2^-9)); payload=RN_E2M1_sat6(x/sf); dequant=payload*sf.

**probability**: 128 consecutive key tokens per group; segment-local order. P payload=RN_E2M1_sat6(6*exp(score-block_max)); E4M3 scale=RN_sat448(exp(block_max-normalizer)/6). Initial normalizer=-inf; update when block_max-old_normalizer>8*ln(2), rescaling numerator and denominator. Denominator sums unrounded FP32 exponentials.

**rounding**: Nearest ties-to-even for E2M1/E4M3 casts, finite saturation; FP32 division defines normalization. Native approximate reciprocal near a midpoint is an implementation discrepancy, not substituted into the independent scoring reference.

**layout**: THD BF16 external inputs and output alias unchanged. Head dimension128, noncausal ragged segments, pv_mode=fp4 as generated. Internal packed layout, CTA and split scheduling remain candidate implementation choices.

**source**: Pinned minimax_h3_varlen_attention cake_backend.py and shipped sm_103a quantizer c91aa0cff6efdec01179 / FP4PV b117806ae1ee4e7876de kernels; explicit delayed normalization policy.

**arithmetic**: Independent FP32 between QDQ with TF32 disabled; no native kernel/internal tensor/hidden prequantized input or FP64 truth.

Numerical comparison uses the supplied standard runner config. No native
kernel or hidden prequantization tensor supplies the scoring truth.
"""
import torch

def e2m1(value):
    levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    mag = value.abs()
    code = torch.zeros_like(mag, dtype=torch.long)
    for i in range(7):
        mid = (levels[i] + levels[i + 1]) / 2
        code += (mag > mid if i % 2 == 0 else mag >= mid).long()
    lut = torch.tensor(levels, device=value.device, dtype=torch.float32)
    return torch.copysign(lut[code], value)

def nvfp4(value, *, min_scale=0.0, norm_const=1.0):
    blocks = value.float().reshape(*value.shape[:-1], -1, 16)
    sf = blocks.abs().amax(-1) * (1.0 / 6.0) * norm_const
    sf = sf.clamp(min_scale, 448.0).to(torch.float8_e4m3fn).float()
    normalized = torch.where(sf[..., None] > 0, blocks * norm_const / sf[..., None], 0.0)
    return (e2m1(normalized) * sf[..., None]).reshape_as(value)

def compute(payload):
    x = payload['call']
    if x['pv_mode'] != 'fp4':
        raise ValueError('Declared profile is pv_mode=fp4')
    q = nvfp4(x['query'], min_scale=2.0 ** (-9))
    k = nvfp4(x['key'], min_scale=2.0 ** (-9))
    out = torch.empty_like(x['out'])
    bounds = x['cu_seqlens'].tolist()
    for a, b in zip(bounds, bounds[1:]):
        if a == b:
            continue
        v = x['value'][a:b].float().permute(1, 2, 0)
        padded = torch.nn.functional.pad(v, (0, -(b - a) % 16))
        v = nvfp4(padded, min_scale=2.0 ** (-9))[..., :b - a].transpose(-1, -2)
        for h in range(q.shape[1]):
            for start in range(a, b, 128):
                end = min(start + 128, b)
                logits = q[start:end, h] @ k[a:b, h].T * x['softmax_scale']
                maximum = torch.full((end - start, 1), -torch.inf, device=q.device)
                denom = torch.zeros_like(maximum)
                numerator = torch.zeros((end - start, q.shape[-1]), device=q.device)
                for left in range(0, b - a, 128):
                    score = logits[:, left:left + 128]
                    block_max = score.amax(-1, keepdim=True)
                    new_max = torch.where(block_max - maximum > 8.0 * __import__('math').log(2.0), block_max, maximum)
                    rescale = (maximum - new_max).exp()
                    p_scale = ((block_max - new_max).exp() / 6.0).clamp(0.0, 448.0).to(torch.float8_e4m3fn).float()
                    probability = e2m1((score - block_max).exp() * 6.0) * p_scale
                    numerator = numerator * rescale + probability @ v[h, left:left + 128]
                    denom = denom * rescale + (score - new_max).exp().sum(-1, keepdim=True)
                    maximum = new_max
                out[start:end, h] = (numerator / denom).to(out.dtype)
    return {'out': out}

class Model(torch.nn.Module):

    def forward(self, query, key, value, cu_seqlens, *, pv_mode='fp8', softmax_scale=None, out=None, cu_seqlens_host=None, backend='cake'):
        torch.backends.cuda.matmul.allow_tf32 = False
        if out is None:
            out = torch.empty_like(query)
        x = dict(query=query, key=key, value=value, cu_seqlens=cu_seqlens, softmax_scale=query.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale, out=out, pv_mode=pv_mode)
        out.copy_(compute({'call': x})['out'])
        return out
