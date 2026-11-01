"""Per-token/head FP8 KV with independent FP32 QK and BF16 weighted P.

For each 32-key normalization group, apply K scale after QK, and round
P * V_scale to BF16 before PV against the stored V payload. The softmax
denominator uses unrounded P. Grouping specifies numerical semantics,
not a required candidate kernel schedule.
"""
import math
import torch

def _attention(q, k, v, out, cu_seqlens_q, seq_lens, block_table, softmax_scale, q_scale, k_scale, v_scale, causal, window_left, sinks, lse):
    ptr = cu_seqlens_q.tolist()
    page = k.shape[2]
    mapping = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // k.shape[1])
    for b, length in enumerate(seq_lens.tolist()):
        first, last = ptr[b:b + 2]
        qlen = last - first
        for start in range(0, qlen, 32):
            end = min(start + 32, qlen)
            query = q[first + start:first + end].float() * q_scale
            visible = length - qlen + torch.arange(start, end, device=q.device)
            maximum = torch.full(query.shape[:2], -torch.inf, device=q.device)
            denom = torch.zeros_like(maximum)
            accum = torch.zeros(*query.shape[:2], v.shape[-1], device=q.device)
            for left in range(0, length, 32):
                positions = torch.arange(left, min(length, left + 32), device=q.device)
                pages = block_table[b, positions // page].long()
                key = k[pages, :, positions % page][:, mapping].float()
                value = v[pages, :, positions % page][:, mapping].float()
                key_scale = k_scale[pages, positions % page][:, mapping].T[None]
                value_scale = v_scale[pages, positions % page][:, mapping].T[None]
                score = torch.einsum('qhd,khd->qhk', query, key) * (softmax_scale * key_scale)
                allowed = torch.ones((end - start, len(positions)), dtype=torch.bool, device=q.device)
                if causal:
                    allowed &= positions[None] <= visible[:, None]
                if window_left >= 0:
                    allowed &= positions[None] >= visible[:, None] - window_left
                score.masked_fill_(~allowed[:, None], -torch.inf)
                next_max = torch.maximum(maximum, score.amax(-1))
                safe = torch.where(torch.isfinite(next_max), next_max, 0.0)
                rescale = (maximum - safe).exp()
                weights = (score - safe[..., None]).exp()
                weighted_p = (weights * value_scale).to(q.dtype).float()
                accum = accum * rescale[..., None] + torch.einsum('qhk,khd->qhd', weighted_p, value)
                denom = denom * rescale + weights.sum(-1)
                maximum = next_max
            base_lse = maximum + denom.log()
            output = accum / denom[..., None]
            total_lse = base_lse
            if sinks is not None:
                total_lse = torch.logaddexp(base_lse, sinks[None])
                output *= (base_lse - total_lse).exp()[..., None]
            out[first + start:first + end].copy_(output)
            if lse is not None:
                lse[first + start:first + end].copy_(total_lse)
    return (out, lse)

def normalized(x):
    kind = 'pth'
    return dict(q=x['q'], k=x['k'].transpose(1, 2), v=x['v'].transpose(1, 2), out=x['out'], cu_seqlens_q=x['cu_seqlens_q'], seq_lens=x['seqused_k'], block_table=x['block_table'], softmax_scale=x['softmax_scale'], q_scale=1.0, k_scale=x['k_scale_cache'], v_scale=x['v_scale_cache'], causal=True, window_left=-1, sinks=None, lse=None)

def returned(x):
    return None

class Model(torch.nn.Module):

    def __init__(self, **prepared):
        super().__init__()
        self.prepared = prepared

    def forward(self, q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal, window_size, block_table, softcap, q_descale, k_descale, v_descale, seq_threshold_3D=None, num_par_softmax_segments=None, softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None, alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None, mm_prefix_range=None, rswa_prefix_lens=None, rswa_window=None, use_alibi_sqrt=False, kv_quant_mode=0, k_scale_cache=None, v_scale_cache=None, chunk_lookback=-1, use_td=False, mm_prefix_clamp_sliding_window=False):
        x = locals().copy()
        x.pop('self')
        torch.backends.cuda.matmul.allow_tf32 = False
        z = normalized(x)
        target = z['out']
        target_lse = z['lse']
        z['out'] = torch.empty_like(target, dtype=torch.float32)
        if target_lse is not None:
            z['lse'] = torch.empty_like(target_lse)
        _attention(**z)
        out.copy_(z['out'].reshape_as(out))
        if target_lse is not None:
            target_lse.copy_(z['lse'] / math.log(2))
        return returned(x)
