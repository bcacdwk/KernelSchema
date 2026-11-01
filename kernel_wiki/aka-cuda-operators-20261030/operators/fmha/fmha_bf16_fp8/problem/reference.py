"""D04: FMHA BF16-Q / FP8-KV. Independent FP32 reference with all scoring helpers in this file."""
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
            for left in range(0, length, 2048):
                positions = torch.arange(left, min(length, left + 2048), device=q.device)
                pages = block_table[b, positions // page].long()
                key = k[pages, :, positions % page][:, mapping].float() * k_scale
                value = v[pages, :, positions % page][:, mapping].float() * v_scale
                score = torch.einsum('qhd,khd->qhk', query, key) * softmax_scale
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
                accum = accum * rescale[..., None] + torch.einsum('qhk,khd->qhd', weights, value)
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

def pointer(lengths, device):
    return torch.tensor([0] + list(__import__('itertools').accumulate(lengths)), dtype=torch.int32, device=device)

def normalized(x):
    kind = 'decode'
    q = x['query']
    lens = x['seq_lens']
    out = x['out']
    k, v = x['kv_cache']
    table = x['block_tables']
    ptr = x.get('cum_seq_lens_q')
    if q.ndim == 4:
        ptr = pointer([q.shape[1]] * q.shape[0], q.device)
        q = q.flatten(0, 1)
        out = out.flatten(0, 1)
    return dict(q=q, k=k, v=v, out=out, cu_seqlens_q=ptr, seq_lens=lens, block_table=table, softmax_scale=x['bmm1_scale'], q_scale=1.0, k_scale=1.0, v_scale=x['bmm2_scale'], causal=x.get('causal', x.get('is_causal', True)), window_left=x.get('window_left', -1), sinks=x.get('sinks', x.get('attention_sinks')), lse=x['lse'])

def returned(x):
    return (x['out'], x['lse']) if x.get('return_lse') else x['out']

class Model(torch.nn.Module):

    def __init__(self, **prepared):
        super().__init__()
        self.prepared = prepared

    def forward(self, query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len, bmm1_scale=1.0, bmm2_scale=1.0, window_left=-1, out=None, out_dtype=None, o_sf_scale=None, o_sf_vec_size=None, sinks=None, kv_layout='HND', enable_pdl=None, backend='auto', q_len_per_req=1, o_scale=1.0, mask=None, max_q_len=None, cum_seq_lens_q=None, skip_softmax_threshold_scale_factor=None, kv_cache_sf=None, uses_shared_paged_kv_idx=True, lse=None, return_lse=False, bmm1_scale_log2=None, multi_ctas_kv_counter_buffer=None, enable_block_sparse_attention=False, cp_world=1, cp_rank=0, causal_seqlens_kv_global=None, bf16q_fp8kv_transform_mode=None, request_order=None, request_order_plan=None):
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
            lse.copy_((z['lse'] / math.log(2)).reshape_as(lse))
        return returned(x)
