"""NVFP4 MLA with independent FP32 computation and explicit quantization.

V: decoded E2M1 * E4M3 -> FP16 -> saturating RNE E4M3.
P: 128-key groups, log2 maximum delayed until it increases by >2,
64 * exp(score - normalizer) -> saturating RNE E4M3 for PV.
Denominator/LSE use unrounded P; a sink seeds only the first split.
Prepared split partials are BF16 before an independent FP32 LSE merge.
"""
import math
import torch

def _split_ranges(lengths, qlen, heads, num_sms):
    """Independent description of the prepared API's numerical partitions.

    A stored BF16 partial exists at every split boundary. Reconstruct only
    these boundaries from the public shape/device policy; no kernel, work
    table, workspace contents, or native intermediate supplies the truth.
    """
    tiles = [(length + 127) // 128 for length in lengths]
    row_groups = (qlen * heads + 127) // 128
    if max(lengths) < 8192:
        width = max(tiles)
        kv_bytes = sum(lengths) * 288
        partial_bytes = len(lengths) * qlen * heads * 512 * 4
        while sum((t + width - 1) // width for t in tiles) * row_groups * 2 < 2 * num_sms:
            candidate = max(8, (width + 1) // 2)
            if candidate >= width or ((max(tiles) + candidate - 1) // candidate) * partial_bytes > .35 * kv_bytes:
                break
            width = candidate
        return [[[(lo * 128, min(lo + width, t) * 128)
                   for lo in range(0, t, width)] for _ in range(row_groups)] for t in tiles]
    total = sum(tiles) * row_groups
    units = max(1, min(num_sms // 2, total // 8))
    while units > 1 and math.ceil(max(tiles) / (total / units)) + 1 > 64:
        units //= 2
    cuts = [u * total // units for u in range(units + 1)]
    result, offset = [], 0
    for t in tiles:
        groups = []
        for _ in range(row_groups):
            groups.append([(max(lo, offset) - offset, min(hi, offset + t) - offset)
                           for lo, hi in zip(cuts, cuts[1:]) if lo < offset + t and hi > offset])
            groups[-1] = [(lo * 128, hi * 128) for lo, hi in groups[-1]]
            offset += t
        result.append(groups)
    return result


def _attention(q, k, v, out, cu_seqlens_q, seq_lens, block_table, softmax_scale, q_scale, k_scale, v_scale, causal, window_left, sinks, lse):
    lengths = seq_lens.tolist()
    ptr = cu_seqlens_q.tolist()
    heads = q.shape[1]
    qlen = ptr[1] - ptr[0]
    partitions = _split_ranges(lengths, qlen, heads, torch.cuda.get_device_properties(q.device).multi_processor_count)
    page = k.shape[2]
    # NVFP4 decoded V is converted through FP16 into finite E4M3 for PV.
    # K retains its decoded NVFP4 value for blockscaled QK.
    v = v.to(torch.float16).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    for b, length in enumerate(lengths):
        query = q[ptr[b]:ptr[b + 1]].flatten(0, 1).float()
        result = out[ptr[b]:ptr[b + 1]].flatten(0, 1)
        result_lse = lse[ptr[b]:ptr[b + 1]].flatten() if lse is not None else None
        for group, start in enumerate(range(0, len(query), 128)):
            end = min(start + 128, len(query))
            rows = torch.arange(start, end, device=q.device)
            visible = length - qlen + rows // heads
            partials, statistics = [], []
            for split, (left, right) in enumerate(partitions[b][group]):
                maximum = torch.full((end - start,), -torch.inf, device=q.device)
                denominator = torch.zeros_like(maximum)
                accum = torch.zeros((end - start, v.shape[-1]), device=q.device)
                if sinks is not None and split == 0:
                    maximum = sinks[rows % heads].float() / math.log(2.0)
                    denominator.fill_(64.0)
                for begin in range(left, min(length, right), 128):
                    positions = torch.arange(begin, min(begin + 128, length, right), device=q.device)
                    pages = block_table[b, positions // page].long()
                    key = k[pages, 0, positions % page].float()
                    value = v[pages, 0, positions % page]
                    score = (query[start:end] @ key.T) * (softmax_scale / math.log(2.0))
                    score.masked_fill_(positions[None] > visible[:, None], -torch.inf)
                    updated = torch.maximum(maximum, score.amax(-1))
                    updated = torch.where(updated - maximum <= 2.0, maximum, updated)
                    safe = torch.where(torch.isfinite(updated), updated, 0.0)
                    rescale = (maximum - safe).exp2()
                    p = (score - safe[:, None] + 6.0).exp2()
                    stored_p = p.clamp(0.0, 448.0).to(torch.float8_e4m3fn).float()
                    accum = accum * rescale[:, None] + stored_p @ value
                    denominator = denominator * rescale + p.sum(-1)
                    maximum = updated
                live = denominator > 0
                partial = accum / torch.where(live, denominator, 1.0)[:, None]
                # Only multi-split paths materialize BF16 partial output.
                if len(partitions[b][group]) > 1:
                    partial = partial.to(torch.bfloat16).float()
                partials.append(partial)
                statistics.append(torch.where(live, (maximum + denominator.log2() - 6.0) * math.log(2.0), -torch.inf))
            logs = torch.stack(statistics)
            total_lse = torch.logsumexp(logs, dim=0)
            weights = (logs - total_lse[None]).exp()
            result[start:end].copy_((torch.stack(partials) * weights[..., None]).sum(0))
            if result_lse is not None:
                result_lse[start:end].copy_(total_lse)
    return out, lse


def pointer(lengths, device):
    return torch.tensor([0] + list(__import__('itertools').accumulate(lengths)), dtype=torch.int32, device=device)

def fp4(payload, scales):
    raw = payload.view(torch.uint8)
    code = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=payload.device)
    return lut[code] * scales.repeat_interleave(16, -1)

def normalized(x):
    kind = 'nvfp4_mla'
    q = fp4(x['query'], x['query_scale'].view(torch.float8_e4m3fn).float())
    k = fp4(x['kv_cache'], x['kv_scale'].view(torch.float8_e4m3fn).float())[:, None]
    b = len(x['seq_lens'])
    ptr = pointer([q.shape[0] // b] * b, q.device)
    return dict(q=q, k=k, v=k, out=x['out'], cu_seqlens_q=ptr, seq_lens=x['seq_lens'], block_table=x['block_tables'], softmax_scale=x['sm_scale'], q_scale=1.0, k_scale=1.0, v_scale=1.0, causal=True, window_left=-1, sinks=x['sinks'], lse=x['lse'])

def returned(x):
    return (x['out'], x['lse']) if x.get('return_lse') else x['out']

class Model(torch.nn.Module):

    def __init__(self, query, query_scale, kv_cache, kv_scale, block_tables, seq_lens, workspace_buffer, *, sm_scale, sinks=None, out=None, lse=None, return_lse=False, seq_lens_cpu=None, backend='cake'):
        super().__init__()
        self.prepared = locals().copy()
        self.prepared.pop('self')
        self.prepared.pop('__class__', None)

    def forward(self):
        x = self.prepared
        torch.backends.cuda.matmul.allow_tf32 = False
        z = normalized(x)
        target = z['out']
        target_lse = z['lse']
        z['out'] = torch.empty_like(target, dtype=torch.float32)
        if target_lse is not None:
            z['lse'] = torch.empty_like(target_lse)
        _attention(**z)
        target.copy_(z['out'])
        if target_lse is not None:
            target_lse.copy_(z['lse'])
        return returned(x)
