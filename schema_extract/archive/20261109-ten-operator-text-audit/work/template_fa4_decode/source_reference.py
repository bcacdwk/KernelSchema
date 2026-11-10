from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        window_left: int,
        sinks: torch.Tensor | None,
        o_sf_scale: torch.Tensor | None,
        out: torch.Tensor,
        q_len_per_req: int,
    ) -> torch.Tensor:
        del workspace_buffer
        if window_left != -1 or sinks is not None or o_sf_scale is not None:
            raise ValueError("captured Qwen3.8-Max path uses full causal attention")
        if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
            raise ValueError("expected HND KV cache")
        if max_seq_len != int(seq_lens.max().item()):
            raise ValueError("max_seq_len and seq_lens disagree")
        batch_size = seq_lens.numel()
        if query.shape[0] != batch_size * q_len_per_req:
            raise ValueError("query rows and q_len_per_req disagree")

        page_size = kv_cache.shape[-2]
        num_q_heads = query.shape[1]
        num_kv_heads = kv_cache.shape[2]
        if num_q_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        q_heads_per_kv = num_q_heads // num_kv_heads

        for request_index, seq_len in enumerate(seq_lens.tolist()):
            page_count = (seq_len + page_size - 1) // page_size
            page_ids = block_tables[request_index, :page_count].to(dtype=torch.long)
            pages = kv_cache.index_select(0, page_ids)
            keys = (
                pages[:, 0]
                .permute(0, 2, 1, 3)
                .reshape(page_count * page_size, num_kv_heads, -1)[:seq_len]
            )
            values = (
                pages[:, 1]
                .permute(0, 2, 1, 3)
                .reshape(page_count * page_size, num_kv_heads, -1)[:seq_len]
            )
            query_start = request_index * q_len_per_req
            query_end = query_start + q_len_per_req
            query_position_start = seq_len - q_len_per_req

            for kv_head in range(num_kv_heads):
                head_start = kv_head * q_heads_per_kv
                head_end = head_start + q_heads_per_kv
                key = keys[:, kv_head].float()
                value = values[:, kv_head].float()
                q_request = query[
                    query_start:query_end,
                    head_start:head_end,
                ].float()
                q_by_head = q_request.permute(1, 0, 2).contiguous()
                scores = torch.matmul(q_by_head, key.transpose(0, 1))
                scores.mul_(bmm1_scale)
                for query_offset in range(q_len_per_req - 1):
                    first_future_key = query_position_start + query_offset + 1
                    scores[:, query_offset, first_future_key:] = float("-inf")
                probabilities = torch.softmax(scores, dim=-1)
                output = torch.matmul(
                    probabilities.reshape(-1, seq_len),
                    value,
                ).reshape(q_heads_per_kv, q_len_per_req, -1)
                output = output.permute(1, 0, 2).contiguous() * bmm2_scale
                out[
                    query_start:query_end,
                    head_start:head_end,
                ].copy_(output.to(dtype=out.dtype))
        return out
