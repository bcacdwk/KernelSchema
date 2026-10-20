from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Pure PyTorch semantics for Qwen3.8 paged CSA sparse GQA."""

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        logical_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_gate: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del use_prefill_config
        if out is None:
            out = torch.empty_like(q)
        scale = q.shape[-1] ** -0.5
        group_size = q.shape[1] // k_cache.shape[2]
        selection_width = logical_indices.shape[1] - 1
        result = torch.empty_like(q)

        # Bound peak temporary memory while preserving the exact paged ABI.
        for begin in range(0, q.shape[0], 32):
            end = min(begin + 32, q.shape[0])
            packed = logical_indices[begin:end]
            indices = packed[:, :selection_width].long()
            valid_count = packed[:, selection_width].long()
            columns = torch.arange(
                selection_width, dtype=torch.int64, device=q.device
            )
            requests = token_to_req[begin:end].long()
            logical_pages = indices.clamp_min(0) // k_cache.shape[1]
            page_offsets = indices.clamp_min(0) % k_cache.shape[1]
            safe_requests = requests[:, None].clamp(0, block_table.shape[0] - 1)
            safe_pages = logical_pages.clamp(0, block_table.shape[1] - 1)
            physical_pages = block_table[safe_requests, safe_pages].long()
            valid = (
                (columns[None, :] < valid_count[:, None])
                & (indices >= 0)
                & (requests[:, None] >= 0)
                & (requests[:, None] < block_table.shape[0])
                & (logical_pages < block_table.shape[1])
                & (physical_pages >= 0)
                & (physical_pages < k_cache.shape[0])
            )
            safe_physical = physical_pages.clamp(0, k_cache.shape[0] - 1)
            keys = k_cache[safe_physical, page_offsets]
            values = v_cache[safe_physical, page_offsets]
            query = q[begin:end].reshape(end - begin, k_cache.shape[2], group_size, q.shape[2])
            scores = torch.einsum("rvgd,rtvd->rvgt", query.float(), keys.float())
            scores.mul_(float(scale)).masked_fill_(~valid[:, None, None, :], -torch.inf)
            probabilities = torch.softmax(scores, dim=-1)
            probabilities = torch.nan_to_num(probabilities, nan=0.0)
            block_out = torch.einsum(
                "rvgt,rtvd->rvgd", probabilities, values.float()
            ).reshape(end - begin, q.shape[1], q.shape[2])
            rounded = block_out.to(q.dtype)
            result[begin:end] = (
                rounded.float() * torch.sigmoid(output_gate[begin:end].float())
            ).to(q.dtype)

        out.copy_(result)
        return out
