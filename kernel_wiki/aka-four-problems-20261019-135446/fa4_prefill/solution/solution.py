"""Task-ABI adapter for the extracted upstream FA4 hd256 2CTA kernel."""

from __future__ import annotations

import torch
import torch.nn as nn

import flash_attn.cute.interface as fa4_interface

try:
    from .kernel import BlackwellFusedMultiHeadAttentionForward
except ImportError:
    from kernel import BlackwellFusedMultiHeadAttentionForward


# Make FA4's preparation/compilation path instantiate the extracted local kernel.
fa4_interface.BlackwellFusedMultiHeadAttentionForward = (
    BlackwellFusedMultiHeadAttentionForward
)


class Model(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        batch_size: int,
        cum_seq_lens_q: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        window_left: int,
        sinks: torch.Tensor | None,
        o_sf_scale: torch.Tensor | None,
        out: torch.Tensor,
    ) -> torch.Tensor:
        del workspace_buffer, cum_seq_lens_kv
        if batch_size != seq_lens.numel():
            raise ValueError("batch_size and seq_lens disagree")
        if bmm2_scale != 1.0:
            raise ValueError("FA4 adapter currently requires bmm2_scale=1")
        if o_sf_scale is not None:
            raise ValueError("FA4 adapter does not implement output scale factors")

        page_size = kv_cache.shape[-2]
        max_pages = (max_kv_len + page_size - 1) // page_size
        page_table = block_tables[:, :max_pages].contiguous()
        key_cache = kv_cache[:, 0].transpose(1, 2)
        value_cache = kv_cache[:, 1].transpose(1, 2)
        window = None if window_left == -1 else window_left

        fa4_interface._flash_attn_fwd(
            query,
            key_cache,
            value_cache,
            cu_seqlens_q=cum_seq_lens_q,
            seqused_k=seq_lens,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_kv_len,
            page_table=page_table,
            softmax_scale=bmm1_scale,
            causal=True,
            window_size_left=window,
            window_size_right=0 if window is not None else None,
            learnable_sink=sinks,
            num_splits=1,
            pack_gqa=False,
            out=out,
        )
        return out
