from __future__ import annotations

import torch
import torch.nn as nn
from flashinfer.prefill import trtllm_batch_context_with_kv_cache


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
        trtllm_batch_context_with_kv_cache(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=batch_size,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            window_left=window_left,
            sinks=sinks,
            o_sf_scale=o_sf_scale,
            out=out,
        )
        return out
