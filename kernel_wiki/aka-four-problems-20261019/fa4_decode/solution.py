from __future__ import annotations

import torch
import torch.nn as nn
from flashinfer.decode import trtllm_batch_decode_with_kv_cache


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
        trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=window_left,
            sinks=sinks,
            o_sf_scale=o_sf_scale,
            out=out,
            q_len_per_req=q_len_per_req,
        )
        return out
