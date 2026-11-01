"""Original-library baseline for dual_cache_latent_bf16.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
from __future__ import annotations
import torch
from typing import Literal, Optional, Union
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

class Model(torch.nn.Module):

    def forward(self, query: torch.Tensor, swa_kv_cache: torch.Tensor, workspace_buffer: torch.Tensor, sparse_indices: torch.Tensor, compressed_kv_cache: Optional[torch.Tensor]=None, sparse_topk_lens: Optional[torch.Tensor]=None, seq_lens: Optional[torch.Tensor]=None, out: Optional[torch.Tensor]=None, bmm1_scale: Union[float, torch.Tensor]=1.0, bmm2_scale: Union[float, torch.Tensor]=1.0, sinks: Optional[torch.Tensor]=None, kv_layout: Literal['HND', 'NHD']='HND', cum_seq_lens_q: Optional[torch.Tensor]=None, max_q_len: Optional[int]=None, enable_pdl: bool | None=None, swa_topk_lens: Optional[torch.Tensor]=None, extra_sparse_indices: Optional[torch.Tensor]=None, extra_sparse_topk_lens: Optional[torch.Tensor]=None):
        return trtllm_batch_decode_sparse_mla_dsv4(query=query, swa_kv_cache=swa_kv_cache, workspace_buffer=workspace_buffer, sparse_indices=sparse_indices, compressed_kv_cache=compressed_kv_cache, sparse_topk_lens=sparse_topk_lens, seq_lens=seq_lens, out=out, bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale, sinks=sinks, kv_layout=kv_layout, cum_seq_lens_q=cum_seq_lens_q, max_q_len=max_q_len, enable_pdl=enable_pdl, swa_topk_lens=swa_topk_lens, extra_sparse_indices=extra_sparse_indices, extra_sparse_topk_lens=extra_sparse_topk_lens)
