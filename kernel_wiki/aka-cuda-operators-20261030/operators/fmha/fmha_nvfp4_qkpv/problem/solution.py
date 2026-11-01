"""Original-library baseline for fmha_nvfp4_qkpv.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch

class Model(torch.nn.Module):
    def forward(self, query, key, value, cu_seqlens, *, pv_mode='fp8', softmax_scale=None, out=None, cu_seqlens_host=None, backend='cake'):
        from flashinfer.prefill import minimax_h3_varlen_nvfp4_attention
        return minimax_h3_varlen_nvfp4_attention(query, key, value, cu_seqlens,
            pv_mode=pv_mode, softmax_scale=softmax_scale, out=out,
            cu_seqlens_host=cu_seqlens_host, backend=backend)
