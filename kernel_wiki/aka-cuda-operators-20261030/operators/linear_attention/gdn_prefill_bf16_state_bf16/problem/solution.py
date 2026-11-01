"""Original-library baseline for gdn_prefill_bf16_state_bf16.

Dependency: vLLM c723a831a81cb4ff89ea6d61b2d15109306ed1bc + FlashInfer d849eca46875; native forward_cuda chooses its own CP policy.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import ChunkGatedDeltaRule

class Model(torch.nn.Module):
    def forward(self, q, k, v, g, beta, initial_state, output_final_state, cu_seqlens=None, chunk_indices=None, chunk_offsets=None, use_qk_l2norm_in_kernel=True, core_attn_out=None):
        return ChunkGatedDeltaRule.forward_cuda(self, q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, core_attn_out=core_attn_out)
