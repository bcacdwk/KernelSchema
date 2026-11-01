"""Original-library baseline for mla_absorbed_fp8.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch

class Model(torch.nn.Module):
    def forward(self, query, kv_cache, workspace_buffer, qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim, block_tables, seq_lens, max_seq_len, sparse_mla_top_k=0, out=None, bmm1_scale=1.0, bmm2_scale=1.0, sinks=None, skip_softmax_threshold_scale_factor=None, enable_pdl=None, backend='auto', is_var_seq=True, uses_shared_paged_kv_idx=True, lse=None, return_lse=False, cute_dsl_impl='auto', kv_scale_format='auto', cum_seq_lens_q=None, max_q_len=None, multi_ctas_kv_counter_buffer=None, sparse_mla_top_k_lens=None, enable_dcp=False, cp_world=1, cp_rank=0, causal_seqlens_kv_global=None, use_fp16_softmax=None, return_lse_base=None):
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
        arguments=locals().copy();arguments.pop("self");arguments.pop("trtllm_batch_decode_with_kv_cache_mla",None)
        return trtllm_batch_decode_with_kv_cache_mla(**arguments)
