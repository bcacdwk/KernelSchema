"""Original-library baseline for fmha_bf16.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch

class Model(torch.nn.Module):
    def forward(self, query, kv_cache, workspace_buffer, block_tables, seq_lens, max_q_len, max_kv_len, bmm1_scale, bmm2_scale, batch_size, cum_seq_lens_q, cum_seq_lens_kv, window_left=-1, out=None, out_dtype=None, o_sf_scale=None, o_sf_vec_size=None, kv_layout='HND', enable_pdl=None, sinks=None, kv_cache_sf=None, skip_softmax_threshold_scale_factor=None, uses_shared_paged_kv_idx=True, causal=True, lse=None, return_lse=False, multi_ctas_kv_counter_buffer=None, use_fp16_softmax=None, uses_spcompress=None, backend='trtllm-gen'):
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache
        arguments=locals().copy();arguments.pop("self");arguments.pop("trtllm_batch_context_with_kv_cache",None)
        return trtllm_batch_context_with_kv_cache(**arguments)
