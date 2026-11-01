"""Original-library baseline for fmha_bf16_fp8.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch

class Model(torch.nn.Module):
    def forward(self, query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len, bmm1_scale=1.0, bmm2_scale=1.0, window_left=-1, out=None, out_dtype=None, o_sf_scale=None, o_sf_vec_size=None, sinks=None, kv_layout='HND', enable_pdl=None, backend='auto', q_len_per_req=1, o_scale=1.0, mask=None, max_q_len=None, cum_seq_lens_q=None, skip_softmax_threshold_scale_factor=None, kv_cache_sf=None, uses_shared_paged_kv_idx=True, lse=None, return_lse=False, bmm1_scale_log2=None, multi_ctas_kv_counter_buffer=None, enable_block_sparse_attention=False, cp_world=1, cp_rank=0, causal_seqlens_kv_global=None, bf16q_fp8kv_transform_mode=None, request_order=None, request_order_plan=None):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        arguments=locals().copy();arguments.pop("self");arguments.pop("trtllm_batch_decode_with_kv_cache",None)
        return trtllm_batch_decode_with_kv_cache(**arguments)
