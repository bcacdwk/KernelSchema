"""Original-library baseline for mla_expanded_bf16.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch

class Model(torch.nn.Module):
    def forward(self, query, key, value, workspace_buffer, seq_lens, max_q_len, max_kv_len, bmm1_scale, bmm2_scale, o_sf_scale, batch_size, window_left, cum_seq_lens_q, cum_seq_lens_kv, enable_pdl, is_causal, return_lse, attention_sinks=None, skip_softmax_threshold_scale_factor=None, out=None, lse=None, sage_attn_sfs=(None, None, None, None), num_elts_per_sage_attn_blk=(0, 0, 0, 0), backend='trtllm-gen', q_seq_lens_cpu=None, kv_seq_lens_cpu=None, use_fp16_softmax=None, uses_spcompress=None, skip_all_rows_active_check=True):
        from flashinfer.prefill import trtllm_ragged_attention_deepseek
        arguments=locals().copy();arguments.pop("self");arguments.pop("trtllm_ragged_attention_deepseek",None)
        return trtllm_ragged_attention_deepseek(**arguments)
