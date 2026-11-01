"""Original-library baseline for fmha_fp8_per_token_head.

Dependency: vLLM c723a831a81cb4ff89ea6d61b2d15109306ed1bc.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
from vllm.v1.attention.ops.triton_unified_attention import KVQuantMode

class Model(torch.nn.Module):
    def forward(self, q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal, window_size, block_table, softcap, q_descale, k_descale, v_descale, seq_threshold_3D=None, num_par_softmax_segments=None, softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None, alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None, mm_prefix_range=None, rswa_prefix_lens=None, rswa_window=None, use_alibi_sqrt=False, kv_quant_mode=KVQuantMode.NONE, k_scale_cache=None, v_scale_cache=None, chunk_lookback=-1, use_td=False, mm_prefix_clamp_sliding_window=False):
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention
        kv_quant_mode = KVQuantMode(kv_quant_mode)
        arguments=locals().copy();arguments.pop("self");arguments.pop("unified_attention",None)
        return unified_attention(**arguments)
