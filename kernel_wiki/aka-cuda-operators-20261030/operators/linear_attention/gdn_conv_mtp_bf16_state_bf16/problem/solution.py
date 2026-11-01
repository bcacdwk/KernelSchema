"""Original-library baseline for gdn_conv_mtp_bf16_state_bf16.

Dependency: vLLM c723a831a81cb4ff89ea6d61b2d15109306ed1bc; zero-copy binding to the selected convolution cache layout.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
from types import SimpleNamespace
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

class Model(torch.nn.Module):
    def forward(self, mixed_qkv, conv_state, conv_weight, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state, output_gate, norm_weight, out=None, scale=128 ** (-0.5), norm_eps=1e-05, output_gate_activation='silu'):
        if out is None:
            out = torch.empty_like(output_gate)
        layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
        torch.nn.Module.__init__(layer)
        layer.kv_cache = (conv_state if is_conv_state_dim_first() else conv_state.transpose(-1, -2), state)
        layer.conv1d = SimpleNamespace(weight=conv_weight.unsqueeze(1), bias=None)
        layer.activation = 'silu'
        layer.A_log = A_log
        layer.dt_bias = dt_bias
        assert scale == 128 ** (-0.5), 'Selected Qwen caller fixes K128 scale'
        layer.head_k_dim = 128
        layer.layer_norm_epsilon = norm_eps
        layer.norm = SimpleNamespace(weight=norm_weight, activation=output_gate_activation)
        meta = SimpleNamespace(spec_state_indices_tensor=state_indices, spec_query_start_loc=cu_seqlens, num_accepted_tokens=num_accepted_tokens, num_spec_decodes=state_indices.shape[0], num_actual_tokens=mixed_qkv.shape[0])
        layer._forward_core_decode_spec_fused_norm(mixed_qkv, b, a, output_gate, out, meta)
        return out
