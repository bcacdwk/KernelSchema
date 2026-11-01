"""Original-library baseline for gdn_decode_core_bf16_state_fp32.

Dependency: vLLM c723a831a81cb4ff89ea6d61b2d15109306ed1bc; zero-copy binding to the selected convolution cache layout.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
from types import SimpleNamespace
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention, qwen_gdn_attention_core_fused_norm_packed
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.forward_context import ForwardContext, override_forward_context

class Model(torch.nn.Module):
    def forward(self, qkvz, ba, conv_weight, conv_state, A_log, dt_bias, state, norm_weight, out, metadata):
        m = GDNAttentionMetadata(**metadata)
        assert m.num_prefills == m.num_spec_decodes == 0 and m.num_decodes == m.num_actual_tokens > 0 and (m.spec_sequence_masks is None)
        hv = state.shape[1]
        h = (conv_weight.shape[0] - hv * 128) // 256
        layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
        torch.nn.Module.__init__(layer)
        layer.prefix = 'gdn'
        layer.tp_size = 1
        layer.tp_rank = 0
        layer.key_dim = h * 128
        layer.value_dim = hv * 128
        layer.num_k_heads = h
        layer.num_v_heads = hv
        layer.head_k_dim = layer.head_v_dim = 128
        layer.disable_tp_for_ba_proj = False
        layer.enable_packed_recurrent_decode = True
        layer.kv_cache = (conv_state if is_conv_state_dim_first() else conv_state.transpose(-1, -2), state)
        layer.conv1d = SimpleNamespace(weight=conv_weight.unsqueeze(1), bias=None)
        layer.activation = 'silu'
        layer.A_log = A_log
        layer.dt_bias = dt_bias
        layer.norm = SimpleNamespace(weight=norm_weight, bias=None, eps=1e-06, group_size=None, norm_before_gate=True, activation='sigmoid')
        context = ForwardContext(no_compile_layers={'gdn': layer}, attn_metadata={'gdn': m}, slot_mapping={})
        with override_forward_context(context):
            return qwen_gdn_attention_core_fused_norm_packed(qkvz, ba, out, 'gdn')
