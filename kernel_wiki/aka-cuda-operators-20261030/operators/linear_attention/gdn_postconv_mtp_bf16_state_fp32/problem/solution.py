"""Original-library baseline for gdn_postconv_mtp_bf16_state_fp32.

Dependency: vLLM c723a831a81cb4ff89ea6d61b2d15109306ed1bc; matching compiled vllm._C extension required.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
from vllm._custom_ops import fused_gdn_decode_post_conv_mtp

class Model(torch.nn.Module):
    def forward(self, mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state, output_gate, norm_weight, out=None, scale=128 ** (-0.5), norm_eps=1e-05, output_gate_activation='silu'):
        return fused_gdn_decode_post_conv_mtp(mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, state_indices=state_indices, cu_seqlens=cu_seqlens, num_accepted_tokens=num_accepted_tokens, state=state, output_gate=output_gate, norm_weight=norm_weight, out=out, scale=scale, norm_eps=norm_eps, output_gate_activation=output_gate_activation)
