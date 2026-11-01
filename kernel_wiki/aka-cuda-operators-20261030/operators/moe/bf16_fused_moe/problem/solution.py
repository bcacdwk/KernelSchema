"""Original-library baseline for bf16_fused_moe.

Dependency: DeepGEMM e1f418c2a4f20818221f6b0e578b4c2f634d4c3f; raw _C ABI (the high-level MegaMoE wrapper has a different signature).
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from deep_gemm import _C
        self.native = _C
        self.native.set_num_sms(max(2,self.native.get_num_sms()-2))
    def forward(self, y, l1_weights, l2_weights, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats, sym_buffer, sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_experts, num_topk, activation, activation_clamp_opt, fast_math, activation_alpha, activation_beta):
        return self.native.bf16_mega_moe(y, l1_weights, l2_weights, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats, sym_buffer, sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_experts, num_topk, activation, activation_clamp_opt, fast_math, activation_alpha, activation_beta)
