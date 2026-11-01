"""Original-library baseline for bf16_grouped_gemm.

Dependency: DeepGEMM e1f418c2a4f20818221f6b0e578b4c2f634d4c3f; raw _C ABI (the high-level MegaMoE wrapper has a different signature).
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from deep_gemm import _C
        self.native = _C
        self.native.use_deterministic_algorithms(True)
    def forward(self, a, b, d, grouped_layout, compiled_dims='nk', use_psum_layout=False, ensure_zero_padding=True, expected_m_for_psum_layout=None):
        return self.native.m_grouped_bf16_gemm_nt_contiguous(a,b,d,grouped_layout,compiled_dims,use_psum_layout,ensure_zero_padding,expected_m_for_psum_layout)
