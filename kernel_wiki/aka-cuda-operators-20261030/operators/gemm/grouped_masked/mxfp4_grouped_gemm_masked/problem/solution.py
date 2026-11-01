"""Original-library baseline for mxfp4_grouped_gemm_masked.

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
    def forward(self, a, b, d, masked_m, expected_m, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False):
        return self.native.m_grouped_fp8_fp4_gemm_nt_masked(a,b,d,masked_m,expected_m,recipe,recipe_a,recipe_b,compiled_dims,disable_ue8m0_cast)
