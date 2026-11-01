"""Original-library baseline for fp8_mxfp4_gemm.

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
    def forward(self, a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, alpha=None):
        return self.native.fp8_fp4_gemm_nt(a,b,d,c,recipe,recipe_a,recipe_b,compiled_dims,disable_ue8m0_cast,alpha)
