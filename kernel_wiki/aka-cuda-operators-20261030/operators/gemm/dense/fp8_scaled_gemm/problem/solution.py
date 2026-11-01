"""Original-library baseline for fp8_scaled_gemm.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b, a_scale, b_scale, scale_major_mode=None, mma_sm=1, scale_granularity_mnk=(1,128,128), out=None, out_dtype=None, backend='cutlass'):
        import flashinfer.gemm
        return flashinfer.gemm.gemm_fp8_nt_groupwise(a,b,a_scale,b_scale,scale_major_mode,mma_sm,scale_granularity_mnk,out,out_dtype,backend)
