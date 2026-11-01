"""Original-library baseline for bf16_nvfp4_gemm.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b, b_descale, alpha=None, *, backend, out_dtype=None, out=None, block_size=16, enable_pdl=True):
        import flashinfer
        return flashinfer.mm_bf16_fp4(a,b,b_descale,alpha,backend=backend,out_dtype=out_dtype,out=out,block_size=block_size,enable_pdl=enable_pdl)
