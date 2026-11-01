"""Original-library baseline for fp8_scaled_grouped_gemm.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
class Model(torch.nn.Module):
    def forward(self, a, b, a_scale, b_scale, m_indptr, scale_granularity_mnk=(1,128,128), scale_major_mode='MN', mma_sm=1, out=None, out_dtype=None, backend='trtllm', segment_alignment=1):
        import flashinfer.gemm
        return flashinfer.gemm.group_gemm_fp8_nt_groupwise(a,b,a_scale,b_scale,m_indptr,scale_granularity_mnk,scale_major_mode,mma_sm,out,out_dtype,backend,segment_alignment)
