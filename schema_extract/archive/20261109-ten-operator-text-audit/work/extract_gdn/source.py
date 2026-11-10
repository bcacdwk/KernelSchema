"""GDN prefill (bf16 in / bf16 state) candidate for L20D — pure CUDA C++/PTX.

Independent implementation of the chunked gated delta rule (no GEMM/attention
library calls, no Triton/CuTe, no Torch numerical fallback). Python only
compiles/loads the extension, allocates tensors and binds the ABI:

    Model.forward(q, k, v, g, beta, initial_state, output_final_state,
                  cu_seqlens, chunk_indices, chunk_offsets,
                  use_qk_l2norm_in_kernel, core_attn_out)
      -> (out bf16 [1,T,HV,128], final_state fp32 [N,HV,128,128] | None)

Numerical work lives in gdn_kernel.cu / gdn_common.cuh next to this file.
"""
import os

import torch
from torch.utils.cpp_extension import load

_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCES = [os.path.join(_DIR, "gdn_kernel.cu")]
_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device required")
        cap = torch.cuda.get_device_capability(0)
        arch = f"{cap[0]}{cap[1]}"
        _ext = load(
            name="gdn_l20d_v2e",
            sources=_SOURCES,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--expt-relaxed-constexpr",
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
            ],
            extra_ldflags=["-lcuda"],   # V5.13b: cuTensorMapEncodeTiled (driver API)
            verbose=False,
        )
    return _ext


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if torch.cuda.is_available():
            _get_ext()  # compile at construction (outside timed paths)

    def forward(self, q, k, v, g, beta, initial_state, output_final_state,
                cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
                use_qk_l2norm_in_kernel=True, core_attn_out=None):
        ext = _get_ext()
        if cu_seqlens is None:
            # Mirror reference fallback: single request covering all tokens.
            T = int(q.shape[1])
            dev = q.device
            cu_seqlens = torch.tensor([0, T], device=dev, dtype=torch.int32)
            nchunk = (T + 63) // 64
            chunk_offsets = torch.tensor([0, nchunk], device=dev, dtype=torch.int32)
            chunk_indices = torch.stack(
                [torch.zeros(nchunk, dtype=torch.int32, device=dev),
                 torch.arange(nchunk, dtype=torch.int32, device=dev)], dim=1)
        out, final = ext.gdn_forward(
            q, k, v, g, beta, initial_state, cu_seqlens, chunk_indices, chunk_offsets,
            bool(use_qk_l2norm_in_kernel), bool(output_final_state), core_attn_out)
        return out, final
