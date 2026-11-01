"""MXFP8 block-scaled GEMM candidate for L20D (sm_103-class, Blackwell).

Independent implementation: all numerical work runs in candidate/mxfp8_kernels.cu,
a multi-stage pipelined kernel built on the 5th-gen tensor core block-scaled
UMMA (tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale, UE8M0 scales per
32 K-elements selected via instruction-descriptor sf_id bytes), TMA operand
loads into 128B-swizzled shared memory, fused exact fp32->UE8M0 scale packing
into the UTCCP-interleaved TMEM scale layout, fp32 accumulation in tensor
memory, and a fused alpha/C epilogue with guarded vectorized stores. Python only compiles/
loads the extension and binds the frozen ABI. Mechanism conventions were studied
from the public PTX ISA and NVIDIA DeepGEMM e1f418c2 / CUTLASS as read-only
references; the code is written independently here. No GEMM library, DSL or Torch
numerical fallback is used.

ABI (frozen): forward(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
compiled_dims='nk', disable_ue8m0_cast=False, alpha=None) -> None; writes D in
place: D = alpha * dequant(A) @ dequant(B)^T + C (C optional; may alias D).
"""
import hashlib
import os
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent / "mxfp8_kernels.cu"
_EXT = None


def _build():
    global _EXT
    if _EXT is not None:
        return _EXT
    # tcgen05 requires the arch-specific target; the device reports cc 10.3.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.3a")
    from torch.utils.cpp_extension import load

    tag = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:10]
    _EXT = load(
        name=f"mxfp8_l20d_{tag}",
        sources=[str(_SRC)],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-Xptxas", "-v"],
        extra_ldflags=["-lcuda"],
        verbose=True,
    )
    return _EXT


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ext = _build()

    def forward(self, a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                compiled_dims="nk", disable_ue8m0_cast=False, alpha=None):
        # Frozen domain: recipe (1,1,32) -> per-row MN granularity 1, UE8M0
        # power-of-two scales at 32-element K granularity (hardware 1X path).
        if recipe is not None and tuple(recipe) != (1, 1, 32):
            raise ValueError(f"unsupported scaling recipe {recipe!r}")
        if recipe_a is not None or recipe_b is not None:
            raise ValueError("per-operand recipes are not part of the frozen domain")
        if disable_ue8m0_cast:
            raise ValueError("disable_ue8m0_cast=True conflicts with the UE8M0 hardware path")
        aq, sa = a
        bq, sb = b
        self.ext.mxfp8_gemm(aq, sa, bq, sb, d, c,
                            1.0 if alpha is None else float(alpha))
        return None
