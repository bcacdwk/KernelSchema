"""mxfp4_gemm candidate — independent CUDA/C++ implementation (no library GEMM).

D = alpha * A @ B^T + C for mxfp4 operands (e2m1 payload, fp32 power-of-two
group scales every 32 elements), bf16 output. Python only allocates scratch
for the fallback path, compiles/loads the extension and binds the ABI; all
numerical work runs in candidate/kernel.cu + candidate/tcgen05_kernels.cuh on
the device (native tcgen05 block-scaled UMMA on sm_100+, with an in-C++
CUDA-graph cache removing per-call launch overhead).
"""
import os
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent
_EXT = None


def _build():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device required")
        maj, min_ = torch.cuda.get_device_capability()
        arch = f"{maj}.{min_}" + ("a" if (maj, min_) >= (9, 0) else "")
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        _EXT = load(
            name="mxfp4_gemm_candidate",
            sources=[str(_DIR / "kernel.cu")],
            extra_cuda_cflags=["-O3", "-std=c++17", "--ptxas-options=-v"],
            extra_cflags=["-O3"],
            extra_ldflags=["-lcuda"],
            verbose=False,
        )
    return _EXT


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = _build()
        self._fwd = getattr(self.ext, "mxfp4_forward", None)
        self._native = torch.cuda.get_device_capability()[0] >= 10

    def forward(self, a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                compiled_dims='nk', disable_ue8m0_cast=False, alpha=None):
        # Phase-2 step-5: the whole hot path (tuple unpack, contiguity fixups,
        # dtype/shape checks, native-vs-fallback gate, graph-cached launch)
        # lives in C++; python overhead inside the engine's timed span drops
        # to a single pybind call.
        if self._fwd is not None:
            self._fwd(a, b, d, c, alpha)
            return None
        # Legacy fallback (ext without mxfp4_forward): previous python path.
        aq, asf = a
        bq, bsf = b
        if not (aq.is_contiguous() and asf.is_contiguous() and
                bq.is_contiguous() and bsf.is_contiguous()):
            aq = aq.contiguous(); asf = asf.contiguous()
            bq = bq.contiguous(); bsf = bsf.contiguous()
        m, k2 = aq.shape
        k = k2 * 2
        n = bq.shape[0]
        has_c = c is not None
        ct = c if has_c else d
        al = alpha if alpha is not None else 1.0
        abuf = torch.empty(m * k, dtype=torch.bfloat16, device=d.device)
        bbuf = torch.empty(n * k, dtype=torch.bfloat16, device=d.device)
        self.ext.mxfp4_run(aq, asf, abuf, bq, bsf, bbuf, d, ct, has_c, al)
        return None
