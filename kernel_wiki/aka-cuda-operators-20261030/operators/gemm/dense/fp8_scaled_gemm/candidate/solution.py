"""Independent SM103 CUDA candidate. Edit adjacent CUDA files directly."""
import hashlib
import json
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent

def _build():
    major, minor = torch.cuda.get_device_capability()
    flags = CUDA_FLAGS + [f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"]
    files = {p.name: p.read_text() for p in sorted(ROOT.iterdir())
             if p.suffix in (".cu", ".cuh")}
    key = hashlib.sha256(json.dumps(
        [files, flags, torch.__version__, torch.version.cuda], sort_keys=True
    ).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("AKA_SF_BUILD_CACHE", Path.home()/".cache/aka-candidates"))
    build = cache / (NAME + "_" + key)
    build.mkdir(parents=True, exist_ok=True)
    return load(name=NAME + "_" + key, sources=[str(ROOT/"launch.cu")],
                extra_cuda_cflags=flags, extra_cflags=CPP_FLAGS,
                build_directory=str(build), verbose=False)

NAME = "aka_fp8_scaled_gemm_v5"
CUDA_FLAGS = ["-O3", "--expt-relaxed-constexpr", "-DFIC_PURE_STANDALONE=1"]
CPP_FLAGS = ["-O3"]

class Model(torch.nn.Module):
    """Blockwise-scaled FP8 GEMM: out = dequant(a, a_scale) @ dequant(b, b_scale)^T.

    The signature is the problem's own ABI, kept argument for argument -- including the
    options this implementation does not vary -- so the evaluator, the reference and the
    original-library baseline are all called identically.
    """

    def __init__(self):
        super().__init__()
        # Compilation happens here, in the standard untimed preparation stage, and does
        # no GPU work of its own.  Import time stays free of both.
        self._ext = _build()

    def forward(self, a, b, a_scale, b_scale, scale_major_mode=None, mma_sm=1,
                scale_granularity_mnk=(1, 128, 128), out=None, out_dtype=None,
                backend='cutlass'):
        gran_m, gran_n, gran_k = scale_granularity_mnk
        if out is None:
            # The problem always hands us a caller-owned buffer; this branch only mirrors
            # what the production entry does when a caller omits it.
            dtype = out_dtype if out_dtype is not None else torch.bfloat16
            out = torch.empty((a.shape[0], b.shape[0]), dtype=dtype, device=a.device)
        self._ext.fp8_scaled_gemm(a, b, a_scale, b_scale, out,
                                  'K' if scale_major_mode is None else str(scale_major_mode),
                                  int(mma_sm), int(gran_m), int(gran_n), int(gran_k))
        # Return the caller's buffer itself, not a copy: metadata declares `out` as an
        # aliased, caller-owned output and lists it under mutates_inputs.
        return out
