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

NAME = "aka_paged_latent_v5"
CUDA_FLAGS = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-Xptxas", "-v"]
CPP_FLAGS = []
_EXT = None

class Model(torch.nn.Module):
    """Production forward ABI for paged_latent_mxfp8_nvfp4."""

    def __init__(self):
        super().__init__()
        self.implementation = "independent CUDA TCGEN05/mma.sync sparse decode (no library)"
        global _EXT
        if _EXT is None:
            _EXT = _build()
        self._fwd = _EXT.forward

    def forward(self, q, k_cache, indices, extra_k_cache, extra_indices, sm_scale, attn_sink, out):
        # out is written in place by the kernel and returned by identity, exactly
        # like the frozen reference; lse is the fresh [B,H,S] strided transpose.
        lse = self._fwd(q, k_cache, indices, extra_k_cache, extra_indices, sm_scale, attn_sink, out)
        return (out, lse)
