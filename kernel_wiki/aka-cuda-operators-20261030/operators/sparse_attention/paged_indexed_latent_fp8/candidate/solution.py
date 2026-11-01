"""Independent SM103 CUDA candidate. Edit adjacent CUDA files directly."""

import hashlib
import json
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load, CUDA_HOME
import subprocess

ROOT = Path(__file__).resolve().parent


def _build():
    major, minor = torch.cuda.get_device_capability()
    flags = CUDA_FLAGS + [
        f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"
    ]
    files = {
        p.name: p.read_text()
        for p in sorted(ROOT.iterdir())
        if p.suffix in (".cu", ".cuh")
    }
    key = hashlib.sha256(
        json.dumps(
            [
                files,
                flags,
                CPP_FLAGS,
                subprocess.check_output(
                    [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
                ),
                torch.__version__,
                torch.version.cuda,
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    cache = Path(
        os.environ.get("AKA_SF_BUILD_CACHE", Path.home() / ".cache/aka-candidates")
    )
    build = cache / (NAME + "_" + key)
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name=NAME + "_" + key,
        sources=[
            str(ROOT / name)
            for name in ("launch.cu", "kernel_m64.cu", "kernel_cluster.cu")
        ],
        extra_cuda_cflags=flags,
        extra_cflags=CPP_FLAGS,
        build_directory=str(build),
        verbose=False,
    )


CPP_FLAGS = ["-O3"]

CUDA_FLAGS = ["-O3", "--expt-relaxed-constexpr"]
NAME = "aka_paged_indexed_latent_fp8"


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self._ext = _build()
        self._call = self._ext.s4_tc_forward

    def forward(
        self,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        sparse_mla_top_k_lens,
        bmm1_scale,
        bmm2_scale,
        workspace_buffer,
        multi_ctas_kv_counter_buffer,
        out,
    ):
        if query.shape[-1] in (512, 576) and query.shape[-2] <= 128:
            if out is None:
                out = torch.empty(
                    (*query.shape[:-1], 512), dtype=torch.bfloat16, device=query.device
                )
            self._call(
                query,
                kv_cache,
                block_tables,
                seq_lens,
                sparse_mla_top_k_lens,
                out,
                float(bmm1_scale),
                float(bmm2_scale),
            )
            return out
        raise ValueError("S4 supports Q/K dimension 512 or 576 and at most 128 heads")

    __call__ = forward
