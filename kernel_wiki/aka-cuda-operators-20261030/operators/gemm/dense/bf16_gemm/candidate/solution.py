"""BF16 GEMM: directly editable CUDA, tuned configurations plus dynamic compilation."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import torch
from torch.utils.cpp_extension import CUDA_HOME, load

ROOT = Path(__file__).resolve().parent
CUDA_FLAGS = [
    "-std=c++20",
    "-O3",
    "--compiler-options=-O3",
    "--ptxas-options=--register-usage-level=10",
    "--compiler-options=-fPIC",
    "--compiler-options=-fconcepts",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--diag-suppress=39,161,174,177,186,940",
    "--compiler-options=-Wno-deprecated-declarations,-Wno-abi",
    "--compiler-options=-g0",
    "--compiler-options=-Wno-psabi",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]
CPP_FLAGS = ["-O3", "-std=c++20"]


def _build():
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}f"
    flags = CUDA_FLAGS + [f"-gencode=arch=compute_{major}{minor}f,code={arch}"]
    compiler = str(Path(CUDA_HOME) / "bin/nvcc")
    version = subprocess.check_output([compiler, "--version"], text=True)
    sources = {
        p.name: p.read_text()
        for p in sorted(ROOT.iterdir())
        if p.suffix in (".cu", ".cuh")
    }
    key = hashlib.sha256(
        json.dumps(
            [sources, flags, CPP_FLAGS, version, torch.__version__, torch.version.cuda],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache = Path(
        os.environ.get("AKA_SF_BUILD_CACHE", Path.home() / ".cache/aka-candidates")
    )
    directory = cache / ("bf16_gemm_" + key[:16])
    directory.mkdir(parents=True, exist_ok=True)
    native = load(
        name="aka_bf16_gemm_" + key[:16],
        sources=[str(ROOT / "launch.cu")],
        extra_include_paths=[str(ROOT)],
        extra_cuda_cflags=flags,
        extra_cflags=CPP_FLAGS,
        build_directory=str(directory),
        verbose=False,
    )

    def resolve(tag, config):
        # This is a configuration, never a frozen shape index. Output dtype,
        # layout, alpha and accumulation can select configurations absent from the table.
        if tag != "sm100_bf16_gemm" or len(config) != 26:
            raise ValueError("Invalid BF16 GEMM configuration")
        if any(not re.fullmatch(r"[A-Za-z0-9_:<> ,.+-]+", v) for v in config):
            raise ValueError("Invalid C++ configuration token")
        digest = hashlib.sha256((key + json.dumps(config)).encode()).hexdigest()
        dest = directory / "dynamic" / digest
        dest.mkdir(parents=True, exist_ok=True)
        cubin = dest / "kernel.cubin"
        if not cubin.exists():
            source = '#include "kernel.cuh"\nusing namespace deep_gemm;\n'
            source += (
                "auto kernel_entry = &sm100_bf16_gemm_impl<"
                + ", ".join(config)
                + ">;\n"
            )
            (dest / "kernel.cu").write_text(source)
            command = [
                compiler,
                *CUDA_FLAGS,
                "-arch=" + arch,
                "--cubin",
                "-I" + str(ROOT),
                str(dest / "kernel.cu"),
                "-o",
                str(dest / "kernel.tmp.cubin"),
            ]
            with (dest / "compile.log").open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=300,
                )
            (dest / "kernel.tmp.cubin").replace(cubin)
            (dest / "config.json").write_text(
                json.dumps(
                    {"config": config, "source_key": key, "command": command}, indent=2
                )
            )
        return str(dest)

    native.init(str(directory))
    native.set_config_resolver(resolve)
    native.use_deterministic_algorithms(True)
    return native


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.native = _build()

    def forward(self, a, b, d, c=None, compiled_dims="nk", alpha=None):
        return self.native.bf16_gemm_nt(a, b, d, c, compiled_dims, alpha)
