"""Local KDA compilation and shape/stream keyed workspace allocation."""

from pathlib import Path
from functools import cache
import hashlib, json, os, subprocess
import torch, tvm_ffi

ROOT = Path(__file__).resolve().parent


@cache
def _module(role, heads=0, sequences=0):
    from torch.utils import cpp_extension
    import tvm_ffi.libinfo

    flags = [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-gencode=arch=compute_103a,code=sm_103a",
        "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=3",
    ]
    if role == "prepare":
        flags += [f"-DAKA_KDA_HEADS={heads}", f"-DAKA_KDA_SEQUENCES={sequences}"]
    compiler = subprocess.check_output(
        [str(Path(cpp_extension.CUDA_HOME) / "bin/nvcc"), "--version"], text=True
    )
    files = {p.name: p.read_text() for p in sorted(ROOT.iterdir()) if p.is_file()}
    key = hashlib.sha256(
        json.dumps(
            [
                files,
                flags,
                compiler,
                torch.__version__,
                torch.version.cuda,
                getattr(tvm_ffi, "__version__", None),
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    build = Path(
        os.environ.get("AKA_SF_BUILD_CACHE", Path.home() / ".cache/aka-candidates")
    ) / ("kda_" + key + "_" + role)
    build.mkdir(parents=True, exist_ok=True)
    source = "preprocess.cu" if role == "preprocess" else role + "_launch.cu"
    path = cpp_extension.load(
        name="aka_kda_" + key + "_" + role,
        sources=[str(ROOT / source)],
        extra_cuda_cflags=flags,
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            str(ROOT),
            tvm_ffi.libinfo.find_include_path(),
            tvm_ffi.libinfo.find_dlpack_include_path(),
        ],
        build_directory=str(build),
        is_python_module=False,
        verbose=False,
    )
    return tvm_ffi.load_module(path)


@cache
def _kda_entry(role, heads=0, sequences=0):
    return _module("preprocess" if role in ("metadata", "beta_pack") else role,
                   heads, sequences)[
        role if role in ("metadata", "beta_pack") else "run"
    ]


@cache
def _kda_plan(total, heads, sequences, device, stream_id, engine):
    capacity = min(total, sequences) + (total - min(total, sequences)) // 16

    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

    if engine:
        return dict(
            capacity=0,
            chunks=empty(sequences + 1, torch.int32),
            mapping=empty(1, torch.int32),
            order=empty(sequences, torch.int32),
            unused_indices=empty(1, torch.int32),
            unused_ckpt=empty(1, torch.bfloat16),
            unused_ckpt_starts=empty(1, torch.int64),
            beta_tma=empty((1, max(total, 16), (heads + 7) // 8 * 8), torch.bfloat16),
        )
    return dict(
        capacity=capacity,
        qd=empty((1, heads, capacity * 16, 128), torch.bfloat16),
        kd=empty((1, heads, capacity * 16, 128), torch.bfloat16),
        w=empty((1, heads, capacity * 16, 128), torch.bfloat16),
        qk=empty((1, heads, capacity, 16, 16), torch.bfloat16),
        diag=empty((1, heads, capacity, 128), torch.float32),
        unused_indices=empty(1, torch.int32),
        unused_ckpt=empty(1, torch.bfloat16),
        unused_ckpt_starts=empty(1, torch.int64),
    )
