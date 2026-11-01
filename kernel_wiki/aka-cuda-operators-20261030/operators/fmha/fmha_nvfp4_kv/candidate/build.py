"""Compile editable local translation units; no source materialization."""

from functools import cache
from pathlib import Path
import hashlib
import json
import os
import subprocess
import torch

ROOT = Path(__file__).resolve().parent


@cache
def _source_identity():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.iterdir())
        if p.is_file()
    }


def compile_units(tag, units, flags, python_module=False, cpp_flags=None):
    from torch.utils import cpp_extension
    import tvm_ffi
    import tvm_ffi.libinfo

    compiler = subprocess.check_output(
        [str(Path(cpp_extension.CUDA_HOME) / "bin/nvcc"), "--version"], text=True
    )
    cpp_flags = cpp_flags or ["-O3", "-std=c++17"]
    identity = [
        _source_identity(),
        units,
        flags,
        cpp_flags,
        compiler,
        torch.__version__,
        torch.version.cuda,
        tvm_ffi.__version__,
        python_module,
    ]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    build = Path(
        os.environ.get("AKA_SF_BUILD_CACHE", Path.home() / ".cache/aka-candidates")
    ) / (tag + "_" + key)
    build.mkdir(parents=True, exist_ok=True)
    library = cpp_extension.load(
        name="aka_" + tag + "_" + key,
        sources=[str(ROOT / p) for p in units],
        extra_cuda_cflags=flags,
        extra_cflags=cpp_flags,
        extra_include_paths=[
            str(ROOT),
            tvm_ffi.libinfo.find_include_path(),
            tvm_ffi.libinfo.find_dlpack_include_path(),
        ],
        extra_ldflags=[] if python_module else ["-lcuda"],
        build_directory=str(build),
        is_python_module=python_module,
        verbose=False,
    )
    return library if python_module else tvm_ffi.load_module(library)


@cache
def build_compat():
    major, minor = torch.cuda.get_device_capability()
    flags = ["-O3", "--expt-relaxed-constexpr", "-use_fast_math"] + [
        f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"
    ]
    return compile_units(
        "fmha_nvfp4_kv_compat",
        ["compat_launch.cu", "compat_bind_context.cpp"],
        flags,
        True,
        ["-O3"],
    )


@cache
def _load_d06tc(
    major,
    minor,
    blocks,
    heads,
    kvheads,
    pack,
    page,
    swizzle,
    has_lse,
    causal,
    has_sink,
    output_dim,
    window=-1,
):
    defs = dict(
        NUM_M_BLOCKS=blocks,
        NUM_Q_HEADS=heads,
        HEADS_PER_GROUP=heads // kvheads,
        PACK_G=pack,
        TOK_PER_STAGE=128 // pack,
        L2_SWIZZLE=swizzle,
        PAGE_SIZE=page,
        CAKE_FMHA_CONTEXT_IS_CAUSAL=int(causal),
        CAKE_FMHA_CONTEXT_RETURN_LSE=int(has_lse),
        CAKE_FMHA_CONTEXT_ENABLE_SINK=int(has_sink),
        D06_OUTPUT_HEAD_DIM=output_dim,
        D06_WINDOW_LEFT=window,
        D06_SOURCE_PAGE_SIZE=page,
    )
    if output_dim == 256:
        defs.update(
            CAKE_FMHA_HD256_FP8=1, CAKE_FMHA_SOURCE_PAGE_SIZE=page, NUM_KV_HEADS=kvheads
        )
        defs.pop("PAGE_SIZE")
    flags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a",
    ]
    flags += [f"-D{k}={v}" for k, v in defs.items()]
    units = [
        f"context_sink{int(has_sink)}_causal{int(causal)}_lse{int(has_lse)}.cu",
        'context_launch.cu',
        'context_d06_dequant.cu',
    ]
    if output_dim == 256:
        units = [
        'context_hd256_causal1_output_bf161.cu',
        'wide_launch.cu',
        'wide_support.cu',
        'context_d06_dequant.cu',
    ]
    return compile_units("fmha_nvfp4_kv__load_d06tc", units, flags)


@cache
def _load_d06tc_short(
    major,
    minor,
    blocks,
    heads,
    kvheads,
    pack,
    page,
    swizzle,
    has_lse,
    causal,
    has_sink,
    output_dim,
    window=-1,
):
    defs = dict(
        NUM_M_BLOCKS=blocks,
        NUM_Q_HEADS=heads,
        HEADS_PER_GROUP=heads // kvheads,
        PACK_G=pack,
        TOK_PER_STAGE=128 // pack,
        L2_SWIZZLE=swizzle,
        PAGE_SIZE=page,
        CAKE_FMHA_CONTEXT_IS_CAUSAL=int(causal),
        CAKE_FMHA_CONTEXT_RETURN_LSE=int(has_lse),
        CAKE_FMHA_CONTEXT_ENABLE_SINK=int(has_sink),
        D06_OUTPUT_HEAD_DIM=output_dim,
        D06_WINDOW_LEFT=window,
        D06_SOURCE_PAGE_SIZE=page,
    )
    if output_dim == 256:
        defs.update(
            CAKE_FMHA_HD256_FP8=1,
            CAKE_FMHA_SOURCE_PAGE_SIZE=page,
            NUM_KV_HEADS=kvheads,
            D06_HD256_PACK_G=pack,
            HEADS_PER_GROUP=heads // kvheads // pack,
        )
        defs.pop("PAGE_SIZE")
    flags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a",
    ]
    flags += [f"-D{k}={v}" for k, v in defs.items()]
    units = [
        f"context_sink{int(has_sink)}_causal{int(causal)}_lse{int(has_lse)}.cu",
        'context_launch.cu',
        'context_d06_dequant.cu',
    ]
    if output_dim == 256:
        units = [
        'short_kernel.cu',
        'short_launch.cu',
        'short_support.cu',
        'context_d06_dequant.cu',
    ]
    return compile_units("fmha_nvfp4_kv__load_d06tc_short", units, flags)


@cache
def _load_d06tc_window_split(
    major,
    minor,
    blocks,
    heads,
    kvheads,
    pack,
    page,
    swizzle,
    has_lse,
    causal,
    has_sink,
    output_dim,
    window=-1,
    splits=1,
):
    defs = dict(
        D06_CONTEXT_SPLITS=splits,
        NUM_M_BLOCKS=blocks,
        NUM_Q_HEADS=heads,
        HEADS_PER_GROUP=heads // kvheads,
        PACK_G=pack,
        TOK_PER_STAGE=128 // pack,
        L2_SWIZZLE=swizzle,
        PAGE_SIZE=page,
        CAKE_FMHA_CONTEXT_IS_CAUSAL=int(causal),
        CAKE_FMHA_CONTEXT_RETURN_LSE=int(has_lse),
        CAKE_FMHA_CONTEXT_ENABLE_SINK=int(has_sink),
        D06_OUTPUT_HEAD_DIM=output_dim,
        D06_WINDOW_LEFT=window,
        D06_SOURCE_PAGE_SIZE=page,
    )
    if output_dim == 256:
        defs.update(
            CAKE_FMHA_HD256_FP8=1, CAKE_FMHA_SOURCE_PAGE_SIZE=page, NUM_KV_HEADS=kvheads
        )
        defs.pop("PAGE_SIZE")
    flags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a",
    ]
    flags += [f"-D{k}={v}" for k, v in defs.items()]
    units = [
        ("window_kernel.cu" if has_sink and causal and has_lse else f"context_sink{int(has_sink)}_causal{int(causal)}_lse{int(has_lse)}.cu"),
        'window_launch.cu',
        'context_d06_dequant.cu',
    ]
    if output_dim == 256:
        units = [
        'context_hd256_causal1_output_bf161.cu',
        'wide_launch.cu',
        'wide_support.cu',
        'context_d06_dequant.cu',
    ]
    return compile_units("fmha_nvfp4_kv__load_d06tc_window_split", units, flags)
