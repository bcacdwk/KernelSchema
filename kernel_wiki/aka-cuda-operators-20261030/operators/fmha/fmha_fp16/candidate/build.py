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
        "fmha_fp16_compat",
        ["compat_launch.cu", "compat_bind_context.cpp"],
        flags,
        True,
        ["-O3"],
    )


@cache
def _load_context(
    major,
    minor,
    blocks,
    heads,
    kv_heads,
    pack,
    page,
    swizzle,
    has_lse,
    causal,
    has_sink,
    head_dim,
):
    definitions = dict(
        AKA_CONTEXT_HEAD_DIM=head_dim,
        NUM_M_BLOCKS=blocks,
        NUM_Q_HEADS=heads,
        HEADS_PER_GROUP=heads // kv_heads,
        PACK_G=pack,
        TOK_PER_STAGE=128 // pack,
        L2_SWIZZLE=swizzle,
        PAGE_SIZE=page,
        CAKE_FMHA_CONTEXT_IS_CAUSAL=int(causal),
        CAKE_FMHA_CONTEXT_RETURN_LSE=int(has_lse),
        CAKE_FMHA_CONTEXT_ENABLE_SINK=int(has_sink),
    )
    flags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a",
    ]
    flags += [f"-D{k}={v}" for k, v in definitions.items()]
    units = [
        f"context_sink{int(has_sink)}_causal{int(causal)}_lse{int(has_lse)}.cu",
        'context_launch.cu',
    ]
    return compile_units("fmha_fp16__load_context", units, flags)


@cache
def _load_smallm(page, group, split):
    component = f"decode_native_bf16_hd256_smallm_n32_p{page}"
    definitions = dict(
        Q_LEN=1,
        GROUP=group,
        Q_BOX_ROWS=32 // group,
        NUM_SPLIT=split,
        CAKE_FMHA_SMALLM_N_ROWS=32,
        CAKE_FMHA_SMALLM_PAGE_SIZE=page,
        CAKE_FMHA_SMALLM_LAUNCH="cake_fmha_launch_" + component,
    )
    flags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        "-gencode=arch=compute_103a,code=sm_103a",
    ]
    flags += [f"-D{k}={v}" for k, v in definitions.items()]
    units = [
        f"smallq_page{page}.cu",
        'smallq_launch.cu',
    ]
    return compile_units("fmha_fp16__load_smallm", units, flags)


@cache
def _load_masked_context(
    major,
    minor,
    blocks,
    heads,
    kv_heads,
    pack,
    page,
    swizzle,
    has_lse,
    causal,
    has_sink,
    head_dim,
):
    definitions = dict(
        AKA_CONTEXT_HEAD_DIM=head_dim,
        NUM_M_BLOCKS=blocks,
        NUM_Q_HEADS=heads,
        HEADS_PER_GROUP=heads // kv_heads,
        PACK_G=pack,
        TOK_PER_STAGE=128 // pack,
        L2_SWIZZLE=swizzle,
        PAGE_SIZE=page,
        CAKE_FMHA_CONTEXT_IS_CAUSAL=int(causal),
        CAKE_FMHA_CONTEXT_RETURN_LSE=int(has_lse),
        CAKE_FMHA_CONTEXT_ENABLE_SINK=int(has_sink),
    )
    flags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a",
    ]
    flags += [f"-D{k}={v}" for k, v in definitions.items()]
    units = [
        'masked_kernel.cu',
        'masked_launch.cu',
    ]
    return compile_units("fmha_fp16__load_masked_context", units, flags)


@cache
def _load_hd256_context(blocks, heads, kvheads, page):
    definitions = dict(
        NUM_M_BLOCKS=blocks,
        NUM_Q_HEADS=heads,
        NUM_KV_HEADS=kvheads,
        HEADS_PER_GROUP=heads // kvheads,
        CAKE_FMHA_HD256_FP8=0,
        CAKE_FMHA_SOURCE_PAGE_SIZE=page,
    )
    flags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        "-gencode=arch=compute_103a,code=sm_103a",
    ]
    flags += [f"-D{k}={v}" for k, v in definitions.items()]
    units = [
        'wide_kernel.cu',
        'wide_support_cake_fmha_hd256_support.cu',
        'wide_launch.cu',
    ]
    return compile_units("fmha_fp16__load_hd256_context", units, flags)


@cache
def _load_split_context(
    major,
    minor,
    blocks,
    heads,
    kv_heads,
    pack,
    page,
    swizzle,
    has_lse,
    causal,
    has_sink,
    head_dim,
    split,
    singleq,
):
    definitions = dict(
        AKA_SINGLE_Q_STAGE=int(singleq),
        AKA_NUM_SPLIT=split,
        AKA_CONTEXT_HEAD_DIM=head_dim,
        NUM_M_BLOCKS=blocks,
        NUM_Q_HEADS=heads,
        HEADS_PER_GROUP=heads // kv_heads,
        PACK_G=pack,
        TOK_PER_STAGE=128 // pack,
        L2_SWIZZLE=swizzle,
        PAGE_SIZE=page,
        CAKE_FMHA_CONTEXT_IS_CAUSAL=int(causal),
        CAKE_FMHA_CONTEXT_RETURN_LSE=int(has_lse),
        CAKE_FMHA_CONTEXT_ENABLE_SINK=int(has_sink),
    )
    flags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a",
    ]
    flags += [f"-D{k}={v}" for k, v in definitions.items()]
    units = [
        'split_kernel.cu',
        'split_launch.cu',
    ]
    return compile_units("fmha_fp16__load_split_context", units, flags)
