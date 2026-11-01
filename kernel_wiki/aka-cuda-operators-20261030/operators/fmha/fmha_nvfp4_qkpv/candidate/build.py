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


CUDA_FLAGS = [
    "-O3",
    "-std=c++17",
    "-gencode=arch=compute_103a,code=sm_103a",
    "--expt-relaxed-constexpr",
    "-static-global-template-stub=false",
    "-Xfatbin=-compress-all",
    "--compress-mode=size",
    "--compiler-options=-fPIC",
    "-DENABLE_F16",
    "-DENABLE_BF16",
    "-DENABLE_FP8_E4M3",
    "-DENABLE_FP8_E5M2",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]
MODULES = {
    "fp8pv_sm_103a_amax": {
        "arch": "sm_103a",
        "sources": ["fp8pv_sm_103a_amax.cu", "fp8pv_sm_103a_amax_launch.cu"],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "v"],
            ["buffer", "v_amax_partial"],
            ["parameter", "num_vectors"],
            ["parameter", "num_chunks"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_workspace_bytes": 0,
    },
    "fp8pv_sm_103a_attention": {
        "arch": "sm_103a",
        "sources": ["fp8pv_sm_103a_attention.cu", "fp8pv_sm_103a_attention_launch.cu"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["buffer", "O"],
            ["buffer", "v_amax"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_workspace_bytes": 0,
    },
    "fp8pv_sm_103a_attention_split": {
        "arch": "sm_103a",
        "sources": [
            "fp8pv_sm_103a_attention_split.cu",
            "fp8pv_sm_103a_attention_split_launch.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["buffer", "O"],
            ["buffer", "v_amax"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["buffer", "cl_kv_begin"],
            ["buffer", "cl_kv_blocks"],
            ["buffer", "cl_ws_slot"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "num_tiles"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_workspace_bytes": 0,
    },
    "fp8pv_sm_103a_quantize": {
        "arch": "sm_103a",
        "sources": ["fp8pv_sm_103a_quantize.cu", "fp8pv_sm_103a_quantize_launch.cu"],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "q_fp4"],
            ["buffer", "k_fp4"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_fp8"],
            ["buffer", "v_amax"],
            ["buffer", "v_amax_partial"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_workspace_bytes": 0,
    },
    "fp4pv_sm_103a_combine": {
        "arch": "sm_103a",
        "sources": ["combine.cu", "combine_launch.cu"],
        "compile_flags": [
            "--use_fast_math",
            "-DAKA_COMBINE_KERNEL=kernel_cake_minimax_h3_varlen_attention_79cc755a24fa6f9758c8",
            "-DAKA_COMBINE_HOST_NAMESPACE=cake_host_shim_31d43c1fd5ee768b",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["buffer", "combine_table"],
            ["buffer", "seg_begin"],
            ["buffer", "seg_len"],
            ["buffer", "O"],
            ["parameter", "num_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_workspace_bytes": 0,
    },
    "fp4pv_sm_103a_attention_split": {
        "arch": "sm_103a",
        "sources": [
            "fp4pv_sm_103a_attention_split.cu",
            "fp4pv_sm_103a_attention_split_launch.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "Vt"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["tma_buffer", "SFVtLo"],
            ["tma_buffer", "SFVtHi"],
            ["buffer", "O"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["buffer", "cl_kv_begin"],
            ["buffer", "cl_kv_blocks"],
            ["buffer", "cl_ws_slot"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "num_tiles"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
            ["buffer", "prefix_max"],
            ["parameter", "prefix_stride"],
            ["buffer", "pf_tile"],
            ["parameter", "pf_total"],
            ["buffer", "pf_flag"],
            ["buffer", "pf_info"],
        ],
        "tma_workspace_bytes": 0,
    },
    "fp4pv_sm_103a_attention": {
        "arch": "sm_103a",
        "sources": ["fp4pv_sm_103a_attention.cu", "fp4pv_sm_103a_attention_launch.cu"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "Vt"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["tma_buffer", "SFVtLo"],
            ["tma_buffer", "SFVtHi"],
            ["buffer", "O"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_workspace_bytes": 0,
    },
    "fp4pv_sm_103a_quantize": {
        "arch": "sm_103a",
        "sources": ["fp4pv_sm_103a_quantize.cu", "fp4pv_sm_103a_quantize_launch.cu"],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "q_fp4"],
            ["buffer", "k_fp4"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_fp4_t"],
            ["buffer", "v_scale_lo"],
            ["buffer", "v_scale_hi"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_workspace_bytes": 0,
    },
    "fp4pv_sm_100a_quantize": {
        "arch": "sm_100a",
        "sources": ["fp4pv_sm_100a_quantize.cu", "fp4pv_sm_100a_quantize_launch.cu"],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "q_fp4"],
            ["buffer", "k_fp4"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_fp4_t"],
            ["buffer", "v_scale_lo"],
            ["buffer", "v_scale_hi"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_workspace_bytes": 0,
    },
    "fp4pv_sm_100a_attention": {
        "arch": "sm_100a",
        "sources": ["fp4pv_sm_100a_attention.cu", "fp4pv_sm_100a_attention_launch.cu"],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "Vt"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["tma_buffer", "SFVtLo"],
            ["tma_buffer", "SFVtHi"],
            ["buffer", "O"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_workspace_bytes": 0,
    },
    "fp4pv_sm_100a_attention_split": {
        "arch": "sm_100a",
        "sources": [
            "fp4pv_sm_100a_attention_split.cu",
            "fp4pv_sm_100a_attention_split_launch.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "Vt"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["tma_buffer", "SFVtLo"],
            ["tma_buffer", "SFVtHi"],
            ["buffer", "O"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["buffer", "cl_kv_begin"],
            ["buffer", "cl_kv_blocks"],
            ["buffer", "cl_ws_slot"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "num_tiles"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
            ["buffer", "prefix_max"],
            ["parameter", "prefix_stride"],
            ["parameter", "prefix_only"],
        ],
        "tma_workspace_bytes": 0,
    },
    "fp4pv_sm_100a_combine": {
        "arch": "sm_100a",
        "sources": ["combine.cu", "combine_launch.cu"],
        "compile_flags": [
            "--use_fast_math",
            "-DAKA_COMBINE_KERNEL=kernel_cake_minimax_h3_varlen_attention_b5202869834cd7822fb1",
            "-DAKA_COMBINE_HOST_NAMESPACE=cake_host_shim_007c63ffd1fd125a",
        ],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["buffer", "combine_table"],
            ["buffer", "seg_begin"],
            ["buffer", "seg_len"],
            ["buffer", "O"],
            ["parameter", "num_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "tma_workspace_bytes": 0,
    },
}
ROUTES = {
    "nvfp4_fp4pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention", "attention_split", "combine"],
        # The sm_103a FP4 split program runs its QK-only prepass as phase 0 of
        # the attention launch, so the host binds one attention_split call.  The
        # sm_100a FP4 split program still takes a separate ``prefix_only`` pass.
        "fused_prefix_prepass": True,
        "modules": {
            "quantize": "fp4pv_sm_103a_quantize",
            "attention": "fp4pv_sm_103a_attention",
            "attention_split": "fp4pv_sm_103a_attention_split",
            "combine": "fp4pv_sm_103a_combine",
        },
    },
    "nvfp4_fp8pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp8pv",
        "stages": ["amax", "quantize", "attention", "attention_split", "combine"],
        "modules": {
            "amax": "fp8pv_sm_103a_amax",
            "quantize": "fp8pv_sm_103a_quantize",
            "attention": "fp8pv_sm_103a_attention",
            "attention_split": "fp8pv_sm_103a_attention_split",
            "combine": "fp4pv_sm_103a_combine",
        },
    },
    "nvfp4_fp4pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention", "attention_split", "combine"],
        "modules": {
            "quantize": "fp4pv_sm_100a_quantize",
            "attention": "fp4pv_sm_100a_attention",
            "attention_split": "fp4pv_sm_100a_attention_split",
            "combine": "fp4pv_sm_100a_combine",
        },
    },
}
STAGES = {
    "bf16": ("attention", "combine"),
    "nvfp4_fp4pv": ("quantize", "attention", "attention_split", "combine"),
    "nvfp4_fp8pv": ("amax", "quantize", "attention", "attention_split", "combine"),
}
VARIANTS = ("bf16", "nvfp4_fp4pv", "nvfp4_fp8pv")


@cache
def load_cake_minimax_h3_varlen_attention_module(name):
    record = MODULES[name]
    flags = [
        f
        for f in CUDA_FLAGS
        if not f.startswith(
            ("-gencode", "--generate-code", "-arch=", "--gpu-architecture=")
        )
    ]
    flags += record["compile_flags"]
    flags.append(
        "-gencode=arch=compute_" + record["arch"][3:] + ",code=" + record["arch"]
    )
    return compile_units("qkpv_" + name, record["sources"], flags)
