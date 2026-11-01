from __future__ import annotations

import os

from typing import Optional

import torch

_EXT = None

_EXT_ERROR: Optional[BaseException] = None

_USE_V1 = False

_HILO = 1

_TARGET_CTAS = 1024

_PYFAST = os.environ.get("MLA_PYFAST", "1") != "0"

_FASTCALL = os.environ.get("MLA_FASTCALL", "1") != "0"

_CFAST = os.environ.get("MLA_CFAST", "1") != "0"

_FAST = None

_REPLAY = None

_CCALL = None


def _install_c_call(ext, model):
    """Hand Model.__call__ to the C entry point once the graph cache is warm.

    _PyObject_LookupSpecial binds descriptors and a PyCFunction is not one, so the
    C entry point is invoked without self; mla_v1_bind_instance gives it the
    instance and the bound forward it falls back to. On a miss the C side calls
    that bound forward and re-arms from the argument vector it just saw, and
    mla_v1_forward clears the active token whenever it launches without a graph,
    so the C cache can only ever hold a graph that the call it is replaying for
    actually selected.
    """
    global _CCALL
    if not _CFAST or _CCALL is not None:
        return
    if not ext.mla_v1_bind_instance(model):
        return
    try:
        _CCALL = ext.mla_v1_make_fast_call()
        type(model).__call__ = _CCALL
    except BaseException:
        _CCALL = None


def _arch_list() -> str:
    major, minor = torch.cuda.get_device_capability()
    suffix = "a" if major >= 9 else ""
    return f"{major}.{minor}{suffix}"


def _ext():
    global _EXT, _EXT_ERROR
    if _EXT is None and _EXT_ERROR is None:
        try:
            _EXT = _build()
            _resolve_impl()
        except BaseException as error:
            _EXT_ERROR = error
    if _EXT is None:
        raise RuntimeError(f"mla_absorbed_fp8 candidate build failed: {_EXT_ERROR!r}")
    return _EXT


def _resolve_impl() -> None:
    global _USE_V1, _HILO
    forced = os.environ.get("MLA_FORCE_IMPL", "auto").lower()
    _HILO = 0 if os.environ.get("MLA_HILO", "1") == "0" else 1
    global _TARGET_CTAS
    try:
        _TARGET_CTAS = int(os.environ.get("MLA_TARGET_CTAS", "1024"))
    except ValueError:
        _TARGET_CTAS = 1024
    if forced == "v0":
        _USE_V1 = False
        return
    if forced == "v1":
        _USE_V1 = True
        return
    major, minor = torch.cuda.get_device_capability()
    _USE_V1 = (major, minor) in ((10, 0), (10, 1), (10, 3))


class Model(torch.nn.Module):
    """Production forward ABI for absorbed FP8 MLA."""

    def forward(
        self,
        query,
        kv_cache,
        workspace_buffer,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        block_tables,
        seq_lens,
        max_seq_len,
        sparse_mla_top_k=0,
        out=None,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        sinks=None,
        skip_softmax_threshold_scale_factor=None,
        enable_pdl=None,
        backend="auto",
        is_var_seq=True,
        uses_shared_paged_kv_idx=True,
        lse=None,
        return_lse=False,
        cute_dsl_impl="auto",
        kv_scale_format="auto",
        cum_seq_lens_q=None,
        max_q_len=None,
        multi_ctas_kv_counter_buffer=None,
        sparse_mla_top_k_lens=None,
        enable_dcp=False,
        cp_world=1,
        cp_rank=0,
        causal_seqlens_kv_global=None,
        use_fp16_softmax=None,
        return_lse_base=None,
    ):
        if out is None:
            raise ValueError(
                "mla_absorbed_fp8 candidate requires a caller-owned out tensor"
            )
        global _FAST, _REPLAY
        key = (
            id(query),
            id(kv_cache),
            id(workspace_buffer),
            id(block_tables),
            id(seq_lens),
            id(out),
            id(lse),
            id(sinks),
            id(cum_seq_lens_q),
            bmm1_scale,
            bmm2_scale,
            return_lse,
            max_q_len,
            max_seq_len,
        )
        fast = _FAST
        if fast is not None:
            if fast[0] == key and _REPLAY(fast[1]):
                return fast[2]
            _FAST = None
        ext = _ext()
        if _USE_V1:
            token = ext.mla_v1_forward(
                query,
                kv_cache,
                block_tables,
                seq_lens,
                cum_seq_lens_q,
                out,
                lse,
                sinks,
                float(bmm1_scale),
                float(bmm2_scale),
                _HILO,
                workspace_buffer,
                int(max_q_len) if max_q_len else 0,
                int(max_seq_len) if max_seq_len else 0,
                _TARGET_CTAS,
            )
            result = (out, lse) if return_lse else out
            if token and _PYFAST:
                _REPLAY = ext.mla_v1_fast_replay() if _FASTCALL else ext.mla_v1_replay
                _FAST = (
                    key,
                    token,
                    result,
                    (
                        query,
                        kv_cache,
                        workspace_buffer,
                        block_tables,
                        seq_lens,
                        out,
                        lse,
                        sinks,
                        cum_seq_lens_q,
                    ),
                )
                _install_c_call(ext, self)
            return result
        ext.mla_v0_forward(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            cum_seq_lens_q,
            out,
            lse,
            sinks,
            float(bmm1_scale),
            float(bmm2_scale),
        )
        return (out, lse) if return_lse else out

    __call__ = forward


from pathlib import Path
import hashlib
import json
import subprocess
from torch.utils.cpp_extension import load, CUDA_HOME

ROOT = Path(__file__).resolve().parent


def _build():
    arch = os.environ.get("MLA_FORCE_ARCH_LIST") or _arch_list()
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"]
    compiler = subprocess.check_output(
        [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
    )
    files = {p.name: p.read_text() for p in sorted(ROOT.iterdir()) if p.is_file()}
    key = hashlib.sha256(
        json.dumps(
            [files, flags, arch, compiler, torch.__version__, torch.version.cuda],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]
    cache = Path(
        os.environ.get(
            "MLA_ABSORBED_FP8_EXT_DIR",
            os.environ.get("AKA_SF_BUILD_CACHE", Path.home() / ".cache/aka-candidates"),
        )
    )
    build = cache / ("mla_absorbed_fp8_" + key)
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="aka_mla_absorbed_fp8_" + key,
        sources=[str(ROOT / "launch.cu")],
        extra_cuda_cflags=flags,
        build_directory=str(build),
        verbose=False,
    )
