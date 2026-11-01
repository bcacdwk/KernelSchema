"""Packed varlen attention ABI; local staged CUDA implementation."""

import hashlib
import importlib
import sys
import types
from pathlib import Path
import torch

_root = Path(__file__).resolve().parent
_package = (
    "_aka_fmha_nvfp4_qkpv_" + hashlib.sha256(str(_root).encode()).hexdigest()[:16]
)
if _package not in sys.modules:
    _module = types.ModuleType(_package)
    _module.__path__ = [str(_root)]
    sys.modules[_package] = _module
_runtime = importlib.import_module(_package + ".runtime")


class Model(torch.nn.Module):
    def forward(
        self,
        query,
        key,
        value,
        cu_seqlens,
        *,
        pv_mode="fp8",
        softmax_scale=None,
        out=None,
        cu_seqlens_host=None,
        backend="cake",
    ):
        return _runtime.minimax_h3_varlen_nvfp4_attention(
            query,
            key,
            value,
            cu_seqlens,
            pv_mode=pv_mode,
            softmax_scale=softmax_scale,
            out=out,
            cu_seqlens_host=cu_seqlens_host,
            backend=backend,
        )
