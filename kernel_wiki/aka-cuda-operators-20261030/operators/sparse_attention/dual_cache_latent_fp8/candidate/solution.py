"""Dual-cache ABI; kernel portfolio is compiled directly."""

import hashlib
import importlib
import sys
import types
from pathlib import Path
import torch

_root = Path(__file__).resolve().parent
_package = (
    "_aka_dual_cache_latent_fp8_" + hashlib.sha256(str(_root).encode()).hexdigest()[:16]
)
if _package not in sys.modules:
    _module = types.ModuleType(_package)
    _module.__path__ = [str(_root)]
    sys.modules[_package] = _module
_runtime = importlib.import_module(_package + ".runtime")

_config = importlib.import_module(_package + ".config")

_ALIGN = _config._ALIGN

_PRIMED_ATTR = _config._PRIMED_ATTR

_selected_workspace_bytes = _runtime._selected_workspace_bytes

_target_arch = _runtime._target_arch

run_cake_dsv4 = _runtime.run_cake_dsv4


class Model(torch.nn.Module):
    """Public dual-cache ABI, editable CUDA portfolio and caller-owned output.

    Uses caller scratch when sufficient, otherwise fresh per-call storage.
    Numerical and performance coverage is recorded separately by source SHA.
    """

    def forward(
        self,
        query,
        swa_kv_cache,
        workspace_buffer,
        sparse_indices,
        compressed_kv_cache=None,
        sparse_topk_lens=None,
        seq_lens=None,
        out=None,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        sinks=None,
        kv_layout="HND",
        cum_seq_lens_q=None,
        max_q_len=None,
        enable_pdl=None,
        swa_topk_lens=None,
        extra_sparse_indices=None,
        extra_sparse_topk_lens=None,
    ):
        if kv_layout != "HND":
            raise ValueError("Frozen dual-cache contract requires HND")
        if swa_topk_lens is not None:
            raise ValueError("Explicit SWA lengths not supported by A")
        if out is None:
            out = torch.empty_like(query, dtype=torch.bfloat16)
        topk = (
            sparse_indices.shape[1]
            if extra_sparse_indices is None
            else 128 + extra_sparse_indices.shape[1]
        )
        size = _selected_workspace_bytes(
            _target_arch(query.device),
            query.dtype,
            sparse_indices.shape[0],
            query.shape[1],
            topk,
            seq_lens.numel(),
            max_q_len,
            cum_seq_lens_q is not None,
            compressed_kv_cache.shape[-2],
            query.device,
        )
        if (
            workspace_buffer.device == query.device
            and workspace_buffer.is_contiguous()
            and (workspace_buffer.numel() * workspace_buffer.element_size() >= size)
            and (workspace_buffer.data_ptr() % _ALIGN == 0)
        ):
            scratch = workspace_buffer
        else:
            scratch = torch.empty(size, dtype=torch.uint8, device=query.device)
        setattr(scratch, _PRIMED_ATTR, False)
        return run_cake_dsv4(
            query=query,
            swa_kv_cache=swa_kv_cache,
            compressed_kv_cache=compressed_kv_cache,
            workspace_buffer=scratch,
            sparse_indices=sparse_indices,
            sparse_topk_lens=sparse_topk_lens,
            seq_lens=seq_lens,
            out=out,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=sinks,
            max_q_len=max_q_len,
            cum_seq_lens_q=cum_seq_lens_q,
            backend="cake",
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_topk_lens,
        )
