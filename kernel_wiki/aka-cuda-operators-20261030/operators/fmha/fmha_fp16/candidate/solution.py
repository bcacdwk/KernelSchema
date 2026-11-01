"""Paged attention: ABI and route selection; CUDA files are directly editable."""

import hashlib
import importlib
import sys
import types
from pathlib import Path
import torch

_root = Path(__file__).resolve().parent
_package = "_aka_fmha_fp16_" + hashlib.sha256(str(_root).encode()).hexdigest()[:16]
if _package not in sys.modules:
    _module = types.ModuleType(_package)
    _module.__path__ = [str(_root)]
    sys.modules[_package] = _module
_runtime = importlib.import_module(_package + ".runtime")

_build = importlib.import_module(_package + ".build")


class Model(torch.nn.Module):
    """Paged ragged GQA attention over caller-owned buffers.

    The signature is the problem's own ABI, kept argument for argument -- including the options
    this implementation does not vary -- so the evaluator, the reference and the original-library
    baseline are all called identically.
    """

    def __init__(self):
        super().__init__()
        self._ext = None

    def forward(
        self,
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        batch_size,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        window_left=-1,
        out=None,
        out_dtype=None,
        o_sf_scale=None,
        o_sf_vec_size=None,
        kv_layout="HND",
        enable_pdl=None,
        sinks=None,
        kv_cache_sf=None,
        skip_softmax_threshold_scale_factor=None,
        uses_shared_paged_kv_idx=True,
        causal=True,
        lse=None,
        return_lse=False,
        multi_ctas_kv_counter_buffer=None,
        use_fp16_softmax=None,
        uses_spcompress=None,
        backend="trtllm-gen",
    ):
        for name, value in (("bmm1_scale", bmm1_scale), ("bmm2_scale", bmm2_scale)):
            if isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{name} must be a host float; the Cake FMHA compatibility route refuses a device tensor and asks the caller to materialise it"
                )
        key_cache, value_cache = kv_cache
        key_block_scales, value_block_scales = (
            kv_cache_sf if kv_cache_sf is not None else (None, None)
        )
        total_q, heads, head_dim = query.shape
        if out is None:
            dtype = out_dtype if out_dtype is not None else torch.float16
            out = torch.empty(
                (total_q, heads, head_dim), dtype=dtype, device=query.device
            )
        if return_lse and lse is None:
            lse = torch.empty(
                (total_q, heads), dtype=torch.float32, device=query.device
            )
        if (
            query.dtype == torch.float16
            and head_dim == 256
            and (max_q_len == 1)
            and (total_q == batch_size)
            and query.is_contiguous()
            and out.is_contiguous()
            and (heads // key_cache.shape[1] == 16)
            and (key_cache.shape[2] in (16, 32, 64))
            and (key_cache.shape == value_cache.shape)
            and (key_cache.shape[-1] == 256)
            and (key_cache.stride(-1) == value_cache.stride(-1) == 1)
            and (window_left == -1)
            and (sinks is None)
            and (bmm2_scale == 1.0)
            and (str(kv_layout).upper() == "HND")
            and uses_shared_paged_kv_idx
            and block_tables.is_contiguous()
            and seq_lens.is_contiguous()
            and (kv_cache_sf is None)
            and (o_sf_scale is None)
            and (o_sf_vec_size is None)
            and (skip_softmax_threshold_scale_factor in (None, 0.0))
            and (not use_fp16_softmax)
            and (not uses_spcompress)
        ):
            _runtime._run_smallm(
                query,
                key_cache,
                value_cache,
                workspace_buffer,
                block_tables,
                seq_lens,
                max_kv_len,
                bmm1_scale,
                batch_size,
                out,
                lse,
            )
            return (out, lse) if return_lse else out
        if (
            query.dtype == torch.float16
            and head_dim in (64, 128)
            and causal
            and (window_left >= 0 or sinks is not None)
            and (bmm2_scale == 1.0)
            and query.is_contiguous()
            and out.is_contiguous()
            and (key_cache.shape == value_cache.shape)
            and (key_cache.shape[-1] == head_dim)
            and (key_cache.stride(-1) == value_cache.stride(-1) == 1)
            and (key_cache.shape[2] in (16, 32, 64, 128, 256, 512, 1024))
            and (str(kv_layout).upper() == "HND")
            and uses_shared_paged_kv_idx
            and (kv_cache_sf is None)
            and (o_sf_scale is None)
            and (o_sf_vec_size is None)
            and (skip_softmax_threshold_scale_factor in (None, 0.0))
            and (not use_fp16_softmax)
            and (not uses_spcompress)
        ):
            runner = (
                _runtime._run_split_context
                if max_q_len <= 16
                else _runtime._run_masked_context
            )
            runner(
                query,
                key_cache,
                value_cache,
                workspace_buffer,
                block_tables,
                seq_lens,
                max_q_len,
                max_kv_len,
                bmm1_scale,
                batch_size,
                cum_seq_lens_q,
                cum_seq_lens_kv,
                out,
                lse,
                uses_shared_paged_kv_idx,
                multi_ctas_kv_counter_buffer,
                causal,
                sinks,
                window_left,
            )
            return (out, lse) if return_lse else out
        if (
            query.dtype == torch.float16
            and head_dim == 256
            and causal
            and query.is_contiguous()
            and out.is_contiguous()
            and (window_left == -1)
            and (sinks is None)
            and (bmm2_scale == 1.0)
            and (key_cache.shape == value_cache.shape)
            and (key_cache.shape[-1] == 256)
            and (key_cache.shape[2] in (16, 32, 64, 128, 256, 512, 1024))
            and (str(kv_layout).upper() == "HND")
            and uses_shared_paged_kv_idx
            and (kv_cache_sf is None)
            and (o_sf_scale is None)
            and (o_sf_vec_size is None)
            and (skip_softmax_threshold_scale_factor in (None, 0.0))
            and (not use_fp16_softmax)
            and (not uses_spcompress)
        ):
            _runtime._run_hd256_context(
                query,
                key_cache,
                value_cache,
                workspace_buffer,
                block_tables,
                seq_lens,
                max_q_len,
                max_kv_len,
                bmm1_scale,
                batch_size,
                cum_seq_lens_q,
                cum_seq_lens_kv,
                out,
                lse,
            )
            return (out, lse) if return_lse else out
        if (
            query.dtype == torch.float16
            and head_dim in (64, 128)
            and (sinks is None or lse is None)
            and (window_left == -1)
            and (bmm2_scale == 1.0)
            and (kv_cache_sf is None)
            and (str(kv_layout).upper() == "HND")
            and (key_cache.shape == value_cache.shape)
            and (key_cache.shape[-1] == head_dim)
            and (key_cache.shape[2] in (16, 32, 64, 128, 256, 512, 1024))
            and (
                query.stride(-1) == key_cache.stride(-1) == value_cache.stride(-1) == 1
            )
            and (out.dtype == torch.float16)
            and out.is_contiguous()
            and (lse is None or lse.is_contiguous())
            and (skip_softmax_threshold_scale_factor in (None, 0.0))
            and (not use_fp16_softmax)
            and (not uses_spcompress)
            and (o_sf_scale is None)
            and (o_sf_vec_size is None)
        ):
            _runtime._run_context(
                query,
                key_cache,
                value_cache,
                workspace_buffer,
                block_tables,
                seq_lens,
                max_q_len,
                max_kv_len,
                bmm1_scale,
                batch_size,
                cum_seq_lens_q,
                cum_seq_lens_kv,
                out,
                lse,
                uses_shared_paged_kv_idx,
                multi_ctas_kv_counter_buffer,
                causal,
                sinks,
            )
            return (out, lse) if return_lse else out
        if self._ext is None:
            self._ext = _build.build_compat()
        self._ext.fmha_compat_context(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            cum_seq_lens_q,
            out,
            lse,
            sinks,
            key_block_scales,
            value_block_scales,
            workspace_buffer,
            float(bmm1_scale),
            float(bmm2_scale),
            int(batch_size),
            0,
            int(window_left),
            bool(causal),
            bool(uses_shared_paged_kv_idx),
            0 if str(kv_layout).upper() == "HND" else 1,
            float(o_sf_scale) if o_sf_scale is not None else 0.0,
            0,
            None,
        )
        return (out, lse) if return_lse else out
