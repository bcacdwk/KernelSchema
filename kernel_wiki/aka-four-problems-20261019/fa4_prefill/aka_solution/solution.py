"""Protected production-ABI adapter for the recovered Increment FA4 R0 seed."""

import os

_STABLE_KNOBS = {
    "FA_HD256_PAGED64_TMA": "1",
    "FA_HD256_EX2EMU": "auto",
    "FA_HD256_KVSTAGE": "0",
    "FA_HD256_LDRED": "1",
    "FA_HD256_FP8_MAX_OFFSET": "1",
    "FA_HD256_STATIC_MASK_LOOPS": "1",
    "FA_HD256_SKIPCORR": "1",
    "FA_HD256_COLEPI": "1",
    "FA_HD256_EARLY_O_RELEASE": "1",
    "FA_HD256_MRASTER": "2",
    "FA_HD256_FLATVARLEN": "1",
    "FA_HD256_FLATLPT": "1",
    "FA_HD256_SHARE_RESCALE": "1",
    "FA_HD256_SPLIT_KV_LOADS": "1",
    "FA_HD256_PACKGQA": "auto",
    "FA_HD256_SPLITKV": "auto",
    "FA_HD256_SPLIT_MIN_BLOCKS": "16",
    "FA_HD256_SPLIT_BLOCKS_PER_SPLIT": "2",
    "FA_HD256_SPLIT_MAX": "64",
    "FA_HD256_SPLIT_COMBINE_MAX_MB": "24",
    "FA_HD256_FASTPATH": "1",
}
for _name, _value in _STABLE_KNOBS.items():
    os.environ.setdefault(_name, _value)

import torch
import torch.nn as nn

from flash_attn.cute.interface import _flash_attn_fwd


class Model(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        batch_size: int,
        cum_seq_lens_q: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        window_left: int,
        sinks: torch.Tensor | None,
        o_sf_scale: torch.Tensor | None,
        out: torch.Tensor,
    ) -> torch.Tensor:
        del workspace_buffer, cum_seq_lens_kv
        fp8 = torch.float8_e4m3fn
        if query.ndim != 3 or tuple(query.shape[1:]) != (16, 256):
            raise ValueError("query must have shape [total_q, 16, 256]")
        if kv_cache.ndim != 5 or tuple(kv_cache.shape[1:]) != (2, 1, 64, 256):
            raise ValueError("kv_cache must have shape [pages, 2, 1, 64, 256]")
        if query.dtype != fp8 or kv_cache.dtype != fp8:
            raise ValueError("query and kv_cache must use FP8 e4m3fn")
        if out.dtype != torch.bfloat16 or tuple(out.shape) != tuple(query.shape):
            raise ValueError("out must be BF16 with the same shape as query")
        if block_tables.ndim != 2 or block_tables.dtype != torch.int32:
            raise ValueError("block_tables must be rank-2 int32")
        if seq_lens.ndim != 1 or seq_lens.dtype != torch.int32:
            raise ValueError("seq_lens must be rank-1 int32")
        if cum_seq_lens_q.ndim != 1 or cum_seq_lens_q.dtype != torch.int32:
            raise ValueError("cum_seq_lens_q must be rank-1 int32")
        if batch_size != seq_lens.numel() or block_tables.shape[0] != batch_size:
            raise ValueError("batch metadata dimensions disagree")
        if cum_seq_lens_q.numel() != batch_size + 1:
            raise ValueError("cum_seq_lens_q must contain batch_size + 1 entries")
        if max_q_len <= 0 or max_kv_len <= 0:
            raise ValueError("maximum sequence lengths must be positive")
        if bmm2_scale != 1.0:
            raise ValueError("Increment R0 requires bmm2_scale == 1")
        if window_left != -1 or sinks is not None or o_sf_scale is not None:
            raise ValueError("Increment R0 supports full causal attention without sinks or output scaling")
        if not (query.is_cuda and kv_cache.is_cuda and block_tables.is_cuda and seq_lens.is_cuda):
            raise ValueError("attention tensors must be CUDA tensors")
        if query.device != kv_cache.device or query.device != out.device:
            raise ValueError("query, kv_cache, and out must share a CUDA device")

        max_pages = (max_kv_len + 63) // 64
        if max_pages > block_tables.shape[1]:
            raise ValueError("block_tables does not cover max_kv_len")

        k_view = kv_cache[:, 0].transpose(1, 2)
        v_view = kv_cache[:, 1].transpose(1, 2)
        page_table = block_tables[:, :max_pages]
        returned = _flash_attn_fwd(
            query,
            k_view,
            v_view,
            cu_seqlens_q=cum_seq_lens_q,
            cu_seqlens_k=None,
            seqused_k=seq_lens,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_pages * 64,
            page_table=page_table,
            softmax_scale=bmm1_scale,
            causal=True,
            num_splits=1,
            pack_gqa=True,
            out=out,
        )[0]
        if returned is not out:
            raise RuntimeError("Increment private API did not preserve the supplied out tensor")
        return out
