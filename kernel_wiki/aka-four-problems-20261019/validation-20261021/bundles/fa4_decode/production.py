from __future__ import annotations

import inspect
import sys

import torch
import torch.nn as nn
from flashinfer.decode import trtllm_batch_decode_with_kv_cache

# Plan A: own the trtllm-gen multi-CTA KV counter buffer instead of letting the
# functional API allocate and zero a fresh one on every call. The kernel resets
# these semaphores at the end of each launch, so one zeroed buffer is valid for
# the whole run -- that is exactly what flashinfer's own docstring says
# (utils.py:_get_trtllm_gen_multi_ctas_kv_counter_buffer) and what vLLM upstream
# does for MLA decode (vllm/v1/attention/backends/mla/flashinfer_mla.py).
# Without it every call emits a grid-(1,1,1) FillFunctor<uint8> launch right
# before the FMHA kernel, which is a baseline artifact, not algorithm work.
#
# Required size is round_up(max(batch * num_qo_heads, sm_count), 8) * 4 bytes,
# so 64 KiB covers every shape in this campaign with a wide margin. The buffer
# is created lazily on the first forward, which the harness runs as warmup
# outside both the CUDA-event window and the NCU collection window.
_COUNTER_BYTES = 1 << 16
_HAS_COUNTER_KWARG = "multi_ctas_kv_counter_buffer" in inspect.signature(
    trtllm_batch_decode_with_kv_cache
).parameters
if _HAS_COUNTER_KWARG:
    print(f"COUNTER_BUFFER_ARMED decode bytes={_COUNTER_BYTES}", flush=True)
else:
    print(
        "COUNTER_BUFFER_UNSUPPORTED decode: trtllm_batch_decode_with_kv_cache "
        "has no multi_ctas_kv_counter_buffer parameter; falling back to the "
        "per-call allocation, FillFunctor will still appear",
        file=sys.stderr,
        flush=True,
    )


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._counter_buffer: torch.Tensor | None = None

    def _counter(self, device: torch.device) -> torch.Tensor | None:
        if not _HAS_COUNTER_KWARG:
            return None
        buffer = self._counter_buffer
        if buffer is None or buffer.device != device:
            buffer = torch.zeros(_COUNTER_BYTES, dtype=torch.uint8, device=device)
            self._counter_buffer = buffer
        return buffer

    def forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        window_left: int,
        sinks: torch.Tensor | None,
        o_sf_scale: torch.Tensor | None,
        out: torch.Tensor,
        q_len_per_req: int,
    ) -> torch.Tensor:
        kwargs = {}
        counter = self._counter(query.device)
        if counter is not None:
            kwargs["multi_ctas_kv_counter_buffer"] = counter
        trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=window_left,
            sinks=sinks,
            o_sf_scale=o_sf_scale,
            out=out,
            q_len_per_req=q_len_per_req,
            **kwargs,
        )
        return out
