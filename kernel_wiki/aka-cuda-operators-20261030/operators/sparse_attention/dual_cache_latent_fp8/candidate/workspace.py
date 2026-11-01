"""Aligned caller scratch and partial-result views."""

import functools
from dataclasses import dataclass
from typing import Any
import torch
from .config import (
    _ALIGN,
    _COUNTER_OFFSET,
    _COUNTER_REGION_BYTES,
    _DESCRIPTOR_SLAB_BYTES,
    _DESCRIPTOR_SLAB_OFFSET,
    _HEAD_DIM,
    _MAX_MERGE_GROUPS,
    _PARTIAL_OFFSET,
    _PRIMED_ATTR,
)


def _align_up(num_bytes: int) -> int:
    return -(-num_bytes // _ALIGN) * _ALIGN


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


@dataclass(frozen=True)
class WorkspaceLayout:
    """Byte ``(offset, size)`` of every region carved from ``workspace_buffer``."""

    descriptor_slab: tuple[int, int]
    counters: tuple[int, int]
    partial_o: tuple[int, int]
    partial_lse: tuple[int, int]
    total_bytes: int


@functools.lru_cache(maxsize=256)
def cake_dsv4_workspace_layout(
    num_query_tokens: int, num_heads: int, num_splits: int
) -> WorkspaceLayout:
    """Deterministic carve of ``workspace_buffer`` for one launch shape."""
    tokens = _positive_int(num_query_tokens, "num_query_tokens")
    heads = _positive_int(num_heads, "num_heads")
    splits = _positive_int(num_splits, "num_splits")
    partial_elems = tokens * heads * splits
    o_bytes = (
        0
        if splits == 1
        else _align_up(partial_elems * _HEAD_DIM * torch.bfloat16.itemsize)
    )
    lse_bytes = _align_up(partial_elems * torch.float32.itemsize)
    o_offset = _PARTIAL_OFFSET
    lse_offset = o_offset + o_bytes
    return WorkspaceLayout(
        descriptor_slab=(_DESCRIPTOR_SLAB_OFFSET, _DESCRIPTOR_SLAB_BYTES),
        counters=(_COUNTER_OFFSET, _COUNTER_REGION_BYTES),
        partial_o=(o_offset, o_bytes),
        partial_lse=(lse_offset, lse_bytes),
        total_bytes=lse_offset + lse_bytes,
    )


def _workspace_bytes(workspace: torch.Tensor) -> torch.Tensor:
    if not isinstance(workspace, torch.Tensor):
        raise TypeError("workspace_buffer must be a torch.Tensor")
    if not workspace.is_contiguous():
        raise ValueError("workspace_buffer must be contiguous")
    raw = (
        workspace
        if workspace.dtype == torch.uint8 and workspace.ndim == 1
        else workspace.view(torch.uint8).reshape(-1)
    )
    if raw.data_ptr() % _ALIGN:
        raise ValueError(f"workspace_buffer must be {_ALIGN}-byte aligned")
    return raw


def _require_workspace_bytes(raw: torch.Tensor, needed: int) -> None:
    if raw.numel() < needed:
        raise ValueError(
            f"workspace_buffer requires at least {needed} bytes for this CAKE DSv4 launch, got {raw.numel()}; size it with get_cake_dsv4_workspace_bytes()"
        )


def _partial_views(
    raw: torch.Tensor,
    out_rows: torch.Tensor,
    num_query_tokens: int,
    num_heads: int,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layout = cake_dsv4_workspace_layout(num_query_tokens, num_heads, num_splits)
    _require_workspace_bytes(raw, layout.total_bytes)
    elems = num_query_tokens * num_heads * num_splits
    lse_offset = layout.partial_lse[0]
    partial_lse = raw[lse_offset : lse_offset + elems * torch.float32.itemsize].view(
        torch.float32
    )
    if num_splits == 1:
        return (out_rows.reshape(-1), partial_lse)
    o_offset = layout.partial_o[0]
    partial_o = raw[
        o_offset : o_offset + elems * _HEAD_DIM * torch.bfloat16.itemsize
    ].view(torch.bfloat16)
    return (partial_o, partial_lse)


def _counters(raw: torch.Tensor, merge_groups: int) -> torch.Tensor:
    if merge_groups > _MAX_MERGE_GROUPS:
        raise ValueError(
            f"CAKE DSv4 split-merge route needs {merge_groups} counters; the workspace counter region holds {_MAX_MERGE_GROUPS}"
        )
    _require_workspace_bytes(raw, _PARTIAL_OFFSET)
    return raw[_COUNTER_OFFSET : _COUNTER_OFFSET + merge_groups * 4].view(torch.uint32)


def _descriptor_workspace(raw: torch.Tensor, num_bytes: int) -> torch.Tensor:
    """Descriptor slab at a fixed offset of the workspace.

    The current SM103 bindings encode descriptors as immutable by-value kernel
    parameters. This caller-owned byte extent remains as a checked compatibility
    argument; it stores no retained descriptor or cross-call tensor state.
    """
    if num_bytes > _DESCRIPTOR_SLAB_BYTES:
        raise ValueError(
            f"CAKE DSv4 variant needs {num_bytes} TMA descriptor bytes; the workspace slab holds {_DESCRIPTOR_SLAB_BYTES}"
        )
    _require_workspace_bytes(raw, _PARTIAL_OFFSET)
    assert _DESCRIPTOR_SLAB_OFFSET == 0
    return raw


def _ensure_counters_zeroed(workspace: torch.Tensor, raw: torch.Tensor) -> None:
    if getattr(workspace, _PRIMED_ATTR, False):
        return
    raw[_COUNTER_OFFSET:_PARTIAL_OFFSET].zero_()
    setattr(workspace, _PRIMED_ATTR, True)
