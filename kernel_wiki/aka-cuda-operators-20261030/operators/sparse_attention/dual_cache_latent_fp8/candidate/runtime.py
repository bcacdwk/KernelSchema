"""Route selection, argument binding and dispatch."""

import functools
from typing import Any, Literal, Optional, Union
import torch
from .config import (
    _DESCRIPTOR_WORKSPACE_NAME,
    _FP8_H64_M64_MIN_TOKENS,
    _FP8_PERSISTENT_LANE_GATHER_MIN_TOKENS,
    _GRID_NAMES,
    _HEAD_DIM,
    _MAX_MERGE_GROUPS,
    _PARTIAL_OFFSET,
    _RAGGED_ONLY_ROUTES,
    _SCALAR_ALIASES,
    _SCALAR_VALUE_NAMES,
    _TENSOR_VALUE_NAMES,
    _TMA_SOURCE_ALIASES,
    _dense_offsets_cache,
    _scale_cache,
    _scale_cache_lock,
)
from .workspace import (
    _counters,
    _descriptor_workspace,
    _ensure_counters_zeroed,
    _partial_views,
    _require_workspace_bytes,
    _workspace_bytes,
    cake_dsv4_workspace_layout,
)
from .build import get_cake_dsv4_module, get_cake_dsv4_spec, get_compute_capability
from .metadata import resolve_cake_dsv4_sparse_metadata


@functools.lru_cache(maxsize=32)
def _target_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(f"CAKE DSv4 requires CUDA tensors, got {device}")
    major, minor = get_compute_capability(device)
    return f"sm_{major}{minor}a"


def _variant_module(variant: str, *, arch: str):
    return get_cake_dsv4_module(variant, arch=arch)


def _stream_ptr(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def _fp8_persistent_program(num_query_tokens: int) -> str:
    if num_query_tokens < _FP8_PERSISTENT_LANE_GATHER_MIN_TOKENS:
        return "fp8_h128_prefill_source_persistent_uniform"
    return "fp8_h128_prefill_source_persistent"


def _fp8_h64_uses_persistent_body(sparse_topk: int, num_query_tokens: int) -> bool:
    """CAKE-624 FP8/H64 rule (Cake seed ``portfolio_v34.uses_persistent_body``)."""
    full_tiles = sparse_topk // 128
    if num_query_tokens <= 12:
        return full_tiles >= 3
    return full_tiles >= 2 or num_query_tokens >= 128


def canonical_arg_name(kind: str, name: str) -> str:
    """Map a registration ``arg_plan`` entry onto the host value vocabulary."""
    if kind == "tma_buffer":
        return _TMA_SOURCE_ALIASES.get(name, name)
    if kind == "parameter":
        return _SCALAR_ALIASES.get(name, name)
    return name


def is_bindable_arg(kind: str, name: str) -> bool:
    """Whether the host can supply this ``arg_plan`` entry at all."""
    canonical = canonical_arg_name(kind, name)
    if kind in ("buffer", "tma_buffer"):
        return canonical in _TENSOR_VALUE_NAMES
    if kind == "parameter":
        return canonical in _SCALAR_VALUE_NAMES
    if kind == "workspace":
        return name == _DESCRIPTOR_WORKSPACE_NAME
    if kind == "grid":
        return name in _GRID_NAMES
    return False


def _grid_values(grid: tuple[int, int, int]) -> dict[str, int]:
    if len(grid) != 3 or any(
        (isinstance(g, bool) or not isinstance(g, int) or g < 1 for g in grid)
    ):
        raise ValueError(f"launch grid must be three positive ints, got {grid!r}")
    return dict(zip(_GRID_NAMES, grid, strict=True))


@functools.lru_cache(maxsize=256)
def _prepared_launch(variant: str, arch: str):
    """Resolve fixed argument names and the FFI callable once, with no tensors.

    The typed C++ entry still validates every invocation's tensor ABI. Python
    forwarding already constructs scalar metadata from validated tensor shapes.
    Only the code/contract and tuple permutation are cached, never tensor views,
    workspace pointers, input values or computed outputs.
    """
    from operator import itemgetter

    contract = get_cake_dsv4_spec(variant, arch=arch)
    plan = contract["arg_plan"]
    tma_bytes = int(contract.get("tma_workspace_bytes", 0) or 0)
    entry = getattr(_variant_module(variant, arch=arch), contract["entry"])
    keys = tuple(
        (
            canonical_arg_name(kind, name)
            for kind, name in plan
            if kind not in ("grid", "workspace")
        )
    )
    assert len(keys) > 1
    indices = []
    value_index = 0
    for kind, name in plan:
        if not is_bindable_arg(kind, name):
            raise ValueError(f"Unresolved host ABI: {variant}/{kind}/{name}")
        if kind == "grid":
            indices.append(len(keys) + _GRID_NAMES.index(name))
        elif kind == "workspace":
            indices.append(len(keys) + 3)
        else:
            indices.append(value_index)
            value_index += 1
    return (entry, itemgetter(*keys), itemgetter(*indices), tma_bytes)


def _launch_variant(variant, *, arch, grid, workspace_raw, values):
    entry, get_values, reorder, tma_bytes = _prepared_launch(variant, arch)
    _grid_values(grid)
    slab = _descriptor_workspace(workspace_raw, tma_bytes) if tma_bytes else None
    return entry(*reorder(get_values(values) + grid + (slab,)))


def _route(
    *,
    arch: str,
    dtype: torch.dtype,
    num_heads: int,
    max_q_len: int,
    ragged: bool,
    sparse_topk: int,
    batch_size: int,
    compressed_page_size: int,
    num_query_tokens: int,
) -> str:
    if max_q_len <= 0:
        raise ValueError("max_q_len must be positive")
    is_swa = sparse_topk == 128
    is_topk4x = not is_swa and compressed_page_size == 64
    is_topk128x = not is_swa and compressed_page_size == 2
    if dtype != torch.float8_e4m3fn:
        raise ValueError(f"unsupported CAKE FP8 DSv4 dtype: {dtype}")
    if arch == "sm_103a" and num_query_tokens <= 32 and (sparse_topk >= 1024):
        return "fp8_m64_partitioned_v4"
    if num_heads == 128:
        return _fp8_persistent_program(num_query_tokens)
    if num_heads not in (8, 16, 32, 64):
        raise ValueError(f"unsupported CAKE FP8 DSv4 head count: {num_heads}")
    if (
        num_heads == 64
        and batch_size == 2
        and (max_q_len == 257)
        and ragged
        and is_topk4x
        and (sparse_topk == 640)
    ):
        return "fp8_h64_source_exact"
    if max_q_len >= 257:
        return "fp8_h64_prefill_source_persistent_m64"
    if (
        num_heads in (8, 16)
        and batch_size == 3
        and (max_q_len == 5)
        and ragged
        and (
            is_topk128x
            and sparse_topk == 260
            or (is_topk4x and sparse_topk == (192 if num_heads == 8 else 256))
        )
    ):
        return "fp8_h8_h16_source_exact"
    if num_heads == 64 and _fp8_h64_uses_persistent_body(sparse_topk, num_query_tokens):
        if num_query_tokens >= _FP8_H64_M64_MIN_TOKENS:
            return "fp8_h64_prefill_source_persistent_m64"
        return _fp8_persistent_program(num_query_tokens)
    if is_swa:
        return "fp8_lowhead_prefill"
    if num_heads == 64:
        if arch == "sm_100a" and sparse_topk >= 640:
            return "fp8_lowhead_h64_split"
        return "fp8_lowhead_h64"
    if sparse_topk > 384 and num_heads in (8, 16):
        return "fp8_h8_h16_source_exact"
    return "fp8_lowhead_one_partition" if sparse_topk <= 384 else "fp8_lowhead_prefill"


def _device_scale(
    value: Union[float, torch.Tensor], *, device: torch.device, name: str
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or value.numel() != 1:
            raise ValueError(f"{name} must be a one-element FP32 tensor")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}, got {value.device}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        return value
    device_index = device.index
    if device_index is None and device.type == "cuda":
        device_index = torch.cuda.current_device()
    key = (device.type, device_index, float(value))
    with _scale_cache_lock:
        result = _scale_cache.get(key)
        if result is None:
            result = torch.tensor([key[2]], dtype=torch.float32, device=device)
            _scale_cache[key] = result
    return result


def _dense_query_offsets(
    batch_size: int, q_len: int, *, device: torch.device
) -> torch.Tensor:
    """``cum_seq_lens_q`` for a dense ``[batch, q_len]`` query (process-lifetime constant).

    Variants that only accept ragged queries read the request boundaries
    from ``cum_seq_lens_q``; a dense query is the ragged query with every
    request ``q_len`` long. The offsets are cached per (device, batch,
    q_len) so no call path allocates and CUDA-graph replay sees a stable
    pointer.
    """
    device_index = device.index
    if device_index is None and device.type == "cuda":
        device_index = torch.cuda.current_device()
    key = (device.type, device_index, int(batch_size), int(q_len))
    with _scale_cache_lock:
        result = _dense_offsets_cache.get(key)
        if result is None:
            result = torch.arange(
                0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=device
            )
            _dense_offsets_cache[key] = result
    return result


def _dense_rows(cache: torch.Tensor, name: str, dtype: torch.dtype) -> torch.Tensor:
    if cache.dtype != dtype:
        raise ValueError(
            f"{name} dtype must match the query dtype {dtype}, got {cache.dtype}"
        )
    if cache.shape[-1] != _HEAD_DIM:
        raise ValueError(
            f"{name} must have head dim {_HEAD_DIM}, got {cache.shape[-1]}"
        )
    if not cache.is_contiguous():
        raise ValueError(
            f"{name} must be a densely packed [..., {_HEAD_DIM}] pool; backend='cake' makes no host copy, so strided page layouts are not supported"
        )
    return cache


def _int32_vector(
    tensor: torch.Tensor, name: str, device: torch.device
) -> torch.Tensor:
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    flat = tensor if tensor.ndim == 1 else tensor.reshape(-1)
    if flat.data_ptr() != tensor.data_ptr() or not flat.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous; backend='cake' makes no host copy"
        )
    return flat


class _Launcher:
    """Per-call launch context: shared values plus per-variant overrides."""

    def __init__(
        self,
        *,
        arch: str,
        workspace: torch.Tensor,
        raw: torch.Tensor,
        stream: int,
        values: dict[str, Any],
    ):
        self.arch = arch
        self.workspace = workspace
        self.raw = raw
        self.stream = stream
        self.values = values

    def variant(self, name: str, *, grid: tuple[int, int, int], **overrides: Any):
        return _launch_variant(
            name,
            arch=self.arch,
            grid=grid,
            workspace_raw=self.raw,
            values={**self.values, **overrides},
        )

    def partials(self, num_splits: int, raw: bool = False) -> dict[str, Any]:
        if raw and self.arch == "sm_103a":
            layout = cake_dsv4_workspace_layout(
                self.values["num_query_tokens"], self.values["num_heads"], num_splits
            )
            _require_workspace_bytes(self.raw, layout.total_bytes)
            return {
                "partial_O": self.raw,
                "partial_lse": self.raw,
                "num_splits": num_splits,
            }
        partial_o, partial_lse = _partial_views(
            self.raw,
            self.values["O"],
            self.values["num_query_tokens"],
            self.values["num_heads"],
            num_splits,
        )
        return {
            "partial_O": partial_o,
            "partial_lse": partial_lse,
            "num_splits": num_splits,
        }

    def counters(self, merge_groups: int) -> torch.Tensor:
        _ensure_counters_zeroed(self.workspace, self.raw)
        return _counters(self.raw, merge_groups)

    def reduce(self, reducer: str, **overrides: Any) -> None:
        tokens = self.values["num_query_tokens"]
        heads = self.values["num_heads"]
        self.variant(reducer, grid=(tokens, heads, 1), **overrides)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def run_cake_dsv4(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: Optional[torch.Tensor],
    out: torch.Tensor,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: Optional[torch.Tensor],
    max_q_len: int,
    cum_seq_lens_q: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
    backend: Literal["cake"],
    extra_sparse_indices: Optional[torch.Tensor] = None,
    extra_sparse_topk_lens: Optional[torch.Tensor] = None,
    sparse_topk_lens_offset: int = 0,
) -> torch.Tensor:
    """Launch the CAKE DSv4 route for flattened ``query [rows, num_heads, 512]``.

    ``query`` / ``out`` may have more rows than the metadata; only the first
    ``num_query_tokens`` (metadata rows) are read and written. No device memory
    is allocated here; see the module docstring for the workspace contract.
    """
    if backend != "cake":
        raise ValueError(f"expected backend='cake', got {backend!r}")
    if query.ndim != 3:
        raise ValueError(
            f"query must be [num_tokens, num_heads, {_HEAD_DIM}], got shape {tuple(query.shape)}"
        )
    query_capacity, num_heads, head_dim = query.shape
    if head_dim != _HEAD_DIM:
        raise ValueError(f"CAKE DSv4 requires head dim {_HEAD_DIM}, got {head_dim}")
    if query.dtype != torch.float8_e4m3fn:
        raise ValueError(f"unsupported CAKE DSv4 dtype: {query.dtype}")
    if not query.is_contiguous():
        raise ValueError("query must be contiguous; backend='cake' makes no host copy")
    device = query.device
    arch = _target_arch(device)
    meta = resolve_cake_dsv4_sparse_metadata(
        sparse_indices,
        sparse_topk_lens,
        extra_sparse_indices=extra_sparse_indices,
        extra_sparse_topk_lens=extra_sparse_topk_lens,
        sparse_topk_lens_offset=sparse_topk_lens_offset,
        query_rows=query_capacity,
    )
    for tensor, name in (
        (meta.swa_indices, "sparse_indices"),
        (meta.compressed_indices, "extra_sparse_indices"),
        (meta.sparse_topk_lens, "sparse_topk_lens"),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    num_query_tokens = meta.num_query_tokens
    sparse_topk = meta.sparse_topk
    if out.dtype != torch.bfloat16:
        raise ValueError(f"out must be bfloat16, got {out.dtype}")
    if out.device != device:
        raise ValueError(f"out must be on {device}, got {out.device}")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous; backend='cake' makes no host copy")
    row_elems = num_heads * _HEAD_DIM
    if out.numel() % row_elems or out.numel() < num_query_tokens * row_elems:
        raise ValueError(
            f"out must hold at least {num_query_tokens} rows of [{num_heads}, {_HEAD_DIM}], got {tuple(out.shape)}"
        )
    query_rows = (
        query if query_capacity == num_query_tokens else query[:num_query_tokens]
    )
    out_rows = (
        out
        if out.ndim == 3
        and tuple(out.shape) == (num_query_tokens, num_heads, _HEAD_DIM)
        else out.view(-1, num_heads, _HEAD_DIM)[:num_query_tokens]
    )
    swa = _dense_rows(swa_kv_cache, "swa_kv_cache", query.dtype)
    compressed = _dense_rows(compressed_kv_cache, "compressed_kv_cache", query.dtype)
    seq_lens = _int32_vector(seq_lens, "seq_lens", device)
    batch_size = seq_lens.numel()
    ragged = cum_seq_lens_q is not None
    if ragged:
        cum_seq_lens_q = _int32_vector(cum_seq_lens_q, "cum_seq_lens_q", device)
    scale1 = _device_scale(bmm1_scale, device=device, name="bmm1_scale")
    scale2 = _device_scale(bmm2_scale, device=device, name="bmm2_scale")
    if sinks is not None:
        if (
            sinks.dtype != torch.float32
            or sinks.device != device
            or (not sinks.is_contiguous())
        ):
            raise ValueError(f"sinks must be a contiguous FP32 tensor on {device}")
    has_sinks = int(sinks is not None)
    sink_tensor = sinks if sinks is not None else scale1
    if workspace_buffer.device != device:
        raise ValueError(
            f"workspace_buffer must be on {device}, got {workspace_buffer.device}"
        )
    raw = _workspace_bytes(workspace_buffer)
    route = _route(
        arch=arch,
        dtype=query.dtype,
        num_heads=num_heads,
        max_q_len=max_q_len,
        ragged=ragged,
        sparse_topk=sparse_topk,
        batch_size=batch_size,
        compressed_page_size=compressed_kv_cache.shape[-2],
        num_query_tokens=meta.num_query_tokens,
    )
    if cum_seq_lens_q is None and route in _RAGGED_ONLY_ROUTES:
        cum_seq_lens_q = _dense_query_offsets(batch_size, max_q_len, device=device)
    is_fp8 = query.dtype == torch.float8_e4m3fn and arch != "sm_103a"
    values: dict[str, Any] = {
        "Q": query_rows.view(torch.uint8) if is_fp8 else query_rows,
        "SWA_cache": swa.view(torch.uint8) if is_fp8 else swa,
        "compressed_KV_cache": compressed.view(torch.uint8) if is_fp8 else compressed,
        "O": out_rows,
        "seq_lens": seq_lens,
        "cum_seq_lens_q": cum_seq_lens_q,
        "sinks": sink_tensor,
        "bmm1_scale": scale1,
        "bmm2_scale": scale2,
        **meta.kernel_kwargs(),
        "sparse_indices": meta.legacy_combined_table,
        "num_heads": num_heads,
        "num_head_tiles": _ceil_div(num_heads, 64),
        "has_sinks": has_sinks,
        "batch_size": batch_size,
        "max_q_len": max_q_len,
        "ragged_query": int(ragged),
        "num_splits": 1,
        "total_work_items": num_query_tokens,
    }
    launcher = _Launcher(
        arch=arch, workspace=workspace_buffer, raw=raw, stream=0, values=values
    )
    _dispatch_route(route, launcher)
    return out


def _dispatch_route(route: str, L: _Launcher) -> None:
    v = L.values
    T = v["num_query_tokens"]
    H = v["num_heads"]
    topk = v["sparse_topk"]
    if route == "fp8_m64_partitioned_v4":
        splits = 32 if topk >= 4096 and T <= 4 else 8
        needed = _PARTIAL_OFFSET + 4 * T * H * splits * (512 + 2)
        if getattr(L, "planning", False):
            L.needed = max(L.needed, needed)
            return
        _require_workspace_bytes(L.raw, needed)
        work = T * ((H + 63) // 64) * splits
        L.variant(route, grid=(work, 1, 1), split_stats=L.raw, total_work_items=work)
        return
    if route == "fp8_h8_h16_source_exact":
        L.variant(route, grid=(v["max_q_len"], H // 8 * 4, v["batch_size"]))
        return
    if route == "fp8_h64_source_exact":
        head_tiles = _ceil_div(H, 64)
        L.variant(
            route,
            grid=(v["max_q_len"], head_tiles, v["batch_size"]),
            num_head_tiles=head_tiles,
            total_work_items=v["max_q_len"] * head_tiles * v["batch_size"],
        )
        return
    if route in (
        "fp8_h128_prefill_source_persistent",
        "fp8_h128_prefill_source_persistent_uniform",
    ):
        parts = {} if L.arch == "sm_103a" else L.partials(1)
        L.variant(route, grid=(T * 2, 1, 1), total_work_items=T, **parts)
        return
    if route == "fp8_h64_prefill_source_persistent_m64":
        parts = {} if L.arch == "sm_103a" else L.partials(1)
        L.variant(route, grid=(T, 1, 1), total_work_items=T, **parts)
        return
    if route in (
        "fp8_lowhead_one_partition",
        "fp8_lowhead_h64",
        "fp8_lowhead_h64_split",
        "fp8_lowhead_prefill",
    ):
        num_splits = 2 if route in ("fp8_lowhead_h64_split",) else 1
        parts = L.partials(num_splits)
        work_factor = 2 if route in ("fp8_lowhead_prefill",) else num_splits
        total_work_items = T * work_factor
        cluster = 1 if route == "fp8_lowhead_prefill" else 2
        producer = dict(parts)
        if num_splits > 1:
            producer["O"] = parts["partial_O"]
        L.variant(
            route,
            grid=(total_work_items * cluster, 1, 1),
            total_work_items=total_work_items,
            **producer,
        )
        if route == "fp8_lowhead_h64_split":
            L.reduce("fp8_h64_split_reduce2", **parts)
        elif num_splits > 1:
            L.reduce("split_reduce", **parts)
        return
    raise RuntimeError(f"unhandled CAKE DSv4 route: {route}")


@functools.lru_cache(maxsize=256)
def _selected_workspace_bytes(
    arch, dtype, tokens, heads, topk, batch, maxq, ragged, page, device
):
    """Cache immutable geometry bytes only; no tensor, pointer, or result survives."""
    from types import SimpleNamespace

    route = _route(
        arch=arch,
        dtype=dtype,
        num_heads=heads,
        max_q_len=maxq,
        ragged=ragged,
        sparse_topk=topk,
        batch_size=batch,
        compressed_page_size=page,
        num_query_tokens=tokens,
    )

    class Plan:
        planning = True

        def __init__(self):
            self.arch = arch
            self.values = {
                "num_query_tokens": tokens,
                "num_heads": heads,
                "sparse_topk": topk,
                "batch_size": batch,
                "max_q_len": maxq,
                "Q": SimpleNamespace(device=device),
            }
            self.needed = _PARTIAL_OFFSET

        def partials(self, splits, raw=False):
            self.needed = max(
                self.needed,
                cake_dsv4_workspace_layout(tokens, heads, splits).total_bytes,
            )
            return {"partial_O": None, "partial_lse": None, "num_splits": splits}

        def counters(self, groups):
            if groups > _MAX_MERGE_GROUPS:
                raise ValueError("Too many merge groups")
            return None

        def variant(self, *args, **kwargs):
            pass

        def reduce(self, *args, **kwargs):
            pass

    plan = Plan()
    _dispatch_route(route, plan)
    return plan.needed
