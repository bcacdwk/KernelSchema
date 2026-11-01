from dataclasses import dataclass
import math
from typing import Optional, Sequence
import torch
import tvm_ffi
from functools import cache

from .plan import *
from .plan import _m_tiles
from pathlib import Path
import hashlib, json, os, subprocess

ROOT = Path(__file__).resolve().parent
MODULES = {
    "cake_nvfp4_mla_decode_sm_103a": {
        "arch": "sm_103a",
        "main": {
            "sources": ["launch.cu", "kernel.cu"],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "QS"],
                ["tma_buffer", "KV"],
                ["tma_buffer", "KVS"],
                ["tma_buffer", "OT"],
                ["tma_buffer", "PO"],
                ["buffer", "O"],
                ["buffer", "LSE"],
                ["buffer", "partial_o"],
                ["buffer", "partial_lse"],
                ["buffer", "work_table"],
                ["buffer", "unit_first"],
                ["buffer", "page_table"],
                ["buffer", "seq_lens"],
                ["buffer", "q_indptr"],
                ["buffer", "sinks"],
                ["parameter", "rows_total"],
                ["parameter", "num_heads"],
                ["parameter", "q_len"],
                ["parameter", "max_pages"],
                ["parameter", "max_splits"],
                ["parameter", "scale_log2"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
        },
        "reduce": {
            "sources": ["combine_launch.cu", "combine.cu"],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "o"],
                ["buffer", "lse"],
                ["buffer", "partial_o"],
                ["buffer", "partial_lse"],
                ["buffer", "row_splits"],
                ["parameter", "total_q"],
                ["parameter", "num_heads"],
                ["parameter", "max_splits"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
        },
    }
}
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


@cache
def load_cake_nvfp4_mla_decode_module(name, stage):
    from torch.utils import cpp_extension
    import tvm_ffi.libinfo

    physical = MODULES[name][stage]
    flags = CUDA_FLAGS + physical["compile_flags"]
    compiler = subprocess.check_output(
        [str(Path(cpp_extension.CUDA_HOME) / "bin/nvcc"), "--version"], text=True
    )
    sources = {p.name: p.read_text() for p in sorted(ROOT.iterdir()) if p.is_file()}
    identity = json.dumps(
        [
            sources,
            stage,
            flags,
            compiler,
            torch.__version__,
            torch.version.cuda,
            getattr(tvm_ffi, "__version__", None),
        ],
        sort_keys=True,
    )
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    root = Path(
        os.environ.get("AKA_SF_BUILD_CACHE", Path.home() / ".cache/aka-candidates")
    ) / ("mla_nvfp4_" + key)
    root.mkdir(parents=True, exist_ok=True)
    library = cpp_extension.load(
        name="aka_mla_" + stage + "_" + key,
        sources=[str(ROOT / p) for p in physical["sources"]],
        extra_cuda_cflags=flags,
        extra_cflags=["-O3", "-std=c++17"],
        extra_include_paths=[
            str(ROOT),
            tvm_ffi.libinfo.find_include_path(),
            tvm_ffi.libinfo.find_dlpack_include_path(),
        ],
        extra_ldflags=["-lcuda"],
        build_directory=str(root),
        is_python_module=False,
        verbose=False,
    )
    return tvm_ffi.load_module(library)


def _carve(flat: torch.Tensor, layout: dict, name: str, dtype, shape):
    offset, nbytes = layout[name]
    return flat[offset : offset + nbytes].view(dtype).view(shape)


@dataclass(frozen=True)
class NVFP4MLADecodeRunner:
    """Launch the prepared decode (and, when planned, the split-KV combine).

    Calling the runner or ``launch()`` writes the caller-owned output and LSE
    with no CUDA allocation or host synchronization and returns ``out`` (or
    ``(out, lse)`` when prepared with ``return_lse=True``). Prepare a new
    runner when sequence lengths, bindings or input values change.
    """

    module_name: str
    plan: WorkPlan
    main_kwargs: dict
    reduce_kwargs: Optional[dict]
    out: torch.Tensor
    lse: torch.Tensor
    return_lse: bool
    main_entry: object
    main_arguments: tuple
    reduce_entry: object
    reduce_arguments: tuple

    def launch(self):
        with tvm_ffi.use_torch_stream():
            self.main_entry(*self.main_arguments)
            if self.reduce_entry is not None:
                self.reduce_entry(*self.reduce_arguments)
        return (self.out, self.lse) if self.return_lse else self.out

    __call__ = launch


def bind_decode_payload(
    arch: str,
    plan: WorkPlan,
    main_kwargs: dict,
    reduce_kwargs: Optional[dict],
    out: torch.Tensor,
    lse: torch.Tensor,
    return_lse: bool,
) -> NVFP4MLADecodeRunner:
    """Bind the prepared buffers to the generated physical argument order."""
    module_name = select_module(arch)
    record = MODULES[module_name]

    def arguments(kwargs, physical):
        grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
        return tuple(
            (
                grid[name] if kind == "grid" else kwargs[name]
                for kind, name in physical["arg_plan"]
            )
        )

    main_module = load_cake_nvfp4_mla_decode_module(module_name, "main")
    main_entry = getattr(main_module, record["main"]["ffi_entry"])
    main_arguments = arguments(main_kwargs, record["main"])
    reduce_entry = None
    reduce_arguments = ()
    if reduce_kwargs is not None:
        reduce_module = load_cake_nvfp4_mla_decode_module(module_name, "reduce")
        reduce_entry = getattr(reduce_module, record["reduce"]["ffi_entry"])
        reduce_arguments = arguments(reduce_kwargs, record["reduce"])
    return NVFP4MLADecodeRunner(
        module_name,
        plan,
        main_kwargs,
        reduce_kwargs,
        out,
        lse,
        return_lse,
        main_entry,
        main_arguments,
        reduce_entry,
        reduce_arguments,
    )


def _check_u8(name: str, tensor: torch.Tensor, shape: tuple) -> None:
    if tensor.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise TypeError(f"{name} must be packed uint8 (or float8_e4m3fn scales)")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def validate_nvfp4_mla_decode_inputs(
    query,
    query_scale,
    kv_cache,
    kv_scale,
    block_tables,
    seq_lens,
    *,
    sinks=None,
    out=None,
    lse=None,
) -> tuple:
    """Shape / dtype validation shared by ``prepare``; returns ``(batch, H, num_pages, q_len)``.

    Device placement and compute capability are checked separately so this
    part runs on host tensors too.
    """
    if query.ndim != 3 or query.shape[-1] != ROW_BYTES:
        raise ValueError(
            f"query must be [total_q, num_heads, {ROW_BYTES}] packed E2M1 bytes"
        )
    total_q, num_heads = (int(query.shape[0]), int(query.shape[1]))
    if block_tables.ndim != 2 or block_tables.dtype != torch.int32:
        raise ValueError("block_tables must be an int32 [batch, max_pages] tensor")
    batch = int(block_tables.shape[0])
    if batch <= 0 or num_heads <= 0 or total_q < batch or total_q % batch:
        raise ValueError(
            "query rows must be batch * q_len with the same number of query tokens per request"
        )
    q_len = total_q // batch
    if total_q * num_heads < ROWS_PER_TILE:
        raise ValueError(
            f"the decode kernel loads query rows in tiles of {ROWS_PER_TILE}: batch * q_len * num_heads must be >= {ROWS_PER_TILE} (got {total_q * num_heads})"
        )
    if seq_lens.shape != (batch,) or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be an int32 [batch] tensor")
    if kv_cache.ndim != 3 or tuple(kv_cache.shape[1:]) != (PAGE_SIZE, ROW_BYTES):
        raise ValueError(
            f"kv_cache must be [num_pages, {PAGE_SIZE}, {ROW_BYTES}] packed E2M1 bytes"
        )
    num_pages = int(kv_cache.shape[0])
    _check_u8("query", query, (total_q, num_heads, ROW_BYTES))
    _check_u8("query_scale", query_scale, (total_q, num_heads, SF_ROW_BYTES))
    _check_u8("kv_cache", kv_cache, (num_pages, PAGE_SIZE, ROW_BYTES))
    _check_u8("kv_scale", kv_scale, (num_pages, PAGE_SIZE, SF_ROW_BYTES))
    if sinks is not None and (
        sinks.shape != (num_heads,) or sinks.dtype != torch.float32
    ):
        raise ValueError("sinks must be a float32 [num_heads] tensor")
    if out is not None and (
        out.shape != (total_q, num_heads, HEAD_DIM) or out.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"out must be a bfloat16 [total_q, num_heads, {HEAD_DIM}] tensor"
        )
    if lse is not None and (
        lse.shape != (total_q, num_heads) or lse.dtype != torch.float32
    ):
        raise ValueError("lse must be a float32 [total_q, num_heads] tensor")
    return (batch, num_heads, num_pages, q_len)


def prepare_nvfp4_batch_decode_with_kv_cache_mla(
    query,
    query_scale,
    kv_cache,
    kv_scale,
    block_tables,
    seq_lens,
    workspace_buffer,
    *,
    sm_scale,
    sinks=None,
    out=None,
    lse=None,
    return_lse=False,
    seq_lens_cpu=None,
    backend="cake",
    schedule="auto",
    tiles_per_split=None,
):
    """Plan and bind one NVFP4 DeepSeek-V4 paged MQA decode batch.

    Host planning (one device-to-host copy of ``seq_lens`` unless
    ``seq_lens_cpu`` is given) and every allocation happen here; the returned
    runner launches with neither. ``schedule`` / ``tiles_per_split`` expose the
    host work-plan policy for tests and benchmarks.
    """
    if backend != "cake":
        raise ValueError("NVFP4 MLA decode supports backend='cake'")
    batch, num_heads, num_pages, q_len = validate_nvfp4_mla_decode_inputs(
        query,
        query_scale,
        kv_cache,
        kv_scale,
        block_tables,
        seq_lens,
        sinks=sinks,
        out=out,
        lse=lse,
    )
    tensors = [query, query_scale, kv_cache, kv_scale, block_tables, seq_lens]
    tensors += [t for t in (sinks, out, lse, workspace_buffer) if t is not None]
    device = query.device
    if not all((t.is_cuda and t.device == device for t in tensors)):
        raise ValueError("Expected all tensors on one CUDA device")
    if not all((t.is_contiguous() for t in tensors)):
        raise ValueError("Expected contiguous tensors")
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            f"NVFP4 MLA decode requires compute capability 10.0 or 10.3 (got {capability[0]}.{capability[1]})"
        )
    if seq_lens_cpu is None:
        seq_lens_cpu = seq_lens.cpu()
    kv_lens = [int(v) for v in seq_lens_cpu.tolist()]
    if len(kv_lens) != batch:
        raise ValueError("seq_lens_cpu must hold one length per request")
    max_pages = int(block_tables.shape[1])
    if min(kv_lens) < q_len or max(kv_lens) > max_pages * PAGE_SIZE:
        raise ValueError(
            f"every request needs q_len ({q_len}) <= seq_len <= max_pages * {PAGE_SIZE}"
        )
    total_q = batch * q_len
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    plan = build_work_plan(
        kv_lens,
        num_heads=num_heads,
        num_sms=num_sms,
        q_len=q_len,
        enable_sink=sinks is not None,
        schedule=schedule,
        tiles_per_split=tiles_per_split,
    )
    layout = workspace_layout(plan, batch=batch, num_heads=num_heads, q_len=q_len)
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(
            f"workspace_buffer needs {layout['total']} bytes for this batch ({plan.max_splits} KV splits), got {flat.numel()}"
        )
    if out is None:
        out = torch.empty(
            (total_q, num_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    if lse is None:
        lse = torch.empty((total_q, num_heads), dtype=torch.float32, device=device)
    max_splits = plan.max_splits
    rows_total = total_q * num_heads
    if max_splits > 1 and max_splits * rows_total * HEAD_DIM > 2**31 - 1:
        raise ValueError(
            f"partial_o [{max_splits}, {total_q}, {num_heads}, {HEAD_DIM}] exceeds the int32 element-offset range of the decode / combine kernels; use fewer splits or a smaller batch"
        )
    o_shape, lse_shape = partial_shapes(plan, total_q=total_q, num_heads=num_heads)
    partial_o = _carve(flat, layout, "partial_o", torch.bfloat16, o_shape)
    partial_lse = _carve(flat, layout, "partial_lse", torch.float32, lse_shape)
    work_table = _carve(
        flat, layout, "work_table", torch.int32, (plan.num_items, ITEM_FIELDS)
    )
    unit_first = _carve(flat, layout, "unit_first", torch.int32, (plan.num_units + 1,))
    row_splits = _carve(flat, layout, "row_splits", torch.int32, (total_q,))
    q_indptr = _carve(flat, layout, "q_indptr", torch.int32, (batch + 1,))
    sink_buffer = _carve(flat, layout, "sinks", torch.float32, (num_heads,))
    partial_o.zero_()
    partial_lse.fill_(float("-inf"))
    work_table.copy_(work_table_rows(plan))
    unit_first.copy_(torch.tensor(plan.unit_first, dtype=torch.int32))
    q_indptr_host = list(range(0, total_q + 1, q_len))
    if max_splits > 1:
        row_splits.copy_(
            torch.tensor(
                token_splits(
                    plan,
                    q_indptr_host,
                    q_len=q_len,
                    num_heads=num_heads,
                    total_q=total_q,
                ),
                dtype=torch.int32,
            )
        )
    else:
        row_splits.fill_(1)
    q_indptr.copy_(torch.tensor(q_indptr_host, dtype=torch.int32))
    if sinks is None:
        sink_buffer.zero_()
    else:
        sink_buffer.copy_(sinks)
    main_kwargs = dict(
        Q=query.view(total_q * num_heads, ROW_BYTES),
        QS=query_scale.view(torch.uint8).view(total_q * num_heads, SF_ROW_BYTES),
        KV=kv_cache.view(num_pages * PAGE_SIZE, ROW_BYTES),
        KVS=kv_scale.view(torch.uint8).view(num_pages * PAGE_SIZE, SF_ROW_BYTES),
        OT=out.view(rows_total, HEAD_DIM),
        PO=(
            partial_o.view(max_splits * rows_total, HEAD_DIM)
            if max_splits > 1
            else partial_o
        ),
        O=out,
        LSE=lse,
        partial_o=partial_o,
        partial_lse=partial_lse,
        work_table=work_table,
        unit_first=unit_first,
        page_table=block_tables,
        seq_lens=seq_lens,
        q_indptr=q_indptr,
        sinks=sink_buffer,
        rows_total=rows_total,
        num_heads=num_heads,
        q_len=q_len,
        max_pages=max_pages,
        max_splits=max_splits,
        scale_log2=float(sm_scale) * LOG2E,
        grid=plan.grid,
    )
    reduce_kwargs = None
    if max_splits > 1:
        reduce_kwargs = dict(
            o=out,
            lse=lse,
            partial_o=partial_o,
            partial_lse=partial_lse,
            row_splits=row_splits,
            total_q=total_q,
            num_heads=num_heads,
            max_splits=max_splits,
            grid=(total_q, math.ceil(num_heads / REDUCE_HEADS_PER_CTA), 1),
        )
    assert tuple(main_kwargs) == MAIN_KWARGS
    assert reduce_kwargs is None or tuple(reduce_kwargs) == REDUCE_KWARGS
    return bind_decode_payload(
        arch, plan, main_kwargs, reduce_kwargs, out, lse, return_lse
    )


def select_module(arch):
    for name, record in MODULES.items():
        if record["arch"] == arch:
            return name
    raise ValueError(f"Unsupported architecture: {arch}")
