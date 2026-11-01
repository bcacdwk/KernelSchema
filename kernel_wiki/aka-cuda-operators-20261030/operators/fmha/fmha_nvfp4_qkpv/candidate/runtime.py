"""Validate inputs and execute quantize / attention / combine stages."""

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union
import torch
import tvm_ffi
from .plan import (
    HEAD_DIM,
    NVFP4_ATTENTION_FP4PV_KWARGS,
    NVFP4_ATTENTION_FP8PV_KWARGS,
    NVFP4_ATTENTION_FUSED_PREFIX_KWARGS,
    NVFP4_ATTENTION_SEPARATE_PREFIX_KWARGS,
    NVFP4_ATTENTION_SPLIT_FP4PV_KWARGS,
    NVFP4_ATTENTION_SPLIT_FP8PV_KWARGS,
    NVFP4_VARIANT,
    PV_MMA_DTYPE,
    PV_MODES,
    PackedSegmentPlan,
    QK_MMA_DTYPE,
    QUANTIZE_QKV_KWARGS,
    QUANTIZE_QK_FP8V_KWARGS,
    QUANTIZE_STAGES,
    RAGGED_STAGE_COST,
    SPLIT_PROGRAM_COST,
    SUPPORTED_COMPUTE_CAPABILITIES,
    TileTables,
    allocate_nvfp4_workspace,
    amax_kwargs,
    build_packed_segment_plan,
    build_tile_tables,
    combine_kwargs,
    normalize_cu_seqlens,
    nvfp4_quantize_grid,
    nvfp4_workspace_shapes,
)
from .build import (
    MODULES,
    ROUTES,
    STAGES,
    VARIANTS,
    load_cake_minimax_h3_varlen_attention_module,
)


def _arch_for(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            f"MiniMax-H3 packed-varlen attention requires compute capability 10.0 or 10.3 (got {capability[0]}.{capability[1]})"
        )
    return arch


def _bind_stage(
    module_name: str, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated argument plan of ``module_name``."""
    record = MODULES[module_name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[name])
        elif name in kwargs:
            arguments.append(kwargs[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} ({kind}); host binding provides {sorted(kwargs)}"
            )
    module = load_cake_minimax_h3_varlen_attention_module(module_name)
    return (getattr(module, record["ffi_entry"]), tuple(arguments))


def _persistent_grid(
    device: torch.device, total_tiles: int, *, at_least_one: bool
) -> tuple[int, int, int]:
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    pairs = min(num_sms // 2, total_tiles)
    if at_least_one:
        pairs = max(1, pairs)
    return (2 * pairs, 1, 1)


def _check_thd(
    name: str,
    tensor: torch.Tensor,
    total_tokens: int,
    num_heads: int,
    device: torch.device,
) -> None:
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must live on {device}")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be a bfloat16 tensor")
    if tuple(tensor.shape) != (total_tokens, num_heads, HEAD_DIM):
        raise ValueError(
            f"{name} must be [T, H, {HEAD_DIM}] = {(total_tokens, num_heads, HEAD_DIM)}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def validate_minimax_h3_varlen_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: Optional[torch.Tensor],
) -> tuple[int, int, torch.device]:
    """Shape / dtype / device validation shared by both families.

    Returns ``(total_tokens, num_heads, device)``.
    """
    if query.ndim != 3:
        raise ValueError(f"query must be THD [T, H, {HEAD_DIM}]")
    total_tokens, num_heads = (int(query.shape[0]), int(query.shape[1]))
    if num_heads <= 0:
        raise ValueError("query must have at least one head")
    device = query.device
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        _check_thd(name, tensor, total_tokens, num_heads, device)
    if out is not None:
        _check_thd("out", out, total_tokens, num_heads, device)
    return (total_tokens, num_heads, device)


@dataclass(frozen=True)
class MiniMaxH3VarlenNVFP4AttentionRunner:
    """Launch the prepared NVFP4 pipeline: quantize Q/K/V, then attend.

    ``launch()`` runs the complete pipeline (for ``pv_mode="fp8"`` one device
    ``amax(V)`` reduction into the workspace, then one fused quantizer launch,
    one attention launch and, for plans with K/V-split units, the ``combine``
    launch) into the caller-owned ``out`` with no CUDA allocation and no host
    synchronization.  ``quantize()`` and ``attention()`` run the stages
    separately so callers can time them on their own.  The attention launch
    uses one of two generated programs, chosen at preparation from the plan:
    the dense ``attention`` module for plans without split units and
    ``attention_split`` otherwise (``attention_stage``).  Packed operands live in ``workspace`` (allocated at
    preparation).  Prepare a new runner when ``cu_seqlens``, shapes or tensor
    bindings change; values may change freely since quantization is part of
    every launch.  CUDA Graph capture belongs to the caller.

    The attention binding is generated with the programmatic-dependent-launch
    attribute (``cudaLaunchAttributeProgrammaticStreamSerialization``): its
    prologue overlaps the quantizer tail and ``griddepcontrol.wait`` orders the
    packed-operand reads.  For ``pv_mode="fp8"`` the quantizer binding carries
    the same attribute towards the preceding ``amax`` stage (its Q/K/V loads
    overlap the partial-max kernel's tail; the fold of the partials waits).
    All launches are enqueued on the current torch stream in order, which is
    all the handshakes require from the host.
    """

    variant: str
    pv_mode: str
    arch: str
    plan: PackedSegmentPlan
    tile_tables: TileTables
    workspace: dict[str, torch.Tensor]
    stage_kwargs: dict[str, dict]
    out: torch.Tensor
    _value: torch.Tensor
    _stages: tuple[tuple[str, Optional[Callable[..., Any]], tuple], ...]
    fused_prefix: bool = False

    def _run(self, names: Sequence[str]) -> None:
        with tvm_ffi.use_torch_stream():
            for name, entry, arguments in self._stages:
                if name not in names or entry is None:
                    continue
                if name == "attention_split" and not self.fused_prefix:
                    # A route whose FP4 split program does not fuse the prepass
                    # runs it as one extra ``prefix_only=1`` launch ahead of the
                    # attention pass.  Fused routes (sm_103a) execute both phases
                    # inside the single launch below, which is what removes the
                    # prepass kernel's own launch and pipeline-init cost from the
                    # measured forward.
                    entry(*arguments[:-1], 1)
                entry(*arguments)

    def quantize(self) -> None:
        self._run(tuple((s for s in STAGES[self.variant] if s in QUANTIZE_STAGES)))

    def attention(self) -> torch.Tensor:
        self._run(tuple((s for s in STAGES[self.variant] if s not in QUANTIZE_STAGES)))
        return self.out

    @property
    def attention_stage(self) -> str:
        """``"attention_split"`` for plans with K/V-split units, else ``"attention"``."""
        return nvfp4_attention_stage(self.tile_tables)

    def launch(self) -> torch.Tensor:
        self._run(STAGES[self.variant])
        return self.out

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        return int(self.stage_kwargs["attention"]["grid"][0])

    @property
    def route_metadata(self) -> dict[str, Any]:
        return dict(
            variant=self.variant,
            pv_mode=self.pv_mode,
            arch=self.arch,
            qk_mma_dtype=QK_MMA_DTYPE[self.variant],
            pv_mma_dtype=PV_MMA_DTYPE[self.variant],
            segment_count=self.plan.num_segments,
            packed_blocks=self.plan.PB,
            cluster_tiles=self.plan.total_clusters,
            tile_count=self.tile_tables.total_tiles,
            kv_split_units=self.tile_tables.num_combine_units,
            max_kv_splits=self.tile_tables.max_kv_splits,
            attention_variant=self.attention_stage,
            fused_prefix_prepass=self.fused_prefix,
            prefix_jobs=self.tile_tables.num_prefix_jobs,
        )


def nvfp4_attention_stage(tiles: TileTables) -> str:
    """The attention program a plan binds: ``attention_split`` when the tile
    tables hold any K/V-split unit (partial slots), else the dense ``attention``.

    Both are the same generated program specialised at build time; the dense
    variant carries no K/V-range or partial-slot code, which keeps the softmax
    block loop of unsplit units at the schedule of the single-program kernel.
    """
    return "attention_split" if int(tiles.num_partial_slots) > 0 else "attention"


def prepare_minimax_h3_varlen_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    pv_mode: str = "fp8",
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    workspace: Optional[dict[str, torch.Tensor]] = None,
    backend: str = "cake",
) -> MiniMaxH3VarlenNVFP4AttentionRunner:
    """Validate, plan and bind one NVFP4 packed-varlen attention pipeline.

    Every allocation happens here (the optional output, the int32 plan
    tables and, unless ``workspace`` is supplied, the packed operand set of
    :func:`nvfp4_workspace_shapes`); the returned runner launches with none.
    """
    if backend != "cake":
        raise ValueError("MiniMax-H3 varlen NVFP4 attention supports backend='cake'")
    if pv_mode not in PV_MODES:
        raise ValueError(f"pv_mode must be one of {PV_MODES}, got {pv_mode!r}")
    variant = NVFP4_VARIANT[pv_mode]
    total_tokens, num_heads, device = validate_minimax_h3_varlen_inputs(
        query, key, value, out
    )
    if isinstance(cu_seqlens, torch.Tensor) and (
        not cu_seqlens.is_cuda or cu_seqlens.device != device
    ):
        raise ValueError("cu_seqlens must be an int32 CUDA tensor on the query device")
    bounds = normalize_cu_seqlens(
        cu_seqlens, total_tokens=total_tokens, cu_seqlens_host=cu_seqlens_host
    )
    arch = _arch_for(device)
    route = select_route(variant, arch)
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** (-0.5)
    if out is None:
        out = torch.empty(
            (total_tokens, num_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    plan = build_packed_segment_plan(bounds, device)
    if workspace is None:
        workspace = allocate_nvfp4_workspace(num_heads, plan.PB, pv_mode, device)
    else:
        expected = nvfp4_workspace_shapes(num_heads, plan.PB, pv_mode)
        for name, (shape, dtype) in expected.items():
            tensor = workspace.get(name)
            if (
                tensor is None
                or tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or (tensor.device != device)
                or (not tensor.is_contiguous())
            ):
                raise ValueError(
                    f"workspace[{name!r}] must be a contiguous {dtype} tensor of shape {shape} on {device}"
                )
    PB = int(plan.PB)
    tiles = build_tile_tables(
        plan,
        num_heads,
        device,
        program_cost=SPLIT_PROGRAM_COST.get((pv_mode, arch), 1.0),
        # FP4 prefix normalization repeats QK and row-max work before attention.
        prefix_cost=0.5 if pv_mode == "fp4" else 0.0,
        # A route whose dense program skips a cluster tile's all-padding second
        # Q-block stage prices that tile accordingly; 1.0 elsewhere.
        ragged_stage_cost=RAGGED_STAGE_COST.get((pv_mode, arch), 1.0),
    )
    quantize_kwargs = dict(
        q=query,
        k=key,
        v=value,
        q_fp4=workspace["q_fp4"],
        k_fp4=workspace["k_fp4"],
        q_scale=workspace["q_scale"],
        k_scale=workspace["k_scale"],
    )
    if pv_mode == "fp4":
        quantize_kwargs.update(
            v_fp4_t=workspace["v_fp4_t"],
            v_scale_lo=workspace["v_scale_lo"],
            v_scale_hi=workspace["v_scale_hi"],
        )
    else:
        quantize_kwargs.update(
            v_fp8=workspace["v_fp8"],
            v_amax=workspace["v_amax"],
            v_amax_partial=workspace["v_amax_partial"],
        )
    quantize_kwargs.update(
        block_token=plan.block_token,
        block_valid=plan.block_valid,
        heads=num_heads,
        PB=PB,
        grid=nvfp4_quantize_grid(num_heads, PB),
    )
    assert tuple(quantize_kwargs) == (
        QUANTIZE_QKV_KWARGS if pv_mode == "fp4" else QUANTIZE_QK_FP8V_KWARGS
    )
    stage_kwargs: dict[str, dict] = {}
    if pv_mode == "fp8":
        stage_kwargs["amax"] = amax_kwargs(value, workspace)
    stage_kwargs["quantize"] = quantize_kwargs
    total_tiles = tiles.total_tiles
    attention_common = dict(
        Q=workspace["q_fp4"],
        K=workspace["k_fp4"],
        SFQ=workspace["q_scale"],
        SFK=workspace["k_scale"],
        O=out,
        cl_head=tiles.cl_head,
        cl_seg_begin=tiles.cl_seg_begin,
        cl_seg_len=tiles.cl_seg_len,
        cl_kv_base=tiles.cl_kv_base,
        cl_q_block=tiles.cl_q_block,
        total_clusters=int(plan.total_clusters),
        heads=num_heads,
        PB=PB,
        softmax_scale_log2=float(softmax_scale) / math.log(2.0),
        grid=_persistent_grid(device, total_tiles, at_least_one=True),
    )
    attention_split_extra = dict(
        cl_kv_begin=tiles.cl_kv_begin,
        cl_kv_blocks=tiles.cl_kv_blocks,
        cl_ws_slot=tiles.cl_ws_slot,
        partial_O=tiles.partial_O,
        partial_ML=tiles.partial_ML,
        num_tiles=total_tiles,
    )
    dense_names: tuple[str, ...]
    split_names: tuple[str, ...]
    fused_prefix = pv_mode == "fp4" and bool(route.get("fused_prefix_prepass", False))
    if pv_mode == "fp4":
        pv_operands = dict(
            Vt=workspace["v_fp4_t"],
            SFVtLo=workspace["v_scale_lo"],
            SFVtHi=workspace["v_scale_hi"],
        )
        dense_names = NVFP4_ATTENTION_FP4PV_KWARGS
        split_names = NVFP4_ATTENTION_SPLIT_FP4PV_KWARGS
        # Appended to the split binding below, once the prepass tables exist.
        prefix_tail = (
            NVFP4_ATTENTION_FUSED_PREFIX_KWARGS
            if fused_prefix
            else NVFP4_ATTENTION_SEPARATE_PREFIX_KWARGS
        )
    else:
        pv_operands = dict(V=workspace["v_fp8"], v_amax=workspace["v_amax"])
        dense_names, split_names = (
            NVFP4_ATTENTION_FP8PV_KWARGS,
            NVFP4_ATTENTION_SPLIT_FP8PV_KWARGS,
        )
        prefix_tail = ()
    attention_kwargs = {**attention_common, **pv_operands}
    attention_split_kwargs = {
        **attention_common,
        **attention_split_extra,
        **pv_operands,
    }
    assert tuple(attention_kwargs) == dense_names
    assert tuple(attention_split_kwargs) == split_names
    stage_kwargs["attention"] = attention_kwargs
    stage_kwargs["attention_split"] = attention_split_kwargs
    stage_kwargs["combine"] = combine_kwargs(
        tiles.partial_O,
        tiles.partial_ML,
        tiles.combine_table,
        tiles.seg_begin,
        tiles.seg_len,
        out,
        num_heads,
        tiles.num_combine_units,
    )
    assert tuple(stage_kwargs) == STAGES[variant]
    attention_stage = nvfp4_attention_stage(tiles)
    stages: list[tuple[str, Optional[Callable[..., Any]], tuple]] = []
    for stage in STAGES[variant]:
        skipped = (
            total_tiles == 0
            or (stage == "combine" and tiles.num_combine_units == 0)
            or (stage in ("attention", "attention_split") and stage != attention_stage)
        )
        if skipped:
            stages.append((stage, None, ()))
            continue
        if stage == "attention_split" and pv_mode == "fp4":
            prefix_stride = max(plan.seg_blocks)
            workspace["prefix_max"] = torch.empty(
                (num_heads * PB * 128, prefix_stride),
                dtype=torch.float32,
                device=device,
            )
            stage_kwargs[stage]["prefix_max"] = workspace["prefix_max"]
            stage_kwargs[stage]["prefix_stride"] = prefix_stride
            if fused_prefix:
                stage_kwargs[stage].update(
                    pf_tile=tiles.pf_tile,
                    pf_total=tiles.pf_total,
                    pf_flag=tiles.pf_flag,
                    pf_info=tiles.pf_info,
                )
            else:
                stage_kwargs[stage]["prefix_only"] = 0
            assert tuple(stage_kwargs[stage]) == (
                split_names + ("prefix_max", "prefix_stride") + prefix_tail
            )
        entry, arguments = _bind_stage(route["modules"][stage], stage_kwargs[stage])
        stages.append((stage, entry, arguments))
    return MiniMaxH3VarlenNVFP4AttentionRunner(
        variant,
        pv_mode,
        arch,
        plan,
        tiles,
        workspace,
        stage_kwargs,
        out,
        value,
        tuple(stages),
        fused_prefix,
    )


def minimax_h3_varlen_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    pv_mode: str = "fp8",
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    backend: str = "cake",
) -> torch.Tensor:
    runner = prepare_minimax_h3_varlen_nvfp4_attention(
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
    return runner.launch()


def route_name(variant: str, arch: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown MiniMax-H3 varlen attention variant {variant!r}")
    return f"{variant}__{arch}"


def select_route(variant: str, arch: str) -> dict[str, Any]:
    """Return the registered route record for ``variant`` on ``arch``."""
    record = ROUTES.get(route_name(variant, arch))
    if record is None:
        raise NotImplementedError(
            f"The generated MiniMax-H3 packed-varlen {variant} attention program for {arch} is not registered in this checkout yet (see flashinfer-ai/flashinfer#4532)"
        )
    if tuple(record["stages"]) != STAGES[variant]:
        raise RuntimeError(
            f"registered route {route_name(variant, arch)!r} has stages {tuple(record['stages'])!r}, expected {STAGES[variant]!r}"
        )
    return record
