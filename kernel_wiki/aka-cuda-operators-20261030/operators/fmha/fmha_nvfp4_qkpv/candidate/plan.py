"""Packed segments, LPT placement, split geometry and quantization workspace."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union
import torch
from functools import lru_cache

HEAD_DIM = 128

BLOCK_M = 128

CLUSTER_Q_BLOCKS = 4

CLUSTER_Q_ROWS = CLUSTER_Q_BLOCKS * BLOCK_M

BLOCK_N = 128

BF16_UNIT_OVERHEAD_BLOCKS = 2

UNIT_WORDS = 4

COMBINE_WORDS = 5

PARTIAL_ROWS = CLUSTER_Q_ROWS

MAX_KV_SPLITS = 8

MAX_KV_BLOCKS = 1 << 16

COMBINE_FIXED_BLOCKS = 3.0

COMBINE_BLOCKS_PER_SLOT = 0.02

KV_SPLIT_MIN_GAIN = 0.03

KV_SPLIT_MAX_WAVES = 4

SPLIT_PROGRAM_COST: dict[tuple[str, str], float] = {
    ("fp8", "sm_103a"): 1.01,
    ("fp4", "sm_100a"): 1.03,
}

# Per-K/V-block rate of a cluster tile whose *second* Q-block stage is entirely
# segment padding, as a fraction of a full tile's rate.  ``qkpv_stage1_dead`` in
# ``fp4pv_sm_103a_attention.cu`` recognises that case (``cl_q_block + 1`` is no
# longer one of the segment's Q blocks -- the one-block tail a length of 2049
# leaves in its last cluster tile) and drops the stage: one of the two QK MMAs,
# one of the two softmax passes, one of the two PV MMA pairs and one whole
# 128x128 accumulator rescale per K/V block.  What it cannot drop is the K/V
# fetch both stages share, which is why the fraction stays above 0.5.  Crediting
# a tail tile with the full price of a dense one made the planner split K/V to
# fill clusters that a cheaper tail tile already balances, and a split plan then
# pays for a prepass, a partial workspace and a whole ``combine`` kernel.
#
# Only a route whose *dense* program implements the skip may be credited, and
# only for a plan that actually ends up all-dense: one K/V-split unit anywhere
# binds the split program, which still runs both stages.  It is used for the LPT
# *placement* of such a plan and deliberately not by :func:`choose_kv_splits`
# (see there for why the program choice is not a scheduling decision).
RAGGED_STAGE_COST: dict[tuple[str, str], float] = {
    ("fp4", "sm_103a"): 0.6,
}

COMBINE_THREADS = 128

COMBINE_ROWS_PER_CTA = COMBINE_THREADS // 32

COMBINE_CTAS_PER_UNIT = PARTIAL_ROWS // COMBINE_ROWS_PER_CTA

SF_VEC = 16

QUANTIZE_SUB_TOKENS = 32

QUANTIZE_SUBS_PER_BLOCK = BLOCK_M // QUANTIZE_SUB_TOKENS

FP8_E4M3_MAX = 448.0

AMAX_CTAS = 512

AMAX_VECTOR = 16

AMAX_CHUNK = 256 * 4

PV_MODES = ("fp4", "fp8")

NVFP4_VARIANT = {"fp4": "nvfp4_fp4pv", "fp8": "nvfp4_fp8pv"}

QUANTIZE_STAGES = ("amax", "quantize")

SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

QK_MMA_DTYPE = {
    "bf16": "f16 bf16 x bf16",
    "nvfp4_fp4pv": "mxf4nvf4 e2m1 x e2m1, ue4m3 block-16 scales",
    "nvfp4_fp8pv": "mxf4nvf4 e2m1 x e2m1, ue4m3 block-16 scales",
}

PV_MMA_DTYPE = {
    "bf16": "f16 bf16 x bf16",
    "nvfp4_fp4pv": "mxf4nvf4 e2m1 x e2m1, ue4m3 block-16 scales (tile-max P scale)",
    "nvfp4_fp8pv": "f8f6f4 e4m3 x e4m3, per-tensor v_scale",
}

COMBINE_KWARGS = (
    "partial_O",
    "partial_ML",
    "combine_table",
    "seg_begin",
    "seg_len",
    "O",
    "num_heads",
    "grid",
)

QUANTIZE_COMMON_KWARGS = ("q", "k", "v", "q_fp4", "k_fp4", "q_scale", "k_scale")

QUANTIZE_TAIL_KWARGS = ("block_token", "block_valid", "heads", "PB", "grid")

QUANTIZE_QKV_KWARGS = (
    QUANTIZE_COMMON_KWARGS
    + ("v_fp4_t", "v_scale_lo", "v_scale_hi")
    + QUANTIZE_TAIL_KWARGS
)

QUANTIZE_QK_FP8V_KWARGS = (
    QUANTIZE_COMMON_KWARGS
    + ("v_fp8", "v_amax", "v_amax_partial")
    + QUANTIZE_TAIL_KWARGS
)

AMAX_KWARGS = ("v", "v_amax_partial", "num_vectors", "num_chunks", "grid")

NVFP4_ATTENTION_COMMON_KWARGS = (
    "Q",
    "K",
    "SFQ",
    "SFK",
    "O",
    "cl_head",
    "cl_seg_begin",
    "cl_seg_len",
    "cl_kv_base",
    "cl_q_block",
    "total_clusters",
    "heads",
    "PB",
    "softmax_scale_log2",
    "grid",
)

NVFP4_ATTENTION_SPLIT_KWARGS = (
    "cl_kv_begin",
    "cl_kv_blocks",
    "cl_ws_slot",
    "partial_O",
    "partial_ML",
    "num_tiles",
)

NVFP4_ATTENTION_FP4PV_KWARGS = NVFP4_ATTENTION_COMMON_KWARGS + (
    "Vt",
    "SFVtLo",
    "SFVtHi",
)

NVFP4_ATTENTION_FP8PV_KWARGS = NVFP4_ATTENTION_COMMON_KWARGS + ("V", "v_amax")

NVFP4_ATTENTION_SPLIT_FP4PV_KWARGS = (
    NVFP4_ATTENTION_COMMON_KWARGS
    + NVFP4_ATTENTION_SPLIT_KWARGS
    + ("Vt", "SFVtLo", "SFVtHi")
)

NVFP4_ATTENTION_SPLIT_FP8PV_KWARGS = (
    NVFP4_ATTENTION_COMMON_KWARGS + NVFP4_ATTENTION_SPLIT_KWARGS + ("V", "v_amax")
)

# FP4 split tail of a route that runs the QK-only prepass as phase 0 of the
# attention launch (sm_103a): the prepass tile table and its length, the zeroed
# handoff flags, and the per-slot ``{flag base, chunk, splits - 1}`` triple from
# which the kernel derives both the flags a chunk waits on and the flags its own
# prepass tile publishes.
NVFP4_ATTENTION_FUSED_PREFIX_KWARGS = ("pf_tile", "pf_total", "pf_flag", "pf_info")

# FP4 split tail of a route that still launches the prepass separately
# (sm_100a): the same maxima table plus the ``prefix_only`` phase selector.
NVFP4_ATTENTION_SEPARATE_PREFIX_KWARGS = ("prefix_only",)


def normalize_cu_seqlens(
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    total_tokens: Optional[int] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> tuple[int, ...]:
    """Validate ``cu_seqlens`` and return its host copy as a tuple of ints.

    When ``cu_seqlens`` is a CUDA tensor and no ``cu_seqlens_host`` is given,
    the values are copied to the host (one synchronization).  Pass the host
    copy explicitly to keep preparation free of device synchronization.
    """
    if isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
            raise ValueError("cu_seqlens must be a one-dimensional int32 tensor")
        if cu_seqlens_host is None:
            cu_seqlens_host = cu_seqlens.tolist()
        elif len(cu_seqlens_host) != int(cu_seqlens.shape[0]):
            raise ValueError("cu_seqlens_host must have the same length as cu_seqlens")
    elif cu_seqlens_host is None:
        cu_seqlens_host = cu_seqlens
    bounds = tuple((int(v) for v in cu_seqlens_host))
    if len(bounds) < 2 or bounds[0] != 0:
        raise ValueError("cu_seqlens must have at least two entries and start at zero")
    if any((b < a for a, b in zip(bounds, bounds[1:], strict=False))):
        raise ValueError("cu_seqlens must be non-decreasing")
    if total_tokens is not None and bounds[-1] != total_tokens:
        raise ValueError(
            f"cu_seqlens[-1]={bounds[-1]} must equal the THD token extent {total_tokens}"
        )
    return bounds


def _table(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values if values else [0], dtype=torch.int32, device=device)


def bf16_grid_clusters(device: torch.device) -> int:
    """Persistent-grid capacity of ``device`` in 2-CTA clusters (``num_SMs / 2``)."""
    return max(1, torch.cuda.get_device_properties(device).multi_processor_count // 2)


def assign_unit_slots(unit_costs: Sequence[float], num_clusters: int) -> list[int]:
    """Longest-processing-time-first placement of units into persistent-grid slots.

    Slot ``k * num_clusters + i`` is the ``k``-th unit of cluster ``i``.  The
    partial tail round receives the cheapest units, and within every full
    round the clusters that also own a tail unit receive that round's cheapest
    units.  Returns ``slot -> unit`` (a permutation).  Equal costs keep the
    enumeration order within a round (both sorts are stable), so a uniform
    plan reproduces the plain enumeration table exactly: consecutive slots
    stream one head's K/V.  Mirrors the production planner's
    ``assign_unit_slots``.
    """
    total = len(unit_costs)
    if total == 0:
        return []
    G = min(num_clusters, total)
    order = sorted(range(total), key=lambda u: -unit_costs[u])
    tail = total % G
    slots = [0] * total
    full_units = order[: total - tail]
    for k in range(len(full_units) // G):
        chunk = full_units[k * G : (k + 1) * G]
        for i, unit in enumerate(sorted(chunk, key=lambda u: unit_costs[u])):
            slots[k * G + i] = unit
    for i, unit in enumerate(order[total - tail :] if tail else []):
        slots[total // G * G + i] = unit
    return slots


def assign_prefix_jobs(
    job_costs: Sequence[float], cluster_loads: Sequence[float], num_clusters: int
) -> list[int]:
    """Place fused-prepass jobs on the persistent clusters with the most slack.

    The FP4 split program runs its QK-only prepass as phase 0 of the *same*
    launch, so a prepass tile adds to the load of the cluster that executes it
    instead of costing a whole extra kernel.  Jobs are handed to the currently
    least loaded cluster, which is how they land on the clusters the LPT slot
    assignment leaves with a single unit: those clusters publish their maxima
    long before the split chunks of phase 1 spin on them.  Returns one entry per
    ``(round, cluster)`` pair -- ``round * num_clusters + cluster`` holds the
    job index, or ``-1`` when that cluster is idle in that round -- matching the
    kernel's ``cluster_id, += num_clusters`` stride.
    """
    total = len(job_costs)
    G = int(num_clusters)
    if total == 0 or G <= 0:
        return []
    rounds = (total + G - 1) // G
    load = [float(cluster_loads[c]) if c < len(cluster_loads) else 0.0 for c in range(G)]
    placed = [-1] * (rounds * G)
    for job in sorted(range(total), key=lambda j: -job_costs[j]):
        cluster = min(range(G), key=lambda c: (load[c], c))
        round_index = 0
        while placed[round_index * G + cluster] != -1:
            round_index += 1
        placed[round_index * G + cluster] = job
        load[cluster] += float(job_costs[job])
    return placed


def split_chunks(blocks: int, splits: int) -> list[tuple[int, int]]:
    """Partition ``blocks`` K/V blocks into ``min(splits, blocks)`` near-equal ``(begin, count)`` ranges."""
    splits = max(1, min(int(splits), blocks))
    base, rem = divmod(blocks, splits)
    chunks: list[tuple[int, int]] = []
    begin = 0
    for j in range(splits):
        count = base + (1 if j < rem else 0)
        chunks.append((begin, count))
        begin += count
    return chunks


def unit_stage1_dead(q_block: int, seg_blocks: int) -> bool:
    """True when a cluster tile's second Q-block stage is all segment padding.

    Host-side mirror of ``qkpv_stage1_dead`` in the SM103 FP4 dense program.  A
    cluster tile owns Q blocks ``q_block .. q_block + 3``: CTA 0 runs ``q_block``
    (stage 0) and ``q_block + 1`` (stage 1), CTA 1 runs ``q_block + 2`` and
    ``q_block + 3``, so stage 1 is dead exactly when ``q_block + 1`` is no longer
    one of the segment's ``seg_blocks`` Q blocks.  ``q_block`` is the tile's
    ``cl_q_block`` entry and ``seg_blocks`` its ``ceil(seg_len / 128)``.
    """
    return q_block + 1 >= seg_blocks


def lpt_makespan(costs: Sequence[float], num_clusters: int) -> float:
    """Makespan (max per-cluster load) of :func:`assign_unit_slots` for ``costs``."""
    if not costs:
        return 0.0
    G = min(num_clusters, len(costs))
    loads = [0.0] * G
    # Costs may be fractional (a tail tile whose dead stage the program skips),
    # so the placement order has to see them unrounded.
    for slot, unit in enumerate(assign_unit_slots(list(costs), G)):
        loads[slot % G] += costs[unit]
    return max(loads)


def choose_kv_splits(
    unit_blocks: Sequence[int],
    num_clusters: int,
    *,
    force: Optional[int] = None,
    program_cost: float = 1.0,
    prefix_cost: float = 0.0,
) -> list[int]:
    """Per-unit K/V split factors of one plan from the simulated makespan.

    ``force`` applies one factor to every unit.  Otherwise, for plans below
    ``KV_SPLIT_MAX_WAVES`` waves, the ``U mod G`` most expensive units (and,
    as the fallback, every unit) are split ``k`` ways for ``k`` in
    ``2..MAX_KV_SPLITS``; each candidate's longest-processing-time-first slot
    assignment is simulated, scaled by ``program_cost`` (the per-unit cost of
    the split-capable attention program relative to the dense one, see
    ``SPLIT_PROGRAM_COST``) and the combine cost added, and the best candidate
    is kept only when it beats the unsplit makespan by more than
    ``KV_SPLIT_MIN_GAIN``.  Mirrors the Cake production planner's
    ``choose_kv_splits``. FP4's separate QK/prefix-max pass costs
    ``prefix_cost`` attention-block equivalents for each split block.

    This search deliberately carries no :data:`RAGGED_STAGE_COST` discount.  It
    decides *which of the route's two programs* serves a geometry, and the two
    programs are not value-identical: measured on shape 8 (2049 tokens, 16 heads,
    peaked), replanning that geometry from the split program onto the dense one
    moved one output row and left 6 of 4196352 elements outside the frozen
    tolerance, while the same change was bit-exact on every other fixed shape.
    Program choice is therefore a correctness decision and keeps its existing
    calibration; only the LPT placement below, which cannot change any delivered
    value, is told what a dead-stage tile really costs.
    """
    U = len(unit_blocks)
    if U == 0:
        return []
    if force is not None:
        return [max(1, min(int(force), int(b))) for b in unit_blocks]
    ones = [1] * U
    G = max(1, int(num_clusters))
    if U >= KV_SPLIT_MAX_WAVES * G:
        return ones
    order = sorted(range(U), key=lambda u: -unit_blocks[u])

    def total(split_of: list[int]) -> float:
        costs: list[float] = []
        prefix_work: list[float] = []
        slots = 0
        for u in range(U):
            chunks = split_chunks(int(unit_blocks[u]), split_of[u])
            if len(chunks) > 1:
                slots += len(chunks)
                prefix_work.extend(count + BF16_UNIT_OVERHEAD_BLOCKS for _, count in chunks)
            costs.extend(float(count + BF16_UNIT_OVERHEAD_BLOCKS) for _, count in chunks)
        combine = (
            COMBINE_FIXED_BLOCKS + COMBINE_BLOCKS_PER_SLOT * slots if slots else 0.0
        )
        span = lpt_makespan(costs, G)
        if slots:
            span *= program_cost
        return span + combine + prefix_cost * lpt_makespan(prefix_work, G)

    base = total(ones)
    tail = U % G
    candidates_n = {U}
    if tail:
        candidates_n.add(tail)
    best, best_split = (base, ones)
    for n in sorted(candidates_n):
        for k in range(2, MAX_KV_SPLITS + 1):
            split_of = ones[:]
            for u in order[:n]:
                split_of[u] = k
            t = total(split_of)
            if t < best:
                best, best_split = (t, split_of)
    if best > base * (1.0 - KV_SPLIT_MIN_GAIN):
        return ones
    return best_split


def _partial_workspace(
    partial_slots: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = max(int(partial_slots), 1)
    return (
        torch.empty(
            slots * PARTIAL_ROWS * HEAD_DIM, dtype=torch.float16, device=device
        ),
        torch.empty(slots * PARTIAL_ROWS * 2, dtype=torch.float32, device=device),
    )


def combine_kwargs(
    partial_O: torch.Tensor,
    partial_ML: torch.Tensor,
    combine_table: torch.Tensor,
    seg_begin: torch.Tensor,
    seg_len: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    num_combine_units: int,
) -> dict[str, Any]:
    """Keyword bindings of the ``combine`` stage (``COMBINE_CTAS_PER_UNIT`` CTAs per split unit)."""
    kwargs = dict(
        partial_O=partial_O,
        partial_ML=partial_ML,
        combine_table=combine_table,
        seg_begin=seg_begin,
        seg_len=seg_len,
        O=out,
        num_heads=int(num_heads),
        grid=(max(int(num_combine_units), 1) * COMBINE_CTAS_PER_UNIT, 1, 1),
    )
    assert tuple(kwargs) == COMBINE_KWARGS
    return kwargs


@dataclass(frozen=True)
class PackedSegmentPlan:
    """Host plan of the NVFP4 packed layout (independent of heads and PV mode).

    Segment ``s`` (non-empty segments only) owns ``seg_blocks[s] =
    ceil(len / 128)`` packed 128-token blocks starting at packed block
    ``seg_tile_base[s]``; ``PB = sum(seg_blocks)``.  Token ``t`` of segment
    ``s`` lands at packed row ``seg_tile_base[s] * 128 + (t - seg_begin[s])``
    of its head; rows past the segment inside its padded region are
    zero-filled with the minimum scale.  ``block_token[p]`` / ``block_valid[p]``
    drive the quantizer (first THD token and valid row count of packed block
    ``p``); the four per-cluster ``cl_*`` tables (cluster tile -> segment
    start token, segment length, first packed block of the segment, first Q
    block of the cluster inside the segment) are the source of the per-tile
    scheduler tables built by :func:`build_tile_tables` for one head count.
    """

    cu_seqlens: tuple[int, ...]
    total_tokens: int
    seg_begin: tuple[int, ...]
    seg_len: tuple[int, ...]
    seg_blocks: tuple[int, ...]
    seg_tile_base: tuple[int, ...]
    cluster_off: tuple[int, ...]
    PB: int
    total_clusters: int
    block_token: torch.Tensor
    block_valid: torch.Tensor
    cl_seg_begin: torch.Tensor
    cl_seg_len: torch.Tensor
    cl_kv_base: torch.Tensor
    cl_q_block: torch.Tensor

    @property
    def num_segments(self) -> int:
        return len(self.seg_len)


def build_packed_segment_plan(
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    device: torch.device,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> PackedSegmentPlan:
    bounds = normalize_cu_seqlens(cu_seqlens, cu_seqlens_host=cu_seqlens_host)
    seg_begin: list[int] = []
    seg_len: list[int] = []
    seg_blocks: list[int] = []
    seg_tile_base: list[int] = []
    cluster_off: list[int] = []
    block_token: list[int] = []
    block_valid: list[int] = []
    cl_seg_begin: list[int] = []
    cl_seg_len: list[int] = []
    cl_kv_base: list[int] = []
    cl_q_block: list[int] = []
    PB = 0
    clusters = 0
    for a, b in zip(bounds, bounds[1:], strict=False):
        length = b - a
        if length <= 0:
            continue
        blocks = (length + BLOCK_M - 1) // BLOCK_M
        seg_begin.append(a)
        seg_len.append(length)
        seg_blocks.append(blocks)
        seg_tile_base.append(PB)
        cluster_off.append(clusters)
        for blk in range(blocks):
            block_token.append(a + blk * BLOCK_M)
            block_valid.append(min(BLOCK_M, length - blk * BLOCK_M))
        n_clusters = (length + CLUSTER_Q_ROWS - 1) // CLUSTER_Q_ROWS
        for c in range(n_clusters):
            cl_seg_begin.append(a)
            cl_seg_len.append(length)
            cl_kv_base.append(PB)
            cl_q_block.append(CLUSTER_Q_BLOCKS * c)
        PB += blocks
        clusters += n_clusters
    cluster_off.append(clusters)
    return PackedSegmentPlan(
        cu_seqlens=bounds,
        total_tokens=bounds[-1],
        seg_begin=tuple(seg_begin),
        seg_len=tuple(seg_len),
        seg_blocks=tuple(seg_blocks),
        seg_tile_base=tuple(seg_tile_base),
        cluster_off=tuple(cluster_off),
        PB=PB,
        total_clusters=clusters,
        block_token=_table(block_token, device),
        block_valid=_table(block_valid, device),
        cl_seg_begin=_table(cl_seg_begin, device),
        cl_seg_len=_table(cl_seg_len, device),
        cl_kv_base=_table(cl_kv_base, device),
        cl_q_block=_table(cl_q_block, device),
    )


@dataclass(frozen=True)
class TileTables:
    """Per-unit scheduler tables of the NVFP4 attention kernel for one ``(plan, heads)``.

    ``total_tiles`` scheduled units (K/V split ranges included) per table.
    Unit ``t`` runs head ``cl_head[t]`` on the cluster tile described by the
    next four tables (segment start token, segment length, first packed block
    of the segment, first Q block of the cluster tile inside the segment) over
    K/V blocks ``cl_kv_begin[t] .. cl_kv_begin[t] + cl_kv_blocks[t]`` of the
    segment, writing partial slot ``cl_ws_slot[t]`` (-1: the BF16 output
    directly).  The persistent grid is statically strided: cluster ``i`` runs
    units ``i, i + G, ...`` with ``G = grid_x / 2``.  ``combine_table``,
    ``seg_begin`` / ``seg_len`` and the partial workspace feed the ``combine``
    stage.

    The FP4 split program fuses its QK-only prepass into the attention launch, so
    three more tables describe it.  ``pf_tile`` lists, per ``(round, cluster)``,
    the slot whose K/V range phase 0 reduces to per-block row maxima (``-1``:
    that cluster has no prepass tile in that round); ``pf_total`` is its length.
    ``pf_info`` holds three int32 per slot -- the slot's flag base, its chunk
    index inside its own K/V-split unit, and that unit's ``splits - 1`` -- which
    is all the kernel needs to derive both the flags a chunk waits on and the
    flags its own prepass tile publishes.  ``pf_flag`` is the zeroed handoff-flag
    buffer; the kernel clears the flags it consumes before it exits, so one
    allocation serves every launch.
    """

    heads: int
    total_tiles: int
    cl_head: torch.Tensor
    cl_seg_begin: torch.Tensor
    cl_seg_len: torch.Tensor
    cl_kv_base: torch.Tensor
    cl_q_block: torch.Tensor
    cl_kv_begin: torch.Tensor
    cl_kv_blocks: torch.Tensor
    cl_ws_slot: torch.Tensor
    combine_table: torch.Tensor
    seg_begin: torch.Tensor
    seg_len: torch.Tensor
    partial_O: torch.Tensor
    partial_ML: torch.Tensor
    num_partial_slots: int
    num_combine_units: int
    max_kv_splits: int
    pf_tile: torch.Tensor
    pf_info: torch.Tensor
    pf_flag: torch.Tensor
    pf_total: int
    num_prefix_jobs: int
    num_prefix_flags: int
    NAMES = (
        "cl_head",
        "cl_seg_begin",
        "cl_seg_len",
        "cl_kv_base",
        "cl_q_block",
        "cl_kv_begin",
        "cl_kv_blocks",
        "cl_ws_slot",
    )


@lru_cache(maxsize=256)
def _d16_split_plan_cached(blocks, clusters, force, program_cost, prefix_cost):
    return tuple(
        choose_kv_splits(blocks, clusters, force=force, program_cost=program_cost,
                         prefix_cost=prefix_cost)
    )


@lru_cache(maxsize=256)
def _d16_unit_slots_cached(costs, clusters):
    return tuple(assign_unit_slots(costs, clusters))


def build_tile_tables(
    plan: PackedSegmentPlan,
    heads: int,
    device: torch.device,
    *,
    num_clusters: Optional[int] = None,
    kv_splits: Optional[int] = None,
    program_cost: float = 1.0,
    prefix_cost: float = 0.0,
    ragged_stage_cost: float = 1.0,
) -> TileTables:
    """Per-unit tables: segment-major enumeration, K/V splits, then LPT slot placement.

    Units are enumerated by segment in descending length (ties keep the
    segment order), then head, then the segment's cluster tiles in order --
    consecutive units share one head's K/V, which keeps the concurrent working
    set L2-resident -- and each unit is cut into its planned number of
    near-equal K/V block ranges (:func:`choose_kv_splits`; ``kv_splits``
    forces one factor).  The units are then placed into the persistent grid's
    statically strided slots longest-processing-time first
    (:func:`assign_unit_slots`; equal costs keep the enumeration order).  The
    kernel's five per-tile tables are allocated first, in the production
    planner's order, so their placement relative to the workspace matches it.
    ``ragged_stage_cost`` is the fraction of the per-block rate a tile with a
    dead second Q-block stage costs on a route whose dense program skips it
    (:data:`RAGGED_STAGE_COST`; 1.0 keeps every tile at full price).
    ``num_clusters`` is the grid capacity in 2-CTA clusters (default
    ``bf16_grid_clusters(device)``, a CUDA device).  Mirrors the Cake
    production planner's ``build_tile_tables``.
    """
    heads = int(heads)
    if heads <= 0:
        raise ValueError("heads must be positive")
    if kv_splits is not None and (not 1 <= int(kv_splits) <= MAX_KV_SPLITS):
        raise ValueError(f"kv_splits must be in [1, {MAX_KV_SPLITS}], got {kv_splits}")
    if num_clusters is None:
        num_clusters = bf16_grid_clusters(device)
    segments = sorted(range(plan.num_segments), key=lambda s: -plan.seg_len[s])
    cl = {
        name: [] for name in ("cl_seg_begin", "cl_seg_len", "cl_kv_base", "cl_q_block")
    }
    for s in range(plan.num_segments):
        count = plan.cluster_off[s + 1] - plan.cluster_off[s]
        cl["cl_seg_begin"].extend([plan.seg_begin[s]] * count)
        cl["cl_seg_len"].extend([plan.seg_len[s]] * count)
        cl["cl_kv_base"].extend([plan.seg_tile_base[s]] * count)
        cl["cl_q_block"].extend((CLUSTER_Q_BLOCKS * c for c in range(count)))
    seg_blocks = [(length + BLOCK_N - 1) // BLOCK_N for length in plan.seg_len]
    # Enumerated exactly like the unit loop below (segment, head, cluster tile),
    # so unit_ragged[u] describes the unit whose cost lands in costs[u].
    unit_blocks: list[int] = []
    unit_ragged: list[bool] = []
    for s in segments:
        for _head in range(heads):
            for c in range(plan.cluster_off[s], plan.cluster_off[s + 1]):
                unit_blocks.append(seg_blocks[s])
                unit_ragged.append(
                    ragged_stage_cost < 1.0
                    and unit_stage1_dead(
                        CLUSTER_Q_BLOCKS * (c - plan.cluster_off[s]), seg_blocks[s]
                    )
                )
    split_of = _d16_split_plan_cached(
        tuple(unit_blocks), int(num_clusters), kv_splits, program_cost, prefix_cost
    )
    # A plan with any K/V-split unit runs in the split program, which executes
    # both Q-block stages, so only an all-dense plan may book the discount -- and
    # only for placement, which never changes a delivered value.
    dense_plan = all(value == 1 for value in split_of)
    units: list[tuple[int, int, int, int, int]] = []
    costs: list[float] = []
    combine: list[int] = []
    # Per enumeration unit: (split-group, chunk index inside it, chunk count).
    # Unsplit units get group -1 and take no part in the prepass handoff.
    chunk_of: list[tuple[int, int, int]] = []
    group_chunks: list[int] = []
    partial_slots = 0
    unit_index = 0
    for s in segments:
        for head in range(heads):
            for c in range(plan.cluster_off[s], plan.cluster_off[s + 1]):
                chunks = split_chunks(seg_blocks[s], split_of[unit_index])
                unit_index += 1
                if len(chunks) == 1:
                    units.append((c, head, 0, seg_blocks[s], -1))
                    if dense_plan and unit_ragged[unit_index - 1]:
                        costs.append(
                            BF16_UNIT_OVERHEAD_BLOCKS + seg_blocks[s] * ragged_stage_cost
                        )
                    else:
                        costs.append(float(seg_blocks[s] + BF16_UNIT_OVERHEAD_BLOCKS))
                    chunk_of.append((-1, 0, 1))
                    continue
                group = len(group_chunks)
                group_chunks.append(len(chunks))
                tile_in_segment = c - plan.cluster_off[s]
                combine.extend(
                    (
                        s,
                        head << 16 | tile_in_segment,
                        partial_slots,
                        len(chunks),
                        # Valid packed rows of this cluster tile.  The combine grid
                        # is uniform (COMBINE_CTAS_PER_UNIT CTAs per split unit), so
                        # a segment-tail tile still gets CTAs for rows that do not
                        # exist; this word lets them leave before fetching partials
                        # the attention program never wrote.
                        max(
                            0,
                            min(
                                CLUSTER_Q_ROWS,
                                plan.seg_len[s] - tile_in_segment * CLUSTER_Q_ROWS,
                            ),
                        ),
                    )
                )
                for chunk_index, (begin, count) in enumerate(chunks):
                    units.append((c, head, begin, count, partial_slots))
                    costs.append(float(count + BF16_UNIT_OVERHEAD_BLOCKS))
                    chunk_of.append((group, chunk_index, len(chunks)))
                    partial_slots += 1
    num_tiles = len(units)
    grid_clusters = min(int(num_clusters), max(num_tiles, 1))
    unit_of_slot = _d16_unit_slots_cached(tuple(costs), grid_clusters)
    slots = [units[u] for u in unit_of_slot]
    # --- fused QK-only prepass tables -------------------------------------
    # A split unit of k chunks needs the row maxima of the blocks preceding
    # each of chunks 1..k-1; chunk j publishes its own range and therefore
    # hands off to readers j+1..k-1.  One flag per (writer, reader) pair keeps
    # each flag single-consumer, so the consumer can clear it and the buffer
    # needs no per-launch reset: k chunks use (k-1)^2 flags.
    group_flag_base: list[int] = []
    prefix_flags = 0
    for chunks_in_group in group_chunks:
        group_flag_base.append(prefix_flags)
        prefix_flags += (chunks_in_group - 1) ** 2
    pf_info = [0] * (3 * max(num_tiles, 1))
    for slot, u in enumerate(unit_of_slot):
        group, chunk_index, chunks_in_group = chunk_of[u]
        if group < 0:
            continue
        pf_info[3 * slot] = group_flag_base[group]
        pf_info[3 * slot + 1] = chunk_index
        pf_info[3 * slot + 2] = chunks_in_group - 1
    # Slot is a prepass tile exactly when a later chunk replays its blocks,
    # i.e. it is not the final chunk of its unit (== needs_prefix_tile).
    prefix_jobs = [
        slot for slot in range(num_tiles) if pf_info[3 * slot + 1] < pf_info[3 * slot + 2]
    ]
    cluster_load = [0.0] * grid_clusters
    for slot, u in enumerate(unit_of_slot):
        cluster_load[slot % grid_clusters] += costs[u]
    placement = assign_prefix_jobs(
        [slots[j][3] + BF16_UNIT_OVERHEAD_BLOCKS for j in prefix_jobs],
        cluster_load,
        grid_clusters,
    )
    pf_total = len(placement)
    # Padded to num_tiles so that a compiler that materialises both arms of the
    # kernel's phase select can never index past the table in the attention
    # phase, where the loop bound is num_tiles rather than pf_total.
    pf_tile = [prefix_jobs[job] if job >= 0 else -1 for job in placement]
    pf_tile += [-1] * max(0, num_tiles - len(pf_tile))
    cl_head = _table([head for _, head, _, _, _ in slots], device)
    cl_tables = {
        name: _table([values[c] for c, _, _, _, _ in slots], device)
        for name, values in cl.items()
    }
    partial_O, partial_ML = _partial_workspace(partial_slots, device)
    # Zeroed on the host and copied, not filled on the device: a CUDA
    # ``torch.zeros`` here would launch an elementwise fill kernel inside every
    # forward, and the measured metric is the sum of kernel durations.  The
    # kernel retires the flags it consumes before it exits, so this one-off
    # initialisation is the only zeroing the buffer ever needs.
    prefix_flag = torch.zeros(max(prefix_flags, 1), dtype=torch.int32).to(device)
    return TileTables(
        heads=heads,
        total_tiles=num_tiles,
        cl_head=cl_head,
        cl_kv_begin=_table([begin for _, _, begin, _, _ in slots], device),
        cl_kv_blocks=_table([count for _, _, _, count, _ in slots], device),
        cl_ws_slot=_table([slot for _, _, _, _, slot in slots], device),
        combine_table=torch.tensor(
            combine or [0] * COMBINE_WORDS, dtype=torch.int32, device=device
        ),
        seg_begin=_table(list(plan.seg_begin), device),
        seg_len=_table(list(plan.seg_len), device),
        partial_O=partial_O,
        partial_ML=partial_ML,
        num_partial_slots=partial_slots,
        num_combine_units=len(combine) // COMBINE_WORDS,
        max_kv_splits=max(split_of, default=1),
        pf_tile=_table(pf_tile, device),
        pf_info=_table(pf_info, device),
        pf_flag=prefix_flag,
        pf_total=pf_total,
        num_prefix_jobs=len(prefix_jobs),
        num_prefix_flags=prefix_flags,
        **cl_tables,
    )


def nvfp4_quantize_grid(heads: int, PB: int) -> tuple[int, int, int]:
    """Launch grid of the fused quantizer: one CTA per 32-token slice of every packed block."""
    return (int(heads) * int(PB) * QUANTIZE_SUBS_PER_BLOCK, 1, 1)


def nvfp4_workspace_shapes(
    heads: int, PB: int, pv_mode: str
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Shapes and dtypes of the packed operand set for ``(heads, PB, pv_mode)``.

    ``PB`` is padded to at least one block so every buffer is non-empty.
    """
    if pv_mode not in PV_MODES:
        raise ValueError(f"pv_mode must be one of {PV_MODES}, got {pv_mode!r}")
    PB = max(int(PB), 1)
    rows = heads * PB * BLOCK_M
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "q_fp4": ((rows, HEAD_DIM // 2), torch.uint8),
        "k_fp4": ((rows, HEAD_DIM // 2), torch.uint8),
        "q_scale": ((heads * PB * 32, 32), torch.uint8),
        "k_scale": ((heads * PB * 32, 32), torch.uint8),
    }
    if pv_mode == "fp4":
        shapes["v_fp4_t"] = ((heads * HEAD_DIM, PB * (BLOCK_M // 2)), torch.uint8)
        shapes["v_scale_lo"] = ((heads * PB * 16, 32), torch.uint8)
        shapes["v_scale_hi"] = ((heads * PB * 16, 32), torch.uint8)
    else:
        shapes["v_fp8"] = ((rows, HEAD_DIM), torch.uint8)
        shapes["v_amax_partial"] = ((AMAX_CTAS,), torch.float32)
        shapes["v_amax"] = ((1,), torch.float32)
    return shapes


def allocate_nvfp4_workspace(
    heads: int, PB: int, pv_mode: str, device: torch.device
) -> dict[str, torch.Tensor]:
    ws = {
        name: torch.empty(shape, dtype=dtype, device=device)
        for name, (shape, dtype) in nvfp4_workspace_shapes(heads, PB, pv_mode).items()
    }
    for name in ("v_amax_partial", "v_amax"):
        if name in ws:
            ws[name].zero_()
    return ws


def amax_kwargs(
    value: torch.Tensor, workspace: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Keyword bindings of the ``amax`` stage for the contiguous BF16 V operand.

    The kernel views ``value`` as ``numel / AMAX_VECTOR`` 32-byte vectors
    (any split of a contiguous tensor yields the same maximum) and each of the
    ``AMAX_CTAS`` CTAs strides over the ``num_chunks`` chunks of ``AMAX_CHUNK``
    vectors; the scalar never visits the host.
    """
    numel = int(value.numel())
    if numel % AMAX_VECTOR != 0:
        raise ValueError(f"V.numel() must be a multiple of {AMAX_VECTOR}, got {numel}")
    if not value.is_contiguous():
        raise ValueError("V must be contiguous for the amax stage")
    num_vectors = numel // AMAX_VECTOR
    kwargs = dict(
        v=value,
        v_amax_partial=workspace["v_amax_partial"],
        num_vectors=num_vectors,
        num_chunks=(num_vectors + AMAX_CHUNK - 1) // AMAX_CHUNK,
        grid=(AMAX_CTAS, 1, 1),
    )
    assert tuple(kwargs) == AMAX_KWARGS
    return kwargs
