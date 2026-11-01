from dataclasses import dataclass

import math

from typing import Optional, Sequence

import torch

import tvm_ffi

HEAD_DIM = 512

PAGE_SIZE = 64

TILE_KV = 128

PAGES_PER_TILE = TILE_KV // PAGE_SIZE

SF_VEC = 16

ROW_BYTES = HEAD_DIM // 2

SF_ROW_BYTES = HEAD_DIM // SF_VEC

DSV4_Q_LEN = 6

CLUSTER_PAIR = 2

ROWS_PER_TILE = 128

V_HALVES = 2

ROWS_PER_CTA = ROWS_PER_TILE // CLUSTER_PAIR

REDUCE_HEADS_PER_CTA = 8

LOG2E = 1.4426950408889634

SUPPORTED_COMPUTE_CAPABILITIES = {(10, 3): "sm_103a"}

ITEM_FIELDS = 8

FLAG_SEED_SINK = 1

FLAG_DIRECT_OUT = 2

MAX_SPLITS = 64

MIN_TILES_PER_UNIT = 8

BALANCED_MIN_KV = 8192

WORKSPACE_ALIGNMENT = 256

MAIN_KWARGS = (
    "Q",
    "QS",
    "KV",
    "KVS",
    "OT",
    "PO",
    "O",
    "LSE",
    "partial_o",
    "partial_lse",
    "work_table",
    "unit_first",
    "page_table",
    "seq_lens",
    "q_indptr",
    "sinks",
    "rows_total",
    "num_heads",
    "q_len",
    "max_pages",
    "max_splits",
    "scale_log2",
    "grid",
)

REDUCE_KWARGS = (
    "o",
    "lse",
    "partial_o",
    "partial_lse",
    "row_splits",
    "total_q",
    "num_heads",
    "max_splits",
    "grid",
)


@dataclass(frozen=True)
class WorkPlan:
    """Flat work table of the persistent decode kernel.

    One item is ``(request, m_tile, v_half, tile_start, tile_end, split,
    num_splits, flags)``; cluster ``u`` (``CLUSTER_PAIR`` CTAs) runs items
    ``unit_first[u] .. unit_first[u + 1]``, always whole (v_half 0, v_half 1)
    pairs of one piece. Plans split per (request, row tile): a piece with more
    than one split goes through the BF16 partials and the split-KV combine
    kernel, a single-split piece keeps ``FLAG_DIRECT_OUT`` and its CTAs write
    ``O`` / ``LSE`` directly, even when other pieces of the batch are split.
    """

    schedule: str
    items: tuple
    unit_first: tuple
    max_splits: int
    tiles_per_split: Optional[int]

    @property
    def num_items(self) -> int:
        return len(self.items)

    @property
    def num_units(self) -> int:
        return len(self.unit_first) - 1

    @property
    def grid(self) -> tuple:
        """Launch grid: one cluster of ``CLUSTER_PAIR`` CTAs per unit."""
        return (CLUSTER_PAIR * self.num_units, 1, 1)


def kv_pages(kv_len: int) -> int:
    return (kv_len + PAGE_SIZE - 1) // PAGE_SIZE


def kv_tiles(kv_len: int) -> int:
    """Decode pipeline tiles (``TILE_KV`` tokens) of one request; the last tile may hold one page."""
    return (kv_len + TILE_KV - 1) // TILE_KV


def _m_tiles(q_len: int, num_heads: int) -> int:
    return math.ceil(q_len * num_heads / ROWS_PER_TILE)


def choose_tiles_per_split(
    kv_lens: Sequence[int],
    *,
    q_len: int,
    num_heads: int,
    num_sms: int,
    min_tiles_per_split: int = 8,
    target_waves: int = 2,
    max_partial_fraction: float = 0.35,
) -> int:
    """Pick the uniform split granularity so the item count fills whole waves.

    Base items (no split) = requests * m_tiles * 2. The split granularity is
    halved until ``items >= target_waves * num_sms``, every split would fall
    below ``min_tiles_per_split`` tiles, or the partial traffic of the
    next split count would exceed ``max_partial_fraction`` of the KV bytes
    streamed. For bs32 / q_len 6 / 64 heads this keeps one split at 8K KV and
    two splits from 16K on.
    """
    m_tiles = _m_tiles(q_len, num_heads)
    total_tiles = [kv_tiles(kv) for kv in kv_lens]
    max_tiles = max(total_tiles)
    kv_bytes = sum(kv_lens) * (ROW_BYTES + SF_ROW_BYTES)
    partial_bytes_per_split = len(kv_lens) * q_len * num_heads * HEAD_DIM * 4
    tiles_per_split = max_tiles
    while True:
        items = (
            sum((math.ceil(t / tiles_per_split) for t in total_tiles))
            * m_tiles
            * V_HALVES
        )
        if items >= target_waves * num_sms:
            break
        candidate = max(min_tiles_per_split, math.ceil(tiles_per_split / 2))
        if candidate >= tiles_per_split:
            break
        next_splits = max((math.ceil(t / candidate) for t in total_tiles))
        if next_splits * partial_bytes_per_split > max_partial_fraction * kv_bytes:
            break
        tiles_per_split = candidate
    return tiles_per_split


def _uniform_plan(kv_lens, *, q_len, num_heads, num_sms, enable_sink, tiles_per_split):
    """Request-major table, one v_half pair per cluster (all CTAs of a split are adjacent)."""
    if tiles_per_split is None:
        tiles_per_split = choose_tiles_per_split(
            kv_lens, q_len=q_len, num_heads=num_heads, num_sms=num_sms
        )
    m_tiles = _m_tiles(q_len, num_heads)
    items = []
    max_splits = 1
    for b, kv_len in enumerate(kv_lens):
        tiles = kv_tiles(kv_len)
        num_splits = math.ceil(tiles / tiles_per_split)
        max_splits = max(max_splits, num_splits)
        for split in range(num_splits):
            start = split * tiles_per_split
            end = min(tiles, start + tiles_per_split)
            flags = 0
            if enable_sink and split == 0:
                flags |= FLAG_SEED_SINK
            if num_splits == 1:
                flags |= FLAG_DIRECT_OUT
            for m_tile in range(m_tiles):
                for v_half in range(V_HALVES):
                    items.append(
                        (b, m_tile, v_half, start, end, split, num_splits, flags)
                    )
    return WorkPlan(
        schedule="uniform",
        items=tuple(items),
        unit_first=tuple(range(0, len(items) + 1, CLUSTER_PAIR)),
        max_splits=max_splits,
        tiles_per_split=tiles_per_split,
    )


def _balanced_plan(kv_lens, *, q_len, num_heads, num_units, enable_sink):
    """Balanced contiguous partition of the concatenated (row tile, page) work.

    Unit ``u`` of ``U`` receives positions ``[floor(u*W/U), floor((u+1)*W/U))``
    of the request-major page sequence. Wherever a unit boundary cuts a
    sequence the sequence becomes two splits; both ``v_half`` items of a piece
    share ``[start, end)`` and the split index, so the LSE written by the
    ``v_half == 0`` item applies to both halves of ``O`` in the combine
    kernel.
    """
    m_tiles = _m_tiles(q_len, num_heads)
    sequences = []
    for b, kv_len in enumerate(kv_lens):
        tiles = kv_tiles(kv_len)
        for m_tile in range(m_tiles):
            sequences.append((b, m_tile, tiles))
    total = sum((seq[2] for seq in sequences))
    if total == 0 or num_units <= 0:
        raise ValueError("balanced plan needs work and at least one unit")
    num_units = max(1, min(num_units, total // MIN_TILES_PER_UNIT))
    longest = max((seq[2] for seq in sequences))
    while num_units > 1 and math.ceil(longest / (total / num_units)) + 1 > MAX_SPLITS:
        num_units //= 2
    pieces: list = [[] for _ in sequences]
    seq_idx = 0
    seq_pos = 0
    for u in range(num_units):
        lo = u * total // num_units
        hi = (u + 1) * total // num_units
        pos = lo
        while pos < hi:
            while seq_pos + sequences[seq_idx][2] <= pos:
                seq_pos += sequences[seq_idx][2]
                seq_idx += 1
            start = pos - seq_pos
            end = min(sequences[seq_idx][2], hi - seq_pos)
            pieces[seq_idx].append((u, start, end))
            pos = seq_pos + end
    per_unit: list = [[] for _ in range(num_units)]
    max_splits = 1
    for (b, m_tile, _tiles), seq_pieces in zip(sequences, pieces, strict=False):
        num_splits = len(seq_pieces)
        max_splits = max(max_splits, num_splits)
        for split, (u, start, end) in enumerate(seq_pieces):
            flags = 0
            if enable_sink and split == 0:
                flags |= FLAG_SEED_SINK
            if num_splits == 1:
                flags |= FLAG_DIRECT_OUT
            for v_half in range(V_HALVES):
                per_unit[u].append(
                    (b, m_tile, v_half, start, end, split, num_splits, flags)
                )
    items = []
    unit_first = [0]
    for u in range(num_units):
        items.extend(per_unit[u])
        unit_first.append(len(items))
    return WorkPlan(
        schedule="balanced",
        items=tuple(items),
        unit_first=tuple(unit_first),
        max_splits=max_splits,
        tiles_per_split=None,
    )


def check_pairs(plan: WorkPlan) -> None:
    """Every unit holds consecutive (v_half 0, v_half 1) items of the same piece."""
    items, unit_first = (plan.items, plan.unit_first)
    for u in range(len(unit_first) - 1):
        lo, hi = (unit_first[u], unit_first[u + 1])
        if (hi - lo) % CLUSTER_PAIR:
            raise ValueError(
                f"unit {u} holds {hi - lo} items; the cluster pairing needs an even count"
            )
        for i in range(lo, hi, CLUSTER_PAIR):
            a, b = (items[i], items[i + 1])
            if (a[2], b[2]) != (0, 1) or a[:2] != b[:2] or a[3:7] != b[3:7]:
                raise ValueError(
                    f"items {i},{i + 1} are not a v_half pair of one piece: {a} / {b}"
                )


def build_work_plan(
    kv_lens: Sequence[int],
    *,
    num_heads: int,
    num_sms: int,
    q_len: int,
    enable_sink: bool = False,
    schedule: str = "auto",
    tiles_per_split: Optional[int] = None,
) -> WorkPlan:
    """Build the host work plan for one batch.

    ``schedule="auto"`` uses the balanced partition over ``num_sms //
    CLUSTER_PAIR`` clusters when the longest request reaches
    ``BALANCED_MIN_KV`` tokens and the uniform split policy otherwise;
    ``"balanced"`` / ``"uniform"`` force one of them. ``tiles_per_split``
    fixes the uniform split granularity (pages per split).
    """
    kv_lens = [int(v) for v in kv_lens]
    if q_len < 1 or not kv_lens or min(kv_lens) < q_len:
        raise ValueError("every request needs kv_len >= q_len >= 1")
    if schedule == "auto":
        schedule = "balanced" if max(kv_lens) >= BALANCED_MIN_KV else "uniform"
    if schedule == "balanced":
        plan = _balanced_plan(
            kv_lens,
            q_len=q_len,
            num_heads=num_heads,
            num_units=num_sms // CLUSTER_PAIR,
            enable_sink=enable_sink,
        )
    elif schedule == "uniform":
        plan = _uniform_plan(
            kv_lens,
            q_len=q_len,
            num_heads=num_heads,
            num_sms=num_sms,
            enable_sink=enable_sink,
            tiles_per_split=tiles_per_split,
        )
    else:
        raise ValueError(f"unknown schedule {schedule!r} (auto | balanced | uniform)")
    if plan.max_splits > MAX_SPLITS:
        raise ValueError(
            f"work plan needs {plan.max_splits} KV splits; the combine kernel supports {MAX_SPLITS}"
        )
    check_pairs(plan)
    return plan


def work_table_rows(plan: WorkPlan) -> torch.Tensor:
    """Host int32 ``[num_items, ITEM_FIELDS]`` table for ``plan``.

    Items are consumed as planned: every item carries ``FLAG_DIRECT_OUT`` iff
    it is the only split of its (request, row tile), whatever the other
    pieces of the batch do; the combine kernel skips the rows of such items
    (``token_splits``).
    """
    return torch.tensor(plan.items, dtype=torch.int32).reshape(-1, ITEM_FIELDS)


def token_splits(
    plan: WorkPlan, q_indptr: Sequence[int], *, q_len: int, num_heads: int, total_q: int
) -> list:
    """Per query token split count consumed by the combine kernel (``row_splits``).

    Plans split per (request, row tile), so one request's tokens may carry
    different counts. Token ``t`` of request ``b`` (row ``q_indptr[b] + t``)
    lies in row tile ``t * num_heads // ROWS_PER_TILE``; ``num_heads`` must
    divide ``ROWS_PER_TILE`` so no token straddles two row tiles. Every item
    must carry ``FLAG_DIRECT_OUT`` iff it is the only split of its (request,
    row tile) -- the decode kernel then writes that row's final ``O`` / ``LSE``
    itself and the combine kernel leaves the row alone -- and every token
    must be covered. Any violation raises; nothing is defaulted.
    """
    if ROWS_PER_TILE % num_heads != 0:
        raise ValueError(
            f"per-token row_splits need num_heads ({num_heads}) to divide ROWS_PER_TILE ({ROWS_PER_TILE}); a token would straddle two row tiles"
        )
    tokens_per_tile = ROWS_PER_TILE // num_heads
    out = [0] * total_q
    for it in plan.items:
        if (it[7] & FLAG_DIRECT_OUT != 0) != (it[6] == 1):
            raise ValueError(
                f"item {it} carries FLAG_DIRECT_OUT inconsistent with its split count"
            )
        first = it[1] * tokens_per_tile
        for t in range(first, min(first + tokens_per_tile, q_len)):
            row = q_indptr[it[0]] + t
            if out[row] not in (0, it[6]):
                raise ValueError(
                    f"token row {row} receives split counts {out[row]} and {it[6]}"
                )
            out[row] = it[6]
    if 0 in out:
        raise ValueError(
            f"query token row {out.index(0)} received no work item (row_splits == 0)"
        )
    return out


def _align(nbytes: int) -> int:
    return (
        (nbytes + WORKSPACE_ALIGNMENT - 1) // WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT
    )


def partial_shapes(plan: WorkPlan, *, total_q: int, num_heads: int) -> tuple:
    """Shapes of BF16 ``partial_o`` (split-major) / FP32 ``partial_lse``.

    Single-split plans keep a ``[ROWS_PER_CTA, 512]`` BF16 dummy: the decode
    kernel encodes its partial-output tensor map from it and never stores to it.
    """
    if plan.max_splits == 1:
        return ((ROWS_PER_CTA, HEAD_DIM), (1,))
    return (
        (plan.max_splits, total_q, num_heads, HEAD_DIM),
        (total_q, num_heads, plan.max_splits),
    )


def workspace_layout(plan: WorkPlan, *, batch: int, num_heads: int, q_len: int) -> dict:
    """Byte offsets and sizes of every workspace region plus ``"total"``.

    Regions: BF16 ``partial_o [max_splits, total_q, H, 512]`` and FP32
    ``partial_lse [total_q, H, max_splits]`` (a 64-row tensor-map dummy and one
    element when the plan has a single split and the kernel writes ``O`` /
    ``LSE`` directly),
    int32 ``work_table``, ``unit_first``, ``row_splits``, ``q_indptr`` and
    FP32 ``sinks [H]``.
    """
    total_q = batch * q_len
    o_shape, lse_shape = partial_shapes(plan, total_q=total_q, num_heads=num_heads)
    sizes = (
        ("partial_o", math.prod(o_shape) * 2),
        ("partial_lse", math.prod(lse_shape) * 4),
        ("work_table", plan.num_items * ITEM_FIELDS * 4),
        ("unit_first", (plan.num_units + 1) * 4),
        ("row_splits", total_q * 4),
        ("q_indptr", (batch + 1) * 4),
        ("sinks", num_heads * 4),
    )
    layout: dict = {}
    offset = 0
    for name, nbytes in sizes:
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


from functools import cache
