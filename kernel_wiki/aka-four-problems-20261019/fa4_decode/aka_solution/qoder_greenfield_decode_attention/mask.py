# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Float32, Optional
class AttentionMask(ABC):
    """ABC interface for attention masking configurations."""

    def __post_init__(self):
        assert is_dataclass(self), f"{type(self)} must be dataclass"

    # Optionally implemented for subclasses
    def can_implement(self, seqlen_q, seqlen_kv, tile_q, tile_kv):
        return

    @cute.jit
    @abstractmethod
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        """Whether the given KV token at idx_kv should be masked as OOB."""
        ...

    @cute.jit
    @abstractmethod
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        num_tiles_kv,
        num_iters_kv,
        kv_splits,
        kv_split_idx,
        warpgroups_kv,
        warpgroup_kv_idx,
    ) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        """Return args to cutlass.range and a mask codegen flag for each masking phase.

        For phases that generate masking code, range args must emit
        tile_kv coordinates (index into total number of tiles KV across entire KV sequence)
        so that kv token index can be correctly calculated.

        Parameters
        ----------
        seqlen_q
            Query sequence length
        seqlen_kv
            Key/Value sequence length
        tile_q
            Query sequence per threadblock
        tile_kv
            Key/Value sequence per threadblock per outer mainloop iteration
        num_tiles_kv
            KV tiles across entire KV sequence for this Q tile
        num_iters_kv
            KV tiles for this KV split, number of outer mainloop iterations
        kv_splits
            Threadblocks per KV sequence
        kv_split_idx
            Threadblock index into total KV splits
        warpgroups_kv
            Warpgroups per threadblock KV sequence
        warpgroup_kv_idx
            Warpgroup index into total KV warpgroups
        """
        ...


@dataclass(frozen=True)
class CausalMask(AttentionMask):
    """Current tokens only attend to past tokens and itself."""

    def can_implement(self, seqlen_q, seqlen_kv, tile_q, tile_kv):
        # Only causal mask up to two tiles
        # Revisit this for prefill integration
        if seqlen_q > tile_kv:
            raise ValueError(
                f"seqlen_q({seqlen_q}) with causal mask can be at most tile_kv({tile_kv})"
            )

    @cute.jit
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        idx_current = seqlen_kv - seqlen_q + idx_q
        return idx_kv > idx_current

    @cute.jit
    def get_range_args(
        self,
        seqlen_q,
        seqlen_kv,
        tile_q,
        tile_kv,
        num_tiles_kv,
        num_iters_kv,
        kv_splits,
        kv_split_idx,
        warpgroups_kv,
        warpgroup_kv_idx,
    ) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        is_last_split = kv_split_idx == (num_tiles_kv - 1) % kv_splits
        is_prev_split = kv_split_idx == (num_tiles_kv - 2) % kv_splits
        is_last_phase = warpgroup_kv_idx == (num_iters_kv - 1) % warpgroups_kv
        is_prev_phase = warpgroup_kv_idx == (num_iters_kv - 2) % warpgroups_kv

        unmasked_start = warpgroup_kv_idx
        unmasked_end = num_iters_kv
        unmasked_step = warpgroups_kv
        masked_start = masked_end = masked_step = 0

        # 2 masked tiles
        if 0 < seqlen_kv % tile_kv < seqlen_q:
            # 1 split masks 2 tiles
            if kv_splits == 1:
                unmasked_end -= 2
                masked_start = num_tiles_kv - 2 + (0 if is_prev_phase else 1)
                masked_end = num_tiles_kv
                masked_step = warpgroups_kv
            # 2 splits mask 1 tile each
            elif (is_last_split or is_prev_split) and is_last_phase:
                unmasked_end -= 1
                masked_start = (
                    (num_tiles_kv - 1) if is_last_split else (num_tiles_kv - 2)
                )
                masked_end = num_tiles_kv
                masked_step = 2
        # 1 masked tile
        elif seqlen_q > 1 or seqlen_kv % tile_kv != 0:
            # 1 split masks 1 tile
            if is_last_split and is_last_phase:
                unmasked_end -= 1
                masked_start = num_tiles_kv - 1
                masked_end = num_tiles_kv
                masked_step = 1

        return (
            (unmasked_start, unmasked_end, unmasked_step, False),
            (masked_start, masked_end, masked_step, True),
        )

