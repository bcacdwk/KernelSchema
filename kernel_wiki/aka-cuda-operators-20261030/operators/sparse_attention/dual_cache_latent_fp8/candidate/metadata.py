"""Sparse index views and metadata validation."""

from dataclasses import dataclass
from typing import Any, Optional
import torch
from .config import KERNEL_METADATA_PARAMS, _SWA_WIDTH


@dataclass(frozen=True)
class SparseMetadata:
    """Resolved kernel-side view of the DSv4 sparse metadata (no copies)."""

    swa_indices: torch.Tensor
    compressed_indices: torch.Tensor
    sparse_topk_lens: torch.Tensor
    swa_index_stride: int
    compressed_index_stride: int
    sparse_topk_lens_offset: int
    sparse_topk: int
    num_query_tokens: int
    separate_tables: bool

    def kernel_kwargs(self) -> dict[str, Any]:
        """Keyword arguments in the shared kernel parameter vocabulary.

        The two index tables are handed over as contiguous spans (see
        ``_flat_index_span``): the generated bindings check every pointer
        buffer for contiguity, and the column-sliced compressed view of a
        combined table is not contiguous even though the kernel only reads
        its base pointer plus the row stride.
        """
        values = {name: getattr(self, name) for name in KERNEL_METADATA_PARAMS}
        values["swa_indices"] = _flat_index_span(self.swa_indices)
        values["compressed_indices"] = _flat_index_span(self.compressed_indices)
        return values

    @property
    def compressed_width(self) -> int:
        return self.sparse_topk - _SWA_WIDTH

    @property
    def legacy_combined_table(self) -> Optional[torch.Tensor]:
        """Combined ``[T, sparse_topk]`` table for bindings that predate the split ABI.

        Only a combined table without an explicit length offset can be handed
        to a binding that still declares the pre-hardening ``sparse_indices``
        argument; separate tables and offsets need regenerated bindings.
        """
        if self.separate_tables or self.sparse_topk_lens_offset != 0:
            return None
        return self.swa_indices


def _flat_index_span(table: torch.Tensor) -> torch.Tensor:
    """Expose a row-strided int32 table as one contiguous span without a copy.

    A contiguous table is returned as is. A column-sliced view (row stride
    larger than its width) becomes the 1-D span from its first to its last
    element, which keeps ``data_ptr`` and the row stride the kernel indexes
    with while satisfying the binding's contiguity check.
    """
    if table.is_contiguous() or table.ndim != 2:
        return table
    rows, cols = (int(table.shape[0]), int(table.shape[1]))
    if rows == 0 or cols == 0:
        return table.reshape(-1)
    return table.as_strided(((rows - 1) * int(table.stride(0)) + cols,), (1,))


def _int32_table(tensor: torch.Tensor, name: str, *, rows: Optional[int] = None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if tensor.ndim < 2:
        raise ValueError(
            f"{name} must be a [rows, columns] table, got shape {tuple(tensor.shape)}"
        )
    if tensor.ndim > 2:
        try:
            tensor = tensor.view(-1, tensor.shape[-1])
        except RuntimeError as exc:
            raise ValueError(
                f"{name} leading dimensions must be densely packed so they fold into rows without a copy"
            ) from exc
    if rows is not None and tensor.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows, got {tensor.shape[0]}")
    if tensor.shape[1] and tensor.stride(1) != 1:
        raise ValueError(
            f"{name} must have a unit column stride; pass a row-strided view instead of a copy"
        )
    return tensor


def _int32_lens(tensor: torch.Tensor, name: str, *, rows: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if tensor.ndim != 1:
        try:
            tensor = tensor.view(-1)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} must flatten to one row per token without a copy"
            ) from exc
    if tensor.numel() != rows:
        raise ValueError(f"{name} must have {rows} entries, got {tensor.numel()}")
    if tensor.numel() and tensor.stride(0) != 1:
        raise ValueError(f"{name} must have unit stride")
    return tensor


def resolve_cake_dsv4_sparse_metadata(
    sparse_indices: torch.Tensor,
    sparse_topk_lens: Optional[torch.Tensor] = None,
    *,
    extra_sparse_indices: Optional[torch.Tensor] = None,
    extra_sparse_topk_lens: Optional[torch.Tensor] = None,
    sparse_topk_lens_offset: int = 0,
    query_rows: int,
) -> SparseMetadata:
    """Resolve the shared metadata ABI from combined or separate host tables.

    Combined form: ``sparse_indices [T, sparse_topk]`` whose first 128 columns
    are SWA slots, with ``sparse_topk_lens`` counting those 128 slots.

    Separate form: ``sparse_indices [T, 128]`` is the SWA table and
    ``extra_sparse_indices [T, topk_c]`` the compressed table. Lengths come from
    ``sparse_topk_lens`` (combined convention) or ``extra_sparse_topk_lens``
    (compressed slots only; the host adds 128 to ``sparse_topk_lens_offset``).

    ``query_rows`` is the number of rows the query tensor provides; the metadata
    may describe fewer tokens (padded batch) but never more.
    """
    if isinstance(sparse_topk_lens_offset, bool) or not isinstance(
        sparse_topk_lens_offset, int
    ):
        raise TypeError("sparse_topk_lens_offset must be an int")
    offset = int(sparse_topk_lens_offset)
    if extra_sparse_indices is not None:
        swa = _int32_table(sparse_indices, "sparse_indices")
        if swa.shape[1] != _SWA_WIDTH:
            raise ValueError(
                f"with extra_sparse_indices, sparse_indices is the SWA table and must have {_SWA_WIDTH} columns, got {swa.shape[1]}"
            )
        rows = int(swa.shape[0])
        compressed = _int32_table(
            extra_sparse_indices, "extra_sparse_indices", rows=rows
        )
        compressed_width = int(compressed.shape[1])
        if compressed_width == 0:
            compressed = swa
        if extra_sparse_topk_lens is not None:
            if sparse_topk_lens is not None:
                raise ValueError(
                    "pass either sparse_topk_lens (combined, counting the 128 SWA slots) or extra_sparse_topk_lens (compressed slots only), not both"
                )
            lens, lens_name = (extra_sparse_topk_lens, "extra_sparse_topk_lens")
            offset += _SWA_WIDTH
        else:
            if sparse_topk_lens is None:
                raise ValueError(
                    "sparse_topk_lens or extra_sparse_topk_lens is required"
                )
            lens, lens_name = (sparse_topk_lens, "sparse_topk_lens")
        separate = True
    else:
        if extra_sparse_topk_lens is not None:
            raise ValueError("extra_sparse_topk_lens requires extra_sparse_indices")
        if sparse_topk_lens is None:
            raise ValueError(
                "sparse_topk_lens is required with a combined sparse_indices table"
            )
        table = _int32_table(sparse_indices, "sparse_indices")
        rows = int(table.shape[0])
        if table.shape[1] < _SWA_WIDTH:
            raise ValueError(
                f"sparse_indices must have at least {_SWA_WIDTH} columns, got {table.shape[1]}"
            )
        swa = table
        compressed_width = int(table.shape[1]) - _SWA_WIDTH
        compressed = table[:, _SWA_WIDTH:] if compressed_width else table
        lens, lens_name = (sparse_topk_lens, "sparse_topk_lens")
        separate = False
    if rows < 1:
        raise ValueError("sparse metadata must describe at least one query token")
    if rows > query_rows:
        raise ValueError(
            f"metadata has {rows} rows but the query only has {query_rows}"
        )
    sparse_topk = _SWA_WIDTH + compressed_width
    if sparse_topk % 4:
        raise ValueError(
            f"sparse_topk (128 + compressed columns) must be a multiple of 4, got {sparse_topk}"
        )
    lens = _int32_lens(lens, lens_name, rows=rows)
    return SparseMetadata(
        swa_indices=swa,
        compressed_indices=compressed,
        sparse_topk_lens=lens,
        swa_index_stride=int(swa.stride(0)),
        compressed_index_stride=int(compressed.stride(0)),
        sparse_topk_lens_offset=offset,
        sparse_topk=sparse_topk,
        num_query_tokens=rows,
        separate_tables=separate,
    )
