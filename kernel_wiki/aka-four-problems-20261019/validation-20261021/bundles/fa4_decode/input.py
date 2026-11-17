from __future__ import annotations

import torch


def _distribution_scales(q_len_per_req: int) -> tuple[float, float]:
    if q_len_per_req == 4:
        return 1.22, 0.96
    if q_len_per_req == 1:
        return 1.40, 2.05
    raise ValueError("captured MTP decode uses q_len_per_req 1 or 4")


def _make_inputs(
    seq_lens_values: list[int],
    q_len_per_req: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    block_table_columns: int,
    workspace_size_bytes: int,
    bmm1_scale: float,
    bmm2_scale: float,
) -> dict[str, object]:
    device = "cuda"
    if not seq_lens_values or any(length <= 0 for length in seq_lens_values):
        raise ValueError("decode sequence lengths must be positive")
    query_scale, kv_scale = _distribution_scales(q_len_per_req)
    if any(length < q_len_per_req for length in seq_lens_values):
        raise ValueError("KV sequence length cannot be shorter than the query tail")

    pages_per_request = [
        (length + page_size - 1) // page_size for length in seq_lens_values
    ]
    if max(pages_per_request) > block_table_columns:
        raise ValueError("block table capacity is too small")

    batch_size = len(seq_lens_values)
    total_pages = sum(pages_per_request)
    query = (
        torch.randn(
            batch_size * q_len_per_req,
            num_q_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * query_scale
    ).to(torch.float8_e4m3fn)
    kv_cache = (
        torch.randn(
            total_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * kv_scale
    ).to(torch.float8_e4m3fn)

    block_tables = torch.full(
        (batch_size, block_table_columns),
        -1,
        dtype=torch.int32,
        device=device,
    )
    page_cursor = 0
    for request_index, page_count in enumerate(pages_per_request):
        block_tables[request_index, :page_count] = torch.arange(
            page_cursor,
            page_cursor + page_count,
            dtype=torch.int32,
            device=device,
        )
        page_cursor += page_count

    return {
        "query": query,
        "kv_cache": kv_cache,
        "workspace_buffer": torch.zeros(
            workspace_size_bytes,
            dtype=torch.uint8,
            device=device,
        ),
        "block_tables": block_tables,
        "seq_lens": torch.tensor(
            seq_lens_values,
            dtype=torch.int32,
            device=device,
        ),
        "max_seq_len": max(seq_lens_values),
        "bmm1_scale": bmm1_scale,
        "bmm2_scale": bmm2_scale,
        "window_left": -1,
        "sinks": None,
        "o_sf_scale": None,
        "out": torch.empty(
            batch_size * q_len_per_req,
            num_q_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        "q_len_per_req": q_len_per_req,
    }
