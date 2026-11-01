"""Host constants and argument vocabulary."""

import threading
from typing import Mapping, Optional
import torch

_HEAD_DIM = 512
_TILE_KV = 128
_SWA_WIDTH = 128
_ALIGN = 128
_DESCRIPTOR_SLAB_OFFSET = 0
_DESCRIPTOR_SLAB_BYTES = 1024
_COUNTER_OFFSET = _DESCRIPTOR_SLAB_OFFSET + _DESCRIPTOR_SLAB_BYTES
_COUNTER_REGION_BYTES = 256 * 1024
_MAX_MERGE_GROUPS = _COUNTER_REGION_BYTES // 4
_PARTIAL_OFFSET = _COUNTER_OFFSET + _COUNTER_REGION_BYTES
_PRIMED_ATTR = "_cake_dsv4_counters_primed"
KERNEL_METADATA_PARAMS = (
    "swa_indices",
    "compressed_indices",
    "sparse_topk_lens",
    "swa_index_stride",
    "compressed_index_stride",
    "sparse_topk_lens_offset",
    "sparse_topk",
    "num_query_tokens",
)
_scale_cache: dict[tuple[str, Optional[int], float], torch.Tensor] = {}
_scale_cache_lock = threading.Lock()
_dense_offsets_cache: dict[tuple[str, Optional[int], int, int], torch.Tensor] = {}
_TMA_SOURCE_ALIASES: Mapping[str, str] = {
    "tmap_q": "Q",
    "tmap_swa_k": "SWA_cache",
    "tmap_swa_v": "SWA_cache",
    "tmap_swa_kv": "SWA_cache",
    "tmap_compressed_k": "compressed_KV_cache",
    "tmap_compressed_v": "compressed_KV_cache",
    "tmap_compressed_kv": "compressed_KV_cache",
    "tmap_o": "O",
}
_SCALAR_ALIASES: Mapping[str, str] = {
    "num_q_heads": "num_heads",
    "num_split": "num_splits",
}
_TENSOR_VALUE_NAMES = frozenset(
    {
        "Q",
        "SWA_cache",
        "compressed_KV_cache",
        "O",
        "partial_O",
        "partial_lse",
        "partition_arrivals",
        "split_stats",
        "seq_lens",
        "cum_seq_lens_q",
        "sinks",
        "bmm1_scale",
        "bmm2_scale",
        "swa_indices",
        "compressed_indices",
        "sparse_topk_lens",
        "sparse_indices",
    }
)
_SCALAR_VALUE_NAMES = frozenset(
    {
        "swa_index_stride",
        "compressed_index_stride",
        "sparse_topk_lens_offset",
        "sparse_topk",
        "num_query_tokens",
        "num_heads",
        "num_head_tiles",
        "has_sinks",
        "num_splits",
        "total_work_items",
        "batch_size",
        "max_q_len",
        "ragged_query",
    }
)
_GRID_NAMES = ("grid_x", "grid_y", "grid_z")
_DESCRIPTOR_WORKSPACE_NAME = "tma_descriptor_workspace"
_RAGGED_ONLY_ROUTES = frozenset(
    {
        "fp8_h128_prefill_source_persistent",
        "fp8_h128_prefill_source_persistent_uniform",
        "fp8_h64_prefill_source_persistent_m64",
    }
)
_FP8_PERSISTENT_LANE_GATHER_MIN_TOKENS = 128
_FP8_H64_M64_MIN_TOKENS = 128
