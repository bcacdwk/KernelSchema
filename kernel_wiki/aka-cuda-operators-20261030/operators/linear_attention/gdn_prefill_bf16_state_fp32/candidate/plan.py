import torch
import tvm_ffi
import math
from dataclasses import dataclass
from functools import cache

GDNCPArch = str

_BLOCK = 64

_HEAD_DIM = 128

_CHUNK_GRANULARITY = 512

_TENSOR_MAP_BYTES = 128

_PREFILL_TENSOR_MAPS = 5

_STATE_IO_THREADS = 256

_STATE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _round_up(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


def _workspace_num_chunks(total_tokens: int, num_seqs: int, chunk_size: int) -> int:
    bounded_prefix = min(num_seqs, total_tokens)
    return bounded_prefix + (total_tokens - bounded_prefix) // chunk_size


def _resolve_max_seqlen(
    *, total_tokens: int, num_seqs: int, max_seqlen: int | None
) -> int:
    if num_seqs <= 0:
        raise ValueError("num_seqs must be positive")
    resolved = (
        max_seqlen if max_seqlen is not None else _ceil_div(total_tokens, num_seqs)
    )
    if type(resolved) is not int or resolved < 0:
        raise ValueError("max_seqlen must be a nonnegative integer")
    if total_tokens and resolved == 0:
        raise ValueError("max_seqlen must be positive when q is nonempty")
    minimum = _ceil_div(total_tokens, num_seqs)
    if resolved < minimum:
        raise ValueError(
            "max_seqlen cannot be smaller than ceil(total_tokens / num_seqs)"
        )
    if resolved > total_tokens:
        raise ValueError("max_seqlen cannot exceed total_tokens")
    return resolved


def _choose_chunk_len(
    *, total_tokens: int, num_seqs: int, max_seqlen: int, num_heads: int, num_sms: int
) -> int:
    approx_ctas = _ceil_div(total_tokens, _CHUNK_GRANULARITY) * num_heads
    if approx_ctas * 2 < num_sms:
        balanced = math.isqrt(max_seqlen * _BLOCK)
        if balanced * balanced < max_seqlen * _BLOCK:
            balanced += 1
        return max(_BLOCK, _round_up(balanced, _BLOCK))
    target_chunks = max(1, num_sms // num_heads)
    return _choose_long_chunk_len(
        total_tokens=total_tokens,
        num_seqs=num_seqs,
        max_seqlen=max_seqlen,
        target_chunks=target_chunks,
    )


def _choose_long_chunk_len(
    *,
    max_seqlen: int,
    total_tokens: int,
    num_seqs: int,
    target_chunks: int,
    chunk_len_granularity: int = _CHUNK_GRANULARITY,
) -> int:
    """Resolve the shared long-path chunk search for an explicit CTA target."""
    remaining_tokens = max(0, total_tokens - max_seqlen)
    remaining_seqs = max(0, num_seqs - 1)

    def bounded_chunks(chunk_len: int) -> int:
        longest = _ceil_div(max_seqlen, chunk_len)
        compact_bound = min(remaining_seqs, remaining_tokens)
        compact_bound += (remaining_tokens - compact_bound) // chunk_len
        return longest + compact_bound

    lo = 1
    hi = max(1, _ceil_div(max_seqlen, chunk_len_granularity))
    while lo < hi:
        mid = (lo + hi) // 2
        if bounded_chunks(mid * chunk_len_granularity) <= target_chunks:
            hi = mid
        else:
            lo = mid + 1
    return lo * chunk_len_granularity


def _default_physical_chunk_len(
    *,
    source_chunk_len: int,
    max_seqlen: int,
    total_tokens: int,
    num_seqs: int,
    num_heads: int,
    num_sms: int,
) -> int:
    """Choose measured physical waves while preserving source-final boundaries."""
    approx_ctas = _ceil_div(total_tokens, _CHUNK_GRANULARITY) * num_heads
    if approx_ctas * 2 < num_sms:
        return source_chunk_len
    target_chunks = max(1, _ceil_div(num_sms, num_heads))
    granularity = _CHUNK_GRANULARITY
    # Small head counts need finer physical chunks to fill one SM wave.
    # Source chunk boundaries still govern state rounding and checkpoints.
    if num_seqs > 1 and num_heads <= 8:
        granularity = 2 * _BLOCK
    if num_sms == 152:
        if num_seqs == 1 and num_heads in (16, 32, 48, 64):
            waves = 1
            if max_seqlen >= 32768 and num_heads in (32, 64):
                waves = num_heads // 16
            target_chunks = max(1, waves * num_sms // num_heads)
            granularity = 2 * _BLOCK
        elif num_seqs == 2 and num_heads == 48:
            if total_tokens == num_seqs * max_seqlen:
                target_chunks = max(1, 2 * num_sms // num_heads)
                granularity = 2 * _BLOCK
            elif 4 * max_seqlen <= 3 * total_tokens:
                target_chunks = max(1, num_sms // num_heads)
                granularity = 2 * _BLOCK
        elif (
            num_seqs == 2
            and num_heads == 64
            and (total_tokens != num_seqs * max_seqlen)
        ):
            if 4 * max_seqlen > 3 * total_tokens:
                return _round_up(_ceil_div(total_tokens, 2), 2 * _BLOCK)
            target_chunks = max(1, 2 * num_sms // num_heads)
            granularity = 2 * _BLOCK
    return _choose_long_chunk_len(
        max_seqlen=max_seqlen,
        total_tokens=total_tokens,
        num_seqs=num_seqs,
        target_chunks=target_chunks,
        chunk_len_granularity=granularity,
    )


def _choose_fixup_kind(num_parallel_states: int, num_sms: int) -> str:
    if num_parallel_states <= num_sms * 2 // (_HEAD_DIM // 4):
        return "state_fixup_simt_row4"
    if num_parallel_states <= num_sms // (_HEAD_DIM // 64):
        return "state_fixup_utcmma64"
    return "state_fixup_utcmma128"


def _require_positive_non_overlapping_state_layout(
    tensor: torch.Tensor, *, name: str
) -> None:
    """Require an injective positive-strided rank-4 state view."""
    shape = tuple((int(size) for size in tensor.shape))
    strides = tuple((int(stride) for stride in tensor.stride()))
    if len(shape) != 4 or len(strides) != 4:
        raise ValueError(f"{name} must be rank 4")
    if any((stride <= 0 for stride in strides)):
        raise ValueError(f"{name} strides must be positive")
    active = [
        (size - 1, stride)
        for size, stride in zip(shape, strides, strict=True)
        if size > 1
    ]
    if len(active) < 2:
        return
    covered_span = 1
    for bound, stride in sorted(active, key=lambda item: item[1]):
        if stride < covered_span:
            break
        covered_span += bound * stride
    else:
        return
    active.sort(key=lambda item: item[0])
    enumerated = active[:-2]
    (left_bound, left_stride), (right_bound, right_stride) = active[-2:]

    def two_stride_solution(target: int, *, require_nonzero: bool) -> bool:
        divisor = math.gcd(left_stride, right_stride)
        if target % divisor:
            return False
        reduced_left = left_stride // divisor
        reduced_right = right_stride // divisor
        reduced_target = target // divisor
        left0 = (
            0
            if reduced_right == 1
            else reduced_target * pow(reduced_left, -1, reduced_right) % reduced_right
        )
        right0 = (target - left_stride * left0) // right_stride

        def ceil_div(value: int, denominator: int) -> int:
            return -(-value // denominator)

        lower = max(
            ceil_div(-left_bound - left0, reduced_right),
            ceil_div(right0 - right_bound, reduced_left),
        )
        upper = min(
            (left_bound - left0) // reduced_right,
            (right0 + right_bound) // reduced_left,
        )
        if lower > upper:
            return False
        if not require_nonzero or lower < upper:
            return True
        left = left0 + lower * reduced_right
        right = right0 - lower * reduced_left
        return left != 0 or right != 0

    def has_overlap(depth: int, subtotal: int, any_nonzero: bool) -> bool:
        if depth == len(enumerated):
            return two_stride_solution(-subtotal, require_nonzero=not any_nonzero)
        bound, stride = enumerated[depth]
        return any(
            (
                has_overlap(
                    depth + 1, subtotal + delta * stride, any_nonzero or delta != 0
                )
                for delta in range(-bound, bound + 1)
            )
        )

    if has_overlap(0, 0, False):
        raise ValueError(f"{name} must have a non-overlapping strided layout")


def _arch_for(device: torch.device) -> GDNCPArch:
    capability = torch.cuda.get_device_capability(device)
    if capability == (10, 0):
        return "sm_100a"
    if capability == (10, 3):
        return "sm_103a"
    raise ValueError(
        f"GDN CP-prefill requires exact compute capability 10.0 or 10.3, got {capability[0]}.{capability[1]}"
    )


@dataclass(frozen=True)
class GDNCPPrefillPlan:
    """Resolved launch and workspace policy for one legal public input."""

    arch: GDNCPArch
    io_dtype: torch.dtype
    num_q_heads: int
    num_k_heads: int
    num_v_heads: int
    num_sab_heads: int
    num_seqs: int
    total_tokens: int
    max_seqlen: int
    num_sms: int
    cp_chunk_len: int
    source_cp_chunk_len: int
    checkpoint_every_n_tokens: int
    checkpoint_count: int
    total_t_blocks: int
    total_cp_chunks: int
    max_t_blocks_per_seq: int
    max_cp_chunks_per_seq: int
    t_kernel: str
    mn_kernel: str
    fixup_kernel: str
    prefill_kernel: str

    @property
    def t_grid(self) -> tuple[int, int, int]:
        return (self.num_sab_heads * self.max_t_blocks_per_seq, self.num_seqs, 1)

    @property
    def cp_grid(self) -> tuple[int, int, int]:
        return (self.num_sab_heads * self.max_cp_chunks_per_seq, self.num_seqs, 1)

    @property
    def fixup_grid(self) -> tuple[int, int, int]:
        row_ctas = {
            "state_fixup_simt_row4": 32,
            "state_fixup_utcmma64": 2,
            "state_fixup_utcmma128": 1,
        }[self.fixup_kernel]
        return (self.num_seqs * self.num_sab_heads * row_ctas, 1, 1)


def _validate_contiguous_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _head_mapping_is_legal(hq: int, hk: int, hv: int) -> bool:
    return bool(
        hq > 0
        and hk > 0
        and (hv > 0)
        and (hq == hk and hv % hq == 0 or (hk == hv and hq % hk == 0))
    )


def _build_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_seqs: int,
    max_seqlen: int | None,
    checkpoint_count: int = 0,
    checkpoint_every_n_tokens: int = 0,
    cp_chunk_len: int | None = None,
) -> GDNCPPrefillPlan:
    arch = _arch_for(q.device)
    total_tokens, hq, head_dim = map(int, q.shape)
    hk = int(k.shape[1])
    hv = int(v.shape[1])
    if head_dim != _HEAD_DIM or not _head_mapping_is_legal(hq, hk, hv):
        raise ValueError(
            f"GDN CP PR4078 CP-prefill requires D=128 and equal-head, GQA, or GVA mapping; got heads={(hq, hk, hv)}, D={head_dim}"
        )
    max_seqlen = _resolve_max_seqlen(
        total_tokens=total_tokens, num_seqs=num_seqs, max_seqlen=max_seqlen
    )
    if checkpoint_count < 0:
        raise ValueError("checkpoint_count must be nonnegative")
    num_sab_heads = max(hq, hv)
    props = torch.cuda.get_device_properties(q.device)
    num_sms = int(props.multi_processor_count)
    if total_tokens == 0:
        source_cp_chunk_len = cp_chunk_len or checkpoint_every_n_tokens or _BLOCK
    else:
        source_cp_chunk_len = _choose_chunk_len(
            total_tokens=total_tokens,
            num_seqs=num_seqs,
            max_seqlen=max_seqlen,
            num_heads=num_sab_heads,
            num_sms=num_sms,
        )
    gb300_long_hv64_override = (
        not checkpoint_every_n_tokens
        and arch == "sm_103a"
        and (num_seqs == 1)
        and (total_tokens == 65536)
        and ((hq, hk, hv) == (16, 16, 64))
    )
    if cp_chunk_len is None:
        split_bf16_two_block_factor = (
            q.dtype == torch.bfloat16 and source_cp_chunk_len == 2 * _BLOCK
        )
        if checkpoint_every_n_tokens:
            cp_chunk_len = checkpoint_every_n_tokens
        elif gb300_long_hv64_override:
            cp_chunk_len = 4096
        elif split_bf16_two_block_factor:
            cp_chunk_len = _BLOCK
        elif total_tokens:
            cp_chunk_len = _default_physical_chunk_len(
                source_chunk_len=source_cp_chunk_len,
                total_tokens=total_tokens,
                num_seqs=num_seqs,
                max_seqlen=max_seqlen,
                num_heads=num_sab_heads,
                num_sms=num_sms,
            )
        else:
            cp_chunk_len = source_cp_chunk_len
    if cp_chunk_len <= 0 or cp_chunk_len % _BLOCK:
        raise ValueError("cp_chunk_len must be a positive multiple of 64")
    if checkpoint_every_n_tokens and checkpoint_every_n_tokens != cp_chunk_len:
        raise ValueError(
            "checkpoint_every_n_tokens requires the same physical CP chunk length"
        )
    dtype_suffix = "_bf16" if q.dtype == torch.bfloat16 else ""
    t_kernel = f"t_precompute{dtype_suffix}"
    if (
        q.dtype == torch.float16
        and arch == "sm_103a"
        and (hk == 16)
        and (
            num_seqs == 1
            and total_tokens == 65536
            and (num_sab_heads == 48)
            or (
                num_seqs == 2
                and total_tokens != num_seqs * max_seqlen
                and (num_sab_heads in (48, 64))
            )
        )
    ):
        t_kernel = "t_precompute_gb300_hv48_min6"
    all_sequences_equal = total_tokens == num_seqs * max_seqlen
    generic_tail = not (all_sequences_equal and max_seqlen % (2 * _BLOCK) == 0)
    if q.dtype == torch.float16 and (not generic_tail) and (hq == hk == hv):
        prefill_kernel = (
            "cp_prefill_equal_head_h32"
            if arch == "sm_103a" and hq == 32
            else "cp_prefill_equal_head"
        )
    elif generic_tail:
        prefill_kernel = f"cp_prefill_generic{dtype_suffix}"
    else:
        prefill_kernel = f"cp_prefill{dtype_suffix}"
    if arch == "sm_100a" and q.dtype == torch.float16:
        prefill_kernel = {
            "cp_prefill": "cp_prefill_checkpoint",
            "cp_prefill_equal_head": "cp_prefill_equal_head_checkpoint",
            "cp_prefill_generic": "cp_prefill_generic_checkpoint",
        }[prefill_kernel]
    fixup_kernel = _choose_fixup_kind(num_seqs * num_sab_heads, num_sms)
    possible_empty_sequence = total_tokens <= (num_seqs - 1) * max_seqlen
    if possible_empty_sequence and not (
        arch == "sm_103a" and fixup_kernel == "state_fixup_utcmma64"
    ):
        fixup_kernel = "state_fixup_simt_row4"
    return GDNCPPrefillPlan(
        arch=arch,
        io_dtype=q.dtype,
        num_q_heads=hq,
        num_k_heads=hk,
        num_v_heads=hv,
        num_sab_heads=num_sab_heads,
        num_seqs=num_seqs,
        total_tokens=total_tokens,
        max_seqlen=max_seqlen,
        num_sms=num_sms,
        cp_chunk_len=cp_chunk_len,
        source_cp_chunk_len=source_cp_chunk_len,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        checkpoint_count=checkpoint_count if checkpoint_every_n_tokens else 0,
        total_t_blocks=_workspace_num_chunks(total_tokens, num_seqs, _BLOCK),
        total_cp_chunks=_workspace_num_chunks(total_tokens, num_seqs, cp_chunk_len),
        max_t_blocks_per_seq=_ceil_div(max_seqlen, _BLOCK),
        max_cp_chunks_per_seq=_ceil_div(max_seqlen, cp_chunk_len),
        t_kernel=t_kernel,
        mn_kernel=f"mn_precompute{dtype_suffix}",
        fixup_kernel=fixup_kernel,
        prefill_kernel=prefill_kernel,
    )


def _validate_state(
    tensor: torch.Tensor,
    *,
    name: str,
    plan: GDNCPPrefillPlan,
    device: torch.device,
    indexed: bool,
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype not in _STATE_DTYPES:
        raise TypeError(f"{name} must have float32, float16, or bfloat16 dtype")
    expected_inner = (plan.num_sab_heads, _HEAD_DIM, _HEAD_DIM)
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != expected_inner:
        raise ValueError(f"{name} must have shape [N, {expected_inner[0]}, 128, 128]")
    _require_positive_non_overlapping_state_layout(tensor, name=name)
    if not indexed and int(tensor.shape[0]) != plan.num_seqs:
        raise ValueError(f"unindexed {name} first dimension must equal num_seqs")


def _state_carrier(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_contiguous():
        return tensor
    span = 1 + sum(
        (
            (int(size) - 1) * int(stride)
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
    )
    return tensor.as_strided((span,), (1,), storage_offset=int(tensor.storage_offset()))
