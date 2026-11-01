import torch
import tvm_ffi
import math
from dataclasses import dataclass
from functools import cache

GDNCPArch = str

from pathlib import Path
import hashlib, json, os, subprocess
from .plan import (
    _BLOCK,
    _HEAD_DIM,
    _CHUNK_GRANULARITY,
    _TENSOR_MAP_BYTES,
    _PREFILL_TENSOR_MAPS,
    _STATE_IO_THREADS,
    _STATE_DTYPES,
    _ceil_div,
    _round_up,
    _workspace_num_chunks,
    _resolve_max_seqlen,
    _choose_chunk_len,
    _choose_long_chunk_len,
    _default_physical_chunk_len,
    _choose_fixup_kind,
    _require_positive_non_overlapping_state_layout,
    _arch_for,
    GDNCPPrefillPlan,
    _validate_contiguous_tensor,
    _head_mapping_is_legal,
    _build_plan,
    _validate_state,
    _state_carrier,
)

ROOT = Path(__file__).resolve().parent
STAGES = {
    "cp_t_precompute_bf16": {
        "kernel": "cp_t_precompute.cu",
        "host": "cp_t_precompute_launch.cc",
        "entry": "run_t_precompute_bf16",
        "module_ident": "flashinfer_blackwell_gdn_cp_prefill_t_precompute_bf16_v1_75841827ff",
    },
    "cp_mn_precompute_bf16": {
        "kernel": "cp_mn_precompute.cu",
        "host": "cp_mn_precompute_launch.cc",
        "entry": "run_mn_precompute_bf16",
        "module_ident": "flashinfer_blackwell_gdn_cp_prefill_mn_precompute_bf16_v1_4081f2c0ac",
    },
    "cp_state_fixup_simt_row4": {
        "kernel": "sm100_state_fixup_simt_row4.cu",
        "host": "sm100_state_fixup_simt_row4_launch.cc",
        "entry": "run_state_fixup_simt_row4",
        "module_ident": "flashinfer_blackwell_gdn_cp_prefill_fixup_v1_0191fade82",
    },
    "cp_state_fixup_utcmma64": {
        "kernel": "sm100_state_fixup_utcmma64.cu",
        "host": "sm100_state_fixup_utcmma64_launch.cc",
        "entry": "run_state_fixup_utcmma64",
        "module_ident": "flashinfer_blackwell_gdn_cp_prefill_fixup_utcmma64_v1_e12079adf3",
    },
    "cp_state_fixup_utcmma128": {
        "kernel": "sm100_state_fixup_utcmma128.cu",
        "host": "sm100_state_fixup_utcmma128_launch.cc",
        "entry": "run_state_fixup_utcmma128",
        "module_ident": "flashinfer_blackwell_gdn_cp_prefill_fixup_utcmma128_v1_c3f835ae4f",
    },
    "cp_cp_prefill_bf16": {
        "kernel": "cp_cp_prefill.cu",
        "host": "cp_cp_prefill_launch.cc",
        "entry": "run_cp_prefill_bf16",
        "module_ident": "flashinfer_blackwell_gdn_cp_prefill_final_bf16_v1_4affd36e88",
    },
    "cp_cp_prefill_generic_bf16": {
        "kernel": "cp_cp_prefill_generic.cu",
        "host": "cp_cp_prefill_generic_launch.cc",
        "entry": "run_cp_prefill_generic_bf16",
        "module_ident": "flashinfer_blackwell_gdn_cp_prefill_final_generic_bf16_v1_3d3b0d5235",
    },
    "noncp_split": {
        "kernel": "prefill_split.cu",
        "host": "prefill_split_launch.cc",
        "entry": "run_prefill_flashinfer_blackwell_gdn_prefill_dvsplit_initial_59521ae9a3c7",
        "module_ident": "flashinfer_blackwell_gdn_prefill_dvsplit_initial_6130877a92",
    },
    "noncp_full": {
        "kernel": "prefill_full.cu",
        "host": "prefill_full_launch.cc",
        "entry": "run_prefill_flashinfer_blackwell_gdn_prefill_initial_f756fe574915",
        "module_ident": "flashinfer_blackwell_gdn_prefill_initial_1230278b65",
    },
}
FIXUPS = {
    "state_fixup_simt_row4": {
        "ffi_entry": "run",
        "tma_workspace_bytes": 0,
        "tma_workspace_arg_index": None,
        "translation_units": {
            "device": "fixup_simt.cu",
            "binding": "fixup_simt_launch.cu",
        },
    },
    "state_fixup_utcmma64": {
        "ffi_entry": "run",
        "tma_workspace_bytes": 256,
        "tma_workspace_arg_index": 11,
        "translation_units": {
            "device": "fixup_mma64.cu",
            "binding": "fixup_mma64_launch.cu",
        },
    },
    "state_fixup_utcmma128": {
        "ffi_entry": "run",
        "tma_workspace_bytes": 256,
        "tma_workspace_arg_index": 11,
        "translation_units": {
            "device": "fixup_mma128.cu",
            "binding": "fixup_mma128_launch.cu",
        },
    },
}


@cache
def _build_cache():
    from torch.utils import cpp_extension

    files = {p.name: p.read_text() for p in sorted(ROOT.iterdir()) if p.is_file()}
    compiler = subprocess.check_output(
        [str(Path(cpp_extension.CUDA_HOME) / "bin/nvcc"), "--version"], text=True
    )
    key = hashlib.sha256(
        json.dumps(
            [
                files,
                compiler,
                torch.__version__,
                torch.version.cuda,
                getattr(tvm_ffi, "__version__", None),
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    root = Path(
        os.environ.get("AKA_SF_BUILD_CACHE", Path.home() / ".cache/aka-candidates")
    ) / ("gdn_prefill_" + key)
    root.mkdir(parents=True, exist_ok=True)
    return root


@cache
def _load_stage(stage, arch, flags=()):
    from filelock import FileLock
    from torch.utils import cpp_extension
    from tvm_ffi import cpp

    record = STAGES[stage]
    root = _build_cache()
    key = hashlib.sha256((root.name + stage + arch + str(flags)).encode()).hexdigest()[
        :24
    ]
    cubin = root / (key + ".cubin")
    with FileLock(str(cubin) + ".lock"):
        if not cubin.exists():
            subprocess.run(
                [
                    str(Path(cpp_extension.CUDA_HOME) / "bin/nvcc"),
                    "--cubin",
                    "-std=c++17",
                    "-O3",
                    "-arch=" + arch,
                    "-I" + str(ROOT),
                    *flags,
                    str(ROOT / record["kernel"]),
                    "-o",
                    str(cubin),
                ],
                check=True,
            )
    module = cpp.load_inline(
        "aka_gdn_" + key,
        cpp_sources=(ROOT / record["host"]).read_text(),
        embed_cubin={record["module_ident"]: cubin.read_bytes()},
        extra_cflags=[f for f in flags if f.startswith("-D")],
        extra_include_paths=[str(Path(cpp_extension.CUDA_HOME) / "include"), str(ROOT)],
        extra_ldflags=["-lcuda"],
    )
    return module[record["entry"]]


@cache
def load_gdn_cp_kernel(name, arch):
    return _load_stage("cp_" + name, arch)


@cache
def _load_generated_fixup(name, arch):
    from torch.utils import cpp_extension
    import tvm_ffi.libinfo

    record = FIXUPS[name]
    root = _build_cache()
    key = hashlib.sha256((root.name + name + arch).encode()).hexdigest()[:24]
    build = root / ("fixup-" + key)
    build.mkdir(exist_ok=True)
    library = cpp_extension.load(
        name="aka_gdn_fixup_" + key,
        sources=[str(ROOT / p) for p in record["translation_units"].values()],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_" + arch.removeprefix("sm_") + ",code=" + arch,
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            str(ROOT),
            tvm_ffi.libinfo.find_include_path(),
            tvm_ffi.libinfo.find_dlpack_include_path(),
        ],
        build_directory=str(build),
        is_python_module=False,
        verbose=False,
    )
    return tvm_ffi.load_module(library)[record["ffi_entry"]], record


@cache
def _load_preprocess():
    from torch.utils import cpp_extension
    import tvm_ffi.libinfo

    root = _build_cache()
    build = root / "preprocess"
    build.mkdir(exist_ok=True)
    library = cpp_extension.load(
        name="aka_gdn_preprocess_" + root.name,
        sources=[str(ROOT / "preprocess.cu")],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_103a,code=sm_103a",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            str(ROOT),
            tvm_ffi.libinfo.find_include_path(),
            tvm_ffi.libinfo.find_dlpack_include_path(),
        ],
        build_directory=str(build),
        is_python_module=False,
        verbose=False,
    )
    return tvm_ffi.load_module(library)


def prepare_gdn_cp_kernel(name, arch, *, device):
    if arch == "sm_103a" and name.startswith("state_fixup_"):
        entry, record = _load_generated_fixup(name, arch)
        if not record["tma_workspace_bytes"]:
            return (entry, ())
        workspace = torch.empty(
            record["tma_workspace_bytes"], dtype=torch.uint8, device=device
        )
        index = record["tma_workspace_arg_index"]

        def launch(*args):
            return entry(*args[:index], workspace, *args[index:])

        return (launch, (workspace,))
    return (load_gdn_cp_kernel(name, arch), ())


def _log2_or_generic(value):
    return value.bit_length() - 1 if value > 0 and value & value - 1 == 0 else -1


@cache
def _device_facts(device):
    return (
        torch.cuda.get_device_properties(device).multi_processor_count,
        _arch_for(device),
    )


@cache
def _noncp_plan(total, hq, hv, num_seqs, output_final_state, device, stream_id):
    num_sms, arch = _device_facts(device)
    split = 2 * num_seqs * hv <= num_sms
    stage = "noncp_split" if split else "noncp_full"
    flags = (
        "--use_fast_math",
        "-DNUM_O_HEADS_LOG2=" + str(_log2_or_generic(hv)),
        "-DHEAD_GROUP_LOG2=" + str(_log2_or_generic(hv // hq)),
        "-DSTORE_FINAL_STATE=" + str(int(output_final_state)),
        "-DAKA_NUM_Q_HEADS=" + str(hq),
        "-DAKA_NUM_V_HEADS=" + str(hv),
    )
    entry = _load_stage(stage, arch, flags)
    tiles = num_seqs * hv * (2 if split else 1)
    grid = min(num_sms, tiles)
    if not split and num_sms in (148, 160) and (tiles == 256):
        grid = 128
    indices = torch.empty(1, dtype=torch.int32, device=device)
    dummy = torch.empty(1, dtype=torch.float32, device=device)
    workspace = torch.empty(grid * 512, dtype=torch.uint8, device=device)
    unused_state = (
        None
        if output_final_state
        else torch.empty((num_seqs, hv, 128, 128), dtype=torch.float32, device=device)
    )
    return (entry, tiles, grid, indices, dummy, workspace, unused_state)


def _noncp(q, k, v, alpha, beta, initial, output_final_state, cu_seqlens, scale):
    total, hq, _ = q.shape
    hv, num_seqs = (v.shape[1], cu_seqlens.numel() - 1)
    stream = torch.cuda.current_stream(q.device)
    entry, tiles, grid, indices, dummy, workspace, unused_state = _noncp_plan(
        total, hq, hv, num_seqs, output_final_state, q.device, int(stream.cuda_stream)
    )
    output = torch.empty_like(v)
    final = (
        torch.empty((num_seqs, hv, 128, 128), dtype=torch.float32, device=q.device)
        if output_final_state
        else unused_state
    )
    with tvm_ffi.use_torch_stream():
        entry(
            q,
            k,
            v,
            output,
            alpha,
            beta,
            cu_seqlens,
            indices,
            initial,
            final,
            dummy,
            indices,
            workspace,
            initial.stride(0),
            final.stride(0),
            0,
            scale,
            num_seqs,
            hq,
            hv,
            tiles,
            grid,
            1,
            1,
        )
    return (output, final if output_final_state else None)


def _copy_tensor(source, destination):
    stream = torch.cuda.current_stream(source.device)
    with tvm_ffi.use_torch_stream():
        _load_preprocess()["copy"](source, destination, int(stream.cuda_stream))


class GDNCPPrefill:
    """Internal fixed-address GDN CP replay.

    Construction eagerly runs the native composite once, producing ``output``
    and any requested ``output_state``. Non-in-place calls may then capture the
    same fixed bindings in a CUDA Graph. In-place preparation snapshots and
    restores the complete caller state pool once before direct replay.
    """

    def __init__(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: torch.Tensor | None,
        output: torch.Tensor,
        output_state: torch.Tensor | None,
        state_indices: torch.Tensor | None,
        state_checkpoints: torch.Tensor | None,
        checkpoint_cu_starts: torch.Tensor | None,
        scale: float,
        output_final_state: bool,
        alpha_was_none: bool,
        use_qk_l2norm_in_kernel: bool,
        plan: GDNCPPrefillPlan,
        capture_graph: bool = True,
    ) -> None:
        self.q = q
        self.k = k
        self.v = v
        self.alpha = alpha
        self.beta = beta
        self.output = output
        self.state_checkpoints = state_checkpoints
        self.checkpoint_cu_starts = checkpoint_cu_starts
        self.plan = plan
        self.scale = scale
        self.output_final_state = output_final_state
        self.alpha_was_none = alpha_was_none
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self._device_index = q.device.index
        if self._device_index is None:
            self._device_index = torch.cuda.current_device()
        self._stream = torch.cuda.current_stream(q.device)
        self.cu_seqlens_input = cu_seqlens
        self._cu_seqlens_staging = torch.empty(
            (plan.num_seqs + 1,), dtype=torch.int64, device=q.device
        )
        self.cu_seqlens = self._cu_seqlens_staging
        if state_indices is None:
            self.state_indices = torch.empty((1,), dtype=torch.int32, device=q.device)
            self._use_state_indices = False
        else:
            self.state_indices = state_indices
            self._use_state_indices = True
        self.checkpoint_indices = torch.empty(
            (plan.checkpoint_count,), dtype=torch.int64, device=q.device
        )
        state_shape = (plan.num_seqs, plan.num_sab_heads, _HEAD_DIM, _HEAD_DIM)
        needs_gather = bool(
            initial_state is not None
            and (
                initial_state.dtype != torch.float32
                or state_indices is not None
                or (not initial_state.is_contiguous())
                or (output_state is initial_state)
            )
        )
        self.initial_state_input = initial_state
        if initial_state is None:
            self.initial_state = torch.zeros(
                state_shape, dtype=torch.float32, device=q.device
            )
        elif needs_gather:
            self.initial_state = torch.empty(
                state_shape, dtype=torch.float32, device=q.device
            )
        else:
            self.initial_state = initial_state
        self.final_state = output_state
        if output_final_state and self.final_state is None:
            self.final_state = torch.empty(
                state_shape, dtype=torch.float32, device=q.device
            )
        needs_scatter = bool(
            self.final_state is not None
            and (
                self.final_state.dtype != torch.float32
                or state_indices is not None
                or (not self.final_state.is_contiguous())
                or (self.final_state is initial_state)
            )
        )
        self.output_state_workspace = (
            torch.empty(state_shape, dtype=torch.float32, device=q.device)
            if self.final_state is None or needs_scatter
            else self.final_state
        )
        matrix_shape = (plan.total_cp_chunks, plan.num_sab_heads, _HEAD_DIM, _HEAD_DIM)
        self.t = torch.empty(
            (plan.total_t_blocks, plan.num_sab_heads, _BLOCK, _BLOCK),
            dtype=q.dtype,
            device=q.device,
        )
        self.q_normalized = torch.empty_like(q) if use_qk_l2norm_in_kernel else q
        self.k_normalized = torch.empty_like(k) if use_qk_l2norm_in_kernel else k
        self.local_transfer = torch.empty(
            matrix_shape, dtype=torch.float32, device=q.device
        )
        self.local_state = torch.empty(
            matrix_shape, dtype=torch.float32, device=q.device
        )
        self.fixed_state = torch.empty(
            matrix_shape, dtype=torch.float32, device=q.device
        )
        self.initial_state_workspace = torch.empty(
            state_shape, dtype=torch.float32, device=q.device
        )
        self.tensormap_workspace = torch.empty(
            (
                plan.num_seqs
                * plan.num_sab_heads
                * plan.max_cp_chunks_per_seq
                * _PREFILL_TENSOR_MAPS
                * _TENSOR_MAP_BYTES,
            ),
            dtype=torch.uint8,
            device=q.device,
        )
        self._qk_norm = None
        self._t = load_gdn_cp_kernel(plan.t_kernel, plan.arch)
        self._mn, self._mn_workspaces = prepare_gdn_cp_kernel(
            plan.mn_kernel, plan.arch, device=q.device
        )
        self._fixup, self._fixup_workspaces = prepare_gdn_cp_kernel(
            plan.fixup_kernel, plan.arch, device=q.device
        )
        prefill_flags = ()
        if plan.prefill_kernel == "cp_prefill_generic_bf16":
            scalars = ("cp_chunk_len", "source_cp_chunk_len", "num_q_heads",
                       "num_k_heads", "num_v_heads", "num_sab_heads")
            prefill_flags = tuple(
                f"-DAKA_{name.upper()}={getattr(plan, name)}" for name in scalars
            )
        self._prefill = _load_stage("cp_" + plan.prefill_kernel, plan.arch, prefill_flags)
        self._gather = None
        self._gather_source = None
        self._scatter = None
        self._scatter_output = None
        self._checkpoint = None
        if needs_gather:
            assert initial_state is not None
            suffix = str(initial_state.dtype).removeprefix("torch.")
            suffix = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[suffix]
            index_suffix = "_int64" if self.state_indices.dtype == torch.int64 else ""
            self._gather = load_gdn_cp_kernel(
                f"state_gather_{suffix}{index_suffix}", plan.arch
            )
            self._gather_source = _state_carrier(initial_state)
        if needs_scatter:
            assert self.final_state is not None
            suffix = str(self.final_state.dtype).removeprefix("torch.")
            suffix = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[suffix]
            index_suffix = "_int64" if self.state_indices.dtype == torch.int64 else ""
            self._scatter = load_gdn_cp_kernel(
                f"state_scatter_{suffix}{index_suffix}", plan.arch
            )
            self._scatter_output = _state_carrier(self.final_state)
        self._inplace_state = (
            initial_state is not None and self.final_state is initial_state
        )
        snapshot = (
            initial_state.clone(memory_format=torch.preserve_format)
            if self._inplace_state
            else None
        )

        def restore_inplace_state() -> None:
            if snapshot is not None:
                initial_state.copy_(snapshot)

        with torch.cuda.device(q.device), tvm_ffi.use_torch_stream():
            self._launch_direct()
        if capture_graph:
            torch.cuda.synchronize(q.device)
        restore_inplace_state()
        if capture_graph and snapshot is not None:
            torch.cuda.synchronize(q.device)
        self._graph: torch.cuda.CUDAGraph | None = None
        if capture_graph and (not self._inplace_state):
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(
                q.device
            ), tvm_ffi.use_torch_stream(), torch.cuda.graph(self._graph):
                self._launch_direct()
        self._refresh_retained_tensors()

    def _refresh_retained_tensors(self) -> None:
        retained = (
            self.q,
            self.k,
            self.v,
            self.alpha,
            self.beta,
            self.output,
            self.initial_state_input,
            self.final_state,
            self.cu_seqlens_input,
            self.cu_seqlens,
            self.state_indices,
            self.checkpoint_indices,
            self.state_checkpoints,
            self.checkpoint_cu_starts,
            self.initial_state,
            self.output_state_workspace,
            self.q_normalized,
            self.k_normalized,
            self.t,
            self.local_transfer,
            self.local_state,
            self.fixed_state,
            self.initial_state_workspace,
            self.tensormap_workspace,
            *self._mn_workspaces,
            *self._fixup_workspaces,
            self._gather_source,
            self._scatter_output,
        )
        self._retained_tensors = tuple(
            (tensor for tensor in retained if tensor is not None)
        )

    def launch_with_bindings(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor | None,
        beta: torch.Tensor | None,
        initial_state: torch.Tensor | None,
        output: torch.Tensor,
        output_state: torch.Tensor | None,
        state_checkpoints: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor | None,
        checkpoint_cu_starts: torch.Tensor | None,
    ) -> None:
        """Launch the direct composite with structurally compatible addresses."""
        if self._graph is not None:
            raise RuntimeError("dynamic GDN CP bindings require a direct composite")
        current = torch.cuda.current_stream(q.device)
        if current.cuda_stream != self._stream.cuda_stream:
            raise RuntimeError("GDNCPPrefill must launch on its preparation stream")
        self.q = q
        self.k = k
        self.v = v
        self.output = output
        self.cu_seqlens_input = cu_seqlens
        if state_indices is not None:
            self.state_indices = state_indices
        if self._qk_norm is None:
            self.q_normalized = q
            self.k_normalized = k
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        self.initial_state_input = initial_state
        if initial_state is not None:
            if self._gather is None:
                self.initial_state = initial_state
            else:
                self._gather_source = _state_carrier(initial_state)
        self.final_state = output_state
        if output_state is not None:
            if self._scatter is None:
                self.output_state_workspace = output_state
            else:
                self._scatter_output = _state_carrier(output_state)
        self.state_checkpoints = state_checkpoints
        self.checkpoint_cu_starts = checkpoint_cu_starts
        with torch.cuda.device(q.device), tvm_ffi.use_torch_stream():
            self._launch_direct()
        self._refresh_retained_tensors()
        external = (
            q,
            k,
            v,
            alpha,
            beta,
            initial_state,
            output,
            output_state,
            cu_seqlens,
            state_indices,
            state_checkpoints,
            checkpoint_cu_starts,
        )
        seen = set()
        for tensor in external:
            if tensor is not None and id(tensor) not in seen:
                tensor.record_stream(self._stream)
                seen.add(id(tensor))

    def _launch_direct(self) -> None:
        p = self.plan
        if self.cu_seqlens_input.dtype == torch.int64:
            self.cu_seqlens = self.cu_seqlens_input
        else:
            _copy_tensor(self.cu_seqlens_input, self._cu_seqlens_staging)
            self.cu_seqlens = self._cu_seqlens_staging
        if p.total_tokens:
            if self._qk_norm is not None:
                self._qk_norm(
                    self.q,
                    self.k,
                    self.q_normalized,
                    self.k_normalized,
                    p.num_q_heads,
                    p.num_k_heads,
                    p.total_tokens,
                    max(p.num_q_heads, p.num_k_heads),
                    1,
                )
            self._t(
                self.k_normalized,
                self.beta,
                self.t,
                self.cu_seqlens,
                p.num_k_heads,
                p.num_sab_heads,
                p.total_t_blocks,
                p.num_seqs,
                *p.t_grid,
            )
            self._mn(
                self.k_normalized,
                self.v,
                self.t,
                self.alpha,
                self.local_transfer,
                self.local_state,
                self.cu_seqlens,
                p.cp_chunk_len,
                p.num_k_heads,
                p.num_v_heads,
                p.num_sab_heads,
                p.total_cp_chunks,
                p.num_seqs,
                *p.cp_grid,
            )
        if self._gather is not None:
            assert (
                self.initial_state_input is not None and self._gather_source is not None
            )
            total_values = p.num_seqs * p.num_sab_heads * _HEAD_DIM * _HEAD_DIM
            self._gather(
                self._gather_source,
                self.state_indices,
                self.initial_state,
                int(self.initial_state_input.stride(0)),
                int(self.initial_state_input.stride(1)),
                int(self.initial_state_input.stride(2)),
                int(self.initial_state_input.stride(3)),
                p.num_sab_heads,
                total_values,
                int(self._use_state_indices),
                _ceil_div(total_values, _STATE_IO_THREADS),
                1,
                1,
            )
        self._fixup(
            self.local_transfer,
            self.local_state,
            self.initial_state,
            self.initial_state_workspace,
            self.fixed_state,
            self.output_state_workspace,
            self.cu_seqlens,
            p.cp_chunk_len,
            p.total_cp_chunks,
            p.num_seqs,
            p.num_sab_heads,
            *p.fixup_grid,
        )
        if self._checkpoint is not None:
            assert self.state_checkpoints is not None
            total_values = p.checkpoint_count * p.num_sab_heads * _HEAD_DIM * _HEAD_DIM
            self._checkpoint(
                self.fixed_state,
                self.checkpoint_indices,
                self.state_checkpoints,
                int(self.fixed_state.stride(0)),
                int(self.fixed_state.stride(1)),
                int(self.fixed_state.stride(2)),
                int(self.fixed_state.stride(3)),
                p.num_sab_heads,
                total_values,
                1,
                _ceil_div(total_values, _STATE_IO_THREADS),
                1,
                1,
            )
        if self._scatter is not None:
            assert self.final_state is not None and self._scatter_output is not None
            total_values = p.num_seqs * p.num_sab_heads * _HEAD_DIM * _HEAD_DIM
            self._scatter(
                self.output_state_workspace,
                self.state_indices,
                self._scatter_output,
                int(self.final_state.stride(0)),
                int(self.final_state.stride(1)),
                int(self.final_state.stride(2)),
                int(self.final_state.stride(3)),
                p.num_sab_heads,
                total_values,
                int(self._use_state_indices),
                _ceil_div(total_values, _STATE_IO_THREADS),
                1,
                1,
            )
        if p.total_tokens:
            self._prefill(
                self.q_normalized,
                self.k_normalized,
                self.v,
                self.t,
                self.output,
                self.alpha,
                self.cu_seqlens,
                self.fixed_state,
                self.initial_state_workspace,
                self.tensormap_workspace,
                p.cp_chunk_len,
                p.source_cp_chunk_len,
                p.num_q_heads,
                p.num_k_heads,
                p.num_v_heads,
                p.num_sab_heads,
                self.scale,
                *p.cp_grid,
            )

    def replay(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Replay the route once on the caller's current stream."""
        current = torch.cuda.current_stream(self.q.device)
        if self._graph is None:
            with torch.cuda.device(self.q.device), tvm_ffi.use_torch_stream():
                self._launch_direct()
            launch_stream = current
        else:
            if current.cuda_stream != self._stream.cuda_stream:
                raise RuntimeError(
                    "captured GDNCPPrefill must replay on its preparation stream"
                )
            self._graph.replay()
            launch_stream = self._stream
        for tensor in self._retained_tensors:
            tensor.record_stream(launch_stream)
        return (self.output, self.final_state if self.output_final_state else None)


def prepare_gdn_cp_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    output: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    cp_chunk_len: int | None = None,
    max_seqlen: int | None = None,
    scale: float | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    _capture_graph: bool = True,
) -> GDNCPPrefill:
    """Run once and optionally capture one internal fixed-address GDN CP graph."""
    if not q.is_cuda:
        raise ValueError("GDN CP-prefill requires CUDA tensors")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must be rank-3 tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q must have float16 or bfloat16 dtype")
    total = int(q.shape[0])
    if total < 0:
        raise ValueError("q token extent must be nonnegative")
    device = q.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must have int32 or int64 dtype")
    if cu_seqlens.ndim != 1 or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be a contiguous rank-1 tensor")
    num_seqs = int(cu_seqlens.shape[0]) - 1
    if num_seqs <= 0:
        raise ValueError("cu_seqlens must contain at least two entries")
    if checkpoint_every_n_tokens < 0 or (
        checkpoint_every_n_tokens and checkpoint_every_n_tokens % _BLOCK
    ):
        raise ValueError(
            "checkpoint_every_n_tokens must be zero or a positive multiple of 64"
        )
    if cp_chunk_len is not None and (cp_chunk_len <= 0 or cp_chunk_len % _BLOCK):
        raise ValueError("cp_chunk_len must be a positive multiple of 64")
    if not isinstance(use_qk_l2norm_in_kernel, bool):
        raise TypeError("use_qk_l2norm_in_kernel must be a bool")
    checkpoint_count = (
        int(state_checkpoints.shape[0])
        if checkpoint_every_n_tokens and state_checkpoints is not None
        else 0
    )
    plan = _build_plan(
        q,
        k,
        v,
        num_seqs=num_seqs,
        max_seqlen=max_seqlen,
        checkpoint_count=checkpoint_count,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        cp_chunk_len=cp_chunk_len,
    )
    _validate_contiguous_tensor(
        q, name="q", shape=(total, plan.num_q_heads, 128), dtype=q.dtype, device=device
    )
    _validate_contiguous_tensor(
        k, name="k", shape=(total, plan.num_k_heads, 128), dtype=q.dtype, device=device
    )
    _validate_contiguous_tensor(
        v, name="v", shape=(total, plan.num_v_heads, 128), dtype=q.dtype, device=device
    )
    _validate_contiguous_tensor(
        cu_seqlens,
        name="cu_seqlens",
        shape=(plan.num_seqs + 1,),
        dtype=cu_seqlens.dtype,
        device=device,
    )
    gate_shape = (total, plan.num_sab_heads)
    alpha_was_none = alpha is None
    if alpha is None:
        alpha = torch.ones(gate_shape, dtype=torch.float32, device=device)
    else:
        _validate_contiguous_tensor(
            alpha, name="alpha", shape=gate_shape, dtype=torch.float32, device=device
        )
    if beta is None:
        beta = torch.ones(gate_shape, dtype=torch.float32, device=device)
    else:
        _validate_contiguous_tensor(
            beta, name="beta", shape=gate_shape, dtype=torch.float32, device=device
        )
    output_shape = (total, plan.num_sab_heads, _HEAD_DIM)
    if output is None:
        output = torch.empty(output_shape, dtype=q.dtype, device=device)
    else:
        _validate_contiguous_tensor(
            output, name="output", shape=output_shape, dtype=q.dtype, device=device
        )
    indexed = state_indices is not None
    if state_indices is not None:
        if state_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("state_indices must have int32 or int64 dtype")
        _validate_contiguous_tensor(
            state_indices,
            name="state_indices",
            shape=(plan.num_seqs,),
            dtype=state_indices.dtype,
            device=device,
        )
    if checkpoint_every_n_tokens:
        if state_checkpoints is None or checkpoint_cu_starts is None:
            raise ValueError(
                "state_checkpoints and checkpoint_cu_starts are required when checkpointing is enabled"
            )
        _validate_contiguous_tensor(
            state_checkpoints,
            name="state_checkpoints",
            shape=(plan.checkpoint_count, plan.num_sab_heads, 128, 128),
            dtype=torch.float32,
            device=device,
        )
        if checkpoint_cu_starts.dtype not in (torch.int32, torch.int64):
            raise TypeError("checkpoint_cu_starts must have int32 or int64 dtype")
        if (
            tuple(checkpoint_cu_starts.shape) != (plan.num_seqs + 1,)
            or not checkpoint_cu_starts.is_contiguous()
        ):
            raise ValueError(
                "checkpoint_cu_starts must be contiguous with shape [num_seqs + 1]"
            )
        if checkpoint_cu_starts.device != device:
            raise ValueError("checkpoint_cu_starts must be on q.device")
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        raise ValueError(
            "state_checkpoints and checkpoint_cu_starts must be None when checkpointing is disabled"
        )
    if initial_state is not None:
        _validate_state(
            initial_state,
            name="initial_state",
            plan=plan,
            device=device,
            indexed=indexed,
        )
    if output_state is not None:
        _validate_state(
            output_state, name="output_state", plan=plan, device=device, indexed=indexed
        )
    if not output_final_state and output_state is not None:
        raise ValueError("output_state must be None when output_final_state=False")
    if indexed and output_final_state and (output_state is None):
        raise ValueError("indexed final state requires an explicit output_state pool")
    pool_rows = {
        int(tensor.shape[0])
        for tensor in (initial_state, output_state)
        if tensor is not None
    }
    if len(pool_rows) > 1:
        raise ValueError(
            "initial_state and output_state pools must have equal row counts"
        )
    if state_indices is not None:
        if not pool_rows:
            raise ValueError(
                "state_indices requires an initial_state or output_state pool"
            )
    if initial_state is not None and output_state is not None:
        aliases = (
            initial_state.untyped_storage().data_ptr()
            == output_state.untyped_storage().data_ptr()
        )
        if aliases and output_state is not initial_state:
            raise ValueError("state storage aliasing must use the same tensor object")
    if output.numel() > 0:
        output_storage = output.untyped_storage().data_ptr()
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("alpha", alpha),
            ("beta", beta),
        ):
            if tensor.untyped_storage().data_ptr() == output_storage:
                raise ValueError(f"output must not alias read-only input {name}")
    resolved_scale = (
        1.0 / math.sqrt(_HEAD_DIM) if scale is None or scale == 0.0 else float(scale)
    )
    if not math.isfinite(resolved_scale):
        raise ValueError("scale must be finite")
    return GDNCPPrefill(
        q=q,
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output=output,
        output_state=output_state,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        scale=resolved_scale,
        output_final_state=output_final_state,
        alpha_was_none=alpha_was_none,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        plan=plan,
        capture_graph=_capture_graph,
    )


def _layout_key(tensor: torch.Tensor | None) -> tuple[object, ...]:
    if tensor is None:
        return (None,)
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


_public_prepared: GDNCPPrefill | None = None

_public_key: tuple[object, ...] | None = None


def _reset_gdn_cp_prefill_cache() -> None:
    """Release the internal public-dispatch plan and its owned workspaces."""
    global _public_key, _public_prepared
    _public_key = None
    _public_prepared = None


def chunk_gated_delta_rule_gdn_cp_sm100(
    output: torch.Tensor,
    output_state: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    scale: float,
    *,
    initial_state: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    output_final_state: bool,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    cp_chunk_len: int | None = None,
    max_seqlen: int | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> None:
    """Public dispatcher target; every accepted SM100/SM103 route uses GDN CP."""
    global _public_key, _public_prepared
    stream = torch.cuda.current_stream(q.device)
    resolved_max_seqlen = _resolve_max_seqlen(
        total_tokens=int(q.shape[0]),
        num_seqs=int(cu_seqlens.shape[0]) - 1,
        max_seqlen=max_seqlen,
    )
    key: tuple[object, ...] = (
        *(
            _layout_key(tensor)
            for tensor in (
                output,
                output_state,
                q,
                k,
                v,
                alpha,
                beta,
                initial_state,
                state_checkpoints,
            )
        ),
        _layout_key(cu_seqlens),
        _layout_key(state_indices),
        _layout_key(checkpoint_cu_starts),
        output_state is initial_state and initial_state is not None,
        float(scale),
        bool(output_final_state),
        int(checkpoint_every_n_tokens),
        cp_chunk_len,
        resolved_max_seqlen,
        bool(use_qk_l2norm_in_kernel),
        int(stream.cuda_stream),
    )
    if _public_prepared is None or _public_key != key:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GDN CP-prefill plan must be warmed before CUDA graph capture"
            )
        _public_prepared = prepare_gdn_cp_prefill(
            q,
            k,
            v,
            alpha,
            beta,
            cu_seqlens,
            initial_state,
            output=output,
            output_state=output_state,
            state_indices=state_indices,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            cp_chunk_len=cp_chunk_len,
            max_seqlen=resolved_max_seqlen,
            scale=scale,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            _capture_graph=False,
        )
        _public_key = key
        if initial_state is not None and output_state is initial_state:
            _public_prepared.replay()
        return
    _public_prepared.launch_with_bindings(
        q=q,
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
        initial_state=initial_state,
        output=output,
        output_state=output_state,
        state_checkpoints=state_checkpoints,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        checkpoint_cu_starts=checkpoint_cu_starts,
    )
