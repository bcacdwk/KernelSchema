"""KDA prefill: engine or prepare/chain; all kernels are local."""

import hashlib, importlib, sys, types
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
_key = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
_package = "_aka_kda_" + _key
if _package not in sys.modules:
    _module = types.ModuleType(_package)
    _module.__path__ = [str(ROOT)]
    sys.modules[_package] = _module
_runtime = importlib.import_module(_package + ".runtime")


import tvm_ffi

_kda_entry = _runtime._kda_entry
_kda_plan = _runtime._kda_plan


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        k,
        v,
        g,
        beta,
        A_log=None,
        dt_bias=None,
        scale=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        lower_bound=None,
        cu_seqlens=None,
        beta_is_logit=False,
        output=None,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
    ):
        assert (
            q.is_cuda
            and q.dtype == torch.bfloat16
            and (q.shape[0] == 1)
            and (q.shape[-1] == 128)
        )
        assert torch.cuda.get_device_capability(q.device) == (10, 3)
        assert all(
            (
                t.is_contiguous()
                for t in (q, k, v, g, A_log, dt_bias, initial_state, cu_seqlens)
            )
        )
        assert (
            output_final_state
            and use_qk_l2norm_in_kernel
            and use_gate_in_kernel
            and beta_is_logit
        )
        assert lower_bound is not None and lower_bound < 0
        assert initial_state.dtype == torch.bfloat16 and cu_seqlens.dtype == torch.int64
        assert checkpoint_every_n_tokens >= 0 and checkpoint_every_n_tokens % 16 == 0
        if checkpoint_every_n_tokens:
            assert state_checkpoints is not None and checkpoint_cu_starts is not None
            assert (
                state_checkpoints.dtype == torch.bfloat16
                and checkpoint_cu_starts.dtype == torch.int64
            )
            assert (
                state_checkpoints.is_contiguous()
                and checkpoint_cu_starts.is_contiguous()
            )
        if output is None:
            output = torch.empty_like(q)
        assert output.shape == q.shape and output.is_contiguous()
        if not beta.is_contiguous() or beta.data_ptr() % 16:
            beta_source = beta
            beta = torch.empty(
                beta_source.shape, dtype=beta_source.dtype, device=beta_source.device
            )
            with tvm_ffi.use_torch_stream():
                _kda_entry("beta_pack")(
                    beta_source,
                    beta,
                    int(torch.cuda.current_stream(beta.device).cuda_stream),
                )
        total, heads = q.shape[1:3]
        sequences = cu_seqlens.numel() - 1
        stream = torch.cuda.current_stream(q.device)
        stream_id = int(stream.cuda_stream)
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        parallel_states = sequences * heads
        engine = checkpoint_every_n_tokens % 32 == 0 and (
            2 * parallel_states > sms
            or (3 * parallel_states > sms and total <= 4096
                and not checkpoint_every_n_tokens)
        )
        p = _kda_plan(total, heads, sequences, q.device, stream_id, engine)
        for t in (
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            cu_seqlens,
            output,
            state_checkpoints,
            checkpoint_cu_starts,
        ):
            if t is not None:
                t.record_stream(stream)
        prep_ctas = (p["capacity"] + 3) // 4 * heads
        with tvm_ffi.use_torch_stream():
            if engine:
                _kda_entry("metadata")(
                    cu_seqlens,
                    p["chunks"],
                    p["mapping"],
                    p["order"],
                    p["capacity"],
                    stream_id,
                    int(engine),
                )
                beta_tma = beta if heads % 8 == 0 and total >= 16 else p["beta_tma"]
                _kda_entry("engine")(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens,
                    p["order"],
                    p["unused_indices"],
                    initial_state,
                    output,
                    initial_state,
                    (
                        state_checkpoints
                        if checkpoint_every_n_tokens
                        else p["unused_ckpt"]
                    ),
                    (
                        checkpoint_cu_starts
                        if checkpoint_every_n_tokens
                        else p["unused_ckpt_starts"]
                    ),
                    heads,
                    beta.stride(-2),
                    initial_state.stride(0),
                    0,
                    1,
                    1,
                    checkpoint_every_n_tokens,
                    float(scale or 128 ** (-0.5)),
                    float(lower_bound),
                    stream_id,
                )
                result = (output, initial_state)
                return (
                    (*result, state_checkpoints)
                    if checkpoint_every_n_tokens
                    else result
                )
            _kda_entry("prepare", heads, sequences)(
                q,
                k,
                g,
                beta,
                A_log,
                dt_bias,
                cu_seqlens,
                p["qd"],
                p["kd"],
                p["w"],
                p["qk"],
                p["diag"],
                p["capacity"],
                heads,
                float(lower_bound),
                prep_ctas,
                1,
                1,
                stream_id,
            )
            _kda_entry("chain")(
                p["qd"],
                p["kd"],
                p["w"],
                p["qk"],
                p["diag"],
                v,
                cu_seqlens,
                p["unused_indices"],
                initial_state,
                output,
                initial_state,
                state_checkpoints if checkpoint_every_n_tokens else p["unused_ckpt"],
                (
                    checkpoint_cu_starts
                    if checkpoint_every_n_tokens
                    else p["unused_ckpt_starts"]
                ),
                checkpoint_every_n_tokens // 16,
                heads,
                initial_state.stride(0),
                0,
                1,
                1,
                float(scale or 128 ** (-0.5)),
                2 * sequences * heads,
                1,
                1,
                stream_id,
            )
        result = (output, initial_state)
        return (*result, state_checkpoints) if checkpoint_every_n_tokens else result
