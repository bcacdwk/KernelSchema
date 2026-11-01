"""GDN prefill: preprocessing, CP/non-CP pipelines and caller state."""

import hashlib, importlib, sys, types
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
_key = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
_package = "_aka_gdn_prefill_bf16_state_fp32_" + _key
if _package not in sys.modules:
    _module = types.ModuleType(_package)
    _module.__path__ = [str(ROOT)]
    sys.modules[_package] = _module
_runtime = importlib.import_module(_package + ".runtime")


import tvm_ffi

_device_facts = _runtime._device_facts
_load_preprocess = _runtime._load_preprocess
chunk_gated_delta_rule_gdn_cp_sm100 = _runtime.chunk_gated_delta_rule_gdn_cp_sm100
_noncp = _runtime._noncp
_copy_tensor = _runtime._copy_tensor


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
        initial_state,
        output_final_state,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_offsets=None,
        use_qk_l2norm_in_kernel=True,
        core_attn_out=None,
    ):
        q, k, v = (x.squeeze(0).contiguous() for x in (q, k, v))
        gate = g.squeeze(0).contiguous()
        beta = beta.squeeze(0).contiguous()
        initial_input = initial_state.contiguous()
        initial = (
            initial_input
            if initial_input.dtype == torch.float32
            else torch.empty_like(initial_input, dtype=torch.float32)
        )
        alpha = torch.empty_like(gate, dtype=torch.float32)
        beta_fp32 = (
            beta
            if beta.dtype == torch.float32
            else torch.empty_like(beta, dtype=torch.float32)
        )
        q_norm = torch.empty_like(q) if use_qk_l2norm_in_kernel else q
        k_norm = torch.empty_like(k) if use_qk_l2norm_in_kernel else k
        make_cu = cu_seqlens is None
        num_seqs = 1 if make_cu else cu_seqlens.numel() - 1
        sms, _ = _device_facts(q.device)
        # Short sequences do not amortize CP precompute/fixup launches.
        use_cp = 4 * num_seqs * v.shape[1] < sms and q.shape[0] > 4096
        cu_dtype = torch.int64 if use_cp else torch.int32
        if make_cu:
            cu_input = cu_seqlens = torch.empty(2, dtype=cu_dtype, device=q.device)
        else:
            cu_input = cu_seqlens.contiguous()
            cu_seqlens = (
                cu_input
                if cu_input.dtype == cu_dtype
                else torch.empty(cu_input.shape, dtype=cu_dtype, device=q.device)
            )
        stream = torch.cuda.current_stream(q.device)
        with tvm_ffi.use_torch_stream():
            _load_preprocess()["run"](
                q,
                k,
                gate,
                beta,
                initial_input,
                q_norm,
                k_norm,
                alpha,
                beta_fp32,
                initial,
                cu_input,
                cu_seqlens,
                make_cu,
                bool(use_qk_l2norm_in_kernel),
                int(stream.cuda_stream),
            )
        q, k, beta = (q_norm, k_norm, beta_fp32)
        scale = 128 ** (-0.5)
        if use_cp:
            output = torch.empty_like(v)
            final = torch.empty_like(initial) if output_final_state else None
            chunk_gated_delta_rule_gdn_cp_sm100(
                output,
                final,
                q,
                k,
                v,
                alpha,
                beta,
                cu_seqlens,
                scale,
                initial_state=initial,
                state_indices=None,
                output_final_state=output_final_state,
                max_seqlen=q.shape[0],
            )
        else:
            output, final = _noncp(
                q, k, v, alpha, beta, initial, output_final_state, cu_seqlens, scale
            )
        output = output.unsqueeze(0)
        if core_attn_out is not None:
            _copy_tensor(output, core_attn_out)
        return (output, final)
