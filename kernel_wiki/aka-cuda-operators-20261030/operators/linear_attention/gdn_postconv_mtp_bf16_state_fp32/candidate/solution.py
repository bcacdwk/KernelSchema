"""Independent SM103 CUDA candidate. Edit adjacent CUDA files directly."""
import hashlib
import json
import os
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent

def _build():
    major, minor = torch.cuda.get_device_capability()
    flags = CUDA_FLAGS + [f"-gencode=arch=compute_{major}{minor}f,code=sm_{major}{minor}f"]
    files = {p.name: p.read_text() for p in sorted(ROOT.iterdir())
             if p.suffix in (".cu", ".cuh")}
    key = hashlib.sha256(json.dumps(
        [files, flags, CPP_FLAGS, torch.__version__, torch.version.cuda], sort_keys=True
    ).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("AKA_SF_BUILD_CACHE", Path.home()/".cache/aka-candidates"))
    build = cache / (NAME + "_" + key)
    build.mkdir(parents=True, exist_ok=True)
    return load(name=NAME + "_" + key, sources=[str(ROOT/"launch.cu")],
                extra_cuda_cflags=flags, extra_cflags=CPP_FLAGS,
                build_directory=str(build), verbose=False)

NAME = "aka_gdn_postconv_mtp_bf16_state_fp32_v5"
CUDA_FLAGS = ["-O3", "-std=c++20", "--use_fast_math"]
CPP_FLAGS = ["-O3", "-std=c++20"]
NULL_BLOCK_ID = 0

class Model(torch.nn.Module):
    """GDN decode/MTP post-convolution: QK L2 normalization, decay/beta gates, delta-rule recurrence
    over the caller's state pool, then gated RMSNorm into the caller's `out` buffer.

    The signature is the problem's own ABI, kept argument for argument -- including the
    defaults this problem never varies -- so the evaluator, the reference and the original
    library baseline are all called identically.  `state` checkpoints and `out` are mutated
    in place; `out` itself is returned, not a copy.
    """

    def __init__(self):
        super().__init__()


        extension = _build()
        self._post = extension.fused_gdn_decode_post_conv_mtp

    def forward(self, mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens,
                num_accepted_tokens, state, output_gate, norm_weight, out=None,
                scale=128 ** (-0.5), norm_eps=1e-05, output_gate_activation='silu'):
        if out is None:


            out = torch.empty_like(output_gate)
        self._post(
            mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens,
            num_accepted_tokens, state, output_gate, norm_weight, out,
            float(scale), float(norm_eps), str(output_gate_activation))


        return out

    # Inference-only candidate: no hooks or autograd are registered by this ABI.
    # Keep the full forward in the timed call without nn.Module hook dispatch.
    __call__ = forward
