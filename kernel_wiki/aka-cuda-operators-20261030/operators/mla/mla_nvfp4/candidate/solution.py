"""Prepared NVFP4 MLA: edit kernel/launch/plan directly."""

import hashlib, importlib, sys, types
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
_key = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
_package = "_aka_mla_nvfp4_" + _key
if _package not in sys.modules:
    _module = types.ModuleType(_package)
    _module.__path__ = [str(ROOT)]
    sys.modules[_package] = _module
_runtime = importlib.import_module(_package + ".runtime")


class Model(torch.nn.Module):

    def __init__(
        self,
        query,
        query_scale,
        kv_cache,
        kv_scale,
        block_tables,
        seq_lens,
        workspace_buffer,
        *,
        sm_scale,
        sinks=None,
        out=None,
        lse=None,
        return_lse=False,
        seq_lens_cpu=None,
        backend="cake"
    ):
        super().__init__()
        self.runner = _runtime.prepare_nvfp4_batch_decode_with_kv_cache_mla(
            query,
            query_scale,
            kv_cache,
            kv_scale,
            block_tables,
            seq_lens,
            workspace_buffer,
            sm_scale=sm_scale,
            sinks=sinks,
            out=out,
            lse=lse,
            return_lse=return_lse,
            seq_lens_cpu=seq_lens_cpu,
            backend=backend,
        )

    def forward(self):
        return self.runner()
