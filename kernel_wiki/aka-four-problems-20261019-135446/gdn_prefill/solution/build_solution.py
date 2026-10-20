"""Build the single-file Agate candidate from pinned upstream sources."""

from __future__ import annotations

from pathlib import Path

import python_minifier


ROOT = Path(__file__).resolve().parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


scheduler = _read("gated_delta_net_tile_scheduler.py")
kernel = _read("gated_delta_net_chunked.py")
adapter = _read("gdn_prefill.py")

kernel_import = """from .gated_delta_net_tile_scheduler import (
    GDNTileSchedulerParams,
    GDNTileScheduler,
)
"""
if kernel_import not in kernel:
    raise RuntimeError("unexpected upstream scheduler import")
kernel = kernel.replace(kernel_import, "", 1)

utility_import = "from flashinfer.cute_dsl.utils import get_num_sm\n"
kernel_import = "from .gated_delta_net_chunked import GatedDeltaNetChunkedKernel\n"
if utility_import not in adapter or kernel_import not in adapter:
    raise RuntimeError("unexpected upstream adapter imports")
adapter = adapter.replace(
    utility_import,
    """@functools.cache
def get_num_sm(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count
""",
    1,
)
adapter = adapter.replace(kernel_import, "", 1)

task_adapter = r'''

class Model(torch.nn.Module):
    """Benchmark ABI adapter for the extracted upstream SM100/SM103 kernel."""

    def __init__(self):
        super().__init__()
        if torch.cuda.get_device_capability() != (10, 3):
            raise RuntimeError("This solution targets NVIDIA L20D / SM103")
        if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
            raise RuntimeError("The extracted SM103 GDN kernel requires CUDA 13+")

    @torch.inference_mode()
    def forward(
        self,
        q,
        k,
        v,
        alpha,
        beta,
        initial_state,
        cu_seqlens,
        scale,
        output,
        output_state,
    ):
        chunk_gated_delta_rule_sm100(
            q,
            k,
            v,
            alpha,
            beta,
            output,
            cu_seqlens.to(torch.int32),
            initial_state,
            output_state,
            scale,
        )
        return output, output_state
'''

header = '''# SPDX-License-Identifier: BSD-3-Clause
# Single-file extraction from FlashInfer commit
# 2ab910c58fdd2392914ea05e2a8714946ac0eef6. See PROVENANCE.json.
'''

source = header + "\n" + scheduler + "\n" + kernel + "\n" + adapter + task_adapter
minified = python_minifier.minify(
    source,
    remove_annotations=False,
    remove_literal_statements=True,
    rename_locals=False,
)
(ROOT / "solution.py").write_text(header + minified + "\n", encoding="utf-8")
