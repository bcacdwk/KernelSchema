"""Build a single-file Agate candidate from the packaged BF16/M64 sources."""

from __future__ import annotations

import argparse
from pathlib import Path

import python_minifier


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT


def _read(name: str) -> str:
    return (SOURCE / name).read_text(encoding="utf-8")


def build(rename_locals: bool) -> str:
    scheduler = _read("gated_delta_net_tile_scheduler.py")
    kernel = _read("gated_delta_net_chunked.py")
    adapter = _read("gdn_prefill.py")

    scheduler_import = """from .gated_delta_net_tile_scheduler import (
    GDNTileSchedulerParams,
    GDNTileScheduler,
)
"""
    if scheduler_import not in kernel:
        raise RuntimeError("unexpected scheduler import")
    kernel = kernel.replace(scheduler_import, "", 1)

    utility_import = "from expert_flashinfer.cute_dsl.utils import get_num_sm\n"
    kernel_import = "from .gated_delta_net_chunked import GatedDeltaNetChunkedKernel\n"
    if utility_import not in adapter or kernel_import not in adapter:
        raise RuntimeError("unexpected adapter imports")
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
    """ABI adapter for qwen38_max_gdn_prefill_sm103_core."""

    def __init__(self):
        super().__init__()
        if torch.cuda.get_device_capability() != (10, 3):
            raise RuntimeError("This solution targets NVIDIA L20D / SM103")
        if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
            raise RuntimeError("The SM103 GDN kernel requires CUDA 13+")

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
        # Always use M64.  Its selector keeps the static full-chunk shortcut
        # only for 128-token pair-aligned B=1 inputs; all other lengths retain
        # the gate=1 / beta=0 padding predicate.
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
            value_rows_override=64,
        )
        return output, output_state
'''

    source = scheduler + "\n" + kernel + "\n" + adapter + task_adapter
    return python_minifier.minify(
        source,
        remove_annotations=False,
        remove_literal_statements=True,
        rename_locals=rename_locals,
        # CuTeDSL's AST lowering recognizes the method receiver by name. Renaming
        # ``self`` changes a compile-time kernel object into a loop-carried value.
        preserve_locals=["self"] if rename_locals else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rename-locals", action="store_true")
    parser.add_argument("--output", default="solution.py")
    args = parser.parse_args()
    header = """# SPDX-License-Identifier: BSD-3-Clause
# BF16/M64 GDN expert candidate; see ../README.md and ../provenance/.
"""
    (ROOT / args.output).write_text(
        header + build(rename_locals=args.rename_locals) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
