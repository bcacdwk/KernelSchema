"""Fixed FlashInfer non-CP performance baseline; not an optimization candidate."""
from pathlib import Path
import hashlib
import json

import torch


def verify_flashinfer_source():
    """Reject an unrelated installed FlashInfer implementation, even if its version matches."""
    import flashinfer

    pin = json.loads(Path(__file__).with_name("flashinfer_pin.json").read_text())
    root = Path(flashinfer.__file__).resolve().parent
    failures = []
    for relative, expected in pin["baseline_source_hashes"].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(
            f"FlashInfer source must match {pin['commit']}; mismatched files: {failures}"
        )
    return pin["commit"]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flashinfer_commit = verify_flashinfer_source()
        if torch.cuda.get_device_capability() != (10, 3):
            raise RuntimeError("This baseline is scoped to SM103")
        if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
            raise RuntimeError("Pinned FlashInfer SM103 GDN requires CUDA 13+")
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
        self._gdn = chunk_gated_delta_rule

    def forward(self, q, k, v, alpha, beta, initial_state, cu_seqlens, scale, output, output_state):
        return self._gdn(
            q, k, v, g=alpha, beta=beta, scale=scale,
            initial_state=initial_state, output_final_state=True,
            cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=False,
            output=output, output_state=output_state, use_cp=False,
        )
