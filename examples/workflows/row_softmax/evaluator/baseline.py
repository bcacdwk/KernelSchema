"""FP32 softmax-plus-copy baseline: Task baseline/V0, not the FP64 oracle.

The trusted adapter validates the Task domain before invoking this Model.
Construction is outside timing; softmax allocation and copy remain inside it.
This implementation is intentionally not compliant with the R0 fused design.
"""

import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, out):
        out.copy_(torch.softmax(x, dim=-1))
        return out
