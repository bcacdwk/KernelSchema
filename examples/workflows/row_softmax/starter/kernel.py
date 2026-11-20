"""Human-authored, deliberately non-executable starting point (C0, never V0)."""

import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, out):
        """Write row softmax into the original out region and return that object."""
        raise NotImplementedError("Implement the fixed Task and selected Schema")
